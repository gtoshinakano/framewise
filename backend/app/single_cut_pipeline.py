"""Checkpointed visual analysis for one or more cuts.

The worker only calls this for jobs explicitly marked single_cut_v2. A job may
hold one cut (the per-cut buttons) or every cut of an episode (batch queueing).
Work always runs stage by stage across all cuts, so each model is loaded once:

    1. initial_frame   - one llama-server (Qwen3.5), the first frame of every cut: the still image
    2. video_analysis  - Qwen3-VL, every cut: camera, motion, speaking and emotion over time

The last frame is no longer described. Who is in frame when the cut ends comes from the user's
last-frame character boxes, and how they got there from the video stage, which is told what the
first frame showed so it can follow each named character. Jobs queued before that change may still
name last_frame; it is ignored, and results it produced earlier stay readable.

Subtitles are not a stage here. They are extracted once for the whole episode by
subtitle_pipeline; these stages only learn the subtitle language and where the strip
sits so they can ignore it.

Each result is checkpointed per cut, so an interrupted or retried job skips the
work it already saved. Legacy jobs retain their original execution path.
"""
import gc
import json
import logging
import threading
import time
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import character_detection, db, frame_model, media, structured_results, video_model, worker
from .visual_contracts import FRAME_INSTRUCTION, VIDEO_INSTRUCTION, video_roster

log = logging.getLogger('framewise.visual_analysis')

# Cuts described at once by the frame model. The llama-server gets one slot per cut, so the
# dense 27B model decodes several frames per pass instead of one.
FRAME_PARALLEL = max(1, int(os.environ.get('FRAME_PARALLEL', '4')))


def release_gpu_models():
    from . import inference_worker
    inference_worker._model = None
    inference_worker._caption_model = None
    inference_worker._caption_processor = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass



ALL_STAGES = ('initial_frame', 'video_analysis')


def ensure_models_available(stages=None):
    stages = set(stages or ALL_STAGES)
    required = [Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server')),
                Path(os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf')),
                Path(os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'))]
    if 'video_analysis' in stages:
        video_dir = Path(os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'))
        try:
            index = json.loads((video_dir / 'model.safetensors.index.json').read_text())
            shards = set(index['weight_map'].values())
            if not shards or any(Path(name).name != name or not (video_dir / name).is_file() for name in shards):
                raise ValueError('Missing model shard')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError('Qwen3-VL 8B video model download is incomplete') from exc
    if 'initial_frame' in stages and any(not path.is_file() for path in required):
        raise RuntimeError('Qwen3.5 frame model or llama-server runtime is not available')

DETECTION_NOTE = ('Boxes are pixel coordinates in the named boundary frame only. When labels_confirmed_by_queue is true, '
                  'the user verified the named boxes when queuing this cut; use those names for the corresponding visible people. '
                  'Otherwise labels are unverified hints. Labels alone do not establish presence, actions, or identity across the entire cut.')


def _subtitle_strip(context):
    """The episode's subtitle language and strip as saved when the job was queued, or None before extraction."""
    subtitles = context.get('subtitles') if isinstance(context, dict) else None
    if not isinstance(subtitles, dict):
        return None
    strip, language = subtitles.get('strip'), subtitles.get('language')
    if not language and not strip:
        return None
    return {'language': language, 'top': strip['top'] if strip else None, 'bottom': strip['bottom'] if strip else None}


def strip_box(strip, width, height):
    """The strip as a pixel box on a frame of ``width`` x ``height``; None when its position is unknown."""
    if not strip or strip.get('top') is None:
        return None
    return [0, round(strip['top'] * height), width, round(strip['bottom'] * height)]


def frame_canvas(episode, width=1280):
    """Size of a frame extracted from the original at the analysis width."""
    source_width, source_height = episode['width'] or width, episode['height'] or round(width * 9 / 16)
    scaled = min(source_width, width)
    return scaled, max(2, round(source_height * scaled / source_width / 2) * 2)


class CutRun:
    """One cut inside an analysis job: its checkpoint, saved results and progress messages."""

    def __init__(self, job, owner, episode, item, stages, batch_size):
        self.job, self.owner, self.episode, self.item, self.stages = job, owner, episode, item, stages
        self.cut_id, self.config, self.context = item['cut_id'], item['config'], item['context']
        self.cut = self.context['cut']
        self.batch_size = batch_size
        self.original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
        self.base_ids = item.get('base_result_ids', {})
        self.strip = _subtitle_strip(self.context)
        self.failure = None  # set when this cut fails; the other cuts keep going
        row = db.one('SELECT * FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job['id'], self.cut_id))
        self.completed = bool(row and row['completed'])
        self.checkpoint = json.loads(row['evidence_json']) if row else {}
        if not isinstance(self.checkpoint, dict):
            raise RuntimeError('Single-cut checkpoint is invalid')
        self.hints = {'selected_context': [{'id': entry['id'], 'kind': entry['kind'], 'title': entry['title'],
                                            'description': entry['content'][:500]} for entry in self.context['context_items']],
                      'additional_context': self.context.get('additional_context', '')[:2000],
                      'character_detections': self.context.get('character_detections'),
                      'detection_note': DETECTION_NOTE,
                      'reference_ids': [ref['id'] for ref in item.get('character_references', [])]}

    # -- progress and validation -------------------------------------------------

    def progress(self, message):
        """Show the stage text, prefixed with the cut number when several cuts share the job."""
        prefix = f"Cut {self.cut['cut_index'] + 1}: " if self.batch_size > 1 else ''
        worker.stage(self.job['id'], self.owner, self.episode['id'], prefix + message)

    def stage(self, key, label):
        return worker.analysis_stage(self.job['id'], self.owner, self.cut_id, key, label)

    def cancel_check(self):
        from . import inference_worker
        inference_worker.check_summary_lease(self.job['id'], self.owner)

    def needs(self, kind):
        """True when the job should run this stage for this cut and has no saved result yet."""
        return kind in self.stages and self.saved(kind) is None

    def check_cut(self, conn):
        current = conn.execute('SELECT active,start_time,end_time FROM cuts WHERE id=? AND episode_id=?', (self.cut_id, self.episode['id'])).fetchone()
        if not current or not current['active'] or current['start_time'] != self.cut['start_time'] or current['end_time'] != self.cut['end_time']:
            raise RuntimeError('The cut changed while analysis was queued or running; start a new analysis')

    def validate(self):
        with db.connect() as conn:
            worker.require_lease(conn, self.job['id'], self.owner)
            self.check_cut(conn)

    # -- checkpointed results ----------------------------------------------------

    def save(self, result_type, data, model_id, prompt, extra=None):
        item = self.item
        provenance = {'cut': self.cut, 'model_context_hints': self.hints,
                      **({'subtitle_strip': self.strip} if self.strip else {}),
                      **({'frame_time': data['frame_time']} if 'frame_time' in data else {})}
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            worker.require_lease(conn, self.job['id'], self.owner)
            self.check_cut(conn)
            new_id = structured_results.save_generated(conn, self.cut_id, self.job['id'], self.owner, result_type, data, model_id, prompt,
                {**self.config, 'provider': 'local', **(extra or {})}, provenance,
                expected_current=self.checkpoint.get(result_type, self.base_ids.get(result_type)))
            if result_type == 'video_analysis':
                conn.execute('DELETE FROM analysis_video_windows WHERE job_id=? AND cut_id=? AND window_index>=?',
                             (self.job['id'], self.cut_id, len(data.get('windows', []))))
            self.checkpoint[result_type] = new_id
            conn.execute('INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES (?,?,?) ON CONFLICT(job_id,cut_id) DO UPDATE SET evidence_json=excluded.evidence_json',
                         (self.job['id'], self.cut_id, json.dumps(self.checkpoint)))

    def saved(self, result_type):
        result_id = self.checkpoint.get(result_type)
        if not result_id:
            return None
        row = db.one('SELECT content FROM analysis_results WHERE id=? AND cut_id=?', (result_id, self.cut_id))
        if not row:
            raise RuntimeError('A completed stage result was deleted while the job was running')
        return json.loads(row['content'])

    def mark_complete(self):
        with db.connect() as conn:
            worker.require_lease(conn, self.job['id'], self.owner)
            conn.execute('UPDATE analysis_job_cuts SET completed=1 WHERE job_id=? AND cut_id=?', (self.job['id'], self.cut_id))
            self.completed = True


def run_batch(job, owner, episode, items):
    """Run every requested stage for all cuts, loading each model once."""
    payload = json.loads(job['payload_json'])
    job_stages = set(payload.get('stages') or ALL_STAGES)
    # A cut may list its own stages when only part of its analysis is missing. Jobs queued before
    # subtitles became an episode-wide step or before the last frame was dropped may still name
    # subtitle_ocr or last_frame; neither exists here any more.
    runs = [CutRun(job, owner, episode, item, set(item.get('stages') or job_stages) & set(ALL_STAGES), len(items)) for item in items]
    runs = [cut_run for cut_run in runs if not cut_run.completed]
    if not runs:
        return
    for cut_run in runs:
        cut_run.validate()
    ensure_models_available(set().union(*(cut_run.stages for cut_run in runs)))
    release_gpu_models()
    try:
        with tempfile.TemporaryDirectory(prefix='framewise-cut-analysis-') as scratch:
            _first_frame_stage(runs, scratch)
            _video_stage(runs, scratch)
    finally:
        release_gpu_models()
    # Finished cuts are kept even when another cut failed; a retry only redoes the failed ones.
    for cut_run in runs:
        if not cut_run.failure:
            cut_run.mark_complete()
    failed = [cut_run for cut_run in runs if cut_run.failure]
    if failed:
        details = '; '.join(f"cut {cut_run.cut['cut_index'] + 1}: {cut_run.failure}" for cut_run in failed[:5])
        more = f' (+{len(failed) - 5} more)' if len(failed) > 5 else ''
        raise RuntimeError(f'{len(failed)} of {len(runs)} cuts failed, the others finished and were kept. {details}{more}')


def _for_each(cut_runs, work, parallel=1):
    """Run work(cut_run) for each cut still healthy. A failure is recorded on that cut and the batch goes on.

    Losing the lease or being cancelled ends the whole job, so those still propagate. With
    ``parallel`` > 1 that many cuts run at once; the first lease or cancel error is raised once
    the cuts already started have stopped.
    """
    from . import inference_worker

    def run(cut_run):
        if cut_run.failure:
            return
        try:
            work(cut_run)
        except (worker.LeaseLost, inference_worker.SummaryCancelled):
            raise
        except Exception as exc:
            cut_run.failure = exc
            log.exception('Visual analysis failed for cut %s; continuing with the other cuts', cut_run.cut_id)

    if parallel <= 1 or len(cut_runs) <= 1:
        for cut_run in cut_runs:
            run(cut_run)
        return
    stop = threading.Event()

    def guarded(cut_run):
        if stop.is_set():
            return
        try:
            run(cut_run)
        except BaseException:
            stop.set()  # lease lost or cancelled: queued cuts are skipped
            raise

    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix='frame-describe') as pool:
        futures = [pool.submit(guarded, cut_run) for cut_run in cut_runs]
        errors = [future.exception() for future in futures]
    for error in errors:
        if error:
            raise error


# -- stage 1: first frame (Qwen3.5 through one llama-server) --------------------

def _first_frame_stage(runs, scratch):
    todo = [cut_run for cut_run in runs if not cut_run.failure and cut_run.needs('initial_frame')]
    if not todo:
        return
    with frame_model.frame_server(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'),
        os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
        os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'),
        cancel_check=todo[0].cancel_check, parallel=FRAME_PARALLEL) as server:
        _for_each(todo, lambda cut_run: _frame_stage(cut_run, server, scratch), parallel=FRAME_PARALLEL)


def marked_people(cut_run, position):
    """The characters the user marked in one boundary frame, with the short handles the frame model sees.

    Only boxes that carry a character count: unnamed boxes are left out, and a cut marked as having no
    characters sends none at all, so the model is not asked about detector noise. Boxes are scaled to a
    0-1000 grid of the whole frame. Returns None when the cut has no detection data for that frame,
    which is different from a frame in which nobody was marked.
    """
    if cut_run.cut.get('no_characters'):
        return []
    detections = cut_run.context.get('character_detections') or {}
    if not any(frame.get('position') == position for frame in detections.get('frames', [])):
        return None
    size = next((frame.get('image_size') for frame in detections.get('frames', []) if frame.get('position') == position), None)
    boxes = [d for d in detections.get('detections', []) if d.get('position') == position and d.get('character_context_id')]
    def left_edge(detection):
        box = detection.get('box')
        return box[0] if isinstance(box, list) and box else 0
    people = []
    for number, detection in enumerate(sorted(boxes, key=left_edge), 1):
        box = detection.get('box')
        grid = None
        if isinstance(box, list) and len(box) == 4 and isinstance(size, list) and len(size) >= 2 and size[0] and size[1]:
            grid = [round(box[0] * 1000 / size[0]), round(box[1] * 1000 / size[1]), round(box[2] * 1000 / size[0]), round(box[3] * 1000 / size[1])]
        people.append({'ref': f'B{number}', 'detection_id': detection['id'], 'character_context_id': detection['character_context_id'],
                       'name': detection.get('character_name') or 'unnamed', 'box': grid})
    return people


def _first_frame_result(cut_run):
    """The first-frame result the video stage builds on, as (content, id), or (None, None).

    It is the one this job just saved, else the cut's current one. It must be a full description of
    this cut as it is now; otherwise the video stage runs without it and only gets the names and boxes.
    """
    result_id = cut_run.checkpoint.get('initial_frame')
    if result_id:
        row = db.one('SELECT id,content,context_json FROM analysis_results WHERE id=? AND cut_id=?', (result_id, cut_run.cut_id))
    else:
        current = next((item for item in structured_results.list_results(cut_run.cut_id) if item['type'] == 'initial_frame'), None)
        row = db.one('SELECT id,content,context_json FROM analysis_results WHERE id=?', (current['id'],)) if current else None
    if not row:
        return None, None
    try:
        content, source = json.loads(row['content']), (json.loads(row['context_json'] or '{}').get('cut') or {})
    except (TypeError, ValueError):
        return None, None
    if not isinstance(content, dict) or content.get('mode') != 'full':
        return None, None
    if source.get('start_time') != cut_run.cut['start_time'] or source.get('end_time') != cut_run.cut['end_time']:
        return None, None
    return content, row['id']


def _frame_stage(cut_run, server, scratch):
    if not cut_run.needs('initial_frame'):
        return
    with cut_run.stage('initial_frame', 'Initial frame'):
        started = time.monotonic()
        usage = {}
        cut_run.progress('Analyzing initial frame')
        cut = cut_run.cut
        image = Path(scratch) / f'boundary-{cut_run.cut_id}-first.jpg'  # cuts share the scratch folder when run in parallel
        first, _ = media.cut_frame_times(cut['start_time'], cut['end_time'])
        actual = media.extract_cut_frame(cut_run.original, image, first, cut['start_time'], cut['end_time'], width=1280)
        box = strip_box(cut_run.strip, *frame_canvas(cut_run.episode))
        regions = [box] if box else []
        people = marked_people(cut_run, 'first') or []
        data = frame_model.describe_frame(server, image, regions, cut_run.hints, cancel_check=cut_run.cancel_check, usage_out=usage,
                                          subtitle_language=(cut_run.strip or {}).get('language'), people=people, frame_label='first frame')
        data['frame_time'] = actual
        data['subtitle_regions'] = regions
        cut_run.save('initial_frame', data, frame_model.MODEL_ID, FRAME_INSTRUCTION,
                     {'model_revision': frame_model.MODEL_REVISION, 'image_width': 1280, 'generation_seconds': time.monotonic() - started, 'usage': usage,
                      'frame_mode': 'full', 'marked_characters': len(people)})


# -- stage 2: video analysis (Qwen3-VL) -----------------------------------------

def _video_stage(runs, scratch):
    todo = [cut_run for cut_run in runs if not cut_run.failure and cut_run.needs('video_analysis')]
    if not todo:
        return
    runtime = video_model.VideoRuntime()
    try:
        _for_each(todo, lambda cut_run: _video_analysis(cut_run, scratch, runtime))
    finally:
        runtime.close()


def _video_analysis(cut_run, scratch, runtime):
    job_id, owner, cut_id = cut_run.job['id'], cut_run.owner, cut_run.cut_id
    proxy = cut_run.original.parent / 'proxy.mp4'

    def load_window(index, start, end):
        row = db.one('SELECT * FROM analysis_video_windows WHERE job_id=? AND cut_id=? AND window_index=?', (job_id, cut_id, index))
        if not row:
            return None
        if row['model_revision'] != video_model.MODEL_REVISION:
            raise RuntimeError('A saved video window no longer matches this run')
        if row['start_time'] != start or row['end_time'] != end:
            return None
        data = json.loads(row['content_json'])
        if not isinstance(data, dict) or not isinstance(data.get('source'), dict) or 'analysis' not in data:
            raise RuntimeError('A saved video window is unreadable')
        return data

    def save_window(index, window):
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            worker.require_lease(conn, job_id, owner)
            cut_run.check_cut(conn)
            conn.execute('''INSERT INTO analysis_video_windows(job_id,cut_id,window_index,start_time,end_time,model_revision,content_json) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(job_id,cut_id,window_index) DO UPDATE SET start_time=excluded.start_time,end_time=excluded.end_time,model_revision=excluded.model_revision,content_json=excluded.content_json''',
                (job_id, cut_id, index, window['source']['source_start'], window['source']['source_end'], video_model.MODEL_REVISION, json.dumps(window, ensure_ascii=False)))

    def save_invalid_response(index, start, end, error, response):
        with db.connect() as conn:
            worker.require_lease(conn, job_id, owner)
            conn.execute('INSERT INTO rejected_video_responses(job_id,cut_id,window_index,start_time,end_time,error,response) VALUES (?,?,?,?,?,?,?)',
                         (job_id, cut_id, index, start, end, error, response))

    with cut_run.stage('video_analysis', 'Video analysis'):
        started = time.monotonic()
        initial, initial_id = _first_frame_result(cut_run)
        roster = video_roster(marked_people(cut_run, 'first'), marked_people(cut_run, 'last'), initial)
        data = video_model.analyze_video(
            proxy if proxy.is_file() else cut_run.original, cut_run.cut, cut_run.config,
            os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'), scratch,
            on_window=lambda start, end: cut_run.progress(f'Analyzing video {start:.1f}–{end:.1f}s'),
            subtitle_strip=cut_run.strip, source_size=(cut_run.episode['width'], cut_run.episode['height']), load_window=load_window, save_window=save_window,
            context_hints=cut_run.hints, on_invalid_response=save_invalid_response, cancel_check=cut_run.cancel_check, runtime=runtime,
            roster=roster, initial=initial, transcript=cut_run.context.get('transcript_segments'))
        if initial_id:
            data['based_on_initial_frame'] = initial_id
        cut_run.save('video_analysis', data, video_model.MODEL_ID, VIDEO_INSTRUCTION,
                     {'model_revision': video_model.MODEL_REVISION, 'generation_seconds': time.monotonic() - started, 'usage': data.get('usage', {}),
                      'video_contract': data.get('contract'), 'followed_characters': len(roster)})
