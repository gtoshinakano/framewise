"""Checkpointed visual analysis for one or more cuts.

The worker only calls this for jobs explicitly marked single_cut_v2. A job may
hold one cut (the per-cut buttons) or every cut of an episode (batch queueing).
Work always runs stage by stage across all cuts, so each model is loaded once:

    1. subtitle_ocr    - WeVisOne, every cut
    2. initial/last    - one llama-server (Qwen3.5), both boundary frames of every cut
    3. video_analysis  - Qwen3-VL, every cut

Each result is checkpointed per cut, so an interrupted or retried job skips the
work it already saved. Legacy jobs retain their original execution path.
"""
import gc
import json
import logging
import time
import os
import tempfile
from pathlib import Path
from . import character_detection, db, frame_model, media, structured_results, subtitle_model, video_model, worker
from .visual_contracts import FRAME_INSTRUCTION, VIDEO_INSTRUCTION

log = logging.getLogger('framewise.visual_analysis')


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



ALL_STAGES = ('subtitle_ocr', 'initial_frame', 'last_frame', 'video_analysis')


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
    if stages.intersection(('initial_frame', 'last_frame')) and any(not path.is_file() for path in required):
        raise RuntimeError('Qwen3.5 frame model or llama-server runtime is not available')

SUBTITLE_PROMPT = 'Detect subtitles; OCR only when audio transcript exists.'
DETECTION_NOTE = ('Boxes are pixel coordinates in the named boundary frame only. When labels_confirmed_by_queue is true, '
                  'the user verified the named boxes when queuing this cut; use those names for the corresponding visible people. '
                  'Otherwise labels are unverified hints. Labels alone do not establish presence, actions, or identity across the entire cut.')


class CutRun:
    """One cut inside an analysis job: its checkpoint, saved results and progress messages."""

    def __init__(self, job, owner, episode, item, stages, batch_size):
        self.job, self.owner, self.episode, self.item, self.stages = job, owner, episode, item, stages
        self.cut_id, self.config, self.context = item['cut_id'], item['config'], item['context']
        self.cut = self.context['cut']
        self.batch_size = batch_size
        self.original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
        self.base_ids = item.get('base_result_ids', {})
        self.subtitles = None
        self.references = []
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
                      **({'subtitle_source_result_id': item['subtitle_input']['result_id']} if item.get('subtitle_input') and result_type != 'subtitle_ocr' else {}),
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
    # A cut may list its own stages when only part of its analysis is missing.
    runs = [CutRun(job, owner, episode, item, set(item.get('stages') or job_stages), len(items)) for item in items]
    runs = [cut_run for cut_run in runs if not cut_run.completed]
    if not runs:
        return
    for cut_run in runs:
        cut_run.validate()
    ensure_models_available(set().union(*(cut_run.stages for cut_run in runs)))
    release_gpu_models()
    try:
        with tempfile.TemporaryDirectory(prefix='framewise-cut-analysis-') as scratch:
            _subtitle_stage(runs, scratch)
            _boundary_frame_stages(runs, scratch)
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


def _for_each(cut_runs, work):
    """Run work(cut_run) for each cut still healthy. A failure is recorded on that cut and the batch goes on.

    Losing the lease or being cancelled ends the whole job, so those still propagate.
    """
    from . import inference_worker
    for cut_run in cut_runs:
        if cut_run.failure:
            continue
        try:
            work(cut_run)
        except (worker.LeaseLost, inference_worker.SummaryCancelled):
            raise
        except Exception as exc:
            cut_run.failure = exc
            log.exception('Visual analysis failed for cut %s; continuing with the other cuts', cut_run.cut_id)


# -- stage 1: subtitle detection and OCR (WeVisOne) -----------------------------

def _subtitle_stage(runs, scratch):
    from . import inference_worker
    todo = [cut_run for cut_run in runs if cut_run.needs('subtitle_ocr')]
    if todo:
        model = inference_worker.get_model()
        try:
            _for_each(todo, lambda cut_run: _detect_subtitles(cut_run, scratch, model))
        finally:
            model = None
            release_gpu_models()
    for cut_run in (run for run in runs if not run.failure):
        cut_run.subtitles = (cut_run.saved('subtitle_ocr')
                             or cut_run.item.get('subtitle_input', {}).get('content')
                             or {'samples': [], 'events': []})


def _detect_subtitles(cut_run, scratch, model):
    from . import inference_worker
    with cut_run.stage('subtitle_ocr', 'Subtitle detection and OCR'):
        started = time.monotonic()
        subtitles = subtitle_model.inspect_subtitles(
            cut_run.original, cut_run.cut, cut_run.config, bool(cut_run.context['transcript_segments']), scratch, model,
            on_frame=lambda i, n: cut_run.progress(f'Subtitle frames {i}/{n}'),
            cancel_check=cut_run.cancel_check)
        cut_run.save('subtitle_ocr', subtitles, inference_worker.MODEL_ID, SUBTITLE_PROMPT,
                     {'model_revision': inference_worker.MODEL_REVISION, 'generation_seconds': time.monotonic() - started})


# -- stage 2: first and last frame (Qwen3.5 through one llama-server) -----------

def _boundary_frame_stages(runs, scratch):
    todo = [cut_run for cut_run in runs if not cut_run.failure and (cut_run.needs('initial_frame') or cut_run.needs('last_frame'))]
    if not todo:
        return
    crops = {}  # reference id -> cropped image, shared by every cut that uses it
    _for_each(todo, lambda cut_run: _load_references(cut_run, crops, scratch))
    with frame_model.frame_server(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'),
        os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
        os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'),
        cancel_check=todo[0].cancel_check) as server:
        _for_each(todo, lambda cut_run: (_frame_stage(cut_run, 'first', server, scratch), _frame_stage(cut_run, 'last', server, scratch)))


def _load_references(cut_run, crops, scratch):
    cut_run.references = []
    for ref in cut_run.item.get('character_references', []):
        if ref['id'] not in crops:
            source = media.episode_dir(ref['project_id'], ref['episode_id']) / f"original{ref['original_ext']}"
            crops[ref['id']] = character_detection.crop_frame(source, ref['frame'], ref['cut_start'], ref['cut_end'], ref['box'], scratch, f"analysis-ref-{ref['id']}")
        cut_run.references.append({**ref, 'image': crops[ref['id']]})


def _frame_stage(cut_run, position, server, scratch):
    kind = 'initial_frame' if position == 'first' else 'last_frame'
    if not cut_run.needs(kind):
        return
    label = 'Initial frame' if position == 'first' else 'Last frame'
    with cut_run.stage(kind, label):
        started = time.monotonic()
        usage = {}
        cut_run.progress(f'Analyzing {label.lower()}')
        cut, subtitles = cut_run.cut, cut_run.subtitles
        image = Path(scratch) / 'boundary.jpg'
        first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
        actual = media.extract_cut_frame(cut_run.original, image, first if position == 'first' else last, cut['start_time'], cut['end_time'], width=1280, last=position == 'last')
        closest = min(subtitles['samples'], key=lambda value: abs(value['time'] - actual), default=None)
        regions = closest['regions'] if closest and abs(closest['time'] - actual) < .1 else []
        data = frame_model.describe_frame(server, image, regions, {**cut_run.hints, 'target_frame': position},
                                          cancel_check=cut_run.cancel_check, reference_images=cut_run.references, usage_out=usage)
        if position == 'first':
            _classify_subtitle_languages(cut_run, server)
        data['frame_time'] = actual
        data['subtitle_regions'] = regions
        cut_run.save(kind, data, frame_model.MODEL_ID, FRAME_INSTRUCTION,
                     {'model_revision': frame_model.MODEL_REVISION, 'image_width': 1280, 'generation_seconds': time.monotonic() - started, 'usage': usage})


def _classify_subtitle_languages(cut_run, server):
    """Reuse the already running frame model to label subtitle languages once per cut."""
    from . import inference_worker
    subtitles = cut_run.subtitles
    if 'subtitle_ocr' not in cut_run.stages or not subtitles.get('events') or subtitles.get('language_model'):
        return
    languages = frame_model.classify_subtitle_languages(server, subtitles['events'], cancel_check=cut_run.cancel_check)
    for event, language in zip(subtitles['events'], languages):
        event['language'] = language
    subtitles['language_model'] = frame_model.MODEL_ID
    subtitles['language_note'] = 'Model-estimated language; unknown means uncertain.'
    cut_run.save('subtitle_ocr', subtitles, inference_worker.MODEL_ID, SUBTITLE_PROMPT,
                 {'model_revision': inference_worker.MODEL_REVISION, 'language_model': frame_model.MODEL_ID, 'language_revision': frame_model.MODEL_REVISION})


# -- stage 3: video analysis (Qwen3-VL) -----------------------------------------

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
        data = video_model.analyze_video(
            proxy if proxy.is_file() else cut_run.original, cut_run.cut, cut_run.config,
            os.environ.get('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b'), scratch,
            on_window=lambda start, end: cut_run.progress(f'Analyzing video {start:.1f}–{end:.1f}s'),
            subtitle_samples=cut_run.subtitles['samples'], load_window=load_window, save_window=save_window,
            context_hints=cut_run.hints, on_invalid_response=save_invalid_response, cancel_check=cut_run.cancel_check, runtime=runtime)
        cut_run.save('video_analysis', data, video_model.MODEL_ID, VIDEO_INSTRUCTION,
                     {'model_revision': video_model.MODEL_REVISION, 'generation_seconds': time.monotonic() - started, 'usage': data.get('usage', {})})
