"""Checkpointed visual analysis for one or more cuts.

The worker only calls this for jobs explicitly marked single_cut_v2. A job may
hold one cut (the per-cut buttons) or every cut of an episode (batch queueing).
Work always runs stage by stage across all cuts, so each model is loaded once:

    1. initial_frame      - one llama-server, the first frame of every cut: the still image
    2. keyframe_analysis  - one llama-server and one DWPose load, every cut: what changes between the
                            keyframes (keyframe_analysis.py); its commit queues the performance beats

Which model runs each stage is the job's ``payload.models`` snapshot (visual_models.py): Qwen3.5-27B
by default for both, or e.g. LFM3-Omni-2B for the first frame. Results record the model that wrote them.

The video stage (Qwen3-VL-8B over the whole clip) was replaced by the keyframe stage on 2026-11-18, and the
last frame is no longer described. Jobs queued before either change may still name video_analysis or
last_frame; they are ignored, and results produced earlier stay readable.

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
from . import character_detection, db, frame_model, keyframe_analysis, media, pose_detail, structured_results, visual_models, worker
from .visual_contracts import FRAME_INSTRUCTION

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



ALL_STAGES = ('initial_frame', 'keyframe_analysis')


def job_models(payload):
    """{stage: VisualModel} this job runs: its snapshot (``payload.models``), else the defaults."""
    return {stage: visual_models.for_job(payload, stage) for stage in ALL_STAGES}


def ensure_models_available(stages=None, models=None):
    """Fail before any model loads when a model a stage needs is missing."""
    models = models or job_models({})
    for stage in sorted(set(stages or ALL_STAGES) & set(ALL_STAGES)):
        visual_models.require(models[stage].key)

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

    def save(self, result_type, data, model_id, prompt, extra=None, after=None):
        """Append the result revision and checkpoint it; ``after(conn, result_id)`` runs in the same transaction."""
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
            if after:
                after(conn, new_id)
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
    # subtitles became an episode-wide step, before the last frame was dropped or before keyframes
    # replaced the video stage may still name subtitle_ocr, last_frame or video_analysis; none runs here.
    runs = [CutRun(job, owner, episode, item, set(item.get('stages') or job_stages) & set(ALL_STAGES), len(items)) for item in items]
    runs = [cut_run for cut_run in runs if not cut_run.completed]
    if not runs:
        return
    for cut_run in runs:
        cut_run.validate()
    models = job_models(payload)
    ensure_models_available(set().union(*(cut_run.stages for cut_run in runs)), models)
    release_gpu_models()
    try:
        with tempfile.TemporaryDirectory(prefix='framewise-cut-analysis-') as scratch:
            _first_frame_stage(runs, scratch, models['initial_frame'])
            keyframe_analysis.stage(runs, scratch, models['keyframe_analysis'])
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


# -- stage 1: first frame (the chosen still model through one llama-server) -----

def _first_frame_stage(runs, scratch, model=None):
    todo = [cut_run for cut_run in runs if not cut_run.failure and cut_run.needs('initial_frame')]
    if not todo:
        return
    model = model or visual_models.MODELS[visual_models.DEFAULTS['initial_frame']]
    with frame_model.frame_server(*model.server_files(), cancel_check=todo[0].cancel_check, parallel=FRAME_PARALLEL,
                                  extra_args=model.server_args) as server:
        _for_each(todo, lambda cut_run: _frame_stage(cut_run, server, scratch, model), parallel=FRAME_PARALLEL)


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


def _frame_stage(cut_run, server, scratch, model=None):
    model = model or visual_models.MODELS[visual_models.DEFAULTS['initial_frame']]
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
        cut_run.save('initial_frame', data, model.model_id, FRAME_INSTRUCTION,
                     {'model_revision': model.revision, 'visual_model': model.key, 'image_width': 1280, 'generation_seconds': time.monotonic() - started, 'usage': usage,
                      'frame_mode': 'full', 'marked_characters': len(people), 'frame_contract': data.get('contract'),
                     'pose': pose_detail.stats(data.get('visible_characters'))})
