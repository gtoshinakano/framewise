"""What each kind of queued job is, for the queue and run history (``runs.py``).

One ``Kind`` per ``jobs.pipeline``: its name, whether it runs on the GPU (and so counts in the episode's
queue counter), when it can be cancelled or retried, which cuts, sequence or stretch of audio it works on,
how many individual GPU steps it holds, and what it is "about" so a failure counts as dealt with once the
same thing is run again. Everything here reads the job's payload only; ``runs.py`` adds what needs the
database (current cut numbers, sequence titles, finished stages).

A new pipeline needs one entry in ``KINDS`` (``tests/test_runs.py`` checks every pipeline a worker claims).
"""
from dataclasses import dataclass, field
from typing import Callable

from . import analysis_context

STAGE_LABELS = {'initial_frame': 'Initial frame', 'keyframe_analysis': 'Keyframe analysis', 'video_analysis': 'Video analysis', 'subtitle_ocr': 'Subtitle detection and OCR',
                'last_frame': 'Last frame'}
# The stages a visual job runs when its payload names none (single_cut_pipeline.ALL_STAGES).
DEFAULT_STAGES = ('initial_frame', 'keyframe_analysis')
SEQUENCE_STEPS = {'compile': 'Dialogue', 'bible': 'Bible', 'plan': 'Plan', 'shot': 'Shot', 'review': 'Review'}


def _cut_ref(cut_id, snapshot=None):
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    return {'id': cut_id, 'snapshot': {key: snapshot.get(key) for key in ('cut_index', 'title', 'start_time', 'end_time')}}


def _one_cut(payload):
    cut_id = payload.get('cut_id')
    if not isinstance(cut_id, str) or not cut_id:
        return []
    context = payload.get('context')
    snapshot = payload.get('cut') or (context.get('cut') if isinstance(context, dict) else None)
    return [_cut_ref(cut_id, snapshot)]


def _analysis_cuts(payload):
    items = payload.get('cuts')
    refs = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get('cut_id'), str):
            context = item.get('context')
            refs.append(_cut_ref(item['cut_id'], context.get('cut') if isinstance(context, dict) else None))
    return refs


def _keyframe_cuts(payload):
    if isinstance(payload.get('cut_ids'), list):
        return [_cut_ref(cut_id) for cut_id in payload['cut_ids'] if isinstance(cut_id, str)]
    return [_cut_ref(item['id'], item) for item in payload.get('cuts') or [] if isinstance(item, dict) and isinstance(item.get('id'), str)]


def _frame_cuts(payload):
    ids = payload.get('cut_ids') if isinstance(payload.get('cut_ids'), list) else []
    ids = [*ids, *(payload[key] for key in ('cut_id', 'next_cut_id') if isinstance(payload.get(key), str))]
    seen = []
    for cut_id in ids:
        if isinstance(cut_id, str) and cut_id not in seen:
            seen.append(cut_id)
    return [_cut_ref(cut_id) for cut_id in seen]


def _export_cuts(payload):
    if payload.get('kind') == 'sequence':
        return []
    brief = payload.get('brief') if isinstance(payload.get('brief'), dict) else {}
    cuts = payload.get('asset_cuts') or brief.get('cuts')
    refs, seen = [], set()
    for item in cuts if isinstance(cuts, list) else []:
        cut_id = item.get('id') if isinstance(item, dict) else None
        if isinstance(cut_id, str) and cut_id not in seen:
            seen.add(cut_id)
            refs.append(_cut_ref(cut_id, {**item, 'cut_index': item.get('cut_index', (item.get('cut_number') or 0) - 1 if item.get('cut_number') else None)}))
    return refs


def _no_cuts(payload):
    return []


def stages_of(payload):
    stages = payload.get('stages')
    if isinstance(stages, list) and stages:
        return [stage for stage in stages if isinstance(stage, str)]
    return list(DEFAULT_STAGES) if payload.get('workflow') == 'single_cut_v2' else []


def _analysis_label(payload):
    stages = stages_of(payload)
    if payload.get('workflow') == 'single_cut_v2':
        return STAGE_LABELS.get(stages[0], 'Visual analysis') if len(stages) == 1 else 'Visual analysis'
    return 'Text analysis (older run)' if payload.get('text_only') else 'Visual analysis (older run)'


def _character_label(payload):
    if payload.get('workflow') == 'target_frames_v1':
        return 'Target frame characters'
    if payload.get('workflow') != 'anime_bulk_v1':
        return 'Character detection (older run)'
    return {'boxes': 'Character boxes', 'recognize': 'Character recognition'}.get(payload.get('phase'), 'Character detection')


def _sequence_label(payload):
    return f"Sequence {SEQUENCE_STEPS.get(payload.get('step'), 'step').lower()}"


def _export_label(payload):
    return 'Sequence download' if payload.get('kind') == 'sequence' else 'Brief download'


def _summary_label(payload):
    return 'Cut summary' if payload.get('result_type', 'summary') == 'summary' or payload.get('production_brief') else 'Text result'


def _transcribe_range(payload):
    span = payload.get('range')
    if isinstance(span, dict) and all(isinstance(span.get(key), (int, float)) for key in ('start_time', 'end_time')):
        return {'start': span['start_time'], 'end': span['end_time']}
    return None


def _voice_range(payload):
    ranges = [item for item in payload.get('ranges') or [] if isinstance(item, dict)]
    starts = [item['start_time'] for item in ranges if isinstance(item.get('start_time'), (int, float))]
    ends = [item['end_time'] for item in ranges if isinstance(item.get('end_time'), (int, float))]
    return {'start': min(starts), 'end': max(ends)} if starts and ends else None


def _transcribe_key(payload):
    span = _transcribe_range(payload)
    return f"transcribe:{span['start']:.3f}-{span['end']:.3f}" if span else 'transcribe:all'


def _cut_keys(prefix):
    return lambda payload, cut_ids: [f'{prefix}:{cut_id}' for cut_id in cut_ids]


def _character_keys(payload, cut_ids):
    # Characters on the evaluated frames are their own work: a first-frame run never deals with their failure.
    prefix = 'target_characters' if payload.get('workflow') == 'target_frames_v1' else 'character_detect'
    return [f'{prefix}:{cut_id}' for cut_id in cut_ids]


def _analysis_keys(payload, cut_ids):
    return [f'analyze:{cut_id}:{stage}' for cut_id in cut_ids for stage in stages_of(payload) or ['legacy']]


@dataclass(frozen=True)
class Scope:
    """Data read and written within an episode. None covers all cuts."""
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    cut_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class Kind:
    """How the queue describes and handles one pipeline."""
    pipeline: str
    title: str
    gpu: bool
    label: Callable[[dict], str] | None = None
    cuts: Callable[[dict], list] = _no_cuts
    # Whether a queued / running job of this kind can be cancelled from the queue.
    cancel_queued: bool = True
    cancel_running: Callable[[dict], bool] = lambda payload: False
    # ``POST /api/jobs/{id}/retry`` supports a failed or cancelled job of this kind.
    retry: Callable[[dict], bool] = lambda payload: False
    # Individual GPU steps the job holds; visual analysis counts one per cut and stage.
    stage_units: bool = False
    # What the job is about: a later job with the same keys deals with a failure (``runs.attention``).
    keys: Callable[[dict, list], list] = lambda payload, cut_ids: []
    # Episode-wide work states the stretch of audio it covers.
    audio_range: Callable[[dict], dict | None] = lambda payload: None
    # Failures of these kinds ask for the user's attention; downloads requested by other apps do not.
    attention: bool = True
    description: str = field(default='')
    resources: frozenset[str] = frozenset()
    scope: Callable[[dict], Scope] | None = None
    priority_class: Callable[[dict], str] | None = None
    inputs: Callable | None = None
    gather: Callable | None = None
    on_inputs_changed: str = 'requeue'
    invalidates: Callable | None = None

    def name(self, payload):
        return self.label(payload) if self.label else self.title

    def can_cancel(self, status, payload):
        if status == 'queued':
            return self.cancel_queued
        return status == 'running' and self.cancel_running(payload)

    def units(self, payload):
        if self.stage_units:
            return max(1, len(self.cuts(payload)) * max(1, len(stages_of(payload))))
        return 1

    def priority(self, payload):
        if self.priority_class:
            return self.priority_class(payload)
        return 'interactive' if len(self.cuts(payload)) == 1 else 'batch'


def _always(payload):
    return True


def _visual_scope(payload):
    return Scope(frozenset({'cut_boundaries', 'detections', 'subtitles', 'context', 'keyframes'}),
                 frozenset(f'results:{stage}' for stage in stages_of(payload)),
                 frozenset(ref['id'] for ref in _analysis_cuts(payload)))


def _visual_invalidates(conn, job):
    from . import structured_results
    structured_results.invalidate_text(conn, job['unit_id'], job['result_type'])


def _detection_scope(payload):
    target = payload.get('workflow') == 'target_frames_v1'
    domains = {'cut_boundaries', 'detections', 'context'}
    if target:
        domains.add('keyframes')
    return Scope(frozenset(domains), frozenset({'target_detections' if target else 'detections'}),
                 frozenset(ref['id'] for ref in _one_cut(payload)))


def _detection_inputs(conn, job):
    from .job_inputs import fingerprint
    inputs = job.get('gathered') or _gather_detection(conn, job)
    return fingerprint({key: value for key, value in inputs.items() if key != 'episode'})


def _gather_detection(conn, job):
    import json
    from . import anime_bulk, target_characters
    payload = json.loads(job['payload_json'])
    gather = target_characters.gather if payload.get('workflow') == target_characters.WORKFLOW else anime_bulk.gather
    return gather(conn, job, payload)


def _detection_invalidates(conn, job):
    import json
    from . import structured_results
    payload = json.loads(job['payload_json'])
    structured_results.invalidate_detections(conn, payload['cut_id'], target=payload.get('workflow') == 'target_frames_v1')


def _preview_scope(payload):
    ids = frozenset(ref['id'] for ref in _frame_cuts(payload))
    return Scope(frozenset({'cut_boundaries'}), frozenset({'cut_boundaries' if payload.get('frame_step') else 'previews'}),
                 None if payload.get('frame_step') else ids or None)


def _preview_gather(conn, job):
    from . import cut_previews
    return cut_previews.gather(conn, job)


def _preview_inputs(conn, job):
    from . import cut_previews
    return cut_previews.inputs(conn, job)


def _preview_invalidates(conn, job):
    from . import cut_previews
    cut_previews.invalidate(conn, job)


def _hardcut_scope(payload):
    writes = {'cut_boundaries', 'previews'}
    if not payload.get('cut_id'):
        writes.add('chunks')
    return Scope(frozenset({'proxy', 'cut_boundaries'}), frozenset(writes))


def _media_gather(conn, job):
    from . import media_jobs
    return media_jobs.gather(conn, job)


def _media_inputs(conn, job):
    from . import media_jobs
    return media_jobs.inputs(conn, job)


def _media_invalidates(conn, job):
    from . import media_jobs
    media_jobs.invalidate(conn, job)


def _selection_scope(payload):
    cuts = frozenset(ref['id'] for ref in _keyframe_cuts(payload))
    return Scope(frozenset({'original', 'cut_boundaries', 'subtitles'}),
                 frozenset({'keyframes', 'target_detections'}), cuts or None)


def _selection_gather(conn, job):
    from . import keyframes
    return keyframes.gather(conn, job)


def _selection_inputs(conn, job):
    from . import keyframes
    return keyframes.inputs(conn, job)


def _selection_invalidates(conn, job):
    from . import keyframes
    keyframes.invalidate(conn, job)


def _transcription_scope(payload):
    return Scope(frozenset({'prepared_audio', 'voices', 'transcript'}), frozenset({'transcript'}))


def _voice_scope(payload):
    reads = {'context', 'voice_upload'} if payload.get('upload_id') else {'context', 'prepared_audio'}
    return Scope(frozenset(reads), frozenset({'voices'}))


def _audio_gather(conn, job):
    from . import audio_worker
    return audio_worker.gather(conn, job)


def _audio_inputs(conn, job):
    from . import audio_worker
    return audio_worker.inputs(conn, job)


def _audio_invalidates(conn, job):
    from . import audio_worker
    audio_worker.invalidate(conn, job)


def _subtitle_gather(conn, job):
    from . import subtitle_pipeline
    return subtitle_pipeline.gather(conn, job)


def _subtitle_inputs(conn, job):
    from . import subtitle_pipeline
    return subtitle_pipeline.inputs(conn, job)


def _subtitle_invalidates(conn, job):
    from . import subtitle_pipeline
    subtitle_pipeline.invalidate(conn, job)


def _beats_gather(conn, job):
    from . import performance_beats
    return performance_beats.gather_job(conn, job)


def _beats_inputs(conn, job):
    from . import performance_beats
    return performance_beats.inputs(conn, job)


def _beats_invalidates(conn, job):
    from . import performance_beats
    performance_beats.invalidate(conn, job)


def _consistency_gather(conn, job):
    from . import summary_queue
    return summary_queue.gather(conn, job)


def _consistency_inputs(conn, job):
    from . import summary_queue
    return summary_queue.inputs(conn, job)


def _consistency_invalidates(conn, job):
    from . import summary_queue
    summary_queue.invalidate(conn, job)


KINDS = {kind.pipeline: kind for kind in (
    Kind('analyze', 'Visual analysis', True, label=_analysis_label, cuts=_analysis_cuts, stage_units=True, keys=_analysis_keys,
         # Older analysis workflows stop safely at their next checkpoint only when they are text-only.
         cancel_running=lambda payload: payload.get('workflow') == 'single_cut_v2' or payload.get('text_only') is True,
         retry=_always, resources=frozenset({'gpu'}), scope=_visual_scope,
         priority_class=lambda payload: 'batch' if payload.get('batch') else 'interactive',
         inputs=analysis_context.visual_fingerprint, gather=analysis_context.visual, invalidates=_visual_invalidates),
    Kind('character_detect', 'Character detection', True, label=_character_label, cuts=_one_cut, keys=_character_keys,
         # Bulk detection keeps one lease per cut and skips a cut whose lease is gone; the older workflow cannot stop midway.
         cancel_running=lambda payload: payload.get('workflow') in ('anime_bulk_v1', 'target_frames_v1'),
         resources=frozenset({'gpu'}), scope=_detection_scope,
         priority_class=lambda payload: 'batch' if payload.get('episode_batch') else 'interactive',
         inputs=_detection_inputs, gather=_gather_detection, invalidates=_detection_invalidates),
    # Describes one labelled box of a character created from it and appends the text to its Details.
    Kind('character_profile', 'Character description', True, cuts=_one_cut, cancel_running=_always, retry=_always,
         keys=lambda payload, cut_ids: [f"character_profile:{payload.get('character_context_id')}"]),
    # The timed acting plan a text model compiles from the keyframe analysis (performance_beats.py).
    Kind('performance_beats', 'Performance beats', True, cuts=_one_cut, keys=_cut_keys('performance_beats'), cancel_running=_always, retry=_always,
         resources=frozenset({'text'}),
         scope=lambda payload: Scope(frozenset({'cut_boundaries', 'results:initial_frame', 'results:keyframe_analysis', 'transcript', 'context', 'chunks'}),
                                     frozenset({'results:performance_beats'}), frozenset({payload['cut_id']}) if payload.get('cut_id') else None),
         priority_class=lambda payload: 'interactive', inputs=_beats_inputs, gather=_beats_gather, invalidates=_beats_invalidates),
    Kind('consistency_summary', 'Consistency summary', True, cuts=_one_cut, keys=_cut_keys('consistency_summary'), cancel_running=_always,
         resources=frozenset({'text'}),
         scope=lambda payload: Scope(frozenset({'cut_boundaries', 'results', 'transcript', 'context', 'chunks', 'detections', 'target_detections', 'subtitles'}),
                                     frozenset({'results:consistency_summary'})),
         priority_class=lambda payload: 'batch' if payload.get('batch') else 'interactive',
         inputs=_consistency_inputs, gather=_consistency_gather, invalidates=_consistency_invalidates),
    Kind('review_answers', 'Production decisions draft', True, cuts=_one_cut, keys=_cut_keys('review_answers'), cancel_running=_always),
    Kind('regen', 'Cut summary', True, label=_summary_label, cuts=_one_cut, keys=_cut_keys('regen'), cancel_running=_always),
    Kind('composition_compact', 'Composition brief', True, cuts=_one_cut, keys=_cut_keys('composition_compact'), cancel_running=_always,
         retry=_always, attention=False),
    Kind('sequence', 'Sequence', True, label=_sequence_label, cancel_running=_always,
         keys=lambda payload, cut_ids: [f"sequence:{payload.get('sequence_id')}:{payload.get('step')}:{payload.get('shot_key') or ''}"]),
    Kind('subtitles', 'Subtitle extraction', True, cancel_running=_always,
         keys=lambda payload, cut_ids: [f"subtitles:{payload.get('language') or ''}"], resources=frozenset({'gpu'}),
         scope=lambda payload: Scope(frozenset({'original', 'proxy', 'transcript'}), frozenset({'subtitles'})),
         priority_class=lambda payload: 'batch', inputs=_subtitle_inputs, gather=_subtitle_gather, invalidates=_subtitle_invalidates),
    Kind('transcribe', 'Transcription', True, audio_range=_transcribe_range, cancel_running=_always,
         keys=lambda payload, cut_ids: [_transcribe_key(payload)], resources=frozenset({'gpu', 'audio'}),
         scope=_transcription_scope, priority_class=lambda payload: 'interactive' if payload.get('range') else 'batch',
         inputs=_audio_inputs, gather=_audio_gather, invalidates=_audio_invalidates),
    Kind('voice_sample', 'Voice sample', True, audio_range=_voice_range, cancel_running=_always,
         keys=lambda payload, cut_ids: [f"voice_sample:{payload.get('character_context_id')}"],
         resources=frozenset({'gpu', 'audio'}), scope=_voice_scope, priority_class=lambda payload: 'interactive',
         inputs=_audio_inputs, gather=_audio_gather, invalidates=_audio_invalidates),
    # Media work runs on the CPU in the API process: listed, never counted, not cancellable while it runs.
    Kind('keyframes', 'Keyframe selection', False, cuts=_keyframe_cuts, cancel_running=_always, retry=_always, keys=lambda payload, cut_ids: ['keyframes'],
         resources=frozenset({'cpu_media'}), scope=_selection_scope,
         priority_class=lambda payload: 'interactive' if len(_keyframe_cuts(payload)) == 1 else 'batch',
         inputs=_selection_inputs, gather=_selection_gather, invalidates=_selection_invalidates),
    Kind('prepare', 'Prepare media', False, cancel_queued=False, keys=lambda payload, cut_ids: ['prepare'],
         resources=frozenset({'cpu_media'}),
         scope=lambda payload: Scope(frozenset({'original'}), frozenset({'proxy', 'prepared_audio', 'chunks', 'previews'})),
         priority_class=lambda payload: 'batch', inputs=_media_inputs, gather=_media_gather, invalidates=_media_invalidates),
    Kind('hardcuts', 'Hard-cut detection', False, cuts=_one_cut, cancel_queued=False,
         keys=lambda payload, cut_ids: [f'hardcuts:{cut_id}' for cut_id in cut_ids] or ['hardcuts'],
         resources=frozenset({'cpu_media'}), scope=_hardcut_scope,
         priority_class=lambda payload: 'interactive' if payload.get('cut_id') else 'batch',
         inputs=_media_inputs, gather=_media_gather, invalidates=_media_invalidates),
    Kind('cutframes', 'Cut previews', False, cuts=_frame_cuts, cancel_queued=False,
         keys=lambda payload, cut_ids: [f'cutframes:{cut_id}' for cut_id in cut_ids],
         resources=frozenset({'cpu_media'}), scope=_preview_scope,
         priority_class=lambda payload: 'interactive' if payload.get('frame_step') else 'batch',
         inputs=_preview_inputs, gather=_preview_gather, invalidates=_preview_invalidates),
    Kind('brief_export', 'Download', False, label=_export_label, cuts=_export_cuts, cancel_queued=False, attention=False),
)}

def kind_of(pipeline):
    """The registry entry of a pipeline; unknown pipelines are listed under their own name and never counted."""
    kind = KINDS.get(pipeline)
    if kind:
        return kind
    return Kind(pipeline, str(pipeline or 'run').replace('_', ' ').capitalize(), False, cancel_queued=False)


GPU_PIPELINES = tuple(name for name, kind in KINDS.items() if kind.gpu)
