"""What each kind of queued job is, for the queue and run history (``runs.py``).

One ``Kind`` per ``jobs.pipeline``: its name, whether it runs on the GPU (and so counts in the episode's
queue counter), when it can be cancelled or retried, which cuts, sequence or stretch of audio it works on,
how many individual GPU steps it holds, and what it is "about" so a failure counts as dealt with once the
same thing is run again. Everything here reads the job's payload only; ``runs.py`` adds what needs the
database (current cut numbers, sequence titles, finished stages).

A new pipeline needs one entry in ``KINDS`` (``tests/test_runs.py`` checks every pipeline a worker claims).
"""
from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

from . import analysis_context

STAGE_LABELS = {'initial_frame': 'Initial frame', 'keyframe_analysis': 'Keyframe analysis', 'video_analysis': 'Video analysis', 'subtitle_ocr': 'Subtitle detection and OCR',
                'last_frame': 'Last frame'}
# The stages a visual job runs when its payload names none (single_cut_pipeline.ALL_STAGES).
DEFAULT_STAGES = ('initial_frame', 'keyframe_analysis')
SEQUENCE_STEPS = {'compile': 'Dialogue', 'bible': 'Bible', 'plan': 'Plan', 'shot': 'Shot', 'shots': 'Shots', 'review': 'Review'}


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
    adjacent_reads: frozenset[str] = frozenset()
    sequence_id: str | None = None
    # Zero-based inclusive cut numbers. These are resolved in the claim transaction.
    cut_range: tuple[int, int] | None = None
    previous_shot_to: str | None = None


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
    lane: str = 'media'
    # Preserve the existing ordering for background cache work and its episode-status behavior.
    claim_order: int = 0
    affects_episode: bool = True
    handler: str | Callable | None = None
    # A GPU job normally gets the whole device: the resident Qwen server is stopped for it. A job whose models all run on that
    # resident server (single-cut visual analysis, a character profile) says False and keeps it loaded.
    needs_device: Callable[[dict], bool] = lambda payload: True

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

    def run(self, job_id, owner):
        handler = self.handler
        if isinstance(handler, str):
            module, name = handler.rsplit('.', 1)
            handler = getattr(import_module('.' + module, __package__), name)
        return handler(job_id, owner)


def _resident_models(payload, roles):
    """Whether the models a job snapshotted for ``roles`` all run on the resident Qwen server (see ``visual_models``)."""
    from . import visual_models
    return visual_models.runs_resident(payload, roles)


def _always(payload):
    return True


def _visual_scope(payload):
    stages = stages_of(payload)
    writes = {f'results:{stage}' for stage in stages}
    reads = {'original', 'cut_boundaries', 'detections', 'subtitles', 'context', 'keyframes'}
    if 'keyframe_analysis' in stages:
        reads.add('target_detections')
    if payload.get('workflow') != 'single_cut_v2':
        reads.update({'transcript', 'chunks', 'results'})
        writes.add('results')
    ids = frozenset(ref['id'] for ref in _analysis_cuts(payload)) or None
    if payload.get('dynamic_cuts'):
        return Scope(frozenset(reads), frozenset(writes), cut_range=(payload.get('from_index', 0), 2**63 - 1))
    nearby = frozenset({'cut_boundaries', 'results'}) if payload.get('workflow') != 'single_cut_v2' else frozenset()
    return Scope(frozenset(reads), frozenset(writes), ids, nearby)


def _visual_invalidates(conn, job):
    from . import structured_results
    if job['result_type'] == 'legacy_visual':
        conn.execute("UPDATE cut_review_drafts SET stale_reason='Cut analysis changed' WHERE cut_id=?", (job['unit_id'],))
    else:
        structured_results.invalidate_text(conn, job['unit_id'], job['result_type'])


def _detection_scope(payload):
    target = payload.get('workflow') == 'target_frames_v1'
    domains = {'original', 'cut_boundaries', 'detections', 'context'}
    if target:
        domains.add('keyframes')
    return Scope(frozenset(domains), frozenset({'target_detections' if target else 'detections'}),
                 frozenset(ref['id'] for ref in _one_cut(payload)) or None)


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


def _profile_gather(conn, job):
    from . import character_profile
    return character_profile.gather(conn, job)


def _profile_inputs(conn, job):
    from . import character_profile
    return character_profile.inputs(conn, job)


def _profile_invalidates(conn, job):
    from . import character_profile
    character_profile.invalidate(conn, job)


def _preview_scope(payload):
    ids = frozenset(ref['id'] for ref in _frame_cuts(payload))
    moves_boundary = bool(payload.get('frame_step'))
    return Scope(frozenset({'proxy', 'cut_boundaries'}), frozenset({'cut_boundaries', 'previews'} if moves_boundary else {'previews'}),
                 None if moves_boundary and not payload.get('next_cut_id') else ids or None)


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


def _missed_cut_gather(conn, job):
    from . import missed_cuts
    return missed_cuts.gather(conn, job)


def _missed_cut_inputs(conn, job):
    from . import missed_cuts
    return missed_cuts.inputs(conn, job)


def _missed_cut_invalidates(conn, job):
    from . import missed_cuts
    missed_cuts.invalidate(conn, job)


def _line_split_gather(conn, job):
    from . import line_splits
    return line_splits.gather(conn, job)


def _line_split_inputs(conn, job):
    from . import line_splits
    return line_splits.inputs(conn, job)


def _line_split_invalidates(conn, job):
    from . import line_splits
    line_splits.invalidate(conn, job)


def _review_decision_gather(conn, job):
    from . import review_decisions
    return review_decisions.gather(conn, job)


def _review_decision_inputs(conn, job):
    from . import review_decisions
    return review_decisions.inputs(conn, job)


def _review_decision_invalidates(conn, job):
    from . import review_decisions
    review_decisions.invalidate(conn, job)


def _transcription_scope(payload):
    return Scope(frozenset({'prepared_audio', 'voices', 'transcript', 'context', 'audio_ranges'}), frozenset({'transcript'}))


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


def _speaker_gather(conn, job):
    from . import speaker_attribution
    return speaker_attribution.gather(conn, job)


def _speaker_inputs(conn, job):
    from . import speaker_attribution
    return speaker_attribution.inputs(conn, job)


def _speaker_invalidates(conn, job):
    from . import speaker_attribution
    speaker_attribution.invalidate(conn, job)


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


def _review_gather(conn, job):
    from . import review_answers
    return review_answers.gather(conn, job)


def _review_inputs(conn, job):
    from . import review_answers
    return review_answers.inputs(conn, job)


def _review_invalidates(conn, job):
    from . import review_answers
    review_answers.invalidate(conn, job)


def _summary_gather(conn, job):
    from . import summary_jobs
    return summary_jobs.gather(conn, job)


def _summary_inputs(conn, job):
    from . import summary_jobs
    return summary_jobs.inputs(conn, job)


def _summary_invalidates(conn, job):
    from . import summary_jobs
    summary_jobs.invalidate(conn, job)


def _sequence_scope(payload):
    sequence_id, step = payload.get('sequence_id'), payload.get('step')
    domain = f'sequence:{sequence_id}'
    reads = {'cut_boundaries', 'prepared_audio', 'transcript', 'subtitles', 'results', 'context', 'decisions'}
    if step == 'shot':
        output = f"{domain}:shot:{payload.get('shot_key')}"
        reads.update({f'{domain}:bible', f'{domain}:plan', f'{domain}:compilation'})
        writes = {output}
    else:
        reads.add(domain)
        writes = {domain}
    return Scope(frozenset(reads), frozenset(writes), sequence_id=sequence_id,
                 previous_shot_to=payload.get('shot_key') if step == 'shot' else None)


def _compaction_gather(conn, job):
    from . import composition_compaction
    return composition_compaction.gather(conn, job)


def _compaction_inputs(conn, job):
    from . import composition_compaction
    return composition_compaction.inputs(conn, job)


def _compaction_invalidates(conn, job):
    from . import composition_compaction
    composition_compaction.invalidate(conn, job)


def _export_scope(payload):
    reads = {'original', 'prepared_audio', 'cut_boundaries', 'results', 'transcript', 'subtitles', 'detections', 'composition_cache'}
    if payload.get('kind') == 'sequence':
        sequence_id = payload.get('sequence_id') or (payload.get('sequence') or {}).get('id')
        reads.update({f'sequence:{sequence_id}', 'context', 'voices'})
        return Scope(frozenset(reads), sequence_id=sequence_id)
    span = (payload['from'] - 1, payload['to'] - 1) if 'from' in payload and 'to' in payload else None
    return Scope(frozenset(reads), cut_range=span)


def _export_gather(conn, job):
    from . import brief
    return brief.gather(conn, job)


def _export_inputs(conn, job):
    from . import brief
    return brief.inputs(conn, job)


def _export_invalidates(conn, job):
    from . import brief
    brief.invalidate(conn, job)


def _sequence_gather(conn, job):
    from . import sequence_pipeline
    return sequence_pipeline.gather(conn, job)


def _sequence_inputs(conn, job):
    from . import sequence_pipeline
    return sequence_pipeline.inputs(conn, job)


def _sequence_invalidates(conn, job):
    from . import sequence_pipeline
    sequence_pipeline.invalidate(conn, job)


def _cut_scope(payload, reads, writes, adjacent_reads=()):
    cut_id = payload.get('cut_id')
    return Scope(frozenset(reads), frozenset(writes), frozenset({cut_id}) if cut_id else None, frozenset(adjacent_reads))


def _episode_request_label(payload):
    return {'consistency': 'Consistency summaries', 'review_answers': 'Production decisions drafts', 'text_chain': 'Cut text steps',
            'cut_summaries': 'Cut summaries', 'text_processing': 'Episode text processing',
            'target_detection': 'Target-frame characters'}.get(payload.get('action'), 'Episode request')


def _episode_request_gather(conn, job):
    from . import episode_requests
    return episode_requests.gather(conn, job)


def _episode_request_inputs(conn, job):
    from . import episode_requests
    return episode_requests.inputs(conn, job)


def _episode_request_invalidates(conn, job):
    from . import episode_requests
    episode_requests.invalidate(conn, job)


KINDS = {kind.pipeline: kind for kind in (
    Kind('episode_request', 'Episode request', False, label=_episode_request_label, cancel_running=_always,
         resources=frozenset({'cpu_media'}), scope=lambda payload: Scope(frozenset({'cut_boundaries'}), frozenset({'queue'})),
         priority_class=lambda payload: 'batch', handler='episode_requests.process', inputs=_episode_request_inputs,
         gather=_episode_request_gather, invalidates=_episode_request_invalidates),
    Kind('analyze', 'Visual analysis', True, label=_analysis_label, cuts=_analysis_cuts, stage_units=True, keys=_analysis_keys,
         # Older analysis workflows stop safely at their next checkpoint only when they are text-only.
         cancel_running=lambda payload: payload.get('workflow') == 'single_cut_v2' or payload.get('text_only') is True,
         retry=_always, lane='inference', resources=frozenset({'gpu'}), scope=_visual_scope,
         needs_device=lambda payload: not (payload.get('workflow') == 'single_cut_v2' and _resident_models(payload, ('initial_frame', 'keyframe_analysis'))),
         priority_class=lambda payload: 'batch' if payload.get('batch') else 'interactive',
         inputs=analysis_context.visual_fingerprint, gather=analysis_context.visual, invalidates=_visual_invalidates),
    Kind('character_detect', 'Character detection', True, label=_character_label, cuts=_one_cut, keys=_character_keys,
         # Bulk detection keeps one lease per cut and skips a cut whose lease is gone; the older workflow cannot stop midway.
         cancel_running=lambda payload: payload.get('workflow') in ('anime_bulk_v1', 'target_frames_v1'),
         lane='inference', resources=frozenset({'gpu'}), scope=_detection_scope,
         priority_class=lambda payload: 'batch' if payload.get('episode_batch') else 'interactive',
         inputs=_detection_inputs, gather=_gather_detection, invalidates=_detection_invalidates),
    # Describes one labelled box of a character created from it and appends the text to its Details.
    Kind('character_profile', 'Character description', True, cuts=_one_cut, cancel_running=_always, retry=_always,
         keys=lambda payload, cut_ids: [f"character_profile:{payload.get('character_context_id')}"],
         lane='inference', handler='character_profile.process', resources=frozenset({'gpu'}),
         needs_device=lambda payload: not _resident_models(payload, ('character_profile',)),
         scope=lambda payload: Scope(frozenset({'original', 'cut_boundaries', 'detections', 'context'}),
                                     frozenset({f"context:{payload.get('character_context_id')}"}),
                                     frozenset({payload['cut_id']}) if payload.get('cut_id') else None),
         priority_class=lambda payload: 'interactive', inputs=_profile_inputs, gather=_profile_gather, invalidates=_profile_invalidates),
    # The timed acting plan a text model compiles from the keyframe analysis (performance_beats.py).
    Kind('performance_beats', 'Performance beats', True, cuts=_one_cut, keys=_cut_keys('performance_beats'), cancel_running=_always, retry=_always,
         lane='text', handler='performance_beats.process', resources=frozenset({'text'}),
         scope=lambda payload: Scope(frozenset({'cut_boundaries', 'results:initial_frame', 'results:keyframe_analysis', 'transcript', 'context', 'chunks'}),
                                     frozenset({'results:performance_beats'}), frozenset({payload['cut_id']}) if payload.get('cut_id') else None),
         priority_class=lambda payload: 'interactive', inputs=_beats_inputs, gather=_beats_gather, invalidates=_beats_invalidates),
    Kind('consistency_summary', 'Consistency summary', True, cuts=_one_cut, keys=_cut_keys('consistency_summary'), cancel_running=_always,
         lane='text', handler='inference_worker.process_consistency_summary', resources=frozenset({'text'}),
         scope=lambda payload: _cut_scope(payload,
             {'original', 'cut_boundaries', 'results:initial_frame', 'results:last_frame', 'results:video_analysis',
              'results:keyframe_analysis', 'results:performance_beats', 'results:consistency_summary', 'results:subtitle_ocr',
              'transcript', 'context', 'chunks', 'detections', 'target_detections', 'subtitles'}, {'results:consistency_summary'},
             {'cut_boundaries', 'results:initial_frame', 'results:keyframe_analysis', 'results:performance_beats', 'results:consistency_summary'}),
         priority_class=lambda payload: 'batch' if payload.get('batch') else 'interactive',
         inputs=_consistency_inputs, gather=_consistency_gather, invalidates=_consistency_invalidates),
    Kind('review_answers', 'Production decisions draft', True, cuts=_one_cut, keys=_cut_keys('review_answers'), cancel_running=_always,
         lane='text', handler='inference_worker.process_review_answers', resources=frozenset({'text'}),
         scope=lambda payload: _cut_scope(payload, {'cut_boundaries', 'results', 'transcript', 'subtitles', 'decisions', 'decision_drafts'},
             {'decision_drafts', 'decisions'} if payload.get('auto_save') else {'decision_drafts'},
             {'cut_boundaries', 'results:consistency_summary'}),
         priority_class=lambda payload: 'batch' if payload.get('auto_save') else 'interactive',
         inputs=_review_inputs, gather=_review_gather, invalidates=_review_invalidates),
    # The autopilot's answers to the review questions (review_decisions.py, M4): a local decider per cut, then the resolver
    # applies or escalates in the publish transaction. A shadow run writes only its own record.
    Kind('review_decisions', 'Review decisions (agent)', True, cuts=_one_cut, keys=_cut_keys('review_decisions'), cancel_running=_always, retry=_always,
         lane='text', handler='review_decisions.process', resources=frozenset({'text'}),
         scope=lambda payload: _cut_scope(payload, {'cut_boundaries', 'results', 'transcript', 'subtitles', 'context', 'detections', 'decisions'},
             set() if payload.get('shadow') else {'decisions'}, {'cut_boundaries', 'results:consistency_summary'}),
         priority_class=lambda payload: 'batch', inputs=_review_decision_inputs, gather=_review_decision_gather, invalidates=_review_decision_invalidates),
    Kind('regen', 'Cut summary', True, label=_summary_label, cuts=_one_cut, keys=_cut_keys('regen'), cancel_running=_always,
         lane='text', handler='inference_worker.process_regen', resources=frozenset({'text'}),
         scope=lambda payload: _cut_scope(payload, {'original', 'cut_boundaries', 'results', 'decisions', 'context', 'chunks', 'detections', 'transcript', 'subtitles'},
             {'results:summary'}, {'cut_boundaries', 'results:consistency_summary'}),
         priority_class=lambda payload: 'interactive', inputs=_summary_inputs, gather=_summary_gather, invalidates=_summary_invalidates),
    Kind('composition_compact', 'Composition brief', True, cuts=_one_cut, keys=_cut_keys('composition_compact'), cancel_running=_always,
         retry=_always, attention=False, lane='inference', handler='composition_compaction.process', affects_episode=False, claim_order=1, resources=frozenset({'text'}),
         scope=lambda payload: Scope(frozenset({'cut_boundaries', 'results', 'detections'}), frozenset({'composition_cache'}),
                                     frozenset({payload['cut_id']}) if payload.get('cut_id') else None),
         priority_class=lambda payload: 'batch', inputs=_compaction_inputs, gather=_compaction_gather, invalidates=_compaction_invalidates),
    Kind('sequence', 'Sequence', True, label=_sequence_label, cancel_running=_always,
         keys=lambda payload, cut_ids: [f"sequence:{payload.get('sequence_id')}:{payload.get('step')}:{payload.get('shot_key') or ''}"],
         lane='text', handler='sequence_pipeline.process', resources=frozenset({'text'}), scope=_sequence_scope,
         priority_class=lambda payload: 'interactive' if payload.get('shot_key') and not payload.get('batch') else 'batch',
         inputs=_sequence_inputs, gather=_sequence_gather, invalidates=_sequence_invalidates),
    Kind('speaker_attribution', 'Speaker attribution', True, audio_range=_transcribe_range, cancel_running=_always, retry=_always,
         keys=lambda payload, cut_ids: [_transcribe_key(payload).replace('transcribe:', 'speaker_attribution:')],
         lane='text', handler='speaker_attribution.process', resources=frozenset({'text'}),
         scope=lambda payload: Scope(frozenset({'transcript', 'subtitles', 'detections', 'context', 'cut_boundaries'}), frozenset({'transcript'})),
         priority_class=lambda payload: 'batch', inputs=_speaker_inputs, gather=_speaker_gather, invalidates=_speaker_invalidates),
    Kind('subtitles', 'Subtitle extraction', True, cancel_running=_always,
         keys=lambda payload, cut_ids: [f"subtitles:{payload.get('language') or ''}"], lane='inference', handler='subtitle_pipeline.process', resources=frozenset({'gpu'}),
         scope=lambda payload: Scope(frozenset({'original', 'proxy', 'transcript'}), frozenset({'subtitles'})),
         priority_class=lambda payload: 'batch', inputs=_subtitle_inputs, gather=_subtitle_gather, invalidates=_subtitle_invalidates),
    Kind('transcribe', 'Transcription', True, audio_range=_transcribe_range, cancel_running=_always,
         keys=lambda payload, cut_ids: [_transcribe_key(payload)], lane='audio', resources=frozenset({'gpu', 'audio'}),
         scope=_transcription_scope, priority_class=lambda payload: 'interactive' if payload.get('range') else 'batch',
         inputs=_audio_inputs, gather=_audio_gather, invalidates=_audio_invalidates),
    Kind('voice_sample', 'Voice sample', True, audio_range=_voice_range, cancel_running=_always,
         keys=lambda payload, cut_ids: [f"voice_sample:{payload.get('character_context_id')}"],
         lane='audio', resources=frozenset({'gpu', 'audio'}), scope=_voice_scope, priority_class=lambda payload: 'interactive',
         inputs=_audio_inputs, gather=_audio_gather, invalidates=_audio_invalidates),
    # Media work runs on the CPU in the API process: listed, never counted, not cancellable while it runs.
    Kind('keyframes', 'Keyframe selection', False, cuts=_keyframe_cuts, cancel_running=_always, retry=_always, keys=lambda payload, cut_ids: ['keyframes'],
         resources=frozenset({'cpu_media'}), scope=_selection_scope,
         priority_class=lambda payload: 'interactive' if len(_keyframe_cuts(payload)) == 1 else 'batch',
         inputs=_selection_inputs, gather=_selection_gather, invalidates=_selection_invalidates),
    # Hard cuts the episode-wide detection missed (missed_cuts.py, autopilot M2): CPU scene scores and picture checks;
    # it splits cuts in its publish transaction, so it writes cut boundaries like a targeted detection.
    Kind('missed_cuts', 'Missed-cut check', False, cuts=_keyframe_cuts, keys=lambda payload, cut_ids: ['missed_cuts'],
         resources=frozenset({'cpu_media'}), handler='missed_cuts.process',
         scope=lambda payload: Scope(frozenset({'proxy', 'cut_boundaries'}), frozenset({'cut_boundaries', 'previews'})),
         priority_class=lambda payload: 'batch', inputs=_missed_cut_inputs, gather=_missed_cut_gather, invalidates=_missed_cut_invalidates),
    # Transcript lines that run across a hard cut, split at a pause or trimmed (line_splits.py, autopilot M3).
    Kind('line_splits', 'Line splits at hard cuts', False, keys=lambda payload, cut_ids: ['line_splits'],
         resources=frozenset({'cpu_media'}), handler='line_splits.process',
         scope=lambda payload: Scope(frozenset({'transcript', 'cut_boundaries', 'prepared_audio'}), frozenset({'transcript'})),
         priority_class=lambda payload: 'batch', inputs=_line_split_inputs, gather=_line_split_gather, invalidates=_line_split_invalidates),
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
    Kind('brief_export', 'Download', False, label=_export_label, cuts=_export_cuts, cancel_queued=False, attention=False,
         lane='export', affects_episode=False, resources=frozenset({'cpu_media'}), scope=_export_scope, priority_class=lambda payload: 'batch',
         inputs=_export_inputs, gather=_export_gather, invalidates=_export_invalidates),
)}

def kind_of(pipeline):
    """The registry entry of a pipeline; unknown pipelines are listed under their own name and never counted."""
    kind = KINDS.get(pipeline)
    if kind:
        return kind
    return Kind(pipeline, str(pipeline or 'run').replace('_', ' ').capitalize(), False, cancel_queued=False)


GPU_PIPELINES = tuple(name for name, kind in KINDS.items() if kind.gpu)


def pipelines_for(lane):
    return tuple(name for name, kind in KINDS.items() if kind.lane == lane)
