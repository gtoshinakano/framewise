"""Snapshot and enqueue a detached single-cut consistency summary."""
import json
import uuid
from . import consistency_summary, prompt_context


def missing_evidence(by_type, pending_beats=False):
    """Why a cut cannot be reviewed yet, or None.

    A cut needs its first frame and its motion: the keyframe analysis with performance beats built from it (since
    2026-11-18), or for a cut analysed before that only an older video analysis. ``pending_beats`` says a chained beats
    job will write them before the summary runs. A cut with keyframes never falls back to an older video analysis.
    """
    if (by_type.get('initial_frame') or {}).get('content') is None:
        return 'Initial frame and keyframe analysis results are required'
    if 'keyframe_analysis' in by_type:
        if pending_beats or prompt_context.beats_current(by_type):
            return None
        return 'The performance beats of this cut are missing or older than its keyframe analysis; write them first'
    video = (by_type.get('video_analysis') or {}).get('content')
    if video is None:
        return 'Initial frame and keyframe analysis results are required'
    if not prompt_context.is_direction(video) and (by_type.get('last_frame') or {}).get('content') is None:
        # A video analysis from before motion directions only describes the cut together with its last frame.
        return 'This cut has an older video analysis without a last frame; run Keyframes'
    return None


def source_ids(results):
    """The results a consistency summary is written from: not itself, and not what a newer motion source replaced."""
    by_type = {item['type']: item for item in results}
    keyframes = 'keyframe_analysis' in by_type
    current_flow = keyframes or prompt_context.is_direction((by_type.get('video_analysis') or {}).get('content'))
    return [item['id'] for item in results if item['type'] != 'consistency_summary'
            and not (item['type'] == 'last_frame' and current_flow) and not (item['type'] == 'video_analysis' and keyframes)]


def payload_for(cut_id, config, context, results, pending_beats=False):
    by_type = {item['type']: item for item in results}
    problem = missing_evidence(by_type, pending_beats)
    if problem:
        raise ValueError(problem)
    evidence = consistency_summary.build_input(results, context['transcript_segments'], context['context_items'], context.get('subtitles'), prompt_context.series_context(context))
    if context.get('character_detections'):
        evidence['character_detections'] = prompt_context.slim_detections(context['character_detections'])
    payload = {'cut_id': cut_id, 'context': {'cut': prompt_context.slim_cut(context['cut'])}, 'config': config,
               'evidence': evidence, 'source_result_ids': source_ids(results), 'base_id': by_type.get('consistency_summary', {}).get('id')}
    if pending_beats:
        # The evidence is built again when the job starts, from the beats the chain wrote by then (consistency_summary.refresh).
        payload['refresh_evidence'] = True
        payload['refresh'] = consistency_summary.refresh_inputs(context)
    return payload


def insert(conn, episode_id, payload):
    if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='consistency_summary' AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?", (episode_id, payload['cut_id'])).fetchone():
        raise ValueError('A consistency summary is already queued for this cut')
    job_id = str(uuid.uuid4())
    conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, episode_id, 'consistency_summary', json.dumps(payload, ensure_ascii=False)))
    return job_id
