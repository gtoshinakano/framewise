"""Snapshot and enqueue a detached single-cut consistency summary."""
import json
import uuid
from . import consistency_summary, prompt_context


def payload_for(cut_id, config, context, results):
    by_type = {item['type']: item for item in results}
    if not all(kind in by_type and by_type[kind]['content'] is not None for kind in ('initial_frame', 'video_analysis')):
        raise ValueError('Initial frame and video analysis results are required')
    if not prompt_context.is_direction(by_type['video_analysis']['content']) and (by_type.get('last_frame') or {}).get('content') is None:
        # A video analysis from before motion directions only describes the cut together with its last frame.
        raise ValueError('This cut has an older video analysis without a last frame; run the video analysis again')
    evidence = consistency_summary.build_input(results, context['transcript_segments'], context['context_items'], context.get('subtitles'))
    if context.get('character_detections'):
        evidence['character_detections'] = prompt_context.slim_detections(context['character_detections'])
    return {'cut_id': cut_id, 'context': {'cut': prompt_context.slim_cut(context['cut'])}, 'config': config,
            'evidence': evidence,
            'source_result_ids': [item['id'] for item in results if item['type'] != 'consistency_summary'
                                  and not (item['type'] == 'last_frame' and prompt_context.is_direction(by_type['video_analysis']['content']))],
            'base_id': by_type.get('consistency_summary', {}).get('id')}


def insert(conn, episode_id, payload):
    if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='consistency_summary' AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?", (episode_id, payload['cut_id'])).fetchone():
        raise ValueError('A consistency summary is already queued for this cut')
    job_id = str(uuid.uuid4())
    conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, episode_id, 'consistency_summary', json.dumps(payload, ensure_ascii=False)))
    return job_id
