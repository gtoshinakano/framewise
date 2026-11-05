"""Snapshot and enqueue a detached single-cut consistency summary."""
import json
import uuid
from . import consistency_summary


def payload_for(cut_id, config, context, results):
    by_type = {item['type']: item for item in results}
    if not all(kind in by_type and by_type[kind]['content'] is not None for kind in ('initial_frame', 'video_analysis', 'last_frame')):
        raise ValueError('Initial frame, video, and last frame results are required')
    return {'cut_id': cut_id, 'context': {'cut': context['cut']}, 'config': config,
            'evidence': consistency_summary.build_input(results, context['transcript_segments'], context['context_items']),
            'source_result_ids': [item['id'] for item in results if item['type'] != 'consistency_summary'],
            'base_id': by_type.get('consistency_summary', {}).get('id')}


def insert(conn, episode_id, payload):
    if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='consistency_summary' AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?", (episode_id, payload['cut_id'])).fetchone():
        raise ValueError('A consistency summary is already queued for this cut')
    job_id = str(uuid.uuid4())
    conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)", (job_id, episode_id, 'consistency_summary', json.dumps(payload, ensure_ascii=False)))
    return job_id
