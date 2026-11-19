"""Cross-process revision counters for the episode event stream."""
import asyncio
import json
import time

from . import db

# Tables written by the API, the media worker and the model workers. List the
# mutable public columns for rows whose private lease is renewed frequently.
EPISODE_TABLES = {
    'episodes': ('id', 'data', 'title,duration,width,height,has_audio,settings_json,subtitle_language'),
    'jobs': ('episode_id', 'jobs', 'status,stage,error,dismissed_at,lease_owner,not_before,payload_json'),
    'cuts': ('episode_id', 'data', None),
    'chunks': ('episode_id', 'data', None),
    'cut_edits': ('episode_id', 'data', None),
    'transcript_segments': ('episode_id', 'data', None),
    'subtitle_runs': ('episode_id', 'data', 'status,language,band_json,error'),
    'subtitle_lines': ('episode_id', 'data', None),
    'sequences': ('episode_id', 'data', None),
    'sequence_results': ('(SELECT episode_id FROM sequences WHERE id={row}.sequence_id)', 'data', None),
    'analysis_results': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', 'title,content,superseded_by'),
    'analysis_job_stages': ('(SELECT episode_id FROM jobs WHERE id={row}.job_id)', 'jobs', 'status,label,error'),
    'analysis_job_cuts': ('(SELECT episode_id FROM jobs WHERE id={row}.job_id)', 'jobs', 'completed'),
    'cut_brief_instructions': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', None),
    'cut_review_drafts': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', None),
    'character_detection_runs': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', None),
    'character_detections': ('''(SELECT c.episode_id FROM character_detection_runs r
        JOIN cuts c ON c.id=r.cut_id WHERE r.id={row}.run_id)''', 'data', None),
    'target_detection_runs': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', None),
    'target_slots': ('''(SELECT c.episode_id FROM target_detection_runs r
        JOIN cuts c ON c.id=r.cut_id WHERE r.id={row}.run_id)''', 'data', None),
    'target_detections': ('''(SELECT c.episode_id FROM target_detection_runs r
        JOIN cuts c ON c.id=r.cut_id WHERE r.id={row}.run_id)''', 'data', None),
    'target_cast_settings': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', None),
    'keyframe_results': ('(SELECT episode_id FROM cuts WHERE id={row}.cut_id)', 'data', None),
    'character_voice_samples': ('episode_id', 'data', None),
}
GLOBAL_TABLES = {
    'episodes': 'status,stage,error',
    'projects': 'title',
    'jobs': 'status,lease_owner',
    'assistant_turns': 'status,stage,cancel_requested',
    'chat_gateway_requests': 'status,stage,cancel_requested',
    'local_text_model_state': 'state,detail,desired_loaded,variant,variants_json',
    'text_generation_settings': None,
    'visual_model_settings': None,
    'visual_model_state': 'models_json',
    'remote_text_state': 'state,detail,models_json,active_jobs',
    'remote_text_server': None,
    'assistant_worker_state': 'providers_json',
}


def _trigger(conn, name, table, action, columns, body):
    when = ''
    if action == 'UPDATE' and columns:
        fields = columns.split(',')
        action += ' OF ' + columns
        when = 'WHEN ' + ' OR '.join(f'NEW.{field} IS NOT OLD.{field}' for field in fields)
    conn.execute(f'CREATE TRIGGER IF NOT EXISTS {name} AFTER {action} ON {table} {when} BEGIN {body}; END')


def _bump_local(field, where):
    return (f'INSERT INTO episode_event_revisions(episode_id,{field}) SELECT id,1 FROM episodes WHERE {where} '
            f'ON CONFLICT(episode_id) DO UPDATE SET {field}={field}+1')


def init_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episode_event_revisions (
            episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
            jobs INTEGER NOT NULL DEFAULT 0, data INTEGER NOT NULL DEFAULT 0, stale INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS queue_event_revision (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);
        INSERT OR IGNORE INTO queue_event_revision VALUES (1,0);
    """)
    for table, (episode, field, columns) in EPISODE_TABLES.items():
        for action, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
            source = episode.format(row=row) if '{row}' in episode else f'{row}.{episode}'
            _trigger(conn, f'episode_events_{table}_{action}', table, action, columns, _bump_local(field, f'id={source}'))
    for action, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
        _trigger(conn, f'episode_events_job_lifecycle_{action}', 'jobs', action, 'status',
                 _bump_local('data', f'id={row}.episode_id'))
    _trigger(conn, 'episode_events_stale', 'analysis_results', 'UPDATE', 'stale_reason',
             _bump_local('stale', 'id=(SELECT episode_id FROM cuts WHERE id=NEW.cut_id)'))
    _trigger(conn, 'episode_events_subtitle_stale', 'subtitle_runs', 'UPDATE', 'stale_reason',
             _bump_local('stale', 'id=NEW.episode_id'))
    for action, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
        _trigger(conn, f'episode_events_context_{action}', 'context_items', action, None,
                 _bump_local('data', f'project_id={row}.project_id AND ({row}.episode_id IS NULL OR id={row}.episode_id)'))
    for table, columns in GLOBAL_TABLES.items():
        for action in ('INSERT', 'UPDATE', 'DELETE'):
            _trigger(conn, f'queue_events_{table}_{action}', table, action, columns, 'UPDATE queue_event_revision SET version=version+1 WHERE id=1')


def snapshot(episode_id):
    with db.connect() as conn:
        conn.execute('BEGIN')
        host = conn.execute("""SELECT version,
            (SELECT heartbeat FROM local_text_model_state WHERE id=1),
            (SELECT heartbeat FROM remote_text_state WHERE id=1),
            (SELECT heartbeat FROM assistant_worker_state WHERE id=1),
            (SELECT updated_at FROM visual_model_state WHERE id=1)
            FROM queue_event_revision WHERE id=1""").fetchone()
        now = time.time()
        online = sum(1 << index for index, (seen, ttl) in enumerate(zip(tuple(host)[1:], (15, 30, 30, 180)))
                     if seen and now - seen <= ttl)
        queue = f'{host[0]}.{online}'
        if episode_id is None:
            return {'version': str(queue), 'queue': queue}
        row = conn.execute("""SELECT COALESCE(r.jobs,0) AS jobs,COALESCE(r.data,0) AS data,COALESCE(r.stale,0) AS stale
            FROM episodes e LEFT JOIN episode_event_revisions r ON r.episode_id=e.id WHERE e.id=?""", (episode_id,)).fetchone()
        if not row:
            return None
        return {'version': f"{queue}.{row['jobs']}.{row['data']}.{row['stale']}", 'queue': queue, **dict(row)}


async def stream(request, episode_id, previous):
    last_heartbeat = asyncio.get_running_loop().time()
    while not await request.is_disconnected():
        state = await asyncio.to_thread(snapshot, episode_id)
        if state is None:
            break
        if state['version'] != previous:
            previous = state['version']
            yield f"id: {previous}\nevent: changed\ndata: {json.dumps(state, separators=(',', ':'))}\n\n"
            last_heartbeat = asyncio.get_running_loop().time()
        elif asyncio.get_running_loop().time() - last_heartbeat >= 15:
            yield ': keepalive\n\n'
            last_heartbeat = asyncio.get_running_loop().time()
        await asyncio.sleep(.65)
