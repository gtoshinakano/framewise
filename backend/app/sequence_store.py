"""Storage for sequences and their revisioned results.

A sequence is a contiguous range of an episode's active cuts (at most three minutes) that is directed
as one short video on the original audio. Its results (``bible``, ``plan``, one ``shot`` per planned
shot and ``review``) are revision chains like ``analysis_results``: ``parent_id`` points back,
``superseded_by`` forward, and the current revision is the one that nothing supersedes. Results have
no approval status. Nothing here imports FastAPI: the inference worker uses it too.
"""
import json
import uuid

from . import db, media, prompt_context, sequence_timeline
from .assistant.transcript_edit import merged_transcript

TYPES = ('bible', 'plan', 'shot', 'review')
MAX_CONTENT = 2_000_000


def loads(value, default):
    try:
        parsed = json.loads(value) if value else default
    except (TypeError, ValueError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


# ------------------------------------------------------------------ sequences

def get(conn, sequence_id):
    row = conn.execute('SELECT * FROM sequences WHERE id=?', (sequence_id,)).fetchone()
    return dict(row) if row else None


def range_cuts(conn, episode_id, first_index, last_index):
    """Active cuts of an episode between two zero-based indices, in order."""
    return [dict(row) for row in conn.execute('SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index BETWEEN ? AND ? ORDER BY cut_index', (episode_id, first_index, last_index))]


def snapshot(cuts):
    return [{'id': cut['id'], 'cut_index': cut['cut_index'], 'start_time': cut['start_time'], 'end_time': cut['end_time']} for cut in cuts]


def current_cuts(conn, sequence):
    """The active cuts the sequence spans now, or None when its first or last cut was archived."""
    first = conn.execute('SELECT cut_index,active FROM cuts WHERE id=? AND episode_id=?', (sequence['first_cut_id'], sequence['episode_id'])).fetchone()
    last = conn.execute('SELECT cut_index,active FROM cuts WHERE id=? AND episode_id=?', (sequence['last_cut_id'], sequence['episode_id'])).fetchone()
    if not first or not last or not first['active'] or not last['active'] or last['cut_index'] < first['cut_index']:
        return None
    return range_cuts(conn, sequence['episode_id'], first['cut_index'], last['cut_index'])


def cuts_changed(sequence, cuts):
    """True when the cuts were re-detected, moved, merged or split since the sequence last snapshotted them."""
    return cuts is None or snapshot(cuts) != loads(sequence['cuts_json'], [])


def resnapshot(conn, sequence, cuts):
    """Adopt the current cuts as the sequence's range (done when a new direction is queued)."""
    conn.execute('UPDATE sequences SET start_time=?,end_time=?,cuts_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                 (cuts[0]['start_time'], cuts[-1]['end_time'], json.dumps(snapshot(cuts)), sequence['id']))
    return {**sequence, 'start_time': cuts[0]['start_time'], 'end_time': cuts[-1]['end_time'], 'cuts_json': json.dumps(snapshot(cuts))}


# ------------------------------------------------------------------- timeline

def transcript_lines(conn, episode_id, start, end):
    """Current spoken lines overlapping ``[start, end)``, merged from word fragments, each with its fragments in ``parts``."""
    rows = [dict(row) for row in conn.execute('''SELECT t.id,t.start_time,t.end_time,t.text,t.speaker_label,t.origin,t.character_context_id,c.title AS character_name
        FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id
        WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>? ORDER BY t.start_time,t.id''', (episode_id, end, start))]
    by_id = {row['id']: row for row in rows}
    lines = merged_transcript([row for row in rows if str(row['text'] or '').strip()])
    for line in lines:
        line['parts'] = [by_id[fragment] for fragment in line['ids']]
    return lines


def event_starts(conn, cuts):
    """Event start times (seconds from the cut's start) of cuts too long for one shot, from their motion direction."""
    events = {}
    for cut in cuts:
        if cut['end_time'] - cut['start_time'] <= sequence_timeline.MAX_SHOT_SECONDS:
            continue
        row = conn.execute("SELECT content FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL AND json_extract(settings_json,'$.structured_type')='video_analysis' ORDER BY rowid DESC LIMIT 1", (cut['id'],)).fetchone()
        view = prompt_context.video_view(loads(row['content'], {})) if row else {}
        starts = []
        for event in (view.get('events') or []) if isinstance(view, dict) else []:
            try:
                starts.append(float(str(event.get('seconds', '')).split('-')[0]))
            except (ValueError, AttributeError):
                continue
        events[cut['id']] = starts
    return events


def timeline_for(conn, sequence, cuts):
    """The sequence clock, speech lines, pauses and candidate boundaries for the given current cuts."""
    episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (sequence['episode_id'],)).fetchone()
    start, end = cuts[0]['start_time'], cuts[-1]['end_time']
    audio = media.episode_dir(episode['project_id'], sequence['episode_id']) / 'audio.wav'
    envelope = sequence_timeline.audio_envelope(audio, start, end) if audio.is_file() else []
    return sequence_timeline.build(cuts, transcript_lines(conn, sequence['episode_id'], start, end), envelope, event_starts(conn, cuts))


# -------------------------------------------------------------------- results

def serialize(row):
    row = dict(row)
    row['content'] = loads(row['content'], {})
    row['settings'] = loads(row.pop('settings_json', None), {})
    row.pop('context_json', None)
    row.pop('prompt', None)
    return row


def current(conn, sequence_id, result_type, shot_key=''):
    row = conn.execute('SELECT * FROM sequence_results WHERE sequence_id=? AND type=? AND shot_key=? AND superseded_by IS NULL ORDER BY rowid DESC LIMIT 1',
                       (sequence_id, result_type, shot_key or '')).fetchone()
    return dict(row) if row else None


def current_shots(conn, sequence_id):
    """Current shot results by shot key."""
    return {row['shot_key']: dict(row) for row in conn.execute("SELECT * FROM sequence_results WHERE sequence_id=? AND type='shot' AND superseded_by IS NULL ORDER BY rowid", (sequence_id,))}


def save(conn, sequence_id, result_type, content, *, shot_key='', rendered=None, origin='model', model_id=None, job_id=None, prompt=None,
         settings=None, context=None, expected_current=...):
    """Append a revision inside the caller's transaction and return its id.

    ``expected_current`` is the id the writer started from (None when there was no result yet); a
    different current revision means someone else wrote in between, and nothing is saved.
    """
    if result_type not in TYPES or not isinstance(content, dict):
        raise ValueError('Invalid sequence result')
    body = json.dumps(content, ensure_ascii=False, allow_nan=False)
    if len(body) > MAX_CONTENT:
        raise ValueError('Sequence result is too large')
    base = current(conn, sequence_id, result_type, shot_key)
    if expected_current is not ... and (base['id'] if base else None) != expected_current:
        raise RuntimeError('The result changed while this was being written; run it again')
    result_id = str(uuid.uuid4())
    conn.execute('''INSERT INTO sequence_results(id,sequence_id,type,shot_key,content,rendered,origin,model_id,job_id,prompt,settings_json,context_json,parent_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                 (result_id, sequence_id, result_type, shot_key or '', body, rendered, origin, model_id, job_id, prompt,
                  json.dumps(settings or {}, ensure_ascii=False), json.dumps(context or {}, ensure_ascii=False), base['id'] if base else None))
    if base:
        conn.execute('UPDATE sequence_results SET superseded_by=? WHERE id=?', (result_id, base['id']))
    conn.execute('UPDATE sequences SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (sequence_id,))
    return result_id


def history(conn, sequence_id, result_type, shot_key=''):
    return [serialize(row) for row in conn.execute('SELECT * FROM sequence_results WHERE sequence_id=? AND type=? AND shot_key=? ORDER BY rowid DESC', (sequence_id, result_type, shot_key or ''))]


# ----------------------------------------------------------------------- jobs

def pending_jobs(conn, sequence_id):
    """Queued or running jobs of this sequence with their step and shot."""
    rows = conn.execute("""SELECT id,status,stage,error,created_at,json_extract(payload_json,'$.step') AS step,json_extract(payload_json,'$.shot_key') AS shot_key
        FROM jobs WHERE pipeline='sequence' AND status IN ('queued','running') AND json_extract(payload_json,'$.sequence_id')=? ORDER BY rowid""", (sequence_id,))
    return [dict(row) for row in rows]


def recent_jobs(conn, sequence_id, limit=60):
    rows = conn.execute("""SELECT id,status,stage,error,created_at,started_at,finished_at,json_extract(payload_json,'$.step') AS step,json_extract(payload_json,'$.shot_key') AS shot_key
        FROM jobs WHERE pipeline='sequence' AND json_extract(payload_json,'$.sequence_id')=? ORDER BY rowid DESC LIMIT ?""", (sequence_id, limit))
    return [dict(row) for row in rows]


def insert_job(conn, sequence, payload):
    job_id = str(uuid.uuid4())
    conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,'sequence',?)", (job_id, sequence['episode_id'], json.dumps({'sequence_id': sequence['id'], **payload}, ensure_ascii=False)))
    return job_id
