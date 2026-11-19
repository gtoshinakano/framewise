"""Storage for sequences and their revisioned results.

A sequence is a contiguous range of an episode's active cuts (at most three minutes) that is directed
as one short video on the original audio. Its results (``bible``, ``plan``, one ``shot`` per planned
shot and ``review``) are revision chains like ``analysis_results``: ``parent_id`` points back,
``superseded_by`` forward, and the current revision is the one that nothing supersedes. Results have
no approval status. Nothing here imports FastAPI: the inference worker uses it too.
"""
import json
import uuid

from . import db, media, prompt_context, sequence_compile, sequence_timeline, subtitle_store
from .assistant.transcript_edit import merged_transcript

TYPES = ('compilation', 'bible', 'plan', 'shot', 'review')
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


def next_cut(conn, sequence, cuts):
    """The first active cut after the sequence's last cut, or None at the end of the episode."""
    row = conn.execute('SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index>? ORDER BY cut_index LIMIT 1', (sequence['episode_id'], cuts[-1]['cut_index'])).fetchone()
    return dict(row) if row else None


def extension(sequence):
    """The cut the user added to the range and is still waiting for the direction to be redone: ``{cut_id, cut, from_time, rewrite_shots}`` or None."""
    found = loads(sequence.get('extension_json'), {})
    return found or None


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
    """Event start times (seconds from the cut's start) of cuts too long for one shot, from their motion (keyframes, else video)."""
    events = {}
    for cut in cuts:
        if cut['end_time'] - cut['start_time'] <= sequence_timeline.MAX_SHOT_SECONDS:
            continue
        motion = prompt_context.current_motion(conn, cut['id'])  # the keyframe analysis, else an older video analysis
        view = prompt_context.video_view(motion['content']) if motion else {}
        starts = []
        for event in (view.get('events') or []) if isinstance(view, dict) else []:
            try:
                starts.append(float(str(event.get('seconds', '')).split('-')[0]))
            except (ValueError, AttributeError):
                continue
        events[cut['id']] = starts
    return events


def source_fingerprint(conn, sequence, cuts):
    """``(hash, transcript fragments, subtitle rows)`` of the range: what a dialogue compilation is built from."""
    start, end = cuts[0]['start_time'], cuts[-1]['end_time']
    fragments = [dict(row) for row in conn.execute("""SELECT id,start_time,end_time,text,speaker_label,character_context_id FROM transcript_segments
        WHERE episode_id=? AND superseded_by IS NULL AND deleted=0 AND start_time<? AND end_time>? ORDER BY start_time,id""", (sequence['episode_id'], end, start))]
    subtitles = subtitle_rows(conn, sequence['episode_id'], start, end)
    return sequence_compile.source_hash(fragments, subtitles), fragments, subtitles


def subtitle_rows(conn, episode_id, start, end):
    """Subtitle lines of the current run overlapping ``[start, end)`` as rows with episode times."""
    return [{'id': line['id'], 'start_time': line['start_time'], 'end_time': line['end_time'], 'text': line['text']}
            for line in subtitle_store.lines(episode_id, start, end, conn=conn)]


def reviewed_compilation(conn, sequence):
    """The dialogue compilation the user confirmed, when it is still the current revision; otherwise None."""
    if not sequence.get('reviewed_compilation_id'):
        return None
    row = current(conn, sequence['id'], 'compilation')
    if not row or row['id'] != sequence['reviewed_compilation_id']:
        return None
    return {**loads(row['content'], {}), 'origin': row['origin']}


def shot_points(sequence):
    """The shot boundaries the user placed on the Dialogue timeline: ``[{after_cut_id, time}]`` in episode seconds."""
    return [item for item in loads(sequence.get('shot_points_json'), []) if isinstance(item, dict)]


def timeline_for(conn, sequence, cuts, points=None):
    """The sequence clock, speech lines, pauses and candidate boundaries for the given current cuts.

    Once the user has confirmed a dialogue compilation its lines replace the raw transcript everywhere.
    The boundaries the user placed (``points``, else the sequence's saved ones) are marked ``fixed``.
    """
    compiled = reviewed_compilation(conn, sequence)
    return build_timeline(conn, sequence, cuts, compiled, points)


def audio_envelope(conn, sequence, cuts):
    episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (sequence['episode_id'],)).fetchone()
    start, end = cuts[0]['start_time'], cuts[-1]['end_time']
    audio = media.episode_dir(episode['project_id'], sequence['episode_id']) / 'audio.wav'
    return sequence_timeline.audio_envelope(audio, start, end) if audio.is_file() else []


def build_timeline(conn, sequence, cuts, compilation, points=None, *, envelope=None, events=None):
    start, end = cuts[0]['start_time'], cuts[-1]['end_time']
    if envelope is None:
        envelope = audio_envelope(conn, sequence, cuts)
    if events is None:
        events = event_starts(conn, cuts)
    lines = sequence_compile.merged_lines(compilation, start) if compilation is not None else transcript_lines(conn, sequence['episode_id'], start, end)
    timeline = sequence_timeline.build(cuts, lines, envelope, events, shot_points(sequence) if points is None else points)
    timing = {item['after_cut_id']: item for item in (compilation or {}).get('timing', [])}
    accepted = {}
    for boundary in timeline['boundaries']:
        item = timing.get(boundary['after_cut_id'])
        if not item or boundary['fixed'] or not boundary['speech'] or item['choice'] not in boundary['options']:
            continue
        key = boundary['id']
        trial = {**accepted, key: {**item, 'id': key}}
        spans = sequence_timeline.decide(timeline, {key: item['choice'] for key, item in trial.items()})
        if not sequence_timeline.check_spans(spans, timeline):
            accepted = trial
    timeline['dialogue_timing'] = accepted
    return timeline


def speech_past_end(conn, sequence, cuts, following):
    """The first spoken line that runs on past the sequence's end into ``following``: ``{text_before, words_after}`` or None.

    Uses the rule of ``sequence_timeline.speech_lines``: a word belongs to the side its midpoint lies on, so the
    words after the end are the ones the sequence leaves out.
    """
    end = cuts[-1]['end_time']
    for line in transcript_lines(conn, sequence['episode_id'], max(cuts[0]['start_time'], end - 30), following['end_time']):
        before = [part for part in line['parts'] if (part['start_time'] + part['end_time']) / 2 < end]
        after = [part for part in line['parts'] if (part['start_time'] + part['end_time']) / 2 >= end]
        if before and after:
            return {'text_before': sequence_timeline.words([{'text': part['text']} for part in before]),
                    'words_after': sequence_timeline.words([{'text': part['text']} for part in after])}
    return None


def raw_timeline_for(conn, sequence, cuts, utterances=False):
    """The same timeline built from the raw transcript, which is what the dialogue compilation starts from."""
    start, end = cuts[0]['start_time'], cuts[-1]['end_time']
    lines = transcript_lines(conn, sequence['episode_id'], start, end)
    if utterances:
        lines = sequence_compile.source_utterances(lines, [cut['start_time'] for cut in cuts[1:]])
    return sequence_timeline.build(cuts, lines)


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
    if not job_id and result_type != 'review':
        invalidate_review(conn, sequence_id)
    if origin != 'model' and not job_id:
        step = 'compile' if result_type == 'compilation' else result_type
        conn.execute("""UPDATE jobs SET input_json=json_set(COALESCE(input_json,'{}'),'$.keep_output',json('true'))
            WHERE pipeline='sequence' AND status IN ('queued','running') AND json_extract(payload_json,'$.sequence_id')=?
            AND json_extract(payload_json,'$.step')=? AND COALESCE(json_extract(payload_json,'$.shot_key'),'')=?""",
            (sequence_id, step, shot_key or ''))
    conn.execute('UPDATE sequences SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (sequence_id,))
    return result_id


def invalidate_review(conn, sequence_id):
    conn.execute("""UPDATE sequence_results SET stale_reason='Sequence evidence changed'
        WHERE sequence_id=? AND type='review' AND superseded_by IS NULL""", (sequence_id,))


def record_sources(conn, sequence_id, result_id, source_hash, subtitle_run_id, transcript_count, subtitle_count):
    """Remember what the current compilation was built from, to tell later when the transcript or subtitles moved on."""
    conn.execute("""INSERT INTO sequence_compilation_sources(sequence_id,result_id,source_hash,subtitle_run_id,transcript_count,subtitle_count) VALUES (?,?,?,?,?,?)
        ON CONFLICT(sequence_id) DO UPDATE SET result_id=excluded.result_id,source_hash=excluded.source_hash,subtitle_run_id=excluded.subtitle_run_id,
        transcript_count=excluded.transcript_count,subtitle_count=excluded.subtitle_count,created_at=CURRENT_TIMESTAMP""",
                 (sequence_id, result_id, source_hash, subtitle_run_id, transcript_count, subtitle_count))


MAX_REMEMBERED = 12
REMEMBERED_CHARS = 6000


def shot_instructions(conn, sequence_id, shot_key):
    """What the user asked for in earlier rewrites of one shot, oldest first: instructions that every later rewrite keeps following.

    Read from the shot's revisions, so there is nothing else to keep in step. Only the user's own instructions count
    (``instruction_source`` ``user``): a suggestion of the automatic review does not become a standing order, and a
    revision from before the source was recorded is not counted. A repeated text keeps its latest place, and the oldest go
    first when there are more than ``MAX_REMEMBERED`` or ``REMEMBERED_CHARS``.
    """
    found = []
    for row in conn.execute("SELECT settings_json FROM sequence_results WHERE sequence_id=? AND type='shot' AND shot_key=? ORDER BY rowid", (sequence_id, shot_key or '')):
        settings = loads(row['settings_json'], {})
        text = settings.get('instruction')
        if settings.get('instruction_source') == 'user' and isinstance(text, str) and text.strip():
            text = text.strip()
            found = [item for item in found if item.casefold() != text.casefold()] + [text]
    found = found[-MAX_REMEMBERED:]
    while len(found) > 1 and sum(len(item) for item in found) > REMEMBERED_CHARS:
        found.pop(0)
    return found


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
    from . import worker
    return worker.insert_job(conn, sequence['episode_id'], 'sequence', {'sequence_id': sequence['id'], **payload})
