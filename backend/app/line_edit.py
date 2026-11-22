"""Merging lines and splitting one line, as single transaction revisions of `transcript_segments`.

The caller opens `BEGIN IMMEDIATE` and passes the connection, so every revision, the superseding
of the sources and the invalidation of dependent results commit together. No function here knows
about HTTP: failures are `LineEditError` with the status the route should answer.
"""
import json
import uuid

from . import structured_results, subtitle_store
from .transcript_parts import aligned_words, join_text

MAX_MERGE = 50
COLUMNS = 'id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json'


class LineEditError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _insert(conn, source, new_id, *, start, end, text, speaker_label, character_context_id, words, speaker_source, confidence, origin='manual_edit'):
    conn.execute(f'INSERT INTO transcript_segments ({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                 (new_id, source['episode_id'], start, end, text, speaker_label, character_context_id, origin,
                  source['model_id'], source['job_id'], source['id'], json.dumps(words, ensure_ascii=False) if words else None,
                  speaker_source, confidence, source['evidence_json']))


def _squashed(text):
    return ''.join(text.split())


def _current(conn, segment_id):
    row = conn.execute('SELECT * FROM transcript_segments WHERE id=?', (segment_id,)).fetchone()
    if not row:
        raise LineEditError(404, 'Transcript segment not found')
    if row['superseded_by'] or row['deleted']:
        raise LineEditError(409, 'This transcript segment is deleted or has a newer revision')
    return row


def _invalidate(conn, episode_id, start, end):
    structured_results.invalidate_transcript(conn, episode_id, start, end)
    subtitle_store.invalidate_probes(conn, episode_id)


def merge(conn, episode_id, segment_ids, text, speaker_label, character_context_id):
    """One line spanning the sources, with the text the user settled on. Returns the new row's id."""
    ids = list(dict.fromkeys(segment_ids))
    if not 2 <= len(ids) <= MAX_MERGE:
        raise LineEditError(400, f'Merge between 2 and {MAX_MERGE} lines')
    rows = [_current(conn, segment_id) for segment_id in ids]
    if any(row['episode_id'] != episode_id for row in rows):
        raise LineEditError(404, 'Transcript segment not found')
    rows.sort(key=lambda row: (row['start_time'], row['id']))
    if not text.strip():
        raise LineEditError(400, 'The merged line needs text')
    joined = ''
    for row in rows:
        joined = join_text(joined, row['text'].strip()) if joined else row['text'].strip()
    words = [word for row in rows for word in (aligned_words(row) or [])] if all(aligned_words(row) for row in rows) and _squashed(text) == _squashed(joined) else None
    same_speaker = all((row['speaker_label'], row['character_context_id']) == (speaker_label, character_context_id) for row in rows)
    first = rows[0]
    start, end = min(row['start_time'] for row in rows), max(row['end_time'] for row in rows)
    new_id = str(uuid.uuid4())
    _insert(conn, first, new_id, start=start, end=end, text=text.strip(), speaker_label=speaker_label, character_context_id=character_context_id,
            words=words, speaker_source=first['speaker_source'] if same_speaker else 'manual', confidence=first['confidence'] if same_speaker else None)
    for row in rows:
        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, row['id']))
    _invalidate(conn, episode_id, start, end)
    return new_id


def split(conn, segment_id, at, left_text, right_text, origin='manual_edit'):
    """Two lines meeting at `at`; the original is superseded by the left one. Returns (left id, right id).

    ``origin`` is the revisions' origin: a person's split is ``manual_edit``; the autopilot's line splitter keeps the
    source line's own origin (``line_splits.py``)."""
    row = _current(conn, segment_id)
    if not row['start_time'] < at < row['end_time']:
        raise LineEditError(400, 'Split inside the line')
    left_text, right_text = left_text.strip(), right_text.strip()
    if not left_text or not right_text:
        raise LineEditError(400, 'Both parts of the split need text')
    words = aligned_words(row)
    left_words = right_words = None
    if words and _squashed(left_text + right_text) == _squashed(row['text']):
        before = [word for word in words if (word[0] + word[1]) / 2 < at]
        after = [word for word in words if (word[0] + word[1]) / 2 >= at]
        if before and after:
            left_words, right_words = before, after
    left_id, right_id = str(uuid.uuid4()), str(uuid.uuid4())
    common = dict(speaker_label=row['speaker_label'], character_context_id=row['character_context_id'], speaker_source=row['speaker_source'],
                  confidence=row['confidence'], origin=origin)
    _insert(conn, row, left_id, start=row['start_time'], end=at, text=left_text, words=left_words, **common)
    _insert(conn, row, right_id, start=at, end=row['end_time'], text=right_text, words=right_words, **common)
    conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (left_id, row['id']))
    _invalidate(conn, row['episode_id'], row['start_time'], row['end_time'])
    return left_id, right_id
