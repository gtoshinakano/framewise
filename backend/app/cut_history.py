"""Undo and redo for edits of an episode's cut boundaries.

Every edit that changes which cuts are active or where they start and end (split, merge, delete,
timing, a one-frame boundary move, hard cuts found inside one cut) stores the active cuts before and
after it in `cut_edits`, in the same transaction as the edit. Undo puts the "before" state back and
redo the "after" state, but only while the cuts look exactly like the other side, so a change made
since is never overwritten. A full redetection replaces every cut and ends the history.

Cut rows are never deleted to undo something, with one exception: a cut the edit itself created
(the second half of a split) is removed again when nothing refers to it and it has no notes. Every
other cut that leaves the timeline is archived like a merge would, so no result history is lost.
"""
import json
import uuid

KEEP = 50
TOLERANCE = 1e-6
# Watermark rows the composition triggers add for every new cut. They hold no user work.
BOOKKEEPING_TABLES = {'composition_changes'}


class Conflict(Exception):
    """The cuts changed since the edit, or there is nothing to undo or redo."""


class Busy(Exception):
    """A media job is changing cuts or their preview frames right now."""


def timecode(seconds: float) -> str:
    total = round(max(0.0, seconds) * 1000)
    whole, ms = divmod(total, 1000)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}.{ms:03d}' if hours else f'{minutes}:{secs:02d}.{ms:03d}'


def snapshot(conn, episode_id: str) -> list[dict]:
    """The episode's active cuts in timeline order."""
    return [{'id': row['id'], 'cut_index': row['cut_index'], 'start_time': row['start_time'], 'end_time': row['end_time'],
             'analysis_enabled': row['analysis_enabled']}
            for row in conn.execute('SELECT id,cut_index,start_time,end_time,analysis_enabled FROM cuts WHERE episode_id=? AND active=1 ORDER BY cut_index', (episode_id,))]


def same(left: list[dict], right: list[dict]) -> bool:
    """Same cuts at the same places. The analysis switch is not part of the timeline, so it is ignored."""
    return len(left) == len(right) and all(
        a['id'] == b['id'] and a['cut_index'] == b['cut_index'] and abs(a['start_time'] - b['start_time']) <= TOLERANCE and abs(a['end_time'] - b['end_time']) <= TOLERANCE
        for a, b in zip(left, right))


def record(conn, episode_id: str, action: str, label: str, cut_id: str | None, before: list[dict], created_ids=()) -> str | None:
    """Store an edit that just happened in this transaction. A new edit ends the redo branch."""
    after = snapshot(conn, episode_id)
    if same(before, after):
        return None
    conn.execute('DELETE FROM cut_edits WHERE episode_id=? AND undone_at IS NOT NULL', (episode_id,))
    edit_id = str(uuid.uuid4())
    conn.execute('INSERT INTO cut_edits (id,episode_id,action,label,cut_id,before_json,after_json,created_ids_json) VALUES (?,?,?,?,?,?,?,?)',
                 (edit_id, episode_id, action, label, cut_id, json.dumps(before), json.dumps(after), json.dumps(sorted(created_ids))))
    conn.execute('DELETE FROM cut_edits WHERE episode_id=? AND id NOT IN (SELECT id FROM cut_edits WHERE episode_id=? ORDER BY rowid DESC LIMIT ?)',
                 (episode_id, episode_id, KEEP))
    return edit_id


def clear(conn, episode_id: str):
    conn.execute('DELETE FROM cut_edits WHERE episode_id=?', (episode_id,))


def state(conn, episode_id: str) -> dict:
    """What Undo and Redo would do now, for the editor's buttons."""
    def entry(row):
        return {'id': row['id'], 'label': row['label'], 'action': row['action'], 'created_at': row['created_at']} if row else None
    undo = conn.execute('SELECT * FROM cut_edits WHERE episode_id=? AND undone_at IS NULL ORDER BY rowid DESC LIMIT 1', (episode_id,)).fetchone()
    redo = conn.execute('SELECT * FROM cut_edits WHERE episode_id=? AND undone_at IS NOT NULL ORDER BY rowid ASC LIMIT 1', (episode_id,)).fetchone()
    return {'undo': entry(undo), 'redo': entry(redo)}


def referencing_columns(conn) -> list[tuple[str, str]]:
    columns = []
    for table in [row['name'] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
        if table in BOOKKEEPING_TABLES:
            continue
        for key in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            if key['table'] == 'cuts':
                columns.append((table, key['from']))
    return columns


def has_history(conn, cut_id: str) -> bool:
    """True when the cut holds user work or anything refers to it, so it must be archived, never deleted."""
    row = conn.execute('SELECT title,description,notes,results_json,no_characters FROM cuts WHERE id=?', (cut_id,)).fetchone()
    if row is None:
        return False
    if row['title'] or row['description'] or row['notes'] or row['no_characters'] or (row['results_json'] or '{}').strip() not in ('', '{}'):
        return True
    for table, column in referencing_columns(conn):
        if conn.execute(f'SELECT 1 FROM "{table}" WHERE "{column}"=? LIMIT 1', (cut_id,)).fetchone():
            return True
    # Sequences name their cuts without a foreign key.
    return conn.execute('SELECT 1 FROM sequences WHERE first_cut_id=? OR last_cut_id=? OR instr(cuts_json, ?) > 0 LIMIT 1', (cut_id, cut_id, cut_id)).fetchone() is not None


def busy(conn, episode_id: str) -> bool:
    # Only an episode-wide replacement owns the entire cut list.
    return conn.execute("""SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')
        AND (pipeline='prepare' OR (pipeline='hardcuts' AND json_extract(payload_json,'$.cut_id') IS NULL))""",
                        (episode_id,)).fetchone() is not None


def _restore(conn, episode_id: str, expected: list[dict], target: list[dict], created_ids: set[str]) -> tuple[set[str], list[str]]:
    """Make the active cuts equal to `target`. Returns the cuts whose frames must be rebuilt and the ids removed."""
    current = snapshot(conn, episode_id)
    if not same(current, expected):
        raise Conflict('The cuts changed after this edit, so it cannot be reversed any more')
    target_ids = {cut['id'] for cut in target}
    # Park every active cut below all existing indices first, so UNIQUE(episode_id, cut_index) holds after every statement.
    lowest = conn.execute('SELECT MIN(cut_index) FROM cuts WHERE episode_id=?', (episode_id,)).fetchone()[0] or 0
    park = min(lowest, 0) - 1
    for offset, cut in enumerate(current):
        conn.execute('UPDATE cuts SET cut_index=? WHERE id=?', (park - offset, cut['id']))
    removed = []
    for cut in current:
        if cut['id'] in target_ids:
            continue
        if cut['id'] in created_ids and not has_history(conn, cut['id']):
            conn.execute('DELETE FROM cuts WHERE id=?', (cut['id'],))
        else:
            conn.execute('UPDATE cuts SET active=0 WHERE id=?', (cut['id'],))
        removed.append(cut['id'])
    changed = set()
    for cut in target:
        row = conn.execute('SELECT active,start_time,end_time FROM cuts WHERE id=? AND episode_id=?', (cut['id'], episode_id)).fetchone()
        if row is None:
            conn.execute('INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time,analysis_enabled,created_at) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                         (cut['id'], episode_id, cut['cut_index'], cut['start_time'], cut['end_time'], cut.get('analysis_enabled', 1)))
            changed.add(cut['id'])
            continue
        if not row['active'] or abs(row['start_time'] - cut['start_time']) > TOLERANCE or abs(row['end_time'] - cut['end_time']) > TOLERANCE:
            changed.add(cut['id'])
        conn.execute('UPDATE cuts SET active=1,cut_index=?,start_time=?,end_time=? WHERE id=?', (cut['cut_index'], cut['start_time'], cut['end_time'], cut['id']))
    for cut_id in changed:
        conn.execute('UPDATE cuts SET first_frame_time=NULL,last_frame_time=NULL WHERE id=?', (cut_id,))
    return changed, removed


def undo(conn, episode_id: str) -> dict:
    if busy(conn, episode_id):
        raise Busy('Wait for media preparation or full hard-cut detection to finish')
    edit = conn.execute('SELECT * FROM cut_edits WHERE episode_id=? AND undone_at IS NULL ORDER BY rowid DESC LIMIT 1', (episode_id,)).fetchone()
    if not edit:
        raise Conflict('There is no cut edit to undo')
    changed, removed = _restore(conn, episode_id, json.loads(edit['after_json']), json.loads(edit['before_json']), set(json.loads(edit['created_ids_json'])))
    conn.execute('UPDATE cut_edits SET undone_at=CURRENT_TIMESTAMP WHERE id=?', (edit['id'],))
    return {'label': edit['label'], 'cut_id': edit['cut_id'], 'changed': changed, 'removed': removed}


def redo(conn, episode_id: str) -> dict:
    if busy(conn, episode_id):
        raise Busy('Wait for media preparation or full hard-cut detection to finish')
    edit = conn.execute('SELECT * FROM cut_edits WHERE episode_id=? AND undone_at IS NOT NULL ORDER BY rowid ASC LIMIT 1', (episode_id,)).fetchone()
    if not edit:
        raise Conflict('There is no cut edit to redo')
    changed, removed = _restore(conn, episode_id, json.loads(edit['before_json']), json.loads(edit['after_json']), set())
    conn.execute('UPDATE cut_edits SET undone_at=NULL WHERE id=?', (edit['id'],))
    return {'label': edit['label'], 'cut_id': edit['cut_id'], 'changed': changed, 'removed': removed}
