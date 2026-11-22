"""The autopilot's record of what it did: runs, decisions and escalations (docs/autopilot/HARNESS.md, milestone M0).

A *run* is one agent session over an episode. A *decision* is one change the agent applied; the code that applies it
calls ``record`` in the same transaction, with the target's state before and after, so ``undo`` can put it back as
long as the target still holds what the agent wrote (the same rule as ``cut_history``). An *escalation* is what the
agent could not settle: one open item per ``(episode, kind, group_key)``, so one face or one voice is one question for
the person however many cuts it touches.

Undo knows how to restore a target through ``UNDO``: a function per ``target_table``. ``restore_columns`` covers the
plain case (a few columns of one row); milestones that change revision chains or cut boundaries register their own.
This is provenance, not an approval status: nothing here gates or hides a result.
"""
import json
import uuid

from . import db

TIERS = ('rules', 'local', 'claude', 'human')
AGENT = 'agent'  # the origin / label_source value of everything the autopilot writes


class UndoConflict(Exception):
    """The target changed after the agent's decision (usually the user edited it), so undo would overwrite that."""


def _dump(value):
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(text, default=None):
    if text in (None, ''):
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


# --- runs --------------------------------------------------------------------------------------------------------------

def start_run(conn, episode_id, agent='', purpose='', policy=None):
    run_id = str(uuid.uuid4())
    conn.execute('INSERT INTO agent_runs(id,episode_id,agent,purpose,policy_json) VALUES (?,?,?,?,?)',
                 (run_id, episode_id, str(agent or '')[:80], str(purpose or '')[:400], _dump(policy or {})))
    return run_id


def finish_run(conn, run_id, summary=None, claude_items=0):
    cursor = conn.execute("""UPDATE agent_runs SET status=CASE WHEN status='running' THEN 'finished' ELSE status END,
        finished_at=COALESCE(finished_at,CURRENT_TIMESTAMP),summary_json=?,claude_items=claude_items+? WHERE id=?""",
                          (_dump(summary or {}), max(0, int(claude_items or 0)), run_id))
    return bool(cursor.rowcount)


def run_view(row, counts=None):
    item = {key: row[key] for key in ('id', 'episode_id', 'agent', 'purpose', 'status', 'claude_items', 'started_at', 'finished_at')}
    item['summary'] = _load(row['summary_json'], {})
    if counts is not None:
        item['decisions'] = counts
    return item


def runs(conn, episode_id, limit=50):
    rows = conn.execute('SELECT * FROM agent_runs WHERE episode_id=? ORDER BY started_at DESC,rowid DESC LIMIT ?', (episode_id, limit)).fetchall()
    counts = {}
    if rows:
        marks = ','.join('?' for _ in rows)
        for run_id, total, undone, overridden in conn.execute(f"""SELECT run_id,COUNT(*),SUM(undone_at IS NOT NULL),SUM(overridden_at IS NOT NULL)
                FROM agent_decisions WHERE run_id IN ({marks}) GROUP BY run_id""", [row['id'] for row in rows]):
            counts[run_id] = {'total': total, 'undone': undone or 0, 'overridden': overridden or 0}
    return [run_view(row, counts.get(row['id'], {'total': 0, 'undone': 0, 'overridden': 0})) for row in rows]


# --- decisions ---------------------------------------------------------------------------------------------------------

def record(conn, *, episode_id, kind, target_table, target_id, before, after, tier='local', run_id=None, cut_id=None,
           evidence=None, confidence=None, summary=''):
    """Log one applied decision. Call it inside the transaction that applied the change."""
    if tier not in TIERS:
        raise ValueError(f'tier must be one of {", ".join(TIERS)}')
    decision_id = str(uuid.uuid4())
    conn.execute("""INSERT INTO agent_decisions(id,run_id,episode_id,cut_id,kind,tier,target_table,target_id,before_json,after_json,
        evidence_json,confidence,summary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (decision_id, run_id, episode_id, cut_id, kind, tier, target_table, str(target_id), _dump(before), _dump(after),
                  _dump(evidence or {}), confidence, str(summary or '')[:500]))
    return decision_id


def decision_view(row):
    item = {key: row[key] for key in ('id', 'run_id', 'episode_id', 'cut_id', 'kind', 'tier', 'target_table', 'target_id',
                                      'confidence', 'summary', 'created_at', 'undone_at', 'overridden_at')}
    item.update(before=_load(row['before_json']), after=_load(row['after_json']), evidence=_load(row['evidence_json'], {}))
    return item


def decisions(conn, *, episode_id=None, run_id=None, kind=None, cut_id=None, limit=200):
    clauses, args = [], []
    for column, value in (('episode_id', episode_id), ('run_id', run_id), ('kind', kind), ('cut_id', cut_id)):
        if value:
            clauses.append(f'{column}=?')
            args.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ''
    rows = conn.execute(f'SELECT * FROM agent_decisions {where} ORDER BY created_at DESC,rowid DESC LIMIT ?', [*args, limit]).fetchall()
    return [decision_view(row) for row in rows]


def note_override(conn, target_table, target_id):
    """A person changed a target the agent had decided. Call it from user edit paths; it feeds the thresholds."""
    conn.execute("""UPDATE agent_decisions SET overridden_at=CURRENT_TIMESTAMP
        WHERE target_table=? AND target_id=? AND undone_at IS NULL AND overridden_at IS NULL""", (target_table, str(target_id)))


# --- undo --------------------------------------------------------------------------------------------------------------

# Columns a plain decision may restore, per table. A milestone adds its table here (or registers a handler in UNDO);
# anything else cannot be undone generically, which keeps undo from writing columns nobody meant it to.
RESTORABLE = {
    'character_detections': ('character_context_id', 'label_source', 'match_reason'),
    'cuts': ('no_characters', 'analysis_enabled'),
}


def restore_columns(conn, decision):
    table, before, after = decision['target_table'], _load(decision['before_json'], {}), _load(decision['after_json'], {})
    allowed = RESTORABLE.get(table)
    if allowed is None:
        raise UndoConflict(f'Decisions on {table} cannot be undone automatically')
    columns = [column for column in after if column in allowed]
    if not columns or any(column not in allowed for column in [*after, *before]):
        raise UndoConflict(f'This decision changed columns of {table} that undo does not restore')
    row = conn.execute(f'SELECT {",".join(columns)} FROM {table} WHERE id=?', (decision['target_id'],)).fetchone()
    if row is None:
        raise UndoConflict('The changed item no longer exists')
    if any(row[column] != after[column] for column in columns):
        raise UndoConflict('Changed since the agent decided; undo would overwrite that change')
    conn.execute(f'UPDATE {table} SET {",".join(f"{column}=?" for column in columns)} WHERE id=?',
                 [before.get(column) for column in columns] + [decision['target_id']])


UNDO = {table: restore_columns for table in RESTORABLE}
# Undo handlers per decision kind; they win over the table's handler (a split and a label are both decisions on `cuts`).
UNDO_KINDS = {}
REVIEWERS = ('claude', 'human')


def undo_decision(conn, decision_id):
    """Undo one decision. Raises ``UndoConflict`` when it cannot; returns False when it was already undone."""
    decision = conn.execute('SELECT * FROM agent_decisions WHERE id=?', (decision_id,)).fetchone()
    if decision is None:
        raise KeyError(decision_id)
    if decision['undone_at']:
        return False
    handler = UNDO_KINDS.get(decision['kind']) or UNDO.get(decision['target_table'])
    if handler is None:
        raise UndoConflict(f'Decisions on {decision["target_table"]} cannot be undone automatically')
    handler(conn, decision)
    conn.execute('UPDATE agent_decisions SET undone_at=CURRENT_TIMESTAMP WHERE id=?', (decision_id,))
    return True


def undo_run(conn, run_id):
    """Undo every decision of a run, newest first. Conflicts are skipped and reported, never forced."""
    undone, conflicts = [], []
    for row in conn.execute('SELECT id,summary FROM agent_decisions WHERE run_id=? AND undone_at IS NULL ORDER BY created_at DESC,rowid DESC', (run_id,)).fetchall():
        conn.execute('SAVEPOINT agent_undo')
        try:
            undo_decision(conn, row['id'])
        except UndoConflict as exc:
            conn.execute('ROLLBACK TO agent_undo')
            conflicts.append({'decision_id': row['id'], 'summary': row['summary'], 'reason': str(exc)})
        else:
            undone.append(row['id'])
        conn.execute('RELEASE agent_undo')
    if not conflicts:
        conn.execute("UPDATE agent_runs SET status='undone' WHERE id=?", (run_id,))
    return {'undone': undone, 'conflicts': conflicts}


# --- escalations -------------------------------------------------------------------------------------------------------

def escalate(conn, *, episode_id, kind, group_key, question, cut_ids=(), options=(), proposal=None, evidence=None, run_id=None, reviewer='human'):
    """Open (or update) the one open escalation of this kind and group; cut ids accumulate. Returns its id.

    ``reviewer`` is who answers it next: ``claude`` (the second reviewer) or ``human``. Updating an open item never hands
    it back from the person to Claude."""
    if reviewer not in REVIEWERS:
        raise ValueError('reviewer must be claude or human')
    existing = conn.execute("SELECT id,cut_ids_json FROM agent_escalations WHERE episode_id=? AND kind=? AND group_key=? AND status='open'",
                            (episode_id, kind, group_key)).fetchone()
    if existing:
        merged = list(dict.fromkeys([*_load(existing['cut_ids_json'], []), *cut_ids]))
        conn.execute("""UPDATE agent_escalations SET cut_ids_json=?,question=?,options_json=?,proposal_json=?,evidence_json=?,
            run_id=COALESCE(?,run_id),reviewer=CASE WHEN ?='human' THEN 'human' ELSE reviewer END,updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                     (_dump(merged), question, _dump(list(options)), _dump(proposal), _dump(evidence or {}), run_id, reviewer, existing['id']))
        return existing['id']
    escalation_id = str(uuid.uuid4())
    conn.execute("""INSERT INTO agent_escalations(id,episode_id,run_id,kind,group_key,cut_ids_json,question,options_json,proposal_json,evidence_json,reviewer)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (escalation_id, episode_id, run_id, kind, group_key, _dump(list(dict.fromkeys(cut_ids))),
                                            question, _dump(list(options)), _dump(proposal), _dump(evidence or {}), reviewer))
    return escalation_id


def escalation_view(row):
    item = {key: row[key] for key in ('id', 'episode_id', 'run_id', 'kind', 'group_key', 'question', 'status', 'reviewer', 'created_at', 'updated_at', 'resolved_at')}
    item.update(cut_ids=_load(row['cut_ids_json'], []), options=_load(row['options_json'], []), proposal=_load(row['proposal_json']),
                evidence=_load(row['evidence_json'], {}), resolution=_load(row['resolution_json']))
    return item


def escalations(conn, episode_id, status='open', kind=None, limit=500, reviewer=None):
    clauses, args = ['episode_id=?'], [episode_id]
    if reviewer:
        clauses.append('reviewer=?')
        args.append(reviewer)
    if status and status != 'all':
        clauses.append('status=?')
        args.append(status)
    if kind:
        clauses.append('kind=?')
        args.append(kind)
    rows = conn.execute(f"SELECT * FROM agent_escalations WHERE {' AND '.join(clauses)} ORDER BY created_at,rowid LIMIT ?", [*args, limit]).fetchall()
    return [escalation_view(row) for row in rows]


def hand_to_person(conn, escalation_id, note=None):
    """Claude could not settle it: the person answers it next. ``note`` is added to its evidence."""
    row = conn.execute("SELECT evidence_json FROM agent_escalations WHERE id=? AND status='open'", (escalation_id,)).fetchone()
    if row is None:
        return False
    evidence = _load(row['evidence_json'], {})
    if note:
        evidence['claude_note'] = str(note)[:1000]
    conn.execute("UPDATE agent_escalations SET reviewer='human',evidence_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (_dump(evidence), escalation_id))
    return True


def close_escalation(conn, escalation_id, status, resolution=None):
    if status not in ('resolved', 'dismissed'):
        raise ValueError('status must be resolved or dismissed')
    cursor = conn.execute("""UPDATE agent_escalations SET status=?,resolution_json=?,resolved_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND status='open'""", (status, _dump(resolution), escalation_id))
    return bool(cursor.rowcount)


def counts(conn, episode_id):
    """Open escalations per kind and agent decisions per kind (not undone), for the inbox header and `fw inbox`."""
    open_items = dict(conn.execute("SELECT kind,COUNT(*) FROM agent_escalations WHERE episode_id=? AND status='open' GROUP BY kind", (episode_id,)).fetchall())
    applied = dict(conn.execute('SELECT kind,COUNT(*) FROM agent_decisions WHERE episode_id=? AND undone_at IS NULL GROUP BY kind', (episode_id,)).fetchall())
    reviewers = dict(conn.execute("SELECT reviewer,COUNT(*) FROM agent_escalations WHERE episode_id=? AND status='open' GROUP BY reviewer", (episode_id,)).fetchall())
    return {'open': open_items, 'open_total': sum(open_items.values()), 'applied': applied, 'applied_total': sum(applied.values()),
            'for_claude': reviewers.get('claude', 0), 'for_person': reviewers.get('human', 0)}


def episode_exists(conn, episode_id):
    return db.one('SELECT 1 FROM episodes WHERE id=?', (episode_id,), conn=conn) is not None
