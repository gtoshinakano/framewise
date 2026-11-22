"""Routes of the missed-cut checker (M2, ``missed_cuts.py``) and the line splitter (M3, ``line_splits.py``), docs/autopilot/HARNESS.md.

Queue a check, read what it found, fetch the strip a reviewer looks at, and answer a question (Claude as the second
reviewer, or the person). The split itself happens in the job's transaction or in the answer's, never over HTTP elsewhere.
"""
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import agent_log, db, episode_requests, line_splits, missed_cuts, worker

router = APIRouter()


class MissedCutCheck(BaseModel):
    from_cut: int | None = Field(default=None, ge=1)
    to_cut: int | None = Field(default=None, ge=1)
    redo: bool = False
    run_id: str | None = None


class MissedCutAnswer(BaseModel):
    answer: Literal['split', 'no_cut', 'unsure']
    tier: Literal['claude', 'human'] = 'human'
    run_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


@router.post('/api/episodes/{episode_id}/missed-cuts', summary='Queue a check for hard cuts the detection missed')
def queue_missed_cut_check(episode_id: str, body: MissedCutCheck):
    span = episode_requests.cut_range(body.from_cut, body.to_cut)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        episode_requests.refuse_range_while_cutting(conn, episode_id, span.get('last_index') is not None)
        try:
            job_id = missed_cuts.queue(conn, episode_id, first_index=span['first_index'], last_index=span.get('last_index'),
                                       redo=body.redo, run_id=body.run_id)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    worker.wake_event.set()
    if job_id is None:
        return {'job_id': None, 'job_ids': [], 'detail': 'Missed-cut checks are switched off in the Autopilot profile'}
    return {'job_id': job_id, 'job_ids': [job_id]}


@router.get('/api/episodes/{episode_id}/missed-cuts', summary="What the checker found: per cut its verdict, the splits applied and the open questions")
def missed_cut_report(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    with db.connect() as conn:
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        first, last = max(0, (from_cut or 1) - 1), (to_cut - 1 if to_cut else None)
        cuts = conn.execute("""SELECT c.id,c.cut_index,c.start_time,c.end_time,c.analysis_enabled,m.verdict,m.cut_start,m.cut_end,m.checked_at,m.result_json
            FROM cuts c LEFT JOIN missed_cut_checks m ON m.cut_id=c.id WHERE c.episode_id=? AND c.active=1 AND c.cut_index>=? AND (? IS NULL OR c.cut_index<=?)
            ORDER BY c.cut_index""", (episode_id, first, last, last)).fetchall()
        counts, items = {'clean': 0, 'apply': 0, 'grey': 0, 'unchecked': 0, 'changed_since': 0, 'analysis_off': 0}, []
        for row in cuts:
            if not row['analysis_enabled']:
                counts['analysis_off'] += 1
                continue
            if row['verdict'] is None:
                counts['unchecked'] += 1
                continue
            if abs(row['cut_start'] - row['start_time']) > 1e-6 or abs(row['cut_end'] - row['end_time']) > 1e-6:
                counts['changed_since'] += 1
                continue
            counts[row['verdict']] += 1
            if row['verdict'] != 'clean':
                result = json.loads(row['result_json'])
                items.append({'cut': row['cut_index'] + 1, 'cut_id': row['id'], 'verdict': row['verdict'], 'split_at': result['split_at'], 'grey': result['grey']})
        splits = agent_log.decisions(conn, episode_id=episode_id, kind=missed_cuts.KIND, limit=500)
        questions = agent_log.escalations(conn, episode_id, 'open', missed_cuts.ESCALATION)
        return {'counts': counts, 'cuts': items, 'splits': splits, 'questions': questions}


@router.get('/api/cuts/{cut_id}/missed-cut-strip', summary='Two frames before and two after a moment, with a red bar between them (JPEG)')
def missed_cut_strip(cut_id: str, time: float):
    with db.connect() as conn:
        cut = conn.execute('SELECT episode_id FROM cuts WHERE id=?', (cut_id,)).fetchone()
        if not cut:
            raise HTTPException(404, 'Cut not found')
        proxy = missed_cuts.proxy_path(conn, cut['episode_id'])
    if not proxy.is_file():
        raise HTTPException(409, 'The preview video is missing')
    try:
        data = missed_cuts.strip_jpeg(proxy, time)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})


@router.post('/api/escalations/{escalation_id}/missed-cut', summary='Answer a missed-cut question: split, no_cut or unsure')
def answer_missed_cut(escalation_id: str, body: MissedCutAnswer):
    from . import cut_previews
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            outcome = missed_cuts.answer(conn, escalation_id, body.answer, body.tier, body.run_id, body.note)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except missed_cuts.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
        cancelled = []
        if outcome['changed']:
            row = conn.execute('SELECT episode_id FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
            episode = conn.execute('SELECT * FROM episodes WHERE id=?', (row['episode_id'],)).fetchone()
            cancelled = cut_previews.after_timeline_edit(conn, episode, set(outcome['changed']))
    worker.after_cancel(cancelled)
    worker.wake_event.set()
    return outcome


# --- line splits at hard cuts (M3, line_splits.py) ----------------------------------------------------------------

class LineSplitRun(BaseModel):
    run_id: str | None = None


@router.post('/api/episodes/{episode_id}/line-splits', summary='Queue the line splitter: lines across a hard cut are split at a pause or trimmed')
def queue_line_splits(episode_id: str, body: LineSplitRun):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        try:
            job_id = line_splits.queue(conn, episode_id, run_id=body.run_id)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    worker.wake_event.set()
    if job_id is None:
        return {'job_id': None, 'job_ids': [], 'detail': 'Line splits are switched off in the Autopilot profile'}
    return {'job_id': job_id, 'job_ids': [job_id]}


@router.get('/api/episodes/{episode_id}/line-splits', summary='What the line splitter would do now with each line that crosses a hard cut (read-only)')
def preview_line_splits(episode_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        items, counts = [], {}
        for row, cut, plan in line_splits.plan_episode(conn, episode_id):
            counts[plan['action']] = counts.get(plan['action'], 0) + 1
            owner = conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND start_time<=? AND end_time>? LIMIT 1',
                                 (episode_id, cut, cut)).fetchone()
            items.append({'line_id': row['id'], 'start_time': row['start_time'], 'end_time': row['end_time'], 'text': row['text'], 'hard_cut': cut,
                          'cut': owner['cut_index'] + 1 if owner else None, 'cut_id': owner['id'] if owner else None,
                          **{key: plan.get(key) for key in ('action', 'why', 'at', 'left', 'right', 'start', 'end')}})
        applied = agent_log.decisions(conn, episode_id=episode_id, kind=line_splits.KIND, limit=1000)
        return {'counts': counts, 'lines': items, 'applied': applied}
