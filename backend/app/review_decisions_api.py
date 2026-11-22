"""Routes of the review-answer resolver (M4, ``review_decisions.py``), docs/autopilot/HARNESS.md.

Queue the decider for a range of cuts (or a shadow run that only measures it), read what it decided and applied, fetch
the picture sheet a reviewer looks at, and answer a cut's escalation (Claude as the second reviewer, or the person).
Answers are written in the job's or the answer's transaction, never anywhere else.
"""
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import agent_log, db, episode_requests, review_decisions, worker

router = APIRouter()


class ReviewDecisionRun(BaseModel):
    from_cut: int | None = Field(default=None, ge=1)
    to_cut: int | None = Field(default=None, ge=1)
    run_id: str | None = None
    shadow: bool = False
    redo: bool = False


class ReviewPick(BaseModel):
    question: str = Field(min_length=1, max_length=20)
    option: str | None = Field(default=None, max_length=4)
    text: str | None = Field(default=None, max_length=600)
    why: str | None = Field(default=None, max_length=300)


class ReviewAnswer(BaseModel):
    answers: list[ReviewPick] = Field(default_factory=list, max_length=10)
    unsure: list[str] = Field(default_factory=list, max_length=10)
    tier: Literal['claude', 'human'] = 'human'
    run_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


@router.post('/api/episodes/{episode_id}/review-decisions', summary='Queue the agent that answers the consistency review questions (shadow: measure only)')
def queue_review_decisions(episode_id: str, body: ReviewDecisionRun):
    span = episode_requests.cut_range(body.from_cut, body.to_cut)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        episode_requests.refuse_range_while_cutting(conn, episode_id, span.get('last_index') is not None)
        try:
            queued = review_decisions.queue(conn, episode_id, first_index=span['first_index'], last_index=span.get('last_index'),
                                            run_id=body.run_id, shadow=body.shadow, redo=body.redo)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not body.shadow:
            review_decisions.close_settled(conn, episode_id)
    worker.wake_event.set()
    if queued is None:
        return {'job_ids': [], 'skipped': {}, 'detail': 'Review answers are switched off in the Autopilot profile'}
    return queued


@router.get('/api/episodes/{episode_id}/review-decisions', summary='What the agent decided per cut, what it applied, the open questions and the trust per topic')
def review_decision_report(episode_id: str, from_cut: int | None = None, to_cut: int | None = None, shadow: bool = False):
    with db.connect() as conn:
        episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
        if not episode:
            raise HTTPException(404, 'Episode not found')
        first, last = max(0, (from_cut or 1) - 1), (to_cut - 1 if to_cut else None)
        cuts = {row['id']: row['cut_index'] + 1 for row in conn.execute(
            'SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR cut_index<=?)', (episode_id, first, last, last))}
        latest = {}
        for row in conn.execute('SELECT * FROM review_decisions WHERE episode_id=? AND shadow=? ORDER BY rowid', (episode_id, int(shadow))):
            if row['cut_id'] in cuts:
                latest[row['cut_id']] = row
        counts, items = {}, []
        for cut_id, row in sorted(latest.items(), key=lambda pair: cuts[pair[0]]):
            outcomes = json.loads(row['outcomes_json'] or '{}')
            for outcome in outcomes.values():
                counts[outcome['action']] = counts.get(outcome['action'], 0) + 1
            items.append({'cut': cuts[cut_id], 'cut_id': cut_id, 'result_id': row['result_id'], 'created_at': row['created_at'],
                          'decisions': json.loads(row['decisions_json'] or '{}'), 'outcomes': outcomes})
        applied = [item for item in agent_log.decisions(conn, episode_id=episode_id, kind=review_decisions.KIND, limit=1000) if item['cut_id'] in cuts]
        questions = [item for item in agent_log.escalations(conn, episode_id, 'open', review_decisions.ESCALATION)
                     if (item['evidence'] or {}).get('cut_id') in cuts]
        return {'counts': counts, 'cuts': items, 'applied': applied, 'questions': questions,
                'trust': review_decisions.trust_table(conn, episode['project_id']), 'trust_needed': review_decisions.TRUST}


@router.get('/api/cuts/{cut_id}/review-sheet', summary="The pictures the cut's consistency review looked at, tiled (JPEG)")
def review_sheet(cut_id: str, labels: str | None = None):
    wanted = [label.strip() for label in (labels or '').split(',') if label.strip()]
    with db.connect() as conn:
        if not conn.execute('SELECT 1 FROM cuts WHERE id=?', (cut_id,)).fetchone():
            raise HTTPException(404, 'Cut not found')
        try:
            data = review_decisions.review_sheet(conn, cut_id, wanted or None)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})


@router.post('/api/escalations/{escalation_id}/review-answers', summary="Answer a cut's review questions with option ids (Claude) or options and text (the person)")
def answer_review_questions(escalation_id: str, body: ReviewAnswer):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            return review_decisions.answer(conn, escalation_id, [pick.model_dump() for pick in body.answers], body.unsure, body.tier, body.run_id, body.note)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except review_decisions.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
