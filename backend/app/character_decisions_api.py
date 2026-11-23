"""Routes of the character agent (M6, ``character_decisions.py``), docs/autopilot/HARNESS.md.

Queue it for an episode (or a cut range), read what it applied and asked, fetch the sheet a reviewer looks at, and answer
one question (Claude as the second reviewer, or the person). Labels, removed boxes and reviewed frames are written only in
the job's or the answer's transaction.
"""
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import agent_log, character_decisions as cd, db, episode_requests, worker

router = APIRouter()


class CharacterRun(BaseModel):
    from_cut: int | None = Field(default=None, ge=1)
    to_cut: int | None = Field(default=None, ge=1)
    run_id: str | None = None
    boxes: bool = True
    targets: bool = True


class LabelAnswer(BaseModel):
    confirm: list[str] = Field(default_factory=list, max_length=200)
    reject: list[str] = Field(default_factory=list, max_length=200)
    character_context_id: str | None = None
    tier: Literal['claude', 'human'] = 'human'
    run_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


class PersonAnswer(BaseModel):
    choice: Literal['character', 'unregistered', 'new', 'background', 'leave', 'unsure']
    character_context_id: str | None = None
    title: str | None = Field(default=None, max_length=200)
    content: str = Field(default='', max_length=20000)
    is_main: bool = False
    episode_only: bool = False
    boxes: list[str] | None = Field(default=None, max_length=200)
    tier: Literal['claude', 'human'] = 'human'
    run_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


class TargetAnswer(BaseModel):
    frames: dict[str, Literal['ok', 'person']]
    tier: Literal['claude', 'human'] = 'human'
    run_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


@router.post('/api/episodes/{episode_id}/character-decisions', summary='Queue the agent that names boxes, groups unknown people and settles target frames')
def queue_character_decisions(episode_id: str, body: CharacterRun):
    span = episode_requests.cut_range(body.from_cut, body.to_cut)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not agent_log.episode_exists(conn, episode_id):
            raise HTTPException(404, 'Episode not found')
        episode_requests.refuse_range_while_cutting(conn, episode_id, span.get('last_index') is not None)
        try:
            job_id = cd.queue(conn, episode_id, run_id=body.run_id, first_index=span['first_index'], last_index=span.get('last_index'),
                              boxes=body.boxes, targets=body.targets)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        cd.close_settled(conn, episode_id)
    worker.wake_event.set()
    if job_id is None:
        return {'job_id': None, 'detail': 'Box names and target frames are switched off in the Autopilot profile'}
    return {'job_id': job_id}


@router.get('/api/episodes/{episode_id}/character-decisions', summary='What the character agent applied, what it asked and its last run')
def character_report(episode_id: str):
    with db.connect() as conn:
        if not agent_log.episode_exists(conn, episode_id):
            raise HTTPException(404, 'Episode not found')
        names = cd.titles(conn, episode_id)
        job = conn.execute("""SELECT id,status,stage,payload_json,updated_at FROM jobs WHERE episode_id=? AND pipeline=?
            ORDER BY rowid DESC LIMIT 1""", (episode_id, cd.PIPELINE)).fetchone()
        last = None
        if job:
            payload = cd._load(job['payload_json'], {}) or {}
            last = {'job_id': job['id'], 'status': job['status'], 'stage': job['stage'], 'updated_at': job['updated_at'], 'result': payload.get('result')}
        applied = agent_log.decisions(conn, episode_id=episode_id, kind=cd.KIND_BOXES, limit=500) + \
            agent_log.decisions(conn, episode_id=episode_id, kind=cd.KIND_TARGETS, limit=500)
        for item in applied:
            item['character'] = names.get((item['after'] or {}).get('character_context_id'))
        questions = [item for kind in cd.ESCALATIONS for item in agent_log.escalations(conn, episode_id, 'open', kind)]
        return {'applied': applied, 'questions': questions, 'last_run': last,
                'characters': [{'id': key, 'title': value} for key, value in sorted(names.items(), key=lambda kv: kv[1])]}


@router.get('/api/escalations/{escalation_id}/box-sheet', summary='Each box of a name question beside the user\'s example of each candidate (JPEG)')
def box_sheet(escalation_id: str):
    with db.connect() as conn:
        try:
            data = cd.box_sheet(conn, escalation_id)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except (RuntimeError, ValueError, cd.Refused) as exc:
            raise HTTPException(422, str(exc)) from exc
    return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})


@router.get('/api/escalations/{escalation_id}/target-sheet', summary='The opening frame and each flagged target frame with every colour (JPEG)')
def target_sheet(escalation_id: str):
    with db.connect() as conn:
        try:
            data = cd.target_sheet(conn, escalation_id)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except (RuntimeError, ValueError, cd.Refused) as exc:
            raise HTTPException(422, str(exc)) from exc
    return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})


def _answer(call):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            return call(conn)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except cd.Refused as exc:
            raise HTTPException(exc.args[1] if len(exc.args) > 1 else 409, str(exc.args[0])) from exc


@router.post('/api/escalations/{escalation_id}/box-labels', summary='Answer a name question: confirm the proposed character on some boxes, reject others')
def answer_labels(escalation_id: str, body: LabelAnswer):
    return _answer(lambda conn: cd.answer_labels(conn, escalation_id, confirm=body.confirm, reject=body.reject, character_id=body.character_context_id,
                                                 tier=body.tier, run_id=body.run_id, note=body.note))


@router.post('/api/escalations/{escalation_id}/person', summary='Answer an unknown-person question: a character, a new one, background, leave or unsure')
def answer_person(escalation_id: str, body: PersonAnswer):
    return _answer(lambda conn: cd.answer_person(conn, escalation_id, choice=body.choice, character_id=body.character_context_id, title=body.title,
                                                 content=body.content, is_main=body.is_main, episode_only=body.episode_only, boxes=body.boxes,
                                                 tier=body.tier, run_id=body.run_id, note=body.note))


@router.post('/api/escalations/{escalation_id}/target-frames', summary='Answer a target-frame question: per frame ok (reviewed) or person')
def answer_target(escalation_id: str, body: TargetAnswer):
    try:
        frames = {int(key): value for key, value in body.frames.items()}
    except ValueError:
        raise HTTPException(422, 'Frames are named by their ordinal') from None
    return _answer(lambda conn: cd.answer_target(conn, escalation_id, frames=frames, tier=body.tier, run_id=body.run_id, note=body.note))
