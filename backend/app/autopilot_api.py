"""Routes of the autopilot (docs/autopilot/HARNESS.md): agent runs, their decisions and undo, escalations, the profile.

Decisions are never written through these routes: the code path that applies a change logs it in the same
transaction (``agent_log.record``). Escalations can be opened here because the supervising agent session works
over HTTP (``scripts/fw``).
"""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import agent_log, autopilot_profile, db, diagnose, triage, worker

router = APIRouter()


def require_episode(conn, episode_id):
    row = conn.execute('SELECT id,project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Episode not found')
    return row


def require_run(conn, run_id):
    row = conn.execute('SELECT * FROM agent_runs WHERE id=?', (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Agent run not found')
    return row


# --- runs and decisions ------------------------------------------------------------------------------------------------

class AgentRunStart(BaseModel):
    agent: str = Field(default='', max_length=80)
    purpose: str = Field(default='', max_length=400)


class AgentRunFinish(BaseModel):
    summary: dict = Field(default_factory=dict)
    claude_items: int = Field(default=0, ge=0, le=10000)


@router.post('/api/episodes/{episode_id}/agent-runs')
def start_agent_run(episode_id: str, body: AgentRunStart):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        episode = require_episode(conn, episode_id)
        profile, _ = autopilot_profile.load(conn, episode['project_id'])
        run_id = agent_log.start_run(conn, episode_id, body.agent, body.purpose, profile)
        return {**agent_log.run_view(require_run(conn, run_id)), 'policy': profile}


@router.post('/api/agent-runs/{run_id}/finish')
def finish_agent_run(run_id: str, body: AgentRunFinish):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require_run(conn, run_id)
        agent_log.finish_run(conn, run_id, body.summary, body.claude_items)
        return agent_log.run_view(require_run(conn, run_id))


@router.get('/api/episodes/{episode_id}/agent-runs')
def list_agent_runs(episode_id: str, limit: int = 50):
    with db.connect() as conn:
        require_episode(conn, episode_id)
        return agent_log.runs(conn, episode_id, max(1, min(limit, 200)))


@router.get('/api/agent-runs/{run_id}')
def get_agent_run(run_id: str):
    with db.connect() as conn:
        run = require_run(conn, run_id)
        return {**agent_log.run_view(run), 'decisions': agent_log.decisions(conn, run_id=run_id, limit=5000)}


@router.get('/api/episodes/{episode_id}/agent-decisions')
def list_agent_decisions(episode_id: str, kind: str | None = None, cut_id: str | None = None, limit: int = 200):
    with db.connect() as conn:
        require_episode(conn, episode_id)
        return agent_log.decisions(conn, episode_id=episode_id, kind=kind, cut_id=cut_id, limit=max(1, min(limit, 2000)))


@router.post('/api/agent-decisions/{decision_id}/undo')
def undo_agent_decision(decision_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            changed = agent_log.undo_decision(conn, decision_id)
        except KeyError:
            raise HTTPException(404, 'Agent decision not found') from None
        except agent_log.UndoConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return {'undone': changed}


@router.post('/api/agent-runs/{run_id}/undo')
def undo_agent_run(run_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require_run(conn, run_id)
        return agent_log.undo_run(conn, run_id)


# --- diagnosis and triage (M1) ------------------------------------------------------------------------------------------

@router.get('/api/episodes/{episode_id}/health', summary='Per cut, ranked problems in one sentence each, worst cuts first')
def episode_health(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    with db.connect() as conn:
        conn.execute('BEGIN')
        return diagnose.health(conn, episode_id, from_cut, to_cut)


@router.get('/api/cuts/{cut_id}/why', summary="Why a cut's step is not done: its prerequisites down to the first broken one")
def cut_why(cut_id: str, step: str | None = None):
    with db.connect() as conn:
        conn.execute('BEGIN')
        try:
            return diagnose.why(conn, cut_id, step)
        except KeyError:
            raise HTTPException(404, 'Cut not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@router.get('/api/episodes/{episode_id}/triage', summary='Failed runs nobody has dealt with, their cause and the fix triage would apply')
def episode_triage(episode_id: str, run_id: str | None = None):
    with db.connect() as conn:
        conn.execute('BEGIN')
        require_episode(conn, episode_id)
        try:
            mode = triage.mode_for(conn, episode_id, run_id)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {**diagnose.triage(conn, episode_id, mode), 'mode': mode}


class TriageApply(BaseModel):
    run_id: str | None = None
    job_ids: list[str] | None = Field(default=None, max_length=2000)


@router.post('/api/episodes/{episode_id}/triage', summary='Apply the fix rules to failed runs; log each fix as an agent decision')
def apply_triage(episode_id: str, body: TriageApply):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require_episode(conn, episode_id)
        try:
            outcome = triage.apply(conn, episode_id, body.run_id, body.job_ids)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    worker.wake_event.set()
    return outcome


# --- escalations -------------------------------------------------------------------------------------------------------

class EscalationInput(BaseModel):
    kind: str = Field(min_length=1, max_length=60)
    group_key: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=2000)
    cut_ids: list[str] = Field(default_factory=list, max_length=2000)
    options: list = Field(default_factory=list, max_length=50)
    proposal: dict | None = None
    evidence: dict = Field(default_factory=dict)
    run_id: str | None = None


class EscalationClose(BaseModel):
    status: Literal['resolved', 'dismissed']
    resolution: dict | None = None


@router.get('/api/episodes/{episode_id}/escalations')
def list_escalations(episode_id: str, status: str = 'open', kind: str | None = None):
    with db.connect() as conn:
        require_episode(conn, episode_id)
        return {'items': agent_log.escalations(conn, episode_id, status, kind), 'counts': agent_log.counts(conn, episode_id)}


@router.post('/api/episodes/{episode_id}/escalations')
def open_escalation(episode_id: str, body: EscalationInput):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        require_episode(conn, episode_id)
        if body.run_id:
            run = require_run(conn, body.run_id)
            if run['episode_id'] != episode_id:
                raise HTTPException(422, 'The agent run belongs to another episode')
        if body.cut_ids:
            marks = ','.join('?' for _ in body.cut_ids)
            found = conn.execute(f'SELECT COUNT(*) FROM cuts WHERE episode_id=? AND id IN ({marks})', [episode_id, *set(body.cut_ids)]).fetchone()[0]
            if found != len(set(body.cut_ids)):
                raise HTTPException(422, 'Every cut must belong to this episode')
        escalation_id = agent_log.escalate(conn, episode_id=episode_id, kind=body.kind, group_key=body.group_key, question=body.question,
                                           cut_ids=body.cut_ids, options=body.options, proposal=body.proposal, evidence=body.evidence, run_id=body.run_id)
        row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
        return agent_log.escalation_view(row)


@router.post('/api/escalations/{escalation_id}/close')
def close_escalation(escalation_id: str, body: EscalationClose):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Escalation not found')
        if not agent_log.close_escalation(conn, escalation_id, body.status, body.resolution):
            raise HTTPException(409, 'This escalation is already closed')
        return agent_log.escalation_view(conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone())


# --- profile -----------------------------------------------------------------------------------------------------------

def profile_response(conn, project_id, profile, updated_at):
    return {
        'profile': profile,
        'updated_at': updated_at,
        'decisions': autopilot_profile.DECISIONS,
        'topics': list(autopilot_profile.TOPICS),
        'topic_stats': autopilot_profile.topic_stats(conn, project_id),
    }


@router.get('/api/projects/{project_id}/autopilot-profile')
def get_autopilot_profile(project_id: str):
    with db.connect() as conn:
        if not conn.execute('SELECT 1 FROM projects WHERE id=?', (project_id,)).fetchone():
            raise HTTPException(404, 'Project not found')
        profile, updated_at = autopilot_profile.load(conn, project_id)
        return profile_response(conn, project_id, profile, updated_at)


@router.put('/api/projects/{project_id}/autopilot-profile')
def put_autopilot_profile(project_id: str, body: dict):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM projects WHERE id=?', (project_id,)).fetchone():
            raise HTTPException(404, 'Project not found')
        try:
            profile = autopilot_profile.save(conn, project_id, body.get('profile', body))
        except autopilot_profile.ProfileError as exc:
            raise HTTPException(422, str(exc)) from exc
        updated_at = conn.execute('SELECT updated_at FROM autopilot_profiles WHERE project_id=?', (project_id,)).fetchone()[0]
        return profile_response(conn, project_id, profile, updated_at)
