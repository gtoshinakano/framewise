"""Routes of the voice resolver (M5, ``voice_resolver.py``), docs/autopilot/HARNESS.md.

Queue the resolver for an episode, read what it applied and what it asked, fetch the frame sheet a reviewer looks at, and
answer one voice question (Claude as the second reviewer, or the person). Speakers are written only in the job's or the
answer's transaction.
"""
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import agent_log, db, voice_resolver, worker

router = APIRouter()


class VoiceRun(BaseModel):
    run_id: str | None = None
    attribution: bool = True


class VoiceAnswer(BaseModel):
    choice: Literal['character', 'unregistered', 'new', 'leave', 'unsure']
    character_context_id: str | None = None
    tier: Literal['claude', 'human'] = 'human'
    run_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


@router.post('/api/episodes/{episode_id}/voice-decisions', summary='Queue the agent that gives unknown voices to characters')
def queue_voice_resolver(episode_id: str, body: VoiceRun):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not agent_log.episode_exists(conn, episode_id):
            raise HTTPException(404, 'Episode not found')
        try:
            job_id = voice_resolver.queue(conn, episode_id, run_id=body.run_id, attribution=body.attribution)
        except KeyError:
            raise HTTPException(404, 'Agent run not found') from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        voice_resolver.close_settled(conn, episode_id)
    worker.wake_event.set()
    if job_id is None:
        return {'job_id': None, 'detail': 'Voices are switched off in the Autopilot profile'}
    return {'job_id': job_id}


@router.get('/api/episodes/{episode_id}/voice-decisions', summary='Open voices, what the agent gave to whom, the open questions and the last run')
def voice_report(episode_id: str):
    with db.connect() as conn:
        if not agent_log.episode_exists(conn, episode_id):
            raise HTTPException(404, 'Episode not found')
        names = voice_resolver.titles(conn, episode_id)
        open_voices = {}
        for row in conn.execute("""SELECT coalesce(speaker_label,'') AS label,COUNT(*) AS lines,MIN(start_time) AS first_time FROM transcript_segments
                WHERE episode_id=? AND superseded_by IS NULL AND deleted=0 AND character_context_id IS NULL GROUP BY 1 ORDER BY 2 DESC""", (episode_id,)):
            open_voices[row['label']] = {'label': row['label'], 'lines': row['lines'], 'first_time': row['first_time']}
        job = conn.execute("""SELECT id,status,stage,payload_json,updated_at FROM jobs WHERE episode_id=? AND pipeline=?
            ORDER BY rowid DESC LIMIT 1""", (episode_id, voice_resolver.PIPELINE)).fetchone()
        last = None
        if job:
            payload = voice_resolver._load(job['payload_json'], {}) or {}
            last = {'job_id': job['id'], 'status': job['status'], 'stage': job['stage'], 'updated_at': job['updated_at'], 'result': payload.get('result')}
        applied = agent_log.decisions(conn, episode_id=episode_id, kind=voice_resolver.KIND, limit=500)
        for item in applied:
            item['character'] = names.get((item['after'] or {}).get('character_context_id'))
        return {'open_voices': list(open_voices.values()), 'applied': applied, 'characters': [{'id': key, 'title': value} for key, value in sorted(names.items(), key=lambda kv: kv[1])],
                'questions': agent_log.escalations(conn, episode_id, 'open', voice_resolver.ESCALATION), 'last_run': last}


@router.get('/api/escalations/{escalation_id}/voice-sheet', summary="Three frames over each sample phrase of a voice question (JPEG)")
def voice_sheet(escalation_id: str):
    with db.connect() as conn:
        try:
            data = voice_resolver.voice_sheet(conn, escalation_id)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})


@router.post('/api/escalations/{escalation_id}/voice', summary='Answer a voice question: a character, new, leave (person) or unsure (Claude)')
def answer_voice(escalation_id: str, body: VoiceAnswer):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            return voice_resolver.answer(conn, escalation_id, body.choice, body.character_context_id, body.tier, body.run_id, body.note)
        except KeyError:
            raise HTTPException(404, 'Escalation not found') from None
        except voice_resolver.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
