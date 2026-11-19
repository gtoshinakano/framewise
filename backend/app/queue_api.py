"""HTTP routes of the episode queue drawer; the logic is in ``runs.py`` and ``job_kinds.py``."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import db, episode_events, runs, worker

router = APIRouter(prefix='/api', tags=['queue'])


def _episode(conn, episode_id):
    if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
        raise HTTPException(404, 'Episode not found')


def _events(request, episode_id):
    return StreamingResponse(episode_events.stream(request, episode_id, request.headers.get('last-event-id')),
                             media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@router.get('/events', summary='Watch queue and host engine changes')
def queue_events(request: Request):
    return _events(request, None)


@router.get('/episodes/{episode_id}/events', summary='Watch episode job, queue and stale-result changes')
def watch_episode(episode_id: str, request: Request):
    with db.connect() as conn:
        _episode(conn, episode_id)
    return _events(request, episode_id)


@router.get('/episodes/{episode_id}/queue', summary='What runs now, what waits and what needs attention in one episode')
def episode_queue(episode_id: str):
    with db.connect() as conn:
        _episode(conn, episode_id)
        return runs.queue_state(conn, episode_id)


@router.get('/episodes/{episode_id}/runs', summary='Finished runs of one episode, newest first')
def episode_runs(episode_id: str, before: int | None = None, limit: int = runs.PAGE_LIMIT, status: Literal['done', 'failed', 'cancelled'] | None = None):
    with db.connect() as conn:
        _episode(conn, episode_id)
        return runs.history(conn, episode_id, before=before, limit=limit, status=status)


class QueueCancelInput(BaseModel):
    include_running: bool = Field(False, description='Also cancel running jobs; they stop at their next checkpoint.')
    group: str | None = Field(None, description='Only this group of the queue (`waiting_groups[].group`).')
    job_ids: list[str] | None = Field(None, description='Only these jobs.')


@router.post('/episodes/{episode_id}/queue/cancel', summary="Cancel this episode's queued (and optionally running) jobs")
def cancel_episode_queue(episode_id: str, body: QueueCancelInput):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _episode(conn, episode_id)
        rows = runs.cancellable(conn, episode_id, body.include_running, body.group, set(body.job_ids) if body.job_ids is not None else None)
        changed = worker.cancel_jobs(conn, rows)
    worker.after_cancel(changed)
    return {'cancelled': len(changed)}


@router.post('/jobs/{job_id}/dismiss', summary='Stop listing a failed run under Needs attention')
def dismiss_job(job_id: str):
    with db.connect() as conn:
        row = conn.execute('SELECT status FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Run not found')
        if row['status'] != 'failed':
            raise HTTPException(409, 'Only failed runs can be dismissed')
        conn.execute('UPDATE jobs SET dismissed_at=CURRENT_TIMESTAMP WHERE id=? AND dismissed_at IS NULL', (job_id,))
    return {'ok': True}


@router.post('/episodes/{episode_id}/queue/dismiss-failures', summary='Dismiss every failure listed under Needs attention')
def dismiss_failures(episode_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _episode(conn, episode_id)
        ids = [row['id'] for row in runs.attention_rows(conn, episode_id)]
        conn.executemany('UPDATE jobs SET dismissed_at=CURRENT_TIMESTAMP WHERE id=?', [(job_id,) for job_id in ids])
    return {'dismissed': len(ids)}


@router.post('/chat-gateway/requests/{request_id}/cancel', summary='Stop a Chat API request from the queue')
def cancel_chat_request(request_id: str):
    from . import chat_gateway
    if not chat_gateway.snapshot(request_id):
        raise HTTPException(404, 'Request not found')
    chat_gateway.request_cancel(request_id)
    return {'ok': True}
