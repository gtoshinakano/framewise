"""Routes of Run Multiple (``batch_runs.py``, docs/autopilot/HARNESS.md D37).

Preview a stage (what each step would run on now, which cuts wait for the person and why), start a run, read it while
it goes, read the episode's runs and cancel one. The steps themselves are carried out by ``batch_runs.tick`` in the API
process, never inside these requests.
"""
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import batch_runs, db, worker

router = APIRouter()
Stage = Literal['prepare', 'prepare_cuts', 'evaluate']


class SubtitleChoice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    language: str = Field(min_length=2, max_length=60, description='Language of the burned-in subtitles; read only when given.')


class BatchRunStart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    stage: Stage
    subtitles: SubtitleChoice | None = Field(default=None, description='Prepare only: read the burned-in subtitles too (optional).')
    from_cut: int | None = Field(default=None, ge=1, description='Evaluate only: the first cut number (inclusive).')
    to_cut: int | None = Field(default=None, ge=1, description='Evaluate only: the last cut number (inclusive).')
    only: Literal['visual', 'beats', 'consistency'] | None = Field(default=None, description='Evaluate only: run just this step.')


def _require_episode(conn, episode_id):
    if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
        raise HTTPException(404, 'Episode not found')


def _require_run(conn, run_id):
    row = conn.execute('SELECT * FROM batch_runs WHERE id=?', (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Run not found')
    return row


@router.get('/api/episodes/{episode_id}/batch-runs/preview', summary='What a Run Multiple stage would do now, step by step, and the cuts that wait for you')
def preview_batch_run(episode_id: str, stage: Stage, subtitles: str | None = None, from_cut: int | None = None,
                      to_cut: int | None = None, only: Literal['visual', 'beats', 'consistency'] | None = None):
    options = {'subtitles': {'language': subtitles} if subtitles else None, 'from_cut': from_cut, 'to_cut': to_cut, 'only': only}
    with db.connect() as conn:
        conn.execute('BEGIN')
        _require_episode(conn, episode_id)
        try:
            return batch_runs.preview(conn, episode_id, stage, options)
        except batch_runs.Refused as exc:
            raise HTTPException(422, str(exc)) from exc


@router.post('/api/episodes/{episode_id}/batch-runs', summary='Start a Run Multiple stage: prepare (once), prepare_cuts or evaluate')
def start_batch_run(episode_id: str, body: BatchRunStart):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _require_episode(conn, episode_id)
        try:
            run_id = batch_runs.start(conn, episode_id, body.stage, body.model_dump(exclude={'stage'}))
        except batch_runs.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
    # The first steps (often skipped ones) are taken at once, off the request; the API's tick carries on from there.
    threading.Thread(target=batch_runs.advance, args=(run_id,), daemon=True).start()
    with db.connect() as conn:
        return batch_runs.view(conn, _require_run(conn, run_id))


@router.get('/api/episodes/{episode_id}/batch-runs', summary="The episode's Run Multiple runs, newest first (the running one included)")
def list_batch_runs(episode_id: str, limit: int = 20):
    with db.connect() as conn:
        conn.execute('BEGIN')
        _require_episode(conn, episode_id)
        return batch_runs.runs_of_episode(conn, episode_id, max(1, min(limit, 100)))


@router.get('/api/batch-runs/{run_id}', summary='One Run Multiple run: its steps, the running step\'s jobs, what each step found')
def get_batch_run(run_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN')
        return batch_runs.view(conn, _require_run(conn, run_id))


@router.post('/api/batch-runs/{run_id}/cancel', summary="Stop a running run: its current step's jobs are cancelled")
def cancel_batch_run(run_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _require_run(conn, run_id)
        try:
            cancelled = batch_runs.cancel(conn, run_id)
        except batch_runs.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
    worker.after_cancel(cancelled)
    with db.connect() as conn:
        return batch_runs.view(conn, _require_run(conn, run_id))
