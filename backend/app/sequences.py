"""HTTP API for sequences: ranges of cuts directed as one short video on the original audio."""
import json
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import db, media, sequence_edit, sequence_store as store, sequence_timeline, sequence_view

router = APIRouter(prefix='/api', tags=['Sequences'])


class SequenceInput(BaseModel):
    title: str = Field(default='', max_length=200)
    from_cut: int = Field(ge=1, description='First cut number as shown in the editor (cut_index + 1).')
    to_cut: int = Field(ge=1, description='Last cut number, inclusive.')
    instruction: str = Field(default='', max_length=8000, description='What this short video should achieve; given to the director model.')


class SequenceUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    instruction: str | None = Field(default=None, max_length=8000)


class SequenceRunInput(BaseModel):
    step: Literal['direct', 'bible', 'plan', 'shots', 'review']
    shot_keys: list[str] | None = Field(default=None, max_length=200, description='Only these shots (step shots); omitted writes every shot that needs it.')
    only_missing: bool = Field(default=True, description='With step shots and no shot_keys: skip shots that are written, current and pass the format check.')
    instruction: str = Field(default='', max_length=4000, description='A correction for the shots being written.')


class SequenceResultUpdate(BaseModel):
    content: dict[str, Any]


class SequenceMergeInput(BaseModel):
    shot_key: str


class SequenceSplitInput(BaseModel):
    shot_key: str
    time: float = Field(description='Episode time in seconds, at least 0.5 s inside the shot.')


def require(conn, sequence_id):
    sequence = store.get(conn, sequence_id)
    if not sequence:
        raise HTTPException(404, 'Sequence not found')
    return sequence


def detail_or_404(sequence_id):
    detail = sequence_view.detail(sequence_id)
    if detail is None:
        raise HTTPException(404, 'Sequence not found')
    return detail


def edit(action):
    """Run an edit in one write transaction and turn its refusals into HTTP errors."""
    try:
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            return action(conn)
    except sequence_edit.EditError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


@router.get('/episodes/{episode_id}/sequences')
def list_sequences(episode_id: str):
    with db.connect() as conn:
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        rows = [dict(row) for row in conn.execute('SELECT * FROM sequences WHERE episode_id=? ORDER BY start_time,created_at,rowid', (episode_id,))]
        return [sequence_view.summary(conn, row) for row in rows]


@router.post('/episodes/{episode_id}/sequences')
def create_sequence(episode_id: str, body: SequenceInput):
    """Create a sequence over an inclusive range of active cuts lasting at most three minutes."""
    sequence_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        if body.to_cut < body.from_cut:
            raise HTTPException(422, 'The last cut must not come before the first cut')
        cuts = store.range_cuts(conn, episode_id, body.from_cut - 1, body.to_cut - 1)
        if len(cuts) != body.to_cut - body.from_cut + 1:
            raise HTTPException(404, 'One or more of those cuts do not exist')
        length = cuts[-1]['end_time'] - cuts[0]['start_time']
        if length > sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001:
            raise HTTPException(422, f'Cuts {body.from_cut} to {body.to_cut} last {length:.1f} s; a sequence may last at most {sequence_timeline.MAX_SEQUENCE_SECONDS:g} s')
        title = body.title.strip() or f'Cuts {body.from_cut}–{body.to_cut}'
        conn.execute('INSERT INTO sequences(id,episode_id,title,instruction,first_cut_id,last_cut_id,start_time,end_time,cuts_json) VALUES (?,?,?,?,?,?,?,?,?)',
                     (sequence_id, episode_id, title, body.instruction.strip(), cuts[0]['id'], cuts[-1]['id'], cuts[0]['start_time'], cuts[-1]['end_time'], json.dumps(store.snapshot(cuts))))
    return detail_or_404(sequence_id)


@router.get('/sequences/{sequence_id}')
def get_sequence(sequence_id: str):
    return detail_or_404(sequence_id)


@router.patch('/sequences/{sequence_id}')
def update_sequence(sequence_id: str, body: SequenceUpdate):
    with db.connect() as conn:
        require(conn, sequence_id)
        fields = {key: value.strip() for key, value in body.model_dump().items() if value is not None}
        if 'title' in fields and not fields['title']:
            raise HTTPException(422, 'Give the sequence a title')
        if fields:
            conn.execute(f"UPDATE sequences SET {','.join(f'{key}=?' for key in fields)},updated_at=CURRENT_TIMESTAMP WHERE id=?", (*fields.values(), sequence_id))
    return detail_or_404(sequence_id)


@router.delete('/sequences/{sequence_id}')
def delete_sequence(sequence_id: str):
    """Delete a sequence with its results and chats. Its jobs are cancelled first; cuts and their analysis are untouched."""
    def action(conn):
        sequence = require(conn, sequence_id)
        sequence_edit.cancel(conn, sequence)
        conn.execute('DELETE FROM sequences WHERE id=?', (sequence_id,))
        return {'ok': True}
    return edit(action)


@router.post('/sequences/{sequence_id}/run')
def run_sequence(sequence_id: str, body: SequenceRunInput):
    """Queue a step of the direction; `direct` writes the bible and the shot plan and then stops for review."""
    job_ids = edit(lambda conn: sequence_edit.queue(conn, require(conn, sequence_id), body.step, body.shot_keys, body.only_missing, body.instruction))
    return {'queued': len(job_ids), 'job_ids': job_ids}


@router.post('/sequences/{sequence_id}/cancel')
def cancel_sequence_jobs(sequence_id: str):
    return {'cancelled': edit(lambda conn: sequence_edit.cancel(conn, require(conn, sequence_id)))}


def result_row(conn, result_id):
    row = conn.execute('SELECT * FROM sequence_results WHERE id=?', (result_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Result not found')
    return dict(row)


@router.put('/sequence-results/{result_id}', responses={409: {'description': 'This result already has a newer revision.'}, 422: {'description': 'The edited content does not pass validation; the detail lists the problems.'}})
def revise_sequence_result(result_id: str, body: SequenceResultUpdate):
    """Append a hand-edited revision of a bible, plan or shot. It is validated like model output; a shot is rendered and format-checked again."""
    def action(conn):
        row = result_row(conn, result_id)
        sequence_edit.revise(conn, require(conn, row['sequence_id']), row, body.content)
        return row['sequence_id']
    return detail_or_404(edit(action))


@router.get('/sequence-results/{result_id}/history')
def sequence_result_history(result_id: str):
    """Every revision of the result this id belongs to, newest first."""
    with db.connect() as conn:
        row = result_row(conn, result_id)
        return store.history(conn, row['sequence_id'], row['type'], row['shot_key'])


@router.post('/sequences/{sequence_id}/plan/merge')
def merge_shot(sequence_id: str, body: SequenceMergeInput):
    """Join a planned shot with the one after it (at most 15 s together)."""
    edit(lambda conn: sequence_edit.reshape(conn, require(conn, sequence_id), body.shot_key))
    return detail_or_404(sequence_id)


@router.post('/sequences/{sequence_id}/plan/split')
def split_shot(sequence_id: str, body: SequenceSplitInput):
    """Divide a planned shot in two at an episode time inside it."""
    edit(lambda conn: sequence_edit.reshape(conn, require(conn, sequence_id), body.shot_key, body.time))
    return detail_or_404(sequence_id)


def shot_frame_spec(sequence_id, shot_key=None, index=None, width=480):
    """Where a planned shot's initial frame lies in the source video, by shot key or one-based index."""
    with db.connect() as conn:
        sequence = require(conn, sequence_id)
        current = sequence_view.state(conn, sequence)
        episode = dict(conn.execute('SELECT id,project_id,original_ext FROM episodes WHERE id=?', (sequence['episode_id'],)).fetchone())
    shot = next((item for item in current['shots'] or [] if item['key'] == shot_key or item['index'] == index), None)
    if shot is None:
        raise HTTPException(404, 'That shot is not part of the current plan')
    cut = next(item for item in current['timeline']['cuts'] if item['id'] == shot['frame']['cut_id'])
    folder = media.episode_dir(episode['project_id'], episode['id'])
    original = folder / f"original{episode['original_ext']}"
    source = original if width is None and original.is_file() else folder / 'proxy.mp4'
    if not source.is_file():
        source = original
    if not source.is_file():
        raise HTTPException(404, 'The source video is unavailable')
    return {'source': source, 'time': shot['frame']['time'], 'start': cut['start_time'], 'end': cut['end_time'], 'width': width, 'last': False, 'seek_tolerance': .000002, 'probe_padding': 0}, episode['id']


@router.get('/sequences/{sequence_id}/shots/{shot_key}/frame', response_class=Response,
            responses={200: {'content': {'image/jpeg': {'schema': {'type': 'string', 'format': 'binary'}}}, 'description': 'Preview of the initial frame of a planned shot'}})
async def shot_frame(sequence_id: str, shot_key: str):
    from . import main
    spec, episode_id = shot_frame_spec(sequence_id, shot_key=shot_key)
    return await main.rendered_frame(spec, None, episode_id)
