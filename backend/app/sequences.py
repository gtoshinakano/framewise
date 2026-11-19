"""HTTP API for sequences: ranges of cuts directed as one short video on the original audio."""
import json
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import db, media, sequence_compile, sequence_edit, worker, sequence_store as store, sequence_timeline, sequence_view

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
    step: Literal['compile', 'direct', 'bible', 'plan', 'shots', 'review']
    shot_keys: list[str] | None = Field(default=None, max_length=200, description='Only these shots (step shots); omitted writes every shot that needs it.')
    only_missing: bool = Field(default=True, description='With step shots and no shot_keys: skip shots that are written, current and pass the format check.')
    instruction: str = Field(default='', max_length=4000, description='A correction for the shots being written.')
    instruction_source: Literal['user', 'review'] = Field(
        default='user', description="user: the user's own instruction, which every later rewrite of that shot keeps following. review: a suggestion of the automatic review, used for this rewrite only.")


class SequenceResultUpdate(BaseModel):
    content: dict[str, Any]


class ShotBoundaryInput(BaseModel):
    after_cut_id: str = Field(description='The cut before the hard cut whose shot boundary moves.')
    choice: Literal['keep_cut', 'before_line', 'after_line'] = Field(
        description='keep_cut: the boundary stays on the hard cut (speech crossing it is divided there). before_line / after_line: it moves into the silence just before or after the phrase the hard cut falls into, so the phrase stays in one shot.')


class ShotPointInput(BaseModel):
    after_cut_id: str = Field(description='The cut before the hard cut whose shot boundary the user placed.')
    time: float = Field(description='Where the shot boundary sits, in episode seconds; anywhere inside the two cuts beside the hard cut, even inside speech.')


class CompilationConfirmInput(BaseModel):
    result_id: str = Field(description='The current dialogue revision the user reviewed.')
    points: list[ShotPointInput] | None = Field(default=None, max_length=500)


class CompilationSaveInput(CompilationConfirmInput):
    content: dict[str, Any] | None = None
    confirm: bool = False


class ShotPointsInput(BaseModel):
    points: list[ShotPointInput] = Field(default_factory=list, max_length=500, description='Every boundary the user placed; a hard cut left out is decided by the director. An empty list hands every boundary back.')


class VoiceSegment(BaseModel):
    start: float = Field(alias='from', ge=0, description='Seconds from the start of the shot; the first segment always starts at 0.')
    delivery: Literal['on_screen', 'off_screen', 'inner_voice', 'narrative'] = Field(
        description='on_screen: visible, lip-synced. off_screen: not in the picture. inner_voice: visible, the mouth does not move. narrative: too far for lip sync; the words play over the picture.')


class ShotVoicesInput(BaseModel):
    shot_key: str
    speaker: str = Field(description='The speaker as the shot names it: the character name, Narrator, or the voice label.')
    segments: list[VoiceSegment] | None = Field(default=None, max_length=20, description='How the speaker is heard from each start until the next; null lets the director decide again.')


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
        # The dialogue is compiled first, so it is ready to review by the time the user opens the sequence.
        sequence_edit.queue_compile(conn, store.get(conn, sequence_id))
        worker.sync_episode_status(conn, episode_id)
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
        jobs = sequence_edit.cancel(conn, sequence)
        conn.execute('DELETE FROM sequences WHERE id=?', (sequence_id,))
        return jobs
    worker.after_cancel(edit(action))
    return {'ok': True}


@router.post('/sequences/{sequence_id}/run')
def run_sequence(sequence_id: str, body: SequenceRunInput):
    """Queue a step of the direction; `direct` writes the bible and the shot plan and then stops for review."""
    job_ids = edit(lambda conn: sequence_edit.queue(conn, require(conn, sequence_id), body.step, body.shot_keys, body.only_missing, body.instruction, instruction_source=body.instruction_source))
    return {'queued': len(job_ids), 'job_ids': job_ids}


@router.post('/sequences/{sequence_id}/extend', responses={409: {'description': 'No cut follows the sequence or its end was archived.'},
                                                    422: {'description': 'With the next cut the sequence would last more than three minutes.'}})
def extend_sequence(sequence_id: str):
    """Add the episode's next cut to the end of the sequence, for speech that runs on past its last cut.

    Only the added cut's dialogue is compiled and appended; the reviewed dialogue keeps its edits. Once the
    dialogue is saved again (at once when nobody speaks in the added cut), the sequence is directed again and,
    if it had shot prompts, every shot is written again.
    """
    edit(lambda conn: sequence_edit.extend(conn, require(conn, sequence_id)))
    return detail_or_404(sequence_id)


@router.post('/sequences/{sequence_id}/cancel')
def cancel_sequence_jobs(sequence_id: str):
    jobs = edit(lambda conn: sequence_edit.cancel(conn, require(conn, sequence_id)))
    worker.after_cancel(jobs)
    return {'cancelled': len(jobs)}


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


@router.post('/sequence-results/{result_id}/review-issues/{index}/dismiss')
def dismiss_review_issue(result_id: str, index: int):
    """Mark a continuity suggestion as requiring no action."""
    def action(conn):
        row = result_row(conn, result_id)
        if row['type'] != 'review' or row['superseded_by']:
            raise sequence_edit.EditError(409, 'The continuity review has changed')
        content = store.loads(row['content'], {})
        issues = content.get('issues', [])
        if not 0 <= index < len(issues):
            raise sequence_edit.EditError(404, 'Review issue not found')
        if not issues[index].get('dismissed'):
            issues[index]['dismissed'] = True
            store.save(conn, row['sequence_id'], 'review', content, origin='manual_edit', model_id=row['model_id'],
                       settings=store.loads(row['settings_json'], {}), expected_current=result_id)
        return row['sequence_id']

    return detail_or_404(edit(action))


@router.put('/sequences/{sequence_id}/compilation')
def save_compilation(sequence_id: str, body: CompilationSaveInput):
    """Save dialogue edits, placed shot boundaries and optional confirmation in one transaction."""
    def action(conn):
        sequence = require(conn, sequence_id)
        row = store.current(conn, sequence_id, 'compilation')
        conflict = 'The dialogue has a newer revision. Your draft is still open.'
        if not row:
            raise sequence_edit.EditError(409, 'The dialogue is no longer available. Your draft is still open.')
        result_id = row['id']
        if result_id != body.result_id:
            # Recover a retry when the dialogue write already landed, including from the older two-request editor.
            if row['parent_id'] != body.result_id or row['origin'] != 'manual_edit' or body.content is None:
                raise sequence_edit.EditError(409, conflict)
            current = sequence_view.state(conn, sequence)
            if current['cuts'] is None:
                raise sequence_edit.EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
            roster = {person['name']: person['id'] for person in current['ctx']['characters']}
            try:
                normal = sequence_compile.validate(body.content, current['timeline']['duration'], roster)
            except ValueError as exc:
                raise sequence_edit.EditError(422, str(exc)) from exc
            saved = store.loads(row['content'], {})
            if any(saved.get(key) != value for key, value in normal.items()):
                raise sequence_edit.EditError(409, conflict)
        elif body.content is not None:
            result_id = sequence_edit.revise(conn, sequence, row, body.content)
            sequence = require(conn, sequence_id)

        points = [point.model_dump() for point in body.points] if body.points is not None else None
        if body.confirm:
            sequence_edit.confirm_compilation(conn, sequence, result_id, points)
        elif points is not None:
            sequence_edit.set_points(conn, sequence, points)

    edit(action)
    return detail_or_404(sequence_id)


@router.post('/sequences/{sequence_id}/compilation/confirm', responses={409: {'description': 'The dialogue has a newer revision than the one confirmed.'}})
def confirm_compilation(sequence_id: str, body: CompilationConfirmInput):
    """Confirm the reviewed dialogue. Direction and the shots use it instead of the raw transcript from now on."""
    edit(lambda conn: sequence_edit.confirm_compilation(conn, require(conn, sequence_id), body.result_id,
                                                       [point.model_dump() for point in body.points] if body.points is not None else None))
    return detail_or_404(sequence_id)


@router.post('/sequences/{sequence_id}/compilation/use-source')
def use_source_dialogue(sequence_id: str):
    """Skip the review: save the original transcript, uncorrected, as the dialogue and confirm it."""
    edit(lambda conn: sequence_edit.use_source_dialogue(conn, require(conn, sequence_id)))
    return detail_or_404(sequence_id)


@router.put('/sequences/{sequence_id}/boundaries', responses={404: {'description': 'No hard cut follows that cut in this sequence.'}, 409: {'description': 'The sequence has no usable shot plan, or its cuts changed.'},
                                                         422: {'description': 'That choice is not possible at this hard cut; the detail says why.'}})
def set_shot_boundary(sequence_id: str, body: ShotBoundaryInput):
    """Choose where the shot boundary sits at one hard cut. Every cut stays one shot; the two shots beside the boundary must be written again."""
    edit(lambda conn: sequence_edit.set_boundary(conn, require(conn, sequence_id), body.after_cut_id, body.choice))
    return detail_or_404(sequence_id)


@router.put('/sequences/{sequence_id}/shot-points', responses={409: {'description': 'A sequence boundary was archived.'},
                                                          422: {'description': 'A point is outside the sequence, or the shots would be out of order, longer than one take or without their own picture.'}})
def set_shot_points(sequence_id: str, body: ShotPointsInput):
    """Save the shot boundaries the user placed on the Dialogue timeline. They are final for the director; with a plan, its shots move onto them
    (the moved shots must be written again) and a boundary handed back gets the app's choice until the shots are planned again."""
    edit(lambda conn: sequence_edit.set_points(conn, require(conn, sequence_id), [item.model_dump() for item in body.points]))
    return detail_or_404(sequence_id)


@router.put('/sequences/{sequence_id}/voices', responses={409: {'description': 'The sequence is not directed, or its cuts changed.'}, 422: {'description': 'The speaker is not heard in that shot, or a segment is invalid.'}})
def set_shot_voices(sequence_id: str, body: ShotVoicesInput):
    """Decide how one speaker is heard in one shot; the plan keeps it and a rewrite of the shot is queued."""
    segments = None if body.segments is None else [{'from': item.start, 'delivery': item.delivery} for item in body.segments]
    edit(lambda conn: sequence_edit.set_voices(conn, require(conn, sequence_id), body.shot_key, body.speaker, segments))
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


@router.get('/sequences/{sequence_id}/shots/{shot_key}/history')
def shot_prompt_history(sequence_id: str, shot_key: str):
    """Every version of a shot's prompt, newest first: why it was written (your instruction, a voice change, a hand edit or an
    Assistant request) and what changed against the version before. Also the rewrite in progress and the Assistant's unapplied proposals."""
    with db.connect() as conn:
        require(conn, sequence_id)
        return sequence_view.shot_history(conn, sequence_id, shot_key)


@router.get('/sequences/{sequence_id}/shots/{shot_key}/frame', response_class=Response,
            responses={200: {'content': {'image/jpeg': {'schema': {'type': 'string', 'format': 'binary'}}}, 'description': 'Preview of the initial frame of a planned shot'}})
async def shot_frame(sequence_id: str, shot_key: str):
    from . import main
    spec, episode_id = shot_frame_spec(sequence_id, shot_key=shot_key)
    return await main.rendered_frame(spec, None, episode_id)
