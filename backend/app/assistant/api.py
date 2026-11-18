"""HTTP API for Cut Assistance sessions, streaming and proposals."""
import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import db, sequence_edit, sequence_store
from . import sequence_tools, skills, store, tools, transcript_edit  # sequence_tools registers the sequence chat's tools

router = APIRouter(tags=['Cut Assistance'])
WORKER_STALE_SECONDS = 30
MODEL_PATTERN = r'^[a-zA-Z0-9_./:@+-]+$'
PROVIDER_PATTERN = r'^[a-z][a-z0-9_-]{1,30}$'


class SessionInput(BaseModel):
    title: str = Field(default='', max_length=200)
    provider: str = Field(default='local', pattern=PROVIDER_PATTERN)
    model: str = Field(default='', max_length=200)
    thinking: bool = True


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    provider: str | None = Field(default=None, pattern=PROVIDER_PATTERN)
    model: str | None = Field(default=None, max_length=200)
    thinking: bool | None = None


class MessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=200, pattern=MODEL_PATTERN)
    thinking: bool = True
    skill: str | None = Field(default=None, max_length=60)
    gpu_fallback: bool = Field(default=False, description='When provider is local and visual work holds the GPU, answer with OpenRouter free instead of waiting.')


class RetryInput(BaseModel):
    provider: str | None = Field(default=None, pattern=PROVIDER_PATTERN)
    model: str | None = Field(default=None, min_length=1, max_length=200, pattern=MODEL_PATTERN)
    thinking: bool | None = None
    gpu_fallback: bool | None = None


# ---------------------------------------------------------------- helpers

def worker_state():
    row = db.one('SELECT providers_json,heartbeat FROM assistant_worker_state WHERE id=1')
    if not row:
        return False, []
    return time.time() - row['heartbeat'] <= WORKER_STALE_SECONDS, store.loads(row['providers_json'], [])


def require_cut(cut_id):
    cut = db.one('SELECT id FROM cuts WHERE id=?', (cut_id,))
    if not cut:
        raise HTTPException(404, 'Cut not found')
    return cut


def require_session(session_id):
    session = db.one('SELECT * FROM assistant_sessions WHERE id=?', (session_id,))
    if not session:
        raise HTTPException(404, 'Chat session not found')
    return session


def serialize_session(row):
    row = dict(row)
    row['thinking'] = bool(row['thinking'])
    return row


def proposal_stale(conn, proposal):
    if proposal['status'] != 'pending':
        return False
    if proposal['target'] == 'review_answers':
        row = conn.execute('SELECT content FROM cut_brief_instructions WHERE cut_id=?', (proposal['cut_id'],)).fetchone()
        return (row['content'] if row else '') != proposal['base_content']
    if proposal['target'] == 'transcript':
        return transcript_edit.plan_stale(conn, proposal['payload_json'])
    if proposal['target'] in tools.SEQUENCE_TARGETS:
        row = conn.execute('SELECT superseded_by FROM sequence_results WHERE id=?', (proposal['base_result_id'],)).fetchone()
        return not row or row['superseded_by'] is not None
    if proposal['base_result_id']:
        row = conn.execute('SELECT superseded_by FROM analysis_results WHERE id=?', (proposal['base_result_id'],)).fetchone()
        return not row or row['superseded_by'] is not None
    return tools.find_current(conn, proposal['cut_id'], proposal['target']) is not None


def serialize_proposal(conn, row):
    row = dict(row)
    row['stale'] = proposal_stale(conn, row)
    compare = row['diff_base'] if row.get('diff_base') is not None else row['base_content']
    old, new = tools.pretty(row['target'], compare), tools.pretty(row['target'], row['content'])
    row['diff'] = tools.diff_segments(old, new)
    row['diff_label'] = row.get('diff_label') or ('the saved version' if row['base_content'] else None)
    row['label'] = tools.LABELS.get(row['target'], row['target'])
    row['content'] = new
    row.pop('base_content')
    row.pop('diff_base', None)
    row.pop('payload_json', None)
    return row


def session_detail(session_id):
    with db.connect() as conn:
        session = conn.execute('SELECT * FROM assistant_sessions WHERE id=?', (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, 'Chat session not found')
        messages = [store.serialize_message(row) for row in conn.execute('SELECT * FROM assistant_messages WHERE session_id=? ORDER BY seq', (session_id,))]
        turns = [store.serialize_turn(row) for row in conn.execute('SELECT * FROM assistant_turns WHERE session_id=? ORDER BY created_at,rowid', (session_id,))]
        proposals = [serialize_proposal(conn, row) for row in conn.execute('SELECT * FROM assistant_proposals WHERE session_id=? ORDER BY created_at,rowid', (session_id,))]
    return {'session': serialize_session(session), 'messages': messages, 'turns': turns, 'proposals': proposals}


def validate_choice(provider, model, skill=None):
    online, providers = worker_state()
    if not online:
        raise HTTPException(503, 'The inference worker is offline; Cut Assistance needs it to run chat turns')
    match = next((item for item in providers if item.get('id') == provider), None)
    if not match:
        raise HTTPException(400, f'Unknown provider: {provider}')
    if not match.get('available'):
        raise HTTPException(400, match.get('detail') or f'{provider} is not available')
    models = [item.get('id') for item in match.get('models') or []]
    if models and not match.get('free_model') and model not in models:
        raise HTTPException(400, f'Unknown model for {provider}: {model}')
    if skill is not None and not skills.get(skill):
        raise HTTPException(400, f'Unknown skill: {skill}')


# -------------------------------------------------------------- endpoints

def default_provider(settings, providers):
    """Chat starts on ``auto`` (whichever server is free) once the remote PC is enabled, else on the saved text provider."""
    auto = next((item for item in providers if item.get('id') == 'auto' and item.get('available')), None)
    if auto and (auto.get('servers') or {}).get('remote'):
        return 'auto'
    return settings.get('text_provider') or 'local'


@router.get('/api/assistant/config')
def assistant_config():
    """Providers published by the inference worker, skills, and defaults."""
    online, providers = worker_state()
    settings = db.one('SELECT text_provider,local_thinking,openrouter_model FROM text_generation_settings WHERE id=1') or {}
    recent = [row['model'] for row in db.all_rows("SELECT model,MAX(updated_at) AS used FROM assistant_sessions WHERE provider!='local' AND model!='' GROUP BY provider,model ORDER BY used DESC LIMIT 12")]
    return {'worker_online': online, 'providers': providers,
            'skills': [{key: skill[key] for key in ('name', 'title', 'description', 'examples', 'scope')} for skill in skills.all_skills()],
            'defaults': {'provider': default_provider(settings, providers), 'openrouter_model': settings.get('openrouter_model') or 'openrouter/free', 'thinking': True},
            'recent_models': recent}


@router.get('/api/cuts/{cut_id}/assistant/sessions')
def list_sessions(cut_id: str):
    require_cut(cut_id)
    rows = db.all_rows('''SELECT s.*,
        (SELECT COUNT(*) FROM assistant_messages m WHERE m.session_id=s.id AND m.role='user') AS message_count,
        (SELECT t.status FROM assistant_turns t WHERE t.session_id=s.id ORDER BY t.created_at DESC, t.rowid DESC LIMIT 1) AS last_status,
        (SELECT COUNT(*) FROM assistant_proposals p WHERE p.session_id=s.id AND p.status='pending') AS pending_proposals
        FROM assistant_sessions s WHERE s.cut_id=? AND s.sequence_id IS NULL ORDER BY s.updated_at DESC, s.rowid DESC''', (cut_id,))
    return [serialize_session(row) for row in rows]


@router.post('/api/cuts/{cut_id}/assistant/sessions')
def create_session(cut_id: str, body: SessionInput):
    require_cut(cut_id)
    session_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('INSERT INTO assistant_sessions(id,cut_id,title,provider,model,thinking) VALUES (?,?,?,?,?,?)',
                     (session_id, cut_id, body.title.strip(), body.provider, body.model, int(body.thinking)))
    return session_detail(session_id)


def require_sequence(sequence_id):
    with db.connect() as conn:
        sequence = sequence_store.get(conn, sequence_id)
    if not sequence:
        raise HTTPException(404, 'Sequence not found')
    return sequence


@router.get('/api/sequences/{sequence_id}/assistant/sessions')
def list_sequence_sessions(sequence_id: str):
    """Chats about a sequence: its bible, shot plan and shot prompts."""
    require_sequence(sequence_id)
    rows = db.all_rows('''SELECT s.*,
        (SELECT COUNT(*) FROM assistant_messages m WHERE m.session_id=s.id AND m.role='user') AS message_count,
        (SELECT t.status FROM assistant_turns t WHERE t.session_id=s.id ORDER BY t.created_at DESC, t.rowid DESC LIMIT 1) AS last_status,
        (SELECT COUNT(*) FROM assistant_proposals p WHERE p.session_id=s.id AND p.status='pending') AS pending_proposals
        FROM assistant_sessions s WHERE s.sequence_id=? ORDER BY s.updated_at DESC, s.rowid DESC''', (sequence_id,))
    return [serialize_session(row) for row in rows]


@router.post('/api/sequences/{sequence_id}/assistant/sessions')
def create_sequence_session(sequence_id: str, body: SessionInput):
    sequence = require_sequence(sequence_id)
    session_id = str(uuid.uuid4())
    with db.connect() as conn:
        # cut_id anchors the session to the project; a sequence chat reads any cut of its range by number.
        conn.execute('INSERT INTO assistant_sessions(id,cut_id,sequence_id,title,provider,model,thinking) VALUES (?,?,?,?,?,?,?)',
                     (session_id, sequence['first_cut_id'], sequence_id, body.title.strip(), body.provider, body.model, int(body.thinking)))
    return session_detail(session_id)


@router.get('/api/assistant/sessions/{session_id}')
def get_session(session_id: str):
    return session_detail(session_id)


@router.patch('/api/assistant/sessions/{session_id}')
def update_session(session_id: str, body: SessionUpdate):
    require_session(session_id)
    fields = {key: (int(value) if isinstance(value, bool) else value.strip() if key == 'title' else value)
              for key, value in body.model_dump().items() if value is not None}
    if fields:
        with db.connect() as conn:
            conn.execute(f"UPDATE assistant_sessions SET {','.join(f'{key}=?' for key in fields)} WHERE id=?", (*fields.values(), session_id))
    return serialize_session(db.one('SELECT * FROM assistant_sessions WHERE id=?', (session_id,)))


@router.delete('/api/assistant/sessions/{session_id}')
def delete_session(session_id: str):
    """Delete a session; a running turn notices on its next checkpoint and stops."""
    with db.connect() as conn:
        if not conn.execute('DELETE FROM assistant_sessions WHERE id=?', (session_id,)).rowcount:
            raise HTTPException(404, 'Chat session not found')
    return {'ok': True}


@router.post('/api/assistant/sessions/{session_id}/messages')
def send_message(session_id: str, body: MessageInput):
    """Store the user's message and queue a turn for the inference worker."""
    session = require_session(session_id)
    validate_choice(body.provider, body.model, body.skill)
    content = body.content.strip()
    if not content:
        raise HTTPException(400, 'Write a message first')
    turn_id, message_id = str(uuid.uuid4()), str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute("SELECT 1 FROM assistant_turns WHERE session_id=? AND status IN ('queued','running')", (session_id,)).fetchone():
            raise HTTPException(409, 'Wait for the current reply to finish or stop it first')
        conn.execute('INSERT INTO assistant_turns(id,session_id,provider,model,thinking,skill,gpu_fallback) VALUES (?,?,?,?,?,?,?)',
                     (turn_id, session_id, body.provider, body.model, int(body.thinking), body.skill, int(body.gpu_fallback)))
        conn.execute("INSERT INTO assistant_messages(id,session_id,turn_id,seq,role,content) VALUES (?,?,?,?,'user',?)",
                     (message_id, session_id, turn_id, store.next_seq(conn, session_id), content))
        title = session['title'] or (content.splitlines()[0][:80] if content else 'New chat')
        conn.execute('UPDATE assistant_sessions SET title=?,provider=?,model=?,thinking=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                     (title, body.provider, body.model, int(body.thinking), session_id))
        turn = store.serialize_turn(conn.execute('SELECT * FROM assistant_turns WHERE id=?', (turn_id,)).fetchone())
        message = store.serialize_message(conn.execute('SELECT * FROM assistant_messages WHERE id=?', (message_id,)).fetchone())
    return {'turn': turn, 'message': message}


@router.post('/api/assistant/turns/{turn_id}/cancel')
def cancel_turn(turn_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        turn = conn.execute('SELECT status FROM assistant_turns WHERE id=?', (turn_id,)).fetchone()
        if not turn:
            raise HTTPException(404, 'Turn not found')
        if turn['status'] == 'queued':
            store.bump(conn, turn_id, status='cancelled', stage='Stopped', finished_at=time.time())
        elif turn['status'] == 'running':
            store.bump(conn, turn_id, cancel_requested=1, stage='Stopping')
        row = conn.execute('SELECT * FROM assistant_turns WHERE id=?', (turn_id,)).fetchone()
    return store.serialize_turn(row)


@router.post('/api/assistant/turns/{turn_id}/retry')
def retry_turn(turn_id: str, body: RetryInput | None = None):
    """Regenerate the latest reply, optionally with another model."""
    body = body or RetryInput()
    with db.connect() as conn:
        turn = conn.execute('SELECT * FROM assistant_turns WHERE id=?', (turn_id,)).fetchone()
        if not turn:
            raise HTTPException(404, 'Turn not found')
    provider = body.provider or turn['provider']
    model = body.model or turn['model']
    thinking = turn['thinking'] if body.thinking is None else int(body.thinking)
    gpu_fallback = turn['gpu_fallback'] if body.gpu_fallback is None else int(body.gpu_fallback)
    validate_choice(provider, model)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        turn = conn.execute('SELECT * FROM assistant_turns WHERE id=?', (turn_id,)).fetchone()
        if not turn or turn['status'] in store.ACTIVE:
            raise HTTPException(409, 'This reply is still running')
        latest = conn.execute('SELECT id FROM assistant_turns WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1', (turn['session_id'],)).fetchone()
        if latest['id'] != turn_id:
            raise HTTPException(409, 'Only the latest reply can be regenerated')
        conn.execute("DELETE FROM assistant_messages WHERE turn_id=? AND role!='user'", (turn_id,))
        conn.execute("DELETE FROM assistant_proposals WHERE turn_id=? AND status='pending'", (turn_id,))
        store.bump(conn, turn_id, status='queued', stage='Queued', error=None, cancel_requested=0, attempt_count=0, usage_json='{}',
                   started_at=None, finished_at=None, provider=provider, model=model, thinking=thinking, gpu_fallback=gpu_fallback, lease_owner=None, lease_until=None)
        conn.execute('UPDATE assistant_sessions SET provider=?,model=?,thinking=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (provider, model, thinking, turn['session_id']))
    return session_detail(turn['session_id'])


def turn_snapshot(turn_id):
    with db.connect() as conn:
        turn = conn.execute('SELECT * FROM assistant_turns WHERE id=?', (turn_id,)).fetchone()
        if not turn:
            return None, [], []
        messages = [store.serialize_message(row) for row in conn.execute('SELECT * FROM assistant_messages WHERE turn_id=? ORDER BY seq', (turn_id,))]
        proposals = [serialize_proposal(conn, row) for row in conn.execute('SELECT * FROM assistant_proposals WHERE turn_id=? ORDER BY created_at,rowid', (turn_id,))]
    return store.serialize_turn(turn), messages, proposals


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@router.get('/api/assistant/turns/{turn_id}/stream')
async def stream_turn(turn_id: str, request: Request):
    """Server-sent events for one turn: snapshot, message, delta, proposal, turn and done."""
    if not await asyncio.to_thread(db.one, 'SELECT id FROM assistant_turns WHERE id=?', (turn_id,)):
        raise HTTPException(404, 'Turn not found')

    async def events():
        version, sent, proposals_seen, turn_seen = None, {}, {}, None
        last_ping = time.monotonic()
        yield 'retry: 1500\n\n'
        while True:
            if await request.is_disconnected():
                return
            row = await asyncio.to_thread(db.one, 'SELECT version,status FROM assistant_turns WHERE id=?', (turn_id,))
            if not row:
                yield sse('gone', {'detail': 'This turn was deleted'})
                return
            if row['version'] != version:
                version = row['version']
                turn, messages, proposals = await asyncio.to_thread(turn_snapshot, turn_id)
                if turn is None:
                    yield sse('gone', {'detail': 'This turn was deleted'})
                    return
                ids = {message['id'] for message in messages}
                if not sent or any(message_id not in ids for message_id in sent):
                    yield sse('snapshot', {'turn': turn, 'messages': messages, 'proposals': proposals})
                    sent = {message['id']: (len(message['content']), len(message['reasoning']), message['streaming']) for message in messages}
                    proposals_seen = {proposal['id']: proposal['status'] for proposal in proposals}
                    turn_seen = (turn['status'], turn['stage'], turn['cancel_requested'])
                else:
                    for message in messages:
                        previous = sent.get(message['id'])
                        current = (len(message['content']), len(message['reasoning']), message['streaming'])
                        if previous == current:
                            continue
                        if previous and message['streaming'] and current[0] >= previous[0] and current[1] >= previous[1]:
                            yield sse('delta', {'id': message['id'], 'content': message['content'][previous[0]:], 'reasoning': message['reasoning'][previous[1]:]})
                        else:
                            yield sse('message', message)
                        sent[message['id']] = current
                    for proposal in proposals:
                        if proposals_seen.get(proposal['id']) != proposal['status']:
                            proposals_seen[proposal['id']] = proposal['status']
                            yield sse('proposal', proposal)
                    # Deltas advance the version too; only send the turn when its state changes.
                    if (turn['status'], turn['stage'], turn['cancel_requested']) != turn_seen:
                        turn_seen = (turn['status'], turn['stage'], turn['cancel_requested'])
                        yield sse('turn', turn)
                if turn['status'] not in store.ACTIVE:
                    yield sse('done', turn)
                    return
            if time.monotonic() - last_ping > 15:
                last_ping = time.monotonic()
                yield ': ping\n\n'
            await asyncio.sleep(0.1)

    return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@router.post('/api/assistant/proposals/{proposal_id}/apply')
def apply_proposal(proposal_id: str):
    """Save a proposal as a new revision (origin assistant_edit) if its base is still current."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        proposal = conn.execute('SELECT * FROM assistant_proposals WHERE id=?', (proposal_id,)).fetchone()
        if not proposal:
            raise HTTPException(404, 'Proposal not found')
        if proposal['status'] != 'pending':
            raise HTTPException(409, f"This proposal was already {proposal['status']}")
        if not conn.execute('SELECT 1 FROM cuts WHERE id=?', (proposal['cut_id'],)).fetchone():
            raise HTTPException(404, 'Cut not found')
        if proposal_stale(conn, proposal):
            raise HTTPException(409, f"The {'transcript' if proposal['target'] == 'transcript' else 'result'} changed after this proposal was made; ask the assistant to propose it again")
        provenance = {'assistant_edit': {'proposal_id': proposal_id, 'session_id': proposal['session_id'], 'provider': proposal['provider'],
                                         'model': proposal['model_id'], 'summary': proposal['summary']}}
        applied_id = None
        if proposal['target'] == 'transcript':
            # Revisions of the episode's transcript segments, not of an analysis result.
            try:
                created = transcript_edit.apply(conn, proposal['payload_json'])
            except transcript_edit.PlanError as exc:
                raise HTTPException(409, str(exc)) from exc
            conn.execute('UPDATE assistant_proposals SET payload_json=? WHERE id=?', (json.dumps({**json.loads(proposal['payload_json']), 'applied_segment_ids': created}), proposal_id))
        elif proposal['target'] == 'review_answers':
            conn.execute('INSERT INTO cut_brief_instructions(cut_id,content) VALUES (?,?) ON CONFLICT(cut_id) DO UPDATE SET content=excluded.content,updated_at=CURRENT_TIMESTAMP',
                         (proposal['cut_id'], proposal['content']))
        elif proposal['target'] in tools.SEQUENCE_TARGETS:
            # A revision of a sequence result, validated and (for a shot) rendered again exactly like a hand edit.
            payload = json.loads(proposal['payload_json'])
            sequence = sequence_store.get(conn, payload['sequence_id'])
            base = conn.execute('SELECT * FROM sequence_results WHERE id=?', (proposal['base_result_id'],)).fetchone()
            if not sequence or not base:
                raise HTTPException(404, 'Sequence not found')
            content = payload[{'sequence_shot': 'direction', 'sequence_plan': 'plan', 'sequence_bible': 'bible'}[proposal['target']]]
            try:
                applied_id = sequence_edit.revise(conn, sequence, dict(base), content, origin='assistant_edit', provenance=provenance)
            except sequence_edit.EditError as exc:
                raise HTTPException(exc.status, exc.detail) from exc
        else:
            base = conn.execute('SELECT * FROM analysis_results WHERE id=?', (proposal['base_result_id'],)).fetchone() if proposal['base_result_id'] else None
            structured = proposal['target'] in tools.STRUCTURED
            settings = {**store.loads(base['settings_json'], {}), **provenance} if structured and base else provenance
            applied_id = str(uuid.uuid4())
            conn.execute('''INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,model_id,settings_json,context_json,parent_id)
                            VALUES (?,?,?,?,?,'assistant_edit',?,?,?,?)''',
                         (applied_id, proposal['cut_id'], proposal['kind'], proposal['title'], proposal['content'], proposal['model_id'],
                          json.dumps(settings, ensure_ascii=False), base['context_json'] if structured and base else '{}', base['id'] if base else None))
            if base:
                conn.execute('UPDATE analysis_results SET superseded_by=? WHERE id=?', (applied_id, base['id']))
            if proposal['target'] == 'keyframe_analysis':
                # The beats are built from the keyframe analysis: an edited one is written into new beats.
                from .. import performance_beats
                episode_id = conn.execute('SELECT episode_id FROM cuts WHERE id=?', (proposal['cut_id'],)).fetchone()['episode_id']
                performance_beats.ensure_queued(conn, episode_id, proposal['cut_id'])
        conn.execute("UPDATE assistant_proposals SET status='applied',applied_result_id=?,resolved_at=CURRENT_TIMESTAMP WHERE id=?", (applied_id, proposal_id))
        if proposal['turn_id']:
            store.bump(conn, proposal['turn_id'])
        result = serialize_proposal(conn, conn.execute('SELECT * FROM assistant_proposals WHERE id=?', (proposal_id,)).fetchone())
    return result


@router.post('/api/assistant/proposals/{proposal_id}/discard')
def discard_proposal(proposal_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        proposal = conn.execute('SELECT status,turn_id FROM assistant_proposals WHERE id=?', (proposal_id,)).fetchone()
        if not proposal:
            raise HTTPException(404, 'Proposal not found')
        if proposal['status'] != 'pending':
            raise HTTPException(409, f"This proposal was already {proposal['status']}")
        conn.execute("UPDATE assistant_proposals SET status='discarded',resolved_at=CURRENT_TIMESTAMP WHERE id=?", (proposal_id,))
        if proposal['turn_id']:
            store.bump(conn, proposal['turn_id'])
        return serialize_proposal(conn, conn.execute('SELECT * FROM assistant_proposals WHERE id=?', (proposal_id,)).fetchone())

