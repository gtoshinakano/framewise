"""Claims queued chat turns, streams provider output into SQLite and runs tools.

Turns use the same lease discipline as jobs: a 60-second renewable lease and a
bounded number of attempts. An interrupted turn restarts from its user message
(partial assistant output and unapplied proposals of that turn are discarded).
Deltas are flushed to ``assistant_messages`` a few times per second and every
flush advances ``assistant_turns.version`` so the API stream can forward them.
"""
import contextlib
import json
import logging
import threading
import time
import uuid

from .. import db, local_text_runtime
from . import compact, providers, sequence_tools, skills, store, tools

log = logging.getLogger(__name__)
LEASE_SECONDS = 60
MAX_ATTEMPTS = 2
MAX_STEPS = 10
FLUSH_SECONDS = 0.12
CHECK_SECONDS = 0.4

SYSTEM_PROMPT = """You are Cut Assistance in Framewise, a video analysis tool for recreating anime/film cuts. You help one production user review the analysis of ONE cut and fix prompts derived from it.

Ground rules
- Saved results are model output or human edits and can be wrong. Base claims on tool results, say which result supports them, and never invent visual details, identities, motion or dialogue.
- Read with tools instead of guessing, but read only what the question needs (use `fields`). The cut card below already lists which results exist.
- Data times are episode-relative seconds unless a tool says clip-relative; prompts and timelines use clip-relative seconds (0 = cut start).
- Subtitles and OCR text are reference only. Never put them into image or animation prompts.
- Human review answers and instructions override model output.
- You cannot change stored data directly. Use `propose_result_edit` for results and `propose_transcript_edit` for the audio transcript; the user applies the proposal. Never claim something is saved.
- Before specialised work, load the matching skill with `load_skill` (once per conversation unless it was already loaded):
{skills}
- Reply in the user's language, in concise Markdown. Put prompts meant for copying in a fenced ```text block.

## Cut card
{card}"""


SEQUENCE_PROMPT = """You are Cut Assistance in Framewise, working on ONE sequence: a range of cuts that is directed as one short video on the original audio and generated shot by shot with MiniMax H3. You help one production user review and correct the direction of that sequence: its bible, its shot plan and the prompts of its shots.

Ground rules
- The direction was written by a model from analysis results; both can be wrong. Base claims on tool results, say what supports them, and never invent visual details, identities, motion or dialogue.
- Read with tools instead of guessing, and read only what the question needs. The sequence card below already lists what exists.
- Plan times are seconds from the start of the sequence; times inside a shot are seconds from the start of that shot.
- A shot's final prompt is rendered by the app from the shot's direction (a JSON object). You change the direction, never the prompt text: the app writes reference labels, timecodes, spoken words and audio lines.
- You cannot change stored data directly. Use `propose_shot_edit`, `propose_plan_edit` and `propose_bible_edit`; the user applies each proposal. Never claim something is saved.
- A change to the plan or the bible makes the affected shots stale; tell the user which shots must be written again.
- Before proposing a change, load the matching skill with `load_skill` (once per conversation unless it was already loaded):
{skills}
- Reply in the user's language, in concise Markdown.

## Sequence card
{card}"""
KEPT_TOOL_RESULTS = ('propose_result_edit', 'propose_transcript_edit', 'propose_shot_edit', 'propose_plan_edit', 'propose_bible_edit')


class Aborted(Exception):
    """The turn was deleted or its lease moved to another worker."""


def system_prompt(session):
    """The standing instructions of a chat: about its sequence when it has one, otherwise about its cut."""
    if session.get('sequence_id'):
        return SEQUENCE_PROMPT.format(skills=skills.index_text('sequence'), card=sequence_tools.sequence_card(session['sequence_id']))
    return SYSTEM_PROMPT.format(skills=skills.index_text(), card=tools.cut_card(session['cut_id']))


# ------------------------------------------------------------------ claiming

def claim_turn():
    now = time.time()
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('''UPDATE assistant_turns SET status='failed',stage='Failed',error='The turn was interrupted; retry it',
                        lease_owner=NULL,lease_until=NULL,finished_at=?,version=version+1
                        WHERE status='running' AND (lease_until IS NULL OR lease_until<=?) AND attempt_count>=?''', (now, now, MAX_ATTEMPTS))
        row = conn.execute('''SELECT * FROM assistant_turns WHERE status='queued' OR (status='running' AND (lease_until IS NULL OR lease_until<=?))
                              ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at, rowid LIMIT 1''', (now,)).fetchone()
        if not row:
            return None
        owner = str(uuid.uuid4())
        if row['status'] == 'running':
            # Restart from the user message; keep nothing half-written.
            conn.execute("DELETE FROM assistant_messages WHERE turn_id=? AND role!='user'", (row['id'],))
            conn.execute("DELETE FROM assistant_proposals WHERE turn_id=? AND status='pending'", (row['id'],))
        store.bump(conn, row['id'], status='running', stage='Starting', lease_owner=owner, lease_until=now + LEASE_SECONDS,
                   attempt_count=row['attempt_count'] + 1, started_at=row['started_at'] or now, error=None, usage_json='{}')
        return row['id'], owner


# ------------------------------------------------------------------- history

def history_messages(session_id, current_turn_id, budget_tokens):
    """Rebuild OpenAI-style history; compact older turns only when over budget."""
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute('SELECT * FROM assistant_messages WHERE session_id=? ORDER BY seq', (session_id,))]
    answered = {row['tool_call_id'] for row in rows if row['role'] == 'tool'}
    turns, order = {}, []
    for row in rows:
        turn_id = row['turn_id'] or row['id']
        if turn_id not in turns:
            turns[turn_id] = []
            order.append(turn_id)
        turns[turn_id].append(row)

    def render(turn_rows, elide_tools):
        out, has_answer = [], False
        for row in turn_rows:
            if row['role'] == 'user':
                out.append({'role': 'user', 'content': row['content']})
            elif row['role'] == 'assistant':
                calls = [call for call in store.loads(row['tool_calls_json'], []) if call.get('id') in answered]
                if calls:
                    out.append({'role': 'assistant', 'content': row['content'] or '',
                                'tool_calls': [{'id': call['id'], 'type': 'function', 'function': {'name': call['name'], 'arguments': call['arguments']}} for call in calls]})
                elif row['content'].strip():
                    out.append({'role': 'assistant', 'content': row['content']})
                    has_answer = True
            elif row['role'] == 'tool':
                content = row['content']
                if elide_tools and row['tool_name'] not in KEPT_TOOL_RESULTS:
                    content = compact.dumps({'elided': True, 'summary': row['tool_summary'] or row['tool_name'], 'note': 'Call the tool again if you need this data.'})
                out.append({'role': 'tool', 'tool_call_id': row['tool_call_id'], 'name': row['tool_name'], 'content': content})
        if turn_rows and turn_rows[0]['turn_id'] != current_turn_id and not has_answer:
            out.append({'role': 'assistant', 'content': '[No answer: this turn was cancelled or failed.]'})
        return out

    def size(blocks):
        return sum(compact.estimate_tokens(json.dumps(block, ensure_ascii=False)) for turn in blocks for block in turn)

    rendered = [render(turns[turn_id], False) for turn_id in order]
    if size(rendered) > budget_tokens:
        rendered = [render(turns[turn_id], turn_id != current_turn_id) for turn_id in order]
    omitted = 0
    while len(rendered) > 1 and size(rendered) > budget_tokens:
        rendered.pop(0)
        omitted += 1
    messages = [block for turn in rendered for block in turn]
    if omitted:
        messages.insert(0, {'role': 'user', 'content': f'[{omitted} earlier exchange(s) omitted to fit the context window.]'})
        messages.insert(1, {'role': 'assistant', 'content': 'Understood.'})
    return messages


# ---------------------------------------------------------------- execution

class TurnRun:
    def __init__(self, turn, owner, session):
        self.turn, self.owner, self.session = turn, owner, session
        self.local = turn['provider'] == 'local'
        self.message_id = None
        self.content = self.reasoning = ''
        self.dirty = False
        self.stage_text = None
        self.last_flush = self.last_check = 0.0
        self.last_lease = self.last_model_beat = time.monotonic()

    def check(self, conn):
        row = conn.execute('SELECT status,cancel_requested,lease_owner FROM assistant_turns WHERE id=?', (self.turn['id'],)).fetchone()
        if not row or row['status'] != 'running' or row['lease_owner'] != self.owner:
            raise Aborted()
        if row['cancel_requested']:
            raise providers.Cancelled()
        now = time.monotonic()
        if now - self.last_lease > 15:
            conn.execute('UPDATE assistant_turns SET lease_until=? WHERE id=? AND lease_owner=?', (time.time() + LEASE_SECONDS, self.turn['id'], self.owner))
            self.last_lease = now
        if self.local and now - self.last_model_beat > 5:
            conn.execute('UPDATE local_text_model_state SET heartbeat=? WHERE id=1', (time.time(),))
            self.last_model_beat = now

    def checkpoint(self, force=False):
        now = time.monotonic()
        flush = self.dirty and (force or now - self.last_flush >= FLUSH_SECONDS)
        check = force or now - self.last_check >= CHECK_SECONDS
        if not flush and not check:
            return
        with db.connect() as conn:
            if flush:
                conn.execute('UPDATE assistant_messages SET content=?,reasoning=? WHERE id=?', (self.content, self.reasoning, self.message_id))
                store.bump(conn, self.turn['id'])
                self.dirty, self.last_flush = False, now
            if check:
                self.check(conn)
                self.last_check = now

    def stage(self, text):
        if text == self.stage_text:
            return
        self.stage_text = text
        with db.connect() as conn:
            self.check(conn)
            store.bump(conn, self.turn['id'], stage=text[:200])

    def add_message(self, role, **values):
        message_id = str(uuid.uuid4())
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self.check(conn)
            conn.execute('''INSERT INTO assistant_messages(id,session_id,turn_id,seq,role,content,reasoning,tool_calls_json,tool_call_id,tool_name,tool_summary,streaming,usage_json)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                         (message_id, self.session['id'], self.turn['id'], store.next_seq(conn, self.session['id']), role, values.get('content', ''), values.get('reasoning', ''),
                          json.dumps(values['tool_calls']) if values.get('tool_calls') is not None else None, values.get('tool_call_id'), values.get('tool_name'),
                          values.get('tool_summary'), int(values.get('streaming', False)), json.dumps(values.get('usage') or {})))
            store.bump(conn, self.turn['id'])
        return message_id

    def start_stream(self):
        self.content = self.reasoning = ''
        self.message_id = self.add_message('assistant', streaming=True)

    def on_delta(self, content, reasoning):
        self.content += content
        self.reasoning += reasoning
        self.dirty = True
        if content.strip():
            self.stage('Writing')
        elif reasoning.strip() and not self.content:
            self.stage('Thinking')
        self.checkpoint()

    def finish_stream(self, tool_calls=None, usage=None):
        with db.connect() as conn:
            conn.execute('UPDATE assistant_messages SET content=?,reasoning=?,tool_calls_json=?,streaming=0,usage_json=? WHERE id=?',
                         (self.content, self.reasoning, json.dumps(tool_calls) if tool_calls else None, json.dumps(usage or {}), self.message_id))
            store.bump(conn, self.turn['id'])
        self.dirty = False


def preload_skill(run, ctx, name):
    if not skills.get(name):
        return
    call = {'id': f'skill-{uuid.uuid4().hex[:8]}', 'name': 'load_skill', 'arguments': json.dumps({'name': name})}
    run.add_message('assistant', tool_calls=[call])
    content, label, ok = tools.execute(ctx, 'load_skill', call['arguments'])
    run.add_message('tool', content=content, tool_call_id=call['id'], tool_name='load_skill', tool_summary=label, usage={'ok': ok, 'preloaded': True})


def merge_usage(total, step, seconds):
    for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'reasoning_tokens_estimated', 'cached_tokens', 'cost'):
        if isinstance(step.get(key), (int, float)):
            total[key] = round(total.get(key, 0) + step[key], 6)
    total['generation_seconds'] = round(total.get('generation_seconds', 0) + seconds, 2)
    if 'prompt_tokens' in step:
        total['context_tokens'] = step['prompt_tokens'] + step.get('completion_tokens', 0)
    total['total_tokens'] = total.get('prompt_tokens', 0) + total.get('completion_tokens', 0)


def connect(stack, run, turn, provider, ctx, usage):
    """Open the endpoint for this turn on ``stack``.

    A turn that chose the local model and allows ``gpu_fallback`` is answered by OpenRouter free when
    visual work holds the GPU (and the worker has an OpenRouter key), instead of waiting for it. The
    decision is made once per turn; the next turn tries the local model again. The turn row keeps the
    user's choice, so Retry and the next message still start from the local model.
    """
    thinking = bool(turn['thinking'])
    checkpoint = lambda: run.checkpoint(force=True)
    fallback = providers.registry().get('openrouter') if turn['gpu_fallback'] and provider.id == 'local' else None
    if fallback is None or not fallback.available()[0]:
        return stack.enter_context(provider.session(turn['model'], thinking, checkpoint, run.stage))
    try:
        return stack.enter_context(provider.session(turn['model'], thinking, checkpoint, run.stage, wait_for_gpu=False))
    except local_text_runtime.GpuBusy:
        pass
    run.local = False
    ctx.provider, ctx.model = fallback.id, providers.FALLBACK_MODEL
    usage['fallback'] = {'reason': 'gpu_busy', 'from': provider.id, 'provider': fallback.id, 'model': providers.FALLBACK_MODEL}
    with db.connect() as conn:
        run.check(conn)
        store.bump(conn, turn['id'], usage_json=json.dumps(usage))  # Visible while the reply streams.
    run.stage('GPU busy · answering with OpenRouter free')
    return stack.enter_context(fallback.session(providers.FALLBACK_MODEL, thinking, checkpoint, run.stage))


def run_turn(turn_id, owner):
    turn = db.one('SELECT * FROM assistant_turns WHERE id=?', (turn_id,))
    session = db.one('SELECT * FROM assistant_sessions WHERE id=?', (turn['session_id'],)) if turn else None
    if not turn or not session:
        return
    run = TurnRun(turn, owner, session)
    provider = providers.registry().get(turn['provider'])
    ctx = tools.ToolContext(cut_id=session['cut_id'], session_id=session['id'], turn_id=turn_id, provider=turn['provider'], model=turn['model'], sequence_id=session.get('sequence_id') or '')
    usage = {'steps': 0, 'tool_calls': 0}
    started = time.monotonic()
    status, error = 'done', None
    try:
        if provider is None:
            raise providers.ProviderError(f"Provider {turn['provider']} is not configured in the inference worker")
        if turn['skill']:
            preload_skill(run, ctx, turn['skill'])
        run.stage('Preparing model')
        with contextlib.ExitStack() as stack:
            endpoint = connect(stack, run, turn, provider, ctx, usage)
            usage['context_limit'] = endpoint.context_limit
            usage['load_seconds'] = endpoint.extra.get('_load_seconds', 0)
            schemas = tools.schemas(tools.scope_of(ctx))
            budget = int((endpoint.context_limit or 32768) * 0.5)
            nudged = False
            failing = None  # A must_pass tool whose latest result failed.
            reminders = 0
            step = 0
            while step < MAX_STEPS:
                last = step == MAX_STEPS - 1
                messages = [{'role': 'system', 'content': system_prompt(session)}, *history_messages(session['id'], turn_id, budget)]
                if last:
                    messages.append({'role': 'user', 'content': '[Tool budget reached. Answer now with the information you have.]'})
                elif failing and reminders:
                    messages.append({'role': 'user', 'content': f'[{failing} reported errors that are not fixed yet. Fix every error and call {failing} again before answering.]'})
                elif nudged:
                    messages.append({'role': 'user', 'content': '[Your previous reply was empty. Give your answer now.]'})
                run.stage('Waiting for first token')
                run.start_stream()
                result = providers.stream_chat(endpoint, messages, None if last else schemas, max_tokens=8192 if turn['thinking'] else 4096,
                                               temperature=0.6 if turn['thinking'] else 0.5, on_delta=run.on_delta, checkpoint=run.checkpoint,
                                               on_tool=lambda name: run.stage(f'Preparing {name}'))
                if result.finish_reason == 'length' and not result.tool_calls:
                    run.content += '\n\n_[Response cut off: the output token limit was reached.]_'
                step_usage = {**result.usage, 'seconds': result.seconds, 'first_token_seconds': result.first_token_seconds, 'finish_reason': result.finish_reason}
                run.finish_stream(result.tool_calls, step_usage)
                usage['steps'] += 1
                usage['model'] = endpoint.model if endpoint.fixed_model else result.model or endpoint.model
                usage.setdefault('first_token_seconds', result.first_token_seconds)
                merge_usage(usage, result.usage, result.seconds)
                if result.usage.get('tokens_per_second'):
                    usage['tokens_per_second'] = result.usage['tokens_per_second']
                if result.tool_calls:
                    for call in result.tool_calls:
                        run.stage(f"Running {call['name']}")
                        try:
                            content, label, ok = tools.execute(ctx, call['name'], call['arguments'])
                        except Exception as exc:  # A tool bug must not end the conversation.
                            log.exception('Assistant tool %s failed', call['name'])
                            content, label, ok = compact.dumps({'error': f'Tool failed: {exc}'}), f"{call['name']} failed", False
                        run.add_message('tool', content=content, tool_call_id=call['id'], tool_name=call['name'], tool_summary=label[:300], usage={'ok': ok, 'chars': len(content)})
                        usage['tool_calls'] += 1
                        spec = tools.REGISTRY.get(call['name'])
                        if spec and spec.must_pass:
                            failing = None if store.loads(content, {}).get('ok') else call['name']
                    step += 1
                    continue
                if failing and reminders < 2 and not last:
                    # The model answered while a required check still fails: send it back once or twice.
                    reminders += 1
                    step += 1
                    continue
                if not run.content.strip() and not nudged and not last:
                    nudged = True
                    step += 1
                    continue
                break
    except providers.Cancelled:
        status, error = 'cancelled', None
    except Aborted:
        return
    except providers.ProviderError as exc:
        status, error = 'failed', str(exc)[:2000]
    except Exception as exc:
        log.exception('Assistant turn %s failed', turn_id)
        status, error = 'failed', f'{type(exc).__name__}: {exc}'[:2000]
    usage['seconds'] = round(time.monotonic() - started, 2)
    if usage.get('generation_seconds') and usage.get('completion_tokens') and 'tokens_per_second' not in usage:
        usage['tokens_per_second'] = round(usage['completion_tokens'] / max(usage['generation_seconds'], 0.01), 1)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT lease_owner FROM assistant_turns WHERE id=?', (turn_id,)).fetchone()
        if not row or row['lease_owner'] != owner:
            return
        if run.message_id:
            conn.execute('UPDATE assistant_messages SET content=?,reasoning=?,streaming=0 WHERE id=? AND streaming=1', (run.content, run.reasoning, run.message_id))
        stage = {'done': 'Complete', 'cancelled': 'Stopped', 'failed': 'Failed'}[status]
        store.bump(conn, turn_id, status=status, stage=stage, error=error, usage_json=json.dumps(usage), lease_owner=None, lease_until=None, finished_at=time.time())
        store.touch_session(conn, session['id'])


# -------------------------------------------------------------------- thread

def publish_state():
    described = []
    for provider in providers.registry().values():
        try:
            described.append(provider.describe())
        except Exception as exc:
            described.append({'id': provider.id, 'label': provider.label, 'available': False, 'detail': str(exc)[:200], 'models': []})
    with db.connect() as conn:
        conn.execute('UPDATE assistant_worker_state SET providers_json=?,heartbeat=? WHERE id=1', (json.dumps(described), time.time()))


def loop(stop_event):
    last_publish = 0.0
    while not stop_event.is_set():
        try:
            if time.monotonic() - last_publish > 5:
                publish_state()
                last_publish = time.monotonic()
            claimed = claim_turn()
            if claimed:
                run_turn(*claimed)
                publish_state()
                last_publish = time.monotonic()
            else:
                stop_event.wait(0.3)
        except Exception:
            log.exception('Assistant worker error')
            stop_event.wait(2)


def start():
    stop_event = threading.Event()
    thread = threading.Thread(target=loop, args=(stop_event,), daemon=True, name='cut-assistance')
    thread.start()
    return stop_event
