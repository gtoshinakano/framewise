"""OpenAI-compatible chat endpoint so other machines on the LAN can use Qwen 3.6 as a chat provider.

``GET /v1/models`` and ``POST /v1/chat/completions`` (streaming or not) live in the API process, which
cannot reach llama-server (it listens on 127.0.0.1 inside the inference worker and moves between ports).
A request is therefore a row in ``chat_gateway_requests``; ``chat_gateway_worker`` claims it in the
inference worker, runs it through ``providers.AutoServer`` (this host's model or the remote llama.cpp
PC, whichever is free, with chat priority over queued jobs) and writes the streamed text back into the
row. Every write advances ``version`` so this module can forward the new text, as assistant turns do.

Access needs ``CHAT_API_KEY`` (a Bearer token). Without it the endpoints answer 503, because the rest of
the API has no authentication and must not become an open model gateway on the LAN.
"""
import asyncio
import hmac
import json
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import db
from .chat_gateway_schema import init_schema, SCHEMA  # noqa: F401  (the worker image has no fastapi, so db.py imports the schema from there)

router = APIRouter(tags=['Chat gateway'])

ACTIVE = ('queued', 'running')
PUBLIC_MODEL = 'qwen3.6-35b-a3b'
MAX_BODY_MESSAGES = 500
WORKER_STALE_SECONDS = 30
KEEP_FINISHED_SECONDS = 24 * 3600
# Request fields forwarded to llama-server as they are. Anything else is ignored.
SAMPLING_FIELDS = ('top_p', 'top_k', 'min_p', 'seed', 'stop', 'presence_penalty', 'frequency_penalty', 'repeat_penalty', 'response_format')


def configured_key():
    return os.environ.get('CHAT_API_KEY', '').strip()


def check_key(request: Request):
    key = configured_key()
    if not key:
        raise HTTPException(503, 'The chat API is disabled: set CHAT_API_KEY for the API container')
    header = request.headers.get('authorization', '')
    supplied = header[7:].strip() if header[:7].lower() == 'bearer ' else ''
    if not supplied or not hmac.compare_digest(supplied.encode(), key.encode()):
        raise HTTPException(401, 'Invalid or missing API key (send Authorization: Bearer <CHAT_API_KEY>)')


def error_body(message, kind='invalid_request_error', code=None):
    return {'error': {'message': message, 'type': kind, 'code': code}}


def openai_error(status, message, kind=None):
    kind = kind or ('authentication_error' if status in (401, 403) else 'invalid_request_error' if status < 500 else 'server_error')
    return JSONResponse(status_code=status, content=error_body(message, kind))


def worker_online():
    row = db.one('SELECT heartbeat FROM chat_gateway_state WHERE id=1')
    return bool(row) and time.time() - (row['heartbeat'] or 0) < WORKER_STALE_SECONDS


def thinking_choice(body):
    """Qwen thinks by default; a client turns it off with ``chat_template_kwargs.enable_thinking`` or ``reasoning_effort: none``."""
    kwargs = body.get('chat_template_kwargs')
    if isinstance(kwargs, dict) and isinstance(kwargs.get('enable_thinking'), bool):
        return kwargs['enable_thinking']
    effort = body.get('reasoning_effort')
    if isinstance(effort, str):
        return effort.lower() not in ('none', 'off', 'minimal')
    reasoning = body.get('reasoning')
    if isinstance(reasoning, dict):
        if reasoning.get('enabled') is False or str(reasoning.get('effort', '')).lower() in ('none', 'off'):
            return False
    return True


def clean_request(body):
    """The part of an OpenAI request the worker needs, or ``ValueError`` with a message for the client."""
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a non-empty array')
    if len(messages) > MAX_BODY_MESSAGES:
        raise ValueError(f'messages is limited to {MAX_BODY_MESSAGES} entries')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
            raise ValueError('every message needs a role of system, user, assistant or tool')
        content = message.get('content')
        if isinstance(content, list):
            if any(isinstance(part, dict) and part.get('type') not in ('text', None) for part in content):
                raise ValueError('only text content is supported by this model endpoint')
    messages = [{**message, 'role': 'system' if message['role'] == 'developer' else message['role']} for message in messages]
    request = {'messages': messages, 'stream': bool(body.get('stream')),
               'include_usage': bool((body.get('stream_options') or {}).get('include_usage')) if isinstance(body.get('stream_options'), dict) else False}
    limit = body.get('max_completion_tokens', body.get('max_tokens'))
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError('max_tokens must be a positive integer')
        request['max_tokens'] = min(limit, 32768)
    temperature = body.get('temperature')
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be between 0 and 2')
        request['temperature'] = float(temperature)
    if isinstance(body.get('tools'), list) and body['tools']:
        request['tools'] = body['tools']
        if body.get('tool_choice') in ('auto', 'none', 'required') or isinstance(body.get('tool_choice'), dict):
            request['tool_choice'] = body['tool_choice']
    request['sampling'] = {key: body[key] for key in SAMPLING_FIELDS if body.get(key) is not None}
    return request


def queue_request(request, model, thinking):
    request_id = 'chatcmpl-' + uuid.uuid4().hex[:24]
    with db.connect() as conn:
        conn.execute('DELETE FROM chat_gateway_requests WHERE status NOT IN (?,?) AND finished_at < ?', (*ACTIVE, time.time() - KEEP_FINISHED_SECONDS))
        conn.execute('INSERT INTO chat_gateway_requests(id,request_json,model,thinking,created_at) VALUES (?,?,?,?,?)',
                     (request_id, json.dumps(request, ensure_ascii=False), model, int(thinking), time.time()))
    return request_id


def request_cancel(request_id):
    with db.connect() as conn:
        conn.execute("UPDATE chat_gateway_requests SET cancel_requested=1,version=version+1 WHERE id=? AND status IN ('queued','running')", (request_id,))
        conn.execute("UPDATE chat_gateway_requests SET status='cancelled',stage='Stopped',finished_at=?,version=version+1 WHERE id=? AND status='queued'", (time.time(), request_id))


def snapshot(request_id):
    return db.one('SELECT * FROM chat_gateway_requests WHERE id=?', (request_id,))


def usage_of(row):
    usage = json.loads(row['usage_json'] or '{}')
    prompt, completion = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    out = {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion}
    if usage.get('reasoning_tokens') is not None:
        out['completion_tokens_details'] = {'reasoning_tokens': usage['reasoning_tokens']}
    if usage.get('cached_tokens') is not None:
        out['prompt_tokens_details'] = {'cached_tokens': usage['cached_tokens']}
    return out


def tool_calls_of(row):
    calls = json.loads(row['tool_calls_json'] or '[]')
    return [{'id': call['id'], 'type': 'function', 'function': {'name': call['name'], 'arguments': call['arguments']}} for call in calls]


def failure_status(row):
    return row['error_status'] or 502


def completion_body(row):
    message = {'role': 'assistant', 'content': row['content'] or None}
    if row['reasoning']:
        message['reasoning_content'] = row['reasoning']
    calls = tool_calls_of(row)
    if calls:
        message['tool_calls'] = calls
    finish = 'tool_calls' if calls else row['finish_reason'] or 'stop'
    return {'id': row['id'], 'object': 'chat.completion', 'created': int(row['created_at']), 'model': row['model'],
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage_of(row)}


def chunk(row, delta, finish=None, usage=None):
    """One SSE event; the usage-only event that ends a stream with ``include_usage`` has no choices."""
    choices = [] if usage is not None else [{'index': 0, 'delta': delta, 'finish_reason': finish}]
    body = {'id': row['id'], 'object': 'chat.completion.chunk', 'created': int(row['created_at']), 'model': row['model'], 'choices': choices}
    if usage is not None:
        body['usage'] = usage
    return 'data: ' + json.dumps(body, ensure_ascii=False) + '\n\n'


async def stream_events(request: Request, request_id, include_usage):
    """Forward the row's text as ``chat.completion.chunk`` events until the request finishes."""
    sent_content = sent_reasoning = 0
    version, started, last_ping = None, False, time.monotonic()
    try:
        while True:
            if await request.is_disconnected():
                await asyncio.to_thread(request_cancel, request_id)
                return
            row = await asyncio.to_thread(snapshot, request_id)
            if not row:
                yield 'data: ' + json.dumps(error_body('The request was removed', 'server_error')) + '\n\n'
                return
            if row['version'] != version or row['status'] not in ACTIVE:
                version = row['version']
                if not started and (row['content'] or row['reasoning'] or row['status'] not in ACTIVE):
                    started = True
                    yield chunk(row, {'role': 'assistant', 'content': ''})
                if row['reasoning'][sent_reasoning:]:
                    yield chunk(row, {'reasoning_content': row['reasoning'][sent_reasoning:]})
                    sent_reasoning = len(row['reasoning'])
                if row['content'][sent_content:]:
                    yield chunk(row, {'content': row['content'][sent_content:]})
                    sent_content = len(row['content'])
                if row['status'] not in ACTIVE:
                    if row['status'] == 'failed':
                        yield 'data: ' + json.dumps(error_body(row['error'] or 'The model request failed', 'server_error')) + '\n\n'
                    elif row['status'] == 'done':
                        calls = tool_calls_of(row)
                        if calls:
                            yield chunk(row, {'tool_calls': [{'index': index, **call} for index, call in enumerate(calls)]})
                        yield chunk(row, {}, 'tool_calls' if calls else row['finish_reason'] or 'stop')
                        if include_usage:
                            yield chunk(row, {}, usage=usage_of(row))
                    yield 'data: [DONE]\n\n'
                    return
            if time.monotonic() - last_ping > 10:
                last_ping = time.monotonic()
                yield ': keep-alive\n\n'
            await asyncio.sleep(0.08)
    except asyncio.CancelledError:
        await asyncio.to_thread(request_cancel, request_id)
        raise


async def wait_for_completion(request: Request, request_id):
    """Block until the row finishes, cancelling it when the client goes away."""
    try:
        while True:
            if await request.is_disconnected():
                await asyncio.to_thread(request_cancel, request_id)
                return None
            row = await asyncio.to_thread(snapshot, request_id)
            if not row or row['status'] not in ACTIVE:
                return row
            await asyncio.sleep(0.15)
    except asyncio.CancelledError:
        await asyncio.to_thread(request_cancel, request_id)
        raise


@router.get('/v1/models', summary='List the chat model (OpenAI-compatible)')
async def list_models(request: Request):
    check_key(request)
    return {'object': 'list', 'data': [{'id': PUBLIC_MODEL, 'object': 'model', 'created': 0, 'owned_by': 'framewise'}]}


@router.post('/v1/chat/completions', summary='Chat with Qwen 3.6 (OpenAI-compatible, streaming supported)')
async def chat_completions(request: Request):
    """Needs ``Authorization: Bearer <CHAT_API_KEY>``. The reply comes from this host's Qwen 3.6 or the remote llama.cpp server, whichever is free."""
    try:
        check_key(request)
    except HTTPException as exc:
        return openai_error(exc.status_code, exc.detail)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('The request body must be a JSON object')
        clean = clean_request(body)
    except ValueError as exc:
        return openai_error(400, str(exc) or 'The request body is not valid JSON')
    if not await asyncio.to_thread(worker_online):
        return openai_error(503, 'The inference worker is not running, so no model can answer', 'server_error')
    model = str(body.get('model') or PUBLIC_MODEL)[:200]
    request_id = await asyncio.to_thread(queue_request, clean, model, thinking_choice(body))
    if clean['stream']:
        return StreamingResponse(stream_events(request, request_id, clean['include_usage']), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    row = await wait_for_completion(request, request_id)
    if row is None:
        return openai_error(499, 'Client closed the request')
    if row['status'] == 'failed':
        return openai_error(failure_status(row), row['error'] or 'The model request failed')
    if row['status'] != 'done':
        return openai_error(503, 'The request was cancelled', 'server_error')
    return JSONResponse(completion_body(row))
