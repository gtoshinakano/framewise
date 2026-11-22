"""OpenAI-compatible chat endpoint so other machines on the LAN can use Qwen 3.6 as a chat provider.

``GET /v1/models`` and ``POST /v1/chat/completions`` (streaming or not) live in the API process, which
cannot reach llama-server (it listens on 127.0.0.1 inside the inference worker and moves between ports).
A request is therefore a row in ``chat_gateway_requests``; ``chat_gateway_worker`` claims it in the
inference worker, runs it through ``providers.AutoServer`` (this host's model or the remote llama.cpp
PC, whichever is free, with chat priority over queued jobs) and writes the streamed text back into the
row. Every write advances ``version`` so this module can forward the new text, as assistant turns do.

Access needs ``CHAT_API_KEY`` (a Bearer token). Without it the endpoints answer 503, because the rest of
the API has no authentication and must not become an open model gateway on the LAN.

User messages may carry ``image_url`` parts (a ``data:`` URL, or an http(s) URL this process downloads).
``clean_request`` turns each into a base64 data URL in a format llama-server decodes, converting others
(WebP, AVIF, TIFF, ...) to PNG with ffmpeg, and counts them in ``request['images']`` so the worker sends
the request only to a server whose model can see (``remote_text.sees``).
"""
import asyncio
import base64
import binascii
import hmac
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

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
MAX_IMAGES = 16
MAX_IMAGE_BYTES = 20 * 1024 * 1024
FETCH_SECONDS = 15
# Formats llama-server's image loader (stb_image) reads as they are; anything else goes through ffmpeg to PNG.
READABLE = ((b'\xff\xd8\xff', 'image/jpeg'), (b'\x89PNG\r\n\x1a\n', 'image/png'), (b'GIF87a', 'image/gif'), (b'GIF89a', 'image/gif'), (b'BM', 'image/bmp'))
# Every column but ``request_json``: the stream polls the row many times a second and must not reread the pictures.
ROW_FIELDS = ('id,status,stage,model,thinking,content,reasoning,tool_calls_json,finish_reason,error,error_status,usage_json,'
              'cancel_requested,attempt_count,version,created_at,started_at,finished_at')


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


def fetch_image(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'framewise-chat-gateway', 'Accept': 'image/*'})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_SECONDS) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except (OSError, ValueError) as exc:  # URLError and HTTPError are OSErrors
        raise ValueError(f'could not download the image at {url[:200]}: {exc}') from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f'an image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB')
    return data


def image_bytes(value):
    """The bytes of an ``image_url`` value: ``{"url": ...}`` or the URL itself, ``data:`` (base64) or http(s)."""
    url = value.get('url') if isinstance(value, dict) else value
    url = url.strip() if isinstance(url, str) else ''
    if url.startswith('data:'):
        header, comma, payload = url.partition(',')
        if not comma or not header.endswith(';base64'):
            raise ValueError('an image data URL must be base64 encoded (data:image/png;base64,...)')
        if len(payload) > MAX_IMAGE_BYTES * 4 // 3 + 16:
            raise ValueError(f'an image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB')
        try:
            data = base64.b64decode(''.join(payload.split()), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError('an image data URL is not valid base64') from exc
    elif url.startswith(('http://', 'https://')):
        data = fetch_image(url)
    else:
        raise ValueError('image_url.url must be a data: URL (data:image/png;base64,...) or an http(s) URL')
    if not data:
        raise ValueError('an image is empty')
    return data


def to_png(data):
    """One frame of any picture ffmpeg can decode (WebP, AVIF, TIFF, ...) as PNG, or ``ValueError``."""
    with tempfile.TemporaryDirectory(prefix='chat-image-') as folder:
        source, target = Path(folder) / 'image', Path(folder) / 'image.png'
        source.write_bytes(data)
        try:
            result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(source), '-frames:v', '1', str(target)], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f'an image could not be converted: {exc}') from exc
        if result.returncode or not target.is_file() or not target.stat().st_size:
            raise ValueError('an image could not be read (send JPEG, PNG, GIF, BMP or WebP)')
        return target.read_bytes()


def image_part(value):
    """An ``image_url`` part the model server can decode: always a base64 data URL of JPEG, PNG, GIF or BMP."""
    data = image_bytes(value)
    mime = next((kind for magic, kind in READABLE if data.startswith(magic)), None)
    if mime is None:
        data, mime = to_png(data), 'image/png'
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')}}


def clean_content(message):
    """A message's content with every image part normalised (``image_part``); text passes as it is."""
    content = message.get('content')
    if not isinstance(content, list):
        return content, 0
    parts, images = [], 0
    for part in content:
        kind = part.get('type') if isinstance(part, dict) else None
        if kind == 'image_url':
            if message['role'] != 'user':
                raise ValueError('images are accepted only in user messages')
            parts.append(image_part(part.get('image_url')))
            images += 1
        elif isinstance(part, dict) and kind not in ('text', None):
            raise ValueError(f"content parts of type {kind!r} are not supported (send text or image_url)")
        else:
            parts.append(part)
    return parts, images


def clean_request(body):
    """The part of an OpenAI request the worker needs, or ``ValueError`` with a message for the client.

    Blocking: it may download pictures or run ffmpeg, so the endpoint calls it on a thread.
    """
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a non-empty array')
    if len(messages) > MAX_BODY_MESSAGES:
        raise ValueError(f'messages is limited to {MAX_BODY_MESSAGES} entries')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
            raise ValueError('every message needs a role of system, user, assistant or tool')
    counted = sum(1 for message in messages if isinstance(message.get('content'), list)
                  for part in message['content'] if isinstance(part, dict) and part.get('type') == 'image_url')
    if counted > MAX_IMAGES:
        raise ValueError(f'a request may carry at most {MAX_IMAGES} images')
    cleaned, images = [], 0
    for message in messages:
        content, found = clean_content(message)
        images += found
        cleaned.append({**message, 'role': 'system' if message['role'] == 'developer' else message['role'], **({'content': content} if found else {})})
    request = {'messages': cleaned, 'stream': bool(body.get('stream')),
               'include_usage': bool((body.get('stream_options') or {}).get('include_usage')) if isinstance(body.get('stream_options'), dict) else False}
    if images:
        request['images'] = images
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
    return db.one(f'SELECT {ROW_FIELDS} FROM chat_gateway_requests WHERE id=?', (request_id,))


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


@router.post('/v1/chat/completions', summary='Chat with Qwen 3.6 (OpenAI-compatible, streaming and images supported)')
async def chat_completions(request: Request):
    """Needs ``Authorization: Bearer <CHAT_API_KEY>``. The reply comes from this host's Qwen 3.6 or the remote llama.cpp server, whichever is free.

    User messages may carry ``image_url`` parts; such a request goes only to a server whose model can read images.
    """
    try:
        check_key(request)
    except HTTPException as exc:
        return openai_error(exc.status_code, exc.detail)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('The request body must be a JSON object')
        clean = await asyncio.to_thread(clean_request, body)
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
