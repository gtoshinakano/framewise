"""Runs queued ``chat_gateway_requests`` in the inference worker (see ``chat_gateway``).

Each request goes through ``providers.AutoServer``: this host's Qwen 3.6 or the remote llama.cpp PC,
whichever is free, with the same chat priority as Cut Assistance (job lanes stop claiming while a chat
waits). Text is flushed into the row a few times per second; every flush advances ``version`` so the API
can stream it. A request interrupted mid-answer is failed, never restarted: a streaming client has
already seen part of the text and a restart would repeat it.
"""
import json
import logging
import re
import threading
import time
import uuid

from . import db
from .assistant import providers

log = logging.getLogger(__name__)
LEASE_SECONDS = 60
FLUSH_SECONDS = 0.12
CHECK_SECONDS = 0.4
MAX_CONCURRENT = 3
DEFAULT_MAX_TOKENS = 16384
STATUS_PATTERN = re.compile(r'^HTTP (\d{3})')


class Aborted(Exception):
    """The lease moved to another worker or the row was deleted."""


def claim():
    now = time.time()
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('''UPDATE chat_gateway_requests SET status='failed',stage='Failed',error='The inference worker restarted while answering',error_status=503,
                        lease_owner=NULL,lease_until=NULL,finished_at=?,version=version+1
                        WHERE status='running' AND (lease_until IS NULL OR lease_until<=?)''', (now, now))
        row = conn.execute("SELECT id FROM chat_gateway_requests WHERE status='queued' ORDER BY created_at,rowid LIMIT 1").fetchone()
        if not row:
            return None
        owner = str(uuid.uuid4())
        conn.execute("UPDATE chat_gateway_requests SET status='running',stage='Starting',lease_owner=?,lease_until=?,attempt_count=attempt_count+1,started_at=?,version=version+1 WHERE id=?",
                     (owner, now + LEASE_SECONDS, now, row['id']))
        return row['id'], owner


def failure_status(message):
    match = STATUS_PATTERN.match(message or '')
    if match and match.group(1) in ('400', '413', '422'):
        return int(match.group(1))
    return 502


class Run:
    def __init__(self, request_id, owner, local=False):
        self.id, self.owner, self.local = request_id, owner, local
        self.content = self.reasoning = ''
        self.dirty = False
        self.stage_text = None
        self.last_flush = self.last_check = 0.0
        self.last_lease = self.last_model_beat = time.monotonic()

    def check(self, conn):
        row = conn.execute('SELECT status,cancel_requested,lease_owner FROM chat_gateway_requests WHERE id=?', (self.id,)).fetchone()
        if not row or row['status'] != 'running' or row['lease_owner'] != self.owner:
            raise Aborted()
        if row['cancel_requested']:
            raise providers.Cancelled()
        now = time.monotonic()
        if now - self.last_lease > 15:
            conn.execute('UPDATE chat_gateway_requests SET lease_until=? WHERE id=? AND lease_owner=?', (time.time() + LEASE_SECONDS, self.id, self.owner))
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
                conn.execute('UPDATE chat_gateway_requests SET content=?,reasoning=?,version=version+1 WHERE id=?', (self.content, self.reasoning, self.id))
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
            conn.execute('UPDATE chat_gateway_requests SET stage=?,version=version+1 WHERE id=?', (text[:200], self.id))

    def on_delta(self, content, reasoning):
        self.content += content
        self.reasoning += reasoning
        self.dirty = True
        self.checkpoint()


def without_images(request):
    """The request as kept after it ran: each picture's data URL replaced by its size, so a finished row stays small for the day it is kept."""
    def slim(part):
        if not (isinstance(part, dict) and part.get('type') == 'image_url'):
            return part
        header, _, data = str((part.get('image_url') or {}).get('url') or '').partition(',')
        return {'type': 'image_url', 'image_url': {'url': f"omitted ({header.removeprefix('data:').removesuffix(';base64')}, {len(data) * 3 // 4} bytes)"}}
    messages = [{**message, 'content': [slim(part) for part in message['content']]} if isinstance(message.get('content'), list) else message
                for message in request['messages']]
    return {**request, 'messages': messages}


def sampling_extra(request, thinking):
    extra = dict(request.get('sampling') or {})
    if request.get('tools') and request.get('tool_choice'):
        extra['tool_choice'] = request['tool_choice']
    extra.setdefault('top_p', 0.95)
    extra.setdefault('top_k', 20)
    return extra


def run_request(request_id, owner):
    row = db.one('SELECT * FROM chat_gateway_requests WHERE id=?', (request_id,))
    if not row:
        return
    request = json.loads(row['request_json'])
    thinking = bool(row['thinking'])
    images = request.get('images') or 0  # counted by ``chat_gateway.clean_request``; such a request needs a server that can see
    run = Run(request_id, owner)
    usage, status, error, error_status = {}, 'done', None, None
    finish, calls = None, []
    started = time.monotonic()
    try:
        provider = providers.AutoServer()
        ok, detail = provider.available(vision=bool(images))
        if not ok:
            raise (providers.NoVision if images else providers.ProviderError)(detail or 'No llama.cpp server is available')
        run.stage('Waiting for a server')
        with provider.session('auto', thinking, lambda: run.checkpoint(force=True), run.stage, vision=bool(images)) as base:
            server = base.extra.get('_server')
            run.local = server == 'local'
            run.stage('Waiting for first token')
            endpoint = providers.Endpoint(base.url, base.model, headers=base.headers, context_limit=base.context_limit, timeout=base.timeout,
                                          fixed_model=base.fixed_model, extra={**base.extra, **sampling_extra(request, thinking)})
            result = providers.stream_chat(endpoint, request['messages'], request.get('tools'), max_tokens=request.get('max_tokens', DEFAULT_MAX_TOKENS),
                                           temperature=request.get('temperature', 0.6 if thinking else 0.7), on_delta=run.on_delta, checkpoint=run.checkpoint)
            run.content, run.reasoning = result.content, result.reasoning
            finish, calls = result.finish_reason, result.tool_calls
            usage = {**result.usage, 'seconds': result.seconds, 'first_token_seconds': result.first_token_seconds, 'server': server, 'context_limit': endpoint.context_limit}
    except providers.Cancelled:
        status = 'cancelled'
    except Aborted:
        return
    except providers.ProviderError as exc:
        status, error, error_status = 'failed', str(exc)[:2000], getattr(exc, 'status', None) or failure_status(str(exc))
    except Exception as exc:
        log.exception('Chat gateway request %s failed', request_id)
        status, error, error_status = 'failed', f'{type(exc).__name__}: {exc}'[:2000], 500
    usage['wall_seconds'] = round(time.monotonic() - started, 2)
    if images:
        usage['images'] = images
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = conn.execute('SELECT lease_owner FROM chat_gateway_requests WHERE id=?', (request_id,)).fetchone()
        if not current or current['lease_owner'] != owner:
            return
        conn.execute('''UPDATE chat_gateway_requests SET status=?,stage=?,content=?,reasoning=?,tool_calls_json=?,finish_reason=?,error=?,error_status=?,usage_json=?,
                        request_json=COALESCE(?,request_json),lease_owner=NULL,lease_until=NULL,finished_at=?,version=version+1 WHERE id=?''',
                     (status, {'done': 'Complete', 'cancelled': 'Stopped', 'failed': 'Failed'}[status], run.content, run.reasoning, json.dumps(calls) if calls else None,
                      finish, error, error_status, json.dumps(usage), json.dumps(without_images(request), ensure_ascii=False) if images else None, time.time(), request_id))


def loop(stop_event):
    active, lock = set(), threading.Lock()

    def work(claimed):
        try:
            run_request(*claimed)
        except Exception:
            log.exception('Chat gateway thread failed')
        finally:
            with lock:
                active.discard(threading.current_thread())

    last_beat = 0.0
    while not stop_event.is_set():
        try:
            if time.monotonic() - last_beat > 5:
                with db.connect() as conn:
                    conn.execute('UPDATE chat_gateway_state SET heartbeat=? WHERE id=1', (time.time(),))
                last_beat = time.monotonic()
            with lock:
                room = len(active) < MAX_CONCURRENT
            claimed = claim() if room else None
            if claimed:
                thread = threading.Thread(target=work, args=(claimed,), daemon=True, name=f'chat-gateway-{claimed[0][-8:]}')
                with lock:
                    active.add(thread)
                thread.start()
            else:
                stop_event.wait(0.3)
        except Exception:
            log.exception('Chat gateway worker error')
            stop_event.wait(2)


def start():
    stop_event = threading.Event()
    threading.Thread(target=loop, args=(stop_event,), daemon=True, name='chat-gateway').start()
    return stop_event
