"""A llama.cpp server on another PC that takes local-provider text jobs next to this host's own.

The user types the PC's address in the Text engine dialog (it has no host name, so a changed IP is
entered again there). One base URL serves several models through the OpenAI-compatible API, and the
request's ``model`` field picks one (llama.cpp router mode or llama-swap). An optional Bearer token (llama-server ``--api-key``) is sent when set.

Routing is a per-pipeline map read when a job is claimed, never snapshotted into payloads:

* ``local``  - only this host's lanes take the pipeline's local-provider jobs (the default)
* ``remote`` - only remote lanes take them, with the route's model
* ``any``    - whichever lane is free first; remote lanes use the route's model

Routes apply only to jobs queued with the ``local`` provider. OpenRouter jobs stay on this host's lanes.

A remote lane runs the ordinary job handler with a thread-local binding (``bind``). ``chat_json`` and
the other local-model helpers check ``bound()`` and talk to the remote instead of ``LocalTextRuntime``,
so remote work never waits for, loads or stops the local GPU model.
"""
import json
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

from . import db

MAX_LANES = 8
DEFAULT_SLOT_CONTEXT = 32768
ROUTE_MODES = ('local', 'remote', 'any')
SERVERS = ('local', 'remote')
URL_PATTERN = re.compile(r'^https?://[^/\s]+(/[^\s]*)?$')
VARIANT_PREFIX = 'remote:'

_binding = threading.local()


def variant_name(model):
    return VARIANT_PREFIX + str(model)


def is_remote_variant(variant):
    return isinstance(variant, str) and variant.startswith(VARIANT_PREFIX)


def parse_routes(value):
    """Valid routes only: ``{pipeline: {'mode', 'model'}}``; a remote route without a model is dropped."""
    from . import worker
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        raw = {}
    routes = {}
    for pipeline in worker.TEXT_PIPELINES:
        item = raw.get(pipeline) if isinstance(raw, dict) else None
        if not isinstance(item, dict) or item.get('mode') not in ROUTE_MODES:
            continue
        model = str(item.get('model') or '').strip()
        if item['mode'] != 'local' and not model:
            continue
        routes[pipeline] = {'mode': item['mode'], 'model': model if item['mode'] != 'local' else ''}
    return routes


def config(conn=None):
    """The saved remote server settings, with ``routes`` parsed (the token is included: worker use only)."""
    row = (conn.execute('SELECT * FROM remote_text_server WHERE id=1').fetchone() if conn else db.one('SELECT * FROM remote_text_server WHERE id=1'))
    row = dict(row) if row else {}
    return {'enabled': bool(row.get('enabled')), 'base_url': (row.get('base_url') or '').rstrip('/'), 'token': row.get('token') or '',
            'parallel': max(1, min(MAX_LANES, int(row.get('parallel') or 1))),
            'routes': parse_routes(row.get('routes_json') or '{}'), 'assistant_fallback': row.get('assistant_fallback') or '',
            'prefer': row.get('prefer') if row.get('prefer') in SERVERS else 'local', 'chat_reserve': max(0, min(MAX_LANES, int(row.get('chat_reserve') if row.get('chat_reserve') is not None else 1)))}


def usable(cfg):
    return bool(cfg.get('enabled') and cfg.get('base_url'))


def chat_model(cfg=None):
    """The remote model that answers chat and unrouted jobs: the one chosen for the assistant, else the first the server lists."""
    cfg = cfg or config()
    listed = [item['id'] for item in state()['models'] if isinstance(item, dict) and item.get('id')]
    if cfg.get('assistant_fallback') and (cfg['assistant_fallback'] in listed or not listed):
        return cfg['assistant_fallback']
    return listed[0] if listed else ''


def effective_routes(cfg):
    """A route for every text pipeline. One without a saved route shares the work: whichever server is free, with the remote's default model."""
    if not usable(cfg):
        return {}
    from . import worker
    default = chat_model(cfg)
    routes = {}
    for pipeline in worker.TEXT_PIPELINES:
        route = cfg['routes'].get(pipeline) or ({'mode': 'any', 'model': default} if default else None)
        if route:
            routes[pipeline] = route
    return routes


def route_for(cfg, pipeline):
    route = effective_routes(cfg).get(pipeline)
    return (route['mode'], route['model']) if route else ('local', '')


def remote_only(cfg):
    """Pipelines whose local-provider jobs this host's lanes must leave to the remote."""
    return tuple(pipeline for pipeline, route in effective_routes(cfg).items() if route['mode'] == 'remote')


def remote_pipelines(cfg):
    """Pipelines a remote lane may claim (local-provider jobs only)."""
    return tuple(pipeline for pipeline, route in effective_routes(cfg).items() if route['mode'] in ('remote', 'any'))


def auth_headers(cfg):
    token = (cfg.get('token') or '').strip()
    return {'Authorization': f'Bearer {token}'} if token else {}


def _get(cfg, path, timeout):
    request = urllib.request.Request(cfg['base_url'] + path, headers=auth_headers(cfg))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body.strip() else {}


def healthy(cfg, timeout=3):
    """The server answers its health check (a model still loading counts as unreachable)."""
    try:
        _get(cfg, '/health', timeout)
        return True
    except urllib.error.HTTPError as exc:
        return exc.code == 401  # reachable; the auth problem is reported by the monitor
    except (OSError, ValueError):
        return False


def probe(cfg, timeout=5):
    """Ask the server for its models and their per-slot context; returns ``{state, detail, models}``."""
    if not cfg.get('base_url'):
        return {'state': 'error', 'detail': 'No address configured', 'models': []}
    try:
        try:
            _get(cfg, '/health', timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise
            if exc.code != 404:  # router front ends may not expose /health; /v1/models decides
                return {'state': 'offline', 'detail': f'Health check answered HTTP {exc.code}', 'models': []}
        listed = _get(cfg, '/v1/models', timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return {'state': 'auth_failed', 'detail': 'The server refused the access token', 'models': []}
        return {'state': 'error', 'detail': f'The server answered HTTP {exc.code}', 'models': []}
    except (OSError, ValueError) as exc:
        reason = getattr(exc, 'reason', None) or exc
        return {'state': 'offline', 'detail': f'Not reachable: {reason}'[:300], 'models': []}
    models = []
    for item in (listed.get('data') or listed.get('models') or []) if isinstance(listed, dict) else []:
        model_id = (item.get('id') or item.get('name') or item.get('model')) if isinstance(item, dict) else None
        if not isinstance(model_id, str) or not model_id or any(m['id'] == model_id for m in models):
            continue
        models.append({'id': model_id, 'n_ctx': _slot_context(cfg, model_id, timeout)})
    if not models:
        return {'state': 'error', 'detail': 'The server lists no models', 'models': []}
    return {'state': 'online', 'detail': f"{len(models)} model{'s' if len(models) != 1 else ''}", 'models': models}


def _slot_context(cfg, model_id, timeout):
    try:
        props = _get(cfg, '/props?model=' + urllib.request.quote(model_id, safe=''), timeout)
    except (OSError, ValueError):
        return None
    settings = props.get('default_generation_settings') if isinstance(props, dict) else None
    value = settings.get('n_ctx') if isinstance(settings, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def state():
    row = db.one('SELECT * FROM remote_text_state WHERE id=1') or {}
    try:
        models = json.loads(row.get('models_json') or '[]')
    except (ValueError, TypeError):
        models = []
    return {'state': row.get('state') or 'disabled', 'detail': row.get('detail') or '', 'models': models if isinstance(models, list) else [],
            'active_jobs': row.get('active_jobs') or 0, 'heartbeat': row.get('heartbeat') or 0}


def online():
    current = state()
    return current['state'] == 'online' and time.time() - current['heartbeat'] < 30


def publish(result, active_jobs=None):
    try:
        with db.connect() as conn:
            conn.execute('UPDATE remote_text_state SET state=?,detail=?,models_json=?,heartbeat=?,active_jobs=COALESCE(?,active_jobs) WHERE id=1',
                         (result['state'], result.get('detail', '')[:500], json.dumps(result.get('models') or []), time.time(), active_jobs))
    except Exception:
        pass  # a locked database must not stop the monitor


def model_context(model):
    """Per-slot context of a remote model as reported by the server (default when unknown)."""
    for item in state()['models']:
        if isinstance(item, dict) and item.get('id') == model and isinstance(item.get('n_ctx'), int):
            return item['n_ctx']
    return DEFAULT_SLOT_CONTEXT


# ------------------------------------------------------------------ binding

class Binding:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.unreachable = False

    @property
    def url(self):
        return self.cfg['base_url'] + '/v1/chat/completions'

    @property
    def variant(self):
        return variant_name(self.model)


@contextmanager
def bind(cfg, model):
    """Run the calling thread's text requests on the remote server with ``model``."""
    previous = getattr(_binding, 'value', None)
    _binding.value = Binding(cfg, model)
    try:
        yield _binding.value
    finally:
        _binding.value = previous


def bound():
    return getattr(_binding, 'value', None)


# ------------------------------------------------------------------ slots and chat priority
#
# Both servers are a pool of slots (this host: ``TEXT_JOB_CONCURRENCY`` llama-server slots, the remote:
# ``parallel``). Text jobs and chat turns each take one. Jobs may only use ``capacity - chat_reserve`` slots
# of a server and claim nothing while a chat turn waits, so a chat message never queues behind a running job.
# The lanes and the assistant thread live in one process, so the counters are plain memory.

class Pool:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {server: 0 for server in SERVERS}
        self.chats = {server: 0 for server in SERVERS}
        self.waiting = 0

    def capacity(self, server, cfg=None):
        if server == 'remote':
            return (cfg or config())['parallel']
        from . import local_text_runtime
        return local_text_runtime.text_job_concurrency()

    def job_limit(self, server, cfg=None):
        cfg = cfg or config()
        capacity = self.capacity(server, cfg)
        return max(1, capacity - min(cfg['chat_reserve'], capacity - 1))

    def chat_waiting(self):
        if self.waiting > 0:
            return True
        with db.connect() as conn:
            return conn.execute("SELECT 1 FROM assistant_turns WHERE status='queued' LIMIT 1").fetchone() is not None

    def try_job(self, server, cfg=None):
        """Take a job slot on ``server`` before claiming a job; ``False`` while chat waits or the job share is used up."""
        if self.chat_waiting():
            return False
        limit = self.job_limit(server, cfg)
        with self.lock:
            if self.jobs[server] >= limit:
                return False
            self.jobs[server] += 1
            return True

    def release_job(self, server):
        with self.lock:
            self.jobs[server] = max(0, self.jobs[server] - 1)

    @contextmanager
    def chat(self, server):
        with self.lock:
            self.chats[server] += 1
        try:
            yield
        finally:
            with self.lock:
                self.chats[server] = max(0, self.chats[server] - 1)

    @contextmanager
    def chat_wait(self):
        """A chat turn is waiting for its server: job lanes claim nothing meanwhile."""
        with self.lock:
            self.waiting += 1
        try:
            yield
        finally:
            with self.lock:
                self.waiting = max(0, self.waiting - 1)

    def free(self, server, cfg=None):
        with self.lock:
            return self.capacity(server, cfg) - self.jobs[server] - self.chats[server]

    def busy_with_chat(self):
        with self.lock:
            return dict(self.chats)


pool = Pool()


def pick_server(cfg=None):
    """``('local', '')`` or ``('remote', model)`` for a chat turn that may use whichever server is free, or ``None``."""
    cfg = cfg or config()
    from . import local_text_runtime
    runtime = local_text_runtime.runtime
    candidates = {}
    if runtime is not None and not runtime.gpu_busy and not runtime.switch_pending() and pool.free('local', cfg) > 0:
        candidates['local'] = ''
    if usable(cfg) and online() and pool.free('remote', cfg) > 0 and chat_model(cfg):
        candidates['remote'] = chat_model(cfg)
    for server in sorted(candidates, key=lambda name: name != cfg['prefer']):
        return server, candidates[server]
    return None
