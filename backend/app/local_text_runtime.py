"""Worker-owned local text model lifecycle; the API controls desired state via SQLite."""
import logging
import os
import json
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
import urllib.error
import urllib.request
from pathlib import Path

from . import db, gpu_device

MODEL_ID = 'Qwen/Qwen3.6-35B-A3B:Q4_K_M'
MODEL_REVISION = 'ggml-org:7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5'
DEFAULT_VARIANT = 'mtp'
LEGACY_VARIANT = 'standard'  # what a stored result without a variant was made with
QWEN_SERVER = ('LLAMA_SERVER_PATH', '/models/llamacpp/llama-server')
QWEN_PROJECTOR = ('QWEN36_PROJECTOR_PATH', '/models/qwen3.6-35b/mmproj-Qwen3.6-35B-A3B-BF16.gguf')
INKLING_ID = 'unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M'
# The local engines the text server can run, one at a time (the selected one is the *resident* server: it answers text jobs, chat and
# the visual stages that run on a resident model, see ``visual_models``). ``standard`` and ``mtp`` are two builds of the same
# Qwen3.6-35B-A3B model; ``mtp`` carries the multi-token-prediction head that llama.cpp uses as its own draft model (speculative
# decoding); it is a larger file, so it lives beside the standard one. ``mtp`` is the default since 2026-11-21 and ``standard`` stays
# selectable in the text engine dialog as a rollback. ``inkling`` is Inkling 2 Mini (Thinking Machines Lab, a 26B looped mixture of
# experts with a vision projector, added 2026-11-21): it needs a newer llama.cpp than Qwen, so every variant names its own
# ``server`` binary, and it switches its reasoning with ``reasoning_effort`` where Qwen uses ``enable_thinking`` (``thinking_kwargs``).
VARIANTS = {
    'standard': {'label': 'Qwen 3.6 35B A3B · Q4_K_M', 'id': MODEL_ID, 'revision': MODEL_REVISION, 'size': 20419565568, 'family': 'qwen',
                 'name': 'Qwen 3.6', 'server': QWEN_SERVER, 'projector': QWEN_PROJECTOR,
                 'model': ('QWEN36_MODEL_PATH', '/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf'),
                 'install': 'scripts/download-qwen36.sh', 'projector_install': 'python3 scripts/install-model.py qwen36-vision'},
    'mtp': {'label': 'Qwen 3.6 35B A3B · UD-Q4_K_M + MTP', 'id': 'unsloth/Qwen3.6-35B-A3B-MTP:UD-Q4_K_M',
            'revision': 'unsloth:5bc3e238d916f48a861bac2f8a1990a0e9b7e98d', 'size': 22663387424, 'family': 'qwen',
            'name': 'Qwen 3.6', 'server': QWEN_SERVER, 'projector': QWEN_PROJECTOR,
            'model': ('QWEN36_MTP_MODEL_PATH', '/models/qwen3.6-35b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'),
            'install': 'scripts/download-qwen36-mtp.sh', 'projector_install': 'python3 scripts/install-model.py qwen36-vision'},
    'inkling': {'label': 'Inkling 2 Mini · UD-Q4_K_M', 'id': INKLING_ID, 'revision': 'unsloth:ae9327c5fbc7333116ba32eee78bb5d2bbefae4f',
                'size': 16610543328, 'family': 'inkling', 'name': 'Inkling 2 Mini',
                'server': ('INKLING_SERVER_PATH', '/models/llamacpp-inkling/llama-server'),
                'projector': ('INKLING_PROJECTOR_PATH', '/models/inkling-2-mini/mmproj-F16.gguf'),
                'model': ('INKLING_MODEL_PATH', '/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf'),
                'install': 'python3 scripts/install-model.py inkling2-mini', 'projector_install': 'python3 scripts/install-model.py inkling2-mini'},
}
MTP_DEFAULT_ARGS = '--spec-type draft-mtp --spec-draft-n-max 3'
# Thinking Machines' recommended sampling for Inkling: temperature 1.0 with no truncation. A request that names its own value wins.
INKLING_SAMPLING = ('--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0.0')


def known_variant(variant):
    """A local variant name, or a ``remote:<model>`` name kept as it is (see ``remote_text``).

    A stored value that names no variant (results saved before variants existed) was made by the standard build.
    """
    if isinstance(variant, str) and variant.startswith('remote:'):
        return variant
    return variant if variant in VARIANTS else LEGACY_VARIANT


def model_id_for(variant):
    variant = known_variant(variant)
    return variant if variant.startswith('remote:') else VARIANTS[variant]['id']


def revision_for(variant):
    """The revision to save with a result; a remote model is recorded by its server-side model id."""
    variant = known_variant(variant)
    return variant if variant.startswith('remote:') else VARIANTS[variant]['revision']


def family(variant):
    """``qwen`` or ``inkling``: what the variant's server flags, sampling and reasoning switch depend on."""
    return VARIANTS[known_variant(variant)]['family'] if not str(variant).startswith('remote:') else 'remote'


def thinking_kwargs(enabled=False):
    """The per-request ``chat_template_kwargs`` that turn the local model's reasoning on or off.

    Qwen reads ``enable_thinking`` and Inkling reads ``reasoning_effort``; each chat template ignores the other's
    variable, so both are always sent and a request never needs to know which engine answers it. Inkling's lowest effort
    is not a guaranteed "off" (its model card says so), which is why JSON answers are grammar-constrained as before.
    """
    return {'enable_thinking': bool(enabled), 'reasoning_effort': 'high' if enabled else 'none'}


def sampling_extra(variant):
    """Request fields for the sampling the engine's authors recommend (Cut Assistance chat)."""
    if family(variant) == 'inkling':
        return {'top_p': 1.0, 'top_k': 0, 'min_p': 0.0}
    return {'top_p': 0.95, 'top_k': 20}


def cancel_local(detail='Cancelled'):
    """Stop the local llama-server to abort a cancelled local generation.

    Does nothing on a thread bound to the remote server: its job never used the local model.
    """
    from . import remote_text
    if remote_text.bound() is None and runtime is not None:
        runtime.stop(detail)


def _configured(entry):
    env, default = entry
    return Path(os.environ.get(env) or default)


def variant_path(variant):
    return _configured(VARIANTS[known_variant(variant)]['model'])


def server_path(variant):
    """The llama-server binary of the variant (Inkling has its own, newer build)."""
    return _configured(VARIANTS[known_variant(variant)]['server'])


def variant_args(variant):
    """Extra llama-server arguments for a variant; the MTP flags can be replaced with QWEN36_MTP_ARGS, Inkling's with INKLING_ARGS."""
    variant = known_variant(variant)
    if variant == 'mtp':
        return shlex.split(os.environ.get('QWEN36_MTP_ARGS') or MTP_DEFAULT_ARGS)
    if variant == 'inkling':
        return [*INKLING_SAMPLING, *shlex.split(os.environ.get('INKLING_ARGS') or '')]
    return []


def server_flags(variant):
    """The flags every start of ``variant``'s server shares (the arguments after the model, host, port, slots and context)."""
    if family(variant) == 'inkling':
        return ['--jinja', '--no-context-shift']
    return ['--jinja', '--no-context-shift', '--reasoning-budget', os.environ.get('QWEN36_THINKING_BUDGET', '2048')]


def projector_path(variant=DEFAULT_VARIANT):
    """The vision projector that lets the text server read images (the same file ``visual_models`` gives the visual jobs)."""
    return _configured(VARIANTS[known_variant(variant)]['projector'])


def vision_enabled(variant=DEFAULT_VARIANT):
    """Whether the text server should be started with the projector: it is installed and ``QWEN36_TEXT_VISION`` is not 0."""
    if os.environ.get('QWEN36_TEXT_VISION', '1').strip().lower() in ('0', 'false', 'off', 'no'):
        return False
    return projector_path(variant).is_file()


def vision_status(runtime, variant):
    """``(chat can read images, why not)`` for the text server of ``variant``: what Cut Assistance shows beside the provider."""
    info = VARIANTS[known_variant(variant)]
    if os.environ.get('QWEN36_TEXT_VISION', '1').strip().lower() in ('0', 'false', 'off', 'no'):
        return False, 'Turned off (QWEN36_TEXT_VISION=0)'
    if not projector_path(variant).is_file():
        return False, f"The vision projector is not installed ({info['projector_install']})"
    if runtime is not None and variant in runtime.vision_failed:
        return False, f"{info['name']} would not start with the vision projector, so it runs without it"
    if runtime is not None and runtime.process and runtime.process.poll() is None and runtime.variant == variant and not runtime.vision:
        return False, 'The running server did not load the vision projector'
    return True, ''


def desired_variant():
    """The variant the user selected in the text engine dialog (the default when nothing valid is saved)."""
    row = db.one('SELECT local_model FROM text_generation_settings WHERE id=1')
    chosen = row['local_model'] if row else None
    return chosen if chosen in VARIANTS else DEFAULT_VARIANT


def resolve_variant(want):
    """``want``, or another installed variant (in registry order, so Qwen first) while ``want`` is not installed."""
    if not variant_path(want).is_file():
        for other in VARIANTS:
            if other != want and variant_path(other).is_file():
                return other
    return want


def selected_variant():
    """The variant new work runs on: the selected one, or the other installed one while it is missing."""
    return resolve_variant(desired_variant())


class ServerExited(RuntimeError):
    """The llama-server process ended while it was starting."""


def variant_status(variant):
    variant = known_variant(variant)
    model = variant_path(variant)
    if model.is_file():
        if family(variant) == 'inkling' and not server_path(variant).is_file():  # its llama.cpp build is installed apart from Qwen's: without it every job would fail
            return {'available': False, 'detail': f"llama-server is missing ({server_path(variant)})"}
        return {'available': True, 'detail': 'Installed'}
    partial = model.with_name(model.name + '.partial')
    if partial.is_file():
        try:
            percent = min(100.0, 100 * partial.stat().st_size / VARIANTS[variant]['size'])
        except OSError:
            percent = 0.0
        return {'available': False, 'detail': f'Downloading {percent:.0f}%'}
    return {'available': False, 'detail': 'Not installed'}


def _env_int(name, default, ceiling=None):
    try:
        value = max(1, int(os.environ.get(name, default)))
    except ValueError:
        value = default
    return min(value, ceiling) if ceiling else value


def text_job_concurrency():
    """Text jobs that may run at once (the share of the server's slots that text jobs and chat use)."""
    return _env_int('TEXT_JOB_CONCURRENCY', 2)


def server_slots():
    """Slots of the one resident llama-server (``--parallel``).

    The server also reads pictures for the visual stages, which run several requests at once
    (``FRAME_PARALLEL``, ``KEYFRAME_PARALLEL``, ``SUBTITLE_READ_PARALLEL``), so by default it has as many
    slots as the widest of them. ``QWEN36_SLOTS`` overrides it; stages ask for ``min(their setting, slots)``.
    """
    widest = max(text_job_concurrency(), _env_int('FRAME_PARALLEL', 4), _env_int('KEYFRAME_PARALLEL', 2), _env_int('SUBTITLE_READ_PARALLEL', 4, 8))
    return _env_int('QWEN36_SLOTS', widest)


def context_pool():
    """``--ctx-size`` of the server: with a unified KV cache it is one pool shared by every slot."""
    return _env_int('QWEN36_CONTEXT_SIZE', 65536)


def server_context():
    """``--ctx-size`` for the resident server.

    Unified: the pool itself. Not unified: llama-server splits ``--ctx-size`` evenly, so it is sized so that each slot
    still has the share a text job budgets (``pool // TEXT_JOB_CONCURRENCY``); that costs more memory.
    """
    if kv_unified():
        return context_pool()
    return context_pool() // text_job_concurrency() * server_slots()


def kv_unified():
    """Share one KV pool between the slots (``--kv-unified``), so more slots do not shrink the room of each text job.

    llama-server splits ``--ctx-size`` evenly between explicit slots unless it is unified; with the pool shared a
    slot may use up to the pool while the others are idle. ``QWEN36_KV_UNIFIED=0`` goes back to an even split.
    """
    return os.environ.get('QWEN36_KV_UNIFIED', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def clamp_parallel(wanted, what='stage'):
    """Threads a stage may use against the server: never more than its slots (the rest would only queue)."""
    slots = server_slots()
    if wanted > slots:
        logging.getLogger('framewise.text').warning('%s asks for %d parallel requests but the server has %d slots (QWEN36_SLOTS); using %d', what, wanted, slots, slots)
        return slots
    return wanted


class GpuBusy(Exception):
    """Visual work holds the GPU, so the local chat model cannot be used right now."""


class LocalTextRuntime:
    def __init__(self):
        self.process = None
        self.log = None
        self.url = None
        self.device = None
        self.failed = False
        # Cut Assistance streams from a second thread. The lock serialises
        # load/unload; chat_users keeps the server alive during a chat turn and
        # gpu_busy keeps chat from loading the model while visual work runs.
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.chat_users = 0
        self.inflight = 0  # generation requests on the wire; the model is never switched under one
        self.variant = None  # variant of the running llama-server
        self.vision = False  # the running server loaded the projector, so chat requests may carry images
        self.vision_failed = {}  # variant -> why a start with the projector failed; it then runs without
        self.held = threading.local()  # chat sessions held by the calling thread
        self.gpu_busy = False
        self.publish('unloaded', 'Worker ready')
        self.publish_variants()

    def publish_variants(self):
        """Tell the API which model files are installed or downloading and which one is loaded."""
        try:
            with db.connect() as conn:
                conn.execute('UPDATE local_text_model_state SET variants_json=? WHERE id=1',
                             (json.dumps({name: {**variant_status(name), 'label': info['label']} for name, info in VARIANTS.items()}),))
        except Exception:
            pass  # a locked database must not stop the worker loop

    def desired_variant(self):
        return desired_variant()

    def target_variant(self):
        """The variant new work should use: the selected one, or the other installed one while it is not installed."""
        return resolve_variant(self.desired_variant())

    def can_switch(self):
        """Nothing is generating or holding a chat session, other than the calling thread."""
        return self.inflight == 0 and self.chat_users - getattr(self.held, 'count', 0) <= 0

    def switch_pending(self):
        """The user selected another variant than the one that is loaded (text lanes then claim no new work)."""
        try:
            return bool(self.process and self.process.poll() is None and self.target_variant() != self.variant)
        except Exception:
            return False

    def publish(self, state, detail='', variant=None):
        """Publish the state; ``variant`` ('' when nothing is loaded) is the variant of the running server."""
        with db.connect() as conn:
            if variant is None:
                conn.execute('UPDATE local_text_model_state SET state=?,detail=?,heartbeat=? WHERE id=1', (state, detail[:1000], time.time()))
            else:
                conn.execute('UPDATE local_text_model_state SET state=?,detail=?,heartbeat=?,variant=? WHERE id=1', (state, detail[:1000], time.time(), variant))

    def heartbeat(self, done):
        """Keep the worker visible during long jobs without changing its state."""
        while not done.wait(5):
            try:
                self.yield_to_audio()
                with db.connect() as conn:
                    conn.execute('UPDATE local_text_model_state SET heartbeat=? WHERE id=1', (time.time(),))
            except Exception:
                # A transient database lock must not end the heartbeat for a
                # job that can legitimately run for many minutes.
                continue

    def desired(self):
        row = db.one('SELECT desired_loaded FROM local_text_model_state WHERE id=1')
        return bool(row and row['desired_loaded'])

    def stop(self, detail='', wait_for_chat=True):
        with self.lock:
            if wait_for_chat:
                # A thread cancelling its own job must not wait for the session it holds itself.
                while self.chat_users > getattr(self.held, 'count', 0):
                    self.changed.wait(1)
            self._stop(detail)

    def _stop(self, detail=''):
        if self.process:
            self.publish('unloading', detail or 'Releasing GPU memory')
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
            self.process = None
        if self.log:
            self.log.close(); self.log = None
        if self.device:
            self.device.close()
            self.device = None
        self.url = None
        self.variant = None
        self.vision = False
        self.publish('unloaded', detail, '')

    @contextmanager
    def request(self, cancel_check=None):
        """Yield ``(server url, variant)`` for one generation request.

        The server is (re)started for the selected variant first, but only when no other request or
        chat session is using the running one; the request is counted so a later switch waits for it.
        """
        with self.lock:
            url = self._ensure(cancel_check)
            self.inflight += 1
            variant = self.variant
        try:
            yield url, variant
        finally:
            with self.changed:
                self.inflight -= 1
                self.changed.notify_all()

    def pending_nonlocal_work(self):
        row = db.one("SELECT pipeline,payload_json FROM jobs WHERE status='queued' AND pipeline IN ('analyze','regen','consistency_summary','review_answers','sequence','character_detect','composition_compact') ORDER BY created_at,rowid LIMIT 1")
        if not row:
            return False
        try:
            payload = json.loads(row['payload_json'])
        except (ValueError, TypeError):
            return True
        if not isinstance(payload, dict):
            return True
        if row['pipeline'] == 'character_detect':
            return True
        if row['pipeline'] == 'analyze':
            if not payload.get('text_only'):
                return True
            cuts = payload.get('cuts')
            return not isinstance(cuts, list) or not cuts or not isinstance(cuts[0], dict) or cuts[0].get('config', {}).get('text_provider', 'openrouter') != 'local'
        provider = payload.get('text_provider', 'openrouter') if row['pipeline'] in ('regen','composition_compact') else payload.get('config', {}).get('text_provider', 'openrouter')
        return provider != 'local'

    @staticmethod
    def model_path(variant=DEFAULT_VARIANT):
        return variant_path(variant)

    def download_status(self, variant=None):
        variant = variant or self.target_variant()
        model = variant_path(variant)
        partial = model.with_name(model.name + '.partial')
        if not model.is_file() and partial.is_file():
            size = partial.stat().st_size
            percent = min(100.0, 100 * size / VARIANTS[variant]['size'])
            return partial, f'Downloading model: {percent:.1f}% (includes checksum verification)'
        return None, ''

    def begin_gpu_work(self):
        """Visual work owns the GPU: block new chat loads, then wait for active chat turns."""
        with self.changed:
            self.gpu_busy = True

    def yield_to_audio(self):
        """Unload the idle model for a leased audio job after any current chat finishes."""
        if self.device and gpu_device.audio_requested():
            with self.changed:
                if self.device and gpu_device.audio_requested():
                    self.stop('Released for audio processing')

    def device_busy(self):
        return self.gpu_busy or gpu_device.audio_requested()

    def end_gpu_work(self):
        with self.changed:
            self.gpu_busy = False
            self.changed.notify_all()

    @contextmanager
    def chat_session(self, cancel_check=None, on_wait=None, wait=True):
        """Hold the text server for one chat turn, waiting while visual work uses the GPU.

        With ``wait=False`` a busy GPU raises ``GpuBusy`` at once, so the caller can answer elsewhere.
        """
        with self.changed:
            if self.device_busy() and not wait:
                raise GpuBusy()
            while self.device_busy():
                if on_wait:
                    on_wait('Waiting for the GPU')
                self.changed.wait(0.5)
                if cancel_check:
                    cancel_check()
            self.chat_users += 1
            self.held.count = getattr(self.held, 'count', 0) + 1
        try:
            yield
        finally:
            with self.changed:
                self.chat_users -= 1
                self.held.count -= 1
                self.changed.notify_all()

    @contextmanager
    def visual_session(self, cancel_check=None, variants=None):
        """Yield the resident server's URL for one visual stage (pictures, subtitle strips, a character profile).

        The same server answers text and chat, so a visual job neither stops it nor starts a private one. The
        session counts like a chat turn: the server is not stopped or switched under it. It is not held back by
        ``device_busy()``, because the visual job itself raised that flag to keep new chat turns out. A server that
        could not load the vision projector is an error here: the stage would send pictures to a model that cannot see them.
        """
        with self.lock:
            gpu_device.release_held()  # a job that had the whole GPU hands it to the server it is about to need
            url = self._ensure(cancel_check)
            if not url:
                raise RuntimeError('The resident model server could not be started')
            if variants and self.variant not in variants:  # the user selected another text engine while this stage waited
                raise RuntimeError(f"The text engine was switched to {VARIANTS[self.variant]['name']} while this stage was waiting; queue it again")
            if not self.vision:
                reason = vision_status(self, self.variant)[1] or 'the server did not report vision in /props'
                raise RuntimeError(f"{VARIANTS[self.variant]['name']} is running without its vision projector ({reason}), so it cannot read pictures")
            self.chat_users += 1
            self.held.count = getattr(self.held, 'count', 0) + 1
        try:
            yield url
        finally:
            with self.changed:
                self.chat_users -= 1
                self.held.count -= 1
                self.changed.notify_all()

    def ensure(self, cancel_check=None, idle_load=False):
        with self.lock:
            return self._ensure(cancel_check, idle_load)

    def _ensure(self, cancel_check=None, idle_load=False):
        variant = self.target_variant()
        if self.process and self.process.poll() is None:
            if variant == self.variant or not self.can_switch():
                self.publish('loaded', 'Ready')
                return self.url
            self._stop(f'Switching to {VARIANTS[variant]["label"]}')
        else:
            self._stop()
        runtime = server_path(variant)
        model = variant_path(variant)
        while not model.is_file():
            partial, detail = self.download_status(variant)
            if partial is None:
                break
            if cancel_check:
                cancel_check()
            if idle_load and (not self.desired() or self.pending_nonlocal_work()):
                self.publish('downloading', detail); return None
            try:
                stale = time.time() - partial.stat().st_mtime > 300
            except FileNotFoundError:
                continue
            if stale:
                self.failed = True
                self.publish('error', 'Model download has stopped; resume it before loading')
                raise RuntimeError('Model download has stopped; resume it before loading')
            self.publish('downloading', detail)
            time.sleep(2)
        if not runtime.is_file() or not model.is_file():
            self.failed = True
            detail = f"{VARIANTS[variant]['name']} or its llama-server is not installed ({VARIANTS[variant]['install']})"
            self.publish('error', detail)
            raise RuntimeError(detail)
        # Chat looks at pictures through the same projector the visual jobs use. A build or a GPU that cannot start with it
        # must not take every text job down, so the server is started again without it and the reason is remembered.
        attempts = [True, False] if vision_enabled(variant) and variant not in self.vision_failed else [False]
        for with_vision in attempts:
            try:
                return self._launch(runtime, model, variant, with_vision, cancel_check, idle_load)
            except ServerExited as exc:
                if not with_vision:
                    raise
                self.vision_failed[variant] = str(exc)[-600:]
                logging.getLogger('framewise.text').warning('%s did not start with its vision projector; starting it without: %s', VARIANTS[variant]['name'], str(exc)[-400:])

    def _launch(self, runtime, model, variant, with_vision, cancel_check, idle_load):
        if idle_load and gpu_device.audio_requested():
            return None
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
        self.log = tempfile.TemporaryFile(mode='w+t')
        env = os.environ.copy()
        env.pop('OPENROUTER_API_KEY', None)
        env['LD_LIBRARY_PATH'] = str(runtime.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        slots = server_slots()
        tags = [name for name, on in (('MTP', variant == 'mtp'), ('vision', with_vision)) if on] + [f'{slots} slots']
        loading = f"Loading {VARIANTS[variant]['name']} into GPU memory" + (f" ({', '.join(tags)})" if tags else '')
        self.publish('loading', loading)
        try:
            self.device = gpu_device.acquire(cancel_check)
            command = [str(runtime), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                '--n-gpu-layers', '99', '--ctx-size', str(server_context()), '--parallel', str(slots), *(['--kv-unified'] if kv_unified() else []),
                *server_flags(variant), *(['--mmproj', str(projector_path(variant))] if with_vision else []), *variant_args(variant)]
            logging.getLogger('framewise.text').info('Starting llama-server: %s', ' '.join(command[1:]))
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, env=env)
            self.url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 600
            while True:
                if cancel_check:
                    cancel_check()
                if idle_load and (not self.desired() or gpu_device.audio_requested() or self.pending_nonlocal_work()):
                    self._stop('Waiting for queued work'); return None
                if self.process.poll() is not None:
                    self.log.seek(0); raise ServerExited('Local model failed to start: ' + self.log.read()[-1000:])
                try:
                    with urllib.request.urlopen(self.url + '/health', timeout=2) as response:
                        if response.status == 200:
                            self.failed = False; self.variant = variant
                            self.vision = with_vision and self.reports_vision(self.url)
                            self.publish('loaded', 'Ready', variant); return self.url
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Local model did not become ready in time')
                self.publish('loading', loading)
                time.sleep(1)
        except Exception as exc:
            self._stop()
            if not (with_vision and isinstance(exc, ServerExited)):  # that one is retried without the projector by the caller
                self.failed = True; self.publish('error', str(exc))
            raise

    @staticmethod
    def reports_vision(url):
        """Whether the started server says it can read images (llama-server lists its ``modalities`` in /props)."""
        try:
            with urllib.request.urlopen(url + '/props', timeout=5) as response:
                modalities = json.load(response).get('modalities')
            return bool(modalities.get('vision')) if isinstance(modalities, dict) else True
        except (OSError, ValueError, urllib.error.URLError, AttributeError):
            return True  # it was started with the projector; an unreadable /props is not a reason to hide the tool

    def reconcile(self):
        with self.lock:
            self._reconcile()

    def _reconcile(self):
        self.publish_variants()
        if gpu_device.audio_requested():
            self.yield_to_audio()
            self.publish('unloaded', 'GPU is processing audio')
            return
        want = self.desired()
        if not want:
            self.failed = False
            if self.process:
                self.stop()
            else:
                partial, detail = self.download_status()
                try:
                    stale = partial and time.time() - partial.stat().st_mtime > 300
                except FileNotFoundError:
                    partial, stale = None, False
                if stale:
                    self.publish('error', 'Model download appears stalled; resume it before loading')
                else:
                    self.publish('downloading' if partial else 'unloaded', detail)
        elif self.process and self.process.poll() is None:
            if self.switch_pending() and self.can_switch():
                self._stop('Switching text model')
                try:
                    self.ensure(idle_load=True)
                except Exception:
                    pass
            else:
                self.publish('loaded', 'Switching text model when running jobs finish' if self.switch_pending() else 'Ready')
        elif not self.failed:
            try:
                self.ensure(idle_load=True)
            except Exception:
                pass
        else:
            with db.connect() as conn:
                conn.execute('UPDATE local_text_model_state SET heartbeat=? WHERE id=1', (time.time(),))


runtime = None
