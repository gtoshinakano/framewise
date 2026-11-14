"""Worker-owned local text model lifecycle; the API controls desired state via SQLite."""
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

from . import db

MODEL_ID = 'Qwen/Qwen3.6-35B-A3B:Q4_K_M'
MODEL_REVISION = 'ggml-org:7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5'
DEFAULT_VARIANT = 'standard'
# Two builds of the same Qwen3.6-35B-A3B text model. ``mtp`` carries the multi-token-prediction
# head that llama.cpp uses as its own draft model (speculative decoding); it is a larger file, so
# it lives beside the standard one and the user switches between them in the text engine dialog.
VARIANTS = {
    'standard': {'label': 'Qwen 3.6 35B A3B · Q4_K_M', 'id': MODEL_ID, 'revision': MODEL_REVISION, 'size': 20419565568},
    'mtp': {'label': 'Qwen 3.6 35B A3B · UD-Q4_K_M + MTP', 'id': 'unsloth/Qwen3.6-35B-A3B-MTP:UD-Q4_K_M',
            'revision': 'unsloth:5bc3e238d916f48a861bac2f8a1990a0e9b7e98d', 'size': 22663387424},
}
MTP_DEFAULT_ARGS = '--spec-type draft-mtp --spec-draft-n-max 3'


def known_variant(variant):
    return variant if variant in VARIANTS else DEFAULT_VARIANT


def model_id_for(variant):
    return VARIANTS[known_variant(variant)]['id']


def revision_for(variant):
    return VARIANTS[known_variant(variant)]['revision']


def variant_path(variant):
    if known_variant(variant) == 'mtp':
        return Path(os.environ.get('QWEN36_MTP_MODEL_PATH', '/models/qwen3.6-35b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'))
    return Path(os.environ.get('QWEN36_MODEL_PATH', '/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf'))


def variant_args(variant):
    """Extra llama-server arguments for a variant; the MTP flags can be replaced with QWEN36_MTP_ARGS."""
    if known_variant(variant) != 'mtp':
        return []
    return shlex.split(os.environ.get('QWEN36_MTP_ARGS') or MTP_DEFAULT_ARGS)


def variant_status(variant):
    variant = known_variant(variant)
    model = variant_path(variant)
    if model.is_file():
        return {'available': True, 'detail': 'Installed'}
    partial = model.with_name(model.name + '.partial')
    if partial.is_file():
        try:
            percent = min(100.0, 100 * partial.stat().st_size / VARIANTS[variant]['size'])
        except OSError:
            percent = 0.0
        return {'available': False, 'detail': f'Downloading {percent:.0f}%'}
    return {'available': False, 'detail': 'Not installed'}


def text_job_concurrency():
    """Text jobs that may run at once; the llama-server gets one slot for each."""
    try:
        return max(1, int(os.environ.get('TEXT_JOB_CONCURRENCY', '2')))
    except ValueError:
        return 2


class GpuBusy(Exception):
    """Visual work holds the GPU, so the local chat model cannot be used right now."""


class LocalTextRuntime:
    def __init__(self):
        self.process = None
        self.log = None
        self.url = None
        self.failed = False
        # Cut Assistance streams from a second thread. The lock serialises
        # load/unload; chat_users keeps the server alive during a chat turn and
        # gpu_busy keeps chat from loading the model while visual work runs.
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.chat_users = 0
        self.inflight = 0  # generation requests on the wire; the model is never switched under one
        self.variant = None  # variant of the running llama-server
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
        row = db.one('SELECT local_model FROM text_generation_settings WHERE id=1')
        return known_variant(row['local_model'] if row else None)

    def target_variant(self):
        """The variant new work should use: the selected one, or the standard one while it is not installed."""
        want = self.desired_variant()
        if want != DEFAULT_VARIANT and not variant_path(want).is_file():
            return DEFAULT_VARIANT
        return want

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
        self.url = None
        self.variant = None
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
            if self.gpu_busy and not wait:
                raise GpuBusy()
            while self.gpu_busy:
                if on_wait:
                    on_wait('Waiting for the GPU: visual analysis is running')
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
        runtime = Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'))
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
            self.publish('error', 'Local text model or llama-server is not installed')
            raise RuntimeError('Local text model or llama-server is not installed')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
        self.log = tempfile.TemporaryFile(mode='w+t')
        env = os.environ.copy()
        env.pop('OPENROUTER_API_KEY', None)
        env['LD_LIBRARY_PATH'] = str(runtime.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        loading = 'Loading Qwen 3.6 into GPU memory' + (' (MTP)' if variant == 'mtp' else '')
        self.publish('loading', loading)
        try:
            self.process = subprocess.Popen([str(runtime), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                '--n-gpu-layers', '99', '--ctx-size', os.environ.get('QWEN36_CONTEXT_SIZE', '65536'), '--parallel', str(text_job_concurrency()), '--jinja', '--no-context-shift', '--reasoning-budget', os.environ.get('QWEN36_THINKING_BUDGET', '2048'), *variant_args(variant)],
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, env=env)
            self.url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 600
            while True:
                if cancel_check:
                    cancel_check()
                if idle_load and (not self.desired() or self.pending_nonlocal_work()):
                    self._stop('Waiting for queued work'); return None
                if self.process.poll() is not None:
                    self.log.seek(0); raise RuntimeError('Local model failed to start: ' + self.log.read()[-1000:])
                try:
                    with urllib.request.urlopen(self.url + '/health', timeout=2) as response:
                        if response.status == 200:
                            self.failed = False; self.variant = variant
                            self.publish('loaded', 'Ready', variant); return self.url
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Local model did not become ready in time')
                self.publish('loading', loading)
                time.sleep(1)
        except Exception as exc:
            self._stop(); self.failed = True; self.publish('error', str(exc)); raise

    def reconcile(self):
        with self.lock:
            self._reconcile()

    def _reconcile(self):
        self.publish_variants()
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
