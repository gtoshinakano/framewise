"""Worker-owned local text model lifecycle; the API controls desired state via SQLite."""
import os
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import db

MODEL_ID = 'Qwen/Qwen3.6-35B-A3B:Q4_K_M'
MODEL_REVISION = 'ggml-org:7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5'


class LocalTextRuntime:
    def __init__(self):
        self.process = None
        self.log = None
        self.url = None
        self.failed = False
        self.publish('unloaded', 'Worker ready')

    def publish(self, state, detail=''):
        with db.connect() as conn:
            conn.execute('UPDATE local_text_model_state SET state=?,detail=?,heartbeat=? WHERE id=1', (state, detail[:1000], time.time()))

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

    def stop(self, detail=''):
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
        self.publish('unloaded', detail)

    def pending_nonlocal_work(self):
        row = db.one("SELECT pipeline,payload_json FROM jobs WHERE status='queued' AND pipeline IN ('analyze','regen','consistency_summary','character_detect','composition_compact') ORDER BY created_at,rowid LIMIT 1")
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
    def model_path():
        return Path(os.environ.get('QWEN36_MODEL_PATH', '/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf'))

    def download_status(self):
        model = self.model_path()
        partial = model.with_name(model.name + '.partial')
        if not model.is_file() and partial.is_file():
            size = partial.stat().st_size
            percent = min(100.0, 100 * size / 20419565568)
            return partial, f'Downloading model: {percent:.1f}% (includes checksum verification)'
        return None, ''

    def ensure(self, cancel_check=None, idle_load=False):
        if self.process and self.process.poll() is None:
            self.publish('loaded', 'Ready')
            return self.url
        self.stop()
        runtime = Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'))
        model = self.model_path()
        while not model.is_file():
            partial, detail = self.download_status()
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
        self.publish('loading', 'Loading Qwen 3.6 into GPU memory')
        try:
            self.process = subprocess.Popen([str(runtime), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                '--n-gpu-layers', '99', '--ctx-size', os.environ.get('QWEN36_CONTEXT_SIZE', '65536'), '--parallel', '1', '--jinja', '--no-context-shift', '--reasoning-budget', os.environ.get('QWEN36_THINKING_BUDGET', '2048')],
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, env=env)
            self.url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 600
            while True:
                if cancel_check:
                    cancel_check()
                if idle_load and (not self.desired() or self.pending_nonlocal_work()):
                    self.stop('Waiting for queued work'); return None
                if self.process.poll() is not None:
                    self.log.seek(0); raise RuntimeError('Local model failed to start: ' + self.log.read()[-1000:])
                try:
                    with urllib.request.urlopen(self.url + '/health', timeout=2) as response:
                        if response.status == 200:
                            self.failed = False; self.publish('loaded', 'Ready'); return self.url
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Local model did not become ready in time')
                self.publish('loading', 'Loading Qwen 3.6 into GPU memory')
                time.sleep(1)
        except Exception as exc:
            self.stop(); self.failed = True; self.publish('error', str(exc)); raise

    def reconcile(self):
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
            self.publish('loaded', 'Ready')
        elif not self.failed:
            try:
                self.ensure(idle_load=True)
            except Exception:
                pass
        else:
            with db.connect() as conn:
                conn.execute('UPDATE local_text_model_state SET heartbeat=? WHERE id=1', (time.time(),))


runtime = None
