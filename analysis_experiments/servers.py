"""A private Qwen3.6 llama-server for the written conclusions (no LocalTextRuntime, no database)."""
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def text_server(startup_timeout=600):
    """Start llama-server for the standard Qwen3.6-35B-A3B file with the flags ``LocalTextRuntime`` uses; one slot."""
    runtime = Path(os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'))
    model = Path(os.environ.get('QWEN36_MODEL_PATH', '/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf'))
    if not runtime.is_file() or not model.is_file():
        raise RuntimeError('Qwen3.6 model or llama-server runtime is missing')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryFile(mode='w+t') as log:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(runtime.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        process = subprocess.Popen([str(runtime), '-m', str(model), '--host', '127.0.0.1', '--port', str(port), '--n-gpu-layers', '99', '--ctx-size',
                                    os.environ.get('QWEN36_CONTEXT_SIZE', '65536'), '--parallel', '1', '--jinja', '--no-context-shift', '--reasoning-budget',
                                    os.environ.get('QWEN36_THINKING_BUDGET', '2048')], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError('Text model server failed to start: ' + log.read()[-1800:])
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('Text model server did not become ready in time')
                time.sleep(1)
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
