"""Bounded, disposable in-memory image cache and background frame queue."""
import asyncio
import concurrent.futures
import logging
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

from . import db, media

log = logging.getLogger(__name__)
MAX_BYTES = 64 * 1024 * 1024
TTL_SECONDS = 120
MAX_PENDING = 128
_lock = threading.Lock()
_cache = OrderedDict()
_inflight = {}
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='frame-preview')


def _prune(now):
    for key, (expires, _) in list(_cache.items()):
        if expires <= now:
            del _cache[key]
    while sum(len(value[1]) for value in _cache.values()) > MAX_BYTES:
        _cache.popitem(last=False)


def _key(spec):
    path = Path(spec['source'])
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size, spec['time'], spec['start'], spec['end'], spec['width'], spec['last'], spec.get('quality'), spec.get('seek_tolerance', 0), spec.get('probe_padding', 0))


def _generate(spec):
    # Existing media helpers write via atomic temporary files; the directory is
    # removed as soon as the bytes are read, even on extraction errors.
    with tempfile.TemporaryDirectory(prefix='framewise-preview-') as tmp:
        output = Path(tmp) / 'frame.jpg'
        media.extract_cut_frame(Path(spec['source']), output, spec['time'], spec['start'], spec['end'], width=spec['width'], last=spec['last'], quality=spec.get('quality'), seek_tolerance=spec.get('seek_tolerance', 0), probe_padding=spec.get('probe_padding', 0), timeout=60)
        return output.read_bytes()


def _complete(key, future):
    with _lock:
        _inflight.pop(key, None)
        if not future.cancelled() and future.exception() is None:
            data = future.result()
            _cache[key] = (time.monotonic() + TTL_SECONDS, data)
            _cache.move_to_end(key)
            _prune(time.monotonic())


def request(spec):
    key = _key(spec)
    with _lock:
        _prune(time.monotonic())
        if key in _cache:
            future = concurrent.futures.Future()
            future.set_result(_cache[key][1])
            _cache.move_to_end(key)
            return future
        if key in _inflight:
            return _inflight[key]
        if len(_inflight) >= MAX_PENDING:
            raise RuntimeError('Frame preview queue is full; try again shortly')
        future = _pool.submit(_generate, dict(spec))
        _inflight[key] = future
    future.add_done_callback(lambda done: _complete(key, done))
    return future


async def get(spec):
    # The FFmpeg work is queued, deduplicated and performed outside the API
    # event loop. Cancelling one request must not cancel another's shared work.
    return await asyncio.shield(asyncio.wrap_future(request(spec)))


def clear_expired():
    with _lock:
        _prune(time.monotonic())


def clear_abandoned_temporary_files():
    # Don't touch worker scratch while any media, inference, or audio job is active.
    active = db.one("SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1")
    prefixes = ('framewise-preview-',) if active else ('framewise-preview-', 'framewise-boundaries-', 'framewise-analysis-')
    for prefix in prefixes:
        for path in Path(tempfile.gettempdir()).glob(prefix + '*'):
            try:
                if not path.is_symlink() and path.is_dir() and time.time() - path.stat().st_mtime > 86400:
                    shutil.rmtree(path)
            except OSError:
                log.warning('Could not remove an abandoned frame directory: %s', path)
