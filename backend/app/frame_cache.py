"""Bounded, disposable in-memory image cache and background frame queue."""
import asyncio
import concurrent.futures
import logging
import math
import subprocess
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
    return (str(path), st.st_mtime_ns, st.st_size, spec['time'], spec['start'], spec['end'], spec['width'], spec['last'], spec.get('quality'), spec.get('seek_tolerance', 0), spec.get('probe_padding', 0), spec.get('ordinal'))


def _target_frame(spec, output):
    """Use fast timestamp seeking only when the target PTS uniquely identifies a frame.

    Duplicate or unprobeable PTS require the slower source-ordinal path. Temporary
    JPEGs live only inside the preview directory owned by the caller.
    """
    source = Path(spec['source'])
    target = spec['time']
    try:
        origin_text = media.run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time', '-of', 'default=noprint_wrappers=1:nokey=1', str(source)], timeout=30).stdout.strip()
        origin = float(origin_text)
        if not math.isfinite(origin): raise ValueError('Invalid video start time')
        target_pts = target + origin
        interval = f'{max(origin, target_pts - .25):.9f}%{target_pts + .25:.9f}'
        probe = media.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-read_intervals', interval,
                           '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'csv=p=0', str(source)], timeout=30)
        times = []
        for line in probe.stdout.splitlines():
            try: times.append(float(line.split(',')[0]))
            except ValueError: continue
        if sum(abs(t - target_pts) <= .000010 for t in times) == 1:
            actual = media.extract_cut_frame(source, output, target, spec['start'], spec['end'], width=spec['width'],
                                             seek_tolerance=.000002, timeout=60)
            if abs(actual - target) <= .000005:
                return output.read_bytes()
    except (RuntimeError, ValueError, OSError):
        pass  # Ordinal decoding is the correctness fallback.
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(source), '-map', '0:v:0',
               '-vf', f"select=eq(n\\,{spec['ordinal']}),scale=w='min(iw,{spec['width']})':h=-2", '-fps_mode', 'passthrough',
               '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=120, check=True)
    if not result.stdout: raise RuntimeError('Could not extract target frame')
    return result.stdout


def _generate(spec):
    # Existing media helpers write via atomic temporary files; the directory is
    # removed as soon as the bytes are read, even on extraction errors.
    with tempfile.TemporaryDirectory(prefix='framewise-preview-') as tmp:
        output = Path(tmp) / 'frame.jpg'
        if spec.get('ordinal') is not None:
            before = Path(spec['source']).stat()
            data = _target_frame(spec, output)
            after = Path(spec['source']).stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError('Original video changed')
            return data
        media.extract_cut_frame(Path(spec['source']), output, spec['time'], spec['start'], spec['end'], width=spec['width'], last=spec['last'], quality=spec.get('quality'), seek_tolerance=spec.get('seek_tolerance', 0), probe_padding=spec.get('probe_padding', 0), timeout=60)
        return output.read_bytes()


def render(spec):
    """One frame now, on the caller's thread: for jobs that need a full-size image. Nothing is cached or queued."""
    return _generate(dict(spec))


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
    prefixes = ('framewise-preview-',) if active else ('framewise-preview-', 'framewise-boundaries-', 'framewise-analysis-', 'framewise-cut-analysis-', 'framewise-short-cut-')
    for prefix in prefixes:
        for path in Path(tempfile.gettempdir()).glob(prefix + '*'):
            try:
                if not path.is_symlink() and path.is_dir() and time.time() - path.stat().st_mtime > 86400:
                    shutil.rmtree(path)
            except OSError:
                log.warning('Could not remove an abandoned frame directory: %s', path)
