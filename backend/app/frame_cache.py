"""Bounded, disposable image caches and background frame queue.

Frames live briefly in memory and, for the pages that show them, in a size-capped disk cache under
``DATA_DIR/cache/frames``: outside the projects, never beside an original, deleted oldest first and safe to remove at
any time. Files are named by the hash of everything that decides the picture (the source file's identity and every
render setting), so a changed source or setting simply misses. The same hash is the frame's HTTP ETag.
"""
import asyncio
import concurrent.futures
import hashlib
import logging
import math
import os
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
DISK_MAX_BYTES = int(os.environ.get('FRAME_CACHE_MB', '1024')) * 1024 * 1024
DISK_PRUNE_TO = .9  # prune below the cap so every write does not trigger another prune
VERSIONED = 'private, max-age=31536000, immutable'  # the URL names every input of the picture
REVALIDATE = 'private, no-cache'  # ask every time; an unchanged frame answers 304 without rendering
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
    key = (str(path), st.st_mtime_ns, st.st_size, spec['time'], spec['start'], spec['end'], spec['width'], spec['last'], spec.get('quality'), spec.get('seek_tolerance', 0), spec.get('probe_padding', 0), spec.get('ordinal'))
    # Only a cropped frame carries the crop, so the keys (file names, ETags) of every other frame stay as they were.
    return key + (tuple(spec['crop']), spec.get('crop_height')) if spec.get('crop') else key


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
        if spec.get('crop'):
            cropped = Path(tmp) / 'crop.jpg'
            media.crop_image(output, cropped, tuple(spec['crop']), timeout=60, height=spec.get('crop_height'))
            return cropped.read_bytes()
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
        future = _pool.submit(_generate_and_store, key, dict(spec))
        _inflight[key] = future
    future.add_done_callback(lambda done: _complete(key, done))
    return future


async def get(spec):
    # The FFmpeg work is queued, deduplicated and performed outside the API
    # event loop. Cancelling one request must not cancel another's shared work.
    # Only pages read the disk cache: storage cleanup and frame_gc compare old JPEGs with a fresh render before
    # deleting them, so they go through request() alone.
    key = _key(spec)
    with _lock:
        cached = _cache.get(key)
    if cached is None or cached[0] <= time.monotonic():
        stored = await asyncio.to_thread(_read_stored, key)
        if stored is not None:
            _remember(key, stored)
            return stored
    return await asyncio.shield(asyncio.wrap_future(request(spec)))


def _digest(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def etag(spec):
    return f'"{_digest(_key(spec))}"'


def http_headers(spec, request=None, versioned=False):
    """ETag and caching of a frame response. A caller without the HTTP request keeps the old short cache."""
    policy = 'private, max-age=120' if request is None else VERSIONED if versioned else REVALIDATE
    return {'ETag': etag(spec), 'Cache-Control': policy}


def unchanged(request, headers):
    """Whether the browser already holds this exact frame (If-None-Match), so a 304 can skip rendering and reading."""
    sent = request.headers.get('if-none-match') if request is not None else None
    return bool(sent) and (sent.strip() == '*' or headers['ETag'] in (tag.strip().removeprefix('W/') for tag in sent.split(',')))


def _stored_path(key):
    digest = _digest(key)
    return db.DATA_DIR / 'cache' / 'frames' / digest[:2] / f'{digest}.jpg'


def _read_stored(key):
    path = _stored_path(key)
    try:
        data = path.read_bytes()
        if not data.startswith(b'\xff\xd8'):
            path.unlink(missing_ok=True)
            return None
        os.utime(path)  # the prune removes the least recently used first
        return data
    except OSError:
        return None


def _store(key, data):
    path = _stored_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='.', suffix='.partial', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        # A full disk must not break previews; the frame is simply rendered again next time.
        log.warning('Could not store a frame in the cache: %s', exc)


def _generate_and_store(key, spec):
    data = _generate(spec)
    _store(key, data)
    return data


def _remember(key, data):
    with _lock:
        _cache[key] = (time.monotonic() + TTL_SECONDS, data)
        _cache.move_to_end(key)
        _prune(time.monotonic())


def prune_disk(max_bytes=None):
    """Delete the least recently used cached frames until the cache is under its cap; returns the bytes removed."""
    cap = DISK_MAX_BYTES if max_bytes is None else max_bytes
    files, total, removed = [], 0, 0
    now = time.time()
    for path in (db.DATA_DIR / 'cache' / 'frames').glob('*/*'):
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.name.endswith('.partial'):
            if now - stat.st_mtime > 3600:  # left by a process that stopped mid-write
                path.unlink(missing_ok=True)
            continue
        files.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size
    if total <= cap:
        return 0
    files.sort()
    for _, size, path in files:
        if total - removed <= cap * DISK_PRUNE_TO:
            break
        try:
            path.unlink()
            removed += size
        except OSError:
            continue
    return removed


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
