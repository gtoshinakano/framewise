"""Keep model teardown and the idle text server inside the shared GPU boundary."""
import fcntl
import threading
import time
from contextlib import contextmanager

from . import db, job_kinds


def acquire(check=None):
    handle = (db.DATA_DIR / '.gpu.lock').open('a+b')
    try:
        while True:
            if check:
                check()
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                time.sleep(0.2)
    except BaseException:
        handle.close()
        raise


_held = threading.local()


@contextmanager
def hold(check=None):
    with acquire(check) as handle:
        stack = _held.__dict__.setdefault('stack', [])
        stack.append(handle)
        try:
            yield
        finally:
            stack.remove(handle)


def release_held():
    """Let go of the device lock the calling thread holds through ``hold``, if it holds one.

    A job that kept the GPU to itself (subtitle locating, the older character matcher) calls this before the resident
    Qwen server loads: the server takes the lock for as long as it runs, and a second lock on the same file from the
    same process would wait for the first one forever. The ``hold`` block still ends normally.
    """
    for handle in getattr(_held, 'stack', ()):
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass


def audio_requested():
    pipelines = job_kinds.pipelines_for('audio')
    marks = ','.join('?' for _ in pipelines)
    return bool(db.one(f'''SELECT 1 FROM jobs WHERE lease_owner IS NOT NULL AND lease_until>?
        AND pipeline IN ({marks}) LIMIT 1''', (time.time(), *pipelines)))
