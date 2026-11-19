"""Keep model teardown and the idle text server inside the shared GPU boundary."""
import fcntl
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


@contextmanager
def hold(check=None):
    with acquire(check):
        yield


def audio_requested():
    pipelines = job_kinds.pipelines_for('audio')
    marks = ','.join('?' for _ in pipelines)
    return bool(db.one(f'''SELECT 1 FROM jobs WHERE lease_owner IS NOT NULL AND lease_until>?
        AND pipeline IN ({marks}) LIMIT 1''', (time.time(), *pipelines)))
