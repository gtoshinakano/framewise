"""Start-time snapshots and atomic publication for queue handlers."""
import hashlib
import json
import time
from contextlib import contextmanager

from . import db, job_kinds

QUIET_SECONDS = 3
RESTART_STAGE = 'Inputs changed, restarting'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def change_wait(conn, job, now):
    """Check edits made during the quiet period before claiming a restarted job."""
    encoded = job['restart_input_json']
    if not encoded:
        return False
    old = json.loads(encoded)
    kind = job_kinds.KINDS[job['pipeline']]
    current = kind.inputs(conn, {**job, 'unit_id': old['unit_id']})
    if current == old['fingerprint']:
        return False
    _delay(conn, job['id'], old['unit_id'], current, now)
    return True


def _delay(conn, job_id, unit_id, stamp, now):
    conn.execute('UPDATE jobs SET not_before=?,restart_input_json=? WHERE id=?',
                 (now + QUIET_SECONDS, json.dumps({'unit_id': unit_id, 'fingerprint': stamp}), job_id))


class JobInputs:
    """A job's inputs, or one checkpointable unit within a batch."""

    def __init__(self, conn, job, owner, unit_id=None):
        self.job = {**job, 'unit_id': unit_id}
        self.kind = job_kinds.KINDS[job['pipeline']]
        self.owner = owner
        self.data = self.kind.gather(conn, self.job)
        self.stamp = self.kind.inputs(conn, {**self.job, 'gathered': self.data})

    @contextmanager
    def publish(self, *, invalidate=True, on_changed=None, **output):
        """Validate and publish within one write transaction.

        Raising after commit preserves the requeue and keeps callers from publishing stale output.
        The worker releases its lease only after it has stopped using its resources.
        """
        from . import worker
        job = self.job
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            worker.require_lease(conn, job['id'], self.owner)
            current = self.kind.inputs(conn, job)
            if current == self.stamp:
                yield conn
                if invalidate:
                    self.kind.invalidates(conn, {**job, **output})
                return
            if self.kind.on_inputs_changed != 'requeue':
                raise ValueError(f"Invalid input change policy for {job['pipeline']}")
            if on_changed:
                on_changed(conn)
            conn.execute("""UPDATE jobs SET status='queued',stage=?,error=NULL,attempt_count=MAX(0,attempt_count-1),
                updated_at=CURRENT_TIMESTAMP WHERE id=?""", (RESTART_STAGE, job['id']))
            now = time.time()
            _delay(conn, job['id'], job['unit_id'], current, now)
            conn.execute("""UPDATE analysis_job_stages SET status='interrupted',finished_at=?
                WHERE job_id=? AND status='running'""", (now, job['id']))
            if job['pipeline'] not in worker.SIDE_PIPELINES:
                worker.sync_episode_status(conn, job['episode_id'])
        raise worker.LeaseLost(RESTART_STAGE)
