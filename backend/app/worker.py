import json
import logging
import math
import re
import threading
import time
import uuid
from contextlib import contextmanager

from pathlib import Path

from . import db, job_inputs, job_kinds, media

log = logging.getLogger(__name__)
worker_thread: threading.Thread | None = None
export_thread: threading.Thread | None = None
wake_event = threading.Event()
stop_event = threading.Event()
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 10
MAX_ATTEMPTS = 3


def start_worker():
    global worker_thread, export_thread
    if worker_thread and worker_thread.is_alive():
        return
    for episode in db.all_rows("SELECT id,duration,settings_json FROM episodes WHERE duration IS NOT NULL AND NOT EXISTS (SELECT 1 FROM chunks WHERE chunks.episode_id=episodes.id)"):
        settings = json.loads(episode["settings_json"])
        sync_chunks(episode["id"], episode["duration"], settings.get("chunk_duration", 60))
    stop_event.clear()
    worker_thread = threading.Thread(target=work_loop, name="media-worker", daemon=True)
    worker_thread.start()
    export_thread = threading.Thread(target=work_loop, args=(("brief_export",),), name="export-worker", daemon=True)
    export_thread.start()


def stop_worker():
    stop_event.set()
    wake_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=2)
    if export_thread and export_thread.is_alive():
        export_thread.join(timeout=2)


def sync_chunks(episode_id: str, duration: float, chunk_duration: int, *, conn=None):
    if not math.isfinite(duration) or duration <= 0:
        return
    if conn is None:
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            sync_chunks(episode_id, duration, chunk_duration, conn=conn)
        return
    previous = db.all_rows("SELECT * FROM chunks WHERE episode_id=? ORDER BY chunk_index", (episode_id,), conn=conn)
    values = []
    for index in range(math.ceil(duration / chunk_duration)):
        start = index * chunk_duration
        end = min(duration, (index + 1) * chunk_duration)
        old = next((chunk for chunk in previous if abs(chunk["start_time"] - start) < 0.1), None)
        values.append((old["id"] if old else str(uuid.uuid4()), episode_id, index, start, end,
                       old["title"] if old else "", old["notes"] if old else ""))
    conn.execute("DELETE FROM chunks WHERE episode_id=?", (episode_id,))
    conn.executemany("INSERT INTO chunks (id,episode_id,chunk_index,start_time,end_time,title,notes) VALUES (?,?,?,?,?,?,?)", values)


def insert_job(conn, episode_id, pipeline, payload, *, priority_class=None, retry_of=None, stage='queued'):
    """Queue a request inside the caller's transaction, reusing an identical active request."""
    kind = job_kinds.KINDS[pipeline]
    choice = {key: value for key, value in payload.items() if key not in ('depends_on', 'batch_id')}
    if choice.get('upload_hash'):
        choice.pop('upload_id', None)
    request_key = job_inputs.fingerprint({'payload': choice, 'retry_of': retry_of})
    existing = conn.execute("""SELECT id,status,payload_json FROM jobs WHERE episode_id=? AND pipeline=? AND request_key=?
        AND status IN ('queued','running')""", (episode_id, pipeline, request_key)).fetchone()
    if existing:
        if existing['status'] == 'queued' and payload.get('depends_on'):
            add_dependencies(conn, existing, payload['depends_on'])
        return existing['id']
    job_id = str(uuid.uuid4())
    conn.execute("""INSERT INTO jobs(id,episode_id,pipeline,payload_json,stage,priority_class,retry_of,request_key)
        VALUES (?,?,?,?,?,?,?,?)""", (job_id, episode_id, pipeline, json.dumps(payload, ensure_ascii=False), stage,
                                      priority_class or kind.priority(payload), retry_of, request_key))
    return job_id


def add_dependencies(conn, job, requested):
    """A reused queued request also waits for its new prerequisites, barring cycles."""
    payload = json.loads(job['payload_json'])
    previous = payload.get('depends_on') or []
    parents = list(previous)
    for parent in dict.fromkeys(requested):
        if parent in parents:
            continue
        cycle = conn.execute("""WITH RECURSIVE ancestors(id) AS (
            SELECT ? UNION SELECT p.value FROM ancestors a JOIN jobs j ON j.id=a.id,
                json_each(j.payload_json,'$.depends_on') p
        ) SELECT 1 FROM ancestors WHERE id=?""", (parent, job['id'])).fetchone()
        if not cycle:
            parents.append(parent)
    if parents != previous:
        conn.execute('UPDATE jobs SET payload_json=? WHERE id=?',
                     (json.dumps({**payload, 'depends_on': parents}, ensure_ascii=False), job['id']))


def pending_jobs(conn, episode_id, pipelines, cut_ids=None):
    """Find the predecessors that can produce inputs for this request."""
    marks = ','.join('?' for _ in pipelines)
    rows = conn.execute(f"""SELECT id,pipeline,payload_json FROM jobs WHERE episode_id=?
        AND pipeline IN ({marks}) AND status IN ('queued','running') ORDER BY rowid""", (episode_id, *pipelines))
    requested = set(cut_ids) if cut_ids is not None else None
    found = []
    for row in rows:
        targets = {ref['id'] for ref in job_kinds.kind_of(row['pipeline']).cuts(json.loads(row['payload_json']))}
        if requested is None or not targets or requested & targets:
            found.append(row['id'])
    return found


def enqueue(episode_id: str, pipeline: str, extra: dict | None = None):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job_id = enqueue_in(conn, episode_id, pipeline, extra)
    wake_event.set()
    return job_id


def enqueue_in(conn, episode_id, pipeline, extra=None):
    """Record settings and prerequisites in a transaction already held by the caller."""
    episode = conn.execute("SELECT settings_json FROM episodes WHERE id=?", (episode_id,)).fetchone()
    if not episode:
        raise ValueError("Episode not found")
    payload = dict(extra or {})
    if pipeline in ('prepare', 'hardcuts', 'cutframes'):
        payload['settings'] = json.loads(episode['settings_json'])
    cut_id = payload.get('cut_id')
    if cut_id and pipeline in ('hardcuts', 'cutframes'):
        cut = conn.execute('SELECT cut_index FROM cuts WHERE id=? AND episode_id=? AND active=1', (cut_id, episode_id)).fetchone()
        if not cut:
            raise ValueError('Active cut not found')
        if pipeline == 'cutframes' and payload.get('frame_step'):
            following = conn.execute('SELECT id FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?',
                                     (episode_id, cut['cut_index'] + 1)).fetchone()
            if not following:
                raise ValueError('There is no following cut at this boundary')
            payload['next_cut_id'] = following['id']
    deps = list(payload.get('depends_on', []))
    if pipeline in ('hardcuts', 'cutframes', 'transcribe', 'subtitles', 'keyframes') or (pipeline == 'voice_sample' and not payload.get('upload_id')):
        deps.extend(pending_jobs(conn, episode_id, ('prepare',)))
    if (pipeline == 'hardcuts' and cut_id) or pipeline == 'cutframes':
        deps.extend(row['id'] for row in conn.execute("""SELECT id FROM jobs WHERE episode_id=? AND pipeline='hardcuts'
            AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id') IS NULL ORDER BY rowid""", (episode_id,)))
    if pipeline == 'subtitles':
        deps.extend(pending_jobs(conn, episode_id, ('transcribe',)))
    if pipeline == 'keyframes':
        targets = None if payload.get('all_cuts') else [ref['id'] for ref in job_kinds.KINDS[pipeline].cuts(payload)]
        deps.extend(pending_jobs(conn, episode_id, ('hardcuts', 'subtitles'), targets))
    if deps:
        payload['depends_on'] = list(dict.fromkeys(deps))
    job_id = insert_job(conn, episode_id, pipeline, payload)
    sync_episode_status(conn, episode_id)
    return job_id


def cleanup_voice_upload(job):
    if job['pipeline'] != 'voice_sample':
        return
    try:
        upload_id = json.loads(job['payload_json']).get('upload_id')
        if upload_id:
            episode = db.one('SELECT project_id FROM episodes WHERE id=?', (job['episode_id'],))
            if episode:
                (media.episode_dir(episode['project_id'], job['episode_id']) / 'voice-uploads' / str(uuid.UUID(upload_id))).unlink(missing_ok=True)
        episode = db.one('SELECT project_id FROM episodes WHERE id=?', (job['episode_id'],))
        if episode and not db.one('SELECT id FROM character_voice_samples WHERE id=?', (job['id'],)):
            media.voice_reference_path(episode['project_id'], str(uuid.UUID(job['id']))).unlink(missing_ok=True)
    except (OSError, ValueError):
        log.exception('Could not clean up voice upload for job %s', job['id'])


# Local text jobs of one episode may run side by side (two llama-server slots); every other
# pipeline still holds its episode alone.
TEXT_PIPELINES = ('regen', 'consistency_summary', 'review_answers', 'sequence', 'performance_beats')
# The provider a text job was queued with: ``regen`` keeps it at the top of its payload, the others in ``config``.
JOB_TEXT_PROVIDER_SQL = "COALESCE(json_extract(jobs.payload_json,'$.text_provider'), json_extract(jobs.payload_json,'$.config.text_provider'), 'openrouter')"
# A text job run by a remote lane uses neither this host's GPU nor its media files, so it does not hold its
# episode against visual analysis, and a running visual analysis does not hold it against a remote lane.
# Remote lanes mark the jobs they hold with this lease-owner prefix; every other pipeline (transcription,
# subtitles, character detection, media preparation, hard cuts, ...) still holds the episode alone.
REMOTE_OWNER = 'remote:'
ALONGSIDE_REMOTE = ('analyze',)
DEPENDENCIES_DONE = """NOT EXISTS (SELECT 1 FROM json_each(jobs.payload_json,'$.depends_on') parent
    LEFT JOIN jobs dep ON dep.id=parent.value WHERE dep.id IS NULL OR dep.status!='done')"""


def claim_job(pipelines=("prepare", "hardcuts", "cutframes", "brief_export"), remote_only=(), local_provider_only=False, remote_lane=False):
    """Lease the next runnable job of ``pipelines``; returns ``(job_id, owner)`` or ``None``.

    Routing to the remote text server (``remote_text``) only concerns jobs queued with the local provider:
    ``remote_only`` names pipelines whose local-provider jobs this caller leaves to the remote lanes,
    and ``local_provider_only`` (a remote lane) skips every job that was not queued for the local model.
    ``remote_lane`` marks the lease as a remote lane's and lets the job start while ``ALONGSIDE_REMOTE``
    jobs of its episode run.
    """
    if not pipelines:
        return None
    now = time.time()
    routing = ''
    if remote_only:
        routing += f" AND NOT (jobs.pipeline IN ({','.join('?' for _ in remote_only)}) AND {JOB_TEXT_PROVIDER_SQL}='local')"
    if local_provider_only:
        routing += f" AND {JOB_TEXT_PROVIDER_SQL}='local'"
    placeholders = ",".join("?" for _ in pipelines)
    text = ",".join(f"'{name}'" for name in TEXT_PIPELINES)
    alongside = ",".join(f"'{name}'" for name in ALONGSIDE_REMOTE)
    # Jobs of ``alongside`` pipelines are never held by a job a remote lane runs; a remote lane is never held by them.
    remote_free = f"AND NOT (jobs.pipeline IN ({alongside}) AND other.lease_owner LIKE '{REMOTE_OWNER}%')" + (f" AND other.pipeline NOT IN ({alongside})" if remote_lane else '')
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        exhausted = conn.execute(f"SELECT * FROM jobs WHERE pipeline IN ({placeholders}) AND status='running' AND (lease_until IS NULL OR lease_until<=?) AND attempt_count>=?", (*pipelines, now, MAX_ATTEMPTS)).fetchall()
        for job in exhausted:
            conn.execute("UPDATE jobs SET status='failed',stage='Failed',error='Interrupted too many times; run the pipeline again',lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
            if job['pipeline'] not in ('brief_export', 'composition_compact'):
                sync_episode_status(conn, job["episode_id"], "Interrupted too many times; run the pipeline again")
        # Sweep abandoned dependencies across lanes. cancel_jobs walks the rest of each chain.
        orphans = conn.execute("""SELECT jobs.* FROM jobs
            WHERE (status='queued' OR (status='running' AND COALESCE(lease_until,0)<=?)) AND EXISTS (
            SELECT 1 FROM json_each(jobs.payload_json,'$.depends_on') parent
            LEFT JOIN jobs dep ON dep.id=parent.value
            WHERE dep.id IS NULL OR dep.status IN ('failed','cancelled'))""", (now,)).fetchall()
        cancelled = cancel_jobs(conn, orphans, WAITED_FOR_FAILED)
        candidates = conn.execute(f"""SELECT * FROM jobs WHERE pipeline IN ({placeholders}){routing} AND
            status IN ('queued','running') AND COALESCE(lease_until,0)<=? AND not_before<=?
            AND {DEPENDENCIES_DONE}
            AND (jobs.pipeline='brief_export' OR NOT EXISTS (SELECT 1 FROM jobs other WHERE other.episode_id=jobs.episode_id AND other.status IN ('queued','running','cancelled') AND other.id!=jobs.id AND other.pipeline!='brief_export'
                -- Only a job with a live lease holds the episode. Interrupted jobs (lease
                -- expired) must not block each other: a bulk batch leaves many at once.
                AND other.lease_until>?
                {remote_free}
                AND NOT (jobs.pipeline IN ({text}) AND other.pipeline IN ({text}))
                AND NOT ((jobs.pipeline='cutframes' AND other.pipeline IN ('analyze','regen')) OR (other.pipeline='cutframes' AND jobs.pipeline IN ('analyze','regen'))))
            )
            ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,
                     CASE pipeline WHEN 'composition_compact' THEN 1 ELSE 0 END, created_at, rowid""", (*pipelines, *remote_only, now, now, now))
        claimed = None
        for job in candidates:
            owner = (REMOTE_OWNER if remote_lane else '') + str(uuid.uuid4())
            if not lease_candidate(conn, job, owner, now):
                continue
            if job['pipeline'] not in ('brief_export', 'composition_compact'):
                conn.execute("UPDATE episodes SET status='processing',stage=?,error=NULL WHERE id=?",
                             ("Resuming" if job["status"] == "running" else "Starting", job["episode_id"]))
            claimed = job["id"], owner
            break
    for old_job in exhausted:
        cleanup_voice_upload(old_job)
    after_cancel(cancelled)
    return claimed


def lease_candidate(conn, job, owner, now):
    """Start a dependency-ready candidate under the claim transaction."""
    if job_inputs.change_wait(conn, dict(job), now):
        return False
    stage = 'Resuming' if job['status'] == 'running' else 'Starting'
    conn.execute("""UPDATE jobs SET status='running',stage=?,error=NULL,lease_owner=?,lease_until=?,
        attempt_count=attempt_count+1,not_before=0,restart_input_json=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?""",
        (stage, owner, now + LEASE_SECONDS, job['id']))
    return True


def requeue_job(job_id, owner, stage='Waiting for a text worker'):
    """Give a running job back to the queue without counting the attempt (its worker went away, not the job)."""
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute("SELECT episode_id FROM jobs WHERE id=? AND status='running' AND lease_owner=? AND stage!='Cancelled'", (job_id, owner)).fetchone()
        if not job:
            return False
        conn.execute("UPDATE jobs SET status='queued',stage=?,lease_owner=NULL,lease_until=NULL,attempt_count=MAX(0,attempt_count-1),updated_at=CURRENT_TIMESTAMP WHERE id=?", (stage, job_id))
        sync_episode_status(conn, job['episode_id'])
        return True


def defer_for_dependencies(conn, job, owner, dependencies, stage):
    """Return to waiting when a handler discovers an input producer it needs."""
    require_lease(conn, job['id'], owner)
    payload = json.loads(job['payload_json'])
    payload['depends_on'] = list(dict.fromkeys([*(payload.get('depends_on') or []), *dependencies]))
    conn.execute("""UPDATE jobs SET status='queued',stage=?,payload_json=?,attempt_count=MAX(0,attempt_count-1),
        updated_at=CURRENT_TIMESTAMP WHERE id=?""", (stage, json.dumps(payload, ensure_ascii=False), job['id']))
    sync_episode_status(conn, job['episode_id'])


WAITED_FOR_FAILED = 'Cancelled because a job it waited for did not finish'
# Pipelines that never move the episode's own status (exports and the composition cache).
SIDE_PIPELINES = ('brief_export', 'composition_compact')


def cancel_jobs(conn, jobs, reason=None):
    """Cancel queued or running jobs inside the caller's transaction and return the rows that changed.

    Cancelled is its own status, never a failure. The stage stays ``Cancelled`` because text handlers
    poll it (``inference_worker.check_summary_lease``). A running job keeps its lease owner: its worker
    holds the GPU until the job's next checkpoint, ``heartbeat`` keeps that lease alive meanwhile (the
    queue shows the job as stopping) and releases it when the worker lets go. Queued jobs that wait for a
    cancelled job (``payload.depends_on``) can never run and are cancelled with it.
    Call ``after_cancel`` with the returned rows once the transaction has committed.
    """
    pending = [(job, reason) for job in jobs]
    changed, seen, episodes = [], set(), set()
    now = time.time()
    while pending:
        job, why = pending.pop(0)
        if job['id'] in seen:
            continue
        seen.add(job['id'])
        current = conn.execute('SELECT * FROM jobs WHERE id=?', (job['id'],)).fetchone()
        if not current or current['status'] not in ('queued', 'running'):
            continue
        running = current['status'] == 'running'
        message = why or ('Cancelled during processing' if running else 'Cancelled before processing')
        lease = '' if current['lease_owner'] and (current['lease_until'] or 0) > now else ',lease_owner=NULL,lease_until=NULL'
        conn.execute(f"UPDATE jobs SET status='cancelled',stage='Cancelled',error=?{lease},updated_at=CURRENT_TIMESTAMP WHERE id=?", (message, current['id']))
        conn.execute("UPDATE analysis_job_stages SET status='cancelled',finished_at=?,error=? WHERE job_id=? AND status='running'", (now, message, current['id']))
        changed.append(current)
        if current['pipeline'] not in SIDE_PIPELINES:
            episodes.add(current['episode_id'])
        for dependent in dependents(conn, current['id']):
            pending.append((dependent, WAITED_FOR_FAILED))
    for episode_id in episodes:
        sync_episode_status(conn, episode_id)
    return changed


def dependents(conn, job_id):
    return conn.execute("""SELECT jobs.* FROM jobs WHERE status='queued' AND EXISTS (
        SELECT 1 FROM json_each(jobs.payload_json,'$.depends_on') parent WHERE parent.value=?)""", (job_id,)).fetchall()


def fail_job(job_id, owner, episode_id, exc, *, keep_stage=False):
    """Fail a leased job and cancel its dependents in the same transaction."""
    message = str(exc)[-1800:]
    stage = "CASE WHEN stage IN ('Queued','Starting') THEN 'Failed' ELSE 'Failed · ' || stage END" if keep_stage else "'Failed'"
    with db.connect() as conn:
        job = conn.execute(f"""UPDATE jobs SET status='failed',stage={stage},error=?,lease_owner=NULL,lease_until=NULL,
            updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='running' AND lease_owner=? RETURNING pipeline""",
            (message, job_id, owner)).fetchone()
        cancelled = cancel_jobs(conn, dependents(conn, job_id), WAITED_FOR_FAILED) if job else []
        if job and job['pipeline'] not in SIDE_PIPELINES:
            sync_episode_status(conn, episode_id, message)
    after_cancel(cancelled)
    return bool(job)


def after_cancel(jobs):
    """Clean-up for cancelled jobs once their transaction has committed."""
    for job in jobs:
        cleanup_voice_upload(job)
    sweep_character_frames({job['episode_id'] for job in jobs if job['pipeline'] == 'character_detect'})


def sweep_character_frames(episode_ids):
    """Remove the temporary first frames of bulk character detection once an episode has no pending detection."""
    for episode_id in episode_ids:
        row = db.one('SELECT project_id FROM episodes WHERE id=?', (episode_id,))
        if row and not db.one("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='character_detect' AND (status IN ('queued','running') OR (status='cancelled' AND lease_owner IS NOT NULL))", (episode_id,)):
            from . import anime_bulk
            anime_bulk.clear_episode_frames(row['project_id'], episode_id)


def release_lease(job_id: str, owner: str):
    """The worker has let go of a job: a cancelled job stops showing as stopping."""
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET lease_owner=NULL,lease_until=NULL WHERE id=? AND lease_owner=? AND status!='running'", (job_id, owner))


def sync_episode_status(conn, episode_id: str, error: str | None = None):
    """Reflect remaining work after a job finishes without hiding its history."""
    pending = conn.execute("SELECT status,stage FROM jobs WHERE episode_id=? AND pipeline NOT IN ('brief_export','composition_compact') AND status IN ('queued','running') ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,created_at,rowid LIMIT 1", (episode_id,)).fetchone()
    if pending:
        conn.execute("UPDATE episodes SET status=?,stage=?,error=NULL WHERE id=?", ('processing' if pending['status'] == 'running' else 'queued', pending['stage'], episode_id))
    elif error:
        conn.execute("UPDATE episodes SET status='failed',stage='Failed',error=? WHERE id=?", (error, episode_id))
    else:
        conn.execute("UPDATE episodes SET status='ready',stage='Ready',error=NULL WHERE id=?", (episode_id,))


class LeaseLost(RuntimeError):
    """The job no longer belongs to this worker; nothing else may be written for it."""


def require_lease(conn, job_id: str, owner: str):
    active = conn.execute("SELECT 1 FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner)).fetchone()
    if not active:
        raise LeaseLost("Job lease was lost")


def stage(job_id: str, owner: str, episode_id: str, label: str):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_lease(conn, job_id, owner)
        conn.execute("UPDATE jobs SET stage=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (label, job_id))
        conn.execute("UPDATE episodes SET status='processing', stage=?, error=NULL WHERE id=? AND (SELECT pipeline FROM jobs WHERE id=?) NOT IN ('brief_export','composition_compact')", (label, episode_id, job_id))


@contextmanager
def analysis_stage(job_id: str, owner: str, cut_id: str, key: str, label: str):
    """Record an attempt under the job lease; interrupted attempts remain visible."""
    stage_id, now = str(uuid.uuid4()), time.time()
    with db.connect() as conn:
        require_lease(conn, job_id, owner)
        conn.execute("UPDATE analysis_job_stages SET status='interrupted',finished_at=last_seen_at WHERE job_id=? AND cut_id=? AND status='running'", (job_id, cut_id))
        conn.execute("INSERT INTO analysis_job_stages(id,job_id,cut_id,stage_key,label,status,started_at,last_seen_at) VALUES (?,?,?,?,?,'running',?,?)", (stage_id, job_id, cut_id, key, label, now, now))
    try:
        yield
    except Exception as exc:
        with db.connect() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner)).fetchone():
                conn.execute("UPDATE analysis_job_stages SET status='failed',finished_at=?,error=? WHERE id=?", (time.time(), str(exc)[-1800:], stage_id))
        raise
    else:
        with db.connect() as conn:
            require_lease(conn, job_id, owner)
            conn.execute("UPDATE analysis_job_stages SET status='done',finished_at=? WHERE id=?", (time.time(), stage_id))


def heartbeat(job_id: str, owner: str, done: threading.Event):
    """Renew the job's lease until ``done`` is set.

    A cancelled job keeps its lease while its worker is still inside it (the GPU is busy until the next
    checkpoint), so the queue can show it as stopping; the lease is released when the worker lets go.
    """
    try:
        while not done.wait(HEARTBEAT_SECONDS):
            try:
                with db.connect() as conn:
                    changed = conn.execute("UPDATE jobs SET lease_until=? WHERE id=? AND status IN ('queued','running','cancelled') AND lease_owner=?",
                                           (time.time() + LEASE_SECONDS, job_id, owner)).rowcount
                    if changed:
                        conn.execute("UPDATE analysis_job_stages SET last_seen_at=? WHERE job_id=? AND status='running'", (time.time(), job_id))
                if not changed:
                    return
            except Exception:
                log.exception("Could not renew media job lease: %s", job_id)
    finally:
        try:
            release_lease(job_id, owner)
        except Exception:
            log.exception('Could not release job lease: %s', job_id)


def work_loop(pipelines=("prepare", "hardcuts", "cutframes", "keyframes")):
    while not stop_event.is_set():
        try:
            claimed = claim_job(pipelines)
            if claimed:
                job_id, owner = claimed
                done = threading.Event()
                beat = threading.Thread(target=heartbeat, args=(job_id, owner, done), name="media-heartbeat", daemon=True)
                beat.start()
                try:
                    process(job_id, owner)
                finally:
                    done.set()
                    beat.join(timeout=1)
                continue
        except Exception:
            log.exception("Media worker error")
        wake_event.wait(timeout=1)
        wake_event.clear()



def discard_obsolete_previews(path):
    """Proxy-derived previews are stale after the proxy is rebuilt."""
    for folder, pattern in ((path / 'frames', r'frame_\d{6}\.jpg'),
                            (path / 'thumbnails', r'(?:cut_\d{6}|[0-9a-f-]{36}_(?:first|last))\.jpg')):
        if folder.is_dir():
            for image in folder.iterdir():
                if image.is_file() and re.fullmatch(pattern, image.name):
                    image.unlink(missing_ok=True)

def process(job_id: str, owner: str):
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    episode = db.one("SELECT * FROM episodes WHERE id=?", (job["episode_id"],))
    if not episode:
        return
    episode_id = episode["id"]
    path = media.episode_dir(episode["project_id"], episode_id)
    original = path / f"original{episode['original_ext']}"
    payload = json.loads(job["payload_json"])
    settings = payload.get("settings") or json.loads(episode["settings_json"])
    try:
        if job['pipeline'] == 'cutframes':
            from . import cut_previews
            cut_previews.process(job, owner, payload, path)
            return
        if job['pipeline'] == 'brief_export':
            from .brief import process_export
            process_export(job, owner, episode, payload)
            with db.connect() as conn:
                require_lease(conn, job_id, owner)
                conn.execute("UPDATE jobs SET status='done',stage='Complete',error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
            return
        if job['pipeline'] == 'keyframes':
            from . import keyframes
            label = keyframes.process(job, owner, episode, payload, original)
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                require_lease(conn, job_id, owner)
                conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (label, job_id))
                sync_episode_status(conn, episode_id)
            return
        from . import media_jobs
        if job['pipeline'] == 'prepare':
            media_jobs.prepare(job, owner, path, settings)
        elif job['pipeline'] == 'hardcuts':
            media_jobs.hardcuts(job, owner, episode, payload, path, settings)
    except LeaseLost:
        return
    except Exception as exc:
        fail_job(job_id, owner, episode_id, exc)
        log.exception("Failed processing episode %s", episode_id)
