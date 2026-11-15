import json
import logging
import math
import re
import sqlite3
import threading
import tempfile
import time
import uuid
from contextlib import contextmanager

from pathlib import Path

from . import cut_history, db, media

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


def sync_chunks(episode_id: str, duration: float, chunk_duration: int):
    if not math.isfinite(duration) or duration <= 0:
        return
    previous = db.all_rows("SELECT * FROM chunks WHERE episode_id=? ORDER BY chunk_index", (episode_id,))
    values = []
    for index in range(math.ceil(duration / chunk_duration)):
        start = index * chunk_duration
        end = min(duration, (index + 1) * chunk_duration)
        old = next((chunk for chunk in previous if abs(chunk["start_time"] - start) < 0.1), None)
        values.append((old["id"] if old else str(uuid.uuid4()), episode_id, index, start, end,
                       old["title"] if old else "", old["notes"] if old else ""))
    with db.connect() as conn:
        conn.execute("DELETE FROM chunks WHERE episode_id=?", (episode_id,))
        conn.executemany("INSERT INTO chunks (id,episode_id,chunk_index,start_time,end_time,title,notes) VALUES (?,?,?,?,?,?,?)", values)


def enqueue(episode_id: str, pipeline: str, extra: dict | None = None):
    try:
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            episode = conn.execute("SELECT settings_json FROM episodes WHERE id=?", (episode_id,)).fetchone()
            if not episode:
                raise ValueError("Episode not found")
            existing = conn.execute("SELECT id FROM jobs WHERE episode_id=? AND pipeline NOT IN ('brief_export','composition_compact') AND status IN ('queued','running')", (episode_id,)).fetchone()
            if existing:
                raise ValueError("An analysis job is already running for this episode")
            if pipeline == 'hardcuts' and extra and extra.get('cut_id'):
                cut = conn.execute('SELECT start_time,end_time FROM cuts WHERE id=? AND episode_id=? AND active=1', (extra['cut_id'], episode_id)).fetchone()
                if not cut:
                    raise ValueError('Active cut not found')
                extra = {**extra, 'cut_start': cut['start_time'], 'cut_end': cut['end_time']}
            if pipeline == 'cutframes' and extra and extra.get('frame_step'):
                cut = conn.execute('SELECT * FROM cuts WHERE id=? AND episode_id=? AND active=1', (extra['cut_id'], episode_id)).fetchone()
                if not cut:
                    raise ValueError('Active cut not found')
                next_cut = conn.execute('SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?', (episode_id, cut['cut_index'] + 1)).fetchone()
                if not next_cut:
                    raise ValueError('There is no following cut at this boundary')
                if abs(cut['end_time'] - next_cut['start_time']) > .000001:
                    raise ValueError('The cuts do not share a boundary')
                extra = {**extra, 'cut_start': cut['start_time'], 'cut_end': cut['end_time'], 'cut_last': cut['last_frame_time'],
                         'next_cut_id': next_cut['id'], 'next_end': next_cut['end_time']}
            job_id = str(uuid.uuid4())
            payload = json.dumps({"settings": json.loads(episode["settings_json"]), **(extra or {})})
            conn.execute("INSERT INTO jobs (id, episode_id, pipeline, payload_json) VALUES (?,?,?,?)", (job_id, episode_id, pipeline, payload))
            conn.execute("UPDATE episodes SET status='queued', stage=? , error=NULL WHERE id=?", (pipeline, episode_id))
    except sqlite3.IntegrityError as exc:
        raise ValueError("An analysis job is already running for this episode") from exc
    wake_event.set()
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
TEXT_PIPELINES = ('regen', 'consistency_summary', 'review_answers', 'sequence')
# The provider a text job was queued with: ``regen`` keeps it at the top of its payload, the others in ``config``.
JOB_TEXT_PROVIDER_SQL = "COALESCE(json_extract(jobs.payload_json,'$.text_provider'), json_extract(jobs.payload_json,'$.config.text_provider'), 'openrouter')"


def claim_job(pipelines=("prepare", "hardcuts", "cutframes", "brief_export"), remote_only=(), local_provider_only=False):
    """Lease the next runnable job of ``pipelines``; returns ``(job_id, owner)`` or ``None``.

    Routing to the remote text server (``remote_text``) only concerns jobs queued with the local provider:
    ``remote_only`` names pipelines whose local-provider jobs this caller leaves to the remote lanes,
    and ``local_provider_only`` (a remote lane) skips every job that was not queued for the local model.
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
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        exhausted = conn.execute(f"SELECT * FROM jobs WHERE pipeline IN ({placeholders}) AND status='running' AND (lease_until IS NULL OR lease_until<=?) AND attempt_count>=?", (*pipelines, now, MAX_ATTEMPTS)).fetchall()
        for job in exhausted:
            conn.execute("UPDATE jobs SET status='failed',stage='Failed',error='Interrupted too many times; run the pipeline again',lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
            if job['pipeline'] not in ('brief_export', 'composition_compact'):
                sync_episode_status(conn, job["episode_id"], "Interrupted too many times; run the pipeline again")
        # A chained text job (payload.wait_for) is held until the job it names is done. If that job
        # failed, was cancelled or is gone, the waiting job can never run, so fail it now.
        while True:  # cancelling one job can orphan the next link of its chain
            orphans = conn.execute(f"""SELECT id,episode_id FROM jobs WHERE status='queued' AND pipeline IN ('regen','review_answers') AND pipeline IN ({placeholders})
                AND json_extract(payload_json,'$.wait_for') IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM jobs dep WHERE dep.id=json_extract(jobs.payload_json,'$.wait_for') AND dep.status IN ('queued','running','done'))""", pipelines).fetchall()
            if not orphans:
                break
            for orphan in orphans:
                conn.execute("UPDATE jobs SET status='failed',stage='Cancelled',error='Cancelled because a job it waited for did not finish',lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (orphan["id"],))
                sync_episode_status(conn, orphan["episode_id"])
        job = conn.execute(f"""SELECT * FROM jobs WHERE pipeline IN ({placeholders}){routing} AND
            (status='queued' OR (status='running' AND (lease_until IS NULL OR lease_until<=?)))
            AND (jobs.pipeline NOT IN ('regen','review_answers') OR json_extract(jobs.payload_json,'$.wait_for') IS NULL
                 OR EXISTS (SELECT 1 FROM jobs dep WHERE dep.id=json_extract(jobs.payload_json,'$.wait_for') AND dep.status='done'))
            AND (jobs.pipeline='brief_export' OR NOT EXISTS (SELECT 1 FROM jobs other WHERE other.episode_id=jobs.episode_id AND other.status='running' AND other.id!=jobs.id AND other.pipeline!='brief_export'
                -- Only a job with a live lease holds the episode. Interrupted jobs (lease
                -- expired) must not block each other: a bulk batch leaves many at once.
                AND other.lease_until>?
                AND NOT (jobs.pipeline IN ({text}) AND other.pipeline IN ({text}))
                AND NOT ((jobs.pipeline='cutframes' AND other.pipeline IN ('analyze','regen')) OR (other.pipeline='cutframes' AND jobs.pipeline IN ('analyze','regen'))))
            )
            ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,
                     CASE pipeline WHEN 'composition_compact' THEN 1 ELSE 0 END, created_at, rowid LIMIT 1""", (*pipelines, *remote_only, now, now)).fetchone()
        claimed = None
        if job:
            owner = str(uuid.uuid4())
            conn.execute("UPDATE jobs SET status='running',stage=?,error=NULL,lease_owner=?,lease_until=?,attempt_count=attempt_count+1,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                         ("Resuming" if job["status"] == "running" else "Starting", owner, now + LEASE_SECONDS, job["id"]))
            if job['pipeline'] not in ('brief_export', 'composition_compact'):
                conn.execute("UPDATE episodes SET status='processing',stage=?,error=NULL WHERE id=?",
                             ("Resuming" if job["status"] == "running" else "Starting", job["episode_id"]))
            claimed = job["id"], owner
    for old_job in exhausted:
        cleanup_voice_upload(old_job)
    return claimed


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
    while not done.wait(HEARTBEAT_SECONDS):
        try:
            with db.connect() as conn:
                changed = conn.execute("UPDATE jobs SET lease_until=? WHERE id=? AND status='running' AND lease_owner=?",
                                       (time.time() + LEASE_SECONDS, job_id, owner)).rowcount
                if changed:
                    conn.execute("UPDATE analysis_job_stages SET last_seen_at=? WHERE job_id=? AND status='running'", (time.time(), job_id))
            if not changed:
                return
        except Exception:
            log.exception("Could not renew media job lease: %s", job_id)


def work_loop(pipelines=("prepare", "hardcuts", "cutframes")):
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
    proxy = path / "proxy.mp4"
    payload = json.loads(job["payload_json"])
    settings = payload.get("settings") or json.loads(episode["settings_json"])
    try:
        if job['pipeline'] == 'brief_export':
            from .brief import process_export
            process_export(job, owner, episode, payload)
            with db.connect() as conn:
                require_lease(conn, job_id, owner)
                conn.execute("UPDATE jobs SET status='done',stage='Complete',error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
            return
        targeted = job["pipeline"] == "hardcuts" and bool(payload.get('cut_id'))
        if job["pipeline"] != "cutframes" and not targeted:
            stage(job_id, owner, episode_id, "Probing video")
            info = media.probe(original)
            with db.connect() as conn:
                require_lease(conn, job_id, owner)
                conn.execute("UPDATE episodes SET duration=?, width=?, height=?, has_audio=? WHERE id=?",
                             (info["duration"], info["width"], info["height"], int(info["has_audio"]), episode_id))
            stage(job_id, owner, episode_id, "Organizing chunks")
            sync_chunks(episode_id, info["duration"], settings.get("chunk_duration", 60))
        if (job["pipeline"] == "cutframes" or targeted) and not proxy.exists():
            raise RuntimeError("Preview video is missing; run Prepare before updating cut frames")
        if job["pipeline"] == "prepare" or not proxy.exists():
            stage(job_id, owner, episode_id, "Creating preview video")
            media.make_proxy(original, proxy, settings["proxy_width"], settings.get("proxy_crf", 28), settings.get("proxy_preset", "veryfast"))
            with db.connect() as conn:
                require_lease(conn, job_id, owner)
                discard_obsolete_previews(path)
            stage(job_id, owner, episode_id, "Indexing frames")
            media.write_frame_manifest(path / "frames", settings["frame_interval"], info["duration"])
            if info["has_audio"]:
                stage(job_id, owner, episode_id, "Extracting audio")
                media.extract_audio(original, path / "audio.wav")
                stage(job_id, owner, episode_id, "Building waveform")
                media.waveform(path / "audio.wav", path / "waveform.json", info["duration"])
            else:
                (path / "waveform.json").write_text(json.dumps({"peaks": [], "duration": info["duration"]}))
        if targeted:
            affected = payload.get('detected_cut_ids')
            if affected is None:
                stage(job_id, owner, episode_id, 'Detecting hard cuts in selected cut')
                segments = media.detect_cuts_in_range(proxy, payload['cut_start'], payload['cut_end'], payload['scene_threshold'])
                with db.connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    require_lease(conn, job_id, owner)
                    cut = conn.execute('SELECT * FROM cuts WHERE id=? AND episode_id=? AND active=1', (payload['cut_id'], episode_id)).fetchone()
                    if not cut or cut['start_time'] != payload['cut_start'] or cut['end_time'] != payload['cut_end']:
                        raise RuntimeError('The selected cut changed before detection completed; run it again')
                    affected = []
                    before = cut_history.snapshot(conn, episode_id)
                    created = set()
                    if len(segments) > 1:
                        affected.append(cut['id'])
                        conn.execute('UPDATE cuts SET end_time=?,first_frame_time=NULL,last_frame_time=NULL WHERE id=?', (segments[0][1], cut['id']))
                        later = conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND cut_index>? ORDER BY cut_index DESC', (episode_id, cut['cut_index'])).fetchall()
                        for item in later:
                            conn.execute('UPDATE cuts SET cut_index=? WHERE id=?', (item['cut_index'] + len(segments) - 1, item['id']))
                        archived = conn.execute('SELECT * FROM cuts WHERE episode_id=? AND active=0 AND start_time>? AND start_time<?', (episode_id, cut['start_time'], cut['end_time'])).fetchall()
                        used = set()
                        for offset, (start, end) in enumerate(segments[1:], 1):
                            match = min((item for item in archived if item['id'] not in used and abs(item['start_time'] - start) < .25), key=lambda item: abs(item['start_time'] - start), default=None)
                            cut_id = match['id'] if match else str(uuid.uuid4())
                            if match:
                                used.add(cut_id)
                                conn.execute('UPDATE cuts SET active=1,cut_index=?,start_time=?,end_time=?,first_frame_time=NULL,last_frame_time=NULL WHERE id=?', (cut['cut_index'] + offset, start, end, cut_id))
                            else:
                                conn.execute('INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time,created_at) VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)', (cut_id, episode_id, cut['cut_index'] + offset, start, end))
                                created.add(cut_id)
                            affected.append(cut_id)
                        cut_history.record(conn, episode_id, 'detect', f"Detected {len(segments)} cuts inside cut {cut['cut_index'] + 1}", cut['id'], before, created)
                    payload['detected_cut_ids'] = affected
                    conn.execute('UPDATE jobs SET payload_json=? WHERE id=?', (json.dumps(payload), job_id))
            stage(job_id, owner, episode_id, 'Updating selected cut previews')
            with tempfile.TemporaryDirectory(prefix='framewise-boundaries-') as scratch:
                for cut_id in affected:
                    cut = db.one('SELECT id,start_time,end_time,first_frame_time,last_frame_time FROM cuts WHERE id=? AND active=1', (cut_id,))
                    if not cut or (cut['first_frame_time'] is not None and cut['last_frame_time'] is not None):
                        continue
                    start, end = cut['start_time'], cut['end_time']
                    first, last = media.cut_frame_times(start, end)
                    first = media.extract_cut_frame(proxy, Path(scratch) / 'first.jpg', first, start, end)
                    last = media.extract_cut_frame(proxy, Path(scratch) / 'last.jpg', last, start, end, last=True, probe_padding=1)
                    with db.connect() as conn:
                        require_lease(conn, job_id, owner)
                        conn.execute('UPDATE cuts SET first_frame_time=?,last_frame_time=? WHERE id=?', (first, last, cut_id))
                        for position in ('first', 'last'):
                            (path / 'thumbnails' / f'{cut_id}_{position}.jpg').unlink(missing_ok=True)
        elif job["pipeline"] == "hardcuts":
            stage(job_id, owner, episode_id, "Detecting hard cuts")
            segments = media.detect_cuts(proxy, info["duration"], settings["scene_threshold"])
            old = db.all_rows("SELECT * FROM cuts WHERE episode_id=? ORDER BY active DESC,start_time", (episode_id,))
            matched_ids = set()
            replacement = []
            for index, (start, end) in enumerate(segments):
                candidates = (cut for cut in old if cut["id"] not in matched_ids and abs(cut["start_time"] - start) < 0.25)
                match = min(candidates, key=lambda cut: (not cut["active"], abs(cut["start_time"] - start)), default=None)
                if match:
                    matched_ids.add(match["id"])
                replacement.append((match["id"] if match else None, index, start, end))
            with db.connect() as conn:
                require_lease(conn, job_id, owner)
                # Every cut is replaced, so earlier cut edits can no longer be reversed.
                cut_history.clear(conn, episode_id)
                next_archive_index = min((cut["cut_index"] for cut in old), default=0) - 1
                for cut in old:
                    if cut["active"]:
                        conn.execute("UPDATE cuts SET active=0,cut_index=? WHERE id=?", (next_archive_index, cut["id"]))
                        next_archive_index -= 1
                for cut_id, index, start, end in replacement:
                    if cut_id:
                        conn.execute("UPDATE cuts SET active=1,cut_index=?,start_time=?,end_time=?,first_frame_time=NULL,last_frame_time=NULL WHERE id=?", (index, start, end, cut_id))
                    else:
                        conn.execute("INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time,created_at) VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                                     (str(uuid.uuid4()), episode_id, index, start, end))
            cut_ids = {cut["cut_index"]: cut["id"] for cut in db.all_rows("SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1", (episode_id,))}
            with tempfile.TemporaryDirectory(prefix='framewise-boundaries-') as scratch:
                for index, (start, end) in enumerate(segments):
                    first, last = media.cut_frame_times(start, end)
                    cut_id = cut_ids[index]
                    first = media.extract_cut_frame(proxy, Path(scratch) / 'first.jpg', first, start, end)
                    last = media.extract_cut_frame(proxy, Path(scratch) / 'last.jpg', last, start, end, last=True, probe_padding=1)
                    with db.connect() as conn:
                        require_lease(conn, job_id, owner)
                        conn.execute("UPDATE cuts SET first_frame_time=?,last_frame_time=? WHERE id=?", (first, last, cut_id))
                        for position in ('first', 'last'):
                            (path / 'thumbnails' / f"{cut_id}_{position}.jpg").unlink(missing_ok=True)
        if job["pipeline"] == "cutframes" and payload.get('frame_step'):
            if not payload.get('frame_step_done'):
                stage(job_id, owner, episode_id, 'Moving cut boundary by one frame')
                # The displayed last frame can deliberately be earlier than the
                # stored boundary. Step from what the user sees when adjusting
                # an end, and close that gap when moving the adjacent start.
                end_anchor = payload.get('frame_anchor', 'end') == 'end' and payload['cut_last'] is not None
                anchor = payload['cut_last'] if end_anchor else payload['cut_end']
                boundary = media.step_frame_boundary(proxy, anchor, payload['frame_step'], after_stepped_frame=end_anchor)
                if not payload['cut_start'] < boundary < payload['next_end']:
                    raise RuntimeError('Moving the boundary would leave a cut without a positive duration')
                with tempfile.TemporaryDirectory(prefix='framewise-boundaries-') as scratch:
                    last = media.extract_cut_frame(proxy, Path(scratch) / 'last.jpg', boundary,
                        payload['cut_start'], boundary, last=True, probe_padding=1)
                    # FFmpeg's seek is rounded to microseconds; seek just before
                    # the boundary so rounding up cannot skip its first frame.
                    first = media.extract_cut_frame(proxy, Path(scratch) / 'first.jpg', boundary,
                        max(0, boundary - .000002), payload['next_end'], seek_tolerance=.000002)
                with db.connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    require_lease(conn, job_id, owner)
                    cut = conn.execute('SELECT * FROM cuts WHERE id=? AND episode_id=? AND active=1', (payload['cut_id'], episode_id)).fetchone()
                    following = conn.execute('SELECT * FROM cuts WHERE id=? AND episode_id=? AND active=1', (payload['next_cut_id'], episode_id)).fetchone()
                    if (not cut or not following or following['cut_index'] != cut['cut_index'] + 1
                            or cut['start_time'] != payload['cut_start'] or cut['end_time'] != payload['cut_end'] or cut['last_frame_time'] != payload['cut_last']
                            or abs(following['start_time'] - payload['cut_end']) > .000001 or following['end_time'] != payload['next_end']):
                        raise RuntimeError('The cuts changed while the boundary was queued; try again')
                    if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND id!=? AND status IN ('queued','running') AND pipeline NOT IN ('brief_export','composition_compact')", (episode_id, job_id)).fetchone():
                        raise RuntimeError('Wait for the current episode job to finish')
                    before = cut_history.snapshot(conn, episode_id)
                    conn.execute('UPDATE cuts SET end_time=?,last_frame_time=? WHERE id=?', (boundary, last, cut['id']))
                    conn.execute('UPDATE cuts SET start_time=?,first_frame_time=? WHERE id=?', (boundary, first, following['id']))
                    cut_history.record(conn, episode_id, 'boundary', f"Moved the end of cut {cut['cut_index'] + 1} one frame {'earlier' if payload['frame_step'] < 0 else 'later'}", cut['id'], before)
                    payload['frame_step_done'] = True
                    conn.execute('UPDATE jobs SET payload_json=? WHERE id=?', (json.dumps(payload), job_id))
                (path / 'thumbnails' / f"{cut['id']}_last.jpg").unlink(missing_ok=True)
                (path / 'thumbnails' / f"{following['id']}_first.jpg").unlink(missing_ok=True)
        elif job["pipeline"] == "cutframes":
            stage(job_id, owner, episode_id, "Updating cut frames")
            with tempfile.TemporaryDirectory(prefix='framewise-boundaries-') as scratch:
                for cut in db.all_rows("SELECT id,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 AND (first_frame_time IS NULL OR last_frame_time IS NULL) ORDER BY cut_index", (episode_id,)):
                    if "cut_ids" in payload and cut["id"] not in payload["cut_ids"]:
                        continue
                    first, last = media.cut_frame_times(cut["start_time"], cut["end_time"])
                    first = media.extract_cut_frame(proxy, Path(scratch) / 'first.jpg', first, cut["start_time"], cut["end_time"])
                    last = media.extract_cut_frame(proxy, Path(scratch) / 'last.jpg', last, cut["start_time"], cut["end_time"], last=True, probe_padding=1)
                    with db.connect() as conn:
                        require_lease(conn, job_id, owner)
                        conn.execute("UPDATE cuts SET first_frame_time=?,last_frame_time=? WHERE id=?", (first, last, cut["id"]))
                        for position in ('first', 'last'):
                            (path / 'thumbnails' / f"{cut['id']}_{position}.jpg").unlink(missing_ok=True)
        with db.connect() as conn:
            require_lease(conn, job_id, owner)
            final_stage = ('Complete · no new cuts found' if targeted and not payload.get('detected_cut_ids') else 'Complete')
            conn.execute("UPDATE jobs SET status='done', stage=?, error=NULL, lease_owner=NULL, lease_until=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?", (final_stage, job_id))
            sync_episode_status(conn, episode_id)
    except Exception as exc:
        message = str(exc)[-1800:]
        with db.connect() as conn:
            changed = conn.execute("UPDATE jobs SET status='failed', stage='Failed', error=?, lease_owner=NULL, lease_until=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='running' AND lease_owner=?",
                                   (message, job_id, owner)).rowcount
            if changed and job['pipeline'] != 'brief_export':
                sync_episode_status(conn, episode_id, message)
        log.exception("Failed processing episode %s", episode_id)
