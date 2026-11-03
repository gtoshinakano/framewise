import json
import logging
import math
import re
import sqlite3
import threading
import tempfile
import time
import uuid

from pathlib import Path

from . import db, media

log = logging.getLogger(__name__)
worker_thread: threading.Thread | None = None
wake_event = threading.Event()
stop_event = threading.Event()
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 10
MAX_ATTEMPTS = 3


def start_worker():
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        return
    for episode in db.all_rows("SELECT id,duration,settings_json FROM episodes WHERE duration IS NOT NULL AND NOT EXISTS (SELECT 1 FROM chunks WHERE chunks.episode_id=episodes.id)"):
        settings = json.loads(episode["settings_json"])
        sync_chunks(episode["id"], episode["duration"], settings.get("chunk_duration", 60))
    stop_event.clear()
    worker_thread = threading.Thread(target=work_loop, name="media-worker", daemon=True)
    worker_thread.start()


def stop_worker():
    stop_event.set()
    wake_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=2)


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
            episode = conn.execute("SELECT settings_json FROM episodes WHERE id=?", (episode_id,)).fetchone()
            if not episode:
                raise ValueError("Episode not found")
            existing = conn.execute("SELECT id FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (episode_id,)).fetchone()
            if existing:
                raise ValueError("An analysis job is already running for this episode")
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


def claim_job(pipelines=("prepare", "hardcuts", "cutframes")):
    now = time.time()
    placeholders = ",".join("?" for _ in pipelines)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        exhausted = conn.execute(f"SELECT * FROM jobs WHERE pipeline IN ({placeholders}) AND status='running' AND (lease_until IS NULL OR lease_until<=?) AND attempt_count>=?", (*pipelines, now, MAX_ATTEMPTS)).fetchall()
        for job in exhausted:
            conn.execute("UPDATE jobs SET status='failed',stage='Failed',error='Interrupted too many times; run the pipeline again',lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
            sync_episode_status(conn, job["episode_id"], "Interrupted too many times; run the pipeline again")
        job = conn.execute(f"""SELECT * FROM jobs WHERE pipeline IN ({placeholders}) AND
            (status='queued' OR (status='running' AND (lease_until IS NULL OR lease_until<=?)))
            AND NOT EXISTS (SELECT 1 FROM jobs other WHERE other.episode_id=jobs.episode_id AND other.status='running' AND other.id!=jobs.id
                AND NOT ((jobs.pipeline='cutframes' AND other.pipeline IN ('analyze','regen')) OR (other.pipeline='cutframes' AND jobs.pipeline IN ('analyze','regen'))))
            ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at, rowid LIMIT 1""", (*pipelines, now)).fetchone()
        claimed = None
        if job:
            owner = str(uuid.uuid4())
            conn.execute("UPDATE jobs SET status='running',stage=?,error=NULL,lease_owner=?,lease_until=?,attempt_count=attempt_count+1,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                         ("Resuming" if job["status"] == "running" else "Starting", owner, now + LEASE_SECONDS, job["id"]))
            conn.execute("UPDATE episodes SET status='processing',stage=?,error=NULL WHERE id=?",
                         ("Resuming" if job["status"] == "running" else "Starting", job["episode_id"]))
            claimed = job["id"], owner
    for old_job in exhausted:
        cleanup_voice_upload(old_job)
    return claimed


def sync_episode_status(conn, episode_id: str, error: str | None = None):
    """Reflect remaining work after a job finishes without hiding its history."""
    pending = conn.execute("SELECT status,stage FROM jobs WHERE episode_id=? AND status IN ('queued','running') ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,created_at,rowid LIMIT 1", (episode_id,)).fetchone()
    if pending:
        conn.execute("UPDATE episodes SET status=?,stage=?,error=NULL WHERE id=?", ('processing' if pending['status'] == 'running' else 'queued', pending['stage'], episode_id))
    elif error:
        conn.execute("UPDATE episodes SET status='failed',stage='Failed',error=? WHERE id=?", (error, episode_id))
    else:
        conn.execute("UPDATE episodes SET status='ready',stage='Ready',error=NULL WHERE id=?", (episode_id,))


def require_lease(conn, job_id: str, owner: str):
    active = conn.execute("SELECT 1 FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner)).fetchone()
    if not active:
        raise RuntimeError("Job lease was lost")


def stage(job_id: str, owner: str, episode_id: str, label: str):
    with db.connect() as conn:
        require_lease(conn, job_id, owner)
        conn.execute("UPDATE jobs SET stage=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (label, job_id))
        conn.execute("UPDATE episodes SET status='processing', stage=?, error=NULL WHERE id=?", (label, episode_id))


def heartbeat(job_id: str, owner: str, done: threading.Event):
    while not done.wait(HEARTBEAT_SECONDS):
        try:
            with db.connect() as conn:
                changed = conn.execute("UPDATE jobs SET lease_until=? WHERE id=? AND status='running' AND lease_owner=?",
                                       (time.time() + LEASE_SECONDS, job_id, owner)).rowcount
            if not changed:
                return
        except Exception:
            log.exception("Could not renew media job lease: %s", job_id)


def work_loop():
    while not stop_event.is_set():
        try:
            claimed = claim_job()
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
        if job["pipeline"] != "cutframes":
            stage(job_id, owner, episode_id, "Probing video")
            info = media.probe(original)
            with db.connect() as conn:
                require_lease(conn, job_id, owner)
                conn.execute("UPDATE episodes SET duration=?, width=?, height=?, has_audio=? WHERE id=?",
                             (info["duration"], info["width"], info["height"], int(info["has_audio"]), episode_id))
            stage(job_id, owner, episode_id, "Organizing chunks")
            sync_chunks(episode_id, info["duration"], settings.get("chunk_duration", 60))
        if job["pipeline"] == "cutframes" and not proxy.exists():
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
        if job["pipeline"] == "hardcuts":
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
                next_archive_index = min((cut["cut_index"] for cut in old), default=0) - 1
                for cut in old:
                    if cut["active"]:
                        conn.execute("UPDATE cuts SET active=0,cut_index=? WHERE id=?", (next_archive_index, cut["id"]))
                        next_archive_index -= 1
                for cut_id, index, start, end in replacement:
                    if cut_id:
                        conn.execute("UPDATE cuts SET active=1,cut_index=?,start_time=?,end_time=?,first_frame_time=NULL,last_frame_time=NULL WHERE id=?", (index, start, end, cut_id))
                    else:
                        conn.execute("INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time) VALUES (?,?,?,?,?)",
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
        if job["pipeline"] == "cutframes":
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
            conn.execute("UPDATE jobs SET status='done', stage='Complete', error=NULL, lease_owner=NULL, lease_until=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
            sync_episode_status(conn, episode_id)
    except Exception as exc:
        message = str(exc)[-1800:]
        with db.connect() as conn:
            changed = conn.execute("UPDATE jobs SET status='failed', stage='Failed', error=?, lease_owner=NULL, lease_until=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='running' AND lease_owner=?",
                                   (message, job_id, owner)).rowcount
            if changed:
                sync_episode_status(conn, episode_id, message)
        log.exception("Failed processing episode %s", episode_id)
