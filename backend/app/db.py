import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "app.sqlite3"


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=30000")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, synopsis TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            title TEXT NOT NULL, original_filename TEXT NOT NULL, original_ext TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'upload',
            error TEXT, duration REAL, width INTEGER, height INTEGER, has_audio INTEGER,
            settings_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS episodes_project_idx ON episodes(project_id);
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            pipeline TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
            stage TEXT NOT NULL DEFAULT 'queued', error TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}', lease_owner TEXT, lease_until REAL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS jobs_episode_idx ON jobs(episode_id);
        CREATE TABLE IF NOT EXISTS cuts (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            cut_index INTEGER NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            title TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '', results_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(episode_id, cut_index)
        );
        CREATE INDEX IF NOT EXISTS cuts_episode_idx ON cuts(episode_id);
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,
            title TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
            UNIQUE(episode_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS chunks_episode_idx ON chunks(episode_id);
        CREATE TABLE IF NOT EXISTS analysis_results (
            id TEXT PRIMARY KEY, cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft', origin TEXT NOT NULL DEFAULT 'manual',
            model_id TEXT, job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            prompt TEXT, settings_json TEXT NOT NULL DEFAULT '{}',
            context_json TEXT NOT NULL DEFAULT '{}',
            parent_id TEXT, superseded_by TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS analysis_results_cut_idx ON analysis_results(cut_id);
        CREATE TABLE IF NOT EXISTS transcript_segments (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            start_time REAL NOT NULL, end_time REAL NOT NULL, text TEXT NOT NULL,
            speaker_label TEXT NOT NULL DEFAULT '',
            character_context_id TEXT REFERENCES context_items(id) ON DELETE SET NULL,
            status TEXT NOT NULL DEFAULT 'draft', origin TEXT NOT NULL DEFAULT 'manual',
            model_id TEXT, job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            parent_id TEXT, superseded_by TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS transcript_episode_time_idx ON transcript_segments(episode_id,start_time,end_time);
        CREATE TABLE IF NOT EXISTS context_items (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            episode_id TEXT REFERENCES episodes(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS context_project_idx ON context_items(project_id);
        """)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(transcript_segments)")}
        if "parent_id" not in columns:
            db.execute("ALTER TABLE transcript_segments ADD COLUMN parent_id TEXT")
        if "superseded_by" not in columns:
            db.execute("ALTER TABLE transcript_segments ADD COLUMN superseded_by TEXT")
        job_columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        for column, definition in (
            ("payload_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("lease_owner", "TEXT"),
            ("lease_until", "REAL"),
            ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in job_columns:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
        cut_columns = {row["name"] for row in db.execute("PRAGMA table_info(cuts)")}
        if "active" not in cut_columns:
            db.execute("ALTER TABLE cuts ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        active = db.execute("SELECT id,episode_id FROM jobs WHERE status IN ('queued','running') ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at, id").fetchall()
        kept = set()
        for job in active:
            if job["episode_id"] in kept:
                db.execute("UPDATE jobs SET status='failed',stage='Failed',error='Duplicate active job superseded during migration',updated_at=CURRENT_TIMESTAMP WHERE id=?", (job["id"],))
            else:
                kept.add(job["episode_id"])
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS jobs_active_episode_idx ON jobs(episode_id) WHERE status IN ('queued','running')")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status,lease_until,created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS transcript_active_idx ON transcript_segments(episode_id,start_time) WHERE superseded_by IS NULL")


def one(sql, args=()):
    with connect() as db:
        row = db.execute(sql, args).fetchone()
        return dict(row) if row else None


def all_rows(sql, args=()):
    with connect() as db:
        return [dict(row) for row in db.execute(sql, args).fetchall()]
