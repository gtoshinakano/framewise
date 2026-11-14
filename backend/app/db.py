import os
import json
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


def remove_legacy_content_status(value):
    """Remove obsolete review metadata from stored result/transcript snapshots."""
    if isinstance(value, list):
        return [remove_legacy_content_status(item) for item in value]
    if isinstance(value, dict):
        content_record = ("kind" in value and "content" in value) or ("text" in value and "start_time" in value and "end_time" in value)
        return {key: remove_legacy_content_status(item) for key, item in value.items()
                if not (content_record and key == "status" and item in ("draft", "approved", "rejected", "active", "deleted"))}
    return value


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
        CREATE TABLE IF NOT EXISTS local_text_model_state (
            id INTEGER PRIMARY KEY CHECK(id=1), desired_loaded INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'unloaded', detail TEXT NOT NULL DEFAULT '',
            heartbeat REAL NOT NULL DEFAULT 0,
            variant TEXT NOT NULL DEFAULT '', variants_json TEXT NOT NULL DEFAULT '{}'
        );
        INSERT OR IGNORE INTO local_text_model_state(id) VALUES (1);
        CREATE TABLE IF NOT EXISTS text_generation_settings (
            id INTEGER PRIMARY KEY CHECK(id=1), text_provider TEXT NOT NULL DEFAULT 'openrouter',
            local_thinking INTEGER NOT NULL DEFAULT 0, openrouter_model TEXT NOT NULL DEFAULT 'openrouter/free',
            local_model TEXT NOT NULL DEFAULT 'standard'
        );
        INSERT OR IGNORE INTO text_generation_settings(id) VALUES (1);
        -- A llama.cpp server on another PC that takes local-provider text jobs (remote_text.py).
        CREATE TABLE IF NOT EXISTS remote_text_server (
            id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL DEFAULT 0,
            base_url TEXT NOT NULL DEFAULT '', token TEXT NOT NULL DEFAULT '',
            parallel INTEGER NOT NULL DEFAULT 1, routes_json TEXT NOT NULL DEFAULT '{}',
            assistant_fallback TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL DEFAULT 0,
            prefer TEXT NOT NULL DEFAULT 'local', chat_reserve INTEGER NOT NULL DEFAULT 1
        );
        INSERT OR IGNORE INTO remote_text_server(id) VALUES (1);
        CREATE TABLE IF NOT EXISTS remote_text_state (
            id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL DEFAULT 'disabled', detail TEXT NOT NULL DEFAULT '',
            models_json TEXT NOT NULL DEFAULT '[]', active_jobs INTEGER NOT NULL DEFAULT 0, heartbeat REAL NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO remote_text_state(id) VALUES (1);
        CREATE TABLE IF NOT EXISTS cuts (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            cut_index INTEGER NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,
            first_frame_time REAL, last_frame_time REAL,
            active INTEGER NOT NULL DEFAULT 1,
            title TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '', results_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(episode_id, cut_index)
        );
        CREATE INDEX IF NOT EXISTS cuts_episode_idx ON cuts(episode_id);
        CREATE TABLE IF NOT EXISTS cut_brief_instructions (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE,
            content TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS cut_review_drafts (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE,
            job_id TEXT NOT NULL, content TEXT NOT NULL, model_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS cut_analysis_configs (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE,
            config_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
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
            origin TEXT NOT NULL DEFAULT 'manual',
            model_id TEXT, job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            prompt TEXT, settings_json TEXT NOT NULL DEFAULT '{}',
            context_json TEXT NOT NULL DEFAULT '{}',
            parent_id TEXT, superseded_by TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS analysis_results_cut_idx ON analysis_results(cut_id);
        CREATE TABLE IF NOT EXISTS analysis_job_cuts (
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            evidence_json TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(job_id,cut_id)
        );
        CREATE TABLE IF NOT EXISTS analysis_job_stages (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            stage_key TEXT NOT NULL, label TEXT NOT NULL,
            status TEXT NOT NULL, started_at REAL NOT NULL, last_seen_at REAL NOT NULL,
            finished_at REAL, error TEXT
        );
        CREATE INDEX IF NOT EXISTS analysis_job_stages_cut_idx ON analysis_job_stages(cut_id, started_at);
        CREATE TABLE IF NOT EXISTS rejected_video_responses (
            id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            window_index INTEGER NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,
            error TEXT NOT NULL, response TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS analysis_video_windows (
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            window_index INTEGER NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,
            model_revision TEXT NOT NULL, content_json TEXT NOT NULL,
            PRIMARY KEY(job_id,cut_id,window_index)
        );
        CREATE TABLE IF NOT EXISTS transcript_segments (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            start_time REAL NOT NULL, end_time REAL NOT NULL, text TEXT NOT NULL,
            speaker_label TEXT NOT NULL DEFAULT '',
            character_context_id TEXT REFERENCES context_items(id) ON DELETE SET NULL,
            deleted INTEGER NOT NULL DEFAULT 0, origin TEXT NOT NULL DEFAULT 'manual',
            model_id TEXT, job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            parent_id TEXT, superseded_by TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS transcript_episode_time_idx ON transcript_segments(episode_id,start_time,end_time);
        CREATE TABLE IF NOT EXISTS subtitle_runs (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            language TEXT NOT NULL, language_code TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            band_json TEXT, scan_hash TEXT,
            model_id TEXT, model_revision TEXT, prompt TEXT,
            settings_json TEXT NOT NULL DEFAULT '{}', diagnostics_json TEXT NOT NULL DEFAULT '{}',
            error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS subtitle_runs_episode_idx ON subtitle_runs(episode_id,status);
        CREATE TABLE IF NOT EXISTS subtitle_readings (
            run_id TEXT NOT NULL REFERENCES subtitle_runs(id) ON DELETE CASCADE,
            event_index INTEGER NOT NULL,
            start_time REAL NOT NULL, end_time REAL NOT NULL, rep_time REAL NOT NULL,
            outcome TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
            legible TEXT NOT NULL DEFAULT 'none', detail TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(run_id,event_index)
        );
        CREATE TABLE IF NOT EXISTS subtitle_lines (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES subtitle_runs(id) ON DELETE CASCADE,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            line_index INTEGER NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,
            text TEXT NOT NULL, confidence TEXT NOT NULL, readings INTEGER NOT NULL DEFAULT 1,
            UNIQUE(run_id,line_index)
        );
        CREATE INDEX IF NOT EXISTS subtitle_lines_time_idx ON subtitle_lines(episode_id,start_time,end_time);
        CREATE TABLE IF NOT EXISTS context_items (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            episode_id TEXT REFERENCES episodes(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS context_project_idx ON context_items(project_id);
        CREATE TABLE IF NOT EXISTS character_detection_runs (
            id TEXT PRIMARY KEY, cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            job_id TEXT UNIQUE REFERENCES jobs(id) ON DELETE SET NULL,
            cut_start REAL NOT NULL, cut_end REAL NOT NULL,
            frames_json TEXT NOT NULL, model_id TEXT NOT NULL, model_revision TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS character_detection_cut_idx ON character_detection_runs(cut_id,created_at);
        CREATE TABLE IF NOT EXISTS character_detections (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES character_detection_runs(id) ON DELETE CASCADE,
            position TEXT NOT NULL, box_json TEXT NOT NULL,
            character_context_id TEXT REFERENCES context_items(id) ON DELETE SET NULL,
            label_source TEXT, match_reason TEXT, match_model_id TEXT, match_model_revision TEXT
        );
        CREATE TABLE IF NOT EXISTS character_image_references (
            id TEXT PRIMARY KEY, character_context_id TEXT NOT NULL REFERENCES context_items(id) ON DELETE CASCADE,
            detection_id TEXT NOT NULL REFERENCES character_detections(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(character_context_id,detection_id)
        );
        CREATE INDEX IF NOT EXISTS character_reference_character_idx ON character_image_references(character_context_id);
        CREATE TABLE IF NOT EXISTS character_reference_embeddings (
            reference_id TEXT NOT NULL REFERENCES character_image_references(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL, model_revision TEXT NOT NULL, preprocessing TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK(dimensions > 0), embedding BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(reference_id,model_id,model_revision,preprocessing)
        );
        CREATE TABLE IF NOT EXISTS character_detection_embeddings (
            detection_id TEXT NOT NULL REFERENCES character_detections(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL, model_revision TEXT NOT NULL, preprocessing TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK(dimensions > 0), embedding BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(detection_id,model_id,model_revision,preprocessing)
        );
        CREATE TABLE IF NOT EXISTS character_voice_samples (
            id TEXT PRIMARY KEY,
            character_context_id TEXT NOT NULL REFERENCES context_items(id) ON DELETE CASCADE,
            episode_id TEXT REFERENCES episodes(id) ON DELETE SET NULL,
            start_time REAL NOT NULL, end_time REAL NOT NULL,
            embedding_json TEXT NOT NULL, model_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS character_voice_samples_character_idx ON character_voice_samples(character_context_id);
        -- Sequences: a user-chosen range of cuts directed as one short video. Their results are
        -- revision chains like analysis_results (parent_id / superseded_by) and have no approval status.
        CREATE TABLE IF NOT EXISTS sequences (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            title TEXT NOT NULL DEFAULT '', instruction TEXT NOT NULL DEFAULT '',
            first_cut_id TEXT NOT NULL, last_cut_id TEXT NOT NULL,
            start_time REAL NOT NULL, end_time REAL NOT NULL,
            cuts_json TEXT NOT NULL DEFAULT '[]', settings_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS sequences_episode_idx ON sequences(episode_id,start_time);
        CREATE TABLE IF NOT EXISTS sequence_results (
            id TEXT PRIMARY KEY, sequence_id TEXT NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
            type TEXT NOT NULL, shot_key TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL, rendered TEXT,
            origin TEXT NOT NULL DEFAULT 'model', model_id TEXT,
            job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            prompt TEXT, settings_json TEXT NOT NULL DEFAULT '{}', context_json TEXT NOT NULL DEFAULT '{}',
            parent_id TEXT, superseded_by TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS sequence_results_idx ON sequence_results(sequence_id,type,shot_key);
        """)
        from .assistant import store as assistant_store
        assistant_store.init_schema(db)
        voice_columns = {row["name"] for row in db.execute("PRAGMA table_info(character_voice_samples)")}
        if "ranges_json" not in voice_columns:
            db.execute("ALTER TABLE character_voice_samples ADD COLUMN ranges_json TEXT")
        if "source_filename" not in voice_columns:
            db.execute("ALTER TABLE character_voice_samples ADD COLUMN source_filename TEXT")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(transcript_segments)")}
        if "parent_id" not in columns:
            db.execute("ALTER TABLE transcript_segments ADD COLUMN parent_id TEXT")
        if "superseded_by" not in columns:
            db.execute("ALTER TABLE transcript_segments ADD COLUMN superseded_by TEXT")
        # Preserve every result and revision while removing obsolete approval
        # metadata. The migration and snapshot cleanup commit as one unit.
        db.commit()
        db.execute("BEGIN IMMEDIATE")
        result_columns = {row["name"] for row in db.execute("PRAGMA table_info(analysis_results)")}
        columns = {row["name"] for row in db.execute("PRAGMA table_info(transcript_segments)")}
        migrating = "status" in result_columns or "status" in columns
        remote_columns = {row["name"] for row in db.execute("PRAGMA table_info(remote_text_server)")}
        if "token" not in remote_columns:  # basic-auth username/password gave way to a Bearer token
            db.execute("ALTER TABLE remote_text_server ADD COLUMN token TEXT NOT NULL DEFAULT ''")
        for name, ddl in (("prefer", "TEXT NOT NULL DEFAULT 'local'"), ("chat_reserve", "INTEGER NOT NULL DEFAULT 1")):
            if name not in remote_columns:
                db.execute(f"ALTER TABLE remote_text_server ADD COLUMN {name} {ddl}")
        for old in ("username", "password"):
            if old in remote_columns:
                db.execute(f"ALTER TABLE remote_text_server DROP COLUMN {old}")
        if "deleted" not in columns:
            db.execute("ALTER TABLE transcript_segments ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0")
        if "status" in columns:
            db.execute("UPDATE transcript_segments SET deleted=1 WHERE status='deleted'")
            db.execute("ALTER TABLE transcript_segments DROP COLUMN status")
        if "status" in result_columns:
            db.execute("ALTER TABLE analysis_results DROP COLUMN status")
        if migrating:
            for table, column in (("analysis_results", "context_json"), ("jobs", "payload_json")):
                for row in db.execute(f"SELECT id,{column} FROM {table}").fetchall():
                    original = json.loads(row[column])
                    cleaned = remove_legacy_content_status(original)
                    if cleaned != original:
                        db.execute(f"UPDATE {table} SET {column}=? WHERE id=?", (json.dumps(cleaned, ensure_ascii=False), row["id"]))
        for table, column, definition in (
            ('character_detection_runs', 'model_revision', 'TEXT'),
            ('character_detection_runs', 'phase', "TEXT NOT NULL DEFAULT 'complete'"),
            ('character_detection_runs', 'timings_json', 'TEXT'),
            ('cuts', 'no_characters', 'INTEGER NOT NULL DEFAULT 0'),
            ('text_generation_settings', 'local_model', "TEXT NOT NULL DEFAULT 'standard'"),
            ('local_text_model_state', 'variant', "TEXT NOT NULL DEFAULT ''"),
            ('local_text_model_state', 'variants_json', "TEXT NOT NULL DEFAULT '{}'"),
            ('episodes', 'subtitle_language', 'TEXT'),
            ('cuts', 'created_at', 'TEXT'),
            ('character_detections', 'match_model_id', 'TEXT'),
            ('character_detections', 'match_model_revision', 'TEXT'),
            # Bulk recognition: ranked suggestions for unconfirmed boxes, the
            # user's override for using a confirmed box as a matching example
            # (NULL automatic, 1 always, 0 never), and the worker's verdict.
            ('character_detections', 'candidates_json', 'TEXT'),
            ('character_detections', 'recognition_use', 'INTEGER'),
            ('character_detections', 'gallery_status', 'TEXT'),
            ('character_detections', 'gallery_note', 'TEXT'),
        ):
            columns = {row['name'] for row in db.execute(f'PRAGMA table_info({table})')}
            if column not in columns:
                db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
        job_columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        stage_columns = {row['name'] for row in db.execute('PRAGMA table_info(analysis_job_stages)')}
        if 'last_seen_at' not in stage_columns:
            db.execute('ALTER TABLE analysis_job_stages ADD COLUMN last_seen_at REAL NOT NULL DEFAULT 0')
            db.execute('UPDATE analysis_job_stages SET last_seen_at=started_at')
        for column, definition in (
            ("payload_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("lease_owner", "TEXT"),
            ("lease_until", "REAL"),
            ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
            ("started_at", "TEXT"),
            ("finished_at", "TEXT"),
            ("retry_of", "TEXT REFERENCES jobs(id) ON DELETE SET NULL"),
        ):
            if column not in job_columns:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
        cut_columns = {row["name"] for row in db.execute("PRAGMA table_info(cuts)")}
        if "active" not in cut_columns:
            db.execute("ALTER TABLE cuts ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        if "analysis_enabled" not in cut_columns:
            db.execute("ALTER TABLE cuts ADD COLUMN analysis_enabled INTEGER NOT NULL DEFAULT 1")
        for column in ("first_frame_time", "last_frame_time"):
            if column not in cut_columns:
                db.execute(f"ALTER TABLE cuts ADD COLUMN {column} REAL")
        db.execute("DROP INDEX IF EXISTS jobs_active_episode_idx")
        db.execute("DROP INDEX IF EXISTS jobs_running_episode_idx")
        db.executescript("""
        CREATE TRIGGER IF NOT EXISTS jobs_started AFTER UPDATE OF status ON jobs
        WHEN NEW.status='running' AND OLD.status='queued'
        BEGIN UPDATE jobs SET started_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;
        CREATE TRIGGER IF NOT EXISTS jobs_finished AFTER UPDATE OF status ON jobs
        WHEN NEW.status IN ('done','failed') AND OLD.status IN ('queued','running')
        BEGIN UPDATE jobs SET finished_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;
        """)
        db.execute("CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status,lease_until,created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS transcript_active_idx ON transcript_segments(episode_id,start_time) WHERE superseded_by IS NULL")
        # Incremental composition reads need a watermark for edits as well as
        # new revisions. SQLite's millisecond clock is used consistently here.
        trigger_drops = '\n'.join('DROP TRIGGER "' + trigger['name'].replace('"', '""') + '";'
                                  for trigger in db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'composition_%'").fetchall())
        db.executescript('BEGIN IMMEDIATE;\n' + trigger_drops + """
        CREATE TABLE IF NOT EXISTS composition_changes (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS composition_compactions (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE,
            input_hash TEXT NOT NULL,
            values_json TEXT NOT NULL,
            model_id TEXT NOT NULL,
            job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            prompt TEXT NOT NULL DEFAULT '',
            settings_json TEXT NOT NULL DEFAULT '{}',
            input_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT OR IGNORE INTO composition_changes(cut_id,updated_at)
            SELECT id,strftime('%Y-%m-%dT%H:%M:%fZ','now') FROM cuts;
        CREATE TRIGGER IF NOT EXISTS composition_cut_insert AFTER INSERT ON cuts BEGIN
            INSERT INTO composition_changes VALUES(NEW.id,strftime('%Y-%m-%dT%H:%M:%fZ','now'));
        END;
        CREATE TRIGGER IF NOT EXISTS composition_cut_update AFTER UPDATE ON cuts BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_result_insert AFTER INSERT ON analysis_results BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_result_update AFTER UPDATE ON analysis_results BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_result_delete AFTER DELETE ON analysis_results BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=OLD.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_transcript_insert AFTER INSERT ON transcript_segments BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=NEW.episode_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_transcript_update AFTER UPDATE ON transcript_segments BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id IN (NEW.episode_id,OLD.episode_id));
        END;
        CREATE TRIGGER IF NOT EXISTS composition_transcript_delete AFTER DELETE ON transcript_segments BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=OLD.episode_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_subtitle_run_update AFTER UPDATE OF status ON subtitle_runs BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=NEW.episode_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_subtitle_run_delete AFTER DELETE ON subtitle_runs BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=OLD.episode_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_detection_run_insert AFTER INSERT ON character_detection_runs BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_detection_run_update AFTER UPDATE ON character_detection_runs BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (NEW.cut_id,OLD.cut_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_detection_run_delete AFTER DELETE ON character_detection_runs BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=OLD.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_detection_insert AFTER INSERT ON character_detections BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT cut_id FROM character_detection_runs WHERE id=NEW.run_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_detection_update AFTER UPDATE ON character_detections BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT cut_id FROM character_detection_runs WHERE id=NEW.run_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_detection_delete AFTER DELETE ON character_detections BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT cut_id FROM character_detection_runs WHERE id=OLD.run_id);
        END;
        CREATE TRIGGER IF NOT EXISTS composition_character_update AFTER UPDATE OF title ON context_items BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT c.id FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE e.project_id=NEW.project_id);
        END;
        -- Sequence jobs write sequence results only, so they leave the cuts' composition watermarks alone.
        DROP TRIGGER IF EXISTS composition_job_insert;
        DROP TRIGGER IF EXISTS composition_job_status;
        CREATE TRIGGER composition_job_insert AFTER INSERT ON jobs WHEN NEW.pipeline NOT IN ('brief_export','composition_compact','sequence') BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=NEW.episode_id);
        END;
        CREATE TRIGGER composition_job_status AFTER UPDATE OF status ON jobs WHEN NEW.pipeline NOT IN ('brief_export','composition_compact','sequence') AND NEW.status!=OLD.status BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=NEW.episode_id);
        END;
        CREATE TRIGGER composition_job_delete AFTER DELETE ON jobs WHEN OLD.pipeline NOT IN ('brief_export','composition_compact','sequence') BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id IN (SELECT id FROM cuts WHERE episode_id=OLD.episode_id);
        END;
        CREATE TRIGGER composition_compact_job_insert AFTER INSERT ON jobs WHEN NEW.pipeline='composition_compact' AND json_valid(NEW.payload_json) BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=json_extract(NEW.payload_json,'$.cut_id');
        END;
        CREATE TRIGGER composition_compact_job_status AFTER UPDATE OF status ON jobs WHEN NEW.pipeline='composition_compact' AND NEW.status!=OLD.status AND json_valid(NEW.payload_json) BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=json_extract(NEW.payload_json,'$.cut_id');
        END;
        CREATE TRIGGER composition_compact_job_delete AFTER DELETE ON jobs WHEN OLD.pipeline='composition_compact' AND json_valid(OLD.payload_json) BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=json_extract(OLD.payload_json,'$.cut_id');
        END;
        CREATE TRIGGER IF NOT EXISTS composition_checkpoint_insert AFTER INSERT ON analysis_job_cuts BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_checkpoint_update AFTER UPDATE OF completed ON analysis_job_cuts BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_checkpoint_delete AFTER DELETE ON analysis_job_cuts BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=OLD.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_compaction_insert AFTER INSERT ON composition_compactions BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_compaction_update AFTER UPDATE ON composition_compactions BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=NEW.cut_id;
        END;
        CREATE TRIGGER IF NOT EXISTS composition_compaction_delete AFTER DELETE ON composition_compactions BEGIN
            UPDATE composition_changes SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE cut_id=OLD.cut_id;
        END;
        COMMIT;
        """)
        db.execute('BEGIN IMMEDIATE')
        compact_columns = {row['name'] for row in db.execute('PRAGMA table_info(composition_compactions)')}
        for column, definition in (('prompt', "TEXT NOT NULL DEFAULT ''"), ('settings_json', "TEXT NOT NULL DEFAULT '{}'"), ('input_json', "TEXT NOT NULL DEFAULT '{}'")):
            if column not in compact_columns:
                db.execute(f'ALTER TABLE composition_compactions ADD COLUMN {column} {definition}')
        db.commit()


def one(sql, args=()):
    with connect() as db:
        row = db.execute(sql, args).fetchone()
        return dict(row) if row else None


def all_rows(sql, args=()):
    with connect() as db:
        return [dict(row) for row in db.execute(sql, args).fetchall()]
