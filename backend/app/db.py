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


def migrate_queue_dependencies(conn):
    flag = 'queue-dependencies-v1'
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    if conn.execute('SELECT 1 FROM app_flags WHERE name=?', (flag,)).fetchone():
        return
    rows = conn.execute("""SELECT id,payload_json FROM jobs
        WHERE json_valid(payload_json) AND (json_type(payload_json,'$.wait_for') IS NOT NULL
           OR json_type(payload_json,'$.wait_for_keyframes') IS NOT NULL
           OR json_type(payload_json,'$.depends_on_consistency') IS NOT NULL)""").fetchall()
    for row in rows:
        payload = json.loads(row['payload_json'])
        parents = [*(payload.get('depends_on') or []), payload.pop('wait_for', None),
                   payload.pop('depends_on_consistency', None), *(payload.pop('wait_for_keyframes', None) or [])]
        payload['depends_on'] = list(dict.fromkeys(parent for parent in parents if parent))
        conn.execute('UPDATE jobs SET payload_json=? WHERE id=?', (json.dumps(payload, ensure_ascii=False), row['id']))
    conn.execute('INSERT INTO app_flags(name) VALUES (?)', (flag,))


# 2026-11-24: only a change of a cut's boundaries makes its first-frame description stale, and audio-side work (subtitles)
# never makes a picture result stale. Marks the older rules left behind are cleared once.
PICTURE_STALE_REASONS = {
    'initial_frame': ('Episode subtitles changed', 'First-frame detections changed', 'First-frame boxes or labels were edited',
                      'The cut’s character setting changed'),
    'keyframe_analysis': ('Episode subtitles changed',),
}


def clear_picture_stale_marks(conn):
    flag = 'picture-stale-rules-v1'
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    if conn.execute('SELECT 1 FROM app_flags WHERE name=?', (flag,)).fetchone():
        return
    for kind, reasons in PICTURE_STALE_REASONS.items():
        conn.execute(f"""UPDATE analysis_results SET stale_reason=NULL WHERE superseded_by IS NULL
            AND json_extract(settings_json,'$.structured_type')=? AND stale_reason IN ({','.join('?' for _ in reasons)})""", (kind, *reasons))
    conn.execute('INSERT INTO app_flags(name) VALUES (?)', (flag,))


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
            local_model TEXT NOT NULL DEFAULT 'mtp',
            openrouter_audio_model TEXT NOT NULL DEFAULT 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'
        );
        INSERT OR IGNORE INTO text_generation_settings(id) VALUES (1);
        -- The model the user chose per visual role (visual_models.py); a missing row means the default.
        CREATE TABLE IF NOT EXISTS visual_model_settings (role TEXT PRIMARY KEY, model TEXT NOT NULL);
        -- Installed/downloading status of every visual model, published by the inference worker.
        CREATE TABLE IF NOT EXISTS visual_model_state (
            id INTEGER PRIMARY KEY CHECK(id=1), models_json TEXT NOT NULL DEFAULT '{}', updated_at REAL NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO visual_model_state(id) VALUES (1);
        -- One-time data changes that must not repeat (a flag row says the change was made).
        CREATE TABLE IF NOT EXISTS app_flags (name TEXT PRIMARY KEY, set_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
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
        CREATE TABLE IF NOT EXISTS keyframe_results (
            cache_key TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE, result_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
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
            evidence_json TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0, source_cut_json TEXT,
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
        CREATE TABLE IF NOT EXISTS project_name_glossary (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            spelling TEXT NOT NULL,
            reading TEXT NOT NULL,
            character_context_id TEXT REFERENCES context_items(id) ON DELETE CASCADE,
            part TEXT CHECK(part IN ('family','given')),
            CHECK((character_context_id IS NULL) = (part IS NULL))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS project_name_glossary_cast_idx
            ON project_name_glossary(project_id,character_context_id,part,spelling,reading) WHERE character_context_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS project_name_glossary_other_idx
            ON project_name_glossary(project_id,spelling,reading) WHERE character_context_id IS NULL;
        CREATE TABLE IF NOT EXISTS character_outfits (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            character_context_id TEXT NOT NULL REFERENCES context_items(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'user' CHECK (origin IN ('seed','user','director')),
            source_sequence_id TEXT,
            needs_review INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS character_outfits_name_idx ON character_outfits(character_context_id,lower(name));
        -- The autopilot (agent_log.py, docs/autopilot/HARNESS.md). A run is one agent session over an episode; each
        -- decision it applied is logged in the transaction that applied it, with the target's state before and after,
        -- so it can be undone while the target is unchanged. cut_id is plain text on purpose: a log entry must never
        -- count as a reference that stops cut_history from removing a cut. This is provenance, never an approval status.
        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            agent TEXT NOT NULL DEFAULT '',
            purpose TEXT NOT NULL DEFAULT '',
            policy_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running','finished','undone')),
            claude_items INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS agent_runs_episode_idx ON agent_runs(episode_id,started_at);
        CREATE TABLE IF NOT EXISTS agent_decisions (
            id TEXT PRIMARY KEY,
            run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            cut_id TEXT,
            kind TEXT NOT NULL,
            tier TEXT NOT NULL CHECK(tier IN ('rules','local','claude','human')),
            target_table TEXT NOT NULL,
            target_id TEXT NOT NULL,
            before_json TEXT,
            after_json TEXT,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            confidence REAL,
            summary TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            undone_at TEXT,
            overridden_at TEXT
        );
        CREATE INDEX IF NOT EXISTS agent_decisions_run_idx ON agent_decisions(run_id);
        CREATE INDEX IF NOT EXISTS agent_decisions_episode_idx ON agent_decisions(episode_id,kind);
        CREATE INDEX IF NOT EXISTS agent_decisions_target_idx ON agent_decisions(target_table,target_id);
        CREATE TABLE IF NOT EXISTS agent_escalations (
            id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            kind TEXT NOT NULL,
            group_key TEXT NOT NULL,
            cut_ids_json TEXT NOT NULL DEFAULT '[]',
            question TEXT NOT NULL,
            options_json TEXT NOT NULL DEFAULT '[]',
            proposal_json TEXT,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved','dismissed')),
            resolution_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS agent_escalations_open_idx ON agent_escalations(episode_id,kind,group_key) WHERE status='open';
        -- The missed-cut checker's last look at each cut (missed_cuts.py, milestone M2). cut_id is plain text, never a foreign
        -- key, so a check never stops cut_history from removing a cut; a check is current while the cut keeps these bounds.
        CREATE TABLE IF NOT EXISTS missed_cut_checks (
            cut_id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            cut_start REAL NOT NULL, cut_end REAL NOT NULL,
            version INTEGER NOT NULL, proxy_key TEXT NOT NULL,
            verdict TEXT NOT NULL,
            result_json TEXT NOT NULL DEFAULT '{}',
            job_id TEXT,
            checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS missed_cut_checks_episode_idx ON missed_cut_checks(episode_id,verdict);
        -- The local decider's answers to one consistency review's questions (review_decisions.py, milestone M4): per
        -- question its choice and confidence, and what the resolver did with it. A shadow row was decided with the user's
        -- answers hidden and changed nothing; it measures the decider. cut_id is plain text, never a foreign key.
        CREATE TABLE IF NOT EXISTS review_decisions (
            id TEXT PRIMARY KEY,
            cut_id TEXT NOT NULL,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            result_id TEXT NOT NULL,
            job_id TEXT, run_id TEXT,
            shadow INTEGER NOT NULL DEFAULT 0,
            decisions_json TEXT NOT NULL DEFAULT '{}',
            outcomes_json TEXT NOT NULL DEFAULT '{}',
            model_id TEXT,
            provenance_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS review_decisions_cut_idx ON review_decisions(cut_id,result_id,shadow);
        CREATE INDEX IF NOT EXISTS review_decisions_episode_idx ON review_decisions(episode_id);
        -- What the user tells the autopilot once per project (autopilot_profile.py, the Autopilot profile dialog).
        CREATE TABLE IF NOT EXISTS autopilot_profiles (
            project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
            profile_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS episode_audio_ranges (
            id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('OP','ED','repeat','manual')),
            source TEXT NOT NULL CHECK(source IN ('fingerprint','user')),
            matched_episodes_json TEXT,
            suppressed INTEGER NOT NULL DEFAULT 0,
            CHECK(start_time >= 0 AND end_time > start_time)
        );
        CREATE INDEX IF NOT EXISTS episode_audio_ranges_time_idx ON episode_audio_ranges(episode_id,start_time);
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
            is_main INTEGER NOT NULL DEFAULT 0,
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
        -- What a dialogue compilation was built from, so the editor can tell when the transcript or subtitles changed since.
        CREATE TABLE IF NOT EXISTS sequence_compilation_sources (
            sequence_id TEXT PRIMARY KEY REFERENCES sequences(id) ON DELETE CASCADE,
            result_id TEXT, source_hash TEXT NOT NULL, subtitle_run_id TEXT, transcript_count INTEGER NOT NULL DEFAULT 0,
            subtitle_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        -- Undo/redo log of cut boundary edits (cut_history.py): the active cuts before and after each edit.
        -- Cut ids live only inside the JSON snapshots, so this table never keeps a cut from being deleted.
        CREATE TABLE IF NOT EXISTS cut_edits (
            id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
            action TEXT NOT NULL, label TEXT NOT NULL, cut_id TEXT,
            before_json TEXT NOT NULL, after_json TEXT NOT NULL, created_ids_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, undone_at TEXT
        );
        CREATE INDEX IF NOT EXISTS cut_edits_episode_idx ON cut_edits(episode_id);
        -- People on the evaluated (target) frames of a cut (target_characters.py). Kept apart from
        -- character_detections on purpose: nothing here is a gallery example or a composition input.
        -- selection_key is the keyframe_results.cache_key the run was built for; a different selection
        -- deletes the run (keyframes.process). The label belongs to the slot, so one change covers every frame.
        CREATE TABLE IF NOT EXISTS target_detection_runs (
            id TEXT PRIMARY KEY, cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            selection_key TEXT NOT NULL, cut_start REAL NOT NULL, cut_end REAL NOT NULL,
            job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL, frames_json TEXT NOT NULL,
            model_id TEXT, model_revision TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            candidates_json TEXT NOT NULL DEFAULT '[]', cast_locked INTEGER, review_version INTEGER NOT NULL DEFAULT 0,
            UNIQUE(cut_id, selection_key)
        );
        CREATE TABLE IF NOT EXISTS target_slots (
            run_id TEXT NOT NULL REFERENCES target_detection_runs(id) ON DELETE CASCADE,
            slot INTEGER NOT NULL, origin TEXT NOT NULL,
            character_context_id TEXT REFERENCES context_items(id) ON DELETE SET NULL,
            label_source TEXT, ignored INTEGER NOT NULL DEFAULT 0, first_detection_id TEXT,
            PRIMARY KEY(run_id, slot)
        );
        CREATE TABLE IF NOT EXISTS target_detections (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES target_detection_runs(id) ON DELETE CASCADE,
            frame_ordinal INTEGER NOT NULL, slot INTEGER NOT NULL, box_json TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto', scores_json TEXT
        );
        CREATE INDEX IF NOT EXISTS target_detections_run_idx ON target_detections(run_id, frame_ordinal);
        CREATE TABLE IF NOT EXISTS target_cast_settings (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE, locked INTEGER NOT NULL
        );
        -- Frames the user added to or removed from a cut's keyframe selection (keyframes.apply_edits). A per-cut
        -- setting, not a result: it outlives a new selection and is ignored once the cut's boundaries change.
        CREATE TABLE IF NOT EXISTS keyframe_edits (
            cut_id TEXT PRIMARY KEY REFERENCES cuts(id) ON DELETE CASCADE,
            cut_start REAL NOT NULL, cut_end REAL NOT NULL,
            added_json TEXT NOT NULL DEFAULT '[]', removed_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        -- User correction history stays available after the keyframe selection and its run are replaced.
        CREATE TABLE IF NOT EXISTS target_corrections (
            id INTEGER PRIMARY KEY, cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL, selection_key TEXT NOT NULL, action TEXT NOT NULL,
            before_json TEXT NOT NULL DEFAULT '[]', after_json TEXT NOT NULL DEFAULT '[]',
            details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS target_corrections_run_idx ON target_corrections(run_id, action, id);
        -- Keyframe analysis (keyframe_analysis.py): Qwen3.5 on the keyframes plus DWPose keypoints and the camera measured
        -- from pixels. One run per cut, keyframe selection and inputs (input_hash); its windows are the checkpoint, so a
        -- run that is not complete is reused by the next job with the same inputs (as subtitle runs are). The merged
        -- answer is also stored as the keyframe_analysis analysis result (result_id); the keypoints live only here.
        -- target_run_id has no foreign key: a new keyframe selection deletes the target runs of the old one.
        CREATE TABLE IF NOT EXISTS keyframe_analysis_runs (
            id TEXT PRIMARY KEY, cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
            selection_key TEXT NOT NULL, cut_start REAL NOT NULL, cut_end REAL NOT NULL,
            job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL, input_hash TEXT NOT NULL,
            phase TEXT NOT NULL DEFAULT 'prepare', method TEXT NOT NULL, roster_source TEXT NOT NULL,
            target_run_id TEXT, roster_json TEXT NOT NULL DEFAULT '[]', provenance_json TEXT NOT NULL,
            timings_json TEXT NOT NULL DEFAULT '{}', result_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS keyframe_analysis_runs_cut_idx ON keyframe_analysis_runs(cut_id, selection_key, phase);
        CREATE TABLE IF NOT EXISTS keyframe_analysis_frames (
            run_id TEXT NOT NULL REFERENCES keyframe_analysis_runs(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, label TEXT NOT NULL, frame_time REAL NOT NULL, relative_time REAL NOT NULL,
            reason TEXT, image_size_json TEXT NOT NULL, pose_json TEXT, states_json TEXT NOT NULL DEFAULT '{}', seconds REAL,
            PRIMARY KEY(run_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS keyframe_analysis_pairs (
            run_id TEXT NOT NULL REFERENCES keyframe_analysis_runs(id) ON DELETE CASCADE,
            pair_index INTEGER NOT NULL, from_ordinal INTEGER NOT NULL, to_ordinal INTEGER NOT NULL,
            start REAL NOT NULL, "end" REAL NOT NULL, camera_json TEXT NOT NULL, changes_json TEXT,
            PRIMARY KEY(run_id, pair_index)
        );
        CREATE TABLE IF NOT EXISTS keyframe_analysis_windows (
            run_id TEXT NOT NULL REFERENCES keyframe_analysis_runs(id) ON DELETE CASCADE,
            window_index INTEGER NOT NULL, labels_json TEXT NOT NULL, answer_json TEXT, rules_json TEXT,
            attempts_json TEXT NOT NULL DEFAULT '[]', prompt TEXT NOT NULL DEFAULT '', ok INTEGER NOT NULL DEFAULT 0, seconds REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id, window_index)
        );
        """)
        from .assistant import store as assistant_store
        assistant_store.init_schema(db)
        from . import chat_gateway_schema
        chat_gateway_schema.init_schema(db)
        from . import point_detect
        point_detect.init_schema(db)
        if "reviewer" not in {row["name"] for row in db.execute("PRAGMA table_info(agent_escalations)")}:
            # Who answers an escalation next: 'claude' (the second reviewer, HARNESS.md tier 2) or 'human' (the inbox).
            db.execute("ALTER TABLE agent_escalations ADD COLUMN reviewer TEXT NOT NULL DEFAULT 'human'")
        sequence_columns = {row["name"] for row in db.execute("PRAGMA table_info(sequences)")}
        if "reviewed_compilation_id" not in sequence_columns:
            # Workflow gate: the compilation revision the user confirmed. Not an approval status on results.
            db.execute("ALTER TABLE sequences ADD COLUMN reviewed_compilation_id TEXT")
        if "shot_points_json" not in sequence_columns:
            # Shot boundaries the user placed on the Dialogue timeline: [{after_cut_id, time}] in episode seconds.
            db.execute("ALTER TABLE sequences ADD COLUMN shot_points_json TEXT")
        if "extension_json" not in sequence_columns:
            # Workflow marker of a cut added to the range: {cut_id, cut, from_time, rewrite_shots}; cleared when the direction is queued again.
            db.execute("ALTER TABLE sequences ADD COLUMN extension_json TEXT")
        if "is_main" not in {row["name"] for row in db.execute("PRAGMA table_info(context_items)")}:
            # A character the user marked as one of the series' main characters.
            db.execute("ALTER TABLE context_items ADD COLUMN is_main INTEGER NOT NULL DEFAULT 0")
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
        target_columns = {
            'target_detection_runs': {'candidates_json': "TEXT NOT NULL DEFAULT '[]'", 'cast_locked': 'INTEGER', 'review_version': 'INTEGER NOT NULL DEFAULT 0'},
            'target_slots': {'ignored': 'INTEGER NOT NULL DEFAULT 0', 'first_detection_id': 'TEXT'},
            'target_detections': {'scores_json': 'TEXT'},
        }
        for table, fields in target_columns.items():
            have = {row['name'] for row in db.execute(f'PRAGMA table_info({table})')}
            for name, definition in fields.items():
                if name not in have:
                    db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')
        result_columns = {row["name"] for row in db.execute("PRAGMA table_info(analysis_results)")}
        columns = {row["name"] for row in db.execute("PRAGMA table_info(transcript_segments)")}
        episode_columns = {row["name"] for row in db.execute("PRAGMA table_info(episodes)")}
        for name in ("transcript_stale_reason", "speaker_attribution_stale_reason"):
            if name not in episode_columns:
                db.execute(f"ALTER TABLE episodes ADD COLUMN {name} TEXT")
        for name in ("words_json", "speaker_source", "confidence", "evidence_json"):
            if name not in columns:
                db.execute(f"ALTER TABLE transcript_segments ADD COLUMN {name} TEXT")
        if "range_embeddings_json" not in {row["name"] for row in db.execute("PRAGMA table_info(character_voice_samples)")}:
            db.execute("ALTER TABLE character_voice_samples ADD COLUMN range_embeddings_json TEXT")
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
            ('character_detection_runs', 'input_json', "TEXT NOT NULL DEFAULT '{}'"),
            ('target_detection_runs', 'input_json', "TEXT NOT NULL DEFAULT '{}'"),
            ('cuts', 'no_characters', 'INTEGER NOT NULL DEFAULT 0'),
            ('text_generation_settings', 'local_model', "TEXT NOT NULL DEFAULT 'mtp'"),
            ('text_generation_settings', 'openrouter_audio_model',
             "TEXT NOT NULL DEFAULT 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'"),
            ('local_text_model_state', 'variant', "TEXT NOT NULL DEFAULT ''"),
            ('local_text_model_state', 'variants_json', "TEXT NOT NULL DEFAULT '{}'"),
            ('episodes', 'subtitle_language', 'TEXT'),
            ('subtitle_runs', 'input_fingerprint', 'TEXT'),
            ('subtitle_runs', 'stale_reason', 'TEXT'),
            ('cuts', 'created_at', 'TEXT'),
            ('analysis_results', 'stale_reason', 'TEXT'),
            ('cut_review_drafts', 'stale_reason', 'TEXT'),
            ('sequence_results', 'stale_reason', 'TEXT'),
            ('character_detections', 'match_model_id', 'TEXT'),
            ('character_detections', 'match_model_revision', 'TEXT'),
            # Bulk recognition: ranked suggestions for unconfirmed boxes, the
            # user's override for using a confirmed box as a matching example
            # (NULL automatic, 1 always, 0 never), and the worker's verdict.
            ('character_detections', 'candidates_json', 'TEXT'),
            ('character_detections', 'recognition_use', 'INTEGER'),
            ('character_detections', 'gallery_status', 'TEXT'),
            ('character_detections', 'gallery_note', 'TEXT'),
            # A numbered Unregistered card (unregistered.py): a minor person with a line or a box but no record.
            ('context_items', 'unregistered', 'INTEGER NOT NULL DEFAULT 0'),
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
            ("priority_class", "TEXT NOT NULL DEFAULT 'batch' CHECK (priority_class IN ('chat','interactive','batch'))"),
            ("request_key", "TEXT"),
            ("not_before", "REAL NOT NULL DEFAULT 0"),
            ("restart_input_json", "TEXT"),
            ("input_json", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if column not in job_columns:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
        from .name_glossary import seed
        seed(db)
        from . import outfit_library
        outfit_library.seed(db)
        migrate_queue_dependencies(db)
        clear_picture_stale_marks(db)
        checkpoint_columns = {row['name'] for row in db.execute('PRAGMA table_info(analysis_job_cuts)')}
        if 'source_cut_json' not in checkpoint_columns:
            db.execute('ALTER TABLE analysis_job_cuts ADD COLUMN source_cut_json TEXT')
        # The queue lists failures that still need the user (runs.attention) until they are run again or dismissed.
        if 'dismissed_at' not in job_columns:
            db.execute('ALTER TABLE jobs ADD COLUMN dismissed_at TEXT')
            # Failures from before the queue tracked them would otherwise all ask for attention at once.
            db.execute("UPDATE jobs SET dismissed_at=CURRENT_TIMESTAMP WHERE status='failed' AND COALESCE(finished_at,updated_at) < datetime('now','-1 day')")
        db.execute('CREATE INDEX IF NOT EXISTS jobs_episode_created_idx ON jobs(episode_id,created_at)')
        db.execute('CREATE INDEX IF NOT EXISTS jobs_request_idx ON jobs(episode_id,pipeline,request_key) WHERE request_key IS NOT NULL')
        # 2026-11-18: Qwen3.6 with its vision projector became the default of every visual role (visual_models.py). A saved
        # 'qwen35-27b' was the old default being restored from the Visual engine dialog, so it would keep the new default
        # from ever applying; drop those rows once. Choices of any other model stay (Qwen3.5 itself was removed on 2026-11-21, so a later choice of it falls back to the default).
        if not db.execute("SELECT 1 FROM app_flags WHERE name='visual-default-qwen36'").fetchone():
            db.execute("DELETE FROM visual_model_settings WHERE model='qwen35-27b'")
            db.execute("INSERT INTO app_flags(name) VALUES ('visual-default-qwen36')")
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
        DROP TRIGGER IF EXISTS jobs_finished;
        CREATE TRIGGER jobs_finished AFTER UPDATE OF status ON jobs
        WHEN NEW.status IN ('done','failed','cancelled') AND OLD.status IN ('queued','running')
        BEGIN UPDATE jobs SET finished_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;
        """)
        db.execute("CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status,lease_until,created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_leases_idx ON jobs(lease_until) WHERE lease_owner IS NOT NULL")
        db.execute("CREATE INDEX IF NOT EXISTS transcript_active_idx ON transcript_segments(episode_id,start_time) WHERE superseded_by IS NULL")
        # The episode's per-cut flags look up the latest run's boxes and the cut's latest job once per cut; without these
        # every lookup scanned the whole table, which made GET /api/episodes/{id} take seconds on long episodes.
        db.execute("CREATE INDEX IF NOT EXISTS character_detections_run_idx ON character_detections(run_id,position)")
        db.execute("CREATE INDEX IF NOT EXISTS analysis_job_cuts_cut_idx ON analysis_job_cuts(cut_id)")
        # Incremental composition reads need a watermark for edits as well as
        # new revisions. SQLite's millisecond clock is used consistently here.
        trigger_drops = '\n'.join('DROP TRIGGER "' + trigger['name'].replace('"', '""') + '";'
                                  for trigger in db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'composition_%'").fetchall())
        db.executescript('BEGIN IMMEDIATE;\n' + trigger_drops + """
        -- Cancelling is its own outcome (2026-11-15). Older cancels were stored as failures with the stage
        -- 'Cancelled'; converting them here, while the composition triggers are dropped, leaves the watermarks alone.
        UPDATE jobs SET status='cancelled' WHERE status='failed' AND stage='Cancelled';
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
        from . import episode_events
        episode_events.init_schema(db)
        # 2026-11-23 (D27): "Your answers" and the Cut summary are off in the autopilot profile unless the user turns them on;
        # the stored profiles held the old defaults, so switch both off once.
        if not db.execute("SELECT 1 FROM app_flags WHERE name='autopilot-answers-summaries-off'").fetchone():
            from . import autopilot_profile
            autopilot_profile.answers_and_summaries_off(db)
            # note_override used to mark a person re-confirming the agent's own label as an override; clear those once.
            db.execute("""UPDATE agent_decisions SET overridden_at=NULL WHERE overridden_at IS NOT NULL AND target_table='character_detections'
                AND json_valid(after_json) AND json_extract(after_json,'$.character_context_id') IS NOT NULL
                AND json_extract(after_json,'$.character_context_id')=(SELECT d.character_context_id FROM character_detections d WHERE d.id=target_id)""")
            db.execute("INSERT INTO app_flags(name) VALUES ('autopilot-answers-summaries-off')")
        # 2026-11-24 (D32): consistency review questions are answered in the cut dialog, never in the inbox.
        if not db.execute("SELECT 1 FROM app_flags WHERE name='inbox-no-review-questions'").fetchone():
            db.execute("""UPDATE agent_escalations SET status='dismissed',resolved_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP,
                resolution_json='{"why": "review questions are answered in the cut dialog, not in the inbox"}'
                WHERE kind='review_question' AND status='open'""")
            db.execute("INSERT INTO app_flags(name) VALUES ('inbox-no-review-questions')")
        db.commit()


def one(sql, args=(), *, conn=None):
    if conn is None:
        with connect() as conn:
            return one(sql, args, conn=conn)
    row = conn.execute(sql, args).fetchone()
    return dict(row) if row else None


def all_rows(sql, args=(), *, conn=None):
    if conn is None:
        with connect() as conn:
            return all_rows(sql, args, conn=conn)
    return [dict(row) for row in conn.execute(sql, args).fetchall()]
