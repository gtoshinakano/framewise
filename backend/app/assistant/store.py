"""SQLite schema and small helpers for Cut Assistance sessions."""
import json
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_sessions (
    id TEXT PRIMARY KEY, cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL DEFAULT 'local',
    model TEXT NOT NULL DEFAULT '', thinking INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS assistant_sessions_cut_idx ON assistant_sessions(cut_id, updated_at);
CREATE TABLE IF NOT EXISTS assistant_turns (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'Queued',
    provider TEXT NOT NULL, model TEXT NOT NULL, thinking INTEGER NOT NULL DEFAULT 1,
    skill TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0, gpu_fallback INTEGER NOT NULL DEFAULT 0,
    priority_class TEXT NOT NULL DEFAULT 'chat', retry_of TEXT,
    lease_owner TEXT, lease_until REAL, attempt_count INTEGER NOT NULL DEFAULT 0,
    error TEXT, usage_json TEXT NOT NULL DEFAULT '{}', version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at REAL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS assistant_turns_claim_idx ON assistant_turns(status, created_at);
CREATE INDEX IF NOT EXISTS assistant_turns_session_idx ON assistant_turns(session_id);
CREATE TABLE IF NOT EXISTS assistant_messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    turn_id TEXT REFERENCES assistant_turns(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '', reasoning TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT, tool_call_id TEXT, tool_name TEXT, tool_summary TEXT,
    streaming INTEGER NOT NULL DEFAULT 0, usage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS assistant_messages_turn_idx ON assistant_messages(turn_id);
CREATE TABLE IF NOT EXISTS assistant_proposals (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    turn_id TEXT REFERENCES assistant_turns(id) ON DELETE CASCADE,
    cut_id TEXT NOT NULL REFERENCES cuts(id) ON DELETE CASCADE,
    target TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL,
    base_result_id TEXT, base_content TEXT NOT NULL DEFAULT '', content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'pending',
    applied_result_id TEXT, model_id TEXT, provider TEXT, diff_base TEXT, diff_label TEXT, payload_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS assistant_proposals_session_idx ON assistant_proposals(session_id);
CREATE TABLE IF NOT EXISTS assistant_worker_state (
    id INTEGER PRIMARY KEY CHECK(id=1), providers_json TEXT NOT NULL DEFAULT '[]', heartbeat REAL NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO assistant_worker_state(id) VALUES (1);
"""

ACTIVE = ('queued', 'running')


def init_schema(conn):
    conn.executescript(SCHEMA)
    columns = {row['name'] for row in conn.execute('PRAGMA table_info(assistant_proposals)')}
    for column in ('diff_base', 'diff_label', 'payload_json'):
        if column not in columns:
            conn.execute(f'ALTER TABLE assistant_proposals ADD COLUMN {column} TEXT')
    turns = {row['name'] for row in conn.execute('PRAGMA table_info(assistant_turns)')}
    for column, definition in (('gpu_fallback', 'INTEGER NOT NULL DEFAULT 0'), ('priority_class', "TEXT NOT NULL DEFAULT 'chat'"), ('retry_of', 'TEXT')):
        if column not in turns:
            conn.execute(f'ALTER TABLE assistant_turns ADD COLUMN {column} {definition}')
    # A chat about a sequence: sequence_id is set and cut_id is the sequence's first cut (the column is
    # NOT NULL and gives the tools their project). Cut chats leave it NULL.
    for table in ('assistant_sessions', 'assistant_proposals'):
        if 'sequence_id' not in {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN sequence_id TEXT REFERENCES sequences(id) ON DELETE CASCADE')


def loads(value, default):
    try:
        parsed = json.loads(value) if value else default
    except (TypeError, ValueError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


def serialize_message(row):
    row = dict(row)
    row['tool_calls'] = loads(row.pop('tool_calls_json', None), [])
    row['usage'] = loads(row.pop('usage_json', None), {})
    row['streaming'] = bool(row['streaming'])
    return row


def serialize_turn(row):
    row = dict(row)
    row['usage'] = loads(row.pop('usage_json', None), {})
    row['thinking'] = bool(row['thinking'])
    row['cancel_requested'] = bool(row['cancel_requested'])
    row['gpu_fallback'] = bool(row.get('gpu_fallback'))
    for key in ('lease_owner', 'lease_until'):
        row.pop(key, None)
    return row


def next_seq(conn, session_id):
    row = conn.execute('SELECT COALESCE(MAX(seq),0)+1 AS seq FROM assistant_messages WHERE session_id=?', (session_id,)).fetchone()
    return row['seq']


def touch_session(conn, session_id):
    conn.execute('UPDATE assistant_sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (session_id,))


def bump(conn, turn_id, **fields):
    """Update turn columns and advance its version so streams notice the change."""
    sets = ''.join(f',{key}=?' for key in fields)
    conn.execute(f'UPDATE assistant_turns SET version=version+1{sets} WHERE id=?', (*fields.values(), turn_id))


def now():
    return time.time()
