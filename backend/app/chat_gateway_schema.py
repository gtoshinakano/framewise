"""SQLite schema of the chat gateway; kept free of FastAPI because db.init_db also runs in the worker images."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_gateway_requests (
    id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'Queued',
    request_json TEXT NOT NULL, model TEXT NOT NULL DEFAULT '', thinking INTEGER NOT NULL DEFAULT 1,
    content TEXT NOT NULL DEFAULT '', reasoning TEXT NOT NULL DEFAULT '', tool_calls_json TEXT,
    finish_reason TEXT, error TEXT, error_status INTEGER, usage_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_until REAL,
    attempt_count INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, started_at REAL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS chat_gateway_claim_idx ON chat_gateway_requests(status, created_at);
CREATE TABLE IF NOT EXISTS chat_gateway_state (id INTEGER PRIMARY KEY CHECK(id=1), heartbeat REAL NOT NULL DEFAULT 0);
INSERT OR IGNORE INTO chat_gateway_state(id) VALUES (1);
"""


def init_schema(conn):
    conn.executescript(SCHEMA)
