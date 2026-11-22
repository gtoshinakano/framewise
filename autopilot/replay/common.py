"""Shared pieces of the replay harness (docs/autopilot/HARNESS.md section 6).

A replay scores one resolver against decisions the user already made by hand, with the human answer hidden. It reads
a *copy* of the database, never the live one, and media read-only. It runs where ``backend/app`` imports work: in a
throwaway container of the API image (``autopilot/replay/run.sh``).
"""
import datetime
import json
import os
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_database(live: Path, target: Path) -> Path:
    """A consistent copy of the live database through SQLite's backup API (the live file is only read)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f'file:{live}?mode=ro', uri=True)
    try:
        copy = sqlite3.connect(target)
        source.backup(copy)
        copy.close()
    finally:
        source.close()
    return target


def open_copy(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def commit() -> str:
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip() or 'unknown'
    except (OSError, subprocess.TimeoutExpired):
        return os.environ.get('GIT_COMMIT', 'unknown')


def write_report(name: str, data: dict, markdown: str, out_dir: Path) -> Path:
    """``<out>/<date>-<name>.json`` and ``.md``, stamped with the date and commit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M')
    data = {'resolver': name, 'date': stamp, 'commit': os.environ.get('GIT_COMMIT') or commit(), **data}
    base = out_dir / f'{stamp}-{name}'
    base.with_suffix('.json').write_text(json.dumps(data, indent=1, ensure_ascii=False))
    base.with_suffix('.md').write_text(f"# Replay: {name}\n\n{stamp} UTC, commit {data['commit']}\n\n{markdown}\n")
    return base.with_suffix('.md')
