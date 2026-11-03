"""Reclaim old frame JPEGs only after verifying an identical replacement.

Run inside the app container: python -m app.frame_gc --apply
Without --apply, only inspect and report candidates.
"""
import argparse
import json
import re
from pathlib import Path

from . import db, frame_cache, storage

KINDS = {'Recreatable result frame', 'Recreatable run frame', 'Recreatable cut preview', 'Recreatable sampled frame'}


def collect(apply=False, progress=print):
    totals = {'examined': 0, 'removed': 0, 'bytes_removed': 0, 'different': 0, 'unavailable': 0, 'changed_or_busy': 0}
    with db.connect() as conn:
        initial = [item for item in storage._inventory(conn)['files'] if item['kind'] in KINDS and item['deletable'] and item['path'].endswith('.jpg')]
        tasks = []
        for item in initial:
            try:
                spec = storage._recreation_spec(conn, item)
                gallery = item['kind'] == 'Recreatable cut preview' and re.fullmatch(r'cut_\d{6}\.jpg', Path(item['path']).name)
                if spec is None and not gallery:
                    totals['unavailable'] += 1
                    continue
                source_stat = None
                if spec:
                    source = Path(spec['source'])
                    stat = source.stat()
                    source_stat = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                tasks.append((item, spec, source_stat))
            except (OSError, TypeError, ValueError, KeyError):
                totals['unavailable'] += 1
    # Nearby sample times share the same small in-memory cache, including a
    # run frame and the corresponding result/detection frames.
    tasks.sort(key=lambda task: (str(task[1]['source']) if task[1] else task[0]['path'], task[1]['time'] if task[1] else -1))
    progress(f"Found {len(initial)} generated JPEGs eligible for verification.")
    for offset in range(0, len(tasks), 48):
        batch = tasks[offset:offset + 48]
        pending = []
        for item, spec, source_stat in batch:
            try:
                future = frame_cache.request(spec) if spec else None
                pending.append((item, spec, source_stat, future))
            except (OSError, ValueError, RuntimeError):
                totals['unavailable'] += 1
        matched = []
        for item, spec, source_stat, future in pending:
            totals['examined'] += 1
            try:
                if future is not None and future.result(timeout=120) != (db.DATA_DIR / item['path']).read_bytes():
                    totals['different'] += 1
                    continue
                matched.append((item, spec, source_stat))
            except Exception:
                totals['unavailable'] += 1
        if apply and matched:
            # Recheck current job state, database links, metadata, and source
            # identity while job creation is serialized by the SQLite lock.
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                current = {item['path']: item for item in storage._inventory(conn)['files']}
                for item, spec, source_stat in matched:
                    latest = current.get(item['path'])
                    if not latest or not latest['deletable'] or latest['kind'] != item['kind'] or latest['size'] != item['size'] or latest['modified_ns'] != item['modified_ns']:
                        totals['changed_or_busy'] += 1
                        continue
                    try:
                        if spec:
                            now = storage._recreation_spec(conn, latest)
                            stat = Path(spec['source']).stat()
                            if now != spec or (stat.st_ino, stat.st_size, stat.st_mtime_ns) != source_stat:
                                totals['changed_or_busy'] += 1
                                continue
                        storage._unlink_checked(item['path'], latest)
                        totals['removed'] += 1
                        totals['bytes_removed'] += latest['size']
                    except (OSError, TypeError, ValueError, KeyError):
                        totals['changed_or_busy'] += 1
        progress(f"Checked {min(offset + len(batch), len(tasks))}/{len(tasks)}; removed {totals['removed']}, retained {totals['different'] + totals['unavailable'] + totals['changed_or_busy']}.")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='remove verified generated JPEGs')
    args = parser.parse_args()
    print(json.dumps(collect(apply=args.apply)), flush=True)


if __name__ == '__main__':
    main()
