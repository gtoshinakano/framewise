"""Resolve a saved result back to its original video frame without JPEG storage."""
import json
import math
import re
from pathlib import Path

from . import db


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def result_frame_spec(result_id, conn=None):
    if conn is None:
        with db.connect() as connection:
            return result_frame_spec(result_id, connection)
    seen = set()
    while result_id and result_id not in seen:
        seen.add(result_id)
        row = conn.execute('''SELECT a.id,a.title,a.origin,a.context_json,a.parent_id,c.start_time,c.end_time,
                                     e.id AS episode_id,e.project_id,e.original_ext
                              FROM analysis_results a JOIN cuts c ON c.id=a.cut_id
                              JOIN episodes e ON e.id=c.episode_id WHERE a.id=?''', (result_id,)).fetchone()
        if not row:
            return None
        # Manual revisions inherit the exact sampled frame from their parent.
        if row['parent_id'] and row['origin'] in ('manual', 'manual_edit') and row['parent_id'] not in seen and conn.execute('SELECT 1 FROM analysis_results WHERE id=?', (row['parent_id'],)).fetchone():
            result_id = row['parent_id']
            continue
        # A legacy image without reconstructible metadata remains usable.
        folder = db.DATA_DIR / 'projects' / row['project_id'] / 'episodes' / row['episode_id']
        legacy = folder / 'analysis' / f"{row['id']}.jpg"
        match = re.fullmatch(r'(Frame|Object detections) at (\d+(?:\.\d+)?)s', row['title'] or '')
        if not match and row['title'] in ('Initial frame description', 'Last frame description'):
            try:
                context = json.loads(row['context_json'] or '{}')
                timestamp = _number(context.get('frame_time'))
                cut = context.get('cut') or {}
                start, end = _number(cut.get('start_time')), _number(cut.get('end_time'))
                original = folder / f"original{row['original_ext']}"
                if timestamp is not None and start is not None and end is not None and 0 <= start <= timestamp < end and original.is_file():
                    return {'source': original, 'time': timestamp, 'start': start, 'end': end, 'width': 1280, 'last': False, 'seek_tolerance': .000002, 'legacy': legacy, 'episode_id': row['episode_id']}
            except (ValueError, TypeError, AttributeError):
                pass
        if match:
            timestamp = float(match.group(2))
            start, end = row['start_time'], row['end_time']
            try:
                context = json.loads(row['context_json'])
                cut = context.get('cut') or {}
                if _number(cut.get('start_time')) is not None and _number(cut.get('end_time')) is not None:
                    start, end = float(cut['start_time']), float(cut['end_time'])
                stored_time = _number(context.get('frame_time'))
                if stored_time is not None:
                    timestamp = stored_time
                frames = context.get('sampled_frames') or []
                if isinstance(frames, list):
                    candidates = [frame for frame in frames if isinstance(frame, dict) and _number(frame.get('time')) is not None]
                    exact = next((f for f in candidates if f.get('frame_id') == row['id']), None)
                    if exact is None and candidates and stored_time is None:
                        closest = min(candidates, key=lambda f: abs(float(f['time']) - timestamp))
                        exact = closest if abs(float(closest['time']) - timestamp) <= .00051 else None
                    if exact:
                        timestamp = float(exact['time'])
            except (ValueError, TypeError, AttributeError):
                pass
            original = folder / f"original{row['original_ext']}"
            if original.is_file() and 0 <= start < end and start <= timestamp < end + .001:
                # Actual decoded timestamps are stored at microsecond precision.
                return {'source': original, 'time': min(timestamp, end - .000001), 'start': start, 'end': end, 'width': 1280, 'last': False, 'seek_tolerance': .000002, 'legacy': legacy, 'episode_id': row['episode_id']}
        if legacy.is_file():
            return {'legacy': legacy}
        result_id = row['parent_id']
    return None
