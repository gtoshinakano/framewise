"""Read the chosen cuts, their keyframes and their context from the database (read-only) and cut them into windows.

A cut's frames are the keyframe selector's choice (``keyframe_results``) for exactly the cut's current
boundaries. When no such row exists, or when ``--kf`` overrides are given, the selection is recomputed here
with ``app.keyframes`` (CPU, nothing written to the database). Frames are labelled F1..Fn per cut and split
into windows of ``size`` frames where each window starts on the previous window's last frame:
A B C D E | E F G H I | I J ...
"""
import json
import re
import sqlite3
from pathlib import Path


def connect_readonly(db_path):
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _json(value, default=None):
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def parse_cuts(spec):
    """'139-158,200,210-212' -> sorted UI cut numbers (1-based, as the app shows them)."""
    numbers = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', part)
        if match:
            lo, hi = int(match[1]), int(match[2])
            if hi < lo:
                lo, hi = hi, lo
            numbers.update(range(lo, hi + 1))
        elif part.isdigit():
            numbers.add(int(part))
        else:
            raise ValueError(f'Invalid cut range: {part!r}')
    if not numbers:
        raise ValueError('No cuts selected')
    return sorted(numbers)


def make_windows(frames, size=5):
    """Windows of ``size`` frames, each starting on the previous window's last frame. One frame -> one window of one."""
    if size < 2:
        raise ValueError('A window needs at least 2 frames')
    if len(frames) <= 1:
        return [{'index': 0, 'frames': list(frames)}]
    windows, start = [], 0
    while start < len(frames) - 1:
        chunk = frames[start:start + size]
        windows.append({'index': len(windows), 'frames': chunk})
        start += size - 1
    return windows


def find_episode(conn, episode):
    row = conn.execute('SELECT * FROM episodes WHERE id=?', (episode,)).fetchone()
    if row is None:
        rows = conn.execute('SELECT * FROM episodes WHERE lower(title)=lower(?)', (episode,)).fetchall()
        if len(rows) > 1:
            raise ValueError(f'Several episodes are titled {episode!r}; pass the id')
        row = rows[0] if rows else None
    if row is None:
        raise ValueError(f'Episode not found: {episode!r}')
    return dict(row)


def subtitle_band(conn, episode_id):
    run = conn.execute("SELECT id, band_json FROM subtitle_runs WHERE episode_id=? AND status='complete' ORDER BY rowid DESC LIMIT 1", (episode_id,)).fetchone()
    band = _json(run['band_json'], {}) if run else {}
    top, bottom = band.get('top'), band.get('bottom')
    if isinstance(top, (int, float)) and isinstance(bottom, (int, float)) and 0 <= top < bottom <= 1 and (top > 0 or bottom < 1):
        return {'top': top, 'bottom': bottom, 'source': 'detected', 'subtitle_run_id': run['id']}
    return None


def cached_keyframes(conn, cut):
    """The newest keyframe_results row computed for exactly these cut boundaries, or None."""
    for row in conn.execute('SELECT result_json FROM keyframe_results WHERE cut_id=? ORDER BY rowid DESC', (cut['id'],)):
        result = _json(row['result_json'])
        if isinstance(result, dict) and abs(result.get('start_timestamp_seconds', -1) - cut['start_time']) < 1e-6 \
                and abs(result.get('end_timestamp_seconds', -1) - cut['end_time']) < 1e-6:
            return result
    return None


def recompute_keyframes(source, cuts, overrides, band):
    """``keyframes.process`` without the database: one decode pass over the requested cuts (sorted by start)."""
    from app import keyframes, media
    settings = keyframes.config(overrides)
    mask = band or {'top': settings['mask_top'], 'bottom': settings['mask_bottom'], 'source': 'fallback', 'subtitle_run_id': None}
    info = media.probe(source)
    ordered = sorted(cuts, key=lambda c: c['start_time'])
    results, index, shot = {}, 0, None
    for ordinal, timestamp, frame in keyframes.decoded(source, info['width'], info['height'], settings['longest_edge']):
        while index < len(ordered) and timestamp >= ordered[index]['end_time']:
            if shot is not None:
                results[ordered[index]['id']] = shot.finish()
            index += 1
            shot = None
        if index == len(ordered):
            break
        cut = ordered[index]
        if timestamp >= cut['start_time']:
            if shot is None:
                shot = keyframes.Shot({'id': cut['id'], 'start_time': cut['start_time'], 'end_time': cut['end_time']}, settings, mask)
            shot.add(ordinal, timestamp, frame)
    if index < len(ordered) and shot is not None:
        results[ordered[index]['id']] = shot.finish()
    return results, settings


def roster(conn, cut, size_hint):
    """Named boxes of the first frame (latest detection run for exactly these boundaries), left to right, 0-1000 grid."""
    if cut['no_characters']:
        return []
    run = conn.execute('SELECT * FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut['id'],)).fetchone()
    if not run or abs((run['cut_start'] or -1) - cut['start_time']) > 1e-3 or abs((run['cut_end'] or -1) - cut['end_time']) > 1e-3:
        return []
    frames = _json(run['frames_json'], [])
    size = next((frame.get('image_size') for frame in frames if frame.get('position') == 'first'), None) or size_hint
    rows = conn.execute('''SELECT d.box_json, d.character_context_id, d.label_source, c.title FROM character_detections d
                           JOIN context_items c ON c.id=d.character_context_id WHERE d.run_id=? AND d.position='first' ORDER BY d.rowid''', (run['id'],)).fetchall()
    people, seen = [], set()
    for row in rows:
        box = _json(row['box_json'])
        if row['title'] in seen or not (isinstance(box, list) and len(box) == 4 and size and size[0] and size[1]):
            continue
        seen.add(row['title'])
        grid = [round(box[0] * 1000 / size[0]), round(box[1] * 1000 / size[1]), round(box[2] * 1000 / size[0]), round(box[3] * 1000 / size[1])]
        centre = (grid[0] + grid[2]) / 2
        side = 'left' if centre < 380 else 'right' if centre > 620 else 'centre'
        people.append({'name': row['title'], 'character_context_id': row['character_context_id'], 'box': grid, 'side': side, 'label_source': row['label_source']})
    people.sort(key=lambda p: p['box'][0])
    return people


def current_result(conn, cut_id, kind, structured):
    row = conn.execute('''SELECT a.content, a.model_id, a.job_id FROM analysis_results a WHERE a.cut_id=? AND a.superseded_by IS NULL AND a.kind=?
                          AND json_extract(a.settings_json,'$.structured_type')=? ORDER BY a.rowid DESC LIMIT 1''', (cut_id, kind, structured)).fetchone()
    return dict(row) if row else None


def transcript(conn, cut):
    rows = conn.execute('''SELECT t.start_time, t.end_time, t.text, t.speaker_label, c.title AS character_name FROM transcript_segments t
                           LEFT JOIN context_items c ON c.id=t.character_context_id
                           WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>? ORDER BY t.start_time''',
                        (cut['episode_id'], cut['end_time'], cut['start_time'])).fetchall()
    return [{'start': round(r['start_time'] - cut['start_time'], 3), 'end': round(r['end_time'] - cut['start_time'], 3), 'text': (r['text'] or '').strip()[:120],
             'speaker': r['character_name'] or r['speaker_label'], 'character': r['character_name']} for r in rows]


def baseline(conn, cut_id):
    """The stored production video analysis as text prompts read it, plus how long its stage ran."""
    row = current_result(conn, cut_id, 'cut_description', 'video_analysis')
    if not row:
        return None
    content = _json(row['content'])
    try:
        from app import prompt_context
        view = prompt_context.video_view(content)
    except Exception:  # noqa: BLE001 - the baseline is optional
        view = None
    seconds = None
    if row.get('job_id'):
        stage = conn.execute("SELECT started_at, finished_at FROM analysis_job_stages WHERE job_id=? AND cut_id=? AND stage_key='video_analysis' AND status='done'",
                             (row['job_id'], cut_id)).fetchone()
        if stage and stage['started_at'] and stage['finished_at']:
            seconds = round(stage['finished_at'] - stage['started_at'], 2)
    return {'view': view, 'model': row['model_id'], 'contract': (content or {}).get('contract') if isinstance(content, dict) else None, 'seconds': seconds}


def first_frame_view(conn, cut_id):
    row = current_result(conn, cut_id, 'first_frame', 'initial_frame')
    if not row:
        return None
    content = _json(row['content'])
    try:
        from app import prompt_context
        return prompt_context.frame_view(content)
    except Exception:  # noqa: BLE001
        return content


def extract_frame(source, output, time, cut, ordinal, width):
    """The exact decoded frame (unique-PTS seek, else source ordinal), as the Target frames tab extracts it."""
    from app import frame_cache
    data = frame_cache._generate({'source': source, 'time': time, 'start': cut['start_time'], 'end': cut['end_time'], 'width': width, 'last': False, 'ordinal': ordinal})
    output.write_bytes(data)


def build(db_path, data_dir, episode, cut_spec, run_dir, kf_overrides=None, window_size=5, image_width=1024, log=print):
    """samples.json entries for the requested cuts (frames extracted into ``run_dir/frames``)."""
    numbers = parse_cuts(cut_spec)
    conn = connect_readonly(db_path)
    try:
        ep = find_episode(conn, episode)
        source = Path(data_dir) / 'projects' / ep['project_id'] / 'episodes' / ep['id'] / f"original{ep['original_ext']}"
        if not source.is_file():
            raise RuntimeError(f'Original video missing: {source}')
        cuts = [dict(r) for r in conn.execute(f'''SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index IN ({','.join('?' * len(numbers))})
                                                 ORDER BY cut_index''', (ep['id'], *[n - 1 for n in numbers]))]
        found = {c['cut_index'] + 1 for c in cuts}
        missing = [n for n in numbers if n not in found]
        if missing:
            log(f'cuts not found (inactive or out of range): {missing}')
        band = subtitle_band(conn, ep['id'])
        keyframes, settings_used, origin = {}, None, {}
        if not kf_overrides:
            for cut in cuts:
                result = cached_keyframes(conn, cut)
                if result:
                    keyframes[cut['id']] = result
                    origin[cut['id']] = 'cached'
        todo = [cut for cut in cuts if cut['id'] not in keyframes]
        if todo:
            log(f'recomputing keyframes for {len(todo)} cut(s) (CPU decode)')
            computed, settings_used = recompute_keyframes(source, todo, kf_overrides or {}, band)
            for cut_id, result in computed.items():
                keyframes[cut_id] = result
                origin[cut_id] = 'recomputed'
        samples = []
        for cut in cuts:
            result = keyframes.get(cut['id'])
            if not result:
                log(f'cut {cut["cut_index"] + 1}: no keyframes; skipped')
                continue
            number = cut['cut_index'] + 1
            sid = f'c{number:03d}'
            cut['episode_id'] = ep['id']
            frames_dir = Path(run_dir) / 'frames' / sid
            frames_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for k, item in enumerate(sorted(result['selected_frames'], key=lambda f: f['source_frame_index']), 1):
                rel = f'frames/{sid}/F{k}.jpg'
                target = Path(run_dir) / rel
                if not target.is_file():
                    extract_frame(source, target, item['timestamp_seconds'], cut, item['source_frame_index'], image_width)
                frames.append({'label': f'F{k}', 'k': k, 't': round(item['relative_timestamp_seconds'], 3), 'time': item['timestamp_seconds'],
                               'source_frame_index': item['source_frame_index'], 'reason': item.get('reason'),
                               'change_score': round(item.get('change_score', 0), 1), 'path': rel})
            people = roster(conn, cut, [ep['width'], ep['height']])
            samples.append({
                'id': sid, 'cut_number': number, 'cut_id': cut['id'], 'episode': ep['title'], 'episode_id': ep['id'], 'source': str(source),
                'start': cut['start_time'], 'end': cut['end_time'], 'duration': round(cut['end_time'] - cut['start_time'], 3),
                'width': ep['width'], 'height': ep['height'], 'no_characters': bool(cut['no_characters']),
                'roster': people, 'first_frame_view': first_frame_view(conn, cut['id']), 'transcript': transcript(conn, cut),
                'subtitle_band': band or result.get('subtitle_mask'),
                'keyframes': {'origin': origin.get(cut['id']), 'settings': settings_used if origin.get(cut['id']) == 'recomputed' else None,
                              'change_level': result.get('change_level'), 'shot_change_score': round(result.get('shot_change_score', 0), 1),
                              'detected_events': result.get('detected_events'), 'coverage_limited': result.get('coverage_limited'),
                              'total_frames': result.get('total_frames')},
                'frames': frames, 'windows': make_windows(frames, window_size), 'baseline': baseline(conn, cut['id']),
            })
            log(f'cut {number}: {len(frames)} frames, {len(samples[-1]["windows"])} window(s), roster {[p["name"] for p in people]}')
        return samples, {'episode': ep['title'], 'episode_id': ep['id'], 'requested': numbers, 'missing': missing, 'subtitle_band': band}
    finally:
        conn.close()


def window_count(samples):
    return sum(len(s['windows']) for s in samples if len(s['frames']) > 1)


def cut_seconds(samples):
    return sum(s['duration'] for s in samples)

