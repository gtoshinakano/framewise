"""Durable user edits and the review flags of a target detection run."""
import json
from collections import Counter, defaultdict

from .frame_confidence import PENALTIES


def box_row(row):
    return {'id': row['id'], 'frame_ordinal': row['frame_ordinal'], 'slot': row['slot'],
            'box': json.loads(row['box_json']), 'source': row['source'],
            'scores': json.loads(row['scores_json'] or 'null')}


def record(conn, run, action, before=(), after=(), **details):
    """Capture scores and geometry before an edit, in the same transaction."""
    conn.execute('''INSERT INTO target_corrections(cut_id,run_id,selection_key,action,before_json,after_json,details_json)
                    VALUES (?,?,?,?,?,?,?)''', (run['cut_id'], run['id'], run['selection_key'], action,
                    json.dumps(list(before)), json.dumps(list(after)), json.dumps(details)))


def archive_selection(conn, cut_id, current_key):
    """Keep the user's old boxes and slot decisions when replacing the frame selection."""
    for run in conn.execute('SELECT id,cut_id,selection_key FROM target_detection_runs WHERE cut_id=? AND selection_key!=?', (cut_id, current_key)):
        boxes = [box_row(row) for row in conn.execute("SELECT * FROM target_detections WHERE run_id=? AND source='manual'", (run['id'],))]
        slots = [dict(row) for row in conn.execute("SELECT * FROM target_slots WHERE run_id=? AND (label_source='manual' OR ignored=1)", (run['id'],))]
        if boxes or slots:
            record(conn, run, 'selection_replaced', boxes, slots=slots)


def decisions(conn, run_id):
    """Per-frame removal tombstones and dismissed candidate crops for a re-detection."""
    removed, rejected = defaultdict(list), defaultdict(list)
    for row in conn.execute("SELECT action,before_json,after_json FROM target_corrections WHERE run_id=? AND action IN ('remove','merge','reject') ORDER BY id", (run_id,)):
        kept = {item['id'] for item in json.loads(row['after_json'])} if row['action'] == 'merge' else set()
        for item in json.loads(row['before_json']):
            if row['action'] != 'reject' and item['id'] not in kept:
                removed[item['frame_ordinal']].append(item['box'])
            elif row['action'] == 'reject':
                rejected[item['frame_ordinal']].append((item['tag'], item['box']))
    return removed, rejected


def update_frames(conn, run, ordinals, *, reviewed=False, candidates=None):
    frames = json.loads(run['frames_json'])
    for frame in frames:
        if frame['ordinal'] in ordinals and not frame.get('mirror'):
            review = frame.setdefault('review', {'score': None, 'flags': {}})
            review['reviewed'] = reviewed
    if candidates is None:
        conn.execute('UPDATE target_detection_runs SET frames_json=? WHERE id=?', (json.dumps(frames), run['id']))
    else:
        conn.execute('UPDATE target_detection_runs SET frames_json=?,candidates_json=? WHERE id=?',
                     (json.dumps(frames), json.dumps(candidates), run['id']))


def visible_reviews(frames, slots, detections, candidates):
    """Reflect manual changes in flags that depend on the currently visible boxes."""
    by_frame, unassigned = defaultdict(list), defaultdict(list)
    names = {slot['slot']: slot['character_context_id'] for slot in slots if slot['character_context_id']}
    for box in detections:
        by_frame[box['frame_ordinal']].append(box)
    for item in candidates:
        unassigned[item['frame_ordinal']].append(item)
    first_count = len(by_frame[frames[0]['ordinal']]) if frames else 0
    for frame in frames:
        review = frame.get('review')
        if not review or frame.get('mirror'):
            continue
        flags = review['flags']
        here = by_frame[frame['ordinal']]
        auto = {box['slot'] for box in here if box['source'] != 'manual'}
        for key in ('low_match', 'jump'):
            if key in flags:
                original = flags.pop(key)
                if affected := [slot for slot in original if slot in auto]:
                    flags[key] = affected
        for key in ('count_changed', 'unmatched', 'suspected_merge', 'duplicate_character'):
            flags.pop(key, None)
        if len(here) != first_count:
            flags['count_changed'] = {'first': first_count, 'here': len(here)}
        for tag, flag in (('unmatched', 'unmatched'), ('merge', 'suspected_merge')):
            if ids := [item['id'] for item in unassigned[frame['ordinal']] if item['tag'] == tag]:
                flags[flag] = ids
        present = {box['slot']: names[box['slot']] for box in here if box['slot'] in names}
        counts = Counter(present.values())
        duplicates = sorted(slot for slot, name in present.items() if counts[name] > 1)
        if duplicates:
            flags['duplicate_character'] = duplicates
        review['score'] = round(max(0., 1. - sum(PENALTIES[key] for key in flags)), 3)
    return frames
