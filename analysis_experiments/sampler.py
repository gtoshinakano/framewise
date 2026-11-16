"""Pick the experiment's cuts from material that is already analysed (read-only).

A cut qualifies when it has a current full first-frame result and a current ``direction-v2`` video
analysis, its character boxes still belong to its boundaries, the boxed characters are the ones the
stored first frame describes, and its original video exists. The 20 picks fill the strata in
``STRATA`` deterministically (hash order) and spread over episodes. Nothing is written to the database.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

LIMIT_LONG = 12.0  # longer cuts need many video windows; the experiment stays affordable


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def connect_readonly(db_path):
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _json(value, default=None):
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


CURRENT = "SELECT {col} FROM analysis_results a WHERE a.cut_id=k.id AND a.superseded_by IS NULL AND a.kind='{kind}' AND json_extract(a.settings_json,'$.structured_type')='{structured}' ORDER BY a.rowid DESC LIMIT 1"
CANDIDATES = f'''SELECT k.id cut_id, k.cut_index, k.start_time, k.end_time, k.first_frame_time, k.last_frame_time, k.no_characters,
    e.id episode_id, e.title episode, e.project_id, e.original_ext, e.width, e.height,
    ({CURRENT.format(col='a.content', kind='first_frame', structured='initial_frame')}) ff,
    ({CURRENT.format(col='a.settings_json', kind='first_frame', structured='initial_frame')}) ff_settings,
    ({CURRENT.format(col='a.context_json', kind='first_frame', structured='initial_frame')}) ff_context,
    ({CURRENT.format(col='a.content', kind='cut_description', structured='video_analysis')}) va,
    ({CURRENT.format(col='a.settings_json', kind='cut_description', structured='video_analysis')}) va_settings
  FROM cuts k JOIN episodes e ON e.id=k.episode_id WHERE k.active=1 AND k.analysis_enabled=1'''


def marked_people(conn, cut):
    """The boxed, named characters of the first frame as the app builds them: left to right, handles B1..., 0-1000 grid.

    Returns None when the stored boxes no longer match the cut (a stale run), which makes the cut unusable.
    """
    if cut['no_characters']:
        return []
    run = conn.execute('SELECT * FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut['cut_id'],)).fetchone()
    if not run:
        return []
    if run['cut_start'] != cut['start_time'] or run['cut_end'] != cut['end_time']:
        return None
    frames = _json(run['frames_json'], [])
    size = next((frame.get('image_size') for frame in frames if frame.get('position') == 'first'), None)
    rows = conn.execute('''SELECT d.box_json, d.character_context_id, c.title FROM character_detections d JOIN context_items c ON c.id=d.character_context_id
                           WHERE d.run_id=? AND d.position='first' ORDER BY d.rowid''', (run['id'],)).fetchall()
    people = []
    for row in sorted(rows, key=lambda r: (_json(r['box_json'], [0]) or [0])[0]):
        box = _json(row['box_json'])
        grid = None
        if isinstance(box, list) and len(box) == 4 and isinstance(size, list) and len(size) >= 2 and size[0] and size[1]:
            grid = [round(box[0] * 1000 / size[0]), round(box[1] * 1000 / size[1]), round(box[2] * 1000 / size[0]), round(box[3] * 1000 / size[1])]
        people.append({'name': row['title'], 'character_context_id': row['character_context_id'], 'box': grid})
    for number, person in enumerate(people, 1):
        person['handle'] = f'B{number}'
    return people


def _transcript(conn, cut):
    rows = conn.execute('''SELECT t.start_time, t.end_time, t.text, c.title AS character_name FROM transcript_segments t
                           LEFT JOIN context_items c ON c.id=t.character_context_id
                           WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>? ORDER BY t.start_time''',
                        (cut['episode_id'], cut['end_time'], cut['start_time'])).fetchall()
    return [{'start_time': r['start_time'], 'end_time': r['end_time'], 'text': _text(r['text'])[:60], 'character_name': r['character_name']} for r in rows]


def load_candidates(conn, data_dir):
    """Every qualifying cut as a plain dict (see the module docstring); ``data_dir`` holds ``projects/``."""
    found = []
    for row in conn.execute(CANDIDATES):
        if not row['ff'] or not row['va']:
            continue
        ff, va = _json(row['ff']), _json(row['va'])
        if not isinstance(ff, dict) or ff.get('mode') != 'full' or not isinstance(va, dict) or va.get('contract') != 'direction-v2':
            continue
        cut = dict(row)
        people = marked_people(conn, cut)
        stored_names = sorted(_text(item.get('character')) for item in ff.get('visible_characters', []) if isinstance(item, dict))
        if people is None or sorted(person['name'] for person in people) != stored_names:
            continue
        episode_dir = Path(data_dir) / 'projects' / cut['project_id'] / 'episodes' / cut['episode_id']
        original = episode_dir / f"original{cut['original_ext']}"
        if not original.is_file():
            continue
        direction = va.get('direction') if isinstance(va.get('direction'), dict) else {}
        video_people = [item for item in direction.get('characters', []) if isinstance(item, dict)]
        background = _text(ff.get('background_people')).lower().rstrip('.')
        composition = ' '.join([_text(ff.get('composition')), *[_text(i.get('position')) for i in ff.get('visible_characters', []) if isinstance(i, dict)]]).lower()
        context = _json(row['ff_context'], {}) or {}
        hints = context.get('model_context_hints') or {}
        found.append({
            'cut_id': cut['cut_id'], 'episode_id': cut['episode_id'], 'project_id': cut['project_id'], 'episode': cut['episode'], 'cut_index': cut['cut_index'],
            'start': cut['start_time'], 'end': cut['end_time'], 'duration': cut['end_time'] - cut['start_time'],
            'first_frame_time': cut['first_frame_time'], 'last_frame_time': cut['last_frame_time'],
            'width': cut['width'], 'height': cut['height'], 'original': str(original), 'proxy': str(episode_dir / 'proxy.mp4'),
            'marked': people, 'n_marked': len(people), 'n_video': len(video_people), 'background': bool(background) and background != 'none',
            'composition_text': composition, 'initial': ff, 'video_direction': direction,
            'hints': {'selected_context': [{'kind': item.get('kind'), 'title': item.get('title'), 'description': _text(item.get('description'))[:300]}
                                           for item in hints.get('selected_context') or [] if isinstance(item, dict)],
                      'additional_context': _text(hints.get('additional_context'))[:1000]},
            'subtitle': context.get('subtitle_strip'),
            'baseline': _baseline(row['ff_settings'], row['va_settings']),
            'transcript': _transcript(conn, cut),
        })
    return found


def _baseline(frame_settings, video_settings):
    frame, video = _json(frame_settings, {}) or {}, _json(video_settings, {}) or {}
    return {'frame_seconds': frame.get('generation_seconds'), 'video_seconds': video.get('generation_seconds'),
            'frame_tokens': (frame.get('usage') or {}).get('completion_tokens'), 'video_tokens': (video.get('usage') or {}).get('completion_tokens')}


def _has(words):
    return lambda c: any(word in c['composition_text'] for word in words)


def _enters_later(c):
    return c['n_marked'] == 0 and c['n_video'] >= 1


def _nobody(c):
    return c['n_marked'] == 0 and c['n_video'] == 0 and not c['background']


# (label, how many, qualifies, preferred among the qualifying)
STRATA = [
    ('Short (2 s or less), one character', 2, lambda c: c['duration'] <= 2 and c['n_marked'] == 1, None),
    ('Short (2 s or less), two characters', 2, lambda c: c['duration'] <= 2 and c['n_marked'] == 2, None),
    ('Mid (2-8 s), one character, close-up', 1, lambda c: 2 < c['duration'] <= 8 and c['n_marked'] == 1, _has(('close-up', 'close up', 'closeup'))),
    ('Mid (2-8 s), one character, wide shot', 1, lambda c: 2 < c['duration'] <= 8 and c['n_marked'] == 1, _has(('wide', 'full-body', 'full body', 'long shot'))),
    ('Mid (2-8 s), two characters facing each other', 1, lambda c: 2 < c['duration'] <= 8 and c['n_marked'] == 2, _has(('facing each other', 'face to face', 'face each other'))),
    ('Mid (2-8 s), two characters, one seen from behind', 1, lambda c: 2 < c['duration'] <= 8 and c['n_marked'] == 2, _has(('from behind', 'facing away', 'back to', 'from the back', 'back of'))),
    ('Long (8-12 s), one character', 2, lambda c: 8 < c['duration'] <= LIMIT_LONG and c['n_marked'] == 1, None),
    ('Long (8-12 s), two characters', 2, lambda c: 8 < c['duration'] <= LIMIT_LONG and c['n_marked'] == 2, None),
    ('Three or more characters, short', 1, lambda c: c['n_marked'] >= 3 and c['duration'] <= 5, None),
    ('Three or more characters, long', 1, lambda c: c['n_marked'] >= 3 and 5 < c['duration'] <= LIMIT_LONG, None),
    ('Nobody marked in the first frame, characters enter later (short)', 1, lambda c: _enters_later(c) and c['duration'] <= 3, None),
    ('Nobody marked in the first frame, characters enter later (mid)', 1, lambda c: _enters_later(c) and 3 < c['duration'] <= 8, None),
    ('Nobody marked in the first frame, characters enter later (long)', 1, lambda c: _enters_later(c) and 8 < c['duration'] <= LIMIT_LONG, None),
    ('Marked characters plus background people', 2, lambda c: c['n_marked'] in (1, 2) and c['background'] and c['duration'] <= 8, None),
    ('No people at all (control)', 1, lambda c: _nobody(c) and c['duration'] <= 8, None),
]


def _rank(candidate):
    return hashlib.sha1(candidate['cut_id'].encode()).hexdigest()


def pick(candidates, strata=STRATA):
    """Fill the strata in order: preferred cuts first, then the least used episode, then hash order. Deterministic."""
    used, episodes, picks = set(), {}, []
    for label, count, qualifies, preferred in strata:
        for _ in range(count):
            pool = [c for c in candidates if c['cut_id'] not in used and qualifies(c)]
            if not pool:
                picks.append({'label': label, 'candidate': None})
                continue
            choice = min(pool, key=lambda c: (0 if preferred and preferred(c) else 1, episodes.get(c['episode_id'], 0), _rank(c)))
            used.add(choice['cut_id'])
            episodes[choice['episode_id']] = episodes.get(choice['episode_id'], 0) + 1
            picks.append({'label': label, 'candidate': choice})
    return picks


def sample_record(index, label, c):
    """The samples.json entry of one pick (everything later stages and the report need, no database access)."""
    initial = c['initial']
    direction = c['video_direction']
    keep = ('character', 'position', 'pose', 'expression', 'emotion', 'gaze', 'action')
    return {
        'id': f's{index:02d}', 'reason': label, 'episode': c['episode'], 'cut_index': c['cut_index'] + 1, 'cut_id': c['cut_id'],
        'start': c['start'], 'end': c['end'], 'duration': round(c['duration'], 3), 'first_frame_time': c['first_frame_time'], 'last_frame_time': c['last_frame_time'],
        'width': c['width'], 'height': c['height'], 'original': c['original'], 'proxy': c['proxy'],
        'marked': [{k: p[k] for k in ('handle', 'name', 'character_context_id', 'box')} for p in c['marked']],
        'background': c['background'], 'hints': c['hints'], 'subtitle': c['subtitle'], 'transcript': c['transcript'],
        'initial': initial,
        'stored': {'first_frame': {'composition': initial.get('composition'), 'background_people': initial.get('background_people'),
                                   'characters': [{k: item.get(k) for k in keep} for item in initial.get('visible_characters', []) if isinstance(item, dict)]},
                   'video': {'summary': direction.get('summary'), 'camera': direction.get('camera'),
                             'characters': [{k: item.get(k) for k in ('character', 'activity', 'travel', 'motion', 'speaking', 'emotion', 'end')}
                                            for item in direction.get('characters', []) if isinstance(item, dict)]},
                   'baseline': c['baseline']},
    }


def build_samples(db_path, data_dir, strata=STRATA):
    """Open the database read-only and return (samples, report lines about empty strata)."""
    conn = connect_readonly(db_path)
    try:
        candidates = load_candidates(conn, data_dir)
    finally:
        conn.close()
    picks = pick(candidates, strata)
    samples, missing = [], []
    for picked in picks:
        if picked['candidate'] is None:
            missing.append(picked['label'])
        else:
            samples.append(sample_record(len(samples) + 1, picked['label'], picked['candidate']))
    return samples, missing, len(candidates)
