"""Scoring of the pose experiment. Pure functions: no database, no model, no files.

There is no hand-labelled truth for head yaw or gaze, so quality is measured with checks that need
none (every check is a ``pass`` or ``fail`` record):

* mirror    the same frame flipped left-right must give the mirrored answer;
* geometry  the answer must fit the character boxes and itself (who looks at whom, camera gaze, ...);
* agreement the answer must not contradict what the stored first-frame / video analysis said.

``score_all`` turns every recorded call into per-approach scores (see ``WEIGHTS``).
"""
import re
from collections import Counter, defaultdict

import pose_schema as ps

WEIGHTS = {'reliability': 0.30, 'quality': 0.40, 'coverage': 0.15, 'efficiency': 0.15}
STAGE_WEIGHTS = {'frame': 0.5, 'video': 0.5}
ANGLE_TOLERANCE = 25  # degrees: a flipped frame or a restated angle may differ this much
SIDE_MARGIN = 60  # box centres closer than this on the 0-1000 grid say nothing about a side

# The pose fields each check speaks about, for the per-field table.
CHECK_FIELDS = {
    'mirror': None,  # the field itself
    'gaze_side': ('gaze_target', 'eye_gaze', 'head_yaw'),
    'camera_gaze': ('gaze_target', 'eye_gaze', 'head_yaw'),
    'shot_size_box': ('shot_size',),
    'elevation_shared': ('camera_elevation',),
    'pitch_gaze': ('head_pitch', 'eye_gaze'),
    'head_body': ('head_yaw', 'body_yaw'),
    'still_stable': ('head_yaw',),
    'body_pose': ('body_pose',),
    'gaze_target': ('gaze_target',),
    'facial_expression': ('facial_expression',),
    'eye_openness': ('eye_openness',),
    'mouth_state': ('mouth_state',),
    'speaking': ('mouth_state',),
}
KIND_OF = {'mirror': 'mirror', 'gaze_side': 'geometry', 'camera_gaze': 'geometry', 'shot_size_box': 'geometry', 'elevation_shared': 'geometry',
           'pitch_gaze': 'geometry', 'head_body': 'geometry', 'still_stable': 'geometry'}


def kind_of(name):
    return KIND_OF.get(name, 'agreement')


def _check(name, who, status, detail, entry=None, fields=None, window=None, moment=None):
    fields = fields or CHECK_FIELDS.get(name) or ()
    confidence = None
    if isinstance(entry, dict) and isinstance(entry.get('confidence'), dict):
        values = [entry['confidence'][f] for f in fields if isinstance(entry['confidence'].get(f), (int, float))]
        confidence = min(values) if values else None
    record = {'kind': kind_of(name), 'name': name, 'character': who, 'fields': list(fields), 'status': status, 'detail': detail, 'conf': confidence}
    if window is not None:
        record['window'] = window
    if moment:
        record['moment'] = moment
    return record


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _centre(box):
    return (box[0] + box[2]) / 2


# -- mirror -------------------------------------------------------------------------------------------------------

def _consistent(field, expected, actual):
    if field in ps.ANGLES:
        return _number(expected) and _number(actual) and ps.circular_difference(expected, actual) <= ANGLE_TOLERANCE
    if field == 'shot_size' and expected in ps.SHOT_SIZES and actual in ps.SHOT_SIZES:
        return abs(ps.SHOT_SIZES.index(expected) - ps.SHOT_SIZES.index(actual)) <= 1
    return expected == actual


def mirror_checks(normal, mirror):
    """Compare the flipped frame's answer with the mirrored original answer, field by field."""
    if not (normal and mirror and normal.get('ok') and mirror.get('ok')):
        return []
    forward = {original: flipped for flipped, original in (mirror.get('mirror_map') or {}).items()}
    actual = {entry.get('character_id'): entry for entry in mirror.get('answer') or [] if isinstance(entry, dict)}
    checks = []
    for entry in normal.get('answer') or []:
        flipped = actual.get(forward.get(entry.get('character_id'), entry.get('character_id')))
        if not flipped:
            continue
        expected = ps.mirror_entry(entry, forward)
        for field in ps.FIELD_ORDER:
            if field not in expected or field not in flipped or not ps.concrete(expected[field]) or not ps.concrete(flipped[field]):
                continue
            ok = _consistent(field, expected[field], flipped[field])
            checks.append(_check('mirror', entry.get('character_id'), 'pass' if ok else 'fail',
                                 f'{field}: expected {expected[field]!r} on the flipped frame, got {flipped[field]!r}', entry, (field,)))
    return checks


# -- geometry ---------------------------------------------------------------------------------------------------------

def _yaw_side(yaw):
    if not _number(yaw) or abs(yaw) < 20 or abs(yaw) > 160:
        return None
    return 'right' if yaw > 0 else 'left'


def _gaze_side(gaze):
    return 'right' if gaze in ('right', 'up_right', 'down_right') else 'left' if gaze in ('left', 'up_left', 'down_left') else None


def pose_checks(who, pose, window=None, moment=None):
    """Checks that need no boxes: camera gaze, pitch against vertical gaze, head against body."""
    checks = []
    yaw, gaze, pitch, body = pose.get('head_yaw'), pose.get('eye_gaze'), pose.get('head_pitch'), pose.get('body_yaw')
    if pose.get('gaze_target') == 'camera':
        parts = []
        if _number(yaw):
            parts.append(abs(yaw) <= 45)
        if ps.concrete(gaze):
            parts.append(gaze in ('center', 'up', 'down'))
        if parts:
            checks.append(_check('camera_gaze', who, 'pass' if all(parts) else 'fail',
                                 f'looks at the camera with head_yaw {yaw!r} and eye_gaze {gaze!r}', pose, window=window, moment=moment))
    if _number(pitch) and abs(pitch) >= 25 and ps.concrete(gaze):
        wrong = (pitch < 0 and gaze in ('up', 'up_left', 'up_right')) or (pitch > 0 and gaze in ('down', 'down_left', 'down_right'))
        checks.append(_check('pitch_gaze', who, 'fail' if wrong else 'pass', f'head_pitch {pitch:g} with eye_gaze {gaze}', pose, window=window, moment=moment))
    if _number(yaw) and _number(body):
        gap = ps.circular_difference(yaw, body)
        checks.append(_check('head_body', who, 'pass' if gap <= 100 else 'fail', f'head_yaw {yaw:g} against body_yaw {body:g} ({gap:g} degrees apart)',
                             pose, window=window, moment=moment))
    return checks


def geometry_checks(sample, answer):
    """Frame checks that use the marked characters' boxes (0-1000 grid)."""
    boxes = {person['handle']: person['box'] for person in sample.get('marked', []) if person.get('box')}
    entries = [entry for entry in answer or [] if isinstance(entry, dict)]
    checks = []
    for entry in entries:
        who = entry.get('character_id')
        target = entry.get('gaze_target')
        if target in boxes and target != who and who in boxes:
            shift = _centre(boxes[target]) - _centre(boxes[who])
            signals = [side for side in (_yaw_side(entry.get('head_yaw')), _gaze_side(entry.get('eye_gaze'))) if side]
            known = _number(entry.get('head_yaw')) or ps.concrete(entry.get('eye_gaze'))
            if abs(shift) >= SIDE_MARGIN and known:
                expected = 'right' if shift > 0 else 'left'
                good = expected in signals and all(side == expected for side in signals)
                checks.append(_check('gaze_side', who, 'pass' if good else 'fail',
                                     f'{target} is to the {expected} of {who}; head_yaw {entry.get("head_yaw")!r}, eye_gaze {entry.get("eye_gaze")!r}', entry))
        checks.extend(pose_checks(who, entry))
        box, size = boxes.get(who), entry.get('shot_size')
        if box and size in ps.SHOT_SIZES:
            share = (box[3] - box[1]) / 1000
            if share <= 0.30 or share >= 0.85:
                allowed = ('medium_long', 'long', 'extreme_long') if share <= 0.30 else ('extreme_close_up', 'close_up', 'medium_close_up', 'medium', 'medium_long')
                checks.append(_check('shot_size_box', who, 'pass' if size in allowed else 'fail', f'box height is {share:.0%} of the frame, shot_size {size}', entry))
    levels = [entry.get('camera_elevation') for entry in entries if ps.concrete(entry.get('camera_elevation'))]
    if len(levels) >= 2:
        same = len(set(levels)) == 1
        for entry in entries:
            if ps.concrete(entry.get('camera_elevation')):
                checks.append(_check('elevation_shared', entry.get('character_id'), 'pass' if same else 'fail', f'camera_elevation values in this frame: {", ".join(levels)}', entry))
    return checks


# -- agreement with the stored analysis ---------------------------------------------------------------------------------

POSE_WORDS = {'standing': 'standing', 'stands': 'standing', 'stood': 'standing', 'sitting': 'sitting', 'seated': 'sitting', 'sits': 'sitting',
              'walking': 'walking', 'walks': 'walking', 'running': 'running', 'runs': 'running', 'lying': 'lying', 'laying': 'lying', 'lies': 'lying'}
OTHER_WORDS = {'crouching', 'kneeling', 'jumping', 'leaning', 'squatting', 'bending', 'hanging', 'climbing', 'dancing', 'bowing', 'stooping', 'crawling'}
FAMILIES = {
    'happy': ('smil', 'happy', 'joy', 'cheer', 'grin', 'delight', 'laugh', 'pleased', 'content'),
    'sad': ('sad', 'tear', 'cry', 'sorrow', 'melanchol', 'grief', 'gloom', 'dejected'),
    'angry': ('angry', 'anger', 'furious', 'irritat', 'annoy', 'glare', 'scowl', 'rage', 'hostile'),
    'worried': ('worr', 'anxious', 'anxiety', 'concern', 'nervous', 'uneasy', 'troubled', 'tense'),
    'surprised': ('surpris', 'shock', 'startl', 'astonish', 'amaz', 'stunned'),
    'embarrassed': ('blush', 'embarrass', 'shy', 'flustered', 'bashful', 'sheepish'),
    'afraid': ('fear', 'afraid', 'scared', 'terrif', 'frighten', 'panic'),
    'disgusted': ('disgust', 'revuls'),
    'smug': ('smug', 'smirk', 'sly', 'cocky'),
    'determined': ('determin', 'resolute', 'focus', 'serious', 'firm'),
    'neutral': ('neutral', 'calm', 'blank', 'composed', 'expressionless', 'relaxed', 'stoic', 'placid'),
}


def pose_word(text):
    """Map the first activity word of a stored text to the body_pose vocabulary, or None when it says nothing usable."""
    words = re.findall(r'[a-z]+', str(text or '').lower())
    if not words:
        return None
    return POSE_WORDS.get(words[0]) or ('other' if words[0] in OTHER_WORDS else None)


def expression_families(*texts):
    text = ' '.join(str(item or '') for item in texts).lower()
    if not text.strip() or 'not readable' in text and len(text) < 40:
        return set()
    return {family for family, stems in FAMILIES.items() if any(stem in text for stem in stems)}


def eyes_expected(text):
    text = str(text or '').lower()
    if re.search(r'eyes? (?:are |is )?(?:closed|shut)|closed eyes|eyes shut|eyelids? closed', text):
        return {'closed', 'half_closed'}
    if re.search(r'half[- ]closed|heavy[- ]lidded|droopy|drooping', text):
        return {'half_closed', 'closed'}
    if re.search(r'wide[- ]eyed|eyes? (?:are )?(?:wide|widened)|wide eyes', text):
        return {'wide'}
    return None


def mouth_expected(text):
    text = str(text or '').lower()
    opened = re.search(r'mouth (?:is |slightly |partly |wide )*(?:open|agape)|open mouth|parted lips|lips (?:are )?parted|mouth wide', text)
    closed = re.search(r'mouth (?:is )?(?:closed|shut)|closed mouth|closed lips|lips (?:are )?(?:closed|pressed|together)|pressed lips', text)
    if opened and not closed:
        return {'slightly_open', 'open', 'wide_open'}
    if closed and not opened:
        return {'closed'}
    return None


def stored_target(text, own_name, others):
    """The handle (or 'camera') a stored gaze text names, or None when it names nobody."""
    text = str(text or '').lower()
    if not text or 'not visible' in text:
        return None
    if re.search(r'\b(camera|viewer|lens)\b', text):
        return 'camera'
    found = set()
    for handle, name in others.items():
        if name == own_name:
            continue
        tokens = [name.lower(), *[token for token in re.findall(r'\w+', name.lower()) if len(token) >= 3]]
        if any(re.search(r'(?<!\w)' + re.escape(token) + r'(?!\w)', text) for token in tokens):
            found.add(handle)
    return found.pop() if len(found) == 1 else None


def frame_agreement(sample, answer):
    names = {person['handle']: person['name'] for person in sample.get('marked', [])}
    stored = {item.get('character'): item for item in (sample.get('stored', {}).get('first_frame') or {}).get('characters', []) if isinstance(item, dict)}
    checks = []
    for entry in answer or []:
        who = entry.get('character_id') if isinstance(entry, dict) else None
        old = stored.get(names.get(who))
        if not old:
            continue
        expected = pose_word(old.get('action'))
        if expected and ps.concrete(entry.get('body_pose')):
            checks.append(_check('body_pose', who, 'pass' if entry['body_pose'] == expected else 'fail', f'stored action "{old.get("action", "")[:50]}" says {expected}, new {entry["body_pose"]}', entry))
        target = stored_target(old.get('gaze'), names.get(who), names)
        if target and ps.concrete(entry.get('gaze_target')):
            checks.append(_check('gaze_target', who, 'pass' if entry['gaze_target'] == target else 'fail', f'stored gaze "{old.get("gaze", "")[:60]}" means {target}, new {entry["gaze_target"]}', entry))
        families = expression_families(old.get('emotion'), old.get('expression'))
        if families and ps.concrete(entry.get('facial_expression')) and entry['facial_expression'] != 'other':
            checks.append(_check('facial_expression', who, 'pass' if entry['facial_expression'] in families else 'fail',
                                 f'stored emotion "{old.get("emotion", "")[:40]}" reads as {", ".join(sorted(families))}, new {entry["facial_expression"]}', entry))
        eyes = eyes_expected(old.get('expression'))
        if eyes and ps.concrete(entry.get('eye_openness')):
            checks.append(_check('eye_openness', who, 'pass' if entry['eye_openness'] in eyes else 'fail', f'stored expression suggests {", ".join(sorted(eyes))}, new {entry["eye_openness"]}', entry))
        mouth = mouth_expected(old.get('expression'))
        if mouth and ps.concrete(entry.get('mouth_state')):
            checks.append(_check('mouth_state', who, 'pass' if entry['mouth_state'] in mouth else 'fail', f'stored expression suggests {", ".join(sorted(mouth))}, new {entry["mouth_state"]}', entry))
    return checks


def _stored_match(stored, name):
    """The stored video character a model-written name means: exact (case-insensitive), or the only one sharing all of its words."""
    key = str(name).casefold()
    if key in stored:
        return stored[key]
    words = set(re.findall(r'\w+', key))
    found = [item for stored_name, item in stored.items() if words and words <= set(re.findall(r'\w+', stored_name))]
    return found[0] if len(found) == 1 else None


def video_checks(sample, windows):
    """Box-free pose checks plus agreement with the stored video direction, for every snapshot of every window."""
    stored = {str(item.get('character', '')).casefold(): item for item in (sample.get('stored', {}).get('video') or {}).get('characters', []) if isinstance(item, dict)}
    checks = []
    for window in windows or []:
        if not window.get('ok'):
            continue
        for entry in window.get('answer') or []:
            who = entry.get('character')
            old = _stored_match(stored, who)
            snapshots = {moment: entry.get(moment) for moment in ('start', 'end') if isinstance(entry.get(moment), dict)}
            for moment, pose in snapshots.items():
                checks.extend(pose_checks(who, pose, window.get('index'), moment))
            if not old:
                continue
            parts = [part.strip() for part in str(old.get('activity', '')).split('→') if part.strip()]
            for moment, part in (('start', parts[0] if parts else None), ('end', parts[-1] if parts else None)):
                expected, pose = pose_word(part), snapshots.get(moment)
                if expected and pose and ps.concrete(pose.get('body_pose')):
                    checks.append(_check('body_pose', who, 'pass' if pose['body_pose'] == expected else 'fail', f'stored activity "{part}" says {expected}, new {pose["body_pose"]} at the {moment}',
                                         pose, window=window.get('index'), moment=moment))
            mouths = [pose.get('mouth_state') for pose in snapshots.values() if ps.concrete(pose.get('mouth_state'))]
            speaking = str(old.get('speaking', '')).lower()
            if mouths and speaking in ('yes', 'no'):
                ok = any(m != 'closed' for m in mouths) if speaking == 'yes' else all(m == 'closed' for m in mouths)
                checks.append(_check('speaking', who, 'pass' if ok else 'fail', f'stored speaking={speaking}, mouth states {", ".join(mouths)}', next(iter(snapshots.values())),
                                     window=window.get('index')))
            yaws = [pose.get('head_yaw') for pose in (snapshots.get('start'), snapshots.get('end')) if pose]
            if str(old.get('motion', '')).strip().lower().rstrip('.') == 'holds still' and len(yaws) == 2 and all(_number(y) for y in yaws):
                gap = ps.circular_difference(*yaws)
                checks.append(_check('still_stable', who, 'pass' if gap <= ANGLE_TOLERANCE else 'fail', f'stored motion "holds still"; head_yaw {yaws[0]:g} -> {yaws[1]:g}', window=window.get('index')))
    return checks


# -- statistics ---------------------------------------------------------------------------------------------------------

def _rate(passed, total):
    return passed / total if total else None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def attach_checks(samples, calls):
    """Compute every check once and store it on its call (``call['checks']``)."""
    by_id = {sample['id']: sample for sample in samples}
    index = {(call['approach'], call['sample'], call.get('variant')): call for call in calls if call['stage'] == 'frame'}
    for call in calls:
        sample = by_id.get(call['sample'])
        if not sample:
            call['checks'] = []
        elif call['stage'] == 'frame':
            if call.get('variant') == 'mirror':
                call['checks'] = []
            else:
                answer = call.get('answer') if call.get('ok') else None
                call['checks'] = (geometry_checks(sample, answer) + frame_agreement(sample, answer) + mirror_checks(call, index.get((call['approach'], call['sample'], 'mirror')))) if answer is not None else []
        else:
            call['checks'] = video_checks(sample, call.get('windows'))
    return calls


def _field_slots(sample, call):
    """Number of (character, snapshot) slots each pose field was requested in, for one call."""
    if call['stage'] == 'frame':
        return len(sample.get('marked', []))
    return sum(max(len(window.get('answer') or []), 0) * 2 for window in call.get('windows', []) if window.get('ok')) or 0


def _poses(call):
    """Every pose object a call produced (frame entries or video snapshots)."""
    if call['stage'] == 'frame':
        return [entry for entry in call.get('answer') or [] if isinstance(entry, dict)] if call.get('ok') else []
    poses = []
    for window in call.get('windows', []):
        if window.get('ok'):
            for entry in window.get('answer') or []:
                poses.extend(entry[moment] for moment in ('start', 'end') if isinstance(entry.get(moment), dict))
    return poses


def stage_summary(samples, calls, stage):
    """Reliability, coverage, quality components, time and tokens of one approach in one stage."""
    by_id = {sample['id']: sample for sample in samples}
    main = [call for call in calls if call['stage'] == stage and call.get('variant', 'normal') == 'normal']
    probes = [call for call in calls if call['stage'] == stage and call.get('variant') == 'mirror']
    graded = main + probes
    fields = {name: Counter() for name in ps.FIELD_ORDER}
    calibration = {'pass': [], 'fail': []}
    parts = {'mirror': [0, 0], 'geometry': [0, 0], 'agreement': [0, 0]}
    per_field = {name: {'mirror': [0, 0], 'geometry': [0, 0], 'agreement': [0, 0], 'conf': []} for name in ps.FIELD_ORDER}
    expected_slots = 0
    concrete_values = 0
    for call in main:
        sample = by_id.get(call['sample'], {})
        slots = _field_slots(sample, call) if call.get('ok') else (len(sample.get('marked', [])) if stage == 'frame' else max(1, len(sample.get('marked', []))) * 2)
        expected_slots += slots * len(ps.FIELD_ORDER)
        for pose in _poses(call):
            for name in ps.FIELD_ORDER:
                if name not in pose:
                    fields[name]['omitted'] += 1
                elif ps.concrete(pose[name]):
                    fields[name]['concrete'] += 1
                    concrete_values += 1
                else:
                    fields[name]['abstained'] += 1
                if name in pose:
                    fields[name]['chosen'] += 1
                    if isinstance(pose.get('confidence'), dict) and isinstance(pose['confidence'].get(name), (int, float)):
                        per_field[name]['conf'].append(pose['confidence'][name])
        for check in call.get('checks', []):
            ok = check['status'] == 'pass'
            parts[check['kind']][0] += ok
            parts[check['kind']][1] += 1
            if check.get('conf') is not None:
                calibration['pass' if ok else 'fail'].append(check['conf'])
            for name in check['fields']:
                if name in per_field:
                    per_field[name][check['kind']][0] += ok
                    per_field[name][check['kind']][1] += 1
    pose_count = sum(len(_poses(call)) for call in main)
    seconds = sum(call.get('seconds', 0) for call in main)
    attempts = [attempt for call in graded for attempt in _attempts(call)]
    failures = sum(1 for attempt in attempts if attempt.get('errors'))
    rel_first = _rate(sum(bool(call.get('first_try_ok')) for call in graded), len(graded))
    rel_final = _rate(sum(bool(call.get('ok')) for call in graded), len(graded))
    components = {name: _rate(*pair) for name, pair in parts.items()}
    available = [value for value in components.values() if value is not None]
    quality = sum(available) / len(available) if available else None
    return {
        'stage': stage, 'calls': len(graded), 'main_calls': len(main), 'ok_calls': sum(bool(call.get('ok')) for call in graded),
        'first_try_rate': rel_first, 'final_rate': rel_final,
        'reliability': None if rel_first is None else 0.5 * rel_first + 0.5 * rel_final,
        'coverage': _rate(concrete_values, expected_slots), 'quality': quality, 'components': components,
        'component_counts': {name: {'pass': pair[0], 'total': pair[1]} for name, pair in parts.items()},
        'seconds': seconds, 'probe_seconds': sum(call.get('seconds', 0) for call in probes),
        'attempts': len(attempts), 'validator_failures': failures, 'retries': sum(max(0, len(_attempts(call)) - 1) for call in graded),
        'hard_failures': sum(not call.get('ok') for call in graded),
        'prompt_tokens': sum(a.get('prompt_tokens') or 0 for a in attempts), 'completion_tokens': sum(a.get('completion_tokens') or 0 for a in attempts),
        'fields': {name: {**dict(fields[name]), 'coverage': _rate(fields[name]['concrete'], pose_count),
                          'mirror': _rate(*per_field[name]['mirror']), 'geometry': _rate(*per_field[name]['geometry']), 'agreement': _rate(*per_field[name]['agreement']),
                          'checks': {kind: per_field[name][kind][1] for kind in ('mirror', 'geometry', 'agreement')}, 'mean_conf': _mean(per_field[name]['conf'])}
                   for name in ps.FIELD_ORDER},
        'calibration': {'n_pass': len(calibration['pass']), 'n_fail': len(calibration['fail']), 'mean_pass': _mean(calibration['pass']), 'mean_fail': _mean(calibration['fail']),
                        'gap': None if not calibration['pass'] or not calibration['fail'] else _mean(calibration['pass']) - _mean(calibration['fail'])},
    }


def _attempts(call):
    if call['stage'] == 'frame':
        return call.get('attempts') or []
    return [attempt for window in call.get('windows', []) for attempt in window.get('attempts') or []]


def score_stage(summary, fastest_seconds):
    """The 0-100 stage score from a ``stage_summary`` and the fastest approach's seconds in that stage."""
    if not summary['calls']:
        return None
    efficiency = (fastest_seconds / summary['seconds']) if summary['seconds'] and fastest_seconds else 0.0
    parts = {'reliability': summary['reliability'] or 0.0, 'quality': summary['quality'] or 0.0, 'coverage': summary['coverage'] or 0.0, 'efficiency': min(1.0, efficiency)}
    return {'score': 100 * sum(WEIGHTS[name] * value for name, value in parts.items()), 'parts': parts}


def a2_choices(calls):
    """How often each field was among the five a top-5 run chose."""
    counts = Counter()
    for call in calls:
        if call.get('variant', 'normal') == 'normal':
            for pose in _poses(call):
                counts.update(name for name in ps.FIELD_ORDER if name in pose)
    return dict(counts)


def agreement_between(samples, calls_by_approach):
    """Share of (sample, character, field) values two approaches agree on (frame stage, normal variant)."""
    answers = {}
    for approach, calls in calls_by_approach.items():
        table = {}
        for call in calls:
            if call['stage'] == 'frame' and call.get('variant', 'normal') == 'normal' and call.get('ok'):
                for entry in call.get('answer') or []:
                    for name in ps.FIELD_ORDER:
                        if name in entry and ps.concrete(entry[name]):
                            table[(call['sample'], entry.get('character_id'), name)] = entry[name]
        answers[approach] = table
    matrix = defaultdict(dict)
    for first in answers:
        for second in answers:
            if first == second:
                continue
            shared = [key for key in answers[first] if key in answers[second]]
            same = sum(_consistent(key[2], answers[first][key], answers[second][key]) for key in shared)
            matrix[first][second] = {'rate': _rate(same, len(shared)), 'n': len(shared)}
    return {a: dict(b) for a, b in matrix.items()}


def score_all(samples, calls):
    """Per-approach scores and statistics from every recorded call (see the module docstring)."""
    calls = attach_checks(samples, calls)
    by_approach = {approach: [call for call in calls if call['approach'] == approach] for approach in ps.APPROACHES}
    summaries = {approach: {stage: stage_summary(samples, by_approach[approach], stage) for stage in STAGE_WEIGHTS} for approach in ps.APPROACHES}
    fastest = {}
    for stage in STAGE_WEIGHTS:
        times = [summaries[approach][stage]['seconds'] for approach in ps.APPROACHES if summaries[approach][stage]['seconds'] > 0]
        fastest[stage] = min(times) if times else 0.0
    result = {}
    for approach in ps.APPROACHES:
        stage_scores = {stage: score_stage(summaries[approach][stage], fastest[stage]) for stage in STAGE_WEIGHTS}
        live = {stage: value for stage, value in stage_scores.items() if value}
        total_weight = sum(STAGE_WEIGHTS[stage] for stage in live)
        final = sum(STAGE_WEIGHTS[stage] * value['score'] for stage, value in live.items()) / total_weight if total_weight else None
        result[approach] = {'final': final, 'stage_scores': stage_scores, 'summary': summaries[approach], 'a2_choices': a2_choices(by_approach[approach]) if approach == 'top5_confidence' else None}
    return {'approaches': result, 'agreement_between': agreement_between(samples, by_approach), 'weights': WEIGHTS, 'stage_weights': STAGE_WEIGHTS}
