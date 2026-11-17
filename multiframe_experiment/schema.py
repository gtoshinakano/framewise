"""The shared answer contract of every prompt style, its validator, and the atomic claims the judge verifies.

Stdlib only. Every style answers the same JSON so the configs are scored the same way:

    {"segments": [{"from": "F1", "to": "F2",
                   "camera": {"move": <CAMERA_MOVES>, "detail": str},
                   "characters": [{"name": str, "walking": yes|no|unclear, "speaking": ..., "blinking": ...,
                                   "changes": [{"part": <PARTS>, "description": str}]}],
                   "other_changes": [{"kind": <OTHER_KINDS>, "description": str}]}],
     "summary": str}

One segment per adjacent pair of the window's frames, in order. Times are not asked for: the app knows them
from the frame labels and writes ``start``/``end`` into the normalised answer.
"""
import hashlib
import json
import re

CAMERA_MOVES = ('static', 'pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'zoom_in', 'zoom_out', 'dolly_in', 'dolly_out',
                'truck', 'orbit', 'shake', 'unclear')
YNU = ('yes', 'no', 'unclear')
PARTS = ('body_pose', 'head', 'eyes', 'mouth', 'arms', 'hands', 'legs', 'position', 'expression')
OTHER_KINDS = ('outfit', 'color', 'lighting', 'object', 'background', 'effect')

# Categories used for precision, recall and the per-category table.
CATEGORIES = ('walking', 'speaking', 'blink', 'head', 'eyes_gaze', 'arms_hands', 'legs', 'body_pose', 'position', 'expression',
              'camera', 'outfit', 'color', 'lighting', 'object', 'background', 'effect')
CATEGORY_LABELS = {'walking': 'Walking', 'speaking': 'Speaking', 'blink': 'Blinking', 'head': 'Head angle', 'eyes_gaze': 'Eyes / gaze',
                   'arms_hands': 'Arms & hands', 'legs': 'Legs', 'body_pose': 'Body pose', 'position': 'Position in frame',
                   'expression': 'Expression', 'camera': 'Camera', 'outfit': 'Outfit', 'color': 'Colours', 'lighting': 'Lighting',
                   'object': 'Objects', 'background': 'Background', 'effect': 'Effects'}
# The user's priorities: movement, camera, speech, pose, head, blink and limbs matter; outfit, colours and lighting less.
IMPORTANCE = {'outfit': .3, 'color': .3, 'lighting': .3, 'object': .5, 'background': .5, 'effect': .5}
PART_CATEGORY = {'body_pose': 'body_pose', 'head': 'head', 'eyes': 'eyes_gaze', 'mouth': 'speaking', 'arms': 'arms_hands',
                 'hands': 'arms_hands', 'legs': 'legs', 'position': 'position', 'expression': 'expression'}
# Categories the judge's reference table marks per pair (recall is measured on these).
REFERENCE_FLAGS = ('walking', 'speaking', 'blink', 'head', 'eyes_gaze', 'arms_hands', 'legs', 'body_pose', 'position', 'expression')

ALIASES = {
    'move': {'none': 'static', 'still': 'static', 'fixed': 'static', 'locked': 'static', 'no movement': 'static', 'pan left': 'pan_left',
             'pan right': 'pan_right', 'tilt up': 'tilt_up', 'tilt down': 'tilt_down', 'zoom in': 'zoom_in', 'zoom out': 'zoom_out',
             'push in': 'dolly_in', 'pull out': 'dolly_out', 'pull back': 'dolly_out', 'dolly in': 'dolly_in', 'dolly out': 'dolly_out',
             'tracking': 'truck', 'track': 'truck', 'rotate': 'orbit', 'rotation': 'orbit', 'unknown': 'unclear'},
    'ynu': {'true': 'yes', 'false': 'no', 'y': 'yes', 'n': 'no', 'unknown': 'unclear', 'not visible': 'unclear', 'n/a': 'unclear'},
    'part': {'body': 'body_pose', 'pose': 'body_pose', 'torso': 'body_pose', 'face': 'expression', 'eye': 'eyes', 'gaze': 'eyes', 'arm': 'arms',
             'hand': 'hands', 'leg': 'legs', 'feet': 'legs', 'foot': 'legs', 'lips': 'mouth', 'location': 'position', 'neck': 'head', 'hair': 'head'},
    'kind': {'colour': 'color', 'colors': 'color', 'clothing': 'outfit', 'clothes': 'outfit', 'light': 'lighting', 'prop': 'object',
             'props': 'object', 'vfx': 'effect', 'effects': 'effect', 'scenery': 'background'},
}


# Bounds in the grammar: a small model otherwise loops (LFM3 repeated "no change" entries up to the token limit).
MAX_CHARACTERS, MAX_CHANGES, MAX_OTHER = 6, 8, 4
TEXT = {'type': 'string', 'maxLength': 240}
NO_CHANGE = re.compile(r'^\s*(no (visible |noticeable )?change|unchanged|not changed|same\b|no movement|remains? |stays? |still )', re.I)


STATIC_WORDS = re.compile(r'\b(remains?|remaining|stays?|unchanged|still|maintains?|maintaining|holds? (the same|still)|same (as|position|pose|expression))\b', re.I)


def is_no_change(text):
    """A "change" that says nothing changed is not a claim (and not a recall hit): "no change…", "arm remains extended", "outfit unchanged"."""
    return bool(NO_CHANGE.search(text or '') or STATIC_WORDS.search(text or ''))


def labels(window):
    return [frame['label'] for frame in window['frames']]


def json_schema(window):
    """A llama-server json_schema grammar for one window: the pair labels and the segment count are fixed."""
    names = labels(window)
    pairs = len(names) - 1
    character = {'type': 'object', 'properties': {
        'name': {'type': 'string'}, 'walking': {'enum': list(YNU)}, 'speaking': {'enum': list(YNU)}, 'blinking': {'enum': list(YNU)},
        'changes': {'type': 'array', 'maxItems': MAX_CHANGES, 'items': {'type': 'object', 'properties': {'part': {'enum': list(PARTS)}, 'description': TEXT},
                                                                     'required': ['part', 'description']}}},
        'required': ['name', 'walking', 'speaking', 'blinking', 'changes']}
    segment = {'type': 'object', 'properties': {
        'from': {'enum': names[:-1]}, 'to': {'enum': names[1:]},
        'camera': {'type': 'object', 'properties': {'move': {'enum': list(CAMERA_MOVES)}, 'detail': TEXT}, 'required': ['move', 'detail']},
        'characters': {'type': 'array', 'maxItems': MAX_CHARACTERS, 'items': character},
        'other_changes': {'type': 'array', 'maxItems': MAX_OTHER, 'items': {'type': 'object', 'properties': {'kind': {'enum': list(OTHER_KINDS)}, 'description': TEXT},
                                                                           'required': ['kind', 'description']}}},
        'required': ['from', 'to', 'camera', 'characters', 'other_changes']}
    state = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'state': {'type': 'string', 'maxLength': 300}}, 'required': ['name', 'state']}
    frame = {'type': 'object', 'properties': {'frame': {'enum': names}, 'characters': {'type': 'array', 'maxItems': MAX_CHARACTERS, 'items': state}},
             'required': ['frame', 'characters']}
    # "frames" comes first so the model writes every frame's state before it compares them.
    return {'type': 'object', 'properties': {'frames': {'type': 'array', 'minItems': len(names), 'maxItems': len(names), 'items': frame},
                                             'segments': {'type': 'array', 'minItems': pairs, 'maxItems': pairs, 'items': segment},
                                             'summary': {'type': 'string', 'maxLength': 600}}, 'required': ['frames', 'segments', 'summary']}


def parse_json(text):
    """The first JSON object in a reply (tolerates code fences and leading prose)."""
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object found')
    depth, in_string, escape = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            escape = (char == '\\') and not escape
            if char == '"' and not escape:
                in_string = False
            elif char != '\\':
                escape = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])
    raise ValueError('the JSON object is not closed')


def _norm(value, allowed, aliases):
    if not isinstance(value, str):
        if isinstance(value, bool) and allowed is YNU:
            return 'yes' if value else 'no'
        return None
    key = value.strip().lower().replace('-', ' ')
    if key.replace(' ', '_') in allowed:
        return key.replace(' ', '_')
    if key in aliases:
        return aliases[key]
    for word, target in aliases.items():
        if key.startswith(word):
            return target
    return None


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def validate(answer, window):
    """(normalised answer or None, error sentences). Coerces enum spellings; never invents content."""
    names = labels(window)
    times = {frame['label']: frame['t'] for frame in window['frames']}
    if not isinstance(answer, dict):
        return None, ['the answer must be one JSON object with "segments" and "summary"']
    segments = answer.get('segments')
    errors = []
    states = []
    raw_frames = answer.get('frames')
    if not isinstance(raw_frames, list) or len(raw_frames) != len(names):
        errors.append(f'"frames" must have exactly {len(names)} entries, one per frame in order: ' + ', '.join(names))
    else:
        for label, item in zip(names, raw_frames):
            if not isinstance(item, dict) or item.get('frame') != label:
                errors.append(f'"frames" entry for {label} must have "frame": "{label}"')
                continue
            people = []
            for person in item.get('characters') if isinstance(item.get('characters'), list) else []:
                if isinstance(person, dict) and _text(person.get('name')) and _text(person.get('state')):
                    people.append({'name': _text(person['name']), 'state': _text(person['state'])})
            states.append({'frame': label, 't': times[label], 'characters': people})
    if not isinstance(segments, list):
        return None, ['"segments" must be an array with one entry per pair of consecutive frames']
    expected = list(zip(names, names[1:]))
    if len(segments) != len(expected):
        errors.append(f'"segments" must have exactly {len(expected)} entries, one per consecutive pair: ' + ', '.join(f'{a}→{b}' for a, b in expected))
    clean = []
    for index, (segment, pair) in enumerate(zip(segments, expected)):
        where = f'segment {index + 1} ({pair[0]}→{pair[1]})'
        if not isinstance(segment, dict):
            errors.append(f'{where} must be an object')
            continue
        if segment.get('from') != pair[0] or segment.get('to') != pair[1]:
            errors.append(f'{where} must have "from": "{pair[0]}" and "to": "{pair[1]}" (got {segment.get("from")!r}→{segment.get("to")!r})')
        camera = segment.get('camera') if isinstance(segment.get('camera'), dict) else {}
        move = _norm(camera.get('move'), CAMERA_MOVES, ALIASES['move'])
        if move is None:
            errors.append(f'{where}: camera.move must be one of ' + ', '.join(CAMERA_MOVES))
        people, seen = [], {}
        raw_people = segment.get('characters')
        if not isinstance(raw_people, list):
            errors.append(f'{where}: "characters" must be an array (empty when nobody is visible)')
            raw_people = []
        for person in raw_people:
            if not isinstance(person, dict) or not _text(person.get('name')):
                errors.append(f'{where}: every character needs a non-empty "name"')
                continue
            name = _text(person['name'])
            flags = {}
            for key in ('walking', 'speaking', 'blinking'):
                flags[key] = _norm(person.get(key), YNU, ALIASES['ynu'])
                if flags[key] is None:
                    errors.append(f'{where}: {name}.{key} must be yes, no or unclear')
            changes = []
            for change in person.get('changes') if isinstance(person.get('changes'), list) else []:
                part = _norm(change.get('part') if isinstance(change, dict) else None, PARTS, ALIASES['part'])
                text = _text(change.get('description')) if isinstance(change, dict) else ''
                if part is None or not text:
                    errors.append(f'{where}: each change of {name} needs "part" (one of {", ".join(PARTS)}) and a non-empty "description"')
                    continue
                changes.append({'part': part, 'description': text})
            earlier = seen.get(name.lower())
            if earlier is not None:
                # The same character listed twice: merge (yes > unclear > no) instead of rejecting, retries rarely fix it.
                rank = {'yes': 2, 'unclear': 1, 'no': 0, None: -1}
                for key, value in flags.items():
                    if rank[value] > rank[earlier[key]]:
                        earlier[key] = value
                earlier['changes'] += [c for c in changes if c not in earlier['changes']]
                earlier['merged'] = True
                continue
            entry = {'name': name, **flags, 'changes': changes}
            seen[name.lower()] = entry
            people.append(entry)
        others = []
        for other in segment.get('other_changes') if isinstance(segment.get('other_changes'), list) else []:
            kind = _norm(other.get('kind') if isinstance(other, dict) else None, OTHER_KINDS, ALIASES['kind'])
            text = _text(other.get('description')) if isinstance(other, dict) else ''
            if kind is None or not text:
                errors.append(f'{where}: each other change needs "kind" (one of {", ".join(OTHER_KINDS)}) and a non-empty "description"')
                continue
            others.append({'kind': kind, 'description': text})
        clean.append({'from': pair[0], 'to': pair[1], 'start': times[pair[0]], 'end': times[pair[1]],
                      'camera': {'move': move, 'detail': _text(camera.get('detail'))}, 'characters': people, 'other_changes': others})
    if errors:
        return None, errors
    return {'frames': states, 'segments': clean, 'summary': _text(answer.get('summary'))}, []


def format_errors(errors):
    lines = '\n'.join(f'- {error}' for error in errors[:12])
    return f'Your answer was rejected:\n{lines}\nReturn the complete corrected JSON object only.'


# -- claims ------------------------------------------------------------------------------------------------------------------

def _claim_id(*parts):
    return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:10]


def importance(category):
    return IMPORTANCE.get(category, 1.0)


def answer_claims(answer, window, config_key, sample_id):
    """Atomic, verifiable statements of one valid answer. Camera moves are claims too, scored by the pixel check, not the judge."""
    claims = []
    for number, segment in enumerate(answer.get('segments', [])):
        base = {'config': config_key, 'sample': sample_id, 'window': window['index'], 'from': segment['from'], 'to': segment['to'],
                'start': segment['start'], 'end': segment['end'], 'segment': number}
        claims.append({**base, 'id': _claim_id(config_key, sample_id, window['index'], number, 'camera'), 'category': 'camera', 'subject': 'camera',
                       'text': f"camera: {segment['camera']['move']}" + (f" ({segment['camera']['detail']})" if segment['camera']['detail'] else ''),
                       'move': segment['camera']['move'], 'judge': False})
        for person in segment['characters']:
            name = person['name']
            for flag, category, phrase in (('walking', 'walking', 'is walking (steps, feet or body travel)'),
                                           ('speaking', 'speaking', 'is speaking (mouth opens or changes shape)'),
                                           ('blinking', 'blink', 'blinks (eyes close or half close)')):
                if person[flag] == 'yes':
                    claims.append({**base, 'id': _claim_id(config_key, sample_id, window['index'], number, name, flag), 'category': category,
                                   'subject': name, 'text': f'{name} {phrase}', 'judge': True})
            for index, change in enumerate(person['changes']):
                if is_no_change(change['description']):
                    continue
                category = PART_CATEGORY[change['part']]
                if category == 'eyes_gaze' and re.search(r'\bblink|eyes? (close|shut)|closes? (her|his|their) eyes', change['description'], re.I):
                    category = 'blink'
                claims.append({**base, 'id': _claim_id(config_key, sample_id, window['index'], number, name, 'c', index), 'category': category,
                               'subject': name, 'text': f"{name} — {change['part']}: {change['description']}", 'judge': True})
        for index, other in enumerate(segment['other_changes']):
            if is_no_change(other['description']):
                continue
            claims.append({**base, 'id': _claim_id(config_key, sample_id, window['index'], number, 'o', index), 'category': other['kind'],
                           'subject': other['kind'], 'text': f"{other['kind']}: {other['description']}", 'judge': True})
    return claims


def segment_categories(segment):
    """The reference-flag categories one normalised segment reports (any character)."""
    found = set()
    for person in segment['characters']:
        if person['walking'] == 'yes':
            found.add('walking')
        if person['speaking'] == 'yes':
            found.add('speaking')
        if person['blinking'] == 'yes':
            found.add('blink')
        for change in person['changes']:
            if is_no_change(change['description']):
                continue
            category = PART_CATEGORY[change['part']]
            found.add(category)
            if category == 'eyes_gaze' and re.search(r'\bblink|eyes? (close|shut)', change['description'], re.I):
                found.add('blink')
    return found


KEYWORDS = [
    ('walking', r'\b(walk|walks|walking|steps?|stride|strides|runs?|running|approach|approaches|enters|exits|leaves|moves? (toward|towards|away|forward))\b'),
    ('speaking', r'\b(speak|speaks|speaking|talk|talks|says|said|mouth|lips|shouts?|whispers?|asks?|replies)\b'),
    ('blink', r'\b(blink|blinks|blinking|closes (her|his|their) eyes|eyes close)\b'),
    ('head', r'\b(head|nods?|tilts?|turns (her|his|their) head|looks? (up|down|back|away))\b'),
    ('eyes_gaze', r'\b(eyes?|gaze|glances?|looks? at|stares?)\b'),
    ('arms_hands', r'\b(arms?|hands?|fingers?|reaches?|reaching|raises?|lowers?|waves?|points?|grabs?|holds?|grips?|releases?)\b'),
    ('legs', r'\b(legs?|kneels?|kneeling|feet|foot|crouch|crouches|squats?)\b'),
    ('body_pose', r'\b(bends?|leans?|sits?|stands?|stands up|lies|turns (her|his|their) body|torso|shoulders?|climbs?)\b'),
    ('expression', r'\b(smiles?|frowns?|expression|cries|crying|tears?|blush|blushes|surprised|grimace)\b'),
]


def classify(text):
    """Keyword categories of a free-text motion statement (the stored baseline has no part enum)."""
    found = [category for category, pattern in KEYWORDS if re.search(pattern, text or '', re.I)]
    return found or ['body_pose']


def _seconds(value):
    if isinstance(value, (int, float)):
        return float(value), float(value)
    numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', str(value or ''))]
    if not numbers:
        return None
    return numbers[0], numbers[-1]


def baseline_answer(view, window):
    """The stored video analysis (``prompt_context.video_view``) laid over one window's pairs, in the shared shape.

    An event goes to every pair its time range overlaps; the camera movement is the cut's for every pair;
    a character marked ``speaking: yes`` without speech events is attributed to every pair (as coarse as the source).
    """
    names = labels(window)
    times = [frame['t'] for frame in window['frames']]
    camera = ((view or {}).get('camera') or {}) if isinstance(view, dict) else {}
    move = _norm(camera.get('movement'), CAMERA_MOVES, ALIASES['move']) or 'unclear'
    events = [event for event in (view or {}).get('events') or [] if isinstance(event, dict)] if isinstance(view, dict) else []
    speakers = {_text(person.get('character')) for person in (view or {}).get('characters') or [] if isinstance(person, dict) and person.get('speaking') == 'yes'} if isinstance(view, dict) else set()
    segments = []
    for index in range(len(names) - 1):
        start, end = times[index], times[index + 1]
        people = {}
        for event in events:
            span = _seconds(event.get('seconds'))
            if span is None:
                continue
            lo, hi = span
            if hi < lo:
                lo, hi = hi, lo
            if hi == lo:
                hi = lo + 0.1
            if lo < end and hi > start:
                name = _text(event.get('subject')) or 'someone'
                person = people.setdefault(name, {'name': name, 'walking': 'no', 'speaking': 'no', 'blinking': 'no', 'changes': []})
                action = _text(event.get('action'))
                cats = classify(action)
                if 'walking' in cats:
                    person['walking'] = 'yes'
                if 'speaking' in cats:
                    person['speaking'] = 'yes'
                if 'blink' in cats:
                    person['blinking'] = 'yes'
                part = next((p for p, c in (('head', 'head'), ('eyes', 'eyes_gaze'), ('arms', 'arms_hands'), ('legs', 'legs'), ('expression', 'expression'),
                                            ('body_pose', 'body_pose')) if c in cats), 'body_pose')
                if action and not set(cats) <= {'walking', 'speaking', 'blink'}:
                    person['changes'].append({'part': part, 'description': action})
        for name in speakers:
            if name and name not in people:
                people[name] = {'name': name, 'walking': 'no', 'speaking': 'yes', 'blinking': 'no', 'changes': []}
            elif name:
                people[name]['speaking'] = 'yes'
        segments.append({'from': names[index], 'to': names[index + 1], 'start': start, 'end': end, 'camera': {'move': move, 'detail': _text(camera.get('detail'))},
                         'characters': list(people.values()), 'other_changes': []})
    return {'segments': segments, 'summary': _text((view or {}).get('summary')) if isinstance(view, dict) else ''}
