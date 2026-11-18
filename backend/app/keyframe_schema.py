"""The keyframe analysis answer: its json_schema grammar, validator and windows (ported from ``multiframe_experiment/schema.py``).

Stdlib only. One Qwen3.5 call answers one window of keyframes:

    {"frames": [{"frame": "F1", "characters": [{"name": str, "state": str}]}],
     "segments": [{"from": "F1", "to": "F2",
                   "camera": {"move": <CAMERA_MOVES>, "detail": str},
                   "characters": [{"name": str, "walking": yes|no|unclear, "speaking": ..., "blinking": ...,
                                   "changes": [{"part": <PARTS>, "description": str}]}],
                   "other_changes": [{"kind": <OTHER_KINDS>, "description": str}]}],
     "summary": str}

The per-frame states come first so the model looks at each frame before comparing them (it stopped copying the
same change into every pair). Times are not asked for: the app knows them from the frame labels and writes
``start``/``end`` into the normalised answer.
"""
import json
import re

CAMERA_MOVES = ('static', 'pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'zoom_in', 'zoom_out', 'dolly_in', 'dolly_out',
                'truck', 'orbit', 'shake', 'unclear')
YNU = ('yes', 'no', 'unclear')
PARTS = ('body_pose', 'head', 'eyes', 'mouth', 'arms', 'hands', 'legs', 'position', 'expression')
OTHER_KINDS = ('outfit', 'color', 'lighting', 'object', 'background', 'effect')

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


# Bounds in the grammar: a model otherwise loops (LFM3 repeated "no change" entries up to the token limit).
MAX_CHARACTERS, MAX_CHANGES, MAX_OTHER = 6, 8, 4
TEXT = {'type': 'string', 'maxLength': 240}
NO_CHANGE = re.compile(r'^\s*(no (visible |noticeable )?change|unchanged|not changed|same\b|no movement|remains? |stays? |still )', re.I)
STATIC_WORDS = re.compile(r'\b(remains?|remaining|stays?|unchanged|still|maintains?|maintaining|holds? (the same|still)|same (as|position|pose|expression))\b', re.I)


def is_no_change(text):
    """A "change" that says nothing changed: "no change…", "arm remains extended", "outfit unchanged"."""
    return bool(NO_CHANGE.search(text or '') or STATIC_WORDS.search(text or ''))


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
    state = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'state': {'type': 'string', 'maxLength': 420}}, 'required': ['name', 'state']}
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
