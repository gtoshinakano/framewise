"""Per-character pose schema: vocabularies, validators, prompt blocks and a JSON schema.

REUSABLE: standard library only, no import from the application. Copy this one file into any
project to validate (and prompt for) the structured pose block of one character:

    body_yaw, body_pose, head_yaw, head_pitch, head_roll, eye_gaze, eye_openness, gaze_target,
    mouth_state, eyebrow_state, facial_expression, camera_elevation, shot_size

Three ways of asking for it are supported (``APPROACHES``):

* ``full_confidence``  every field plus a 0.0-1.0 confidence per field; angles are free numbers.
* ``top5_confidence``  only the five fields the model is surest about, with confidence, best first.
* ``strict_detailed``  every field, no confidence; angles on a 45 degree grid; long explanations.

Conventions used everywhere: left/right are the IMAGE's left/right; yaw 0 faces the camera, positive
turns toward the image's right, +-90 is profile, 180 is the back of the head; pitch is negative when
looking down. Abstaining is allowed per field (``unclear`` for enums, ``null`` for angles).

Frame entries look like ``{"character_id": "B1", "head_yaw": 45, ..., "confidence": {...}}``.
Video entries look like ``{"character": "Mai", "start": {<pose>} | null, "end": {<pose>} | null}``.
"""
import json
import math

APPROACHES = ('full_confidence', 'top5_confidence', 'strict_detailed')
APPROACH_LABELS = {
    'full_confidence': 'A1 · Full JSON + confidence',
    'top5_confidence': 'A2 · Top-5 fields + confidence',
    'strict_detailed': 'A3 · Strict, detailed, no confidence',
}
TOP_N = 5
ABSTAIN = 'unclear'
GAZE_SPECIAL = ('camera', 'off_screen', 'none', ABSTAIN)

FIELD_ORDER = ('body_yaw', 'body_pose', 'head_yaw', 'head_pitch', 'head_roll', 'eye_gaze', 'eye_openness',
               'gaze_target', 'mouth_state', 'eyebrow_state', 'facial_expression', 'camera_elevation', 'shot_size')

GAZE_NINE = ('up_left', 'up', 'up_right', 'left', 'center', 'right', 'down_left', 'down', 'down_right')
SHOT_SIZES = ('extreme_close_up', 'close_up', 'medium_close_up', 'medium', 'medium_long', 'long', 'extreme_long')

ENUMS = {
    'body_pose': ('standing', 'walking', 'lying', 'sitting', 'running', 'other'),
    'head_roll': ('neutral', 'tilt_left', 'tilt_right'),
    'eye_gaze': GAZE_NINE,
    'eye_openness': ('wide', 'normal', 'half_closed', 'closed'),
    'mouth_state': ('closed', 'slightly_open', 'open', 'wide_open', 'other'),
    'eyebrow_state': ('neutral', 'raised', 'furrowed', 'lowered', 'asymmetric'),
    'facial_expression': ('neutral', 'happy', 'sad', 'angry', 'worried', 'surprised', 'embarrassed', 'afraid',
                          'disgusted', 'smug', 'determined', 'other'),
    'camera_elevation': ('eye_level', 'high_angle', 'low_angle', 'birds_eye', 'worms_eye'),
    'shot_size': SHOT_SIZES,
}
ANGLES = {
    'body_yaw': {'range': (-180, 180), 'grid': (-135, -90, -45, 0, 45, 90, 135, 180)},
    'head_yaw': {'range': (-180, 180), 'grid': (-135, -90, -45, 0, 45, 90, 135, 180)},
    'head_pitch': {'range': (-90, 90), 'grid': (-90, -45, 0, 45, 90)},
}

MIRROR_GAZE = {'up_left': 'up_right', 'up_right': 'up_left', 'left': 'right', 'right': 'left',
               'down_left': 'down_right', 'down_right': 'down_left'}
MIRROR_ROLL = {'tilt_left': 'tilt_right', 'tilt_right': 'tilt_left'}


# -- small helpers -----------------------------------------------------------------------------

def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def concrete(value):
    """True when a field carries a real value: present and not an abstention."""
    return value is not None and value != ABSTAIN


def circular_difference(a, b):
    """Smallest absolute difference between two angles in degrees (180 and -180 are the same)."""
    return abs((a - b + 180) % 360 - 180)


def values_of(entry):
    """The pose fields an entry carries, in schema order."""
    return {name: entry[name] for name in FIELD_ORDER if isinstance(entry, dict) and name in entry}


def parse_json(raw):
    """Parse a model answer that may be wrapped in a Markdown fence; raises ValueError."""
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1] if '\n' in text else ''
        text = text.rsplit('```', 1)[0]
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f'the answer is not valid JSON ({exc})') from exc


# -- validators -----------------------------------------------------------------------------------

def check_value(field, value, approach, known_ids=()):
    """None when ``value`` is acceptable for ``field`` under ``approach``, else a sentence for the model."""
    if field in ANGLES:
        if value is None:
            return None
        spec = ANGLES[field]
        low, high = spec['range']
        if not _number(value):
            return f'{field} must be a number of degrees or null, not {value!r}'
        if not low <= value <= high:
            return f'{field} {value:g} is outside {low} to {high}'
        if approach == 'strict_detailed' and value not in spec['grid']:
            return f'{field} {value:g} is not on the 45-degree grid; use one of {", ".join(str(v) for v in spec["grid"])} (the nearest)'
        return None
    if field == 'gaze_target':
        if not isinstance(value, str) or not value.strip():
            return 'gaze_target must be a non-empty string'
        if value in GAZE_SPECIAL or value in known_ids:
            return None
        if value.startswith('object:') and len(value) > len('object:'):
            return None
        listed = ', '.join([*known_ids, *GAZE_SPECIAL, 'object:<noun>'])
        return f'gaze_target {value!r} is not allowed; use one of {listed}'
    allowed = ENUMS[field]
    if value == ABSTAIN or value in allowed:
        return None
    return f'{field} {value!r} is not allowed; use one of {", ".join(allowed)}, or {ABSTAIN}'


def _confidence_errors(entry, fields):
    confidence = entry.get('confidence')
    if not isinstance(confidence, dict):
        return ['confidence must be an object with one number from 0.0 to 1.0 per field']
    errors = []
    missing = [name for name in fields if name not in confidence]
    extra = [name for name in confidence if name not in fields]
    if missing:
        errors.append('confidence is missing ' + ', '.join(missing))
    if extra:
        errors.append('confidence has unexpected keys ' + ', '.join(extra))
    for name, value in confidence.items():
        if name in fields and (not _number(value) or not 0 <= value <= 1):
            errors.append(f'confidence for {name} must be a number from 0.0 to 1.0, not {value!r}')
    return errors


def validate_pose(entry, approach, known_ids=(), allow_extra=()):
    """Errors in one pose object (the fields and ``confidence``; ids are checked by the callers)."""
    if approach not in APPROACHES:
        raise ValueError(f'unknown approach {approach!r}')
    if not isinstance(entry, dict):
        return ['a pose entry must be a JSON object']
    allowed = {*FIELD_ORDER, 'confidence', *allow_extra}
    errors = [f'unexpected key {key}' for key in entry if key not in allowed]
    present = [name for name in FIELD_ORDER if name in entry]
    if approach == 'top5_confidence':
        if len(present) != TOP_N:
            errors.append(f'return exactly {TOP_N} fields besides confidence (you returned {len(present)})')
    else:
        missing = [name for name in FIELD_ORDER if name not in entry]
        if missing:
            errors.append('missing fields: ' + ', '.join(missing))
    for name in present:
        problem = check_value(name, entry[name], approach, known_ids)
        if problem:
            errors.append(problem)
    if approach == 'strict_detailed':
        if 'confidence' in entry:
            errors.append('confidence is not wanted here; remove it')
    else:
        errors.extend(_confidence_errors(entry, present))
        if approach == 'top5_confidence' and not errors:
            chosen = [name for name in entry if name in FIELD_ORDER]  # the order the model wrote them in
            levels = [entry['confidence'][name] for name in chosen]
            if any(later > earlier for earlier, later in zip(levels, levels[1:])):
                errors.append('list the five fields from the highest to the lowest confidence')
    return errors


def validate_entry(entry, approach, known_ids):
    """Errors in one frame entry; ``known_ids`` are the box handles offered in the prompt."""
    if not isinstance(entry, dict):
        return ['a pose_detail entry must be a JSON object']
    errors = []
    identity = entry.get('character_id')
    if identity not in known_ids:
        errors.append(f'character_id {identity!r} must be one of {", ".join(known_ids) or "(nobody is marked)"}')
    errors.extend(validate_pose(entry, approach, known_ids, allow_extra=('character_id',)))
    return errors


def validate_answer(value, approach, known_ids):
    """Errors in the whole ``pose_detail`` array of a frame answer."""
    known_ids = list(known_ids)
    if not isinstance(value, list):
        return ['pose_detail must be an array with one entry per marked character']
    errors = []
    if len(value) != len(known_ids):
        errors.append(f'pose_detail needs exactly {len(known_ids)} entries ({", ".join(known_ids) or "none"}), it has {len(value)}')
    seen = []
    for index, entry in enumerate(value):
        label = entry.get('character_id') if isinstance(entry, dict) else None
        for problem in validate_entry(entry, approach, known_ids):
            errors.append(f'pose_detail[{index}]{f" ({label})" if label else ""}: {problem}')
        if label in seen:
            errors.append(f'pose_detail[{index}]: character_id {label} appears twice')
        seen.append(label)
    return errors


def validate_video_entry(entry, approach, names):
    """Errors in one video entry ``{"character", "start", "end"}``; ``names`` are the answer's characters."""
    if not isinstance(entry, dict):
        return ['a pose_detail entry must be a JSON object']
    errors = [f'unexpected key {key}' for key in entry if key not in ('character', 'start', 'end')]
    if entry.get('character') not in names:
        errors.append(f'character {entry.get("character")!r} must be one of the names used in characters: {", ".join(names) or "(none)"}')
    for moment in ('start', 'end'):
        if moment not in entry:
            errors.append(f'{moment} is missing (use null when the character is not visible then)')
        elif entry[moment] is not None:
            errors.extend(f'{moment}: {problem}' for problem in validate_pose(entry[moment], approach, names))
    return errors


def validate_video_answer(value, approach, names):
    """Errors in the ``pose_detail`` array of one video window, one entry per character in the answer."""
    names = list(names)
    if not isinstance(value, list):
        return ['pose_detail must be an array with one entry per entry of characters']
    errors = []
    if len(value) != len(names):
        errors.append(f'pose_detail needs exactly {len(names)} entries (one per entry of characters), it has {len(value)}')
    seen = []
    for index, entry in enumerate(value):
        label = entry.get('character') if isinstance(entry, dict) else None
        for problem in validate_video_entry(entry, approach, names):
            errors.append(f'pose_detail[{index}]{f" ({label})" if label else ""}: {problem}')
        if label in seen:
            errors.append(f'pose_detail[{index}]: character {label} appears twice')
        seen.append(label)
    return errors


def format_errors(errors, limit=12):
    """The error list as the bullet text sent back to the model."""
    shown = errors[:limit] + ([f'... and {len(errors) - limit} more'] if len(errors) > limit else [])
    return '\n'.join('- ' + error for error in shown)


# -- mirroring (a flipped image must give the mirrored answer) ----------------------------------------

def mirror_value(field, value, id_map=None):
    """What ``value`` of ``field`` becomes when the image is flipped left to right."""
    if not concrete(value):
        return value
    if field in ('body_yaw', 'head_yaw'):
        if not _number(value):
            return value
        return 180 if abs(value) == 180 else -value
    if field == 'eye_gaze':
        return MIRROR_GAZE.get(value, value)
    if field == 'head_roll':
        return MIRROR_ROLL.get(value, value)
    if field == 'gaze_target' and id_map:
        return id_map.get(value, value)
    return value


def mirror_entry(entry, id_map=None):
    """A copy of a pose entry as it should read on the flipped image (``id_map`` renames character handles)."""
    result = dict(entry)
    for name in FIELD_ORDER:
        if name in result:
            result[name] = mirror_value(name, result[name], id_map)
    if id_map and 'character_id' in result:
        result['character_id'] = id_map.get(result['character_id'], result['character_id'])
    return result


# -- prompt blocks -------------------------------------------------------------------------------------

CONVENTIONS = ('Conventions: left and right always mean the image\'s left and right as the viewer sees it, never the character\'s own side. '
               'Angles are in degrees. Yaw 0 = facing the camera, positive = turned toward the image\'s right, negative = toward the image\'s left, '
               '+-90 = profile, 180 = back toward the camera. Pitch 0 = level, positive = looking up, negative = looking down. '
               'Judge only what the pixels show.')

ABSTAIN_RULE = ('When a feature is hidden, cropped, turned away or too small to judge, answer unclear for an enum field and null for an angle; '
                'otherwise commit to the closest value. Never leave a requested key out.')

SHORT_FIELDS = {
    'body_yaw': 'body_yaw: the direction the torso faces, an integer from -180 to 180, or null when the torso is not visible.',
    'body_pose': 'body_pose: standing | walking | lying | sitting | running | other.',
    'head_yaw': 'head_yaw: the direction the face points, an integer from -180 to 180, or null when the head is not visible.',
    'head_pitch': 'head_pitch: head up or down, an integer from -90 to 90, or null.',
    'head_roll': 'head_roll: neutral | tilt_left | tilt_right (top of the head leaning toward the image\'s left or right).',
    'eye_gaze': 'eye_gaze: the line of sight in the image: up_left | up | up_right | left | center | right | down_left | down | down_right.',
    'eye_openness': 'eye_openness: wide | normal | half_closed | closed.',
    'gaze_target': 'gaze_target: what the character looks at: a box handle of another marked character, camera, off_screen, none, or object:<noun>.',
    'mouth_state': 'mouth_state: closed | slightly_open | open | wide_open | other.',
    'eyebrow_state': 'eyebrow_state: neutral | raised | furrowed | lowered | asymmetric.',
    'facial_expression': 'facial_expression: neutral | happy | sad | angry | worried | surprised | embarrassed | afraid | disgusted | smug | determined | other.',
    'camera_elevation': 'camera_elevation: the camera height relative to this character: eye_level | high_angle | low_angle | birds_eye | worms_eye.',
    'shot_size': 'shot_size: how much of this character the frame shows: extreme_close_up | close_up | medium_close_up | medium | medium_long | long | extreme_long.',
}

LONG_FIELDS = {
    'body_yaw': ('body_yaw: where the chest and shoulders face, on a 45-degree grid. 0 = chest toward the camera; 45 = turned 45 degrees toward the image\'s right '
                 '(three-quarter view); 90 = right profile; 135 = back three-quarter, turned toward the image\'s right; 180 = back toward the camera; '
                 'the same with a minus sign for the image\'s left (-45, -90, -135). Allowed: -135, -90, -45, 0, 45, 90, 135, 180. null when no torso is visible.'),
    'body_pose': ('body_pose: the posture the still shows. standing = upright on the feet and not mid-stride; walking = upright and caught mid-stride at a '
                  'walking pace; running = mid-stride with a running posture; sitting = on a seat or the floor; lying = body horizontal; other = anything '
                  'else (crouching, kneeling, jumping, leaning, hanging). Allowed: standing, walking, lying, sitting, running, other.'),
    'head_yaw': ('head_yaw: where the FACE points, same convention and the same grid as body_yaw. 0 = face toward the camera; 45 = three-quarter view '
                 'toward the image\'s right; 90 = right profile; 135 = mostly the back of the head, face turned toward the image\'s right; 180 = back of '
                 'the head; negative values mirror this toward the image\'s left. Allowed: -135, -90, -45, 0, 45, 90, 135, 180. null when the head is not visible.'),
    'head_pitch': ('head_pitch: the head\'s up or down tilt. 0 = level, chin neither raised nor lowered; 45 = chin raised, face tipped up; -45 = head bowed '
                   'down; 90 / -90 = the face points almost straight up / down. Allowed: -90, -45, 0, 45, 90. null when the head is not visible.'),
    'head_roll': ('head_roll: sideways tilt of the head. neutral = upright; tilt_left = the top of the head leans toward the image\'s left; tilt_right = '
                  'toward the image\'s right. Allowed: neutral, tilt_left, tilt_right.'),
    'eye_gaze': ('eye_gaze: the overall line of sight as seen in the image, combining head turn and eye position, in nine sectors: center = straight out of '
                 'the screen toward the camera, or too close to call; left / right = toward the image\'s left / right; up / down = toward the top / bottom '
                 'of the image; up_left, up_right, down_left, down_right = the diagonals. Allowed: up_left, up, up_right, left, center, right, down_left, '
                 'down, down_right.'),
    'eye_openness': ('eye_openness: wide = eyes opened wider than usual; normal = ordinary open eyes; half_closed = lids lowered over part of the eye; '
                     'closed = shut. Allowed: wide, normal, half_closed, closed.'),
    'gaze_target': ('gaze_target: what the line of sight is aimed at. The box handle (B1, B2, ...) of another marked character; camera = looking into the '
                    'lens; off_screen = at something outside the frame; none = at nothing in particular (eyes closed, empty stare); object:<noun> = a '
                    'visible object, for example object:book. Never the character\'s own handle.'),
    'mouth_state': ('mouth_state: closed = lips together; slightly_open = lips parted by a small gap; open = clearly open, as in speech; wide_open = '
                    'shouting, yawning, gasping; other = a shape that is none of these (pursed, tongue out). Allowed: closed, slightly_open, open, wide_open, other.'),
    'eyebrow_state': ('eyebrow_state: neutral = relaxed; raised = lifted; furrowed = drawn together and down (frown); lowered = low and relaxed or '
                      'sagging, without the pinch; asymmetric = one brow raised or lowered. Allowed: neutral, raised, furrowed, lowered, asymmetric.'),
    'facial_expression': ('facial_expression: the emotion the face shows. neutral, happy, sad, angry, worried, surprised, embarrassed (blush, shyness), '
                          'afraid, disgusted, smug (self-satisfied smirk), determined (set jaw, focused), other. Allowed: neutral, happy, sad, angry, '
                          'worried, surprised, embarrassed, afraid, disgusted, smug, determined, other.'),
    'camera_elevation': ('camera_elevation: the camera\'s height relative to this character\'s face. eye_level = level with the eyes; high_angle = above, '
                         'looking down; low_angle = below, looking up; birds_eye = almost straight down; worms_eye = almost straight up. Allowed: '
                         'eye_level, high_angle, low_angle, birds_eye, worms_eye.'),
    'shot_size': ('shot_size: how much of this character the frame holds. extreme_close_up = part of the face; close_up = the face fills the frame; '
                  'medium_close_up = head and chest; medium = waist up; medium_long = knees up; long = the whole body with surroundings; extreme_long = a '
                  'tiny figure in a wide scene. Allowed: extreme_close_up, close_up, medium_close_up, medium, medium_long, long, extreme_long.'),
}

APPROACH_TAIL = {
    'full_confidence': ('Every field above is mandatory in every entry. Also give confidence: an object with one number from 0.0 to 1.0 for every one of '
                        'the fields (0.0 = a pure guess or cannot tell, 1.0 = certain), judged only from what is visible; use low numbers when the face '
                        'is small, turned away, hidden or blurred. '),
    'top5_confidence': (f'Do NOT return every field. Choose the {TOP_N} fields you can judge most reliably for this character and return only those '
                        f'{TOP_N}, ordered from the highest to the lowest confidence, followed by confidence: an object with one number from 0.0 to 1.0 '
                        f'for each of those {TOP_N} (0.0 = a pure guess, 1.0 = certain). Leave the other fields out completely. '),
    'strict_detailed': ('Every field is mandatory in every entry and each value must be exactly one of the allowed values written in its definition. '
                        'Do not add confidence values or any other key. '),
}


def _field_text(approach):
    table = LONG_FIELDS if approach == 'strict_detailed' else SHORT_FIELDS
    return '\n'.join('- ' + table[name] for name in FIELD_ORDER)


def prompt_block(approach, kind='frame'):
    """The paragraph appended to a visual instruction to request ``pose_detail`` in ``approach``."""
    if approach not in APPROACHES:
        raise ValueError(f'unknown approach {approach!r}')
    if kind == 'frame':
        head = ('POSE DETAIL\nAdd one more key at the end of the JSON object: pose_detail. It is an array with exactly one entry for every marked character, '
                'in the listed order, and nobody else (an empty array when nobody is marked). Each entry measures that one character in this still. '
                'Inside pose_detail refer to characters only by their box handle (B1, B2, ...), never by name. Each entry has the key character_id (the '
                'handle) and the pose fields below.')
        order = 'character_id, '
    else:
        head = ('POSE DETAIL\nAdd one more key at the end of the JSON object: pose_detail. It is an array with exactly one entry for every entry of '
                'characters, in the same order. Each entry has the keys character (exactly the name used in characters), start and end. start measures the '
                'character in the first frames of this clip, end in the final frames; use null for a moment in which the character is not visible. Each '
                'of start and end is an object with the pose fields below. For gaze_target use the exact names used in characters instead of handles.')
        order = ''
    keys = ', '.join(FIELD_ORDER)
    if approach == 'top5_confidence':
        shape = f'Keys of a pose object: {order}your {TOP_N} chosen fields, confidence.'
    elif approach == 'full_confidence':
        shape = f'Keys of a pose object, in this order: {order}{keys}, confidence.'
    else:
        shape = f'Keys of a pose object, in this order: {order}{keys}.'
    return '\n\n'.join([head, CONVENTIONS, 'Pose fields:\n' + _field_text(approach), ABSTAIN_RULE + ' ' + APPROACH_TAIL[approach].strip(), shape])


# -- JSON schema (a llama.cpp grammar) for the pose_detail array ---------------------------------------------

def _field_schema(field, approach, known_ids):
    if field in ANGLES:
        spec = ANGLES[field]
        if approach == 'strict_detailed':
            return {'enum': [*spec['grid'], None]}
        return {'type': ['integer', 'null'], 'minimum': spec['range'][0], 'maximum': spec['range'][1]}
    if field == 'gaze_target':
        return {'type': 'string'}
    return {'enum': [*ENUMS[field], ABSTAIN]}


def json_schema(approach, known_ids, kind='frame'):
    """JSON schema of the ``pose_detail`` array. Best effort: it constrains values, not the top-5 choice or order."""
    fields = {name: _field_schema(name, approach, known_ids) for name in FIELD_ORDER}
    pose = {'type': 'object', 'additionalProperties': False, 'properties': dict(fields),
            'required': [] if approach == 'top5_confidence' else list(FIELD_ORDER)}
    if approach != 'strict_detailed':
        pose['properties']['confidence'] = {'type': 'object', 'additionalProperties': {'type': 'number', 'minimum': 0, 'maximum': 1}}
        pose['required'] = [*pose['required'], 'confidence']
    if kind == 'frame':
        item = {**pose, 'properties': {'character_id': {'enum': list(known_ids)}, **pose['properties']}, 'required': ['character_id', *pose['required']]}
        return {'type': 'array', 'items': item, 'minItems': len(known_ids), 'maxItems': len(known_ids)}
    moment = {'anyOf': [pose, {'type': 'null'}]}
    item = {'type': 'object', 'additionalProperties': False, 'required': ['character', 'start', 'end'],
            'properties': {'character': {'type': 'string'}, 'start': moment, 'end': moment}}
    return {'type': 'array', 'items': item}
