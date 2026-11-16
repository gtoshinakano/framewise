"""The structured pose block of one marked character in a first frame.

Standard library only. The first-frame model measures what a still can show reliably: the body's
state, which way the torso and the head face, where the eyes go and what they look at. Everything the
experiment in ``analysis_experiments/FINDINGS.md`` found unreliable is left out (face-detail enums,
head roll, per-field confidence, the 45-degree yaw grid).

The model's answer is *coerced*, never rejected for vocabulary: aliases are mapped, angles are snapped
to 15 degrees, handles become character names, anything unreadable becomes ``unclear``/``null``.
Geometry checks against the character boxes produce ``flags`` and change nothing; the consistency
review decides what a flag means.
"""
import math
import re

ANGLE_STEP = 15
SIDE_MARGIN = 60  # box centres closer than this on the 0-1000 grid say nothing about a side
ABSTAIN = 'unclear'

BODY_POSE = ('standing', 'walking', 'running', 'sitting', 'lying', 'crouching', 'kneeling', 'leaning', 'jumping', 'other', ABSTAIN)
GAZE_SECTORS = ('up_left', 'up', 'up_right', 'left', 'center', 'right', 'down_left', 'down', 'down_right')
EYE_GAZE = (*GAZE_SECTORS, 'closed', 'not_visible', ABSTAIN)
TARGET_KINDS = ('camera', 'off_screen', 'object', 'none', ABSTAIN)
ANGLES = {'body_yaw': 180, 'head_yaw': 180, 'head_pitch': 90}
FIELDS = ('body_pose', 'body_yaw', 'head_yaw', 'head_pitch', 'eye_gaze', 'gaze_target', 'gaze_object')

_BODY_WORDS = (
    (('stand', 'stands', 'standing', 'stood', 'upright'), 'standing'),
    (('sit', 'sits', 'sitting', 'seated', 'seat', 'perched', 'perching'), 'sitting'),
    (('walk', 'walks', 'walking', 'walked', 'stride', 'striding', 'stroll', 'strolling'), 'walking'),
    (('run', 'runs', 'running', 'sprint', 'sprinting', 'dash', 'dashing'), 'running'),
    (('lie', 'lies', 'lying', 'lay', 'laying', 'laid', 'reclining', 'sleeping'), 'lying'),
    (('crouch', 'crouches', 'crouching', 'crouched', 'squat', 'squats', 'squatting', 'bend', 'bends', 'bending', 'bent', 'stoop', 'stooping', 'hunched', 'hunching'), 'crouching'),
    (('kneel', 'kneels', 'kneeling', 'knelt'), 'kneeling'),
    (('lean', 'leans', 'leaning', 'leaned'), 'leaning'),
    (('jump', 'jumps', 'jumping', 'jumped', 'leap', 'leaping', 'hop', 'hopping'), 'jumping'),
)
_GAZE_ALIASES = {
    'upper_left': 'up_left', 'top_left': 'up_left', 'left_up': 'up_left', 'upleft': 'up_left',
    'upper_right': 'up_right', 'top_right': 'up_right', 'right_up': 'up_right', 'upright': 'up_right',
    'lower_left': 'down_left', 'bottom_left': 'down_left', 'left_down': 'down_left', 'downleft': 'down_left',
    'lower_right': 'down_right', 'bottom_right': 'down_right', 'right_down': 'down_right', 'downright': 'down_right',
    'upper': 'up', 'top': 'up', 'above': 'up', 'lower': 'down', 'bottom': 'down', 'below': 'down',
    'straight': 'center', 'centre': 'center', 'middle': 'center', 'forward': 'center', 'ahead': 'center', 'front': 'center', 'camera': 'center',
    'shut': 'closed', 'eyes_closed': 'closed', 'close': 'closed',
    'none': 'not_visible', 'null': 'not_visible', 'n/a': 'not_visible', 'na': 'not_visible', 'hidden': 'not_visible', 'not_readable': 'not_visible',
    'cannot_see': 'not_visible', 'obscured': 'not_visible', 'unknown': ABSTAIN, 'unsure': ABSTAIN,
}
_TARGET_WORDS = {
    'camera': 'camera', 'viewer': 'camera', 'lens': 'camera', 'audience': 'camera', 'the_camera': 'camera',
    'off_screen': 'off_screen', 'offscreen': 'off_screen', 'off_frame': 'off_screen', 'out_of_frame': 'off_screen', 'outside': 'off_screen',
    'none': 'none', 'nothing': 'none', 'null': 'none', 'no_one': 'none', 'nobody': 'none', 'empty': 'none',
    'unclear': ABSTAIN, 'unknown': ABSTAIN, 'unsure': ABSTAIN,
}


def _key(value):
    return re.sub(r'[\s\-]+', '_', str(value).strip().lower()) if isinstance(value, str) else ''


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def snap(value, limit):
    """An angle as an integer on the 15-degree step within ``limit``; ``None`` when it is not a number."""
    if isinstance(value, str):
        found = re.search(r'-?\d+(?:\.\d+)?', value)
        value = float(found.group()) if found else None
    if not _number(value):
        return None
    value = max(-limit, min(limit, value))
    snapped = int(round(value / ANGLE_STEP) * ANGLE_STEP)
    snapped = max(-limit, min(limit, snapped))
    return 180 if snapped == -180 else snapped


def body_pose(value):
    """``(body_pose, changed)``: the model's word mapped to the vocabulary."""
    key = _key(value)
    if key in BODY_POSE:
        return key, False
    if not key or key in ('none', 'null', 'unknown', 'not_visible', 'not_readable'):
        return ABSTAIN, bool(key)
    first = re.split(r'[_,;:/.]', key)[0]
    for words, result in _BODY_WORDS:
        if first in words:
            return result, True
    return 'other', True


def eye_gaze(value):
    key = _key(value)
    if key in EYE_GAZE:
        return key, False
    if key in _GAZE_ALIASES:
        return _GAZE_ALIASES[key], True
    flat = key.replace('_', '')
    for sector in GAZE_SECTORS:
        if flat == sector.replace('_', ''):
            return sector, True
    return ABSTAIN, bool(key)


def gaze_target(value, noun, own_ref, people):
    """``(target, target_id, object, changed)``: a character name, a special kind, or ``object`` with its noun."""
    text = value.strip() if isinstance(value, str) else ''
    noun = noun.strip()[:60] if isinstance(noun, str) else ''
    key = _key(text)
    if not text:
        return ABSTAIN, None, '', True
    handle = text.upper()
    if handle == own_ref:
        return 'none', None, '', True  # a character never looks at their own box
    for person in people:
        if handle == person['ref'] or key == _key(person['name']):
            return person['name'], person['character_context_id'], '', False
    if key == 'object':
        return 'object', None, noun, False
    if key in TARGET_KINDS:
        return key, None, '', False
    if key in _TARGET_WORDS:
        return _TARGET_WORDS[key], None, '', True
    if key.startswith(('object:', 'object_')):
        return 'object', None, re.sub(r'^object[:_\s]*', '', text, flags=re.I).strip()[:60] or noun, True
    words = set(re.findall(r'\w+', key))
    matches = [person for person in people if person['ref'] != own_ref and words and words <= set(re.findall(r'\w+', _key(person['name'])))]
    if len(matches) == 1:
        return matches[0]['name'], matches[0]['character_context_id'], '', True
    return 'object', None, text[:60], True  # an unknown noun is most likely a thing, kept as written


def _view(yaw):
    """The torso or head direction as a view of the camera, with the image side it turns toward."""
    if yaw is None:
        return None
    side = 'right' if yaw > 0 else 'left'
    size = abs(yaw)
    if size <= 15:
        return 'front'
    if size <= 60:
        return f'three_quarter_{side}'
    if size <= 105:
        return f'profile_{side}'
    if size <= 150:
        return f'back_three_quarter_{side}'
    return 'back'


def _centre(box):
    return (box[0] + box[2]) / 2


def _box(person):
    box = person.get('box') if isinstance(person, dict) else None
    return box if isinstance(box, list) and len(box) == 4 and all(_number(value) for value in box) else None


def _yaw_side(yaw):
    if not isinstance(yaw, int) or abs(yaw) < 20 or abs(yaw) > 160:
        return None
    return 'right' if yaw > 0 else 'left'


def _gaze_side(gaze):
    return 'right' if gaze in ('right', 'up_right', 'down_right') else 'left' if gaze in ('left', 'up_left', 'down_left') else None


def _gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def flags(pose, person, people):
    """Geometry checks that disagree: ``[{code, text}]``. They never change a value."""
    found = []
    name = person.get('name') or 'the character'
    yaw, pitch, body, eyes = pose.get('head_yaw'), pose.get('head_pitch'), pose.get('body_yaw'), pose.get('eye_gaze')
    target = next((other for other in people if other['name'] == pose.get('gaze_target') and other['ref'] != person['ref']), None)
    if target and _box(target) and _box(person):
        shift = _centre(_box(target)) - _centre(_box(person))
        signals = [side for side in (_yaw_side(yaw), _gaze_side(eyes)) if side]
        if abs(shift) >= SIDE_MARGIN and signals:
            expected = 'right' if shift > 0 else 'left'
            if not all(side == expected for side in signals):
                found.append({'code': 'gaze_side', 'text': f'{name} looks at {target["name"]}, who is on the image {expected}, but the head or eyes point to the image {" and ".join(sorted(set(signals)))}'})
    if pose.get('gaze_target') == 'camera':
        bad = (isinstance(yaw, int) and abs(yaw) > 45) or eyes not in ('center', 'up', 'down', ABSTAIN, 'not_visible', 'closed')
        if bad:
            found.append({'code': 'camera_gaze', 'text': f'{name} is said to look into the camera while the head or eyes point away'})
    if isinstance(pitch, int) and abs(pitch) >= 25 and eyes in GAZE_SECTORS:
        if (pitch < 0 and eyes in ('up', 'up_left', 'up_right')) or (pitch > 0 and eyes in ('down', 'down_left', 'down_right')):
            found.append({'code': 'pitch_gaze', 'text': f'{name}: the head tilts {"down" if pitch < 0 else "up"} but the eyes look {"up" if pitch < 0 else "down"}'})
    if isinstance(yaw, int) and isinstance(body, int) and _gap(yaw, body) > 100:
        found.append({'code': 'head_body', 'text': f'{name}: the head and the torso face directions {_gap(yaw, body)} degrees apart'})
    return found


def resolve(raw, person, people=()):
    """The stored ``pose_detail`` of one marked character. Never raises.

    ``person`` is the marked character (``ref``, ``name``, ``character_context_id``, ``box``) and ``people``
    all of them. A missing or unusable block becomes ``{'status': 'missing'}``.
    """
    if not isinstance(raw, dict) or not any(name in raw for name in FIELDS):
        return {'status': 'missing'}
    people = [item for item in people or [] if isinstance(item, dict) and item.get('ref')]
    changed = []
    pose, moved = body_pose(raw.get('body_pose'))
    if moved:
        changed.append('body_pose')
    result = {'body_pose': pose}
    for name, limit in ANGLES.items():
        value = snap(raw.get(name), limit)
        if isinstance(raw.get(name), str) and value is not None:
            changed.append(name)  # snapping a number to the 15-degree step is not a correction, a number written as text is
        result[name] = value
    gaze, moved = eye_gaze(raw.get('eye_gaze'))
    if moved:
        changed.append('eye_gaze')
    result['eye_gaze'] = gaze
    target, target_id, noun, moved = gaze_target(raw.get('gaze_target'), raw.get('gaze_object'), person.get('ref'), people)
    if moved:
        changed.append('gaze_target')
    result.update({'gaze_target': target, 'gaze_object': noun})
    if target_id:
        result['gaze_target_id'] = target_id
    result['body_view'], result['head_view'] = _view(result['body_yaw']), _view(result['head_yaw'])
    box = _box(person)
    if box:
        result['frame_fill'] = round(max(0, min(1000, box[3]) - max(0, box[1])) / 1000, 2)
    found = flags(result, person, people)
    if found:
        result['flags'] = found
    if changed:
        result['coerced'] = changed
    return result


def stats(characters):
    """How the pose blocks of one frame fared, for the result's settings."""
    entries = [item.get('pose_detail') for item in characters or [] if isinstance(item, dict)]
    return {'entries': len(entries), 'missing': sum(1 for item in entries if not isinstance(item, dict) or item.get('status') == 'missing'),
            'coerced': sum(len(item.get('coerced') or []) for item in entries if isinstance(item, dict)),
            'flags': sum(len(item.get('flags') or []) for item in entries if isinstance(item, dict))}


# -- the words of still_vocabulary.yaml ---------------------------------------------------------------------------
# Prompts that read a pose block and the shot writer's keyword lists share one vocabulary (viewer-relative:
# screen-left and screen-right). tests/test_still_vocabulary.py checks that every phrase below exists in the dictionary.

def _screen(yaw):
    return 'screen-right' if yaw > 0 else 'screen-left'


def torso_phrase(yaw):
    """``body.orientation`` of the dictionary for a torso direction, or None when it is not visible."""
    if not isinstance(yaw, int):
        return None
    size = abs(yaw)
    if size <= 15:
        return 'facing the camera'
    if size <= 30:
        return f'slightly turned toward {_screen(yaw)}'
    if size <= 60:
        return f'three-quarter view toward {_screen(yaw)}'
    if size <= 105:
        return f'side profile facing {_screen(yaw)}'
    if size <= 150:
        return f'rear three-quarter view with the front of the torso toward {_screen(yaw)}'
    return 'facing away from the camera'


def head_phrase(yaw):
    """``head.yaw`` of the dictionary for a head direction, or None when it is not visible."""
    if not isinstance(yaw, int):
        return None
    size = abs(yaw)
    if size <= 15:
        return 'facing the camera'
    if size <= 30:
        return f'slightly turned toward {_screen(yaw)}'
    if size <= 60:
        return f'three-quarter view toward {_screen(yaw)}'
    if size <= 105:
        return f'side profile facing {_screen(yaw)}'
    if size <= 150:
        return f'rear three-quarter view toward {_screen(yaw)}'
    return 'facing away from the camera'


def pitch_phrase(pitch):
    """``head.pitch`` of the dictionary; None for a head that is level or not visible."""
    if not isinstance(pitch, int) or abs(pitch) < 15:
        return None
    if pitch < 0:
        return 'chin slightly lowered' if pitch > -30 else 'chin lowered' if pitch > -45 else 'chin strongly lowered'
    return 'chin slightly raised' if pitch < 30 else 'chin raised' if pitch < 45 else 'chin strongly raised'


_GAZE_WORDS = {'up_left': 'diagonally upward toward screen-left', 'up': 'upward', 'up_right': 'diagonally upward toward screen-right', 'left': 'toward screen-left',
               'center': 'straight ahead', 'right': 'toward screen-right', 'down_left': 'diagonally downward toward screen-left', 'down': 'downward',
               'down_right': 'diagonally downward toward screen-right'}


def gaze_phrase(sector, target=None):
    """``eyes.gaze_direction`` of the dictionary (``eyes.openness`` for closed eyes); None when the eyes are not visible or unclear."""
    if sector == 'closed':
        return 'fully closed'
    if sector == 'center' and target == 'camera':
        return 'directly into the camera'
    return _GAZE_WORDS.get(sector)


def target_phrase(pose):
    """``eyes.gaze_target`` of the dictionary: a character by name, or the dictionary's words for the other targets."""
    target = pose.get('gaze_target')
    if not target or target == ABSTAIN:
        return None
    if target == 'object':
        return f"an object ({pose['gaze_object']})" if pose.get('gaze_object') else 'an object'
    return {'camera': 'the camera', 'off_screen': 'a point outside the frame', 'none': 'no specific target'}.get(target, target)


def compact(pose):
    """The short form text prompts read: state, torso, head, eyes, what is looked at, and the app's flags."""
    if not isinstance(pose, dict) or pose.get('status') == 'missing':
        return None
    if 'body_pose' not in pose:
        return pose  # already compact
    out = {}
    if pose.get('body_pose') not in (None, ABSTAIN):
        out['state'] = pose['body_pose']
    for key, phrase, yaw in (('torso', torso_phrase, 'body_yaw'), ('head', head_phrase, 'head_yaw')):
        words = phrase(pose.get(yaw))
        if words:
            out[key] = words if words == 'facing the camera' else f'{words} ({pose[yaw]:+d}°)'
    tilt = pitch_phrase(pose.get('head_pitch'))
    if tilt:
        out['head'] = f"{out['head']}, {tilt}" if out.get('head') else tilt
    eyes = gaze_phrase(pose.get('eye_gaze'), pose.get('gaze_target'))
    if eyes:
        out['eyes'] = eyes
    looking = target_phrase(pose)
    if looking:
        out['looking_at'] = looking
    if pose.get('flags'):
        out['check'] = [flag['text'] for flag in pose['flags'] if isinstance(flag, dict) and flag.get('text')]
    return out or None


def brief(pose):
    """One line for a reader that only compares states: ``sitting; torso three-quarter view toward screen-right; head side profile facing screen-left; looking at Rio``."""
    if not isinstance(pose, dict) or 'body_pose' not in pose:
        return ''
    parts = [pose['body_pose']] if pose.get('body_pose') not in (None, ABSTAIN) else []
    for label, phrase, yaw in (('torso', torso_phrase, 'body_yaw'), ('head', head_phrase, 'head_yaw')):
        words = phrase(pose.get(yaw))
        if words:
            parts.append(f'{label} {words}')
    looking = target_phrase(pose)
    if looking:
        parts.append('looking at ' + looking)
    return '; '.join(parts)


# -- prompt block and grammar ---------------------------------------------------------------------------------------

POSE_RULES = '''pose_detail: an object that measures this one character in the still, with exactly the keys body_pose, body_yaw, head_yaw, head_pitch, eye_gaze, gaze_target and gaze_object. Judge only what the pixels show.
Conventions: left and right are always the image's left and right as the viewer sees it, never the character's own side. Angles are whole degrees. Yaw 0 means facing the camera, a positive yaw means turned toward the image's right, a negative yaw toward the image's left, plus or minus 90 is profile and 180 is the back toward the camera. Pitch 0 is level, positive is looking up, negative is looking down.
- body_pose: the state the still shows, chosen from standing, walking, running, sitting, lying, crouching, kneeling, leaning, jumping, other. A figure caught mid-stride is walking or running; an upright figure whose legs are planted is standing; unclear when the body is not visible.
- body_yaw: the direction the chest and shoulders face; null when no torso is visible.
- head_yaw: the direction the face points; null when the head is not visible. Back of the head toward the camera is 180.
- head_pitch: how far the head tips up or down; null when the head is not visible.
- eye_gaze: the line of sight as seen in the image, combining head turn and eye position, chosen from up_left, up, up_right, left, center, right, down_left, down, down_right; closed when the eyes are shut; not_visible when the eyes are hidden, covered or turned away; unclear when too small to tell. center means straight out of the screen.
- gaze_target: what the line of sight is aimed at: the handle of another marked character, camera, off_screen (something outside the frame), object (a visible thing, named in gaze_object), none (nothing in particular, or the eyes are closed), or unclear. Never the character's own handle. gaze_object is the noun of the object, otherwise an empty string.
Do not force a value: choose not_visible, unclear or null when the feature is hidden, cropped, turned away or too small to judge. Every key is always present.'''


def json_schema(handles):
    """JSON schema (a llama.cpp grammar) of one character's pose block."""
    angle = lambda limit: {'type': ['integer', 'null'], 'minimum': -limit, 'maximum': limit}
    properties = {'body_pose': {'enum': list(BODY_POSE)}, 'body_yaw': angle(180), 'head_yaw': angle(180), 'head_pitch': angle(90),
                  'eye_gaze': {'enum': list(EYE_GAZE)}, 'gaze_target': {'enum': [*handles, *TARGET_KINDS]}, 'gaze_object': {'type': 'string'}}
    return {'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}
