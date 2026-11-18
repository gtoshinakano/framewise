"""DWPose keypoints -> per-frame states -> the shared segment JSON (setup C), and the DWPose + Qwen merge (setup D).

Stdlib only. Wording follows ``backend/app/still_vocabulary.yaml`` (arm.position, arm.elbow, hand.position,
leg.knee, character_position.horizontal, action.locomotion) and ``pose_detail.head_phrase``/``torso_phrase``
(screen-left/screen-right, viewer relative). Limbs are named by their side of the screen in each frame, never by
DWPose's anatomical left/right: that keeps the wording right for a character facing the camera and makes a DWPose
left/right flip harmless.

Nothing here reads faces: blinking, speaking, mouth and expression stay ``unclear``/empty (DWPose's face points
are unreliable on anime). Head turn is a rough yaw from nose, eyes and ears.

Thresholds are deliberately coarse; a change is reported only when a limb moves by a large angle or crosses
into another named region, so the rules prefer silence to jitter.
"""
import math
import re

try:  # the app's phrases when the backend is importable (container, host checkout); same wording otherwise
    from app.pose_detail import head_phrase, torso_phrase
except Exception:  # noqa: BLE001
    def _screen(yaw):
        return 'screen-right' if yaw > 0 else 'screen-left'

    def head_phrase(yaw):
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

    def torso_phrase(yaw):
        return head_phrase(yaw).replace('rear three-quarter view toward', 'rear three-quarter view with the front of the torso toward')

KP = 0.3
CONFIG_C = 'dwpose|rules|keypoints'
CONFIG_D = 'hybrid|observer|frames'


def hybrid_config(merge_with):
    """D's config key for the Qwen config it merges into: hybrid|<style>|<context>."""
    _, style, context = merge_with.split('|')
    return f'hybrid|{style}|{context}'
BODY_PARTS = ('arms', 'hands', 'legs', 'body_pose', 'position')  # D takes these (and walking) from DWPose where usable

NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

ARM_STEP, ELBOW_STEP, KNEE_STEP, HEAD_STEP = 35, 45, 40, 45     # degrees a part must move before it counts as a change
POSITION_STEP, DEPTH_STEP, HAND_MOVE = 0.06, 0.25, 0.6          # share of width, relative box height, share of torso length

# Walking (2026-11-18): the person travels through the world (camera removed) AND the ankles alternate.
LOWER_BODY = (LHIP, RHIP, LKN, RKN, LAN, RAN)
NOT_ON_FEET = ('sitting', 'crouching', 'lying')   # the sitting state: nobody walks while seated, crouched or lying
TRAVEL_KNOWN, TRAVEL_UNKNOWN = POSITION_STEP * 0.6, POSITION_STEP * 1.5   # share of width; a pair without a camera estimate needs more
SIZE_TRAVEL = 0.15                    # relative box height change that counts as walking toward or away from the camera (estimate known)
GAIT_MIN = 0.12                       # torso lengths an ankle must be apart (sideways or in height) in both frames to count as a lead
SWING_STEP = 0.25                     # torso lengths the ankle order must change to count as part of a step
NEAR_SECONDS = 1.0                    # a step in a neighbouring pair this close in time completes a half step
TRACKED_HEIGHT = 0.8                  # a person this tall in the picture can fool the camera estimate (it follows them)


def _ok(points, *indices):
    return all(points[i] and points[i][2] >= KP for i in indices)


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c):
    """Angle at b in degrees (180 = straight)."""
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if not n:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))))


def _from_down(a, b):
    """Angle of the vector a->b from straight down (0 = hanging, 90 = horizontal, 180 = straight up)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if not (dx or dy):
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dy / math.hypot(dx, dy)))))


def horizontal(cx):
    """character_position.horizontal for a box centre (0-1 of the width)."""
    for edge, phrase in ((0.15, 'far screen-left'), (0.35, 'screen-left'), (0.45, 'slightly left of center'), (0.55, 'centered'),
                         (0.65, 'slightly right of center'), (0.85, 'screen-right')):
        if cx < edge:
            return phrase
    return 'far screen-right'


def elbow_phrase(angle):
    if angle is None:
        return None
    for edge, phrase in ((75, 'sharply bent'), (100, 'bent at approximately a right angle'), (135, 'moderately bent'), (160, 'slightly bent')):
        if angle < edge:
            return phrase
    return 'fully extended'


def knee_phrase(angle):
    return {'fully extended': 'straight'}.get(elbow_phrase(angle), elbow_phrase(angle))


def arm_phrase(elevation, elbow, wrist_above_shoulder):
    if elevation is None:
        return None
    if elevation >= 120 or (elevation >= 90 and wrist_above_shoulder):
        return 'raised above the head'
    if elevation >= 70:
        return 'raised to shoulder height'
    if elevation >= 35:
        return 'raised to chest height'
    if elbow is not None and elbow < 100:
        return 'bent in front of the chest'
    return 'hanging naturally at the side'


def _yaw(points):
    """Rough head yaw (degrees, + toward screen-right, 180 facing away) from nose, eyes and ears; None when unseen."""
    nose = points[NOSE] if _ok(points, NOSE) else None
    eyes = [points[i] for i in (LEYE, REYE) if _ok(points, i)]
    ears = [points[i] for i in (LEAR, REAR) if _ok(points, i)]
    if not nose and not eyes:
        return 180 if len(ears) == 2 else None
    if len(ears) == 2 and nose:
        span = abs(ears[0][0] - ears[1][0]) / 2
        if span < 1:
            return None
        offset = max(-1.0, min(1.0, (nose[0] - (ears[0][0] + ears[1][0]) / 2) / span))
        return int(round(math.degrees(math.asin(offset)) / 15) * 15)
    if len(ears) == 1 and nose:
        return 90 if nose[0] > ears[0][0] else -90
    if len(eyes) == 2 and nose:
        span = abs(eyes[0][0] - eyes[1][0]) / 2
        if span < 1:
            return None
        offset = max(-1.0, min(1.0, (nose[0] - (eyes[0][0] + eyes[1][0]) / 2) / (2 * span)))
        return int(round(math.degrees(math.asin(offset)) / 15) * 15)
    return None


def person_state(person, size):
    """Measured state of one person in one keyframe: numbers for comparing, phrases for the text."""
    w, h = size
    # A hip, knee or ankle placed outside the picture was never seen (a bust shot): DWPose still invents the lower body there,
    # so a person cannot sit, lie or walk on those points.
    p = [None if (i in LOWER_BODY and pt and not (-0.01 * w <= pt[0] <= 1.01 * w and -0.01 * h <= pt[1] <= 1.01 * h)) else pt
         for i, pt in enumerate(person['points'])]
    box = person['box']
    state = {'name': person['name'], 'box': box, 'cx': (box[0] + box[2]) / 2 / w, 'cy': (box[1] + box[3]) / 2 / h,
             'height': (box[3] - box[1]) / h, 'usable': bool(person.get('usable')), 'parts': {'position': True}}
    shoulders = _ok(p, LSH, RSH)
    hips = [i for i in (LHIP, RHIP) if _ok(p, i)]
    torso_len = None
    if shoulders and hips:
        sh = _mid(p[LSH], p[RSH])
        hip = _mid(p[hips[0]], p[hips[-1]])
        torso_len = _dist(sh, hip)
        lean = _from_down(hip, sh)  # 180 = upright
        state['torso_tilt'] = None if lean is None else 180 - lean
    state['torso_len'] = torso_len or (box[3] - box[1]) * 0.3
    # Hands and feet are compared relative to the body, so a whole-body move is not reported as a hand or foot move.
    anchor = _mid(p[LSH], p[RSH]) if shoulders else centre_of(box)
    hip_anchor = _mid(p[hips[0]], p[hips[-1]]) if hips else centre_of(box)
    # Arms by screen side: the shoulder further left on screen is the screen-left arm.
    arms = []
    for s, e, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        if _ok(p, s, e):
            elevation = _from_down(p[s], p[e])
            elbow = _angle(p[s], p[e], p[wr]) if _ok(p, wr) else None
            above = _ok(p, wr) and p[wr][1] < p[s][1] - 0.3 * state['torso_len']
            arms.append({'x': p[s][0], 'elevation': elevation, 'elbow': elbow, 'phrase': arm_phrase(elevation, elbow, above), 'elbow_phrase': elbow_phrase(elbow)})
    arms.sort(key=lambda a: a['x'])
    state['arms'] = _sided(arms)
    state['parts']['arms'] = bool(arms)
    # Hands: wrist relative to head and torso.
    head = [p[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if _ok(p, i)]
    head_c = (sum(q[0] for q in head) / len(head), sum(q[1] for q in head) / len(head)) if head else None
    radius = max(0.35 * state['torso_len'], (max(q[0] for q in head) - min(q[0] for q in head)) if head else 0)
    hands = []
    for wr in (LWR, RWR):
        if not _ok(p, wr):
            continue
        q = p[wr]
        region = 'beside the body'
        if head_c and _dist(q, head_c) < 1.3 * radius:
            region = 'near the face'
        elif head_c and q[1] < head_c[1] - radius:
            region = 'above the head'
        elif shoulders and q[1] < min(p[LSH][1], p[RSH][1]):
            region = 'beside the head'
        elif shoulders and hips:
            x0, x1 = sorted((p[LSH][0], p[RSH][0]))
            pad = 0.15 * state['torso_len']
            hip_y = max(p[i][1] for i in hips)
            if x0 - pad <= q[0] <= x1 + pad and min(p[LSH][1], p[RSH][1]) <= q[1] <= hip_y - pad:  # a hand hanging at the hip is beside the body
                region = 'in front of the torso'
        hands.append({'x': q[0], 'pt': (q[0], q[1]), 'rel': ((q[0] - anchor[0]) / state['torso_len'], (q[1] - anchor[1]) / state['torso_len']), 'region': region})
    hands.sort(key=lambda a: a['x'])
    state['hands'] = _sided(hands)
    state['parts']['hands'] = bool(hands)
    # Legs: knee angle and lift, by screen side.
    legs = []
    for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        if _ok(p, hp, kn):
            knee = _angle(p[hp], p[kn], p[an]) if _ok(p, an) else None
            thigh = _from_down(p[hp], p[kn])
            ankle = (p[an][0], p[an][1]) if _ok(p, an) else None
            rel = ((ankle[0] - hip_anchor[0]) / state['torso_len'], (ankle[1] - hip_anchor[1]) / state['torso_len']) if ankle else None
            legs.append({'x': p[hp][0], 'knee': knee, 'thigh': thigh, 'ankle': ankle, 'rel': rel, 'phrase': knee_phrase(knee)})
    legs.sort(key=lambda a: a['x'])
    if len(legs) == 2 and legs[0]['ankle'] and legs[1]['ankle']:
        lift = legs[0]['ankle'][1] - legs[1]['ankle'][1]
        if abs(lift) > 0.3 * state['torso_len']:
            (legs[1] if lift > 0 else legs[0])['raised'] = True
        state['stride'] = abs(legs[0]['ankle'][0] - legs[1]['ankle'][0]) / state['torso_len']
    state['legs'] = _sided(legs)
    state['parts']['legs'] = bool(legs)
    # Gait: the two ankles by the person's own left/right (DWPose's anatomical labels, not the screen side), relative to the hips in torso
    # lengths. gait = (left - right sideways, left - right in height); it changes sign when the other foot takes the lead.
    state['gait'] = None
    if torso_len and hips and _ok(p, LAN) and _ok(p, RAN):
        state['gait'] = ((p[LAN][0] - p[RAN][0]) / torso_len, (p[LAN][1] - p[RAN][1]) / torso_len)
    # Which hip is on the screen-right: if it changes, the body turned (or DWPose swapped its sides) and a flipped gait proves nothing.
    state['hip_order'] = 0
    if _ok(p, LHIP, RHIP) and torso_len and abs(p[LHIP][0] - p[RHIP][0]) >= 0.15 * torso_len:
        state['hip_order'] = 1 if p[LHIP][0] > p[RHIP][0] else -1
    # Body pose from torso tilt and knees.
    knees = [leg['knee'] for leg in legs if leg['knee'] is not None]
    thighs = [leg['thigh'] for leg in legs if leg['thigh'] is not None]
    tilt = state.get('torso_tilt')
    pose = None
    aspect = (box[2] - box[0]) / max(box[3] - box[1], 1)
    foot_up = bool(hips) and any(leg['ankle'] and leg['ankle'][1] < min(p[i][1] for i in hips) for leg in legs)  # a kick or a high step
    state['aspect'] = aspect
    state['foot_up'] = foot_up
    if tilt is not None:
        if tilt > 60 or aspect > 1.3:
            pose = 'lying'  # a body clearly wider than tall is not standing, even seen from behind (cut 144's dive)
        elif thighs and min(thighs) > 50 and tilt < 40 and not foot_up:
            pose = 'crouching' if knees and min(knees) < 75 else 'sitting'  # every visible thigh near horizontal
        elif tilt >= 25:
            pose = 'leaning'
        else:
            pose = 'standing' if knees or thighs else 'upright'  # one thigh up with the other leg down: standing on one leg
    state['pose'] = pose
    state['parts']['body_pose'] = pose is not None
    yaw = None
    if shoulders and hips and pose != 'lying':  # shoulder width says nothing about turning when the body lies
        width = abs(p[LSH][0] - p[RSH][0]) / state['torso_len']
        facing_camera = p[LSH][0] > p[RSH][0]  # the person's left shoulder on screen-right
        nose_side = 1 if (_ok(p, NOSE) and p[NOSE][0] > (p[LSH][0] + p[RSH][0]) / 2) else -1
        if width >= 0.55:
            yaw = 0 if facing_camera else 180
        elif width >= 0.3:
            yaw = (45 if facing_camera else 135) * nose_side
        else:
            yaw = 90 * nose_side
    state['torso_yaw'] = yaw
    state['head_yaw'] = _yaw(p)
    state['parts']['head'] = state['head_yaw'] is not None
    state['parts']['walking'] = sum(1 for leg in legs if leg['ankle']) >= 1
    return state


def centre_of(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _sided(items):
    if len(items) == 2:
        return {'screen-left': items[0], 'screen-right': items[1]}
    if len(items) == 1:
        return {'one': items[0]}
    return {}


POSE_WORDS = {'standing': 'standing upright', 'upright': 'torso upright', 'sitting': 'sitting', 'crouching': 'crouching', 'lying': 'lying down', 'leaning': 'leaning'}


def state_text(state):
    if not state['usable']:
        return f"{horizontal(state['cx'])}; body keypoints not usable"
    bits = []
    if state['pose']:
        bits.append(POSE_WORDS[state['pose']])
    if state['torso_yaw'] is not None:
        bits.append('torso ' + torso_phrase(state['torso_yaw']))
    if state['head_yaw'] is not None:
        bits.append('head ' + head_phrase(state['head_yaw']))
    for side, arm in state['arms'].items():
        label = 'arm' if side == 'one' else f'{side} arm'
        bits.append(f"{label} {arm['phrase']}" + (f", elbow {arm['elbow_phrase']}" if arm['elbow_phrase'] else ''))
    for side, hand in state['hands'].items():
        bits.append(f"{'hand' if side == 'one' else side + ' hand'} {hand['region']}")
    for side, leg in state['legs'].items():
        if leg['phrase'] or leg.get('raised'):
            bits.append(f"{'leg' if side == 'one' else side + ' leg'} {leg['phrase'] or ''}{' raised' if leg.get('raised') else ''}".strip())
    bits.append(f"{horizontal(state['cx'])}, box {state['height'] * 100:.0f}% of the image height")
    return '; '.join(bits)


def camera_known(camera):
    """True when the pixel estimate gave a transform (static counts: that is a measurement too; a missing dx is 0)."""
    return bool(camera) and camera.get('move') not in (None, 'unclear')


def _own_shift(a, b, camera):
    """The person's own move between two frames: centre shift and size ratio with the camera's transform removed.

    A pan that follows the walker leaves them still in the picture and moves the background, so their own shift is
    the opposite of the camera's: the walker still travels through the world.
    """
    dx, scale = 0.0, 1.0
    if camera_known(camera):
        dx, scale = camera.get('dx') or 0.0, camera.get('scale') or 1.0
    expected = (a['cx'] - 0.5) * scale + 0.5 + dx
    return b['cx'] - expected, (b['height'] / (a['height'] * scale)) if a['height'] else 1.0


def alternation(a, b):
    """Do the ankles alternate between two frames? {'flip', 'swing'} or None when the gait cannot be read in both.

    flip: the other foot is now in the lead, sideways (a stride seen from the side) or in height (seen toward or away
    from the camera, the nearer or the lifted foot changes), with both leads clearly apart. It counts only while the
    same hip stays on the same side of the picture, so a turned body or a DWPose left/right swap is no alternation.
    swing: how far the ankle order moved in torso lengths (a foot swinging past or back, whether or not it crossed).
    """
    ga, gb = a.get('gait'), b.get('gait')
    if not ga or not gb:
        return None
    swing = max(abs(gb[0] - ga[0]), abs(gb[1] - ga[1]))
    same_side = not (a.get('hip_order') and b.get('hip_order') and a['hip_order'] != b['hip_order'])
    flip = same_side and any(x * y < 0 and min(abs(x), abs(y)) >= GAIT_MIN for x, y in ((ga[0], gb[0]), (ga[1], gb[1])))
    return {'flip': flip, 'swing': swing}


def walking_call(a, b, shift, size, known, near_flip=False):
    """'yes', 'no' or 'unclear' for one person between two keyframes (2026-11-18 rule).

    Walking needs the person to travel through the world with the camera removed (or, when the camera follows them, the
    background to travel the other way) AND the ankles to alternate. The rule says 'no' only for what it can see is not
    a walk, and 'unclear' where the keypoints cannot decide, so the hybrid leaves that call to the language model.
    * 'unclear': an unusable body, one ankle missing, or the person travels with legs that neither alternate nor swing
      (a far walker's small feet, or two similar phases of one stride).
    * 'no': sitting, crouching or lying in either frame (the sitting state), a foot above the hips (a kick), no travel.
    * 'yes': travel and a flip of the lead foot, or travel and a clear swing next to a flip within NEAR_SECONDS.
    Without a camera estimate the travel needs a larger shift, and a person who steps without travelling is 'unclear':
    the camera may be following them (as it may with anyone who fills the picture).
    """
    if not (a['usable'] and b['usable'] and a['parts']['walking'] and b['parts']['walking']):
        return 'unclear'
    if a['pose'] in NOT_ON_FEET or b['pose'] in NOT_ON_FEET:
        return 'no'
    if a.get('foot_up') or b.get('foot_up'):
        return 'no'
    alt = alternation(a, b)
    if alt is None:
        return 'unclear'
    travel = abs(shift) >= (TRAVEL_KNOWN if known else TRAVEL_UNKNOWN) or (known and abs(size - 1) >= SIZE_TRAVEL)
    if not travel:
        followed = not known or min(a['height'], b['height']) >= TRACKED_HEIGHT
        return 'unclear' if alt['flip'] and followed else 'no'
    if alt['flip'] or (alt['swing'] >= SWING_STEP and near_flip):
        return 'yes'
    return 'unclear'


def compare(a, b, camera=None, near_flip=False):
    """walking flag and changes (part, description) between two states of one person."""
    changes = []
    shift, size = _own_shift(a, b, camera)
    if abs(shift) >= POSITION_STEP:
        way = 'screen-right' if shift > 0 else 'screen-left'
        where = f"from {horizontal(a['cx'])} to {horizontal(b['cx'])}" if horizontal(a['cx']) != horizontal(b['cx']) else f"staying {horizontal(b['cx'])}"
        changes.append(('position', f'moves toward {way} ({where})'))
    if size >= 1 + DEPTH_STEP:
        changes.append(('position', 'comes closer to the camera (appears larger)'))
    elif size <= 1 / (1 + DEPTH_STEP):
        changes.append(('position', 'moves away from the camera (appears smaller)'))
    walking = 'unclear'
    if a['usable'] and b['usable']:
        walking = walking_call(a, b, shift, size, camera_known(camera), near_flip)
        if a['pose'] and b['pose'] and a['pose'] != b['pose'] and {a['pose'], b['pose']} != {'standing', 'upright'}:
            changes.append(('body_pose', f"{POSE_WORDS[a['pose']]} → {POSE_WORDS[b['pose']]}"))
        elif a.get('torso_tilt') is not None and b.get('torso_tilt') is not None and abs(a['torso_tilt'] - b['torso_tilt']) >= 25:
            more = b['torso_tilt'] > a['torso_tilt']
            changes.append(('body_pose', f"{'leans further' if more else 'straightens up'} (torso tilt {a['torso_tilt']:.0f}° → {b['torso_tilt']:.0f}°)"))
        if a['torso_yaw'] is not None and b['torso_yaw'] is not None and _turn(a['torso_yaw'], b['torso_yaw']) >= 60:
            changes.append(('body_pose', f"turns the body: {torso_phrase(a['torso_yaw'])} → {torso_phrase(b['torso_yaw'])}"))
        if a['head_yaw'] is not None and b['head_yaw'] is not None and _turn(a['head_yaw'], b['head_yaw']) >= HEAD_STEP \
                and head_phrase(a['head_yaw']) != head_phrase(b['head_yaw']):
            changes.append(('head', f"head turns: {head_phrase(a['head_yaw'])} → {head_phrase(b['head_yaw'])}"))
        for side in sorted(set(a['arms']) & set(b['arms'])):
            x, y = a['arms'][side], b['arms'][side]
            label = 'arm' if side == 'one' else f'{side} arm'
            moved = abs(x['elevation'] - y['elevation']) >= ARM_STEP
            bent = x['elbow'] is not None and y['elbow'] is not None and abs(x['elbow'] - y['elbow']) >= ELBOW_STEP
            if moved and x['phrase'] != y['phrase']:
                changes.append(('arms', f"{label}: {x['phrase']} → {y['phrase']}"))
            elif moved:
                changes.append(('arms', f"{label} {'rises' if y['elevation'] > x['elevation'] else 'lowers'} ({y['phrase']})"))
            elif bent and x['elbow_phrase'] != y['elbow_phrase']:
                changes.append(('arms', f"{label}: elbow {x['elbow_phrase']} → {y['elbow_phrase']}"))
        for side in sorted(set(a['hands']) & set(b['hands'])):
            x, y = a['hands'][side], b['hands'][side]
            label = 'hand' if side == 'one' else f'{side} hand'
            if x['region'] != y['region']:
                changes.append(('hands', f"{label}: {x['region']} → {y['region']}"))
            elif _dist(x['rel'], y['rel']) >= HAND_MOVE:
                dx = y['rel'][0] - x['rel'][0]
                dy = y['rel'][1] - x['rel'][1]
                way = ('toward screen-right' if dx > 0 else 'toward screen-left') if abs(dx) >= abs(dy) else ('downward' if dy > 0 else 'upward')
                changes.append(('hands', f"{label} moves {way}, staying {y['region']}"))
        for side in sorted(set(a['legs']) & set(b['legs'])):
            x, y = a['legs'][side], b['legs'][side]
            label = 'leg' if side == 'one' else f'{side} leg'
            if x.get('raised') != y.get('raised'):
                changes.append(('legs', f"{label} {'lifts' if y.get('raised') else 'comes down'}"))
            elif x['knee'] is not None and y['knee'] is not None and abs(x['knee'] - y['knee']) >= KNEE_STEP and x['phrase'] != y['phrase']:
                changes.append(('legs', f"{label}: knee {x['phrase']} → {y['phrase']}"))
            elif x['thigh'] is not None and y['thigh'] is not None and abs(x['thigh'] - y['thigh']) >= KNEE_STEP:
                changes.append(('legs', f"{label} swings {'forward or up' if y['thigh'] > x['thigh'] else 'down'}"))
    if walking == 'yes':
        changes = [c for c in changes if not c[1].startswith('moves toward')]  # the walk says it
    return walking, [{'part': part, 'description': text} for part, text in changes]


def _turn(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def frame_states(pose_sample):
    """{label: {name: state}} for every keyframe of one sample."""
    out = {}
    for frame in pose_sample['frames']:
        out[frame['label']] = {person['name']: person_state(person, frame['size']) for person in frame['persons']}
    return out


def _flips(states, labels, times):
    """{name: {pair index: True}} for the pairs of one window where the ankles of that person alternate."""
    out = {}
    for index, (a, b) in enumerate(zip(labels, labels[1:])):
        for name in set(states.get(a, {})) & set(states.get(b, {})):
            first, second = states[a][name], states[b][name]
            if first['usable'] and second['usable'] and first['pose'] not in NOT_ON_FEET and second['pose'] not in NOT_ON_FEET:
                alt = alternation(first, second)
                if alt and alt['flip']:
                    out.setdefault(name, {})[index] = True
    return out


def _near_flip(flipped, index, times, labels):
    """Another pair of this person's window with alternating ankles, at most NEAR_SECONDS away from pair ``index``."""
    for other in flipped:
        if other == index:
            continue
        gap = times[labels[index]] - times[labels[other + 1]] if other < index else times[labels[other]] - times[labels[index + 1]]
        if gap <= NEAR_SECONDS:
            return True
    return False


def answer_for_window(states, window, pairs):
    """The normalised answer (schema.validate shape) of setup C for one window."""
    labels = [f['label'] for f in window['frames']]
    times = {f['label']: f['t'] for f in window['frames']}
    measured = {(p['from'], p['to']): p for p in pairs or []}
    frames = [{'frame': label, 't': times[label], 'characters': [{'name': name, 'state': state_text(st)} for name, st in states.get(label, {}).items()]}
              for label in labels]
    segments, total = [], 0
    flips = _flips(states, labels, times)
    for index, (a, b) in enumerate(zip(labels, labels[1:])):
        camera = (measured.get((a, b)) or {}).get('camera') or {}
        first, second = states.get(a, {}), states.get(b, {})
        people = []
        for name in list(dict.fromkeys([*first, *second])):
            if name in first and name in second:
                walking, changes = compare(first[name], second[name], camera, near_flip=_near_flip(flips.get(name) or {}, index, times, labels))
            elif name in second:
                walking, changes = 'unclear', [{'part': 'position', 'description': f"enters the frame ({horizontal(second[name]['cx'])})"}]
            else:
                walking, changes = 'unclear', [{'part': 'position', 'description': 'leaves the frame'}]
            total += len(changes)
            people.append({'name': name, 'walking': walking, 'speaking': 'unclear', 'blinking': 'unclear', 'changes': changes})
        segments.append({'from': a, 'to': b, 'start': times[a], 'end': times[b], 'camera': {'move': camera.get('move') or 'unclear', 'detail': 'pixel estimate'},
                         'characters': people, 'other_changes': []})
    return {'frames': frames, 'segments': segments, 'summary': f'DWPose keypoint rules: {total} body changes over {len(segments)} pairs (no face, mouth or expression).'}


def usable_parts(states, label, name):
    state = states.get(label, {}).get(name)
    if not state or not state['usable']:
        return {'position'} if state else set()
    return {part for part, ok in state['parts'].items() if ok}


# -- merge (setup D) -----------------------------------------------------------------------------------------------------------

PLACE = re.compile(r'\b(?:(?:on|at|to|in|toward|towards) the (left|right|center|centre|middle)|(left|right) (?:side|edge|half)|screen[- ](left|right)|'
                   r'(left|right) of the (?:frame|image|picture)|(center|centre|middle) of the (?:frame|image|picture))\b', re.I)
DEPTH = re.compile(r'\b(background|distance|far|small)\b|\b(foreground|close[- ]up|large|fills)\b', re.I)
GENERIC_NAME = re.compile(r'^unknown person \d+$')
# The sitting state in the language model's frame text (either frame of the pair): nobody walks while seated, riding, kneeling or lying.
SEATED = re.compile(r'\b(?:sit|sits|sitting|sat|seated|riding|rides|rode|cycling|pedal(?:s|l?ing)|on (?:a|the|his|her) (?:bicycle|bike)|kneel(?:s|ing)?|crouch(?:es|ing)?|squat(?:s|ting)?|lying|lies|lay|prone)\b', re.I)


def seated_by_text(text):
    return bool(SEATED.search(text or ''))


def _sides(text):
    found = set()
    for match in PLACE.finditer(text or ''):
        word = next(g for g in match.groups() if g).lower()
        found.add({'centre': 'center', 'middle': 'center'}.get(word, word))
    return found


def _bucket(cx):
    return 'left' if cx < 0.4 else 'right' if cx > 0.6 else 'center'


def _names_match(a, b):
    a, b = a.lower(), b.lower()
    return a == b or a in b or b in a or any(part in a.split() for part in b.split() if len(part) > 2)


def match_characters(a_people, c_people, a_texts, c_states):
    """Pairs (a name, c name): by name first, then by the place words of A's state text against C's box, then the only pair left."""
    pairs, left_a, left_c = {}, list(a_people), list(c_people)
    for a in list(left_a):
        for c in list(left_c):
            if not GENERIC_NAME.match(c) and _names_match(a, c):
                pairs[a] = c
                left_a.remove(a)
                left_c.remove(c)
                break
    scored = []
    for a in left_a:
        sides = _sides(a_texts.get(a, ''))
        depth = DEPTH.search(a_texts.get(a, '') or '')
        for c in left_c:
            state = c_states.get(c)
            if not state:
                continue
            score = 0.0
            if sides:
                score += 1 if _bucket(state['cx']) in sides else -1
            if depth:
                far = bool(depth.group(1))
                score += 0.5 if far == (state['height'] < 0.4) else -0.5
            scored.append((score, a, c))
    for score, a, c in sorted(scored, reverse=True):
        if score > 0 and a in left_a and c in left_c:
            pairs[a] = c
            left_a.remove(a)
            left_c.remove(c)
    if len(left_a) == 1 and len(left_c) == 1:
        last = [s for s, a, c in scored if a == left_a[0] and c == left_c[0]]
        if not last or last[0] >= 0:  # one of each left and nothing contradicts it
            pairs[left_a[0]] = left_c[0]
    return pairs


def merge(a_answer, c_answer, states, pairs_measured):
    """Setup D: body parts and walking from DWPose where its keypoints pass in both frames, the rest from Qwen (A).

    No new model call. Eyes, mouth, expression, head, speaking, blinking and other_changes stay Qwen's. The camera
    is the pixel estimate.
    """
    measured = {(p['from'], p['to']): p for p in pairs_measured or []}
    a_states = {f['frame']: {c['name']: c['state'] for c in f['characters']} for f in a_answer.get('frames') or []}
    c_segments = {(s['from'], s['to']): s for s in c_answer['segments']}
    segments, notes = [], {'from_dwpose': 0, 'from_qwen': 0, 'dwpose_only_characters': 0}
    for segment in a_answer['segments']:
        key = (segment['from'], segment['to'])
        c_segment = c_segments.get(key) or {'characters': []}
        c_people = {p['name']: p for p in c_segment['characters']}
        a_people = {p['name']: p for p in segment['characters']}
        texts = {name: ' '.join([a_states.get(segment['from'], {}).get(name, ''), a_states.get(segment['to'], {}).get(name, '')]) for name in a_people}
        c_here = {name: states.get(segment['from'], {}).get(name) or states.get(segment['to'], {}).get(name) for name in c_people}
        pairing = match_characters(list(a_people), list(c_people), texts, c_here)
        people = []
        for name, a in a_people.items():
            c_name = pairing.get(name)
            if not c_name:
                people.append(a)
                notes['from_qwen'] += len(a['changes'])
                continue
            c = c_people[c_name]
            usable = usable_parts(states, segment['from'], c_name) & usable_parts(states, segment['to'], c_name)
            if c_name not in states.get(segment['from'], {}) or c_name not in states.get(segment['to'], {}):
                usable = {'position'}  # enters or leaves: DWPose knows only that
            changes = [ch for ch in a['changes'] if not (ch['part'] in BODY_PARTS and ch['part'] in usable)]
            notes['from_qwen'] += len(changes)
            body = [ch for ch in c['changes'] if ch['part'] in usable]
            notes['from_dwpose'] += len(body)
            walking = c['walking'] if 'walking' in usable and c['walking'] != 'unclear' else a['walking']
            if seated_by_text(texts.get(name)):
                walking = 'no'  # the language model sees what the keypoints cannot: a bicycle, a chair, the ground
            people.append({'name': c_name if not GENERIC_NAME.match(c_name) else name, 'walking': walking, 'speaking': a['speaking'], 'blinking': a['blinking'],
                           'changes': body + changes, 'merged_from': {'qwen': name, 'dwpose': c_name}})
        for c_name, c in c_people.items():
            if c_name in pairing.values():
                continue
            usable = usable_parts(states, segment['from'], c_name) & usable_parts(states, segment['to'], c_name)
            body = [ch for ch in c['changes'] if ch['part'] in usable or (ch['part'] == 'position' and ch['description'].startswith(('enters', 'leaves')))]
            if body or (c['walking'] == 'yes'):
                notes['dwpose_only_characters'] += 1
                notes['from_dwpose'] += len(body)
                people.append({'name': c_name, 'walking': c['walking'] if 'walking' in usable else 'unclear', 'speaking': 'unclear', 'blinking': 'unclear',
                               'changes': body, 'merged_from': {'qwen': None, 'dwpose': c_name}})
        camera = (measured.get(key) or {}).get('camera') or {}
        segments.append({**segment, 'camera': {'move': camera.get('move') or 'unclear', 'detail': 'pixel estimate'}, 'characters': people})
    return {'frames': a_answer.get('frames') or [], 'segments': segments, 'summary': a_answer.get('summary', ''), 'merge': notes}
