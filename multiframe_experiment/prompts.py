"""Prompt styles and context levels. A config is (model, style, context); add a style or a context here and nothing else changes.

Every style asks for the same JSON (``schema.py``); they differ in framing and in how a change is phrased.
Every context level adds information on top of the frames; the stored video analysis is never given (it is the baseline).
"""
import json

COMMON_RULES = '''Frames are labelled F<n> with their time in seconds from the start of the shot. They are keyframes picked where the picture changed, so the time between two frames varies.
Return one JSON object only:
{"frames": [one entry per frame, in order: {"frame": "F1", "characters": [{"name": "...", "state": "..."}]}],
 "segments": [one entry per pair of consecutive frames, in order: {"from": "F1", "to": "F2", "camera": {"move": ..., "detail": "..."}, "characters": [{"name": "...", "walking": "yes|no|unclear", "speaking": "yes|no|unclear", "blinking": "yes|no|unclear", "changes": [{"part": "body_pose|head|eyes|mouth|arms|hands|legs|position|expression", "description": "..."}]}], "other_changes": [{"kind": "outfit|color|lighting|object|background|effect", "description": "..."}]}],
 "summary": "..."}
First fill "frames": look at each frame on its own and write, for every visible character, a short state of that frame only: pose, head angle, where the eyes look and whether they are open or closed, whether the mouth is open or closed, arm, hand and leg positions, place in the image. Then write "segments" by comparing the states of the two frames of each pair: a change exists only where the two states differ. Consecutive pairs usually differ, so do not repeat the same change for every pair unless the frames really show it each time.
camera.move is one of: static, pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out, dolly_in, dolly_out, truck, orbit, shake, unclear.
Priorities, most important first: who walks; whether and how the camera moves; who speaks (mouth movement); pose changes; head angle changes; eye blinks and gaze; arm, hand and leg positions. Outfit, colours and lighting matter only when they visibly change.
"changes" and "other_changes" list only what differs between the two frames; never list a part that stays the same (an empty list is fine). Keep each description to one short sentence.
Left and right always mean the viewer's left and right of the image.
A burned-in subtitle strip, if any, is not part of the picture: ignore its text and its appearance.'''

STYLES = {
    'animator': {
        'label': 'Animator: "how would you animate these frames?"',
        'system': '''You are a key animator for an anime production. You receive the keyframes of one shot. Your job: tell the in-between animator exactly how to animate from each keyframe to the next one.
For every pair of consecutive frames write the animation directions: how the camera moves, and for each character which body parts move, from where to where, in which direction and how strongly (for example a small head turn versus a full turn). Write each change as a direction an animator can follow, with the start and end state of the part.
''' + COMMON_RULES + '''
"walking", "speaking" and "blinking" say whether that motion must be animated between the two frames. "changes" lists every other part that must move. "summary" is the animation plan of the whole window in two or three sentences.''',
        'retry_hint': 'Keep the animation directions concrete.',
    },
    'observer': {
        'label': 'Observer: "describe exactly what changed between frames"',
        'system': '''You compare consecutive keyframes of one anime shot and report exactly what changed between them.
Only report what you can see change between the two images. Small changes count: a blink, a slight head tilt, a mouth opening, a hand shifting a few pixels. Do not report something that stays the same, and do not guess motion that the frames do not show. When a body part is hidden or too small to judge, say "unclear" or leave it out; "no" and an empty "changes" list are valid answers.
''' + COMMON_RULES + '''
"walking", "speaking" and "blinking" say whether that motion is visible between the two frames. "changes" lists every other part that visibly changed, each with what it looked like in the first frame and what it looks like in the second. "summary" says in two or three sentences what happens across the window.''',
        'retry_hint': 'Report only visible changes.',
    },
    'inspector': {
        'label': 'Inspector: observer + fixed state fields per frame + field-by-field comparison',
        'system': '''You compare consecutive keyframes of one anime shot and report exactly what changed between them, for every character: the main ones, small ones in the background, and anyone entering or leaving the picture.
Only report what you can see change between the two images. Small changes count: a blink, a slight head turn, a mouth opening, a hand shifting, a few steps far away. Never guess motion the frames do not show.
''' + COMMON_RULES + '''
How to write "frames": each character's "state" is these fields in this order, separated by semicolons, a few words each:
place (screen-left / center / screen-right, near or far, how big in the picture); pose (standing, mid-step, sitting, crouching, kneeling, lying, leaning, jumping, kicking, falling); head (which way it faces, tilt); eyes (open, half or closed; where they look); mouth (open or closed, shape); arms; hands; legs and feet.
Write "not visible" for a hidden or cropped field.
How to write "segments": for each character, compare its two states field by field and report every field that differs, as "before → after":
- place → part "position" (moves toward screen-left or screen-right, comes closer, moves away, enters, leaves the picture);
- the feet step or the body travels over the ground → "walking": "yes", also for a small character far away;
- pose → "body_pose"; head → "head"; eyes looking elsewhere → "eyes"; eyes closing or opening → "blinking": "yes";
- mouth opening, closing or changing shape → "speaking": "yes" and part "mouth";
- arms → "arms"; hands → "hands"; legs and feet → "legs"; the face's emotion → "expression".
When the whole body moves (falls, is pushed, is carried by a step), report that as position, walking or body_pose, not as every limb moving with it.
When a field is "not visible" in either frame, report nothing for it. A field that reads the same in both states is not a change.
"summary" says in two or three sentences what happens across the window.''',
        'retry_hint': 'Compare the state fields one by one.',
    },
}


# Walker (2026-11-18): observer plus a definition of "walking". The misses of the observer prompt were upper-body crops of a person leaving the
# picture and a runner cut by the frame edge; the labels (and the judge) count running and stepping to another place as walking.
WALK_OLD = '"walking", "speaking" and "blinking" say whether that motion is visible between the two frames.'
WALK_NEW = ('"walking" is "yes" when a character travels over the ground on foot between the two frames: walking, running or stepping from one place '
            'to another, including a character who moves into or out of the picture and a small or distant one. Judge it from how the whole body moves '
            'along the ground, so it is "yes" also when the legs are cropped out of the picture. It is "no" when the character only turns, leans, gestures '
            'or shifts in place, is carried, pushed or falling, or sits or rides a vehicle. "speaking" and "blinking" say whether that motion is visible '
            'between the two frames.')
STYLES['walker'] = {
    'label': 'Walker: observer + a definition of walking (running, leaving the picture, cropped legs)',
    'system': STYLES['observer']['system'].replace(WALK_OLD, WALK_NEW),
    'retry_hint': STYLES['observer']['retry_hint'],
}
assert STYLES['walker']['system'] != STYLES['observer']['system']


def _roster_lines(sample):
    if sample.get('no_characters'):
        return ['The user marked this shot as having no characters.']
    people = sample.get('roster') or []
    if not people:
        return ['No character was identified in the first frame; describe anyone visible as "unknown person 1", "unknown person 2" and so on.']
    lines = ['Characters identified in the first frame (box on a 0-1000 grid of the image: left, top, right, bottom). Use exactly these names; '
             'anyone else is "unknown person 1", "unknown person 2" and so on:']
    for person in people:
        lines.append(f"- {person['name']}: box {person['box']} ({person['side']})")
    return lines


def _subtitle_line(sample):
    band = sample.get('subtitle_band') or {}
    if band.get('top') is None:
        return []
    return [f"A burned-in subtitle strip covers {band['top'] * 100:.0f}-{band['bottom'] * 100:.0f}% of the image height; ignore it."]


def _transcript_lines(sample, window):
    start, end = window['frames'][0]['t'], window['frames'][-1]['t']
    lines = [line for line in sample.get('transcript') or [] if line['start'] < end + 0.05 and line['end'] > start - 0.05]
    if not lines:
        return ['Transcript: nobody speaks during these frames (the audio may still have off-screen sounds).']
    out = ['Transcript lines heard during these frames (seconds from the start of the shot; the speaker may be off screen):']
    for line in lines:
        out.append(f"- {line['start']:.2f}-{line['end']:.2f} s, {line.get('speaker') or 'unknown speaker'}: {line['text']}")
    return out


def _first_frame_lines(sample):
    view = sample.get('first_frame_view')
    if not isinstance(view, dict):
        return []
    keep = {key: view.get(key) for key in ('composition', 'camera', 'background_people') if view.get(key)}
    people = []
    for person in view.get('visible_characters') or []:
        if isinstance(person, dict):
            people.append({key: person.get(key) for key in ('character', 'position', 'pose', 'expression', 'gaze', 'action', 'pose_detail') if person.get(key)})
    if people:
        keep['characters'] = people
    return ['Description of the shot\'s first frame (F1) written earlier; use it to know who is who, but report only what changes between frames:',
            json.dumps(keep, ensure_ascii=False)]


def _previous_lines(previous):
    if not previous:
        return []
    last = previous['segments'][-1] if previous.get('segments') else None
    note = {'summary': previous.get('summary'), 'last_segment': last}
    return ['Your answer for the frames just before these (the first frame here is the last frame there):', json.dumps(note, ensure_ascii=False)]


def _camera_lines(sample, window, previous=None):
    """The camera measured from the background alone (people masked out), per pair of the window, as one line."""
    measured = {(p['from'], p['to']): (p.get('camera') or {}).get('move') for p in sample.get('pairs') or []}
    names = [frame['label'] for frame in window['frames']]
    parts = []
    for a, b in zip(names, names[1:]):
        move = measured.get((a, b))
        parts.append(f"{a}→{b} {move if move and move != 'unclear' else 'not measurable'}")
    return ['Camera movement measured from the background alone, people masked out (an automatic measurement): ' + '; '.join(parts) + '. '
            'Use a measured move as the pair\'s camera.move; where it says not measurable, judge the camera yourself. A person who only seems to move '
            'because the camera moved has not changed place.']


CONTEXTS = {
    'frames': {'label': 'Frames + timestamps only', 'tells_speech': False, 'chained': False,
               'build': lambda sample, window, previous: []},
    'roster': {'label': 'Frames + character roster (names, boxes) + subtitle band', 'tells_speech': False, 'chained': False,
               'build': lambda sample, window, previous: [*_roster_lines(sample), *_subtitle_line(sample)]},
    'roster_cam': {'label': 'Roster + subtitle band + the camera measured from the background', 'tells_speech': False, 'chained': False,
                   'build': lambda sample, window, previous: [*_roster_lines(sample), *_subtitle_line(sample), *_camera_lines(sample, window)]},
    'full': {'label': 'Roster + first-frame description + transcript + previous window', 'tells_speech': True, 'chained': True,
             'build': lambda sample, window, previous: [*_roster_lines(sample), *_subtitle_line(sample), *_first_frame_lines(sample),
                                                        *_transcript_lines(sample, window), *_previous_lines(previous)]},
}


def frame_heading(frame):
    return f"{frame['label']} · t={frame['t']:.2f} s"


def build_messages(sample, window, style, context, images, previous=None):
    """The chat messages for one window: system = style, user = context lines then each frame label followed by its image."""
    names = [frame['label'] for frame in window['frames']]
    pairs = ', '.join(f'{a}→{b}' for a, b in zip(names, names[1:]))
    intro = [f"Shot length {sample['duration']:.2f} s. These are frames {names[0]} to {names[-1]} of the shot ({len(names)} frames).",
             *CONTEXTS[context]['build'](sample, window, previous)]
    content = [{'type': 'text', 'text': '\n'.join(intro)}]
    for frame, image in zip(window['frames'], images):
        content.append({'type': 'text', 'text': frame_heading(frame)})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}})
    content.append({'type': 'text', 'text': f'Answer with exactly {len(names) - 1} segments: {pairs}. JSON only.'})
    return [{'role': 'system', 'content': STYLES[style]['system']}, {'role': 'user', 'content': content}]


def prompt_text(messages):
    """The messages with images replaced by a placeholder (stored with each call for the report)."""
    out = []
    for message in messages:
        if isinstance(message['content'], str):
            out.append(f"[{message['role']}]\n{message['content']}")
            continue
        parts = [part['text'] if part['type'] == 'text' else '<image>' for part in message['content']]
        out.append(f"[{message['role']}]\n" + '\n'.join(parts))
    return '\n\n'.join(out)
