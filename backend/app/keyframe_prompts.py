"""The keyframe analysis prompt: the ``observer`` style with the ``roster`` context of ``multiframe_experiment/prompts.py``.

That pair was the one the experiment kept (PIPELINE_HANDOFF.md section 2): observer is the most precise style and the
roster stops invented names. ``SYSTEM`` is the experiment's text byte for byte (a test pins its hash): the measured
numbers belong to it, so change it only with a new experiment run and a new ``PROMPT_VERSION``.

Stdlib only. The DWPose numbers and the measured camera are never put into the prompt; the app merges them afterwards.
"""
PROMPT_VERSION = 'observer-roster-v1'

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

SYSTEM = '''You compare consecutive keyframes of one anime shot and report exactly what changed between them.
Only report what you can see change between the two images. Small changes count: a blink, a slight head tilt, a mouth opening, a hand shifting a few pixels. Do not report something that stays the same, and do not guess motion that the frames do not show. When a body part is hidden or too small to judge, say "unclear" or leave it out; "no" and an empty "changes" list are valid answers.
''' + COMMON_RULES + '''
"walking", "speaking" and "blinking" say whether that motion is visible between the two frames. "changes" lists every other part that visibly changed, each with what it looked like in the first frame and what it looks like in the second. "summary" says in two or three sentences what happens across the window.'''
RETRY_HINT = 'Report only visible changes.'


def roster_lines(sample):
    """Who the model may name. ``sample['roster']``: ``[{name, box (0-1000 grid), side, first_seen?}]`` left to right.

    A person named on the Target frames tab who is not in the first keyframe carries ``first_seen`` (their first
    keyframe's label) and the box from that frame.
    """
    if sample.get('no_characters'):
        return ['The user marked this shot as having no characters.']
    people = sample.get('roster') or []
    if not people:
        return ['No character was identified in the first frame; describe anyone visible as "unknown person 1", "unknown person 2" and so on.']
    lines = ['Characters identified in the first frame (box on a 0-1000 grid of the image: left, top, right, bottom). Use exactly these names; '
             'anyone else is "unknown person 1", "unknown person 2" and so on:']
    for person in people:
        seen = f", first seen in {person['first_seen']}" if person.get('first_seen') and person['first_seen'] != 'F1' else ''
        lines.append(f"- {person['name']}: box {person['box']} ({person['side']}{seen})")
    return lines


def subtitle_line(sample):
    band = sample.get('subtitle_band') or {}
    if band.get('top') is None:
        return []
    return [f"A burned-in subtitle strip covers {band['top'] * 100:.0f}-{band['bottom'] * 100:.0f}% of the image height; ignore it."]


def frame_heading(frame):
    return f"{frame['label']} · t={frame['t']:.2f} s"


def build_messages(sample, window, images):
    """The chat messages for one window: system prompt, then the roster lines and each frame label followed by its image.

    ``sample``: ``{duration, no_characters, roster, subtitle_band}``; ``images``: base64 JPEGs in window order.
    """
    names = [frame['label'] for frame in window['frames']]
    pairs = ', '.join(f'{a}→{b}' for a, b in zip(names, names[1:]))
    intro = [f"Shot length {sample['duration']:.2f} s. These are frames {names[0]} to {names[-1]} of the shot ({len(names)} frames).",
             *roster_lines(sample), *subtitle_line(sample)]
    content = [{'type': 'text', 'text': '\n'.join(intro)}]
    for frame, image in zip(window['frames'], images):
        content.append({'type': 'text', 'text': frame_heading(frame)})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}})
    content.append({'type': 'text', 'text': f'Answer with exactly {len(names) - 1} segments: {pairs}. JSON only.'})
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]


def prompt_text(messages):
    """The messages with images replaced by a placeholder (stored with each window)."""
    out = []
    for message in messages:
        if isinstance(message['content'], str):
            out.append(f"[{message['role']}]\n{message['content']}")
            continue
        parts = [part['text'] if part['type'] == 'text' else '<image>' for part in message['content']]
        out.append(f"[{message['role']}]\n" + '\n'.join(parts))
    return '\n\n'.join(out)
