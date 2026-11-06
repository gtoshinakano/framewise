"""Contracts shared by the single-cut visual models and their consumers."""
import json
import math
import re

SUBTITLE_INSTRUCTION = '''The footage may contain burned-in Portuguese or English subtitles. Ignore all on-screen writing as visual content. Never quote, describe, translate, recreate, or animate subtitles or other text. Do not infer identities or speech from subtitles. The video is silent to you: do not claim to hear speech, assign voices, or infer dialogue from lip movement. If a person appears to speak, mark it as a visual observation only and describe uncertainty.'''

FRAME_INSTRUCTION = '''Analyze this single frame in detail for recreating its visual composition. Return one JSON object with exactly these keys: composition, setting, lighting, camera, visible_characters, other_visual_details, uncertainties. The first four and other_visual_details and uncertainties must be strings. visible_characters must be a list of objects with keys appearance, position, pose, expression and identity_evidence, all strings. For each visible character, use position to describe their location in the frame and apparent body and face orientation relative to the camera: facing camera, facing away, profile, or three-quarter view, including visible left/right direction when clear. Say unknown when the orientation is obscured. Use pose to describe visible leg and foot placement and any posture consistent with a step or walking; if these details are obscured, say so. A single frame cannot establish whether a person is walking or stationary: explicitly state that walking is unconfirmed, and do not treat a walking-like pose as proof of motion. Describe visible detail only. When supplied character detections have labels_confirmed_by_queue set to true, the user has verified each named box: identify the person in the matching target frame by that name in appearance and identity_evidence, citing the user-verified box. Describe that person’s visible pose directly; do not hedge the name merely because it originated in detection. Unnamed boxes and labels without that flag do not confirm identity. Do not infer identity for other people. ''' + SUBTITLE_INSTRUCTION

VIDEO_INSTRUCTION = '''Analyze this silent video for visual reconstruction. Describe only what the frames support.

Return one complete JSON object, with no Markdown or additional fields. Its keys are overview (string), events (array), and uncertainties (string). Each event has exactly these keys: start_seconds and end_seconds (numbers or null), subject, visual_action, camera, and uncertainty (strings).

In overview, describe the setting, composition, lighting, important objects, and each person's position, orientation, pose, and expression when visible. Say whether the camera appears static or moving and describe any supported movement. Give useful appearance details for unidentified people. For a known character, use their name without repeating their physical appearance or clothing; focus on their position, pose, actions, and expression.

For identities, a named box marked labels_confirmed_by_queue is user-verified for its specified boundary frame. Use that name for the corresponding visible person, and follow them across frames only when continuity supports it. Other names and detections are unverified hints; do not assume who is present or what they do.

In events, describe distinct visible changes in time order. Identify the subject and describe the movement of body parts or objects, including initial and final positions and direction when visible. Include changes in gaze, expression and mouth shape when discernible. Distinguish actual displacement from a walking-like pose. Do not infer hidden movements or paths. If only poses differ, describe the difference without inventing the transition. Combine repeated or continuous motion in one event; use overlapping events for simultaneous actions. Do not repeat events or invent changes to fill time. Use an empty array if nothing changes.

Use approximate start_seconds and end_seconds relative to the start of this clip, within its duration. If timing is unknown, use null for both and explain the limitation; do not use 0 as a placeholder. Each event's camera field describes supported camera movement or says "static" or "unclear". Its uncertainty field states a specific limitation or "none". Summarize only material limitations in uncertainties.

Only in the final window of a cut, check for a clear abrupt switch to a different shot at the end. Exclude the next-shot frames and end events at that boundary. If the transition is unclear, keep the frames and note the uncertainty; ordinary pose or lighting changes are not hard cuts.

Ignore subtitles and all other on-screen writing: do not quote, describe, translate, recreate, or use it to infer identity or action. The video is silent: do not claim to hear speech, identify voices, or infer dialogue or speech from mouth movement.'''


def parse_object(raw):
    raw = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', raw, re.IGNORECASE)
    value = json.loads(fenced.group(1) if fenced else raw)
    if not isinstance(value, dict):
        raise ValueError('Model response must be a JSON object')
    return value


def validate_frame(raw):
    value = parse_object(raw)
    text_fields = ('composition', 'setting', 'lighting', 'camera', 'other_visual_details', 'uncertainties')
    if set(value) != set(text_fields) | {'visible_characters'} or any(not isinstance(value[key], str) for key in text_fields):
        raise ValueError('Invalid frame description fields')
    characters = value['visible_characters']
    fields = {'appearance', 'position', 'pose', 'expression', 'identity_evidence'}
    if not isinstance(characters, list) or any(not isinstance(item, dict) or set(item) != fields or any(not isinstance(item[key], str) for key in fields) for item in characters):
        raise ValueError('Invalid frame character fields')
    return value


def validate_video(raw, duration, episode_start):
    """Accept any JSON value and add episode times only for usable event timing.

    The model's observations and original event order are retained even when a
    field is missing, malformed, or outside the clip. Derived times are only
    provided for finite, ordered intervals within the source window.
    """
    text = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    def invalid_constant(token):
        raise ValueError(f'Invalid JSON constant: {token}')
    value = json.loads(fenced.group(1) if fenced else text, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        return value
    events = value.get('events')
    if not isinstance(events, list):
        return value
    for event in events:
        if not isinstance(event, dict):
            continue
        start, end = event.get('start_seconds'), event.get('end_seconds')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (start, end)):
            continue
        try:
            valid = all(math.isfinite(x) for x in (start, end)) and -.05 <= start <= end <= duration + .05
        except OverflowError:
            valid = False
        if valid:
            event.setdefault('episode_start_seconds', episode_start + max(0, min(start, duration)))
            event.setdefault('episode_end_seconds', episode_start + max(0, min(end, duration)))
    return value
