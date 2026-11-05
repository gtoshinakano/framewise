"""Contracts shared by the single-cut visual models and their consumers."""
import json
import math
import re

SUBTITLE_INSTRUCTION = '''The footage may contain burned-in Portuguese or English subtitles. Ignore all on-screen writing as visual content. Never quote, describe, translate, recreate, or animate subtitles or other text. Do not infer identities or speech from subtitles. The video is silent to you: do not claim to hear speech, assign voices, or infer dialogue from lip movement. If a person appears to speak, mark it as a visual observation only and describe uncertainty.'''

FRAME_INSTRUCTION = '''Analyze this single frame in detail for recreating its visual composition. Return one JSON object with exactly these keys: composition, setting, lighting, camera, visible_characters, other_visual_details, uncertainties. The first four and other_visual_details and uncertainties must be strings. visible_characters must be a list of objects with keys appearance, position, pose, expression and identity_evidence, all strings. Describe visible detail only. A single frame cannot prove motion or identity; state uncertainty. ''' + SUBTITLE_INSTRUCTION

VIDEO_INSTRUCTION = '''Analyze the supplied silent video for animation reconstruction. Return one JSON object with exactly the keys overview, events, uncertainties. overview and uncertainties are strings; events is a chronological list of objects with keys start_seconds, end_seconds, subject, visual_action, camera, uncertainty. Times are numeric seconds relative to the start of this video clip. Describe motion, appearances, expressions, poses, transitions, and camera changes with precise observed timing. When visible, describe a person's mouth movements and their observed start and end, including who is visibly moving their mouth, without claiming that speech is audible or assigning a voice. State uncertainty where frames do not establish an event. Do not infer unseen movement between samples. ''' + SUBTITLE_INSTRUCTION


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
    value = parse_object(raw)
    if set(value) != {'overview', 'events', 'uncertainties'} or not isinstance(value['overview'], str) or not isinstance(value['uncertainties'], str) or not isinstance(value['events'], list):
        raise ValueError('Invalid video analysis fields')
    fields = {'start_seconds', 'end_seconds', 'subject', 'visual_action', 'camera', 'uncertainty'}
    previous = 0.0
    for event in value['events']:
        if not isinstance(event, dict) or set(event) != fields:
            raise ValueError('Invalid video event fields')
        start, end = event['start_seconds'], event['end_seconds']
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start <= end <= duration or start < previous:
            raise ValueError('Video event times must be ordered within the cut')
        if any(not isinstance(event[key], str) for key in ('subject', 'visual_action', 'camera', 'uncertainty')):
            raise ValueError('Invalid video event description')
        previous = start
        event['episode_start_seconds'] = episode_start + start
        event['episode_end_seconds'] = episode_start + end
    return value
