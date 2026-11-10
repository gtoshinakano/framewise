"""Contracts shared by the single-cut visual models and their consumers."""
import json
import math
import re

def subtitle_note(language, boxes, space):
    """Closing sentence for visual prompts: the one subtitle language and where the strip sits, if known."""
    if not language and not boxes:
        return ''
    parts = ['\nSubtitles:']
    if language:
        parts.append(f'every burned-in subtitle in this video is in {language}; that is certain.')
    if boxes:
        parts.append(f'The subtitle strip is {json.dumps(boxes)} in {space} (detected for the whole video; edges may be imprecise).')
    parts.append('Treat everything in that strip as subtitles and never read, describe, translate or reproduce it.')
    return ' '.join(parts)


SUBTITLE_INSTRUCTION = '''Burned-in subtitles, if any, are always in one single language for the whole video; the language and strip are given below when known. Ignore subtitles and all other on-screen writing: never quote, describe, translate or recreate it, and never infer identity, speech or action from it. A mouth shape is a visual observation only.'''

OUTFIT_RULE = '''Appearance comes only from pixels you can see in this image. Series notes are background and are NOT evidence of what anyone wears or looks like here: never copy garments, hair or accessories from them or from what is usual for the character. Write a garment, hair detail or accessory only if it is visible in this frame. A part that is cropped, covered, out of frame, turned away or too small is written as "not visible" (for example "lower body not visible"); a character who is only partly in frame gets only the visible parts described.'''

TRUST_RULE = '''The user has verified every marked character's identity: it is certain. Never question, hedge or explain it. Say "unclear" for a feature only when it is hidden or too small to see. Do not discuss walking or motion; state posture only.'''

FRAME_INSTRUCTION = f'''You write production notes for ONE still frame of an anime, so an artist can rebuild the shot with AI. Describe only what is visible. Write compact noun phrases, not full sentences; outfit and hair detail is the exception and should be thorough. Never repeat a fact in another field and never restate a caveat.

Return one JSON object with exactly these keys, in this order:
- composition: shot size, angle, framing, and where the main elements sit (max 2 short sentences).
- setting: place and key background props (1 sentence).
- lighting: direction, quality, colour and time of day (1 sentence).
- camera: angle, lens feel and depth of field of this still (1 short sentence). One frame cannot show movement.
- visible_characters: one entry for every marked character listed in the message, in that order, and nobody else. Each entry has: box (the reference, e.g. B1), appearance (the visible outfit, garment by garment from head to toe: type, colour, pattern, cut or fit, material or sheen, accessories, footwear; plus hair colour, length and style, eye colour and any distinguishing marks; "not visible" for each part that is out of frame, hidden or too small; never the name), position (where in the frame, plus body and face orientation: facing camera, away, profile or three-quarter, left/right when clear), pose (posture, hands, and legs/feet only when visible), expression (eyes, brow, mouth, gaze direction), action (what they are doing, holding or reacting to).
- background_people: every person who is NOT marked, one item per person or similar group separated by semicolons: count, position in frame, outfit (garment types, colours, patterns, accessories), hair, and what they do. Describe outfits as carefully as for marked characters, but skip expression and pose nuance. "none" when there are no others.
- notes: only material details the fields above cannot hold (objects, effects, art-style traits), otherwise "none".

{TRUST_RULE}

{OUTFIT_RULE}

''' + SUBTITLE_INSTRUCTION

LAST_FRAME_INSTRUCTION = f'''You write production notes for the LAST frame of an anime cut, as CHANGES from its first frame. The first frame is already recorded and is given in the message: never repeat anything it already states, and never describe outfits again. Describe only what is visible. Write short noun phrases.

Return one JSON object with exactly these keys, in this order:
- composition, setting, lighting, camera: "unchanged", or only what differs from the first frame (shot size, angle or framing change; new light; set or prop change; where the camera ended up).
- visible_characters: one entry for every marked character listed in the message, in that order, and nobody else. Each entry has: box, appearance, position, pose, expression, action.
  - A character who was in the first frame: write "unchanged" for every field that still matches the first frame, and only the new state otherwise. appearance stays "unchanged" unless garments or hair visibly differ in this frame, and then describe only the new garments or hair. A part that was visible in the first frame but is cropped or hidden now is "not visible", not "unchanged"; a part that was not visible before is described only if it is visible now. A pose or expression worded differently but not visibly different is "unchanged". position changes only if the character moved relative to the frame or to others.
  - A character marked new (not in the first frame): describe every field in full, including the visible outfit as for a first frame ("not visible" for parts out of frame).
- background_people: "unchanged", or items separated by semicolons for unmarked people who entered, left, moved or look different; describe any newcomer's visible outfit (garment types, colours, patterns, accessories) and hair in full.
- notes: new material details only, otherwise "none".

{TRUST_RULE}

{OUTFIT_RULE}

''' + SUBTITLE_INSTRUCTION

PERSON_TEXT = ('appearance', 'position', 'pose', 'expression', 'action')
FRAME_TEXT = ('composition', 'setting', 'lighting', 'camera')
UNCHANGED = 'unchanged'


def frame_schema(handles):
    """JSON schema (a llama.cpp grammar) for one frame answer: fixed key order, a box from ``handles`` per marked character."""
    text = lambda name: {'type': 'string'}  # no maxLength: it expands into a large grammar; the prompt keeps answers short and max_tokens caps them
    person = {'type': 'object', 'additionalProperties': False, 'required': ['box', *PERSON_TEXT],
              'properties': {'box': {'enum': list(handles)}, **{name: text(name) for name in PERSON_TEXT}}}
    properties = {**{name: text(name) for name in FRAME_TEXT},
                  'visible_characters': {'type': 'array', 'items': person, 'minItems': len(handles), 'maxItems': len(handles)},
                  'background_people': text('background_people'), 'notes': text('notes')}
    return {'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def resolve_frame(raw, people, *, changes=False):
    """Validate a frame answer and replace its box references with the real records.

    ``people`` are the marked characters sent with the prompt, each with ``ref`` (the handle the model
    saw), ``character_context_id``, ``name`` and, for a last frame in changes mode,
    ``new``. The model only sees handles. This gives every entry the real IDs, drops entries that
    cite a handle that was never offered, folds duplicates into one, and adds an entry for a person
    the model skipped, so no stored reference points at nothing and no marked person goes missing.
    """
    value = parse_object(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError('Model response must be a JSON object')
    entries = value.get('visible_characters')
    if not isinstance(entries, list):
        raise ValueError('Invalid frame character list')
    blank = UNCHANGED if changes else ''
    by_ref = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ref = _text(entry.get('box')).upper()
        if not any(person['ref'] == ref for person in people):
            continue  # a handle that was never offered
        seen = by_ref.setdefault(ref, {name: '' for name in PERSON_TEXT})
        for name in PERSON_TEXT:
            if not seen[name]:
                seen[name] = _text(entry.get(name))
    characters = []
    for person in people:
        details = by_ref.get(person['ref'], {name: '' for name in PERSON_TEXT})
        if changes and not person.get('new'):
            details = {name: details[name] or UNCHANGED for name in PERSON_TEXT}
        entry = {'character': person['name'], 'character_context_id': person['character_context_id'], **{name: details[name] for name in PERSON_TEXT}}
        if changes:
            entry['new_in_frame'] = bool(person.get('new'))
        characters.append(entry)
    result = {name: _text(value.get(name)) or blank for name in FRAME_TEXT}
    result['visible_characters'] = characters
    result['background_people'] = _text(value.get('background_people')) or ('none' if not changes else UNCHANGED)
    result['notes'] = _text(value.get('notes')) or 'none'
    result['mode'] = 'changes' if changes else 'full'
    return result


def person_context(people, frame_label, changes=False):
    """The 'Marked characters' block of the user message: handles, names and boxes on a 0-1000 grid."""
    if not people:
        return 'Marked characters: none. Put everyone in background_people and leave visible_characters empty.'
    lines = [f'Marked characters in the {frame_label} (box = x1,y1,x2,y2 on a 0-1000 grid over the whole frame):']
    for person in people:
        tag = (' [new: not in the first frame]' if person.get('new') else ' [was in the first frame]') if changes else ''
        lines.append(f"- {person['ref']}: {person['name']}{tag}" + (f", box {person['box']}" if person.get('box') else ''))
    return '\n'.join(lines)


def previous_notes(content):
    """The first frame's stored result as the compact notes a last-frame prompt compares against."""
    drop = {'mode', 'frame_time', 'subtitle_regions', 'based_on_initial_frame', 'left_frame'}
    notes = {key: value for key, value in content.items() if key not in drop and key != 'visible_characters'}
    notes['visible_characters'] = [{key: value for key, value in item.items() if key in ('character', *PERSON_TEXT)}
                                   for item in content.get('visible_characters', []) if isinstance(item, dict)]
    return notes


def full_last_frame(initial, last):
    """A last frame in changes mode as full per-character states, for readers that compare two frames.

    "unchanged" is replaced by the first frame's value; a first-frame character without an entry is
    left out, which readers take as having left the frame. Other results pass through untouched.
    """
    if not isinstance(last, dict) or last.get('mode') != 'changes' or not isinstance(initial, dict):
        return last
    def keep(new, old):
        return old if not _text(new) or _text(new).lower() == UNCHANGED else new
    before = {item.get('character_context_id'): item for item in initial.get('visible_characters', []) if isinstance(item, dict) and item.get('character_context_id')}
    people = []
    for item in last.get('visible_characters', []):
        if not isinstance(item, dict):
            continue
        base = before.get(item.get('character_context_id')) or {}
        people.append({**item, **{name: keep(item.get(name), base.get(name, '')) for name in PERSON_TEXT}})
    return {**last, **{name: keep(last.get(name), initial.get(name, '')) for name in FRAME_TEXT}, 'visible_characters': people}


VIDEO_INSTRUCTION = '''Analyze this video clip for visual reconstruction. Describe only what the frames support.

Return one complete JSON object, with no Markdown or additional fields. Its keys are overview (string), events (array) and uncertainties (string).

overview: at most 3 sentences. Say whether the camera is static or moving and how, give the overall action, and give each visible person's starting position and orientation in a few words. Setting, lighting and outfits are recorded elsewhere: do not repeat them. Describe an unidentified person briefly only when it helps tell people apart.

Identities: a named box marked labels_confirmed_by_queue is verified by the user for its boundary frame, so that name is certain; use it for that person and follow them across frames while continuity supports it. Other names and detections are unverified hints. Refer to a known character by name only and never describe their clothing or hair.

events: distinct visible changes in time order, at most 6. Each event has start_seconds and end_seconds (numbers relative to the start of this clip and within its duration; null for both if timing is unknown, never 0 as a placeholder), subject, and visual_action (one sentence: which body part or object moves, from where to where and in which direction; changes in gaze, expression and mouth shape when discernible; a mouth shape is only a visual observation). Add camera only when the camera moves during that event, and uncertainty only for a specific limitation; omit both otherwise. Combine repeated or continuous motion in one event and use overlapping events for simultaneous actions. Report displacement only when a position actually changes; otherwise state posture. If only poses differ, describe the difference without inventing the transition or hidden paths. Use an empty array if nothing changes.

uncertainties: material limitations only, otherwise "none".

Ignore subtitles and all other on-screen writing: do not quote, describe, translate, recreate or use it to infer identity or action.

Example shape: {"overview": "Static camera. Mia stands left, facing right.", "events": [{"start_seconds": 0.0, "end_seconds": 1.5, "subject": "Mia", "visual_action": "Raises her right hand from her side to shoulder height and turns her head to the right."}], "uncertainties": "none"}'''


def parse_object(raw):
    raw = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', raw, re.IGNORECASE)
    value = json.loads(fenced.group(1) if fenced else raw)
    if not isinstance(value, dict):
        raise ValueError('Model response must be a JSON object')
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
