"""Contracts shared by the single-cut visual models and their consumers."""
import json
import math
import re

from . import pose_detail, unregistered

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


def series_note(context):
    """Series notes worth sending with a frame or clip: selected context items except character notes, and the user's extra text, never box data."""
    if not isinstance(context, dict):
        return ''
    lines = [f"- {item.get('kind', 'note')} {item.get('title', '')}: {str(item.get('description', ''))[:300]}" for item in context.get('selected_context') or [] if isinstance(item, dict) and item.get('kind') != 'character']  # character notes carry outfits that get copied into the answer; the marked list has the names
    extra = str(context.get('additional_context') or '').strip()
    return ('Series notes (background only, not evidence of what is visible):\n' + '\n'.join(lines) if lines else '') + (('\n' if lines else '') + 'Extra notes: ' + extra[:1000] if extra else '')


SUBTITLE_INSTRUCTION = '''Burned-in subtitles, if any, are always in one single language for the whole video; the language and strip are given below when known. Ignore subtitles and all other on-screen writing: never quote, describe, translate or recreate it, and never infer identity, speech or action from it. A mouth shape is a visual observation only.'''

OUTFIT_RULE = '''Appearance comes only from pixels you can see in this image. Series notes are background and are NOT evidence of what anyone wears or looks like here: never copy garments, hair or accessories from them or from what is usual for the character. Write a garment, hair detail or accessory only if it is visible in this frame. A part that is cropped, covered, out of frame, turned away or too small is written as "not visible" (for example "lower body not visible"); a character who is only partly in frame gets only the visible parts described.'''

TRUST_RULE = '''The user has verified every marked character's identity: it is certain. Never question, hedge or explain it. Say "unclear" for a feature only when it is hidden or too small to see. State what the still shows without hedging: a stride caught mid-step is "walking", an open mouth is "mouth open"; never add that motion or speech cannot be confirmed from one frame.'''

NAME_RULE = '''Names: outside a character's own entry, always refer to a marked character by the exact name given in the message (in composition, props, notes, and in another character's gaze or action), never as "a girl", "the male figure" or a box reference. Refer to an unmarked person by a short visible label such as "girl with the red umbrella".'''

FRAME_CONTRACT = 'frame-v2'
POSE_RULES = pose_detail.POSE_RULES.replace('\n', '\n    ')

FRAME_INSTRUCTION = f'''You write production notes for the FIRST frame of an anime cut, so the shot can be rebuilt as a still image with an AI image model and then animated. Describe only what is visible. Write compact noun phrases, not full sentences; outfit and hair detail is the exception and should be thorough. Never repeat a fact in another field and never restate a caveat.

Return one JSON object with exactly these keys, in this order:
- composition: shot size, angle, framing, and where the main elements sit, naming the marked characters (max 2 short sentences).
- setting: place, key background props, weather and time of day when visible (1 sentence).
- lighting: direction, quality, colour and the dominant colour palette (1 sentence).
- camera: angle, lens feel and depth of field of this still (1 short sentence). One frame cannot show camera movement.
- mood: the emotional tone of the whole image in 2 to 6 words (for example "tense, quiet standoff" or "bright, playful").
- visible_characters: one entry for every marked character listed in the message, in that order, and nobody else. Each entry has:
  - box: the reference, e.g. B1.
  - appearance: the visible outfit, garment by garment from head to toe: type, colour, pattern, cut or fit, material or sheen, accessories, footwear; plus hair colour, length and style, eye colour and any distinguishing marks; "not visible" for each part that is out of frame, hidden or too small; never the name.
  - position: where in the frame, depth (foreground, midground, background), and body and face orientation: facing camera, away, profile or three-quarter, left/right when clear.
  - pose: posture, hands, and legs/feet only when visible.
  - expression: the face as drawn: eyes, brows, mouth (open or closed), blush, sweat or tears.
  - emotion: the feeling that face and body read as, with its strength, in 1 to 4 words (for example "mild worry", "furious", "shy happiness", "calm, neutral"); "not readable" when the face is hidden.
  - gaze: where the eyes look, naming the marked character or object looked at (for example "at Mai Sakurajima", "down at the box", "at the camera"); "not visible" when the eyes are hidden.
  - action: starts with the activity the still shows, chosen from standing, sitting, crouching, kneeling, lying, walking, running, jumping, leaning, or another single verb, then what they are doing, holding or reacting to and with whom (for example "sitting, talking to Mai Sakurajima across the desk").
  - {POSE_RULES}
- background_people: EVERY person who is not in the marked list, wherever they are, including someone prominent or in the foreground (the key name is historical). One item per person or similar group separated by semicolons: count, position in frame, outfit (garment types, colours, patterns, accessories), hair, and what they do. Describe outfits as carefully as for marked characters, but skip expression and pose nuance. "none" only when the marked characters are the only people in the image.
- props: the objects that matter for rebuilding the shot and who holds or uses them, separated by semicolons; "none" when there are none.
- notes: only material details the fields above cannot hold (visual effects, art-style traits), otherwise "none".

{TRUST_RULE}

{NAME_RULE}

{OUTFIT_RULE}

''' + SUBTITLE_INSTRUCTION

# The stored per-character text of a first frame. Results written before emotion and gaze existed lack those two.
PERSON_TEXT = ('appearance', 'position', 'pose', 'expression', 'emotion', 'gaze', 'action')
# Frame-wide text, in answer order. ``mood`` and ``props`` are absent from older results.
FRAME_TEXT = ('composition', 'setting', 'lighting', 'camera', 'mood')
FRAME_TAIL = ('background_people', 'props', 'notes')
UNCHANGED = 'unchanged'


POSE_KEY = 'pose_detail'


def frame_schema(handles):
    """JSON schema (a llama.cpp grammar) for one frame answer: fixed key order, a box from ``handles`` per marked character."""
    text = lambda name: {'type': 'string'}  # no maxLength: it expands into a large grammar; the prompt keeps answers short and max_tokens caps them
    person = {'type': 'object', 'additionalProperties': False, 'required': ['box', *PERSON_TEXT, POSE_KEY],
              'properties': {'box': {'enum': list(handles)}, **{name: text(name) for name in PERSON_TEXT}, POSE_KEY: pose_detail.json_schema(handles)}}
    # With no marked characters the list must stay empty. Leave out the per-person schema: its empty box
    # enum is an invalid grammar (llama-server answers HTTP 500 and the answer fell back to plain JSON).
    characters = ({'type': 'array', 'items': person, 'minItems': len(handles), 'maxItems': len(handles)} if handles
                  else {'type': 'array', 'maxItems': 0})
    properties = {**{name: text(name) for name in FRAME_TEXT},
                  'visible_characters': characters,
                  **{name: text(name) for name in FRAME_TAIL}}
    return {'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def resolve_frame(raw, people):
    """Validate a frame answer and replace its box references with the real records.

    ``people`` are the marked characters sent with the prompt, each with ``ref`` (the handle the model
    saw), ``character_context_id`` and ``name``. The model only sees handles. This gives every entry the
    real IDs, drops entries that cite a handle that was never offered, folds duplicates into one, and
    adds an entry for a person the model skipped, so no stored reference points at nothing and no
    marked person goes missing.
    """
    value = parse_object(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError('Model response must be a JSON object')
    entries = value.get('visible_characters')
    if not isinstance(entries, list):
        raise ValueError('Invalid frame character list')
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
        if not isinstance(seen.get(POSE_KEY), dict) and isinstance(entry.get(POSE_KEY), dict):
            seen[POSE_KEY] = entry[POSE_KEY]  # a repeated handle keeps the first pose block
    characters = []
    for person in people:
        details = by_ref.get(person['ref'], {name: '' for name in PERSON_TEXT})
        characters.append({'character': person['name'], 'character_context_id': person['character_context_id'], **{name: details[name] for name in PERSON_TEXT},
                           POSE_KEY: pose_detail.resolve(details.get(POSE_KEY), person, people),
                           **({'unregistered': True} if unregistered.is_card(person['name'] or '') else {})})
    result = {name: _text(value.get(name)) for name in FRAME_TEXT}
    result['visible_characters'] = characters
    for name in FRAME_TAIL:
        result[name] = _text(value.get(name)) or 'none'
    result['mode'] = 'full'
    result['contract'] = FRAME_CONTRACT
    return result


def person_context(people, frame_label):
    """The 'Marked characters' block of the user message: handles, names and boxes on a 0-1000 grid."""
    if not people:
        return 'Marked characters: none. Put everyone in background_people and leave visible_characters empty.'
    lines = [f'Marked characters in the {frame_label} (box = x1,y1,x2,y2 on a 0-1000 grid over the whole frame):']
    for person in people:
        lines.append(f"- {person['ref']}: {person['name']}" + (f", box {person['box']}" if person.get('box') else ''))
    if unregistered.present(person['name'] for person in people):
        lines.append(unregistered.RULE)
    return '\n'.join(lines)


def full_last_frame(initial, last):
    """A stored last frame in changes mode as full per-character states, for readers of older results.

    Last frames are no longer analyzed; cuts analyzed before that keep theirs. "unchanged" is replaced
    by the first frame's value; a first-frame character without an entry is left out, which readers
    take as having left the frame. Other results pass through untouched.
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
        people.append({**item, **{name: keep(item.get(name), base.get(name, '')) for name in PERSON_TEXT if name in item or name in base}})
    return {**last, **{name: keep(last.get(name), initial.get(name, '')) for name in FRAME_TEXT if name in last or name in initial}, 'visible_characters': people}


# -- video analysis: motion direction for one window of a cut -----------------------------------

VIDEO_CONTRACT = 'direction-v3'
CAMERA_MOVES = ('static', 'pan left', 'pan right', 'tilt up', 'tilt down', 'zoom in', 'zoom out', 'push in', 'pull out', 'tracking', 'handheld', 'other')
VIDEO_PERSON_TEXT = ('activity', 'travel', 'motion', 'speaking', 'emotion', 'end')
EVENT_KINDS = ('body', 'gaze', 'mouth', 'expression', 'object', 'travel', 'camera')

VIDEO_INSTRUCTION = '''You watch one clip of an anime cut and write motion directions, so the shot can be re-created by animating its opening still with an image-to-video model. A sharper model has already described the opening still: for each listed character you are given the state, facing and gaze it read from that one image. Your job is what a single still cannot show: what the characters really do, what changes and when. Do not describe outfits, hair, setting or lighting. Report only what the frames show.

Return one complete JSON object, with no Markdown or additional fields. Its keys are overview, camera, characters, events and uncertainties.

overview: 1 or 2 sentences saying what happens in this clip, using the given character names.

camera: an object with movement and detail. movement is exactly one of: static, pan left, pan right, tilt up, tilt down, zoom in, zoom out, push in, pull out, tracking, handheld, other. detail gives the speed, what the camera follows and how the framing ends, or "none". Use static when the viewpoint does not change, even if everything in the picture moves.

First-frame states are read from one still, so they can be wrong about anything that needs motion: an upright figure may be mid-stride, a seated one may be getting up. Decide every character's state from the frames of this clip. When the clip shows a different state than the first frame gave, write the state you see; when it is the same, write the same. Gaze, head turns, mouth and hand movements are only reported when you see them happen.

characters: one entry for every listed character who is visible at any moment of this clip, in the listed order, then one entry for each other clearly visible person. Each entry has:
- name: the exact name from the list. Never give a listed name to anyone else, and never invent a name. For an unlisted person write a short label made only of what this clip shows about them (what they wear or hold, or the visible body part when only part of them is in view). List only people you can see in these frames.
- activity: the state of the body during the clip, in one or two words you can see: standing, sitting, crouching, kneeling, lying, walking, running, jumping, turning, bowing, reaching, eating, writing, crying, fighting, or another verb. A person who changes state writes both in order, for example "sitting, then standing". Never guess it from the story.
- travel: how the person moves through the picture: "none" when they stay in place, otherwise "left", "right", "toward camera", "away from camera", "up", "down", "enters from left", "enters from right", "exits left" or "exits right". Directions are the image's left and right.
- motion: one sentence on the visible body movement: which part moves, from where to where (a head turn, a raised arm, a nod, a step, hair or clothes moving). "holds still" when nothing moves.
- gaze: one sentence on where the eyes and the head go during the clip and when they shift, naming who or what is looked at, or "holds" when the gaze stays where the first frame had it.
- speaking: "yes" when the mouth opens and closes as in speech, "no" when the mouth stays shut, "unclear" when the mouth is hidden, turned away or too small.
- emotion: the feeling the face and body show over the clip. When the list gives a character's emotion in the first frame, start from it (it was read from a sharper image) and say whether it stays or how it changes: "stays calm", "worried, then relieved". "not readable" when the face is hidden throughout.
- objects: an array of short strings, one per object the person touches, holds, moves, points at or looks at, saying what happens to it; an empty array when there is none.
- end: where and how the person is in the final frames of this clip (place in the picture, posture, facing), or "left frame" when they are no longer visible at the end.

events: the distinct visible changes in time order, at most 6. Each event has start_seconds and end_seconds (numbers relative to the start of this clip and within its duration; null for both if timing is unknown, never 0 as a placeholder), subject (a name used in characters, or "camera"), kind (exactly one of body, gaze, mouth, expression, object, travel, camera) and visual_action (one sentence: who or what moves, from where to where and in which direction). Give a gaze shift, a mouth movement and a handled object their own events when they are discernible. Combine repeated or continuous motion in one event and use overlapping events for simultaneous actions. Report travel only when a position actually changes. Use an empty array only if nothing at all changes.

uncertainties: material limitations only, otherwise "none".

Names: the listed characters were identified by the user in the first frame of the cut, so those names are certain. Follow each one through the clip by place and look, use the exact name everywhere, and never call a listed character "a girl", "a boy" or "a person". The video itself determines where each character ends; no end-frame box is supplied.

Speech: when speech times are given they come from the audio track. A listed speaker who is on screen is probably the one talking then; check the mouth and report what you see. A voice can also come from off screen or be a thought, so a closed mouth during speech is a valid observation. "another voice" is someone who is not a listed character: never give that voice's owner a name, and label an unlisted person only by what you see.

Ignore subtitles and all other on-screen writing: do not quote, describe, translate, recreate or use it to infer identity or action.

Example shape: {"overview": "Mia turns to Ken and starts talking while he keeps walking away.", "camera": {"movement": "static", "detail": "none"}, "characters": [{"name": "Mia", "activity": "standing", "travel": "none", "motion": "Turns her head from the window to the right and raises her right hand to shoulder height.", "gaze": "Looks at the window, then shifts to Ken after the head turn.", "speaking": "yes", "emotion": "calm, then annoyed", "objects": [], "end": "left of frame, facing right, hand raised"}], "events": [{"start_seconds": 0.0, "end_seconds": 1.5, "subject": "Mia", "kind": "gaze", "visual_action": "Turns her head to the right and shifts her gaze from the window to Ken."}], "uncertainties": "none"}'''


def _zone(box):
    """A 0-1000 grid box as a plain place in the picture."""
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)):
        return ''
    x = (box[0] + box[2]) / 2
    return 'left' if x < 333 else 'right' if x > 667 else 'centre'


def _clip_words(text, limit):
    words = _text(text).split()
    return ' '.join(words[:limit]).rstrip(' ,;.') + ('…' if len(words) > limit else '')


def video_roster(first_people, last_people, initial=None):
    """Who the video model should follow: the marked characters of the first frame.

    ``first_people`` are the marked characters of the first frame. ``last_people`` is retained as
    an optional compatibility input for older callers, but the current pipeline supplies None.
    The first-frame result helps the video model follow named characters without being asked to
    describe them again.

    The video model reports whether someone remains in frame or leaves; the current workflow does
    not run character detection on the last frame.
    """
    described = {item.get('character_context_id'): item for item in (initial or {}).get('visible_characters', []) if isinstance(item, dict)}
    roster, seen = [], {}
    for label, people in (('start', first_people), ('end', last_people)):
        for person in people or []:
            key = person['character_context_id']
            if key not in seen:
                notes = described.get(key) or {}
                seen[key] = {'name': person['name'], 'character_context_id': key, 'at_start': False if first_people is not None else None, 'at_end': None,
                             **{name: _text(notes.get(name)) for name in ('position', 'pose', 'emotion') if _text(notes.get(name))},
                             **({'pose_detail': notes['pose_detail']} if isinstance(notes.get('pose_detail'), dict) and notes['pose_detail'].get('body_pose') else {})}
                roster.append(seen[key])
            seen[key]['at_' + label] = True
            if person.get('box') and label + '_box' not in seen[key]:
                seen[key][label + '_box'] = person['box']
    return roster


def _pose_facts(pose):
    """A character's first-frame pose block as the short clause the video prompt reads."""
    small = pose_detail.compact(pose) if isinstance(pose, dict) else None
    if not small:
        return ''
    labels = (('state', 'state'), ('torso', 'torso'), ('head', 'head'), ('eyes', 'eyes'), ('looking_at', 'looking at'))
    return ', '.join(f'{label} {small[key]}' for key, label in labels if small.get(key))


def roster_note(roster, initial=None):
    """The roster as the 'Characters' block of the video prompt."""
    lines = []
    if roster:
        lines.append('Characters in this cut (identified by the user; boxes are x1,y1,x2,y2 on a 0-1000 grid over the frame):')
    for person in roster:
        parts = []
        if person.get('at_start'):
            place = _clip_words(person.get('position'), 16) or _zone(person.get('start_box'))
            start = 'first frame: ' + (place or 'in frame') + (f", box {person['start_box']}" if person.get('start_box') else '')
            if person.get('pose'):
                start += '; ' + _clip_words(person['pose'], 12)
            parts.append(start)
            facts = _pose_facts(person.get('pose_detail'))
            if facts:
                parts.append('first-frame reading (from one still, may miss motion): ' + facts)
            if person.get('emotion') and person['emotion'].lower().rstrip('.') != 'not readable':
                parts.append('emotion in the first frame: ' + _clip_words(person['emotion'], 8))
        elif person.get('at_start') is False:
            parts.append('not marked in the first frame')
        if person.get('at_end'):
            parts.append('last frame of the cut: in frame' + (f" ({_zone(person['end_box'])}, box {person['end_box']})" if person.get('end_box') else '')
                         + (', so this character probably enters during the cut' if person.get('at_start') is False else ''))
        lines.append(f"- {person['name']}: " + '. '.join(parts) + '.')
    if not roster:
        lines.append('Characters in this cut: nobody was identified. Give each visible person a short visual label.')
    if isinstance(initial, dict):
        others = _text(initial.get('background_people'))
        if others and others.lower().rstrip('.') != 'none':
            lines.append('Unnamed people in the first frame: ' + _clip_words(others, 50))
        if _text(initial.get('composition')):
            lines.append('Opening shot: ' + _clip_words(initial['composition'], 40))
    return '\n'.join(lines)


def speech_note(segments, start, end, roster=()):
    """Who is heard during ``start``-``end`` (episode seconds), as clip-relative spans without the words.

    Only characters of the ``roster`` are named. Any other speaker is "another voice", so the model
    cannot pin a transcript name on an unmarked person it happens to see.
    """
    listed = {person['name'] for person in roster or []}
    turns = []
    for segment in sorted((item for item in segments or [] if isinstance(item, dict)), key=lambda item: item.get('start_time') or 0):
        if segment.get('deleted') or not _text(segment.get('text')):
            continue
        try:
            begin, finish = max(start, float(segment['start_time'])), min(end, float(segment['end_time']))
        except (KeyError, TypeError, ValueError):
            continue
        if finish <= begin:
            continue
        speaker = _text(segment.get('character_name'))
        speaker = speaker if speaker in listed else 'another voice'
        if turns and turns[-1][0] == speaker and begin - turns[-1][2] <= .6:
            turns[-1][2] = max(turns[-1][2], finish)
        else:
            turns.append([speaker, begin, finish])
    if not turns:
        return 'Speech heard in this clip: none in the transcript.'
    return 'Speech heard in this clip (from the audio track, seconds from the clip start): ' + '; '.join(
        f'{begin - start:.1f}-{finish - start:.1f} {speaker}' for speaker, begin, finish in turns) + '.'


# Added to the video prompt when the clip carries its soundtrack (a model that hears audio, such as LFM3-Omni).
AUDIO_NOTE = ('This clip includes its soundtrack. Use the sound only to judge who is speaking and when (a mouth that moves with a voice, '
              'an off-screen voice, a voice over a closed mouth) and when a sound marks an event. Never quote, translate or summarise the words, '
              'and do not describe music or ambience.')


def earlier_note(previous):
    """What the previous window of the same cut reported, so the next one continues from it."""
    analysis = previous.get('analysis') if isinstance(previous, dict) else None
    if not isinstance(analysis, dict):
        return ''
    places = [f"- {person.get('character') or person.get('name')}: {_clip_words(person['end'], 16)}"
              for person in analysis.get('characters') or [] if isinstance(person, dict) and _text(person.get('end'))]
    before = _clip_words(analysis.get('overview'), 40)
    if not before and not places:
        return ''
    return ('Background, not to be repeated in your answer: this clip continues the same cut.'
            + (f' Just before it: {before}' if before else '')
            + ('\nWhere everyone is when this clip starts:\n' + '\n'.join(places) if places else '')
            + '\nDescribe only what happens in this clip.')


def _match_name(name, roster):
    """The roster entry a model-written name means: exact, or the only one sharing all of its words."""
    wanted = _text(name).casefold()
    if not wanted:
        return None
    for person in roster:
        if person['name'].casefold() == wanted:
            return person
    words = set(re.findall(r'\w+', wanted))
    matches = [person for person in roster if words and words <= set(re.findall(r'\w+', person['name'].casefold()))]
    return matches[0] if len(matches) == 1 else None


_KIND_ALIASES = {'movement': 'body', 'motion': 'body', 'gesture': 'body', 'posture': 'body', 'head': 'body', 'look': 'gaze', 'eyes': 'gaze', 'eye': 'gaze',
                 'speech': 'mouth', 'speaking': 'mouth', 'talk': 'mouth', 'face': 'expression', 'emotion': 'expression', 'prop': 'object', 'item': 'object',
                 'move': 'travel', 'walk': 'travel', 'position': 'travel'}


def event_kind(value):
    """An event's kind as one of EVENT_KINDS; whatever the model wrote that is not one becomes "other"."""
    word = _text(value).lower().replace(' ', '_')
    word = word.split('_')[0] if word not in EVENT_KINDS else word
    return word if word in EVENT_KINDS else _KIND_ALIASES.get(word, 'other')


def _objects(value):
    """The objects a character handles as a short list of strings, however the model wrote them."""
    items = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    found = []
    for item in items:
        text = _text(item) if isinstance(item, str) else _text(item.get('object') or item.get('name')) if isinstance(item, dict) else ''
        if text and text.lower().rstrip('.') not in ('none', 'n/a', 'no objects', 'nothing') and text not in found:
            found.append(text[:160])
    return found[:6]


def link_video(value, roster):
    """Tie one window's answer to the character records: names become canonical and carry their context id.

    Only well-formed parts are touched. Anything unusual the model returned is kept as it is, because a
    partial observation is still worth storing.
    """
    if not isinstance(value, dict):
        return value
    roster = list(roster or [])
    camera = value.get('camera')
    if isinstance(camera, str):
        camera = {'movement': camera, 'detail': ''}
    if isinstance(camera, dict):
        movement = _text(camera.get('movement')).lower()
        value['camera'] = {'movement': movement if movement in CAMERA_MOVES else (movement or 'other'), 'detail': _text(camera.get('detail')) or 'none'}
    people = value.get('characters')
    if isinstance(people, list):
        linked = []
        for item in people:
            if not isinstance(item, dict):
                continue
            match = _match_name(item.get('name') or item.get('character'), roster)
            entry = {'character': match['name'] if match else _text(item.get('name') or item.get('character')) or 'unknown',
                     'character_context_id': match['character_context_id'] if match else None,
                     **{name: _text(item.get(name)) for name in VIDEO_PERSON_TEXT}}
            spoken = entry['speaking'].lower()
            entry['speaking'] = 'yes' if spoken.startswith('yes') else 'no' if spoken.startswith('no') else 'unclear'
            if _text(item.get('gaze')):
                entry['gaze'] = _text(item.get('gaze'))
            objects = _objects(item.get('objects'))
            if objects:
                entry['objects'] = objects
            linked.append(entry)
        value['characters'] = linked
    events = value.get('events')
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and 'kind' in event:
                event['kind'] = event_kind(event['kind'])
            if isinstance(event, dict) and isinstance(event.get('subject'), str):
                match = _match_name(event['subject'], roster)
                if match:
                    event['subject'], event['character_context_id'] = match['name'], match['character_context_id']
                    continue
                # One event for several people ("Mai, Rio and Tomoe walk off") is linked to each of them.
                named = [person for person in roster if re.search(r'(?<!\w)' + re.escape(person['name']) + r'(?!\w)', event['subject'], re.I)]
                if len(named) > 1:
                    event['character_context_ids'] = [person['character_context_id'] for person in named]
    return value


def _join(values, skip=(), separator=' → '):
    kept = []
    for value in values:
        value = _text(value)
        if value and value.lower().rstrip('.') not in skip and value.casefold() not in (item.casefold() for item in kept):
            kept.append(value)
    return separator.join(kept)


def _state_correction(first_state, activity):
    """``{first_frame, clip}`` when the clip's first state differs from what the still read, else None.

    Only states both sides name in the shared vocabulary count; "turning" or "writing" says nothing
    about the body's state. A clip that starts like the still and then changes is a transition, not a correction.
    """
    if first_state in (None, '', 'unclear', 'other'):
        return None
    clip, _ = pose_detail.body_pose(activity)
    if clip in ('other', 'unclear') or clip == first_state:
        return None
    return {'first_frame': first_state, 'clip': clip}


def cut_direction(windows, roster=()):
    """One motion direction for the whole cut, merged from its windows without calling a model.

    Camera moves, activities and emotions are listed in the order the windows report them; a
    character's ``end`` is the last window's. Legacy callers may provide a roster with last-frame
    detections, which still adds the historical positive ``in_last_frame`` confirmation. The current
    workflow provides no such boxes, so new directions rely on the video model's ``end`` field.
    """
    analyses = [window['analysis'] for window in windows if isinstance(window, dict) and isinstance(window.get('analysis'), dict)]
    cameras = [analysis['camera'] for analysis in analyses if isinstance(analysis.get('camera'), dict)]
    moves = _join(camera.get('movement') for camera in cameras)
    direction = {'summary': _join((analysis.get('overview') for analysis in analyses), separator=' Then: '),
                 'camera': {'movement': moves or 'unknown', 'detail': _join((camera.get('detail') for camera in cameras), skip=('none',), separator='; ') or 'none'}}
    order, merged = [], {}
    for index, analysis in enumerate(analyses):
        for person in analysis.get('characters') or []:
            if not isinstance(person, dict):
                continue
            key = person.get('character_context_id') or _text(person.get('character')).casefold()
            if key not in merged:
                merged[key] = {'character': person.get('character'), 'character_context_id': person.get('character_context_id'), 'parts': []}
                order.append(key)
            merged[key]['parts'].append((index, person))
    first_states = {person.get('character_context_id'): (person.get('pose_detail') or {}).get('body_pose') for person in roster or [] if isinstance(person, dict) and person.get('character_context_id')}
    people = []
    for key in order:
        parts = [person for _, person in merged[key]['parts']]
        spoken = [person.get('speaking') for person in parts]
        last_index, last = merged[key]['parts'][-1]
        entry = {'character': merged[key]['character'], 'character_context_id': merged[key]['character_context_id'],
                 'activity': _join(person.get('activity') for person in parts),
                 'travel': _join((person.get('travel') for person in parts), skip=('none',)) or 'none',
                 'motion': _join((person.get('motion') for person in parts), skip=('holds still',), separator=' Then: ') or 'holds still',
                 'speaking': 'yes' if 'yes' in spoken else 'no' if spoken and all(value == 'no' for value in spoken) else 'unclear',
                 'emotion': _join(person.get('emotion') for person in parts),
                 'end': _text(last.get('end')) if last_index == len(analyses) - 1 else 'not seen in the final part of the cut'}
        gaze = _join((person.get('gaze') for person in parts), skip=('holds',), separator=' Then: ')
        if gaze:
            entry['gaze'] = gaze
        objects = _objects([item for person in parts for item in (person.get('objects') if isinstance(person.get('objects'), list) else [])])
        if objects:
            entry['objects'] = objects
        correction = _state_correction(first_states.get(merged[key]['character_context_id']), parts[0].get('activity'))
        if correction:
            entry['state_correction'] = correction
        people.append(entry)
    confirmed = [person for person in roster or [] if person.get('at_end')]
    boxed = {person['character_context_id'] for person in confirmed}
    for person in people:
        if person['character_context_id'] in boxed:
            person['in_last_frame'] = True
    direction['characters'] = people
    if confirmed:
        direction['end_state'] = {'in_frame': [person['name'] for person in confirmed], 'source': 'character boxes of the last frame'}
    return direction


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
