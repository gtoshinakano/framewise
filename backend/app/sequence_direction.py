"""The director's side of sequence direction: prompts, answer schemas and validators.

Four calls direct a sequence: ``bible`` (what the sequence is and who is in it), ``plan`` (where
the shots begin and end), one ``shot`` call per planned shot, and ``review`` (continuity across the
written shots). The model decides and writes prose; ids, times, labels and spoken words are the
app's (``sequence_timeline``, ``sequence_prompt``). Each validator returns the normalised answer
or raises ``ValueError`` with sentences that are sent back to the model, or shown to a person who
edited the result by hand.
"""
import math
import re

from . import sequence_evidence, sequence_prompt, sequence_timeline

FRAMING = """You are the director of a short video of at most three minutes that re-creates a sequence of an anime episode with AI, shot by shot. Each shot is generated on its own: first a still image of its opening frame, then a MiniMax H3 video of at most 15 seconds that animates that still in sync with the ORIGINAL audio of the episode. The original audio is kept unchanged under the whole video, so it is the spine every decision follows.

You do not see pictures and you do not hear audio. Your evidence is text written by other systems (vision models, a transcription, the user's notes); it can be incomplete or wrong. The user's own notes, character links and decisions are authoritative."""

RULES = """## RULES
- Base everything on the supplied evidence. Never invent characters, places, actions, props or dialogue.
- Emotion comes from the pictures and from the meaning of the words. No acoustic analysis was made: never describe vocal tone or delivery as heard.
- Subtitles and other on-screen text are reference only and never part of a picture.
- Write in English. Keep character names exactly as they are given, in full."""

BIBLE_SYSTEM = FRAMING + """

Before any shot is planned you write the SEQUENCE BIBLE: the shared understanding that keeps every shot consistent.

## WHAT YOU RECEIVE
One JSON object:
- project, episode and director_instruction (the user's intent for this sequence; follow it when present).
- sequence: its length in seconds and its first and last cut number.
- characters: the project's character records (exact names, notes). For a character who appears in these cuts, seen_in_cuts lists where, and appearance_in_first_frames quotes how the vision model described that character in the opening frames. scenario_notes and context_notes: the user's notes on places and story.
- cuts: one digest per source cut, in order. t is its start and end in seconds from the start of the sequence; shot is the size and angle of its opening frame; camera its movement; on_screen the named characters the user marked; unnamed_people true when other people are visible; speaking the named characters whose mouth visibly moves; action what happens; setting, lighting, mood and props describe the opening frame. "analysis": "missing" means the cut was not analysed.
- transcript: every spoken line as `L<n> start-end VOICE LABEL / Character: words`. A character after the slash was linked by the user and is certain. A line with only a voice label (such as SPEAKER_03) has no known speaker yet. A line shown as "Unknown speaker" has no voice label at all: such lines may come from different people, so they get no entry in voices.
- voices: per voice label, how much it says, which characters the user linked it to (linked_by_user), and for its unlinked lines who was visibly speaking meanwhile and who was the only named person on screen. These are hints, not proof.
- pauses: quiet stretches of the audio, as start and end seconds.

## YOUR DECISIONS
1. Story (logline, synopsis, arc): what this sequence is about and how tension and emotion develop from the first cut to the last.
2. Scenes: group consecutive cuts that share one place and time. Every cut belongs to exactly one scene and scenes follow each other without gaps. Name the place concretely, with its time of day and light. If a scenario note describes that place, put its exact title in "scenario"; otherwise null.
3. Characters: every named character who is on screen in these cuts (seen_in_cuts), with the exact name of the character record. anchors are 3 to 6 visible features that stay constant in this sequence (build, hair, face, main garment and its colours). outfit is what the character wears here. Take both from appearance_in_first_frames, which is what the frames show; the record notes describe the series and may show another outfit. When the frames disagree about a garment or a colour, keep what most of them say and name the disagreement in open_questions. arc is what the character wants or feels across the sequence, in one sentence.
4. Voices: decide who owns each voice label. A label the user linked to one character belongs to that character. For an unlinked label weigh who visibly speaks while it is heard, who is alone on screen, what the words mean and who would say them to whom, and whether another label already belongs to that character. kind is character, narrator (narration, a title voice) or unknown. confidence is high only when two independent hints agree, otherwise medium or low. When nothing supports a choice, use kind unknown and character null: a wrong owner is worse than none.
5. Style: one sentence that starts with "The target video is" and locks the look of every shot (medium, line work, colour, shading), from the project context. It is copied into every shot prompt, so keep it to the look, with no story in it.
6. spoken_language: the language of the spoken lines, as one English word (for example Japanese).
7. Pacing: how the shots should breathe with the audio: where the sequence is quick, where it rests, where a held look matters.
8. Continuity: concrete rules every shot must respect (outfits, props carried from one cut to the next, who stands where, time of day, weather).
9. open_questions: only what a person must decide because the evidence cannot.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"logline": "one sentence",
 "synopsis": "at most 120 words",
 "style": "The target video is ...",
 "spoken_language": "Japanese",
 "scenes": [{"id": "S1", "from_cut": 12, "to_cut": 15, "location": "...", "time_of_day": "...", "lighting": "...", "mood": "...", "scenario": "exact scenario note title or null"}],
 "characters": [{"name": "exact record name", "role": "who this is in the sequence", "anchors": "3 to 6 visible features", "outfit": "...", "arc": "..."}],
 "voices": [{"speaker_label": "SPEAKER_01", "character": "exact record name or null", "kind": "character | narrator | unknown", "confidence": "high | medium | low", "reason": "the hints that decided it"}],
 "arc": "...", "pacing": "...",
 "continuity": ["one rule per item"],
 "open_questions": ["..."]}
Return JSON only."""

PLAN_SYSTEM = FRAMING + """

The sequence bible is written. Now you plan the SHOTS: you divide the sequence into consecutive shots along its original audio.

## HOW A SHOT WORKS
- A shot is ONE continuous camera take, generated from ONE initial frame (a still). It cannot contain an edit.
- A shot lasts at most 15 seconds. H3 works best between 5 and 12 seconds. A shot shorter than 5 seconds is generated at 5 seconds and trimmed, which is wasteful: avoid it when an honest merge exists, accept it when the source really is a quick shot.
- The original audio plays unchanged underneath, so every boundary is also an audio decision: a boundary inside a spoken line splits that line between two takes.

## WHAT YOU RECEIVE
One JSON object: director_instruction, the bible, sequence_seconds, the cuts digests (as described by their keys: t in sequence seconds, shot, camera, setting, on_screen, speaking, action), the transcript (`L<n> start-end speaker: words`, times in sequence seconds), pauses, and candidates.

candidates are the ONLY places where a shot may start or end. B0 is the start of the sequence and the last candidate is its end. Each has t (sequence seconds) and kind:
- "cut": a hard cut of the source (cut = the cut that starts there; crosses_line = a spoken line that continues across it).
- "pause": a silence inside a long cut (pause = its length in seconds).
- "event": a change of action inside a very long cut. "quiet": a low point of the audio. Both exist so that a long cut can always be divided.

## YOUR DECISIONS
For every shot choose "from" and "to" among the candidate ids. The default is one shot per source cut, from one "cut" candidate to the next. Depart from it only for a reason:
- MERGE consecutive cuts into one shot when one continuous take can honestly carry them: the same place with the framing continued; a short insert or reaction that can become a camera move or stay off screen under continuing dialogue; several very short cuts (under about 3 seconds each) of one action. The picture of one cut then carries the whole shot. One take has ONE framing unless the camera moves: when the merged cuts differ in shot size or in what they show (a close insert and a wide shot, for example), either keep one cut's framing for the entire time and say that the other picture is dropped or happens off screen, or name the single camera move (a push in, a pull out, a pan) that leads from one framing to the other. A static take cannot change its framing. Say in "staging" what the camera shows for the entire time, any camera move, and what becomes of the other cuts' pictures. Do not merge across a change of scene, place or time, or when the cuts show different speakers whose lips must be seen.
- SPLIT a cut that is longer than 15 seconds (you must), or one above about 10 seconds whose action clearly has two phases. Prefer a "pause" candidate; use "event" or "quiet" only when no pause fits. The later part continues the same take from the frame where the earlier part stopped; say so in "staging".
- frame_cut is the number of the covered cut whose opening frame becomes the initial frame of the shot. For an unmerged shot it is that cut. For a merged shot choose the cut whose picture carries the take (usually the longest, or the one whose speaker is on screen); when a camera move connects two framings, it is the cut the move starts from.

Direct the rhythm against the audio: let a line finish in the shot where its speaker is visible, put boundaries in pauses, keep listening and reaction shots as long as the source held them, and follow the bible's pacing.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"shots": [{"from": "B0", "to": "B1", "frame_cut": 12, "scene": "S1",
            "intent": "what this shot must achieve in the sequence, one sentence",
            "staging": "for a merged or split shot: how one continuous take covers it; otherwise an empty string"}],
 "notes": "one or two sentences on the rhythm you chose"}
The first shot starts at B0, every shot starts where the previous one ends, the last one ends at the final candidate, and no shot is longer than 15 seconds.
Return JSON only."""

SHOT_SYSTEM = FRAMING + """

The bible and the shot plan are written. Now you direct ONE shot. Your answer is structured direction, and the app turns it into the final MiniMax H3 prompt: it writes the section headers, the reference labels, the timecodes, the camera sentence, the spoken words and the audio lines itself. Therefore write names and never reference labels, put no timecodes inside your sentences, and never quote dialogue.

## WHAT YOU RECEIVE
One JSON object:
- sequence: synopsis, the style lock, the spoken language, the arc and the continuity rules of the bible. director_instruction is the user's intent for the sequence; instruction_for_this_shot, when present, is a correction you must follow.
- scene and scenario_reference (a picture of the place that the generator receives, when there is one).
- shot: its number, its length in seconds, the source cuts it covers, its intent and staging from the plan, and initial_frame: which cut supplies the opening still and whether that still is the cut's first frame or a later moment.
- previous_shot and next_shot: what surrounds this shot. ends_with is how the analysis says the previous shot's characters end, opens_with is how the opening frame of the next shot shows them, and hands_over is the state the previous shot was written to end in.
- characters: the named characters that appear in the covered cuts, with the bible's anchors and outfit and the user's notes.
- cuts: the evidence for each covered cut. plays_at is when that cut is heard and seen on THIS shot's clock. first_frame describes the cut's opening still. motion is the motion direction: summary, camera, per character activity, travel, motion, speaking (yes, no, unclear), emotion and end, plus events whose seconds are already on this shot's clock. cut_summary is a written brief of the cut and user_decisions are the user's answers to open review points.
- dialogue: the lines heard during this shot, with id, t (start and end on this shot's clock), the speaker and how the speaker is known. subtitles_for_meaning_only may help you understand the words; never show them.

## PRECEDENCE
user_decisions and instruction_for_this_shot, then cut_summary, then motion, then first_frame; the neighbouring shots only for continuity. The first frame decides how the shot starts, the motion decides what changes over time, and speaking decides whether lips move.

## WHAT YOU WRITE
- subjects: every character or person who is visible at any point of the shot, the most important first, at most 5. name is the exact name from "characters"; for someone without a record use a short visual label such as "the shop clerk", never a guessed name. anchors: 3 to 6 visible features (build, hair, main garment and colour) that agree with the bible and the first frame. keep: what must stay identical to that character's reference, as one clause.
- picture_line: who is where in the opening still, in one sentence, with only what is visible in it. keep_picture: what the video must preserve from the opening still (composition, style, palette, lighting), as one clause.
- still: the opening frame as ONE frozen moment for an image generator. shot: size, angle, camera height and where the frame crops the subject. subjects: for each visible character the place in frame, orientation, pose, expression, gaze and visible outfit. layout: foreground, midground and background and how they relate. setting. lighting: source, softness, time of day, palette. Poses, not actions; no lettering.
- summary: one sentence with what happens in the shot and how the camera behaves.
- shot_line: shot size, angle and who is where at the start, in one or two sentences. framing_lock: one short sentence that locks the crop you gave in still.shot, naming the same upper and lower edge of the frame and ending with "never wider" (or "never closer" for a wide shot); it must agree with still.shot. light: one sentence on the light.
- camera: move is ONE of static, push_in, pull_out, pan_left, pan_right, truck_left, truck_right, tilt_up, tilt_down, pedestal_up, pedestal_down, arc, tracking, shake. If the motion evidence says static, use static. amplitude is small, medium or large; speed is slow, medium or fast. detail is one optional sentence on where the move ends or what it reveals; leave it empty for a static shot.
- beats: the action in order, about 2 to 3 beats per 10 seconds, at least one. end is the second on this shot's clock at which the beat ends; the last beat ends at the shot's length. action is a visible start state, then one action, then a visible end state, in 2 to 4 concrete sentences (body, hands, face, gaze, cloth, hair, light). Keep every event at its real time from the motion evidence, because the picture must stay in sync with the original audio. Do not mention the camera in a beat. Do not write spoken words and do not write that someone "says" something: describe the mouth and the acting only (lips moving in speech, a breath, a glance).
- speech: one entry for EVERY line in "dialogue". delivery is lip_sync (the speaker is one of your subjects and visibly speaks: speaking is yes, or the user decided so), off_screen (the speaker is not in the picture, or the voice is unconfirmed), inner_voice (the speaker is on screen with closed lips: a thought or that character's narration) or narration (a narrator outside the scene). subject is the exact subject name for lip_sync and inner_voice, otherwise null. Being the only person on screen is not proof of speaking; for an unconfirmed voice choose off_screen and add an open question.
- scenario_anchors and scenario_keep: only when scenario_reference is present: 3 to 5 visible features of the place, and what must stay as in its reference.
- soundscape: 1 to 3 sentences of ambience and action sounds that fit the picture (room tone, steps, cloth, wind). No speech and no music.
- continuity: from_previous is what must match the end of the previous shot; to_next is the state this shot hands over (positions, poses, props, gaze). One sentence each. When the evidence of this shot contradicts a neighbour (someone sits here and stands in the next opening frame, an outfit differs), follow this shot's own evidence and name the contradiction in open_questions instead of hiding it.
- open_questions: only decisions the evidence cannot make.

## RULES FOR H3
- Never write the word "cut", not even "no cut": H3 reads it as an edit. Say "shot" or "take".
- No lettering anywhere: write "blank" or "unmarked". Subtitles and on-screen text never appear.
- A merged shot is ONE continuous take as its staging describes it; do not describe an edit.
- When the initial frame is a later moment of its cut, derive the opening state from first_frame plus the motion up to that moment, and add an open question if that state is uncertain.
- Write 180 to 380 words across shot_line, framing_lock, light and the beats together. Every sentence must be concrete and supported; no filler.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"subjects": [{"name": "...", "anchors": "...", "keep": "..."}],
 "picture_line": "...", "keep_picture": "...",
 "still": {"shot": "...", "subjects": "...", "layout": "...", "setting": "...", "lighting": "..."},
 "summary": "...", "shot_line": "...", "framing_lock": "...", "light": "...",
 "camera": {"move": "static", "amplitude": "small", "speed": "slow", "detail": ""},
 "beats": [{"end": 2.4, "action": "..."}],
 "speech": [{"line": "L7", "delivery": "lip_sync", "subject": "exact subject name or null"}],
 "scenario_anchors": "", "scenario_keep": "",
 "soundscape": "...",
 "continuity": {"from_previous": "...", "to_next": "..."},
 "open_questions": ["..."]}
Return JSON only."""

REVIEW_SYSTEM = FRAMING + """

Every shot of the sequence is written. Now you REVIEW the whole sequence before anything is generated, as the director who has to answer for continuity.

## WHAT YOU RECEIVE
One JSON object: the bible (story, scenes, characters with anchors and outfits, continuity rules), the voices it decided, and shots in order. Each shot has t (start and end in sequence seconds), the source cuts, scene, intent, staging, opening_still, subjects, summary, camera, beats ("end second: action"), speech (line id, delivery, subject, first words) and continuity (from_previous, to_next). A shot without "written" has no direction yet.

## WHAT TO CHECK
1. Continuity between consecutive shots of one scene: positions, poses, props, outfits, light and time of day; each shot's to_next against the opening still of the next shot.
2. Speech: the same voice lip-synced by different characters; a line that continues into the next shot delivered differently there; lip sync given to someone who is not a subject of the shot.
3. Staging: a merged shot that describes two takes; a camera move that fights the intent; a split shot whose second part does not continue the first.
4. Rhythm: a shot that is too busy or too empty for its length, against the bible's pacing.
5. Bible: anchors, outfits and continuity rules respected by every shot.
6. Shots that are not written.

Report only problems that would change a generated picture, its motion or its lip sync. Differences of wording are not problems. If the sequence is consistent, return no issues.

""" + RULES + """

## OUTPUT
Return exactly one JSON object:
{"verdict": "two or three sentences on whether the sequence holds together",
 "issues": [{"shot": 4, "severity": "fix | check", "problem": "what is inconsistent, naming the shots involved",
             "suggestion": "an instruction the writer of that shot can follow to repair it"}]}
severity "fix" means wrong as written; "check" means a person must decide. Return JSON only."""

# max_tokens includes the model's reasoning; temperature is lower where the answer is a decision and higher where it is prose.
STEPS = {
    'bible': {'system': BIBLE_SYSTEM, 'max_tokens': 6144, 'temperature': 0.3},
    'plan': {'system': PLAN_SYSTEM, 'max_tokens': 8192, 'temperature': 0.3},
    'shot': {'system': SHOT_SYSTEM, 'max_tokens': 6144, 'temperature': 0.6},
    'review': {'system': REVIEW_SYSTEM, 'max_tokens': 6144, 'temperature': 0.2},
}
STRING = {'type': 'string'}
LANGUAGE = re.compile(r'^[A-Za-z][A-Za-z -]{1,30}$')
BANNED = (
    (re.compile(r'\bcuts?\b', re.IGNORECASE), 'contains the word "cut"; say "shot" or "take"'),
    (re.compile(r'<\s*(?:Subject|Picture|Audio|Video)\b', re.IGNORECASE), 'contains a reference label; write names, the app adds the labels'),
    (re.compile(r'</?d>'), 'quotes dialogue; describe the acting only, the app adds the spoken words'),
    (re.compile(r'\b\d{2}:\d{2}(?:\.\d+)?\b'), 'contains a timecode; the app writes the times'),
)


def _object(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(required if required is not None else properties), 'additionalProperties': False}


def _format(name, schema):
    return {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}


def _str(value):
    return re.sub(r'\s+', ' ', value).strip() if isinstance(value, str) else ''


def _strings(value, limit):
    return [_str(item) for item in value if _str(item)][:limit] if isinstance(value, list) else []


def _fail(errors):
    raise ValueError(' '.join(f'({number}) {error}' if len(errors) > 1 else error for number, error in enumerate(errors, 1)))


def _canonical(name, names):
    """The record name a model meant, matched without regard to case."""
    wanted = _str(name).casefold()
    return next((item for item in names if item.casefold() == wanted), None)


# ---------------------------------------------------------------------- bible

def bible_format():
    return _format('sequence_bible', _object({
        'logline': STRING, 'synopsis': STRING, 'style': STRING, 'spoken_language': STRING,
        'scenes': {'type': 'array', 'items': _object({'id': STRING, 'from_cut': {'type': 'integer'}, 'to_cut': {'type': 'integer'}, 'location': STRING, 'time_of_day': STRING,
                                                      'lighting': STRING, 'mood': STRING, 'scenario': {'type': ['string', 'null']}})},
        'characters': {'type': 'array', 'items': _object({'name': STRING, 'role': STRING, 'anchors': STRING, 'outfit': STRING, 'arc': STRING})},
        'voices': {'type': 'array', 'items': _object({'speaker_label': STRING, 'character': {'type': ['string', 'null']}, 'kind': {'type': 'string', 'enum': ['character', 'narrator', 'unknown']},
                                                      'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']}, 'reason': STRING})},
        'arc': STRING, 'pacing': STRING, 'continuity': {'type': 'array', 'items': STRING}, 'open_questions': {'type': 'array', 'items': STRING}}))


def validate_bible(output, ctx, timeline):
    """Check and normalise a bible against the character records, scenario notes, cuts and voice labels of the sequence."""
    if not isinstance(output, dict):
        _fail(['The answer must be one JSON object'])
    errors = []
    bible = {key: _str(output.get(key)) for key in ('logline', 'synopsis', 'style', 'spoken_language', 'arc', 'pacing')}
    for key in ('logline', 'synopsis', 'style', 'spoken_language'):
        if not bible[key]:
            errors.append(f'"{key}" is empty')
    if bible['style'] and not bible['style'].startswith(sequence_prompt.STYLE_START):
        errors.append(f'"style" must be one sentence that starts with "{sequence_prompt.STYLE_START}"')
    if bible['spoken_language'] and not LANGUAGE.match(bible['spoken_language']):
        errors.append('"spoken_language" must be the language name in English letters, for example Japanese')
    first, last = timeline['cuts'][0]['number'], timeline['cuts'][-1]['number']
    scenario_titles = [item['title'] for item in ctx['scenarios']]
    scenario_ids = {item['title']: item['id'] for item in ctx['scenarios']}
    scenes, expected = [], first
    for scene in output.get('scenes') if isinstance(output.get('scenes'), list) else []:
        if not isinstance(scene, dict) or not all(isinstance(scene.get(key), int) and not isinstance(scene.get(key), bool) for key in ('from_cut', 'to_cut')):
            errors.append('every scene needs integer from_cut and to_cut')
            continue
        if scene['from_cut'] != expected or scene['to_cut'] < scene['from_cut'] or scene['to_cut'] > last:
            errors.append(f"scenes must cover cuts {first} to {last} in order without gaps or overlaps; the scene starting at cut {scene['from_cut']} should start at cut {expected} and end by cut {last}")
            break
        expected = scene['to_cut'] + 1
        title = None
        if _str(scene.get('scenario')) and _str(scene.get('scenario')).lower() != 'null':
            title = _canonical(scene['scenario'], scenario_titles)
            if title is None and scenario_titles:
                errors.append(f"scenario \"{_str(scene['scenario'])}\" is not a scenario note; use one of: {', '.join(scenario_titles)}, or null")
        scenes.append({'id': f'S{len(scenes) + 1}', 'from_cut': scene['from_cut'], 'to_cut': scene['to_cut'], **{key: _str(scene.get(key)) for key in ('location', 'time_of_day', 'lighting', 'mood')},
                       'scenario': title, 'scenario_context_id': scenario_ids.get(title)})
    if not scenes:
        errors.append('"scenes" must list at least one scene')
    elif expected != last + 1 and not errors:
        errors.append(f'scenes stop at cut {expected - 1}; they must cover every cut up to {last}')
    names = [character['name'] for character in ctx['characters']]
    ids = {character['name']: character['id'] for character in ctx['characters']}
    characters = []
    for person in output.get('characters') if isinstance(output.get('characters'), list) else []:
        name = _canonical(person.get('name'), names) if isinstance(person, dict) else None
        if name is None:
            errors.append(f"character \"{_str(person.get('name')) if isinstance(person, dict) else person}\" has no character record; use exact names from: {', '.join(names) or 'none (there are no records)'}")
            continue
        if any(item['name'] == name for item in characters):
            continue
        if not _str(person.get('anchors')):
            errors.append(f'character "{name}" needs anchors: 3 to 6 visible features')
        characters.append({'name': name, 'character_context_id': ids[name], **{key: _str(person.get(key)) for key in ('role', 'anchors', 'outfit', 'arc')}})
    evidence = {voice['speaker_label']: voice for voice in sequence_evidence.voice_evidence(ctx, timeline)}
    given = {_str(voice.get('speaker_label')): voice for voice in output.get('voices') if isinstance(voice, dict)} if isinstance(output.get('voices'), list) else {}
    voices = []
    for label, facts in evidence.items():
        voice = given.get(label) or {}
        linked = facts.get('linked_by_user') or {}
        if len(linked) == 1:
            # The user's own link decides; the model's answer is only kept as the reason it gave.
            name = next(iter(linked))
            voices.append({'speaker_label': label, 'character': name, 'character_context_id': ids.get(name), 'kind': 'character', 'confidence': 'high', 'source': 'transcript',
                           'reason': 'Linked to this character in the transcript.'})
            continue
        name = _canonical(voice.get('character'), names) if _str(voice.get('character')) and _str(voice.get('character')).lower() != 'null' else None
        if _str(voice.get('character')) and _str(voice.get('character')).lower() != 'null' and name is None:
            errors.append(f"voice {label}: \"{_str(voice['character'])}\" has no character record; use an exact name or null")
        kind = voice.get('kind') if voice.get('kind') in ('character', 'narrator', 'unknown') else 'unknown'
        kind = 'character' if name else ('narrator' if kind == 'narrator' else 'unknown')
        voices.append({'speaker_label': label, 'character': name, 'character_context_id': ids.get(name), 'kind': kind,
                       'confidence': voice.get('confidence') if voice.get('confidence') in ('high', 'medium', 'low') else 'low', 'source': 'director', 'reason': _str(voice.get('reason'))})
    if errors:
        _fail(errors)
    return {**bible, 'scenes': scenes, 'characters': characters, 'voices': voices, 'continuity': _strings(output.get('continuity'), 16), 'open_questions': _strings(output.get('open_questions'), 12)}


# ----------------------------------------------------------------------- plan

def plan_format(timeline, bible):
    ids = [candidate['id'] for candidate in timeline['candidates']]
    scenes = [scene['id'] for scene in bible.get('scenes') or []] or ['S1']
    shot = _object({'from': {'type': 'string', 'enum': ids[:-1]}, 'to': {'type': 'string', 'enum': ids[1:]},
                    'frame_cut': {'type': 'integer', 'enum': [cut['number'] for cut in timeline['cuts']]}, 'scene': {'type': 'string', 'enum': scenes},
                    'intent': STRING, 'staging': STRING})
    return _format('shot_plan', _object({'shots': {'type': 'array', 'minItems': 1, 'items': shot}, 'notes': STRING}))


def validate_plan(output, timeline, bible):
    """Check a plan written with candidate ids; returns ``{'shots': [...], 'notes': str}`` still in candidate ids."""
    shots = output.get('shots') if isinstance(output, dict) else None
    errors = sequence_timeline.check_model_plan(shots, timeline)
    if errors:
        _fail(errors)
    by_id = {candidate['id']: candidate for candidate in timeline['candidates']}
    scene_ids = {scene['id'] for scene in bible.get('scenes') or []}
    normal = []
    for shot in shots:
        numbers = [cut['number'] for cut in sequence_timeline.covered_cuts(timeline, by_id[shot['from']]['t'], by_id[shot['to']]['t'])]
        frame_cut = shot.get('frame_cut') if shot.get('frame_cut') in numbers else numbers[0]
        scene = shot.get('scene') if shot.get('scene') in scene_ids else (sequence_evidence.scene_for(bible, frame_cut) or {}).get('id')
        normal.append({'from': shot['from'], 'to': shot['to'], 'frame_cut': frame_cut, 'scene': scene, 'intent': _str(shot.get('intent')), 'staging': _str(shot.get('staging'))})
    return {'shots': normal, 'notes': _str(output.get('notes'))}


# ----------------------------------------------------------------------- shot

def shot_format(shot, roster, dialogue):
    lines = [line['line'] for line in dialogue]
    speech = _object({'line': {'type': 'string', 'enum': lines} if lines else STRING, 'delivery': {'type': 'string', 'enum': list(sequence_prompt.DELIVERIES)}, 'subject': {'type': ['string', 'null']}})
    return _format('shot_direction', _object({
        'subjects': {'type': 'array', 'maxItems': 5, 'items': _object({'name': STRING, 'anchors': STRING, 'keep': STRING})},
        'picture_line': STRING, 'keep_picture': STRING,
        'still': _object({key: STRING for key in ('shot', 'subjects', 'layout', 'setting', 'lighting')}),
        'summary': STRING, 'shot_line': STRING, 'framing_lock': STRING, 'light': STRING,
        'camera': _object({'move': {'type': 'string', 'enum': list(sequence_prompt.CAMERA_MOVES)}, 'amplitude': {'type': 'string', 'enum': list(sequence_prompt.AMPLITUDES)},
                           'speed': {'type': 'string', 'enum': list(sequence_prompt.SPEEDS)}, 'detail': STRING}),
        'beats': {'type': 'array', 'minItems': 1, 'items': _object({'end': {'type': 'number'}, 'action': STRING})},
        'speech': {'type': 'array', 'items': speech},
        'scenario_anchors': STRING, 'scenario_keep': STRING, 'soundscape': STRING,
        'continuity': _object({'from_previous': STRING, 'to_next': STRING}), 'open_questions': {'type': 'array', 'items': STRING}}))


def _prose(content):
    """Every field the model writes as prose, with the name an error message gives it."""
    yield from ((key, content.get(key)) for key in ('picture_line', 'keep_picture', 'summary', 'shot_line', 'framing_lock', 'light', 'soundscape', 'scenario_anchors', 'scenario_keep'))
    yield 'camera.detail', content['camera'].get('detail')
    yield from ((f'still.{key}', value) for key, value in content['still'].items())
    yield from ((f'beat {number}', beat['action']) for number, beat in enumerate(content['beats'], 1))
    for subject in content['subjects']:
        yield f"subject {subject['name']}", f"{subject['anchors']} {subject['keep']}"
    yield from ((f'continuity.{key}', value) for key, value in content['continuity'].items())


def validate_shot(output, shot, bible, roster, scenario, dialogue):
    """Check and normalise one shot's direction, render it and apply the H3 format rules.

    ``roster`` are the named characters of the covered cuts and ``dialogue`` the resolved lines of the
    shot. Returns the normalised direction; problems (including those of the rendered prompt) are raised.
    """
    if not isinstance(output, dict):
        _fail(['The answer must be one JSON object'])
    errors = []
    names = [person['name'] for person in roster]
    subjects = []
    for item in output.get('subjects') if isinstance(output.get('subjects'), list) else []:
        name = _str(item.get('name')) if isinstance(item, dict) else ''
        name = _canonical(name, names) or name
        if not name or any(subject['name'].casefold() == name.casefold() for subject in subjects):
            continue
        if not _str(item.get('anchors')):
            errors.append(f'subject "{name}" needs anchors: 3 to 6 visible features')
        subjects.append({'name': name, 'anchors': _str(item.get('anchors')), 'keep': _str(item.get('keep'))})
    if len(subjects) > 5:
        errors.append('list at most 5 subjects, the most important first')
    content = {'subjects': subjects[:5], **{key: _str(output.get(key)) for key in ('picture_line', 'keep_picture', 'summary', 'shot_line', 'framing_lock', 'light', 'soundscape', 'scenario_anchors', 'scenario_keep')}}
    for key in ('picture_line', 'summary', 'shot_line', 'soundscape'):
        if not content[key]:
            errors.append(f'"{key}" is empty')
    still = output.get('still') if isinstance(output.get('still'), dict) else {}
    content['still'] = {key: _str(still.get(key)) for key in ('shot', 'subjects', 'layout', 'setting', 'lighting')}
    for key in ('shot', 'setting', 'lighting'):
        if not content['still'][key]:
            errors.append(f'"still.{key}" is empty')
    camera = output.get('camera') if isinstance(output.get('camera'), dict) else {}
    if camera.get('move') not in sequence_prompt.CAMERA_MOVES:
        errors.append(f"camera.move must be one of: {', '.join(sequence_prompt.CAMERA_MOVES)}")
    content['camera'] = {'move': camera.get('move') if camera.get('move') in sequence_prompt.CAMERA_MOVES else 'static',
                         'amplitude': camera.get('amplitude') if camera.get('amplitude') in sequence_prompt.AMPLITUDES else 'small',
                         'speed': camera.get('speed') if camera.get('speed') in sequence_prompt.SPEEDS else 'slow',
                         'detail': '' if camera.get('move') == 'static' else _str(camera.get('detail'))}
    duration, beats, previous = shot['duration'], [], 0.0
    for number, beat in enumerate(output.get('beats') if isinstance(output.get('beats'), list) else [], 1):
        end = beat.get('end') if isinstance(beat, dict) else None
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or not _str(beat.get('action')):
            errors.append(f'beat {number} needs a numeric "end" in seconds and an "action"')
            continue
        if end <= previous + 0.2:
            errors.append(f'beat {number} ends at {end:g} s, which is not after the previous beat ({previous:g} s)')
            continue
        if end > duration + 0.5:
            errors.append(f'beat {number} ends at {end:g} s but the shot lasts {duration:g} s; times are seconds from the start of this shot')
            continue
        previous = min(float(end), duration)
        beats.append({'end': round(previous, 3), 'action': _str(beat['action'])})
    if not beats:
        errors.append(f'"beats" needs at least one beat; the last one ends at {duration:g} s')
    else:
        beats[-1]['end'] = duration
        limit = max(2, math.ceil(duration / 2.5))
        if len(beats) > limit:
            errors.append(f'{len(beats)} beats are too many for {duration:g} s; write at most {limit}, each one start state, one action, end state')
    content['beats'] = beats
    decisions = {}
    for item in output.get('speech') if isinstance(output.get('speech'), list) else []:
        if isinstance(item, dict) and _str(item.get('line')):
            decisions.setdefault(_str(item['line']), item)
    speech = []
    for line in dialogue:
        item = decisions.get(line['line'])
        if not item or item.get('delivery') not in sequence_prompt.DELIVERIES:
            errors.append(f"speech needs one entry for line {line['line']} with delivery {', '.join(sequence_prompt.DELIVERIES)}")
            continue
        subject = None
        if item['delivery'] in ('lip_sync', 'inner_voice'):
            subject = _canonical(item.get('subject'), [entry['name'] for entry in content['subjects']])
            if subject is None:
                errors.append(f"speech {line['line']}: {item['delivery']} needs \"subject\" to be one of your subjects ({', '.join(entry['name'] for entry in content['subjects']) or 'you listed none'}); use off_screen for a speaker who is not in the picture")
                continue
        speech.append({'line': line['line'], 'delivery': item['delivery'], 'subject': subject})
    content['speech'] = speech
    continuity = output.get('continuity') if isinstance(output.get('continuity'), dict) else {}
    content['continuity'] = {key: _str(continuity.get(key)) for key in ('from_previous', 'to_next')}
    content['open_questions'] = _strings(output.get('open_questions'), 8)
    if scenario and not content['scenario_anchors']:
        errors.append('"scenario_anchors" is empty: give 3 to 5 visible features of the place, because the shot receives its scenario picture')
    if not scenario:
        content['scenario_anchors'] = content['scenario_keep'] = ''
    for field, value in _prose(content):
        for pattern, problem in BANNED:
            if value and pattern.search(value):
                errors.append(f'{field} {problem}')
    if errors:
        _fail(errors)
    _, verdict, _ = sequence_prompt.build(shot, content, bible, roster, scenario, dialogue)
    if not verdict['ok']:
        _fail([error.replace('detailed_description has', 'shot_line, framing_lock, light and beats together have') for error in verdict['errors']])
    return content


# --------------------------------------------------------------------- review

def review_format(count):
    issue = _object({'shot': {'type': 'integer', 'minimum': 1, 'maximum': max(1, count)}, 'severity': {'type': 'string', 'enum': ['fix', 'check']}, 'problem': STRING, 'suggestion': STRING})
    return _format('sequence_review', _object({'verdict': STRING, 'issues': {'type': 'array', 'items': issue}}))


def validate_review(output, count):
    if not isinstance(output, dict) or not _str(output.get('verdict')):
        _fail(['"verdict" is empty'])
    issues = []
    for item in output.get('issues') if isinstance(output.get('issues'), list) else []:
        if not isinstance(item, dict) or isinstance(item.get('shot'), bool) or not isinstance(item.get('shot'), int) or not 1 <= item['shot'] <= count or not _str(item.get('problem')):
            continue
        issues.append({'shot': item['shot'], 'severity': item.get('severity') if item.get('severity') in ('fix', 'check') else 'check',
                       'problem': _str(item['problem']), 'suggestion': _str(item.get('suggestion'))})
    return {'verdict': _str(output['verdict']), 'issues': issues}
