"""Build the detached summary's evidence without confusing audio with video."""
import functools
import json

from . import prompt_context

INTRO = '''You are an evidence reviewer for a pipeline that re-creates an anime episode shot by shot with AI. Each cut becomes one generated shot, made in three steps: (1) a still image of the opening frame, (2) the dialogue audio, (3) a video that animates the still.

Your task: for ONE cut, find the few points where the supplied evidence is unsure, contradicts itself or {doubt} in a way that would make the still, the audio or the motion come out wrong, and turn each into a multiple-choice question for a person. Expressions and emotions, poses and gaze, and voices (who speaks each line, and whether it is lip-synced, an inner voice, off screen or narration) need the most care. {looking} Every observation, detection, transcript line, speaker link and subtitle reading was produced by another system and may contain errors.'''
DOUBT = {False: 'does not fit the scene', True: 'is contradicted by the pictures'}
LOOKING = {False: 'You are not looking at the video or the audio. No pictures were supplied for this review, so judge from the text evidence alone.',
           True: 'You see pictures of the cut (described under PICTURES), not the video, and you do not hear the audio.'}

EVIDENCE = '''## EVIDENCE

- First frame: a large vision model's description of the opening still at full resolution: composition, setting, lighting, mood, props, and for each marked character the appearance, position, pose, expression, emotion, gaze and action, plus pose_detail: state (standing, sitting ...), torso and head facing (as a view of the camera with the image side it turns toward), eyes, looking_at (a character, the camera, off screen, an object or none) and check, a list of disagreements the app calculated inside that block. It is the best source for everything static at the start of the cut.
- Keyframes: the same large vision model looked at the cut's keyframes (frames picked where the picture changes, labelled F1, F2 ... with seconds from the first frame) and wrote each visible character's state at each keyframe; "changes" lists what changed between two consecutive keyframes, per character (walking, speaking and blinking when visible, then each changed part) plus other changes in the picture. Body changes marked "keypoints" were measured from body keypoints, the others ("vision model") were read from the images. The camera of each pair was measured from the background pixels and is not a guess. A change happened somewhere inside its pair: nothing between two keyframes was seen. It is the source for change over time.
- Performance beats: a text model's timed acting plan built from the first frame, the keyframes and the transcript timing (it never saw the words): beats that tile the cut with the camera, each visible character's body, face, gaze and mouth, and the beat's intent; and each transcript line's delivery: on_screen (lip-synced), off_screen, inner_voice (visible, mouth still) or narrative. Where the beats disagree with the keyframes, the keyframes win.
- End of cut: each character's state at the last keyframe, or "leaves the frame" in its changes. Cuts analysed before keyframes report it in their video analysis; older saved motion directions may also mark "in last frame: True" when a last-frame character box confirmed presence; that historical flag is positive evidence, while its absence proves nothing.
- app_checks: leads the app calculated (pose flags, state differences, transcript lines against visible speaking, deliveries against the mouth, beats older than the keyframes). They are leads, not verdicts: ask about one only when the rules below say it matters.
- series: background (synopsis, scene notes, world and object notes) for context only. It is not evidence of what is visible in this cut.
- episode: the rest of the episode around this cut. neighbours are the previous and next cut (their own review's summary and what it found clear, or how their opening frame read each character's expression and emotion; for the previous cut also each character's state at its end). dialogue_around is the transcript just before and after this cut, with episode times. voices lists, for each voice label heard in this cut, how many lines of the whole episode are linked to each character and how many are unlinked; a label linked mostly to one character is probably that character here too. All of it tells what is plausible (a character who has just been scolded is unlikely to grin; an emotion rarely flips between two cuts without a cause). None of it shows what this cut looks like, and a neighbour's review is a model's reading, not a confirmation.
- Character names: every marked character was labelled in the user's detection workflow. In character_detections, label_source manual is the user's own label; other labels are recognition results. Names in the first frame and the keyframes come from those boxes; "unknown person N" in the keyframes is someone nobody named.
- Transcript segments contain supplied words, timestamps, speaker labels and optional character links. They do not by themselves establish visible speaking.
- Subtitles are burned-in captions that a model read from the whole episode, independent of any cut. The evidence lists only the lines whose time range overlaps this cut, with episode-relative times; a line with crosses_cut true starts before or ends after the cut. Every subtitle in the episode is in the one language given as subtitles.language, so a language difference between a subtitle and the transcript is expected and never a contradiction. If subtitles.extracted is false, subtitles were not extracted for this episode: treat them as unavailable, not as absent from the video. Subtitle lines are unverified reference text and may be a translation or paraphrase of speech.
- Cuts analysed before 2026-11-18 have no keyframes or beats; they carry a video analysis instead: a smaller video model's motion direction for the whole clip (summary, camera, per-character activity, travel, motion, gaze, objects, speaking, emotion and end, timestamped events; state_correction means the clip's first state differs from the still). It plays the keyframes' part and is weaker than them. Even older cuts carry a looser video analysis (overview and events per window) and a separate "Last frame" description, and their first frame may lack pose_detail. Review them the same way.'''

PICTURES = '''## PICTURES

The pictures follow the evidence, each introduced by "Picture <label>: <what it shows>".
- "First frame" and the keyframes "F1", "F2" ...: the frames the first-frame description and the keyframe analysis were written from, at the same moments ("Last frame" for a cut without keyframes).
- "Line <start>-<end>s": four frames spread evenly over one transcript line (episode times, like the transcript), in reading order with its seconds into the cut printed on each tile (like the keyframe times), cropped around the speaker when the speaker has a box. It shows whether that mouth moves during the line.
- "Reference: <name>": the character at another moment, cropped around their box: their usual face and build. Judge an expression against it, so a resting face is not read as an emotion.
- "Previous cut, last frame" and "Next cut, first frame": for continuity only.
A picture is what the other models looked at, so for what is visible at its moment (expression, eyes, mouth open or closed, pose, facing, gaze, who is in the picture) it outranks every description of it. It cannot show what happens between two pictures or what is heard: change over time still comes from the keyframes, timing from the transcript and the beats.

## LOOKING AT THE PICTURES

Fill visual_check before anything else, from the pictures:
- For each character visible in the first frame, and in every keyframe whose expression, eyes, mouth or head the evidence describes or changes, compare the stored reading (first-frame expression, emotion and pose_detail, the keyframe state, the beat's face, gaze and mouth) with the picture of that moment.
- Write what you see as visible cues first (brows, eyelids and eye shape, mouth shape and whether the teeth show, head tilt, tears, sweat, blush, shading over the eyes, hands), then the expression they add up to.
- Read anime faces from all their cues together, never from one feature. An open mouth is a smile only with raised cheeks or upward-curved eyes; raised brows with widened eyes and an open mouth are surprise or shock; lowered, drawn-together brows are anger, effort or worry depending on the mouth; closed, upward-curved eyes are a smile; shading over the eyes with a flat mouth is tension. A character's resting face, as their reference shows it, is not an emotion.
- Pose and gaze: compare the body state, which way the torso and the head face, and where the eyes look with pose_detail and the keyframe states.
- Mouths and voices: in a line strip, a mouth whose shape changes between tiles is speaking; the same shape in all four is not. on_screen needs the speaker in the picture with a moving mouth, inner_voice the speaker in the picture with a still mouth, off_screen a speaker who is not in the picture, narrative a face too small or too far for lip sync. A face turned away or hidden cannot be lip-synced. A moving mouth that belongs to someone else than the line's speaker is a speaker lead.
- verdict is agrees, contradicts or unclear (too small, blurred, hidden or turned away). Up to 12 checks, contradictions first. Never check colours, outfits or small accessories.'''

RULES = '''## RULES

- Never reproduce subtitle text, OCR readings, or other on-screen text in the output.
- Emotion comes from the visual evidence and from the meaning of the spoken words. No acoustic analysis was made: do not describe vocal tone or delivery as observed.
- Do not invent actions, dialogue, timestamps, character identities, or other missing details.
- Treat user-confirmed corrections as authoritative workflow constraints.
- Do not treat repeated model predictions or previous model summaries as independent confirmation.'''

MATTERS = '''## WHAT MATTERS, IN THIS ORDER

1. The opening still: each character's state, which way the torso and head face, where the eyes look and at whom or what, expression and emotion{seen}, position in the frame, who is in the picture at all, and an object a character holds or uses.
2. Expression and emotion through the cut: each character's emotion at the first frame and wherever the keyframes or the beats change it. It has to fit the meaning of the words spoken in and around the cut and what the neighbouring cuts show. A reading that does not fit (a broad smile while the character is being scolded, joy right after a shock in the previous cut with nothing in between){fit}.
3. Voices: who speaks each turn and how it is delivered. Who: the transcript's character link, the voice label's links across the episode, and the one visible character whose mouth moves. A turn with an unlinked speaker label while exactly one visible character's mouth moves is a likely link. A turn whose speaker is a named character with no box in this cut, while an unknown or unmarked person's mouth moves, probably means that person is that character and is missing a box. How: each turn's delivery in the beats against the mouth its character shows{mouth}. on_screen with a mouth that never moves (it is probably an inner voice or narration, or the keyframes missed the mouth), or inner_voice with a moving mouth, is a lead. A turn whose speaker is not in frame is an off-screen voice and not a problem. For older cuts, the video analysis' speaking value plays this part.
4. The motion: the states and changes the keyframes report, gaze shifts, travel, objects handled, the end state, and whether the beats follow the keyframes (a beat that has a character walk where no keyframe pair shows it, or a character who leaves the frame and is still acting in a later beat, is a real problem). A character who is absent from the historical end-of-cut confirmation is not thereby gone.'''
MATTERS_TEXT = {'seen': '', 'fit': ' is a question when the evidence cannot settle it', 'mouth': ' at the keyframes'}
MATTERS_SEEN = {'seen': ' as the pictures show them', 'fit': ' is a reason to look at that picture again; it is a question when the picture does not settle it', 'mouth': ' in its line strip and at the keyframes'}

NOT_A_QUESTION = '''## NOT A QUESTION

- Colour shades or tones of hair, eyes or clothes, small accessories, and any difference of wording or of detail between two sources.
- Angles that differ by less than a quarter turn, or a gaze that differs by one sector.
- A state or gaze that the keyframes show changing after the first frame. The first frame decides how the cut starts, the keyframes (for older cuts, the video analysis) decide what changes afterwards, so a difference between them is movement, not a contradiction. For older cuts, state_correction means the still caught a mid-motion pose and read it as static: the video's activity is the character's state during the clip, decided by this rule.
- A transcript line from someone who is not in frame; characters who speak while stationary; a change dated to its pair of keyframes rather than to an exact second; a delivery that differs from the app's suggestion when the keyframes support it.{pictures}
- Anything that would not change the generated picture, motion or lip sync. When in doubt, leave it out. An empty questions list is the right answer for a cut whose evidence agrees.'''
NOT_A_QUESTION_SEEN = '\n- A reading the pictures confirm, and a difference no picture can settle (too small, blurred, hidden).'

QUESTIONS = '''## QUESTIONS

- Ask at most 5, most important first: the still, then expression and emotion, then voices, then the motion. Merge points about the same decision. Fewer is better.
- One decision per question, written as one sentence that names the character. Give 2 to 4 options. Each option is a complete statement that could be pasted into the production brief as a fact, in the same words the evidence uses, and each option is something the evidence supports or at least allows. Never offer "unknown", "other" or "not sure": the person can always type their own answer.{seen}
- recommended is the id of the option the evidence favours (the source rule above decides ties), or null when it is really open. why is one short sentence naming the sources that disagree (first frame, keyframes, performance beats, video analysis, transcript, app check, episode{why}).
- topic is one of: pose (torso, head, body state at the start), gaze, state (what the body does during the clip), speaking, emotion, position, identity, camera, objects, other. character is the exact name, or an empty string when the question is about the whole shot.'''
QUESTIONS_SEEN = {'seen': ' A contradiction you saw in a picture is a question: one option states what the picture shows, in the evidence\'s words, and is the recommended one; another states the stored reading.', 'why': ', and the picture label such as F3 or the line strip'}

OUTPUT = '''## OUTPUT FIELDS

Return exactly one JSON object with these fields:
{check}
**summary:** What the cut shows, in at most three sentences: who is on screen and where, what happens (motion and camera), and who speaks. Use the character names.

**clear:** What can be used for generation as it is. Three short parts in this order: "Still:" (the opening image, with each character's expression and emotion), "Motion:" (camera, each character's state, travel, gaze, objects, emotion change and end state, beat by beat when the beats hold), "Speech:" (each turn's speaker, whether they are on screen and its delivery). Use the character names.

**confusing:** Only material contradictions or evidence gaps that would change the still, the audio or the motion, naming the sources, claims and timestamps. If none exist, return "None identified."

**questions:** The array described above, [] when nothing needs a decision. Each question is {{"id": "q1", "topic": "...", "character": "...", "question": "...", "options": [{{"id": "a", "text": "..."}}, {{"id": "b", "text": "..."}}], "recommended": "a", "why": "..."}}.

Before returning, check that every question concerns a genuine disagreement or gap that affects generation, that no option merely restates another, and that the JSON is syntactically valid.

Return JSON only. No Markdown, explanations, or additional fields.'''
OUTPUT_CHECK = '''
**visual_check:** Written first. The checks described under LOOKING AT THE PICTURES, [] when no picture shows a face, pose or mouth the evidence describes. Each is {"character": "...", "picture": "<picture label>", "source": "first_frame|keyframes|beats|transcript", "claim": "the stored reading, in a few words", "seen": "the visible cues, then what they add up to", "verdict": "agrees|contradicts|unclear"}.
'''


def system(vision=False):
    """The reviewer's instructions; ``vision`` adds the pictures and the visual checks."""
    words = MATTERS_SEEN if vision else MATTERS_TEXT
    parts = [INTRO.format(doubt=DOUBT[vision], looking=LOOKING[vision]), EVIDENCE]
    if vision:
        parts.append(PICTURES)
    parts += [RULES, MATTERS.format(**words), NOT_A_QUESTION.format(pictures=NOT_A_QUESTION_SEEN if vision else ''),
              QUESTIONS.format(**(QUESTIONS_SEEN if vision else {'seen': '', 'why': ''})), OUTPUT.format(check=OUTPUT_CHECK if vision else '')]
    return '\n\n'.join(parts)


SYSTEM = system(False)
FIELDS = ['summary', 'clear', 'confusing', 'questions']
CHECK_FIELDS = ['visual_check', *FIELDS]
LEGACY_FIELDS = ['summary', 'clear', 'confusing', 'human_review', 'recommended_checks']
TOPICS = ('pose', 'gaze', 'state', 'speaking', 'emotion', 'position', 'identity', 'camera', 'objects', 'other')
MAX_QUESTIONS = 5
MAX_CHECKS = 12
CHECK_SOURCES = ('first_frame', 'keyframes', 'beats', 'transcript')
VERDICTS = ('contradicts', 'unclear', 'agrees')  # the stored order
CHECKED_PICTURES = ('first', 'keyframe', 'line')  # pictures of the cut itself, which a visual check may name
MAX_OPTIONS = 4
OPTION_IDS = 'abcd'
_TOPIC_ALIASES = {'facing': 'pose', 'orientation': 'pose', 'head': 'pose', 'torso': 'pose', 'posture': 'state', 'activity': 'state', 'action': 'state', 'look': 'gaze', 'eyes': 'gaze',
                  'speech': 'speaking', 'dialogue': 'speaking', 'voice': 'speaking', 'expression': 'emotion', 'face': 'emotion', 'prop': 'objects', 'object': 'objects',
                  'character': 'identity', 'who': 'identity', 'framing': 'camera', 'location': 'position', 'placement': 'position'}


def question_format(labels=None):
    """``response_format`` (a llama.cpp grammar) for the consistency summary: three strings and at most five questions.

    With picture ``labels`` the answer starts with ``visual_check``, so the model writes what it sees before it
    summarizes or asks; each check names one of those pictures.
    """
    string = {'type': 'string'}
    option = {'type': 'object', 'additionalProperties': False, 'required': ['id', 'text'], 'properties': {'id': {'enum': list(OPTION_IDS)}, 'text': string}}
    question = {'type': 'object', 'additionalProperties': False, 'required': ['id', 'topic', 'character', 'question', 'options', 'recommended', 'why'],
                'properties': {'id': string, 'topic': {'enum': list(TOPICS)}, 'character': string, 'question': string,
                               'options': {'type': 'array', 'items': option, 'minItems': 2, 'maxItems': MAX_OPTIONS},
                               'recommended': {'type': ['string', 'null'], 'enum': [*OPTION_IDS, None]}, 'why': string}}
    properties = {'summary': string, 'clear': string, 'confusing': string, 'questions': {'type': 'array', 'items': question, 'maxItems': MAX_QUESTIONS}}
    required = FIELDS
    if labels:
        check = {'type': 'object', 'additionalProperties': False, 'required': ['character', 'picture', 'source', 'claim', 'seen', 'verdict'],
                 'properties': {'character': string, 'picture': {'enum': list(labels)}, 'source': {'enum': list(CHECK_SOURCES)}, 'claim': string, 'seen': string,
                                'verdict': {'enum': list(VERDICTS)}}}
        properties, required = {'visual_check': {'type': 'array', 'items': check, 'maxItems': MAX_CHECKS}, **properties}, CHECK_FIELDS
    schema = {'type': 'object', 'additionalProperties': False, 'required': required, 'properties': properties}
    return {'type': 'json_schema', 'json_schema': {'name': 'consistency_summary', 'strict': True, 'schema': schema}}


def _clean(value, limit):
    return ' '.join(value.split())[:limit] if isinstance(value, str) else ''


def normalize_questions(items):
    """The model's questions as stored: at most five, each with two to four distinct options and ids that follow the order.

    Never raises for a malformed entry; it is dropped (a question with fewer than two usable options cannot be answered
    by choosing). ``recommended`` may be an option id, its position or its text, and becomes an id or None.
    """
    found = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or len(found) >= MAX_QUESTIONS:
            continue
        text = _clean(item.get('question'), 300)
        raw = item.get('options') if isinstance(item.get('options'), list) else []
        options, originals = [], []
        for index, option in enumerate(raw):
            label = _clean(option.get('text') or option.get('label') or option.get('option') if isinstance(option, dict) else option, 240)
            if not label or label.casefold() in (existing['text'].casefold() for existing in options) or len(options) >= MAX_OPTIONS:
                continue
            options.append({'id': OPTION_IDS[len(options)], 'text': label})
            originals.append((index, option.get('id') if isinstance(option, dict) else None, label))
        if not text or len(options) < 2:
            continue
        wanted = item.get('recommended')
        recommended = None
        for new, (index, old_id, label) in zip(options, originals):
            if wanted is not None and (wanted == old_id or wanted == index or (isinstance(wanted, str) and (wanted.strip().casefold() == label.casefold() or wanted.strip().lower() == new['id']))):
                recommended = new['id']
                break
        topic = _clean(item.get('topic'), 30).lower()
        topic = topic if topic in TOPICS else _TOPIC_ALIASES.get(topic, 'other')
        found.append({'id': f'q{len(found) + 1}', 'topic': topic, 'character': _clean(item.get('character'), 80), 'question': text, 'options': options,
                      'recommended': recommended, 'why': _clean(item.get('why'), 240)})
    return found


def normalize_checks(items, labels):
    """The visual checks as stored: each names an offered picture and a known verdict, contradictions first, at most twelve."""
    known = {label.casefold(): label for label in labels or []}
    found = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        picture = known.get(_clean(item.get('picture'), 80).casefold())
        verdict = _clean(item.get('verdict'), 20).lower()
        claim, seen = _clean(item.get('claim'), 300), _clean(item.get('seen'), 400)
        if not picture or verdict not in VERDICTS or not (claim or seen):
            continue
        source = _clean(item.get('source'), 20).lower()
        found.append({'character': _clean(item.get('character'), 80), 'picture': picture, 'source': source if source in CHECK_SOURCES else 'first_frame',
                      'claim': claim, 'seen': seen, 'verdict': verdict})
    found.sort(key=lambda item: VERDICTS.index(item['verdict']))
    return found[:MAX_CHECKS]


def validate(output, labels=None):
    """The parsed answer as stored. A ValueError goes back to the model with its answer.

    With picture ``labels`` (the review looked at pictures) the result keeps ``visual_check``; without, it has none.
    """
    summary = _clean(output.get('summary'), 4000)
    if not summary:
        raise ValueError('summary must be a non-empty string')
    if not isinstance(output.get('questions'), list):
        raise ValueError('questions must be an array of questions (use [] when nothing needs a decision)')
    stored = {'summary': summary, 'clear': _clean(output.get('clear'), 6000), 'confusing': _clean(output.get('confusing'), 6000) or 'None identified.',
              'questions': normalize_questions(output['questions'])}
    if labels:
        if not isinstance(output.get('visual_check'), list):
            raise ValueError('visual_check must be an array of checks (use [] when no picture shows what the evidence describes)')
        stored = {'visual_check': normalize_checks(output['visual_check'], labels), **stored}
    return stored


def _as_text(value, indent=0):
    if isinstance(value, dict):
        return '\n'.join('  ' * indent + str(key).replace('_', ' ') + ': ' + (_as_text(child, indent + 1).lstrip() if not isinstance(child, (dict, list)) else '\n' + _as_text(child, indent + 1)) for key, child in value.items())
    if isinstance(value, list):
        return '\n'.join('  ' * indent + '- ' + (_as_text(child, indent + 1).lstrip() if not isinstance(child, (dict, list)) else '\n' + _as_text(child, indent + 1)) for child in value) or '  ' * indent + '(none)'
    return str(value) if value is not None else 'unknown'


def subtitle_evidence(subtitles):
    """Subtitle lines overlapping the cut, in the form the reviewer is told about."""
    subtitles = subtitles if isinstance(subtitles, dict) else {}
    lines = [{key: line.get(key) for key in ('start_time', 'end_time', 'text', 'confidence', 'crosses_cut')}
             for line in subtitles.get('lines') or [] if isinstance(line, dict)]
    return {'extracted': bool(subtitles.get('extracted')), 'language': subtitles.get('language'), 'lines': lines}


def _without_ids(frame):
    """A frame description with the characters named only: database ids mean nothing to a reviewer."""
    if not isinstance(frame, dict) or not isinstance(frame.get('visible_characters'), list):
        return frame
    return {**frame, 'visible_characters': [{key: value for key, value in person.items() if key != 'character_context_id'} if isinstance(person, dict) else person
                                            for person in frame['visible_characters']]}


def app_checks(result_map, segments):
    """Leads the app calculates from the evidence, so the reviewer does not have to find them: ``[{kind, ...}]``.

    Pose flags come from the first-frame geometry checks; a state difference is the video disagreeing with
    what the still read (already decided by the source rule); a transcript line is set against the visible
    speaking of its character (from the keyframes, else the video analysis); a beat line's delivery is set
    against the mouth the keyframes show. Nothing here is a verdict.
    """
    checks = []
    frame = (result_map.get('initial_frame') or {}).get('content')
    source = prompt_context.motion_source(result_map)
    video = source['content'] if source else None
    for person in (frame.get('visible_characters') if isinstance(frame, dict) else None) or []:
        pose = person.get('pose_detail') if isinstance(person, dict) else None
        for flag in (pose.get('flags') if isinstance(pose, dict) else None) or []:
            if isinstance(flag, dict) and flag.get('text'):
                checks.append({'kind': 'pose_' + str(flag.get('code')), 'detail': flag['text']})
    beats = (result_map.get('performance_beats') or {}).get('content')
    if 'keyframe_analysis' in result_map and not prompt_context.beats_current(result_map):
        checks.append({'kind': 'beats_stale', 'detail': 'the performance beats are missing or were written from an older keyframe analysis'})
    elif isinstance(beats, dict):
        for line in beats.get('speech') or []:
            checks.extend(delivery_checks(line, video))
        for item in beats.get('checks') or []:
            if isinstance(item, dict) and item.get('kind') in ('mouth_without_line', 'lip_sync_without_mouth', 'on_screen_speaker_not_seen'):
                checks.append({key: value for key, value in item.items() if key in ('kind', 'beat', 'line', 'character', 'speaker')})
    if not prompt_context.is_direction(video):
        return checks[:8]
    people = [person for person in video['direction'].get('characters') or [] if isinstance(person, dict)]
    for person in people:
        fix = person.get('state_correction')
        if isinstance(fix, dict):
            checks.append({'kind': 'state_differs', 'character': person.get('character'), 'first_frame': fix.get('first_frame'), 'video': fix.get('clip'),
                           'rule': 'the video decides the state during the clip; not a question'})
    speaking = {person.get('character'): person.get('speaking') for person in people if person.get('character')}
    speaking_yes = [name for name, value in speaking.items() if value == 'yes']
    reported = set()
    for segment in prompt_context.slim_transcript(segments):
        who, at = segment.get('character_name'), f"{segment['start_time']:.1f}-{segment['end_time']:.1f}s"
        if who and speaking.get(who) == 'no' and ('quiet', who) not in reported:
            reported.add(('quiet', who))
            checks.append({'kind': 'line_without_visible_speaking', 'character': who, 'at': at})
        elif not who and len(speaking_yes) == 1 and ('unlinked', at) not in reported:
            reported.add(('unlinked', at))
            checks.append({'kind': 'unlinked_speaker_may_be', 'character': speaking_yes[0], 'at': at})
    return checks[:8]


def delivery_checks(line, motion):
    """A beat line whose delivery disagrees with the mouth its speaker shows at the keyframes."""
    if not isinstance(line, dict) or not prompt_context.is_direction(motion):
        return []
    speaking = {person.get('character'): person.get('speaking') for person in motion['direction'].get('characters') or [] if isinstance(person, dict)}
    mouth = speaking.get(line.get('speaker'))
    if (line.get('delivery') == 'on_screen' and mouth == 'no') or (line.get('delivery') == 'inner_voice' and mouth == 'yes'):
        return [{'kind': 'delivery_against_mouth', 'character': line.get('speaker'), 'line_at': f"{line.get('start', 0):.1f}-{line.get('end', 0):.1f}s",
                 'delivery': line.get('delivery'), 'mouth_moves': mouth}]
    return []


def refresh_inputs(context):
    """What ``refresh`` needs to build the evidence again when a chained job starts: named transcript lines and the characters."""
    names = {item['id']: item['title'] for item in context.get('context_items') or [] if item.get('kind') == 'character'}
    segments = [{**prompt_context.slim_transcript([segment])[0], 'character_context_id': segment.get('character_context_id')}
                for segment in context.get('transcript_segments') or [] if prompt_context.slim_transcript([segment])]
    return {'transcript_segments': [{**segment, 'character_name': segment.get('character_name') or names.get(segment.get('character_context_id'))} for segment in segments],
            'context_items': [{'id': key, 'title': value, 'kind': 'character'} for key, value in names.items()]}


def refresh(payload, results):
    """Build the evidence again from the cut's current results (a chained job: the beats were written after it was queued).

    The transcript, subtitles, series notes and character boxes stay as they were queued.
    """
    saved, inputs = payload['evidence'], payload.get('refresh') or {}
    fresh = build_input(results, inputs.get('transcript_segments') or [], inputs.get('context_items') or [], saved.get('subtitles'), saved.get('series'))
    if saved.get('character_detections'):
        fresh['character_detections'] = saved['character_detections']
    from .summary_queue import source_ids
    payload['evidence'], payload['source_result_ids'] = fresh, source_ids(results)
    return payload


def build_input(results, transcripts, context_items, subtitles=None, series=None, episode=None):
    result_map = {item['type']: item for item in results}
    parts = []
    video = (result_map.get('video_analysis') or {}).get('content')
    keyframes = result_map.get('keyframe_analysis')
    if keyframes and keyframes.get('content') is not None:
        # Current analyses: the opening still, what changes between the keyframes, and the beats built from them.
        first = result_map.get('initial_frame')
        beats = result_map.get('performance_beats')
        parts.append('First frame:\n' + (_as_text(_without_ids(prompt_context.frame_view(first['content']))) if first and first.get('content') is not None else 'Unavailable.'))
        parts.append('Keyframes:\n' + _as_text(prompt_context.keyframe_view(keyframes['content'], 'review')))
        parts.append('Performance beats:\n' + (_as_text(prompt_context.beats_view(beats['content'])) if prompt_context.beats_current(result_map) else 'Not written yet.'))
    elif prompt_context.is_direction(video):
        # Current analyses: the opening still and one motion direction for the cut. A last-frame
        # description left over from an older analysis is not evidence about this run.
        first = result_map.get('initial_frame')
        parts.append('First frame:\n' + (_as_text(_without_ids(prompt_context.frame_view(first['content']))) if first and first.get('content') is not None else 'Unavailable.'))
        parts.append('Video analysis:\n' + _as_text(prompt_context.video_view(video)))
    else:
        for kind, label in (('initial_frame', 'Initial frame'), ('video_analysis', 'Video analysis'), ('last_frame', 'Last frame')):
            result = result_map.get(kind)
            parts.append(label + ':\n' + (_as_text(result['content']) if result and result.get('content') is not None else 'Unavailable.'))
    names = {item['id']: item['title'] for item in context_items if item.get('kind') == 'character'}
    named = [{**segment, 'character_name': segment.get('character_name') or names.get(segment.get('character_context_id'))} for segment in transcripts]
    turns = [prompt_context.transcript_line(segment) for segment in prompt_context.slim_transcript(named)]
    evidence = {'visual_evidence_text': '\n\n'.join(parts),
                'transcript_text': '\n'.join(turns) if turns else 'No transcript segments available.',
                'subtitles': subtitle_evidence(subtitles)}
    checks = app_checks(result_map, named)
    if checks:
        evidence['app_checks'] = checks
    if series:
        evidence['series'] = series
    if episode:
        evidence['episode'] = episode
    return evidence


def generate(config, evidence, progress=None, cancel_check=None, usage_out=None, pictures=None):
    """``(output, model)``. ``pictures`` are ``(picture, jpeg bytes)`` pairs (``consistency_evidence.render``); with them the
    reviewer gets the picture instructions and writes ``visual_check``. Without a picture of the cut itself (only
    references or neighbours could be cut) nothing is attached."""
    from .inference_worker import chat_json
    labels = [picture['label'] for picture, _ in pictures or [] if picture['kind'] in CHECKED_PICTURES]
    images = [(f"{picture['label']}: {picture['caption']}", data) for picture, data in pictures or []] if labels else []
    result, model = chat_json(config['openrouter_model'], system(bool(images)), json.dumps(evidence, ensure_ascii=False), CHECK_FIELDS if images else FIELDS, progress=progress,
                              cancel_check=cancel_check, provider=config.get('text_provider', 'openrouter'), thinking=config.get('local_thinking', False), usage_out=usage_out,
                              validator=functools.partial(validate, labels=labels or None), response_format=question_format(labels or None), images=images or None)
    return result, model
