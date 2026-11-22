"""Render a directed shot as a MiniMax H3 prompt and a still-image prompt.

The director model writes prose and decisions; everything that must be exact is written here:
the six sections and their order, ``<Picture N>``/``<Subject N>``/``<Audio N>`` numbering, timecodes,
contiguous beats, spoken words copied from the transcript, the camera sentence in H3 vocabulary and
the audio lines. The result is checked with the same rules Cut Assistance uses (``h3_prompt.check``).

Reference convention (declared in the download so the generating app can attach its files in this
order): ``<Picture 1>`` is the shot's initial frame, then one character sheet per named on-screen
character, then one scenario picture when the scene has a scenario note; ``<Audio 1>`` is the shot's
slice of the original audio.
"""
import re

from .assistant import h3_prompt

MAX_SHEETS = 4  # character sheets sent with one shot; further characters are described from the initial frame
CAMERA_MOVES = {
    'static': None, 'push_in': 'pushes in', 'pull_out': 'pulls out', 'pan_left': 'pans left', 'pan_right': 'pans right',
    'truck_left': 'trucks left', 'truck_right': 'trucks right', 'tilt_up': 'tilts up', 'tilt_down': 'tilts down',
    'pedestal_up': 'pedestals up', 'pedestal_down': 'pedestals down', 'arc': 'moves in an arc shot', 'tracking': 'follows in a tracking shot', 'shake': 'shakes slightly',
}
AMPLITUDES = ('small', 'medium', 'large')
SPEEDS = ('slow', 'medium', 'fast')
LOOK_SENTENCE = 'The shot keeps the look of <Picture 1> unchanged: its rendering, palette and lighting.'
# How a speaker is heard during part of a shot. on_screen: visible with the mouth visible, lip-synced.
# off_screen: not in the picture. inner_voice: visible, the mouth does not move for the words (a thought,
# or the face turned away). narrative: visible or not, too far from the camera to lip-sync; the words play over the picture.
DELIVERIES = ('on_screen', 'off_screen', 'inner_voice', 'narrative')
NEEDS_SUBJECT = ('on_screen', 'inner_voice')
LEGACY_DELIVERY = {'lip_sync': 'on_screen', 'narration': 'narrative'}  # per-line values of shots written before 2026-11-14
NARRATOR = 'Narrator'
UNKNOWN_SPEAKER = 'Unknown speaker'


def sentence(text):
    """One clean line ending in punctuation."""
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    return text if not text or text[-1] in '.!?…”"' else text + '.'


def clause(text):
    """A phrase used inside a sentence the app finishes: no trailing period."""
    return re.sub(r'\s+', ' ', str(text or '')).strip().rstrip('.').strip()


def audio_file(shot):
    return f"shot-{shot['index']:02d}.wav"


def assign(content, roster, scenario=None):
    """Number the subjects and pictures of a shot.

    ``roster`` maps exact character names to their context ids. A subject whose name is on the roster
    is a character and gets a sheet picture (the first ``MAX_SHEETS`` of them); any other subject is
    an unnamed person described from the initial frame. The scenario, when there is one, is the last
    subject and the last picture.
    """
    ids = {person['name']: person['character_context_id'] for person in roster}
    subjects, pictures = [], [{'label': '<Picture 1>', 'role': 'initial_frame'}]
    for item in content.get('subjects') or []:
        number = len(subjects) + 1
        subject = {'number': number, 'label': f'<Subject {number}>', 'name': item['name'], 'kind': 'person', 'character_context_id': None, 'picture': None,
                   'anchors': item.get('anchors') or '', 'keep': item.get('keep') or ''}
        if item['name'] in ids:
            subject.update(kind='character', character_context_id=ids[item['name']])
            if sum(1 for picture in pictures if picture['role'] == 'character_sheet') < MAX_SHEETS:
                subject['picture'] = len(pictures) + 1
                pictures.append({'label': f"<Picture {subject['picture']}>", 'role': 'character_sheet', 'character_context_id': ids[item['name']], 'name': item['name']})
        subjects.append(subject)
    if scenario:
        number = len(subjects) + 1
        picture = len(pictures) + 1
        pictures.append({'label': f'<Picture {picture}>', 'role': 'scenario', 'context_id': scenario['id'], 'name': scenario['title']})
        subjects.append({'number': number, 'label': f'<Subject {number}>', 'name': scenario['title'], 'kind': 'scenario', 'character_context_id': None, 'context_id': scenario['id'],
                         'picture': picture, 'anchors': content.get('scenario_anchors') or '', 'keep': content.get('scenario_keep') or ''})
    return subjects, [*pictures, {'label': '<Audio 1>', 'role': 'original_audio'}]


def _aliases(subjects):
    """Texts that stand for a subject in prose, longest first: the full name, and single name words no other subject shares."""
    words = {}
    for subject in subjects:
        if subject['kind'] == 'character':
            for word in subject['name'].split():
                words.setdefault(word, set()).add(subject['number'])
    aliases = [(subject['name'], subject) for subject in subjects]
    for subject in subjects:
        if subject['kind'] == 'character' and len(subject['name'].split()) > 1:
            aliases.extend((word, subject) for word in subject['name'].split() if len(word) >= 3 and len(words[word]) == 1)
    return sorted(aliases, key=lambda pair: -len(pair[0]))


def label_names(text, subjects):
    """Replace subject names in prose with their ``<Subject N>`` labels."""
    text = str(text or '')
    for alias, subject in _aliases(subjects):
        ascii_name = alias.isascii()
        pattern = (r'(?<![\w<])' if ascii_name else '') + re.escape(alias) + (r'(?!\w)' if ascii_name else '')
        text = re.sub(pattern, subject['label'], text, flags=0 if subject['kind'] == 'character' else re.IGNORECASE)
    return text


def subject_number(name, subjects):
    wanted = str(name or '').strip().casefold()
    return next((subject['number'] for subject in subjects if subject['kind'] != 'scenario' and subject['name'].casefold() == wanted), None)


def beat_ranges(content, shot):
    """Contiguous beats from 0 to the generated length; a shot shorter than H3's minimum ends on a hold."""
    ranges, start = [], 0.0
    beats = content.get('beats') or []
    for index, beat in enumerate(beats):
        end = shot['duration'] if index == len(beats) - 1 else min(float(beat['end']), shot['duration'])
        if end - start > 0.001:
            ranges.append({'start': round(start, 3), 'end': round(end, 3), 'action': beat['action']})
            start = end
    if shot['generate_seconds'] - shot['duration'] > 0.001:
        ranges.append({'start': round(shot['duration'], 3), 'end': shot['generate_seconds'], 'hold': True,
                       'action': 'Everyone holds the final pose and the camera stays where it ended; nothing new enters the frame.'})
    return ranges


def camera_sentence(camera):
    camera = camera or {}
    verb = CAMERA_MOVES.get(camera.get('move'))
    if not verb:
        return 'The camera holds a static shot.'
    if camera.get('move') == 'shake':
        text = f"The camera {verb} at {camera.get('speed') or 'slow'} speed."
    else:
        text = f"The camera {verb} with {camera.get('amplitude') or 'small'} amplitude at {camera.get('speed') or 'slow'} speed."
    return text


def delivery_of(value):
    """A delivery in the current vocabulary, translating the per-line values of older shots; None when unknown."""
    value = LEGACY_DELIVERY.get(value, value)
    return value if value in DELIVERIES else None


def speaker_key(line):
    """The name a shot's voices use for the speaker of a line: the resolved character (or Narrator), else the voice label."""
    return line.get('character_name') or line.get('speaker_label') or UNKNOWN_SPEAKER


def speakers(dialogue):
    """The speakers heard in a shot, in the order they first speak."""
    return list(dict.fromkeys(speaker_key(line) for line in dialogue))


def normal_voices(entries, duration):
    """Voice entries per speaker in time order: the first starts at 0, a later entry at the same time wins,
    neighbours that change nothing are joined, and each entry gets ``to`` (the next entry's start, or the shot's end)."""
    by_speaker = {}
    for entry in entries:
        by_speaker.setdefault(entry['speaker'], []).append(entry)
    out = []
    for items in by_speaker.values():
        joined = []
        for entry in sorted(items, key=lambda item: item['from']):
            entry = {key: entry.get(key) for key in ('speaker', 'from', 'delivery', 'subject', 'fixed')}
            if joined and entry['from'] - joined[-1]['from'] < 0.05:
                joined[-1] = {**entry, 'from': joined[-1]['from']}
            elif not joined or (entry['delivery'], entry['subject'], bool(entry['fixed'])) != (joined[-1]['delivery'], joined[-1]['subject'], bool(joined[-1]['fixed'])):
                joined.append(entry)
        joined[0]['from'] = 0.0
        for entry, following in zip(joined, [*joined[1:], None]):
            entry['from'] = round(min(max(0.0, float(entry['from'])), duration), 3)
            entry['to'] = round(following['from'] if following else duration, 3)
            entry['fixed'] = bool(entry['fixed']) or None
        out.extend({key: value for key, value in entry.items() if key != 'fixed' or value} for entry in joined if entry['to'] - entry['from'] > 0.001 or len(joined) == 1)
    return out


def overlay(entries, fixed, duration):
    """The director's voice entries with the user's ranges laid over them; a user range keeps the director's subject for that speaker.

    ``entries`` are ``{speaker, from, delivery, subject}``; ``fixed`` are ``{speaker, from, to, delivery}`` on the shot's clock.
    """
    normal = normal_voices(entries, duration) if entries else []
    out = []
    for speaker in dict.fromkeys([*(entry['speaker'] for entry in normal), *(item['speaker'] for item in fixed)]):
        own = [entry for entry in normal if entry['speaker'] == speaker]
        mine = [item for item in fixed if item['speaker'] == speaker]
        if not mine:
            out.extend(own)
            continue
        points = sorted({0.0, duration, *(item['from'] for item in mine), *(min(item['to'], duration) for item in mine), *(entry['from'] for entry in own)})
        pieces = []
        for start, end in zip(points, points[1:]):
            if end - start < 0.001:
                continue
            middle = (start + end) / 2
            user = next((item for item in reversed(mine) if item['from'] <= middle < item['to']), None)
            base = next((entry for entry in own if entry['from'] <= middle < entry['to']), None)
            if user:
                subject = (base or {}).get('subject') or (speaker if user['delivery'] in NEEDS_SUBJECT else None)
                pieces.append({'speaker': speaker, 'from': start, 'delivery': user['delivery'], 'subject': subject if user['delivery'] != 'off_screen' else None, 'fixed': True})
            elif base:
                pieces.append({**base, 'from': start, 'fixed': None})
        if pieces:
            out.extend(normal_voices(pieces, duration))
    return out


def voices_of(content, dialogue, duration):
    """A direction's voices; shots written before voices existed are read from their per-line ``speech``."""
    if isinstance(content.get('voices'), list):
        entries = [{**entry, 'delivery': delivery_of(entry.get('delivery')) or 'off_screen'} for entry in content['voices']
                   if isinstance(entry, dict) and entry.get('speaker') and isinstance(entry.get('from'), (int, float)) and not isinstance(entry.get('from'), bool)]
        return normal_voices(entries, duration) if entries else []
    by_line = {line['line']: line for line in dialogue}
    entries = []
    for item in content.get('speech') or []:
        line = by_line.get(item.get('line')) if isinstance(item, dict) else None
        if line is None:
            continue
        entries.append({'speaker': speaker_key(line), 'from': line['t0'], 'delivery': delivery_of(item.get('delivery')) or 'off_screen', 'subject': item.get('subject')})
    return normal_voices(entries, duration) if entries else []


def voice_at(voices, speaker, t):
    """The voice entry of ``speaker`` in force at second ``t`` of the shot."""
    items = [entry for entry in voices if entry['speaker'] == speaker]
    return next((entry for entry in reversed(items) if entry['from'] <= t + 1e-6), items[0] if items else None)


def spoken(dialogue, content, subjects, duration=None):
    """Each spoken line with the delivery its speaker has at the middle of the line, and the subject number that carries it."""
    duration = duration if duration is not None else max([line['t1'] for line in dialogue] or [0.0])
    voices = voices_of(content, dialogue, duration)
    lines = []
    for line in dialogue:
        speaker = speaker_key(line)
        entry = voice_at(voices, speaker, (line['t0'] + line['t1']) / 2) or {}
        delivery = delivery_of(entry.get('delivery')) or 'off_screen'
        number = subject_number(entry.get('subject'), subjects) if delivery != 'off_screen' else None
        if delivery in NEEDS_SUBJECT and number is None:
            delivery = 'off_screen'  # nobody visible can carry the voice; the validator normally prevents this
        lines.append({**line, 'speaker': speaker, 'delivery': delivery, 'subject': number,
                      'range': [entry['from'], entry['to']] if 'to' in entry else None, 'fixed_by_user': bool(entry.get('fixed'))})
    return lines


def speech_sentence(line, language, slot=None):
    """One spoken line as H3 writes it; with ``slot`` (``1.M``) its start and words become placeholders of the prompt template."""
    at = f'{{start {slot}}}' if slot else h3_prompt.fmt(line['t0'])
    text = f'{{line {slot}}}' if slot else line['text']
    words = f'<d>[{language}] {text}</d>'
    if line['delivery'] == 'on_screen':
        return f"At {at}, <Subject {line['subject']}> (S{line['subject']}) says: {words}"
    if line['delivery'] == 'inner_voice':
        return f"At {at}, the voice of <Subject {line['subject']}> is heard while its lips stay closed: {words}"
    if line['delivery'] == 'narrative':
        if line['subject']:
            return f"At {at}, the voice of <Subject {line['subject']}> plays over the picture without lip sync: {words}"
        return f"At {at}, {'a narrator' if line.get('speaker') == NARRATOR else 'a voice'} is heard over the picture: {words}"
    return f"At {at}, an off-screen voice is heard: {words}"


# The sound section never describes the audio: nobody who writes the prompt has heard it, and a guess ("a quiet room")
# fights whatever music, steps or ambience the original really carries. It names generic kinds of sound only to say that
# every one of them is kept (2026-11-22: H3 dropped secondary voices, background talk and splashes it was not told to keep),
# and asks for the picture's actions to land on their own sounds.
SOUND_SENTENCE = ('Reproduce <Audio 1> completely, exactly as recorded: every sound in it keeps its own time and level, the speech and also any secondary voices, '
                  'indistinct background talk, cries, effects, ambience and music. Nothing in it is removed, muted, cleaned up or replaced, and nothing is added. '
                  'Each visible action that makes a sound lands exactly on that sound in <Audio 1>.')
NO_SPEECH = '<Audio 1> holds no dialogue for anyone on screen; any voices in it are background sound and stay as recorded.'
SOUND = {'sentence': SOUND_SENTENCE, 'silent': NO_SPEECH}
# The sound wording of shots written before 2026-11-22. Their stored prompts keep it, so their template is rendered with it.
LEGACY_SOUND = {'sentence': 'Follow <Audio 1> exactly for all sound: its background sound, music and effects stay as recorded, nothing is added to them, '
                            'and every movement keeps to its timing.',
                'silent': '<Audio 1> carries no speech.'}


def audio_lines(lines, slots=None, silent=NO_SPEECH):
    """What ``<Audio 1>`` drives, per speaker and voice range: whose lips follow it, and which words nobody on screen mouths.

    With ``slots`` (line id to ``1.M``) a run's span names the start of its first line and the end of its last one as placeholders.
    """
    if not lines:
        return [silent]
    runs = {}
    for line in lines:
        key = (line['speaker'], tuple(line['range'] or ()), line['delivery'], line['subject'])
        run = runs.setdefault(key, {'t0': line['t0'], 't1': line['t1'], **line, 'first': line['line'], 'last': line['line']})
        if line['t0'] < run['t0']:
            run['t0'], run['first'] = line['t0'], line['line']
        if line['t1'] > run['t1']:
            run['t1'], run['last'] = line['t1'], line['line']
    out = []
    for run in sorted(runs.values(), key=lambda item: item['t0']):
        if slots:
            span = f"from {{start {slots[run['first']]}}} to {{end {slots[run['last']]}}}"
        else:
            span = f"from {h3_prompt.fmt(run['t0'])} to {h3_prompt.fmt(run['t1'])}"
        subject = f"<Subject {run['subject']}>" if run['subject'] else None
        if run['delivery'] == 'on_screen':
            out.append(f"{subject} speaks the words of <Audio 1> in its voice, lips in sync, {span}.")
        elif run['delivery'] == 'inner_voice':
            out.append(f"The words of <Audio 1> {span} are the voice of {subject}; its lips stay closed.")
        elif run['delivery'] == 'narrative':
            who = f"the voice of {subject}, too far from the camera for lip sync" if subject else ('a narrator' if run['speaker'] == NARRATOR else 'a voice heard over the picture')
            out.append(f"The words of <Audio 1> {span} are {who}; they play over the picture and no lips move for them.")
        else:
            out.append(f"An off-screen voice speaks the words of <Audio 1> {span}; no lips on screen move for them.")
    return out


def voice_notes(subjects, lines):
    """One sentence per person on screen about the mouth: closed, not lip-synced, or moving only for its own words."""
    notes = []
    for subject in subjects:
        if subject['kind'] == 'scenario':
            continue
        own = {line['delivery'] for line in lines if line['subject'] == subject['number']}
        label = subject['label']
        if not own or own == {'inner_voice'}:
            if not own:
                notes.append(f"{label} does not speak; the lips stay closed." if subject['kind'] == 'character' else f"{label} does not speak.")
            else:
                notes.append(f"{label} keeps the lips closed while its voice is heard.")
        elif 'on_screen' not in own:
            notes.append(f"{label} is too far from the camera for lip sync; its voice plays over the picture.")
        elif own != {'on_screen'}:
            notes.append(f"{label} moves the lips only while its own on-screen words are heard.")
    return notes


def render(shot, content, bible, subjects, dialogue, slots=None, sound=None):
    """The H3 prompt of a shot, from ``subject_definitions:`` to ``non_diegetic_music:``.

    With ``slots`` (line id to ``1.M``, see ``line_slots``) the result is the prompt template: the spoken language, each
    line's words and the times taken from a line are placeholders, every other time stays as written. ``sound`` is the
    app's sound wording (``SOUND``, or ``LEGACY_SOUND`` to rebuild a prompt written before it changed).
    """
    sound = sound or SOUND
    language = '{language}' if slots else spoken_language(bible)
    lines = spoken(dialogue, content, subjects, shot['duration'])

    definitions = [f"<Picture 1> is the first frame of [Shot 1]: {sentence(content.get('picture_line'))}"]
    for subject in subjects:
        if subject['kind'] == 'scenario':
            definitions.append(f"{subject['label']} is {subject['name']}, the setting from <Picture {subject['picture']}>: {sentence(subject['anchors'])}")
        elif subject['picture']:
            definitions.append(f"{subject['label']} is {subject['name']}, whose face, hair and clothing come from <Picture {subject['picture']}>: {sentence(subject['anchors'])}")
        else:
            definitions.append(f"{subject['label']} is {subject['name']}, seen in <Picture 1>: {sentence(subject['anchors'])}")
    carries = 'its voices, sound and timing to follow' if lines else 'its sound and timing to follow'
    definitions.append(f"<Audio 1> is the original sound of this shot, “{audio_file(shot)}”: {carries}.")

    retention = [f"<Picture 1> ([Shot 1] first frame): fully_preserved - {sentence(content.get('keep_picture') or 'composition, palette and lighting of the opening frame')}"]
    retention.extend(f"{subject['label']} (appears in [Shot 1]): fully_preserved - {sentence(subject['keep'] or 'appearance follows its reference exactly')}" for subject in subjects)

    # What each person's mouth does: H3 would otherwise move any visible lips to the audio.
    opening = ' '.join(part for part in (
        f"[Shot 1] The shot begins from <Picture 1>: {sentence(label_names(content.get('shot_line'), subjects))}",
        sentence(label_names(content.get('framing_lock'), subjects)), sentence(label_names(content.get('light'), subjects)), *voice_notes(subjects, lines)) if part)
    ranges = beat_ranges(content, shot)
    description = [LOOK_SENTENCE, opening]
    for index, beat in enumerate(ranges):
        parts = [sentence(beat['action'] if beat.get('hold') else label_names(beat['action'], subjects))]
        if index == 0:
            parts.append(camera_sentence(content.get('camera')))
            detail = sentence(label_names((content.get('camera') or {}).get('detail'), subjects))
            if detail:
                parts.append(detail)
        last = index == len(ranges) - 1
        parts.extend(speech_sentence(line, language, slots[line['line']] if slots else None) for line in lines
                     if beat['start'] <= line['t0'] < beat['end'] or (last and line['t0'] >= beat['end']))
        description.append(f"{h3_prompt.fmt(beat['start'])}-{h3_prompt.fmt(beat['end'])}: {' '.join(parts)}")

    soundscape = ' '.join([sound['sentence'], *audio_lines(lines, slots, sound['silent'])])
    return '\n'.join([
        'subject_definitions:', *definitions, '',
        'summary:', f"{h3_prompt.OMNI_PREFIX} {sentence(label_names(content.get('summary'), subjects))}", '',
        'retention_analysis:', *retention, '',
        'detailed_description:', *description, '',
        'overall_soundscape:', soundscape, '',
        'non_diegetic_music:', 'N/A'])


def spoken_language(bible):
    return (bible or {}).get('spoken_language') or 'Japanese'


# Placeholders of a prompt template: the spoken language, and per line of the shot's one audio file its
# words, start, end and length (``duration`` is never written by the app; a consumer may use it).
SLOT = re.compile(r'\{(language|(?:line|start|end|duration) \d+\.\d+)\}')


def line_slots(dialogue):
    """``line_map`` of a shot: each spoken line's slot ``1.M``, M its order by start time among the lines of the shot's audio."""
    ordered = sorted(dialogue, key=lambda line: (line['t0'], line['line']))
    return {line['line']: f'1.{number}' for number, line in enumerate(ordered, 1)}


def slot_values(dialogue, slots, bible):
    """What each placeholder stands for in the prompt as written."""
    values = {'language': spoken_language(bible)}
    for line in dialogue:
        slot = slots[line['line']]
        values.update({f'line {slot}': line['text'], f'start {slot}': h3_prompt.fmt(line['t0']), f'end {slot}': h3_prompt.fmt(line['t1']),
                       f'duration {slot}': f"{max(0.0, line['t1'] - line['t0']):.3f}"})
    return values


def fill(text, values):
    """A prompt template with its placeholders replaced; a placeholder without a value is left as it is."""
    return SLOT.sub(lambda match: values.get(match.group(1), match.group(0)), text)


def template(shot, content, bible, subjects, dialogue, prompt):
    """``(prompt_template, line_map)`` of a written shot.

    The template is rendered from the stored direction like the prompt itself, never by searching the prose,
    and kept only when filling it with the shot's own words and times gives back the stored prompt exactly;
    otherwise (the plan or the bible changed since the shot was written) it is None. A prompt written with the
    earlier sound wording gets its template in that wording.
    """
    slots = line_slots(dialogue)
    if not prompt:
        return None, slots
    values = slot_values(dialogue, slots, bible)
    for sound in (SOUND, LEGACY_SOUND):
        try:
            text = render(shot, content, bible, subjects, dialogue, slots, sound)
        except (KeyError, TypeError, ValueError, AttributeError):
            return None, slots
        if fill(text, values) == prompt:
            return text, slots
    return None, slots


def still_prompt(content, bible):
    """The opening still, described for Qwen-Image-2.1."""
    still = content.get('still') or {}
    visuals = ' '.join(sentence(still[key]) for key in ('subjects', 'placement', 'setting', 'layout', 'shot', 'lighting') if clause(still.get(key)))
    look = still.get('camera_look') or ''
    camera = ' '.join(sentence(line) for line in look.splitlines() if clause(line))
    return '\n\n'.join(part for part in (visuals, f'Cinematic camera and finish. {camera}' if camera else '',
                                          'A single frozen frame with no text, subtitles, captions, logos, overlays or watermarks. Include only the described people.') if part)


def check(prompt):
    """Format verdict of a rendered prompt (errors block, warnings advise)."""
    result = h3_prompt.check(prompt)
    return {'ok': result['ok'], 'errors': result['errors'], 'warnings': result['warnings'], 'stats': result['stats']}


def build(shot, content, bible, roster, scenario, dialogue):
    """Render a shot and everything the download declares about it.

    Returns ``(prompt, check, derived)`` where ``derived`` holds the numbered subjects and references,
    the dialogue with the delivery each line gets, the shot's voice ranges, the still prompt and the
    audio file name the prompt quotes.
    """
    subjects, references = assign(content, roster, scenario)
    prompt = render(shot, content, bible, subjects, dialogue)
    lines = spoken(dialogue, content, subjects, shot['duration'])
    voices = [{**entry, 'subject_number': subject_number(entry.get('subject'), subjects), 'fixed_by_user': bool(entry.get('fixed'))}
              for entry in voices_of(content, dialogue, shot['duration'])]
    derived = {'subjects': subjects, 'references': references, 'audio_file': audio_file(shot), 'still_prompt': still_prompt(content, bible),
               'voices': [{key: entry.get(key) for key in ('speaker', 'from', 'to', 'delivery', 'subject', 'subject_number', 'fixed_by_user')} for entry in voices],
               'dialogue': [{key: line.get(key) for key in ('line', 't0', 't1', 'text', 'speaker', 'speaker_label', 'character_context_id', 'character_name', 'attribution', 'delivery', 'subject',
                                                          'fixed_by_user', 'crosses_start', 'crosses_end')} for line in lines]}
    return prompt, check(prompt), derived
