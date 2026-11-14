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
DELIVERIES = ('lip_sync', 'off_screen', 'inner_voice', 'narration')
STYLE_START = 'The target video is'


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


def spoken(dialogue, content, subjects):
    """Each spoken line with the director's delivery decision and the subject number it belongs to."""
    decisions = {item['line']: item for item in content.get('speech') or []}
    lines = []
    for line in dialogue:
        decision = decisions.get(line['line']) or {}
        number = subject_number(decision.get('subject'), subjects)
        delivery = decision.get('delivery') if decision.get('delivery') in DELIVERIES else 'off_screen'
        if delivery in ('lip_sync', 'inner_voice') and number is None:
            delivery = 'off_screen'
        lines.append({**line, 'delivery': delivery, 'subject': number if delivery in ('lip_sync', 'inner_voice') else None})
    return lines


def speech_sentence(line, language):
    at, words = h3_prompt.fmt(line['t0']), f"<d>[{language}] {line['text']}</d>"
    if line['delivery'] == 'lip_sync':
        return f"At {at}, <Subject {line['subject']}> (S{line['subject']}) says: {words}"
    if line['delivery'] == 'inner_voice':
        return f"At {at}, the inner voice of <Subject {line['subject']}> is heard while the lips stay closed: {words}"
    if line['delivery'] == 'narration':
        return f"At {at}, a narrator is heard: {words}"
    return f"At {at}, an off-screen voice is heard: {words}"


def audio_lines(lines):
    """What ``<Audio 1>`` drives: whose lips follow it, and which words nobody on screen mouths."""
    if not lines:
        return ['<Audio 1> carries no speech; match its ambience and timing.']
    out, seen = [], set()
    for line in lines:
        key = (line['delivery'], line['subject'])
        if key in seen:
            continue
        seen.add(key)
        at = h3_prompt.fmt(line['t0'])
        if line['delivery'] == 'lip_sync':
            out.append(f"<Subject {line['subject']}> speaks the words of <Audio 1> in its voice, lips in sync, from {at}.")
        elif line['delivery'] == 'inner_voice':
            out.append(f"The words of <Audio 1> from {at} are the inner voice of <Subject {line['subject']}>; the lips stay closed.")
        elif line['delivery'] == 'narration':
            out.append(f"A narrator speaks the words of <Audio 1> from {at}; no lips on screen move for them.")
        else:
            out.append(f"An off-screen voice speaks the words of <Audio 1> from {at}; no lips on screen move for them.")
    return out


def render(shot, content, bible, subjects, dialogue):
    """The H3 prompt of a shot, from ``subject_definitions:`` to ``non_diegetic_music:``."""
    language = bible.get('spoken_language') or 'Japanese'
    lines = spoken(dialogue, content, subjects)
    people = [subject for subject in subjects if subject['kind'] != 'scenario']

    definitions = [f"<Picture 1> is the first frame of [Shot 1]: {sentence(content.get('picture_line'))}"]
    for subject in subjects:
        if subject['kind'] == 'scenario':
            definitions.append(f"{subject['label']} is {subject['name']}, the setting from <Picture {subject['picture']}>: {sentence(subject['anchors'])}")
        elif subject['picture']:
            definitions.append(f"{subject['label']} is {subject['name']}, whose face, hair and clothing come from <Picture {subject['picture']}>: {sentence(subject['anchors'])}")
        else:
            definitions.append(f"{subject['label']} is {subject['name']}, seen in <Picture 1>: {sentence(subject['anchors'])}")
    carries = 'the voices and ambience to keep in sync' if lines else 'the ambience and timing to match'
    definitions.append(f"<Audio 1> is the original sound of this shot, “{audio_file(shot)}”: {carries}.")

    retention = [f"<Picture 1> ([Shot 1] first frame): fully_preserved - {sentence(content.get('keep_picture') or 'composition, style, palette and lighting of the opening frame')}"]
    retention.extend(f"{subject['label']} (appears in [Shot 1]): fully_preserved - {sentence(subject['keep'] or 'appearance follows its reference exactly')}" for subject in subjects)

    speaking = {line['subject'] for line in lines if line['delivery'] == 'lip_sync'}
    # A named character has a face whose lips H3 might move to the audio; a subject without a record can be anything
    # the director listed (a passer-by, a hand), so it only gets the plain statement.
    silent = [f"{subject['label']} does not speak; the lips stay closed." if subject['kind'] == 'character' else f"{subject['label']} does not speak."
              for subject in people if subject['number'] not in speaking]
    opening = ' '.join(part for part in (
        f"[Shot 1] The shot begins from <Picture 1>: {sentence(label_names(content.get('shot_line'), subjects))}",
        sentence(label_names(content.get('framing_lock'), subjects)), sentence(label_names(content.get('light'), subjects)), *silent) if part)
    ranges = beat_ranges(content, shot)
    description = [sentence(bible.get('style') or f'{STYLE_START} an animated shot in the style of <Picture 1>'), opening]
    for index, beat in enumerate(ranges):
        parts = [sentence(beat['action'] if beat.get('hold') else label_names(beat['action'], subjects))]
        if index == 0:
            parts.append(camera_sentence(content.get('camera')))
            detail = sentence(label_names((content.get('camera') or {}).get('detail'), subjects))
            if detail:
                parts.append(detail)
        last = index == len(ranges) - 1
        parts.extend(speech_sentence(line, language) for line in lines if beat['start'] <= line['t0'] < beat['end'] or (last and line['t0'] >= beat['end']))
        description.append(f"{h3_prompt.fmt(beat['start'])}-{h3_prompt.fmt(beat['end'])}: {' '.join(parts)}")

    soundscape = ' '.join([sentence(label_names(content.get('soundscape'), subjects)), *audio_lines(lines)]).strip()
    return '\n'.join([
        'subject_definitions:', *definitions, '',
        'summary:', f"{h3_prompt.OMNI_PREFIX} {sentence(label_names(content.get('summary'), subjects))}", '',
        'retention_analysis:', *retention, '',
        'detailed_description:', *description, '',
        'overall_soundscape:', soundscape, '',
        'non_diegetic_music:', 'N/A'])


def still_prompt(content, bible):
    """A generator-agnostic prompt for the opening still: one frozen moment, no motion, no text."""
    still = content.get('still') or {}
    style = clause(bible.get('style'))
    style = style[len(STYLE_START):].strip() if style.startswith(STYLE_START) else style
    rows = [('Shot', still.get('shot')), ('Subjects', still.get('subjects')), ('Objects and layout', still.get('layout')), ('Setting', still.get('setting')),
            ('Lighting and colour', still.get('lighting')), ('Style', style[:1].upper() + style[1:] if style else None),
            ('Constraints', 'one frozen moment; no text, subtitles, captions, logos or watermarks; no additional characters')]
    return '\n'.join(f'{title}: {sentence(value)}' for title, value in rows if clause(value))


def check(prompt):
    """Format verdict of a rendered prompt (errors block, warnings advise)."""
    result = h3_prompt.check(prompt)
    return {'ok': result['ok'], 'errors': result['errors'], 'warnings': result['warnings'], 'stats': result['stats']}


def build(shot, content, bible, roster, scenario, dialogue):
    """Render a shot and everything the download declares about it.

    Returns ``(prompt, check, derived)`` where ``derived`` holds the numbered subjects and references,
    the dialogue with deliveries, the still prompt and the audio file name the prompt quotes.
    """
    subjects, references = assign(content, roster, scenario)
    prompt = render(shot, content, bible, subjects, dialogue)
    lines = spoken(dialogue, content, subjects)
    derived = {'subjects': subjects, 'references': references, 'audio_file': audio_file(shot), 'still_prompt': still_prompt(content, bible),
               'dialogue': [{key: line.get(key) for key in ('line', 't0', 't1', 'text', 'speaker_label', 'character_context_id', 'character_name', 'attribution', 'delivery', 'subject', 'crosses_start', 'crosses_end')} for line in lines]}
    return prompt, check(prompt), derived
