"""Deterministic checks for MiniMax H3 (FastH3 OmniRef) shot prompts.

The director's format has six sections in a fixed order. Picture, audio and
video references are assigned by the generating app (stills, then character
sheets, then scenarios), so a corrected prompt must keep exactly the reference
numbers and names of the prompt it fixes. Errors block the result; warnings
are advisory, mirroring what the generating app refuses or warns about.
"""
import re

SECTIONS = ('subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music')
HEADER = re.compile(r'^\s*(' + '|'.join(SECTIONS) + r')\s*:\s*(.*)$')
LABEL = re.compile(r'<(Picture|Audio|Video|Subject) (\d+)>')
BAD_LABEL = re.compile(r'@Image\s*\d+|\[\s*image\s*\d+\s*\]|<\s*(?:picture|audio|video|subject)\s*\d+\s*>|\b(?:Picture|Audio|Video|Subject)\d+\b', re.IGNORECASE)
SPOKEN_AT = re.compile(r'\bAt (\d{2}):(\d{2}(?:\.\d{1,3})?)\b')
BEAT = re.compile(r'^\s*(\d{2}):(\d{2}(?:\.\d{1,3})?)\s*[-–]\s*(\d{2}):(\d{2}(?:\.\d{1,3})?)\s*:')
CAMERA = re.compile(r'\b(push(?:es|ing)? in|pull(?:s|ing)? out|pan(?:s|ning)? (?:left|right)|truck(?:s|ing)? (?:left|right)|tilt(?:s|ing)? (?:up|down)|pedestal(?:s|ing)? (?:up|down)|arc shot|tracking shot|static shot|shakes? slightly)\b', re.IGNORECASE)
PROMPT_START = re.compile(r'^\s*(?:<(?:Picture|Audio|Video) \d+>|subject_definitions\s*:)', re.MULTILINE)
OMNI_PREFIX = '[keyframe completion + reference generation]'
MAX_CHARS = 7000
MAX_SECONDS = 15.0


def strip_fences(text):
    lines = [line for line in (text or '').strip().splitlines() if not re.fullmatch(r'\s*```[\w-]*\s*', line)]
    return '\n'.join(lines).strip()


def extract(message):
    """Return the H3 prompt pasted in a chat message (from its picture lines or first section), or None."""
    text = strip_fences(message)
    if 'subject_definitions' not in text:
        return None
    match = PROMPT_START.search(text)
    return text[match.start():].strip() if match else None


def parse(text):
    """Split into preamble (app picture lines), ordered sections and their bodies."""
    preamble, sections, order, current = [], {}, [], None
    for line in strip_fences(text).splitlines():
        header = HEADER.match(line)
        if header and header.group(1) not in sections:
            current = header.group(1)
            order.append(current)
            sections[current] = [header.group(2)] if header.group(2).strip() else []
        elif current is None:
            preamble.append(line)
        else:
            sections[current].append(line)
    return {'preamble': '\n'.join(preamble).strip(), 'order': order, 'sections': {key: '\n'.join(value).strip() for key, value in sections.items()}}


def labels(text):
    found = {'Picture': set(), 'Audio': set(), 'Video': set(), 'Subject': set()}
    for kind, number in LABEL.findall(text or ''):
        found[kind].add(int(number))
    return found


def mode(text):
    found = labels(text)
    if found['Video'] or '[video editing' in (text or ''):
        return 'fix'
    if not found['Picture']:
        return 'first_last_frame'
    return 'omni_reference'


def fmt(value):
    minutes, rest = divmod(value, 60)
    return f'{int(minutes):02d}:{rest:06.3f}'


def seconds(minutes, secs):
    return int(minutes) * 60 + float(secs)


def beats(description):
    out = []
    for line in description.splitlines():
        match = BEAT.match(line)
        if match:
            out.append((seconds(match.group(1), match.group(2)), seconds(match.group(3), match.group(4))))
    return out


def prose(text):
    """Text the model wrote itself: without quoted names, file names or spoken lines, which stay verbatim."""
    text = re.sub(r'<d>.*?</d>', ' ', text, flags=re.S)
    text = re.sub(r'“[^”]*”|"[^"\n]*"', ' ', text)
    return re.sub(r'\S+\.(?:wav|mp3|m4a|flac|ogg|png|jpe?g|webp|mp4|mov|webm)\b', ' ', text, flags=re.IGNORECASE)


def misplaced_times(description):
    """Timed sentences ("At 00:00.433, …") written inside a beat paragraph whose range does not contain them."""
    out, current = [], None
    for line in description.splitlines():
        if not line.strip():
            current = None
            continue
        match = BEAT.match(line)
        if match:
            current = (seconds(match.group(1), match.group(2)), seconds(match.group(3), match.group(4)))
        for spoken in SPOKEN_AT.finditer(line[match.end():] if match else line):
            at = seconds(spoken.group(1), spoken.group(2))
            if current is not None and not current[0] - 1e-6 <= at <= current[1] + 1e-6:
                out.append((f'{spoken.group(1)}:{spoken.group(2)}', f'{fmt(current[0])}-{fmt(current[1])}'))
    return out


def summary_prefix(summary):
    match = re.match(r'\s*(\[[^\]]+\])', summary or '')
    return match.group(1) if match else None


FILE_NAME = re.compile(r'[^\s“”"]+\.(?:wav|mp3|m4a|flac|ogg|png|jpe?g|webp|mp4|mov|webm)\b', re.IGNORECASE)


def restore_verbatim(prompt, original):
    """Put back what must never change: the app's picture/audio lines and curly-quoted names.

    Models often re-type “quotes” as "quotes"; this is restored in code rather than spent on retries.
    """
    source = parse(original)
    body = strip_fences(prompt)
    draft = parse(body)
    if draft['preamble']:
        match = re.search(r'^\s*subject_definitions\s*:', body, re.MULTILINE)
        body = body[match.start():].strip() if match else body
    if source['preamble']:
        body = source['preamble'] + '\n' + body
    for quoted in set(re.findall(r'“([^”\n]+)”', original)):
        body = body.replace(f'"{quoted}"', f'“{quoted}”')
    return body


def check(prompt, original=None):
    """Validate a prompt; ``original`` is the prompt being fixed (labels and names to keep).

    Returns errors, warnings, stats and ``prompt``: the text to deliver, with verbatim parts restored.
    """
    prompt = restore_verbatim(prompt, original) if original else strip_fences(prompt)
    errors, warnings = [], []
    parsed = parse(prompt)
    source = parse(original) if original else None
    sections = parsed['sections']
    prompt_mode = mode(original or prompt)

    missing = [name for name in SECTIONS if name not in sections]
    if missing:
        errors.append('Missing sections: ' + ', '.join(f'{name}:' for name in missing) + '. Use all six, each header on its own line.')
    elif parsed['order'] != list(SECTIONS):
        errors.append('Sections must be in this order: ' + ', '.join(SECTIONS) + '.')
    for name in SECTIONS:
        if name in sections and not sections[name] and name != 'non_diegetic_music':
            errors.append(f'{name}: is empty.')

    if parsed['preamble'] and (not source or not source['preamble']):
        errors.append('Write nothing before subject_definitions: (the app adds its picture lines there).')
    if source:
        before, after = subject_names(original), subject_names(prompt)
        for number, name in sorted(before.items()):
            if number in after and after[number] != name:
                errors.append(f'<Subject {number}> was "{name}" in the original; keep record names exactly (it is now "{after[number]}").')
        missing_files = sorted({name for name in FILE_NAME.findall(original)} - {name for name in FILE_NAME.findall(prompt)})
        if missing_files:
            errors.append('File names from the original are missing or changed: ' + ', '.join(missing_files) + '.')
        # The app quotes reference names (audio windows, files) in subject_definitions; they bind to existing files.
        names = set(re.findall(r'“([^”\n]+)”', source['sections'].get('subject_definitions', '')))
        missing_names = sorted(name for name in names if f'“{name}”' not in prompt)
        if missing_names:
            errors.append('Quoted reference names from subject_definitions are missing or changed: ' + ', '.join(f'“{name}”' for name in missing_names) + '. Keep them exactly.')

    bad = sorted({match.group(0) for match in BAD_LABEL.finditer(prompt) if not LABEL.fullmatch(match.group(0))})
    if bad:
        errors.append('Malformed reference labels: ' + ', '.join(bad) + '. Use exactly <Picture N>, <Audio N>, <Video N>, <Subject N>.')

    used = labels(prompt)
    if source:
        kept = labels(original)
        for kind in ('Picture', 'Audio', 'Video'):
            added = sorted(used[kind] - kept[kind])
            dropped = sorted(kept[kind] - used[kind])
            if added:
                errors.append(f"{', '.join(f'<{kind} {n}>' for n in added)} not in the original prompt; never add or renumber {kind.lower()} references.")
            if dropped:
                errors.append(f"{', '.join(f'<{kind} {n}>' for n in dropped)} from the original must still be defined and used.")
        added_subjects = sorted(used['Subject'] - kept['Subject'])
        if added_subjects:
            warnings.append(f"New subject labels {', '.join(f'<Subject {n}>' for n in added_subjects)}: keep the original subjects unless the user asked for more.")
        defined = sections.get('subject_definitions', '')
        for kind in ('Picture', 'Audio', 'Video'):
            for number in sorted(kept[kind]):
                if f'<{kind} {number}>' not in defined and kind != 'Picture':
                    warnings.append(f'<{kind} {number}> is not named in subject_definitions; name each reference where it is used.')
    elif prompt_mode == 'first_last_frame' and used['Picture']:
        errors.append('First/last frame prompts must not name pictures.')
    for kind in ('Picture', 'Audio', 'Video', 'Subject'):
        numbers = sorted(used[kind])
        if numbers and numbers != list(range(1, numbers[-1] + 1)) and not source:
            warnings.append(f'{kind} labels skip numbers ({numbers}); check them against the app’s inputs.')
    if used['Video'] - {1} or used['Audio'] - {1, 2, 3}:
        errors.append('Only <Video 1> and up to <Audio 3> can be sent.')

    if re.search(r'\bcuts?\b', prose('\n'.join(sections.values())), re.IGNORECASE):
        errors.append('The word "cut" appears; H3 reads it as an edit. Rephrase (e.g. "the shot").')
    if source:
        dropped_subjects = sorted(labels(original)['Subject'] - used['Subject'])
        if dropped_subjects:
            warnings.append(f"{', '.join(f'<Subject {n}>' for n in dropped_subjects)} from the original no longer appear.")
    if re.search(r'^\s*---\s*$', prompt, re.MULTILINE):
        errors.append('Remove "---" lines: one prompt, one take.')
    if len(prompt) > MAX_CHARS:
        errors.append(f'Prompt is {len(prompt)} characters; keep it under {MAX_CHARS}.')

    summary = sections.get('summary', '')
    expected_prefix = summary_prefix(source['sections'].get('summary')) if source else None
    if expected_prefix is None and prompt_mode == 'omni_reference':
        expected_prefix = OMNI_PREFIX
    if expected_prefix and summary_prefix(summary) != expected_prefix:
        errors.append(f'summary: must start with {expected_prefix}.')
    if prompt_mode == 'fix' and not (summary_prefix(summary) or '').startswith('[video editing'):
        errors.append('summary: a character-fix prompt starts with [video editing + reference generation (+ audio reuse)].')

    description = sections.get('detailed_description', '')
    shots = re.findall(r'\[Shot (\d+)\]', prompt)
    if 'detailed_description' in sections:
        if description.count('[Shot 1]') != 1:
            errors.append('detailed_description: needs exactly one [Shot 1].')
        if any(shot != '1' for shot in shots):
            errors.append('Only [Shot 1] is allowed: one continuous shot.')
        if not description.lstrip().startswith('The target video is'):
            warnings.append('detailed_description: open with the style sentence "The target video is …".')
    timeline = beats(description)
    duration = timeline[-1][1] if timeline else None
    if prompt_mode != 'fix' and 'detailed_description' in sections:
        if not timeline:
            errors.append('detailed_description: add timed beats like "00:00.000-00:02.000: …".')
        else:
            if abs(timeline[0][0]) > 1e-6:
                errors.append('The first beat must start at 00:00.000.')
            for (start, end), (next_start, _) in zip(timeline, timeline[1:]):
                if abs(next_start - end) > 1e-6:
                    errors.append(f'Beats must be contiguous: a beat ends at {end:.3f}s but the next starts at {next_start:.3f}s.')
                    break
            if any(end <= start for start, end in timeline):
                errors.append('Each beat must end after it starts.')
            if duration > MAX_SECONDS:
                errors.append(f'The shot is {duration:.3f}s; H3 takes at most {MAX_SECONDS:.0f}s.')
            elif duration < 5:
                warnings.append(f'The shot is {duration:.3f}s; H3 is trained for about 5–15s.')
            source_beats = beats(source['sections'].get('detailed_description', '')) if source else []
            if source_beats and abs(source_beats[-1][1] - duration) > 1e-6:
                errors.append(f'The shot length is set by the app: the last beat must end at {fmt(source_beats[-1][1])} as in the original (now {fmt(duration)}). '
                              'Keep the source action at its real times and hold the end pose for the rest.')
    moves = {re.sub(r'(es|s|ing)\b', '', match.lower().split()[0]) for match in CAMERA.findall(description)}
    if prompt_mode != 'fix' and description and not moves:
        warnings.append('No camera move found; state it as type + amplitude + speed, e.g. "The camera holds a static shot."')
    elif len(moves - {'static'}) > 1:
        warnings.append(f'Several camera moves ({", ".join(sorted(moves))}); H3 handles one move per shot best.')
    words = len(description.split())
    if prompt_mode != 'fix' and description:
        if words < 150:
            errors.append(f'detailed_description has {words} words; H3 needs 200–500 concrete words (composition, positions, visible features, light, motion, camera, sound). Add evidence-based detail, not filler.')
        elif not 200 <= words <= 500:
            warnings.append(f'detailed_description has {words} words; aim for 200–500.')

    # Advisory only: the director may put a global timed note (e.g. a voice-over) after the beats.
    stray = misplaced_times(description)
    if stray and not (source and misplaced_times(source['sections'].get('detailed_description', ''))):
        warnings.extend(f'"At {at}" sits in the {span} beat; consider moving it into the beat that contains its time.' for at, span in stray)
    if source:
        body = '\n'.join(value for key, value in sections.items() if key != 'subject_definitions')
        for number in sorted(labels(original)['Audio']):
            if f'<Audio {number}>' not in body:
                warnings.append(f'<Audio {number}> is defined but never used: say what it drives, e.g. in overall_soundscape "<Subject 1> speaks the words of <Audio {number}> in its voice, lips in sync, from 00:00.070", or "An off-screen voice speaks the words of <Audio {number}>".')

    if '<d>' in prompt or '</d>' in prompt:
        if prompt.count('<d>') != prompt.count('</d>'):
            errors.append('Unbalanced <d>…</d> dialogue tags.')
        elif re.search(r'<d>(?!\[[A-Za-z][A-Za-z -]*\] )', prompt):
            errors.append('Dialogue must look like <d>[Language] words</d>.')
    soundscape = sections.get('overall_soundscape', '')
    if '<d>' in soundscape:
        errors.append('overall_soundscape: ambience and action sounds only; put speech in detailed_description.')
    music = sections.get('non_diegetic_music', '').strip()
    if 'non_diegetic_music' in sections and not re.fullmatch(r'N/A\.?', music):
        errors.append('non_diegetic_music: must be N/A (the score belongs to the edit).')

    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'prompt': prompt,
            'stats': {'mode': prompt_mode, 'chars': len(prompt), 'description_words': words, 'beats': len(timeline), 'seconds': duration,
                      'pictures': sorted(used['Picture']), 'audio': sorted(used['Audio']), 'subjects': sorted(used['Subject'])}}


def subject_names(text):
    """Map <Subject N> to the record name its definition gives ("<Subject 1> is Anna, the woman …")."""
    names = {}
    for number, name in re.findall(r'<Subject (\d+)> is (?:the )?([^,:\n<]+?)(?:,| from | whose |:)', parse(text)['sections'].get('subject_definitions', '')):
        names[int(number)] = name.strip()
    return names


def speaking_subjects(text):
    """Subjects given speech: '<Subject 1> (S1) … says: <d>' or '<Subject 1> speaks the words of <Audio N>'."""
    found = set()
    for number in re.findall(r'<Subject (\d+)>(?: \(S\d+\))?[^.\n<]{0,80}?\bsays:\s*<d>', text):
        found.add(int(number))
    for number in re.findall(r'<Subject (\d+)> speaks the words of <Audio \d+>', text):
        found.add(int(number))
    return found
