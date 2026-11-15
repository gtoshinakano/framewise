"""Dialogue compilation of a sequence: transcript, burned-in subtitles and visual evidence reconciled.

Pure functions, no database. The app does the arithmetic (pairing a transcript line with the subtitle
lines that overlap it, hints where Japanese wording was cut in half, retiming words after a correction,
snapping times to real edges, validation); the text model only chooses text and speaker from the
table the app prepares. How a line is played (on screen, off screen, inner voice, narrative) is not
decided here but per shot (``sequence_prompt``). All times are seconds on the sequence clock.

A compiled line (``C1…``)::

    {id, t0, t1, text, original_text, subtitle_text, speaker_label, character, character_context_id,
     continues_next, confidence, note, source_ids, subtitle_ids, parts: [{t0, t1, text}]}

``parts`` are the words (or short character groups) with their own times, so that a shot boundary can
still split a line by word, exactly as it does for the raw transcript.
"""
import difflib
import hashlib
import json
import re

from .sequence_timeline import EPSILON, PARTICLES, SENTENCE_END, join_text

CONFIDENCE = ('high', 'medium', 'low')
NARRATOR = 'Narrator'
MAX_TEXT = 600
SNAP_SECONDS = 0.4
MERGE_GAP = 0.6
SAME_SPEAKER_OVERLAP = 0.3

_STRIP = re.compile(r'[\s　、。，,.!?！？…・「」『』（）()\[\]―ー〜~"\'“”]+')
_CJK = re.compile(r'[぀-ヿ㐀-鿿ｦ-ﾟ]')


def normalize(text):
    return _STRIP.sub('', str(text or '')).casefold()


def similarity(left, right):
    left, right = normalize(left), normalize(right)
    if not left or not right:
        return 0.0
    return round(difflib.SequenceMatcher(None, left, right, autojunk=False).ratio(), 3)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


# ------------------------------------------------------------------- pairing

def pair_subtitles(lines, subtitles):
    """For each transcript line the subtitle lines that overlap it in time, with the overlap and the text similarity.

    ``lines`` have ``id``, ``t0``, ``t1``, ``text``; ``subtitles`` have ``id``, ``t0``, ``t1``, ``text``.
    A subtitle is shown a little before and after the speech, so a margin of half a second counts.
    """
    pairs = {}
    for line in lines:
        found = []
        for sub in subtitles:
            seconds = overlap(line['t0'] - 0.5, line['t1'] + 0.5, sub['t0'], sub['t1'])
            if seconds <= EPSILON:
                continue
            found.append({'subtitle': sub['id'], 'overlap': round(overlap(line['t0'], line['t1'], sub['t0'], sub['t1']), 2), 'similarity': similarity(line['text'], sub['text'])})
        pairs[line['id']] = found
    return pairs


def continuity_hints(lines, subtitles, cuts=()):
    """Places where one utterance may have been cut into two lines: ``[{a, b, reasons}]`` for consecutive lines."""
    hints = []
    for left, right in zip(lines, lines[1:]):
        reasons = []
        gap = right['t0'] - left['t1']
        same = not (left.get('speaker_label') and right.get('speaker_label')) or left['speaker_label'] == right['speaker_label']
        if same and gap < MERGE_GAP:
            reasons.append('short gap, same or unknown voice')
        if not SENTENCE_END.search(str(left['text']).strip()) and gap < 1.5:
            reasons.append('the first line ends without a sentence ending')
        if str(right['text']).strip().startswith(PARTICLES) and gap < 1.5:
            reasons.append('the second line starts like the middle of a sentence')
        if any(sub['t0'] < right['t0'] and sub['t1'] > left['t1'] and overlap(left['t0'], left['t1'], sub['t0'], sub['t1']) > 0.15
               and overlap(right['t0'], right['t1'], sub['t0'], sub['t1']) > 0.15 for sub in subtitles):
            reasons.append('one subtitle spans both lines')
        if any(left['t1'] - 0.8 <= cut['t0'] <= right['t0'] + 0.8 and cut['t0'] > EPSILON for cut in cuts):
            reasons.append('a cut falls between them')
        if reasons and (same or 'one subtitle spans both lines' in reasons):
            hints.append({'a': left['id'], 'b': right['id'], 'reasons': reasons})
    return hints


# --------------------------------------------------------------------- parts

def _units(text):
    """Words of Latin text and groups of up to three characters of Japanese, in order."""
    units = []
    for token in re.findall(r'[A-Za-z0-9\']+|[^\sA-Za-z0-9\']+', str(text)):
        if _CJK.search(token):
            units.extend(token[index:index + 3] for index in range(0, len(token), 3))
        else:
            units.append(token)
    return units


def retime_parts(text, t0, t1):
    """Spread the text over ``[t0, t1]`` in proportion to its characters, giving word-sized parts."""
    units = _units(text)
    if not units or t1 <= t0:
        return [{'t0': round(t0, 3), 't1': round(max(t1, t0), 3), 'text': str(text).strip()}] if str(text).strip() else []
    total = sum(len(unit) for unit in units)
    parts, cursor = [], t0
    for index, unit in enumerate(units):
        end = t1 if index == len(units) - 1 else cursor + (t1 - t0) * len(unit) / total
        parts.append({'t0': round(cursor, 3), 't1': round(end, 3), 'text': unit})
        cursor = end
    return parts


def parts_text(parts):
    text = ''
    for part in parts:
        text = join_text(text, part['text'])
    return text


def fit_parts(parts, t0, t1):
    """Move and scale the words of a line linearly onto ``[t0, t1]``."""
    if not parts:
        return []
    old0, old1 = parts[0]['t0'], max(part['t1'] for part in parts)
    if abs(old0 - t0) < 0.001 and abs(old1 - t1) < 0.001:
        return [dict(part) for part in parts]
    scale = (t1 - t0) / (old1 - old0) if old1 - old0 > 0.001 else 0
    out = []
    for part in parts:
        if scale:
            a, b = t0 + (part['t0'] - old0) * scale, t0 + (part['t1'] - old0) * scale
        else:
            a, b = t0, t1
        out.append({'t0': round(a, 3), 't1': round(b, 3), 'text': part['text']})
    return out


# ------------------------------------------------------------------- building

def line_from_source(line, index):
    """A compiled line that is exactly one transcript line (nothing corrected)."""
    parts = [{'t0': part['t0'], 't1': part['t1'], 'text': part['text']} for part in line.get('parts') or []]
    return {'id': f'C{index}', 't0': line['t0'], 't1': line['t1'], 'text': line['text'], 'original_text': line['text'], 'subtitle_text': None,
            'speaker_label': line.get('speaker_label') or '', 'character': line.get('character_name'), 'character_context_id': line.get('character_context_id'),
            'continues_next': False, 'confidence': 'high' if line.get('character_context_id') else 'medium', 'note': '',
            'source_ids': [part['id'] for part in line.get('parts') or [] if part.get('id')], 'subtitle_ids': [], 'parts': parts or retime_parts(line['text'], line['t0'], line['t1'])}


def source_lines(timeline):
    """The raw transcript as a compilation: the fallback, and what "use the original transcript" saves."""
    lines = [line_from_source(line, number) for number, line in enumerate(timeline['lines'], 1)]
    return {'lines': lines, 'voices': voices_of_lines(lines, []), 'open_questions': []}


def voices_of_lines(lines, given):
    """One voice row per label: the given decision when there is one, else the character most of its lines name."""
    by_label = {voice['speaker_label']: voice for voice in given if voice.get('speaker_label')}
    voices = []
    for label in dict.fromkeys(line['speaker_label'] for line in lines if line['speaker_label']):
        if label in by_label:
            voices.append(by_label[label])
            continue
        names = {}
        for line in lines:
            if line['speaker_label'] == label and line['character']:
                names[(line['character'], line['character_context_id'])] = names.get((line['character'], line['character_context_id']), 0) + 1
        best = max(names, key=names.get) if names else None
        voices.append({'speaker_label': label, 'character': best[0] if best else None, 'character_context_id': best[1] if best else None,
                       'kind': 'narrator' if best and best[0] == NARRATOR else 'character' if best else 'unknown', 'confidence': 'high' if best else 'low',
                       'reason': 'linked on the transcript' if best else ''})
    return voices


def snap(lines, anchors, duration):
    """Move line edges onto the nearest real edge (a transcript word or a subtitle) within ``SNAP_SECONDS``.

    ``anchors`` are times: word edges of the source transcript and the edges of the paired subtitle lines.
    Lines of one voice may not overlap by more than ``SAME_SPEAKER_OVERLAP`` seconds.
    """
    def nearest(value):
        close = [anchor for anchor in anchors if abs(anchor - value) <= SNAP_SECONDS]
        return min(close, key=lambda anchor: abs(anchor - value)) if close else value

    for line in lines:
        t0, t1 = nearest(line['t0']), nearest(line['t1'])
        t0, t1 = max(0.0, min(t0, duration)), max(0.0, min(t1, duration))
        if t1 - t0 < 0.2:
            t0, t1 = max(0.0, min(line['t0'], duration)), max(0.0, min(line['t1'], duration))
        line['t0'], line['t1'] = round(t0, 3), round(t1, 3)
    ordered = sorted(lines, key=lambda line: (line['t0'], line['t1']))
    last = {}
    for line in ordered:
        label = line['speaker_label']
        before = last.get(label)
        if label and before and before['t1'] - line['t0'] > SAME_SPEAKER_OVERLAP:
            middle = round((before['t1'] + line['t0']) / 2, 3)
            if before['t0'] + 0.2 < middle < line['t1'] - 0.2:
                before['t1'], line['t0'] = middle, middle
        last[label] = line
    return ordered


def merge_lines(group):
    """One line from consecutive compiled lines (used when the model joins a split utterance)."""
    first, last = group[0], group[-1]
    parts = [dict(part) for line in group for part in line['parts']]
    text = ''
    for line in group:
        text = join_text(text, line['text'])
    return {**first, 't0': first['t0'], 't1': max(line['t1'] for line in group), 'text': text, 'original_text': ' '.join(line['original_text'] for line in group),
            'parts': parts, 'source_ids': [source for line in group for source in line['source_ids']], 'subtitle_ids': list(dict.fromkeys(sub for line in group for sub in line['subtitle_ids'])),
            'continues_next': last['continues_next']}


# ----------------------------------------------------------------- validation

def _text(value):
    return re.sub(r'[ \t]+', ' ', value).strip() if isinstance(value, str) else ''


def validate(content, duration, roster):
    """Check and normalise a compilation (a model answer, a hand edit or an assistant edit).

    ``roster`` maps a character name to its context id. Returns the content in time order with ids
    ``C1…``; the words of a line are fitted to its times and rebuilt when its text no longer matches them.
    Raises ``ValueError`` with every problem found.
    """
    if not isinstance(content, dict) or not isinstance(content.get('lines'), list):
        raise ValueError('The compilation needs a "lines" list')
    errors, lines = [], []
    by_name = {name.casefold(): (name, context_id) for name, context_id in roster.items()}
    for position, raw in enumerate(content['lines'], 1):
        label = f'line {position}'
        if not isinstance(raw, dict):
            errors.append(f'{label} is not an object')
            continue
        try:
            t0, t1 = float(raw.get('t0')), float(raw.get('t1'))
        except (TypeError, ValueError):
            errors.append(f'{label} needs numeric t0 and t1')
            continue
        text = _text(raw.get('text'))
        if not text:
            errors.append(f'{label} has no text; delete the line instead')
            continue
        if len(text) > MAX_TEXT:
            errors.append(f'{label} is longer than {MAX_TEXT} characters; split it')
            continue
        if not (-EPSILON <= t0 < t1 <= duration + EPSILON):
            errors.append(f'{label} must satisfy 0 <= t0 < t1 <= {duration:g} (it is {t0:g}-{t1:g})')
            continue
        name = _text(raw.get('character')) or None
        context_id = None
        if name:
            if name.casefold() == NARRATOR.casefold():
                name = NARRATOR
            elif name.casefold() in by_name:
                name, context_id = by_name[name.casefold()]
            else:
                errors.append(f'{label}: "{name}" is not a character of this project')
                continue
        confidence = raw.get('confidence') if raw.get('confidence') in CONFIDENCE else 'medium'
        t0, t1 = max(0.0, round(t0, 3)), min(duration, round(t1, 3))
        parts = [{'t0': float(part['t0']), 't1': float(part['t1']), 'text': str(part['text'])} for part in raw.get('parts') or []
                 if isinstance(part, dict) and isinstance(part.get('text'), str) and isinstance(part.get('t0'), (int, float)) and isinstance(part.get('t1'), (int, float))]
        parts = fit_parts(parts, t0, t1) if parts and normalize(parts_text(parts)) == normalize(text) else retime_parts(text, t0, t1)
        lines.append({'id': '', 't0': t0, 't1': t1, 'text': text, 'original_text': _text(raw.get('original_text')) or text, 'subtitle_text': _text(raw.get('subtitle_text')) or None,
                      'speaker_label': _text(raw.get('speaker_label')), 'character': name, 'character_context_id': context_id,
                      'continues_next': bool(raw.get('continues_next')), 'confidence': confidence, 'note': _text(raw.get('note'))[:300],
                      'source_ids': [str(item) for item in raw.get('source_ids') or [] if isinstance(item, str)][:200],
                      'subtitle_ids': [str(item) for item in raw.get('subtitle_ids') or [] if isinstance(item, str)][:50], 'parts': parts})
    if errors:
        raise ValueError('; '.join(errors[:12]) + (f' (and {len(errors) - 12} more)' if len(errors) > 12 else ''))
    lines.sort(key=lambda line: (line['t0'], line['t1']))
    for number, line in enumerate(lines, 1):
        line['id'] = f'C{number}'
    voices = []
    for voice in content.get('voices') if isinstance(content.get('voices'), list) else []:
        if not isinstance(voice, dict) or not _text(voice.get('speaker_label')):
            continue
        name = _text(voice.get('character')) or None
        kind = voice.get('kind') if voice.get('kind') in ('character', 'narrator', 'unknown') else 'unknown'
        if name and name.casefold() in by_name:
            name = by_name[name.casefold()][0]
        elif name and name.casefold() == NARRATOR.casefold():
            name, kind = NARRATOR, 'narrator'
        else:
            name = None
        voices.append({'speaker_label': _text(voice.get('speaker_label')), 'character': name, 'character_context_id': by_name[name.casefold()][1] if name and name.casefold() in by_name else None,
                       'kind': kind if name else 'unknown', 'confidence': voice.get('confidence') if voice.get('confidence') in CONFIDENCE else 'medium', 'reason': _text(voice.get('reason'))[:300]})
    # A voice row exists for every label the lines use, so the bible can copy them.
    known = {voice['speaker_label'] for voice in voices}
    voices.extend(voice for voice in voices_of_lines(lines, []) if voice['speaker_label'] not in known)
    return {'lines': lines, 'voices': voices, 'open_questions': [_text(item) for item in content.get('open_questions') or [] if _text(item)][:12]}


# ------------------------------------------------------------------ extension

def tail_lines(lines, from_t):
    """The timeline lines from ``from_t`` on, for compiling only a cut added at the end of a sequence.

    A word belongs to the side its midpoint lies on, exactly as ``sequence_timeline.speech_lines`` clipped the
    range before it grew, so no word is compiled twice. Returns ``(tail, crossed)``: ``crossed`` are the ids of
    the words before ``from_t`` of lines that run on past it (the phrase the old end cut in half).
    """
    tail, crossed = [], set()
    for line in lines:
        after = [part for part in line['parts'] if (part['t0'] + part['t1']) / 2 >= from_t + EPSILON]
        if not after:
            continue
        before = [part for part in line['parts'] if (part['t0'] + part['t1']) / 2 < from_t + EPSILON]
        crossed.update(part['id'] for part in before if part.get('id'))
        tail.append({**line, 't0': after[0]['t0'], 't1': max(part['t1'] for part in after), 'text': parts_text(after), 'parts': after})
    return tail, crossed


def append_lines(existing, added, voices, questions, crossed):
    """An existing compilation with the lines of an added cut after it; ready for ``validate``.

    The existing lines and voice decisions stay exactly as the user left them. The line whose phrase the old
    end cut in half (it holds a word of ``crossed``) is marked ``continues_next``, so the planner keeps the
    phrase whole if it can and the user sees it among the lines to review. Only voice labels that are new are added.
    """
    lines = [dict(line) for line in existing.get('lines') or []]
    holder = max((line for line in lines if crossed & set(line.get('source_ids') or [])), key=lambda line: (line['t0'], line['t1']), default=None)
    if holder is not None:
        holder['continues_next'] = True
    known = {voice.get('speaker_label') for voice in existing.get('voices') or []}
    return {'lines': lines + [dict(line) for line in added],
            'voices': [*(existing.get('voices') or []), *(voice for voice in voices if voice['speaker_label'] not in known)],
            'open_questions': [*questions, *(existing.get('open_questions') or [])][:12]}  # the added cut's questions first: they are the ones to answer now


# ------------------------------------------------------------------ consumers

def merged_lines(content, origin):
    """The compilation in the shape ``sequence_timeline.speech_lines`` reads (episode times)."""
    lines = []
    for line in content.get('lines') or []:
        lines.append({'start_time': origin + line['t0'], 'end_time': origin + line['t1'], 'text': line['text'], 'speaker_label': line.get('speaker_label') or '',
                      'character_context_id': line.get('character_context_id'), 'character_name': line.get('character'),
                      'continues_next': bool(line.get('continues_next')),
                      'compiled': content.get('origin') != 'source',
                      'parts': [{'start_time': origin + part['t0'], 'end_time': origin + part['t1'], 'text': part['text']} for part in line.get('parts') or []]})
    return lines


def source_hash(transcript, subtitles):
    """Fingerprint of the transcript fragments and subtitle lines a compilation was built from."""
    body = json.dumps([[row['id'], row['start_time'], row['end_time'], row['text'], row.get('speaker_label'), row.get('character_context_id')] for row in transcript]
                      + [[row['id'], row['start_time'], row['end_time'], row['text']] for row in subtitles], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:20]


def needs_review(line):
    """True when a person should look at the line: unsure, unspoken-for, or continuing in the next line."""
    return line.get('confidence') == 'low' or not line.get('character') or bool(line.get('continues_next'))


# ------------------------------------------------------------------- assembly

def deterministic_groups(window_lines):
    """The answer the app gives when the model could not: every line alone, its text and its linked speaker unchanged."""
    return {'lines': [{'ids': [line['id']], 'text': line['text'], 'speaker': line.get('character_name'),
                       'subtitle_ids': [], 'confidence': 'high' if line.get('character_name') else 'low', 'note': '', 'continues_next': False} for line in window_lines],
            'voices': [], 'open_questions': []}


def assemble(answer, window_lines, subtitles, roster):
    """Compiled lines from a validated answer: times and words come from the source lines, the model's text replaces them.

    ``window_lines`` are timeline lines of the window; ``subtitles`` is ``{'U1': {...}}`` with the real ``id`` under ``real``.
    A character the user linked on the transcript always wins over the model's speaker. ``roster`` maps names to ids.
    """
    by_id = {line['id']: line for line in window_lines}
    out = []
    for group in answer['lines']:
        sources = [line_from_source(by_id[line_id], 0) for line_id in group['ids']]
        base = merge_lines(sources)
        text = _text(group['text']) or base['text']
        linked = next((line for line in sources if line['character_context_id']), None)
        name = linked['character'] if linked else group.get('speaker')
        context_id = linked['character_context_id'] if linked else roster.get(name)
        confidence = group.get('confidence') or 'medium'
        used = [subtitles[item] for item in group.get('subtitle_ids') or [] if item in subtitles]
        if normalize(text) == normalize(base['text']):
            parts = base['parts']
            text = base['text']
        else:
            parts = retime_parts(text, base['t0'], base['t1'])
        out.append({**base, 'text': text, 'parts': parts, 'character': name, 'character_context_id': context_id,
                    'continues_next': bool(group.get('continues_next')), 'confidence': confidence, 'note': group.get('note') or '',
                    'subtitle_ids': [item['real'] for item in used], 'subtitle_text': ' '.join(item['text'] for item in used) or None})
    return out


def settle_voices(lines, given, raw_lines):
    """Voice rows for the compilation: the model's decisions, overridden by what the user linked on the transcript.

    A label whose raw lines were all linked to one character belongs to that character whatever the model said.
    """
    linked = {}
    for line in raw_lines:
        if line.get('speaker_label') and line.get('character_context_id'):
            linked.setdefault(line['speaker_label'], {}).setdefault((line['character_name'], line['character_context_id']), 0)
            linked[line['speaker_label']][(line['character_name'], line['character_context_id'])] += 1
    voices = []
    for voice in given:
        label = voice['speaker_label']
        if len(linked.get(label, {})) == 1:
            (name, context_id), = linked[label]
            voice = {**voice, 'character': name, 'character_context_id': context_id, 'kind': 'character', 'confidence': 'high', 'reason': 'Linked to this character in the transcript.'}
        voices.append(voice)
    return voices
