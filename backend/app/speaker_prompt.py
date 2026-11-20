"""The JSON contract and conversation windows for speaker attribution."""
import json

CONFIDENCE = ['high', 'medium', 'low']
REASONS = ['voice', 'addressed', 'pronoun', 'on_screen', 'turn', 'subtitle', 'other']

SYSTEM_OPEN = """You attribute lines of dialogue from one episode of a Japanese anime to the character who speaks them.

You see a stretch of the conversation. Lines marked "decided" already have their speaker from a clear match of the line's own voice with a recorded sample of that character; they are settled and you do not answer them, but they tell you who is talking around the open lines. You answer only the open lines listed at the end.

An open line comes with: its id and times in seconds, the Japanese text from a speech recognizer, a diarization label (a voice cluster; a decided line with the same label nearby is good evidence of the same voice, but one character can have several labels and one label can hold several characters), the characters whose recorded voice is closest to this line (weak: it did not decide the line, often because the line is very short), the characters recognised in the first frame of the cut when known (off-screen speech and inner monologue are normal), and a loose Brazilian Portuguese translation of the burned-in subtitles ("pt").

Rules.
1. Decide who speaks each open line from turn taking with the decided lines around it, the diarization label, who is addressed, first-person pronouns and sentence endings, answers to questions, and a line that continues the previous line of the same speaker. Who is on screen is weak evidence.
2. A name or honorific used to address someone names the listener, not the speaker.
3. The Portuguese translation is loose: use it for where sentences begin and end, whether two people speak inside one subtitle (a leading dash marks a second speaker) and rough meaning only.
4. speaker is one of the listed characters, "other" for a person who is not listed (announcer, passer-by, crowd), or "unknown" when the evidence does not decide. Prefer unknown to a guess.
5. confidence: high when two independent cues agree, medium when one clear cue supports it, low when the evidence is weak. reason is the single main cue."""


def answer_schema(ids, names):
    speakers = list(names) + ['other', 'unknown']
    item = {'type': 'object', 'additionalProperties': False,
            'required': ['id', 'speaker', 'confidence', 'reason'],
            'properties': {'id': {'type': 'string', 'enum': list(ids)}, 'speaker': {'type': 'string', 'enum': speakers},
                           'confidence': {'type': 'string', 'enum': CONFIDENCE}, 'reason': {'type': 'string', 'enum': REASONS}}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['lines'],
            'properties': {'lines': {'type': 'array', 'items': item, 'minItems': len(ids), 'maxItems': len(ids)}}}


def make_windows(items, size=40, overlap=4):
    """Index ranges [(start, stop)] over `items`."""
    windows, start = [], 0
    stop = min(len(items), size)
    while start < len(items):
        windows.append((start, stop))
        if stop >= len(items):
            break
        start = stop - overlap
        stop = min(len(items), start + size)
    return windows


def validate(answer, window, names):
    items = answer.get('lines') if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return {}, ['the answer must be an object with a "lines" array']
    expected = [line['id'] for line in window]
    seen = [item.get('id') for item in items if isinstance(item, dict)]
    errors, valid = [], {}
    if seen != expected or len(seen) != len(items):
        missing = [line for line in expected if line not in seen]
        errors.append(f'answer every line exactly once and in order; missing {missing[:6]}')
    allowed = [*names, 'other', 'unknown']
    for item in items:
        if not isinstance(item, dict) or item.get('id') not in expected:
            continue
        lid = item['id']
        if item.get('speaker') not in allowed:
            errors.append(f'{lid}: speaker must be one of the listed characters, "other" or "unknown"')
        elif item.get('confidence') not in CONFIDENCE or item.get('reason') not in REASONS:
            errors.append(f'{lid}: use one of the listed confidence levels and reasons')
        else:
            valid[lid] = {key: item[key] for key in ('speaker', 'reason')}
    return valid, errors


def user_message(window, names, shown):
    return ('Characters of this episode: ' + ', '.join(names) + '.\n\n'
            + f'Conversation ({len(shown)} lines; a line with "decided" already has its speaker and is not answered):\n'
            + json.dumps(shown, ensure_ascii=False)
            + f'\n\nAnswer these {len(window)} lines, in this order: ' + json.dumps([line['id'] for line in window]))


def attribute_open(items, ask):
    windows = make_windows(items)
    chosen = {}
    for number, (start, stop) in enumerate(windows, 1):
        shown = items[start:stop]
        asked = [item for item in shown if 'decided' not in item]
        if not asked:
            continue
        answers = ask(asked, shown, number, len(windows))
        for offset, item in enumerate(shown):
            lid = item['id']
            if lid not in answers:
                continue
            distance = min(offset, len(shown) - 1 - offset)
            if lid not in chosen or distance >= chosen[lid][0]:
                chosen[lid] = (distance, answers[lid])
    return {lid: value[1] for lid, value in chosen.items()}
