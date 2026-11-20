"""Word timing for line rows and compatibility with the original fragment rows."""
import difflib
import json


def join_text(left, right):
    if left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return f'{left} {right}'
    return left + right


def aligned_words(row):
    words = dict(row).get('words_json')
    return json.loads(words) if isinstance(words, str) else words


def timings(row):
    words = aligned_words(row)
    return [(start, end) for start, end, _ in words] if words else [(row['start_time'], row['end_time'])]


def parts(row):
    row = dict(row)
    words = aligned_words(row)
    if not words:
        return [row]
    # Model spelling fixes can change the line while keeping the original aligner times.
    original, owners = '', []
    for index, (_, _, word) in enumerate(words):
        combined = join_text(original, word.strip())
        owners.extend([index] * (len(combined) - len(original)))
        original = combined
    corrected = [''] * len(words)
    for op, a, b, c, d in difflib.SequenceMatcher(None, original, row['text'], autojunk=False).get_opcodes():
        if op == 'equal':
            for index, char in zip(owners[a:b], row['text'][c:d]):
                corrected[index] += char
        elif op in ('replace', 'insert') and owners:
            # If a name spans several words, keep the replacement on those same time slots.
            slots = list(dict.fromkeys(owners[a:b] or [owners[max(0, a - 1)]]))
            for index, char in enumerate(row['text'][c:d]):
                corrected[slots[index * len(slots) // (d - c)]] += char
    return [{**row, 'id': f"{row.get('id') or ''}:{index}", 'start_time': start, 'end_time': end,
             'text': corrected[index].strip(), 'words_json': None}
            for index, (start, end, _) in enumerate(words) if corrected[index].strip()] or [row]


def within(row, start, end):
    """Retain just the words belonging to a timed excerpt of an aligned line."""
    if not row.get('words_json') or start <= row['start_time'] and end >= row['end_time']:
        return row
    selected = [part for part in parts(row) if start <= (part['start_time'] + part['end_time']) / 2 < end]
    if not selected:
        return None
    text = ''
    for part in selected:
        text = join_text(text, part['text'])
    return {**row, 'start_time': selected[0]['start_time'], 'end_time': max(part['end_time'] for part in selected),
            'text': text, 'words_json': None}


def manual_metadata(old, new):
    old = dict(old)
    speaker_changed = any(old[key] != new[key] for key in ('speaker_label', 'character_context_id'))
    alignment_changed = any(old[key] != new[key] for key in ('text', 'start_time', 'end_time'))
    return (None if alignment_changed else old.get('words_json'),
            'manual' if speaker_changed else old.get('speaker_source'),
            None if speaker_changed else old.get('confidence'), old.get('evidence_json'))
