"""Word-level text differences, shown as inserted and deleted runs.

Used by the assistant's proposal cards and by the history of a sequence shot's prompt, so every
change made by a model or by hand is shown the same way.
"""
import difflib
import re


def diff_segments(old, new, limit=60000):
    """Word-level diff segments: runs of `equal`, `delete` and `insert` text."""
    if len(old) + len(new) > limit:
        lines = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=2)
        return [{'op': 'equal', 'text': '\n'.join(list(lines)[2:])}]
    # ASCII words, whitespace runs and single other characters: JSON quotes stay out of
    # word edits and Japanese text (no spaces) is compared character by character.
    split = lambda text: re.findall(r'[A-Za-z0-9_]+|\s+|[^\sA-Za-z0-9_]', text)
    a, b = split(old), split(new)
    opcodes = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    segments, deleted, inserted = [], '', ''

    def close():
        nonlocal deleted, inserted
        if deleted:
            segments.append({'op': 'delete', 'text': deleted})
        if inserted:
            segments.append({'op': 'insert', 'text': inserted})
        deleted = inserted = ''

    for index, (op, i1, i2, j1, j2) in enumerate(opcodes):
        if op == 'equal':
            text = ''.join(a[i1:i2])
            between = 0 < index < len(opcodes) - 1 and (deleted or inserted)
            if between and len(text) <= 2 and not text.strip():
                # Merge "a person" -> "Tomoe Koga" into one edit instead of word-by-word pairs.
                deleted, inserted = deleted + text, inserted + text
                continue
            close()
            segments.append({'op': 'equal', 'text': text})
        else:
            deleted += ''.join(a[i1:i2])
            inserted += ''.join(b[j1:j2])
    close()
    return segments
