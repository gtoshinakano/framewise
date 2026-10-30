"""Reconcile aligned words, diarization turns, and project voice embeddings.

All times are in seconds on the original episode timeline. Model output can be
edited by a person; speaker assignments are suggestions.
"""
import math
from bisect import bisect_right
from collections import defaultdict


def overlap_index(intervals):
    """Build a compact interval index for efficient transcript reruns."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    starts = [start for start, _ in merged]

    def overlaps(start, end):
        index = bisect_right(starts, start) - 1
        return (index >= 0 and merged[index][1] > start) or (index + 1 < len(merged) and merged[index + 1][0] < end)

    return overlaps


def restore_aligned_text(text, words):
    """Retain punctuation and spacing that the aligner's tokenization removes.

    Return None if its tokens cannot be reconciled with the transcription;
    callers can then retain the original sentence as a single editable span.
    """
    cursor = 0
    restored = []
    for word in words:
        token = word['text']
        # Work in the original string so Unicode case-folding cannot shift offsets.
        import re
        match = re.search(re.escape(token), text[cursor:], flags=re.IGNORECASE)
        if not match:
            return None
        end = cursor + match.end()
        restored.append({**word, 'text': text[cursor:end]})
        cursor = end
    if restored:
        restored[-1]['text'] += text[cursor:]
    return restored


def match_voices(speakers, references, threshold=0.7, margin=0.08):
    """Return confident speaker -> character matches using cosine similarity.

    References may contain multiple embeddings per character. Invalid vectors
    are ignored and ambiguous matches are deliberately left unassigned.
    """
    def cosine(a, b):
        if not a or len(a) != len(b):
            return None
        if any(not math.isfinite(float(x)) for x in (*a, *b)):
            return None
        an = math.sqrt(sum(x * x for x in a))
        bn = math.sqrt(sum(x * x for x in b))
        if not an or not bn:
            return None
        return sum(x * y for x, y in zip(a, b)) / (an * bn)

    matches = {}
    for speaker, vector in speakers.items():
        scores = defaultdict(list)
        for ref in references:
            score = cosine(vector, ref['embedding'])
            if score is not None:
                scores[ref['character_id']].append(score)
        ranked = sorted(((max(values), character) for character, values in scores.items()), reverse=True)
        if ranked and ranked[0][0] >= threshold and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= margin):
            matches[speaker] = {'character_id': ranked[0][1], 'similarity': ranked[0][0]}
    return matches


def assign_words(words, turns):
    """Assign each aligned word to its greatest-overlap exclusive speaker turn.

    Words without a positive overlap stay unattributed. Invalid timings are
    rejected rather than silently producing misleading cut context.
    """
    result = []
    ordered = sorted(turns, key=lambda turn: turn['start_time'])
    ends = [turn['end_time'] for turn in ordered]
    ordered_ends = all(a <= b for a, b in zip(ends, ends[1:]))
    for word in words:
        start, end = word['start_time'], word['end_time']
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('Invalid aligned word timing')
        overlaps = defaultdict(float)
        # Exclusive diarization turns have monotonic ends. Binary search skips
        # earlier turns so long recordings do not require a full scan per word.
        first = bisect_right(ends, start) if ordered_ends else 0
        for index in range(first, len(ordered)):
            turn = ordered[index]
            if turn['start_time'] >= end:
                break
            overlap = min(end, turn['end_time']) - max(start, turn['start_time'])
            if overlap > 0:
                overlaps[turn['speaker']] += overlap
        speaker = max(overlaps, key=overlaps.get) if overlaps else None
        result.append({**word, 'speaker': speaker})
    return result
