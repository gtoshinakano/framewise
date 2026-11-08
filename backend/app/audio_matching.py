"""Reconcile aligned words, diarization turns, and project voice embeddings.

All times are in seconds on the original episode timeline. Model output can be
edited by a person; speaker assignments are suggestions.
"""
import math
import re
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict


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
        match = re.search(re.escape(token), text[cursor:], flags=re.IGNORECASE)
        if not match:
            return None
        start = cursor + match.start()
        end = cursor + match.end()
        if restored:
            # Closing punctuation belongs to the preceding timed word. Leave
            # whitespace and opening punctuation with the next word so the
            # original transcription can still be reconstructed exactly.
            while cursor < start:
                char = text[cursor]
                category = unicodedata.category(char)
                if not category.startswith('P') or category in ('Ps', 'Pi'):
                    break
                restored[-1]['text'] += char
                cursor += 1
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


def reconcile_speakers(turns, existing, matches, redo, min_overlap=.5):
    """Name the speakers heard in a re-run excerpt after the labels the episode already uses.

    Diarizing an excerpt gives local labels (SPEAKER_00 ...) that can mean other
    people than the same labels elsewhere in the episode. ``turns`` are the
    excerpt's speaker turns on the episode timeline, ``existing`` the episode's
    current segments (start_time, end_time, speaker_label, character_context_id),
    ``matches`` the voice-reference matches of the local speakers, and ``redo``
    the (start, end) being replaced. The old labels inside ``redo`` are the
    suspect ones, so they are not evidence.

    A local speaker takes the existing label it overlaps most in the surrounding
    speech (one-to-one), else the label already used for the character its voice
    matched, else a new label no segment uses yet. Its character is the voice
    match, else the character most of that label's segments are linked to.
    Returns {local speaker: {'label': str, 'character_id': str | None}}.
    """
    redo_start, redo_end = redo
    outside = [s for s in existing if s['speaker_label'] and not (s['start_time'] < redo_end and s['end_time'] > redo_start)]
    overlap = defaultdict(float)
    if turns:
        low, high = min(t['start_time'] for t in turns), max(t['end_time'] for t in turns)
        nearby = [s for s in outside if s['end_time'] > low and s['start_time'] < high]
        for turn in turns:
            for segment in nearby:
                shared = min(turn['end_time'], segment['end_time']) - max(turn['start_time'], segment['start_time'])
                if shared > 0:
                    overlap[(turn['speaker'], segment['speaker_label'])] += shared
    names, used = {}, set()
    for (speaker, label), shared in sorted(overlap.items(), key=lambda item: (-item[1], item[0])):
        if shared >= min_overlap and speaker not in names and label not in used:
            names[speaker] = label
            used.add(label)
    by_label, labels_of = defaultdict(Counter), defaultdict(Counter)
    for segment in outside:
        by_label[segment['speaker_label']][segment['character_context_id']] += 1
        if segment['character_context_id']:
            labels_of[segment['character_context_id']][segment['speaker_label']] += 1
    speakers = sorted({turn['speaker'] for turn in turns})
    for speaker in speakers:
        character = matches.get(speaker, {}).get('character_id')
        if speaker not in names and character:
            for label, _ in sorted(labels_of[character].items(), key=lambda item: (-item[1], item[0])):
                if label not in used:
                    names[speaker] = label
                    used.add(label)
                    break
    numbers = [int(m.group(1)) for s in existing if (m := re.fullmatch(r'SPEAKER_(\d+)', s['speaker_label'] or ''))]
    fresh = max(numbers, default=-1) + 1
    for speaker in speakers:
        if speaker not in names:
            names[speaker] = f'SPEAKER_{fresh:02d}'
            fresh += 1
    result = {}
    for speaker in speakers:
        label = names[speaker]
        character = matches.get(speaker, {}).get('character_id')
        if not character:
            counts = by_label.get(label)
            top = counts.most_common(1)[0] if counts else None
            # Inherit a link only when most of the label's lines carry it.
            if top and top[0] and top[1] * 2 > sum(counts.values()):
                character = top[0]
        result[speaker] = {'label': label, 'character_id': character}
    return result
