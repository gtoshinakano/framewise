"""Confidence and review flags for one target frame, derived from tracking evidence."""
import math
from collections import Counter

from .merged_box import centre
from .target_tracking import UNMATCHED_COST

PENALTIES = {
    'low_match': .25,
    'count_changed': .15,
    'unmatched': .18,
    'suspected_merge': .28,
    'jump': .15,
    'duplicate_character': .35,
}


def review(placed, candidates, unmatched, slots, first_count, predicted, size, merged=(), *, no_match=UNMATCHED_COST):
    """Return a 0–1 score and named flags, with the affected colours and candidates.

    The score orders human review. It is not a calibrated probability or an
    authorization to apply a model edit.
    """
    flags = {}
    for entry in placed:
        slot = entry['slot']
        evidence = entry.get('scores')
        box = candidates[entry['detection']]['box']
        if entry['new'] or (evidence and evidence['cost'] >= no_match - .07):
            flags.setdefault('low_match', []).append(slot)
        previous = predicted.get(slot)
        if previous and math.dist(centre(previous), centre(box)) > .16 * size[0]:
            flags.setdefault('jump', []).append(slot)
    if len(placed) != first_count:
        flags['count_changed'] = {'first': first_count, 'here': len(placed)}
    if unmatched:
        flags['unmatched'] = list(unmatched)
    if merged:
        flags['suspected_merge'] = list(merged)
    visible = {entry['slot'] for entry in placed}
    characters = {slot.number: slot.character_id for slot in slots if slot.number in visible and slot.character_id}
    counts = Counter(characters.values())
    duplicates = sorted(number for number, character in characters.items() if counts[character] > 1)
    if duplicates:
        flags['duplicate_character'] = duplicates
    score = max(0., 1. - sum(PENALTIES[reason] for reason in flags))
    return {'score': round(score, 3), 'flags': flags}
