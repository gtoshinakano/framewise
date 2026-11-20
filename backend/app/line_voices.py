"""Per-line voice attribution against individual reference excerpts."""
from collections import defaultdict

from . import audio_lines as L

FIX_TOP = .45
FIX_GAP = .06
PAD_REACH = 1.0
MIN_CLIP = .6
SOURCES = ('line', 'pad')

def scores(cos_by_range, refs_meta, topk=2):
    """Character -> mean of its `topk` best cosines over its reference ranges,."""
    owner = {r['range_id']: r['character'] for r in refs_meta}
    per = defaultdict(list)
    for range_id, value in (cos_by_range or {}).items():
        if range_id not in owner:
            continue
        per[owner[range_id]].append(value)
    result = {}
    for character, values in per.items():
        best = sorted(values, reverse=True)[:topk]
        result[character] = sum(best) / len(best)
    return result


def ranked(score_map):
    return sorted(score_map.items(), key=lambda kv: -kv[1])


def decide(score_map, threshold=.35, margin=.06):
    """(character or None, top score, gap to the runner-up)."""
    order = ranked(score_map)
    if not order:
        return None, 0., 0.
    gap = order[0][1] - order[1][1] if len(order) > 1 else order[0][1]
    return (order[0][0] if order[0][1] >= threshold and gap + 1e-9 >= margin else None), order[0][1], gap


def pad_span(ordered, index, reach=PAD_REACH):
    """The line's span grown into the silence before and after it, never into a neighbouring line."""
    line = ordered[index]
    left = line['start_time'] - reach
    right = line['end_time'] + reach
    if index > 0:
        left = max(left, ordered[index - 1]['end_time'])
    if index + 1 < len(ordered):
        right = min(right, ordered[index + 1]['start_time'])
    return round(min(left, line['start_time']), 3), round(max(right, line['end_time']), 3)


def cascade(scores_by_source, top_min=None, gap_min=None):
    """First source whose scores decide the line at the FIX thresholds.

    -> {speaker, source, top, gap, evidence: best top-3 among the sources tried, evidence_source}. speaker None when nothing decides.
    """
    best = None
    for source in SOURCES:
        sc = scores_by_source.get(source)
        if not sc:
            continue
        who, top, gap = decide(sc, FIX_TOP if top_min is None else top_min, FIX_GAP if gap_min is None else gap_min)
        ordered = ranked(sc)
        if best is None or ordered[0][1] > best[1][0][1]:
            best = (source, ordered)
        if who:
            return {'speaker': who, 'source': source, 'top': round(top, 3), 'gap': round(gap, 3),
                    'evidence': [[c, round(v, 3)] for c, v in ordered[:3]], 'evidence_source': source}
    if best is None:
        return {'speaker': None, 'source': None, 'evidence': [], 'evidence_source': None}
    source, ordered = best
    return {'speaker': None, 'source': None, 'top': round(ordered[0][1], 3), 'gap': round(ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.), 3),
            'evidence': [[c, round(v, 3)] for c, v in ordered[:3]], 'evidence_source': source}


def split_by_mask(lines, mask, share=.5):
    """-> (kept, dropped): a line is dropped when at least `share` of it lies inside the mask."""
    kept, dropped = [], []
    for line in lines:
        (dropped if mask and L.inside_any(line['start_time'], line['end_time'], mask, share) else kept).append(line)
    return kept, dropped




def range_embeddings(samples, ranges, embed):
    """One embedding per usable range, keeping the longest of all-short samples."""
    selected = [row for row in ranges if row['end_time'] - row['start_time'] >= 1.]
    if not selected and ranges:
        selected = [max(ranges, key=lambda row: row['end_time'] - row['start_time'])]
    result = []
    for row in selected:
        start, end = row['start_time'], row['end_time']
        vector = embed(samples[round(start * 16000):round(end * 16000)])
        if vector is not None:
            result.append({'start': start, 'end': end, 'embedding': vector.tolist()})
    return result


def score_clip(vector, refs):
    import numpy as np
    if vector is None:
        return {}
    norm = np.linalg.norm(vector)
    if not norm:
        return {}
    cosines = {row['range_id']: round(float(np.dot(vector / norm, row['unit'])), 4)
               for row in refs if len(row['unit']) == len(vector)}
    return scores(cosines, refs)
