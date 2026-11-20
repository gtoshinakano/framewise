"""Voice evidence as plain tables (stdlib only), so scoring and `rescore` run on the host without numpy.

The audio stage stores cosine similarities: line -> {reference range id: cos}, cluster -> {range id: cos} and cluster -> {sample id: cos}.
Everything that ranks, thresholds or leaves a reference out (leave-one-out) happens here on those tables.
"""
import math
from collections import defaultdict

import lines as L


def overlapping_ranges(refs_meta, episode, start, end):
    """range_ids of references whose audio overlaps [start, end] in that episode: left out when scoring that span."""
    return {r['range_id'] for r in refs_meta if r['episode'] == episode and L.overlap(start, end, r['start'], r['end']) > 0}


def scores(cos_by_range, refs_meta, exclude=frozenset(), topk=2):
    """Character -> mean of its `topk` best cosines over its reference ranges, without the excluded ranges."""
    owner = {r['range_id']: r['character'] for r in refs_meta}
    per = defaultdict(list)
    for range_id, value in (cos_by_range or {}).items():
        if range_id in exclude or range_id not in owner:
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
    return (order[0][0] if order[0][1] >= threshold and gap >= margin else None), order[0][1], gap


def match_clusters_app(cluster_sample_cos, samples_meta, threshold=.7, margin=.08):
    """The app's rule (audio_matching.match_voices): best cosine against the stored whole-sample embeddings per character."""
    owner = {s['id']: s['character'] for s in samples_meta}
    matches = {}
    for label, by_sample in cluster_sample_cos.items():
        best = defaultdict(lambda: -2.)
        for sample_id, value in by_sample.items():
            character = owner.get(sample_id)
            if character is not None:
                best[character] = max(best[character], value)
        order = sorted(((v, c) for c, v in best.items()), reverse=True)
        if order and order[0][0] >= threshold and (len(order) == 1 or order[0][0] - order[1][0] >= margin):
            matches[label] = order[0][1]
    return matches


def match_clusters_range(cluster_range_cos, refs_meta, exclude=frozenset(), threshold=.45, margin=.05):
    matches = {}
    for label, by_range in cluster_range_cos.items():
        character, _, _ = decide(scores(by_range, refs_meta, exclude), threshold, margin)
        if character:
            matches[label] = character
    return matches


def softmax(score_map, temperature=.07):
    if not score_map:
        return {}
    top = max(score_map.values())
    weights = {k: math.exp((v - top) / temperature) for k, v in score_map.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}
