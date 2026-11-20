"""Voice references, per-line scores and cluster resolution.

References are embeddings of each voice-sample RANGE (not only the whole sample), so a line can be scored with the range it
overlaps left out (leave-one-out): the reference audio never helps to label itself. Pure numpy; no models are loaded here.
"""
import math
from collections import Counter, defaultdict

import numpy as np

import lines as L
from voice_tables import decide, overlapping_ranges, ranked, scores, softmax  # noqa: F401  (pure twins live there)

MIN_EMBED_SECONDS = 1.0


def unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def cosine(a, b):
    return float(np.dot(unit(a), unit(b)))


def clip_samples(wav, start, end, rate=16000, minimum=.3):
    a, b = max(0, int(start * rate)), min(len(wav), int(end * rate))
    clip = wav[a:b]
    return clip if len(clip) >= minimum * rate else None


def build_references(samples, wavs, embed_span):
    """One reference per sample range of at least MIN_EMBED_SECONDS.

    ``samples`` as data.voice_samples, ``wavs`` {episode number: float32 array}, ``embed_span(episode, start, end, clip) -> vector``.
    A sample whose ranges are all shorter than that keeps one reference built from the longest range padded by its neighbours.
    """
    refs = []
    for sample in samples:
        wav = wavs.get(sample['episode'])
        if wav is None:
            continue
        added = 0
        for index, rng in enumerate(sample['ranges']):
            if rng['end'] - rng['start'] < MIN_EMBED_SECONDS:
                continue
            clip = clip_samples(wav, rng['start'], rng['end'])
            if clip is None:
                continue
            refs.append({'character': sample['character'], 'sample_id': sample['id'], 'range_id': f"{sample['id']}:{index}",
                         'episode': sample['episode'], 'start': rng['start'], 'end': rng['end'], 'vec': unit(embed_span(sample['episode'], rng['start'], rng['end'], clip))})
            added += 1
        if not added and sample['ranges']:
            longest = max(sample['ranges'], key=lambda r: r['end'] - r['start'])
            clip = clip_samples(wav, longest['start'], longest['end'])
            if clip is not None:
                refs.append({'character': sample['character'], 'sample_id': sample['id'], 'range_id': f"{sample['id']}:0",
                             'episode': sample['episode'], 'start': longest['start'], 'end': longest['end'], 'vec': unit(embed_span(sample['episode'], longest['start'], longest['end'], clip)), 'short': True})
    return refs


# ------------------------------------------------------------------ episode-level resolution (R1)

def resolve_clusters(lines, min_lines=5, min_share=.6, min_mass=.4):
    """Link each diarization cluster to a character from its own lines' evidence.

    Each line contributes its Qwen vote (weighted by confidence) and its voice-score softmax; the cluster takes the
    character with the most mass when it holds at least `min_share` of the cluster's mass over at least `min_lines`
    lines. Clusters that map to the same character are reported as merged.
    ``lines`` carry `diar`, `qwen` (name or None), `qwen_confidence`, `voice_scores`.
    """
    weight = {'high': 1., 'medium': .6, 'low': .25}
    mass, count = defaultdict(Counter), Counter()
    for line in lines:
        label = line.get('diar')
        if not label:
            continue
        count[label] += 1
        if line.get('qwen') and line['qwen'] not in ('unknown', 'other'):
            mass[label][line['qwen']] += weight.get(line.get('qwen_confidence'), .4)
        for character, share in softmax(line.get('voice_scores') or {}).items():
            mass[label][character] += min_mass * share
    result = {}
    for label, counter in mass.items():
        total = sum(counter.values())
        if not total or count[label] < min_lines:
            continue
        character, value = counter.most_common(1)[0]
        if value / total >= min_share:
            result[label] = {'character': character, 'share': round(value / total, 3), 'lines': count[label]}
    merged = defaultdict(list)
    for label, info in result.items():
        merged[info['character']].append(label)
    return result, {c: labels for c, labels in merged.items() if len(labels) > 1}


def enrol(lines, min_conf=('high',), per_character_max=60):
    """Per-character centroid from the lines the methods agree on (embedding present, confident speaker)."""
    pool = defaultdict(list)
    for line in lines:
        if line.get('embedding') is not None and line.get('speaker') not in (None, 'unknown', 'other') and line.get('confidence') in min_conf:
            pool[line['speaker']].append(unit(line['embedding']))
    centroids = {}
    for character, vectors in pool.items():
        if len(vectors) >= 3:
            centroids[character] = {'vec': unit(np.mean(vectors[:per_character_max], axis=0)), 'lines': len(vectors)}
    return centroids


def enrolled_scores(vec, centroids):
    vec = unit(vec)
    return {c: float(np.dot(vec, v['vec'])) for c, v in centroids.items()}
