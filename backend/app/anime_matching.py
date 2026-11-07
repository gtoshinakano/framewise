"""Matching primitives for the bulk anime-character workflow.

Only explicitly saved character references belong in the persistent index.
Automatic detections and temporal tracks must never be promoted into it.
"""
import math
import os
import struct


def normalize(vector):
    values = [float(value) for value in vector]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError('Embedding must contain finite values')
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('Embedding has zero or invalid magnitude')
    return [value / norm for value in values]


def pack_embedding(vector):
    values = normalize(vector)
    return len(values), struct.pack(f'<{len(values)}f', *values)


def unpack_embedding(blob, dimensions):
    if dimensions <= 0 or len(blob) != dimensions * 4:
        raise ValueError('Invalid cached embedding')
    return normalize(struct.unpack(f'<{dimensions}f', blob))


def rank_references(vector, references):
    """Return the best cosine similarity per character, in descending order."""
    target = normalize(vector)
    scores = {}
    for reference in references:
        candidate = normalize(reference['embedding'])
        if len(candidate) != len(target):
            raise ValueError('Reference embedding dimensions do not match model output')
        score = sum(a * b for a, b in zip(target, candidate))
        character_id = reference['character_context_id']
        scores[character_id] = max(score, scores.get(character_id, -1.0))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _env_fraction(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f'{name} must be between 0 and 1')
    return number


def confident_match(ranked, threshold=None, margin=None):
    """Return a match only with both adequate similarity and separation.

    Defaults (0.85 similarity, 0.03 margin) were tuned on labelled DINOv3
    detections, whose different characters still score close together, so a
    wide margin rejected nearly all correct matches. They are not calibrated
    probabilities. Override with ANIME_REID_MIN_SIMILARITY / ANIME_REID_MIN_MARGIN.
    All other cases need a Qwen review or remain unidentified.
    """
    if threshold is None:
        threshold = _env_fraction('ANIME_REID_MIN_SIMILARITY', 0.85)
    if margin is None:
        margin = _env_fraction('ANIME_REID_MIN_MARGIN', 0.03)
    if not ranked or ranked[0][1] < threshold:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return None
    return ranked[0][0]


def intersection_over_union(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_tracks(previous, current, max_gap=1.0, min_iou=0.65, min_similarity=0.90, margin=0.05):
    """Link short, same-cut observations only when both directions are unique.

    Tracks are deliberately broken at cuts, long time gaps, and ambiguous
    assignments. The caller still determines which linked labels to reuse.
    """
    options = []
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            gap = new['time'] - old['time']
            if old['cut_id'] != new['cut_id'] or gap < 0 or gap > max_gap:
                continue
            if old['image_size'] != new['image_size']:
                continue
            overlap = intersection_over_union(old['box'], new['box'])
            if overlap < min_iou:
                continue
            left, right = normalize(old['embedding']), normalize(new['embedding'])
            if len(left) != len(right):
                raise ValueError('Track embedding dimensions do not match')
            similarity = sum(a * b for a, b in zip(left, right))
            options.append((i, j, similarity, overlap))
    matches = {}
    for i, j, similarity, overlap in options:
        if similarity < min_similarity:
            continue
        old_alternatives = [other for old_i, new_j, other, _ in options if old_i == i and new_j != j]
        new_alternatives = [other for old_i, new_j, other, _ in options if new_j == j and old_i != i]
        if any(similarity - other < margin for other in old_alternatives + new_alternatives):
            continue
        if (old_alternatives and similarity < max(old_alternatives)) or (new_alternatives and similarity < max(new_alternatives)):
            continue
        matches[j] = i
    return matches


def match_adjacent_appearances(previous, current, min_similarity=0.97, margin=0.05):
    """Conservatively associate appearance across adjacent cuts, never geometry.

    Both directions must have a unique best match. The caller is responsible
    for checking cut adjacency and for allowing only still-valid identities.
    """
    options = []
    for i, old in enumerate(previous):
        left = normalize(old['embedding'])
        for j, new in enumerate(current):
            right = normalize(new['embedding'])
            if len(left) != len(right):
                raise ValueError('Track embedding dimensions do not match')
            similarity = sum(a * b for a, b in zip(left, right))
            options.append((i, j, similarity))
    matches = {}
    for i, j, similarity in options:
        if similarity < min_similarity:
            continue
        alternatives = [score for old_i, new_j, score in options
                        if (old_i == i and new_j != j) or (new_j == j and old_i != i)]
        if any(similarity - other < margin for other in alternatives):
            continue
        matches[j] = i
    return matches


def read_cached_reference(conn, reference_id, model_id, model_revision, preprocessing):
    row = conn.execute('''SELECT dimensions,embedding FROM character_reference_embeddings
        WHERE reference_id=? AND model_id=? AND model_revision=? AND preprocessing=?''',
        (reference_id, model_id, model_revision, preprocessing)).fetchone()
    return unpack_embedding(row['embedding'], row['dimensions']) if row else None


def write_cached_reference(conn, reference_id, model_id, model_revision, preprocessing, vector):
    dimensions, blob = pack_embedding(vector)
    conn.execute('''INSERT OR REPLACE INTO character_reference_embeddings
        (reference_id,model_id,model_revision,preprocessing,dimensions,embedding)
        VALUES (?,?,?,?,?,?)''', (reference_id, model_id, model_revision, preprocessing, dimensions, blob))


def read_cached_detection(conn, detection_id, model_id, model_revision, preprocessing):
    row = conn.execute('''SELECT dimensions,embedding FROM character_detection_embeddings
        WHERE detection_id=? AND model_id=? AND model_revision=? AND preprocessing=?''',
        (detection_id, model_id, model_revision, preprocessing)).fetchone()
    return unpack_embedding(row['embedding'], row['dimensions']) if row else None


def write_cached_detection(conn, detection_id, model_id, model_revision, preprocessing, vector):
    dimensions, blob = pack_embedding(vector)
    conn.execute('''INSERT OR REPLACE INTO character_detection_embeddings
        (detection_id,model_id,model_revision,preprocessing,dimensions,embedding)
        VALUES (?,?,?,?,?,?)''', (detection_id, model_id, model_revision, preprocessing, dimensions, blob))
