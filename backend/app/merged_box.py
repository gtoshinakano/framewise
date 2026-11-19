"""Find two distinct tracked people inside one detection, in frame pixels."""
import math
from itertools import combinations

from .target_tracking import iou


def area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def intersect(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def coverage(outer, inner):
    size = area(inner)
    return area(intersect(outer, inner)) / size if size else 0.


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _parts(outer, a, b):
    if min(coverage(outer, item) for item in (a, b)) < .80 or iou(a, b) >= .65:
        return None
    children = [intersect(outer, item) for item in (a, b)]
    if min(area(item) for item in children) < .10 * area(outer) or max(area(item) for item in children) > .88 * area(outer):
        return None
    # Duplicate poses from different crops do not constitute two people.
    distance = math.dist(centre(a), centre(b))
    if distance < .15 * min(max(a[2] - a[0], a[3] - a[1]), max(b[2] - b[0], b[3] - b[1])):
        return None
    return children


def _pose_box(item):
    detector = item['box']
    points = [(x, y) for x, y, score in item['points'][:17] if score >= .3]
    if len(points) < 6:
        return None
    xs, ys = zip(*points)
    dx, dy = (detector[2] - detector[0]) * .15, (detector[3] - detector[1]) * .16
    extent = [min(xs) - dx, min(ys) - dy, max(xs) + dx, max(ys) + dy]
    # Keep the YOLOX crop envelope on partly visible poses, but constrain gross keypoint outliers.
    return intersect(detector, extent) if area(intersect(detector, extent)) >= .55 * area(detector) else detector


def propose(box, predicted, skeletons=(), others=()):
    """A two-person proposal or None. Each child can carry its predicted slot.

    ``others`` are the other raw boxes in this frame, so an already isolated person
    does not become a duplicate split child. This decision has no identity side effects.
    """
    by_pose = [pose for item in skeletons if (pose := _pose_box(item)) is not None]
    pairs = []
    for a, b in combinations(by_pose, 2):
        children = _parts(box, a, b)
        if children:
            evidence = sum(coverage(box, child) for child in (a, b)) - iou(a, b)
            pairs.append((evidence, children))
    if pairs:
        children = max(pairs, key=lambda item: item[0])[1]
        ranked = [sorted(((iou(child, p), slot) for slot, p in predicted.items()), reverse=True) for child in children]
        hints = [matches[0][1] if matches and matches[0][0] >= .35 else None for matches in ranked]
        if hints[0] == hints[1]:
            hints = [None, None]
        source = 'dwpose'
    else:
        choices = []
        for (aa, a), (bb, b) in combinations(sorted(predicted.items()), 2):
            children = _parts(box, a, b)
            if children:
                choices.append((coverage(box, a) + coverage(box, b) - iou(a, b), children, (aa, bb)))
        if not choices:
            return None
        _, children, hints = max(choices, key=lambda item: item[0])
        source = 'prediction'
    if any(iou(child, other) >= .50 for child in children for other in others):
        return None
    return {'source': source, 'parts': [{'box': child, 'slot_hint': hint} for child, hint in zip(children, hints, strict=True)]}
