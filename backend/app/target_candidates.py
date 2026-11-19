"""Build the shared candidate set for the target-frame tracker and its offline replay."""

from . import keyframe_pose, merged_box, target_flow, target_tracking as tracking


def keep_candidate(item, size, band):
    x0, y0, x1, y1 = item['box']
    if min(x1 - x0, y1 - y0) < .012 * min(size):
        return False
    if band and y0 >= band['top'] * size[1] and y1 <= band['bottom'] * size[1] and item['small']:
        return False
    return not (item['small'] and item['rank'] >= 3)


def split_candidates(candidates, predicted, skeletons):
    prepared, splits = [], []
    for item in candidates:
        others = [other['box'] for other in candidates if other is not item]
        proposal = merged_box.propose(item['box'], predicted, skeletons, others)
        if not proposal:
            prepared.append(item)
            continue
        splits.append({'raw_index': item['raw_index'], **proposal})
        prepared.extend({**item, **part, 'vector': None, 'gallery': {}, 'split': True} for part in proposal['parts'])
    return prepared, splits


def prepare(raw, slots, size, camera, band, skeletons=(), *, split=False, previous=None, current=None, verified=(), suppressed=(), protected=()):
    predicted = {slot.number: keyframe_pose.predict(slot.box, camera, *size) for slot in slots if slot.box}
    known = {slot.character_id for slot in slots if slot.character_id}

    def allowed(item):
        box = item['box']
        return (all(tracking.iou(box, other) < tracking.IOU_ACCEPT for other in suppressed)
                and sum(merged_box.coverage(box, other) >= .80 for other in protected) < 2)

    candidates = [{**item, 'can_open': tracking.can_open_slot(item, size, known)} for item in raw
                  if keep_candidate(item, size, band) and allowed(item)]
    splits = []
    if split:
        candidates, splits = split_candidates(candidates, predicted, skeletons)
    if previous is not None and current is not None:
        for slot in slots:
            if slot.number not in verified or slot.number not in predicted:
                continue
            expected = predicted[slot.number]
            if any(tracking.iou(expected, item['box']) >= .47 for item in candidates):
                continue
            found = target_flow.follow(previous, current, slot.box, expected, size)
            if found and all(tracking.iou(found['box'], item['box']) < .45 for item in candidates):
                candidates.append({**found, 'raw_index': None, 'slot_hint': slot.number, 'can_open': False, 'flow': True})
        for pose in skeletons:
            if any(tracking.iou(pose['box'], item['box']) >= .35 for item in candidates):
                continue
            ranks = sorted(((tracking.iou(pose['box'], box), slot) for slot, box in predicted.items()), reverse=True)
            if ranks and ranks[0][0] >= .42 and (len(ranks) == 1 or ranks[0][0] - ranks[1][0] >= .17):
                candidates.append({'box': pose['box'], 'raw_index': None, 'slot_hint': ranks[0][1], 'can_open': False, 'pose': True})
    return [item for item in candidates if allowed(item)], predicted, splits
