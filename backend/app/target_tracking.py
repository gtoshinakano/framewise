"""Keeps one person in one numbered slot across the evaluated (target) frames of a cut.

Pure code: no database, no models. ``target_characters`` feeds it boxes and identity vectors.

A slot is a person of the cut. Slots 0..n-1 start as the first frame's boxes in the order the first-frame
editor colours them, so a colour means the same person in every frame. A person who is not in the first frame
gets the next free slot when they appear. A frame stores at most ``MAX_BOXES`` boxes, and a cut at most
``MAX_SLOTS`` slots (the editor has that many colours).

AniRef returns boxes without a score, best first, so the order of detections is their confidence rank.
"""
MAX_BOXES = 3
MAX_SLOTS = 8
IOU_WEIGHT = .35        # how much a box overlapping the slot's last box adds to an identity match
IOU_ACCEPT = .5         # overlap that may stand in for a weaker identity match
IDENTITY_SLACK = .15    # how far below the same-person similarity an overlapping box may fall
KEPT_VECTORS = 3        # recent identity vectors a slot remembers besides its first


class Slot:
    """One person: who they are (``character_id``), where they were last seen and what they looked like."""

    def __init__(self, number, origin, character_id=None, label_source=None, box=None, vector=None):
        self.number = number
        self.origin = origin                  # 'first_frame' or 'appeared'
        self.character_id = character_id
        self.label_source = label_source
        self.box = list(box) if box else None
        self.first = vector
        self.recent = []

    @property
    def known(self):
        return self.character_id is not None

    def vectors(self):
        return ([self.first] if self.first is not None else []) + self.recent

    def seen(self, box, vector):
        self.box = list(box)
        if vector is not None:
            if self.first is None:
                self.first = vector
            else:
                self.recent = (self.recent + [vector])[-KEPT_VECTORS:]


def iou(a, b):
    if not a or not b:
        return 0.
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.
    inside = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inside
    return inside / union if union > 0 else 0.


def similarity(vector, slot):
    """Best dot product between a box's identity vector and what the slot looked like, or None."""
    if vector is None:
        return None
    best = None
    for other in slot.vectors():
        value = sum(a * b for a, b in zip(vector, other))
        best = value if best is None or value > best else best
    return best


def free_number(slots):
    used = {slot.number for slot in slots}
    return next((number for number in range(MAX_SLOTS) if number not in used), None)


def assign_frame(slots, detections, same_person, max_boxes=MAX_BOXES, blocked=()):
    """Place one frame's detections in slots; new people are appended to ``slots``.

    ``detections`` are dicts with ``box``, ``vector`` (None when it cannot be told) and ``small`` (too small to
    identify: it may follow a slot by overlap but never starts one), best first. ``same_person`` is the
    similarity from which two boxes of one cut are one person. ``blocked`` are slot numbers that already hold a
    box in this frame (placed by the user): they take no detection, and ``max_boxes`` is what is left of the cap.

    Returns ``[{'detection': index, 'slot': number, 'new': bool}]`` by slot number, at most ``max_boxes``.
    Matched slots win over new people, and slots with a known character win over unknown ones, both in slot
    order; people who do not fit are left out.
    """
    max_boxes = max(0, max_boxes)
    pairs = []
    for index, detection in enumerate(detections):
        vector = None if detection.get('small') else detection.get('vector')
        for slot in slots:
            if slot.number in blocked:
                continue
            overlap = iou(detection['box'], slot.box)
            likeness = similarity(vector, slot)
            same = likeness is not None and likeness >= same_person
            followed = overlap >= IOU_ACCEPT and (likeness is None or likeness >= same_person - IDENTITY_SLACK)
            if same or followed:
                pairs.append((((likeness if likeness is not None else same_person) + IOU_WEIGHT * overlap), -index, -slot.number, index, slot))
    taken, used = {}, set()
    for _, _, _, index, slot in sorted(pairs, key=lambda item: item[:3], reverse=True):
        if index not in taken and slot.number not in used:
            taken[index] = slot
            used.add(slot.number)
    kept = sorted(taken.items(), key=lambda item: (not item[1].known, item[1].number))[:max_boxes]
    placed = [{'detection': index, 'slot': slot.number, 'new': False} for index, slot in kept]
    for index, slot in kept:
        slot.seen(detections[index]['box'], None if detections[index].get('small') else detections[index].get('vector'))
    for index, detection in enumerate(detections):
        if len(placed) >= max_boxes:
            break
        if index in taken or detection.get('small'):
            continue
        number = free_number(slots)
        if number is None:
            break
        slot = Slot(number, 'appeared')
        slot.seen(detection['box'], detection.get('vector'))
        slots.append(slot)
        placed.append({'detection': index, 'slot': number, 'new': True})
    return sorted(placed, key=lambda item: item['slot'])
