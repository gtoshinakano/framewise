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
        self.manual_anchor = None
        self.ignored = False
        self.first_detection_id = None

    @property
    def known(self):
        return self.character_id is not None

    def vectors(self):
        anchor = self.manual_anchor if self.manual_anchor is not None else self.first
        return ([anchor] if anchor is not None else []) + self.recent

    def anchor(self, box, vector):
        """A user correction supersedes the track's accumulated automatic appearance."""
        self.box = list(box)
        self.manual_anchor = vector
        self.recent = []

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


def hungarian(costs):
    """Minimum-cost column for each row of a rectangular matrix (rows <= columns)."""
    if not costs:
        return []
    rows, cols = len(costs), len(costs[0])
    if rows > cols or any(len(row) != cols for row in costs):
        raise ValueError('Assignment needs a rectangular matrix with at least one column per row')
    u, v = [0.] * (rows + 1), [0.] * (cols + 1)
    owner, parent = [0] * (cols + 1), [0] * (cols + 1)
    for row in range(1, rows + 1):
        owner[0], col = row, 0
        distance = [float('inf')] * (cols + 1)
        used = [False] * (cols + 1)
        while owner[col]:
            used[col] = True
            active = owner[col]
            delta, next_col = float('inf'), 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                cost = costs[active - 1][candidate - 1] - u[active] - v[candidate]
                if cost < distance[candidate]:
                    distance[candidate], parent[candidate] = cost, col
                if distance[candidate] < delta:
                    delta, next_col = distance[candidate], candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            col = next_col
        while col:
            previous = parent[col]
            owner[col] = owner[previous]
            col = previous
    result = [-1] * rows
    for col, row in enumerate(owner[1:]):
        if row:
            result[row - 1] = col
    return result


UNMATCHED_COST = .50


def match_evidence(slot, detection, predicted, same_person):
    """Costs use only available identity, geometry and gallery evidence for this pair."""
    overlap = iou(detection['box'], predicted)
    appearance = similarity(None if detection.get('small') else detection.get('vector'), slot)
    scores = detection.get('gallery') or {}
    gallery = scores.get(slot.character_id)
    rival = max((value for key, value in scores.items() if key != slot.character_id), default=None)
    hinted = detection.get('slot_hint') == slot.number
    gallery_leads = gallery is not None and (rival is None or gallery >= rival)
    shape_match = (overlap >= IOU_ACCEPT and (appearance is None or appearance >= same_person - IDENTITY_SLACK)
                   or overlap >= .36 and appearance is not None and appearance >= same_person - .22
                   or overlap >= .28 and gallery_leads
                   or overlap >= .72 and (gallery is None or rival is None or rival - gallery <= .10))
    gallery_match = gallery_leads and gallery >= same_person - .09 and (rival is None or gallery >= rival + .03)
    possible = hinted or shape_match or gallery_match or (appearance is not None and appearance >= same_person)
    # Absent crop features can follow the geometry, but weak or contradictory measured features still count.
    looks = appearance if appearance is not None else same_person - IDENTITY_SLACK
    cost = .62 * (1 - looks) + .38 * (1 - overlap)
    if gallery is not None:
        cost += .22 * (max(0., (rival or 0.) - gallery) - max(0., gallery - same_person + .10))
    if shape_match:
        cost = min(cost, .53 - .22 * overlap)
    if gallery_match:
        cost = min(cost, .48 - .28 * (gallery - .75) - .12 * overlap)
    if hinted:
        cost = min(cost, .22 + .15 * (1 - overlap))
    return {'cost': max(0., cost) if possible else 2., 'iou': overlap, 'similarity': appearance, 'gallery': gallery}


def can_open_slot(item, size, known):
    """Keep small or weakly identified extras as candidates when the cast is open."""
    width, height = size
    x0, y0, x1, y1 = item['box']
    share = (x1 - x0) * (y1 - y0) / (width * height)
    ranked = sorted((item.get('gallery') or {}).items(), key=lambda row: row[1], reverse=True)
    if ranked:
        character, score = ranked[0]
        clear_lead = len(ranked) == 1 or score >= ranked[1][1] + .03
        if character in known and score >= .76 and clear_lead:
            return False  # a likely duplicate, or an existing track with poor geometry
        if share >= .045 and score >= .78 and clear_lead:
            return True
    return share >= .17 and (not ranked or item.get('rank', 0) == 0)


def assign_global(slots, detections, same_person, predicted, *, locked, max_boxes=MAX_BOXES, blocked=(), no_match=UNMATCHED_COST):
    """One Hungarian pass for a frame, with a dummy no-match column per live slot.

    `predicted` is the last box of each slot transformed into this frame's pixels.
    Locked cuts keep every unclaimed candidate out of the coloured slots.
    """
    active = sorted((slot for slot in slots if slot.number not in blocked), key=lambda slot: slot.number)
    evidence = [[match_evidence(slot, item, predicted.get(slot.number), same_person) for item in detections] for slot in active]
    assignments = hungarian([[item['cost'] for item in row] + [no_match] * len(active) for row in evidence])
    matched = [{'detection': col, 'slot': slot.number, 'new': False, 'scores': row[col]}
               for slot, row, col in zip(active, evidence, assignments)
               if col < len(row) and row[col]['cost'] < no_match]
    known = {slot.number for slot in active if slot.known}
    matched.sort(key=lambda entry: (entry['slot'] not in known, entry['scores']['cost'], entry['slot']))
    reserved = {entry['detection'] for entry in matched}
    placed = matched[:max(0, max_boxes)]
    by_number = {slot.number: slot for slot in slots}
    for entry in placed:
        item = detections[entry['detection']]
        vector = None if item.get('small') or item.get('split') else item.get('vector')
        by_number[entry['slot']].seen(item['box'], vector)
    if not locked:
        for index, item in enumerate(detections):
            if len(placed) >= max_boxes:
                break
            if index in reserved or item.get('small') or item.get('split'):
                continue
            if item.get('can_open') is False or any(iou(item['box'], detections[entry['detection']]['box']) >= .55 for entry in placed):
                continue
            number = free_number(slots)
            if number is None:
                break
            slot = Slot(number, 'appeared', box=item['box'], vector=item.get('vector'))
            slots.append(slot)
            placed.append({'detection': index, 'slot': number, 'new': True, 'scores': None})
    used = {entry['detection'] for entry in placed}
    return sorted(placed, key=lambda entry: entry['slot']), [i for i in range(len(detections)) if i not in used]
