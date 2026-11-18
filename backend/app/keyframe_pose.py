"""DWPose whole-body keypoints on the keyframes of one cut, on the CPU (onnxruntime), with names from the character boxes.

Ported from ``multiframe_experiment/pose.py``. Weights (Apache-2.0, https://huggingface.co/yzd-v/DWPose, pinned in
``model_manifest.json`` as ``dwpose``): ``yolox_l.onnx`` (person detector) and ``dw-ll_ucoco_384.onnx`` (top-down
whole-body pose, 133 COCO-WholeBody keypoints, SimCC heads), mounted read-only at ``DWPOSE_DIR`` (``/models/dwpose``).
The sessions use ``CPUExecutionProvider`` only: this runs inside the keyframe stage while the GPU holds Qwen3.5.

People per keyframe (``keyframe_analysis`` builds the anchors):
* anchor boxes: the Target frames tab's boxes of that keyframe when its detection run is current, else the first
  frame's character boxes on the first keyframe;
* anyone the anchors miss comes from the YOLOX detector;
* people are tracked across keyframes by box overlap through the camera transform, else the nearest centre. A track
  takes the name of any named anchor it meets; two anchors of different Target frames slots never share a track.

The geometry and tracking are stdlib; only ``DWPose`` needs onnxruntime, numpy and cv2 (imported lazily).
"""
import math
import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get('DWPOSE_DIR', '/models/dwpose'))
DETECTOR, POSE = 'yolox_l.onnx', 'dw-ll_ucoco_384.onnx'
KP_THRESHOLD = 0.3         # keypoint score under which a point counts as not found (DWPose draws at 0.3)
USABLE_BODY_POINTS = 6     # a body is usable with at least this many of the 17 body points over the threshold
DET_SCORE = 0.3
CLOSE_UP = 0.8             # box height share of the image above which a person is a close-up


# -- geometry (pure) ----------------------------------------------------------------------------------------------------------

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def body_usable(points):
    return sum(1 for p in points[:17] if p and p[2] >= KP_THRESHOLD) >= USABLE_BODY_POINTS


def body_confidence(points):
    scores = [p[2] for p in points[:17] if p]
    return sum(scores) / len(scores) if scores else 0.0


def suspected_swap(prev, cur, box_prev, box_cur):
    """Left and right wrists traded places while the box stayed put: a DWPose left/right flip, not a motion."""
    if iou(box_prev, box_cur) < 0.7:
        return False
    a, b = (prev[9], prev[10]), (cur[9], cur[10])
    if not all(p and p[2] >= KP_THRESHOLD for p in (*a, *b)):
        return False
    d = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])  # noqa: E731
    straight = d(a[0], b[0]) + d(a[1], b[1])
    crossed = d(a[0], b[1]) + d(a[1], b[0])
    height = box_cur[3] - box_cur[1]
    return crossed < 0.5 * straight and straight > 0.15 * height


def predict(box, camera, width, height):
    """Where a box lands after the pair's camera transform (pixel estimate: scale about the centre, then shift)."""
    if not camera or camera.get('move') in (None, 'unclear') or camera.get('scale') is None:
        return box
    s, dx, dy = camera['scale'], (camera.get('dx') or 0) * width, (camera.get('dy') or 0) * width
    return [(box[0] - width / 2) * s + width / 2 + dx, (box[1] - height / 2) * s + height / 2 + dy,
            (box[2] - width / 2) * s + width / 2 + dx, (box[3] - height / 2) * s + height / 2 + dy]


def _match_score(person, item, width):
    if person.get('name') and item['name'] and person['name'] != item['name']:
        return None
    if person.get('slot') is not None and item.get('slot') is not None and person['slot'] != item['slot']:
        return None  # two Target frames slots are two people, whatever their boxes look like
    overlap = iou(person['box'], item['box'])
    (cx, cy), (tx, ty) = centre(person['box']), centre(item['box'])
    near = 1 - math.hypot(cx - tx, cy - ty) / (0.3 * width)
    h1, h2 = person['box'][3] - person['box'][1], item['box'][3] - item['box'][1]
    size = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
    named = 1.0 if (person.get('name') and person['name'] == item['name']) or (person.get('slot') is not None and person['slot'] == item.get('slot')) else 0.0
    if not (overlap > 0.1 or near > 0 or named):
        return None
    return max(overlap * 2, near) * (0.5 + 0.5 * size) + named


def track(frames_people, width, height=None, cameras=None):
    """Assign a track id to every person of every frame (in place).

    Track boxes are carried through the camera transform of each pair (``cameras[i]`` is the pixel estimate from
    frame i to i+1), then matched globally: best (overlap or nearness) x size similarity first, a shared name wins,
    two different names never match.
    """
    tracks, next_id = [], 1  # each: {'id', 'box', 'name'}
    for index, people in enumerate(frames_people):
        if index and cameras and height:
            for item in tracks:
                item['box'] = predict(item['box'], cameras[index - 1] if index - 1 < len(cameras) else None, width, height)
        candidates = sorted(((score, n, item['id']) for n, person in enumerate(people) for item in tracks
                             if (score := _match_score(person, item, width)) is not None), reverse=True)
        taken_people, taken_tracks, chosen = set(), set(), {}
        for score, n, track_id in candidates:
            if n in taken_people or track_id in taken_tracks:
                continue
            chosen[n] = track_id
            taken_people.add(n)
            taken_tracks.add(track_id)
        by_id = {item['id']: item for item in tracks}
        for n, person in enumerate(people):
            item = by_id.get(chosen.get(n))
            if item is None:
                item = {'id': next_id, 'box': person['box'], 'name': person.get('name'), 'slot': person.get('slot'), 'character_context_id': person.get('character_context_id')}
                next_id += 1
                tracks.append(item)
            item['box'] = person['box']
            if person.get('name') and not item['name']:
                item['name'], item['character_context_id'] = person['name'], person.get('character_context_id')
            if person.get('slot') is not None and item.get('slot') is None:
                item['slot'] = person['slot']
            person['track'] = item['id']
    names = {t['id']: t['name'] for t in tracks}
    ids = {t['id']: t.get('character_context_id') for t in tracks}
    unnamed = {}
    for people in frames_people:
        for person in people:
            name = names.get(person['track'])
            if not name:
                name = unnamed.setdefault(person['track'], f'unknown person {len(unnamed) + 1}')
            person['name'] = name
            person['character_context_id'] = ids.get(person['track'])
    return frames_people


def track_stats(frames):
    """Per character: keyframes seen, share with a usable body, mean body confidence, suspected swaps, close-ups."""
    stats, last = {}, {}
    for frame in frames:
        for person in frame['persons']:
            item = stats.setdefault(person['name'], {'name': person['name'], 'frames': 0, 'usable': 0, 'confidence': [], 'swaps': 0, 'close_ups': 0,
                                                      'close_up_usable': 0, 'sources': set()})
            item['frames'] += 1
            item['usable'] += person['usable']
            item['confidence'].append(person['confidence'])
            item['sources'].add(person['source'])
            if person['close_up']:
                item['close_ups'] += 1
                item['close_up_usable'] += person['usable']
            previous = last.get(person['name'])
            if previous and suspected_swap(previous['points'], person['points'], previous['box'], person['box']):
                item['swaps'] += 1
                person['swap'] = True
            last[person['name']] = person
    out = []
    for item in stats.values():
        values = item.pop('confidence')
        out.append({**item, 'sources': sorted(item['sources']), 'usable_share': round(item['usable'] / item['frames'], 3) if item['frames'] else None,
                    'mean_confidence': round(sum(values) / len(values), 3) if values else None})
    return out


# -- DWPose (onnxruntime, numpy, cv2) -------------------------------------------------------------------------------------------

class DWPose:
    def __init__(self, folder=MODEL_DIR, threads=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        providers = ['CPUExecutionProvider']  # never the GPU: Qwen3.5 holds it while this runs
        self.det = ort.InferenceSession(str(Path(folder) / DETECTOR), options, providers=providers)
        self.pose = ort.InferenceSession(str(Path(folder) / POSE), options, providers=providers)
        self.files = {name: str(Path(folder) / name) for name in (DETECTOR, POSE)}

    def detect(self, bgr):
        """YOLOX-L person boxes [x0, y0, x1, y1, score] in image pixels."""
        import cv2
        import numpy as np
        size = 640
        h, w = bgr.shape[:2]
        ratio = min(size / h, size / w)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        resized = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.det.run(None, {self.det.get_inputs()[0].name: blob})[0][0]
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = size // stride
            ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        grids, strides = np.concatenate(grids), np.concatenate(strides)
        xy = (out[:, :2] + grids) * strides
        wh = np.exp(out[:, 2:4]) * strides
        scores = out[:, 4] * out[:, 5]  # objectness x person class
        keep = scores > DET_SCORE
        xy, wh, scores = xy[keep], wh[keep], scores[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / ratio
        order = scores.argsort()[::-1]
        chosen = []
        for i in order:
            box = boxes[i]
            if all(iou(box, boxes[j]) < 0.45 for j in chosen):
                chosen.append(i)
        return [[float(max(0, boxes[i][0])), float(max(0, boxes[i][1])), float(min(w, boxes[i][2])), float(min(h, boxes[i][3])), float(scores[i])] for i in chosen]

    def keypoints(self, rgb, box):
        """133 keypoints [x, y, score] in image pixels for one person box (top-down, padding 1.25, SimCC split 2)."""
        import cv2
        import numpy as np
        in_w, in_h = 288, 384
        x0, y0, x1, y1 = box[:4]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = (x1 - x0) * 1.25, (y1 - y0) * 1.25
        if bw / bh > in_w / in_h:
            bh = bw * in_h / in_w
        else:
            bw = bh * in_w / in_h
        src = np.float32([[cx, cy], [cx, cy - bh / 2], [cx - bw / 2, cy]])
        dst = np.float32([[in_w / 2, in_h / 2], [in_w / 2, 0], [0, in_h / 2]])
        matrix = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(rgb, matrix, (in_w, in_h), flags=cv2.INTER_LINEAR).astype(np.float32)
        crop = (crop - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        blob = crop.transpose(2, 0, 1)[None].astype(np.float32)
        simcc_x, simcc_y = self.pose.run(None, {self.pose.get_inputs()[0].name: blob})
        sx, sy = simcc_x[0], simcc_y[0]
        px, py = sx.argmax(1) / 2.0, sy.argmax(1) / 2.0
        score = np.minimum(sx.max(1), sy.max(1))
        # back from the 288x384 crop to the image
        gx = px / in_w * bw + cx - bw / 2
        gy = py / in_h * bh + cy - bh / 2
        return [[round(float(x), 1), round(float(y), 1), round(float(s), 3)] for x, y, s in zip(gx, gy, score)]


def people_for_frame(model, bgr, anchors):
    """Anchors first; detector boxes overlapping no anchor are added. Each person gets keypoints."""
    rgb = bgr[:, :, ::-1].copy()
    people = [dict(a) for a in anchors]
    detected = model.detect(bgr)
    for box in detected:
        if all(iou(box, a['box']) < 0.3 for a in anchors):
            people.append({'box': box[:4], 'name': None, 'source': 'detector', 'det_score': round(box[4], 3)})
    for person in people:
        person['box'] = [round(v, 1) for v in person['box']]
        person['points'] = model.keypoints(rgb, person['box'])
    return people, len(detected)


def describe_people(sizes, frames_people):
    """Usability, confidence and close-up flags of every tracked person (in place); ``sizes``: ``[w, h]`` per keyframe."""
    for (_, height), people in zip(sizes, frames_people):
        for person in people:
            person['usable'] = body_usable(person['points'])
            person['confidence'] = round(body_confidence(person['points']), 3)
            person['close_up'] = (person['box'][3] - person['box'][1]) / height >= CLOSE_UP
    return frames_people
