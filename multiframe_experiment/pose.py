"""DWPose whole-body keypoints for every keyframe of a cut, on the CPU (onnxruntime), with names from the character boxes.

Weights (Apache-2.0, https://huggingface.co/yzd-v/DWPose, see README): ``yolox_l.onnx`` (person detector) and
``dw-ll_ucoco_384.onnx`` (top-down whole-body pose, 133 COCO-WholeBody keypoints, SimCC heads). They are mounted
read-only at ``/models/dwpose`` (``DWPOSE_DIR``). The sessions use ``CPUExecutionProvider`` only, so this runs in
the ``prepare`` stage while the inference worker keeps the GPU.

Persons per keyframe:
* anchor boxes: the cut's character boxes from the latest detection run (labelled or not) whose frame time matches
  the keyframe (the run's first frame and, when the cut kept its end, its last frame);
* every other keyframe, and anyone the anchors miss, come from the YOLOX detector;
* people are tracked across keyframes by box overlap, else the nearest centre, and a track takes the name of any
  named anchor it meets (cut 158: the officer is named in F1, Tomoe and Sakuta only in the last frame).

Output: ``pose.json`` in the run folder (per sample: frames with persons, keypoints, box, track, seconds; per track
usability stats) and overlays ``frames/<sample>/pose/F<n>.jpg``.
"""
import json
import logging
import math
import os
import time
from pathlib import Path

log = logging.getLogger('mfx.pose')

MODEL_DIR = Path(os.environ.get('DWPOSE_DIR', '/models/dwpose'))
DETECTOR, POSE = 'yolox_l.onnx', 'dw-ll_ucoco_384.onnx'
KP_THRESHOLD = 0.3         # keypoint score under which a point counts as not found (DWPose draws at 0.3)
USABLE_BODY_POINTS = 6     # a body is usable with at least this many of the 17 body points over the threshold
DET_SCORE = 0.3
ANCHOR_SECONDS = 0.1       # a detection-run frame applies to a keyframe this close in time
CLOSE_UP = 0.8             # box height share of the image above which a person is a close-up

BODY_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (0, 2), (1, 3), (2, 4), (15, 17), (15, 19), (16, 20), (16, 22)]
HAND_LIMBS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


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
    overlap = iou(person['box'], item['box'])
    (cx, cy), (tx, ty) = centre(person['box']), centre(item['box'])
    near = 1 - math.hypot(cx - tx, cy - ty) / (0.3 * width)
    h1, h2 = person['box'][3] - person['box'][1], item['box'][3] - item['box'][1]
    size = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
    named = 1.0 if person.get('name') and person['name'] == item['name'] else 0.0
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
                item = {'id': next_id, 'box': person['box'], 'name': person.get('name')}
                next_id += 1
                tracks.append(item)
            item['box'] = person['box']
            if person.get('name') and not item['name']:
                item['name'] = person['name']
            person['track'] = item['id']
    names = {t['id']: t['name'] for t in tracks}
    unnamed = {}
    for people in frames_people:
        for person in people:
            name = names.get(person['track'])
            if not name:
                name = unnamed.setdefault(person['track'], f'unknown person {len(unnamed) + 1}')
            person['name'] = name
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
        providers = ['CPUExecutionProvider']  # never the GPU: this runs beside the inference worker
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


def anchors_for(sample, frame, scale):
    """The detection run's boxes that belong to this keyframe (scaled to the extracted frame)."""
    out = []
    for item in sample.get('boxes') or []:
        if item.get('time') is None or abs(item['time'] - (sample['start'] + frame['t'])) > ANCHOR_SECONDS:
            continue
        box = [v * scale for v in item['box']]
        out.append({'box': box, 'name': item.get('name'), 'source': f"box:{item['position']}:{item.get('label_source') or '?'}"})
    return out


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


def draw(bgr, people, path):
    import cv2
    image = bgr.copy()
    h = image.shape[0]
    palette = [(80, 200, 255), (120, 255, 120), (255, 160, 80), (220, 120, 255), (90, 90, 255), (255, 255, 120)]
    for person in people:
        color = palette[(person['track'] - 1) % len(palette)]
        x0, y0, x1, y1 = [int(v) for v in person['box']]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        pts = person['points']
        ok = lambda i: pts[i][2] >= KP_THRESHOLD  # noqa: E731
        for a, b in BODY_LIMBS:
            if ok(a) and ok(b):
                cv2.line(image, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 3, cv2.LINE_AA)
        for base in (91, 112):
            for a, b in HAND_LIMBS:
                if ok(base + a) and ok(base + b):
                    cv2.line(image, (int(pts[base + a][0]), int(pts[base + a][1])), (int(pts[base + b][0]), int(pts[base + b][1])), (255, 255, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(pts[:23]):
            shade = max(0, min(255, int(255 * p[2])))
            cv2.circle(image, (int(p[0]), int(p[1])), 4 if p[2] >= KP_THRESHOLD else 2, (0, shade, 255 - shade) if p[2] >= KP_THRESHOLD else (60, 60, 60), -1)
        label = f"{person['name']} {person['confidence']:.2f}{' usable' if person['usable'] else ' NOT usable'}{' SWAP?' if person.get('swap') else ''}"
        y = max(18, y0 - 6) if y0 > 24 else min(h - 6, y1 + 18)
        cv2.putText(image, label, (x0 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, (x0 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_sample(model, sample, run_dir):
    import cv2
    scale = None
    frames, raw_people = [], []
    for frame in sample['frames']:
        bgr = cv2.imread(str(Path(run_dir) / frame['path']))
        if bgr is None:
            raise RuntimeError(f"cannot read {frame['path']}")
        scale = bgr.shape[1] / sample['width']
        began_wall, began_cpu = time.monotonic(), time.process_time()
        people, detected = people_for_frame(model, bgr, anchors_for(sample, frame, scale))
        frames.append({'label': frame['label'], 't': frame['t'], 'size': [bgr.shape[1], bgr.shape[0]], 'detected': detected,
                       'seconds': round(time.monotonic() - began_wall, 3), 'cpu_seconds': round(time.process_time() - began_cpu, 3)})
        raw_people.append(people)
    width, height = frames[0]['size'] if frames else (1, 1)
    track(raw_people, width, height, [pair.get('camera') for pair in sample.get('pairs') or []])
    for frame, people in zip(frames, raw_people):
        h = frame['size'][1]
        for person in people:
            person['usable'] = body_usable(person['points'])
            person['confidence'] = round(body_confidence(person['points']), 3)
            person['close_up'] = (person['box'][3] - person['box'][1]) / h >= CLOSE_UP
        frame['persons'] = people
    stats = track_stats(frames)
    folder = Path(run_dir) / 'frames' / sample['id'] / 'pose'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(folder, 0o777)
    except OSError:
        pass
    for frame, source in zip(frames, sample['frames']):
        path = folder / f"{frame['label']}.jpg"
        draw(cv2.imread(str(Path(run_dir) / source['path'])), frame['persons'], path)
        frame['overlay'] = f"frames/{sample['id']}/pose/{frame['label']}.jpg"
    return {'sample': sample['id'], 'frames': frames, 'characters': stats,
            'seconds': round(sum(f['seconds'] for f in frames), 2), 'cpu_seconds': round(sum(f['cpu_seconds'] for f in frames), 2)}


def run(samples, run_dir, progress=None):
    """pose.json for every sample with at least one frame. Loads both ONNX sessions once (CPU)."""
    began = time.monotonic()
    model = DWPose()
    load = round(time.monotonic() - began, 2)
    out = {'model': {'detector': DETECTOR, 'pose': POSE, 'source': 'https://huggingface.co/yzd-v/DWPose', 'license': 'apache-2.0',
                     'threshold': KP_THRESHOLD, 'usable_points': USABLE_BODY_POINTS, 'provider': 'CPUExecutionProvider', 'load_seconds': load},
           'samples': {}}
    for number, sample in enumerate(samples, 1):
        result = run_sample(model, sample, run_dir)
        out['samples'][sample['id']] = result
        log.info('pose %s: %d frames, %s, %.1fs wall / %.1fs cpu', sample['id'], len(result['frames']),
                 ', '.join(f"{c['name']} {c['usable']}/{c['frames']} usable" for c in result['characters']) or 'nobody', result['seconds'], result['cpu_seconds'])
        if progress:
            progress(number, len(samples), f"pose {sample['id']}")
    return out


def save(run_dir, data):
    path = Path(run_dir) / 'pose.json'
    temp = path.with_name('pose.json.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, default=list))
    try:
        os.chmod(temp, 0o666)
    except OSError:
        pass
    os.replace(temp, path)
