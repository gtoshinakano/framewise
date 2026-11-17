"""Pixel evidence per pair of consecutive frames (CPU, cv2): the dominant camera transform and how much changed where.

These are proxies, not ground truth:
* camera: ORB matches + ``cv2.estimateAffinePartial2D`` (RANSAC) on the picture without the subtitle band. The dominant
  transform is usually the background's; a character that fills the frame can dominate it, and too few inliers give
  ``unclear`` (not scored).
* change: ``app.keyframes.compare`` indices (global, regional; 0-1, grayscale levels / 12) for the whole picture and the
  same mean difference inside each first-frame character box (boxes do not follow the character; a big move leaves its box).
"""
import math

STATIC_SHIFT = 0.012   # share of the image width
STATIC_SCALE = 0.015   # relative scale change
STILL_CHANGE = 0.04    # regional index under which a pair counts as visually still
MOVING_BOX = 0.25      # box index above which a character's region clearly changed


def _load(path, width=640):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'cannot read {path}')
    scale = width / image.shape[1]
    return cv2.resize(image, (width, max(2, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def _mask(shape, band):
    import numpy as np
    mask = np.ones(shape, dtype=bool)
    if band and band.get('top') is not None:
        mask[int(band['top'] * shape[0]):math.ceil(band['bottom'] * shape[0]), :] = False
    return mask


def camera_estimate(a, b, mask):
    """{'move', 'dx', 'dy', 'scale', 'rotation', 'inliers', 'ratio'}; dx/dy are the content shift as a share of the width."""
    import cv2
    import numpy as np
    orb = cv2.ORB_create(nfeatures=1500)
    m8 = mask.astype(np.uint8) * 255
    ka, da = orb.detectAndCompute(a, m8)
    kb, db = orb.detectAndCompute(b, m8)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return {'move': 'unclear', 'reason': 'too few features'}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 12:
        return {'move': 'unclear', 'reason': 'too few matches'}
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000, confidence=0.995)
    if matrix is None or inliers is None:
        return {'move': 'unclear', 'reason': 'no transform'}
    count = int(inliers.sum())
    ratio = count / len(matches)
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    width = a.shape[1]
    # Translation of the image centre (so a zoom about the centre is not read as a pan).
    cx, cy = a.shape[1] / 2, a.shape[0] / 2
    dx = float(matrix[0, 0] * cx + matrix[0, 1] * cy + matrix[0, 2] - cx) / width
    dy = float(matrix[1, 0] * cx + matrix[1, 1] * cy + matrix[1, 2] - cy) / width
    out = {'dx': round(dx, 4), 'dy': round(dy, 4), 'scale': round(scale, 4), 'rotation': round(rotation, 2), 'inliers': count, 'ratio': round(ratio, 3)}
    if count < 10 or ratio < 0.25:
        return {**out, 'move': 'unclear', 'reason': 'few inliers'}
    if abs(scale - 1) >= STATIC_SCALE:
        move = 'zoom_in' if scale > 1 else 'zoom_out'
    elif max(abs(dx), abs(dy)) >= STATIC_SHIFT:
        # Content moving left means the camera turned right.
        move = ('pan_right' if dx < 0 else 'pan_left') if abs(dx) >= abs(dy) else ('tilt_up' if dy > 0 else 'tilt_down')
    elif abs(rotation) >= 2:
        move = 'orbit'
    else:
        move = 'static'
    return {**out, 'move': move}


def _box_change(a, b, mask, box, normalization):
    import numpy as np
    h, w = a.shape
    x0, y0, x1, y1 = [int(round(v / 1000 * (w if i % 2 == 0 else h))) for i, v in enumerate(box)]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    region = mask[y0:y1, x0:x1]
    if x1 - x0 < 4 or y1 - y0 < 4 or not region.any():
        return None
    diff = np.abs(a[y0:y1, x0:x1].astype(np.float32) - b[y0:y1, x0:x1].astype(np.float32))
    return round(min(1.0, float(diff[region].mean()) / normalization), 3)


def measure_sample(sample, run_dir):
    """Per consecutive pair of the cut's frames: camera estimate, change indices, per-character box change."""
    import cv2
    from app import keyframes
    settings = keyframes.config({})
    frames = sample['frames']
    images = [_load(f"{run_dir}/{frame['path']}") for frame in frames]
    pairs = []
    for i in range(len(frames) - 1):
        a, b = images[i], images[i + 1]
        mask = _mask(a.shape, sample.get('subtitle_band'))
        blurred = [cv2.GaussianBlur(img, (settings['blur'], settings['blur']), 0) for img in (a, b)]
        for img in blurred:
            img[~mask] = 0
        g, r, coverage = keyframes.compare(blurred[1], blurred[0], mask, settings)
        boxes = {}
        for person in sample.get('roster') or []:
            value = _box_change(blurred[0], blurred[1], mask, person['box'], settings['normalization'])
            if value is not None:
                boxes[person['name']] = value
        pairs.append({'from': frames[i]['label'], 'to': frames[i + 1]['label'], 'start': frames[i]['t'], 'end': frames[i + 1]['t'],
                      'camera': camera_estimate(a, b, mask), 'global': round(g, 3), 'regional': round(r, 3), 'coverage': round(coverage, 3),
                      'still': max(g, r) < STILL_CHANGE, 'boxes': boxes})
    return pairs
