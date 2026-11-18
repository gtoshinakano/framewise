"""The camera between two keyframes, measured from pixels (cv2), never asked of a model (ported from ``multiframe_experiment/measure.py``).

ORB matches and ``cv2.estimateAffinePartial2D`` (RANSAC) on the picture without the subtitle band give the dominant
transform. With every person masked out (``camera_without_people``) it is the backdrop's: in a talking-heads cut the
faces otherwise dominate it and a static shot reads as a zoom or pan. Too few features or inliers give ``unclear``;
no estimate beats a wrong one. A close-up leaves no backdrop, so ``unclear`` is common there.
"""
import math

STATIC_SHIFT = 0.012   # share of the image width
STATIC_SCALE = 0.015   # relative scale change
WIDTH = 640            # the estimate works on grayscale copies this wide
PAD = 0.05             # a masked person box grows by this share of its size on every side


def gray(bgr, width=WIDTH):
    """A grayscale copy ``width`` pixels wide."""
    import cv2
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    scale = width / image.shape[1]
    return cv2.resize(image, (width, max(2, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def band_mask(shape, band):
    """True where the picture counts: everything but the subtitle band (``{top, bottom}`` as shares of the height)."""
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


def mask_people(mask, boxes, size, pad=PAD):
    """Mask out person boxes (pixels of a ``size`` = ``[w, h]`` frame) on a mask of the grayscale copy (in place)."""
    sx, sy = mask.shape[1] / size[0], mask.shape[0] / size[1]
    for x0, y0, x1, y1 in boxes:
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        mask[max(0, int((y0 - py) * sy)):int((y1 + py) * sy) + 1, max(0, int((x0 - px) * sx)):int((x1 + px) * sx) + 1] = False
    return mask


def pair_cameras(grays, band, boxes=None, sizes=None):
    """One estimate per consecutive pair of ``grays``; with ``boxes`` (person boxes per frame) and ``sizes``, people are masked in both frames."""
    out = []
    for i in range(len(grays) - 1):
        a, b = grays[i], grays[i + 1]
        mask = band_mask(a.shape, band)
        if boxes is not None:
            for k in (i, i + 1):
                mask_people(mask, boxes[k], sizes[k])
        out.append(camera_estimate(a, b, mask))
    return out
