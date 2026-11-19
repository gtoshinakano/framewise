"""Short-gap propagation of an existing box from tracked image features."""
import math


def follow(previous, current, box, predicted, size):
    """Return a box and flow support, or None when the pixels cannot corroborate it.

    Images are the grayscale keyframe copies. Start on the inner part of the old
    box to avoid following surrounding scenery. Require forward-backward agreement
    and a robust similarity transform shared by its feature points.
    """
    import cv2
    import numpy as np

    height, width = previous.shape
    scale = width / size[0]
    x0, y0, x1, y1 = [x * scale for x in box]
    dx, dy = (x1 - x0) * .16, (y1 - y0) * .12
    left, right = max(1, math.ceil(x0 + dx)), min(width - 1, math.floor(x1 - dx))
    top, bottom = max(1, math.ceil(y0 + dy)), min(height - 1, math.floor(y1 - dy))
    if right - left < 16 or bottom - top < 16:
        return None
    mask = np.zeros(previous.shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    points = cv2.goodFeaturesToTrack(previous, maxCorners=75, qualityLevel=.018, minDistance=5, mask=mask)
    if points is None or len(points) < 8:
        return None
    px, py, qx, qy = [v * scale for v in predicted]
    projected = points.copy()
    projected[..., 0] = px + (points[..., 0] - x0) * (qx - px) / (x1 - x0)
    projected[..., 1] = py + (points[..., 1] - y0) * (qy - py) / (y1 - y0)
    landed, tracked, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, projected, winSize=(25, 25), maxLevel=3,
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if landed is None or not tracked.any():
        return None
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, landed, points.copy(), winSize=(25, 25), maxLevel=3,
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if back is None:
        return None
    agree = (tracked[:, 0] > 0) & (reverse[:, 0] > 0) & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.5)
    if agree.sum() < 8:
        return None
    a, b = points[agree, 0], landed[agree, 0]
    transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if transform is None or inliers is None:
        return None
    strong = inliers[:, 0] > 0
    support = float(strong.sum() / len(points))
    if strong.sum() < 8 or support < .55 or np.ptp(a[strong, 0]) < (right - left) * .28 or np.ptp(a[strong, 1]) < (bottom - top) * .28:
        return None
    zoom = math.hypot(transform[0, 0], transform[1, 0])
    if not .60 <= zoom <= 1.65 or abs(transform[1, 0]) > .2 * zoom:
        return None
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float32)
    transformed = corners @ transform[:, :2].T + transform[:, 2]
    lo, hi = transformed.min(axis=0), transformed.max(axis=0)
    box = [max(0., lo[0] / scale), max(0., lo[1] / scale), min(size[0], hi[0] / scale), min(size[1], hi[1] / scale)]
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    return {'box': box, 'support': round(support, 3)}
