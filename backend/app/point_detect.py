"""Find the person under a double-clicked point and return their box (the inference worker side).

The API (``point_detect_api``) writes one ``point_detection_requests`` row per double-click with the frame's
render spec, the clicked point and the boxes already in that frame, then waits for the row. This thread
renders the frame, runs AniRef on the CPU at a low confidence (the user says someone is there), first on the
whole frame and, when nothing there holds the point, on a square crop around it (small or distant people),
and keeps the detection that contains the point and is not a person already boxed. It only answers: the
frontend saves the box through the ordinary add-box endpoints, and recognition names it later. Not a job
kind: it is sub-second CPU work that must answer while the job lanes hold the GPU for minutes.
"""
import io
import json
import logging
import os
import threading
import time
import uuid

from . import db, frame_cache
from .target_tracking import iou

log = logging.getLogger(__name__)
LEASE_SECONDS = 60
POLL_SECONDS = 0.1
BEAT_SECONDS = 5
CONFIDENCE = 0.05
SAME_PERSON_IOU = 0.6
TIE = 0.1
CROP_MIN = 320
CROP_SHARE = 0.45

SCHEMA = """
CREATE TABLE IF NOT EXISTS point_detection_requests (
    id TEXT PRIMARY KEY, surface TEXT NOT NULL, run_id TEXT NOT NULL, frame_key TEXT NOT NULL,
    point_json TEXT NOT NULL, spec_json TEXT NOT NULL, image_size_json TEXT NOT NULL, existing_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'queued', box_json TEXT, score REAL, pass TEXT, error TEXT,
    lease_owner TEXT, lease_until REAL, created_at REAL NOT NULL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS point_detection_claim_idx ON point_detection_requests(status, created_at);
CREATE TABLE IF NOT EXISTS point_detection_state (id INTEGER PRIMARY KEY CHECK(id=1), heartbeat REAL NOT NULL DEFAULT 0, device TEXT, error TEXT);
INSERT OR IGNORE INTO point_detection_state(id) VALUES (1);
"""


def init_schema(conn):
    conn.executescript(SCHEMA)


# --- choosing the box (pure) ---------------------------------------------------------------------------------

def contains(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def pick(found, point, existing, label='full'):
    """The detection under ``point``: ``found`` is ``[(box, score)]``. A box that is mostly a person already in the
    frame is skipped; the most confident wins and, when two are close, the smaller (the person, not a group)."""
    candidates = [(box, score) for box, score in found
                  if contains(box, point) and not any(iou(box, other) >= SAME_PERSON_IOU for other in existing)]
    if not candidates:
        return None
    best = max(score for _, score in candidates)
    box, score = min((item for item in candidates if item[1] >= best - TIE),
                     key=lambda item: ((item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), -item[1]))
    return {'box': list(box), 'score': round(score, 3), 'pass': label}


def crop_around(point, size):
    """A square around the point, kept inside the frame: ``[x1, y1, x2, y2]`` in whole pixels."""
    width, height = size
    side = min(width, height, max(CROP_MIN, round(CROP_SHARE * min(width, height))))
    x1 = min(max(0, round(point[0] - side / 2)), width - side)
    y1 = min(max(0, round(point[1] - side / 2)), height - side)
    return [x1, y1, x1 + side, y1 + side]


def uncrop(box, crop):
    return [box[0] + crop[0], box[1] + crop[1], box[2] + crop[0], box[3] + crop[1]]


def clamp(box, size):
    x1, y1, x2, y2 = (round(value) for value in box)
    return [max(0, min(size[0], x1)), max(0, min(size[1], y1)), max(0, min(size[0], x2)), max(0, min(size[1], y2))]


def detect(image, point, existing, detector):
    """``{box, score, pass}`` for the person under ``point`` in a PIL image, or None."""
    size = list(image.size)
    found = detector.detect([image], confidence=CONFIDENCE, scores=True)[0]
    chosen = pick(found, point, existing, 'full')
    if not chosen:
        crop = crop_around(point, size)
        found = detector.detect([image.crop(tuple(crop))], confidence=CONFIDENCE, scores=True)[0]
        chosen = pick([(uncrop(box, crop), score) for box, score in found], point, existing, 'crop')
    if not chosen:
        return None
    box = clamp(chosen['box'], size)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    return {**chosen, 'box': box}


# --- the worker thread ---------------------------------------------------------------------------------------

def claim():
    now = time.time()
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('''UPDATE point_detection_requests SET status='failed',error='The inference worker restarted while detecting',
                        lease_owner=NULL,lease_until=NULL,finished_at=? WHERE status='running' AND (lease_until IS NULL OR lease_until<=?)''', (now, now))
        row = conn.execute("SELECT * FROM point_detection_requests WHERE status='queued' ORDER BY created_at,rowid LIMIT 1").fetchone()
        if not row:
            return None
        owner = str(uuid.uuid4())
        conn.execute("UPDATE point_detection_requests SET status='running',lease_owner=?,lease_until=? WHERE id=?",
                     (owner, now + LEASE_SECONDS, row['id']))
        return dict(row), owner


def frame_image(row):
    from PIL import Image
    spec = json.loads(row['spec_json'])
    picture = spec['registration_image'] if row['surface'] == 'registration' else io.BytesIO(frame_cache.render(spec))
    with Image.open(picture) as source:
        image = source.convert('RGB')
    size = tuple(json.loads(row['image_size_json']))
    # Boxes are stored in the frame's detection size; a rounding difference in the render must not shift them.
    return image if image.size == size else image.resize(size)


def process(row, owner, detector_factory, render=frame_image):
    """Answer one claimed request; the row is written only while this thread still holds it."""
    status, found, error = 'done', None, None
    try:
        found = detect(render(row), json.loads(row['point_json']), json.loads(row['existing_json']), detector_factory())
    except Exception as exc:
        log.exception('Point detection %s failed', row['id'])
        status, error = 'failed', f'{type(exc).__name__}: {exc}'[:1000]
    with db.connect() as conn:
        conn.execute('''UPDATE point_detection_requests SET status=?,box_json=?,score=?,pass=?,error=?,lease_owner=NULL,lease_until=NULL,finished_at=?
                        WHERE id=? AND lease_owner=?''',
                     (status, json.dumps(found['box']) if found else None, found and found['score'], found and found['pass'],
                      error, time.time(), row['id'], owner))


class Detector:
    """One CPU AniRef for the thread's life, loaded on the first request."""

    def __init__(self):
        self.device = os.environ.get('POINT_DETECT_DEVICE', 'cpu')
        self.model = None

    def __call__(self):
        if self.model is None:
            from . import anime_models
            self.model = anime_models.AniRefDetector(os.environ.get('ANIREF_MODEL_PATH', '/models/aniref/AniRef40000-s-epoch40.pt'), self.device)
        return self.model


def loop(stop_event):
    detector = Detector()
    last_beat = 0.0
    while not stop_event.is_set():
        try:
            if time.monotonic() - last_beat > BEAT_SECONDS:
                with db.connect() as conn:
                    conn.execute('UPDATE point_detection_state SET heartbeat=?,device=? WHERE id=1', (time.time(), detector.device))
                last_beat = time.monotonic()
            claimed = claim()
            if claimed:
                process(*claimed, detector)
            else:
                stop_event.wait(POLL_SECONDS)
        except Exception:
            log.exception('Point detection worker error')
            stop_event.wait(2)


def start():
    stop_event = threading.Event()
    threading.Thread(target=loop, args=(stop_event,), daemon=True, name='point-detect').start()
    return stop_event
