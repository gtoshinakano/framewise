"""``POST /api/point-detections``: the box of the person under a double-clicked point (answered by ``point_detect``)."""
import asyncio
import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import db, media
from .target_characters_api import _frame as target_frame, _frame_boxes, _slots

router = APIRouter()
WAIT_SECONDS = 30  # the first request also loads the detector
STALE_HEARTBEAT = 20
KEEP_SECONDS = 600


class PointDetection(BaseModel):
    surface: Literal['first', 'target']
    run_id: str
    position: Literal['first', 'last'] | None = None
    frame_ordinal: int | None = None
    point: tuple[float, float]


def first_frame_spec(source, run, frame, width=1280):
    """The picture ``/api/character-detection-runs/{id}/frame/{position}`` shows and the boxes are drawn on."""
    return {'source': source, 'time': frame['time'], 'start': run['cut_start'], 'end': run['cut_end'], 'width': width, 'last': False, 'seek_tolerance': .000002}


def _original(conn, cut_id):
    row = conn.execute('SELECT e.id,e.project_id,e.original_ext FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=?', (cut_id,)).fetchone()
    source = media.episode_dir(row['project_id'], row['id']) / f"original{row['original_ext']}" if row else None
    if not source or not source.is_file():
        raise HTTPException(404, 'Frame unavailable')
    return source


def _request(conn, body):
    """``(frame_key, spec, image_size, existing boxes)`` of the frame the user double-clicked."""
    if body.surface == 'first':
        run = conn.execute('SELECT * FROM character_detection_runs WHERE id=?', (body.run_id,)).fetchone()
        if not run:
            raise HTTPException(404, 'Detection run not found')
        frame = next((f for f in json.loads(run['frames_json']) if f['position'] == (body.position or 'first')), None)
        if not frame:
            raise HTTPException(400, 'Frame not found')
        existing = [json.loads(row['box_json']) for row in conn.execute(
            'SELECT box_json FROM character_detections WHERE run_id=? AND position=?', (run['id'], frame['position']))]
        return frame['position'], first_frame_spec(_original(conn, run['cut_id']), run, frame, frame['image_size'][0]), frame['image_size'], existing
    run = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (body.run_id,)).fetchone()
    if not run:
        raise HTTPException(404, 'Detection run not found')
    if body.frame_ordinal is None:
        raise HTTPException(400, 'Frame not found')
    frame = target_frame(run, body.frame_ordinal)
    hidden = {number for number, slot in _slots(conn, run['id']).items() if slot['ignored']}
    existing = [json.loads(row['box_json']) for row in _frame_boxes(conn, run['id'], frame['ordinal']) if row['slot'] not in hidden]
    # The frame the target job detected on (target_characters._extract_frames), at the size its boxes use.
    spec = {'source': _original(conn, run['cut_id']), 'time': frame['time'], 'start': run['cut_start'], 'end': run['cut_end'],
            'width': frame['image_size'][0], 'last': False, 'ordinal': frame['ordinal']}
    return str(frame['ordinal']), spec, frame['image_size'], existing


@router.post('/api/point-detections')
async def detect_at_point(body: PointDetection):
    """Find the person under ``point`` (frame pixels). Returns ``{box, score, pass}``, ``box`` null when nobody is there.
    Nothing is saved: the caller adds the box through the run's ordinary add-box endpoint."""
    now = time.time()
    request_id = str(uuid.uuid4())
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        state = conn.execute('SELECT heartbeat FROM point_detection_state WHERE id=1').fetchone()
        if not state or now - state['heartbeat'] > STALE_HEARTBEAT:
            raise HTTPException(503, 'The inference worker is not running, so the box cannot be found; draw it instead')
        frame_key, spec, size, existing = _request(conn, body)
        x, y = body.point
        if not (0 <= x <= size[0] and 0 <= y <= size[1]):
            raise HTTPException(400, 'The point is outside the frame')
        conn.execute("DELETE FROM point_detection_requests WHERE status IN ('done','failed') AND finished_at<?", (now - KEEP_SECONDS,))
        conn.execute('''INSERT INTO point_detection_requests(id,surface,run_id,frame_key,point_json,spec_json,image_size_json,existing_json,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?)''',
                     (request_id, body.surface, body.run_id, frame_key, json.dumps([x, y]), json.dumps({**spec, 'source': str(spec['source'])}),
                      json.dumps(size), json.dumps(existing), now))
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(.1)
        row = db.one('SELECT status,box_json,score,pass,error FROM point_detection_requests WHERE id=?', (request_id,))
        if row['status'] == 'done':
            return {'box': json.loads(row['box_json']) if row['box_json'] else None, 'score': row['score'], 'pass': row['pass']}
        if row['status'] == 'failed':
            raise HTTPException(502, row['error'] or 'Detection failed')
    with db.connect() as conn:
        conn.execute("UPDATE point_detection_requests SET status='failed',error='Timed out',lease_owner=NULL,finished_at=? WHERE id=? AND status IN ('queued','running')",
                     (time.time(), request_id))
    raise HTTPException(504, 'The detector did not answer in time; draw the box instead')
