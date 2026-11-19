"""Queue, read and edit the people on a cut's evaluated frames (see ``target_characters``)."""
import json
import math
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import db, episode_requests, keyframes, structured_results, target_characters, target_reviews, target_tracking, worker

router = APIRouter()


class TargetSlotLabel(BaseModel):
    character_context_id: str | None = None


class TargetSlotMerge(BaseModel):
    into: int


class TargetBoxCreate(BaseModel):
    frame_ordinal: int
    box: tuple[float, float, float, float]
    slot: int | None = None


class TargetBoxUpdate(BaseModel):
    box: tuple[float, float, float, float]


class TargetBoxSlot(BaseModel):
    slot: int


class CastLock(BaseModel):
    locked: bool


class SlotIgnored(BaseModel):
    ignored: bool


class FrameReview(BaseModel):
    reviewed: bool


class SwapFrom(BaseModel):
    other: int
    frame_ordinal: int


class CandidateDecision(BaseModel):
    action: Literal['promote', 'split', 'reject']
    slot: int | None = None


@router.get('/api/cuts/{cut_id}/target-detections')
def get_target_detections(cut_id: str):
    result = target_characters.state(cut_id)
    if result is None:
        raise HTTPException(404, 'Active cut not found')
    return result


def queue_targets(conn, episode_id, cuts, *, cut_id=None, only_missing=True):
    skipped = {'not_evaluated': [], 'analysis_disabled': [], 'no_characters': [], 'up_to_date': [], 'already_queued': []}
    job_ids = []
    for cut in cuts:
        cid = cut['id']
        if not cut_id and (not cut['analysis_enabled'] or cut['no_characters']):
            reason = 'analysis_disabled' if not cut['analysis_enabled'] else 'no_characters'
            skipped[reason].append(cut['cut_index'] + 1)
            continue
        deps = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts'), [cid])
        deps.extend(target_characters.pending_opening_detection(conn, episode_id, cid))
        deps.extend(keyframes.ensure_selection(conn, episode_id, [cid]))
        payload = {'cut_id': cid, 'workflow': target_characters.WORKFLOW,
                   **({'episode_batch': True, 'only_missing': only_missing} if cut_id is None else {})}
        if deps:
            payload['depends_on'] = list(dict.fromkeys(deps))
        job_ids.append(worker.insert_job(conn, episode_id, 'character_detect', payload))
    return {'queued': len(job_ids), 'job_ids': job_ids, 'skipped': skipped}


def _queue(episode_id, cut_id=None, from_cut=None, only_missing=True):
    """Queue one request per cut. Each worker reads the current selection and anchors."""
    first_index = max(0, (from_cut or 1) - 1)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM episodes WHERE id=?', (episode_id,)).fetchone():
            raise HTTPException(404, 'Episode not found')
        dispatch = None if cut_id else episode_requests.defer(conn, episode_id, 'target_detection',
                                                               {'first_index': first_index, 'only_missing': only_missing})
        if dispatch:
            result = {'queued': 0, 'job_ids': [dispatch], 'skipped': {}, 'deferred': True}
        else:
            cuts = episode_requests.cuts(conn, episode_id, first_index, cut_id)
            if cut_id and not cuts:
                raise HTTPException(404, 'Active cut not found')
            result = queue_targets(conn, episode_id, cuts, cut_id=cut_id, only_missing=only_missing)
        if result['job_ids']:
            worker.sync_episode_status(conn, episode_id)
    worker.wake_event.set()
    return result


@router.post('/api/cuts/{cut_id}/target-detection')
def queue_cut(cut_id: str):
    """Find the people on this cut's evaluated frames. Detecting again keeps the user's labels and boxes."""
    cut = db.one('SELECT episode_id FROM cuts WHERE id=? AND active=1', (cut_id,))
    if not cut:
        raise HTTPException(404, 'Active cut not found')
    result = _queue(cut['episode_id'], cut_id=cut_id)
    return {'job_id': result['job_ids'][0], 'queued': 1}


@router.post('/api/episodes/{episode_id}/target-detection')
def queue_episode(episode_id: str, from_cut: int | None = None, only_missing: bool = True):
    """Queue enabled cuts from the given timeline number and evaluate missing target frames as needed."""
    return _queue(episode_id, from_cut=from_cut, only_missing=only_missing)


# --- editing ---------------------------------------------------------------------------------------------------

def _editable_run(conn, run_id):
    run = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, 'Detection run not found')
    structured_results.invalidate_detections(conn, run['cut_id'], target=True, reason='Target frame edits changed')
    return run


@router.put('/api/cuts/{cut_id}/target-cast')
def set_cast_lock(cut_id: str, body: CastLock):
    """A cut's explicit cast choice, used the next time its target frames are detected."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM cuts WHERE id=? AND active=1', (cut_id,)).fetchone():
            raise HTTPException(404, 'Active cut not found')
        conn.execute('''INSERT INTO target_cast_settings(cut_id,locked) VALUES (?,?)
                        ON CONFLICT(cut_id) DO UPDATE SET locked=excluded.locked''', (cut_id, body.locked))
    return {'locked': body.locked}


def _slots(conn, run_id):
    return {row['slot']: dict(row) for row in conn.execute('SELECT * FROM target_slots WHERE run_id=? ORDER BY slot', (run_id,))}


def _frame_boxes(conn, run_id, ordinal):
    return conn.execute('SELECT * FROM target_detections WHERE run_id=? AND frame_ordinal=? ORDER BY slot', (run_id, ordinal)).fetchall()


def _manual_box(conn, run_id, ordinal, slot, box):
    box_id = str(uuid.uuid4())
    conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json,source) VALUES (?,?,?,?,?,'manual')",
                 (box_id, run_id, ordinal, slot, json.dumps(list(box))))
    return box_id


def _check_box(frame, box):
    width, height = frame['image_size']
    x1, y1, x2, y2 = box
    if not all(math.isfinite(value) for value in box) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise HTTPException(400, 'Box must fit inside the frame')


def _frame(run, ordinal, editing=True):
    frame = next((item for item in json.loads(run['frames_json']) if item['ordinal'] == ordinal), None)
    if not frame:
        raise HTTPException(400, 'Frame not found')
    if editing and frame.get('mirror'):
        raise HTTPException(409, "This frame is the first frame as you left it. Change its boxes in Characters in this cut, then detect again.")
    return frame


@router.put('/api/target-slots/{run_id}/{slot}')
def label_slot(run_id: str, slot: int, body: TargetSlotLabel):
    """Name a person once: every box of this colour in the cut's evaluated frames follows."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        if not conn.execute('SELECT 1 FROM target_slots WHERE run_id=? AND slot=?', (run_id, slot)).fetchone():
            raise HTTPException(404, 'Slot not found')
        character_id = body.character_context_id
        if character_id and not conn.execute('''SELECT 1 FROM context_items ci JOIN cuts c ON c.id=? JOIN episodes e ON e.id=c.episode_id
                                                WHERE ci.id=? AND ci.kind='character' AND ci.project_id=e.project_id''', (run['cut_id'], character_id)).fetchone():
            raise HTTPException(400, 'Character not found in this project')
        conn.execute("UPDATE target_slots SET character_context_id=?,label_source='manual' WHERE run_id=? AND slot=?",
                     (character_id, run_id, slot))
    return {'ok': True}


@router.put('/api/target-slots/{run_id}/{slot}/ignored')
def ignore_slot(run_id: str, slot: int, body: SlotIgnored):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        source = _slots(conn, run_id).get(slot)
        if not source:
            raise HTTPException(404, 'Slot not found')
        if bool(source['ignored']) != body.ignored:
            boxes = [target_reviews.box_row(row) for row in conn.execute('SELECT * FROM target_detections WHERE run_id=? AND slot=?', (run_id, slot))]
            conn.execute('UPDATE target_slots SET ignored=? WHERE run_id=? AND slot=?', (body.ignored, run_id, slot))
            target_reviews.record(conn, run, 'ignore' if body.ignored else 'restore', boxes if body.ignored else (),
                                  () if body.ignored else boxes, slot=slot)
    return {'ok': True}


@router.post('/api/target-slots/{run_id}/{slot}/swap-from')
def swap_slots_from_frame(run_id: str, slot: int, body: SwapFrom):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        _frame(run, body.frame_ordinal)
        slots = _slots(conn, run_id)
        if body.other == slot or any(number not in slots or slots[number]['ignored'] for number in (slot, body.other)):
            raise HTTPException(400, 'Pick two visible colours')
        rows = conn.execute('''SELECT * FROM target_detections WHERE run_id=? AND frame_ordinal>=? AND slot IN (?,?)''',
                            (run_id, body.frame_ordinal, slot, body.other)).fetchall()
        if not rows:
            raise HTTPException(409, 'These colours have no boxes from this frame onward')
        before = [target_reviews.box_row(row) for row in rows]
        after = [{**row, 'slot': body.other if row['slot'] == slot else slot, 'source': 'manual'} for row in before]
        conn.executemany("UPDATE target_detections SET slot=?,source='manual' WHERE id=?", ((row['slot'], row['id']) for row in after))
        target_reviews.record(conn, run, 'swap', before, after, from_ordinal=body.frame_ordinal, slots=[slot, body.other])
        target_reviews.update_frames(conn, run, {row['frame_ordinal'] for row in after})
    return {'swapped': len(after)}


@router.post('/api/target-slots/{run_id}/{slot}/merge')
def merge_slot(run_id: str, slot: int, body: TargetSlotMerge):
    """One colour was really another: every box of ``slot`` joins colour ``into`` and ``slot`` is removed.

    A person is in a frame once, so where both colours have a box the one the user drew or moved wins, else
    ``into``'s stays. The moved boxes count as the user's, so detecting again keeps the merge. ``into`` keeps its
    name; without one it takes ``slot``'s. A colour that has a box on the mirrored first frame is a separate
    person there and cannot be merged away.
    """
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        if body.into == slot:
            raise HTTPException(400, 'Pick another colour to merge into')
        slots = {row['slot']: row for row in conn.execute('SELECT * FROM target_slots WHERE run_id=? AND slot IN (?,?)', (run_id, slot, body.into))}
        if slot not in slots or body.into not in slots or slots[slot]['ignored'] or slots[body.into]['ignored']:
            raise HTTPException(404, 'Slot not found')
        mirrored = {frame['ordinal'] for frame in json.loads(run['frames_json']) if frame.get('mirror')}
        boxes = conn.execute('SELECT * FROM target_detections WHERE run_id=? AND slot IN (?,?)', (run_id, slot, body.into)).fetchall()
        mine = [box for box in boxes if box['slot'] == slot]
        theirs = {box['frame_ordinal']: box for box in boxes if box['slot'] == body.into}
        if any(box['frame_ordinal'] in mirrored for box in mine):
            raise HTTPException(409, 'This colour is in the first frame as a different person. Change the first frame in Characters in this cut, then detect again.')
        moved = dropped = 0
        for box in mine:
            other = theirs.get(box['frame_ordinal'])
            if other and not (box['source'] == 'manual' and other['source'] != 'manual'):
                conn.execute('DELETE FROM target_detections WHERE id=?', (box['id'],))
                dropped += 1
                continue
            if other:
                conn.execute('DELETE FROM target_detections WHERE id=?', (other['id'],))
                dropped += 1
            conn.execute("UPDATE target_detections SET slot=?,source='manual' WHERE id=?", (body.into, box['id']))
            moved += 1
        if slots[slot]['character_context_id'] and not slots[body.into]['character_context_id']:
            conn.execute('UPDATE target_slots SET character_context_id=?,label_source=? WHERE run_id=? AND slot=?',
                         (slots[slot]['character_context_id'], slots[slot]['label_source'], run_id, body.into))
        conn.execute('DELETE FROM target_slots WHERE run_id=? AND slot=?', (run_id, slot))
        before = [target_reviews.box_row(box) for box in boxes]
        after = [target_reviews.box_row(box) for box in conn.execute('SELECT * FROM target_detections WHERE run_id=? AND slot=?', (run_id, body.into))]
        target_reviews.record(conn, run, 'merge', before, after, old_slot=slot, new_slot=body.into)
        target_reviews.update_frames(conn, run, {box['frame_ordinal'] for box in mine})
    return {'moved': moved, 'dropped': dropped}


@router.post('/api/target-runs/{run_id}/boxes')
def add_box(run_id: str, body: TargetBoxCreate):
    """Draw a box the detector missed. It takes the first colour not used in that frame unless ``slot`` says which."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        _check_box(_frame(run, body.frame_ordinal), body.box)
        slots = _slots(conn, run_id)
        visible = {number for number, row in slots.items() if not row['ignored']}
        used = {row['slot'] for row in _frame_boxes(conn, run_id, body.frame_ordinal) if row['slot'] in visible}
        limit = target_characters.frame_capacity(slots.values(), bool(run['cast_locked']))
        if len(used) >= limit:
            raise HTTPException(409, f'A frame holds at most {limit} boxes')
        created = False
        if body.slot is not None:
            if body.slot not in visible:
                raise HTTPException(404, 'Slot not found')
            if body.slot in used:
                raise HTTPException(409, 'That colour already has a box in this frame')
            slot = body.slot
        else:
            slot = next((number for number in slots if number in visible and number not in used), None)
            if slot is None:
                slot = next((number for number in range(target_tracking.MAX_SLOTS) if number not in slots), None)
                if slot is None:
                    raise HTTPException(409, 'Every colour is already in use')
                conn.execute("INSERT INTO target_slots(run_id,slot,origin) VALUES (?,?,'appeared')", (run_id, slot))
                created = True
        box_id = _manual_box(conn, run_id, body.frame_ordinal, slot, body.box)
        target_reviews.record(conn, run, 'add', after=[target_reviews.box_row(_box_row(conn, box_id))])
        target_reviews.update_frames(conn, run, {body.frame_ordinal})
    return {'id': box_id, 'slot': slot, 'created_slot': created}


def _box_row(conn, detection_id):
    row = conn.execute('SELECT * FROM target_detections WHERE id=?', (detection_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Box not found')
    return row


@router.put('/api/target-detections/{detection_id}/box')
def move_box(detection_id: str, body: TargetBoxUpdate):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = _box_row(conn, detection_id)
        run = _editable_run(conn, row['run_id'])
        _check_box(_frame(run, row['frame_ordinal']), body.box)
        conn.execute("UPDATE target_detections SET box_json=?,source='manual' WHERE id=?", (json.dumps(list(body.box)), detection_id))
        target_reviews.record(conn, run, 'move', [target_reviews.box_row(row)], [target_reviews.box_row(_box_row(conn, detection_id))])
        target_reviews.update_frames(conn, run, {row['frame_ordinal']})
    return {'ok': True}


@router.put('/api/target-detections/{detection_id}/slot')
def move_box_to_slot(detection_id: str, body: TargetBoxSlot):
    """Say this box is another person. If that colour already has a box in the frame, the two swap."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = _box_row(conn, detection_id)
        run = _editable_run(conn, row['run_id'])
        _frame(run, row['frame_ordinal'])
        if not conn.execute('SELECT 1 FROM target_slots WHERE run_id=? AND slot=? AND ignored=0', (row['run_id'], body.slot)).fetchone():
            raise HTTPException(404, 'Slot not found')
        other = conn.execute('SELECT * FROM target_detections WHERE run_id=? AND frame_ordinal=? AND slot=? AND id!=?',
                             (row['run_id'], row['frame_ordinal'], body.slot, detection_id)).fetchone()
        if other:
            conn.execute("UPDATE target_detections SET slot=?,source='manual' WHERE id=?", (row['slot'], other['id']))
        conn.execute("UPDATE target_detections SET slot=?,source='manual' WHERE id=?", (body.slot, detection_id))
        before = [target_reviews.box_row(box) for box in (row, other) if box]
        target_reviews.record(conn, run, 'swap' if other else 'recolour', before,
                              [target_reviews.box_row(_box_row(conn, box['id'])) for box in before])
        target_reviews.update_frames(conn, run, {row['frame_ordinal']})
    return {'ok': True}


@router.delete('/api/target-detections/{detection_id}')
def delete_box(detection_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = _box_row(conn, detection_id)
        run = _editable_run(conn, row['run_id'])
        _frame(run, row['frame_ordinal'])
        conn.execute('DELETE FROM target_detections WHERE id=?', (detection_id,))
        target_reviews.record(conn, run, 'remove', [target_reviews.box_row(row)])
        target_reviews.update_frames(conn, run, {row['frame_ordinal']})
    return {'ok': True}


@router.put('/api/target-runs/{run_id}/frames/{ordinal}/reviewed')
def review_frame(run_id: str, ordinal: int, body: FrameReview):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        _frame(run, ordinal)
        target_reviews.update_frames(conn, run, {ordinal}, reviewed=body.reviewed)
    return {'ok': True}


def _promote_candidate(conn, run, candidate, row, slots, boxes, requested):
    taken = {box['slot'] for box in boxes}
    visible = {number for number, slot in slots.items() if not slot['ignored']}
    limit = target_characters.frame_capacity(slots.values(), bool(run['cast_locked']))
    if len(taken & visible) >= limit:
        raise HTTPException(409, f'This frame already has {limit} boxes')
    number = requested if requested is not None else candidate.get('suggested_slot')
    if number is None:
        number = next((i for i in range(target_tracking.MAX_SLOTS) if i not in slots), None)
        if number is None:
            raise HTTPException(409, 'Every colour is in use. Choose an existing colour.')
        conn.execute("INSERT INTO target_slots(run_id,slot,origin) VALUES (?,?,'appeared')", (run['id'], number))
    elif number not in visible or number in taken:
        raise HTTPException(409, 'Choose a visible colour without a box in this frame')
    _check_box(row, candidate['box'])
    return [], [_manual_box(conn, run['id'], candidate['frame_ordinal'], number, candidate['box'])]


def _accept_split(conn, run, candidate, frame, slots, boxes):
    parts = candidate.get('parts') or []
    numbers = {part.get('slot_hint') for part in parts}
    if len(parts) != 2 or len(numbers) != 2 or any(number not in slots or slots[number]['ignored'] for number in numbers):
        raise HTTPException(409, 'The split needs two visible colours. You can draw the boxes by hand.')
    replaced = [box for box in boxes if box['slot'] in numbers or target_tracking.iou(json.loads(box['box_json']), candidate['box']) >= .8]
    if any(box['source'] == 'manual' or slots[box['slot']]['ignored'] for box in replaced):
        raise HTTPException(409, 'This split overlaps a box you edited. Edit the frame by hand to keep your changes.')
    limit = target_characters.frame_capacity(slots.values(), bool(run['cast_locked']))
    if sum(not slots[box['slot']]['ignored'] for box in boxes) - len(replaced) + 2 > limit:
        raise HTTPException(409, f'This frame can hold at most {limit} boxes')
    for part in parts:
        _check_box(frame, part['box'])
    conn.executemany('DELETE FROM target_detections WHERE id=?', ((box['id'],) for box in replaced))
    ids = [_manual_box(conn, run['id'], candidate['frame_ordinal'], part['slot_hint'], part['box']) for part in parts]
    return [target_reviews.box_row(row) for row in replaced], ids


@router.post('/api/target-runs/{run_id}/candidates/{candidate_id}')
def decide_candidate(run_id: str, candidate_id: str, body: CandidateDecision):
    """Promote a crop or apply a geometric split with explicit permission for this frame."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        candidates = json.loads(run['candidates_json'])
        candidate = next((item for item in candidates if item['id'] == candidate_id), None)
        if not candidate:
            raise HTTPException(404, 'Candidate not found')
        ordinal = candidate['frame_ordinal']
        frame = _frame(run, ordinal)
        boxes, slots = _frame_boxes(conn, run_id, ordinal), _slots(conn, run_id)
        before, ids = [], []
        if body.action == 'promote' and candidate['tag'] == 'unmatched':
            before, ids = _promote_candidate(conn, run, candidate, frame, slots, boxes, body.slot)
        elif body.action == 'split' and candidate['tag'] == 'merge':
            before, ids = _accept_split(conn, run, candidate, frame, slots, boxes)
        elif body.action != 'reject':
            raise HTTPException(400, 'That action does not apply to this candidate')
        target_reviews.record(conn, run, body.action, [candidate, *before],
                              [target_reviews.box_row(_box_row(conn, box_id)) for box_id in ids])
        target_reviews.update_frames(conn, run, {ordinal}, candidates=[item for item in candidates if item['id'] != candidate_id])
    return {'ok': True, 'boxes': ids}
