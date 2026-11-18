"""Queue, read and edit the people on a cut's evaluated frames (see ``target_characters``)."""
import json
import math
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import db, target_characters, target_tracking, worker

router = APIRouter()
PENDING = "SELECT 1 FROM jobs WHERE pipeline='character_detect' AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?"


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


@router.get('/api/cuts/{cut_id}/target-detections')
def get_target_detections(cut_id: str):
    result = target_characters.state(cut_id)
    if result is None:
        raise HTTPException(404, 'Active cut not found')
    return result


def _queue(episode_id, cut_id=None, from_cut=None, only_missing=True):
    """One job per cut with a current evaluation. A single cut is explicit; a batch skips cuts the user left out."""
    if not db.one('SELECT id FROM episodes WHERE id=?', (episode_id,)):
        raise HTTPException(404, 'Episode not found')
    first_index = max(0, from_cut - 1) if from_cut else 0
    cuts = db.all_rows('''SELECT id,episode_id,cut_index,title,start_time,end_time,analysis_enabled,no_characters FROM cuts
                          WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR id=?) ORDER BY cut_index''',
                       (episode_id, first_index, cut_id, cut_id))
    if cut_id and not cuts:
        raise HTTPException(404, 'Active cut not found')
    skipped = {'not_evaluated': [], 'analysis_disabled': [], 'no_characters': [], 'up_to_date': [], 'already_queued': []}
    ready = []
    for cut in cuts:
        number = cut['cut_index'] + 1
        if not cut_id and not cut['analysis_enabled']:
            skipped['analysis_disabled'].append(number)
            continue
        if not cut_id and cut['no_characters']:
            skipped['no_characters'].append(number)
            continue
        if target_characters.pending_job(cut['id']):
            if cut_id:
                raise HTTPException(409, 'Character detection is already queued for this cut')
            skipped['already_queued'].append(number)
            continue
        _, _, key, _ = target_characters.current_selection(cut['id'])
        if not key:
            if cut_id:
                raise HTTPException(409, 'Evaluate this cut first: it has no current target frames')
            skipped['not_evaluated'].append(number)
            continue
        run = target_characters.run_for(cut['id'], key)
        if only_missing and not cut_id and run and run['cut_start'] == cut['start_time'] and run['cut_end'] == cut['end_time']:
            skipped['up_to_date'].append(number)
            continue
        ready.append((cut, key))
    job_ids = []
    if ready:
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline!='character_detect' AND status IN ('queued','running')", (episode_id,)).fetchone():
                raise HTTPException(409, 'Wait for the current episode job to finish')
            batch_id = str(uuid.uuid4())
            for cut, key in ready:
                if conn.execute(PENDING, (cut['id'],)).fetchone():
                    skipped['already_queued'].append(cut['cut_index'] + 1)
                    continue
                snapshot = {name: cut[name] for name in ('id', 'episode_id', 'cut_index', 'title', 'start_time', 'end_time')}
                payload = {'cut_id': cut['id'], 'cut': snapshot, 'workflow': target_characters.WORKFLOW, 'batch_id': batch_id, 'selection_key': key}
                job_id = str(uuid.uuid4())
                conn.execute('INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,?,?)', (job_id, episode_id, 'character_detect', json.dumps(payload)))
                job_ids.append(job_id)
            if job_ids:
                worker.sync_episode_status(conn, episode_id)
    return {'queued': len(job_ids), 'job_ids': job_ids, 'skipped': skipped}


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
    """Queue every evaluated cut (from cut ``from_cut``, the number shown in the editor). Cuts already done are skipped."""
    return _queue(episode_id, from_cut=from_cut, only_missing=only_missing)


# --- editing ---------------------------------------------------------------------------------------------------

def _editable_run(conn, run_id):
    run = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, 'Detection run not found')
    if conn.execute(PENDING, (run['cut_id'],)).fetchone():
        raise HTTPException(409, 'Wait for detection to finish before editing')
    return run


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
        conn.execute('UPDATE target_slots SET character_context_id=?,label_source=? WHERE run_id=? AND slot=?',
                     (character_id, 'manual' if character_id else None, run_id, slot))
    return {'ok': True}


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
        if slot not in slots or body.into not in slots:
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
    return {'moved': moved, 'dropped': dropped}


@router.post('/api/target-runs/{run_id}/boxes')
def add_box(run_id: str, body: TargetBoxCreate):
    """Draw a box the detector missed. It takes the first colour not used in that frame unless ``slot`` says which."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = _editable_run(conn, run_id)
        _check_box(_frame(run, body.frame_ordinal), body.box)
        used = {row['slot'] for row in conn.execute('SELECT slot FROM target_detections WHERE run_id=? AND frame_ordinal=?', (run_id, body.frame_ordinal))}
        if len(used) >= target_characters.MAX_BOXES:
            raise HTTPException(409, f'A frame holds at most {target_characters.MAX_BOXES} boxes')
        slots = [row['slot'] for row in conn.execute('SELECT slot FROM target_slots WHERE run_id=? ORDER BY slot', (run_id,))]
        created = False
        if body.slot is not None:
            if body.slot not in slots:
                raise HTTPException(404, 'Slot not found')
            if body.slot in used:
                raise HTTPException(409, 'That colour already has a box in this frame')
            slot = body.slot
        else:
            slot = next((number for number in slots if number not in used), None)
            if slot is None:
                slot = next((number for number in range(target_tracking.MAX_SLOTS) if number not in slots), None)
                if slot is None:
                    raise HTTPException(409, 'Every colour is already in use')
                conn.execute("INSERT INTO target_slots(run_id,slot,origin) VALUES (?,?,'appeared')", (run_id, slot))
                created = True
        box_id = str(uuid.uuid4())
        conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json,source) VALUES (?,?,?,?,?,'manual')",
                     (box_id, run_id, body.frame_ordinal, slot, json.dumps(list(body.box))))
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
    return {'ok': True}


@router.put('/api/target-detections/{detection_id}/slot')
def move_box_to_slot(detection_id: str, body: TargetBoxSlot):
    """Say this box is another person. If that colour already has a box in the frame, the two swap."""
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = _box_row(conn, detection_id)
        _frame(_editable_run(conn, row['run_id']), row['frame_ordinal'])
        if not conn.execute('SELECT 1 FROM target_slots WHERE run_id=? AND slot=?', (row['run_id'], body.slot)).fetchone():
            raise HTTPException(404, 'Slot not found')
        other = conn.execute('SELECT id FROM target_detections WHERE run_id=? AND frame_ordinal=? AND slot=? AND id!=?',
                             (row['run_id'], row['frame_ordinal'], body.slot, detection_id)).fetchone()
        if other:
            conn.execute("UPDATE target_detections SET slot=?,source='manual' WHERE id=?", (row['slot'], other['id']))
        conn.execute("UPDATE target_detections SET slot=?,source='manual' WHERE id=?", (body.slot, detection_id))
    return {'ok': True}


@router.delete('/api/target-detections/{detection_id}')
def delete_box(detection_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = _box_row(conn, detection_id)
        _frame(_editable_run(conn, row['run_id']), row['frame_ordinal'])
        conn.execute('DELETE FROM target_detections WHERE id=?', (detection_id,))
    return {'ok': True}
