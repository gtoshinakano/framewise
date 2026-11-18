"""People on the evaluated (target) frames of a cut.

The keyframe selector picks the frames that matter in a cut (``keyframes.py``). This module finds the people
on them, in the order of the first frame: slot 0 is the first frame's first box, and so on, so a colour is the
same person in every frame (``target_tracking``). The label belongs to the slot. A person who appears later
gets the next free slot, and a frame keeps at most three boxes.

Nothing here is a gallery example: the rows live in their own tables (``target_*``), the gallery only reads
the user's first-frame labels, and the recognition and composition code never see these boxes.

The selector always keeps the cut's first frame as its first target frame (``initial_state``). That frame is
not detected again: it shows the first-frame boxes and labels exactly as the user left them (a copy, made
whenever detection runs; it is read only on the tab, the first-frame editor is where it changes). Only the
frames after it are detected and followed.

Runs as a ``character_detect`` job with ``workflow: 'target_frames_v1'`` in the inference worker. Detection
again keeps what the user did on this tab: slot labels they set, and boxes they drew or moved. It saves
everything in one transaction at the end, so an interrupted job leaves the previous result alone.
"""
import json
import logging
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import anime_bulk, anime_gallery, anime_matching, db, frame_cache, media, target_tracking, worker

log = logging.getLogger('framewise.character_detect')
WORKFLOW = 'target_frames_v1'
FRAME_WIDTH = 1280
MAX_BOXES = target_tracking.MAX_BOXES


def is_target_job(payload):
    return isinstance(payload, dict) and payload.get('workflow') == WORKFLOW


def current_selection(cut_id):
    """``(cut, source path, key, result)`` of the cut's latest evaluation; key and result are None when it is missing or stale.

    The cut is None when it does not exist or is not active. The key is the ``keyframe_results`` cache key,
    which also names the evaluation a run was built for.
    """
    from . import keyframes
    cut = db.one('''SELECT c.*,e.project_id,e.original_ext FROM cuts c JOIN episodes e ON e.id=c.episode_id
                    WHERE c.id=? AND c.active=1''', (cut_id,))
    if not cut:
        return None, None, None, None
    source = media.episode_dir(cut['project_id'], cut['episode_id']) / f"original{cut['original_ext']}"
    job = db.one('''SELECT j.payload_json FROM jobs j WHERE j.episode_id=? AND j.pipeline='keyframes'
                    AND EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.cuts') item WHERE json_extract(item.value,'$.id')=?)
                    ORDER BY j.rowid DESC LIMIT 1''', (cut['episode_id'], cut_id))
    if not job:
        return cut, source, None, None
    settings = json.loads(job['payload_json'])['keyframe_settings']
    try:
        key = keyframes.cache_key(source, cut, settings, keyframes.mask_for(cut['episode_id'], settings))
    except OSError:
        return cut, source, None, None
    row = db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (key,))
    return (cut, source, key, json.loads(row['result_json'])) if row else (cut, source, None, None)


def pending_job(cut_id):
    return db.one("""SELECT id FROM jobs WHERE pipeline='character_detect' AND status IN ('queued','running')
                     AND json_extract(payload_json,'$.cut_id')=?""", (cut_id,))


def latest_job(cut_id):
    return db.one("""SELECT id,status,stage,error FROM jobs WHERE pipeline='character_detect'
                     AND json_extract(payload_json,'$.workflow')=? AND json_extract(payload_json,'$.cut_id')=?
                     ORDER BY rowid DESC LIMIT 1""", (WORKFLOW, cut_id))


def run_for(cut_id, key):
    return db.one('SELECT * FROM target_detection_runs WHERE cut_id=? AND selection_key=?', (cut_id, key))


def state(cut_id):
    """What the Target frames tab shows: the evaluation, the latest run with its slots and boxes, and the job."""
    cut, _, key, _ = current_selection(cut_id)
    if not cut:
        return None
    run = db.one('SELECT * FROM target_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut_id,))
    result = {'evaluation': 'current' if key else 'missing', 'selection_key': key, 'max_boxes': MAX_BOXES, 'run': None, 'job': latest_job(cut_id)}
    if run:
        slots = db.all_rows('''SELECT s.slot,s.origin,s.character_context_id,s.label_source,c.title AS character_name
                               FROM target_slots s LEFT JOIN context_items c ON c.id=s.character_context_id
                               WHERE s.run_id=? ORDER BY s.slot''', (run['id'],))
        boxes = db.all_rows('SELECT id,frame_ordinal,slot,box_json,source FROM target_detections WHERE run_id=? ORDER BY rowid', (run['id'],))
        result['run'] = {
            'id': run['id'], 'selection_key': run['selection_key'],
            'stale': key is None or run['selection_key'] != key or run['cut_start'] != cut['start_time'] or run['cut_end'] != cut['end_time'],
            'frames': json.loads(run['frames_json']), 'slots': slots,
            'detections': [{'id': box['id'], 'frame_ordinal': box['frame_ordinal'], 'slot': box['slot'], 'box': json.loads(box['box_json']), 'source': box['source']} for box in boxes],
        }
    return result


# --- the job -------------------------------------------------------------------------------------------------

def _extract_frames(source, cut, frames):
    """The evaluated frames as RGB images, extracted in parallel at detection size."""
    from PIL import Image
    import io

    def one(frame):
        spec = {'source': source, 'time': frame['timestamp_seconds'], 'start': cut['start_time'], 'end': cut['end_time'],
                'width': FRAME_WIDTH, 'last': False, 'ordinal': frame['source_frame_index']}
        with Image.open(io.BytesIO(frame_cache.render(spec))) as image:
            return image.convert('RGB')
    with ThreadPoolExecutor(anime_gallery.FRAME_THREADS) as pool:
        return list(pool.map(one, frames))


def _first_frame_image(original, cut, frame):
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='framewise-targets-') as scratch:
        target = Path(scratch) / 'first.jpg'
        media.extract_cut_frame(original, target, frame['time'], cut['start_time'], cut['end_time'], width=1280, last=False, seek_tolerance=.000002)
        with Image.open(target) as source:
            image = source.convert('RGB')
    if list(image.size) != frame['image_size']:
        raise RuntimeError('Detection frame dimensions changed; run character detection again')
    return image


def _first_frame(job_id, owner, original, cut, model):
    """The first frame as the user left it: ``{'people': [...], 'size': [w, h]}``, or None when it was never detected.

    One entry per box in the order the first-frame editor shows them (the slots), with the box exactly as
    saved and its identity vector. A reviewed frame without people is a frame with no entries, not None.
    """
    run = anime_bulk.latest_current_run(cut)
    frame = next((item for item in json.loads(run['frames_json']) if item['position'] == 'first'), None) if run else None
    if not run or not frame:
        return None
    # More than the palette has colours cannot be told apart; the editor's first boxes keep their numbers.
    rows = db.all_rows("SELECT * FROM character_detections WHERE run_id=? AND position='first' ORDER BY rowid", (run['id'],))[:target_tracking.MAX_SLOTS]
    people = []
    if rows:
        image = _first_frame_image(original, cut, frame)
        vectors = anime_bulk._target_vectors(job_id, owner, rows, [anime_gallery.crop_jpeg(image, json.loads(row['box_json'])) for row in rows], model)
        people = [{'row': row, 'box': json.loads(row['box_json']), 'vector': vector} for row, vector in zip(rows, vectors)]
    return {'people': people, 'size': list(frame['image_size'])}


def _seed_slots(first, existing_slots, existing_boxes):
    """Starting slots: the first frame's people, then the slots the user's own edits keep alive.

    Returns ``(slots, manual boxes, renumbered)``. A slot label the user set (``manual``) wins over the first
    frame's. A slot the first frame no longer claims survives only through the user's edits (a label they set
    or a box they drew), as an ``appeared`` slot; it keeps its number unless a first-frame slot needs it, and
    ``renumbered`` maps the old number to the new one for the boxes.
    """
    saved = {row['slot']: row for row in existing_slots}
    slots = []
    for number, person in enumerate(first):
        mine = saved.get(number)
        if mine and mine['origin'] == 'first_frame' and mine['label_source'] == 'manual':
            character_id, label = mine['character_context_id'], 'manual'
        else:
            character_id = person['row']['character_context_id']
            label = 'inherited' if character_id else None
        slots.append(target_tracking.Slot(number, 'first_frame', character_id, label, person['box'], person['vector']))
    manual = [dict(row) for row in existing_boxes if row['source'] == 'manual']
    renumbered = {}
    for row in sorted(existing_slots, key=lambda item: item['slot']):
        if row['origin'] == 'first_frame' and row['slot'] < len(first):
            continue  # the first frame still has this person
        kept_label = row['label_source'] == 'manual'
        if not kept_label and not any(box['slot'] == row['slot'] for box in manual):
            continue
        number = row['slot'] if not any(slot.number == row['slot'] for slot in slots) else target_tracking.free_number(slots)
        if number is None:
            continue
        renumbered[row['slot']] = number
        slots.append(target_tracking.Slot(number, 'appeared', row['character_context_id'] if kept_label else None, 'manual' if kept_label else None))
    numbers = {slot.number for slot in slots}
    kept = []
    for box in manual:
        if box['slot'] in renumbered or box['slot'] in numbers:
            box['slot'] = renumbered.get(box['slot'], box['slot'])
            box['box'] = json.loads(box['box_json'])
            kept.append(box)
    return slots, kept, renumbered


def _gallery_labels(runtime, episode, slots, lease_check, timings, model):
    """Name people who are not in the first frame by the user's labelled boxes (read only, never added to)."""
    todo = sorted((slot for slot in slots if slot.origin == 'appeared' and slot.label_source != 'manual' and slot.first is not None), key=lambda slot: slot.number)
    if not todo:
        return
    runtime.gallery = None  # labels confirmed since the last job count
    gallery = runtime.get_gallery(episode['project_id'], lease_check, timings)
    if not len(gallery):
        return
    defaults = model.defaults
    threshold = anime_matching.env_fraction('ANIME_REID_MIN_SIMILARITY', defaults['similarity'])
    margin = anime_matching.env_fraction('ANIME_REID_MIN_MARGIN', defaults['margin'])
    taken = {slot.character_id for slot in slots if slot.character_id}
    for slot in todo:
        ranked = gallery.rank(slot.first)
        character_id = anime_matching.confident_match(ranked, threshold, margin)
        if not character_id and defaults.get('wide'):
            character_id = anime_matching.confident_match(ranked, *defaults['wide'])
        if character_id and character_id not in taken:
            slot.character_id, slot.label_source = character_id, 'auto'
            taken.add(character_id)


def track(runtime, model, frames, images, slots, manual, lease_check, same_person, mirror=None):
    """Detect people on the frames after the first and place them in slots. Returns the stored boxes as ``(ordinal, slot, box)``.

    ``mirror`` is the first frame's boxes as ``(ordinal, slot, box)`` when the first evaluated frame is the first
    frame: they are stored as they are and that frame is neither detected nor edited.
    """
    detector = runtime.get_detector()
    skip = {0} if mirror is not None else set()
    todo = [number for number in range(len(images)) if number not in skip]
    answered = detector.detect([images[number] for number in todo]) if todo else []
    if len(answered) != len(todo):
        raise RuntimeError('AniRef returned an unexpected number of frames')
    all_boxes = [[] for _ in images]
    for number, boxes in zip(todo, answered):
        all_boxes[number] = boxes
    detections = [[{'box': box, 'small': bool(anime_gallery.box_problem(box, image.size)), 'vector': None} for box in boxes]
                  for boxes, image in zip(all_boxes, images)]
    index = {frame['source_frame_index']: number for number, frame in enumerate(frames)}
    manual_here = {number: [] for number in range(len(frames))}
    for box in manual:
        number = index.get(box['frame_ordinal'])
        if number is not None and number not in skip:
            manual_here[number].append({**box, 'small': bool(anime_gallery.box_problem(box['box'], images[number].size)), 'vector': None})
    owners, crops = [], []
    for number, image in enumerate(images):
        for item in [*detections[number], *manual_here[number]]:
            if not item['small']:
                owners.append(item)
                crops.append(anime_gallery.open_jpeg(anime_gallery.crop_jpeg(image, item['box'])))
    if crops:
        lease_check()
        vectors = model.embed(crops)
        if len(vectors) != len(crops):
            raise RuntimeError('Character embedder returned an unexpected number of target embeddings')
        for item, vector in zip(owners, vectors):
            item['vector'] = vector
    by_number = lambda: {slot.number: slot for slot in slots}
    stored = []
    for number, frame in enumerate(frames):
        lease_check()
        if number in skip:
            stored.extend(mirror)
            continue
        placed_by_user = manual_here[number]
        for item in placed_by_user:
            slot = by_number().get(item['slot'])
            if slot:
                slot.seen(item['box'], item['vector'])
        auto = [item for item in detections[number]
                if not any(target_tracking.iou(item['box'], other['box']) >= target_tracking.IOU_ACCEPT for other in placed_by_user)]
        placed = target_tracking.assign_frame(slots, auto, same_person, max_boxes=MAX_BOXES - len(placed_by_user),
                                              blocked={item['slot'] for item in placed_by_user})
        stored.extend((frame['source_frame_index'], item['slot'], auto[item['detection']]['box']) for item in placed)
    return stored


def process(job_id, owner, runtime):
    """Find and order the people on one cut's evaluated frames."""
    from . import inference_worker
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    try:
        payload = json.loads(job['payload_json'])
        cut = payload['cut']
        episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
        started = time.perf_counter()
        with db.connect() as conn:
            anime_bulk._valid_cut(conn, job_id, owner, episode['id'], cut)
        lease_check = lambda: inference_worker.check_summary_lease(job_id, owner)
        _, source, key, result = current_selection(cut['id'])
        if key is None or key != payload.get('selection_key'):
            raise RuntimeError('The evaluation of this cut changed; evaluate the cut and detect characters again')
        frames = result['selected_frames']
        timings = {}
        worker.stage(job_id, owner, episode['id'], 'Extracting evaluated frames')
        with anime_bulk.timed('target_frames', timings, cut=cut.get('cut_index'), frames=len(frames)):
            images = _extract_frames(source, cut, frames)
        lease_check()
        model = runtime.get_embedder()
        existing = run_for(cut['id'], key)
        existing_slots = db.all_rows('SELECT * FROM target_slots WHERE run_id=?', (existing['id'],)) if existing else []
        existing_boxes = db.all_rows('SELECT * FROM target_detections WHERE run_id=?', (existing['id'],)) if existing else []
        worker.stage(job_id, owner, episode['id'], 'Reading the first frame order')
        first = _first_frame(job_id, owner, source, cut, model)
        people = first['people'] if first else []
        # The selector always keeps the cut's first frame first. Once the user has reviewed that frame, its
        # evaluated copy is theirs as they left it: no detection, no tracking, and no edits of its own.
        mirrored = bool(first) and frames[0].get('reason') == 'initial_state'
        if mirrored and list(images[0].size) != first['size']:
            raise RuntimeError('Detection frame dimensions changed; run character detection again')
        ordinal = frames[0]['source_frame_index']
        if mirrored:
            existing_boxes = [box for box in existing_boxes if box['frame_ordinal'] != ordinal]
        mirror = [(ordinal, number, person['box']) for number, person in enumerate(people)] if mirrored else None
        slots, manual, renumbered = _seed_slots(people, existing_slots, existing_boxes)
        worker.stage(job_id, owner, episode['id'], 'Finding people on the evaluated frames')
        same_person = anime_matching.env_fraction('TARGET_SAME_PERSON_SIMILARITY', model.defaults['track'])
        with anime_bulk.timed('target_detect', timings, cut=cut.get('cut_index'), frames=len(frames)):
            stored = track(runtime, model, frames, images, slots, manual, lease_check, same_person, mirror)
        _gallery_labels(runtime, episode, slots, lease_check, timings, model)
        meta = [{'ordinal': frame['source_frame_index'], 'time': frame['timestamp_seconds'],
                 'relative_time': frame['relative_timestamp_seconds'], 'image_size': list(image.size),
                 **({'mirror': True} if mirrored and number == 0 else {})} for number, (frame, image) in enumerate(zip(frames, images))]
        _save(job, owner, episode, cut, key, meta, slots, stored, renumbered, getattr(runtime.get_detector(), 'revision', None), ordinal if mirrored else None)
        anime_bulk._finish(job_id, owner, episode['id'], cut)
        log.info('character-detect target frames cut=%s finished total=%.2fs boxes=%d slots=%d', cut.get('cut_index'), time.perf_counter() - started, len(stored), len(slots))
    except Exception as exc:
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
        inference_worker.log.exception('Target frame detection failed for job %s', job_id)


def _save(job, owner, episode, cut, key, frames, slots, stored, renumbered, revision, mirror_ordinal=None):
    from . import anime_models
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        anime_bulk._valid_cut(conn, job['id'], owner, episode['id'], cut)
        if not conn.execute('SELECT 1 FROM keyframe_results WHERE cache_key=?', (key,)).fetchone():
            raise RuntimeError('The evaluation of this cut changed; evaluate the cut and detect characters again')
        run = conn.execute('SELECT id FROM target_detection_runs WHERE cut_id=? AND selection_key=?', (cut['id'], key)).fetchone()
        if run:
            run_id = run['id']
            conn.execute('UPDATE target_detection_runs SET cut_start=?,cut_end=?,job_id=?,frames_json=?,model_id=?,model_revision=? WHERE id=?',
                         (cut['start_time'], cut['end_time'], job['id'], json.dumps(frames), anime_models.ANIREF_MODEL_ID, revision, run_id))
            # The mirrored first frame is replaced whole; everywhere else the user's own boxes stay.
            conn.execute("DELETE FROM target_detections WHERE run_id=? AND (source='auto' OR frame_ordinal=?)", (run_id, mirror_ordinal))
            conn.execute('DELETE FROM target_slots WHERE run_id=?', (run_id,))
            for old, new in renumbered.items():
                if old != new:
                    conn.execute("UPDATE target_detections SET slot=? WHERE run_id=? AND source='manual' AND slot=?", (-1 - new, run_id, old))
            conn.execute("UPDATE target_detections SET slot=-1-slot WHERE run_id=? AND slot<0", (run_id,))
        else:
            run_id = str(uuid.uuid4())
            conn.execute('INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,job_id,frames_json,model_id,model_revision) VALUES (?,?,?,?,?,?,?,?,?)',
                         (run_id, cut['id'], key, cut['start_time'], cut['end_time'], job['id'], json.dumps(frames), anime_models.ANIREF_MODEL_ID, revision))
        numbers = set()
        for slot in sorted(slots, key=lambda item: item.number):
            character_id = slot.character_id
            if character_id and not conn.execute("SELECT 1 FROM context_items WHERE id=? AND project_id=? AND kind='character'", (character_id, episode['project_id'])).fetchone():
                character_id = None
            conn.execute('INSERT INTO target_slots(run_id,slot,origin,character_context_id,label_source) VALUES (?,?,?,?,?)',
                         (run_id, slot.number, slot.origin, character_id, slot.label_source if character_id else None))
            numbers.add(slot.number)
        # The user's own boxes outlive a slot only when the slot does.
        # ``IN (NULL)`` would match nothing, so an empty list is spelled with a number no slot has.
        conn.execute(f"DELETE FROM target_detections WHERE run_id=? AND source='manual' AND slot NOT IN ({','.join('?' * len(numbers)) or '-1'})", (run_id, *numbers))
        for ordinal, slot, box in stored:
            conn.execute('INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json,source) VALUES (?,?,?,?,?,?)',
                         (str(uuid.uuid4()), run_id, ordinal, slot, json.dumps(list(box)), 'auto'))
    return run_id
