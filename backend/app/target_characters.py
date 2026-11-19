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
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import anime_bulk, anime_gallery, anime_matching, db, frame_cache, frame_confidence, keyframe_camera, keyframe_pose, media, target_candidates, target_reviews, target_tracking, worker

log = logging.getLogger('framewise.character_detect')
WORKFLOW = 'target_frames_v1'
FRAME_WIDTH = 1280
MAX_BOXES = target_tracking.MAX_BOXES
REVIEW_VERSION = 1


def is_target_job(payload):
    return isinstance(payload, dict) and payload.get('workflow') == WORKFLOW


def current_selection(cut_id):
    """``(cut, source path, key, result)`` of the cut's latest evaluation (``keyframes.current_selection``)."""
    from . import keyframes
    return keyframes.current_selection(cut_id)


def pending_job(cut_id):
    return db.one("""SELECT id FROM jobs WHERE pipeline='character_detect' AND status IN ('queued','running')
                     AND json_extract(payload_json,'$.cut_id')=?""", (cut_id,))


def latest_job(cut_id):
    return db.one("""SELECT id,status,stage,error FROM jobs WHERE pipeline='character_detect'
                     AND json_extract(payload_json,'$.workflow')=? AND json_extract(payload_json,'$.cut_id')=?
                     ORDER BY rowid DESC LIMIT 1""", (WORKFLOW, cut_id))


def run_for(cut_id, key):
    return db.one('SELECT * FROM target_detection_runs WHERE cut_id=? AND selection_key=?', (cut_id, key))


def frame_capacity(slots, locked=False):
    """Keep the open-cast cap, with room for every reviewed opening colour on a locked cut."""
    opening = sum(slot['origin'] == 'first_frame' and not slot.get('ignored') for slot in slots)
    return max(MAX_BOXES, opening) if locked else MAX_BOXES


def state(cut_id):
    """What the Target frames tab shows: the evaluation, the latest run with its slots and boxes, and the job."""
    cut, _, key, _ = current_selection(cut_id)
    if not cut:
        return None
    run = db.one('SELECT * FROM target_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut_id,))
    cast = db.one('SELECT locked FROM target_cast_settings WHERE cut_id=?', (cut_id,))
    locked = bool((cast or {}).get('locked'))
    result = {'evaluation': 'current' if key else 'missing', 'selection_key': key, 'max_boxes': MAX_BOXES,
              'cast_locked': locked, 'run': None, 'job': latest_job(cut_id)}
    if run:
        slots = db.all_rows('''SELECT s.slot,s.origin,s.character_context_id,s.label_source,s.ignored,c.title AS character_name
                               FROM target_slots s LEFT JOIN context_items c ON c.id=s.character_context_id
                               WHERE s.run_id=? ORDER BY s.slot''', (run['id'],))
        boxes = db.all_rows('''SELECT d.* FROM target_detections d JOIN target_slots s ON s.run_id=d.run_id AND s.slot=d.slot
                              WHERE d.run_id=? AND s.ignored=0 ORDER BY d.rowid''', (run['id'],))
        result['max_boxes'] = frame_capacity(slots, bool(run['cast_locked']))
        candidates = json.loads(run['candidates_json'])
        detections = [target_reviews.box_row(box) for box in boxes]
        result['run'] = {
            'id': run['id'], 'selection_key': run['selection_key'],
            'stale': key is None or run['selection_key'] != key or run['cut_start'] != cut['start_time'] or run['cut_end'] != cut['end_time'],
            'frames': target_reviews.visible_reviews(json.loads(run['frames_json']), slots, detections, candidates),
            'slots': [{**slot, 'ignored': bool(slot['ignored'])} for slot in slots],
            'detections': detections, 'candidates': candidates,
            'unmatched_count': sum(candidate['tag'] == 'unmatched' for candidate in candidates),
            'review_version': run['review_version'], 'cast_locked': bool(run['cast_locked']),
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
    opening = {row['first_detection_id']: row for row in existing_slots if row.get('first_detection_id') and row['origin'] == 'first_frame'}
    slots, renumbered, inherited = [], {}, set()
    for number, person in enumerate(first):
        original = person['row']
        mine = opening.get(original['id'])
        if not mine:
            old = saved.get(number)
            if old and old['origin'] == 'first_frame' and not old.get('first_detection_id'):
                mine = old  # old runs only have the first-frame order
        if mine:
            inherited.add(mine['slot'])
            renumbered[mine['slot']] = number
        if mine and mine['label_source'] == 'manual':
            character_id, label = mine['character_context_id'], 'manual'
        else:
            character_id = original['character_context_id']
            label = 'inherited' if character_id else None
        slot = target_tracking.Slot(number, 'first_frame', character_id, label, person['box'], person['vector'])
        slot.first_detection_id = original['id']
        slot.ignored = bool(mine and mine.get('ignored'))
        slots.append(slot)
    manual = [dict(row) for row in existing_boxes if row['source'] == 'manual']
    for row in sorted(existing_slots, key=lambda item: item['slot']):
        if row['slot'] in inherited:
            continue  # the first frame still has this person
        kept_label = row['label_source'] == 'manual'
        if not kept_label and not row.get('ignored') and not any(box['slot'] == row['slot'] for box in manual):
            continue
        number = row['slot'] if not any(slot.number == row['slot'] for slot in slots) else target_tracking.free_number(slots)
        if number is None:
            raise RuntimeError('The first-frame cast and your manual edits use more colours than fit. Finish the first-frame review before detecting again.')
        renumbered[row['slot']] = number
        slot = target_tracking.Slot(number, 'appeared', row['character_context_id'] if kept_label or row.get('ignored') else None,
                                    row['label_source'] if kept_label or row.get('ignored') else None)
        slot.ignored = bool(row.get('ignored'))
        slots.append(slot)
    numbers = {slot.number for slot in slots}
    kept = []
    for box in manual:
        if box['slot'] in renumbered or box['slot'] in numbers:
            box['slot'] = renumbered.get(box['slot'], box['slot'])
            box['box'] = json.loads(box['box_json'])
            kept.append(box)
    return slots, kept, renumbered


def _gallery_labels(gallery, slots, model):
    """Name people who are not in the first frame by the user's labelled boxes (read only, never added to)."""
    todo = sorted((slot for slot in slots if slot.origin == 'appeared' and slot.label_source != 'manual' and slot.first is not None), key=lambda slot: slot.number)
    if not todo:
        return
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


def _pose_model(runtime):
    if not hasattr(runtime, 'target_pose'):
        folder = keyframe_pose.MODEL_DIR
        available = all((folder / name).is_file() for name in (keyframe_pose.DETECTOR, keyframe_pose.POSE))
        runtime.target_pose = None
        if available:
            try:
                runtime.target_pose = keyframe_pose.DWPose(threads=int(os.environ.get('DWPOSE_THREADS') or 3))
            except Exception:
                log.exception('Target pose models could not be loaded; continuing with predicted boxes')
    return runtime.target_pose


def track(runtime, model, frames, images, slots, manual, lease_check, same_person, mirror=None, *, gallery,
          band=None, locked=False, removed=None, rejected=None, hidden=()):
    """Detect and globally assign once per non-mirrored frame, with manual anchors held fixed."""
    import numpy as np

    detector = runtime.get_detector()
    skip = {0} if mirror is not None else set()
    todo = [number for number in range(len(images)) if number not in skip]
    answered = detector.detect([images[number] for number in todo]) if todo else []
    if len(answered) != len(todo):
        raise RuntimeError('AniRef returned an unexpected number of frames')
    all_boxes = [[] for _ in images]
    for number, boxes in zip(todo, answered):
        all_boxes[number] = boxes
    detections = [[{'box': box, 'rank': rank, 'raw_index': rank, 'small': bool(anime_gallery.box_problem(box, image.size)),
                    'vector': None, 'gallery': {}} for rank, box in enumerate(boxes)]
                  for boxes, image in zip(all_boxes, images)]
    index = {frame['source_frame_index']: number for number, frame in enumerate(frames)}
    manual_here = {number: [] for number in range(len(frames))}
    for box in manual:
        number = index.get(box['frame_ordinal'])
        if number is not None and number not in skip and not any(slot.number == box['slot'] and slot.ignored for slot in slots):
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
            if len(gallery):
                item['gallery'] = dict(gallery.rank(vector))
    bgr = [np.asarray(image)[:, :, ::-1].copy() for image in images]
    grays = [keyframe_camera.gray(image) for image in bgr]
    masked = [list(boxes) for boxes in all_boxes]
    if mirror is not None:
        masked[0] = [item[2] for item in mirror]
    cameras = [None, *keyframe_camera.pair_cameras(grays, band, masked, [image.size for image in images])]
    pose = _pose_model(runtime)
    frozen = {slot.number for slot in slots if slot.ignored}
    manual_slots = {box['slot'] for box in manual}
    active = [slot for slot in slots if slot.number not in frozen and (not locked or slot.origin == 'first_frame' or slot.label_source == 'manual' or slot.number in manual_slots)]
    first_count = sum(slot.origin == 'first_frame' for slot in active)
    limit = max(MAX_BOXES, first_count) if locked else MAX_BOXES
    verified = {slot.number for slot in active if slot.box}
    hidden_by_frame = {}
    for item in hidden:
        hidden_by_frame.setdefault(item['frame_ordinal'], []).append(json.loads(item['box_json']))
    stored, reviews, review_candidates = [], {}, []
    for number, frame in enumerate(frames):
        lease_check()
        ordinal = frame['source_frame_index']
        if number in skip:
            stored.extend((*item, None) for item in mirror)
            reviews[ordinal] = {'score': 1., 'flags': {}}
            continue
        placed_by_user = manual_here[number]
        by_number = {slot.number: slot for slot in active}
        for item in placed_by_user:
            slot = by_number.get(item['slot'])
            if slot:
                slot.anchor(item['box'], item['vector'])
        skeletons = []
        if pose:
            rgb = bgr[number][:, :, ::-1]
            for box in pose.detect(bgr[number]):
                points = pose.keypoints(rgb, box)
                if keyframe_pose.body_usable(points):
                    skeletons.append({'box': box[:4], 'points': points[:17]})
        blocked = {item['slot'] for item in placed_by_user} | frozen | ({slot.number for slot in slots} - set(by_number))
        protected = [item['box'] for item in placed_by_user]
        excluded = protected + list((removed or {}).get(ordinal, ())) + hidden_by_frame.get(ordinal, [])
        candidates, predicted, merges = target_candidates.prepare(detections[number], [slot for slot in active if slot.number not in blocked],
            images[number].size, cameras[number], band, skeletons, split=True, previous=grays[number - 1] if number else None,
            current=grays[number], verified=verified, suppressed=excluded, protected=protected)
        placed, unmatched = target_tracking.assign_global(slots, candidates, same_person, predicted, locked=locked,
                                                          max_boxes=limit - len(placed_by_user), blocked=blocked)
        if not number and not first_count:
            first_count = len(placed) + len(placed_by_user)
        active = [slot for slot in slots if not slot.ignored and (not locked or slot.number in by_number)]
        verified = {item['slot'] for item in placed} | {item['slot'] for item in placed_by_user}
        for slot in active:
            if slot.number not in verified and slot.number in predicted:
                slot.box = predicted[slot.number]
        stored.extend((ordinal, entry['slot'], candidates[entry['detection']]['box'], entry['scores']) for entry in placed)
        combined = [*candidates, *placed_by_user]
        anchors = [{'detection': len(candidates) + i, 'slot': item['slot'], 'new': False} for i, item in enumerate(placed_by_user)]
        items = _review_candidates(ordinal, detections[number], candidates, [*placed, *anchors], unmatched, merges, active, predicted,
                                   same_person, (rejected or {}).get(ordinal, ()))
        review_candidates.extend(items)
        reviews[ordinal] = frame_confidence.review([*placed, *anchors], combined,
            [item['id'] for item in items if item['tag'] == 'unmatched'], active, first_count, predicted, images[number].size,
            [item['id'] for item in items if item['tag'] == 'merge'])
    return stored, reviews, review_candidates


def _review_candidates(ordinal, raw, candidates, placed, unmatched, merges, slots, predicted, same_person, rejected):
    """Keep unmatched crops and split proposals independently of the coloured boxes."""
    out = []
    taken = {entry['slot'] for entry in placed}

    def add(tag, box, **extra):
        if not any(tag == kind and target_tracking.iou(box, old) >= .50 for kind, old in rejected):
            out.append({'id': str(uuid.uuid4()), 'frame_ordinal': ordinal, 'tag': tag, 'box': box, 'source': 'tracker', **extra})

    merged_indexes = {item['raw_index'] for item in merges}
    for index in unmatched:
        item = candidates[index]
        if item['raw_index'] is None or item['raw_index'] in merged_indexes:
            continue
        scores = [(target_tracking.match_evidence(slot, item, predicted.get(slot.number), same_person), slot.number)
                  for slot in slots if slot.number not in taken]
        evidence, slot = min(scores, key=lambda pair: pair[0]['cost'], default=({}, None))
        add('unmatched', item['box'], suggested_slot=slot if evidence.get('cost', 2) < target_tracking.UNMATCHED_COST + .07 else None,
            scores=evidence or None)
    for item in merges:
        add('merge', raw[item['raw_index']]['box'], parts=item['parts'], method=item['source'])
    return out


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
        ignored = {slot['slot'] for slot in existing_slots if slot.get('ignored')}
        hidden = [item for item in existing_boxes if item['slot'] in ignored]
        with db.connect() as conn:
            cast = conn.execute('SELECT locked FROM target_cast_settings WHERE cut_id=?', (cut['id'],)).fetchone()
            locked = bool(cast and cast['locked'])
            removed, rejected = target_reviews.decisions(conn, existing['id']) if existing else ({}, {})
        runtime.gallery = None
        gallery = runtime.get_gallery(episode['project_id'], lease_check, timings)
        gallery = anime_gallery.Gallery([item for item in gallery.items if item['cut_id'] != cut['id']])
        mask = result.get('subtitle_mask') or {}
        band = mask if mask.get('source') == 'detected' else None
        worker.stage(job_id, owner, episode['id'], 'Finding people on the evaluated frames')
        same_person = anime_matching.env_fraction('TARGET_SAME_PERSON_SIMILARITY', model.defaults['track'])
        with anime_bulk.timed('target_detect', timings, cut=cut.get('cut_index'), frames=len(frames)):
            stored, reviews, candidates = track(runtime, model, frames, images, slots, manual, lease_check, same_person, mirror,
                gallery=gallery, band=band, locked=locked, removed=removed, rejected=rejected, hidden=hidden)
        _gallery_labels(gallery, slots, model)
        meta = [{'ordinal': frame['source_frame_index'], 'time': frame['timestamp_seconds'],
                 'relative_time': frame['relative_timestamp_seconds'], 'image_size': list(image.size),
                 'review': reviews[frame['source_frame_index']],
                 **({'mirror': True} if mirrored and number == 0 else {})} for number, (frame, image) in enumerate(zip(frames, images))]
        _save(job, owner, episode, cut, key, meta, slots, stored, renumbered, getattr(runtime.get_detector(), 'revision', None),
              ordinal if mirrored else None, candidates, locked)
        anime_bulk._finish(job_id, owner, episode['id'], cut)
        log.info('character-detect target frames cut=%s finished total=%.2fs boxes=%d slots=%d', cut.get('cut_index'), time.perf_counter() - started, len(stored), len(slots))
    except Exception as exc:
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
        inference_worker.log.exception('Target frame detection failed for job %s', job_id)


def _save(job, owner, episode, cut, key, frames, slots, stored, renumbered, revision, mirror_ordinal, candidates, locked):
    from . import anime_models
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        anime_bulk._valid_cut(conn, job['id'], owner, episode['id'], cut)
        if not conn.execute('SELECT 1 FROM keyframe_results WHERE cache_key=?', (key,)).fetchone():
            raise RuntimeError('The evaluation of this cut changed; evaluate the cut and detect characters again')
        run = conn.execute('SELECT id FROM target_detection_runs WHERE cut_id=? AND selection_key=?', (cut['id'], key)).fetchone()
        values = (json.dumps(frames), json.dumps(candidates), int(locked), REVIEW_VERSION)
        if run:
            run_id = run['id']
            conn.execute('''UPDATE target_detection_runs SET cut_start=?,cut_end=?,job_id=?,model_id=?,model_revision=?,
                         frames_json=?,candidates_json=?,cast_locked=?,review_version=? WHERE id=?''',
                         (cut['start_time'], cut['end_time'], job['id'], anime_models.ANIREF_MODEL_ID, revision, *values, run_id))
            # F1 is always a fresh mirror. Archive any manual rows left on it by the older editable implementation.
            retired = [target_reviews.box_row(row) for row in conn.execute('''SELECT * FROM target_detections
                WHERE run_id=? AND frame_ordinal=? AND source='manual' ''', (run_id, mirror_ordinal))]
            if retired:
                target_reviews.record(conn, {'id': run_id, 'cut_id': cut['id'], 'selection_key': key}, 'mirror_migrated', retired)
            # Keep manual and hidden rows elsewhere, including their stable IDs.
            conn.execute('''DELETE FROM target_detections WHERE run_id=? AND (frame_ordinal=? OR (source!='manual'
                         AND slot NOT IN (SELECT slot FROM target_slots WHERE run_id=? AND ignored=1)))''', (run_id, mirror_ordinal, run_id))
            conn.execute('DELETE FROM target_slots WHERE run_id=?', (run_id,))
            for old, new in renumbered.items():
                if old != new:
                    conn.execute('UPDATE target_detections SET slot=? WHERE run_id=? AND slot=?', (-1 - new, run_id, old))
            conn.execute("UPDATE target_detections SET slot=-1-slot WHERE run_id=? AND slot<0", (run_id,))
        else:
            run_id = str(uuid.uuid4())
            conn.execute('''INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,job_id,model_id,model_revision,
                         frames_json,candidates_json,cast_locked,review_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                         (run_id, cut['id'], key, cut['start_time'], cut['end_time'], job['id'], anime_models.ANIREF_MODEL_ID, revision, *values))
        numbers = set()
        for slot in sorted(slots, key=lambda item: item.number):
            character_id = slot.character_id
            if character_id and not conn.execute("SELECT 1 FROM context_items WHERE id=? AND project_id=? AND kind='character'", (character_id, episode['project_id'])).fetchone():
                character_id = None
            label_source = slot.label_source if character_id or slot.label_source == 'manual' else None
            conn.execute('''INSERT INTO target_slots(run_id,slot,origin,character_context_id,label_source,ignored,first_detection_id)
                            VALUES (?,?,?,?,?,?,?)''', (run_id, slot.number, slot.origin, character_id, label_source, int(slot.ignored), slot.first_detection_id))
            numbers.add(slot.number)
        if any(row['slot'] not in numbers for row in conn.execute('SELECT slot FROM target_detections WHERE run_id=?', (run_id,))):
            raise RuntimeError('Detecting again would orphan a user edit. The previous run was kept.')
        for ordinal, slot, box, scores in stored:
            conn.execute('INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json,source,scores_json) VALUES (?,?,?,?,?,?,?)',
                         (str(uuid.uuid4()), run_id, ordinal, slot, json.dumps(list(box)), 'auto', json.dumps(scores) if scores else None))
    return run_id
