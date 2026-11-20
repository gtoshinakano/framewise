"""Bulk character detection and recognition.

This workflow is isolated from the original per-cut WeVisOne pipeline. It
stores the same run and detection records so the existing editor can review
the results. A runtime may be reused for successive jobs from a bulk request.

Recognition compares each unconfirmed box with the gallery of boxes the user
labelled by hand (``anime_gallery``), never with automatic labels. Manual
labels are never changed. Boxes the gallery cannot settle keep their top-3
``candidates_json`` so the user picks one on screen; no vision model is asked.
"""
import contextlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import anime_gallery, anime_matching, anime_models, db, job_inputs, media, worker

log = logging.getLogger('framewise.character_detect')
WORKFLOW = 'anime_bulk_v1'
PHASES = ('boxes', 'embeddings', 'complete')
BATCH_SIZE = 16

# Kept for callers that crop outside the gallery module.
_crop_jpeg = anime_gallery.crop_jpeg
_open_jpeg = anime_gallery.open_jpeg


@contextlib.contextmanager
def timed(label, timings=None, **fields):
    """Log the wall-clock time of one step and optionally record it in milliseconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[label] = timings.get(label, 0) + round(elapsed * 1000)
        log.info('character-detect %s took %.2fs %s', label, elapsed, ' '.join(f'{k}={v}' for k, v in fields.items()))


def frame_cache_dir(episode):
    """Temporary first frames shared by the phases; removed when detection completes."""
    return media.episode_dir(episode['project_id'], episode['id']) / 'character-frames'


def clear_cut_frames(episode, cut_id):
    cache = frame_cache_dir(episode)
    for position in ('first', 'last'):
        (cache / f'{cut_id}-{position}.jpg').unlink(missing_ok=True)
    try:
        cache.rmdir()  # only succeeds once every cut's frames are gone
    except OSError:
        pass


def clear_episode_frames(project_id, episode_id):
    shutil.rmtree(media.episode_dir(project_id, episode_id) / 'character-frames', ignore_errors=True)


class BulkRuntime:
    def __init__(self):
        self.detector = None
        self.embedder = None
        self.gallery = None
        self.gallery_project = None
        self.prepared_frames = {}
        self.inputs = {}

    def get_detector(self):
        if self.detector is None:
            self.detector = anime_models.AniRefDetector(
                os.environ.get('ANIREF_MODEL_PATH', '/models/aniref/AniRef40000-s-epoch40.pt'),
                os.environ.get('ANIREF_DEVICE', '0'))
        return self.detector

    def get_embedder(self):
        if self.embedder is None:
            self.embedder = anime_models.create_embedder()
        return self.embedder

    def get_gallery(self, project_id, lease_check=None, timings=None, evidence=None):
        """Confirmed examples for the project, reloaded once per batch so new labels count."""
        key = (project_id, job_inputs.fingerprint(evidence) if evidence is not None else None)
        if self.gallery is None or self.gallery_project != key:
            with timed('gallery', timings, project=project_id):
                self.gallery = anime_gallery.load(project_id, self.get_embedder, lease_check, evidence)
            self.gallery_project = key
        return self.gallery

    def close(self):
        import gc
        self.detector = self.embedder = self.gallery = None
        self.prepared_frames.clear()
        self.inputs.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _extract_frames(original, cut, cache, saved=None):
    """Return the first frame and RGB image, reusing its cached JPEG when present."""
    from PIL import Image
    times = media.cut_frame_times(cut['start_time'], cut['end_time'])
    frames, images = [], []
    for index, position in enumerate(('first',)):
        existing = next((frame for frame in saved or [] if frame['position'] == position), None)
        target = Path(cache) / f"{cut['id']}-{position}.jpg"
        if not (existing and target.is_file()):
            actual = media.extract_cut_frame(original, target, existing['time'] if existing else times[index],
                cut['start_time'], cut['end_time'], width=1280, last=False,
                seek_tolerance=.000002 if existing else 0)
        else:
            actual = existing['time']
        with Image.open(target) as source:
            image = source.convert('RGB')
        if existing and list(image.size) != existing['image_size']:
            raise RuntimeError('Detection frame dimensions changed; run character detection again')
        frames.append(existing or {'position': position, 'time': actual, 'image_size': list(image.size)})
        images.append(image)
    return frames, images


def _save_run(job, inputs, cut, frames, detector, timings):
    with inputs.publish() as conn:
        run_id = str(uuid.uuid4())
        preserve_checkpoint(conn, job['id'])
        conn.execute('INSERT INTO character_detection_runs(id,cut_id,job_id,cut_start,cut_end,frames_json,model_id,model_revision,phase,timings_json,input_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (run_id, cut['id'], job['id'], cut['start_time'], cut['end_time'], json.dumps(frames), anime_models.ANIREF_MODEL_ID,
             detector.revision, 'boxes', json.dumps(timings), json.dumps({'boxes': {'fingerprint': inputs.stamp}})))
        for frame in frames:
            for box in frame['boxes']:
                conn.execute('INSERT INTO character_detections(id,run_id,position,box_json) VALUES (?,?,?,?)',
                    (str(uuid.uuid4()), run_id, frame['position'], json.dumps(box)))
        inputs.capture(conn)
    return run_id


def preserve_checkpoint(conn, job_id):
    # Old databases identify one current detection run per job. Keep an obsolete
    # checkpoint tied to its original job in provenance when its boundary moves.
    conn.execute("""UPDATE character_detection_runs SET job_id=NULL,input_json=json_set(input_json,'$.original_job_id',?)
        WHERE job_id=?""", (job_id, job_id))


def enqueue(conn, episode_id, phase, *, rerun=False, cut_id=None, depends_on=(), first_index=0, last_index=None):
    # Episode-wide work after a boundary replacement must discover the new cuts.
    media_jobs = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts'), None if cut_id is None else [cut_id])
    waits = list(dict.fromkeys([*depends_on, *media_jobs]))
    if cut_id is None and waits:
        payload = {'workflow': WORKFLOW, 'episode_batch': True, 'select_cuts': True,
                   'phase': phase, 'rerun': rerun, 'depends_on': waits}
        return [worker.insert_job(conn, episode_id, 'character_detect', payload)]
    return _enqueue_cuts(conn, episode_id, phase, rerun=rerun, cut_id=cut_id, depends_on=waits,
                         first_index=first_index, last_index=last_index)


def eligible_cuts(conn, episode_id, phase, *, rerun=False, cut_id=None, depends_on=(), first_index=0, last_index=None):
    # The latest run for the cut's current boundaries decides what is still missing.
    latest = """(SELECT r2.id FROM character_detection_runs r2 WHERE r2.cut_id=c.id AND r2.cut_start=c.start_time AND r2.cut_end=c.end_time ORDER BY r2.rowid DESC LIMIT 1)"""
    # Current detection runs save one frame. Older runs may still have both
    # boundary frames, so completeness means the first frame is present.
    incomplete_frames = "(CASE WHEN json_valid(r.frames_json) THEN NOT EXISTS (SELECT 1 FROM json_each(r.frames_json) f WHERE json_extract(f.value,'$.position')='first') ELSE 1 END)"
    if rerun:
        need = f"""AND ({latest} IS NULL OR EXISTS (SELECT 1 FROM character_detection_runs r WHERE r.id={latest} AND {incomplete_frames})
            OR EXISTS (SELECT 1 FROM character_detections d WHERE d.run_id={latest}
            AND d.position='first' AND (d.label_source IS NULL OR d.label_source!='manual')))"""
    elif phase == 'boxes':
        need = f"AND ({latest} IS NULL OR EXISTS (SELECT 1 FROM character_detection_runs r WHERE r.id={latest} AND {incomplete_frames}))"
    elif phase == 'full':
        # Boxes and recognition in one pass: anything without a finished run.
        need = f"AND ({latest} IS NULL OR EXISTS (SELECT 1 FROM character_detection_runs r WHERE r.id={latest} AND (r.phase!='complete' OR {incomplete_frames})))"
    else:
        need = f"""AND (EXISTS (SELECT 1 FROM character_detection_runs r WHERE r.id={latest} AND (r.phase!='complete' OR {incomplete_frames}))
            OR EXISTS (SELECT 1 FROM jobs j WHERE j.episode_id=c.episode_id AND j.pipeline='character_detect'
                AND j.status IN ('queued','running') AND json_extract(j.payload_json,'$.cut_id')=c.id
                AND json_extract(j.payload_json,'$.phase') IN ('boxes','full')))"""
    if cut_id:
        scope, params = "AND c.id=?", [cut_id]
    else:
        scope = "AND c.analysis_enabled=1 AND c.no_characters=0 AND c.cut_index>=? AND (? IS NULL OR c.cut_index<=?)"
        params = [first_index, last_index, last_index]
    if cut_id and depends_on:
        need = ''  # a cut producer can make its current evidence obsolete
    return conn.execute(f"""SELECT c.id FROM cuts c WHERE c.episode_id=? AND c.active=1 {scope}
        {need} ORDER BY c.cut_index""", (episode_id, *params)).fetchall()


def _enqueue_cuts(conn, episode_id, phase, *, rerun, cut_id=None, depends_on=(), first_index=0, last_index=None):
    cuts = eligible_cuts(conn, episode_id, phase, rerun=rerun, cut_id=cut_id, depends_on=depends_on,
                         first_index=first_index, last_index=last_index)
    job_ids, batch_id = [], str(uuid.uuid4())
    for row in cuts:
        cid = row['id']
        payload = {'cut_id': cid, 'workflow': WORKFLOW, 'phase': phase, 'batch_id': batch_id,
                   **({'rerun': True} if rerun else {}), **({'episode_batch': True} if cut_id is None else {})}
        deps = list(depends_on)
        if phase == 'recognize':
            deps.extend(row['id'] for row in conn.execute("""SELECT id FROM jobs WHERE episode_id=? AND pipeline='character_detect'
                AND status IN ('queued','running') AND json_extract(payload_json,'$.cut_id')=?
                AND json_extract(payload_json,'$.phase') IN ('boxes','full')""", (episode_id, cid)))
        if deps:
            payload['depends_on'] = list(dict.fromkeys(deps))
        job_ids.append(worker.insert_job(conn, episode_id, 'character_detect', payload))
    return job_ids


def dispatch_episode(job, owner, payload):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job['id'], owner)
        if payload.get('expanded'):
            finish(conn, job)
            return
        children = _enqueue_cuts(conn, job['episode_id'], payload['phase'], rerun=payload.get('rerun', False))
        if not children:
            finish(conn, job, 'Complete · No matching cuts need character detection')
            return
        request = {**job, 'payload_json': json.dumps({**payload, 'expanded': True})}
        worker.defer_for_dependencies(conn, request, owner, children, 'Waiting for cut detections')


def latest_current_run(cut, conn=None):
    return db.one('SELECT * FROM character_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=? ORDER BY rowid DESC LIMIT 1',
                  (cut['id'], cut['start_time'], cut['end_time']), conn=conn)


def cut_evidence(conn, job, payload):
    cut = db.one('SELECT id,active,start_time,end_time,cut_index,analysis_enabled,no_characters FROM cuts WHERE id=? AND episode_id=?',
                 (payload['cut_id'], job['episode_id']), conn=conn)
    if not cut or not cut['active']:
        return {'skip': 'Cut archived or removed'}
    if payload.get('episode_batch') and (not cut['analysis_enabled'] or cut['no_characters']):
        return {'skip': 'Cut disabled or marked as having no characters'}
    episode = db.one('SELECT id,project_id,original_ext FROM episodes WHERE id=?', (job['episode_id'],), conn=conn)
    run = latest_current_run(cut, conn)
    if run and not any(frame['position'] == 'first' for frame in json.loads(run['frames_json'])):
        run = None
    if run:
        # Display notes and cached candidate scores are not read by the models.
        run = {key: run[key] for key in ('id', 'frames_json', 'phase', 'timings_json')}
        boxes = db.all_rows("""SELECT id,run_id,position,box_json,character_context_id,label_source FROM character_detections
            WHERE run_id=? AND position='first' ORDER BY rowid""", (run['id'],), conn=conn)
    else:
        boxes = []
    return {'cut': cut, 'episode': episode, 'run': run, 'boxes': boxes}


def gather(conn, job, payload):
    data = cut_evidence(conn, job, payload)
    if 'skip' in data:
        return data
    if payload.get('workflow') != WORKFLOW:
        from . import character_detection
        cut = data['cut']
        saved = db.one("""SELECT id,frames_json FROM character_detection_runs WHERE job_id=?
            AND cut_start=? AND cut_end=?""", (job['id'], cut['start_time'], cut['end_time']), conn=conn)
        detected = db.all_rows("""SELECT id,position,box_json,label_source,character_context_id
            FROM character_detections WHERE run_id=? ORDER BY rowid""", (saved['id'],), conn=conn) if saved else []
        return {**data, 'saved': saved, 'detected': detected,
                'references': character_detection.reference_snapshots(conn, data['episode']['project_id'])}
    phase = payload.get('phase', 'full')
    data['mode'] = 'full' if payload.get('rerun') and not data['run'] else phase
    if phase != 'boxes':
        data['gallery'] = anime_gallery.evidence(conn, data['episode']['project_id'])
    return data


def job_snapshot(job, owner, runtime):
    snapshots = runtime.inputs
    if job['id'] not in snapshots:
        with db.connect() as conn:
            conn.execute('BEGIN')
            worker.require_lease(conn, job['id'], owner)
            snapshots[job['id']] = job_inputs.JobInputs(conn, job, owner)
    return snapshots[job['id']]


def _target_vectors(job_id, owner, detections, crops, model):
    vectors, missing = [None] * len(detections), []
    with db.connect() as conn:
        worker.require_lease(conn, job_id, owner)
        for index, detection in enumerate(detections):
            vector = anime_matching.read_cached_detection(conn, detection['id'], model.model_id,
                                                          model.revision, model.preprocessing)
            if vector is None:
                missing.append(index)
            else:
                vectors[index] = vector
    if missing:
        computed = model.embed([_open_jpeg(crops[index]) for index in missing])
        if len(computed) != len(missing):
            raise RuntimeError('Character embedder returned an unexpected number of target embeddings')
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            worker.require_lease(conn, job_id, owner)
            for index, vector in zip(missing, computed):
                vectors[index] = vector
                if conn.execute('SELECT 1 FROM character_detections WHERE id=? AND box_json=?', (detections[index]['id'], detections[index]['box_json'])).fetchone():
                    anime_matching.write_cached_detection(conn, detections[index]['id'], model.model_id,
                        model.revision, model.preprocessing, vector)
    return vectors


def _detection_crop(images, detection):
    """JPEG crop of a detection from the first frame."""
    return _crop_jpeg(images[0], json.loads(detection['box_json']))


def finish(conn, job, stage='Complete'):
    conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                 (stage, job['id']))
    worker.sync_episode_status(conn, job['episode_id'])


def _recognize(inputs, runtime, frames, images, detections, lease_check, timings):
    """Use the gallery captured at start and return candidates and proposed labels."""
    job, owner = inputs.job, inputs.owner
    gallery = runtime.get_gallery(inputs.data['episode']['project_id'], lease_check, timings, inputs.data['gallery'])
    if not len(gallery) or not detections:
        return {}, []
    model = runtime.get_embedder()
    defaults = model.defaults
    worker.stage(job['id'], owner, job['episode_id'], f'Matching characters with {model.label}')
    with timed('embeddings', timings, cut=inputs.data['cut']['cut_index'], detections=len(detections)):
        vectors = _target_vectors(job['id'], owner, detections, [_detection_crop(images, d) for d in detections], model)
    threshold = anime_matching.env_fraction('ANIME_REID_MIN_SIMILARITY', defaults['similarity'])
    margin = anime_matching.env_fraction('ANIME_REID_MIN_MARGIN', defaults['margin'])
    image_size = next(frame['image_size'] for frame in frames if frame['position'] == 'first')
    candidates, proposals = {}, []
    for index, detection in enumerate(detections):
        if detection['label_source'] is not None:
            continue
        lease_check()
        ranked = gallery.rank(vectors[index])
        candidates[detection['id']] = [{'character_id': cid, 'score': round(score, 3)} for cid, score in ranked[:3]]
        box = json.loads(detection['box_json'])
        if anime_gallery.box_problem(box, image_size):
            continue
        character_id = anime_matching.confident_match(ranked, threshold, margin)
        if not character_id and defaults.get('wide'):
            character_id = anime_matching.confident_match(ranked, *defaults['wide'])
        if character_id:
            runner_up = f", next {gallery.names.get(ranked[1][0], 'other')} {ranked[1][1]:.2f}" if len(ranked) > 1 else ''
            reason = f'{model.label} similarity {ranked[0][1]:.2f} to your confirmed labels{runner_up}'
            proposals.append((ranked[0][1], detection['id'], character_id, reason, model.model_id, model.revision))
    return candidates, sorted(proposals, key=lambda item: -item[0])


def _save_recognition(inputs, candidates, proposals, timings, rerun):
    run = inputs.data['run']
    boxes = inputs.data['boxes']
    previous = {box['id']: box['character_context_id'] for box in boxes}
    present = {box['character_context_id'] for box in boxes if box['character_context_id'] and (not rerun or box['label_source'] != 'auto')}
    assigned = {}
    for _, box_id, character_id, reason, model, revision in proposals:
        if character_id not in present:
            assigned[box_id] = (character_id, reason, model, revision)
            present.add(character_id)
    changed = any(previous[box_id] != value[0] for box_id, value in assigned.items()) or (
        rerun and any(box['label_source'] == 'auto' and box['id'] not in assigned for box in boxes))
    with inputs.publish(invalidate=changed) as conn:
        if rerun:
            conn.execute("""UPDATE character_detections SET character_context_id=NULL,label_source=NULL,match_reason=NULL,
                match_model_id=NULL,match_model_revision=NULL,candidates_json=NULL
                WHERE run_id=? AND position='first' AND (label_source='auto' OR label_source IS NULL)""", (run['id'],))
        for box_id, (character_id, reason, model, revision) in assigned.items():
            conn.execute("""UPDATE character_detections SET character_context_id=?,label_source='auto',match_reason=?,
                match_model_id=?,match_model_revision=? WHERE id=? AND label_source IS NULL""",
                (character_id, reason[:2000], model, revision, box_id))
        for box_id, ranked in candidates.items():
            conn.execute("UPDATE character_detections SET candidates_json=? WHERE id=? AND (label_source IS NULL OR label_source='auto')",
                         (json.dumps(ranked), box_id))
        provenance = {'fingerprint': inputs.stamp, 'run_id': run['id'], 'boxes': list(previous),
                      'exemplars': [row['id'] for row in inputs.data['gallery'][0]]}
        conn.execute("""UPDATE character_detection_runs SET phase='complete',timings_json=?,
            input_json=json_set(input_json,'$.recognition',json(?)) WHERE id=?""",
            (json.dumps(timings), json.dumps(provenance), run['id']))
        conn.execute("UPDATE jobs SET input_json=? WHERE id=?", (json.dumps(provenance), inputs.job['id']))
        finish(conn, inputs.job)


def process(job_id, owner, runtime):
    """Detect boxes and recognize people using the leased cut's current evidence."""
    from . import inference_worker
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    try:
        inputs = job_snapshot(job, owner, runtime)
        data = inputs.data
        if data.get('skip'):
            with inputs.publish(invalidate=False) as conn:
                finish(conn, job, f"Skipped · {data['skip']}")
            return
        rerun = json.loads(job['payload_json']).get('rerun', False)
        cut, episode, run, mode = (data[key] for key in ('cut', 'episode', 'run', 'mode'))
        if (mode == 'recognize' and not run) or (run and (mode == 'boxes' or (run['phase'] == 'complete' and not rerun))):
            with inputs.publish(invalidate=False) as conn:
                finish(conn, job, 'Skipped · Detect first-frame boxes first' if not run else 'Complete')
            return
        original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
        cache = frame_cache_dir(episode)
        cache.mkdir(parents=True, exist_ok=True)
        timings = json.loads(run['timings_json'] or '{}') if run else {}
        ready = runtime.prepared_frames.pop(job_id, None)
        if not run:
            with timed('frames', timings, cut=cut['cut_index']):
                ready = ready or _extract_frames(original, cut, cache)
            frames, images = ready
            worker.stage(job_id, owner, episode['id'], 'Detecting characters with AniRef')
            detector = runtime.get_detector()
            with timed('boxes', timings, cut=cut['cut_index']):
                boxes = detector.detect(images)
            if len(boxes) != len(frames):
                raise RuntimeError('AniRef returned an unexpected number of frames')
            for frame, frame_boxes in zip(frames, boxes):
                frame['boxes'] = frame_boxes
            _save_run(job, inputs, cut, frames, detector, timings)
            run = inputs.data['run']
        if mode == 'boxes':
            with inputs.publish(invalidate=False) as conn:
                finish(conn, job)
            return
        detections = [{**box, 'label_source': None, 'character_context_id': None} if rerun and box['label_source'] == 'auto' else box
                      for box in inputs.data['boxes']]
        candidates, proposals = {}, []
        if any(box['label_source'] is None for box in detections):
            frames, images = ready or _extract_frames(original, cut, cache, json.loads(run['frames_json']))
            lease_check = lambda: inference_worker.check_summary_lease(job_id, owner)
            candidates, proposals = _recognize(inputs, runtime, frames, images, detections, lease_check, timings)
        _save_recognition(inputs, candidates, proposals, timings, rerun)
        clear_cut_frames(episode, cut['id'])
    except (worker.LeaseLost, inference_worker.SummaryCancelled):
        return
    except Exception as exc:
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
        inference_worker.log.exception('Bulk anime character detection failed for job %s', job_id)
    finally:
        runtime.inputs.pop(job_id, None)


def _claim_batch(first_job, first_owner, limit=BATCH_SIZE):
    """Claim related queued cuts atomically while retaining an individual lease each."""
    payload = json.loads(first_job['payload_json'])
    batch_id = payload.get('batch_id')
    claimed = [(first_job['id'], first_owner)]
    if not batch_id or limit <= 1:
        return claimed
    now = time.time()
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, first_job['id'], first_owner)
        candidates = conn.execute(f'''SELECT * FROM jobs WHERE episode_id=? AND pipeline='character_detect'
            AND status='queued' AND COALESCE(lease_until,0)<=? AND not_before<=?
            AND json_extract(payload_json,'$.workflow')='anime_bulk_v1'
            AND json_extract(payload_json,'$.batch_id')=? AND {worker.DEPENDENCIES_DONE}
            ORDER BY created_at,rowid''', (first_job['episode_id'], now, now, batch_id)).fetchall()
        active = worker.job_scopes.active_leases(conn, now)
        for row in candidates:
            if worker.lease_candidate(conn, row, first_owner, now, active):
                claimed.append((row['id'], first_owner))
            if len(claimed) >= limit:
                break
    return claimed


def _prepare_frames(claimed, runtime):
    """Extract the first frame of every claimed cut that still needs it, in parallel."""
    from . import inference_worker
    work = []
    for job_id, owner in claimed:
        job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
        if not job:
            continue
        try:
            inputs = job_snapshot(job, owner, runtime)
        except worker.LeaseLost:
            continue
        except Exception as exc:
            inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
            continue
        data = inputs.data
        if 'skip' in data:
            continue
        run, mode, cut, episode = (data[key] for key in ('run', 'mode', 'cut', 'episode'))
        rerun = json.loads(job['payload_json']).get('rerun', False)
        if mode == 'recognize' and not run:
            continue
        if run and (mode == 'boxes' or (run['phase'] == 'complete' and not rerun)
                    or not any(box['label_source'] is None or (rerun and box['label_source'] == 'auto') for box in data['boxes'])):
            continue
        work.append((job, owner, cut, episode, run))

    def extract(item):
        job, owner, cut, episode, run = item
        try:
            with db.connect() as conn:
                worker.require_lease(conn, job['id'], owner)
            cache = frame_cache_dir(episode)
            cache.mkdir(parents=True, exist_ok=True)
            original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
            return job['id'], _extract_frames(original, cut, cache, json.loads(run['frames_json']) if run else None), None
        except Exception as exc:  # noqa: BLE001 - reported per job below
            return job['id'], None, exc

    with ThreadPoolExecutor(anime_gallery.FRAME_THREADS) as pool:
        for (job, owner, cut, episode, run), (job_id, result, error) in zip(work, pool.map(extract, work)):
            if error is not None:
                inference_worker.fail_job(job_id, owner, episode['id'], error)
                inference_worker.log.exception('Could not prepare frames for character detection %s', job_id, exc_info=error)
            else:
                runtime.prepared_frames[job_id] = result


def _detect_batch(claimed, runtime):
    """Detect first-frame boxes for several leased cuts in one GPU invocation."""
    from . import inference_worker
    prepared = []
    for job_id, owner in claimed:
        job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
        if not job:
            continue
        try:
            inputs = job_snapshot(job, owner, runtime)
            data = inputs.data
            if 'skip' in data or data['run'] or data['mode'] == 'recognize':
                continue
            cut, episode = data['cut'], data['episode']
            worker.stage(job_id, owner, episode['id'], 'Preparing frames for AniRef batch')
            timings = {}
            ready = runtime.prepared_frames.pop(job_id, None)
            with timed('frames', timings, cut=cut.get('cut_index')):
                if ready is None:
                    cache = frame_cache_dir(episode)
                    cache.mkdir(parents=True, exist_ok=True)
                    original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
                    ready = _extract_frames(original, cut, cache)
            frames, images = ready
            prepared.append((job, inputs, cut, episode, frames, images, timings))
        except worker.LeaseLost:
            continue
        except Exception as exc:
            inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
            inference_worker.log.exception('Could not prepare cut for anime detection batch %s', job_id)
    if not prepared:
        return
    try:
        detector = runtime.get_detector()
        all_images = [image for item in prepared for image in item[5]]
        batch_start = time.perf_counter()
        with timed('boxes_batch', None, cuts=len(prepared), images=len(all_images)):
            all_boxes = detector.detect(all_images)
        if len(all_boxes) != len(all_images):
            raise RuntimeError('AniRef returned an unexpected number of frames')
        per_cut_ms = round((time.perf_counter() - batch_start) * 1000 / len(prepared))
    except Exception:
        # Retry cuts individually so a malformed frame does not fail the group.
        inference_worker.log.exception('AniRef batch failed; retrying cuts individually')
        for job, inputs, cut, episode, frames, images, timings in prepared:
            runtime.prepared_frames[job['id']] = (frames, images)
        return
    box_index = 0
    for job, inputs, cut, episode, frames, images, timings in prepared:
        try:
            for frame in frames:
                frame['boxes'] = all_boxes[box_index]
                box_index += 1
            timings['boxes'] = per_cut_ms
            _save_run(job, inputs, cut, frames, detector, timings)
            if json.loads(job['payload_json']).get('phase') != 'boxes':
                runtime.prepared_frames[job['id']] = (frames, images)
        except worker.LeaseLost:
            continue
        except Exception as exc:
            inference_worker.fail_job(job['id'], inputs.owner, episode['id'], exc)
            inference_worker.log.exception('Could not save anime detection batch result %s', job['id'])


def process_batch(first_job, first_owner, runtime, batch_size=BATCH_SIZE):
    """Run several cut jobs with separate leases, one model at a time across all of them."""
    payload = json.loads(first_job.get('payload_json') or '{}')
    if payload.get('select_cuts'):
        dispatch_episode(first_job, first_owner, payload)
        return
    claimed = _claim_batch(first_job, first_owner, batch_size)
    extra_beats = []
    try:
        for job_id, owner in claimed[1:]:
            done = threading.Event()
            thread = threading.Thread(target=worker.heartbeat, args=(job_id, owner, done), daemon=True)
            thread.start()
            extra_beats.append((done, thread))
        runtime.gallery = None  # reload: labels confirmed since the previous batch now count
        # One pass per model, each over every claimed cut:
        _prepare_frames(claimed, runtime)                    # ffmpeg, in parallel
        _detect_batch(claimed, runtime)                      # AniRef boxes
        for job_id, owner in claimed:
            process(job_id, owner, runtime)                  # embeddings + gallery
    finally:
        for job_id, _ in claimed:
            runtime.prepared_frames.pop(job_id, None)
            runtime.inputs.pop(job_id, None)
        for done, _ in extra_beats:
            done.set()
        for _, thread in extra_beats:
            thread.join(timeout=1)
