"""Bulk character detection with cached reference embeddings.

This workflow is isolated from the original per-cut WeVisOne pipeline. It
stores the same run and detection records so the existing editor can review
the results. A runtime may be reused for successive jobs from a bulk request.
"""
import io
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from . import anime_matching, anime_models, character_detection, db, frame_model, media, worker


class BulkRuntime:
    def __init__(self):
        self.detector = None
        self.reid = None
        self.reference_images = {}
        self.prepared_frames = {}
        self.previous_cut = None
        self.qwen_context = None
        self.qwen_url = None
        self.lease_check = None

    def get_detector(self):
        if self.detector is None:
            self.detector = anime_models.AniRefDetector(
                os.environ.get('ANIREF_MODEL_PATH', '/models/aniref/AniRef40000-s-epoch40.pt'),
                os.environ.get('ANIREF_DEVICE', '0'))
        return self.detector

    def get_reid(self):
        if self.reid is None:
            self.reid = anime_models.AnimeReId(os.environ.get('ANIME_REID_MODEL_PATH', '/models/anime-reid/dino_v3.onnx'))
        return self.reid

    def get_qwen(self, lease_check):
        self.lease_check = lease_check
        if self.qwen_url is None:
            context = frame_model.frame_server(
                os.environ.get('QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'),
                os.environ.get('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
                os.environ.get('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf'),
                cancel_check=lambda: self.lease_check())
            self.qwen_url = context.__enter__()
            self.qwen_context = context
        return self.qwen_url

    def close(self):
        import gc
        if self.qwen_context:
            self.qwen_context.__exit__(None, None, None)
        self.qwen_context = self.qwen_url = None
        self.detector = self.reid = None
        self.reference_images.clear()
        self.prepared_frames.clear()
        self.previous_cut = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _crop_jpeg(image, box):
    crop = image.crop(tuple(box)).convert('RGB')
    crop.thumbnail((512, 512))
    buffer = io.BytesIO()
    crop.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def _open_jpeg(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        return image.convert('RGB')


def _valid_cut(conn, job_id, owner, episode_id, cut):
    worker.require_lease(conn, job_id, owner)
    current = conn.execute('SELECT start_time,end_time,active FROM cuts WHERE id=? AND episode_id=?', (cut['id'], episode_id)).fetchone()
    if not current or not current['active'] or current['start_time'] != cut['start_time'] or current['end_time'] != cut['end_time']:
        raise RuntimeError('Cut boundaries changed; run character detection again')


def _extract_frames(original, cut, scratch, saved=None):
    from PIL import Image
    times = media.cut_frame_times(cut['start_time'], cut['end_time'])
    frames, images = [], []
    for index, position in enumerate(('first', 'last')):
        existing = next((frame for frame in saved or [] if frame['position'] == position), None)
        target = Path(scratch) / f'{position}.jpg'
        actual = media.extract_cut_frame(original, target, existing['time'] if existing else times[index],
            cut['start_time'], cut['end_time'], width=1280, last=(position == 'last' and existing is None),
            seek_tolerance=.000002 if existing else 0)
        with Image.open(target) as source:
            image = source.convert('RGB')
        if existing and list(image.size) != existing['image_size']:
            raise RuntimeError('Detection frame dimensions changed; run character detection again')
        frames.append(existing or {'position': position, 'time': actual, 'image_size': list(image.size)})
        images.append(image)
    return frames, images


def _references(job, owner, payload, runtime, scratch):
    references = []
    for index, ref in enumerate(payload.get('references', [])):
        # Only user-saved references are eligible, including after deletion.
        if not db.one('SELECT id FROM character_image_references WHERE id=? AND character_context_id=?',
                      (ref['id'], ref['character_context_id'])):
            continue
        image = runtime.reference_images.get(ref['id'])
        if image is None:
            source = media.episode_dir(ref['project_id'], ref['episode_id']) / f"original{ref['original_ext']}"
            image = character_detection.crop_frame(source, ref['frame'], ref['cut_start'], ref['cut_end'], ref['box'], scratch, f'ref-{index}')
            runtime.reference_images[ref['id']] = image
        references.append({**ref, 'image': image})
    if not references:
        return []
    model = runtime.get_reid()
    missing = []
    with db.connect() as conn:
        worker.require_lease(conn, job['id'], owner)
        for ref in references:
            vector = anime_matching.read_cached_reference(conn, ref['id'], anime_models.REID_MODEL_ID,
                                                          model.revision, anime_models.REID_PREPROCESSING)
            if vector is None:
                missing.append(ref)
            else:
                ref['embedding'] = vector
    if missing:
        vectors = model.embed([_open_jpeg(ref['image']) for ref in missing])
        if len(vectors) != len(missing):
            raise RuntimeError('Anime Re-ID returned an unexpected number of reference embeddings')
        with db.connect() as conn:
            worker.require_lease(conn, job['id'], owner)
            for ref, vector in zip(missing, vectors):
                ref['embedding'] = vector
                if conn.execute('SELECT 1 FROM character_image_references WHERE id=? AND character_context_id=?',
                                (ref['id'], ref['character_context_id'])).fetchone():
                    anime_matching.write_cached_reference(conn, ref['id'], anime_models.REID_MODEL_ID,
                        model.revision, anime_models.REID_PREPROCESSING, vector)
    # If a reference was removed during embedding, do not match it.
    return [ref for ref in references if db.one('SELECT id FROM character_image_references WHERE id=? AND character_context_id=?',
                                                (ref['id'], ref['character_context_id']))]


def _review(runtime, image, references, lease_check):
    ids = list(dict.fromkeys(ref['character_context_id'] for ref in references))
    results = []
    server = runtime.get_qwen(lease_check)
    for offset in range(0, len(ids), 2):
        lease_check()
        selected = []
        for character_id in ids[offset:offset + 2]:
            selected.extend([ref for ref in references if ref['character_context_id'] == character_id][:2])
        result = frame_model.match_character(server, image, selected, cancel_check=lease_check)
        if result and result['confidence'] == 'high' and result['character_id']:
            results.append(result)
    return results[0] if len(results) == 1 else None


def _target_vectors(job_id, owner, detections, crops, model):
    vectors, missing = [None] * len(detections), []
    with db.connect() as conn:
        worker.require_lease(conn, job_id, owner)
        for index, detection in enumerate(detections):
            vector = anime_matching.read_cached_detection(conn, detection['id'], anime_models.REID_MODEL_ID,
                                                          model.revision, anime_models.REID_PREPROCESSING)
            if vector is None:
                missing.append(index)
            else:
                vectors[index] = vector
    if missing:
        computed = model.embed([_open_jpeg(crops[index]) for index in missing])
        if len(computed) != len(missing):
            raise RuntimeError('Anime Re-ID returned an unexpected number of target embeddings')
        with db.connect() as conn:
            worker.require_lease(conn, job_id, owner)
            for index, vector in zip(missing, computed):
                vectors[index] = vector
                if conn.execute('SELECT 1 FROM character_detections WHERE id=?', (detections[index]['id'],)).fetchone():
                    anime_matching.write_cached_detection(conn, detections[index]['id'], anime_models.REID_MODEL_ID,
                        model.revision, anime_models.REID_PREPROCESSING, vector)
    return vectors


def process(job_id, owner, runtime):
    """Process one restartable cut job using a bulk runtime shared across jobs."""
    from . import inference_worker
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    try:
        payload = json.loads(job['payload_json'])
        cut = payload['cut']
        episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
        with db.connect() as conn:
            _valid_cut(conn, job_id, owner, episode['id'], cut)
        original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
        lease_check = lambda: inference_worker.check_summary_lease(job_id, owner)
        next_tracks = []
        with tempfile.TemporaryDirectory(prefix='framewise-anime-') as scratch:
            run = db.one('SELECT * FROM character_detection_runs WHERE job_id=?', (job_id,))
            prepared = runtime.prepared_frames.pop(job_id, None)
            frames, images = prepared if prepared else _extract_frames(original, cut, scratch, json.loads(run['frames_json']) if run else None)
            if not run:
                worker.stage(job_id, owner, episode['id'], 'Detecting characters with AniRef')
                detector = runtime.get_detector()
                boxes = detector.detect(images)
                if len(boxes) != len(frames):
                    raise RuntimeError('AniRef returned an unexpected number of frames')
                for frame, frame_boxes in zip(frames, boxes):
                    frame['boxes'] = frame_boxes
                with db.connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    _valid_cut(conn, job_id, owner, episode['id'], cut)
                    run_id = str(uuid.uuid4())
                    conn.execute('INSERT INTO character_detection_runs(id,cut_id,job_id,cut_start,cut_end,frames_json,model_id,model_revision) VALUES (?,?,?,?,?,?,?,?)',
                        (run_id, cut['id'], job_id, cut['start_time'], cut['end_time'], json.dumps(frames), anime_models.ANIREF_MODEL_ID, detector.revision))
                    for frame in frames:
                        for box in frame['boxes']:
                            conn.execute('INSERT INTO character_detections(id,run_id,position,box_json) VALUES (?,?,?,?)',
                                (str(uuid.uuid4()), run_id, frame['position'], json.dumps(box)))
                run = db.one('SELECT * FROM character_detection_runs WHERE id=?', (run_id,))
            detections = db.all_rows('SELECT * FROM character_detections WHERE run_id=? ORDER BY rowid', (run['id'],))
            pending = [d for d in detections if d['label_source'] is None]
            if pending and payload.get('references'):
                worker.stage(job_id, owner, episode['id'], 'Preparing character references')
                references = _references(job, owner, payload, runtime, scratch)
                if references:
                    worker.stage(job_id, owner, episode['id'], 'Comparing character embeddings')
                    # Include previously matched detections on retry for short temporal links.
                    crops = [_crop_jpeg(images[0 if d['position'] == 'first' else 1], json.loads(d['box_json'])) for d in detections]
                    vectors = _target_vectors(job_id, owner, detections, crops, runtime.get_reid())
                    observations = []
                    for d, vector in zip(detections, vectors):
                        frame = next(f for f in frames if f['position'] == d['position'])
                        observations.append({'cut_id': cut['id'], 'time': frame['time'], 'box': json.loads(d['box_json']),
                                             'image_size': frame['image_size'], 'embedding': vector})
                    first_indices = [i for i, d in enumerate(detections) if d['position'] == 'first']
                    last_indices = [i for i, d in enumerate(detections) if d['position'] == 'last']
                    links = anime_matching.match_tracks([observations[i] for i in first_indices], [observations[i] for i in last_indices])
                    linked = {last_indices[j]: first_indices[i] for j, i in links.items()}
                    previous = getattr(runtime, 'previous_cut', None)
                    across = {}
                    previous_tracks = []
                    if (previous and previous['episode_id'] == episode['id'] and
                            abs(previous['end_time'] - cut['start_time']) <= .000001):
                        # A previous identity must appear only once on its boundary;
                        # otherwise the source track is already ambiguous.
                        all_tracks = previous['tracks']
                        previous_tracks = [track for track in all_tracks if sum(t['character_id'] == track['character_id'] for t in all_tracks) == 1]
                        matches = anime_matching.match_adjacent_appearances(previous_tracks, [observations[i] for i in first_indices])
                        across = {first_indices[j]: previous_tracks[i] for j, i in matches.items()}
                    for index, detection in enumerate(detections):
                        if detection['label_source'] is not None:
                            continue
                        lease_check()
                        ranked = anime_matching.rank_references(vectors[index], references)
                        character_id = anime_matching.confident_match(ranked)
                        model_id, revision = anime_models.REID_MODEL_ID, runtime.get_reid().revision
                        reason = f'Anime Re-ID cosine similarity {ranked[0][1]:.3f}' if character_id else ''
                        prior = detections[linked[index]] if index in linked else None
                        if not character_id and prior and prior['character_context_id']:
                            character_id = prior['character_context_id']
                            model_id, revision = anime_models.REID_MODEL_ID, runtime.get_reid().revision
                            reason = 'Short same-cut track with matching appearance and position'
                        if not character_id and index in across:
                            character_id = across[index]['character_id']
                            model_id, revision = anime_models.REID_MODEL_ID, runtime.get_reid().revision
                            reason = 'Strong unique appearance match to the preceding cut boundary'
                        if not character_id:
                            worker.stage(job_id, owner, episode['id'], f'Reviewing ambiguous character {index + 1}/{len(detections)}')
                            match = _review(runtime, crops[index], references, lease_check)
                            if match:
                                character_id, reason = match['character_id'], match['reason']
                                model_id, revision = frame_model.MODEL_ID, frame_model.MODEL_REVISION
                        if character_id:
                            with db.connect() as conn:
                                _valid_cut(conn, job_id, owner, episode['id'], cut)
                                # Do not assign identities whose reference or character was removed.
                                if conn.execute('''SELECT 1 FROM context_items c JOIN character_image_references r ON r.character_context_id=c.id
                                    WHERE c.id=? AND c.project_id=? AND c.kind='character' AND r.id IN (%s)''' % ','.join('?' * len(references)),
                                    (character_id, episode['project_id'], *(ref['id'] for ref in references))).fetchone():
                                    updated = conn.execute("UPDATE character_detections SET character_context_id=?,label_source='auto',match_reason=?,match_model_id=?,match_model_revision=? WHERE id=? AND label_source IS NULL",
                                        (character_id, reason[:2000], model_id, revision, detection['id'])).rowcount
                                    if updated:
                                        detection['character_context_id'] = character_id
                                        detection['label_source'] = 'auto'
                    next_tracks = [{'character_id': detections[i]['character_context_id'], 'embedding': vectors[i]}
                                   for i in last_indices if detections[i]['character_context_id']]
            with db.connect() as conn:
                _valid_cut(conn, job_id, owner, episode['id'], cut)
                conn.execute("UPDATE jobs SET status='done',stage='Complete',error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
                worker.sync_episode_status(conn, episode['id'])
            runtime.previous_cut = {'episode_id': episode['id'], 'end_time': cut['end_time'], 'tracks': next_tracks}
    except Exception as exc:
        runtime.previous_cut = None
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
        inference_worker.log.exception('Bulk anime character detection failed for job %s', job_id)


def _claim_batch(first_job, first_owner, limit=8):
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
        candidates = conn.execute('''SELECT id FROM jobs WHERE episode_id=? AND pipeline='character_detect'
            AND status='queued' AND json_extract(payload_json,'$.workflow')='anime_bulk_v1'
            AND json_extract(payload_json,'$.batch_id')=? ORDER BY created_at,rowid LIMIT ?''',
            (first_job['episode_id'], batch_id, limit - 1)).fetchall()
        for row in candidates:
            owner = str(uuid.uuid4())
            changed = conn.execute("UPDATE jobs SET status='running',stage='Starting',error=NULL,lease_owner=?,lease_until=?,attempt_count=attempt_count+1,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='queued'",
                                   (owner, now + worker.LEASE_SECONDS, row['id'])).rowcount
            if changed:
                claimed.append((row['id'], owner))
    return claimed


def _detect_batch(claimed, runtime):
    """Detect boundary frames for several leased cuts in one GPU invocation."""
    from . import inference_worker
    prepared = []
    for job_id, owner in claimed:
        job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
        if not job or db.one('SELECT id FROM character_detection_runs WHERE job_id=?', (job_id,)):
            continue
        try:
            cut = json.loads(job['payload_json'])['cut']
            episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
            with db.connect() as conn:
                _valid_cut(conn, job_id, owner, episode['id'], cut)
            worker.stage(job_id, owner, episode['id'], 'Preparing frames for AniRef batch')
            original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
            with tempfile.TemporaryDirectory(prefix='framewise-anime-batch-') as scratch:
                frames, images = _extract_frames(original, cut, scratch)
            prepared.append((job, owner, cut, episode, frames, images))
        except Exception as exc:
            inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
            inference_worker.log.exception('Could not prepare cut for anime detection batch %s', job_id)
    if not prepared:
        return
    try:
        detector = runtime.get_detector()
        all_images = [image for _, _, _, _, _, images in prepared for image in images]
        all_boxes = detector.detect(all_images)
        if len(all_boxes) != len(all_images):
            raise RuntimeError('AniRef returned an unexpected number of frames')
    except Exception:
        # Retry cuts individually so a malformed frame does not fail the group.
        inference_worker.log.exception('AniRef batch failed; retrying cuts individually')
        return
    for index, (job, owner, cut, episode, frames, images) in enumerate(prepared):
        try:
            for frame, boxes in zip(frames, all_boxes[index * 2:index * 2 + 2]):
                frame['boxes'] = boxes
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                _valid_cut(conn, job['id'], owner, episode['id'], cut)
                run_id = str(uuid.uuid4())
                conn.execute('INSERT INTO character_detection_runs(id,cut_id,job_id,cut_start,cut_end,frames_json,model_id,model_revision) VALUES (?,?,?,?,?,?,?,?)',
                    (run_id, cut['id'], job['id'], cut['start_time'], cut['end_time'], json.dumps(frames), anime_models.ANIREF_MODEL_ID, detector.revision))
                for frame in frames:
                    for box in frame['boxes']:
                        conn.execute('INSERT INTO character_detections(id,run_id,position,box_json) VALUES (?,?,?,?)',
                            (str(uuid.uuid4()), run_id, frame['position'], json.dumps(box)))
            runtime.prepared_frames[job['id']] = (frames, images)
        except Exception as exc:
            inference_worker.fail_job(job['id'], owner, episode['id'], exc)
            inference_worker.log.exception('Could not save anime detection batch result %s', job['id'])


def process_batch(first_job, first_owner, runtime, batch_size=8):
    """Run several cut jobs with separate leases and shared GPU models."""
    claimed = _claim_batch(first_job, first_owner, batch_size)
    extra_beats = []
    try:
        for job_id, owner in claimed[1:]:
            done = threading.Event()
            thread = threading.Thread(target=worker.heartbeat, args=(job_id, owner, done), daemon=True)
            thread.start()
            extra_beats.append((done, thread))
        _detect_batch(claimed, runtime)
        for job_id, owner in claimed:
            process(job_id, owner, runtime)
    finally:
        for done, _ in extra_beats:
            done.set()
        for _, thread in extra_beats:
            thread.join(timeout=1)
