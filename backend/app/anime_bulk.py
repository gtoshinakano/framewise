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

from . import anime_gallery, anime_matching, anime_models, db, media, worker

log = logging.getLogger('framewise.character_detect')
PHASES = ('boxes', 'embeddings', 'complete')
SAME_TRACK_REASON = 'Short same-cut track with matching appearance and position'
ADJACENT_REASON = 'Strong unique appearance match to the preceding cut boundary'
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
    """Temporary boundary frames shared by the phases; removed when detection completes."""
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
        self.previous_cut = None

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

    def get_gallery(self, project_id, lease_check=None, timings=None):
        """Confirmed examples for the project, reloaded once per batch so new labels count."""
        if self.gallery is None or self.gallery_project != project_id:
            with timed('gallery', timings, project=project_id):
                self.gallery = anime_gallery.load(project_id, self.get_embedder, lease_check)
            self.gallery_project = project_id
        return self.gallery

    def close(self):
        import gc
        self.detector = self.embedder = self.gallery = None
        self.prepared_frames.clear()
        self.previous_cut = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _valid_cut(conn, job_id, owner, episode_id, cut):
    worker.require_lease(conn, job_id, owner)
    current = conn.execute('SELECT start_time,end_time,active FROM cuts WHERE id=? AND episode_id=?', (cut['id'], episode_id)).fetchone()
    if not current or not current['active'] or current['start_time'] != cut['start_time'] or current['end_time'] != cut['end_time']:
        raise RuntimeError('Cut boundaries changed; run character detection again')


def _extract_frames(original, cut, cache, saved=None):
    """Return frame metadata and RGB images, reusing cached temporary JPEGs when present."""
    from PIL import Image
    times = media.cut_frame_times(cut['start_time'], cut['end_time'])
    frames, images = [], []
    for index, position in enumerate(('first', 'last')):
        existing = next((frame for frame in saved or [] if frame['position'] == position), None)
        target = Path(cache) / f"{cut['id']}-{position}.jpg"
        if not (existing and target.is_file()):
            actual = media.extract_cut_frame(original, target, existing['time'] if existing else times[index],
                cut['start_time'], cut['end_time'], width=1280, last=(position == 'last' and existing is None),
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


def _save_run(job, owner, episode, cut, frames, detector, timings):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _valid_cut(conn, job['id'], owner, episode['id'], cut)
        run_id = str(uuid.uuid4())
        conn.execute('INSERT INTO character_detection_runs(id,cut_id,job_id,cut_start,cut_end,frames_json,model_id,model_revision,phase,timings_json) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (run_id, cut['id'], job['id'], cut['start_time'], cut['end_time'], json.dumps(frames), anime_models.ANIREF_MODEL_ID, detector.revision, 'boxes', json.dumps(timings)))
        for frame in frames:
            for box in frame['boxes']:
                conn.execute('INSERT INTO character_detections(id,run_id,position,box_json) VALUES (?,?,?,?)',
                    (str(uuid.uuid4()), run_id, frame['position'], json.dumps(box)))
    return run_id


def _set_phase(job_id, owner, run_id, phase, timings):
    with db.connect() as conn:
        worker.require_lease(conn, job_id, owner)
        conn.execute('UPDATE character_detection_runs SET phase=?,timings_json=? WHERE id=?', (phase, json.dumps(timings), run_id))


def latest_current_run(cut):
    return db.one('SELECT * FROM character_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=? ORDER BY rowid DESC LIMIT 1',
                  (cut['id'], cut['start_time'], cut['end_time']))


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
            worker.require_lease(conn, job_id, owner)
            for index, vector in zip(missing, computed):
                vectors[index] = vector
                if conn.execute('SELECT 1 FROM character_detections WHERE id=?', (detections[index]['id'],)).fetchone():
                    anime_matching.write_cached_detection(conn, detections[index]['id'], model.model_id,
                        model.revision, model.preprocessing, vector)
    return vectors


def _detection_crop(images, detection):
    """JPEG crop of one detection from its boundary frame (images are [first, last])."""
    return _crop_jpeg(images[0 if detection['position'] == 'first' else 1], json.loads(detection['box_json']))


def _observations(cut, frames, detections, vectors):
    """Pair each detection with its frame time so boxes can be linked across frames."""
    times = {frame['position']: frame for frame in frames}
    return [{'cut_id': cut['id'], 'time': times[d['position']]['time'], 'box': json.loads(d['box_json']),
             'image_size': times[d['position']]['image_size'], 'embedding': vector} for d, vector in zip(detections, vectors)]


def _same_cut_links(detections, observations, min_similarity=None):
    """Map each last-frame detection index to the first-frame one it clearly continues."""
    first = [i for i, d in enumerate(detections) if d['position'] == 'first']
    last = [i for i, d in enumerate(detections) if d['position'] == 'last']
    options = {} if min_similarity is None else {'min_similarity': min_similarity}
    links = anime_matching.match_tracks([observations[i] for i in first], [observations[i] for i in last], **options)
    return {last[j]: first[i] for j, i in links.items()}


def _finish(job_id, owner, episode_id, cut):
    with db.connect() as conn:
        _valid_cut(conn, job_id, owner, episode_id, cut)
        conn.execute("UPDATE jobs SET status='done',stage='Complete',error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        worker.sync_episode_status(conn, episode_id)


def _frame_size(frames, detection):
    return next(frame['image_size'] for frame in frames if frame['position'] == detection['position'])


class _Labeller:
    """Assigns automatic labels to one cut's boxes without touching confirmed ones."""

    def __init__(self, job_id, owner, episode, cut, detections):
        self.job_id, self.owner, self.episode, self.cut = job_id, owner, episode, cut
        self.detections = detections
        # A character appears at most once per frame.
        self.present = {'first': set(), 'last': set()}
        for detection in detections:
            if detection['character_context_id']:
                self.present[detection['position']].add(detection['character_context_id'])

    def free(self, index, character_id):
        detection = self.detections[index]
        return detection['label_source'] is None and character_id not in self.present[detection['position']]

    def assign(self, index, character_id, reason, model_id, revision):
        detection = self.detections[index]
        if not self.free(index, character_id):
            return False
        with db.connect() as conn:
            _valid_cut(conn, self.job_id, self.owner, self.episode['id'], self.cut)
            # Do not assign identities whose character was removed.
            if not conn.execute("SELECT 1 FROM context_items WHERE id=? AND project_id=? AND kind='character'",
                                (character_id, self.episode['project_id'])).fetchone():
                return False
            updated = conn.execute("UPDATE character_detections SET character_context_id=?,label_source='auto',match_reason=?,match_model_id=?,match_model_revision=? WHERE id=? AND label_source IS NULL",
                (character_id, reason[:2000], model_id, revision, detection['id'])).rowcount
        if updated:
            detection['character_context_id'] = character_id
            detection['label_source'] = 'auto'
            self.present[detection['position']].add(character_id)
        return bool(updated)

    def propagate(self, linked, model_id, revision):
        """Copy a label along each same-cut track, in either direction, until nothing changes."""
        changed = True
        while changed:
            changed = False
            for last, first in linked.items():
                for source, target in ((first, last), (last, first)):
                    character_id = self.detections[source]['character_context_id']
                    if character_id and self.assign(target, character_id, SAME_TRACK_REASON, model_id, revision):
                        changed = True


def _store_candidates(job_id, owner, detections, candidates):
    with db.connect() as conn:
        worker.require_lease(conn, job_id, owner)
        for index, ranked in candidates.items():
            conn.execute("UPDATE character_detections SET candidates_json=? WHERE id=? AND (label_source IS NULL OR label_source='auto')",
                         (json.dumps([{'character_id': cid, 'score': round(score, 3)} for cid, score in ranked[:3]]), detections[index]['id']))
            detections[index]['candidates_json'] = json.dumps([{'character_id': cid, 'score': round(score, 3)} for cid, score in ranked[:3]])


def _recognize(job_id, owner, runtime, episode, cut, frames, images, detections, lease_check, timings):
    """Embed the cut's boxes, match them with the gallery and propagate along tracks.

    Returns (linked, next_tracks) for the next cut.
    """
    gallery = runtime.get_gallery(episode['project_id'], lease_check, timings)
    if not len(gallery) or not detections:
        return {}, []
    model = runtime.get_embedder()
    defaults = model.defaults
    worker.stage(job_id, owner, episode['id'], f'Matching characters with {model.label}')
    with timed('embeddings', timings, cut=cut.get('cut_index'), detections=len(detections)):
        vectors = _target_vectors(job_id, owner, detections, [_detection_crop(images, d) for d in detections], model)
    observations = _observations(cut, frames, detections, vectors)
    linked = _same_cut_links(detections, observations, defaults['track'])
    labeller = _Labeller(job_id, owner, episode, cut, detections)
    threshold = anime_matching.env_fraction('ANIME_REID_MIN_SIMILARITY', defaults['similarity'])
    margin = anime_matching.env_fraction('ANIME_REID_MIN_MARGIN', defaults['margin'])
    candidates, proposals = {}, []
    for index, detection in enumerate(detections):
        if detection['label_source'] is not None:
            continue
        lease_check()
        ranked = gallery.rank(vectors[index])
        candidates[index] = ranked
        if anime_gallery.box_problem(json.loads(detection['box_json']), _frame_size(frames, detection)):
            continue  # too small to identify; keep only the suggestions
        character_id = anime_matching.confident_match(ranked, threshold, margin)
        if not character_id and defaults.get('wide'):
            character_id = anime_matching.confident_match(ranked, *defaults['wide'])
        if character_id:
            runner_up = f", next {gallery.names.get(ranked[1][0], 'other')} {ranked[1][1]:.2f}" if len(ranked) > 1 else ''
            proposals.append((ranked[0][1], index, character_id,
                              f'{model.label} similarity {ranked[0][1]:.2f} to your confirmed labels{runner_up}'))
    for score, index, character_id, reason in sorted(proposals, key=lambda item: -item[0]):
        labeller.assign(index, character_id, reason, model.model_id, model.revision)
    labeller.propagate(linked, model.model_id, model.revision)
    previous = runtime.previous_cut
    if previous and previous['episode_id'] == episode['id'] and abs(previous['end_time'] - cut['start_time']) <= .000001:
        # A previous identity must appear only once on its boundary;
        # otherwise the source track is already ambiguous.
        tracks = [track for track in previous['tracks'] if sum(t['character_id'] == track['character_id'] for t in previous['tracks']) == 1]
        first = [i for i, d in enumerate(detections) if d['position'] == 'first']
        matches = anime_matching.match_adjacent_appearances(tracks, [observations[i] for i in first], defaults['adjacent'])
        for j, i in matches.items():
            labeller.assign(first[j], tracks[i]['character_id'], ADJACENT_REASON, model.model_id, model.revision)
        labeller.propagate(linked, model.model_id, model.revision)
    _store_candidates(job_id, owner, detections, candidates)
    next_tracks = [{'character_id': d['character_context_id'], 'embedding': vectors[i]}
                   for i, d in enumerate(detections) if d['position'] == 'last' and d['character_context_id']]
    return linked, next_tracks


def process(job_id, owner, runtime):
    """Process one restartable cut job. Phases: 'boxes' (AniRef only), 'recognize' (gallery matching), or 'full'."""
    from . import inference_worker
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    try:
        payload = json.loads(job['payload_json'])
        mode = payload.get('phase', 'full')
        cut = payload['cut']
        episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
        started = time.perf_counter()
        with db.connect() as conn:
            _valid_cut(conn, job_id, owner, episode['id'], cut)
        original = media.episode_dir(episode['project_id'], episode['id']) / f"original{episode['original_ext']}"
        lease_check = lambda: inference_worker.check_summary_lease(job_id, owner)
        cache = frame_cache_dir(episode)
        cache.mkdir(parents=True, exist_ok=True)
        next_tracks = []
        label = f"cut={cut.get('cut_index')}"
        run = db.one('SELECT * FROM character_detection_runs WHERE job_id=?', (job_id,))
        if not run and mode == 'recognize':
            run = latest_current_run(cut)
            if not run:
                raise RuntimeError('Run Detect Character Boxes for this cut first')
        if run and mode == 'boxes':
            _finish(job_id, owner, episode['id'], cut)  # boxes were saved by the batch step
            return
        timings = json.loads(run['timings_json'] or '{}') if run else {}
        loaded = []

        def frames_and_images():
            if not loaded:
                prepared = runtime.prepared_frames.pop(job_id, None)
                with timed('frames', timings if not run else None, cut=cut.get('cut_index')):
                    loaded.append(prepared if prepared else _extract_frames(original, cut, cache, json.loads(run['frames_json']) if run else None))
            return loaded[0]

        if not run:
            frames, images = frames_and_images()
            worker.stage(job_id, owner, episode['id'], 'Detecting characters with AniRef')
            detector = runtime.get_detector()
            with timed('boxes', timings, cut=cut.get('cut_index')):
                boxes = detector.detect(images)
            if len(boxes) != len(frames):
                raise RuntimeError('AniRef returned an unexpected number of frames')
            for frame, frame_boxes in zip(frames, boxes):
                frame['boxes'] = frame_boxes
            run = db.one('SELECT * FROM character_detection_runs WHERE id=?', (_save_run(job, owner, episode, cut, frames, detector, timings),))
            if mode == 'boxes':
                _finish(job_id, owner, episode['id'], cut)
                return
        detections = db.all_rows('SELECT * FROM character_detections WHERE run_id=? ORDER BY rowid', (run['id'],))
        phase = run['phase']
        if phase == 'boxes':
            if any(d['label_source'] is None for d in detections):
                frames, images = frames_and_images()
                _, next_tracks = _recognize(job_id, owner, runtime, episode, cut, frames, images, detections, lease_check, timings)
            _set_phase(job_id, owner, run['id'], 'embeddings', timings)
            phase = 'embeddings'
        if phase == 'embeddings':
            _set_phase(job_id, owner, run['id'], 'complete', timings)
        _finish(job_id, owner, episode['id'], cut)
        clear_cut_frames(episode, cut['id'])
        runtime.previous_cut = {'episode_id': episode['id'], 'end_time': cut['end_time'], 'tracks': next_tracks}
        log.info('character-detect %s finished mode=%s total=%.2fs timings_ms=%s', label, mode, time.perf_counter() - started, timings)
    except Exception as exc:
        runtime.previous_cut = None
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
        inference_worker.log.exception('Bulk anime character detection failed for job %s', job_id)


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


def _prepare_frames(claimed, runtime):
    """Extract the boundary frames of every claimed cut that still needs them, in parallel."""
    from . import inference_worker
    work = []
    for job_id, owner in claimed:
        job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
        if not job:
            continue
        payload = json.loads(job['payload_json'])
        cut = payload['cut']
        run = db.one('SELECT * FROM character_detection_runs WHERE job_id=?', (job_id,))
        if not run and payload.get('phase') == 'recognize':
            run = latest_current_run(cut)
        if run and (payload.get('phase') == 'boxes' or run['phase'] == 'complete' or not db.one(
                'SELECT 1 FROM character_detections WHERE run_id=? AND label_source IS NULL LIMIT 1', (run['id'],))):
            continue  # nothing left to recognize in this cut
        episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
        work.append((job, owner, cut, episode, run))

    def extract(item):
        job, owner, cut, episode, run = item
        try:
            with db.connect() as conn:
                _valid_cut(conn, job['id'], owner, episode['id'], cut)
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
    """Detect boundary frames for several leased cuts in one GPU invocation."""
    from . import inference_worker
    prepared = []
    for job_id, owner in claimed:
        job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
        if not job or db.one('SELECT id FROM character_detection_runs WHERE job_id=?', (job_id,)):
            continue
        try:
            payload = json.loads(job['payload_json'])
            if payload.get('phase') == 'recognize':
                continue
            cut = payload['cut']
            episode = db.one('SELECT * FROM episodes WHERE id=?', (job['episode_id'],))
            with db.connect() as conn:
                _valid_cut(conn, job_id, owner, episode['id'], cut)
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
            prepared.append((job, owner, cut, episode, frames, images, timings))
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
        for job, owner, cut, episode, frames, images, timings in prepared:
            runtime.prepared_frames[job['id']] = (frames, images)
        return
    for index, (job, owner, cut, episode, frames, images, timings) in enumerate(prepared):
        try:
            for frame, boxes in zip(frames, all_boxes[index * 2:index * 2 + 2]):
                frame['boxes'] = boxes
            timings['boxes'] = per_cut_ms
            _save_run(job, owner, episode, cut, frames, detector, timings)
            if json.loads(job['payload_json']).get('phase') != 'boxes':
                runtime.prepared_frames[job['id']] = (frames, images)
        except Exception as exc:
            inference_worker.fail_job(job['id'], owner, episode['id'], exc)
            inference_worker.log.exception('Could not save anime detection batch result %s', job['id'])


def process_batch(first_job, first_owner, runtime, batch_size=BATCH_SIZE):
    """Run several cut jobs with separate leases, one model at a time across all of them."""
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
        for done, _ in extra_beats:
            done.set()
        for _, thread in extra_beats:
            thread.join(timeout=1)
