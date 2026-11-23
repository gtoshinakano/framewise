"""User-confirmed examples for bulk character recognition.

The gallery holds every box the user labelled by hand (``label_source =
'manual'``, which includes saved references) in the latest detection run of
each cut, plus every saved reference. Automatic labels and track links are
never added, so the gallery only grows from the user's own decisions.

Some confirmed boxes cannot identify anyone: a distant figure, a hand, a
shoe or a strand of hair. They stay labelled (analysis still uses them) but
are left out of matching:

* ``small``: the box covers too little of the frame to show a face.
* ``atypical``: it does not resemble its own character's other examples
  and looks more like another character.

The verdict is written to ``character_detections.gallery_status`` and
``gallery_note`` so the editor can show it; ``recognition_use`` (1 always,
0 never, NULL automatic) lets the user override it per box.
"""
import io
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import db, media

MIN_AREA = 0.006  # fraction of the frame
MIN_SIDE = 36     # pixels on the 1280-wide detection frame
TOP_K = 3         # a character's score is the mean of its best TOP_K example similarities
FRAME_THREADS = 6


def crop_jpeg(image, box):
    crop = image.crop(tuple(box)).convert('RGB')
    crop.thumbnail((512, 512))
    buffer = io.BytesIO()
    crop.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def open_jpeg(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        return image.convert('RGB')


def box_problem(box, image_size):
    """'small' when a box is too small to identify anyone, else None."""
    width, height = max(0, box[2] - box[0]), max(0, box[3] - box[1])
    frame_area = max(1, image_size[0] * image_size[1])
    if width * height / frame_area < MIN_AREA or min(width, height) < MIN_SIDE:
        return 'small'
    return None


def exemplar_rows(project_id, conn=None):
    """Confirmed boxes in each cut's latest run, plus saved references from any run. Unregistered cards stand for different
    people (unregistered.py): never an example, so recognition never proposes one."""
    return db.all_rows('''SELECT d.id,d.character_context_id,d.position,d.box_json,d.recognition_use,d.gallery_status,d.gallery_note,
            r.id AS run_id,r.frames_json,r.cut_start,r.cut_end,c.id AS cut_id,e.id AS episode_id,e.project_id,e.original_ext,
            EXISTS(SELECT 1 FROM character_image_references x WHERE x.detection_id=d.id) AS is_reference
        FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id JOIN cuts c ON c.id=r.cut_id
        JOIN episodes e ON e.id=c.episode_id
        JOIN context_items ci ON ci.id=d.character_context_id AND ci.kind='character' AND ci.project_id=e.project_id
            AND COALESCE(ci.unregistered,0)=0
        WHERE e.project_id=? AND d.label_source='manual' AND (
            r.rowid=(SELECT max(r2.rowid) FROM character_detection_runs r2 WHERE r2.cut_id=r.cut_id)
            OR EXISTS(SELECT 1 FROM character_image_references x WHERE x.detection_id=d.id))
        ORDER BY d.rowid''', (project_id,), conn=conn)


def evidence(conn, project_id):
    rows = exemplar_rows(project_id, conn)
    for row in rows:
        row.pop('gallery_status')
        row.pop('gallery_note')
    names = {row['id']: row['title'] for row in db.all_rows(
        "SELECT id,title FROM context_items WHERE project_id=? AND kind='character' AND COALESCE(unregistered,0)=0", (project_id,), conn=conn)}
    return rows, names


def extract_crops(rows, scratch, lease_check=None):
    """JPEG crops for detection rows, extracting each source frame once and in parallel."""
    from PIL import Image
    frames = {}
    for row in rows:
        frames.setdefault((row['run_id'], row['position']), row)

    def extract(key):
        row = frames[key]
        frame = next(f for f in json.loads(row['frames_json']) if f['position'] == row['position'])
        target = Path(scratch) / f'{key[0]}-{key[1]}.jpg'
        source = media.episode_dir(row['project_id'], row['episode_id']) / f"original{row['original_ext']}"
        media.extract_cut_frame(source, target, frame['time'], row['cut_start'], row['cut_end'], width=1280, seek_tolerance=.000002)
        with Image.open(target) as image:
            image = image.convert('RGB')
        target.unlink(missing_ok=True)
        if list(image.size) != frame['image_size']:
            raise RuntimeError('Reference frame dimensions changed; save a new reference')
        return key, image

    images = {}
    with ThreadPoolExecutor(FRAME_THREADS) as pool:
        for key, image in pool.map(extract, frames):
            if lease_check:
                lease_check()
            images[key] = image
    return {row['id']: crop_jpeg(images[(row['run_id'], row['position'])], json.loads(row['box_json'])) for row in rows}


def cached_vectors(conn, detection_ids, model):
    from . import anime_matching
    vectors = {}
    for start in range(0, len(detection_ids), 500):
        chunk = detection_ids[start:start + 500]
        for row in conn.execute(f'''SELECT detection_id,dimensions,embedding FROM character_detection_embeddings
                WHERE model_id=? AND model_revision=? AND preprocessing=? AND detection_id IN ({','.join('?' * len(chunk))})''',
                (model.model_id, model.revision, model.preprocessing, *chunk)):
            vectors[row['detection_id']] = anime_matching.unpack_embedding(row['embedding'], row['dimensions'])
    return vectors


def store_vectors(conn, vectors, model):
    from . import anime_matching
    for detection_id, vector in vectors.items():
        if conn.execute('SELECT 1 FROM character_detections WHERE id=?', (detection_id,)).fetchone():
            anime_matching.write_cached_detection(conn, detection_id, model.model_id, model.revision, model.preprocessing, vector)


class Gallery:
    """Embedded confirmed examples, scored per character."""

    def __init__(self, items, names=None):
        import numpy as np
        self.items = items
        self.names = names or {}
        self.characters = sorted({item['character_id'] for item in items})
        self.matrix = np.array([item['vector'] for item in items], dtype=np.float32) if items else np.zeros((0, 1), dtype=np.float32)
        self.owner = np.array([self.characters.index(item['character_id']) for item in items], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def similarities(self, vector):
        import numpy as np
        return self.matrix @ np.asarray(vector, dtype=np.float32)

    def rank(self, vector, top_k=TOP_K):
        """[(character_id, score)] best first; score is the mean of the top_k example similarities."""
        import numpy as np
        if not self.items:
            return []
        sims = self.similarities(vector)
        ranked = []
        for index, character_id in enumerate(self.characters):
            values = np.sort(sims[self.owner == index])[::-1][:top_k]
            ranked.append((character_id, float(values.mean())))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def best_examples(self, character_id, vectors, limit=2):
        """The example items of one character most similar to any of the given vectors (references first)."""
        import numpy as np
        indices = [i for i, item in enumerate(self.items) if item['character_id'] == character_id]
        if not indices:
            return []
        scores = np.max(self.matrix[indices] @ np.asarray(vectors, dtype=np.float32).T, axis=1) if len(vectors) else np.zeros(len(indices))
        order = sorted(range(len(indices)), key=lambda j: (not self.items[indices[j]]['is_reference'], -scores[j]))
        return [self.items[indices[j]] for j in order[:limit]]


def judge(items, defaults, top_k=TOP_K):
    """Mark examples that look more like another character than their own as atypical.

    Items are dicts with id, character_id, cut_id, vector, is_reference and
    forced (user said always use). Only examples of characters with at
    least three examples from other cuts can be judged. Returns {id: (status, note_character_id)}.
    """
    import numpy as np
    verdicts = {}
    if not items:
        return verdicts
    matrix = np.array([item['vector'] for item in items], dtype=np.float32)
    sims = matrix @ matrix.T
    characters = sorted({item['character_id'] for item in items})
    owner = np.array([characters.index(item['character_id']) for item in items])
    cuts = np.array([item['cut_id'] for item in items], dtype=object)
    for i, item in enumerate(items):
        if item['is_reference'] or item['forced']:
            verdicts[item['id']] = ('used', None)
            continue
        other_cut = cuts != item['cut_id']  # same-cut boxes are near duplicates, not evidence
        scores = {}
        for index, character_id in enumerate(characters):
            peers = sims[i, (owner == index) & other_cut]
            if len(peers):
                scores[character_id] = (float(np.sort(peers)[::-1][:top_k].mean()), len(peers))
        own = scores.get(item['character_id'])
        if not own or own[1] < 3:
            verdicts[item['id']] = ('used', None)
            continue
        rival = max(((score, cid) for cid, (score, _) in scores.items() if cid != item['character_id']), default=None)
        if own[0] < defaults['atypical'] and rival and rival[0] > own[0]:
            verdicts[item['id']] = ('atypical', rival[1])
        else:
            verdicts[item['id']] = ('used', None)
    return verdicts


def load(project_id, get_model, lease_check=None, inputs=None):
    """Embed (or read cached vectors for) the project's confirmed examples and return the usable Gallery.

    get_model is called only when there is something to embed or look up,
    so a project without confirmed labels never loads the embedder.
    """
    if inputs is None:
        with db.connect() as conn:
            conn.execute('BEGIN')
            inputs = evidence(conn, project_id)
    rows, names = inputs
    statuses, candidates = {}, []
    for row in rows:
        if row['recognition_use'] == 0:
            statuses[row['id']] = ('excluded', None)
            continue
        frame = next((f for f in json.loads(row['frames_json']) if f['position'] == row['position']), None)
        if frame is None:
            continue
        forced = row['recognition_use'] == 1
        if not (forced or row['is_reference']) and box_problem(json.loads(row['box_json']), frame['image_size']):
            statuses[row['id']] = ('small', None)
            continue
        candidates.append(row)
    if not candidates:
        _write_statuses(rows, statuses, names)
        return Gallery([], names)
    model = get_model()
    with db.connect() as conn:
        vectors = cached_vectors(conn, [row['id'] for row in candidates], model)
    missing = [row for row in candidates if row['id'] not in vectors]
    if missing:
        with tempfile.TemporaryDirectory(prefix='framewise-gallery-') as scratch:
            crops = extract_crops(missing, scratch, lease_check)
        computed = model.embed([open_jpeg(crops[row['id']]) for row in missing])
        if len(computed) != len(missing):
            raise RuntimeError('Character embedder returned an unexpected number of gallery vectors')
        fresh = {row['id']: vector for row, vector in zip(missing, computed)}
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            valid = {row['id']: fresh[row['id']] for row in missing if conn.execute(
                'SELECT 1 FROM character_detections WHERE id=? AND box_json=?', (row['id'], row['box_json'])).fetchone()}
            store_vectors(conn, valid, model)
        vectors.update(fresh)
    items = [{'id': row['id'], 'character_id': row['character_context_id'], 'cut_id': row['cut_id'], 'vector': vectors[row['id']],
              'is_reference': bool(row['is_reference']), 'forced': row['recognition_use'] == 1, 'row': row} for row in candidates]
    for detection_id, verdict in judge(items, model.defaults).items():
        statuses[detection_id] = verdict
    _write_statuses(rows, statuses, names)
    usable = [item for item in items if statuses.get(item['id'], ('used',))[0] == 'used']
    return Gallery(usable, names)


def _note(status, rival, names):
    if status == 'small':
        return 'Too small to identify anyone'
    if status == 'atypical':
        return f"Looks more like {names.get(rival, 'another character')} than its own examples"
    if status == 'excluded':
        return 'You chose not to use this box for matching'
    return None


def _write_statuses(rows, statuses, names):
    changes = []
    for row in rows:
        status, rival = statuses.get(row['id'], (None, None))
        if status is None:
            continue
        note = _note(status, rival, names)
        changes.append((status, note, row['id'], row['box_json'], row['character_context_id'], row['recognition_use'], status, note))
    if changes:
        with db.connect() as conn:
            conn.executemany('''UPDATE character_detections SET gallery_status=?,gallery_note=?
                WHERE id=? AND box_json=? AND character_context_id IS ? AND recognition_use IS ?
                AND (gallery_status IS NOT ? OR gallery_note IS NOT ?)''', changes)
