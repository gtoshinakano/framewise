"""Episode audio ranges and the interval shape used by voice samples."""
import hashlib
import json
import tempfile
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from . import db, media


class VoiceRange(BaseModel):
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)


def rows(conn, episode_id, *, user_only=False):
    where = " AND source='user'" if user_only else ''
    return db.all_rows(f'SELECT * FROM episode_audio_ranges WHERE episode_id=?{where} ORDER BY start_time,end_time,id',
                       (episode_id,), conn=conn)


def public(conn, episode_id):
    return [{key: value for key, value in row.items() if key not in ('suppressed', 'matched_episodes_json')}
            | {'matched_episodes': json.loads(row['matched_episodes_json'] or '[]')}
            for row in rows(conn, episode_id) if not row['suppressed']]


def add(conn, episode_id, start, end, kind):
    conn.execute("""INSERT INTO episode_audio_ranges(id,episode_id,start_time,end_time,kind,source)
        VALUES (?,?,?,?,?,'user')""", (str(uuid.uuid4()), episode_id, start, end, kind))
    hide_overlaps(conn, episode_id, start, end)
    mark_stale(conn, episode_id)


def hide_overlaps(conn, episode_id, start, end):
    # Record excluded suggestions so a later scan cannot undo an edit.
    conn.execute("""UPDATE episode_audio_ranges SET source='user',suppressed=1
        WHERE episode_id=? AND source='fingerprint' AND start_time<? AND end_time>?""", (episode_id, end, start))


def remove(conn, episode_id, range_id):
    row = db.one('SELECT * FROM episode_audio_ranges WHERE episode_id=? AND id=? AND suppressed=0',
                 (episode_id, range_id), conn=conn)
    if not row:
        return False
    if row['source'] == 'fingerprint':
        conn.execute("UPDATE episode_audio_ranges SET source='user',suppressed=1 WHERE id=?", (range_id,))
    else:
        conn.execute('DELETE FROM episode_audio_ranges WHERE id=?', (range_id,))
    mark_stale(conn, episode_id)
    return True


def mark_stale(conn, episode_id):
    conn.execute("""UPDATE episodes SET transcript_stale_reason='Excluded audio ranges changed'
        WHERE id=? AND EXISTS(SELECT 1 FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0)""",
                 (episode_id, episode_id))


def audio_files(conn, project_id):
    files = {}
    for row in conn.execute('SELECT id FROM episodes WHERE project_id=? ORDER BY rowid', (project_id,)):
        path = media.episode_dir(project_id, row['id']) / 'audio.wav'
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        files[row['id']] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    return files


def cached_fingerprint(path, identity):
    import numpy as np
    import soundfile
    from .audio_fingerprint import fingerprint

    key = hashlib.sha256(json.dumps(['v1', str(path), identity]).encode()).hexdigest()
    directory = db.DATA_DIR / 'cache' / 'fingerprints'
    target = directory / f'{key}.npy'
    try:
        return np.load(target, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        samples, rate = soundfile.read(path, dtype='float32', always_2d=True)
        result = fingerprint(samples.mean(axis=1), rate)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.npy', delete=False) as temp:
        np.save(temp, result, allow_pickle=False)
    Path(temp.name).replace(target)
    return result


def detect(episode, files, overrides):
    from . import audio_fingerprint as fp
    episode_id, project_id = episode['id'], episode['project_id']
    if len(files) < 3 or episode_id not in files:
        return []
    own = cached_fingerprint(media.episode_dir(project_id, episode_id) / 'audio.wav', files[episode_id])
    partners = [key for key in files if key != episode_id]
    pairs = {(0, i): fp.match_pair(own, cached_fingerprint(media.episode_dir(project_id, key) / 'audio.wav', files[key]))
             for i, key in enumerate(partners, 1)}
    found = fp.consolidate(pairs, len(files), min_support=2).get(0, [])
    result = []
    for item in fp.label_ranges(found, episode['duration']):
        start, end = max(0., item['start']), min(episode['duration'], item['end'])
        if end <= start or any(row['start_time'] < end and row['end_time'] > start for row in overrides):
            continue
        result.append({'start_time': start, 'end_time': end, 'kind': item['kind'] if item['kind'] in ('OP', 'ED') else 'repeat',
                       'matched_episodes': [partners[i - 1] for i in item['partners']]})
    return result


def replace_fingerprints(conn, episode_id, ranges):
    conn.execute("DELETE FROM episode_audio_ranges WHERE episode_id=? AND source='fingerprint'", (episode_id,))
    conn.executemany("""INSERT INTO episode_audio_ranges(id,episode_id,start_time,end_time,kind,source,matched_episodes_json)
        VALUES (?,?,?,?,?,'fingerprint',?)""",
        [(str(uuid.uuid4()), episode_id, row['start_time'], row['end_time'], row['kind'], json.dumps(row['matched_episodes'])) for row in ranges])
