"""Prepare media and find cut boundaries using the original upload."""
import json
import tempfile
import uuid
from pathlib import Path

from . import cut_history, cut_previews, db, job_inputs, media, worker


def gather(conn, job):
    episode = db.one("SELECT id,project_id,original_ext,duration FROM episodes WHERE id=?",
                     (job['episode_id'],), conn=conn)
    if job['pipeline'] == 'prepare':
        return episode
    payload = json.loads(job['payload_json'])
    cut_id = job['unit_id'] or payload.get('cut_id')
    if cut_id:
        cut = db.one("SELECT id,start_time,end_time FROM cuts WHERE id=? AND episode_id=? AND active=1",
                     (cut_id, episode['id']), conn=conn)
        return (cut or {'skip': 'Cut archived or removed'}) if job['unit_id'] else {**episode, 'cut': cut}
    return episode


def inputs(conn, job):
    gathered = job.get('gathered') or gather(conn, job)
    return job_inputs.fingerprint(gathered)


def invalidate(conn, job):
    # Timing changes invalidate their consumers as part of replacing the boundaries.
    pass


def prepare(job, owner, path, settings):
    job_id, episode_id = job['id'], job['episode_id']
    with db.connect() as conn:
        conn.execute('BEGIN')
        snapshot = job_inputs.JobInputs(conn, job, owner)
    original = path / f"original{snapshot.data['original_ext']}"
    worker.stage(job_id, owner, episode_id, 'Probing video')
    info = media.probe(original)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=path) as directory:
        scratch = Path(directory)
        worker.stage(job_id, owner, episode_id, 'Creating preview video')
        media.make_proxy(original, scratch / 'proxy.mp4', settings['proxy_width'],
                         settings.get('proxy_crf', 28), settings.get('proxy_preset', 'veryfast'))
        worker.stage(job_id, owner, episode_id, 'Indexing frames')
        media.write_frame_manifest(scratch / 'frames', settings['frame_interval'], info['duration'])
        if info['has_audio']:
            worker.stage(job_id, owner, episode_id, 'Extracting audio')
            media.extract_audio(original, scratch / 'audio.wav')
            worker.stage(job_id, owner, episode_id, 'Building waveform')
            media.waveform(scratch / 'audio.wav', scratch / 'waveform.json', info['duration'])
        else:
            (scratch / 'waveform.json').write_text(json.dumps({'peaks': [], 'duration': info['duration']}))
        with snapshot.publish() as conn:
            worker.discard_obsolete_previews(path)
            (path / 'frames').mkdir(exist_ok=True)
            for name in ('proxy.mp4', 'waveform.json', 'frames/manifest.json'):
                (scratch / name).replace(path / name)
            if info['has_audio']:
                (scratch / 'audio.wav').replace(path / 'audio.wav')
            else:
                (path / 'audio.wav').unlink(missing_ok=True)
            worker.sync_chunks(episode_id, info['duration'], settings.get('chunk_duration', 60), conn=conn)
            conn.execute('UPDATE episodes SET duration=?,width=?,height=?,has_audio=? WHERE id=?',
                         (info['duration'], info['width'], info['height'], int(info['has_audio']), episode_id))
            conn.execute('UPDATE jobs SET input_json=? WHERE id=?',
                         (json.dumps({'fingerprint': snapshot.stamp, **snapshot.data}), job_id))
            if job['pipeline'] == 'prepare':
                cut_previews.finish(conn, job)


def split_cut(conn, episode_id, cut_id, segments):
    """Retain the first section's ID and restore prior IDs for rediscovered cuts."""
    if len(segments) <= 1:
        return []
    cut = conn.execute('SELECT * FROM cuts WHERE id=?', (cut_id,)).fetchone()
    before = cut_history.snapshot(conn, episode_id)
    conn.execute('UPDATE cuts SET end_time=?,first_frame_time=NULL,last_frame_time=NULL WHERE id=?', (segments[0][1], cut_id))
    later = conn.execute("""SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1
        AND cut_index>? ORDER BY cut_index DESC""", (episode_id, cut['cut_index'])).fetchall()
    for row in later:
        conn.execute('UPDATE cuts SET cut_index=? WHERE id=?', (row['cut_index'] + len(segments) - 1, row['id']))
    archived = conn.execute('SELECT * FROM cuts WHERE episode_id=? AND active=0 AND start_time>? AND start_time<?',
                            (episode_id, cut['start_time'], cut['end_time'])).fetchall()
    affected, used, created = [cut_id], set(), set()
    for offset, (start, end) in enumerate(segments[1:], 1):
        candidates = (row for row in archived if row['id'] not in used and abs(row['start_time'] - start) < .25)
        match = min(candidates, key=lambda row: abs(row['start_time'] - start), default=None)
        affected.append(save_cut(conn, episode_id, cut['cut_index'] + offset, start, end, match, created))
        conn.execute('UPDATE cuts SET first_frame_time=NULL,last_frame_time=NULL WHERE id=?', (affected[-1],))
        if match:
            used.add(match['id'])
    cut_history.record(conn, episode_id, 'detect', f"Detected {len(segments)} cuts inside cut {cut['cut_index'] + 1}",
                       cut_id, before, created)
    return affected


def save_cut(conn, episode_id, index, start, end, existing, created):
    if existing:
        cut_id = existing['id']
        conn.execute('UPDATE cuts SET active=1,cut_index=?,start_time=?,end_time=? WHERE id=?',
                     (index, start, end, cut_id))
    else:
        cut_id = str(uuid.uuid4())
        conn.execute('INSERT INTO cuts (id,episode_id,cut_index,start_time,end_time,created_at) VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)',
                     (cut_id, episode_id, index, start, end))
        created.add(cut_id)
    return cut_id


def replace_cuts(conn, episode_id, segments):
    old = conn.execute('SELECT * FROM cuts WHERE episode_id=? ORDER BY active DESC,start_time', (episode_id,)).fetchall()
    cut_history.clear(conn, episode_id)
    archive_index = min((cut['cut_index'] for cut in old), default=0) - 1
    active = {cut['id'] for cut in old if cut['active']}
    for cut_id in sorted(active):
        conn.execute('UPDATE cuts SET active=0,cut_index=? WHERE id=?', (archive_index, cut_id))
        archive_index -= 1
    used, changed, created = set(), [], set()
    for index, (start, end) in enumerate(segments):
        candidates = (cut for cut in old if cut['id'] not in used and abs(cut['start_time'] - start) < .25)
        match = min(candidates, key=lambda cut: (not cut['active'], abs(cut['start_time'] - start)), default=None)
        cut_id = save_cut(conn, episode_id, index, start, end, match, created)
        used.add(cut_id)
        if not match or not match['active'] or (match['start_time'], match['end_time']) != (start, end):
            changed.append(cut_id)
    return changed, active - used


def hardcuts(job, owner, episode, payload, path, settings):
    job_id, episode_id = job['id'], episode['id']
    proxy = path / 'proxy.mp4'
    targeted = bool(payload.get('cut_id'))
    if not proxy.exists():
        if targeted:
            raise RuntimeError('Preview video is missing; run Prepare before detecting cuts within a cut')
        prepare(job, owner, path, settings)
    affected = payload.get('detected_cut_ids')
    if affected is None:
        with db.connect() as conn:
            conn.execute('BEGIN')
            snapshot = job_inputs.JobInputs(conn, job, owner)
        if targeted:
            cut = snapshot.data['cut']
            if not cut:
                with snapshot.publish(invalidate=False) as conn:
                    cut_previews.finish(conn, job, 'Skipped · cut archived or removed')
                return
            worker.stage(job_id, owner, episode_id, 'Detecting hard cuts in selected cut')
            segments = media.detect_cuts_in_range(proxy, cut['start_time'], cut['end_time'], payload['scene_threshold'])
        else:
            worker.stage(job_id, owner, episode_id, 'Detecting hard cuts')
            duration = snapshot.data['duration'] or media.probe(path / f"original{episode['original_ext']}")['duration']
            segments = media.detect_cuts(proxy, duration, settings['scene_threshold'])
        with snapshot.publish() as conn:
            if targeted:
                affected, removed = split_cut(conn, episode_id, cut['id'], segments), ()
            else:
                worker.sync_chunks(episode_id, duration, settings.get('chunk_duration', 60), conn=conn)
                affected, removed = replace_cuts(conn, episode_id, segments)
            cancelled = cut_previews.after_timeline_edit(conn, episode, affected, removed, rebuilding=affected)
            payload['detected_cut_ids'] = affected
            conn.execute('UPDATE jobs SET payload_json=?,input_json=? WHERE id=?',
                         (json.dumps(payload), json.dumps({'fingerprint': snapshot.stamp, **snapshot.data}), job_id))
        worker.after_cancel(cancelled)
    worker.stage(job_id, owner, episode_id, 'Updating cut previews')
    cut_previews.rebuild(job, owner, path, affected if targeted else None)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job_id, owner)
        label = 'Complete · no new cuts found' if targeted and not affected else 'Complete'
        cut_previews.finish(conn, job, label)
