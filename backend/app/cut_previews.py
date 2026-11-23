"""Preview builds and queued one-frame boundary moves."""
import json
import tempfile
from pathlib import Path

from . import agent_log, cut_history, db, job_inputs, job_kinds, media, structured_results, worker


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    cut_id = job['unit_id'] or payload.get('cut_id')
    cut = db.one('SELECT id,start_time,end_time,active FROM cuts WHERE id=? AND episode_id=?', (cut_id, job['episode_id']), conn=conn)
    if not cut or not cut['active']:
        return {'skip': 'Cut archived or removed'}
    if not payload.get('frame_step'):
        return cut
    position = db.one('SELECT cut_index,last_frame_time FROM cuts WHERE id=?', (cut_id,), conn=conn)
    following = db.one('SELECT id,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?',
                       (job['episode_id'], position['cut_index'] + 1), conn=conn)
    if not following:
        return {'skip': 'There is no following cut at this boundary'}
    if abs(following['start_time'] - cut['end_time']) > .000001:
        return {'skip': 'The cuts do not share a boundary'}
    return {**cut, **position, 'following': following}


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    if job.get('updated_boundaries'):
        structured_results.invalidate_cut_timing(conn, job['updated_boundaries'])


def finish(conn, job, stage='Complete'):
    conn.execute("""UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,
        updated_at=CURRENT_TIMESTAMP WHERE id=?""", (stage, job['id']))
    worker.sync_episode_status(conn, job['episode_id'])


def move(job, owner, payload, path):
    job_id = job['id']
    with db.connect() as conn:
        conn.execute('BEGIN')
        worker.require_lease(conn, job_id, owner)
        if payload.get('frame_step_done'):
            finish(conn, job)
            return
        snapshot = job_inputs.JobInputs(conn, job, owner)
    cut = snapshot.data
    if cut.get('skip'):
        with snapshot.publish(invalidate=False) as conn:
            finish(conn, job, f"Skipped · {cut['skip']}")
        return
    following, step = cut['following'], payload['frame_step']
    worker.stage(job_id, owner, job['episode_id'], 'Moving cut boundary by one frame')
    end_anchor = payload.get('frame_anchor', 'end') == 'end' and cut['last_frame_time'] is not None
    anchor = cut['last_frame_time'] if end_anchor else cut['end_time']
    proxy = path / 'proxy.mp4'
    boundary = media.step_frame_boundary(proxy, anchor, step, after_stepped_frame=end_anchor)
    if not cut['start_time'] < boundary < following['end_time']:
        raise RuntimeError('Moving the boundary would leave a cut without a positive duration')
    with tempfile.TemporaryDirectory(prefix='framewise-boundaries-') as scratch:
        last = media.extract_cut_frame(proxy, Path(scratch) / 'last.jpg', boundary, cut['start_time'], boundary, last=True, probe_padding=1)
        # Seek just before the boundary so FFmpeg's rounded seek cannot skip the first frame.
        first = media.extract_cut_frame(proxy, Path(scratch) / 'first.jpg', boundary,
                                       max(0, boundary - .000002), following['end_time'], seek_tolerance=.000002)
    with snapshot.publish(updated_boundaries=[cut['id'], following['id']]) as conn:
        before = cut_history.snapshot(conn, job['episode_id'])
        conn.execute('UPDATE cuts SET end_time=?,last_frame_time=? WHERE id=?', (boundary, last, cut['id']))
        conn.execute('UPDATE cuts SET start_time=?,first_frame_time=? WHERE id=?', (boundary, first, following['id']))
        direction = 'earlier' if step < 0 else 'later'
        cut_history.record(conn, job['episode_id'], 'boundary',
                           f"Moved the end of cut {cut['cut_index'] + 1} one frame {direction}", cut['id'], before)
        finish(conn, job)
    (path / 'thumbnails' / f"{cut['id']}_last.jpg").unlink(missing_ok=True)
    (path / 'thumbnails' / f"{following['id']}_first.jpg").unlink(missing_ok=True)


def process(job, owner, payload, path):
    if not (path / 'proxy.mp4').exists():
        raise RuntimeError('Preview video is missing; run Prepare before updating cut frames')
    if payload.get('frame_step'):
        move(job, owner, payload, path)
        return
    worker.stage(job['id'], owner, job['episode_id'], 'Updating cut frames')
    rebuild(job, owner, path, payload.get('cut_ids'))
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job['id'], owner)
        finish(conn, job)


def rebuild(job, owner, path, requested=None):
    with db.connect() as conn:
        conn.execute('BEGIN')
        todo = [row['id'] for row in conn.execute("""SELECT id FROM cuts WHERE episode_id=? AND active=1
            AND (first_frame_time IS NULL OR last_frame_time IS NULL) ORDER BY cut_index""", (job['episode_id'],))]
        requested = set(todo if requested is None else requested)
        units = [job_inputs.JobInputs(conn, job, owner, cut_id) for cut_id in todo if cut_id in requested]
    with tempfile.TemporaryDirectory(prefix='framewise-boundaries-') as scratch:
        proxy = path / 'proxy.mp4'
        for unit in units:
            cut = unit.data
            first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
            first = media.extract_cut_frame(proxy, Path(scratch) / 'first.jpg', first, cut['start_time'], cut['end_time'])
            last = media.extract_cut_frame(proxy, Path(scratch) / 'last.jpg', last, cut['start_time'], cut['end_time'], last=True, probe_padding=1)
            with unit.publish(invalidate=False) as conn:
                conn.execute('UPDATE cuts SET first_frame_time=?,last_frame_time=? WHERE id=?', (first, last, cut['id']))
            for position in ('first', 'last'):
                (path / 'thumbnails' / f"{cut['id']}_{position}.jpg").unlink(missing_ok=True)


def after_timeline_edit(conn, episode, changed, removed=(), *, rebuilding=()):
    """Replace previews and cancel jobs whose entire target was archived."""
    changed, removed = set(changed), set(removed)
    obsolete = changed | removed
    if not obsolete:
        return []
    candidates = conn.execute("SELECT * FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (episode['id'],)).fetchall()
    cancelled = []
    rebuild = set(changed)
    for job in candidates:
        payload = json.loads(job['payload_json'])
        targets = {ref['id'] for ref in job_kinds.kind_of(job['pipeline']).cuts(payload)}
        if job['pipeline'] == 'cutframes' and not payload.get('frame_step') and (not targets or targets & obsolete):
            remaining = targets or {row['id'] for row in conn.execute("""SELECT id FROM cuts WHERE episode_id=? AND active=1
                AND (first_frame_time IS NULL OR last_frame_time IS NULL)""", (episode['id'],))}
            rebuild.update(remaining - removed)
            cancelled.extend(worker.cancel_jobs(conn, [job], 'Cancelled because cut previews changed'))
        elif (targets and targets <= removed) or (payload.get('frame_step') and payload.get('cut_id') in removed):
            cancelled.extend(worker.cancel_jobs(conn, [job], 'Cancelled because the target cut was archived'))
    structured_results.invalidate_cut_timing(conn, changed | removed)
    conn.executemany('UPDATE cuts SET first_frame_time=NULL,last_frame_time=NULL WHERE id=?', [(cut_id,) for cut_id in changed])
    rebuild.difference_update(rebuilding)
    if rebuild:
        worker.insert_job(conn, episode['id'], 'cutframes',
                          {'settings': json.loads(episode['settings_json']), 'cut_ids': sorted(rebuild)})
    worker.sync_episode_status(conn, episode['id'])
    agent_log.settle(conn, episode['id'])  # a missed-cut question about a cut the person just changed is settled
    return cancelled
