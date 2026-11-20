"""Dispatch existing episode-wide actions once the episode has its current cuts."""
import json

from fastapi import HTTPException

from . import db, job_inputs, worker


def defer(conn, episode_id, action, options, *, config=None):
    parents = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts'))
    if not parents:
        return None
    payload = {'action': action, 'options': options, 'depends_on': parents}
    if config:
        payload['choices'] = config()
    return worker.insert_job(conn, episode_id, 'episode_request', payload)


def cut_range(from_cut, to_cut):
    """Batch options for the displayed (1-based, inclusive) cut numbers. ``last_index`` is only set for a closed range."""
    first_index = max(0, (from_cut or 1) - 1)
    if to_cut is None:
        return {'first_index': first_index}
    if to_cut < first_index + 1:
        raise HTTPException(422, 'to_cut must not be before from_cut')
    return {'first_index': first_index, 'last_index': to_cut - 1}


def refuse_range_while_cutting(conn, episode_id, ranged):
    # A range names cut numbers, which a pending cut detection can still renumber.
    if ranged and worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts')):
        raise HTTPException(409, 'Cuts are still being detected; queue a cut range once detection has finished')


def cuts(conn, episode_id, first_index, cut_id=None, last_index=None):
    return db.all_rows('''SELECT id,cut_index,analysis_enabled,no_characters,start_time,end_time FROM cuts
        WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR cut_index<=?) AND (? IS NULL OR id=?) ORDER BY cut_index''',
        (episode_id, first_index, last_index, last_index, cut_id, cut_id), conn=conn)


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    options = payload['options']
    return cuts(conn, job['episode_id'], options.get('first_index', 0), last_index=options.get('last_index'))


def inputs(conn, job):
    data = job['gathered'] if 'gathered' in job else gather(conn, job)
    return job_inputs.fingerprint(data)


def invalidate(conn, job):
    """Dispatch only inserts jobs. Each child declares its own invalidation."""


def process(job_id, owner):
    from . import main, target_characters_api
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    episode_id = job['episode_id']
    with db.connect() as conn:
        conn.execute('BEGIN')
        live = job_inputs.JobInputs(conn, job, owner)
    with live.publish() as conn:
        if payload['action'] == 'target_detection':
            result = target_characters_api.queue_targets(conn, episode_id, live.data, only_missing=payload['options']['only_missing'])
        else:
            result = main.run_text_batch(conn, episode_id, payload['action'], payload['options'], live.data, payload['choices'])
        child_ids = result['job_ids']
        stage = f'Queued {len(child_ids)} jobs' if child_ids else 'Skipped · no eligible cuts'
        conn.execute('''UPDATE jobs SET status='done',stage=?,error=NULL,input_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?''',
                     (stage, json.dumps({'fingerprint': live.stamp, 'cut_ids': [cut['id'] for cut in live.data], **result}), job_id))
        worker.sync_episode_status(conn, episode_id)
    worker.wake_event.set()
