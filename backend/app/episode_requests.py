"""Dispatch existing episode-wide actions once the episode has its current cuts."""
import json

from . import db, job_inputs, worker


def defer(conn, episode_id, action, options, *, config=None):
    parents = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts'))
    if not parents:
        return None
    payload = {'action': action, 'options': options, 'depends_on': parents}
    if config:
        payload['choices'] = config()
    return worker.insert_job(conn, episode_id, 'episode_request', payload)


def cuts(conn, episode_id, first_index, cut_id=None):
    return db.all_rows('''SELECT id,cut_index,analysis_enabled,no_characters,start_time,end_time FROM cuts
        WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR id=?) ORDER BY cut_index''',
        (episode_id, first_index, cut_id, cut_id), conn=conn)


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    return cuts(conn, job['episode_id'], payload['options'].get('first_index', 0))


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
