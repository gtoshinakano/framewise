"""Per-cut processing history for the Analysis timeline.

Durations only count execution time (job start to finish, or summed stage
runs), never the time spent queued. Events that cannot be dated because older
data did not record it are still listed, with no time and 0 seconds.
"""
import json
import time
from datetime import datetime, timezone

from . import db

# Canonical order, also the order of "pending" placeholders.
KINDS = ('created', 'character_boxes', 'character_identity', 'visual_analysis',
         'consistency_summary', 'review_answers', 'answers_saved', 'cut_summary')
LABELS = {'created': 'Cut created', 'character_boxes': 'Character boxes', 'character_identity': 'Character identity',
          'visual_analysis': 'Visual analysis', 'consistency_summary': 'Consistency Summary',
          'review_answers': 'AI answers', 'answers_saved': 'Answers saved', 'cut_summary': 'Cut summary'}


def _epoch(value):
    """Parse SQLite CURRENT_TIMESTAMP text (UTC) into epoch seconds."""
    if not value:
        return None
    try:
        return datetime.strptime(value[:19], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if epoch is not None else None


def _event(kind, status, at=None, seconds=0.0, job_id=None, detail=None, label=None):
    return {'kind': kind, 'label': label or LABELS[kind], 'status': status, 'at': _iso(at), 'seconds': round(max(0.0, seconds), 2),
            'job_id': job_id, 'detail': detail}


def _job_status(job):
    if job['status'] == 'done':
        return 'done'
    return job['status']  # queued / running / failed / cancelled


def _job_seconds(job, now):
    start = _epoch(job['started_at'])
    if start is None:
        return 0.0
    end = _epoch(job['finished_at'])
    if end is None:
        end = now if job['status'] == 'running' else start
    return end - start


def _job_event(kind, job, now, seconds=None, label=None):
    status = _job_status(job)
    detail = job['error'] if status == 'failed' else None
    at = _epoch(job['started_at']) or _epoch(job['created_at'])
    return _event(kind, status, at, _job_seconds(job, now) if seconds is None else seconds, job['id'], detail, label)


def build(cut_id):
    now = time.time()
    cut = db.one('SELECT id,created_at FROM cuts WHERE id=?', (cut_id,))
    if not cut:
        return None
    events = [_event('created', 'done', _epoch(cut['created_at']))]

    columns = 'id,pipeline,status,stage,error,created_at,started_at,finished_at'
    jobs = db.all_rows(f"""SELECT {columns} FROM jobs WHERE pipeline IN ('character_detect','analyze','consistency_summary','review_answers','regen')
        AND (json_extract(payload_json,'$.cut_id')=? OR EXISTS (SELECT 1 FROM json_each(payload_json,'$.cuts') i WHERE json_extract(i.value,'$.cut_id')=?)
             OR id IN (SELECT job_id FROM analysis_job_cuts WHERE cut_id=?)) ORDER BY created_at,rowid""", (cut_id, cut_id, cut_id))
    stage_seconds = {}
    for row in db.all_rows('SELECT job_id,status,started_at,last_seen_at,finished_at FROM analysis_job_stages WHERE cut_id=?', (cut_id,)):
        end = row['finished_at'] or (now if row['status'] == 'running' else row['last_seen_at'])
        stage_seconds[row['job_id']] = stage_seconds.get(row['job_id'], 0.0) + max(0.0, (end or row['started_at']) - row['started_at'])

    runs = db.all_rows('SELECT id,job_id,created_at,phase FROM character_detection_runs WHERE cut_id=? ORDER BY rowid', (cut_id,))
    run_jobs = {run['job_id'] for run in runs}
    identified = {row['job_id'] for row in db.all_rows(
        '''SELECT DISTINCT r.job_id FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id
           WHERE r.cut_id=? AND d.character_context_id IS NOT NULL''', (cut_id,))}
    consistency_jobs = set()
    for job in jobs:
        pipeline = job['pipeline']
        if pipeline == 'character_detect':
            payload = db.one('SELECT json_extract(payload_json,\'$.phase\') AS phase FROM jobs WHERE id=?', (job['id'],))
            phase = payload['phase'] if payload else None
            if phase == 'boxes':
                events.append(_job_event('character_boxes', job, now))
            elif phase == 'recognize':
                events.append(_job_event('character_identity', job, now))
            else:  # single-cut flow localizes and labels in one job
                events.append(_job_event('character_boxes', job, now))
                if job['id'] in identified:
                    events.append(_event('character_identity', 'done', _epoch(job['finished_at']), 0, job['id']))
        elif pipeline == 'analyze':
            seconds = stage_seconds.get(job['id'])
            events.append(_job_event('visual_analysis', job, now, seconds))
        elif pipeline == 'consistency_summary':
            consistency_jobs.add(job['id'])
            events.append(_job_event('consistency_summary', job, now))
        elif pipeline == 'review_answers':
            events.append(_job_event('review_answers', job, now))
        elif pipeline == 'regen':
            payload = db.one('SELECT json_extract(payload_json,\'$.result_type\') AS result_type FROM jobs WHERE id=?', (job['id'],))
            if payload and payload['result_type'] == 'summary':
                events.append(_job_event('cut_summary', job, now))
    # Older runs whose job row is gone, and summaries created outside a summary job.
    for run in runs:
        if run['job_id'] not in {job['id'] for job in jobs}:
            events.append(_event('character_boxes', 'done', _epoch(run['created_at'])))
    for row in db.all_rows("SELECT job_id,created_at,settings_json FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL", (cut_id,)):
        try:
            structured = json.loads(row['settings_json'] or '{}').get('structured_type')
        except ValueError:
            structured = None
        if structured == 'consistency_summary' and row['job_id'] not in consistency_jobs and not any(e['kind'] == 'consistency_summary' for e in events):
            events.append(_event('consistency_summary', 'done', _epoch(row['created_at'])))
    saved = db.one('SELECT content,updated_at FROM cut_brief_instructions WHERE cut_id=?', (cut_id,))
    if saved and saved['content'].strip():
        events.append(_event('answers_saved', 'done', _epoch(saved['updated_at'])))

    rank = {kind: index for index, kind in enumerate(KINDS)}
    head, rest = events[:1], events[1:]
    rest.sort(key=lambda e: (e['at'] or '9999', rank[e['kind']]))
    present = {event['kind'] for event in events}
    pending = [_event(kind, 'pending') for kind in KINDS if kind not in present]
    events = head + rest + pending
    return {'events': events, 'total_seconds': round(sum(event['seconds'] for event in events), 2)}
