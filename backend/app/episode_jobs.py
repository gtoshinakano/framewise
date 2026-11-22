"""The job list the editor reads: ``episode_response``'s ``jobs`` and ``GET /api/cuts/{id}/jobs``.

A long-used episode holds thousands of finished jobs, megabytes of them with whole analysis contexts in
``payload_json``. The editor needs the jobs that are still active, the newest job of each pipeline, a short
recent history, and, inside the Analysis dialog, the complete history of the one cut it shows. So the
episode response lists a bounded set (``listed``) and the dialog asks for its cut's jobs (``for_cut``).
Both return the same job objects, built by ``serialize``.

Payloads are read only for the jobs that are returned. Ordering never reads past ``payload_json``: columns
stored after it (``created_at`` among them) cost the whole payload per row, so the id selection below sorts
through ``jobs_episode_created_idx`` and reads only ``id``, ``pipeline`` and ``status``.
"""
import math

from . import job_kinds
from .analysis_context import json_object_or_empty

ACTIVE = ('queued', 'running')
RECENT = 25
# Pipelines whose payload names the cuts they work on (``serialize`` builds ``cuts`` for these).
CUT_PIPELINES = ('analyze', 'character_detect', 'regen', 'consistency_summary', 'review_answers', 'review_decisions', 'performance_beats')
COLUMNS = 'id,episode_id,pipeline,status,stage,error,attempt_count,created_at,updated_at,started_at,finished_at,retry_of,priority_class,payload_json'


def _time_value(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _rows(conn, ids):
    """The listed jobs, newest first (the order of the episode's job history)."""
    found = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for row in conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE id IN ({','.join('?' for _ in chunk)})", chunk):
            found[row['id']] = dict(row)
    return [found[job_id] for job_id in ids if job_id in found]


def serialize(conn, episode_id, rows, current_cuts):
    """Job metadata for the UI: the stored payload stays in SQLite for the worker, only its summary is returned."""
    completed = {(row['job_id'], row['cut_id']) for row in conn.execute(
        "SELECT a.job_id,a.cut_id FROM analysis_job_cuts a JOIN jobs j ON j.id=a.job_id WHERE j.episode_id=? AND a.completed=1", (episode_id,))}
    for job in rows:
        payload = json_object_or_empty(job.pop('payload_json'))
        job['workflow'] = payload.get('workflow') if isinstance(payload.get('workflow'), str) else None
        job['stages'] = payload.get('stages') if isinstance(payload.get('stages'), list) else None
        # Sequence jobs belong to a sequence, not to a cut: the editor shows them in the sequence workspace.
        job['sequence'] = {'id': payload.get('sequence_id'), 'step': payload.get('step'), 'shot_key': payload.get('shot_key')} if job['pipeline'] == 'sequence' else None
        kind = job_kinds.kind_of(job['pipeline'])
        job['actions'] = {'cancel': job['status'] in ACTIVE and kind.can_cancel(job['status'], payload),
                          'retry': job['status'] in ('failed', 'cancelled') and kind.retry(payload)}
        if job['pipeline'] == 'analyze':
            items = payload.get('cuts', [])
        elif job['pipeline'] == 'character_detect':
            items = [{'cut_id': payload.get('cut_id'), 'context': {'cut': payload.get('cut', {})}}]
        elif job['pipeline'] in ('regen', 'consistency_summary', 'review_answers', 'review_decisions', 'performance_beats'):
            items = [{'cut_id': payload.get('cut_id'), 'context': payload.get('context', {})}]
        else:
            items = []
        job['cuts'] = []
        if not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            context = item.get('context')
            cut = context.get('cut') if isinstance(context, dict) else {}
            if not isinstance(cut, dict):
                cut = {}
            cut_id = item.get('cut_id')
            if isinstance(cut_id, str) and cut_id:
                cut = cut or current_cuts.get(cut_id, {})
                index = cut.get('cut_index')
                title = cut.get('title')
                job['cuts'].append({'id': cut_id, 'cut_index': index if isinstance(index, int) and not isinstance(index, bool) else None,
                                    'title': title if isinstance(title, str) else '', 'start_time': _time_value(cut.get('start_time')),
                                    'end_time': _time_value(cut.get('end_time')),
                                    'completed': (job['id'], cut_id) in completed if job['pipeline'] == 'analyze' else job['status'] == 'done'})
    return rows


def listed_ids(conn, episode_id):
    """Ids, newest first, of what the episode response lists: every queued or running job, the newest job of each pipeline,
    the newest finished Prepare (the editor's frame and waveform versions) and the most recent ``RECENT`` jobs."""
    ids, pipelines, prepared = [], set(), False
    for position, row in enumerate(conn.execute("SELECT id,pipeline,status FROM jobs WHERE episode_id=? ORDER BY created_at DESC, rowid DESC", (episode_id,))):
        keep = row['status'] in ACTIVE or position < RECENT or row['pipeline'] not in pipelines
        if row['pipeline'] == 'prepare' and row['status'] == 'done' and not prepared:
            prepared, keep = True, True
        pipelines.add(row['pipeline'])
        if keep:
            ids.append(row['id'])
    return ids


def listed(conn, episode_id, current_cuts):
    return serialize(conn, episode_id, _rows(conn, listed_ids(conn, episode_id)), current_cuts)


def for_cut(conn, episode_id, cut_id):
    """Every job of the episode that works on this cut, newest first, as the Analysis dialog's run history and panels read them."""
    marks = ','.join('?' for _ in CUT_PIPELINES)
    ids = [row['id'] for row in conn.execute(
        f"SELECT id FROM jobs WHERE episode_id=? AND pipeline IN ({marks}) AND instr(payload_json,?)>0 ORDER BY created_at DESC, rowid DESC",
        (episode_id, *CUT_PIPELINES, cut_id))]
    current_cuts = {row['id']: dict(row) for row in conn.execute("SELECT id,cut_index,title,start_time,end_time FROM cuts WHERE episode_id=?", (episode_id,))}
    jobs = serialize(conn, episode_id, _rows(conn, ids), current_cuts)
    # The text match only narrows the search; a job counts when its cut list names this cut.
    return [job for job in jobs if any(item['id'] == cut_id for item in job['cuts'])]
