"""The episode's queue and run history: what runs now, what waits, what needs the user, and what ran.

Built on ``job_kinds`` (what each pipeline is) and the ``jobs`` table, and free of FastAPI so the worker
images can import it (the HTTP routes are in ``queue_api.py``).

* Cancelled is its own status, never a failure. A cancelled job whose worker has not let go yet (it still
  holds its lease, see ``worker.heartbeat``) is reported as ``stopping``: the GPU is still busy with it.
* The counter (``summary``) counts the individual GPU steps of one episode during its current busy period:
  from the oldest of its active GPU jobs, extended back over jobs that were still running when a later one
  was queued. It is empty when the episode's GPU work is idle.
* ``attention`` lists failures nobody has dealt with: not dismissed, and not followed by a later run of the
  same thing (``Kind.keys``) or a retry.

Payloads can hold whole analysis contexts, so every query reads a slim copy built by SQLite (``SLIM``).
"""
import calendar
import json
import sqlite3
import time

from . import job_kinds

ACTIVE = ('queued', 'running')
FINISHED = ('done', 'failed', 'cancelled')
LISTED_CUTS = 8
WAITING_LIMIT = 200
PAGE_LIMIT = 50
STOPPING_SQL = "(status='cancelled' AND lease_owner IS NOT NULL AND lease_until>?)"


def _snapshot(path):
    """The few fields of a cut snapshot the queue shows, never the whole stored context."""
    return (f"CASE WHEN json_type(payload_json,'{path}')='object' THEN json_object("
            f"'cut_index',json_extract(payload_json,'{path}.cut_index'),'title',json_extract(payload_json,'{path}.title'),"
            f"'start_time',json_extract(payload_json,'{path}.start_time'),'end_time',json_extract(payload_json,'{path}.end_time')) END")


def _field(name, path=None):
    return f"'{name}',json_extract(payload_json,'{path or '$.' + name}')"


SLIM = ("CASE WHEN json_valid(payload_json) THEN json_object("
        + ','.join(_field(name) for name in ('workflow', 'text_only', 'phase', 'cut_id', 'cut_ids', 'next_cut_id', 'step', 'shot_key',
                                               'kind', 'result_type', 'production_brief', 'range', 'ranges', 'language', 'character_context_id'))
        + ",'stages',CASE WHEN json_type(payload_json,'$.stages')='array' THEN json_extract(payload_json,'$.stages') END"
        + f",'cut',{_snapshot('$.cut')},'context',json_object('cut',{_snapshot('$.context.cut')})"
        + ",'cuts',CASE WHEN json_type(payload_json,'$.cuts')='array' THEN (SELECT json_group_array(json_object("
          "'cut_id',json_extract(e.value,'$.cut_id'),'context',json_object('cut',json_object("
          "'cut_index',json_extract(e.value,'$.context.cut.cut_index'),'title',json_extract(e.value,'$.context.cut.title'),"
          "'start_time',json_extract(e.value,'$.context.cut.start_time'),'end_time',json_extract(e.value,'$.context.cut.end_time')))))"
          " FROM json_each(payload_json,'$.cuts') e WHERE e.type='object') END"
        + ",'asset_cuts',(SELECT json_group_array(json_object('id',json_extract(e.value,'$.id'),'cut_number',json_extract(e.value,'$.cut_number')))"
          " FROM json_each(payload_json,CASE WHEN json_type(payload_json,'$.asset_cuts')='array' THEN '$.asset_cuts'"
          " WHEN json_type(payload_json,'$.brief.cuts')='array' THEN '$.brief.cuts' ELSE '$.no_cuts' END) e WHERE e.type='object')"
        + ",'sequence_id',COALESCE(json_extract(payload_json,'$.sequence_id'),CASE WHEN json_type(payload_json,'$.sequence')='object'"
          " THEN json_extract(payload_json,'$.sequence.id') END)"
        + ") ELSE '{}' END")

COLUMNS = ('rowid AS seq,id,episode_id,pipeline,status,stage,error,created_at,started_at,finished_at,updated_at,retry_of,priority_class,'
           f'lease_owner,lease_until,dismissed_at,{SLIM} AS slim')


def _placeholders(values):
    return ','.join('?' for _ in values)


def iso(value):
    """SQLite ``CURRENT_TIMESTAMP`` text (UTC) or epoch seconds as ISO 8601 with ``Z``."""
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(value))
    text = str(value).replace(' ', 'T')
    return text if text.endswith('Z') or '+' in text[10:] else text + 'Z'


def epoch(value):
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(calendar.timegm(time.strptime(str(value).replace('T', ' ').rstrip('Z')[:19], '%Y-%m-%d %H:%M:%S')))
    except ValueError:
        return None


def is_stopping(row, now):
    return row['status'] == 'cancelled' and bool(row['lease_owner']) and (row['lease_until'] or 0) > now


def _loads(row):
    try:
        value = json.loads(row['slim'] or '{}')
    except (TypeError, ValueError):
        value = {}
    payload = value if isinstance(value, dict) else {}
    # SQLite gives back empty arrays and objects for missing pieces; the registry expects them absent.
    if payload.get('cuts') is None:
        payload.pop('cuts', None)
    for key in ('cut', 'context'):
        if not payload.get(key) or (key == 'context' and not payload['context'].get('cut')):
            payload.pop(key, None)
    return payload


def _chunks(values, size=400):
    values = list(values)
    for index in range(0, len(values), size):
        yield values[index:index + size]


class _Progress:
    """Finished cuts and stages of visual analysis jobs, read once for a batch of runs."""

    def __init__(self, conn, job_ids):
        self.completed, self.stages, self.running = set(), set(), {}
        for chunk in _chunks(job_ids):
            marks = _placeholders(chunk)
            for row in conn.execute(f'SELECT job_id,cut_id FROM analysis_job_cuts WHERE completed=1 AND job_id IN ({marks})', chunk):
                self.completed.add((row['job_id'], row['cut_id']))
            for row in conn.execute(f"SELECT job_id,cut_id,stage_key,status FROM analysis_job_stages WHERE job_id IN ({marks}) AND status IN ('done','reused','complete','running')", chunk):
                if row['status'] == 'running':
                    self.running[row['job_id']] = self.running.get(row['job_id'], 0) + 1
                else:
                    self.stages.add((row['job_id'], row['cut_id'], row['stage_key']))

    def units(self, kind, row, payload):
        """``(done, total)`` GPU steps of one job."""
        total = kind.units(payload)
        if row['status'] == 'done':
            return total, total
        if not kind.stage_units:
            return 0, total
        stages = job_kinds.stages_of(payload)
        done = 0
        for ref in kind.cuts(payload):
            if (row['id'], ref['id']) in self.completed:
                done += max(1, len(stages))
            else:
                done += sum(1 for stage in stages if (row['id'], ref['id'], stage) in self.stages)
        return min(done, total), total


def _progress_for(conn, items):
    return _Progress(conn, [row['id'] for row, _ in items if job_kinds.kind_of(row['pipeline']).stage_units])


def _sequences(conn, ids):
    """Title, and shot number and picture cut by shot key, of each sequence."""
    found = {}
    for sequence_id in ids:
        row = conn.execute('SELECT id,episode_id,title FROM sequences WHERE id=?', (sequence_id,)).fetchone()
        if not row:
            continue
        shots = {}
        plan = conn.execute("SELECT content FROM sequence_results WHERE sequence_id=? AND type='plan' AND superseded_by IS NULL ORDER BY rowid DESC LIMIT 1", (sequence_id,)).fetchone()
        try:
            planned = json.loads(plan['content']).get('shots') if plan else None
        except (TypeError, ValueError, AttributeError):
            planned = None
        for index, shot in enumerate(planned if isinstance(planned, list) else []):
            if isinstance(shot, dict) and isinstance(shot.get('start_time'), (int, float)) and isinstance(shot.get('end_time'), (int, float)):
                key = f"{round(shot['start_time'] * 1000)}-{round(shot['end_time'] * 1000)}"  # sequence_timeline.shot_key
                shots[key] = {'number': index + 1, 'cut_id': shot.get('frame_cut_id')}
        found[sequence_id] = {'id': row['id'], 'episode_id': row['episode_id'], 'title': row['title'], 'shots': shots}
    return found


def _shot_cut(conn, episode_id, shot_key):
    """The cut whose picture a shot shows, from the times in its key, for a shot the current plan no longer has."""
    try:
        start, end = (int(part) / 1000 for part in str(shot_key).split('-'))
    except (TypeError, ValueError):
        return None
    row = conn.execute('SELECT id FROM cuts WHERE episode_id=? AND active=1 AND start_time<? AND end_time>? ORDER BY MIN(end_time,?)-MAX(start_time,?) DESC LIMIT 1',
                       (episode_id, end, start, end, start)).fetchone()
    return row['id'] if row else None


def describe(conn, rows, now=None):
    """Runs as the queue shows them, in the order given."""
    now = time.time() if now is None else now
    items = [(dict(row), _loads(row)) for row in rows]
    if not items:
        return []
    progress = _progress_for(conn, items)
    sequences = _sequences(conn, {payload['sequence_id'] for row, payload in items if isinstance(payload.get('sequence_id'), str)})
    refs = {}
    for row, payload in items:
        kind = job_kinds.kind_of(row['pipeline'])
        found = list(kind.cuts(payload))
        sequence = sequences.get(payload.get('sequence_id'))
        if row['pipeline'] == 'sequence' and sequence and payload.get('shot_key'):
            shot = sequence['shots'].get(payload['shot_key']) or {}
            cut_id = shot.get('cut_id') or _shot_cut(conn, row['episode_id'], payload['shot_key'])
            if cut_id:
                found = [{'id': cut_id, 'snapshot': {}}]
        refs[row['id']] = found
    cut_ids = {ref['id'] for found in refs.values() for ref in found}
    cuts = {}
    for chunk in _chunks(cut_ids):
        for cut in conn.execute(f'SELECT id,cut_index,title,active,start_time,end_time FROM cuts WHERE id IN ({_placeholders(chunk)})', chunk):
            cuts[cut['id']] = dict(cut)
    runs = []
    for row, payload in items:
        kind = job_kinds.kind_of(row['pipeline'])
        status = 'stopping' if is_stopping(row, now) else row['status']
        done, total = progress.units(kind, row, payload)
        listed = []
        for ref in refs[row['id']]:
            current, snapshot = cuts.get(ref['id']), ref['snapshot']
            index = current['cut_index'] if current and current['active'] else snapshot.get('cut_index')
            listed.append({'id': ref['id'], 'number': index + 1 if isinstance(index, int) and not isinstance(index, bool) and index >= 0 else None,
                           'title': (current['title'] if current else snapshot.get('title')) or '',
                           'archived': bool(current) and not current['active'], 'missing': current is None,
                           'start_time': current['start_time'] if current else snapshot.get('start_time'),
                           'end_time': current['end_time'] if current else snapshot.get('end_time'),
                           'completed': (row['id'], ref['id']) in progress.completed if kind.stage_units else row['status'] == 'done'})
        sequence = sequences.get(payload.get('sequence_id'))
        shot = sequence['shots'].get(payload.get('shot_key')) if sequence and payload.get('shot_key') else None
        label = kind.name(payload)
        started, finished = epoch(row['started_at']), epoch(row['finished_at'])
        end = finished if finished is not None else (now if status in ('running', 'stopping') else None)
        retry = row['status'] in ('failed', 'cancelled') and kind.retry(payload)
        runs.append({
            'id': row['id'], 'seq': row['seq'], 'kind': row['pipeline'], 'label': label, 'group': f"{row['pipeline']}:{label}", 'gpu': kind.gpu,
            'status': status, 'stage': row['stage'], 'error': row['error'], 'priority_class': row['priority_class'],
            'cuts': listed[:LISTED_CUTS], 'cut_count': len(listed),
            'cut_span': {'first': min(numbers), 'last': max(numbers)} if (numbers := [cut['number'] for cut in listed if cut['number'] is not None]) else None,
            'sequence': {'id': sequence['id'], 'title': sequence['title'], 'step': payload.get('step'), 'shot_key': payload.get('shot_key'),
                         'shot_number': shot['number'] if shot else None} if sequence else None,
            'range': kind.audio_range(payload),
            'units': {'done': done, 'total': total},
            'created_at': iso(row['created_at']), 'started_at': iso(row['started_at']), 'finished_at': iso(row['finished_at']),
            'duration_seconds': max(0, round(end - started)) if started is not None and end is not None else None,
            'retry_of': row['retry_of'], 'dismissed': bool(row['dismissed_at']),
            'actions': {'cancel': row['status'] in ACTIVE and kind.can_cancel(row['status'], payload),
                        'retry': retry,
                        'dismiss': row['status'] == 'failed' and not row['dismissed_at'] and kind.attention,
                        'delete': row['status'] in FINISHED and status != 'stopping'},
        })
    return runs


def _gpu():
    return job_kinds.GPU_PIPELINES


def attention_rows(conn, episode_id):
    """Failed jobs of the episode nobody has dealt with, newest first."""
    failed = [row for row in conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND status='failed' AND dismissed_at IS NULL ORDER BY rowid DESC", (episode_id,))
              if job_kinds.kind_of(row['pipeline']).attention]
    if not failed:
        return []
    pipelines = sorted({row['pipeline'] for row in failed})
    oldest = min(row['seq'] for row in failed)
    latest, retried = {}, set()
    for row in conn.execute(f"SELECT rowid AS seq,pipeline,retry_of,{SLIM} AS slim FROM jobs WHERE episode_id=? AND rowid>? AND status!='cancelled' AND pipeline IN ({_placeholders(pipelines)})",
                            (episode_id, oldest, *pipelines)):
        kind, payload = job_kinds.kind_of(row['pipeline']), _loads(row)
        for key in kind.keys(payload, [ref['id'] for ref in kind.cuts(payload)]):
            latest[key] = max(latest.get(key, 0), row['seq'])
        if row['retry_of']:
            retried.add(row['retry_of'])
    progress = _Progress(conn, [row['id'] for row in failed if job_kinds.kind_of(row['pipeline']).stage_units])
    open_rows = []
    for row in failed:
        if row['id'] in retried:
            continue
        kind, payload = job_kinds.kind_of(row['pipeline']), _loads(row)
        cut_ids = [ref['id'] for ref in kind.cuts(payload)]
        if kind.stage_units:  # a batch that stopped partway is dealt with once its unfinished cuts ran again
            cut_ids = [cut_id for cut_id in cut_ids if (row['id'], cut_id) not in progress.completed] or cut_ids
        keys = kind.keys(payload, cut_ids)
        if keys and all(latest.get(key, 0) > row['seq'] for key in keys):
            continue
        open_rows.append(row)
    return open_rows


def summary(conn, episode_id, now=None):
    """The header counter: individual GPU steps of this episode in its current busy period."""
    now = time.time() if now is None else now
    gpu = _gpu()
    marks = _placeholders(gpu)
    attention = len(attention_rows(conn, episode_id))
    other_active = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE episode_id=? AND status IN ('queued','running') AND pipeline NOT IN ({marks})", (episode_id, *gpu)).fetchone()[0]
    base = {'busy': False, 'since': None, 'done': 0, 'failed': 0, 'running': 0, 'waiting': 0, 'stopping': 0, 'cancelled': 0,
            'total': 0, 'finished': 0, 'attention': attention, 'other_active': other_active}
    first = conn.execute(f"SELECT MIN(created_at) FROM jobs WHERE episode_id=? AND pipeline IN ({marks}) AND (status IN ('queued','running') OR {STOPPING_SQL})",
                         (episode_id, *gpu, now)).fetchone()[0]
    if not first:
        return base
    start = first
    while True:  # extend back over jobs that were still unfinished when a later one was queued
        earlier = conn.execute(f"SELECT MIN(created_at) FROM jobs WHERE episode_id=? AND pipeline IN ({marks}) AND created_at<? AND COALESCE(finished_at,updated_at)>=?",
                               (episode_id, *gpu, start, start)).fetchone()[0]
        if not earlier or earlier >= start:
            break
        start = earlier
    items = [(dict(row), _loads(row)) for row in conn.execute(f'SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND pipeline IN ({marks}) AND created_at>=?', (episode_id, *gpu, start))]
    progress = _progress_for(conn, items)
    counts = {key: 0 for key in ('done', 'failed', 'running', 'waiting', 'stopping', 'cancelled')}
    for row, payload in items:
        kind = job_kinds.kind_of(row['pipeline'])
        done, total = progress.units(kind, row, payload)
        status = row['status']
        if status == 'done':
            counts['done'] += total
        elif status == 'failed':  # the failing step counts once; the rest of a failed batch leaves the queue
            counts['done'] += done
            counts['failed'] += 1
        elif status == 'cancelled':
            counts['done'] += done
            counts['cancelled'] += total - done
            counts['stopping'] += is_stopping(row, now)
        elif status == 'running':
            running = max(1, min(total - done, progress.running.get(row['id'], 1)))
            counts['done'] += done
            counts['running'] += running
            counts['waiting'] += total - done - running
        else:
            counts['waiting'] += total
    finished = counts['done'] + counts['failed']
    return {**base, **counts, 'busy': True, 'since': iso(start), 'finished': finished, 'total': finished + counts['running'] + counts['waiting']}


def _chat(conn, episode_id):
    """This episode's Cut Assistance replies that are queued or running (listed, never counted)."""
    try:
        rows = conn.execute("""SELECT t.id,t.status,t.stage,t.provider,t.model,t.created_at,t.started_at,t.cancel_requested,s.cut_id,s.sequence_id,s.title AS session_title,
                c.cut_index,c.title AS cut_title,c.active,q.title AS sequence_title
            FROM assistant_turns t JOIN assistant_sessions s ON s.id=t.session_id JOIN cuts c ON c.id=s.cut_id LEFT JOIN sequences q ON q.id=s.sequence_id
            WHERE c.episode_id=? AND t.status IN ('queued','running') ORDER BY t.created_at""", (episode_id,)).fetchall()
    except sqlite3.OperationalError:  # the assistant tables are created by the assistant module
        return []
    now = time.time()
    return [{'id': row['id'], 'kind': 'chat', 'label': 'Cut Assistance reply', 'status': 'stopping' if row['status'] == 'running' and row['cancel_requested'] else row['status'],
             'stage': row['stage'], 'provider': row['provider'], 'model': row['model'], 'title': row['session_title'],
             'cut': {'id': row['cut_id'], 'number': row['cut_index'] + 1 if row['active'] and row['cut_index'] is not None and row['cut_index'] >= 0 else None, 'title': row['cut_title'] or '', 'archived': not row['active']},
             'sequence': {'id': row['sequence_id'], 'title': row['sequence_title']} if row['sequence_id'] else None,
             'created_at': iso(row['created_at']), 'started_at': iso(row['started_at']),
             'duration_seconds': max(0, round(now - row['started_at'])) if row['started_at'] else None} for row in rows]


def _elsewhere(conn, episode_id, now):
    """GPU work of other episodes and Chat API requests: why this episode's work may be waiting."""
    gpu = _gpu()
    marks = _placeholders(gpu)
    episodes = []
    for row in conn.execute(f"""SELECT j.episode_id,e.title,p.title AS project_title,SUM(j.status='running') AS running,SUM(j.status='queued') AS waiting
            FROM jobs j JOIN episodes e ON e.id=j.episode_id JOIN projects p ON p.id=e.project_id
            WHERE j.episode_id!=? AND j.pipeline IN ({marks}) AND j.status IN ('queued','running') GROUP BY j.episode_id ORDER BY MIN(j.created_at)""", (episode_id, *gpu)):
        current = conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND status='running' AND pipeline IN ({marks}) ORDER BY started_at,rowid LIMIT 1", (row['episode_id'], *gpu)).fetchone()
        episodes.append({'episode_id': row['episode_id'], 'title': row['title'], 'project_title': row['project_title'], 'running': row['running'], 'waiting': row['waiting'],
                         'current': describe(conn, [current], now)[0] if current else None})
    try:
        chat_api = [{'id': row['id'], 'status': 'stopping' if row['status'] == 'running' and row['cancel_requested'] else row['status'], 'stage': row['stage'],
                     'model': row['model'], 'created_at': iso(row['created_at']), 'started_at': iso(row['started_at']),
                     'duration_seconds': max(0, round(now - row['started_at'])) if row['started_at'] else None}
                    for row in conn.execute("SELECT id,status,stage,model,created_at,started_at,cancel_requested FROM chat_gateway_requests WHERE status IN ('queued','running') ORDER BY created_at")]
    except sqlite3.OperationalError:
        chat_api = []
    return {'episodes': episodes, 'chat_api': chat_api}


def queue_state(conn, episode_id, now=None):
    """Everything the queue drawer shows above the history."""
    now = time.time() if now is None else now
    running = conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND (status='running' OR {STOPPING_SQL}) ORDER BY COALESCE(started_at,created_at),rowid", (episode_id, now)).fetchall()
    queued = conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND status='queued' ORDER BY created_at,rowid", (episode_id,)).fetchall()
    groups = {}
    for row in queued:
        kind, payload = job_kinds.kind_of(row['pipeline']), _loads(row)
        label = kind.name(payload)
        group = groups.setdefault(f"{row['pipeline']}:{label}", {'group': f"{row['pipeline']}:{label}", 'kind': row['pipeline'], 'label': label, 'gpu': kind.gpu, 'count': 0, 'cancel': False})
        group['count'] += 1
        group['cancel'] = group['cancel'] or kind.can_cancel('queued', payload)
    return {'summary': summary(conn, episode_id, now), 'running': describe(conn, running, now),
            'waiting': describe(conn, queued[:WAITING_LIMIT], now), 'waiting_total': len(queued), 'waiting_groups': list(groups.values()),
            'attention': describe(conn, attention_rows(conn, episode_id), now), 'chat': _chat(conn, episode_id), 'elsewhere': _elsewhere(conn, episode_id, now)}


def history(conn, episode_id, before=None, limit=PAGE_LIMIT, status=None, now=None):
    """Finished runs of the episode, newest first, one page at a time (``next`` is the cursor of the following page)."""
    statuses = (status,) if status in FINISHED else FINISHED
    limit = max(1, min(int(limit or PAGE_LIMIT), 200))
    args = [episode_id, *statuses]
    cursor = ''
    if before is not None:
        cursor = ' AND rowid<?'
        args.append(int(before))
    rows = conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND status IN ({_placeholders(statuses)}){cursor} ORDER BY rowid DESC LIMIT ?", (*args, limit + 1)).fetchall()
    counts = {key: 0 for key in FINISHED}
    for row in conn.execute("SELECT status,COUNT(*) AS n FROM jobs WHERE episode_id=? AND status IN ('done','failed','cancelled') GROUP BY status", (episode_id,)):
        counts[row['status']] = row['n']
    return {'runs': describe(conn, rows[:limit], now), 'next': rows[limit - 1]['seq'] if len(rows) > limit else None, 'counts': counts}


def cancellable(conn, episode_id, include_running=False, group=None, job_ids=None):
    """The episode's queued (and, when asked, running) jobs the queue may cancel, optionally one group or some ids."""
    statuses = ('queued', 'running') if include_running else ('queued',)
    rows = conn.execute(f"SELECT {COLUMNS} FROM jobs WHERE episode_id=? AND status IN ({_placeholders(statuses)}) ORDER BY rowid", (episode_id, *statuses)).fetchall()
    chosen = []
    for row in rows:
        kind, payload = job_kinds.kind_of(row['pipeline']), _loads(row)
        if not kind.can_cancel(row['status'], payload):
            continue
        if group and f"{row['pipeline']}:{kind.name(payload)}" != group:
            continue
        if job_ids is not None and row['id'] not in job_ids:
            continue
        chosen.append(row)
    return chosen


def job_payload(conn, job_id):
    """The job row with its slim payload, for one-job decisions (cancel, dismiss)."""
    row = conn.execute(f'SELECT {COLUMNS} FROM jobs WHERE id=?', (job_id,)).fetchone()
    return (row, _loads(row)) if row else (None, None)
