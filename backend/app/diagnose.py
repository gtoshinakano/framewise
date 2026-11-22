"""Why a cut is not finished, and what a failed run needs: the autopilot's diagnosis (docs/autopilot/HARNESS.md, milestone M1).

Read-only. Three views over facts that are already stored:

* ``health``: per cut, a ranked list of problems in one sentence each (failed steps, stale results, picture contradictions,
  lines that run across the cut, checkpoints for a person, the next missing step), worst cuts first.
* ``why``: for one cut and one step, the chain of prerequisites down to the first broken link, e.g.
  "Consistency review failed ← Performance beats are older than the keyframe analysis (redone 2026-11-20)".
* ``triage``: every failed run nobody has dealt with (``runs.attention_rows``), classified by cause (``classify``) with the fix
  rule that applies and whether the autopilot may apply it now, should wait, or must ask a person. ``triage.py`` applies it.

A cut's analysis is modelled as eight steps in the editor's order (``STEPS``); each has a state:
``off`` (analysis switched off), ``running``/``queued``, ``failed``, ``missing``, ``stale``, ``person`` (a checkpoint for a
person) or ``ok``. Nothing here is an approval status, and nothing is saved.
"""
import bisect
import json
import re
import time

from . import job_kinds, review_report, runs

STEPS = ('boxes', 'target_frames', 'first_frame', 'keyframes', 'beats', 'consistency', 'answers', 'summary')
LABELS = {'boxes': 'First-frame character boxes', 'target_frames': 'Target-frame people', 'first_frame': 'First-frame analysis',
          'keyframes': 'Keyframe analysis', 'beats': 'Performance beats', 'consistency': 'Consistency review',
          'answers': 'Answers to the review questions', 'summary': 'Cut summary'}
# What each step reads. The first broken one is where the chain continues.
PREREQS = {'boxes': (), 'target_frames': ('boxes',), 'first_frame': ('boxes',), 'keyframes': ('target_frames',),
           'beats': ('keyframes',), 'consistency': ('first_frame', 'beats'), 'answers': ('consistency',), 'summary': ('answers',)}
# Where a cut link opens for the step (scripts/fw PLACES) and the fw command that queues it.
PLACES = {'boxes': 'characters', 'target_frames': 'target-frames', 'first_frame': 'run', 'keyframes': 'run',
          'beats': 'analysis', 'consistency': 'analysis', 'answers': 'analysis', 'summary': 'analysis'}
FW_STEP = {'boxes': 'boxes', 'target_frames': 'target-frames', 'first_frame': 'visual', 'keyframes': 'visual', 'beats': 'beats',
           'consistency': 'consistency', 'answers': 'answers', 'summary': 'summary'}
STAGE_STEPS = {'initial_frame': 'first_frame', 'keyframe_analysis': 'keyframes', 'video_analysis': 'keyframes'}

# Problem weights: a cut's score is the sum, cuts are listed by their worst problem first.
SEVERITY = {'failed': 100, 'stale': 60, 'contradiction': 50, 'person': 40, 'missing': 30, 'line_crossing': 20,
            'auto_labels': 10, 'no_boxes': 10}
CROSSING = 0.15  # a line runs across a hard cut when it lasts more than this on both sides (HARNESS.md section 3)
ERROR_CHARS = 240

# --- failure causes and their fix rules ------------------------------------------------------------------------------

# (cause, pattern). The first match wins, so the specific patterns come first.
RULES = (
    ('prerequisite_missing', r'has no keyframe analysis|run Keyframes first'),
    ('prerequisite_stale', r'missing or older than (?:its|the) keyframe analysis'),
    ('gpu_memory', r'out of memory|failed to fit params|miopenStatus|hipError|HIP error|Local model failed to start|free device memory'),
    ('context_overflow', r'exceeds the available context size'),
    ('provider', r'OpenRouter returned|OpenRouter request|IncompleteRead|response was empty|rate limit|HTTP 429|HTTP 50\d|timed out|Connection (?:refused|reset)'),
    ('stale_previews', r'preview frame timestamps are invalid|refresh the cut frames'),
    ('legacy_pipeline', r'Character comparison failed|Invalid character match response|queue visual analysis again'),
    ('validator_loop', r'invalid analysis after|Invalid .* response|[Rr]esponse was truncated|Unterminated string|must be |must match|missing or invalid field|Invalid JSON'),
    ('code_error', r'No module named|ImportError|NameError|AttributeError|TypeError|KeyError|Traceback'),
    ('media_error', r'muxer|ffmpeg|ffprobe|Invalid data found|Error submitting|moov atom|Input #0'),
)
CAUSES = {
    'prerequisite_missing': 'a step it needs had not run',
    'prerequisite_stale': 'it would have read results that are out of date',
    'gpu_memory': 'the GPU had no free memory',
    'context_overflow': 'the request was longer than the model can read',
    'provider': 'the text provider returned nothing usable',
    'stale_previews': 'the cut preview frames were out of date',
    'validator_loop': 'the model answer failed validation on every attempt',
    'legacy_pipeline': 'it belongs to an older workflow',
    'code_error': 'a program error',
    'media_error': 'reading or writing the media failed',
    'unknown': 'no rule knows this error',
    'obsolete': 'what it would have produced exists now',
    'out_of_scope': 'sequences are the user\'s work',
}
# cause -> (action, attempts the autopilot may make before it asks a person, what the fix does)
#   rerun: retry the job from its snapshot when its kind supports it, else queue the step again; requeue: queue the step again
#   with the current settings; prerequisite: queue the step it needs, then the step; rerun_when_idle: rerun once no GPU job runs.
FIXES = {
    'prerequisite_missing': ('prerequisite', 2, 'queue the step it needs, then run it again after that'),
    'prerequisite_stale': ('requeue', 2, 'queue the out-of-date step again first, then this one after it'),
    'gpu_memory': ('rerun_when_idle', 2, 'run it again once no GPU work is running'),
    'provider': ('requeue', 1, 'queue it again with the current text engine'),
    'stale_previews': ('requeue', 1, 'queue it again with the current workflow, which reads fresh frames'),
    'validator_loop': ('rerun', 1, 'run it once more; a second failure goes to a person with the validator\'s message'),
    'legacy_pipeline': ('requeue', 1, 'queue it again with the current workflow'),
    'code_error': ('rerun', 1, 'run it once more (the code may have been fixed since)'),
    'media_error': ('rerun', 1, 'run it once more'),
    'context_overflow': ('escalate', 0, 'a person must shorten its input or split the cut'),
    'unknown': ('escalate', 0, 'a person must look at the error'),
    'obsolete': ('dismiss', 1, 'dismiss it: nothing is missing any more'),
    'out_of_scope': ('leave', 0, 'leave it: sequences are out of the autopilot\'s scope'),
}
# Pipelines triage can queue again with the current settings (triage.requeue); the others only through their own retry.
REQUEUE = ('analyze', 'performance_beats', 'consistency_summary', 'review_answers', 'review_decisions', 'regen', 'missed_cuts', 'line_splits', 'voice_resolver', 'character_decisions')
OUT_OF_SCOPE = ('sequence', 'brief_export')


def classify(pipeline, error):
    """``(cause, text)`` of a failure from its pipeline and error message."""
    if pipeline in OUT_OF_SCOPE:
        return 'out_of_scope', CAUSES['out_of_scope']
    for cause, pattern in RULES:
        if re.search(pattern, error or ''):
            return cause, CAUSES[cause]
    return 'unknown', CAUSES['unknown']


def sentence(error, number=None):
    """The part of an error that says what went wrong for one cut: the per-cut part of a batch message, after "N attempts:"."""
    text = (error or '').strip()
    if number is not None:
        found = dict((int(n), part.strip()) for n, part in re.findall(r'cut (\d+): (.+?)(?=; cut \d+: |$)', text, re.S))
        text = found.get(number, text)
    text = re.sub(r'^\d+ of \d+ cuts failed, the others finished and were kept\. ', '', text)
    text = re.split(r'after (?:\d+|one repair) attempts?: ', text, maxsplit=1)[-1]
    return text[:ERROR_CHARS]


def job_steps(pipeline, payload):
    """The cut steps a job of this pipeline produces."""
    if pipeline == 'analyze':
        stages = job_kinds.stages_of(payload) or list(job_kinds.DEFAULT_STAGES)
        return list(dict.fromkeys(STAGE_STEPS[stage] for stage in stages if stage in STAGE_STEPS))
    if pipeline == 'character_detect':
        return ['target_frames' if payload.get('workflow') == 'target_frames_v1' else 'boxes']
    if pipeline == 'regen':
        return ['summary'] if job_kinds.kind_of('regen').name(payload) == 'Cut summary' else []
    return {'keyframes': ['keyframes'], 'performance_beats': ['beats'], 'consistency_summary': ['consistency'],
            'review_answers': ['answers'], 'review_decisions': ['answers']}.get(pipeline, [])


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def _date(value):
    return str(value or '')[:10]


# --- facts ---------------------------------------------------------------------------------------------------------

class Facts:
    """Everything the views need about one episode (or a cut range), read once."""

    def __init__(self, conn, episode_id, from_cut=None, to_cut=None):
        self.conn, self.episode_id, self.now = conn, episode_id, time.time()
        self.report = review_report.report(conn, episode_id, from_cut, to_cut)
        first, last = max(0, (from_cut or 1) - 1), (to_cut - 1 if to_cut is not None else None)
        self.cuts = [dict(row) for row in conn.execute("""SELECT id,cut_index,start_time,end_time,analysis_enabled,no_characters FROM cuts
            WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR cut_index<=?) ORDER BY cut_index""", (episode_id, first, last, last))]
        self.by_number = {cut['cut_index'] + 1: cut for cut in self.cuts}
        self.numbers = {cut['id']: cut['cut_index'] + 1 for cut in self.cuts}
        self._sets()
        self._results()
        self._jobs()

    def _sets(self):
        progress, gates, checks = self.report['progress'], self.report['gates'], self.report['checks']
        self.missing = {key: set(progress.get(key) or []) for key in progress if key.endswith('_missing')}
        self.gates = {name: {item['cut']: item for item in items} for name, items in gates.items() if name != 'voices'}
        self.contradictions = {item['cut']: item['checks'] for item in checks.get('picture_contradictions') or []}
        self.auto_labels = {item['cut']: item['labels'] for item in checks.get('auto_labels') or []}
        self.no_boxes = set(checks.get('no_boxes') or [])

    def _results(self):
        """Current results per cut and type: id, date, stale reason, and for beats the keyframe analysis they were built from."""
        self.results = {}
        ids = list(self.numbers)
        for chunk in range(0, len(ids), 400):
            part = ids[chunk:chunk + 400]
            marks = ','.join('?' for _ in part)
            for row in self.conn.execute(f"""SELECT cut_id,id,title,created_at,stale_reason,
                    json_extract(settings_json,'$.structured_type') AS type,
                    CASE WHEN json_extract(settings_json,'$.structured_type')='performance_beats'
                         THEN json_extract(content,'$.based_on.keyframe_analysis') END AS based_on
                    FROM analysis_results WHERE superseded_by IS NULL AND cut_id IN ({marks}) ORDER BY rowid""", part):
                kind = row['type'] or ('summary' if row['title'] == 'Cut summary' else None)
                if kind:
                    self.results.setdefault(row['cut_id'], {})[kind] = dict(row)

    def _jobs(self):
        """Queued and running jobs, and failed runs nobody has dealt with, per (cut, step)."""
        self.active, self.failures, self.attention = {}, {}, []
        rows = self.conn.execute(f"SELECT {runs.COLUMNS} FROM jobs WHERE episode_id=? AND status IN ('queued','running') ORDER BY rowid",
                                 (self.episode_id,)).fetchall()
        for row in rows:
            payload = runs._loads(row)
            for ref in job_kinds.kind_of(row['pipeline']).cuts(payload):
                for step in job_steps(row['pipeline'], payload):
                    self.active.setdefault((ref['id'], step), {'id': row['id'], 'status': row['status'], 'stage': row['stage']})
        failed = runs.attention_rows(self.conn, self.episode_id)
        completed = set()
        stage_jobs = [row['id'] for row in failed if job_kinds.kind_of(row['pipeline']).stage_units]
        for chunk in range(0, len(stage_jobs), 400):
            part = stage_jobs[chunk:chunk + 400]
            marks = ','.join('?' for _ in part)
            completed |= {(row[0], row[1]) for row in self.conn.execute(
                f'SELECT job_id,cut_id FROM analysis_job_cuts WHERE completed=1 AND job_id IN ({marks})', part)}
        for row in failed:
            payload = runs._loads(row)
            kind = job_kinds.kind_of(row['pipeline'])
            cut_ids = [ref['id'] for ref in kind.cuts(payload)]
            if kind.stage_units:
                cut_ids = [cut_id for cut_id in cut_ids if (row['id'], cut_id) not in completed] or cut_ids
            item = {'row': row, 'payload': payload, 'cut_ids': cut_ids, 'steps': job_steps(row['pipeline'], payload),
                    'finished': runs.epoch(row['finished_at'] or row['updated_at'])}
            item['obsolete'] = self._obsolete(item)
            self.attention.append(item)
            if item['obsolete']:
                continue
            for cut_id in cut_ids:
                for step in item['steps']:
                    self.failures.setdefault((cut_id, step), item)

    def _result_ok(self, cut_id, step):
        """The step's own output exists and is current, whatever its jobs did; returns its date (epoch) or True, else None."""
        number = self.numbers.get(cut_id)
        found = self.results.get(cut_id, {})
        if step in ('boxes', 'target_frames', 'answers'):
            if number is None:
                return None
            if step == 'boxes':
                return True if number not in self.missing.get('boxes_missing', ()) else None
            if step == 'target_frames':
                return True if number not in self.missing.get('target_missing', ()) else None
            return True if number not in self.gates.get('decisions', {}) and 'consistency_summary' in found else None
        kinds = {'first_frame': ('initial_frame',), 'keyframes': ('keyframe_analysis', 'video_analysis'), 'beats': ('performance_beats',),
                 'consistency': ('consistency_summary',), 'summary': ('summary',)}[step]
        result = next((found[kind] for kind in kinds if kind in found), None)
        if not result or result['stale_reason']:
            return None
        if step == 'beats' and 'keyframe_analysis' in found and result['based_on'] != found['keyframe_analysis']['id']:
            return None
        return runs.epoch(result['created_at']) or True

    def _obsolete(self, item):
        """A failure is obsolete when every cut it left unfinished is archived, or has the step's output made after it failed."""
        if not item['cut_ids'] or not item['steps']:
            return False
        live = set(row[0] for row in self.conn.execute(
            f"SELECT id FROM cuts WHERE active=1 AND id IN ({','.join('?' for _ in item['cut_ids'])})", item['cut_ids']))
        for cut_id in item['cut_ids']:
            if cut_id not in live:
                continue
            if cut_id not in self.numbers:  # outside the range: unknown here, so not obsolete
                return False
            for step in item['steps']:
                made = self._result_ok(cut_id, step)
                if made is None or (made is not True and item['finished'] and made < item['finished']):
                    return False
        return True

    # --- one step of one cut ---------------------------------------------------------------------------------------

    def state(self, cut, step):
        """``{state, text}`` of one step of one cut (and ``job``, ``error``, ``cause`` where they apply)."""
        number, cut_id = cut['cut_index'] + 1, cut['id']
        if not cut['analysis_enabled']:
            return {'state': 'off', 'text': 'analysis is switched off for this cut'}
        if step in ('boxes', 'target_frames') and cut['no_characters']:
            return {'state': 'ok', 'text': 'marked as having no characters'}
        if active := self.active.get((cut_id, step)):
            return {'state': active['status'], 'job': active['id'],
                    'text': f"{'running' if active['status'] == 'running' else 'queued'}" + (f" ({active['stage']})" if active['stage'] and active['status'] == 'running' else '')}
        if failure := self.failures.get((cut_id, step)):
            row = failure['row']
            cause, cause_text = classify(row['pipeline'], row['error'])
            return {'state': 'failed', 'job': row['id'], 'cause': cause, 'error': sentence(row['error'], number),
                    'text': f"failed {_date(row['finished_at'])}: {cause_text} ({sentence(row['error'], number)[:120]})"}
        found = self.results.get(cut_id, {})
        missing = self.missing
        if step == 'boxes':
            if number in missing.get('boxes_missing', ()):
                return {'state': 'missing', 'text': 'no first-frame character detection yet'}
            if gate := self.gates['first_frame'].get(number):
                return {'state': 'person', 'text': f"{gate['unlabeled_boxes']} first-frame box(es) have no character"}
            return {'state': 'ok', 'text': 'detected'}
        if step == 'target_frames':
            if number in missing.get('target_missing', ()):
                return {'state': 'missing', 'text': 'people on the target frames not detected yet'}
            if gate := self.gates['target_frames'].get(number):
                parts = [f"{len(gate['frames'])} flagged frame(s)"] if gate['frames'] else []
                parts += [f"{gate['unnamed_people']} unnamed person(s)"] if gate['unnamed_people'] else []
                return {'state': 'person', 'text': 'target frames need review: ' + ', '.join(parts)}
            return {'state': 'ok', 'text': 'detected'}
        if step == 'first_frame':
            return self._result_state(found.get('initial_frame'), 'no first-frame analysis yet')
        if step == 'keyframes':
            return self._result_state(found.get('keyframe_analysis') or found.get('video_analysis'), 'no keyframe analysis yet')
        if step == 'beats':
            keyframes = found.get('keyframe_analysis')
            if not keyframes:
                if found.get('video_analysis'):
                    return {'state': 'ok', 'text': 'analysed before keyframes: the video analysis is its motion'}
                return {'state': 'missing', 'text': 'no performance beats (no keyframe analysis yet)'}
            beats = found.get('performance_beats')
            if not beats:
                return {'state': 'missing', 'text': 'no performance beats yet'}
            if beats['based_on'] != keyframes['id']:
                return {'state': 'stale', 'text': f"built from an older keyframe analysis (keyframes redone {_date(keyframes['created_at'])})"}
            return self._result_state(beats, '')
        if step == 'consistency':
            return self._result_state(found.get('consistency_summary'), 'no consistency review yet')
        if step == 'answers':
            if 'consistency_summary' not in found:
                return {'state': 'missing', 'text': 'no review questions yet (no consistency review)'}
            if gate := self.gates['decisions'].get(number):
                return {'state': 'person', 'text': f"{gate['questions']} review question(s) without saved answers"
                        + (' (a suggested draft is ready)' if gate['draft'] else '')}
            return {'state': 'ok', 'text': 'answered, or nothing to answer'}
        return self._result_state(found.get('summary'), 'no Cut summary yet')

    @staticmethod
    def _result_state(result, missing_text):
        if not result:
            return {'state': 'missing', 'text': missing_text}
        if result['stale_reason']:
            return {'state': 'stale', 'text': f"stale: {result['stale_reason']}", 'since': _date(result['created_at'])}
        return {'state': 'ok', 'text': f"made {_date(result['created_at'])}"}

    def chain(self, cut, step):
        """From ``step`` down its prerequisites to the first broken link: ``[{step, label, state, text, …}]``, root last."""
        links, current, seen = [], step, set()
        while current and current not in seen:
            seen.add(current)
            state = self.state(cut, current)
            links.append({'step': current, 'label': LABELS[current], **state})
            if state['state'] in ('ok', 'off'):
                break
            broken = [name for name in PREREQS[current] if self.state(cut, name)['state'] not in ('ok',)]
            current = broken[0] if broken else None
        return links


# --- what to do about the root ---------------------------------------------------------------------------------------

def fix_for(link, cut_number):
    """One sentence and, where a command does it, the fw command for the root link of a chain."""
    step, state = link['step'], link['state']
    command = FW_STEP[step]
    if state == 'ok':
        return {'text': 'nothing to fix: this step is done', 'command': None}
    if state in ('running', 'queued'):
        return {'text': 'wait: it is in the queue', 'command': None}
    if state == 'off':
        return {'text': 'switch analysis on for this cut if it should be analysed', 'command': None}
    if state == 'person':
        if step in ('boxes', 'target_frames'):
            return {'text': f"the character agent names what it can and asks about the rest; then a person settles it ({PLACES[step]})",
                    'command': 'fw cast run EP', 'place': PLACES[step]}
        if step == 'answers':
            return {'text': f"the review-answer agent answers what it can; then a person settles it ({PLACES[step]})",
                    'command': 'fw review-decisions run EP', 'place': PLACES[step]}
        return {'text': f"a person settles it in the editor ({PLACES[step]})", 'command': None, 'place': PLACES[step]}
    if state == 'failed':
        action, _, text = FIXES[link['cause']]
        return {'text': text, 'command': 'fw triage EP --go' if action not in ('escalate', 'leave') else None}
    if state == 'stale':
        redo = '' if step == 'beats' else ' --redo'
        return {'text': f"run {LABELS[step].lower()} again", 'command': f'fw {command} EP {cut_number}{redo}'}
    return {'text': f"queue {LABELS[step].lower()}", 'command': f'fw {command} EP {cut_number}'}


def why(conn, cut_id, step=None):
    """The root cause of one step of one cut (default: the earliest step that is not done, else the Cut summary)."""
    cut = conn.execute('SELECT episode_id,cut_index,active FROM cuts WHERE id=?', (cut_id,)).fetchone()
    if not cut:
        raise KeyError(cut_id)
    if not cut['active']:
        return {'cut_id': cut_id, 'cut': None, 'step': step, 'chain': [], 'root': None,
                'fix': {'text': 'this cut is archived: nothing runs on it', 'command': None}}
    number = cut['cut_index'] + 1
    facts = Facts(conn, cut['episode_id'], number, number)
    row = facts.by_number[number]
    if step is None:
        step = next((name for name in STEPS if facts.state(row, name)['state'] not in ('ok',)), 'summary')
    if step not in STEPS:
        raise ValueError(f"step must be one of {', '.join(STEPS)}")
    chain = facts.chain(row, step)
    root = chain[-1]
    return {'cut_id': cut_id, 'cut': number, 'step': step, 'chain': chain, 'root': root, 'fix': fix_for(root, number), 'place': PLACES[root['step']],
            'text': ' ← '.join(f"{link['label']}: {link['text']}" for link in chain),
            'steps': {name: facts.state(row, name) for name in STEPS}}


# --- per-cut health --------------------------------------------------------------------------------------------------

def crossing_lines(conn, episode_id, cuts):
    """{cut id: [{start, end, text, edge}]} of current transcript lines that run across the cut's start or end."""
    lines = conn.execute("""SELECT start_time,end_time,text FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL
        AND deleted=0 AND end_time-start_time>? ORDER BY start_time""", (episode_id, 2 * CROSSING)).fetchall()
    starts = [line['start_time'] for line in lines]
    longest = max((line['end_time'] - line['start_time'] for line in lines), default=0)
    found = {}
    for cut in cuts:
        for edge, at in (('start', cut['start_time']), ('end', cut['end_time'])):
            for line in lines[bisect.bisect_left(starts, at - longest):bisect.bisect_left(starts, at - CROSSING)]:
                if line['end_time'] > at + CROSSING:
                    found.setdefault(cut['id'], []).append({'start': round(line['start_time'], 2), 'end': round(line['end_time'], 2),
                                                            'edge': edge, 'at': round(at, 3), 'text': line['text'][:60]})
    return found


def health(conn, episode_id, from_cut=None, to_cut=None):
    facts = Facts(conn, episode_id, from_cut, to_cut)
    crossing = crossing_lines(conn, episode_id, facts.cuts)
    listed, counts = [], {}
    for cut in facts.cuts:
        number = cut['cut_index'] + 1
        problems = []

        def add(kind, text, **extra):
            problems.append({'kind': kind, 'severity': SEVERITY[kind], 'text': text, **extra})
            counts[kind] = counts.get(kind, 0) + 1

        states = {step: facts.state(cut, step) for step in STEPS}
        busy = [step for step, state in states.items() if state['state'] in ('running', 'queued')]
        for step, state in states.items():
            if state['state'] == 'failed':
                action, _, fix_text = FIXES[state['cause']]
                add('failed', f"{LABELS[step]} failed: {CAUSES[state['cause']]} — {state['error'][:140]}", step=step, job_id=state['job'],
                    cause=state['cause'], fix=fix_text, place=PLACES[step])
            elif state['state'] == 'stale':
                add('stale', f"{LABELS[step]} is {state['text']}", step=step, place=PLACES[step], fix=fix_for({'step': step, **state}, number))
            elif state['state'] == 'person':
                add('person', f"{LABELS[step]}: {state['text']}", step=step, place=PLACES[step])
        for check in facts.contradictions.get(number, []):
            add('contradiction', f"Picture {check.get('picture')} contradicts \"{(check.get('claim') or '')[:100]}\" ({check.get('character') or 'scene'})",
                place='analysis')
        # The next step to run: the earliest one missing, explained by its chain (it may wait on a failure or a person).
        first_missing = next((step for step in STEPS if states[step]['state'] == 'missing'), None)
        if first_missing and cut['analysis_enabled']:
            chain = facts.chain(cut, first_missing)
            root = chain[-1]
            reason = '' if root['step'] == first_missing else f" ← {LABELS[root['step']]}: {root['text']}"
            add('missing', f"Needs {LABELS[first_missing].lower()}{reason}", step=first_missing, place=PLACES[first_missing],
                root=root['step'], fix=fix_for(root, number))
        if lines := crossing.get(cut['id']):
            edges = sorted({line['edge'] for line in lines})
            add('line_crossing', f"{len(lines)} transcript line(s) run across the cut's {' and '.join(edges)}"
                + f" (first: {lines[0]['start']}–{lines[0]['end']}s \"{lines[0]['text']}\")", place='analysis', lines=lines[:5])
        if labels := facts.auto_labels.get(number):
            add('auto_labels', f"{labels} automatic label(s) not confirmed", place='characters')
        if number in facts.no_boxes:
            add('no_boxes', 'Detected without any box (mark No characters if that is right)', place='characters')
        if not problems:
            continue
        problems.sort(key=lambda item: -item['severity'])
        listed.append({'cut': number, 'cut_id': cut['id'], 'start_time': cut['start_time'], 'end_time': cut['end_time'],
                       'score': sum(item['severity'] for item in problems), 'worst': problems[0]['severity'],
                       'busy': busy, 'problems': problems})
    listed.sort(key=lambda item: (-item['worst'], -item['score'], item['cut']))
    return {'episode': facts.report['episode'], 'range': facts.report['range'], 'cuts': listed, 'counts': counts,
            'episode_problems': episode_problems(facts), 'checked': len(facts.cuts)}


def episode_problems(facts):
    report, items = facts.report, []
    missing = [name for name, done in report['setup'].items() if not done]
    if missing:
        items.append({'kind': 'setup', 'text': f"Setup is incomplete: {', '.join(missing)}", 'command': 'fw setup EP'})
    for voice in report['gates'].get('voices') or []:
        items.append({'kind': 'voice', 'text': f"Voice {voice['speaker_label'] or '(no label)'}: {voice['lines']} line(s) without a character",
                      'first_time': voice['first_time'], 'command': 'fw voices run EP'})
    checks = report['checks']
    if checks.get('speaker_attribution_stale'):
        items.append({'kind': 'stale', 'text': f"Speaker attribution is stale: {checks['speaker_attribution_stale']}", 'command': 'fw speakers EP'})
    if checks.get('transcript_stale'):
        items.append({'kind': 'stale', 'text': f"The transcript is stale: {checks['transcript_stale']}"})
    for item in facts.attention:
        if not item['cut_ids'] and not item['obsolete']:
            row = item['row']
            cause, text = classify(row['pipeline'], row['error'])
            items.append({'kind': 'failed', 'text': f"{job_kinds.kind_of(row['pipeline']).name(item['payload'])} failed: {text} — {sentence(row['error'])[:140]}",
                          'job_id': row['id'], 'cause': cause})
    return items


# --- triage: failed runs and their fix rules -------------------------------------------------------------------------

def attempts(conn, episode_id, key):
    """Fixes the autopilot already applied to this failure key (decisions not undone)."""
    return conn.execute("""SELECT COUNT(*) FROM agent_decisions WHERE episode_id=? AND kind='failed_runs' AND undone_at IS NULL
        AND json_extract(evidence_json,'$.key')=?""", (episode_id, key)).fetchone()[0]


def gpu_busy(conn):
    marks = ','.join('?' for _ in job_kinds.GPU_PIPELINES)
    return bool(conn.execute(f"SELECT 1 FROM jobs WHERE status='running' AND pipeline IN ({marks}) LIMIT 1", job_kinds.GPU_PIPELINES).fetchone())


def current_provider(conn):
    row = conn.execute('SELECT text_provider FROM text_generation_settings WHERE id=1').fetchone()
    return row['text_provider'] if row else 'openrouter'


def _provider_of(payload):
    configs = [payload.get('config')] + [item.get('config') for item in payload.get('cuts') or [] if isinstance(item, dict)]
    for config in configs:
        if isinstance(config, dict) and config.get('text_provider'):
            return config['text_provider']
    return payload.get('text_provider')


def _can_rerun(pipeline, payload):
    """Whether ``rerun`` retries from the job's snapshot (resuming its checkpoints) rather than queueing the step fresh."""
    if not job_kinds.kind_of(pipeline).retry(payload):
        return False
    return pipeline != 'analyze' or payload.get('workflow') == 'single_cut_v2'


def plan_item(conn, facts, item, mode='apply', gpu=None):
    """The decision for one failed run: ``action`` is what triage does now (dismiss, rerun, requeue, prerequisite, wait,
    escalate, leave) and ``why`` says why that and not the rule's own action."""
    row, payload = item['row'], item['payload']
    pipeline = row['pipeline']
    kind = job_kinds.kind_of(pipeline)
    cause, cause_text = ('obsolete', CAUSES['obsolete']) if item['obsolete'] else classify(pipeline, row['error'])
    action, limit, fix_text = FIXES[cause]
    keys = sorted(kind.keys(payload, item['cut_ids'])) or [f'{pipeline}:{row["id"]}']
    key = '|'.join(keys)
    tried = attempts(conn, facts.episode_id, key)
    numbers = sorted(facts.numbers[cut_id] for cut_id in item['cut_ids'] if cut_id in facts.numbers)
    why_text = ''
    if action == 'rerun_when_idle':
        action = 'wait' if (gpu if gpu is not None else gpu_busy(conn)) else 'rerun'
        why_text = 'GPU work is running; try again when it is idle' if action == 'wait' else ''
    if cause == 'provider' and _provider_of(payload) not in (None, current_provider(conn)):
        limit = 2  # a different engine now: worth a fresh request even after a failed one
    if action in ('rerun', 'requeue', 'prerequisite'):
        if action == 'rerun' and not _can_rerun(pipeline, payload):
            action = 'requeue'
        if action in ('requeue', 'prerequisite') and pipeline not in REQUEUE:
            action, why_text = 'escalate', f"no automatic way to queue {kind.name(payload).lower()} again"
        elif action == 'prerequisite' and pipeline != 'performance_beats':
            action = 'requeue'
    if action in ('rerun', 'requeue', 'prerequisite', 'wait') and tried >= limit:
        action, why_text = 'escalate', f"the autopilot already fixed it {tried} time(s) and it failed again"
    if action not in ('leave',) and mode == 'off':
        action, why_text = 'leave', 'failed-run fixes are switched off in the Autopilot profile'
    elif action in ('rerun', 'requeue', 'prerequisite', 'dismiss') and mode == 'ask':
        why_text = f"the Autopilot profile asks first (proposed: {action})"
        action = 'escalate'
    first = numbers[0] if numbers else None
    return {'job_id': row['id'], 'pipeline': pipeline, 'label': kind.name(payload), 'cut_ids': item['cut_ids'], 'cuts': numbers,
            'steps': item['steps'], 'error': sentence(row['error'], first if len(numbers) == 1 else None),
            'finished_at': runs.iso(row['finished_at']), 'cause': cause, 'cause_text': cause_text, 'action': action,
            'rule': FIXES[cause][0], 'fix': fix_text, 'why': why_text, 'attempts': tried, 'limit': limit, 'key': key}


def triage(conn, episode_id, mode='apply', job_ids=None):
    """Every failed run nobody has dealt with, with the fix triage would apply now."""
    facts = Facts(conn, episode_id)
    gpu = gpu_busy(conn)
    items = [plan_item(conn, facts, item, mode, gpu) for item in facts.attention if job_ids is None or item['row']['id'] in job_ids]
    counts = {}
    for item in items:
        counts[item['action']] = counts.get(item['action'], 0) + 1
    return {'items': items, 'counts': counts, 'gpu_busy': gpu}
