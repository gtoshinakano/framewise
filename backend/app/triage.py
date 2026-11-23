"""Apply the fix rules of ``diagnose.triage`` to failed runs (docs/autopilot/HARNESS.md, milestone M1).

One call settles every failed run of an episode that nobody has dealt with, in one write transaction, one savepoint per run:

* ``dismiss``: what the run would have produced exists now (or its cuts are archived), so it is only dismissed.
* ``rerun``: the job's own retry, from its snapshot (a visual run resumes its saved checkpoints).
* ``requeue``: the step again with the current settings (a dead provider or an old workflow is not repeated); the user's
  choices in the failed request (a Cut summary instruction) are kept.
* ``prerequisite``: the step it needed first (performance beats without a keyframe analysis queue the keyframes), then the
  step after it.
* ``wait``: GPU memory failures wait until no GPU job runs; ``leave``: sequences are the user's work and are never touched.
* ``escalate``: no rule, the rule's attempts are used up, or the profile says ask. One open question per cause and pipeline
  (``agent_escalations``, kind ``failed_run``); the cuts and runs accumulate on it, and it closes itself once none of its
  runs fails any more.

Each applied fix is an ``agent_decisions`` row (kind ``failed_runs``, tier ``rules``, target the failed job) written in the
savepoint that applied it; its ``evidence.key`` (the failure's ``Kind.keys``) counts the attempts. Undo dismisses nothing
back and cancels nothing that already started: it cancels the jobs the fix created while they are still queued, or clears
the dismissal while it is the agent's.
"""
import json

from fastapi import HTTPException

from . import agent_log, autopilot_profile, diagnose, job_kinds, runs, worker

KIND = 'failed_runs'
ESCALATION = 'failed_run'
FIXING = ('dismiss', 'rerun', 'requeue', 'prerequisite')
LISTED_JOBS = 30


class Unfixable(Exception):
    """The fix rule applies but cannot be carried out for this run (the message says why)."""


def _cuts(conn, cut_ids):
    if not cut_ids:
        return []
    marks = ','.join('?' for _ in cut_ids)
    rows = conn.execute(f'SELECT id,cut_index,start_time,end_time,episode_id FROM cuts WHERE active=1 AND id IN ({marks}) ORDER BY cut_index', cut_ids)
    return [dict(row) for row in rows]


def _by_type(conn, cut_id):
    from . import structured_results
    return {item['type']: item for item in structured_results.list_results(cut_id, conn=conn)}


def _config(conn, cut):
    from . import main
    return main.snapshot_analysis_config(conn, cut).model_dump()


def _text_choices(config):
    return {key: config[key] for key in ('text_provider', 'local_thinking', 'openrouter_model')}


def _visual(conn, episode_id, cut, stages=None):
    """Queue the visual stages a cut is missing (an active visual job of the cut is reused)."""
    from . import main
    pending = worker.pending_jobs(conn, episode_id, ('analyze',), [cut['id']])
    return pending[0] if pending else main.insert_visual_job(conn, episode_id, [cut], stages=stages, only_missing=True)


def _consistency(conn, episode_id, cut, config, retry_of=None):
    """Queue the consistency review of a cut after whatever it is missing: the visual stages, then beats built from the
    current keyframes. Returns the job id."""
    from . import performance_beats, prompt_context, summary_queue
    by_type = _by_type(conn, cut['id'])
    if not worker.pending_jobs(conn, episode_id, ('analyze',), [cut['id']]) and (
            'initial_frame' not in by_type or not ({'keyframe_analysis', 'video_analysis'} & by_type.keys())):
        _visual(conn, episode_id, cut)
        by_type = _by_type(conn, cut['id'])
    parents = []
    beats = performance_beats.pending_job(conn, cut['id'])
    if not beats and 'keyframe_analysis' in by_type and not prompt_context.beats_current(by_type):
        beats = performance_beats.insert(conn, episode_id, cut['id'], _text_choices(config))
    if beats:
        parents.append(beats)
    if not worker.pending_jobs(conn, episode_id, ('analyze',), [cut['id']]) and (problem := summary_queue.missing_evidence(by_type, pending_beats=bool(beats))):
        raise Unfixable(f"cut {cut['cut_index'] + 1}: {problem}")
    return summary_queue.insert(conn, episode_id, {'cut_id': cut['id'], 'config': config, 'batch': True, 'depends_on': parents}, retry_of=retry_of)


def _consistency_ready(conn, episode_id, cut, config):
    """The consistency job a later text step waits for: a pending one, a new one when the review is missing or stale, else None."""
    pending = worker.pending_jobs(conn, episode_id, ('consistency_summary',), [cut['id']])
    if pending:
        return pending[0]
    review = _by_type(conn, cut['id']).get('consistency_summary')
    if review and not review['stale_reason']:
        return None
    return _consistency(conn, episode_id, cut, config)


def requeue(conn, job, payload, cut_ids, prerequisite=False):
    """Queue the failed step again for its cuts with the current settings. Returns the ids the queue answered with."""
    from . import performance_beats, review_answers, summary_jobs
    episode_id, pipeline = job['episode_id'], job['pipeline']
    if pipeline == 'line_splits':
        from . import line_splits
        again = line_splits.queue(conn, episode_id, run_id=payload.get('run_id'))
        if not again:
            raise Unfixable('line splits are switched off in the Autopilot profile')
        return [again]
    if pipeline == 'character_decisions':
        from . import character_decisions
        parts = payload.get('parts') or {}
        again = character_decisions.queue(conn, episode_id, run_id=payload.get('run_id'), first_index=payload.get('first_index') or 0,
                                          last_index=payload.get('last_index'), boxes=parts.get('boxes', True), targets=parts.get('targets', True))
        if not again:
            raise Unfixable('box names and target frames are switched off in the Autopilot profile')
        return [again]
    if pipeline == 'voice_resolver':
        from . import voice_resolver
        again = voice_resolver.queue(conn, episode_id, run_id=payload.get('run_id'), attribution=payload.get('attribution', True))
        if not again:
            raise Unfixable('voices are switched off in the Autopilot profile')
        return [again]
    if pipeline == 'review_decisions':
        from . import review_decisions
        queued = review_decisions.queue(conn, episode_id, cut_ids=cut_ids, run_id=payload.get('run_id'), shadow=bool(payload.get('shadow')), redo=True)
        if queued is None:
            raise Unfixable('review answers are switched off in the Autopilot profile')
        if not queued['job_ids']:
            raise Unfixable('nothing is left to decide: ' + ', '.join(queued['skipped']) if queued['skipped'] else 'nothing is left to decide')
        return queued['job_ids']
    if pipeline == 'missed_cuts':
        from . import missed_cuts
        checked = missed_cuts.queue(conn, episode_id, first_index=payload.get('first_index') or 0, last_index=payload.get('last_index'),
                                    cut_ids=payload.get('cut_ids'), redo=bool(payload.get('redo')), run_id=payload.get('run_id'))
        if not checked:
            raise Unfixable('missed-cut checks are switched off in the Autopilot profile')
        return [checked]
    if pipeline == 'subtitles':
        from . import main
        language = payload.get('language') or (conn.execute('SELECT subtitle_language FROM episodes WHERE id=?', (episode_id,)).fetchone() or [None])[0]
        if not language:
            raise Unfixable('the episode has no subtitle language')
        # the new run reuses every saved reading; accept_unread lets it finish when only a handful stay unreadable
        return [main.queue_subtitles(conn, episode_id, language, accept_unread=True)]
    cuts = _cuts(conn, cut_ids)
    if not cuts:
        raise Unfixable('its cuts are archived')
    ids = []
    if pipeline == 'analyze':
        stages = job_kinds.stages_of(payload) if payload.get('workflow') == 'single_cut_v2' else None
        from . import main
        ids.append(main.insert_visual_job(conn, episode_id, cuts, stages=stages, batch=len(cuts) > 1, only_missing=True))
        return ids
    for cut in cuts:
        config = _config(conn, cut)
        if pipeline == 'performance_beats':
            if prerequisite or 'keyframe_analysis' not in _by_type(conn, cut['id']):
                ids.append(_visual(conn, episode_id, cut, ['keyframe_analysis']))
            ids.append(performance_beats.insert(conn, episode_id, cut['id'], _text_choices(config), retry_of=job['id']))
        elif pipeline == 'consistency_summary':
            ids.append(_consistency(conn, episode_id, cut, config, retry_of=job['id']))
        elif pipeline == 'review_answers':
            parent = _consistency_ready(conn, episode_id, cut, config)
            ids += [parent] if parent else []
            ids.append(review_answers.insert(conn, episode_id, {'cut_id': cut['id'], 'config': config, 'auto_save': bool(payload.get('auto_save')),
                                                                 'depends_on': [parent] if parent else []}))
        elif pipeline == 'regen':
            parent = _consistency_ready(conn, episode_id, cut, config)
            ids += [parent] if parent else []
            request = summary_jobs.choices(cut['id'], config, mode=payload.get('mode') or 'regenerate', instruction=payload.get('instruction') or '',
                                           edited_prompt=payload.get('edited_prompt'), request_override=payload.get('request_override'))
            ids.append(summary_jobs.insert(conn, episode_id, request, parents=[parent] if parent else [], batch=True, retry_of=job['id']))
        else:
            raise Unfixable(f'no automatic way to queue {pipeline} again')
    return list(dict.fromkeys(ids))


def _created_since(conn, episode_id, rowid):
    return [row[0] for row in conn.execute('SELECT id FROM jobs WHERE episode_id=? AND rowid>? ORDER BY rowid', (episode_id, rowid))]


def _escalate(conn, episode_id, item, run_id):
    """Add the run to the open question for its cause and pipeline."""
    group = f"{item['cause']}:{item['pipeline']}"
    existing = conn.execute("SELECT evidence_json FROM agent_escalations WHERE episode_id=? AND kind=? AND group_key=? AND status='open'",
                            (episode_id, ESCALATION, group)).fetchone()
    jobs = {entry['job_id']: entry for entry in (json.loads(existing['evidence_json']).get('jobs', []) if existing and existing['evidence_json'] else [])}
    jobs[item['job_id']] = {'job_id': item['job_id'], 'cuts': item['cuts'], 'error': item['error'][:300], 'finished_at': item['finished_at'],
                            'why': item['why']}
    listed = list(jobs.values())[-LISTED_JOBS:]
    count = len(jobs)
    question = (f"{item['label']} failed ({diagnose.CAUSES.get(item['cause'], item['cause'])}) and the autopilot will not fix it on its own"
                f"{': ' + item['why'] if item['why'] else ''}. Last error: {item['error'][:300]}"
                f"{f' ({count} runs)' if count > 1 else ''}. Retry, change its input, or dismiss it?")
    return agent_log.escalate(conn, episode_id=episode_id, kind=ESCALATION, group_key=group, question=question[:2000],
                              cut_ids=item['cut_ids'], options=['Retry', 'Dismiss'], proposal={'rule': item['rule'], 'fix': item['fix']},
                              evidence={'cause': item['cause'], 'pipeline': item['pipeline'], 'jobs': listed}, run_id=run_id)


def close_settled(conn, episode_id, failing):
    """Close the failed-run questions none of whose runs is still failing (the person retried or dismissed them)."""
    closed = 0
    for row in conn.execute("SELECT id,evidence_json FROM agent_escalations WHERE episode_id=? AND kind=? AND status='open'",
                            (episode_id, ESCALATION)).fetchall():
        jobs = [entry.get('job_id') for entry in json.loads(row['evidence_json'] or '{}').get('jobs', [])]
        if jobs and not set(jobs) & failing:
            closed += agent_log.close_escalation(conn, row['id'], 'resolved', {'by': 'triage', 'reason': 'none of its runs is failing any more'})
    return closed


def settle(conn, episode_id):
    """``close_settled`` with the runs failing now (after a retry or dismiss in the queue drawer or the cut dialog)."""
    return close_settled(conn, episode_id, {row['id'] for row in runs.attention_rows(conn, episode_id)})


def mode_for(conn, episode_id, run_id=None):
    """The profile's mode for failed-run fixes: the run's snapshot when there is a run, else the project's profile."""
    if run_id:
        row = conn.execute('SELECT policy_json,episode_id FROM agent_runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row['episode_id'] != episode_id:
            raise ValueError('The agent run belongs to another episode')
        policy = json.loads(row['policy_json'] or '{}')
        if (policy.get('modes') or {}).get(KIND):
            return policy['modes'][KIND]
    project = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    profile, _ = autopilot_profile.load(conn, project['project_id'])
    return profile['modes'][KIND]


def apply(conn, episode_id, run_id=None, job_ids=None):
    """Settle the failed runs inside the caller's write transaction. Returns what happened to each, grouped by outcome."""
    mode = mode_for(conn, episode_id, run_id)
    plan = diagnose.triage(conn, episode_id, mode, set(job_ids) if job_ids else None)
    outcome = {'applied': [], 'escalated': [], 'waiting': [], 'left': [], 'errors': [], 'mode': mode, 'gpu_busy': plan['gpu_busy']}
    rows = {row['id']: row for row in runs.attention_rows(conn, episode_id)}
    for item in plan['items']:
        if item['action'] == 'wait':
            outcome['waiting'].append(item)
            continue
        if item['action'] == 'leave':
            outcome['left'].append(item)
            continue
        if item['action'] == 'escalate':
            item['escalation_id'] = _escalate(conn, episode_id, item, run_id)
            outcome['escalated'].append(item)
            continue
        job = rows[item['job_id']]
        payload = json.loads(conn.execute('SELECT payload_json FROM jobs WHERE id=?', (job['id'],)).fetchone()[0] or '{}')
        conn.execute('SAVEPOINT triage_fix')
        try:
            before_rowid = conn.execute('SELECT COALESCE(MAX(rowid),0) FROM jobs').fetchone()[0]
            if item['action'] == 'dismiss':
                conn.execute('UPDATE jobs SET dismissed_at=CURRENT_TIMESTAMP WHERE id=?', (job['id'],))
                stamp = conn.execute('SELECT dismissed_at FROM jobs WHERE id=?', (job['id'],)).fetchone()[0]
                before, after, verb = {'dismissed_at': None}, {'dismissed_at': stamp}, 'Dismissed'
            else:
                if item['action'] == 'rerun':
                    from . import main
                    answered = [main.retry_job(conn, job['id'])]
                else:
                    answered = requeue(conn, dict(job), payload, item['cut_ids'], prerequisite=item['action'] == 'prerequisite')
                created = _created_since(conn, episode_id, before_rowid)
                item['job_ids'] = answered
                before, after = {'queued': []}, {'queued': created, 'answered': answered}
                verb = {'rerun': 'Retried', 'requeue': 'Queued again', 'prerequisite': 'Queued the missing step and then'}[item['action']]
            cuts = f" (cut{'s' if len(item['cuts']) > 1 else ''} {', '.join(map(str, item['cuts'][:6]))})" if item['cuts'] else ''
            item['decision_id'] = agent_log.record(
                conn, episode_id=episode_id, run_id=run_id, cut_id=item['cut_ids'][0] if len(item['cut_ids']) == 1 else None,
                kind=KIND, tier='rules', target_table='jobs', target_id=job['id'], before=before, after=after,
                evidence={'key': item['key'], 'cause': item['cause'], 'action': item['action'], 'pipeline': item['pipeline'],
                          'error': item['error'][:300], 'attempt': item['attempts'] + 1},
                summary=f"{verb} {item['label'].lower()}{cuts}: {item['cause_text']}")
        except (Unfixable, HTTPException, ValueError, KeyError) as exc:
            conn.execute('ROLLBACK TO triage_fix')
            conn.execute('RELEASE triage_fix')
            item['why'] = getattr(exc, 'detail', None) or str(exc)
            item['action'] = 'escalate'
            item['escalation_id'] = _escalate(conn, episode_id, item, run_id)
            outcome['errors'].append({'job_id': item['job_id'], 'error': item['why']})
            outcome['escalated'].append(item)
            continue
        conn.execute('RELEASE triage_fix')
        outcome['applied'].append(item)
    failing = {row['id'] for row in runs.attention_rows(conn, episode_id)}
    outcome['closed_escalations'] = close_settled(conn, episode_id, failing)
    return outcome


def undo_fix(conn, decision):
    """Undo one failed-run fix: clear the agent's dismissal, or cancel the jobs it created while none of them has started."""
    after = json.loads(decision['after_json'] or '{}')
    if decision['kind'] != KIND or not ({'dismissed_at', 'queued'} & after.keys()):
        raise agent_log.UndoConflict('Only failed-run fixes on jobs can be undone automatically')
    if 'dismissed_at' in after:
        row = conn.execute('SELECT dismissed_at FROM jobs WHERE id=?', (decision['target_id'],)).fetchone()
        if row is None:
            raise agent_log.UndoConflict('The failed run no longer exists')
        if row['dismissed_at'] != after['dismissed_at']:
            raise agent_log.UndoConflict('The run was dismissed or restored again since; undo would overwrite that')
        conn.execute('UPDATE jobs SET dismissed_at=NULL WHERE id=?', (decision['target_id'],))
        return
    ids = after.get('queued') or []
    if not ids:
        return
    marks = ','.join('?' for _ in ids)
    rows = conn.execute(f'SELECT * FROM jobs WHERE id IN ({marks})', ids).fetchall()
    if any(row['status'] not in ('queued', 'cancelled') for row in rows):
        raise agent_log.UndoConflict('A job this fix queued has already started or finished; cancel it from the queue instead')
    worker.cancel_jobs(conn, [row for row in rows if row['status'] == 'queued'], 'Undone: the autopilot fix was taken back')


agent_log.UNDO['jobs'] = undo_fix
