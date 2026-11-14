"""Changes to a sequence that come from a person or the assistant: queueing steps and revising results.

A hand edit is checked exactly like a model answer (the same validators, the same H3 format check),
so nothing invalid can reach the download. Used by the HTTP API and by Cut Assistance when a
proposal is previewed or applied; it does not import FastAPI.
"""
import uuid

from . import sequence_direction as direction, sequence_evidence as evidence, sequence_prompt, sequence_store as store, sequence_timeline, sequence_view, worker

DIRECT_STEPS = ('direct', 'bible', 'plan')


class EditError(Exception):
    """A request that cannot be carried out; ``status`` is the HTTP status the API answers with."""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


def config_snapshot(conn):
    """The text engine a job will use, as chosen in the text engine dialog when it is queued."""
    row = conn.execute('SELECT text_provider,local_thinking,openrouter_model FROM text_generation_settings WHERE id=1').fetchone()
    if not row:
        return {'text_provider': 'openrouter', 'local_thinking': False, 'openrouter_model': 'openrouter/free'}
    return {'text_provider': row['text_provider'], 'local_thinking': bool(row['local_thinking']), 'openrouter_model': row['openrouter_model']}


def needs_writing(view):
    return not view['result'] or bool(view['stale']) or not (view['result']['check'] or {}).get('ok')


def queue(conn, sequence, step, shot_keys=None, only_missing=True, instruction=''):
    """Queue a step of the direction and return the new job ids.

    ``direct`` writes the bible and then the plan and stops there; ``shots`` writes every shot that is
    missing, stale or failing its check (or exactly ``shot_keys``) and reviews the sequence after a whole
    batch; ``bible``, ``plan`` and ``review`` run that one step again.
    """
    if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts','character_detect') AND status IN ('queued','running')", (sequence['episode_id'],)).fetchone():
        raise EditError(409, 'Wait for media preparation or detection to finish')
    pending = store.pending_jobs(conn, sequence['id'])
    config = config_snapshot(conn)
    cuts = store.current_cuts(conn, sequence)
    if cuts is None:
        raise EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
    job_ids = []
    if step in DIRECT_STEPS:
        if pending:
            raise EditError(409, 'This sequence already has jobs queued or running; wait for them or cancel them')
        length = cuts[-1]['end_time'] - cuts[0]['start_time']
        if length > sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001:
            raise EditError(409, f'The cuts of this sequence now last {length:.1f} s; a sequence may last at most {sequence_timeline.MAX_SEQUENCE_SECONDS:g} s')
        missing = evidence.missing_cuts(evidence.load(conn, sequence, cuts), cuts)
        if missing:
            raise EditError(409, 'These cuts need a first-frame and a video analysis first: ' + ', '.join(map(str, missing)))
        if store.cuts_changed(sequence, cuts):
            sequence = store.resnapshot(conn, sequence, cuts)
        if step == 'plan':
            if not store.current(conn, sequence['id'], 'bible'):
                raise EditError(409, 'Write the sequence bible first')
            base = store.current(conn, sequence['id'], 'plan')
            job_ids.append(store.insert_job(conn, sequence, {'step': 'plan', 'config': config, 'base_id': base['id'] if base else None}))
        else:
            base = store.current(conn, sequence['id'], 'bible')
            job_ids.append(store.insert_job(conn, sequence, {'step': 'bible', 'config': config, 'base_id': base['id'] if base else None, 'then_plan': step == 'direct'}))
    elif step == 'shots':
        if any(job['step'] in ('bible', 'plan') for job in pending):
            raise EditError(409, 'Wait for the bible and the shot plan to finish')
        current = sequence_view.state(conn, sequence)
        if current['changed']:
            raise EditError(409, 'The cuts of this sequence changed; direct the sequence again')
        if not current['bible'] or not current['shots']:
            raise EditError(409, 'The shot plan no longer fits the cuts; direct the sequence again' if current['plan_problems'] else 'Direct the sequence first: it needs a bible and a shot plan')
        views = sequence_view.shot_views(current)
        if shot_keys is not None:
            unknown = set(shot_keys) - {view['key'] for view in views}
            if unknown:
                raise EditError(404, 'That shot is not part of the current plan')
            targets = [view for view in views if view['key'] in shot_keys]
        else:
            targets = [view for view in views if needs_writing(view) or not only_missing]
        busy = {job['shot_key'] for job in pending if job['step'] == 'shot'}
        targets = [view for view in targets if view['key'] not in busy]
        if not targets:
            raise EditError(409, 'That shot is already queued' if shot_keys is not None else 'Every shot is already written and current')
        batch = str(uuid.uuid4())
        for view in targets:
            job_ids.append(store.insert_job(conn, sequence, {'step': 'shot', 'shot_key': view['key'], 'config': config, 'base_id': view['result']['id'] if view['result'] else None,
                                                            'instruction': instruction.strip(), 'batch_id': batch, 'then_review': shot_keys is None}))
    elif step == 'review':
        if any(job['step'] == 'review' for job in pending):
            raise EditError(409, 'A review is already queued or running')
        if not store.current_shots(conn, sequence['id']):
            raise EditError(409, 'Write the shot prompts before reviewing them')
        base = store.current(conn, sequence['id'], 'review')
        job_ids.append(store.insert_job(conn, sequence, {'step': 'review', 'config': config, 'base_id': base['id'] if base else None}))
    else:
        raise EditError(422, 'Unknown step')
    worker.sync_episode_status(conn, sequence['episode_id'])
    return job_ids


def cancel(conn, sequence):
    """Cancel every queued or running job of the sequence; a running one stops at its next checkpoint."""
    jobs = store.pending_jobs(conn, sequence['id'])
    for job in jobs:
        message = 'Cancelled before processing' if job['status'] == 'queued' else 'Cancelled during processing'
        conn.execute("UPDATE jobs SET status='failed',stage='Cancelled',error=?,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status IN ('queued','running')", (message, job['id']))
    worker.sync_episode_status(conn, sequence['episode_id'])
    return len(jobs)


def normal_spans(shots, timeline):
    """Check hand-edited plan shots and keep only the fields a plan stores."""
    problems = sequence_timeline.check_spans(shots, timeline)
    if problems:
        raise sequence_timeline.PlanError(problems)
    spans = []
    for shot in shots:
        t0, t1 = shot['start_time'] - timeline['origin'], shot['end_time'] - timeline['origin']
        covered = {cut['id'] for cut in sequence_timeline.covered_cuts(timeline, t0, t1)}
        spans.append({'start_time': shot['start_time'], 'end_time': shot['end_time'], 'frame_cut_id': shot.get('frame_cut_id') if shot.get('frame_cut_id') in covered else None,
                      'scene': shot.get('scene') if isinstance(shot.get('scene'), str) else None,
                      'intent': shot['intent'].strip() if isinstance(shot.get('intent'), str) else '', 'staging': shot['staging'].strip() if isinstance(shot.get('staging'), str) else ''})
    return spans


def preview_shot(current, shot, content):
    """Validate and render a shot's direction against the current bible and plan.

    Returns ``(direction, prompt, check, derived, hashes)``; raises ``ValueError`` with the problems.
    """
    roster, scenario, dialogue = evidence.shot_parts(current['ctx'], current['bible'], shot)
    normal = direction.validate_shot(content, shot, current['bible'], roster, scenario, dialogue)
    prompt, verdict, derived = sequence_prompt.build(shot, normal, current['bible'], roster, scenario, dialogue)
    return normal, prompt, verdict, derived, evidence.shot_hashes(current['ctx'], current['bible'], shot)


def revise(conn, sequence, row, content, origin='manual_edit', provenance=None, current=None):
    """Append a hand or assistant revision of a bible, plan or shot and return its id."""
    if row['superseded_by']:
        raise EditError(409, 'This result already has a newer revision')
    current = current or sequence_view.state(conn, sequence)
    if current['cuts'] is None:
        raise EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
    settings = dict(provenance or {})
    try:
        if row['type'] == 'bible':
            normal = direction.validate_bible(content, current['ctx'], current['timeline'])
            return store.save(conn, sequence['id'], 'bible', normal, origin=origin, settings=settings, expected_current=row['id'])
        if row['type'] == 'plan':
            spans = normal_spans(content.get('shots'), current['timeline'])
            notes = content.get('notes') if isinstance(content.get('notes'), str) else ''
            return store.save(conn, sequence['id'], 'plan', {'shots': spans, 'notes': notes}, origin=origin, settings=settings, expected_current=row['id'])
        if row['type'] == 'shot':
            if not current['bible']:
                raise EditError(409, 'Write the sequence bible first')
            shot = next((item for item in current['shots'] or [] if item['key'] == row['shot_key']), None)
            if shot is None:
                raise EditError(409, 'This shot is no longer part of the plan')
            normal, prompt, verdict, derived, hashes = preview_shot(current, shot, content)
            settings.update(check=verdict, derived=derived, hashes=hashes)
            return store.save(conn, sequence['id'], 'shot', normal, shot_key=row['shot_key'], rendered=prompt, origin=origin, settings=settings, expected_current=row['id'])
    except ValueError as exc:
        raise EditError(422, str(exc)) from exc
    except RuntimeError as exc:
        raise EditError(409, str(exc)) from exc
    raise EditError(409, 'A review cannot be edited; run it again')


def reshape(conn, sequence, shot_key, time=None, origin='manual_edit', provenance=None):
    """Merge a planned shot with the next one, or split it at an episode time; returns the new plan revision id."""
    current = sequence_view.state(conn, sequence)
    row, plan = current['plan_row'], current['plan']
    if not row or not plan or current['timeline'] is None:
        raise EditError(409, 'Plan the shots first')
    spans = plan['shots']
    index = next((position for position, span in enumerate(spans) if sequence_timeline.shot_key(span['start_time'], span['end_time']) == shot_key), None)
    if index is None:
        raise EditError(404, 'That shot is not part of the current plan')
    try:
        changed = sequence_timeline.merge_spans(spans, index) if time is None else sequence_timeline.split_span(spans, index, time)
        return revise(conn, sequence, row, {'shots': changed, 'notes': plan.get('notes')}, origin, provenance, current)
    except sequence_timeline.PlanError as exc:
        raise EditError(422, ' '.join(exc.errors)) from exc
