"""Changes to a sequence that come from a person or the assistant: queueing steps and revising results.

A hand edit is checked exactly like a model answer (the same validators, the same H3 format check),
so nothing invalid can reach the download. Used by the HTTP API and by Cut Assistance when a
proposal is previewed or applied; it does not import FastAPI.
"""
import json
import logging
import math
import uuid

from . import sequence_compile, sequence_direction as direction, sequence_evidence as evidence, sequence_prompt, sequence_store as store, sequence_timeline, sequence_view, worker

DIRECT_STEPS = ('direct', 'bible', 'plan')
BUSY_EPISODE = "SELECT 1 FROM jobs WHERE episode_id=? AND pipeline IN ('prepare','hardcuts','character_detect') AND status IN ('queued','running')"
log = logging.getLogger(__name__)


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


def require_reviewed(conn, sequence, cuts):
    """Direction starts from the dialogue the user reviewed; a sequence in which nobody speaks has none to review."""
    if store.reviewed_compilation(conn, sequence) is not None:
        return
    if store.current(conn, sequence['id'], 'compilation') is None and not store.transcript_lines(conn, sequence['episode_id'], cuts[0]['start_time'], cuts[-1]['end_time']):
        return
    raise EditError(409, 'Review the dialogue first: open the Dialogue tab, check it and save it')


def queue_compile(conn, sequence, extend_from=None, extend_cut=None):
    """Start the dialogue compilation in the transaction that created the sequence (or added a cut to it).

    With ``extend_from`` (episode seconds, the old end) only the added cut is compiled and appended to the
    current dialogue; a sequence that has no dialogue yet is compiled whole.
    """
    base = store.current(conn, sequence['id'], 'compilation')
    payload = {'step': 'compile', 'config': config_snapshot(conn), 'base_id': base['id'] if base else None}
    if extend_from is not None and base:
        payload.update(extend_from=extend_from, extend_cut=extend_cut)
    return store.insert_job(conn, sequence, payload)


def extend(conn, sequence):
    """Add the episode's next cut to the end of the sequence and start updating everything built on the range.

    The range and its snapshot move in this transaction, together with the job that compiles only the added
    cut's dialogue (the user's edits stay). ``extension_json`` remembers what was added until the direction
    is queued again: once the dialogue is confirmed, ``continue_extension`` directs the sequence again and,
    when it had shot prompts, writes them again and reviews them.
    """
    if conn.execute(BUSY_EPISODE, (sequence['episode_id'],)).fetchone():
        raise EditError(409, 'Wait for media preparation or detection to finish')
    if store.pending_jobs(conn, sequence['id']):
        raise EditError(409, 'This sequence already has jobs queued or running; wait for them or cancel them')
    cuts = store.current_cuts(conn, sequence)
    if cuts is None:
        raise EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
    following = store.next_cut(conn, sequence, cuts)
    if following is None:
        raise EditError(409, f"Cut {cuts[-1]['cut_index'] + 1} is the last cut of the episode")
    number = following['cut_index'] + 1
    length = following['end_time'] - cuts[0]['start_time']
    if length > sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001:
        raise EditError(422, f'With cut {number} the sequence would last {length:.1f} s; a sequence may last at most {sequence_timeline.MAX_SEQUENCE_SECONDS:g} s')
    if evidence.missing_cuts(evidence.load(conn, sequence, [following]), [following]):
        raise EditError(409, f'Cut {number} needs a first-frame and a video analysis first; run them from the cut, then add it')
    earlier = store.extension(sequence) or {}
    marker = {'cut_ids': [*(earlier.get('cut_ids') or []), following['id']], 'cuts': [*(earlier.get('cuts') or []), number],
              'from_time': earlier.get('from_time', cuts[-1]['end_time']),
              'rewrite_shots': bool(earlier.get('rewrite_shots')) or bool(store.current_shots(conn, sequence['id']))}
    title = sequence['title']
    if title == f"Cuts {cuts[0]['cut_index'] + 1}–{cuts[-1]['cut_index'] + 1}":
        title = f"Cuts {cuts[0]['cut_index'] + 1}–{number}"  # the default title follows the range
    conn.execute('UPDATE sequences SET last_cut_id=?,title=?,extension_json=? WHERE id=?', (following['id'], title, json.dumps(marker), sequence['id']))
    sequence = store.resnapshot(conn, {**sequence, 'last_cut_id': following['id'], 'title': title, 'extension_json': json.dumps(marker)}, [*cuts, following])
    queue_compile(conn, sequence, extend_from=cuts[-1]['end_time'], extend_cut=number)
    worker.sync_episode_status(conn, sequence['episode_id'])
    return marker


def continue_extension(conn, sequence):
    """After a cut was added and the dialogue is confirmed: direct the sequence again, and write its shots again if it had any.

    Does nothing while no cut waits for it or the dialogue still needs review. A refusal (for example while
    preparation runs) leaves the marker, so the next confirmation tries again. Returns the queued job ids.
    """
    marker = store.extension(sequence)
    if not marker:
        return []
    try:
        return queue(conn, sequence, 'direct', then_shots=bool(marker.get('rewrite_shots')))
    except EditError as exc:
        log.info('Sequence %s: direction after the added cut not queued yet: %s', sequence['id'], exc.detail)
        return []


def confirm_compilation(conn, sequence, result_id):
    """Mark the current compilation as the dialogue the user reviewed; Direction and every shot use it from now on."""
    row = store.current(conn, sequence['id'], 'compilation')
    if not row or row['id'] != result_id:
        raise EditError(409, 'The dialogue changed; reload it and save again')
    conn.execute('UPDATE sequences SET reviewed_compilation_id=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (result_id, sequence['id']))
    continue_extension(conn, {**sequence, 'reviewed_compilation_id': result_id})
    return result_id


def use_source_dialogue(conn, sequence):
    """Save the raw transcript as the dialogue (no corrections) and confirm it."""
    if store.pending_jobs(conn, sequence['id']):
        raise EditError(409, 'This sequence already has jobs queued or running; wait for them or cancel them')
    cuts = store.current_cuts(conn, sequence)
    if cuts is None:
        raise EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
    timeline = store.raw_timeline_for(conn, sequence, cuts)
    roster = {row['title']: row['id'] for row in conn.execute("SELECT c.id,c.title FROM context_items c JOIN episodes e ON e.project_id=c.project_id WHERE e.id=? AND c.kind='character'", (sequence['episode_id'],))}
    content = sequence_compile.validate(sequence_compile.source_lines(timeline), timeline['duration'], roster)
    base = store.current(conn, sequence['id'], 'compilation')
    source, fragments, subtitles = store.source_fingerprint(conn, sequence, cuts)
    result_id = store.save(conn, sequence['id'], 'compilation', content, origin='source', settings={'note': 'The original transcript, without corrections'}, expected_current=base['id'] if base else None)
    store.record_sources(conn, sequence['id'], result_id, source, None, len(fragments), len(subtitles))
    conn.execute('UPDATE sequences SET reviewed_compilation_id=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (result_id, sequence['id']))
    continue_extension(conn, {**sequence, 'reviewed_compilation_id': result_id})
    return result_id


def needs_writing(view):
    return not view['result'] or bool(view['stale']) or not (view['result']['check'] or {}).get('ok')


def queue(conn, sequence, step, shot_keys=None, only_missing=True, instruction='', trigger=None, then_shots=False, instruction_source='user'):
    """Queue a step of the direction and return the new job ids.

    ``direct`` writes the bible and then the plan and stops there; ``shots`` writes every shot that is
    missing, stale or failing its check (or exactly ``shot_keys``) and reviews the sequence after a whole
    batch; ``bible``, ``plan`` and ``review`` run that one step again. ``trigger`` says why shots are
    rewritten when the user did not write an instruction (``voices``), for the shot's history. With
    ``then_shots``, ``direct`` goes on after the plan and writes every shot that needs it (a cut was added).
    ``instruction_source`` is ``user`` for what the user wrote (every later rewrite of the shot keeps following it) or ``review``
    for a suggestion of the automatic review (used once, never remembered).
    """
    if conn.execute(BUSY_EPISODE, (sequence['episode_id'],)).fetchone():
        raise EditError(409, 'Wait for media preparation or detection to finish')
    pending = store.pending_jobs(conn, sequence['id'])
    config = config_snapshot(conn)
    cuts = store.current_cuts(conn, sequence)
    if cuts is None:
        raise EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
    job_ids = []
    if step == 'compile':
        if pending:
            raise EditError(409, 'This sequence already has jobs queued or running; wait for them or cancel them')
        base = store.current(conn, sequence['id'], 'compilation')
        job_ids.append(store.insert_job(conn, sequence, {'step': 'compile', 'config': config, 'base_id': base['id'] if base else None}))
    elif step in DIRECT_STEPS:
        if pending:
            raise EditError(409, 'This sequence already has jobs queued or running; wait for them or cancel them')
        require_reviewed(conn, sequence, cuts)
        length = cuts[-1]['end_time'] - cuts[0]['start_time']
        if length > sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001:
            raise EditError(409, f'The cuts of this sequence now last {length:.1f} s; a sequence may last at most {sequence_timeline.MAX_SEQUENCE_SECONDS:g} s')
        missing = evidence.missing_cuts(evidence.load(conn, sequence, cuts), cuts)
        if missing:
            raise EditError(409, 'These cuts need a first-frame and a video analysis first: ' + ', '.join(map(str, missing)))
        if store.cuts_changed(sequence, cuts):
            sequence = store.resnapshot(conn, sequence, cuts)
        # The direction is being redone now: an added cut no longer waits for it.
        conn.execute('UPDATE sequences SET extension_json=NULL WHERE id=?', (sequence['id'],))
        if step == 'plan':
            if not store.current(conn, sequence['id'], 'bible'):
                raise EditError(409, 'Write the sequence bible first')
            base = store.current(conn, sequence['id'], 'plan')
            job_ids.append(store.insert_job(conn, sequence, {'step': 'plan', 'config': config, 'base_id': base['id'] if base else None}))
        else:
            base = store.current(conn, sequence['id'], 'bible')
            job_ids.append(store.insert_job(conn, sequence, {'step': 'bible', 'config': config, 'base_id': base['id'] if base else None, 'then_plan': step == 'direct',
                                                            **({'then_shots': True} if then_shots and step == 'direct' else {})}))
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
                                                            'instruction': instruction.strip(), 'batch_id': batch, 'then_review': shot_keys is None,
                                                            **({'instruction_source': 'review' if instruction_source == 'review' else 'user'} if instruction.strip() else {}),
                                                            **({'trigger': trigger} if trigger else {})}))
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
    worker.cancel_jobs(conn, jobs)
    worker.sync_episode_status(conn, sequence['episode_id'])
    return len(jobs)


def normal_spans(shots, timeline):
    """Check hand-edited plan shots (one per cut, see ``sequence_timeline.check_spans``) and keep only the fields a plan stores."""
    problems = sequence_timeline.check_spans(shots, timeline)
    if problems:
        raise sequence_timeline.PlanError(problems)
    return [{'start_time': shot['start_time'], 'end_time': shot['end_time'], 'frame_cut_id': sequence_timeline.owner_of(shot, timeline)['id'],
             'scene': shot.get('scene') if isinstance(shot.get('scene'), str) else None,
             'intent': shot['intent'].strip() if isinstance(shot.get('intent'), str) else ''} for shot in shots]


def normal_decisions(items, timeline):
    """The plan's boundary decisions: one per hard cut inside speech that the user did not place, ``{after_cut_id, choice, reason, by}``."""
    speaking = {boundary['after_cut_id'] for boundary in timeline['boundaries'] if boundary['speech'] and not boundary.get('fixed')}
    order = {boundary['after_cut_id']: index for index, boundary in enumerate(timeline['boundaries'])}
    found = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get('after_cut_id') in speaking and item.get('choice') in sequence_timeline.CHOICES:
            found[item['after_cut_id']] = {'after_cut_id': item['after_cut_id'], 'choice': item['choice'], 'reason': item['reason'].strip() if isinstance(item.get('reason'), str) else '',
                                           'by': item.get('by') if item.get('by') in ('director', 'app', 'user') else 'user'}
    return sorted(found.values(), key=lambda item: order[item['after_cut_id']])


def boundary_plan(plan, timeline, after_cut_id, choice, reason=''):
    """A plan with the shot boundary at the hard cut after ``after_cut_id`` moved to ``choice``; the other boundaries, intents and voices stay."""
    boundary = next((item for item in timeline['boundaries'] if item['after_cut_id'] == after_cut_id), None)
    if boundary is None:
        raise EditError(404, 'There is no hard cut after that cut in this sequence')
    try:
        spans = sequence_timeline.move_boundary(plan['shots'], timeline, boundary['id'], choice)
    except sequence_timeline.PlanError as exc:
        raise EditError(422, ' '.join(exc.errors)) from exc
    decisions = [item for item in plan.get('boundaries') or [] if isinstance(item, dict) and item.get('after_cut_id') != after_cut_id]
    if boundary['speech']:
        decisions.append({'after_cut_id': after_cut_id, 'choice': choice, 'reason': reason.strip(), 'by': 'user'})
    return {**plan, 'shots': spans, 'boundaries': normal_decisions(decisions, timeline)}


def set_boundary(conn, sequence, after_cut_id, choice, origin='manual_edit', provenance=None):
    """Move one hard cut's shot boundary (keep the cut, or before or after the phrase it falls into) and save the plan; returns its revision id.

    The two shots on either side get new intervals, so they must be written again; every other shot keeps its prompt.
    """
    current = sequence_view.state(conn, sequence)
    row, plan = current['plan_row'], current['plan']
    if current['cuts'] is None or current['changed']:
        raise EditError(409, 'The cuts of this sequence changed; direct the sequence again')
    if not row or not current['shots']:
        raise EditError(409, 'The shot plan no longer fits the cuts; plan the shots again' if current['plan_problems'] else 'Plan the shots first')
    if any(job['step'] in ('bible', 'plan') for job in store.pending_jobs(conn, sequence['id'])):
        raise EditError(409, 'Wait for the shot plan to finish')
    content = boundary_plan(plan, current['timeline'], after_cut_id, choice)
    if content['shots'] == plan['shots'] and content['boundaries'] == (plan.get('boundaries') or []):
        return row['id']
    return revise(conn, sequence, row, content, origin, provenance, current)


RELEASED = 'Chosen by the app when you handed this boundary back; plan the shots again to let the director decide.'
BESIDE = 'Put back on its hard cut by the app to fit beside a boundary you placed; plan the shots again to let the director decide.'


def normal_points(points, timeline):
    """Check the user's placed boundaries: at most one ``{after_cut_id, time}`` per hard cut, inside the sequence; returns them in order."""
    order = {boundary['after_cut_id']: index for index, boundary in enumerate(timeline['boundaries'])}
    start, end = timeline['origin'], timeline['origin'] + timeline['duration']
    normal, errors, seen = [], [], set()
    for number, item in enumerate(points if isinstance(points, list) else [], 1):
        cut_id = item.get('after_cut_id') if isinstance(item, dict) else None
        time = item.get('time') if isinstance(item, dict) else None
        if cut_id not in order:
            errors.append(f'point {number}: no hard cut follows that cut in this sequence')
        elif cut_id in seen:
            errors.append(f'point {number}: that hard cut already has a boundary')
        elif isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or not start < time < end:
            errors.append(f'point {number} must lie inside the sequence ({start:g} to {end:g} s, episode time)')
        else:
            seen.add(cut_id)
            normal.append({'after_cut_id': cut_id, 'time': round(float(time), 6)})
    if errors:
        raise EditError(422, '; '.join(errors))
    return sorted(normal, key=lambda item: order[item['after_cut_id']])


def points_plan(plan, before, timeline, ctx, bible):
    """The plan with its shots on the user's placed boundaries, or None when it cannot be made to fit (plan the shots again).

    ``before`` are the ids of the boundaries that were placed until now; one that is no longer placed goes
    back to the app's choice (``direction.default_choice``, else its hard cut). A director's move next to a
    placed boundary that no longer fits goes back to its hard cut. Intents, scenes, voices and the parts
    of cuts that touch no moved boundary stay as they are.
    """
    boundaries = timeline['boundaries']
    fixed = sequence_timeline.placed(timeline)
    decisions = {item['after_cut_id']: item for item in plan.get('boundaries') or [] if isinstance(item, dict) and item.get('after_cut_id')}
    positions = {}

    def spans_for(trial):
        try:
            spans = sequence_timeline.apply_points(plan['shots'], timeline, trial)
        except sequence_timeline.PlanError:
            return None
        return None if sequence_timeline.check_spans(spans, timeline) else spans

    def to_cut(boundary, reason):
        positions[boundary['id']] = (boundary['t'], boundary['time'])
        decisions.pop(boundary['after_cut_id'], None)
        if boundary['speech']:
            decisions[boundary['after_cut_id']] = {'after_cut_id': boundary['after_cut_id'], 'choice': 'keep_cut', 'reason': reason, 'by': 'app'}

    released = [boundary for boundary in boundaries if boundary['id'] in before and boundary['id'] not in fixed]
    for boundary in released:
        to_cut(boundary, RELEASED)
    spans = spans_for(positions)
    if spans is None:
        index = {boundary['id']: position for position, boundary in enumerate(boundaries)}
        near = sorted({index[key] + step for key in fixed for step in (-1, 1)})
        for position in near:
            if 0 <= position < len(boundaries) and boundaries[position]['id'] not in fixed and boundaries[position]['id'] not in positions:
                to_cut(boundaries[position], BESIDE)
        spans = spans_for(positions)
        if spans is None:
            return None
    # A boundary handed back keeps its phrase whole when that still fits.
    for boundary in released:
        if not boundary['speech']:
            continue
        choice = direction.default_choice(boundary, ctx, timeline, bible or {})
        if choice == 'keep_cut':
            continue
        option = boundary['options'][choice]
        trial = {**positions, boundary['id']: (option['t'], option['time'])}
        better = spans_for(trial)
        if better is not None:
            positions, spans = trial, better
            decisions[boundary['after_cut_id']] = {'after_cut_id': boundary['after_cut_id'], 'choice': choice, 'reason': RELEASED, 'by': 'app'}
    return {**plan, 'shots': spans, 'boundaries': normal_decisions(list(decisions.values()), timeline)}


def set_points(conn, sequence, points, origin='manual_edit', provenance=None):
    """Save the shot boundaries the user placed on the Dialogue timeline and put the plan's shots on them; returns the saved points.

    ``points`` are ``[{after_cut_id, time}]`` in episode seconds; an empty list hands every boundary back
    to the director. A placed boundary is final for the director and only the hard rules apply
    (``sequence_timeline.check_points``): it may sit anywhere, even inside speech. With a usable plan, a plan
    revision that puts its shots on the points is saved in the same transaction; the moved shots become
    stale and are written again with Write shots.
    """
    current = sequence_view.state(conn, sequence)
    if current['cuts'] is None:
        raise EditError(409, 'The first or last cut of this sequence was archived; create the sequence again')
    if any(job['step'] in ('bible', 'plan') for job in store.pending_jobs(conn, sequence['id'])):
        raise EditError(409, 'Wait for the bible and the shot plan to finish')
    normal = normal_points(points, current['timeline'])
    timeline = store.timeline_for(conn, sequence, current['cuts'], normal)
    problems = sequence_timeline.check_points(timeline)
    if problems:
        raise EditError(422, ' '.join(problem[0].upper() + problem[1:] + ('' if problem.endswith('.') else '.') for problem in problems))
    body = json.dumps(normal)
    if body != (sequence.get('shot_points_json') or '[]'):
        conn.execute('UPDATE sequences SET shot_points_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (body, sequence['id']))
    row, plan = current['plan_row'], current['plan']
    if not row or not plan or not plan.get('shots') or current['changed']:
        return normal
    content = points_plan(plan, set(sequence_timeline.placed(current['timeline'])), timeline, current['ctx'], current['bible'])
    if content is None or (content['shots'] == plan['shots'] and content['boundaries'] == (plan.get('boundaries') or [])):
        return normal
    revise(conn, {**sequence, 'shot_points_json': body}, row, content, origin, provenance, {**current, 'timeline': timeline})
    return normal


def normal_plan_voices(voices, timeline):
    """Check the user's voice ranges of a plan: ``[{speaker, from, to, delivery}]`` in episode seconds inside the sequence."""
    if not isinstance(voices, list):
        raise ValueError('"voices" must be a list of {speaker, from, to, delivery}')
    start, end = timeline['origin'], timeline['origin'] + timeline['duration']
    normal, errors = [], []
    for number, item in enumerate(voices, 1):
        if not isinstance(item, dict) or not isinstance(item.get('speaker'), str) or not item['speaker'].strip():
            errors.append(f'voice range {number} needs a speaker')
            continue
        if not all(isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool) for key in ('from', 'to')) or not start - 0.01 <= item['from'] < item['to'] <= end + 0.01:
            errors.append(f'voice range {number} needs from < to inside the sequence ({start:g} to {end:g} s, episode time)')
            continue
        delivery = sequence_prompt.delivery_of(item.get('delivery'))
        if delivery is None:
            errors.append(f"voice range {number}: delivery must be one of {', '.join(sequence_prompt.DELIVERIES)}")
            continue
        normal.append({'speaker': item['speaker'].strip(), 'from': round(max(start, float(item['from'])), 6), 'to': round(min(end, float(item['to'])), 6), 'delivery': delivery})
    if errors:
        raise ValueError('; '.join(errors))
    return sorted(normal, key=lambda item: (item['from'], item['speaker']))


def set_voices(conn, sequence, shot_key, speaker, segments):
    """Decide how one speaker is heard in one shot, and rewrite that shot around it.

    ``segments`` are ``[{from, delivery}]`` in seconds from the shot's start, or None to let the director
    decide again. The ranges are saved on the plan (episode time) and a rewrite of the shot is queued in
    the same transaction; a rewrite that is still queued reads them when it starts, one already running
    is cancelled because it started from the old voices. Returns the new plan revision id.
    """
    current = sequence_view.state(conn, sequence)
    row, plan = current['plan_row'], current['plan']
    if current['cuts'] is None or current['changed']:
        raise EditError(409, 'The cuts of this sequence changed; direct the sequence again')
    if not current['bible'] or not row or not current['shots']:
        raise EditError(409, 'Direct the sequence first: it needs a bible and a shot plan')
    shot = next((item for item in current['shots'] if item['key'] == shot_key), None)
    if shot is None:
        raise EditError(404, 'That shot is not part of the current plan')
    heard = sequence_prompt.speakers(evidence.shot_dialogue(shot, current['bible']))
    if speaker not in heard:
        raise EditError(422, f"{speaker} is not heard in this shot" + (f"; the speakers are {', '.join(heard)}" if heard else '; nobody speaks in it'))
    normal = []
    for item in segments or []:
        delivery = sequence_prompt.delivery_of(item.get('delivery')) if isinstance(item, dict) else None
        start = item.get('from') if isinstance(item, dict) else None
        if delivery is None or isinstance(start, bool) or not isinstance(start, (int, float)) or not 0 <= start < shot['duration']:
            raise EditError(422, f"Each range needs a delivery ({', '.join(sequence_prompt.DELIVERIES)}) and a start inside the shot (0 to {shot['duration']:g} s)")
        normal.append({'from': float(start), 'delivery': delivery})
    if len({round(item['from'], 2) for item in normal}) != len(normal):
        raise EditError(422, 'Two ranges start at the same second')
    voices = sequence_timeline.replace_voices(plan.get('voices') or [], speaker, shot['start_time'], shot['end_time'], normal)
    if voices == (plan.get('voices') or []):
        return row['id']
    plan_id = revise(conn, sequence, row, {**plan, 'voices': voices}, current=current)
    for job in store.pending_jobs(conn, sequence['id']):
        if job['step'] == 'shot' and job['shot_key'] == shot_key and job['status'] == 'running':
            worker.cancel_jobs(conn, [job], 'Cancelled: the voices of this shot changed')
    if not any(job['step'] == 'shot' and job['shot_key'] == shot_key for job in store.pending_jobs(conn, sequence['id'])):
        queue(conn, sequence, 'shots', shot_keys=[shot_key], trigger='voices')
    return plan_id


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
        if row['type'] == 'compilation':
            roster = {person['name']: person['id'] for person in current['ctx']['characters']}
            normal = sequence_compile.validate(content, current['timeline']['duration'], roster)
            new_id = store.save(conn, sequence['id'], 'compilation', normal, origin=origin, settings=settings, expected_current=row['id'])
            # A new revision has not been reviewed yet: Direction waits for it to be confirmed again.
            conn.execute('UPDATE sequences SET reviewed_compilation_id=NULL WHERE id=?', (sequence['id'],))
            return new_id
        if row['type'] == 'bible':
            normal = direction.validate_bible(content, current['ctx'], current['timeline'])
            return store.save(conn, sequence['id'], 'bible', normal, origin=origin, settings=settings, expected_current=row['id'])
        if row['type'] == 'plan':
            spans = normal_spans(content.get('shots'), current['timeline'])
            notes = content.get('notes') if isinstance(content.get('notes'), str) else ''
            earlier = store.loads(row['content'], {})
            # The user's voice ranges and the boundary decisions stay with the plan unless this revision sets them.
            voices = normal_plan_voices(content['voices'], current['timeline']) if 'voices' in content else (earlier.get('voices') or [])
            decisions = normal_decisions(content['boundaries'] if 'boundaries' in content else earlier.get('boundaries'), current['timeline'])
            saved = {'shots': spans, 'boundaries': decisions, 'notes': notes, **({'voices': voices} if voices else {})}
            return store.save(conn, sequence['id'], 'plan', saved, origin=origin, settings=settings, expected_current=row['id'])
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
