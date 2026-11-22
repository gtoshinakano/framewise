"""Inference-worker side of sequence direction (pipeline ``sequence``, one job per step).

``payload.step`` is ``bible``, ``plan``, ``shot`` or ``review``. A job reads the current cuts, results
and transcript when it starts (never a snapshot taken at queue time), asks the text model, and saves
one revision under its lease. Follow-up jobs are inserted in the same transaction as the result that
makes them possible: the bible queues the plan. Continuity review runs on request. A
job whose result was already committed before a crash only marks itself done.
"""
import json
import logging
import os
import time

from . import db, job_inputs, local_text_runtime, outfit_library, worker
from . import sequence_compile, sequence_direction as direction, sequence_edit, sequence_evidence as evidence, sequence_prompt, sequence_store as store, sequence_timeline, sequence_view, subtitle_store

log = logging.getLogger(__name__)


def finish(conn, job, stage='Complete'):
    conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (stage, job['id']))
    worker.sync_episode_status(conn, job['episode_id'])


def load_state(conn, sequence, adopt_cuts=False):
    """Current cuts, evidence and timeline of a sequence; steps after the plan refuse cuts that changed under it."""
    cuts = store.current_cuts(conn, sequence)
    if cuts is None:
        raise RuntimeError('The first or last cut of this sequence was archived; create the sequence again')
    if cuts[-1]['end_time'] - cuts[0]['start_time'] > sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001:
        raise RuntimeError('The cuts of this sequence now last more than three minutes; create a shorter sequence')
    changed = store.cuts_changed(sequence, cuts)
    if changed and not adopt_cuts:
        raise RuntimeError('The cuts of this sequence changed; direct the sequence again before writing shots')
    ctx = evidence.load(conn, sequence, cuts)
    missing = evidence.missing_cuts(ctx, cuts)
    if missing:
        raise RuntimeError('Complete first-frame analysis, keyframes and performance beats for cuts: ' + ', '.join(map(str, missing)))
    return {'cuts': cuts, 'changed': changed, 'ctx': ctx, 'timeline': store.timeline_for(conn, sequence, cuts)}


def content_of(row):
    return store.loads(row['content'], {}) if row else None


def output_type(step):
    return 'compilation' if step == 'compile' else step


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    sequence = store.get(conn, payload.get('sequence_id'))
    if not sequence:
        return {'skip': 'Sequence removed'}
    sequence.pop('updated_at', None)
    step = payload['step']
    if store.current_cuts(conn, sequence) is None:
        return {'skip': 'Sequence boundaries were archived'}
    try:
        if step == 'compile':
            data = compile_inputs(conn, sequence, payload)
        elif step == 'shots':
            current = sequence_view.state(conn, sequence)
            if current['changed'] or not current['shots'] or not current['bible']:
                return {'missing': 'Direct the sequence first to establish a current shot plan'}
            data = {'views': sequence_view.shot_views(current)}
        else:
            sequence_edit.require_reviewed(conn, sequence, store.current_cuts(conn, sequence))
            data = load_state(conn, sequence, adopt_cuts=step in ('bible', 'plan'))
            if step == 'bible':
                data['reviewed_compilation'] = store.reviewed_compilation(conn, sequence)
            if step in ('plan', 'shot', 'review'):
                bible = store.current(conn, sequence['id'], 'bible')
                data['bible_id'], data['bible'] = (bible['id'], content_of(bible)) if bible else (None, None)
            if step in ('shot', 'review'):
                plan = store.current(conn, sequence['id'], 'plan')
                data['plan_id'] = plan['id'] if plan else None
                data['shots'] = planned_shots(conn, sequence, data['timeline'])
            if step == 'shot':
                data.update(shot_inputs(conn, sequence, data, payload))
            if step == 'review':
                written = store.current_shots(conn, sequence['id'])
                data['shot_result_ids'] = {key: row['id'] for key, row in written.items()}
                data['written'] = {key: content_of(row) for key, row in written.items()}
                data['asked'] = {shot['key']: store.shot_instructions(conn, sequence['id'], shot['key'])
                                 for shot in data['shots'] if shot['key'] in written}
    except sequence_edit.EditError as exc:
        return {'missing': exc.detail}
    except RuntimeError as exc:
        if step == 'shot' and str(exc) == 'This shot is no longer part of the plan':
            return {'skip': str(exc)}
        return {'missing': str(exc)}
    base = store.current(conn, sequence['id'], output_type(step), payload.get('shot_key'))
    protected = db.one("SELECT json_extract(input_json,'$.keep_output') AS enabled FROM jobs WHERE id=?", (job['id'],), conn=conn)
    return {'sequence': sequence, 'state': data, 'base': base, 'keep_output': bool(protected and protected['enabled'])}


def inputs(conn, job):
    values = job.get('gathered') or gather(conn, job)
    return job_inputs.fingerprint({**values, 'base': (values.get('base') or {}).get('id')})


def invalidate(conn, job):
    payload = json.loads(job['payload_json'])
    if payload.get('step') in ('compile', 'bible', 'plan', 'shot'):
        store.invalidate_review(conn, payload.get('sequence_id'))


def result_settings(settings, snapshot):
    state = snapshot.data['state']
    revisions = {name: state[name] for name in ('bible_id', 'plan_id', 'shot_result_ids') if name in state}
    base = snapshot.data['base']
    return {**settings, 'input_fingerprint': snapshot.stamp, 'input_revisions': {
        **revisions, 'previous_result_id': base['id'] if base else None,
        'reviewed_compilation_id': snapshot.data['sequence'].get('reviewed_compilation_id'),
    }}


def ask(job, owner, payload, step, build, response_format, validator, label, fallback=None):
    """One director call. Returns ``(answer, model, settings, input)``.

    ``fallback`` is used when every attempt was rejected by the validator; it returns the answer to
    save instead, so a step that the app can complete by calculation never fails on form.
    """
    from .inference_worker import chat_json, check_summary_lease
    config = payload.get('config') or {}
    provider = config.get('text_provider', 'openrouter')
    spec = direction.STEPS[step]
    user_input, level = evidence.fit(build, spec['system'], spec['max_tokens'], provider)
    usage, started, repaired = {}, time.monotonic(), None
    try:
        output, model = chat_json(config.get('openrouter_model', 'openrouter/free'), spec['system'], json.dumps(user_input, ensure_ascii=False), [], max_tokens=spec['max_tokens'],
                                  progress=lambda attempt, total: worker.stage(job['id'], owner, job['episode_id'], f'{label} · attempt {attempt}/{total}'),
                                  cancel_check=lambda: check_summary_lease(job['id'], owner), provider=provider, thinking=config.get('local_thinking', False), usage_out=usage,
                                  response_format=response_format, validator=validator, temperature=spec['temperature'])
    except worker.LeaseLost:
        raise
    except RuntimeError as exc:
        if fallback is None or 'returned an invalid analysis' not in str(exc):
            raise
        output, repaired = fallback(), str(exc)[-600:]
        model = local_text_runtime.model_id_for(usage.get('model_variant')) if provider == 'local' else config.get('openrouter_model')
    local = provider == 'local'
    settings = {'provider': provider, 'model_revision': local_text_runtime.revision_for(usage.get('model_variant')) if local else None,
                'thinking': config.get('local_thinking', False) if local else None,
                'thinking_budget_tokens': int(os.environ.get('QWEN36_THINKING_BUDGET', '2048')) if local and config.get('local_thinking') else None,
                'summary_generation_seconds': time.monotonic() - started, 'usage': usage, 'detail_level': level}
    if repaired:
        settings['completed_by_app'] = repaired
    return output, model, settings, user_input


# ---------------------------------------------------------------------- steps

WINDOW_LINES = 40
WINDOW_SECONDS = 45.0


def windows_of(lines):
    """Consecutive runs of lines of at most ``WINDOW_LINES`` lines and ``WINDOW_SECONDS`` seconds, one model call each."""
    runs, run = [], []
    for line in lines:
        if run and (len(run) >= WINDOW_LINES or line['t1'] - run[0]['t0'] > WINDOW_SECONDS):
            runs.append(run)
            run = []
        run.append(line)
    if run:
        runs.append(run)
    return [{'index': index + 1, 'of': len(runs), 'lines': items, 'next': runs[index + 1][:2] if index + 1 < len(runs) else []}
            for index, items in enumerate(runs)]


def compile_windows(job, owner, payload, ctx, timeline, source_lines, subtitles, previous=(), decided=()):
    """Ask Qwen for the dialogue of ``source_lines``, window by window.

    ``previous`` are compiled lines before them and ``decided`` the voices already settled; both are context
    the model continues from. Returns the compiled lines, the voices it newly decided, its open questions and
    the provenance of the last call.
    """
    roster = {person['name']: person['id'] for person in ctx['characters']}
    names = list(roster)
    lines, found, questions = [], [], []
    out = {'model': None, 'settings': {}, 'input': {}, 'seconds': 0.0, 'windows': 0}
    for window in windows_of(source_lines):
        ids = [line['id'] for line in window['lines']]
        nearby = evidence.window_subtitles(window['lines'], subtitles)
        by_subtitle = {item['id']: item for item in nearby}
        sub_ids = list(by_subtitle)
        label = f"Aligning dialogue · part {window['index']}/{window['of']}"
        worker.stage(job['id'], owner, job['episode_id'], label)
        answer, out['model'], out['settings'], out['input'] = ask(
            job, owner, payload, 'compile',
            lambda level, window=window: evidence.compile_input(ctx, timeline, window, subtitles, [*previous, *lines], [*decided, *found], level),
            direction.compile_format(ids, sub_ids, names), lambda output: direction.validate_compile(output, ids, set(sub_ids), names, window['lines']),
            label, fallback=lambda window=window: sequence_compile.deterministic_groups(window['lines']))
        out['seconds'] += out['settings'].get('summary_generation_seconds') or 0.0
        out['windows'] += 1
        lines.extend(sequence_compile.assemble(answer, window['lines'], by_subtitle, roster))
        known = {voice['speaker_label'] for voice in [*decided, *found]}
        found.extend(voice for voice in answer['voices'] if voice['speaker_label'] not in known)
        questions.extend(answer['open_questions'])
    return {**out, 'lines': lines, 'decided': found, 'questions': questions}


def compile_timing(job, owner, payload, sequence, cuts, ctx, content, captured):
    """Select the shot starts on the finished dialogue so the reviewer can preview them immediately."""
    timeline = store.build_timeline(None, sequence, cuts, content, envelope=captured['audio'], events=captured['event_starts'])
    boundaries = direction.open_boundaries(timeline)
    if not boundaries:
        return {}, {}
    settings, user_input = {}, {}
    if any(len(boundary['options']) > 1 for boundary in boundaries):
        seen = None

        def validate(answer):
            nonlocal seen
            seen = answer
            return direction.validate_decisions(answer, timeline, timing=True)

        worker.stage(job['id'], owner, job['episode_id'], 'Aligning shot starts to phrases')
        answer, model, settings, user_input = ask(
            job, owner, payload, 'timing', lambda level: evidence.timing_input(ctx, timeline, level),
            direction.timing_format(timeline), validate, 'Aligning shot starts to phrases',
            fallback=lambda: direction.complete_plan(seen, timeline, {}, ctx, timing=True))
        settings = {**settings, 'model_id': model}
    else:
        answer = direction.complete_plan(None, timeline, {}, ctx, timing=True)
    by_id = {boundary['id']: boundary for boundary in boundaries}
    content['timing'] = [{'after_cut_id': by_id[item['id']]['after_cut_id'], **{key: item[key] for key in ('choice', 'reason', 'by')}}
                         for item in answer['decisions']]
    return settings, user_input


def compile_inputs(conn, sequence, payload):
    cuts = store.current_cuts(conn, sequence)
    if cuts is None:
        raise RuntimeError('The first or last cut of this sequence was archived; create the sequence again')
    if cuts[-1]['end_time'] - cuts[0]['start_time'] > sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001:
        raise RuntimeError('The cuts of this sequence now last more than three minutes; create a shorter sequence')
    changed = store.cuts_changed(sequence, cuts)
    ctx = evidence.load(conn, sequence, cuts)
    marker = store.extension(sequence) or {}
    extend_from = (marker.get('compiled_to') or marker.get('from_time')) if payload.get('extend') else payload.get('extend_from')
    timeline = store.raw_timeline_for(conn, sequence, cuts, utterances=extend_from is None)
    source, fragments, subtitle_source = store.source_fingerprint(conn, sequence, cuts)
    run = subtitle_store.current_run(sequence['episode_id'], conn)
    if run and not ctx['subtitle_language']:
        ctx['subtitle_language'] = run['language']
    base = store.current(conn, sequence['id'], 'compilation')
    was_current = False
    if extend_from is not None and base is not None and cuts[0]['start_time'] < extend_from < cuts[-1]['end_time']:
        # The sources were current before the cut was added only if they matched the range as it was then.
        earlier = [cut for cut in cuts if cut['end_time'] <= extend_from + 0.001]
        saved = conn.execute('SELECT source_hash FROM sequence_compilation_sources WHERE sequence_id=?', (sequence['id'],)).fetchone()
        was_current = not saved or bool(earlier) and saved['source_hash'] == store.source_fingerprint(conn, sequence, earlier)[0]
    else:
        extend_from = None
    return {'cuts': cuts, 'changed': changed, 'ctx': ctx, 'timeline': timeline, 'source': source,
            'fragments': fragments, 'subtitle_source': subtitle_source, 'subtitle_run': run, 'extend_from': extend_from,
            'was_current': was_current, 'event_starts': store.event_starts(conn, cuts), 'audio': store.audio_envelope(conn, sequence, cuts),
            'extension_done': bool(payload.get('extend') and (not marker or marker.get('compiled_to', 0) >= cuts[-1]['end_time']))}


def mark_compiled_extension(conn, sequence, end):
    marker = store.extension(sequence)
    if marker:
        conn.execute('UPDATE sequences SET extension_json=? WHERE id=?', (json.dumps({**marker, 'compiled_to': end}), sequence['id']))


def run_compile(job, owner, payload, sequence, snapshot):
    """Compile current speech, with an append-only path for an extension."""
    values = snapshot.data['state']
    cuts, changed, ctx, timeline = (values[key] for key in ('cuts', 'changed', 'ctx', 'timeline'))
    source, fragments, subtitle_source, run = (values[key] for key in ('source', 'fragments', 'subtitle_source', 'subtitle_run'))
    base, extend_from, was_current = snapshot.data['base'], values['extend_from'], values['was_current']
    if payload.get('extend'):
        marker = store.extension(sequence) or {}
        payload = {**payload, 'extend_cut': (marker.get('cuts') or [None])[-1]}
    if values['extension_done']:
        with snapshot.publish(invalidate=False) as conn:
            finish(conn, job, 'Complete · added dialogue already compiled')
        return
    origin, duration = timeline['origin'], timeline['duration']
    subtitles = [{'id': f'U{number}', 'real': row['id'], 't0': round(row['start_time'] - origin, 3), 't1': round(row['end_time'] - origin, 3), 'text': row['text']}
                 for number, row in enumerate(subtitle_source, 1)]
    roster = {person['name']: person['id'] for person in ctx['characters']}
    if extend_from is not None:
        return extend_compile(job, owner, payload, sequence, cuts, changed, ctx, timeline, subtitles, roster, base, extend_from - origin, was_current, snapshot,
                              (source, run, fragments, subtitle_source))
    found = compile_windows(job, owner, payload, ctx, timeline, timeline['lines'], subtitles)
    anchors = [edge for line in timeline['lines'] for part in line['parts'] for edge in (part['t0'], part['t1'])] + [edge for item in subtitles for edge in (item['t0'], item['t1'])]
    lines = sequence_compile.snap(found['lines'], anchors, duration)
    voices = sequence_compile.settle_voices(lines, found['decided'], timeline['lines'])
    content = sequence_compile.validate({'lines': lines, 'voices': voices, 'open_questions': found['questions'][:12]}, duration, roster)
    settings = {**found['settings'], 'summary_generation_seconds': found['seconds'], 'windows': found['windows'], 'subtitle_lines': len(subtitle_source), 'source_hash': source}
    timing, timing_input = compile_timing(job, owner, payload, sequence, cuts, ctx, content, values)
    settings['timing'] = timing
    with snapshot.publish() as conn:
        if changed:
            store.resnapshot(conn, sequence, cuts)
        result_id = store.save(conn, sequence['id'], 'compilation', content, model_id=found['model'], job_id=job['id'], prompt=direction.COMPILE_SYSTEM, settings=result_settings(settings, snapshot),
                               context={'dialogue': found['input'], 'timing': timing_input})
        store.record_sources(conn, sequence['id'], result_id, source, run['id'] if run else None, len(fragments), len(subtitle_source))
        if not content['lines']:
            conn.execute('UPDATE sequences SET reviewed_compilation_id=? WHERE id=?', (result_id, sequence['id']))  # nothing is spoken: there is nothing to review
        else:
            conn.execute('UPDATE sequences SET reviewed_compilation_id=NULL WHERE id=?', (sequence['id'],))
        mark_compiled_extension(conn, sequence, cuts[-1]['end_time'])
        finish(conn, job)
        sequence_edit.continue_extension(conn, store.get(conn, sequence['id']))


def extend_compile(job, owner, payload, sequence, cuts, changed, ctx, timeline, subtitles, roster, base, old_duration, was_current, snapshot, sources):
    """Compile the words after ``old_duration`` (the cut added at the end) and append them to the current dialogue.

    The model sees the user's last lines and voice decisions as context. The new lines are merged onto
    whatever dialogue is current when the job commits, so a save the user made meanwhile is kept. When the
    added cut has no speech, no revision is written and a confirmed dialogue stays confirmed, so the
    direction is redone at once (``sequence_edit.continue_extension``).
    """
    source, run, fragments, subtitle_source = sources
    duration = timeline['duration']
    tail, crossed = sequence_compile.tail_lines(timeline['lines'], old_duration)
    existing = store.loads(base['content'], {})
    found = {'lines': [], 'decided': [], 'questions': [], 'model': None, 'settings': {}, 'input': {}, 'seconds': 0.0, 'windows': 0}
    if tail:
        found = compile_windows(job, owner, payload, ctx, timeline, tail, subtitles, existing.get('lines') or [], existing.get('voices') or [])
    nearby = [item for item in subtitles if item['t1'] > old_duration]
    anchors = [edge for line in tail for part in line['parts'] for edge in (part['t0'], part['t1'])] + [edge for item in nearby for edge in (item['t0'], item['t1'])]
    added = sequence_compile.snap(found['lines'], anchors, duration)
    voices = sequence_compile.settle_voices(added, found['decided'], tail)
    settings = {**found['settings'], 'summary_generation_seconds': found['seconds'], 'windows': found['windows'], 'subtitle_lines': len(subtitle_source), 'source_hash': source,
                'extended_with_cut': payload.get('extend_cut'), 'extended_from': round(old_duration, 3)}
    timing_content = sequence_compile.validate(sequence_compile.append_lines(existing, added, voices, found['questions'], crossed), duration, roster)
    timing, timing_input = compile_timing(job, owner, payload, sequence, cuts, ctx, timing_content, snapshot.data['state']) if added else ({}, {})
    settings['timing'] = timing
    with snapshot.publish() as conn:
        if changed:
            store.resnapshot(conn, sequence, cuts)
        latest = base
        if added:
            content = sequence_compile.validate(sequence_compile.append_lines(store.loads(latest['content'], {}), added, voices, found['questions'], crossed), duration, roster)
            if latest['id'] == base['id'] and timing_content.get('timing'):
                content['timing'] = timing_content['timing']
            result_id = store.save(conn, sequence['id'], 'compilation', content, model_id=found['model'], job_id=job['id'], prompt=direction.COMPILE_SYSTEM, settings=result_settings(settings, snapshot),
                                   context={'dialogue': found['input'], 'timing': timing_input})
            # The added lines have not been reviewed yet.
            conn.execute('UPDATE sequences SET reviewed_compilation_id=NULL WHERE id=?', (sequence['id'],))
        else:
            result_id = latest['id']
        if was_current:
            store.record_sources(conn, sequence['id'], result_id, source, run['id'] if run else None, len(fragments), len(subtitle_source))
        if payload.get('extend'):
            mark_compiled_extension(conn, sequence, cuts[-1]['end_time'])
        finish(conn, job, 'Complete' if added else 'Complete · no speech in the added cut')
        sequence_edit.continue_extension(conn, store.get(conn, sequence['id']))


def run_bible(job, owner, payload, sequence, snapshot):
    state = snapshot.data['state']
    ctx, timeline, compiled = (state[key] for key in ('ctx', 'timeline', 'reviewed_compilation'))
    # A compilation saved as "the original transcript" was not reviewed by anyone: its voices are only hints.
    final_voices = compiled['voices'] if compiled is not None and compiled['origin'] != 'source' else None
    worker.stage(job['id'], owner, job['episode_id'], 'Writing the sequence bible')
    output, model, settings, user_input = ask(job, owner, payload, 'bible', lambda level: evidence.bible_input(ctx, sequence, timeline, level, final_voices),
                                              direction.bible_format(), lambda answer: direction.validate_bible(answer, ctx, timeline, require_outfits=True), 'Sequence bible')
    if final_voices is not None:
        # The user reviewed who owns each voice; the model only wrote the story around it.
        output = {**output, 'voices': [{**voice, 'source': 'reviewed'} for voice in final_voices]}
    with snapshot.publish() as conn:
        if state['changed']:
            store.resnapshot(conn, sequence, state['cuts'])
        output = outfit_library.adopt_for(conn, sequence, output)  # creates the outfits this bible proposes, in the transaction that saves it
        store.save(conn, sequence['id'], 'bible', output, model_id=model, job_id=job['id'], prompt=direction.BIBLE_SYSTEM, settings=result_settings(settings, snapshot), context=user_input)
        if payload.get('clear_extension'):
            conn.execute('UPDATE sequences SET extension_json=NULL WHERE id=?', (sequence['id'],))
        if payload.get('then_plan'):
            store.insert_job(conn, sequence, {'step': 'plan', 'config': payload.get('config') or {}, 'depends_on': [job['id']],
                                              **({'then_shots': True} if payload.get('then_shots') else {})})
        finish(conn, job)


def run_plan(job, owner, payload, sequence, snapshot):
    state = snapshot.data['state']
    bible = state['bible']
    if not bible:
        raise RuntimeError('Write the sequence bible first')
    ctx, timeline = state['ctx'], state['timeline']
    worker.stage(job['id'], owner, job['episode_id'], 'Planning the shots')
    seen = {}

    def validate(answer):
        seen['answer'] = answer
        return direction.validate_plan(answer, timeline, bible)

    output, model, settings, user_input = ask(job, owner, payload, 'plan', lambda level: evidence.plan_input(ctx, sequence, timeline, bible, level),
                                              direction.plan_format(timeline, bible), validate, 'Shot plan',
                                              # The model could not produce a valid answer: keep its usable choices and let the app decide the rest.
                                              fallback=lambda: direction.complete_plan(seen.get('answer'), timeline, bible, ctx))
    content = direction.plan_content(output, timeline, bible)
    if sequence.get('reviewed_compilation_id'):
        content['compilation_id'] = sequence['reviewed_compilation_id']
    with snapshot.publish() as conn:
        if state['changed']:
            store.resnapshot(conn, sequence, state['cuts'])
        # The user's voice ranges are in episode time, so they survive a new partition of the same audio.
        earlier = content_of(store.current(conn, sequence['id'], 'plan')) or {}
        if earlier.get('voices'):
            content['voices'] = earlier['voices']
        store.save(conn, sequence['id'], 'plan', content, model_id=model, job_id=job['id'], prompt=direction.PLAN_SYSTEM, settings=result_settings(settings, snapshot), context=user_input)
        finish(conn, job)
        if payload.get('then_shots'):
            # A cut was added to a sequence whose shots were written: they are written again on the new plan.
            try:
                sequence_edit.queue(conn, store.get(conn, sequence['id']), 'shots')
            except sequence_edit.EditError as exc:
                log.info('Sequence %s: shots not queued after the plan: %s', sequence['id'], exc.detail)


def planned_shots(conn, sequence, timeline):
    plan = content_of(store.current(conn, sequence['id'], 'plan'))
    if not plan or not plan.get('shots'):
        raise RuntimeError('Plan the shots first')
    problems = sequence_timeline.check_spans(plan['shots'], timeline)
    if problems:
        raise RuntimeError('The shot plan no longer fits the cuts (' + '; '.join(problems) + '); plan the shots again')
    return sequence_timeline.expand(plan['shots'], timeline, plan.get('voices'), plan.get('boundaries'))


def shot_inputs(conn, sequence, state, payload):
    bible = state['bible']
    if not bible:
        raise RuntimeError('Write the sequence bible first')
    shots = state['shots']
    index = next((position for position, shot in enumerate(shots) if shot['key'] == payload.get('shot_key')), None)
    if index is None:
        raise RuntimeError('This shot is no longer part of the plan')
    previous = content_of(store.current(conn, sequence['id'], 'shot', shots[index - 1]['key'])) if index else None
    ctx, timeline, shot = state['ctx'], state['timeline'], shots[index]
    instruction = (payload.get('instruction') or '').strip()
    from_review = bool(instruction and payload.get('instruction_source') == 'review')
    earlier = [text for text in store.shot_instructions(conn, sequence['id'], shot['key']) if text.casefold() != instruction.casefold() or from_review]
    planned = [evidence.shot_input(conn, ctx, sequence, timeline, bible, shots, index, instruction, previous, level,
                                  earlier=earlier, from_review=from_review) for level in range(evidence.LEVELS)]
    return {'shot': shot, 'instruction': instruction, 'from_review': from_review, 'earlier': earlier, 'requests': planned}


def run_shot(job, owner, payload, sequence, snapshot):
    state = snapshot.data['state']
    ctx, shot, bible, shots = (state[key] for key in ('ctx', 'shot', 'bible', 'shots'))
    instruction, from_review, earlier, requests = (state[key] for key in ('instruction', 'from_review', 'earlier', 'requests'))
    roster, scenario, dialogue = evidence.shot_parts(ctx, bible, shot)
    worker.stage(job['id'], owner, job['episode_id'], f"Writing shot {shot['index']} of {len(shots)}")
    output, model, settings, user_input = ask(job, owner, payload, 'shot', requests.__getitem__, direction.shot_format(shot, roster, dialogue),
                                              lambda answer: direction.validate_shot(answer, shot, bible, roster, scenario, dialogue, generated=True), f"Shot {shot['index']}")
    camera = evidence.still_camera(ctx['results'].get(shot['cut_id'], {}))
    if camera:
        output['still']['camera_look'] = camera
    prompt, verdict, derived = sequence_prompt.build(shot, output, bible, roster, scenario, dialogue)
    settings.update(check=verdict, derived=derived, hashes=evidence.shot_hashes(ctx, bible, shot), instruction=instruction or None,
                    instruction_source=('review' if from_review else 'user') if instruction else None, earlier_instructions=len(earlier) or None,
                    trigger=payload.get('trigger'))
    with snapshot.publish() as conn:
        store.save(conn, sequence['id'], 'shot', output, shot_key=shot['key'], rendered=prompt, model_id=model, job_id=job['id'], prompt=direction.SHOT_SYSTEM,
                   settings=result_settings(settings, snapshot), context=user_input)
        finish(conn, job)


def run_review(job, owner, payload, sequence, snapshot):
    state = snapshot.data['state']
    bible, shots, written, asked = (state[key] for key in ('bible', 'shots', 'written', 'asked'))
    if not bible:
        raise RuntimeError('Write the sequence bible first')
    if not any(shot['key'] in written for shot in shots):
        raise RuntimeError('Write the shot prompts before reviewing them')
    worker.stage(job['id'], owner, job['episode_id'], 'Reviewing continuity')
    output, model, settings, user_input = ask(job, owner, payload, 'review', lambda level: evidence.review_input(bible, shots, written, level, evidence.shot_states(state['ctx'], shots), asked),
                                              direction.review_format(len(shots)), lambda answer: direction.validate_review(answer, len(shots)), 'Sequence review')
    for issue in output['issues']:
        issue['shot_key'] = shots[issue['shot'] - 1]['key']  # shot numbers move when the plan is edited; the key does not
    with snapshot.publish() as conn:
        store.save(conn, sequence['id'], 'review', output, model_id=model, job_id=job['id'], prompt=direction.REVIEW_SYSTEM, settings=result_settings(settings, snapshot), context=user_input)
        finish(conn, job)


def run_shots(job, owner, payload, sequence, snapshot):
    with snapshot.publish(invalidate=False) as conn:
        children = sequence_edit.queue_shot_views(conn, sequence, payload, snapshot.data['state']['views'])
        if not children:
            finish(conn, job, 'Complete · every shot is current')
            return
        expanded = {**job, 'payload_json': json.dumps({**payload, 'expanded': True})}
        worker.defer_for_dependencies(conn, expanded, owner, children, 'Waiting for shot prompts')


STEPS = {'compile': run_compile, 'bible': run_bible, 'plan': run_plan, 'shots': run_shots, 'shot': run_shot, 'review': run_review}


def process(job_id, owner):
    from .inference_worker import SummaryCancelled, fail_job
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = {}
    try:
        payload = json.loads(job['payload_json'])
        with db.connect() as conn:
            conn.execute('BEGIN')
            if payload.get('expanded') or conn.execute('SELECT 1 FROM sequence_results WHERE job_id=?', (job_id,)).fetchone():
                # A crash after the result was committed and before the job was marked done.
                worker.require_lease(conn, job_id, owner)
                finish(conn, job)
                return
            snapshot = job_inputs.JobInputs(conn, job, owner)
        if payload.get('step') not in STEPS:
            raise RuntimeError(f"Unknown sequence step: {payload.get('step')}")
        values = snapshot.data
        if values.get('missing'):
            with snapshot.publish(invalidate=False):
                raise RuntimeError(values['missing'])
        current = values.get('base') or {}
        base = json.loads(job['input_json'] or '{}').get('output_base', current.get('id'))
        edited = current.get('id') != base and current.get('origin') != 'model'
        if values.get('skip') or ((edited or values['keep_output']) and not values['state'].get('extend_from')):
            with snapshot.publish(invalidate=False) as conn:
                finish(conn, job, f"Skipped · {values.get('skip') or 'sequence result edited by hand'}")
            return
        provenance = {'fingerprint': snapshot.stamp, 'output_base': base,
                      **result_settings({}, snapshot)['input_revisions']}

        def record_inputs(conn):
            conn.execute("UPDATE jobs SET input_json=json_patch(COALESCE(input_json,'{}'),?) WHERE id=?",
                         (json.dumps(provenance, ensure_ascii=False), job_id))

        with snapshot.publish(invalidate=False, on_changed=record_inputs) as conn:
            record_inputs(conn)
        STEPS[payload['step']](job, owner, payload, values['sequence'], snapshot)
    except SummaryCancelled:
        if (payload.get('config') or {}).get('text_provider') == 'local':
            local_text_runtime.cancel_local('Cancelled')
        log.info('Sequence job %s was cancelled', job_id)
    except worker.LeaseLost:
        log.info('Sequence job %s lost its lease', job_id)
    except Exception as exc:
        fail_job(job_id, owner, job['episode_id'], exc)
        log.exception('Sequence direction failed for job %s', job_id)
