"""Inference-worker side of sequence direction (pipeline ``sequence``, one job per step).

``payload.step`` is ``bible``, ``plan``, ``shot`` or ``review``. A job reads the current cuts, results
and transcript when it starts (never a snapshot taken at queue time), asks the text model, and saves
one revision under its lease. Follow-up jobs are inserted in the same transaction as the result that
makes them possible: the bible queues the plan, and the last shot of a batch queues the review. A
job whose result was already committed before a crash only marks itself done.
"""
import json
import logging
import os
import time

from . import db, local_text_runtime, worker
from . import sequence_direction as direction, sequence_evidence as evidence, sequence_prompt, sequence_store as store, sequence_timeline

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
        raise RuntimeError('These cuts need a first-frame and a video analysis first: ' + ', '.join(map(str, missing)))
    return {'cuts': cuts, 'changed': changed, 'ctx': ctx, 'timeline': store.timeline_for(conn, sequence, cuts)}


def content_of(row):
    return store.loads(row['content'], {}) if row else None


def base_of(payload):
    """The revision a job was queued against; a job queued without one accepts whatever is current."""
    return payload['base_id'] if 'base_id' in payload else ...


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

def run_bible(job, owner, payload, sequence):
    with db.connect() as conn:
        state = load_state(conn, sequence, adopt_cuts=True)
    ctx, timeline = state['ctx'], state['timeline']
    worker.stage(job['id'], owner, job['episode_id'], 'Writing the sequence bible')
    output, model, settings, user_input = ask(job, owner, payload, 'bible', lambda level: evidence.bible_input(ctx, sequence, timeline, level),
                                              direction.bible_format(), lambda answer: direction.validate_bible(answer, ctx, timeline), 'Sequence bible')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job['id'], owner)
        if state['changed']:
            store.resnapshot(conn, sequence, state['cuts'])
        store.save(conn, sequence['id'], 'bible', output, model_id=model, job_id=job['id'], prompt=direction.BIBLE_SYSTEM, settings=settings, context=user_input,
                   expected_current=base_of(payload))
        if payload.get('then_plan'):
            plan = store.current(conn, sequence['id'], 'plan')
            store.insert_job(conn, sequence, {'step': 'plan', 'config': payload.get('config') or {}, 'base_id': plan['id'] if plan else None})
        finish(conn, job)


def run_plan(job, owner, payload, sequence):
    with db.connect() as conn:
        state = load_state(conn, sequence, adopt_cuts=True)
        bible = content_of(store.current(conn, sequence['id'], 'bible'))
    if not bible:
        raise RuntimeError('Write the sequence bible first')
    ctx, timeline = state['ctx'], state['timeline']
    worker.stage(job['id'], owner, job['episode_id'], 'Planning the shots')
    seen = {}

    def validate(answer):
        seen['answer'] = answer
        return direction.validate_plan(answer, timeline, bible)

    def complete():
        # The model could not produce a valid partition; keep its usable boundaries and add the missing ones.
        answer = seen.get('answer') if isinstance(seen.get('answer'), dict) else {}
        shots = sequence_timeline.auto_fix(answer.get('shots'), timeline)
        for shot in shots:
            shot['scene'] = shot.get('scene') or (evidence.scene_for(bible, shot['frame_cut']) or {}).get('id')
        return {'shots': shots, 'notes': answer.get('notes') if isinstance(answer.get('notes'), str) else ''}

    output, model, settings, user_input = ask(job, owner, payload, 'plan', lambda level: evidence.plan_input(ctx, sequence, timeline, bible, level),
                                              direction.plan_format(timeline, bible), validate, 'Shot plan', fallback=complete)
    content = {'shots': sequence_timeline.to_spans(output['shots'], timeline), 'notes': output.get('notes') or ''}
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job['id'], owner)
        if state['changed']:
            store.resnapshot(conn, sequence, state['cuts'])
        store.save(conn, sequence['id'], 'plan', content, model_id=model, job_id=job['id'], prompt=direction.PLAN_SYSTEM, settings=settings, context=user_input,
                   expected_current=base_of(payload))
        finish(conn, job)


def planned_shots(conn, sequence, timeline):
    plan = content_of(store.current(conn, sequence['id'], 'plan'))
    if not plan or not plan.get('shots'):
        raise RuntimeError('Plan the shots first')
    problems = sequence_timeline.check_spans(plan['shots'], timeline)
    if problems:
        raise RuntimeError('The shot plan no longer fits the cuts (' + '; '.join(problems) + '); direct the sequence again')
    return sequence_timeline.expand(plan['shots'], timeline)


def run_shot(job, owner, payload, sequence):
    with db.connect() as conn:
        state = load_state(conn, sequence)
        bible = content_of(store.current(conn, sequence['id'], 'bible'))
        if not bible:
            raise RuntimeError('Write the sequence bible first')
        shots = planned_shots(conn, sequence, state['timeline'])
        index = next((position for position, shot in enumerate(shots) if shot['key'] == payload.get('shot_key')), None)
        if index is None:
            raise RuntimeError('This shot is no longer part of the plan')
        previous = content_of(store.current(conn, sequence['id'], 'shot', shots[index - 1]['key'])) if index else None
        ctx, timeline, shot = state['ctx'], state['timeline'], shots[index]
        # The subtitle lines are read here because the input is built several times while fitting the context.
        inputs = {level: evidence.shot_input(conn, ctx, sequence, timeline, bible, shots, index, payload.get('instruction') or '', previous, level) for level in range(evidence.LEVELS)}
    roster, scenario, dialogue = evidence.shot_parts(ctx, bible, shot)
    worker.stage(job['id'], owner, job['episode_id'], f"Writing shot {shot['index']} of {len(shots)}")
    output, model, settings, user_input = ask(job, owner, payload, 'shot', inputs.__getitem__, direction.shot_format(shot, roster, dialogue),
                                              lambda answer: direction.validate_shot(answer, shot, bible, roster, scenario, dialogue), f"Shot {shot['index']}")
    prompt, verdict, derived = sequence_prompt.build(shot, output, bible, roster, scenario, dialogue)
    settings.update(check=verdict, derived=derived, hashes=evidence.shot_hashes(ctx, bible, shot), instruction=payload.get('instruction') or None)
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job['id'], owner)
        store.save(conn, sequence['id'], 'shot', output, shot_key=shot['key'], rendered=prompt, model_id=model, job_id=job['id'], prompt=direction.SHOT_SYSTEM,
                   settings=settings, context=user_input, expected_current=base_of(payload))
        queue_review_after(conn, job, payload, sequence)
        finish(conn, job)


def queue_review_after(conn, job, payload, sequence):
    """Queue the review when this was the last shot of its batch and none of its shots failed."""
    batch = payload.get('batch_id')
    if not batch or not payload.get('then_review'):
        return
    others = [row['status'] for row in conn.execute("SELECT status FROM jobs WHERE pipeline='sequence' AND id!=? AND json_extract(payload_json,'$.batch_id')=?", (job['id'], batch))]
    if any(status in ('queued', 'running', 'failed') for status in others):
        return
    if conn.execute("SELECT 1 FROM jobs WHERE pipeline='sequence' AND status IN ('queued','running') AND json_extract(payload_json,'$.sequence_id')=? AND json_extract(payload_json,'$.step')='review'", (sequence['id'],)).fetchone():
        return
    review = store.current(conn, sequence['id'], 'review')
    store.insert_job(conn, sequence, {'step': 'review', 'config': payload.get('config') or {}, 'base_id': review['id'] if review else None})


def run_review(job, owner, payload, sequence):
    with db.connect() as conn:
        state = load_state(conn, sequence)
        bible = content_of(store.current(conn, sequence['id'], 'bible'))
        if not bible:
            raise RuntimeError('Write the sequence bible first')
        shots = planned_shots(conn, sequence, state['timeline'])
        written = {key: store.loads(row['content'], {}) for key, row in store.current_shots(conn, sequence['id']).items()}
    if not any(shot['key'] in written for shot in shots):
        raise RuntimeError('Write the shot prompts before reviewing them')
    worker.stage(job['id'], owner, job['episode_id'], 'Reviewing continuity')
    output, model, settings, user_input = ask(job, owner, payload, 'review', lambda level: evidence.review_input(bible, shots, written, level),
                                              direction.review_format(len(shots)), lambda answer: direction.validate_review(answer, len(shots)), 'Sequence review')
    for issue in output['issues']:
        issue['shot_key'] = shots[issue['shot'] - 1]['key']  # shot numbers move when the plan is edited; the key does not
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        worker.require_lease(conn, job['id'], owner)
        store.save(conn, sequence['id'], 'review', output, model_id=model, job_id=job['id'], prompt=direction.REVIEW_SYSTEM, settings=settings, context=user_input,
                   expected_current=base_of(payload))
        finish(conn, job)


STEPS = {'bible': run_bible, 'plan': run_plan, 'shot': run_shot, 'review': run_review}


def process(job_id, owner):
    from .inference_worker import SummaryCancelled, fail_job
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = {}
    try:
        payload = json.loads(job['payload_json'])
        with db.connect() as conn:
            sequence = store.get(conn, payload.get('sequence_id'))
            if not sequence:
                raise RuntimeError('The sequence was deleted')
            if conn.execute('SELECT 1 FROM sequence_results WHERE job_id=?', (job_id,)).fetchone():
                # A crash after the result was committed and before the job was marked done.
                worker.require_lease(conn, job_id, owner)
                finish(conn, job)
                return
        if payload.get('step') not in STEPS:
            raise RuntimeError(f"Unknown sequence step: {payload.get('step')}")
        STEPS[payload['step']](job, owner, payload, sequence)
    except SummaryCancelled:
        if (payload.get('config') or {}).get('text_provider') == 'local' and local_text_runtime.runtime:
            local_text_runtime.runtime.stop('Cancelled')
        log.info('Sequence job %s was cancelled', job_id)
    except worker.LeaseLost:
        log.info('Sequence job %s lost its lease', job_id)
    except Exception as exc:
        fail_job(job_id, owner, job['episode_id'], exc)
        log.exception('Sequence direction failed for job %s', job_id)
