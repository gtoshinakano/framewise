"""The autopilot answers the consistency review's questions (docs/autopilot/HARNESS.md, milestone M4).

A cut's consistency review asks up to five multiple-choice questions; the person's answer is one line per question in
"Your answers" (``cut_brief_instructions``, format of ``consistencyView.setAnswer``: ``- Topic (Character): option text``).
This module decides them in three tiers:

1. **Local decider** (job kind ``review_decisions``, a text job on the local model): for each question an option id or
   none, whether it is a fact or a creative choice, a confidence, and whether one of the user's standing rules (the
   Autopilot profile's per-topic rule) decides it. Grammar-constrained, one answer per question.
2. **Resolver** (``resolve``, pure): applies an answer when a standing rule decides it with high confidence, or when
   the decider agrees with the reviewer's recommended option with high confidence *and* that topic is trusted
   (``trust_table``: measured agreement with the user's own answers reaches ``TRUST``). Otherwise the topic's
   "when unsure" setting decides: take the recommended option, ask the person, or follow the evidence, which sends the
   cut to Claude (the second reviewer, ``reviewer='claude'``) while its budget lasts, else to the person.
3. **Escalations**: one per cut and reviewer (``review_question``, group ``cut:<id>`` for the person,
   ``cut:<id>:claude`` for Claude), answered through ``answer`` with option ids; Claude's ``unsure`` moves a question
   to the person.

Answers the user wrote are never touched: a question whose line is already in "Your answers" is kept. Every applied
answer is an ``agent_decisions`` row (kind ``review_answers``, target ``cut_brief_instructions``) with the content
before and after; undo restores it while the content is still what the agent wrote. A **shadow** job decides with
the user's answers hidden and writes nothing but its ``review_decisions`` row: that is how the decider is measured on
the user's own past answers (``autopilot/replay/review_answers.py``), and the measurement feeds ``trust_table``.

Provenance, not an approval status: nothing here gates or hides an answer.
"""
import functools
import json
import re
import uuid

from . import agent_log, analysis_context, db, job_inputs, prompt_context, review_answers, structured_results, worker

KIND = 'review_answers'        # the decision kind (autopilot_profile.DECISIONS) and agent_decisions.kind
PIPELINE = 'review_decisions'  # the job kind
ESCALATION = 'review_question'
PROMPT_VERSION = 1
CONFIDENCE = ('high', 'medium', 'low')
QUESTION_KINDS = ('fact', 'creative')
TRUST = 0.9          # D12: answers need 90% agreement with the user before the agent applies them on its own
MIN_SCORED = 10      # fewer scored answers than this in a topic say nothing about it
OPTION_IDS = 'abcd'


# --- answer lines (a port of frontend/src/consistencyView.ts: answerLabel, selection, setAnswer) -------------------

BULLET_LINE = re.compile(r'^\s*[-*•–]\s+')


def topic_label(topic):
    return topic[0].upper() + topic[1:] if topic else 'Other'


def answer_label(question):
    character = question.get('character') or ''
    return f"{topic_label(question.get('topic') or '')}{f' ({character})' if character else ''}"


def answer_line(question, answer):
    return f"- {answer_label(question)}: {' '.join(str(answer).split())}"


def _norm(value):
    return re.sub(r'[.。]+$', '', ' '.join(str(value).split()).lower())


def _locate(lines, question, others):
    bullets = []
    index = 0
    while index < len(lines):
        if not BULLET_LINE.match(lines[index]):
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip() and not BULLET_LINE.match(lines[end]):
            end += 1
        body = BULLET_LINE.sub('', ' '.join(lines[index:end]), count=1)
        colon = body.find(':')
        bullets.append({'start': index, 'end': end, 'label': _norm(body[:colon]) if colon > 0 else '',
                        'answer': body[colon + 1:].strip() if colon > 0 else body.strip()})
        index += 1
    mine = {_norm(option['text']) for option in question.get('options') or []}
    foreign = {_norm(option['text']) for other in others if other is not question for option in other.get('options') or []}
    for bullet in bullets:
        if _norm(bullet['answer']) in mine:
            return bullet
    label = _norm(answer_label(question))
    for bullet in bullets:
        if bullet['label'] == label and _norm(bullet['answer']) not in foreign:
            return bullet
    return None


def _lines(answers):
    return str(answers or '').replace('\r\n', '\n').replace('\r', '\n').split('\n')


def selection(answers, question, questions=None):
    """``{option, custom}`` of the line answering ``question``, or None when it is unanswered (``consistencyView.selection``)."""
    found = _locate(_lines(answers), question, questions or [question])
    if not found:
        return None
    for option in question.get('options') or []:
        if _norm(option['text']) == _norm(found['answer']):
            return {'option': option['id'], 'custom': ''}
    return {'option': None, 'custom': found['answer']}


def set_answer(answers, question, answer, questions=None):
    """The answers with this question's line replaced or added at the end (``consistencyView.setAnswer``)."""
    lines = _lines(answers)
    found = _locate(lines, question, questions or [question])
    line = answer_line(question, answer) if answer and str(answer).strip() else None
    if found:
        lines[found['start']:found['end']] = [line] if line else []
    elif line:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.append(line)
    text = '\n'.join(lines).lstrip('\n')
    return re.sub(r'\n{3,}', '\n\n', text)


def option_text(question, option_id):
    return next((option['text'] for option in question.get('options') or [] if option['id'] == option_id), None)


# --- the local decider ------------------------------------------------------------------------------------------------

SYSTEM = '''You decide the open review questions of ONE cut, for a pipeline that re-creates an anime episode shot by shot with AI (a still of the opening frame, then the dialogue audio, then a video that animates the still). A reviewer model compared the cut's evidence and asked multiple-choice questions where it was unsure. Your decisions are written into the production brief, so a wrong one spreads into the generated shot.

## INPUT

A JSON evidence package: "questions" (each with id, topic, optional character, question, options with ids, the reviewer's "recommended" option id or null, and "why"), "review" (what the reviewer found clear or confusing, and "contradicted_by_pictures": readings it saw contradicted in a picture of this cut), "current_cut" (first frame, keyframes and performance beats as text, the transcript and the burned-in subtitles), the neighbouring cuts' reviews, "existing_answers" (the user's own answers to this cut, when there are any), "standing_rules" (the user's rules for this project, per question topic) and "project_notes".

## HOW TO DECIDE

- standing_rules are the user's own decisions for every cut of this project. When a rule decides a question, follow it even against the recommended option and the pictures, set rule to true and say which rule. A rule decides a question only when it speaks about exactly that kind of question; never stretch it.
- existing_answers are the user's own decisions for this cut: never contradict them.
- Otherwise decide from the evidence with the source rule: the first frame decides how the cut starts (appearance, pose, expression, emotion); the keyframes decide what changes afterwards (motion, camera, mouth, emotion change, end state); the performance beats are the timed acting plan and each line's delivery, and where a beat contradicts the keyframes the keyframes win.
- "recommended" is the reviewer's reading, another model's. It saw pictures, but agreeing with it is not a second source: judge each option from the evidence yourself.
- Weak evidence, learned from the user's own answers: a mouth read as closed in a four-tile line strip or in one keyframe (small tiles miss lip movement; a visible speaker of their own line is usually lip-synced unless the face is turned away, hidden or too far); a transcript timing that starts long before or runs past the picture (lines often bleed across a hard cut); a picture at the very end of a cut that shows another shot (a missed hard cut, not a change inside this cut). Never give high confidence from weak evidence alone.

## FIELDS, one decision per question

- question: the question's id.
- kind: "fact" when the question asks what is visible, said or physically true in this cut (the evidence can settle it); "creative" when it asks how to play or stage something the evidence cannot show.
- choice: the id of the option the evidence (or a rule) supports, or null when no option is right or nothing settles it.
- confidence: "high" only when at least two independent sources support the choice and none contradicts it, or a standing rule decides it; "medium" when the evidence leans to it; "low" when it is a guess. Repetition by the same model is not independent.
- rule: true only when a standing rule decides it.
- reason: one short sentence naming the sources (or the rule) that decide it.

Never invent actions, characters, timings or dialogue. Return JSON only.'''


def decision_format(questions):
    """``response_format`` (a llama.cpp grammar): exactly one decision per question, ids and options from enums."""
    ids = [question['id'] for question in questions]
    item = {'type': 'object', 'additionalProperties': False, 'required': ['question', 'kind', 'choice', 'confidence', 'rule', 'reason'],
            'properties': {'question': {'enum': ids}, 'kind': {'enum': list(QUESTION_KINDS)},
                           'choice': {'type': ['string', 'null'], 'enum': [*OPTION_IDS, None]},
                           'confidence': {'enum': list(CONFIDENCE)}, 'rule': {'type': 'boolean'}, 'reason': {'type': 'string'}}}
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['decisions'],
              'properties': {'decisions': {'type': 'array', 'items': item, 'minItems': len(ids), 'maxItems': len(ids)}}}
    return {'type': 'json_schema', 'json_schema': {'name': 'review_decisions', 'strict': True, 'schema': schema}}


def validate(output, questions):
    """``{question id: decision}``. A ValueError goes back to the model with its answer."""
    items = output.get('decisions') if isinstance(output, dict) else None
    if not isinstance(items, list):
        raise ValueError('decisions must be an array with one decision per question')
    by_id = {question['id']: question for question in questions}
    found, problems = {}, []
    for item in items:
        if not isinstance(item, dict) or item.get('question') not in by_id:
            problems.append(f"{item.get('question') if isinstance(item, dict) else item!r} is not a question id")
            continue
        question = by_id[item['question']]
        if item['question'] in found:
            problems.append(f"{item['question']} is decided twice")
            continue
        choice = item.get('choice')
        if choice is not None and option_text(question, choice) is None:
            problems.append(f"{item['question']} has no option {choice}; choose one of {', '.join(option['id'] for option in question['options'])} or null")
            continue
        found[item['question']] = {'choice': choice, 'kind': item.get('kind') if item.get('kind') in QUESTION_KINDS else 'fact',
                                   'confidence': item.get('confidence') if item.get('confidence') in CONFIDENCE else 'low',
                                   'rule': bool(item.get('rule')), 'reason': ' '.join(str(item.get('reason') or '').split())[:300]}
    missing = [question_id for question_id in by_id if question_id not in found]
    if missing:
        problems.append(f"no decision for {', '.join(missing)}")
    if problems:
        raise ValueError('; '.join(problems))
    return found


def evidence_for(base, questions, policy):
    """The decider's input: the answers prompt's evidence, the questions first, plus the user's standing rules."""
    review = dict(base.get('review') or {})
    review.pop('questions', None)
    evidence = {'questions': questions, 'review': review, **{key: value for key, value in base.items() if key != 'review'}}
    topics = (policy or {}).get('topics') or {}
    rules = {topic: entry['rule'] for topic, entry in topics.items() if isinstance(entry, dict) and entry.get('rule')}
    wanted = {question.get('topic') for question in questions}
    evidence['standing_rules'] = {topic: rule for topic, rule in rules.items() if topic in wanted} or None
    if (policy or {}).get('notes'):
        evidence['project_notes'] = policy['notes']
    return evidence


def generate(config, evidence, questions, progress=None, cancel_check=None, usage_out=None):
    from .inference_worker import chat_json
    return chat_json(config['openrouter_model'], SYSTEM, json.dumps(evidence, ensure_ascii=False), ['decisions'], max_tokens=4096,
                     progress=progress, cancel_check=cancel_check, provider=config.get('text_provider', 'openrouter'),
                     thinking=config.get('local_thinking', False), usage_out=usage_out,
                     validator=functools.partial(validate, questions=questions), response_format=decision_format(questions), temperature=0.1)


# --- the resolver (pure) ------------------------------------------------------------------------------------------------

def resolve(question, decision, topic_policy=None, trust=None, mode='apply'):
    """What to do with one question: ``{action: apply|claude|human, option, tier, why}``.

    ``trust`` is the topic's measured agreement with the user (``trust_table``), None when unmeasured. ``claude``
    means "the second reviewer should look"; the caller turns it into ``human`` when Claude is off or out of budget.
    """
    topic_policy = topic_policy or {}
    recommended = question.get('recommended')
    decision = decision or {}
    choice, confident = decision.get('choice'), decision.get('confidence') == 'high'
    proposal = choice or recommended
    if choice and confident and decision.get('rule') and topic_policy.get('rule'):
        outcome = {'action': 'apply', 'option': choice, 'tier': 'local', 'why': 'a standing rule decides it'}
    elif decision.get('kind') == 'creative':
        outcome = {'action': 'human', 'option': proposal, 'tier': None, 'why': 'a creative choice no standing rule covers'}
    elif choice and confident and choice == recommended and trust is not None and trust >= TRUST:
        outcome = {'action': 'apply', 'option': choice, 'tier': 'local', 'why': f'the decider agrees with the reviewer; this topic agrees with you {trust:.0%}'}
    else:
        when = topic_policy.get('when_unsure', 'evidence')
        if choice and recommended and choice != recommended:
            doubt = 'the decider and the reviewer disagree'
        elif not choice:
            doubt = 'the evidence does not settle it'
        elif not confident:
            doubt = f"the decider is only {decision.get('confidence', 'low')} on it"
        else:
            doubt = 'this topic is not trusted yet' if trust is None else f'this topic agrees with you only {trust:.0%}'
        if when == 'recommended' and recommended:
            outcome = {'action': 'apply', 'option': recommended, 'tier': 'rules', 'why': f'{doubt}; your profile takes the recommended option'}
        elif when == 'ask':
            outcome = {'action': 'human', 'option': proposal, 'tier': None, 'why': f'{doubt}; your profile asks you'}
        else:
            outcome = {'action': 'claude', 'option': proposal, 'tier': None, 'why': doubt}
    if mode == 'ask' and outcome['action'] == 'apply':
        outcome = {**outcome, 'action': 'human', 'tier': None, 'why': outcome['why'] + '; your profile asks before answering'}
    return outcome


# --- trust: how often each topic's agreed answer matches the user -------------------------------------------------------

def _current_reviews(conn, project_id=None, cut_ids=None):
    """``{cut_id: (result id, questions)}`` of current consistency reviews that ask questions."""
    clauses, args = ["r.title='Consistency summary'", 'r.superseded_by IS NULL', 'c.active=1'], []
    if project_id:
        clauses.append('e.project_id=?')
        args.append(project_id)
    if cut_ids is not None:
        if not cut_ids:
            return {}
        clauses.append(f"r.cut_id IN ({','.join('?' for _ in cut_ids)})")
        args += list(cut_ids)
    found = {}
    for row in conn.execute(f"""SELECT r.id,r.cut_id,r.content FROM analysis_results r JOIN cuts c ON c.id=r.cut_id JOIN episodes e ON e.id=c.episode_id
            WHERE {' AND '.join(clauses)} ORDER BY r.rowid""", args):
        try:
            content = json.loads(row['content'])
        except (TypeError, ValueError):
            continue
        questions = content.get('questions') if isinstance(content, dict) else None
        if isinstance(questions, list) and questions:
            found[row['cut_id']] = (row['id'], questions)
    return found


def agent_held(conn, cut_ids):
    """Cuts whose saved answers still hold an agent decision nobody changed since: their answers are not the user's."""
    if not cut_ids:
        return set()
    marks = ','.join('?' for _ in cut_ids)
    return {row[0] for row in conn.execute(f"""SELECT DISTINCT target_id FROM agent_decisions WHERE kind=? AND undone_at IS NULL AND overridden_at IS NULL
        AND target_id IN ({marks})""", [KIND, *cut_ids])}


def trust_table(conn, project_id):
    """Per topic ``{rate, scored, source}``: how often an answer the agent would apply alone matches the user's own.

    Measured from shadow decisions (the decider agreeing with the reviewer at high confidence) where the user answered
    the same questions; until a topic has ``MIN_SCORED`` of those, the user's acceptance of the reviewer's recommended
    option stands in for it (a lower bound: the decider can only add a second opinion). Answers the agent wrote are
    never counted as the user's.
    """
    reviews = _current_reviews(conn, project_id)
    answers = {row['cut_id']: row['content'] for row in conn.execute(
        f"SELECT cut_id,content FROM cut_brief_instructions WHERE cut_id IN ({','.join('?' for _ in reviews) or 'NULL'})", list(reviews))}
    held = agent_held(conn, list(answers))
    shadow, history = {}, {}
    for cut_id, (result_id, questions) in reviews.items():
        text = answers.get(cut_id)
        if not text or cut_id in held:
            continue
        picked = {question['id']: (selection(text, question, questions) or {}).get('option') for question in questions}
        for question in questions:
            if picked[question['id']] is None:
                continue
            entry = history.setdefault(question.get('topic') or 'other', [0, 0])
            entry[0] += 1
            entry[1] += picked[question['id']] == question.get('recommended')
        row = conn.execute('SELECT decisions_json FROM review_decisions WHERE cut_id=? AND result_id=? AND shadow=1 ORDER BY rowid DESC LIMIT 1',
                           (cut_id, result_id)).fetchone()
        decided = json.loads(row['decisions_json']) if row else {}
        for question in questions:
            decision = decided.get(question['id']) or {}
            if picked[question['id']] is None or decision.get('confidence') != 'high' or not decision.get('choice') or decision['choice'] != question.get('recommended'):
                continue
            entry = shadow.setdefault(question.get('topic') or 'other', [0, 0])
            entry[0] += 1
            entry[1] += picked[question['id']] == decision['choice']
    table = {}
    for topic in {*shadow, *history}:
        scored, right = shadow.get(topic, (0, 0))
        if scored >= MIN_SCORED:
            table[topic] = {'rate': right / scored, 'scored': scored, 'source': 'shadow'}
            continue
        answered, recommended = history.get(topic, (0, 0))
        table[topic] = ({'rate': recommended / answered, 'scored': answered, 'source': 'recommended'} if answered >= MIN_SCORED
                        else {'rate': None, 'scored': answered + scored, 'source': 'too few'})
    return table


# --- queueing -----------------------------------------------------------------------------------------------------------

def policy_for(conn, episode_id, run_id=None):
    """The job's snapshot: the mode and Claude settings of ``review_answers`` plus the topics' rules and the notes."""
    from . import autopilot_profile
    profile = autopilot_profile.profile_for(conn, episode_id, run_id, KIND)
    claude = profile.get('claude') or {}
    return {'mode': profile['modes'].get(KIND, 'apply'), 'claude': bool(claude.get('enabled', True)),
            'claude_items': int(claude.get('max_items_per_run', 0) or 0), 'topics': profile.get('topics') or {}, 'notes': profile.get('notes') or ''}


def _decided(conn, cut_id, result_id, shadow):
    return conn.execute('SELECT 1 FROM review_decisions WHERE cut_id=? AND result_id=? AND shadow=?', (cut_id, result_id, int(shadow))).fetchone() is not None


def queue(conn, episode_id, *, first_index=0, last_index=None, run_id=None, shadow=False, redo=False, cut_ids=None):
    """One decider job per cut with open review questions, inside the caller's transaction.

    Returns ``{job_ids, skipped: {reason: [cut numbers]}}``, or None when the profile switched answers off. A cut waits
    for its pending consistency review. A shadow run ignores the switch (it writes nothing) and takes only the cuts
    whose current review the user answered themselves: it exists to be scored against those answers."""
    from . import main
    policy = policy_for(conn, episode_id, run_id)
    if policy['mode'] == 'off' and not shadow:
        return None
    wanted = set(cut_ids) if cut_ids else None
    cuts = [row for row in conn.execute("""SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR cut_index<=?)
        ORDER BY cut_index""", (episode_id, first_index or 0, last_index, last_index)) if wanted is None or row['id'] in wanted]
    reviews = _current_reviews(conn, cut_ids=[cut['id'] for cut in cuts])
    stale = {row[0] for row in conn.execute(f"""SELECT cut_id FROM analysis_results WHERE title='Consistency summary' AND superseded_by IS NULL
        AND stale_reason IS NOT NULL AND cut_id IN ({','.join('?' for _ in cuts) or 'NULL'})""", [cut['id'] for cut in cuts])}
    answers = {row['cut_id']: row['content'] for row in conn.execute(
        f"SELECT cut_id,content FROM cut_brief_instructions WHERE cut_id IN ({','.join('?' for _ in cuts) or 'NULL'})", [cut['id'] for cut in cuts])}
    held = agent_held(conn, list(answers)) if shadow else set()
    skipped = {'analysis_off': [], 'no_questions': [], 'review_stale': [], 'answered': [], 'no_user_answers': [], 'decided': []}
    job_ids = []
    for cut in cuts:
        number = cut['cut_index'] + 1
        if not cut['analysis_enabled']:
            skipped['analysis_off'].append(number)
            continue
        pending = worker.pending_jobs(conn, episode_id, ('consistency_summary',), [cut['id']])
        if shadow and (pending or cut['id'] not in reviews or cut['id'] in held
                       or not any(selection(answers.get(cut['id']), question, reviews[cut['id']][1]) for question in reviews[cut['id']][1])):
            skipped['no_user_answers'].append(number)
            continue
        if not pending:
            if cut['id'] not in reviews:
                skipped['no_questions'].append(number)
                continue
            if cut['id'] in stale and not shadow:  # the user answered a stale review too: still a fair measurement
                skipped['review_stale'].append(number)
                continue
            result_id, questions = reviews[cut['id']]
            if not shadow and all(selection(answers.get(cut['id']), question, questions) for question in questions):
                skipped['answered'].append(number)
                continue
            if not redo and _decided(conn, cut['id'], result_id, shadow):
                skipped['decided'].append(number)
                continue
        config = main.snapshot_analysis_config(conn, cut).model_dump()
        payload = {'cut_id': cut['id'], 'config': config, 'policy': policy, 'run_id': run_id, 'shadow': bool(shadow), 'depends_on': pending}
        job_ids.append(worker.insert_job(conn, episode_id, PIPELINE, payload, priority_class='batch'))
    if job_ids:
        worker.sync_episode_status(conn, episode_id)
    return {'job_ids': job_ids, 'skipped': {key: value for key, value in skipped.items() if value}}


# --- the job ------------------------------------------------------------------------------------------------------------

def _review(conn, cut_id):
    row = conn.execute("""SELECT id,content,stale_reason FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL
        AND title='Consistency summary' ORDER BY rowid DESC LIMIT 1""", (cut_id,)).fetchone()
    if not row:
        return None
    try:
        content = json.loads(row['content'])
    except (TypeError, ValueError):
        return None
    return {'id': row['id'], 'content': content if isinstance(content, dict) else {}, 'stale_reason': row['stale_reason']}


def gather(conn, job):
    payload = json.loads(job['payload_json'])
    cut_id = payload['cut_id']
    context = analysis_context.selected(conn, cut_id, payload['config'])
    if not context or not context['cut']['active']:
        return {'skip': 'Cut archived or removed'}
    review = _review(conn, cut_id)
    questions = (review or {}).get('content', {}).get('questions') or []
    if not review or not isinstance(questions, list) or not questions:
        return {'skip': 'No review questions'}
    if review['stale_reason'] and not payload.get('shadow'):
        return {'skip': 'The consistency review is stale; write it again first'}
    answers = review_answers.saved_answers(conn, cut_id)
    results = structured_results.list_results(cut_id, conn=conn)
    try:
        base = review_answers.build_evidence(results, context, '' if payload.get('shadow') else answers, prompt_context.slim_neighbors(context))
    except ValueError as exc:
        return {'missing': str(exc)}
    return {'result_id': review['id'], 'questions': questions, 'answers': answers, 'evidence': evidence_for(base, questions, payload.get('policy')),
            'cut_index': context['cut']['cut_index'], 'source_result_ids': [item['id'] for item in results]}


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    """Applied answers mark the Cut summary stale themselves (``review_answers.save_hand_answers``)."""


def _done(conn, job_id, stage, payload=None):
    if payload is not None:
        conn.execute("UPDATE jobs SET payload_json=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), job_id))
    conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (stage, job_id))


def process(job_id, owner):
    from . import inference_worker, local_text_runtime
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    config = payload.get('config') or {}
    try:
        with db.connect() as conn:
            conn.execute('BEGIN')
            snapshot = job_inputs.JobInputs(conn, job, owner)
        values = snapshot.data
        if values.get('missing'):
            with snapshot.publish(invalidate=False):
                raise RuntimeError(values['missing'])
        open_questions = values.get('questions') or []
        if not payload.get('shadow') and not values.get('skip'):
            open_questions = [question for question in open_questions if selection(values['answers'], question, values['questions']) is None]
        if values.get('skip') or not open_questions:
            with snapshot.publish(invalidate=False) as conn:
                if values.get('result_id'):
                    close_settled(conn, job['episode_id'], [payload['cut_id']])
                _done(conn, job_id, f"Skipped · {values.get('skip') or 'every question is answered'}")
                worker.sync_episode_status(conn, job['episode_id'])
            return
        worker.stage(job_id, owner, job['episode_id'], f'Deciding {len(open_questions)} review question(s)')
        telemetry = {}
        decisions, model = generate(config, values['evidence'], values['questions'],
            progress=lambda attempt, total: worker.stage(job_id, owner, job['episode_id'], f'Review decisions · attempt {attempt}/{total}'),
            cancel_check=lambda: inference_worker.check_summary_lease(job_id, owner), usage_out=telemetry)
        local = config.get('text_provider') == 'local'
        provenance = {'model': model, 'prompt_version': PROMPT_VERSION, 'fingerprint': snapshot.stamp, 'usage': telemetry,
                      'model_revision': local_text_runtime.revision_for(telemetry.get('model_variant')) if local else None}
        with snapshot.publish(invalidate=False) as conn:
            counts = publish(conn, job, payload, values, decisions, provenance)
            told = ', '.join(f'{count} {name}' for name, count in counts.items() if count)
            _done(conn, job_id, f"Complete · {'shadow · ' if payload.get('shadow') else ''}{told}", {**payload, 'result': counts})
            worker.sync_episode_status(conn, job['episode_id'])
    except inference_worker.SummaryCancelled:
        if config.get('text_provider') == 'local':
            local_text_runtime.cancel_local('Cancelled')
    except worker.LeaseLost:
        return
    except Exception as exc:
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)


def publish(conn, job, payload, values, decisions, provenance):
    """Save the decisions; unless shadow, apply what the resolver applies and escalate the rest. Returns counts."""
    episode_id, cut_id, questions = job['episode_id'], payload['cut_id'], values['questions']
    policy = payload.get('policy') or {}
    project = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    trust = trust_table(conn, project['project_id'])
    shadow = bool(payload.get('shadow'))
    outcomes = {}
    for question in questions:
        if not shadow and selection(values['answers'], question, questions) is not None:
            outcomes[question['id']] = {'action': 'keep', 'why': 'you answered it'}
            continue
        topic = question.get('topic') or 'other'
        outcomes[question['id']] = resolve(question, decisions.get(question['id']), (policy.get('topics') or {}).get(topic),
                                           (trust.get(topic) or {}).get('rate'), policy.get('mode', 'apply'))
    claude_left = claude_budget(conn, episode_id, payload.get('run_id'), policy) if not shadow else 0
    wants_claude = any(outcome['action'] == 'claude' for outcome in outcomes.values())
    if wants_claude and claude_left <= 0:
        for outcome in outcomes.values():
            if outcome['action'] == 'claude':
                outcome.update(action='human', why=outcome['why'] + ('; Claude is off in your profile' if not policy.get('claude') else '; the Claude budget of this run is used up'))
    conn.execute("""INSERT INTO review_decisions(id,cut_id,episode_id,result_id,job_id,run_id,shadow,decisions_json,outcomes_json,model_id,provenance_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (str(uuid.uuid4()), cut_id, episode_id, values['result_id'], job['id'], payload.get('run_id'), int(shadow),
                                            json.dumps(decisions, ensure_ascii=False), json.dumps(outcomes, ensure_ascii=False), provenance.get('model'),
                                            json.dumps(provenance, ensure_ascii=False)))
    counts = {'applied': 0, 'for Claude': 0, 'for you': 0, 'kept': 0}
    for outcome in outcomes.values():
        counts[{'apply': 'applied', 'claude': 'for Claude', 'human': 'for you', 'keep': 'kept'}[outcome['action']]] += 1
    if shadow:
        return counts
    by_id = {question['id']: question for question in questions}
    picks = [{'question': by_id[question_id], 'option': outcome['option'], 'tier': outcome['tier'], 'why': outcome['why'],
              'decider': decisions.get(question_id)} for question_id, outcome in outcomes.items() if outcome['action'] == 'apply']
    for tier in ('local', 'rules'):
        chosen = [pick for pick in picks if pick['tier'] == tier]
        if chosen:
            apply_answers(conn, episode_id=episode_id, cut_id=cut_id, result_id=values['result_id'], questions=questions, picks=chosen, tier=tier,
                          run_id=payload.get('run_id'), model=provenance.get('model'))
    for reviewer, action in (('claude', 'claude'), ('human', 'human')):
        items = [{**_question_view(by_id[question_id]), 'proposal': outcome['option'], 'why_asked': outcome['why'],
                  'decider': decisions.get(question_id)} for question_id, outcome in outcomes.items() if outcome['action'] == action]
        if items:
            escalate_questions(conn, episode_id, cut_id, values['result_id'], items, reviewer, payload.get('run_id'))
    close_settled(conn, episode_id, [cut_id])
    return counts


def claude_budget(conn, episode_id, run_id, policy):
    """How many more cuts this run (or, without a run, this episode) may hand to Claude."""
    if not policy.get('claude'):
        return 0
    if run_id:
        used = conn.execute("SELECT COUNT(*) FROM agent_escalations WHERE kind=? AND run_id=? AND group_key LIKE '%:claude'", (ESCALATION, run_id)).fetchone()[0]
    else:
        used = conn.execute("SELECT COUNT(*) FROM agent_escalations WHERE kind=? AND episode_id=? AND status='open' AND reviewer='claude'",
                            (ESCALATION, episode_id)).fetchone()[0]
    return int(policy.get('claude_items', 0)) - used


def _question_view(question):
    return {key: question[key] for key in ('id', 'topic', 'character', 'question', 'options', 'recommended', 'why') if question.get(key) not in (None, '')}


# --- applying answers ---------------------------------------------------------------------------------------------------

def apply_answers(conn, *, episode_id, cut_id, result_id, questions, picks, tier, run_id=None, model=None, note=None, escalation_id=None):
    """Write one line per pick into the cut's answers and log it as one decision. Picks whose question the user answered
    meanwhile are skipped. ``pick``: ``{question, option, text?, why?, decider?}``. Returns the decision id or None."""
    before = review_answers.saved_answers(conn, cut_id)
    content, lines = before, []
    for pick in picks:
        question = pick['question']
        if selection(content, question, questions) is not None:
            continue
        text = option_text(question, pick['option']) if pick.get('option') else pick.get('text')
        if not text:
            continue
        content = set_answer(content, question, text, questions)
        lines.append({'question': question['id'], 'topic': question.get('topic'), 'character': question.get('character') or '', 'option': pick.get('option'),
                      'text': text, 'recommended': question.get('recommended'), **({'why': pick['why']} if pick.get('why') else {}),
                      **({'decider': pick['decider']} if pick.get('decider') else {})})
    if not lines:
        return None
    review_answers.save_hand_answers(conn, cut_id, content)
    topics = sorted({line['topic'] or 'other' for line in lines})
    agreed = sum(line['option'] is not None and line['option'] == line['recommended'] for line in lines)
    return agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=cut_id, kind=KIND, tier=tier,
                            target_table='cut_brief_instructions', target_id=cut_id, before={'content': before}, after={'content': content, 'lines': lines},
                            evidence={'result_id': result_id, 'model': model, **({'note': str(note)[:1000]} if note else {}),
                                      **({'escalation_id': escalation_id} if escalation_id else {})},
                            confidence=None, summary=f"Answered {len(lines)} review question(s) ({', '.join(topics)}; {agreed} recommended) [{tier}]")


def undo(conn, decision):
    """Put the answers back while they still are what the agent wrote."""
    before = json.loads(decision['before_json'] or '{}')
    after = json.loads(decision['after_json'] or '{}')
    current = review_answers.saved_answers(conn, decision['target_id'])
    if current != after.get('content'):
        raise agent_log.UndoConflict('The answers changed since; undo would overwrite that')
    review_answers.save_answers(conn, decision['target_id'], before.get('content') or '')
    review_answers.invalidate_saved_brief(conn, decision['target_id'])


agent_log.UNDO_KINDS[KIND] = undo


# --- escalations --------------------------------------------------------------------------------------------------------

def _group(cut_id, reviewer):
    return f'cut:{cut_id}' + (':claude' if reviewer == 'claude' else '')


def _cut_number(conn, cut_id):
    row = conn.execute('SELECT cut_index FROM cuts WHERE id=?', (cut_id,)).fetchone()
    return row['cut_index'] + 1 if row else None


def picture_labels(conn, result_id):
    """The labels of the pictures the review looked at (its stored references), in order."""
    row = conn.execute('SELECT context_json FROM analysis_results WHERE id=?', (result_id,)).fetchone()
    try:
        context = json.loads(row['context_json']) if row else {}
    except (TypeError, ValueError):
        context = {}
    sent = ((context.get('pictures') or {}).get('sent') or []) if isinstance(context, dict) else []
    return [picture['label'] for picture in sent if isinstance(picture, dict) and picture.get('label')]


def cited(question, labels):
    """The picture labels a question's ``why`` names (``F3``, ``line strip``, ``first frame``)."""
    why = str(question.get('why') or '')
    found = [label for label in labels if re.search(rf'(?<![\w]){re.escape(label)}(?![\w])', why, re.I)]
    if re.search(r'line strip', why, re.I):
        found += [label for label in labels if label.lower().startswith('line ') and label not in found]
    if re.search(r'first frame', why, re.I) and 'First frame' in labels and 'First frame' not in found:
        found.append('First frame')
    return found


def escalate_questions(conn, episode_id, cut_id, result_id, items, reviewer, run_id=None):
    """Open (or extend) the cut's escalation for ``reviewer`` with these questions; a question already listed is replaced."""
    group = _group(cut_id, reviewer)
    existing = conn.execute("SELECT evidence_json FROM agent_escalations WHERE episode_id=? AND kind=? AND group_key=? AND status='open'",
                            (episode_id, ESCALATION, group)).fetchone()
    evidence = json.loads(existing['evidence_json'] or '{}') if existing else {}
    if evidence.get('result_id') not in (None, result_id):
        evidence = {}
    labels = picture_labels(conn, result_id)
    listed = {item['id']: item for item in evidence.get('questions') or []}
    for item in items:
        listed[item['id']] = {**item, 'pictures': cited(item, labels)}
    questions = sorted(listed.values(), key=lambda item: item['id'])
    number = _cut_number(conn, cut_id)
    wanted = sorted({label for item in questions for label in item.get('pictures') or []}, key=labels.index) or labels[:4]
    evidence = {'cut_id': cut_id, 'cut': number, 'result_id': result_id, 'questions': questions, 'pictures': labels,
                'sheet': f"/api/cuts/{cut_id}/review-sheet" + (f"?labels={','.join(wanted)}" if wanted else '')}
    question = f"Cut {number}: {len(questions)} review question(s) to decide (" + ', '.join(f"{item.get('topic')}" + (f" · {item['character']}" if item.get('character') else '') for item in questions) + ')'
    proposal = {item['id']: item.get('proposal') for item in questions}
    return agent_log.escalate(conn, episode_id=episode_id, kind=ESCALATION, group_key=group, question=question, cut_ids=[cut_id],
                              proposal=proposal, evidence=evidence, run_id=run_id, reviewer=reviewer)


def close_settled(conn, episode_id, cut_ids=None):
    """Close open review-question escalations whose review was replaced (dismissed) or whose questions all have an answer
    now (resolved; the person may have answered in the Results tab). Returns how many were closed."""
    rows = conn.execute("SELECT * FROM agent_escalations WHERE episode_id=? AND kind=? AND status='open'", (episode_id, ESCALATION)).fetchall()
    closed = 0
    for row in rows:
        evidence = json.loads(row['evidence_json'] or '{}')
        cut_id = evidence.get('cut_id')
        if cut_ids is not None and cut_id not in cut_ids:
            continue
        review = _review(conn, cut_id) if cut_id else None
        if not review or review['id'] != evidence.get('result_id'):
            agent_log.close_escalation(conn, row['id'], 'dismissed', {'why': 'the consistency review was written again'})
            closed += 1
            continue
        questions = review['content'].get('questions') or []
        answers = review_answers.saved_answers(conn, cut_id)
        listed = {item['id'] for item in evidence.get('questions') or []}
        if all(selection(answers, question, questions) is not None for question in questions if question['id'] in listed):
            agent_log.close_escalation(conn, row['id'], 'resolved', {'why': 'every question has an answer'})
            closed += 1
    return closed


class Refused(Exception):
    pass


def answer(conn, escalation_id, picks, unsure=(), tier='human', run_id=None, note=None):
    """Apply Claude's or the person's answers to one cut's questions.

    ``picks``: ``[{question, option}]`` (the person may give ``text`` instead of an option). ``unsure``: question ids
    Claude cannot settle; they move to the person's escalation of the cut. Returns ``{status, decision_id, open}``."""
    if tier not in ('claude', 'human'):
        raise Refused('tier must be claude or human')
    row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
    if row is None:
        raise KeyError(escalation_id)
    if row['kind'] != ESCALATION:
        raise Refused('This is not a review-question escalation')
    if row['status'] != 'open':
        raise Refused('This question is already closed')
    if tier == 'claude' and row['reviewer'] != 'claude':
        raise Refused('These questions wait for a person')
    evidence = json.loads(row['evidence_json'] or '{}')
    cut_id, episode_id = evidence.get('cut_id'), row['episode_id']
    review = _review(conn, cut_id) if cut_id else None
    if not review or review['id'] != evidence.get('result_id'):
        agent_log.close_escalation(conn, escalation_id, 'dismissed', {'why': 'the consistency review was written again'})
        return {'status': 'outdated', 'decision_id': None, 'open': []}
    questions = review['content'].get('questions') or []
    listed = {item['id']: item for item in evidence.get('questions') or []}
    by_id = {question['id']: question for question in questions}
    chosen, problems = [], []
    for pick in picks or []:
        question = by_id.get(pick.get('question'))
        if question is None or question['id'] not in listed:
            problems.append(f"{pick.get('question')} is not one of this escalation's questions")
        elif pick.get('option'):
            if option_text(question, pick['option']) is None:
                problems.append(f"{question['id']} has no option {pick['option']}")
            else:
                chosen.append({'question': question, 'option': pick['option'], 'why': pick.get('why')})
        elif pick.get('text') and tier == 'human':
            chosen.append({'question': question, 'option': None, 'text': ' '.join(str(pick['text']).split())[:600]})
        else:
            problems.append(f"{question['id']}: answer with an option id" + (' (Claude only picks options)' if tier == 'claude' else ' or a text'))
    unsure = [question_id for question_id in unsure or [] if question_id in listed]
    if problems:
        raise Refused('; '.join(problems))
    decision_id = apply_answers(conn, episode_id=episode_id, cut_id=cut_id, result_id=review['id'], questions=questions, picks=chosen, tier=tier,
                                run_id=run_id or row['run_id'], note=note, escalation_id=escalation_id) if chosen else None
    if unsure and tier == 'claude':
        moved = [{**listed[question_id], 'why_asked': f"Claude was unsure{': ' + str(note)[:300] if note else ''}"} for question_id in unsure]
        escalate_questions(conn, episode_id, cut_id, review['id'], moved, 'human', run_id or row['run_id'])
    answers = review_answers.saved_answers(conn, cut_id)
    left = [question_id for question_id in listed if not (tier == 'claude' and question_id in unsure)
            and question_id in by_id and selection(answers, by_id[question_id], questions) is None]
    resolution = {'by': tier, 'decision_id': decision_id, **({'note': str(note)[:1000]} if note else {})}
    if tier == 'claude':
        # Claude has one look per cut: what it was unsure of or left unanswered goes to the person.
        if left:
            moved = [{**listed[question_id], 'why_asked': 'Claude did not answer it'} for question_id in left]
            escalate_questions(conn, episode_id, cut_id, review['id'], moved, 'human', run_id or row['run_id'])
        agent_log.close_escalation(conn, escalation_id, 'resolved', {**resolution, 'handed_to_person': unsure + left})
        return {'status': 'handed_to_person' if unsure or left else 'resolved', 'decision_id': decision_id, 'open': []}
    if not left:
        agent_log.close_escalation(conn, escalation_id, 'resolved', resolution)
        return {'status': 'resolved', 'decision_id': decision_id, 'open': []}
    return {'status': 'open', 'decision_id': decision_id, 'open': left}


# --- the review sheet Claude looks at -----------------------------------------------------------------------------------

MAX_SHEET = 6


def review_sheet(conn, cut_id, labels=None):
    """JPEG bytes: the pictures the cut's current review looked at (``labels`` or the first few), tiled two per row."""
    import tempfile
    from pathlib import Path
    from . import consistency_evidence, media
    review = _review(conn, cut_id)
    if not review:
        raise ValueError('This cut has no consistency review')
    row = conn.execute('SELECT context_json FROM analysis_results WHERE id=?', (review['id'],)).fetchone()
    context = json.loads(row['context_json'] or '{}')
    sent = [picture for picture in ((context.get('pictures') or {}).get('sent') or []) if isinstance(picture, dict)]
    if not sent:
        raise ValueError('The review looked at no pictures (it was written without vision)')
    if labels:
        wanted = {label.casefold() for label in labels}
        sent = [picture for picture in sent if str(picture.get('label', '')).casefold() in wanted]
    sent = [picture for picture in sent if picture.get('kind') in ('first', 'keyframe', 'line', 'last')][:MAX_SHEET] or sent[:MAX_SHEET]
    size = conn.execute('SELECT e.width,e.height FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=?', (cut_id,)).fetchone()
    aspect = (size['height'] / size['width']) if size and size['width'] and size['height'] else 9 / 16
    with tempfile.TemporaryDirectory(prefix='framewise-review-sheet-') as temp:
        items = []
        for index, picture in enumerate(sent):
            path = Path(temp) / f'{index}.jpg'
            try:
                path.write_bytes(consistency_evidence.render_one(picture))
            except Exception:  # noqa: BLE001 - a picture that cannot be cut shows as a black tile labelled missing
                path = None
            items.append((path, str(picture.get('label'))))
        output = Path(temp) / 'sheet.jpg'
        media.tile_images(items, output, 2, (512, max(160, int(512 * aspect))))
        return output.read_bytes()
