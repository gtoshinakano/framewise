"""Cut Assistance tools for the chat about a sequence.

A sequence chat reads the bible, the shot plan and the shots, looks up the analysis of any cut in
the range, and proposes changes. A shot's prompt is never edited as text: the assistant changes the
shot's direction (a JSON object), the app renders the prompt again and checks it, and only a
direction that passes becomes a proposal. Plan and bible proposals are validated the same way a
model answer is. Nothing is saved until the user applies the proposal.
"""
import json

from .. import db, prompt_context, sequence_direction as direction, sequence_edit, sequence_evidence as evidence, sequence_prompt, sequence_store, sequence_timeline, sequence_view
from . import tools
from .tools import ToolError, tool

EDIT_NOTE = 'Not saved yet. Tell the user to review and apply it on the proposal card.'
CHANGE_PROPERTIES = {
    'replacements': {'type': 'array', 'items': {'type': 'object', 'properties': {'find': {'type': 'string'}, 'replace': {'type': 'string'}, 'count': {'type': 'integer'}}, 'required': ['find', 'replace']},
                     'description': 'Exact text to replace inside string values (all occurrences unless count).'},
    'set': {'type': 'array', 'items': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'value': {}}, 'required': ['path', 'value']},
            'description': 'Set one value by path, for example beats.1.action or camera.move.'},
    'content': {'type': 'string', 'description': 'The full new JSON object, only when most of it changes.'},
    'summary': {'type': 'string', 'description': 'One line describing the change'},
}


def load(ctx):
    with db.connect() as conn:
        sequence = sequence_store.get(conn, ctx.sequence_id)
        if not sequence:
            raise ToolError('The sequence no longer exists')
        current = sequence_view.state(conn, sequence)
        pending = sequence_store.pending_jobs(conn, sequence['id'])
    if current['cuts'] is None:
        raise ToolError('The first or last cut of this sequence was archived; the user must create the sequence again')
    return sequence, current, pending


def views(current, pending=()):
    return sequence_view.shot_views(current, pending)


def find_shot(shots, number):
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= len(shots):
        raise ToolError(f'There is no shot {number}; the plan has {len(shots)} shot(s)' if shots else 'This sequence has no shot plan yet')
    return shots[number - 1]


def check_label(view):
    if not view['result']:
        return 'not written'
    verdict = view['result']['check'] or {}
    return 'ok' if verdict.get('ok') else f"{len(verdict.get('errors') or [])} format error(s)"


def shot_row(view, bible=None):
    return {'shot': view['index'], 't': tools.span(view['t0'], view['t1']), 'seconds': view['duration'], 'cut': view['cut'], 'part': view['part'],
            'opens': evidence.edge_brief(view['opens']), 'closes': evidence.edge_brief(view['closes']), 'intent': view['intent'][:140], 'prompt': check_label(view),
            'stale_because': view['stale'] or None, 'lines': [line['line'] for line in view['lines']] or None,
            'voices': evidence.voices_text((view['result'] or {}).get('content') or {}, view, bible) if view['result'] else None,
            'voices_set_by_user': view['fixed_voices'] or None, 'job': view['job']}


def sequence_card(sequence_id):
    """Terse status block for the system prompt so simple questions need no tool call."""
    with db.connect() as conn:
        sequence = sequence_store.get(conn, sequence_id)
        if not sequence:
            return 'The sequence no longer exists.'
        current = sequence_view.state(conn, sequence)
        pending = sequence_store.pending_jobs(conn, sequence_id)
        names = conn.execute('SELECT e.title AS episode,p.title AS project FROM episodes e JOIN projects p ON p.id=e.project_id WHERE e.id=?', (sequence['episode_id'],)).fetchone()
    info = sequence_view.sequence_view(sequence, current['cuts'], current['changed'])
    lines = [f"Sequence \"{sequence['title']}\" of episode \"{names['episode']}\" (project \"{names['project']}\"): cuts {info['first_cut']}–{info['last_cut']}, "
             f"{info['duration']:.1f} s, episode time {info['start_time']:.3f}–{info['end_time']:.3f}s."]
    if sequence['instruction'].strip():
        lines.append(f"Director instruction: {sequence['instruction'].strip()[:400]}")
    if info['cuts_missing']:
        return '\n'.join([*lines, 'Its first or last cut was archived: it must be created again.'])
    if info['cuts_changed']:
        lines.append('The cuts changed since it was directed: the user must direct the sequence again before shots can be written.')
    if current['missing']:
        lines.append('Cuts without first-frame or video analysis: ' + ', '.join(map(str, current['missing'])) + '.')
    shots = views(current, pending)
    lines.append(f"Bible: {'written (' + current['bible_row']['origin'] + ')' if current['bible_row'] else 'missing'}. "
                 f"Plan: {str(len(shots)) + ' shots' if shots else ('does not fit the cuts' if current['plan_problems'] else 'missing')}.")
    if shots:
        written = [view for view in shots if view['result']]
        stale = [str(view['index']) for view in shots if view['stale']]
        failing = [str(view['index']) for view in written if not (view['result']['check'] or {}).get('ok')]
        lines.append(f"Shots written: {len(written)}/{len(shots)}; stale: {', '.join(stale) or 'none'}; failing the format check: {', '.join(failing) or 'none'}.")
    review = sequence_store.loads(current['review_row']['content'], {}) if current['review_row'] else None
    if review:
        lines.append('Review: ' + (f"{len(review['issues'])} issue(s) in shots {', '.join(sorted({str(issue['shot']) for issue in review['issues']}))}." if review.get('issues') else 'no issues.'))
    if pending:
        steps = [job['step'] for job in pending]
        lines.append('Running/queued: ' + ', '.join(f'{step} ×{steps.count(step)}' for step in dict.fromkeys(steps)) + '.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------- read

@tool('get_sequence', 'Read this sequence. part: overview (status, story, every shot in one row, review issues; the default), bible (the full sequence bible), plan (every shot with its cut, times, frame and intent, and every hard cut that falls inside speech with the choice made there and its options) or review.',
      {'part': {'type': 'string', 'enum': ['overview', 'bible', 'plan', 'review']}},
      summary=lambda args, value: f"Read sequence {args.get('part') or 'overview'}", limit=9000, scope='sequence')
def get_sequence(ctx, args):
    sequence, current, pending = load(ctx)
    part = args.get('part') or 'overview'
    bible, shots = current['bible'], views(current, pending)
    review = sequence_store.loads(current['review_row']['content'], {}) if current['review_row'] else None
    if part == 'bible':
        return {'bible': bible or 'not written yet', 'origin': current['bible_row']['origin'] if current['bible_row'] else None}
    if part == 'review':
        return {'review': review or 'not reviewed yet'}
    if part == 'plan':
        return {'plan_problems': current['plan_problems'] or None, 'notes': (current['plan'] or {}).get('notes'),
                'shots': [{'shot': view['index'], 't': tools.span(view['t0'], view['t1']), 'seconds': view['duration'], 'generate_seconds': view['generate_seconds'], 'cut': view['cut'],
                           'part': view['part'], 'audio_from_cuts': view['source_cuts'] if len(view['source_cuts']) > 1 else None, 'frame': view['frame']['strategy'],
                           'opens': evidence.edge_brief(view['opens']), 'closes': evidence.edge_brief(view['closes']),
                           'scene': view['scene'], 'intent': view['intent'], 'lines': [line['line'] for line in view['lines']] or None} for view in shots] or 'no shot plan yet',
                'boundaries_in_speech': [boundary_row(boundary) for boundary in sequence_view.boundary_views(current) if boundary['speech'] or boundary.get('fixed')] or None}
    info = sequence_view.sequence_view(sequence, current['cuts'], current['changed'])
    return {'sequence': {'title': info['title'], 'cuts': [info['first_cut'], info['last_cut']], 'seconds': info['duration'], 'instruction': sequence['instruction'] or None,
                         'cuts_changed': info['cuts_changed'] or None, 'cuts_without_analysis': current['missing'] or None},
            'story': {key: bible.get(key) for key in ('logline', 'synopsis', 'spoken_language', 'pacing', 'open_questions')} if bible else 'no bible yet',
            'shots': [shot_row(view, bible) for view in shots] or 'no shot plan yet',
            'review': {'verdict': review.get('verdict'), 'issues': review.get('issues')} if review else 'not reviewed yet',
            'jobs': [{'step': job['step'], 'status': job['status']} for job in pending] or None}


def boundary_row(boundary):
    """A hard cut inside speech (or one the user placed): where the boundary sits now, why, and where else it could sit (t in sequence seconds).

    A boundary the user placed on the Dialogue timeline is final: no plan edit may move it.
    """
    speech = boundary['speech'] or {}
    fixed = boundary.get('fixed')
    row = {'after_cut': boundary['after_cut'], 'hard_cut_t': boundary['t'], 'lines': speech.get('lines'), 'before_cut': speech.get('words_before'), 'after_cut_words': speech.get('words_after'),
           'choice': boundary['choice'], 'reason': boundary['reason'], 'decided_by': boundary['by']}
    if fixed:
        return {**row, 'placed_by_user': {'t': fixed['t'], 'shift': fixed['shift']}}
    return {**row, 'options': {choice: {'t': option['t'], 'shift': option['shift']} for choice, option in boundary['options'].items()},
            'not_possible': boundary['unavailable'] or None}


def dialogue_rows(view, bible):
    """The shot's lines with the delivery each one gets from its speaker's voice range in the written direction."""
    decided = {line['line']: line for line in ((view['result'] or {}).get('derived') or {}).get('dialogue') or []}
    rows = []
    for line in evidence.shot_dialogue(view, bible or {}):
        made = decided.get(line['line']) or {}
        rows.append({'line': line['line'], 't': tools.span(line['t0'], line['t1']), 'speaker': sequence_prompt.speaker_key(line), 'known_from': line['attribution'],
                     'delivery': sequence_prompt.delivery_of(made.get('delivery')), 'set_by_user': made.get('fixed_by_user') or None, 'text': line['text'],
                     'continues_from_previous_shot': line['crosses_start'] or None, 'continues_in_next_shot': line['crosses_end'] or None})
    return rows


@tool('get_shot', "Read one shot by number. direction is the JSON object that propose_shot_edit changes (paths such as beats.0.action, camera.move, voices.0.delivery). voices says per speaker how they are heard in this shot from which second (on_screen, off_screen, inner_voice, narrative); voices_set_by_user are ranges the user decided, which no edit may change. user_instructions are what the user asked for in earlier rewrites of this shot; every rewrite keeps following them, so no edit may undo them. Times are seconds from the start of the shot. fields narrows to: plan, dialogue, direction, check, prompt (the rendered H3 prompt; only returned when asked for).",
      {'shot': {'type': 'integer'}, 'fields': {'type': 'array', 'items': {'type': 'string', 'enum': ['plan', 'dialogue', 'direction', 'check', 'prompt']}}}, ('shot',),
      summary=lambda args, value: f"Read shot {args['shot']}", limit=10000, scope='sequence')
def get_shot(ctx, args):
    _, current, pending = load(ctx)
    view = find_shot(views(current, pending), args['shot'])
    fields = args.get('fields') or ['plan', 'dialogue', 'direction', 'check']
    result = view['result'] or {}
    out = {'shot': view['index'], 'written': bool(result), 'stale_because': view['stale'] or None, 'origin': result.get('origin')}
    if 'plan' in fields and view['fixed_voices']:
        out['voices_set_by_user'] = view['fixed_voices']
    if 'plan' in fields:
        with db.connect() as conn:
            asked = sequence_store.shot_instructions(conn, ctx.sequence_id, view['key'])
        if asked:
            out['user_instructions'] = asked
    if 'plan' in fields:
        out['plan'] = {'t_in_sequence': tools.span(view['t0'], view['t1']), 'seconds': view['duration'], 'generate_seconds': view['generate_seconds'], 'trim_to': view['trim_seconds'],
                       'cut': view['cut'], 'part': view['part'], 'frame': view['frame']['strategy'], 'opens': evidence.edge_words(view, 'opens'), 'closes': evidence.edge_words(view, 'closes'),
                       'scene': view['scene'], 'intent': view['intent']}
    if 'dialogue' in fields:
        out['dialogue'] = dialogue_rows(view, current['bible']) or 'nobody speaks during this shot'
    if 'direction' in fields:
        out['direction'] = result.get('content') or 'not written yet: the user runs "Write shot prompts"'
    if 'check' in fields and result:
        verdict = result.get('check') or {}
        out['check'] = {'ok': verdict.get('ok'), 'errors': verdict.get('errors') or None, 'warnings': verdict.get('warnings') or None, 'stats': verdict.get('stats')}
    if 'prompt' in fields:
        out['prompt'] = result.get('prompt') or 'not written yet'
    return out


@tool('get_cut_evidence', "Read a current analysis result of one source cut of this sequence, by cut number: what the direction was written from. Times are seconds from the start of that cut.",
      {'cut': {'type': 'integer', 'description': 'Cut number as shown in the editor'},
       'type': {'type': 'string', 'enum': ['initial_frame', 'keyframe_analysis', 'performance_beats', 'video_analysis', 'consistency_summary', 'production_brief', 'review_answers']},
       'fields': {'type': 'array', 'items': {'type': 'string'}}}, ('cut', 'type'),
      summary=lambda args, value: f"Read {tools.LABELS[args['type']]} of cut {args['cut']}" + (' (missing)' if isinstance(value, dict) and value.get('missing') else ''), limit=9000, scope='sequence')
def get_cut_evidence(ctx, args):
    _, current, _ = load(ctx)
    span = next((cut for cut in current['timeline']['cuts'] if cut['number'] == args['cut']), None)
    if span is None:
        raise ToolError(f"Cut {args['cut']} is not part of this sequence (cuts {current['timeline']['cuts'][0]['number']}–{current['timeline']['cuts'][-1]['number']})")
    cut = tools.load_cut(span['id'])
    with db.connect() as conn:
        row = tools.find_current(conn, span['id'], args['type'])
        motion = prompt_context.current_motion(conn, span['id']) if args['type'] == 'video_analysis' else None
    if motion and motion['type'] == 'keyframe_analysis':
        # The cut's motion comes from its keyframe analysis now; a video analysis kept from before it is out of date.
        view = prompt_context.video_view(motion['content'])
        return {'cut': args['cut'], 'plays_at_in_sequence': tools.span(span['t0'], span['t1']), 'type': args['type'], 'source': 'keyframe_analysis',
                'content': {key: value for key, value in view.items() if key != 'kind'} if isinstance(view, dict) else view}
    if row is None:
        return {'cut': args['cut'], 'type': args['type'], 'missing': True}
    view = tools.project(args['type'], row, cut)
    fields = args.get('fields')
    if fields and isinstance(view, dict) and set(view) != {'text'}:
        selected = {key: view[key] for key in fields if key in view}
        if not selected:
            raise ToolError(f"None of those fields exist. Available: {', '.join(view)}")
        view = selected
    return {'cut': args['cut'], 'plays_at_in_sequence': tools.span(span['t0'], span['t1']), 'type': args['type'], **tools.meta(row), 'content': view}


@tool('get_sequence_transcript', "Spoken lines of the sequence with their ids (L1…), speaker and how the speaker is known (transcript = the user's link, director = the bible's decision, unknown). Times are seconds from the start of the sequence; with shot, only that shot's lines on the shot's clock.",
      {'shot': {'type': 'integer'}}, summary=lambda args, value: 'Read sequence transcript' + (f" (shot {args['shot']})" if args.get('shot') else ''), limit=9000, scope='sequence')
def get_sequence_transcript(ctx, args):
    _, current, _ = load(ctx)
    bible = current['bible'] or {}
    if args.get('shot') is not None:
        view = find_shot(views(current), args['shot'])
        return {'shot': view['index'], 'seconds': view['duration'], 'lines': dialogue_rows(view, bible) or 'nobody speaks during this shot'}
    rows = []
    for line in current['timeline']['lines']:
        who = evidence.speaker_of(line, bible)
        rows.append({'line': line['id'], 't': tools.span(line['t0'], line['t1']), 'voice': line['speaker_label'] or 'unknown', 'speaker': who['name'], 'known_from': who['attribution'], 'text': line['text']})
    return {'seconds': current['timeline']['duration'], 'lines': rows or 'no spoken lines in this range'}


# ------------------------------------------------------------------ proposals

def changed_json(data, args, label):
    """A copy of a JSON result with the requested replacements, path sets or full content applied."""
    if args.get('content') is None and not args.get('replacements') and not args.get('set'):
        raise ToolError('Provide replacements, set, or content')
    if args.get('content') is not None:
        try:
            data = json.loads(args['content'])
        except ValueError as exc:
            raise ToolError(f'content must be a JSON object: {exc}') from exc
        if not isinstance(data, dict):
            raise ToolError('content must be a JSON object')
    else:
        data = json.loads(json.dumps(data))
    for item in args.get('set') or []:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or 'value' not in item:
            raise ToolError('Each set item needs path and value')
        tools.set_path(data, item['path'], item['value'])
    for item in args.get('replacements') or []:
        find, replace = tools.validate_replacement(item)
        counter = [int(item.get('count') or 0) or -1, 0]
        data = tools.replace_in_json(data, find, replace, counter)
        if counter[1] == 0:
            raise ToolError(f'Text not found in the {label}: "{find[:80]}". Copy the exact wording from the tool result.')
    return data


def propose(ctx, target, row, before, after, summary, payload):
    """Store a pending proposal; an earlier unsaved proposal of this turn for the same result is replaced."""
    if ctx.turn_id:
        with db.connect() as conn:
            conn.execute("UPDATE assistant_proposals SET status='discarded',resolved_at=CURRENT_TIMESTAMP WHERE turn_id=? AND target=? AND base_result_id=? AND status='pending'", (ctx.turn_id, target, row['id']))
    return tools.create_proposal(ctx, target, {'id': row['id']}, after, summary, before=before, payload={'sequence_id': ctx.sequence_id, **payload})


def not_possible(reason):
    """An edit that cannot be made at all (as opposed to one the model can correct): nothing is left for it to fix."""
    return {'ok': True, 'proposal_id': None, 'not_possible': reason}


@tool('propose_shot_edit', "Propose a change to one shot's direction. The app renders the H3 prompt again and checks it; if it returns errors, fix them and call again. The user must apply the proposal; nothing is saved until then. Prefer set (path + value) or replacements over full content.",
      {'shot': {'type': 'integer'}, **CHANGE_PROPERTIES}, ('shot', 'summary'),
      summary=lambda args, value: (f"Proposed a change to shot {args['shot']}" if value.get('proposal_id') else f"Shot {args['shot']}: {'no change possible' if value.get('ok') else 'direction has errors'}"),
      limit=5000, must_pass=True, scope='sequence')
def propose_shot_edit(ctx, args):
    _, current, _ = load(ctx)
    view = find_shot(views(current), args['shot'])
    row = current['shot_rows'].get(view['key'])
    if not row:
        return not_possible(f"Shot {view['index']} has no direction yet. Tell the user to run \"Write shot prompts\" first.")
    if not current['bible']:
        return not_possible('The sequence has no bible; the user must direct the sequence first.')
    old = sequence_store.loads(row['content'], {})
    data = changed_json(old, args, 'shot direction')
    try:
        normal, prompt, verdict, _, _ = sequence_edit.preview_shot(current, view, data)
    except ValueError as exc:
        return {'ok': False, 'errors': str(exc), 'note': 'Fix every problem and call propose_shot_edit again. Write names, not labels; the app writes labels, timecodes and spoken words.'}
    if normal == old:
        raise ToolError('The proposal does not change the shot')
    proposal_id = propose(ctx, 'sequence_shot', row, row['rendered'] or '', prompt, args['summary'], {'title': f"Shot {view['index']} prompt", 'shot_key': view['key'], 'direction': normal})
    return {'ok': True, 'proposal_id': proposal_id, 'status': 'pending_user_approval', 'shot': view['index'], 'warnings': verdict['warnings'] or None,
            'diff': tools.short_diff(row['rendered'] or '', prompt), 'note': EDIT_NOTE}


def plan_text(shots):
    """The plan as one readable line per shot, for the proposal card."""
    return '\n'.join(f"{shot['index']}. {shot['t0']:.2f}–{shot['t1']:.2f} s ({shot['duration']:.2f} s) · cut {shot['cut']}"
                     + (f" part {shot['part'][0]}/{shot['part'][1]}" if shot.get('part') else '')
                     + ''.join(f" · {side} {evidence.edge_brief(shot[side])}" for side in ('opens', 'closes') if evidence.edge_brief(shot[side]) and shot[side]['how'] != 'continuation')
                     + f" — {shot['intent']}" for shot in shots)


@tool('propose_plan_edit', "Propose a change to the shot plan. Every cut is exactly one shot (a cut longer than 15 s is divided by the app); shots are never merged or split. "
      "action boundary chooses where the shot boundary sits at the hard cut after cut `cut` when speech crosses it: keep_cut (on the hard cut; the speech is divided there), "
      "before_line or after_line (in the silence just before or after the phrase, so it stays in one shot); get_sequence part=plan lists these hard cuts with their options. "
      "A boundary with placed_by_user was placed by the user on the Dialogue timeline and is final: never propose to move it. "
      "action update changes the intent of the shot(s) of cut `cut`. The user must apply it. Shots whose interval or plan entry changes become stale and must be written again.",
      {'action': {'type': 'string', 'enum': ['boundary', 'update']}, 'cut': {'type': 'integer', 'description': 'Cut number: for boundary, the cut before the hard cut'},
       'choice': {'type': 'string', 'enum': list(sequence_timeline.CHOICES)}, 'reason': {'type': 'string', 'description': 'Why the boundary sits there (boundary)'},
       'intent': {'type': 'string'}, 'summary': {'type': 'string', 'description': 'One line describing the change'}}, ('action', 'cut', 'summary'),
      summary=lambda args, value: f"Proposed a plan change ({args['action']} at cut {args['cut']})" if value.get('proposal_id') else 'Plan change not possible', limit=5000, scope='sequence')
def propose_plan_edit(ctx, args):
    _, current, _ = load(ctx)
    row, plan, timeline = current['plan_row'], current['plan'], current['timeline']
    if not row or not current['shots']:
        return not_possible('There is no usable shot plan; the user must direct the sequence first.')
    cut = next((item for item in timeline['cuts'] if item['number'] == args['cut']), None)
    if cut is None:
        raise ToolError(f"Cut {args['cut']} is not part of this sequence (cuts {timeline['cuts'][0]['number']}–{timeline['cuts'][-1]['number']})")
    try:
        if args['action'] == 'boundary':
            if args.get('choice') not in sequence_timeline.CHOICES:
                raise ToolError(f"boundary needs choice: {', '.join(sequence_timeline.CHOICES)}")
            content = sequence_edit.boundary_plan(plan, timeline, cut['id'], args['choice'], args.get('reason') or '')
        else:
            if not isinstance(args.get('intent'), str) or not args['intent'].strip():
                raise ToolError('update needs intent')
            content = {**plan, 'shots': [{**span, 'intent': args['intent'].strip()} if sequence_timeline.owner_of(span, timeline)['id'] == cut['id'] else span for span in plan['shots']]}
        spans = sequence_edit.normal_spans(content['shots'], timeline)
        after = sequence_timeline.expand(spans, timeline, plan.get('voices'), content.get('boundaries'))
    except sequence_edit.EditError as exc:
        raise ToolError(exc.detail) from exc
    except sequence_timeline.PlanError as exc:
        raise ToolError(' '.join(exc.errors)) from exc
    before_text, after_text = plan_text(current['shots']), plan_text(after)
    if before_text == after_text and content.get('boundaries') == plan.get('boundaries'):
        raise ToolError('The proposal does not change the plan')
    kept = {shot['key'] for shot in after}
    proposal_id = propose(ctx, 'sequence_plan', row, before_text, after_text, args['summary'],
                          {'title': 'Shot plan', 'plan': {'shots': spans, 'boundaries': content.get('boundaries') or [], 'notes': plan.get('notes') or ''}})
    return {'ok': True, 'proposal_id': proposal_id, 'status': 'pending_user_approval', 'shots_after': len(after),
            'shots_to_write_again': [shot['index'] for shot in after if shot['key'] not in current['shot_rows']] or None,
            'written_shots_that_disappear': [shot['index'] for shot in current['shots'] if shot['key'] in current['shot_rows'] and shot['key'] not in kept] or None,
            'diff': tools.short_diff(before_text, after_text), 'note': EDIT_NOTE}


@tool('propose_bible_edit', "Propose a change to the sequence bible (story, scenes, characters' anchors and outfits, voices, continuity rules). The answer is checked like the director's own; if it returns errors, fix them and call again. The user must apply it. Shots written from the changed parts become stale.",
      CHANGE_PROPERTIES, ('summary',),
      summary=lambda args, value: 'Proposed a bible change' if value.get('proposal_id') else ('Bible change not possible' if value.get('ok') else 'Bible change has errors'), limit=5000, must_pass=True, scope='sequence')
def propose_bible_edit(ctx, args):
    _, current, _ = load(ctx)
    row, old = current['bible_row'], current['bible']
    if not row:
        return not_possible('The sequence has no bible yet; the user must direct the sequence first.')
    data = changed_json(old, args, 'bible')
    try:
        normal = direction.validate_bible(data, current['ctx'], current['timeline'])
    except ValueError as exc:
        return {'ok': False, 'errors': str(exc), 'note': 'Fix every problem and call propose_bible_edit again.'}
    if normal == old:
        raise ToolError('The proposal does not change the bible')
    before_text, after_text = json.dumps(old, ensure_ascii=False, indent=2), json.dumps(normal, ensure_ascii=False, indent=2)
    proposal_id = propose(ctx, 'sequence_bible', row, before_text, after_text, args['summary'], {'title': 'Sequence bible', 'bible': normal})
    return {'ok': True, 'proposal_id': proposal_id, 'status': 'pending_user_approval', 'diff': tools.short_diff(before_text, after_text), 'note': EDIT_NOTE}
