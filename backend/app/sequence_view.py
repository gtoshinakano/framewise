"""Read models of a sequence: the editor's detail view, built on ``state``.

Everything is assembled from current data on every read: the cuts as they are now, the current bible,
plan, shots and review, and for each shot whether what it was written from has changed since
(``stale``). The download for other apps (contract ``sequence-v2``) is ``sequence_package``.
"""
from . import db, sequence_compile, sequence_evidence as evidence, sequence_prompt, sequence_store as store, sequence_timeline, text_diff

REQUEST_CHARS = 1000
# What the user typed in the Assistant chat turn that made a proposal ``p``.
CHAT_REQUEST = "(SELECT m.content FROM assistant_messages m WHERE m.turn_id=p.turn_id AND m.role='user' ORDER BY m.seq LIMIT 1)"


def state(conn, sequence):
    """Everything known about a sequence right now, without raising for what is missing."""
    cuts = store.current_cuts(conn, sequence)
    current = {'sequence': sequence, 'cuts': cuts, 'changed': store.cuts_changed(sequence, cuts), 'ctx': None, 'timeline': None, 'missing': [],
               'bible_row': store.current(conn, sequence['id'], 'bible'), 'plan_row': store.current(conn, sequence['id'], 'plan'),
               'compilation_row': store.current(conn, sequence['id'], 'compilation'),
               'review_row': store.current(conn, sequence['id'], 'review'), 'shot_rows': store.current_shots(conn, sequence['id']), 'shots': None, 'plan_problems': []}
    current['bible'] = store.loads(current['bible_row']['content'], {}) if current['bible_row'] else None
    current['plan'] = store.loads(current['plan_row']['content'], {}) if current['plan_row'] else None
    if cuts is None:
        return current
    current['ctx'] = evidence.load(conn, sequence, cuts)
    current['missing'] = evidence.missing_cuts(current['ctx'], cuts)
    current['timeline'] = store.timeline_for(conn, sequence, cuts)
    if current['plan'] and current['plan'].get('shots'):
        current['plan_problems'] = sequence_timeline.check_spans(current['plan']['shots'], current['timeline'])
        if not current['plan_problems'] and current['plan'].get('compilation_id') != sequence.get('reviewed_compilation_id'):
            timing = current['timeline'].get('dialogue_timing', {})
            starts = {cut['id']: spans[0]['start_time'] for cut, spans in sequence_timeline.groups(current['plan']['shots'], current['timeline'])}
            if any(abs(starts[boundary['before_cut_id']] - boundary['options'][timing[boundary['id']]['choice']]['time']) >= 0.002
                   for boundary in current['timeline']['boundaries'] if boundary['id'] in timing):
                current['plan_problems'] = ['The compiled dialogue has new shot starts. Replan to apply them.']

        if not current['plan_problems']:
            current['shots'] = sequence_timeline.expand(current['plan']['shots'], current['timeline'], current['plan'].get('voices'), current['plan'].get('boundaries'))
    return current


def sequence_view(sequence, cuts, changed):
    numbers = [cut['cut_index'] + 1 for cut in cuts] if cuts else [item['cut_index'] + 1 for item in store.loads(sequence['cuts_json'], [])]
    start, end = (cuts[0]['start_time'], cuts[-1]['end_time']) if cuts else (sequence['start_time'], sequence['end_time'])
    return {'id': sequence['id'], 'episode_id': sequence['episode_id'], 'title': sequence['title'], 'instruction': sequence['instruction'],
            'start_time': start, 'end_time': end, 'duration': round(end - start, 3), 'first_cut': numbers[0] if numbers else None, 'last_cut': numbers[-1] if numbers else None,
            'cuts_changed': bool(changed and cuts is not None), 'cuts_missing': cuts is None, 'shot_points': store.shot_points(sequence),
            'extension': store.extension(sequence),
            'created_at': sequence['created_at'], 'updated_at': sequence['updated_at']}


def dialogue_state(conn, sequence, row=None, pending=None):
    """Where the dialogue stands: none (never compiled), running, review (waiting for the user) or confirmed."""
    row = row if row is not None else store.current(conn, sequence['id'], 'compilation')
    pending = pending if pending is not None else store.pending_jobs(conn, sequence['id'])
    if any(job['step'] == 'compile' for job in pending):
        return 'running'
    if not row:
        return 'none'
    return 'confirmed' if sequence.get('reviewed_compilation_id') == row['id'] else 'review'


def compilation_view(conn, sequence, current, pending):
    """The dialogue compilation with what it was built from, for the Dialogue tab."""
    row, cuts = current['compilation_row'], current['cuts']
    view = {'state': dialogue_state(conn, sequence, row, pending), 'result': result_view(row), 'confirmed': False, 'stale': False, 'source': [], 'subtitles': [], 'has_speech': False}
    view['confirmed'] = view['state'] == 'confirmed'
    if cuts is None:
        return view
    origin = cuts[0]['start_time']
    hash_now, fragments, subtitle_source = store.source_fingerprint(conn, sequence, cuts)
    saved = conn.execute('SELECT source_hash FROM sequence_compilation_sources WHERE sequence_id=?', (sequence['id'],)).fetchone()
    view['stale'] = bool(row and saved and saved['source_hash'] != hash_now)
    view['has_speech'] = bool(fragments)
    view['subtitles'] = [{'id': item['id'], 't0': round(item['start_time'] - origin, 3), 't1': round(item['end_time'] - origin, 3), 'text': item['text']} for item in subtitle_source]
    view['source'] = [{'id': line['id'], 't0': round(line['start_time'] - origin, 3), 't1': round(line['end_time'] - origin, 3), 'text': line['text'], 'speaker_label': line['speaker_label'] or ''}
                      for line in store.transcript_lines(conn, sequence['episode_id'], cuts[0]['start_time'], cuts[-1]['end_time'])]
    if row and not current['changed'] and not (current['shots'] and current['plan'].get('compilation_id') == row['id']):
        compilation = store.loads(row['content'], {})
        draft = None
        if compilation.get('timing'):
            draft = current['timeline'] if sequence.get('reviewed_compilation_id') == row['id'] else store.build_timeline(conn, sequence, cuts, compilation)
        if draft:
            view['shot_starts'] = [{'after_cut_id': item['after_cut_id'], 'reason': item['reason'], 'by': item['by'],
                                    'time': boundary['options'][item['choice']]['time'], 'speech': bool(boundary['speech'])}
                                   for boundary in draft['boundaries'] if (item := draft['dialogue_timing'].get(boundary['id']))]
    view['characters'] = [{'id': person['id'], 'name': person['name']} for person in current['ctx']['characters']]
    return view


def summary(conn, sequence):
    """One row of the episode's sequence list; cheap enough to compute for every sequence."""
    cuts = store.current_cuts(conn, sequence)
    plan = store.current(conn, sequence['id'], 'plan')
    planned = len(store.loads(plan['content'], {}).get('shots') or []) if plan else 0
    return {**sequence_view(sequence, cuts, store.cuts_changed(sequence, cuts)),
            'has_bible': store.current(conn, sequence['id'], 'bible') is not None, 'has_plan': plan is not None, 'has_review': store.current(conn, sequence['id'], 'review') is not None,
            'shots_planned': planned, 'shots_written': len(store.current_shots(conn, sequence['id'])) if planned else 0,
            'dialogue': dialogue_state(conn, sequence), 'pending_jobs': len(store.pending_jobs(conn, sequence['id']))}


def stale_reasons(current, shot, row):
    """Which inputs of a written shot changed since: its plan entry, the bible, or the analysis of its cuts."""
    if not row or not current['bible']:
        return []
    saved = store.loads(row['settings_json'], {}).get('hashes') or {}
    now = evidence.shot_hashes(current['ctx'], current['bible'], shot)
    return [part for part in ('plan', 'bible', 'evidence') if saved.get(part) != now[part]]


def shot_result(row):
    if not row:
        return None
    settings = store.loads(row['settings_json'], {})
    usage = settings.get('usage') or {}
    return {'id': row['id'], 'content': store.loads(row['content'], {}), 'prompt': row['rendered'], 'check': settings.get('check'), 'derived': settings.get('derived') or {},
            'generation_seconds': settings.get('summary_generation_seconds'), 'model_variant': usage.get('model_variant'), 'decode_tokens_per_second': usage.get('decode_tokens_per_second'),
            'origin': row['origin'], 'model_id': row['model_id'], 'job_id': row['job_id'], 'parent_id': row['parent_id'], 'created_at': row['created_at'],
            'instruction': settings.get('instruction'), 'instruction_source': settings.get('instruction_source'), 'trigger': settings.get('trigger'),
            'reference_frame_source': settings.get('reference_frame_source'), 'reference_picture': settings.get('reference_picture'),
            'completed_by_app': settings.get('completed_by_app'),
            'assistant_summary': (settings.get('assistant_edit') or {}).get('summary')}


def clip(text, limit=REQUEST_CHARS):
    text = (text or '').strip()
    return text[:limit] + ('…' if len(text) > limit else '') if text else None


def shot_history(conn, sequence_id, shot_key):
    """Every version of one shot's prompt, newest first, with why it was written and what changed.

    Each revision is diffed against the version it replaced and says what caused it: the user's
    instruction, a voice change (``trigger``), a hand edit, or an Assistant chat request (``assistant``).
    ``remembered`` are the user's instructions of earlier rewrites, which every later rewrite keeps following.
    ``pending`` is a rewrite that is queued or running; ``proposals`` are the Assistant tab's changes to
    this shot that wait for the user to apply them.
    """
    rows = [dict(row) for row in conn.execute("""SELECT id,parent_id,origin,model_id,rendered,settings_json,created_at,json_extract(content,'$.reference_frame') AS reference
        FROM sequence_results WHERE sequence_id=? AND type='shot' AND shot_key=? ORDER BY rowid""", (sequence_id, shot_key))]
    settings = [store.loads(row['settings_json'], {}) for row in rows]
    edits = [item.get('assistant_edit') if isinstance(item.get('assistant_edit'), dict) else None for item in settings]
    asked = [edit['proposal_id'] for edit in edits if edit and edit.get('proposal_id')]
    requests = {}
    if asked:
        requests = {row['id']: row['request'] for row in conn.execute(
            f"SELECT p.id,{CHAT_REQUEST} AS request FROM assistant_proposals p WHERE p.id IN ({','.join('?' * len(asked))})", asked)}
    by_id = {row['id']: row for row in rows}
    revisions = []
    for number, (row, extra, edit) in enumerate(zip(rows, settings, edits), 1):
        before = by_id.get(row['parent_id']) if row['parent_id'] else (rows[number - 2] if number > 1 else None)
        prompt = row['rendered'] or ''
        revisions.append({
            'id': row['id'], 'number': number, 'origin': row['origin'], 'model_id': row['model_id'], 'created_at': row['created_at'],
            'instruction': extra.get('instruction'), 'instruction_source': extra.get('instruction_source'), 'earlier_instructions': extra.get('earlier_instructions'),
            'trigger': extra.get('trigger'),
            'reference': store.loads(row['reference'], {}) or None, 'reference_source': extra.get('reference_frame_source'),
            'reference_picture': extra.get('reference_picture'),
            'assistant': {'request': clip(requests.get(edit.get('proposal_id'))), 'summary': edit.get('summary') or '', 'provider': edit.get('provider'),
                          'model': edit.get('model'), 'session_id': edit.get('session_id')} if edit else None,
            'model_variant': (extra.get('usage') or {}).get('model_variant'), 'generation_seconds': extra.get('summary_generation_seconds'),
            'check_ok': (extra.get('check') or {}).get('ok'), 'prompt': prompt,
            'diff': text_diff.diff_segments(before['rendered'] or '', prompt) if before else None})
    job = conn.execute("""SELECT status,json_extract(payload_json,'$.instruction') AS instruction,json_extract(payload_json,'$.trigger') AS reason,
        json_extract(payload_json,'$.reference_frame') AS reference FROM jobs
        WHERE pipeline='sequence' AND status IN ('queued','running') AND json_extract(payload_json,'$.sequence_id')=? AND json_extract(payload_json,'$.step')='shot'
        AND json_extract(payload_json,'$.shot_key')=? ORDER BY rowid DESC LIMIT 1""", (sequence_id, shot_key)).fetchone()
    current = rows[-1]['id'] if rows else None
    proposals = [{'id': row['id'], 'created_at': row['created_at'], 'summary': row['summary'], 'request': clip(row['request']), 'provider': row['provider'],
                  'model': row['model_id'], 'stale': row['base_result_id'] != current, 'prompt': row['content'], 'diff': text_diff.diff_segments(row['base_content'] or '', row['content'])}
                 for row in conn.execute(f"""SELECT p.id,p.created_at,p.summary,p.provider,p.model_id,p.base_result_id,p.base_content,p.content,{CHAT_REQUEST} AS request
                     FROM assistant_proposals p WHERE p.sequence_id=? AND p.target='sequence_shot' AND p.status='pending' AND json_extract(p.payload_json,'$.shot_key')=?
                     ORDER BY p.created_at DESC,p.rowid DESC""", (sequence_id, shot_key))]
    return {'revisions': revisions[::-1], 'remembered': store.shot_instructions(conn, sequence_id, shot_key), 'pending': {'status': job['status'], 'instruction': job['instruction'] or None, 'trigger': job['reason'],
                                                                                                             **({'reference': store.loads(job['reference'], {})} if job['reference'] else {})} if job else None,
            'proposals': proposals}


def result_view(row):
    if not row:
        return None
    settings = store.loads(row['settings_json'], {})
    return {'id': row['id'], 'content': store.loads(row['content'], {}), 'origin': row['origin'], 'model_id': row['model_id'], 'job_id': row['job_id'],
            'parent_id': row['parent_id'], 'created_at': row['created_at'], 'completed_by_app': settings.get('completed_by_app'),
            'stale_reason': row.get('stale_reason'),
            'usage': settings.get('usage'), 'generation_seconds': settings.get('summary_generation_seconds')}


def shot_views(current, pending=()):
    """Planned shots with their written direction, staleness and queue state."""
    waiting = {job['shot_key']: job['status'] for job in pending if job.get('step') == 'shot'}
    views = []
    for shot in current['shots'] or []:
        row = current['shot_rows'].get(shot['key'])
        views.append({**shot, 'result': shot_result(row), 'stale': stale_reasons(current, shot, row), 'job': waiting.get(shot['key']),
                      'speakers': sequence_prompt.speakers(evidence.shot_dialogue(shot, current['bible'] or {}))})
    return views


def boundary_views(current):
    """Every hard cut of the sequence with its options and, once there is a usable plan, the choice made there and by whom.

    ``fixed`` is the user's placed boundary (``{t, time, shift}``) or None; a placed boundary is decided ``by`` the user.
    """
    timeline = current['timeline']
    if timeline is None:
        return []
    choices = sequence_timeline.choices_of(current['plan']['shots'], timeline) if current['shots'] else {}
    decided = {item.get('after_cut_id'): item for item in (current['plan'] or {}).get('boundaries') or [] if isinstance(item, dict)}
    views = []
    for boundary in timeline['boundaries']:
        choice = choices.get(boundary['id'])
        if boundary.get('fixed'):
            views.append({**boundary, 'choice': choice, 'reason': None, 'by': 'user'})
            continue
        made = decided.get(boundary['after_cut_id']) or {}
        views.append({**boundary, 'choice': choice, 'reason': made.get('reason') if made.get('choice') == choice else None,
                      'by': made.get('by') if made.get('choice') == choice else None})
    return views


def is_ready(current, shots):
    """True when the download holds a current, format-clean prompt for every planned shot."""
    return bool(shots) and not current['changed'] and all(shot['result'] and (shot['result']['check'] or {}).get('ok') and not shot['stale'] for shot in shots)


def next_cut_view(conn, sequence, cuts):
    """The cut that could be added at the end: whether it fits and is analysed, and the words that run on into it."""
    following = store.next_cut(conn, sequence, cuts) if cuts else None
    if following is None:
        return None
    seconds = following['end_time'] - cuts[0]['start_time']
    return {'id': following['id'], 'number': following['cut_index'] + 1, 'start_time': following['start_time'], 'end_time': following['end_time'],
            'seconds': round(following['end_time'] - following['start_time'], 3), 'fits': seconds <= sequence_timeline.MAX_SEQUENCE_SECONDS + 0.001,
            'ready': not evidence.missing_cuts(evidence.load(conn, sequence, [following]), [following]),
            'speech': store.speech_past_end(conn, sequence, cuts, following)}


def detail(sequence_id):
    with db.connect() as conn:
        sequence = store.get(conn, sequence_id)
        if not sequence:
            return None
        current = state(conn, sequence)
        pending = store.pending_jobs(conn, sequence_id)
        jobs = store.recent_jobs(conn, sequence_id)
        compilation = compilation_view(conn, sequence, current, pending)
        following = next_cut_view(conn, sequence, current['cuts'])
    cuts, timeline = current['cuts'], current['timeline']
    shots = shot_views(current, pending)
    return {'sequence': sequence_view(sequence, cuts, current['changed']),
            'cuts': [{'id': cut['id'], 'number': cut['number'], 't0': cut['t0'], 't1': cut['t1'], 'start_time': cut['start_time'], 'end_time': cut['end_time'],
                      'ready': cut['number'] not in current['missing']} for cut in timeline['cuts']] if timeline else [],
            'missing': current['missing'],
            'timeline': {'duration': timeline['duration'], 'pauses': timeline['pauses'], 'boundaries': boundary_views(current),
                         'lines': [{key: line[key] for key in ('id', 't0', 't1', 'text', 'speaker_label', 'character_name', 'character_context_id')} for line in timeline['lines']]} if timeline else None,
            'compilation': compilation, 'bible': result_view(current['bible_row']), 'plan': result_view(current['plan_row']), 'plan_problems': current['plan_problems'],
            'shots': shots, 'review': result_view(current['review_row']),
            'ready': is_ready(current, shots), 'jobs': jobs, 'next_cut': following,
            'limits': {'sequence_seconds': sequence_timeline.MAX_SEQUENCE_SECONDS, 'shot_seconds': sequence_timeline.MAX_SHOT_SECONDS, 'min_generate_seconds': sequence_timeline.MIN_GENERATE_SECONDS}}
