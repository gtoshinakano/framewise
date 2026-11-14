"""Read models of a sequence: the editor's detail view and the download package.

Both are assembled from current data on every read: the cuts as they are now, the current bible,
plan, shots and review, and for each shot whether what it was written from has changed since
(``stale``). ``package`` is the contract ``sequence-v1`` that the video generation app downloads.
"""
from . import db, sequence_evidence as evidence, sequence_prompt, sequence_store as store, sequence_timeline

CONTRACT = 'sequence-v1'


def state(conn, sequence):
    """Everything known about a sequence right now, without raising for what is missing."""
    cuts = store.current_cuts(conn, sequence)
    current = {'sequence': sequence, 'cuts': cuts, 'changed': store.cuts_changed(sequence, cuts), 'ctx': None, 'timeline': None, 'missing': [],
               'bible_row': store.current(conn, sequence['id'], 'bible'), 'plan_row': store.current(conn, sequence['id'], 'plan'),
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
        if not current['plan_problems']:
            current['shots'] = sequence_timeline.expand(current['plan']['shots'], current['timeline'])
    return current


def sequence_view(sequence, cuts, changed):
    numbers = [cut['cut_index'] + 1 for cut in cuts] if cuts else [item['cut_index'] + 1 for item in store.loads(sequence['cuts_json'], [])]
    start, end = (cuts[0]['start_time'], cuts[-1]['end_time']) if cuts else (sequence['start_time'], sequence['end_time'])
    return {'id': sequence['id'], 'episode_id': sequence['episode_id'], 'title': sequence['title'], 'instruction': sequence['instruction'],
            'start_time': start, 'end_time': end, 'duration': round(end - start, 3), 'first_cut': numbers[0] if numbers else None, 'last_cut': numbers[-1] if numbers else None,
            'cuts_changed': bool(changed and cuts is not None), 'cuts_missing': cuts is None, 'created_at': sequence['created_at'], 'updated_at': sequence['updated_at']}


def summary(conn, sequence):
    """One row of the episode's sequence list; cheap enough to compute for every sequence."""
    cuts = store.current_cuts(conn, sequence)
    plan = store.current(conn, sequence['id'], 'plan')
    planned = len(store.loads(plan['content'], {}).get('shots') or []) if plan else 0
    return {**sequence_view(sequence, cuts, store.cuts_changed(sequence, cuts)),
            'has_bible': store.current(conn, sequence['id'], 'bible') is not None, 'has_plan': plan is not None, 'has_review': store.current(conn, sequence['id'], 'review') is not None,
            'shots_planned': planned, 'shots_written': len(store.current_shots(conn, sequence['id'])) if planned else 0,
            'pending_jobs': len(store.pending_jobs(conn, sequence['id']))}


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
    return {'id': row['id'], 'content': store.loads(row['content'], {}), 'prompt': row['rendered'], 'check': settings.get('check'), 'derived': settings.get('derived') or {},
            'origin': row['origin'], 'model_id': row['model_id'], 'job_id': row['job_id'], 'parent_id': row['parent_id'], 'created_at': row['created_at'],
            'instruction': settings.get('instruction'), 'completed_by_app': settings.get('completed_by_app')}


def result_view(row):
    if not row:
        return None
    settings = store.loads(row['settings_json'], {})
    return {'id': row['id'], 'content': store.loads(row['content'], {}), 'origin': row['origin'], 'model_id': row['model_id'], 'job_id': row['job_id'],
            'parent_id': row['parent_id'], 'created_at': row['created_at'], 'completed_by_app': settings.get('completed_by_app'),
            'usage': settings.get('usage'), 'generation_seconds': settings.get('summary_generation_seconds')}


def shot_views(current, pending=()):
    """Planned shots with their written direction, staleness and queue state."""
    waiting = {job['shot_key']: job['status'] for job in pending if job.get('step') == 'shot'}
    views = []
    for shot in current['shots'] or []:
        row = current['shot_rows'].get(shot['key'])
        views.append({**shot, 'result': shot_result(row), 'stale': stale_reasons(current, shot, row), 'job': waiting.get(shot['key'])})
    return views


def is_ready(current, shots):
    """True when the download holds a current, format-clean prompt for every planned shot."""
    return bool(shots) and not current['changed'] and all(shot['result'] and (shot['result']['check'] or {}).get('ok') and not shot['stale'] for shot in shots)


def detail(sequence_id):
    with db.connect() as conn:
        sequence = store.get(conn, sequence_id)
        if not sequence:
            return None
        current = state(conn, sequence)
        pending = store.pending_jobs(conn, sequence_id)
        jobs = store.recent_jobs(conn, sequence_id)
    cuts, timeline = current['cuts'], current['timeline']
    shots = shot_views(current, pending)
    return {'sequence': sequence_view(sequence, cuts, current['changed']),
            'cuts': [{'id': cut['id'], 'number': cut['number'], 't0': cut['t0'], 't1': cut['t1'], 'start_time': cut['start_time'], 'end_time': cut['end_time'],
                      'ready': cut['number'] not in current['missing']} for cut in timeline['cuts']] if timeline else [],
            'missing': current['missing'],
            'timeline': {'duration': timeline['duration'], 'pauses': timeline['pauses'], 'candidates': timeline['candidates'],
                         'lines': [{key: line[key] for key in ('id', 't0', 't1', 'text', 'speaker_label', 'character_name', 'character_context_id')} for line in timeline['lines']]} if timeline else None,
            'bible': result_view(current['bible_row']), 'plan': result_view(current['plan_row']), 'plan_problems': current['plan_problems'],
            'shots': shots, 'review': result_view(current['review_row']),
            'ready': is_ready(current, shots), 'jobs': jobs,
            'limits': {'sequence_seconds': sequence_timeline.MAX_SEQUENCE_SECONDS, 'shot_seconds': sequence_timeline.MAX_SHOT_SECONDS, 'min_generate_seconds': sequence_timeline.MIN_GENERATE_SECONDS}}


def package(sequence_id):
    """The sequence as the video generation app downloads it (contract ``sequence-v1``), or None."""
    with db.connect() as conn:
        sequence = store.get(conn, sequence_id)
        if not sequence:
            return None
        current = state(conn, sequence)
    cuts, timeline, bible = current['cuts'], current['timeline'], current['bible'] or {}
    base = f'/api/v1/sequences/{sequence_id}'
    views, shots = shot_views(current), []
    for shot in views:
        result = shot['result'] or {}
        content, derived = result.get('content') or {}, result.get('derived') or {}
        dialogue = derived.get('dialogue')
        if dialogue is None:  # not written yet: the lines with their speakers, without a delivery decision
            dialogue = [{key: line.get(key) for key in ('line', 't0', 't1', 'text', 'speaker_label', 'character_context_id', 'character_name', 'attribution', 'crosses_start', 'crosses_end')}
                        for line in evidence.shot_dialogue(shot, bible)]
        shots.append({
            'index': shot['index'], 'key': shot['key'], 'start_time': shot['start_time'], 'end_time': shot['end_time'], 't0': shot['t0'], 't1': shot['t1'],
            'duration': shot['duration'], 'generate_seconds': shot['generate_seconds'], 'trim_seconds': shot['trim_seconds'],
            'source_cuts': shot['source_cuts'], 'cut_ids': shot['cut_ids'], 'merged': shot['merged'], 'starts_mid_cut': shot['starts_mid_cut'], 'ends_mid_cut': shot['ends_mid_cut'],
            'scene': shot['scene'], 'intent': shot['intent'], 'staging': shot['staging'],
            'initial_frame': {'cut_id': shot['frame']['cut_id'], 'cut_number': shot['frame']['cut'], 'time': shot['frame']['time'], 'strategy': shot['frame']['strategy'],
                              'continues_previous_shot': shot['frame']['continues_previous_shot'], 'description': content.get('picture_line'), 'still': content.get('still'),
                              'still_prompt': derived.get('still_prompt'), 'frame_url': f"{base}/shots/{shot['index']}/frame", 'file': f"shot-{shot['index']:02d}-initial.jpg"},
            'audio': {'start_time': shot['start_time'], 'end_time': shot['end_time'], 'url': f"{base}/shots/{shot['index']}/audio", 'file': sequence_prompt.audio_file(shot), 'has_speech': bool(dialogue)},
            'dialogue': dialogue,
            'subjects': [{key: subject.get(key) for key in ('number', 'label', 'name', 'kind', 'character_context_id', 'context_id', 'picture', 'anchors')} for subject in derived.get('subjects') or []],
            'references': derived.get('references') or [],
            'camera': content.get('camera'), 'beats': sequence_prompt.beat_ranges(content, shot) if content.get('beats') else [],
            'soundscape': content.get('soundscape'), 'continuity': content.get('continuity'), 'open_questions': content.get('open_questions') or [],
            'prompt': result.get('prompt'), 'check': result.get('check'), 'written': bool(result), 'stale': shot['stale'],
            'result': {key: result.get(key) for key in ('id', 'origin', 'model_id', 'created_at')} if result else None})
    review = store.loads(current['review_row']['content'], {}) if current['review_row'] else None
    return {'contract': CONTRACT, **sequence_view(sequence, cuts, current['changed']),
            'cuts': [{'id': cut['id'], 'cut_number': cut['number'], 'start_time': cut['start_time'], 'end_time': cut['end_time']} for cut in timeline['cuts']] if timeline else [],
            'audio_url': f'{base}/audio', 'ready': is_ready(current, views),
            'pending': [shot['index'] for shot in views if not shot['result'] or shot['stale'] or not (shot['result']['check'] or {}).get('ok')],
            'plan_problems': current['plan_problems'],
            'bible': bible or None, 'shots': shots, 'review': review}
