"""Queue Cut summaries and assemble the evidence used by the text worker."""
import json
import re

from . import analysis_context, db, frame_source, job_inputs, prompt_context, worker

SUMMARY_TITLE = 'Cut summary'
ANIMATION_TITLE = 'MiniMax animation prompt'


def choices(cut_id, config, mode='regenerate', instruction='', edited_prompt=None, request_override=None):
    return {'production_brief': True, 'result_type': 'summary', 'cut_id': cut_id, 'mode': mode,
            'instruction': instruction.strip(), 'edited_prompt': edited_prompt,
            'model': config['openrouter_model'], 'text_provider': config['text_provider'], 'local_thinking': config['local_thinking'],
            'context_ids': config.get('context_ids', []), 'additional_context': config.get('additional_context', ''),
            **({'request_override': request_override} if request_override is not None else {})}


def current(conn, cut_id, title=SUMMARY_TITLE):
    return db.one("""SELECT id,content,origin FROM analysis_results WHERE cut_id=? AND title=?
        AND superseded_by IS NULL ORDER BY rowid DESC LIMIT 1""", (cut_id, title), conn=conn)


def insert(conn, episode_id, request, *, parents=(), batch=False, retry_of=None):
    from . import performance_beats
    cut_id = request['cut_id']
    pipelines = ('prepare', 'hardcuts') if request.get('request_override') is not None else (
        'prepare', 'hardcuts', 'analyze', 'character_detect', 'performance_beats', 'review_answers')
    waits = worker.pending_jobs(conn, episode_id, pipelines, [cut_id])
    if request.get('request_override') is None:
        beats = performance_beats.after_visual(conn, episode_id, cut_id, {
            **request, 'openrouter_model': request['model'],
        }, waits)
        if beats:
            waits.append(beats)
    return worker.insert_job(conn, episode_id, 'regen', {**request, 'depends_on': list(dict.fromkeys([*parents, *waits]))},
                             priority_class='batch' if batch else 'interactive', retry_of=retry_of)


def _frame_evidence(conn, cut, rows):
    samples = []
    for row in rows:
        match = re.fullmatch(r'(?:Frame|Object detections) at (\d+(?:\.\d+)?)s', row['title'] or '')
        if not match or not frame_source.result_frame_spec(row['id'], conn):
            continue
        timestamp = float(match.group(1))
        try:
            source = json.loads(row['context_json'] or '{}')
            if isinstance(source.get('frame_time'), (int, float)):
                timestamp = float(source['frame_time'])
            observation = next((item for item in source.get('sampled_frames', []) if item.get('frame_id') == row['id']), None)
            if observation:
                timestamp = float(observation['time'])
        except (ValueError, TypeError, AttributeError):
            pass
        if cut['start_time'] <= timestamp < cut['end_time']:
            samples.append({'frame_id': row['id'], 'time': timestamp, 'description': row['content']})
    return samples


def gather(conn, job, *, preview=False):
    payload = json.loads(job['payload_json'])
    cut_id = payload['cut_id']
    cut = db.one('SELECT * FROM cuts WHERE id=?', (cut_id,), conn=conn)
    if not cut or (not preview and not cut['active']):
        return {'skip': 'Cut archived or removed'}
    if not preview and not cut['analysis_enabled']:
        return {'skip': 'Analysis disabled for this cut'}
    legacy = payload.get('result_type') != 'summary'
    title = ANIMATION_TITLE if legacy else SUMMARY_TITLE
    base = current(conn, cut_id, title)
    edited_prompt = payload.get('edited_prompt')
    if 'edited_prompt' not in payload and not payload.get('production_brief'):
        edited_prompt = payload.get('current_prompt') or None
    if payload.get('mode') == 'edit' and not base and not (edited_prompt or '').strip() and not payload.get('request_override'):
        return {'skip': 'No saved result to edit'}
    text = edited_prompt if edited_prompt is not None else (base['content'] if base else '')
    if payload.get('mode') != 'edit':
        text = ''
    old_context = payload.get('context') or {}
    config = {'context_ids': payload.get('context_ids', [item['id'] for item in old_context.get('context_items') or [] if item.get('id')]),
              'additional_context': payload.get('additional_context', old_context.get('additional_context', ''))}
    brief = bool(payload.get('production_brief')) and not legacy
    values = {'base': {name: base[name] for name in ('id', 'origin')} if base else {}, 'current_prompt': text,
              'frame_time': cut['first_frame_time'],
              'results': [], 'sampled_frames': [], 'source_result_ids': [], 'human_instructions': ''}
    if payload.get('request_override') is not None:
        # An explicitly edited request already contains exactly the evidence chosen by the user.
        values['context'] = {'cut': {'start_time': cut['start_time'], 'end_time': cut['end_time']}}
        return values
    values['context'] = prompt_context.slim_context(analysis_context.selected(conn, cut_id, config), dialogue=not brief)
    rows = db.all_rows("""SELECT id,kind,title,content,origin,context_json FROM analysis_results
        WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid""", (cut_id,), conn=conn)
    evidence = [row for row in rows if row['title'] != title]
    drop = (prompt_context.CONSISTENCY_TITLE,) if brief else ()
    values['results'] = prompt_context.slim_results(evidence, drop=drop)
    included = {row.get('title') for row in values['results']}
    values['source_result_ids'] = [row['id'] for row in evidence if row['title'] in included]
    values['sampled_frames'] = _frame_evidence(conn, cut, rows) if not legacy else []
    saved = db.one('SELECT content FROM cut_brief_instructions WHERE cut_id=?', (cut_id,), conn=conn)
    values['human_instructions'] = saved['content'] if saved else ''
    return values


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    from .structured_results import invalidate_sequence_reviews
    cut_id = json.loads(job['payload_json'])['cut_id']
    invalidate_sequence_reviews(conn, [cut_id], 'Cut summary changed')
