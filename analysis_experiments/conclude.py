"""The written conclusions: Qwen 3.6 reads a compact digest of each run and says what it means.

``digest`` is pure (numbers only); ``run`` calls any ``chat(system, user) -> (text, seconds, usage)``.
Each answer goes through a validator and, like every other model call here, is retried with the validator's
complaints (up to three attempts).
"""
import json
import re
import time

import pose_schema as ps
import scoring

VERDICTS = ('adopt', 'adopt_with_changes', 'not_ready')
MAX_ATTEMPTS = 3

SYSTEM = ('You are an evaluation analyst. You receive measurements from an experiment in which two local vision models (Qwen3.5-27B for first frames, Qwen3-VL-8B for '
          'video clips) were asked to describe each character\'s head, gaze and body pose as a structured JSON block, using one of three prompt styles. There is no '
          'hand-labelled truth. Quality numbers come from consistency checks: mirror (a left-right flipped frame must give the mirrored answer), geometry (answers must '
          'fit the character boxes and themselves) and agreement (answers must not contradict the earlier free-text analysis of the same cut). Say "consistent" or '
          '"inconsistent", never "accurate" or "correct". Use only the figures in the evidence, quote the important ones, and never invent a number. A field that the '
          'model abstained from (unclear / null) is honest, not a failure. Write plain English for a developer deciding whether to add this block to the pipeline.')

APPROACH_NOTES = {
    'full_confidence': 'A1: every field mandatory, short value lists, free angles, a 0.0-1.0 confidence per field.',
    'top5_confidence': 'A2: the model returns only the 5 fields it is surest about per character, with confidence, best first.',
    'strict_detailed': 'A3: every field mandatory, no confidence, a detailed prompt, angles on a 45-degree grid, nine gaze sectors.',
}


def _round(value, digits=2):
    return None if value is None else round(value, digits)


def _stage_digest(summary):
    calls = max(summary['calls'], 1)
    return {'calls': summary['calls'], 'first_try_rate': _round(summary['first_try_rate']), 'final_rate': _round(summary['final_rate']),
            'validator_failures': summary['validator_failures'], 'retries': summary['retries'], 'hard_failures': summary['hard_failures'],
            'seconds': _round(summary['seconds'], 0), 'seconds_per_call': _round(summary['seconds'] / max(summary['main_calls'], 1), 1),
            'completion_tokens_per_call': _round(summary['completion_tokens'] / calls, 0), 'coverage': _round(summary['coverage']), 'quality': _round(summary['quality']),
            'components': {name: {'rate': _round(rate), **summary['component_counts'][name]} for name, rate in summary['components'].items()},
            'calibration': {k: _round(v) for k, v in summary['calibration'].items()}}


def _field_table(summary):
    return {name: {key: _round(value) for key, value in row.items() if key in ('coverage', 'mirror', 'geometry', 'agreement', 'mean_conf')} for name, row in summary['fields'].items()}


def sample_ranking(samples, calls, approach):
    """Samples ordered by the pass rate of their checks (frame and video together) with the failing details."""
    by_id = {sample['id']: sample for sample in samples}
    rows = []
    for call in calls:
        if call['approach'] != approach or call.get('variant', 'normal') != 'normal':
            continue
        row = next((item for item in rows if item['sample'] == call['sample']), None)
        if row is None:
            sample = by_id.get(call['sample'], {})
            row = {'sample': call['sample'], 'reason': sample.get('reason'), 'passed': 0, 'total': 0, 'failures': []}
            rows.append(row)
        for check in call.get('checks', []):
            row['total'] += 1
            row['passed'] += check['status'] == 'pass'
            if check['status'] == 'fail' and len(row['failures']) < 3:
                row['failures'].append(f'{check["kind"]}/{check["name"]}: {check["detail"]}')
    rows = [row for row in rows if row['total']]
    rows.sort(key=lambda row: (row['passed'] / row['total'], -row['total']))
    return rows


def digest(approach, scored, samples, calls):
    block = scored['approaches'][approach]
    frame, video = block['summary']['frame'], block['summary']['video']
    baseline = [sample['stored']['baseline'] for sample in samples]
    frame_base = [b['frame_seconds'] for b in baseline if b.get('frame_seconds')]
    video_base = [b['video_seconds'] for b in baseline if b.get('video_seconds')]
    ranking = sample_ranking(samples, calls, approach)
    return {'approach': approach, 'description': APPROACH_NOTES[approach], 'final_score': _round(block['final'], 1),
            'stage_scores': {stage: {'score': _round(item['score'], 1), 'parts': {k: _round(v) for k, v in item['parts'].items()}} for stage, item in block['stage_scores'].items() if item},
            'weights': scored['weights'], 'frame_stage': _stage_digest(frame), 'video_stage': _stage_digest(video),
            'mirror_probe_seconds': _round(frame['probe_seconds'], 0),
            'stored_baseline_seconds_per_cut': {'first_frame': _round(sum(frame_base) / len(frame_base), 1) if frame_base else None, 'video': _round(sum(video_base) / len(video_base), 1) if video_base else None},
            'frame_fields': _field_table(frame), 'video_fields': _field_table(video), 'top5_field_choices': block['a2_choices'],
            'agreement_with_other_approaches': {other: {'rate': _round(item['rate']), 'n': item['n']} for other, item in scored['agreement_between'].get(approach, {}).items()},
            'best_samples': [{k: row[k] for k in ('sample', 'reason', 'passed', 'total')} for row in ranking[-3:][::-1]],
            'worst_samples': [{k: row[k] for k in ('sample', 'reason', 'passed', 'total', 'failures')} for row in ranking[:3]]}


RUN_RULES = ('Return one JSON object with exactly these keys: verdict (one of adopt, adopt_with_changes, not_ready), headline (one sentence, at most 200 characters), '
             'strengths (2 to 5 short strings), weaknesses (2 to 5 short strings), reliable_fields (field names whose numbers look dependable), unreliable_fields '
             '(field names whose numbers look weak), recommendation (2 to 4 sentences: what to do with this prompt style). Field names must come from: ' + ', '.join(ps.FIELD_ORDER) + '.')
OVERALL_RULES = ('Return one JSON object with exactly these keys: best_approach (one of ' + ', '.join(ps.APPROACHES) + '), headline (one sentence, at most 200 characters), '
                 'reasoning (3 to 6 sentences comparing the three runs with figures), fields_to_add (field names worth adding to the analysis), fields_to_avoid (field names '
                 'not worth adding yet), next_steps (2 to 5 short strings), caveats (1 to 4 short strings). Field names must come from: ' + ', '.join(ps.FIELD_ORDER) + '.')


def _strings(value, key, low, high, limit=300):
    if not isinstance(value, list) or not low <= len(value) <= high or not all(isinstance(item, str) and item.strip() and len(item) <= limit for item in value):
        return [f'{key} must be a list of {low} to {high} non-empty strings of at most {limit} characters']
    return []


def _fields(value, key):
    if not isinstance(value, list) or any(item not in ps.FIELD_ORDER for item in value):
        return [f'{key} must be a list of field names from: {", ".join(ps.FIELD_ORDER)}']
    return []


def _text(value, key, limit):
    return [] if isinstance(value, str) and value.strip() and len(value) <= limit else [f'{key} must be a non-empty string of at most {limit} characters']


def validate_run(value):
    if not isinstance(value, dict):
        return ['the answer must be one JSON object']
    errors = [f'unexpected key {key}' for key in value if key not in ('verdict', 'headline', 'strengths', 'weaknesses', 'reliable_fields', 'unreliable_fields', 'recommendation')]
    if value.get('verdict') not in VERDICTS:
        errors.append('verdict must be one of ' + ', '.join(VERDICTS))
    errors += _text(value.get('headline'), 'headline', 200) + _text(value.get('recommendation'), 'recommendation', 700)
    errors += _strings(value.get('strengths'), 'strengths', 1, 6) + _strings(value.get('weaknesses'), 'weaknesses', 1, 6)
    errors += _fields(value.get('reliable_fields'), 'reliable_fields') + _fields(value.get('unreliable_fields'), 'unreliable_fields')
    return errors


def validate_overall(value):
    if not isinstance(value, dict):
        return ['the answer must be one JSON object']
    errors = [f'unexpected key {key}' for key in value if key not in ('best_approach', 'headline', 'reasoning', 'fields_to_add', 'fields_to_avoid', 'next_steps', 'caveats')]
    if value.get('best_approach') not in ps.APPROACHES:
        errors.append('best_approach must be one of ' + ', '.join(ps.APPROACHES))
    errors += _text(value.get('headline'), 'headline', 200) + _text(value.get('reasoning'), 'reasoning', 1400)
    errors += _fields(value.get('fields_to_add'), 'fields_to_add') + _fields(value.get('fields_to_avoid'), 'fields_to_avoid')
    errors += _strings(value.get('next_steps'), 'next_steps', 1, 6) + _strings(value.get('caveats'), 'caveats', 1, 5)
    return errors


def ask(chat, user, validator):
    """One validated JSON answer: ``{'ok', 'value', 'attempts': [{seconds, errors, ...}], 'seconds'}``."""
    attempts, value, messages_user = [], None, user
    for number in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        text, seconds, usage = chat(SYSTEM, messages_user)
        errors = []
        text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)  # a server that leaves the reasoning inside the content
        try:
            value = ps.parse_json(text)
            errors = validator(value)
        except ValueError as exc:
            errors = [str(exc)]
        attempts.append({'n': number, 'seconds': round(seconds or time.monotonic() - started, 1), 'errors': errors, 'completion_tokens': (usage or {}).get('completion_tokens')})
        if not errors:
            return {'ok': True, 'value': value, 'attempts': attempts, 'seconds': round(sum(a['seconds'] for a in attempts), 1)}
        messages_user = user + '\n\nYour previous answer was rejected:\n' + ps.format_errors(errors) + '\nReturn the complete corrected JSON object.'
    return {'ok': False, 'value': None, 'attempts': attempts, 'seconds': round(sum(a['seconds'] for a in attempts), 1), 'error': '; '.join(attempts[-1]['errors'][:3])}


def run(chat, scored, samples, calls, previous=None, on_done=None):
    """Four conclusions: one per approach and an overall comparison; ``previous`` results that already succeeded are kept."""
    previous = previous or {}
    result = {'runs': dict(previous.get('runs', {})), 'overall': previous.get('overall')}
    digests = {}
    for approach in ps.APPROACHES:
        digests[approach] = digest(approach, scored, samples, calls)
        if result['runs'].get(approach, {}).get('ok'):
            continue
        user = f'Evidence for {ps.APPROACH_LABELS[approach]}:\n' + json.dumps(digests[approach], indent=1) + '\n\n' + RUN_RULES
        result['runs'][approach] = ask(chat, user, validate_run)
        if on_done:
            on_done(result)
    if not (result['overall'] or {}).get('ok'):
        brief = {approach: {key: digests[approach][key] for key in ('final_score', 'stage_scores', 'frame_stage', 'video_stage', 'agreement_with_other_approaches')} for approach in ps.APPROACHES}
        verdicts = {approach: (result['runs'][approach].get('value') or {}).get('verdict') for approach in ps.APPROACHES}
        user = 'Evidence for all three runs:\n' + json.dumps({'runs': brief, 'per_run_verdicts': verdicts}, indent=1) + '\n\n' + OVERALL_RULES
        result['overall'] = ask(chat, user, validate_overall)
        if on_done:
            on_done(result)
    return result
