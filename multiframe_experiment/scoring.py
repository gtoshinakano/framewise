"""Pure scoring (stdlib only): every config and the stored baseline, from samples, answer records and judge records.

Components (each 0-1; a missing component is left out and the weights renormalised):
* precision   judge-verified claims: true / (true + false), weighted by category importance (outfit/colour/lighting 0.3)
* recall      per pair and category, the judge's reference flags the answer also reports
* camera      agreement of camera.move with the pixel estimate (pairs where the estimate is not ``unclear``)
* speaking    speaking=yes against transcript speech in the pair's interval (precision and on-screen recall)
* reliability half valid on the first attempt, half valid after retries
* efficiency  cheapest compute seconds per second of footage / this config's
"""
import re

import infer
import schema

WEIGHTS = {'precision': 35, 'recall': 20, 'camera': 10, 'speaking': 10, 'reliability': 10, 'efficiency': 15}
GENERIC = re.compile(r'\b(unknown|person|people|someone|somebody|figure|character|man|men|woman|women|girl|boy|child|kid|student|guy|lady|crowd|background|'
                     r'passer|pedestrian|stranger|other|second|first|left|right|older|younger|teen|adult|baby|mother|father)\b', re.I)


def move_class(move):
    return {'zoom_in': 'in', 'dolly_in': 'in', 'zoom_out': 'out', 'dolly_out': 'out', 'truck': 'pan', 'shake': 'static'}.get(move, move)


def camera_agreement(model_move, cv_move):
    """1 exact class, 0.5 same family (pan_left vs truck...), 0 otherwise; None when either side is unclear."""
    if model_move in (None, 'unclear') or cv_move in (None, 'unclear'):
        return None
    a, b = move_class(model_move), move_class(cv_move)
    if a == b:
        return 1.0
    family = lambda m: m.split('_')[0]  # noqa: E731
    if a != 'static' and b != 'static' and family(a) == family(b):
        return 0.5
    return 0.0


def name_ok(name, roster):
    low = name.lower()
    for person in roster:
        other = person['name'].lower()
        if low == other or low in other or other in low or any(part in low.split() for part in other.split() if len(part) > 2):
            return True
    return bool(GENERIC.search(name))


def _speech(sample, start, end):
    return [line for line in sample.get('transcript') or [] if line['start'] < end and line['end'] > start]


def _ratio(num, den):
    return num / den if den else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_rows(samples, answers, judge_records, labels=None):
    by_sample = {s['id']: s for s in samples}
    references = {(r['sample'], r['window']): r['pairs'] for r in judge_records.values() if r.get('kind') == 'reference' and r.get('ok')}
    verdicts = {}
    for record in judge_records.values():
        if record.get('kind') == 'verify' and record.get('ok'):
            for text_key, verdict in record['verdicts'].items():
                verdicts[(record['sample'], record['window'], text_key)] = verdict
    from judge import claim_text_key

    # Gather (config -> list of (sample, window, normalised answer, record or None)).
    grouped = {}
    for record in answers.values():
        grouped.setdefault(record['config'], []).append(record)
    for sample in samples:
        view = (sample.get('baseline') or {}).get('view')
        if not view or len(sample['frames']) < 2:
            continue
        for window in sample['windows']:
            grouped.setdefault(infer.BASELINE_KEY, []).append({'config': infer.BASELINE_KEY, 'sample': sample['id'], 'window': window['index'], 'ok': True,
                                                               'first_try_ok': True, 'attempts': [], 'seconds': None, 'answer': schema.baseline_answer(view, window),
                                                               'baseline': True})
    rows = {}
    for config, records in grouped.items():
        baseline = config == infer.BASELINE_KEY
        model, style, context = config.split('|')
        cat = {c: {'true': 0, 'false': 0, 'not_visible': 0, 'unjudged': 0, 'ref': 0, 'hit': 0} for c in schema.CATEGORIES}
        w_true = w_false = 0.0
        cam_scores, cam_abstain = [], 0
        speak_yes = speak_yes_ok = speak_onscreen = speak_onscreen_hit = 0
        names_total = names_bad = 0
        bad_names = set()
        still_claims = still_pairs = missed_boxes = moving_boxes = 0
        gt = {'tp': 0, 'fp': 0, 'fn': 0}
        cut_seconds = {}
        valid = first = attempts = 0
        prompt_tokens, completion_tokens, decode_rates = [], [], []
        for record in records:
            sample = by_sample.get(record['sample'])
            if not sample or record['window'] >= len(sample['windows']):
                continue
            window = sample['windows'][record['window']]
            if not baseline:
                attempts += len(record['attempts'])
                valid += bool(record['ok'])
                first += bool(record['first_try_ok'])
                cut_seconds[sample['id']] = cut_seconds.get(sample['id'], 0) + (record['seconds'] or 0)
                for attempt in record['attempts']:
                    if attempt.get('prompt_tokens'):
                        prompt_tokens.append(attempt['prompt_tokens'])
                    if attempt.get('completion_tokens'):
                        completion_tokens.append(attempt['completion_tokens'])
                    if attempt.get('predicted_per_second'):
                        decode_rates.append(attempt['predicted_per_second'])
            if not record['ok']:
                continue
            answer = record['answer']
            pairs = {(p['from'], p['to']): p for p in sample.get('pairs') or []}
            reference = references.get((sample['id'], window['index']))
            ref_pairs = {(p['from'], p['to']): p for p in reference or []}
            label_pairs = {(p['from'], p['to']): p for p in ((labels or {}).get(sample['id']) or {}).get('pairs', [])}
            for claim in schema.answer_claims(answer, window, config, sample['id']):
                if not claim['judge']:
                    continue
                verdict = verdicts.get((sample['id'], window['index'], claim_text_key(claim)))
                bucket = cat[claim['category']]
                if verdict is None:
                    bucket['unjudged'] += 1
                    continue
                bucket[verdict] += 1
                weight = schema.importance(claim['category'])
                if verdict == 'true':
                    w_true += weight
                elif verdict == 'false':
                    w_false += weight
            for segment in answer['segments']:
                key = (segment['from'], segment['to'])
                measured = pairs.get(key)
                found = schema.segment_categories(segment)
                if measured:
                    score = camera_agreement(segment['camera']['move'], measured['camera'].get('move'))
                    if score is None:
                        cam_abstain += segment['camera']['move'] == 'unclear'
                    else:
                        cam_scores.append(score)
                    claims_here = sum(len(p['changes']) + sum(p[f] == 'yes' for f in ('walking', 'speaking', 'blinking')) for p in segment['characters'])
                    if measured['still']:
                        still_pairs += 1
                        still_claims += claims_here
                    for value in measured['boxes'].values():
                        if value >= 0.25:
                            moving_boxes += 1
                            missed_boxes += claims_here == 0
                ref = ref_pairs.get(key)
                if ref is not None:
                    for category in ref['categories']:
                        cat[category]['ref'] += 1
                        cat[category]['hit'] += category in found
                truth = label_pairs.get(key)
                if truth:
                    flags = {c for c, v in (truth.get('flags') or {}).items() if v}
                    gt['tp'] += len(flags & found)
                    gt['fp'] += len(found - flags)
                    gt['fn'] += len(flags - found)
                speech = _speech(sample, segment['start'], segment['end'])
                says = any(p['speaking'] == 'yes' for p in segment['characters'])
                if says:
                    speak_yes += 1
                    speak_yes_ok += bool(speech)
                roster_names = {p['name'] for p in sample.get('roster') or []}
                if any(line.get('character') in roster_names for line in speech):
                    speak_onscreen += 1
                    speak_onscreen_hit += says
                for person in segment['characters']:
                    names_total += 1
                    if not name_ok(person['name'], sample.get('roster') or []):
                        names_bad += 1
                        bad_names.add(person['name'])
        total_true = sum(c['true'] for c in cat.values())
        total_false = sum(c['false'] for c in cat.values())
        total_nv = sum(c['not_visible'] for c in cat.values())
        total_ref = sum(c['ref'] for c in cat.values())
        total_hit = sum(c['hit'] for c in cat.values())
        windows = len(records)
        footage = sum(by_sample[s]['duration'] for s in cut_seconds if s in by_sample)
        compute = sum(cut_seconds.values())
        if baseline:
            stage = [(by_sample[s]['baseline'] or {}).get('seconds') for s in {r['sample'] for r in records} if s in by_sample]
            stage = [v for v in stage if v]
            footage_b = sum(by_sample[s]['duration'] for s in {r['sample'] for r in records} if s in by_sample and (by_sample[s]['baseline'] or {}).get('seconds'))
            compute, footage = (sum(stage), footage_b) if stage else (None, None)
        speaking_parts = [_ratio(speak_yes_ok, speak_yes), _ratio(speak_onscreen_hit, speak_onscreen)]
        components = {
            'precision': _ratio(w_true, w_true + w_false),
            'recall': _ratio(total_hit, total_ref),
            'camera': _mean(cam_scores),
            'speaking': _mean(speaking_parts),
            'reliability': None if baseline or not windows else 0.5 * first / windows + 0.5 * valid / windows,
        }
        rows[config] = {
            'config': config, 'model': model, 'style': style, 'context': context, 'baseline': baseline,
            'windows': windows, 'valid': valid if not baseline else windows, 'first_try': first if not baseline else windows, 'attempts': attempts,
            'components': components,
            'claims': {'true': total_true, 'false': total_false, 'not_visible': total_nv, 'unjudged': sum(c['unjudged'] for c in cat.values())},
            'hallucination_rate': _ratio(total_false, total_true + total_false),
            'unverifiable_rate': _ratio(total_nv, total_true + total_false + total_nv),
            'recall_raw': {'hit': total_hit, 'ref': total_ref},
            'categories': {c: {**v, 'precision': _ratio(v['true'], v['true'] + v['false']), 'recall': _ratio(v['hit'], v['ref'])} for c, v in cat.items()},
            'camera': {'scored_pairs': len(cam_scores), 'unclear_answers': cam_abstain},
            'speaking': {'yes_pairs': speak_yes, 'yes_with_speech': speak_yes_ok, 'onscreen_speech_pairs': speak_onscreen, 'onscreen_hits': speak_onscreen_hit,
                         'told': context == 'full'},
            'names': {'entries': names_total, 'invented': names_bad, 'examples': sorted(bad_names)[:12]},
            'pixel': {'still_pairs': still_pairs, 'claims_on_still': still_claims, 'moving_boxes': moving_boxes, 'missed_moving_boxes': missed_boxes},
            'ground_truth': gt if sum(gt.values()) else None,
            'cost': {'compute_seconds': round(compute, 1) if compute is not None else None, 'footage_seconds': round(footage, 2) if footage else None,
                     'per_footage_second': round(compute / footage, 2) if compute and footage else None,
                     'per_window': round(compute / windows, 2) if compute is not None and windows and not baseline else None,
                     'prompt_tokens': round(_mean(prompt_tokens)) if prompt_tokens else None,
                     'completion_tokens': round(_mean(completion_tokens)) if completion_tokens else None,
                     'decode_tps': round(_mean(decode_rates), 1) if decode_rates else None},
        }
    costs = [row['cost']['per_footage_second'] for row in rows.values() if row['cost']['per_footage_second']]
    cheapest = min(costs) if costs else None
    for row in rows.values():
        per = row['cost']['per_footage_second']
        row['components']['efficiency'] = cheapest / per if cheapest and per else None
        available = {k: v for k, v in row['components'].items() if v is not None}
        weight = sum(WEIGHTS[k] for k in available)
        row['score'] = round(100 * sum(WEIGHTS[k] * v for k, v in available.items()) / weight, 1) if weight else None
        row['score_basis'] = sorted(available)
    return rows


def pareto(rows):
    """Configs no cheaper config beats on score (cost = compute seconds per footage second)."""
    items = sorted((r for r in rows.values() if r['score'] is not None and r['cost']['per_footage_second']), key=lambda r: r['cost']['per_footage_second'])
    front, best = [], -1
    for row in items:
        if row['score'] > best:
            front.append(row['config'])
            best = row['score']
    return front


def _avg(rows, key, value):
    groups = {}
    for row in rows:
        if row['baseline'] or row['score'] is None:
            continue
        groups.setdefault(key(row), []).append(value(row))
    return {k: round(sum(v) / len(v), 1) for k, v in groups.items() if v}


def findings(rows, complete):
    """Deterministic sentences for report.md and the page (no model involved)."""
    candidates = [r for r in rows.values() if not r['baseline'] and r['score'] is not None]
    if not candidates:
        return ['No scored config yet.']
    out = []
    best = max(candidates, key=lambda r: r['score'])
    out.append(f"Best overall: {best['config']} with {best['score']} (basis: {', '.join(best['score_basis'])}).")
    for model in sorted({r['model'] for r in candidates}):
        mine = [r for r in candidates if r['model'] == model]
        top = max(mine, key=lambda r: r['score'])
        out.append(f"Best for {model}: {top['style']}/{top['context']} at {top['score']}; {top['cost']['per_window']} s per window.")
    styles = _avg(candidates, lambda r: r['style'], lambda r: r['score'])
    if len(styles) > 1:
        out.append('Mean score by style: ' + ', '.join(f'{k} {v}' for k, v in sorted(styles.items(), key=lambda kv: -kv[1])) + '.')
    contexts = _avg(candidates, lambda r: f"{r['model']}/{r['context']}", lambda r: r['score'])
    if contexts:
        out.append('Mean score by model and context: ' + ', '.join(f'{k} {v}' for k, v in sorted(contexts.items())) + '.')
    halluc = [r for r in candidates if r['hallucination_rate'] is not None]
    if halluc:
        low = min(halluc, key=lambda r: r['hallucination_rate'])
        out.append(f"Fewest false claims: {low['config']} ({low['hallucination_rate'] * 100:.0f}% of judged claims false).")
    rec = [r for r in candidates if r['components']['recall'] is not None]
    if rec:
        top = max(rec, key=lambda r: r['components']['recall'])
        out.append(f"Highest recall of the judge's reference changes: {top['config']} ({top['components']['recall'] * 100:.0f}%).")
    cam = [r for r in candidates if r['components']['camera'] is not None]
    if cam:
        top = max(cam, key=lambda r: r['components']['camera'])
        out.append(f"Best camera agreement with the pixel estimate: {top['config']} ({top['components']['camera'] * 100:.0f}%).")
    speak = [r for r in candidates if r['components']['speaking'] is not None and not r['speaking']['told']]
    if speak:
        top = max(speak, key=lambda r: r['components']['speaking'])
        out.append(f"Best speaking agreement without being given the transcript: {top['config']} ({top['components']['speaking'] * 100:.0f}%).")
    front = pareto(rows)
    if front:
        out.append('Cost/quality frontier (each beats every cheaper config): ' + ' → '.join(front) + '.')
    near = [r for r in candidates if r['score'] >= best['score'] - 5 and r['cost']['per_footage_second']]
    if near:
        cheap = min(near, key=lambda r: r['cost']['per_footage_second'])
        out.append(f"Cheapest within 5 points of the best: {cheap['config']} ({cheap['score']}, {cheap['cost']['per_footage_second']} s compute per footage second).")
    base = rows.get(infer.BASELINE_KEY)
    if base and base['score'] is not None:
        out.append(f"Stored Qwen3-VL video analysis (baseline): {base['score']} on {', '.join(base['score_basis'])}; "
                   f"{base['cost']['per_footage_second']} s per footage second." if base['cost']['per_footage_second'] else
                   f"Stored Qwen3-VL video analysis (baseline): {base['score']} on {', '.join(base['score_basis'])}.")
    if not complete:
        out.append('The run is not complete; these numbers will change.')
    return out
