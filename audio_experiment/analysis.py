"""Turns the checkpoints of a run into results.json (stdlib only; runs in the container at the end and on the host for `run.sh report` / `rescore`)."""
import json
import time
from collections import Counter, defaultdict

import common
import glossary
import lines as L
import oped_report
import scoring as S
import voice_tables as VT

DIAR_NAMES = ['D1', 'D0', 'thr0.45', 'thr0.75', 'thr0.9', 'hint3-8', 'max6_thr0.75']
Q_NAMES = ['Q1', 'Q2', 'Q1s', 'Q3', 'R1']


def read_jsonl(path):
    out = []
    try:
        for row in open(path):
            try:
                out.append(json.loads(row))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def asr_records(run, n, variant):
    return read_jsonl(run.path(f'ep{n}', 'asr', f'{variant}.jsonl'))


def label_anchors(labels, n, base_lines):
    """Anchors from the spot-check file {"ep3": {"L12": "Mai Sakurajima"}}."""
    mine = (labels or {}).get(f'ep{n}') or (labels or {}).get(str(n)) or {}
    by_id = {l['id']: l for l in base_lines}
    return [{'episode': n, 'kind': 'label', 'character': who, 'start': by_id[i]['start_time'], 'end': by_id[i]['end_time'], 'range_id': f'label:{i}'}
            for i, who in mine.items() if i in by_id and who and who not in ('unknown',)]


def with_speaker(lines, mapping, speaker_of):
    return [{**l, 'speaker': speaker_of(l)} for l in lines]


def on_screen_flag(line, ep):
    for cut in ep['cuts']:
        if cut['start_time'] <= line['start_time'] < cut['end_time']:
            shown = (ep.get('roster') or {}).get(cut['id'])
            if shown:
                return line.get('speaker') in shown
    return None


def method_scores(lines, anchors, ep, mask, q1_lines=None, v3_lines=None, loo=True):
    out = {'distribution': S.distribution([{**l, 'on_screen': on_screen_flag(l, ep)} for l in lines]),
           'contradictions': S.addressed_contradictions(lines), 'pronouns': S.pronoun_consistency(lines), 'loo': loo}
    kinds = {k: S.anchor_accuracy(lines, anchors, kind=k) for k in ('sample', 'manual', 'label')}
    out['anchors'] = {'all': S.anchor_accuracy(lines, anchors), **kinds}
    if q1_lines is not None:
        out['agreement_with_Q1'] = S.agreement(lines, q1_lines)
    if v3_lines is not None:
        out['agreement_with_V3'] = S.agreement(lines, v3_lines)
    return out


def analyse_episode(run, prep, n, labels=None):
    ep = prep['episodes'][str(n)]
    samples = prep['samples']
    mask = [(r['start'], r['end']) for r in (run.load('oped.json') or {}).get('ranges', {}).get(str(n), [])]
    base = run.load(f'ep{n}', 'lines', 'A1.json')
    if not base:
        return None
    anchors = ep['anchors'] + label_anchors(labels, n, base)
    refs = run.load('emb', 'refs.json', default=[])
    tables = run.load(f'ep{n}', 'voice_tables.json', default={'ok': {}, 'line_ref_cos': {}})
    clusters = run.load(f'ep{n}', 'cluster_tables.json', default={'cluster_ref_cos': {}, 'cluster_sample_cos': {}})
    glossary_data = run.load('glossary.json', default={}) or {}
    names = list(glossary.VERIFIED[t]['ja'] for t in glossary.VERIFIED)
    result = {'episode': n, 'title': ep['title'], 'anchors_n': {k: sum(1 for a in anchors if a['kind'] == k) for k in ('sample', 'manual', 'label')},
              'mask': [list(m) for m in mask], 'methods': {}, 'text': {}, 'diar': {}}

    # ---------------------------------------------------------------- B0
    b0 = ep['baseline_lines']
    result['methods']['B0'] = {**method_scores(b0, anchors, ep, mask, loo=False),
                               'note': 'the stored transcript; its character links use the same voice samples the anchors come from, so its anchor score is inflated',
                               'job_seconds': ep.get('baseline_job_seconds')}
    result['text']['B0'] = {'segmentation': S.segmentation_stats(b0, ep['subtitles'], mask, S.baseline_speaker_switches(ep['baseline_rows'])),
                            'garbage': S.garbage_stats(b0, mask, names)}

    # ---------------------------------------------------------------- voice-only methods on the base lines
    exclusions = {l['id']: VT.overlapping_ranges(refs, n, l['start_time'], l['end_time']) for l in base}

    def voice_scores(line):
        return VT.scores(tables['line_ref_cos'].get(line['id'], {}), refs, exclusions[line['id']]) if tables['ok'].get(line['id']) else {}

    v3 = []
    v3raw = []
    for line in base:
        sc = voice_scores(line)
        who, top, gap = VT.decide(sc)
        v3.append({**line, 'speaker': who, 'confidence': 'high' if who and top >= .5 else 'medium' if who else None, 'reason': 'voice', 'voice': VT.ranked(sc)[:3]})
        raw = VT.ranked(sc)[0][0] if sc else None
        v3raw.append({**line, 'speaker': raw})
    qlines = {}
    for name in Q_NAMES:
        q = run.load(f'ep{n}', 'qwen', f'{name}.json')
        if q:
            qlines[name] = [{**l, 'speaker': (q['answers'].get(l['id']) or {}).get('speaker'), 'confidence': (q['answers'].get(l['id']) or {}).get('confidence'),
                             'reason': (q['answers'].get(l['id']) or {}).get('reason'), 'fix': (q['answers'].get(l['id']) or {}).get('text_fix'),
                             'fix_reason': (q['answers'].get(l['id']) or {}).get('fix_reason'), 'redecided': (q['answers'].get(l['id']) or {}).get('redecided')} for l in base]
    q1 = qlines.get('Q1')
    result['methods']['V3'] = method_scores(v3, anchors, ep, mask, q1, None)
    result['methods']['V3raw'] = method_scores(v3raw, anchors, ep, mask, q1, None)
    enrol = run.load(f'ep{n}', 'enrol_scores.json')
    if enrol:
        lines_e = []
        for line in base:
            sc = enrol['scores'].get(line['id'])
            who, top, gap = VT.decide(sc or {}, .3, .05)
            lines_e.append({**line, 'speaker': who})
        result['methods']['ENROL'] = {**method_scores(lines_e, anchors, ep, mask, q1, v3), 'sources': enrol['sources'], 'characters': enrol['characters']}
    for name, lines_q in qlines.items():
        result['methods'][name] = {**method_scores(lines_q, anchors, ep, mask, q1 if name != 'Q1' else None, v3),
                                   'seconds': (run.load(f'ep{n}', 'qwen', f'{name}.json') or {}).get('seconds'),
                                   'unanswered': sum(1 for l in lines_q if l['speaker'] is None)}
    if q1 and 'Q1s' in qlines:
        result['stability'] = S.agreement(q1, qlines['Q1s'])
    if q1 and v3:
        result['voice_vs_qwen'] = S.agreement(v3, q1)
    # combination: voice and Qwen agree
    if q1:
        combo = []
        for a, b in zip(v3, q1):
            agree = a['speaker'] and a['speaker'] == b['speaker']
            combo.append({**b, 'speaker': b['speaker'] if agree else ('unknown' if a['speaker'] and b['speaker'] and a['speaker'] != b['speaker'] else b['speaker'] or a['speaker'])})
        result['methods']['V3+Q1'] = method_scores(combo, anchors, ep, mask, q1, v3)

    # ---------------------------------------------------------------- diarization settings (voices per setting, cluster -> character, LOO)
    index_of = {}
    records = asr_records(run, n, 'A1')
    for name in DIAR_NAMES:
        diar = run.load(f'ep{n}', 'diar', f'{name}.json')
        if not diar:
            continue
        index = L.TurnIndex(diar['turns'])
        lines_s = L.build_variant_lines(records, index)
        app_map = VT.match_clusters_app(clusters['cluster_sample_cos'].get(name, {}), samples)
        cr = clusters['cluster_ref_cos'].get(name, {})
        cache = {}

        def range_speaker(line, cr=cr, cache=cache):
            exclude = frozenset(VT.overlapping_ranges(refs, n, line['start_time'], line['end_time']))
            if exclude not in cache:
                cache[exclude] = VT.match_clusters_range(cr, refs, exclude)
            return cache[exclude].get(line.get('diar'))

        entry = {'voices': diar.get('n_voices'), 'seconds': diar.get('wall_seconds'), 'turns': len(diar['turns']), 'lines': len(lines_s),
                 'mapped_app': len(app_map), 'mapped_range': len(VT.match_clusters_range(cr, refs)),
                 'speech_seconds': round(sum(t['end_time'] - t['start_time'] for t in diar['turns']), 1), 'mixed_lines': sum(1 for l in lines_s if l.get('mixed'))}
        entry['app'] = method_scores(with_speaker(lines_s, app_map, lambda l: app_map.get(l.get('diar'))), anchors, ep, mask, loo=False)
        entry['range'] = method_scores(with_speaker(lines_s, None, range_speaker), anchors, ep, mask, loo=True)
        result['diar'][name] = entry

    # ---------------------------------------------------------------- text variants
    variants = {'A1': base}
    for name in ('A2', 'A3'):
        loaded = run.load(f'ep{n}', 'lines', f'{name}.json')
        if loaded:
            variants[name] = loaded
    a0 = asr_records(run, n, 'A0')
    d0 = run.load(f'ep{n}', 'diar', 'D0.json')
    if a0 and d0:
        variants['A0'] = L.build_variant_lines(a0, L.TurnIndex(d0['turns']))
    ctx_names = [glossary.VERIFIED[t]['ja'] for t in glossary.VERIFIED]
    for name, lines_v in variants.items():
        switches = sum(1 for l in lines_v if l.get('mixed'))
        result['text'][name] = {'segmentation': S.segmentation_stats(lines_v, ep['subtitles'], mask, switches), 'garbage': S.garbage_stats(lines_v, mask, ctx_names),
                                'asr_seconds': sum(r['seconds'] for r in asr_records(run, n, name)) or None,
                                'languages': dict(Counter(r.get('language') for r in asr_records(run, n, name)))}
    pairs = [('B0', 'A1'), ('A1', 'A2'), ('A1', 'A3'), ('A2', 'A3'), ('A1', 'A0'), ('B0', 'A3')]
    all_lines = {'B0': b0, **variants}
    result['text']['agreement'] = {f'{a}~{b}': S.text_agreement(all_lines[a], all_lines[b]) for a, b in pairs if a in all_lines and b in all_lines}
    second = run.load(f'ep{n}', 'second.json')
    if second:
        done = second['lines']
        settled = 0
        voice_agree = 0
        for line_id, info in done.items():
            tv = info.get('tight_voice') or {}
            qa = (qlines.get('R1') or qlines.get('Q1') or [])
            q_by = {l['id']: l for l in qa}
            qspeaker = (q_by.get(line_id) or {}).get('speaker')
            if tv.get('decision') and qspeaker not in (None, 'unknown'):
                voice_agree += tv['decision'] == qspeaker
            if info['cer'] <= .2 and tv.get('decision'):
                settled += 1
        cers = [i['cer'] for i in done.values()]
        result['second'] = {'considered': second['considered'], 'reread': len(done), 'median_cer': L.median(cers), 'disagree_over_20pct': sum(1 for c in cers if c > .2),
                            'settled_by_voice': settled, 'voice_agrees_with_qwen': voice_agree, 'seconds': second['seconds']}
    # Qwen text fixes, for the user to read
    fixes = []
    for name in ('Q1', 'Q2', 'Q3'):
        for l in qlines.get(name, []):
            if l.get('fix'):
                fixes.append({'method': name, 'id': l['id'], 'time': l['start_time'], 'before': l['text'], 'after': l['fix'], 'reason': l.get('fix_reason'), 'cer': round(L.cer(l['text'], l['fix']), 3)})
    result['fixes'] = fixes[:120]
    result['lines_base'] = len(base)
    return result, {'base': base, 'v3': v3, 'q': qlines, 'enrol': enrol}


def disagreement_samples(run, prep, n, extras, per_episode=40):
    """Lines where the methods disagree most, with every method's text and speaker (for the report and the spot check)."""
    ep = prep['episodes'][str(n)]
    base, v3, q = extras['base'], extras['v3'], extras['q']
    v3_by = {l['id']: l for l in v3}
    q_by = {k: {l['id']: l for l in v} for k, v in q.items()}
    a2 = run.load(f'ep{n}', 'lines', 'A2.json', default=[])
    a3 = run.load(f'ep{n}', 'lines', 'A3.json', default=[])

    def text_in(lines_v, line):
        pool = [m for m in lines_v if L.overlap(line['start_time'], line['end_time'], m['start_time'], m['end_time']) > 0]
        return ''.join(m['text'] for m in sorted(pool, key=lambda m: m['start_time'])) or None

    scored = []
    for line in base:
        answers = {}
        for name, by in q_by.items():
            if line['id'] in by and by[line['id']]['speaker']:
                answers[name] = by[line['id']]
        if v3_by[line['id']]['speaker']:
            answers['V3'] = v3_by[line['id']]
        speakers = {a['speaker'] for a in answers.values() if a['speaker'] not in (None, 'unknown')}
        texts = {text_in(ep['baseline_lines'], line), line['text'], text_in(a2, line), text_in(a3, line)} - {None}
        scored.append((len(speakers) * 3 + (len(texts) - 1) + (1 if any(a.get('confidence') == 'low' for a in answers.values()) else 0), line, answers, speakers))
    scored.sort(key=lambda t: (-t[0], t[1]['start_time']))
    picks = []
    for score, line, answers, speakers in scored:
        if score <= 0 or len(picks) >= per_episode:
            break
        cues = L.subtitle_overlap(line, ep['subtitles'])
        picks.append({'id': line['id'], 'start': line['start_time'], 'end': line['end_time'], 'diar': line.get('diar'),
                      'text': {'B0': text_in(ep['baseline_lines'], line), 'A1': line['text'], 'A2': text_in(a2, line), 'A3': text_in(a3, line)},
                      'pt': ' / '.join(c['text'].replace('\n', ' ') for c in cues) or None,
                      'speakers': {k: {'speaker': a['speaker'], 'confidence': a.get('confidence'), 'reason': a.get('reason')} for k, a in answers.items()},
                      'voice': v3_by[line['id']].get('voice'), 'candidates': sorted(speakers)})
    return sorted(picks, key=lambda p: p['start'])


def compute(run, labels=None):
    prep = run.load('prep.json')
    if not prep:
        return None
    out = {'run_id': run.id, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'excerpt': prep.get('excerpt'),
           'episodes': {}, 'samples': {}, 'breakdown_used': False,
           'breakdown_available': prep.get('breakdown_available'), 'notes': run.load('notes.json', default=[]), 'timings': run.load('timings.json', default={}),
           'progress': run.load('progress.json', default={}), 'glossary': run.load('glossary.json'), 'lfm3': run.load('lfm3.json'),
           'oped': oped_report.summary(run, prep)}
    extras_all = {}
    for n in common.EVALUATED:
        if str(n) not in prep['episodes']:
            continue
        got = analyse_episode(run, prep, n, labels)
        if got:
            out['episodes'][str(n)], extras = got
            out['samples'][str(n)] = disagreement_samples(run, prep, n, extras)
            extras_all[n] = extras
    out['ranking'] = ranking(out)
    out['recommended_time'] = recommended_time(out)
    return out


def method_row(ep_results, name, group='methods'):
    m = ep_results[group].get(name)
    return m


def ranking(out):
    """Speaker methods ordered by anchor accuracy (episodes 1+2 pooled, leave-one-out where the method uses references), then contradictions, agreement, stability."""
    rows = []
    names = set()
    for ep in out['episodes'].values():
        names.update(ep['methods'])
        for setting, entry in ep['diar'].items():
            names.add(f'diar:{setting}:range')
            names.add(f'diar:{setting}:app')
    for name in sorted(names):
        n_all = hits = answered = 0
        contra_n = contra_bad = 0
        agree_n = agree_same = 0
        stab_n = stab_same = 0
        loo = None
        for ep in out['episodes'].values():
            if name.startswith('diar:'):
                _, setting, kind = name.split(':')
                m = (ep['diar'].get(setting) or {}).get(kind)
            else:
                m = ep['methods'].get(name)
            if not m:
                continue
            loo = m.get('loo')
            a = m['anchors']['all']
            n_all += a['n']
            hits += a['correct']
            answered += a['answered']
            contra_n += m['contradictions']['addressed_lines']
            contra_bad += m['contradictions']['contradictions']
            key = 'agreement_with_Q1' if name.startswith(('V3', 'ENROL')) or name.startswith('diar:') else 'agreement_with_V3'
            ag = m.get(key)
            if ag:
                agree_n += ag['both_answered']
                agree_same += ag['same']
            if name == 'Q1' and ep.get('stability'):
                stab_n += ep['stability']['both_answered']
                stab_same += ep['stability']['same']
        rows.append({'method': name, 'anchor_n': n_all, 'anchor_strict': hits / n_all if n_all else None, 'anchor_answered': hits / answered if answered else None,
                     'anchor_coverage': answered / n_all if n_all else None, 'contradiction_rate': contra_bad / contra_n if contra_n else None,
                     'contradictions': f'{contra_bad}/{contra_n}', 'agreement': agree_same / agree_n if agree_n else None,
                     'stability': stab_same / stab_n if stab_n else None, 'loo': loo})
    return S.rank_speaker_methods(rows)


def recommended_time(out):
    """Seconds per 25-minute episode for the recommended pipeline = base diarization + A3 ASR + embeddings + Q1 + R1 + second opinion, from episode 3."""
    t = out['timings']
    parts = {'diarization': (t.get('ep3.diar.D1') or {}).get('seconds'), 'asr': (t.get('ep3.asr.A3') or t.get('ep3.asr.A1') or {}).get('seconds'),
             'qwen_Q1': (t.get('ep3.qwen.Q1') or {}).get('seconds'), 'qwen_R1': (t.get('ep3.qwen.R1redo') or {}).get('seconds'), 'second_opinion': (t.get('ep3.second') or {}).get('seconds')}
    known = {k: v for k, v in parts.items() if v}
    return {'parts': parts, 'total_seconds': round(sum(known.values()), 1) if known else None,
            'baseline_seconds': (t.get('baseline_rerun.ep3') or {}).get('seconds'), 'note': 'model load times are separate (see the time table)'}
