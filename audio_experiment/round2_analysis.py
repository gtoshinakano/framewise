#!/usr/bin/env python3
"""Round 2 results and report (stdlib only, runs in the final2 container and on the host).

  python3 round2_analysis.py <round-2 run dir> --base <round-1 run dir> [--labels timed.json ...] [--id-labels spotcheck-export.json ...]

Timed labels ({"labels": [{episode, start, end, speaker, set}]}, see labels/round1.json) apply to any run; a spot-check export of this
round's page uses the base run's line ids and is turned into timed labels with set "round2".
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import lines as L  # noqa: E402
import names as N  # noqa: E402
import report  # noqa: E402
import round2_rules as R  # noqa: E402
import voice_tables as VT  # noqa: E402

FINAL = 'voice cascade → Q4 (round 2)'
SWEEP_TOP = (.35, .4, .45, .5, .55)
SWEEP_GAP = (.04, .06, .1)


def b0_speaker(line, b0):
    best, share = None, 0.
    for row in b0:
        o = L.overlap(line['start_time'], line['end_time'], row['start_time'], row['end_time'])
        if o > share:
            best, share = row.get('speaker'), o
    return best


def answered(speaker):
    return speaker not in (None, '', 'unknown')


def analyse_episode(run, base, prep, n, labelled):
    ep = prep['episodes'][str(n)]
    refs = base.load('emb', 'refs.json', default=[])
    music = run.load('music.json') or {}
    base_ranges = (base.load('oped.json') or {}).get('ranges', {}).get(str(n), [])
    ranges = (music.get('ranges') or {}).get(str(n), base_ranges)
    lines = sorted(base.load(f'ep{n}', 'lines', 'A1.json', default=[]), key=lambda l: l['start_time'])
    kept, dropped = R.split_by_mask(lines, [(r['start'], r['end']) for r in ranges])
    _, dropped_before = R.split_by_mask(lines, [(r['start'], r['end']) for r in base_ranges])
    before_ids = {l['id'] for l in dropped_before}
    tables = base.load(f'ep{n}', 'voice_tables.json', default={'ok': {}, 'line_ref_cos': {}})
    voice2 = run.load(f'ep{n}', 'voice2.json', default={})
    scores = {l['id']: R.source_scores(l, tables, voice2, refs, n) for l in kept}
    dec = {lid: R.cascade(sc) for lid, sc in scores.items()}

    def answers(run_, name):
        return ((run_.load(f'ep{n}', 'qwen', f'{name}.json') if run_ is base else run_.load(f'ep{n}', f'{name}.json')) or {}).get('answers', {})
    q1, q1s, q4 = answers(base, 'Q1'), answers(base, 'Q1s'), answers(run, 'Q4')
    sp = lambda table, lid: (table.get(lid) or {}).get('speaker')  # noqa: E731
    b0 = ep['baseline_lines']
    v3 = {}
    for l in kept:
        if tables['ok'].get(l['id']):
            sc = VT.scores(tables['line_ref_cos'].get(l['id'], {}), refs, VT.overlapping_ranges(refs, n, l['start_time'], l['end_time']))
            v3[l['id']] = VT.decide(sc)[0]
    by_source = lambda sources: {lid: (d['speaker'] if d['source'] in sources else None) for lid, d in dec.items()}  # noqa: E731
    voice_line, voice_pad, voice_all = by_source(('line',)), by_source(('line', 'pad')), by_source(R.SOURCES)
    borrowed = R.borrow(kept, voice_all)
    methods = {
        'B0 stored transcript': {l['id']: b0_speaker(l, b0) for l in kept},
        'Q1 (round 1)': {l['id']: sp(q1, l['id']) for l in kept},
        'Q1s (round 1)': {l['id']: sp(q1s, l['id']) for l in kept},
        'V3 voice (round 1 rule)': {l['id']: v3.get(l['id']) for l in kept},
        'voice: line clip': voice_line,
        'voice: + padded clip': voice_pad,
        'voice: + diarization turn': voice_all,
        'voice: + borrowed neighbour': {lid: voice_all[lid] or borrowed.get(lid) for lid in voice_all},
        'voice line clip → Q1': {lid: voice_line[lid] or sp(q1, lid) for lid in voice_line},
        'voice cascade → Q1': {lid: voice_all[lid] or sp(q1, lid) for lid in voice_all},
    }
    if q4:
        methods[FINAL] = {lid: voice_all[lid] or sp(q4, lid) for lid in voice_all}
        methods['voice cascade → borrowed → Q4'] = {lid: voice_all[lid] or borrowed.get(lid) or sp(q4, lid) for lid in voice_all}
    final_name = FINAL if q4 else 'voice cascade → Q1'
    final = methods[final_name]
    conf = {l['id']: R.confidence(voice_all[l['id']], final[l['id']], sp(q1, l['id']), (dec[l['id']]['evidence'] or [[None]])[0][0]) for l in kept}

    # ------------------------------------------------------------ labels
    mine = labelled.get(n, {})
    rows = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'answered': 0, 'right': 0}))
    for lid, (truth, set_name) in mine.items():
        if lid not in final:
            continue
        for name, table in methods.items():
            for key in (set_name, 'all'):
                r = rows[key][name]
                r['n'] += 1
                if answered(table.get(lid)):
                    r['answered'] += 1
                    r['right'] += table[lid] == truth
    buckets = defaultdict(lambda: {'n': 0, 'right': 0})
    sources = defaultdict(lambda: {'n': 0, 'right': 0})
    for lid, (truth, _) in mine.items():
        if lid in final:
            buckets[conf[lid]]['n'] += 1
            buckets[conf[lid]]['right'] += final[lid] == truth
            src = dec[lid]['source'] or ('Q4' if q4 else 'Q1')
            sources[src]['n'] += 1
            sources[src]['right'] += final[lid] == truth
    sweep = {}
    for top in SWEEP_TOP:
        for gap in SWEEP_GAP:
            n_dec = right = fin = 0
            for lid, (truth, _) in mine.items():
                if lid not in scores:
                    continue
                who = R.cascade(scores[lid], top, gap)['speaker']
                n_dec += bool(who)
                right += who == truth
                fin += (who or sp(q4, lid) or sp(q1, lid)) == truth
            sweep[f'{top}/{gap}'] = {'decided': n_dec, 'right': right, 'final_right': fin, 'n': sum(1 for lid in mine if lid in scores)}
    labels_dropped = [{'id': lid, 'speaker': truth, 'set': s, 'text': next(l['text'] for l in lines if l['id'] == lid)} for lid, (truth, s) in mine.items() if lid not in final]

    # ------------------------------------------------------------ names
    names_run = run.load(f'ep{n}', 'names.json') or {}
    aliased, fixes = names_run.get('aliased') or {}, names_run.get('fixes') or {}
    text_now = {l['id']: fixes.get(l['id']) or aliased.get(l['id']) or l['text'] for l in kept}
    hits = {'A1': Counter(), 'aliases': Counter(), 'aliases + Qwen': Counter()}
    for l in kept:
        hits['A1'].update(N.hits(l['text']))
        hits['aliases'].update(N.hits(aliased.get(l['id'], l['text'])))
        hits['aliases + Qwen'].update(N.hits(text_now[l['id']]))
    changes = []
    for l in kept:
        if text_now[l['id']] != l['text']:
            changes.append({'id': l['id'], 'time': l['start_time'], 'before': l['text'], 'after': text_now[l['id']],
                            'by': 'Qwen' if l['id'] in fixes else 'alias'})

    # ------------------------------------------------------------ music map
    windows = (music.get('windows') or {}).get(str(n), [])
    centres = [((l['start_time'] + l['end_time']) / 2) for l in kept]
    speech_windows = [w for w in windows if any(w['start'] <= c < w['end'] for c in centres)]
    music_stats = {'windows': len(windows), 'labels': dict(Counter(w['label'] for w in windows)),
                   'windows_with_kept_lines': len(speech_windows),
                   'of_those_called_music_or_singing': sum(1 for w in speech_windows if w['label'] in R.MUSIC)}
    newly = [{'id': l['id'], 'time': l['start_time'], 'text': l['text'], 'q1': sp(q1, l['id'])} for l in dropped if l['id'] not in before_ids]

    # ------------------------------------------------------------ spot-check picks (not labelled yet): changed by round 2 first, then low confidence
    q1_of = lambda lid: sp(q1, lid)  # noqa: E731
    picks = [l for l in kept if l['id'] not in mine and final[l['id']] != q1_of(l['id'])]
    picks += [l for l in kept if l['id'] not in mine and conf[l['id']] == 'low' and l not in picks]
    samples = []
    for l in picks[:40]:
        cues = L.subtitle_overlap(l, ep['subtitles'])
        cands = sorted({x for x in (final[l['id']], q1_of(l['id']), sp(q4, l['id']), voice_all[l['id']], borrowed.get(l['id'])) if answered(x) and x != 'other'})
        samples.append({'id': l['id'], 'start': l['start_time'], 'end': l['end_time'], 'text': {'A1': text_now[l['id']]},
                        'pt': ' / '.join(c['text'].replace('\n', ' ') for c in cues) or None, 'candidates': cands})
    samples.sort(key=lambda p: p['start'])

    source_share = Counter(d['source'] or ('Q4' if q4 else 'Q1') for d in dec.values())
    return {'episode': n, 'title': ep['title'], 'lines': len(lines), 'kept': len(kept), 'dropped_total': len(dropped), 'dropped_new': newly,
            'ranges_before': base_ranges, 'ranges_after': ranges, 'music': music_stats,
            'decided_by': dict(source_share), 'confidence': dict(Counter(conf.values())), 'final_method': final_name,
            'unknown_final': sum(1 for v in final.values() if not answered(v)),
            'label_rows': {k: dict(v) for k, v in rows.items()}, 'buckets': dict(buckets), 'sources': dict(sources), 'sweep': sweep,
            'labels_dropped': labels_dropped, 'names': {k: dict(v) for k, v in hits.items()}, 'name_changes': changes,
            'q4': {k: v for k, v in ((run.load(f'ep{n}', 'Q4.json') or {}).items()) if k in ('asked', 'decided', 'answered', 'seconds')},
            'samples': samples}


def merge_counts(dicts):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for d in dicts:
        for set_name, methods in d.items():
            for name, row in methods.items():
                for k, v in row.items():
                    out[set_name][name][k] += v
    return {s: {m: dict(r) for m, r in ms.items()} for s, ms in out.items()}


def add(dicts):
    out = defaultdict(lambda: defaultdict(int))
    for d in dicts:
        for k, row in d.items():
            for kk, v in row.items():
                out[k][kk] += v
    return {k: dict(v) for k, v in out.items()}


def compute(run, base, timed=()):
    prep = base.load('prep.json')
    if not prep:
        return None
    base_lines = {n: base.load(f'ep{n}', 'lines', 'A1.json', default=[]) for n in common.EVALUATED}
    labelled, conflicts = R.label_lines(list(timed), base_lines)
    episodes = {}
    for n in common.EVALUATED:
        if str(n) in prep['episodes'] and base_lines[n]:
            episodes[str(n)] = analyse_episode(run, base, prep, n, labelled)
    eps = list(episodes.values())
    return {'run_id': run.id, 'base_run': base.id, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'thresholds': {'fix_top': R.FIX_TOP, 'fix_gap': R.FIX_GAP, 'ref_guard': R.REF_GUARD, 'pad_reach': R.PAD_REACH, 'turn_reach': R.TURN_REACH},
            'episodes': episodes, 'labels': {'n': sum(len(v) for v in labelled.values()), 'sets': dict(Counter(s for v in labelled.values() for _, s in v.values())),
                                             'conflicts': conflicts},
            'label_rows': merge_counts(e['label_rows'] for e in eps), 'buckets': add(e['buckets'] for e in eps), 'sources': add(e['sources'] for e in eps),
            'sweep': add(e['sweep'] for e in eps), 'timings': {**base.load('timings.json', default={}), **run.load('timings.json', default={})},
            'notes': run.load('notes.json', default=[]), 'progress': run.load('progress.json', default={}),
            'characters': prep.get('characters', []), 'samples': {k: e['samples'] for k, e in episodes.items()}}


# ====================================================================== report

def rate(right, n):
    return f'{right}/{n} = {right / n:.0%}' if n else '–'


def verdict(r):
    rows = r['label_rows']
    parts = []
    for set_name in sorted(rows, key=lambda s: (s == 'all', s)):
        m = rows[set_name]
        final = next((k for k in (FINAL, 'voice cascade → Q1') if k in m), None)
        if not final:
            continue
        q1 = m.get('Q1 (round 1)', {})
        parts.append(f"{set_name} labels: {final} {rate(m[final]['right'], m[final]['n'])} against Q1 {rate(q1.get('right', 0), q1.get('n', 0))}")
    eps = r['episodes'].values()
    lines = sum(e['kept'] for e in eps)
    decided = sum(v for e in eps for k, v in e['decided_by'].items() if k in R.SOURCES)
    dropped = sum(len(e['dropped_new']) for e in eps)
    fixes = sum(len(e['name_changes']) for e in eps)
    text = '; '.join(parts) or 'no labels matched'
    return (f"{text}. The voice decides {decided} of {lines} lines ({decided / max(lines, 1):.0%}); Qwen answers the rest with the decided lines as context. "
            f"The music map drops {dropped} more lines as OP/ED/singing (listed below to check). Names: {fixes} lines changed. "
            f"Labels: {r['labels']['n']} lines ({', '.join(f'{k} {v}' for k, v in r['labels']['sets'].items())}). The fix thresholds were chosen on the round-1 labels, "
            f"so the numbers on those sets are optimistic; labels from this round's spot check are the honest test.")


def build(r):
    d = report.Doc()
    d.h(1, f"Audio experiment round 2 · {r['run_id']} (base {r['base_run']})")
    d.p(f"Generated {r['generated_at']}. Progress: {r['progress'].get('status') or '–'}.", 'mut')
    d.h(2, 'Verdict')
    d.p(verdict(r), 'verdict')
    d.p(f"A voice score is final at a top score of at least {R.FIX_TOP} and a gap of at least {R.FIX_GAP} to the runner-up, with every reference range within "
        f"{R.REF_GUARD:.0f} s of the clip left out. The cascade tries the line's own clip, then the clip grown into the silence around it (up to {R.PAD_REACH} s each side), "
        f"then the diarization turn around it (up to {R.TURN_REACH} s past the line).", 'mut')

    d.h(2, '1 · Speaker methods on the labelled lines')
    d.p('"typical" = the dry run spot check (episode openings, consecutive lines), "hard" = round 1 disagreement picks, "round2" = this round\'s spot check '
        '(changed and low-confidence lines). Strict: an unanswered line counts as wrong.', 'mut')
    for set_name in sorted(r['label_rows'], key=lambda s: (s == 'all', s)):
        m = r['label_rows'][set_name]
        d.h(3, set_name)
        d.table(['method', 'n', 'right (strict)', 'answered', 'right when answered'],
                [[k, v['n'], rate(v['right'], v['n']), v['answered'], rate(v['right'], v['answered'])] for k, v in sorted(m.items(), key=lambda kv: -kv[1]['right'] / max(kv[1]['n'], 1))])

    d.h(2, '2 · Who decides a line, and how often it is right')
    rows = []
    for n, e in r['episodes'].items():
        rows.append([f'Ep{n}', e['kept'], *[e['decided_by'].get(s, 0) for s in (*R.SOURCES, 'Q4', 'Q1')], e['unknown_final'],
                     e['q4'].get('asked', '–'), e['q4'].get('seconds', '–')])
    d.table(['ep', 'lines', 'voice: line', 'voice: padded', 'voice: turn', 'Qwen Q4', 'Qwen Q1 (no Q4)', 'final unknown', 'Q4 asked', 'Q4 seconds'], rows)
    d.table(['decided by', 'labelled lines', 'right'], [[k, v['n'], rate(v['right'], v['n'])] for k, v in sorted(r['sources'].items())])
    d.h(3, 'Confidence (from agreement, not from Qwen)')
    share = Counter()
    for e in r['episodes'].values():
        share.update(e['confidence'])
    total = sum(share.values()) or 1
    d.table(['confidence', 'share of all lines', 'labelled', 'right'],
            [[k, f'{share.get(k, 0) / total:.0%}', r['buckets'].get(k, {}).get('n', 0), rate(r['buckets'].get(k, {}).get('right', 0), r['buckets'].get(k, {}).get('n', 0))]
             for k in ('high', 'medium', 'low')])
    d.h(3, 'Threshold sweep on the labels (voice cascade, then Q4 / Q1)')
    d.table(['top / gap', 'voice decides', 'voice right', 'final right'],
            [[k, f"{v['decided']}/{v['n']}", rate(v['right'], v['decided']), rate(v['final_right'], v['n'])] for k, v in r['sweep'].items()])

    d.h(2, '3 · OP / ED and singing (LFM3 music map)')
    rows = []
    for n, e in r['episodes'].items():
        after = '; '.join(f"{report.clock(x['start'])}–{report.clock(x['end'])} {x.get('kind') or ''} (+{x['grown'][0]:.0f}s before, +{x['grown'][1]:.0f}s after, {x['source']})" for x in e['ranges_after'])
        rows.append([f'Ep{n}', len(e['ranges_before']), after or '–', len(e['dropped_new']),
                     f"{e['music']['of_those_called_music_or_singing']}/{e['music']['windows_with_kept_lines']}", json.dumps(e['music']['labels'])])
    d.table(['ep', 'fingerprint ranges', 'ranges now', 'lines newly dropped', '5 s windows with dialogue called music/singing', 'labels'], rows)
    for n, e in r['episodes'].items():
        if e['dropped_new']:
            d.p(f'Ep{n} lines newly dropped (check that none is dialogue): ' + ' · '.join(f"{report.clock(x['time'])} {x['id']} {x['text'][:40]} ({x['q1']})" for x in e['dropped_new']))
        if e['labels_dropped']:
            d.p(f'Ep{n} labelled lines now inside the mask: ' + ' · '.join(f"{x['id']} {x['text'][:30]} ({x['speaker']}, {x['set']})" for x in e['labels_dropped']))

    d.h(2, '4 · Character names')
    for n, e in r['episodes'].items():
        spellings = sorted(set().union(*[set(v) for v in e['names'].values()]))
        d.table([f'Ep{n} spelling', *e['names'].keys()], [[s, *[v.get(s, 0) for v in e['names'].values()]] for s in spellings])
    changes = [[f"Ep{n}", report.clock(c['time']), c['by'], c['before'], c['after']] for n, e in r['episodes'].items() for c in e['name_changes']]
    d.p(f'{len(changes)} lines changed; every change is listed so a wrong one is easy to spot.', 'mut')
    d.table(['ep', 'time', 'by', 'before', 'after'], changes[:200])

    d.h(2, '5 · Time')
    keys = sorted(k for k in r['timings'] if 'round2' in k or k.startswith('phase.'))
    d.table(['stage', 'seconds', 'detail'], [[k, report.num(r['timings'][k].get('seconds')), ', '.join(f'{a}={b}' for a, b in r['timings'][k].items() if a != 'seconds')] for k in keys])
    t = r['timings']
    parts = {'diarization D1': (t.get('ep3.diar.D1') or {}).get('seconds'), 'ASR A1': (t.get('ep3.asr.A1') or {}).get('seconds'),
             'voice clips': (t.get('ep3.round2.voice') or {}).get('seconds'), 'music map': (t.get('round2.music') or {}).get('seconds', 0) / max(len(r['episodes']), 1),
             'Q4': (t.get('ep3.round2.Q4') or {}).get('seconds'), 'names': (t.get('ep3.round2.names') or {}).get('seconds')}
    known = {k: v for k, v in parts.items() if v}
    d.p(f"Per 25-minute episode (Episode 3, model loads excluded): {', '.join(f'{k} {v:.0f}s' for k, v in known.items())}; total {sum(known.values()):.0f}s, "
        f"against {report.num((t.get('baseline_rerun.ep3') or {}).get('seconds'), 0)}s for the current pipeline.")

    d.h(2, '6 · Notes')
    if r['labels']['conflicts']:
        d.p('Label conflicts (left out): ' + json.dumps(r['labels']['conflicts'], ensure_ascii=False))
    d.table(['when', 'kind', 'note'], [[x['at'], x['kind'], x['text']] for x in r['notes']] or [['–', '–', 'none']])
    return d


def load_labels(paths, id_paths, base):
    timed = []
    for path in paths:
        data = json.load(open(path))
        timed.extend(data['labels'] if isinstance(data, dict) and 'labels' in data else data)
    base_lines = {n: base.load(f'ep{n}', 'lines', 'A1.json', default=[]) for n in common.EVALUATED}
    for path in id_paths:
        timed.extend(R.timed_from_ids(json.load(open(path)), base_lines, 'round2', Path(path).name))
    return timed


def write(run, base, timed):
    results = compute(run, base, timed)
    if results is None:
        print('nothing to report yet (the base run has no prep.json)')
        return None
    run.save(results, 'results2.json')
    doc = build(results)
    (run.dir / 'report.html').write_text(doc.html(f'Audio experiment round 2 {run.id}', refresh=results['progress'].get('status') == 'running'))
    (run.dir / 'report.md').write_text(doc.md())
    page = report.spotcheck(results).replace(f'experiment.py rescore {run.id} --labels file.json', f'audio_experiment/run.sh report2 {run.id} file.json')
    (run.dir / 'spotcheck.html').write_text(page)
    for name in ('report.html', 'report.md', 'spotcheck.html', 'results2.json'):
        try:
            (run.dir / name).chmod(0o666)
        except OSError:
            pass
    return results


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir')
    parser.add_argument('--base', required=True)
    parser.add_argument('--labels', action='append', default=[])
    parser.add_argument('--id-labels', action='append', default=[])
    args = parser.parse_args(argv[1:])
    run_dir, base_dir = Path(args.run_dir), Path(args.base)
    run, base = common.Run(run_dir.parent, run_dir.name), common.Run(base_dir.parent, base_dir.name)
    if write(run, base, load_labels(args.labels, args.id_labels, base)) is None:
        return 1
    print('wrote', run_dir / 'report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
