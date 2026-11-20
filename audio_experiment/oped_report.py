"""OP/ED section of the results (stdlib only): ranges, consistency, what the baseline lines inside them look like, 1.b signals."""
import lines as L
import oped_signals


def clock(t):
    return f'{int(t // 60)}:{t % 60:04.1f}'


def summary(run, prep):
    data = run.load('oped.json') or {}
    ranges = data.get('ranges', {})
    out = {'method': data.get('method'), 'episodes': {}, 'pairs': data.get('pairs', {}), 'consistency': {}}
    kinds = {}
    for n, items in ranges.items():
        for item in items:
            kinds.setdefault(item['kind'], []).append((n, item))
    for kind, entries in kinds.items():
        lengths = [e['end'] - e['start'] for _, e in entries]
        out['consistency'][kind] = {'episodes': sorted({n for n, _ in entries}), 'median_seconds': L.median(lengths), 'min_seconds': min(lengths), 'max_seconds': max(lengths)}
    for n in (prep.get('episodes') or {}):
        ep = prep['episodes'][n]
        mask = [(r['start'], r['end']) for r in ranges.get(n, [])]
        lines = ep['baseline_lines']
        inside = [l for l in lines if mask and L.inside_any(l['start_time'], l['end_time'], mask, .5)]
        garbage = [l for l in inside if L.garbage_reasons(l['text'])]
        probably_real = [l for l in inside if not L.garbage_reasons(l['text']) and L.japanese_ratio(l['text']) >= .8 and len(l['text']) >= 3]
        outside_garbage = [l for l in lines if l not in inside and L.garbage_reasons(l['text'])]
        lyric = oped_signals.lyric(lines)
        signals = run.load(f'ep{n}', 'signals.json', default={}) or {}
        cues_inside = [c for c in ep['subtitles'] if mask and L.inside_any(c['start_time'], c['end_time'], mask, .5)]
        out['episodes'][n] = {
            'ranges': [{**r, 'from': clock(r['start']), 'to': clock(r['end'])} for r in ranges.get(n, [])],
            'baseline_lines': len(lines), 'inside': len(inside), 'garbage_removed': len(garbage), 'probably_real_dialogue_lost': len(probably_real),
            'real_examples': [{'time': clock(l['start_time']), 'text': l['text']} for l in probably_real[:12]],
            'garbage_left_outside': len(outside_garbage), 'garbage_left_examples': [{'time': clock(l['start_time']), 'text': l['text'][:60]} for l in outside_garbage[:8]],
            'subtitle_cues_inside': len(cues_inside), 'lyric_signal': lyric, 'lyric_vs_ranges': oped_signals.compare(lyric['ranges'], mask),
            'music_like': [[round(a, 1), round(b, 1)] for a, b in (signals.get('music_like') or [])],
            'music_like_vs_ranges': oped_signals.compare(signals.get('music_like') or [], mask)}
    return out
