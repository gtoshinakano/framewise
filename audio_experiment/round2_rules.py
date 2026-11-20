"""Round 2 rules (stdlib only, so `run.sh report2` works on the host): which voice evidence decides a line, how a short line gets
a usable clip, how the OP/ED mask is extended from the LFM3 music map, and how a final line gets its confidence.

Round 1 measured on 166 labelled lines (see FINDINGS.md): when the voice decides it is right far more often than Qwen, Qwen overruled it
in 11 of its 26 mistakes on hard lines, and 22 % of lines (mostly under 0.8 s) had no voice decision at all.
"""
from bisect import bisect_left

import lines as L
import voice_tables as VT

# A voice decision is final (Qwen does not answer the line) above these. Chosen on the round-1 labels (96 % right there, ±60 s guard);
# the report sweeps them again on every label set, so check that they hold on labels that were not used to pick them.
FIX_TOP = .45
FIX_GAP = .06
# Every voice score leaves out the reference ranges within this many seconds of the clip (round 1 only left out overlapping ranges,
# and each character has one sample whose ranges sit seconds apart in the same scene).
REF_GUARD = 60.
PAD_REACH = 1.0          # a short line's clip grows into the silence around it, at most this far on each side
TURN_REACH = 3.0         # the diarization turn around a line, clipped to this far past the line
MIN_CLIP = .6            # shorter clips are not embedded (as in round 1)
SOURCES = ('line', 'pad', 'turn')
MUSIC = ('singing', 'music')


def guarded_scores(cos_by_range, refs_meta, episode, start, end, guard=REF_GUARD):
    exclude = {r['range_id'] for r in refs_meta if r['episode'] == episode and L.overlap(start - guard, end + guard, r['start'], r['end']) > 0}
    return VT.scores(cos_by_range or {}, refs_meta, exclude)


def pad_span(ordered, index, reach=PAD_REACH):
    """The line's span grown into the silence before and after it, never into a neighbouring line."""
    line = ordered[index]
    left = line['start_time'] - reach
    right = line['end_time'] + reach
    if index > 0:
        left = max(left, ordered[index - 1]['end_time'])
    if index + 1 < len(ordered):
        right = min(right, ordered[index + 1]['start_time'])
    return round(min(left, line['start_time']), 3), round(max(right, line['end_time']), 3)


class Turns:
    """Diarization turns with the merged same-speaker stretch around a time."""

    def __init__(self, turns):
        self.turns = sorted(turns, key=lambda t: t['start_time'])
        self.starts = [t['start_time'] for t in self.turns]

    def stretch(self, start, end, speaker, reach=TURN_REACH, join=.3):
        """Span of the turns of `speaker` that overlap or touch [start, end], joined across gaps up to `join`, clipped to `reach` past the line."""
        lo, hi = None, None
        index = max(0, bisect_left(self.starts, start - 30.) - 1)
        mine = [t for t in self.turns[index:] if t['speaker'] == speaker and t['start_time'] < end + reach + join]
        for turn in mine:
            if turn['end_time'] < start - join or turn['start_time'] > end + join:
                continue
            lo = turn['start_time'] if lo is None else min(lo, turn['start_time'])
            hi = turn['end_time'] if hi is None else max(hi, turn['end_time'])
        if lo is None:
            return None
        changed = True
        while changed:                                  # follow the same speaker across short gaps
            changed = False
            for turn in mine:
                if turn['end_time'] >= lo - join and turn['start_time'] <= hi + join and (turn['start_time'] < lo or turn['end_time'] > hi):
                    lo, hi, changed = min(lo, turn['start_time']), max(hi, turn['end_time']), True
        return round(max(lo, start - reach), 3), round(min(hi, end + reach), 3)


def decide(score_map, top=None, gap=None):
    return VT.decide(score_map, FIX_TOP if top is None else top, FIX_GAP if gap is None else gap)


def source_scores(line, tables, voice2, refs, episode, sources=SOURCES):
    """{source: character scores} for one line: its own clip (round 1 tables) and the round-2 padded clip and turn, all with the ±60 s guard."""
    out = {}
    if 'line' in sources and tables['ok'].get(line['id']):
        out['line'] = guarded_scores(tables['line_ref_cos'].get(line['id'], {}), refs, episode, line['start_time'], line['end_time'])
    for source, entry in (voice2.get(line['id']) or {}).items():
        if source in sources:
            a, b = entry['span']
            out[source] = guarded_scores(entry['cos'], refs, episode, min(a, line['start_time']), max(b, line['end_time']))
    return out


def decisions(lines, tables, voice2, refs, episode, sources=SOURCES, top=None, gap=None):
    """Per line id: the voice cascade (line -> padded clip -> diarization turn)."""
    return {line['id']: cascade(source_scores(line, tables, voice2, refs, episode, sources), top, gap) for line in lines}


def cascade(scores_by_source, top_min=None, gap_min=None):
    """First source whose scores decide the line at the FIX thresholds.

    -> {speaker, source, top, gap, evidence: best top-3 among the sources tried, evidence_source}. speaker None when nothing decides.
    """
    best = None
    for source in SOURCES:
        sc = scores_by_source.get(source)
        if not sc:
            continue
        who, top, gap = decide(sc, top_min, gap_min)
        ranked = VT.ranked(sc)
        if best is None or ranked[0][1] > best[1][0][1]:
            best = (source, ranked)
        if who:
            return {'speaker': who, 'source': source, 'top': round(top, 3), 'gap': round(gap, 3),
                    'evidence': [[c, round(v, 3)] for c, v in ranked[:3]], 'evidence_source': source}
    if best is None:
        return {'speaker': None, 'source': None, 'evidence': [], 'evidence_source': None}
    source, ranked = best
    return {'speaker': None, 'source': None, 'top': round(ranked[0][1], 3), 'gap': round(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.), 3),
            'evidence': [[c, round(v, 3)] for c, v in ranked[:3]], 'evidence_source': source}


def borrow(ordered, decided, max_gap=2.0):
    """Undecided line -> the speaker of the nearest decided line (before or after, at most `max_gap` s away) with the same diarization label."""
    out = {}
    for index, line in enumerate(ordered):
        if decided.get(line['id']) or not line.get('diar'):
            continue
        best, distance = None, None
        for step in (-1, 1):
            j = index + step
            while 0 <= j < len(ordered):
                other = ordered[j]
                gap = line['start_time'] - other['end_time'] if step < 0 else other['start_time'] - line['end_time']
                if gap > max_gap:
                    break
                if other.get('diar') == line['diar'] and decided.get(other['id']):
                    if distance is None or gap < distance:
                        best, distance = decided[other['id']], gap
                    break
                j += step
        if best:
            out[line['id']] = best
    return out


def extend_ranges(ranges, windows, max_extend=75., min_new=20.):
    """Grow each OP/ED range over the adjacent windows LFM3 called singing or music, up to the farthest singing one; add runs of singing of at least `min_new` s.

    ``ranges`` [{start, end, kind}], ``windows`` [{start, end, label}] in time order (fixed length, no overlap).
    -> [{start, end, kind, grown: [seconds before, seconds after], source}].
    """
    windows = sorted(windows, key=lambda w: w['start'])
    out = []
    for item in ranges:
        start, end = item['start'], item['end']
        # walk outwards over singing/music windows, but stop the growth at the farthest SINGING window: dialogue over background
        # music may be called "music", lyrics are what leave text behind
        for w in reversed([w for w in windows if w['start'] < item['start']]):      # nearest first: the window holding the edge
            if w['label'] not in MUSIC or item['start'] - w['start'] > max_extend:
                break
            if w['label'] == 'singing':
                start = min(start, w['start'])
        for w in [w for w in windows if w['end'] > item['end']]:
            if w['label'] not in MUSIC or w['end'] - item['end'] > max_extend:
                break
            if w['label'] == 'singing':
                end = max(end, w['end'])
        out.append({'start': start, 'end': end, 'kind': item.get('kind'), 'grown': [round(item['start'] - start, 1), round(end - item['end'], 1)], 'source': 'fingerprint'})
    run = []
    for w in windows + [{'start': 1e9, 'end': 1e9, 'label': 'end'}]:
        if w['label'] == 'singing' and (not run or abs(w['start'] - run[-1]['end']) < .5):
            run.append(w)
            continue
        if run and run[-1]['end'] - run[0]['start'] >= min_new:
            a, b = run[0]['start'], run[-1]['end']
            if not any(L.overlap(a, b, r['start'], r['end']) > 0 for r in out):
                out.append({'start': a, 'end': b, 'kind': 'singing', 'grown': [0., 0.], 'source': 'lfm3'})
        run = [w] if w['label'] == 'singing' else []
    return sorted(out, key=lambda r: r['start'])


def split_by_mask(lines, mask, share=.5):
    """-> (kept, dropped): a line is dropped when at least `share` of it lies inside the mask."""
    kept, dropped = [], []
    for line in lines:
        (dropped if mask and L.inside_any(line['start_time'], line['end_time'], mask, share) else kept).append(line)
    return kept, dropped


def confidence(voice_speaker, final, q1, evidence_top1=None):
    """Confidence from agreement between independent signals (round 1: Qwen's own confidence was 'high' on 70 % of its hard-line errors).

    high: the voice decided and round-1 Qwen (which did not see that decision as final) agrees; medium: the voice decided against Qwen,
    or Qwen decided and both its runs and the strongest voice evidence agree; low: everything else.
    """
    if voice_speaker:
        return 'high' if q1 == voice_speaker else 'medium'
    if final and final == q1 and evidence_top1 == final:
        return 'medium'
    return 'low'


def label_lines(timed, base_lines, min_share=.6):
    """Timed labels [{episode, start, end, speaker, set}] -> {episode: {line id: (speaker, set)}} for the lines lying mostly inside a label.

    A line that two labels give to different characters is left out and listed in `conflicts`.
    """
    out, conflicts = {}, []
    for label in timed:
        for line in base_lines.get(label['episode'], []):
            duration = max(line['end_time'] - line['start_time'], 1e-6)
            if L.overlap(line['start_time'], line['end_time'], label['start'], label['end']) / duration < min_share:
                continue
            mine = out.setdefault(label['episode'], {})
            old = mine.get(line['id'])
            if old and old[0] != label['speaker']:
                conflicts.append({'episode': label['episode'], 'id': line['id'], 'speakers': [old[0], label['speaker']]})
                mine[line['id']] = (None, old[1])
            elif not old:
                mine[line['id']] = (label['speaker'], label['set'])
    for mine in out.values():
        for key in [k for k, v in mine.items() if v[0] is None]:
            del mine[key]
    return out, conflicts


def timed_from_ids(id_labels, base_lines, set_name, source):
    """Spot-check export {"ep1": {"L12": "Mai Sakurajima"}} made on the base run's line ids -> timed labels."""
    out = []
    for key, mapping in (id_labels or {}).items():
        episode = int(str(key).replace('ep', ''))
        by_id = {l['id']: l for l in base_lines.get(episode, [])}
        for line_id, speaker in mapping.items():
            if line_id in by_id and speaker not in (None, '', 'unknown', 'other'):
                line = by_id[line_id]
                out.append({'episode': episode, 'start': line['start_time'], 'end': line['end_time'], 'speaker': speaker, 'set': set_name,
                            'from': f'{source}:{line_id}', 'text': line['text']})
    return out
