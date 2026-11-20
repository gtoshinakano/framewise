"""Scoring without ground truth (stdlib only). Everything takes plain lists of dicts, so `run.sh report` works on the host.

A *method result* is a list of lines `{id, start_time, end_time, text, speaker, confidence?, reason?, ...}` where
`speaker` is a character name (db title), 'unknown', 'other' or None. Anchors are
`{episode, kind: 'sample'|'manual', character, start, end, range_id}`.
"""
import math
import re
from collections import Counter, defaultdict

import glossary
import lines as L

UNKNOWN = (None, '', 'unknown')


def answered(speaker):
    return speaker not in UNKNOWN


def anchor_pairs(lines, anchors, min_share=.6):
    """(line, anchor) for each line that lies mostly inside an anchor span. A line matches at most one anchor (the largest overlap)."""
    pairs = []
    for line in lines:
        duration = max(line['end_time'] - line['start_time'], 1e-6)
        best, best_share = None, 0.
        for anchor in anchors:
            shared = L.overlap(line['start_time'], line['end_time'], anchor['start'], anchor['end'])
            share = shared / duration if anchor['kind'] == 'sample' else shared / max(anchor['end'] - anchor['start'], 1e-6)
            need = min_share if anchor['kind'] == 'sample' else .5
            if share >= need and share > best_share:
                best, best_share = anchor, share
        if best:
            pairs.append((line, best))
    return pairs


def anchor_accuracy(lines, anchors, kind=None, speaker_of=None):
    """Accuracy against the anchors: strict = correct / all anchor lines, answered = correct / answered anchor lines."""
    speaker_of = speaker_of or (lambda line: line.get('speaker'))
    pairs = [(l, a) for l, a in anchor_pairs(lines, anchors) if kind in (None, a['kind'])]
    per = defaultdict(lambda: {'n': 0, 'answered': 0, 'correct': 0})
    for line, anchor in pairs:
        speaker = speaker_of(line)
        row = per[anchor['character']]
        row['n'] += 1
        if answered(speaker):
            row['answered'] += 1
            row['correct'] += speaker == anchor['character']
    n = sum(r['n'] for r in per.values())
    got = sum(r['answered'] for r in per.values())
    ok = sum(r['correct'] for r in per.values())
    return {'n': n, 'answered': got, 'correct': ok,
            'strict': ok / n if n else None, 'answered_accuracy': ok / got if got else None,
            'coverage': got / n if n else None, 'by_character': dict(per)}


def addressed_contradictions(lines, speaker_of=None):
    """Lines that address a character by name + honorific but are attributed to that same character."""
    speaker_of = speaker_of or (lambda line: line.get('speaker'))
    patterns = {c: re.compile(p) for c in glossary.VERIFIED if (p := glossary.addressed_patterns(c))}
    addressed, bad, examples = 0, 0, []
    for line in lines:
        speaker = speaker_of(line)
        hit = [c for c, rx in patterns.items() if rx.search(line['text'])]
        if not hit:
            continue
        if not answered(speaker):
            continue
        addressed += 1
        if speaker in hit:
            bad += 1
            if len(examples) < 12:
                examples.append({'id': line.get('id'), 'time': line['start_time'], 'text': line['text'], 'speaker': speaker})
    return {'addressed_lines': addressed, 'contradictions': bad, 'rate': bad / addressed if addressed else None, 'examples': examples}


def pronoun_class(text):
    """Counts of first-person pronoun classes in a text (longest spelling first, so あたし is not わたし)."""
    counts = Counter()
    for cls, spellings in glossary.PRONOUNS.items():
        for spelling in spellings:
            counts[cls] += text.count(spelling)
    # 'あたし' contains 'たし' only; 'わたし' contains neither 'あたし' nor 'わし'; the classes do not overlap
    return +counts


def entropy(counter):
    total = sum(counter.values())
    if not total:
        return None
    return -sum(c / total * math.log2(c / total) for c in counter.values() if c)


def pronoun_consistency(lines, speaker_of=None, min_uses=6):
    """Per character: pronoun class distribution and its entropy (lower = more consistent); weighted mean over characters."""
    speaker_of = speaker_of or (lambda line: line.get('speaker'))
    per = defaultdict(Counter)
    for line in lines:
        speaker = speaker_of(line)
        if answered(speaker) and speaker != 'other':
            per[speaker].update(pronoun_class(line['text']))
    rows, weighted, weight = {}, 0., 0
    for character, counts in per.items():
        total = sum(counts.values())
        h = entropy(counts)
        rows[character] = {'uses': total, 'entropy': h, 'distribution': dict(counts)}
        if h is not None and total >= min_uses:
            weighted += h * total
            weight += total
    return {'by_character': rows, 'weighted_entropy': weighted / weight if weight else None, 'uses_scored': weight}


def agreement(a, b, speaker_a=None, speaker_b=None):
    """Share of lines (matched by id) where two methods give the same speaker, among those both answered."""
    speaker_a = speaker_a or (lambda line: line.get('speaker'))
    speaker_b = speaker_b or (lambda line: line.get('speaker'))
    index = {line['id']: line for line in b}
    both, same = 0, 0
    for line in a:
        other = index.get(line['id'])
        if not other:
            continue
        x, y = speaker_a(line), speaker_b(other)
        if answered(x) and answered(y):
            both += 1
            same += x == y
    return {'both_answered': both, 'same': same, 'agreement': same / both if both else None}


def distribution(lines, total_lines=None):
    counts = Counter(l.get('speaker') if answered(l.get('speaker')) else 'unknown' for l in lines)
    n = len(lines) or 1
    confidences = Counter(l.get('confidence') for l in lines if l.get('confidence'))
    on_screen = [l for l in lines if l.get('on_screen') is not None]
    return {'lines': len(lines), 'speakers': dict(counts.most_common()), 'voices': len([k for k in counts if k != 'unknown']),
            'unknown_share': counts.get('unknown', 0) / n,
            'low_share': confidences.get('low', 0) / n if confidences else None,
            'confidence': dict(confidences),
            'on_screen_share': (sum(1 for l in on_screen if l['on_screen']) / len(on_screen)) if on_screen else None}


# ------------------------------------------------------------------ segmentation

def segmentation_stats(lines, subtitles=None, mask=(), speaker_switches=None):
    """Shape of a line set: lengths, short and unfinished lines, cut-by-window lines, edges against the subtitle cue edges."""
    kept = [l for l in lines if not L.inside_any(l['start_time'], l['end_time'], mask, .5)] if mask else list(lines)
    durations = [l['end_time'] - l['start_time'] for l in kept]
    chars = [len(L.normalise_for_cer(l['text'])) for l in kept]
    result = {'lines': len(kept), 'median_seconds': L.median(durations), 'median_chars': L.median(chars),
              'short_lines': sum(1 for d in durations if d < .8),
              'unfinished': sum(1 for l in kept if not L.ends_sentence(l['text'])),
              'window_cut': sum(1 for l in kept if l.get('window_edge') and not L.ends_sentence(l['text'])),
              'mixed_speaker_lines': sum(1 for l in kept if l.get('mixed'))}
    if speaker_switches is not None:
        result['speaker_switches_inside_sentence'] = speaker_switches
    if subtitles:
        cues = [c for c in subtitles if not L.inside_any(c['start_time'], c['end_time'], mask, .5)] if mask else subtitles
        starts = [l['start_time'] for l in kept]
        ends = [l['end_time'] for l in kept]
        result['edge_start'] = L.edge_match(starts, [c['start_time'] for c in cues])
        result['edge_end'] = L.edge_match(ends, [c['end_time'] for c in cues])
    return result


def baseline_speaker_switches(rows):
    """Sentences (by final punctuation, speaker ignored) of stored word rows whose rows carry more than one speaker label."""
    count = 0
    for sentence in L.sentences_from_rows(rows):
        labels = {r['speaker_label'] for r in sentence if r.get('speaker_label')}
        count += len(labels) > 1
    return count


def garbage_stats(lines, mask=(), glossary_names=()):
    flagged, inside, reasons = 0, 0, Counter()
    for line in lines:
        in_mask = bool(mask) and L.inside_any(line['start_time'], line['end_time'], mask, .5)
        inside += in_mask
        found = L.garbage_reasons(line['text'])
        if found and not in_mask:
            flagged += 1
            reasons.update(found)
    outside = [l for l in lines if not (mask and L.inside_any(l['start_time'], l['end_time'], mask, .5))]
    names = sum(1 for l in outside for n in glossary_names if n in l['text'])
    return {'lines': len(lines), 'inside_mask': inside, 'garbage_outside_mask': flagged, 'reasons': dict(reasons),
            'garbage_rate': flagged / len(outside) if outside else None, 'glossary_hits': names}


def text_agreement(lines_a, lines_b, minimum=.5):
    """CER between two independent readings of the same speech: each line of A against the text of B inside its span."""
    scores = []
    for line in lines_a:
        pool = [m for m in lines_b if L.overlap(line['start_time'], line['end_time'], m['start_time'], m['end_time']) > 0]
        if not pool:
            continue
        text_b = ''.join(m['text'] for m in sorted(pool, key=lambda m: m['start_time']))
        share = sum(L.overlap(line['start_time'], line['end_time'], m['start_time'], m['end_time']) for m in pool) / max(line['end_time'] - line['start_time'], 1e-6)
        if share >= minimum:
            scores.append(L.cer(line['text'], text_b))
    return {'n': len(scores), 'mean_cer': sum(scores) / len(scores) if scores else None, 'median_cer': L.median(scores)}


def rank_speaker_methods(rows):
    """Order methods the way the task asks: anchor accuracy first, then contradictions, then agreement and stability.

    `rows` is a list of dicts with `method`, `anchor_strict`, `contradiction_rate`, `agreement`, `stability`; None sorts last.
    No composite number is built.
    """
    def key(row):
        def hi(value):
            return -value if value is not None else 1.
        def lo(value):
            return value if value is not None else 9.
        return (hi(row.get('anchor_strict')), lo(row.get('contradiction_rate')), hi(row.get('agreement')), hi(row.get('stability')))
    return sorted(rows, key=key)
