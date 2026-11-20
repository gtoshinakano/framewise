"""Cross-episode matching of recurring audio by sub-fingerprints."""
import numpy as np

from . import audio_lines as L

HOP = 1600                 # 0.1 s at 16 kHz
FRAME = 4096
BANDS = 33
FRAME_SECONDS = HOP / 16000.
_POP = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)


def popcount32(values):
    values = values.astype(np.uint32)
    return _POP[values & 0xFFFF] + _POP[values >> 16]


def fingerprint(samples, rate=16000):
    """uint32 sub-fingerprint per 0.1 s."""
    if rate != 16000:
        raise ValueError('expected 16 kHz audio')
    edges = np.geomspace(300., 3000., BANDS + 1)
    bins = np.fft.rfftfreq(FRAME, 1. / rate)
    index = np.searchsorted(bins, edges)
    window = np.hanning(FRAME).astype(np.float32)
    n_frames = max(0, (len(samples) - FRAME) // HOP + 1)
    energy = np.zeros((n_frames, BANDS), dtype=np.float32)
    for start in range(0, n_frames, 1500):
        stop = min(n_frames, start + 1500)
        idx = (np.arange(start, stop)[:, None] * HOP) + np.arange(FRAME)[None, :]
        spectrum = np.abs(np.fft.rfft(samples[idx] * window, axis=1)) ** 2
        cumulative = np.concatenate([np.zeros((spectrum.shape[0], 1), dtype=np.float64), np.cumsum(spectrum, axis=1)], axis=1)
        energy[start:stop] = (cumulative[:, index[1:]] - cumulative[:, index[:-1]]).astype(np.float32)
    energy = np.log(energy + 1e-6)
    diff = energy[:, :-1] - energy[:, 1:]                       # (n, 32)
    bits = (diff[1:] - diff[:-1]) > 0                           # (n-1, 32)
    weights = (1 << np.arange(32, dtype=np.uint64)).astype(np.uint64)
    packed = (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.uint32)
    return packed


def _halves(hashes):
    """Each 32-bit hash as two 16-bit keys (low, high + 65536): a few flipped bits still leave one half intact."""
    keys = np.concatenate([(hashes & 0xFFFF).astype(np.int64), (hashes >> 16).astype(np.int64) + 65536])
    frames = np.concatenate([np.arange(len(hashes)), np.arange(len(hashes))])
    keep = ~np.isin(keys, [0, 0xFFFF, 65536, 65536 + 0xFFFF])
    return keys[keep], frames[keep]


def offset_votes(ha, hb, max_occurrence=40):
    """Counts of (frame in A) - (frame in B) over matching 16-bit half hashes, ignoring silence-like and very common keys."""
    ka, fa = _halves(ha)
    kb, fb = _halves(hb)
    order = np.argsort(kb, kind='stable')
    sb = kb[order]
    lo = np.searchsorted(sb, ka, 'left')
    hi = np.searchsorted(sb, ka, 'right')
    counts = hi - lo
    usable = (counts >= 1) & (counts <= max_occurrence)
    ia = np.nonzero(usable)[0]
    if not len(ia):
        return {}
    reps = counts[ia]
    a_frames = np.repeat(fa[ia], reps)
    starts = np.repeat(lo[ia], reps)
    within = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    b_frames = fb[order[starts + within]]
    offsets = a_frames - b_frames
    values, votes = np.unique(offsets, return_counts=True)
    return dict(zip(values.tolist(), votes.tolist()))


def peak_offsets(votes, minimum=30, ratio=6., exclusion=40, limit=8):
    if not votes:
        return []
    offsets = np.array(sorted(votes))
    counts = np.array([votes[o] for o in offsets])
    lookup = dict(zip(offsets.tolist(), counts.tolist()))
    smooth = np.array([c + lookup.get(o - 1, 0) + lookup.get(o + 1, 0) for o, c in zip(offsets.tolist(), counts.tolist())], dtype=float)
    rest = np.sort(smooth)[:-20] if len(smooth) > 40 else np.array([1.])
    typical = max(float(np.median(rest)) if len(rest) else 1., 1.)
    chosen = []
    for i in np.argsort(-smooth):
        if smooth[i] < minimum or smooth[i] < ratio * typical:
            break
        if all(abs(int(offsets[i]) - c) > exclusion for c in chosen):
            chosen.append(int(offsets[i]))
        if len(chosen) >= limit:
            break
    return chosen


def verify_diagonal(ha, hb, offset, max_bits=11, block=20, band=14, step=8, merge_gap=20, min_frames=100):
    """Runs where the two fingerprints agree along a diagonal that may drift a little.

    Another encode can be a fraction of a percent faster or slower, so the offset is searched per 2 s block within
    +-`band` frames of the previous block's (the first block: of the vote peak). Returns [(a_start, a_end, match_share)]
    in frames of A.
    """
    n_a, n_b = len(ha), len(hb)
    good_blocks = []                        # (a_start, ok, offset)
    current = offset
    first = True
    for a0 in range(0, n_a - block, block):
        centre = current
        reach = band * 2 if first else band
        offsets = np.arange(max(centre - reach, a0 + block - n_b), min(centre + reach, a0) + 1)
        best, best_offset = None, None
        if offsets.size:
            positions = a0 - offsets[:, None] + np.arange(block)
            means = popcount32(ha[a0:a0 + block] ^ hb[positions]).mean(axis=1)
            index = int(np.argmin(means))
            best, best_offset = float(means[index]), int(offsets[index])
        ok = best is not None and best <= max_bits - 1.5
        if ok and (first or abs(best_offset - current) <= step + band):
            current, first = best_offset, False
        good_blocks.append((a0, bool(ok), best_offset))
        if not ok and not first and len(good_blocks) > 6 and not any(g[1] for g in good_blocks[-6:]):
            current, first = offset, True   # lost it: look again around the vote peak
    runs, start = [], None
    for a0, ok, _ in good_blocks:
        if ok and start is None:
            start = a0
        elif not ok and start is not None:
            runs.append([start, a0])
            start = None
    if start is not None:
        runs.append([start, good_blocks[-1][0] + block])
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    result = []
    for s0, e0 in merged:
        if e0 - s0 >= min_frames:
            share = sum(1 for a0, ok, _ in good_blocks if ok and s0 <= a0 < e0) * block / max(e0 - s0, 1)
            result.append((s0, e0, round(share, 3)))
    return result


def local_offset(ha, hb, a_frame, guess, band=40, block=20):
    """Best offset (A frame - B frame) around `guess` for the block of A that starts at `a_frame`."""
    best, best_off = None, guess
    for off in range(guess - band, guess + band + 1):
        b0 = a_frame - off
        if b0 < 0 or b0 + block > len(hb) or a_frame + block > len(ha):
            continue
        value = float(popcount32(ha[a_frame:a_frame + block] ^ hb[b0:b0 + block]).mean())
        if best is None or value < best:
            best, best_off = value, off
    return best_off


def match_pair(ha, hb):
    """Segments both episodes share: [{a: (s, e), b: (s, e), score}] in seconds (B's times follow the local offset at each end)."""
    segments, seen = [], []
    for offset in peak_offsets(offset_votes(ha, hb)):
        for a_start, a_end, score in verify_diagonal(ha, hb, offset):
            if any(L.overlap(a_start, a_end, s, e) > .6 * (a_end - a_start) for s, e in seen):
                continue                     # the same stretch found through a neighbouring vote peak
            seen.append((a_start, a_end))
            first = local_offset(ha, hb, a_start, offset)
            last = local_offset(ha, hb, max(a_start, a_end - 20), offset)
            segments.append({'a': (a_start * FRAME_SECONDS, a_end * FRAME_SECONDS),
                             'b': ((a_start - first) * FRAME_SECONDS, (a_end - last) * FRAME_SECONDS), 'score': round(score, 3)})
    return segments


def consolidate(pairs, n_episodes, min_support=None, min_seconds=10., joined=.5):
    """Combine pairwise matches into recurring ranges per episode.

    ``pairs`` is {(i, j): [segment, ...]} with i < j (segment['a'] belongs to i, ['b'] to j). Returns {episode: [{start, end,
    support, partners, min_start, max_end}]}; a range is kept when it matches in at least ``min_support`` other episodes
    (default: half of them).
    """
    min_support = min_support or 2
    found = {}
    for (i, j), segments in pairs.items():
        for seg in segments:
            found.setdefault(i, []).append((seg['a'][0], seg['a'][1], j))
            found.setdefault(j, []).append((seg['b'][0], seg['b'][1], i))
    result = {}
    for episode, items in found.items():
        clusters = []
        for start, end, partner in sorted(items):
            for cluster in clusters:
                shorter = min(end - start, cluster['end'] - cluster['start'])
                if L.overlap(start, end, cluster['start'], cluster['end']) >= joined * max(shorter, 1e-6):
                    cluster['members'].append((start, end, partner))
                    cluster['start'] = min(m[0] for m in cluster['members'])
                    cluster['end'] = max(m[1] for m in cluster['members'])
                    break
            else:
                clusters.append({'start': start, 'end': end, 'members': [(start, end, partner)]})
        kept = []
        for cluster in clusters:
            members = cluster['members']
            partners = sorted({m[2] for m in members})
            if len(partners) < min_support:
                continue
            starts = sorted(m[0] for m in members)
            ends = sorted(m[1] for m in members)
            start, end = starts[len(starts) // 2], ends[len(ends) // 2]      # medians resist a partner that matched only part of it
            if end - start >= min_seconds:
                kept.append({'start': round(start, 2), 'end': round(end, 2), 'support': len(partners), 'partners': partners,
                             'min_start': round(starts[0], 2), 'max_end': round(ends[-1], 2)})
        result[episode] = sorted(kept, key=lambda r: r['start'])
    return result


def label_ranges(ranges, duration):
    """OP / ED / jingle / repeat by length and place."""
    for item in ranges:
        length = item['end'] - item['start']
        if length >= 45:
            item['kind'] = 'OP' if item['start'] < duration * .4 else 'ED' if item['end'] > duration * .6 else 'long repeat'
        else:
            item['kind'] = 'jingle'
    return ranges


# ------------------------------------------------------------------ 1.b signals inside one episode

def rms_levels(samples, rate=16000, hop_seconds=.05):
    """Amplitude (RMS) per `hop_seconds`; also returns the hop."""
    hop = int(rate * hop_seconds)
    n = len(samples) // hop
    if not n:
        return [], hop_seconds
    frames = samples[:n * hop].reshape(n, hop)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)).tolist(), hop_seconds
