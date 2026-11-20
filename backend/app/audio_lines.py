"""Pause-cut ASR windows and sentence lines on the episode clock."""
from bisect import bisect_right
from collections import defaultdict

from .transcript_parts import join_text

SENTENCE_END = '。！？!?…．'
CLOSERS = '」』）)】"”’'

def ends_sentence(text):
    """True when the text ends in a sentence-final form (final punctuation, optionally followed by a closing quote)."""
    stripped = (text or '').rstrip().rstrip(CLOSERS)
    return bool(stripped) and stripped[-1] in SENTENCE_END


def merge_intervals(intervals, gap=0.):
    merged = []
    for start, end in sorted((s, e) for s, e in intervals if e > s):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def subtract_intervals(intervals, cuts):
    """Remove the `cuts` intervals from `intervals`."""
    result = []
    cuts = merge_intervals(cuts)
    for start, end in intervals:
        pieces = [(start, end)]
        for cut_start, cut_end in cuts:
            next_pieces = []
            for a, b in pieces:
                if cut_end <= a or cut_start >= b:
                    next_pieces.append((a, b))
                    continue
                if cut_start > a:
                    next_pieces.append((a, cut_start))
                if cut_end < b:
                    next_pieces.append((cut_end, b))
            pieces = next_pieces
        result.extend(p for p in pieces if p[1] - p[0] > 1e-6)
    return result


def overlap(a_start, a_end, b_start, b_end):
    return max(0., min(a_end, b_end) - max(a_start, b_start))


def inside_any(start, end, intervals, share=.5):
    """True when at least `share` of [start, end] lies inside the union of `intervals`."""
    length = max(end - start, 1e-6)
    return sum(overlap(start, end, s, e) for s, e in merge_intervals(intervals)) / length >= share


# ------------------------------------------------------------------ windows at pauses

def quietest_cut(rms, hop, low, high):
    """Time of the quietest frame in [low, high] (rms: one dB or amplitude value per `hop` seconds, smoothed over 3 frames)."""
    first, last = max(0, int(low / hop)), min(len(rms) - 1, int(high / hop))
    if last <= first:
        return (low + high) / 2
    best, best_value = first, None
    for i in range(first, last + 1):
        window = rms[max(0, i - 1):i + 2]
        value = sum(window) / len(window)
        if best_value is None or value < best_value:
            best, best_value = i, value
    return min(high, max(low, best * hop + hop / 2))


def plan_windows(regions, rms, hop, mask=(), cap=25., min_len=.8, pad=.25, max_gap=6., join_gap=.35, split_floor=.55):
    """Cut speech into ASR windows of at most `cap` seconds at the pauses.

    ``regions`` are speech regions [(start, end)] (diarization turns or voiced stretches, already without the mask),
    ``rms`` the level per ``hop`` seconds, ``mask`` the OP/ED ranges that no window may cross. Regions closer than
    ``join_gap`` are one stretch; a stretch longer than ``cap`` is cut at its quietest frame in the last 45 percent;
    stretches are packed into a window while it stays within ``cap``, and when the next one does not fit the window is
    closed at the longest pause in the second half of the window (never inside speech when a pause exists).
    A window never spans a masked range or a pause longer than ``max_gap``.
    """
    mask = merge_intervals(mask)
    stretches = []
    for start, end in merge_intervals(regions, gap=join_gap):
        if end - start >= min_len / 2:
            stretches.append((start, end))
    cut = []
    for start, end in stretches:
        while end - start > cap - 2 * pad:
            point = quietest_cut(rms, hop, start + split_floor * cap, start + cap - 2 * pad)
            cut.append((start, point))
            start = point
        cut.append((start, end))
    stretches = cut

    def blocked(a_end, b_start):
        return b_start - a_end > max_gap or any(overlap(a_end, b_start, ms, me) > 0 or (ms >= a_end and me <= b_start) for ms, me in mask)

    windows, group = [], []

    def close(items):
        if not items:
            return
        start, end = items[0][0], items[-1][1]
        if end - start >= min_len:
            windows.append((max(0., start - pad), end + pad))

    for item in stretches:
        if not group:
            group = [item]
            continue
        if blocked(group[-1][1], item[0]):
            close(group)
            group = [item]
            continue
        if item[1] - group[0][0] <= cap - 2 * pad:
            group.append(item)
            continue
        # does not fit: close at the longest pause in the second half of the group
        halfway = group[0][0] + cap * .5
        best, gap_best = None, .0
        for index in range(1, len(group)):
            gap = group[index][0] - group[index - 1][1]
            if group[index][0] >= halfway and gap > gap_best:
                best, gap_best = index, gap
        if best is None or gap_best < .3:
            close(group)
            group = [item]
        else:
            close(group[:best])
            group = group[best:] + [item]
            while group[-1][1] - group[0][0] > cap - 2 * pad and len(group) > 1:
                close(group[:1])
                group = group[1:]
    close(group)
    # keep windows from overlapping and from entering the mask
    result = []
    for start, end in windows:
        for piece in subtract_intervals([(start, end)], mask):
            if piece[1] - piece[0] >= min_len:
                result.append(piece)
    result.sort()
    cleaned = []
    for start, end in result:
        if cleaned and start < cleaned[-1][1]:
            start = cleaned[-1][1]
        if end - start >= min_len:
            cleaned.append((start, end))
    return cleaned


# ------------------------------------------------------------------ speaker per token / sentence

class TurnIndex:
    """Exclusive diarization turns with a binary-search lookup of the speaker with the most overlap."""

    def __init__(self, turns):
        self.turns = sorted(turns, key=lambda t: (t['start_time'], t['end_time']))
        self.starts = [t['start_time'] for t in self.turns]
        self.max_len = max((t['end_time'] - t['start_time'] for t in self.turns), default=0.)

    def overlaps(self, start, end):
        result = defaultdict(float)
        index = max(0, bisect_right(self.starts, start - self.max_len) - 1)
        for turn in self.turns[index:]:
            if turn['start_time'] >= end:
                break
            shared = min(end, turn['end_time']) - max(start, turn['start_time'])
            if shared > 0:
                result[turn['speaker']] += shared
        return result

    def speaker(self, start, end):
        shares = self.overlaps(start, end)
        return max(shares, key=shares.get) if shares else None

    def majority(self, start, end, tokens=None):
        """Speaker with the largest summed overlap with the given word spans (or one span); also returns its share."""
        total = defaultdict(float)
        for s, e in tokens or [(start, end)]:
            for speaker, shared in self.overlaps(s, e).items():
                total[speaker] += shared
        if not total:
            return None, 0.
        speaker = max(total, key=total.get)
        return speaker, total[speaker] / max(sum(total.values()), 1e-9)


def split_sentences(tokens, long_pause=1.2):
    """Group aligned tokens ({start, end, text}) into sentences: after final punctuation, or at a pause of `long_pause`."""
    sentences, current = [], []
    for index, token in enumerate(tokens):
        current.append(token)
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if ends_sentence(token['text']) or (following and following['start'] - token['end'] >= long_pause):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def split_by_speaker(sentence, index, min_pause=.3):
    """Split one sentence where the speaker changes AND there is a pause of at least `min_pause` between the two tokens.

    Returns the pieces; a speaker change without a pause leaves the sentence whole (the majority speaker takes it).
    """
    labels = [index.speaker(t['start'], t['end']) for t in sentence]
    cuts = []
    last_known = None
    for i, label in enumerate(labels):
        if label is None:
            continue
        if last_known is not None and label != labels[last_known]:
            # the change happened somewhere between tokens last_known and i: take the longest pause there
            best, best_gap = None, -1.
            for j in range(last_known, i):
                gap = sentence[j + 1]['start'] - sentence[j]['end']
                if gap > best_gap:
                    best, best_gap = j, gap
            if best_gap >= min_pause:
                cuts.append(best + 1)
        last_known = i
    pieces, begin = [], 0
    for cut in cuts + [len(sentence)]:
        if cut > begin:
            pieces.append(sentence[begin:cut])
            begin = cut
    return pieces


def build_lines(tokens, index, window_id=None, window_edges=None, min_pause=.3, long_pause=1.2):
    """Aligned tokens -> lines with one diarization speaker each (majority of the word overlap), tokens kept as `words`.

    ``window_edges`` is (start, end) of the ASR window: a line touching either edge is flagged `window_edge`.
    """
    lines = []
    for sentence in split_sentences(tokens, long_pause):
        for piece in split_by_speaker(sentence, index, min_pause):
            text = ''
            for token in piece:
                text = join_text(text, token['text']) if token['text'][:1].isascii() else text + token['text']
            start, end = piece[0]['start'], piece[-1]['end']
            speaker, share = index.majority(start, end, [(t['start'], t['end']) for t in piece])
            token_speakers = [index.speaker(t['start'], t['end']) for t in piece]
            line = {'start_time': round(start, 3), 'end_time': round(end, 3), 'text': text.strip(), 'diar': speaker,
                    'diar_share': round(share, 3), 'window': window_id,
                    'mixed': len({s for s in token_speakers if s}) > 1,
                    'words': [{'s': round(t['start'], 3), 'e': round(t['end'], 3), 't': t['text'], 'spk': sp} for t, sp in zip(piece, token_speakers)]}
            if window_edges:
                line['window_edge'] = start - window_edges[0] < .35 or window_edges[1] - end < .35
            if line['text']:
                lines.append(line)
    return lines


def subtitle_overlap(line, subtitles, minimum=.25):
    """Subtitle cues that overlap a line (at least `minimum` of the line or of the cue)."""
    found = []
    duration = max(line['end_time'] - line['start_time'], 1e-6)
    for cue in subtitles:
        shared = overlap(line['start_time'], line['end_time'], cue['start_time'], cue['end_time'])
        if shared <= 0:
            continue
        if shared / duration >= minimum or shared / max(cue['end_time'] - cue['start_time'], 1e-6) >= minimum:
            found.append(cue)
    return found


