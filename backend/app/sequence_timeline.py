"""Timeline arithmetic for sequences: everything a model must not be asked to calculate.

A sequence is a contiguous range of cuts played on the original audio, directed as ONE SHOT PER CUT:
shots are never merged, and only a cut longer than one shot (15 s) is divided into consecutive parts
of one take. This module turns the cuts, the transcript and the audio level into one clock (seconds
from the first cut), speech lines ``L1…``, pauses and the hard-cut boundaries ``K1…``. Where a hard
cut falls inside speech, its shot boundary may instead sit in the silence just before or just after
that phrase, so the phrase stays in one shot; a director model only chooses among those options. The
exact positions (snapped to where the voice really starts or stops in the audio), the limits, the
division of long cuts, the initial frame, the edges and the dialogue of each shot are computed and
checked here. Nothing in this file touches the database.
"""
import array
import itertools
import math
import re
import wave
from functools import lru_cache
from operator import mul
from pathlib import Path

MAX_SEQUENCE_SECONDS = 180.0
MAX_SHOT_SECONDS = 15.0       # one H3 take (assistant.h3_prompt.MAX_SECONDS)
MIN_GENERATE_SECONDS = 5.0    # H3 is trained for about 5-15 s; shorter shots are generated longer and trimmed
MIN_SHOT_SECONDS = 0.04       # one frame: every cut is a shot, however short
EDGE_SECONDS = 2.5            # a long cut is divided no closer than this to its hard cuts
SHIFT_SECONDS = 3.0           # how far a boundary may move off a hard cut so that a phrase stays in one shot
OWN_SHARE = 0.5               # every cut keeps at least this share of its own picture in its shot
EDGE_PAD = 0.06               # a moved boundary sits this far into the silence before or after the voice
SPEECH_TAIL = 0.6             # how far past a transcript edge the voice may still be heard
HOLD_SECONDS = 0.1            # the level must stay low this long to count as the end (or the start) of a voice
MIN_PAUSE = 0.3
WINDOW = 0.05
EPSILON = 1e-6
CONTINUE_GAP = 1.0            # two lines of one voice closer than this may be one sentence cut in half
CHOICES = ('keep_cut', 'before_line', 'after_line')

SENTENCE_END = re.compile(r'[。.!?！？…」』）)]$')
# A line that starts with one of these is almost never the start of a sentence.
PARTICLES = ('は', 'が', 'を', 'に', 'で', 'と', 'も', 'の', 'へ', 'や', 'か', 'ね', 'よ', 'て', 'って', 'から', 'まで', 'けど', 'ので', 'のに', 'し', 'ば')


class PlanError(ValueError):
    """A plan that cannot be used; ``errors`` are sentences a director model or a person can act on."""

    def __init__(self, errors):
        super().__init__('; '.join(errors))
        self.errors = list(errors)


def shot_key(start_time, end_time):
    """Stable identity of a shot: its episode interval in milliseconds, so unchanged shots survive a plan edit."""
    return f'{round(start_time * 1000)}-{round(end_time * 1000)}'


def join_text(left, right):
    """Join transcript fragments: no space between CJK fragments, one between Latin words."""
    if left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return f'{left} {right}'
    return left + right


def words(parts):
    text = ''
    for part in parts:
        text = join_text(text, part['text'])
    return text


# ------------------------------------------------------------------- audio

@lru_cache(maxsize=16)
def _envelope(path, mtime_ns, size, start, end, window):
    """RMS level per window of a 16-bit PCM WAV between two episode times, 0..1."""
    levels = []
    try:
        with wave.open(path, 'rb') as source:
            if source.getsampwidth() != 2:
                return ()
            rate, channels = source.getframerate(), source.getnchannels()
            first = max(0, min(source.getnframes(), math.floor(start * rate)))
            last = max(first, min(source.getnframes(), math.ceil(end * rate)))
            step = max(1, round(window * rate))
            source.setpos(first)
            remaining = last - first
            while remaining > 0:
                data = source.readframes(min(step, remaining))
                if not data:
                    break
                values = array.array('h')
                values.frombytes(data[:len(data) - len(data) % 2])
                remaining -= len(values) // channels
                if not values:
                    break
                levels.append(math.sqrt(sum(map(mul, values, values)) / len(values)) / 32768)
    except (OSError, wave.Error, EOFError):
        return ()
    return tuple(levels)


def audio_envelope(path, start, end, window=WINDOW):
    """Level of the prepared episode audio over ``[start, end)``; empty when there is no usable audio."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return []
    return list(_envelope(str(path), stat.st_mtime_ns, stat.st_size, round(start, 6), round(end, 6), window))


def quiet_threshold(envelope):
    """Level under which the audio counts as a pause: just above its own noise floor."""
    if not envelope:
        return 0.0
    ordered = sorted(envelope)
    low, high = ordered[len(ordered) // 10], ordered[min(len(ordered) - 1, len(ordered) * 95 // 100)]
    return max(0.003, low + 0.12 * (high - low))


def find_pauses(envelope, window=WINDOW, min_pause=MIN_PAUSE):
    """Quiet stretches as ``(t0, t1)`` in seconds from the start of the envelope."""
    threshold = quiet_threshold(envelope)
    pauses, begin = [], None
    for index, level in enumerate([*envelope, math.inf]):
        if level <= threshold and begin is None:
            begin = index
        elif level > threshold and begin is not None:
            if (index - begin) * window >= min_pause - EPSILON:
                pauses.append((round(begin * window, 3), round(index * window, 3)))
            begin = None
    return pauses


def quietest(envelope, t0, t1, window=WINDOW):
    """The quietest moment inside ``[t0, t1]``; the midpoint without an envelope."""
    first, last = max(0, math.ceil(t0 / window)), min(len(envelope), math.floor(t1 / window))
    if last <= first:
        return round((t0 + t1) / 2, 3)
    # Prefer the middle of the quietest run, so a boundary does not sit on the edge of a word.
    span = envelope[first:last]
    lowest = min(span)
    run = [index for index, level in enumerate(span) if level <= lowest * 1.25 + 1e-9]
    best, current = [], []
    for index in run:
        if current and index != current[-1] + 1:
            best, current = max(best, current, key=len), []
        current.append(index)
    best = max(best, current, key=len)
    return round((first + (best[0] + best[-1] + 1) / 2) * window, 3)


def speech_edge(envelope, t, side, low, high, window=WINDOW):
    """Where a voice really stops after ``t`` (side ``after``) or sets in before ``t`` (side ``before``), padded into the silence.

    Transcript word times are often a little early or late; the audio level shows where the voice fades
    out or begins. The search never goes the other way past the transcript edge (so the boundary never
    lands inside the words) and the result stays inside ``(low, high)``, the silence between this line
    and its neighbour. Without a usable envelope, or when the level does not drop in that gap (music,
    noise), the transcript time is used with a short pad.
    """
    room = high - low
    fallback = round(t + min(0.15, room / 2) if side == 'after' else t - min(0.15, room / 2), 3)
    if not envelope or room <= 0.02:
        return fallback
    threshold = quiet_threshold(envelope)
    if threshold >= max(envelope):
        return fallback  # a flat level says nothing about where a voice starts or stops
    hold = max(1, round(HOLD_SECONDS / window))
    quiet = [level <= threshold for level in envelope]
    if side == 'after':
        first = max(0, math.floor(t / window + EPSILON))
        last = min(len(quiet) - hold, math.floor(min(high, t + SPEECH_TAIL) / window))
        for index in range(first, last + 1):
            if all(quiet[index:index + hold]):
                edge = max(t, index * window) + EDGE_PAD
                return round(min(edge, (max(t, index * window) + high) / 2), 3) if edge >= high - 0.02 else round(edge, 3)
        return fallback
    first = min(len(quiet), math.ceil(t / window - EPSILON))
    last = max(hold, math.ceil(max(low, t - SPEECH_TAIL) / window))
    for index in range(first, last - 1, -1):
        if all(quiet[index - hold:index]):
            edge = min(t, index * window) - EDGE_PAD
            return round(max(edge, (min(t, index * window) + low) / 2), 3) if edge <= low + 0.02 else round(edge, 3)
    return fallback


# ------------------------------------------------------------------- lines

def speech_lines(merged, origin, duration):
    """Merged transcript lines on the sequence clock, clipped to it, numbered ``L1…``.

    ``merged`` are lines from ``merged_transcript`` with their word fragments in ``parts``. A line that
    crosses the sequence edge keeps only the fragments whose midpoint lies inside.
    """
    lines = []
    for line in merged:
        parts = []
        for part in line.get('parts') or [line]:
            middle = (part['start_time'] + part['end_time']) / 2 - origin
            text = str(part.get('text') or '').strip()
            if text and -EPSILON <= middle < duration + EPSILON:
                parts.append({'id': part.get('id'), 't0': round(max(0.0, part['start_time'] - origin), 3), 't1': round(min(duration, part['end_time'] - origin), 3), 'text': text})
        if not parts:
            continue
        lines.append({'id': f'L{len(lines) + 1}', 't0': parts[0]['t0'], 't1': max(part['t1'] for part in parts), 'text': words(parts),
                      'speaker_label': line.get('speaker_label') or '', 'character_context_id': line.get('character_context_id'),
                      'character_name': line.get('character_name'), 'parts': parts, 'compiled': bool(line.get('compiled')),
                      'continues_next': bool(line.get('continues_next'))})
    for line, following in zip(lines, [*lines[1:], None]):
        # A reviewed compilation says itself whether a line continues; the raw transcript is judged from its wording.
        if not line['compiled']:
            line['continues_next'] = following is not None and sentence_continues(line, following)
    return lines


def sentence_continues(left, right):
    """True when ``right`` most likely continues the sentence ``left`` started: one voice, a short gap, an open ending."""
    if right['t0'] - left['t1'] >= CONTINUE_GAP:
        return False
    if left.get('speaker_label') and right.get('speaker_label') and left['speaker_label'] != right['speaker_label']:
        return False
    return not SENTENCE_END.search(str(left['text']).strip()) or str(right['text']).strip().startswith(PARTICLES)


def joined(left, right):
    """True when ``right`` goes on with the sentence ``left`` began, without a real pause between them."""
    return bool(left.get('continues_next')) and right['t0'] - left['t1'] < CONTINUE_GAP


def line_at(lines, t):
    """The speech line that is being spoken at ``t``, if any."""
    return next((line for line in lines if line['t0'] + 0.05 < t < line['t1'] - 0.05), None)


def sentence_at(lines, t):
    """``(left, right)`` when ``t`` lies in the gap between two lines that form one sentence, else None."""
    for left, right in zip(lines, lines[1:]):
        if joined(left, right) and left['t1'] - 0.05 <= t <= right['t0'] + 0.05:
            return left, right
    return None


ONSET = 0.2  # a boundary this close to the start or end of a line, with no word on that side, sits at the line's edge


def split_at(lines, t):
    """The speech a boundary at ``t`` divides: ``{lines, before, after}`` (words by midpoint), or None when it divides nothing.

    A boundary at the very start or end of a line (no word on one side) divides speech only when that
    line goes on with the sentence of its neighbour; then it lies between the two halves of that sentence.
    """
    crossing = line_at(lines, t)
    if crossing:
        before = [part for part in crossing['parts'] if (part['t0'] + part['t1']) / 2 < t]
        after = [part for part in crossing['parts'] if (part['t0'] + part['t1']) / 2 >= t]
        index = lines.index(crossing)
        if not before and t - crossing['t0'] < ONSET:
            if index and joined(lines[index - 1], crossing):
                return {'lines': [lines[index - 1]['id'], crossing['id']], 'before': lines[index - 1]['text'], 'after': crossing['text']}
            return None
        if not after and crossing['t1'] - t < ONSET:
            if index + 1 < len(lines) and joined(crossing, lines[index + 1]):
                return {'lines': [crossing['id'], lines[index + 1]['id']], 'before': crossing['text'], 'after': lines[index + 1]['text']}
            return None
        return {'lines': [crossing['id']], 'before': words(before), 'after': words(after)}
    halves = sentence_at(lines, t)
    if halves:
        return {'lines': [halves[0]['id'], halves[1]['id']], 'before': halves[0]['text'], 'after': halves[1]['text']}
    return None


# ------------------------------------------------------------ boundaries

def build(cuts, merged_lines, envelope=None, events=None):
    """The sequence timeline.

    ``cuts`` are the active cuts of the range in order (``id``, ``cut_index``, ``start_time``,
    ``end_time``, ``first_frame_time``). ``events`` maps a cut id to event start times in seconds from
    that cut's start (from the video analysis), used to divide cuts longer than one shot.
    """
    if not cuts:
        raise ValueError('A sequence needs at least one cut')
    origin = cuts[0]['start_time']
    duration = round(cuts[-1]['end_time'] - origin, 3)
    envelope = list(envelope or [])
    lines = speech_lines(merged_lines, origin, duration)
    pauses = [pause for pause in find_pauses(envelope) if pause[1] <= duration + EPSILON] if envelope else []
    spans = []
    for cut in cuts:
        spans.append({'id': cut['id'], 'number': cut['cut_index'] + 1, 't0': round(cut['start_time'] - origin, 3), 't1': round(cut['end_time'] - origin, 3),
                      'start_time': cut['start_time'], 'end_time': cut['end_time'], 'first_frame_time': cut.get('first_frame_time')})
    boundaries = [_boundary(number, left, right, lines, envelope, origin) for number, (left, right) in enumerate(zip(spans, spans[1:]), 1)]
    splits = {}
    for span in spans:
        if span['t1'] - span['t0'] > MAX_SHOT_SECONDS + EPSILON:
            found = _inside(span, lines, pauses, envelope, (events or {}).get(span['id']) or [], origin)
            for candidate in found:
                crossing = line_at(lines, candidate['t'])
                if crossing:
                    candidate['crosses_line'] = crossing['id']
                elif sentence_at(lines, candidate['t']):
                    candidate['splits_sentence'] = [line['id'] for line in sentence_at(lines, candidate['t'])]
            splits[span['id']] = found
    return {'origin': origin, 'duration': duration, 'cuts': spans, 'lines': lines, 'pauses': [list(pause) for pause in pauses if pause[1] - pause[0] >= 0.4],
            'boundaries': boundaries, 'splits': splits}


def _boundary(number, left, right, lines, envelope, origin):
    """The hard cut between two cuts and where its shot boundary may sit.

    ``keep_cut`` is the hard cut itself. When the cut falls inside a phrase (a spoken line, or the pause
    between two halves of one sentence), ``before_line`` and ``after_line`` put the boundary in the
    silence just before or just after the whole phrase, so it stays in one shot. Such a move is offered
    only up to ``SHIFT_SECONDS`` away, never taking more than ``OWN_SHARE`` of the cut it moves into and
    never making a shot of a cut that fits one take longer than that; ``unavailable`` says why an option is missing.
    """
    hard = left['t1']
    boundary = {'id': f'K{number}', 'after_cut': left['number'], 'before_cut': right['number'], 'after_cut_id': left['id'], 'before_cut_id': right['id'],
                't': hard, 'time': right['start_time'], 'speech': None, 'options': {'keep_cut': {'t': hard, 'time': right['start_time'], 'shift': 0.0}}, 'unavailable': {}}
    split = split_at(lines, hard)
    if not split:
        return boundary
    anchor = next(line for line in lines if line['id'] == split['lines'][0])
    first = last = lines.index(anchor)
    while first > 0 and joined(lines[first - 1], lines[first]):
        first -= 1
    while last + 1 < len(lines) and joined(lines[last], lines[last + 1]):
        last += 1
    phrase = lines[first:last + 1]
    boundary['speech'] = {'kind': 'inside_line' if len(split['lines']) == 1 else 'between_sentence_halves', 'lines': split['lines'],
                          'words_before': split['before'], 'words_after': split['after'], 'keeps': [line['id'] for line in phrase],
                          'phrase': [phrase[0]['t0'], phrase[-1]['t1']], 'speaker_label': anchor['speaker_label'],
                          'character_name': anchor.get('character_name'), 'character_context_id': anchor.get('character_context_id')}
    name = phrase[0]['id'] if len(phrase) == 1 else f"{phrase[0]['id']}–{phrase[-1]['id']}"

    def offer(choice, t, into, other):
        """Add an option unless it moves too far, takes too much of the cut ``into`` or makes the shot of ``other`` too long."""
        shift = round(t - hard, 3)
        own = into['t1'] - into['t0']
        grows = (t - other['t0']) if shift > 0 else (other['t1'] - t)
        if abs(shift) > SHIFT_SECONDS + EPSILON:
            boundary['unavailable'][choice] = f"{name} {'starts' if shift < 0 else 'ends'} {abs(shift):.2f} s {'before' if shift < 0 else 'after'} the hard cut; a boundary moves at most {SHIFT_SECONDS:g} s"
        elif abs(shift) > OWN_SHARE * own + EPSILON:
            boundary['unavailable'][choice] = f"it would take {abs(shift):.2f} s of cut {into['number']}'s {own:.2f} s picture; at most half of a cut may be given up"
        elif other['t1'] - other['t0'] <= MAX_SHOT_SECONDS + EPSILON and grows > MAX_SHOT_SECONDS + EPSILON:
            boundary['unavailable'][choice] = f"the shot of cut {other['number']} would last {grows:.1f} s; a shot lasts at most {MAX_SHOT_SECONDS:g} s"
        else:
            drops = {'cut': into['number'], 'from': round(t - into['t0'], 3), 'to': round(hard - into['t0'], 3)} if shift < 0 else {'cut': into['number'], 'from': 0.0, 'to': shift}
            boundary['options'][choice] = {'t': round(t, 3), 'time': round(origin + t, 6), 'shift': shift, 'drops': drops}

    start = phrase[0]['t0']
    prior = lines[first - 1]['t1'] if first else 0.0
    if first and start - prior <= 0.02:
        boundary['unavailable']['before_line'] = f"{phrase[0]['id']} follows {lines[first - 1]['id']} without a pause"
    else:
        offer('before_line', speech_edge(envelope, start, 'before', prior, start), left, right)
    end = phrase[-1]['t1']
    following = lines[last + 1]['t0'] if last + 1 < len(lines) else right['t1']
    if following - end <= 0.02:
        boundary['unavailable']['after_line'] = f"{lines[last + 1]['id']} follows {phrase[-1]['id']} without a pause" if last + 1 < len(lines) else f"{name} runs on to the end of cut {right['number']}"
    else:
        offer('after_line', speech_edge(envelope, end, 'after', end, following), right, left)
    return boundary


def _inside(span, lines, pauses, envelope, events, origin):
    """Places where a cut longer than one shot may be divided: speech gaps, silences, event starts, and quiet points where needed."""
    length = span['t1'] - span['t0']
    low, high = span['t0'] + EDGE_SECONDS, span['t1'] - EDGE_SECONDS
    found = []

    def offer(t, kind, pause=None):
        t = round(t, 3)
        if not low <= t <= high or line_at(lines, t) or sentence_at(lines, t) or any(abs(t - other['t']) < 0.75 for other in found):
            return
        found.append({'t': t, 'time': round(origin + t, 6), 'kind': kind, 'cut': span['number'], **({'pause': round(pause, 2)} if pause else {})})

    inside = sorted((line for line in lines if line['t1'] > span['t0'] and line['t0'] < span['t1']), key=lambda line: line['t0'])
    for before, after in zip(inside, inside[1:]):
        gap = after['t0'] - before['t1']
        if gap >= MIN_PAUSE:
            offer(quietest(envelope, before['t1'], after['t0']), 'pause', gap)
    for t0, t1 in pauses:
        if t1 - t0 >= MIN_PAUSE:
            offer(quietest(envelope, t0, t1), 'pause', t1 - t0)
    for seconds in events:
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            offer(span['t0'] + seconds, 'event')
    # Keep the list short: the longest pauses first, a handful per cut.
    limit = max(2, math.ceil(length / 4))
    found = sorted(sorted(found, key=lambda item: -(item.get('pause') or 0))[:limit], key=lambda item: item['t'])
    # A valid division must exist even when both of the cut's boundaries move out by the most they may.
    for _ in range(64):
        edges = [span['t0'] - SHIFT_SECONDS, *[item['t'] for item in found], span['t1'] + SHIFT_SECONDS]
        gap = max(zip(edges, edges[1:]), key=lambda pair: pair[1] - pair[0])
        if gap[1] - gap[0] <= MAX_SHOT_SECONDS:
            break
        quarter = (gap[1] - gap[0]) / 4
        a, b = max(low, gap[0] + quarter), min(high, gap[1] - quarter)
        t = quietest(envelope, a, b) if b > a else round((a + b) / 2, 3)
        spoken = line_at(lines, t)
        if spoken and a <= spoken['t1'] + 0.05 <= b:
            t = round(spoken['t1'] + 0.05, 3)  # rather after the line than inside it
        found = sorted([*found, {'t': t, 'time': round(origin + t, 6), 'kind': 'quiet', 'cut': span['number']}], key=lambda item: item['t'])
    return found


def _split_cost(candidate):
    """How bad a place is for dividing a long cut: inside speech is worst, a long pause best."""
    return (4.0 if candidate.get('crosses_line') or candidate.get('splits_sentence') else 0.0) - min(2.0, candidate.get('pause') or 0.0) \
        + {'quiet': 0.5, 'event': 0.25}.get(candidate['kind'], 0.0)


def partition(timeline, cut, start, end):
    """Edges ``[(t, time), …]`` of the parts of one cut's shot span ``start``–``end`` (each ``(t, time)``).

    A span that fits one take is one part. A longer one is divided at the fewest places, preferring
    pauses, never inside speech when that can be avoided, and as evenly as possible.
    """
    if end[0] - start[0] <= MAX_SHOT_SECONDS + EPSILON:
        return [start, end]
    points = [item for item in timeline['splits'].get(cut['id']) or [] if start[0] + MIN_SHOT_SECONDS < item['t'] < end[0] - MIN_SHOT_SECONDS]
    for count in range(1, min(len(points), 8) + 1):
        best = None
        for chosen in itertools.combinations(points, count):
            edges = [start[0], *[item['t'] for item in chosen], end[0]]
            parts = [b - a for a, b in zip(edges, edges[1:])]
            if max(parts) > MAX_SHOT_SECONDS + EPSILON:
                continue
            even = (end[0] - start[0]) / (count + 1)
            cost = sum(_split_cost(item) for item in chosen) + sum(abs(part - even) for part in parts) / 4
            if best is None or cost < best[0]:
                best = (cost, chosen)
        if best:
            return [start, *[(item['t'], item['time']) for item in best[1]], end]
    raise PlanError([f"cut {cut['number']} cannot be divided into shots of at most {MAX_SHOT_SECONDS:g} s"])


def unavailable(boundary, choice):
    """Why a boundary cannot take a choice, as a sentence."""
    if choice not in CHOICES:
        return f"{boundary['id']}: choose one of {', '.join(CHOICES)}"
    if not boundary['speech']:
        return f"{boundary['id']}: no speech crosses the hard cut between cut {boundary['after_cut']} and cut {boundary['before_cut']}, so its boundary stays on the cut"
    reason = boundary['unavailable'].get(choice) or 'there is no such position'
    return f"{boundary['id']}: {choice} is not possible here ({reason}); choose one of {', '.join(boundary['options'])}"


def decide(timeline, choices=None, intents=None, scenes=None):
    """The plan's spans: one shot per cut in order (a cut longer than one take in parts), each hard cut's boundary where ``choices`` puts it.

    ``choices`` maps boundary ids to ``keep_cut``, ``before_line`` or ``after_line`` (keep_cut when
    missing); ``intents`` and ``scenes`` map cut numbers to the text stored on that cut's shots.
    """
    choices = choices or {}
    edges = [(0.0, timeline['cuts'][0]['start_time'])]
    for boundary in timeline['boundaries']:
        option = boundary['options'].get(choices.get(boundary['id']) or 'keep_cut')
        if option is None:
            raise PlanError([unavailable(boundary, choices.get(boundary['id']))])
        edges.append((option['t'], option['time']))
    edges.append((timeline['duration'], timeline['cuts'][-1]['end_time']))
    return _spans(timeline, edges, intents or {}, scenes or {})


def _spans(timeline, edges, intents, scenes, keep=None):
    """Spans for each cut between its two edges; ``keep`` maps a cut id to parts that stay as they are."""
    spans = []
    for cut, start, end in zip(timeline['cuts'], edges, edges[1:]):
        if keep and cut['id'] in keep:
            spans.extend(keep[cut['id']])
            continue
        if end[0] - start[0] < MIN_SHOT_SECONDS:
            # No picture left: the span is kept so that check_spans can say which boundaries took it.
            spans.append({'start_time': start[1], 'end_time': end[1], 'frame_cut_id': cut['id'], 'scene': scenes.get(cut['number']), 'intent': intents.get(cut['number']) or ''})
            continue
        points = partition(timeline, cut, start, end)
        for (_, a), (_, b) in zip(points, points[1:]):
            spans.append({'start_time': a, 'end_time': b, 'frame_cut_id': cut['id'], 'scene': scenes.get(cut['number']), 'intent': intents.get(cut['number']) or ''})
    return spans


def owner_of(span, timeline):
    """The cut whose shot a span is: its ``frame_cut_id``, or the cut that fills most of it."""
    found = next((cut for cut in timeline['cuts'] if cut['id'] == span.get('frame_cut_id')), None)
    if found:
        return found
    t0, t1 = span['start_time'] - timeline['origin'], span['end_time'] - timeline['origin']
    return main_cut(timeline, t0, t1) if covered_cuts(timeline, t0, t1) else None


def groups(spans, timeline):
    """Consecutive spans grouped by the cut they belong to, as ``[(cut, [spans])]``."""
    found = []
    for span in spans:
        owner = owner_of(span, timeline)
        if found and owner is not None and found[-1][0] is not None and found[-1][0]['id'] == owner['id']:
            found[-1][1].append(span)
        else:
            found.append((owner, [span]))
    return found


def move_boundary(spans, timeline, boundary_id, choice):
    """The plan's spans with one hard cut's boundary moved to another of its options.

    The other boundaries, every intent and scene, and the parts of cuts that do not touch this boundary stay as they are.
    """
    problems = check_spans(spans, timeline)
    if problems:
        raise PlanError(problems)
    position = next((index for index, boundary in enumerate(timeline['boundaries']) if boundary['id'] == boundary_id), None)
    if position is None:
        raise PlanError([f'{boundary_id} is not a hard cut of this sequence'])
    boundary = timeline['boundaries'][position]
    option = boundary['options'].get(choice)
    if option is None:
        raise PlanError([unavailable(boundary, choice)])
    found = groups(spans, timeline)
    origin = timeline['origin']
    edges = [(parts[0]['start_time'] - origin, parts[0]['start_time']) for _, parts in found] + [(found[-1][1][-1]['end_time'] - origin, found[-1][1][-1]['end_time'])]
    edges[position + 1] = (option['t'], option['time'])
    keep = {cut['id']: parts for index, (cut, parts) in enumerate(found) if index not in (position, position + 1)}
    intents = {cut['number']: parts[0].get('intent') or '' for cut, parts in found}
    scenes = {cut['number']: parts[0].get('scene') for cut, parts in found}
    moved = _spans(timeline, edges, intents, scenes, keep)
    problems = check_spans(moved, timeline)
    if problems:
        raise PlanError(problems)
    return moved


# -------------------------------------------------------------------- plans

def covered_cuts(timeline, t0, t1):
    return [cut for cut in timeline['cuts'] if cut['t1'] > t0 + EPSILON and cut['t0'] < t1 - EPSILON]


def main_cut(timeline, t0, t1):
    """The covered cut that fills most of ``[t0, t1]`` (the first one on a tie)."""
    cuts = covered_cuts(timeline, t0, t1)
    return max(cuts, key=lambda cut: (round(min(cut['t1'], t1) - max(cut['t0'], t0), 3), -cut['t0']))


def check_spans(spans, timeline):
    """Problems of stored spans against the current cuts, as sentences a person or a director model can act on; empty when usable.

    One shot per cut in order (only a cut longer than one take may be several), every boundary on its
    hard cut unless it moved at most ``SHIFT_SECONDS`` into the silence around speech that crosses that
    cut, every cut keeping at least ``OWN_SHARE`` of its picture, and no shot longer than one take.
    """
    if not isinstance(spans, list) or not spans:
        return ['The plan has no shots']
    origin, errors = timeline['origin'], []
    for number, span in enumerate(spans, 1):
        if not isinstance(span, dict) or not all(isinstance(span.get(key), (int, float)) and not isinstance(span.get(key), bool) for key in ('start_time', 'end_time')):
            return [f'shot {number} needs numeric start_time and end_time']
    if abs(spans[0]['start_time'] - origin) > 0.001:
        errors.append('the first shot must start where the sequence starts')
    if abs(spans[-1]['end_time'] - (origin + timeline['duration'])) > 0.002:
        errors.append('the last shot must end where the sequence ends')
    for number, (span, following) in enumerate(zip(spans, spans[1:]), 1):
        if abs(span['end_time'] - following['start_time']) > 0.001:
            errors.append(f'shot {number} and shot {number + 1} must meet without a gap or overlap')
    if errors:
        return errors
    found = groups(spans, timeline)
    if [cut['id'] if cut else None for cut, _ in found] != [cut['id'] for cut in timeline['cuts']]:
        return [f'This plan merges, drops or reorders cuts: every cut is its own shot, in order (only a cut longer than {MAX_SHOT_SECONDS:g} s is divided). Plan the shots again']
    shifts = {}
    for (left, _), (right, parts), boundary in zip(found, found[1:], timeline['boundaries']):
        at = parts[0]['start_time'] - origin
        shift = at - boundary['t']
        if abs(shift) <= 0.002:
            continue
        name = f"{boundary['id']} (the hard cut between cut {left['number']} and cut {right['number']})"
        shifts[boundary['id']] = shift
        if not boundary['speech']:
            errors.append(f'{name} moved {shift:+.2f} s, but no speech crosses that cut: its boundary stays on the hard cut')
        elif abs(shift) > SHIFT_SECONDS + 0.002:
            errors.append(f'{name} moved {shift:+.2f} s; a boundary moves at most {SHIFT_SECONDS:g} s off its hard cut')
        elif line_at(timeline['lines'], at) or sentence_at(timeline['lines'], at):
            errors.append(f'{name} moved into speech at {at:.2f} s; a boundary leaves its hard cut only to keep a phrase whole')
    for index, (cut, parts) in enumerate(found):
        length = cut['t1'] - cut['t0']
        start, end = parts[0]['start_time'] - origin, parts[-1]['end_time'] - origin
        kept = min(end, cut['t1']) - max(start, cut['t0'])
        opening = timeline['boundaries'][index - 1] if index else None
        closing = timeline['boundaries'][index] if index < len(timeline['boundaries']) else None
        # Moves into this cut take picture from it; moves away from it lengthen its shot.
        into = [f"{item['id']} {choice}" for item, choice, sign in ((opening, 'after_line', 1), (closing, 'before_line', -1)) if item and shifts.get(item['id'], 0) * sign > 0]
        out = [f"{item['id']} {choice}" for item, choice, sign in ((opening, 'before_line', -1), (closing, 'after_line', 1)) if item and shifts.get(item['id'], 0) * sign > 0]
        if kept < OWN_SHARE * length - 0.002:
            errors.append(f"cut {cut['number']} keeps only {max(0.0, kept):.2f} s of its {length:.2f} s picture ({', '.join(into)}); keep {'one of these hard cuts' if len(into) > 1 else 'its hard cut'}")
            continue
        if len(parts) > 1 and length <= MAX_SHOT_SECONDS + EPSILON:
            errors.append(f"cut {cut['number']} lasts {length:.1f} s and is one shot; only a cut longer than {MAX_SHOT_SECONDS:g} s is divided")
            continue
        for part in parts:
            seconds = part['end_time'] - part['start_time']
            if seconds > MAX_SHOT_SECONDS + 0.001:
                errors.append(f"the shot of cut {cut['number']} lasts {seconds:.1f} s{' with ' + ', '.join(out) if out else ''}; a shot lasts at most {MAX_SHOT_SECONDS:g} s")
            elif seconds < MIN_SHOT_SECONDS:
                errors.append(f"a shot of cut {cut['number']} is shorter than {MIN_SHOT_SECONDS:g} s")
    return errors


def choices_of(spans, timeline):
    """The current choice at every hard cut of a usable plan, by boundary id."""
    found, origin = groups(spans, timeline), timeline['origin']
    result = {}
    for (_, parts), boundary in zip(found[1:], timeline['boundaries']):
        shift = parts[0]['start_time'] - origin - boundary['t']
        result[boundary['id']] = 'keep_cut' if abs(shift) <= 0.002 else ('before_line' if shift < 0 else 'after_line')
    return result


def shot_voices(voices, start_time, end_time):
    """The user's voice ranges of a plan (episode times) that fall inside one shot, on the shot's clock."""
    found = []
    for item in voices or []:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool) for key in ('from', 'to')):
            continue
        if item['to'] <= start_time + 0.001 or item['from'] >= end_time - 0.001:
            continue
        found.append({'speaker': item.get('speaker'), 'from': round(max(0.0, item['from'] - start_time), 3), 'to': round(min(end_time, item['to']) - start_time, 3), 'delivery': item.get('delivery')})
    return sorted(found, key=lambda item: (str(item['speaker']), item['from']))


def replace_voices(voices, speaker, start_time, end_time, segments):
    """The plan's voice ranges with one speaker's ranges inside one shot replaced.

    ``segments`` are ``[{from, delivery}]`` in seconds from the shot's start (the first one is moved to 0
    and each lasts until the next or the end of the shot); an empty list hands the shot back to the director.
    Ranges of other speakers, and the parts of this speaker's ranges outside the shot, are kept.
    """
    kept = []
    for item in voices or []:
        if item.get('speaker') != speaker or item['to'] <= start_time + 0.001 or item['from'] >= end_time - 0.001:
            kept.append(dict(item))
            continue
        if item['from'] < start_time - 0.001:
            kept.append({**item, 'to': round(start_time, 6)})
        if item['to'] > end_time + 0.001:
            kept.append({**item, 'from': round(end_time, 6)})
    ordered = sorted(segments or [], key=lambda segment: segment['from'])
    for index, segment in enumerate(ordered):
        start = start_time if index == 0 else start_time + segment['from']
        end = start_time + ordered[index + 1]['from'] if index + 1 < len(ordered) else end_time
        if end - start > 0.001:
            kept.append({'speaker': speaker, 'from': round(start, 6), 'to': round(end, 6), 'delivery': segment['delivery']})
    return sorted(kept, key=lambda item: (item['from'], str(item['speaker'])))


def _edge(timeline, owners, index, side, at, decisions):
    """How a shot meets its neighbour on one side.

    ``how`` is ``start``/``end`` (the sequence edge), ``continuation`` (the same long cut goes on),
    ``keep_cut`` (exactly on the hard cut), ``before_line`` or ``after_line`` (moved off the hard cut by
    ``shift`` seconds to keep a phrase whole). ``split`` is the speech the boundary divides, if any;
    ``reason`` and ``by`` come from the plan's decision for that hard cut.
    """
    if (side == 'opens' and index == 0) or (side == 'closes' and index == len(owners) - 1):
        edge = {'how': 'start' if side == 'opens' else 'end'}
    else:
        before, after = (owners[index - 1], owners[index]) if side == 'opens' else (owners[index], owners[index + 1])
        if before['id'] == after['id']:
            edge = {'how': 'continuation'}
        else:
            boundary = next(item for item in timeline['boundaries'] if item['after_cut_id'] == before['id'])
            shift = round(at - boundary['t'], 3)
            how = 'keep_cut' if abs(shift) <= 0.002 else ('before_line' if shift < 0 else 'after_line')
            edge = {'how': how, 'boundary': boundary['id'], 'shift': 0.0 if how == 'keep_cut' else shift, 'hard_cut': boundary['t']}
            if how != 'keep_cut' and boundary['speech']:
                edge['keeps'] = boundary['speech']['keeps']
            decided = decisions.get(before['id'])
            if decided and decided.get('choice') == how:
                edge.update({key: decided.get(key) for key in ('reason', 'by') if decided.get(key)})
    split = split_at(timeline['lines'], at) if edge['how'] not in ('start', 'end') else None
    if split:
        edge['split'] = split
    return edge


def expand(spans, timeline, voices=None, decisions=None):
    """Everything the app derives for each shot of a plan: times, its cut, the frame, the edges, lengths, dialogue and the user's voice ranges.

    ``voices`` are the plan's user-set voice ranges in episode times (``plan['voices']``) and
    ``decisions`` the plan's boundary decisions (``plan['boundaries']``), which supply each edge's reason.
    """
    origin, shots = timeline['origin'], []
    owners = [owner_of(span, timeline) for span in spans]
    reasons = {item.get('after_cut_id'): item for item in decisions or [] if isinstance(item, dict)}
    for index, span in enumerate(spans):
        t0, t1 = round(span['start_time'] - origin, 3), round(span['end_time'] - origin, 3)
        duration = round(t1 - t0, 3)
        cuts = covered_cuts(timeline, t0, t1)
        own = owners[index]
        if not cuts or own is None:
            raise PlanError([f'shot {index + 1} lies outside the sequence'])
        starts_mid = t0 - own['t0'] > 0.002
        if starts_mid:
            # A later moment of the cut, never analysed: the next part of a long cut, or the moment after a phrase the previous shot finished.
            continues = bool(shots) and shots[-1]['frame']['cut_id'] == own['id']
            frame = {'cut': own['number'], 'cut_id': own['id'], 'time': span['start_time'], 'strategy': 'frame_at_time', 'continues_previous_shot': continues}
        else:
            first = own.get('first_frame_time')
            time = first if isinstance(first, (int, float)) and own['start_time'] <= first < own['end_time'] else own['start_time']
            frame = {'cut': own['number'], 'cut_id': own['id'], 'time': time, 'strategy': 'cut_first_frame', 'continues_previous_shot': False}
        parts = [position for position, owner in enumerate(owners) if owner['id'] == own['id']]
        shots.append({'index': index + 1, 'key': shot_key(span['start_time'], span['end_time']), 't0': t0, 't1': t1,
                      'start_time': span['start_time'], 'end_time': span['end_time'], 'duration': duration,
                      'generate_seconds': round(max(duration, MIN_GENERATE_SECONDS), 3), 'trim_seconds': duration if duration < MIN_GENERATE_SECONDS else None,
                      'cut': own['number'], 'cut_id': own['id'], 'part': [parts.index(index) + 1, len(parts)] if len(parts) > 1 else None,
                      'source_cuts': [cut['number'] for cut in cuts], 'cut_ids': [cut['id'] for cut in cuts],
                      'merged': False, 'starts_mid_cut': starts_mid, 'ends_mid_cut': own['t1'] - t1 > 0.002, 'frame': frame,
                      'opens': _edge(timeline, owners, index, 'opens', t0, reasons), 'closes': _edge(timeline, owners, index, 'closes', t1, reasons),
                      'scene': span.get('scene'), 'intent': span.get('intent') or '', 'lines': shot_lines(timeline['lines'], t0, t1),
                      'fixed_voices': shot_voices(voices, span['start_time'], span['end_time'])})
    return shots


def shot_lines(lines, t0, t1):
    """The words spoken inside ``[t0, t1)`` on the shot's own clock; a line cut by a boundary is split by word."""
    spoken = []
    for line in lines:
        if line['t1'] <= t0 or line['t0'] >= t1:
            continue
        parts = [part for part in line['parts'] if t0 - EPSILON <= (part['t0'] + part['t1']) / 2 < t1]
        if not parts:
            continue
        spoken.append({'line': line['id'], 't0': round(max(0.0, parts[0]['t0'] - t0), 3), 't1': round(min(t1, max(part['t1'] for part in parts)) - t0, 3), 'text': words(parts),
                       'speaker_label': line['speaker_label'], 'character_context_id': line['character_context_id'], 'character_name': line['character_name'],
                       'compiled': line.get('compiled', False),
                       'crosses_start': len(parts) < len(line['parts']) and parts[0] is not line['parts'][0],
                       'crosses_end': len(parts) < len(line['parts']) and parts[-1] is not line['parts'][-1]})
    return spoken
