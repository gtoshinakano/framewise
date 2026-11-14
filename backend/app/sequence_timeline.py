"""Timeline arithmetic for sequences: everything a model must not be asked to calculate.

A sequence is a contiguous range of cuts played on the original audio. This module turns the cuts,
the transcript and the audio level into one clock (seconds from the first cut), speech lines
``L1…``, pauses and candidate shot boundaries ``B0…``. A director model only chooses among those
candidates; coverage, the 15-second shot limit, the initial frame time, the audio interval and the
dialogue of each shot are computed and checked here. Nothing in this file touches the database.
"""
import array
import math
import wave
from functools import lru_cache
from operator import mul
from pathlib import Path

MAX_SEQUENCE_SECONDS = 180.0
MAX_SHOT_SECONDS = 15.0       # one H3 take (assistant.h3_prompt.MAX_SECONDS)
MIN_GENERATE_SECONDS = 5.0    # H3 is trained for about 5-15 s; shorter shots are generated longer and trimmed
SPLIT_FROM_SECONDS = 8.0      # cuts at least this long are offered boundaries inside them
EDGE_SECONDS = 2.5            # no boundary this close to a hard cut: a split must not leave a sliver of a shot
MIN_PAUSE = 0.3
WINDOW = 0.05
EPSILON = 1e-6


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
                parts.append({'t0': round(max(0.0, part['start_time'] - origin), 3), 't1': round(min(duration, part['end_time'] - origin), 3), 'text': text})
        if not parts:
            continue
        text = ''
        for part in parts:
            text = join_text(text, part['text'])
        lines.append({'id': f'L{len(lines) + 1}', 't0': parts[0]['t0'], 't1': max(part['t1'] for part in parts), 'text': text,
                      'speaker_label': line.get('speaker_label') or '', 'character_context_id': line.get('character_context_id'),
                      'character_name': line.get('character_name'), 'parts': parts})
    return lines


def line_at(lines, t):
    """The speech line that is being spoken at ``t``, if any."""
    return next((line for line in lines if line['t0'] + 0.05 < t < line['t1'] - 0.05), None)


# -------------------------------------------------------------- candidates

def build(cuts, merged_lines, envelope=None, events=None):
    """The sequence timeline.

    ``cuts`` are the active cuts of the range in order (``id``, ``cut_index``, ``start_time``,
    ``end_time``, ``first_frame_time``). ``events`` maps a cut id to event start times in seconds from
    that cut's start (from the video analysis), used as boundaries only inside very long cuts.
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
    candidates = [{'t': 0.0, 'time': cuts[0]['start_time'], 'kind': 'start', 'cut': spans[0]['number']}]
    for span in spans:
        candidates.extend(_inside(span, lines, pauses, envelope, (events or {}).get(span['id']) or [], origin))
        last = span is spans[-1]
        candidates.append({'t': span['t1'], 'time': span['end_time'], 'kind': 'end' if last else 'cut', 'cut': None if last else span['number'] + 1})
    for index, candidate in enumerate(candidates):
        candidate['id'] = f'B{index}'
        crossing = line_at(lines, candidate['t']) if candidate['kind'] == 'cut' else None
        if crossing:
            candidate['crosses_line'] = crossing['id']
    return {'origin': origin, 'duration': duration, 'cuts': spans, 'lines': lines, 'pauses': [list(pause) for pause in pauses if pause[1] - pause[0] >= 0.4], 'candidates': candidates}


def _inside(span, lines, pauses, envelope, events, origin):
    """Boundary candidates inside one cut: speech gaps, silences, event starts, and quiet points where needed."""
    length = span['t1'] - span['t0']
    if length < SPLIT_FROM_SECONDS:
        return []
    low, high = span['t0'] + EDGE_SECONDS, span['t1'] - EDGE_SECONDS
    found = []

    def offer(t, kind, pause=None):
        t = round(t, 3)
        if not low <= t <= high or line_at(lines, t) or any(abs(t - other['t']) < 0.75 for other in found):
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
    if length > MAX_SHOT_SECONDS:
        for seconds in events:
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                offer(span['t0'] + seconds, 'event')
    # Keep the list short: the longest pauses first, a handful per cut.
    limit = max(2, math.ceil(length / 4))
    found = sorted(sorted(found, key=lambda item: -(item.get('pause') or 0))[:limit], key=lambda item: item['t'])
    # A valid partition must exist: no stretch between neighbouring candidates may exceed one shot.
    while True:
        edges = [span['t0'], *[item['t'] for item in found], span['t1']]
        gap = max(zip(edges, edges[1:]), key=lambda pair: pair[1] - pair[0])
        if gap[1] - gap[0] <= MAX_SHOT_SECONDS:
            return found
        quarter = (gap[1] - gap[0]) / 4
        t = quietest(envelope, gap[0] + quarter, gap[1] - quarter)
        spoken = line_at(lines, t)
        if spoken and gap[0] + quarter <= spoken['t1'] + 0.05 <= gap[1] - quarter:
            t = round(spoken['t1'] + 0.05, 3)  # rather after the line than inside it
        found = sorted([*found, {'t': t, 'time': round(origin + t, 6), 'kind': 'quiet', 'cut': span['number']}], key=lambda item: item['t'])


def describe(candidate):
    """A candidate as an error message names it, e.g. ``B12 (pause 0.6 s, 41.3 s)``."""
    kind = f"pause {candidate['pause']:g} s" if candidate.get('pause') else {'cut': 'hard cut', 'event': 'action change', 'quiet': 'quiet point'}.get(candidate['kind'], candidate['kind'])
    return f"{candidate['id']} ({kind}, {candidate['t']:g} s)"


# -------------------------------------------------------------------- plans

def covered_cuts(timeline, t0, t1):
    return [cut for cut in timeline['cuts'] if cut['t1'] > t0 + EPSILON and cut['t0'] < t1 - EPSILON]


def check_model_plan(shots, timeline, strict=True):
    """Problems of a plan written with candidate ids, as sentences the model can fix. Empty when it is usable."""
    by_id = {candidate['id']: candidate for candidate in timeline['candidates']}
    last_id = timeline['candidates'][-1]['id']
    if not isinstance(shots, list) or not shots:
        return ['"shots" must be a non-empty list']
    errors = []
    for number, shot in enumerate(shots, 1):
        if not isinstance(shot, dict) or shot.get('from') not in by_id or shot.get('to') not in by_id:
            errors.append(f'shot {number}: "from" and "to" must be candidate ids from the list (B0…{last_id})')
            continue
        start, end = by_id[shot['from']], by_id[shot['to']]
        if end['t'] <= start['t']:
            errors.append(f"shot {number}: \"to\" ({end['id']}) must come after \"from\" ({start['id']})")
            continue
        length = end['t'] - start['t']
        if length > MAX_SHOT_SECONDS + EPSILON:
            between = [describe(candidate) for candidate in timeline['candidates'] if start['t'] < candidate['t'] < end['t']]
            errors.append(f"shot {number} ({start['id']}→{end['id']}) is {length:.1f} s; a shot may last at most {MAX_SHOT_SECONDS:g} s. End it earlier at one of: {', '.join(between)}")
        cuts = covered_cuts(timeline, start['t'], end['t'])
        numbers = [cut['number'] for cut in cuts]
        if shot.get('frame_cut') is not None and shot['frame_cut'] not in numbers:
            errors.append(f"shot {number}: frame_cut {shot['frame_cut']!r} is not one of the cuts this shot covers ({', '.join(map(str, numbers))})")
        reshaped = len(cuts) > 1 or start['kind'] not in ('start', 'cut') or end['kind'] not in ('cut', 'end')
        if strict and reshaped and not str(shot.get('staging') or '').strip():
            errors.append(f'shot {number} merges or splits source cuts; describe in "staging" how one continuous take covers it')
        if strict and not str(shot.get('intent') or '').strip():
            errors.append(f'shot {number}: "intent" is empty')
    if errors:
        return errors
    if shots[0]['from'] != 'B0':
        errors.append('the first shot must start at B0')
    if shots[-1]['to'] != last_id:
        errors.append(f'the last shot must end at {last_id}, the end of the sequence')
    for number, (shot, following) in enumerate(zip(shots, shots[1:]), 1):
        if shot['to'] != following['from']:
            errors.append(f"shot {number} ends at {shot['to']} but shot {number + 1} starts at {following['from']}; shots must follow each other without gaps or overlaps")
    return errors


def auto_fix(shots, timeline):
    """A valid plan as close as possible to an invalid one: keep its usable boundaries, add the best missing ones."""
    candidates = timeline['candidates']
    by_id = {candidate['id']: candidate for candidate in candidates}
    proposed = [shot for shot in shots if isinstance(shot, dict) and shot.get('from') in by_id and shot.get('to') in by_id] if isinstance(shots, list) else []
    chosen = {0.0, candidates[-1]['t']}
    for shot in proposed:
        chosen.update((by_id[shot['from']]['t'], by_id[shot['to']]['t']))
    if not proposed:
        chosen.update(candidate['t'] for candidate in candidates if candidate['kind'] == 'cut')  # fall back to the source cuts
    edges = sorted(chosen)
    while True:
        gap = next(((a, b) for a, b in zip(edges, edges[1:]) if b - a > MAX_SHOT_SECONDS + EPSILON), None)
        if gap is None:
            break
        inside = [candidate for candidate in candidates if gap[0] < candidate['t'] < gap[1]]
        middle = (gap[0] + gap[1]) / 2
        # Hard cuts first, then long pauses, each as close to the middle as possible.
        best = min(inside, key=lambda candidate: (abs(candidate['t'] - middle) - (3 if candidate['kind'] == 'cut' else 0) - min(2.0, candidate.get('pause') or 0)))
        edges = sorted({*edges, best['t']})
    by_time = {candidate['t']: candidate for candidate in candidates}
    fixed = []
    for start, end in zip(edges, edges[1:]):
        source = next((shot for shot in proposed if by_id[shot['from']]['t'] == start and by_id[shot['to']]['t'] == end), None)
        entry = dict(source) if source else {}
        if source is None:
            middle = (start + end) / 2
            near = next((shot for shot in proposed if by_id[shot['from']]['t'] <= middle < by_id[shot['to']]['t']), None) or {}
            entry = {'scene': near.get('scene'), 'intent': near.get('intent') or 'Follow the source cuts.', 'staging': near.get('staging') or '', 'auto_fixed': True}
        entry['from'], entry['to'] = by_time[start]['id'], by_time[end]['id']
        numbers = [cut['number'] for cut in covered_cuts(timeline, start, end)]
        if entry.get('frame_cut') not in numbers:
            entry['frame_cut'] = numbers[0]
        fixed.append(entry)
    return fixed


def to_spans(shots, timeline):
    """A checked candidate-id plan as stored spans: episode times and the id of the cut that supplies the frame."""
    by_id = {candidate['id']: candidate for candidate in timeline['candidates']}
    spans = []
    for shot in shots:
        start, end = by_id[shot['from']], by_id[shot['to']]
        cuts = covered_cuts(timeline, start['t'], end['t'])
        frame = next((cut for cut in cuts if cut['number'] == shot.get('frame_cut')), cuts[0])
        span = {'start_time': start['time'], 'end_time': end['time'], 'frame_cut_id': frame['id'],
                'scene': shot.get('scene') or None, 'intent': str(shot.get('intent') or '').strip(), 'staging': str(shot.get('staging') or '').strip()}
        if shot.get('auto_fixed'):
            span['auto_fixed'] = True
        spans.append(span)
    return spans


def check_spans(spans, timeline):
    """Problems of stored spans against the current cuts: coverage, order and the shot limit."""
    if not isinstance(spans, list) or not spans:
        return ['The plan has no shots']
    origin, errors = timeline['origin'], []
    for number, span in enumerate(spans, 1):
        if not isinstance(span, dict) or not all(isinstance(span.get(key), (int, float)) and not isinstance(span.get(key), bool) for key in ('start_time', 'end_time')):
            return [f'shot {number} needs numeric start_time and end_time']
        if span['end_time'] - span['start_time'] < 0.2:
            errors.append(f'shot {number} is shorter than 0.2 s')
        if span['end_time'] - span['start_time'] > MAX_SHOT_SECONDS + 0.001:
            errors.append(f"shot {number} is {span['end_time'] - span['start_time']:.1f} s; a shot may last at most {MAX_SHOT_SECONDS:g} s")
    if abs(spans[0]['start_time'] - origin) > 0.001:
        errors.append('the first shot must start where the sequence starts')
    if abs(spans[-1]['end_time'] - (origin + timeline['duration'])) > 0.002:
        errors.append('the last shot must end where the sequence ends')
    for number, (span, following) in enumerate(zip(spans, spans[1:]), 1):
        if abs(span['end_time'] - following['start_time']) > 0.001:
            errors.append(f'shot {number} and shot {number + 1} must meet without a gap or overlap')
    return errors


def expand(spans, timeline):
    """Everything the app derives for each shot of a plan: times, source cuts, the frame, lengths and dialogue."""
    origin, shots = timeline['origin'], []
    for index, span in enumerate(spans):
        t0, t1 = round(span['start_time'] - origin, 3), round(span['end_time'] - origin, 3)
        duration = round(t1 - t0, 3)
        cuts = covered_cuts(timeline, t0, t1)
        if not cuts:
            raise PlanError([f'shot {index + 1} lies outside the sequence'])
        frame_cut = next((cut for cut in cuts if cut['id'] == span.get('frame_cut_id')), cuts[0])
        starts_mid = t0 - cuts[0]['t0'] > 0.002
        ends_mid = cuts[-1]['t1'] - t1 > 0.002
        if frame_cut is cuts[0] and starts_mid:
            # The take continues the previous shot inside one source cut; the frame at this moment was never analysed.
            frame = {'cut': frame_cut['number'], 'cut_id': frame_cut['id'], 'time': span['start_time'], 'strategy': 'frame_at_time', 'continues_previous_shot': True}
        else:
            first = frame_cut.get('first_frame_time')
            time = first if isinstance(first, (int, float)) and frame_cut['start_time'] <= first < frame_cut['end_time'] else frame_cut['start_time']
            frame = {'cut': frame_cut['number'], 'cut_id': frame_cut['id'], 'time': time, 'strategy': 'cut_first_frame', 'continues_previous_shot': False}
        shots.append({'index': index + 1, 'key': shot_key(span['start_time'], span['end_time']), 't0': t0, 't1': t1,
                      'start_time': span['start_time'], 'end_time': span['end_time'], 'duration': duration,
                      'generate_seconds': round(max(duration, MIN_GENERATE_SECONDS), 3), 'trim_seconds': duration if duration < MIN_GENERATE_SECONDS else None,
                      'source_cuts': [cut['number'] for cut in cuts], 'cut_ids': [cut['id'] for cut in cuts],
                      'merged': len(cuts) > 1, 'starts_mid_cut': starts_mid, 'ends_mid_cut': ends_mid, 'frame': frame,
                      'scene': span.get('scene'), 'intent': span.get('intent') or '', 'staging': span.get('staging') or '',
                      'auto_fixed': bool(span.get('auto_fixed')), 'lines': shot_lines(timeline['lines'], t0, t1)})
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
        text = ''
        for part in parts:
            text = join_text(text, part['text'])
        spoken.append({'line': line['id'], 't0': round(max(0.0, parts[0]['t0'] - t0), 3), 't1': round(min(t1, max(part['t1'] for part in parts)) - t0, 3), 'text': text,
                       'speaker_label': line['speaker_label'], 'character_context_id': line['character_context_id'], 'character_name': line['character_name'],
                       'crosses_start': len(parts) < len(line['parts']) and parts[0] is not line['parts'][0],
                       'crosses_end': len(parts) < len(line['parts']) and parts[-1] is not line['parts'][-1]})
    return spoken


def merge_spans(spans, index):
    """Join shot ``index`` (0-based) with the one after it."""
    if not 0 <= index < len(spans) - 1:
        raise PlanError(['There is no following shot to merge with'])
    first, second = spans[index], spans[index + 1]
    if second['end_time'] - first['start_time'] > MAX_SHOT_SECONDS + 0.001:
        raise PlanError([f"The merged shot would last {second['end_time'] - first['start_time']:.1f} s; a shot may last at most {MAX_SHOT_SECONDS:g} s"])
    merged = {**first, 'end_time': second['end_time'], 'staging': ' '.join(part for part in (first.get('staging'), second.get('staging')) if part),
              'intent': ' '.join(part for part in (first.get('intent'), second.get('intent')) if part)}
    merged.pop('auto_fixed', None)
    return [*spans[:index], merged, *spans[index + 2:]]


def split_span(spans, index, time):
    """Cut shot ``index`` (0-based) in two at an episode time inside it."""
    if not 0 <= index < len(spans):
        raise PlanError(['That shot does not exist'])
    span = spans[index]
    if not isinstance(time, (int, float)) or isinstance(time, bool) or not span['start_time'] + 0.5 <= time <= span['end_time'] - 0.5:
        raise PlanError(['Choose a split time at least 0.5 s inside the shot'])
    first = {**span, 'end_time': time}
    second = {**span, 'start_time': time, 'frame_cut_id': None}
    for part in (first, second):
        part.pop('auto_fixed', None)
    return [*spans[:index], first, second, *spans[index + 1:]]
