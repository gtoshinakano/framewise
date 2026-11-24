"""How the keyframe analysis and the performance beats read in prompts, and the adapter for older readers.

Stdlib only. ``keyframe_view`` and ``beats_view`` are the compact forms text prompts get (no ids, no keypoints,
no window bookkeeping). ``direction_from_keyframes`` presents a ``keyframes-v1`` result in the motion-direction
shape the video analysis had (``prompt_context.is_direction``/``video_view``), so the readers written for it
(sequence evidence, composition brief, ``/api/v1`` ``motion``, assistant) keep working until they are rewritten.
"""
from .keyframe_schema import is_no_change

KEYFRAME_VIEW = 'keyframes of the cut: each frame\'s state, then what changed between consecutive frames; times are seconds from the first frame'
BEATS_VIEW = 'performance beats; times are seconds from the cut\'s first frame'
SOURCE_WORDS = {'dwpose': 'keypoints', 'qwen': 'vision model'}
# keyframes-v1 camera moves in the motion-direction vocabulary (visual_contracts.CAMERA_MOVES).
CAMERA_WORDS = {'static': 'static', 'pan_left': 'pan left', 'pan_right': 'pan right', 'tilt_up': 'tilt up', 'tilt_down': 'tilt down',
                'zoom_in': 'zoom in', 'zoom_out': 'zoom out', 'dolly_in': 'push in', 'dolly_out': 'pull out', 'truck': 'tracking',
                'shake': 'handheld', 'orbit': 'other'}
EVENT_KIND = {'body_pose': 'body', 'arms': 'body', 'hands': 'body', 'legs': 'body', 'head': 'gaze', 'eyes': 'gaze', 'mouth': 'mouth',
              'expression': 'expression', 'position': 'travel'}
MAX_EVENTS = 40


def _present(**values):
    return {key: value for key, value in values.items() if value not in (None, '', [], {})}


def is_keyframes(content):
    return isinstance(content, dict) and content.get('contract') == 'keyframes-v1' and isinstance(content.get('segments'), list)


def is_beats(content):
    """Stored performance beats (a view made by ``beats_view`` is not)."""
    return isinstance(content, dict) and isinstance(content.get('beats'), list) and content.get('kind') != BEATS_VIEW


def _seconds(start, end):
    return f'{start:.2f}-{end:.2f}s'


def _changes(person, with_source):
    out = []
    for change in person.get('changes') or []:
        if not isinstance(change, dict) or is_no_change(change.get('description')):
            continue
        source = f" ({SOURCE_WORDS[change['source']]})" if with_source and change.get('source') in SOURCE_WORDS else ''
        out.append(f"{change.get('part')}: {change.get('description')}{source}")
    return out


def keyframe_view(content, level='review'):
    """A keyframe analysis as a prompt reads it.

    ``review`` (consistency summary, assistant) and ``beats`` (the performance beats) carry each frame's states,
    the body state DWPose measured and every pair's changes. ``brief`` (the Cut summary) carries
    only the changes. Anything that is not a keyframe analysis is returned as it is.
    """
    if not is_keyframes(content):
        return content
    frames = []
    if level != 'brief':
        for frame in content.get('frames') or []:
            people = []
            for person in frame.get('characters') or []:
                body = person.get('pose')  # DWPose's measured body: the review settles body state and sides with it
                people.append(_present(name=person.get('name'), state=person.get('state'), body_measured=body))
            frames.append(_present(frame=frame.get('label'), t=frame.get('t'), characters=[p for p in people if p.get('state') or p.get('body_measured')]))
    pairs = []
    for segment in content['segments']:
        people = []
        for person in segment.get('characters') or []:
            flags = [flag for flag in ('walking', 'speaking', 'blinking') if person.get(flag) == 'yes']
            changes = _changes(person, level != 'brief')
            if flags or changes:
                people.append(_present(name=person.get('name'), visible=', '.join(flags), changes=changes))
        others = [f"{item.get('kind')}: {item.get('description')}" for item in segment.get('other_changes') or [] if isinstance(item, dict) and not is_no_change(item.get('description'))]
        camera = (segment.get('camera') or {}).get('move')
        pairs.append(_present(pair=f"{segment.get('from')}→{segment.get('to')}", at=_seconds(segment.get('start', 0), segment.get('end', 0)),
                              camera=camera if camera and camera != 'unclear' else 'not measurable', characters=people, other=others))
    return _present(kind=KEYFRAME_VIEW, summary=content.get('summary'), frames=frames, changes=pairs,
                    measured='camera from the background pixels; body parts marked "keypoints" from body keypoints' if content.get('method') == 'hybrid' else 'camera from the background pixels')


def beats_view(content):
    """Performance beats as a prompt reads them (no ids, no provenance)."""
    if not is_beats(content):
        return content
    beats = []
    for beat in content['beats']:
        if not isinstance(beat, dict):
            continue
        camera = beat.get('camera') if isinstance(beat.get('camera'), dict) else {}
        people = [_present(name=person.get('character'), body=person.get('body'), face=person.get('face'), gaze=person.get('gaze'), mouth=person.get('mouth'))
                  for person in beat.get('characters') or [] if isinstance(person, dict)]
        speech = [_present(line=item.get('line'), speaker=item.get('speaker'), at=_seconds(item.get('start', 0), item.get('end', 0)), delivery=item.get('delivery'))
                  for item in beat.get('speech') or [] if isinstance(item, dict)]
        beats.append(_present(at=_seconds(beat.get('start', 0), beat.get('end', 0)), camera=_present(move=camera.get('move'), detail=camera.get('detail')),
                              intent=beat.get('intent'), characters=people, speech=speech))
    return _present(kind=BEATS_VIEW, summary=content.get('summary'), beats=beats)


# -- the adapter -------------------------------------------------------------------------------------------------------

def _join(items, empty):
    items = [item for item in items if item]
    return '; '.join(dict.fromkeys(items)) if items else empty


def _camera(segments):
    moves, spans = [], []
    for segment in segments:
        move = CAMERA_WORDS.get((segment.get('camera') or {}).get('move'))
        if not move:
            continue
        if not moves or moves[-1] != move:
            moves.append(move)
        if move != 'static':
            spans.append(f"{move} {segment.get('start', 0):.1f}-{segment.get('end', 0):.1f}s")
    return {'movement': ' → '.join(moves) or 'static', 'detail': 'measured from the background pixels' + (': ' + ', '.join(spans) if spans else '')}


def _people(content):
    """One motion-direction character per name, in order of appearance."""
    order, people = [], {}
    for frame in content.get('frames') or []:
        for person in frame.get('characters') or []:
            name = person.get('name')
            if name and name not in people:
                order.append(name)
                people[name] = {'character': name, 'character_context_id': person.get('character_context_id'), 'states': [], 'parts': {}, 'flags': {}}
            if name:
                people[name]['states'].append(person.get('state') or person.get('pose'))
    for segment in content['segments']:
        for person in segment.get('characters') or []:
            name = person.get('name')
            if not name:
                continue
            if name not in people:
                order.append(name)
                people[name] = {'character': name, 'character_context_id': person.get('character_context_id'), 'states': [], 'parts': {}, 'flags': {}}
            entry = people[name]
            for flag in ('walking', 'speaking', 'blinking'):
                entry['flags'].setdefault(flag, []).append(person.get(flag))
            for change in person.get('changes') or []:
                if isinstance(change, dict) and not is_no_change(change.get('description')):
                    entry['parts'].setdefault(change.get('part'), []).append(change.get('description'))
    out = []
    for name in order:
        entry = people[name]
        states = [state for state in entry['states'] if state]
        flags, parts = entry['flags'], entry['parts']
        speaking = flags.get('speaking') or []
        spoken = 'yes' if 'yes' in speaking else 'no' if speaking and all(value == 'no' for value in speaking) else 'unclear'
        positions = parts.get('position') or []
        leaves = any(str(text).lower().startswith('leaves') for text in positions)
        activity = states[0] if states else ''
        if 'yes' in (flags.get('walking') or []):
            activity = 'walking' + (f'; {activity}' if activity else '')
        out.append(_present(character=name, character_context_id=entry['character_context_id'], activity=activity or 'not readable',
                            travel=_join(positions, 'none'), motion=_join([*parts.get('body_pose', []), *parts.get('arms', []), *parts.get('hands', []), *parts.get('legs', [])], 'holds still'),
                            speaking=spoken, emotion=_join(parts.get('expression', []), 'not readable'),
                            gaze=_join([*parts.get('head', []), *parts.get('eyes', [])], ''),
                            end='leaves the frame' if leaves else (states[-1] if states else '')))
    return out


def _events(content):
    events = []
    for segment in content['segments']:
        start, end = segment.get('start', 0), segment.get('end', 0)
        move = (segment.get('camera') or {}).get('move')
        if move in CAMERA_WORDS and move != 'static':
            events.append({'start_seconds': start, 'end_seconds': end, 'subject': 'camera', 'visual_action': f'camera: {CAMERA_WORDS[move]}', 'kind': 'camera'})
        for person in segment.get('characters') or []:
            name = person.get('name')
            for flag, kind, text in (('walking', 'travel', 'walks'), ('speaking', 'mouth', 'mouth moves (speaking)'), ('blinking', 'expression', 'blinks')):
                if person.get(flag) == 'yes':
                    events.append(_present(start_seconds=start, end_seconds=end, subject=name, character_context_id=person.get('character_context_id'), visual_action=text, kind=kind))
            for change in person.get('changes') or []:
                if isinstance(change, dict) and not is_no_change(change.get('description')):
                    events.append(_present(start_seconds=start, end_seconds=end, subject=name, character_context_id=person.get('character_context_id'),
                                           visual_action=f"{change.get('part')}: {change.get('description')}", kind=EVENT_KIND.get(change.get('part'), 'body')))
        for other in segment.get('other_changes') or []:
            if isinstance(other, dict) and not is_no_change(other.get('description')):
                events.append({'start_seconds': start, 'end_seconds': end, 'subject': other.get('kind') or 'scene', 'visual_action': other.get('description'),
                               'kind': 'object' if other.get('kind') == 'object' else 'body'})
    return events[:MAX_EVENTS]


def direction_from_keyframes(content, beats=None):
    """A ``keyframes-v1`` result in the motion-direction shape (``prompt_context.is_direction`` is true for it).

    Event times stay clip-relative; the one window carries the cut's episode times (``content.cut``) like a video
    window did. ``beats`` (current performance beats) give the summary when present.
    """
    if not is_keyframes(content):
        return None
    segments = content['segments']
    cut = content.get('cut') if isinstance(content.get('cut'), dict) else {}
    cut_start = cut.get('start_time') if isinstance(cut.get('start_time'), (int, float)) else 0.0
    duration = cut['end_time'] - cut_start if isinstance(cut.get('end_time'), (int, float)) else (segments[-1].get('end', 0) if segments else 0)
    summary = (beats or {}).get('summary') if is_beats(beats) else None
    return {'contract': 'direction-from-keyframes-v1', 'source': {'type': 'keyframe_analysis', 'run_id': content.get('run_id'), 'method': content.get('method')},
            'direction': {'summary': summary or content.get('summary') or '', 'camera': _camera(segments), 'characters': _people(content), 'end_state': {}},
            'windows': [{'source': {'source_start': cut_start, 'source_end': cut_start + duration},
                         'analysis': {'events': _events(content), 'uncertainties': ''}}]}
