"""The reference keyframe of a shot (2026-11-22): one frame of the shot's own picture that the user picks on the Shots tab.

For a scene that words alone stage poorly (several people, props, a busy setting) the user chooses one keyframe of the
shot's cut. A rewrite of the shot then:

- gives the director that frame's analysis (``evidence``: the keyframe states, the measured body, the changes into and out
  of it, the Target-frame boxes as screen side and size) and, when the text engine can read images, the frame itself;
- adds the frame to the H3 prompt as one more ``<Picture N>`` after the others, with the app's sentences that take only
  the staging from it (``sequence_prompt.reference_definition``); its look is never a reference;
- exports it in the download as ``shot-NN-reference.jpg`` (the exact source frame, by its ordinal).

The choice is stored on the shot revision as ``content.reference_frame`` (``{cut_id, ordinal, time, at, frame}``), set by
the app and never by the model, and every later rewrite of the shot keeps it until the user picks another frame or clears
it (``resolve``). ``ordinal`` is the source frame index, so the exact frame is cut even after the keyframe selection changes.
No FastAPI here: the inference worker imports this module.
"""
import json

CLEAR = {'clear': True}  # what a job payload carries when the user cleared the reference
EDGE = 0.001  # a frame on the shot's last instant still belongs to it


def span(shot, cut):
    """The shot's own picture in episode seconds: the part of its cut it plays (a borrowed neighbour's audio has no picture)."""
    return max(shot['start_time'], cut['start_time']), min(shot['end_time'], cut['end_time'])


def selection(conn, cut_id):
    """``(cut, key, frames)``: the cut's current keyframe selection, frames sorted and labelled F1…Fn as everywhere else."""
    from . import keyframes  # imports OpenCV; only the callers that need the selection pay for it
    cut, _, key, result = keyframes.current_selection(cut_id, conn)
    frames = sorted((result or {}).get('selected_frames') or [], key=lambda item: item['source_frame_index'])
    return cut, key, [{'frame': f'F{number}', 'ordinal': item['source_frame_index'], 'time': float(item['timestamp_seconds'])}
                      for number, item in enumerate(frames, 1)]


def analysed_ordinals(results):
    """Frame ordinals the cut's keyframe analysis described."""
    content = (results.get('keyframe_analysis') or {}).get('content') or {}
    return {frame.get('ordinal') for frame in content.get('frames') or [] if isinstance(frame, dict)}


def candidates(conn, shot, results=None):
    """What the Shots tab offers for one shot: the current keyframes of its cut that lie inside the shot's own picture.

    Returns ``{cut_id, revision, status, frames}``; each frame has ``frame`` (F label), ``ordinal``, ``time`` (episode
    seconds), ``at`` (seconds on the shot's clock) and ``analysed`` (the keyframe analysis has states for it).
    """
    cut, key, frames = selection(conn, shot['cut_id'])
    if cut is None:
        return {'cut_id': shot['cut_id'], 'revision': None, 'status': 'missing', 'frames': []}
    start, end = span(shot, cut)
    analysed = analysed_ordinals(results or {})
    inside = [{**item, 'at': round(item['time'] - shot['start_time'], 3), 'analysed': item['ordinal'] in analysed}
              for item in frames if start - EDGE <= item['time'] < end + EDGE]
    return {'cut_id': shot['cut_id'], 'revision': key, 'status': 'current' if key else 'missing', 'frames': inside}


def request(conn, shot, ordinal):
    """The payload of a user's choice: ``{cut_id, ordinal, time}``. Raises ``ValueError`` with a sentence for the user."""
    cut, key, frames = selection(conn, shot['cut_id'])
    if cut is None or not key:
        raise ValueError(f"Cut {shot['cut']} has no current keyframes; select its keyframes again before choosing a reference")
    item = next((frame for frame in frames if frame['ordinal'] == ordinal), None)
    if item is None:
        raise ValueError(f"Frame {ordinal} is not one of the current keyframes of cut {shot['cut']}")
    start, end = span(shot, cut)
    if not start - EDGE <= item['time'] < end + EDGE:
        raise ValueError(f"{item['frame']} of cut {shot['cut']} is outside shot {shot['index']}; choose a keyframe this shot shows")
    return {'cut_id': shot['cut_id'], 'ordinal': item['ordinal'], 'time': item['time']}


def _usable(value):
    return (isinstance(value, dict) and isinstance(value.get('cut_id'), str) and isinstance(value.get('ordinal'), int) and not isinstance(value.get('ordinal'), bool)
            and isinstance(value.get('time'), (int, float)) and not isinstance(value.get('time'), bool))


def resolve(conn, shot, chosen, previous):
    """The reference a shot job writes with, and where it came from.

    ``chosen`` is the job payload's ``reference_frame`` (``{cut_id, ordinal, time}``, ``CLEAR`` or None) and ``previous``
    the content of the shot's current revision. Returns ``(reference or None, source)``; source is ``user`` (chosen for
    this rewrite), ``kept`` (inherited from the revision before), ``cleared``, ``dropped: why`` or None (never had one).
    """
    if isinstance(chosen, dict) and chosen.get('clear'):
        return None, 'cleared'
    source, value = ('user', chosen) if _usable(chosen) else ('kept', (previous or {}).get('reference_frame'))
    if not _usable(value):
        return None, None
    cut, _, frames = selection(conn, shot['cut_id'])
    if value['cut_id'] != shot['cut_id'] or cut is None:
        return None, 'dropped: the reference keyframe belongs to another cut'
    start, end = span(shot, cut)
    if not start - EDGE <= value['time'] < end + EDGE:
        return None, 'dropped: the reference keyframe is no longer inside this shot'
    label = next((item['frame'] for item in frames if item['ordinal'] == value['ordinal']), None) or value.get('frame') or 'reference'
    return {'cut_id': value['cut_id'], 'ordinal': value['ordinal'], 'time': round(float(value['time']), 6),
            'at': round(float(value['time']) - shot['start_time'], 3), 'frame': label}, source


def horizontal(cx):
    """Screen side of a box centre (0-1 of the width), in the still vocabulary's viewer-relative words."""
    from .keyframe_rules import horizontal as phrase
    return phrase(cx)


def boxes(conn, cut_id, ordinal):
    """The Target-frame boxes on one frame: ``[{name, position, height_in_frame}]``, from the cut's latest run, ignored colours left out."""
    run = conn.execute('SELECT id,frames_json FROM target_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut_id,)).fetchone()
    if not run:
        return []
    frame = next((item for item in json.loads(run['frames_json'] or '[]') if isinstance(item, dict) and item.get('ordinal') == ordinal), None)
    size = (frame or {}).get('image_size')
    if not size or len(size) != 2 or not size[0] or not size[1]:
        return []
    out = []
    for row in conn.execute('''SELECT d.box_json,c.title AS name FROM target_detections d
                               JOIN target_slots s ON s.run_id=d.run_id AND s.slot=d.slot
                               LEFT JOIN context_items c ON c.id=s.character_context_id
                               WHERE d.run_id=? AND d.frame_ordinal=? AND s.ignored=0 ORDER BY d.slot''', (run['id'], ordinal)):
        box = json.loads(row['box_json'] or '[]')
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        out.append({'name': row['name'] or 'unnamed person', 'position': horizontal((x1 + x2) / 2 / size[0]),
                    'height_in_frame': round(max(0.0, min(1.0, (y2 - y1) / size[1])), 2)})
    return out


def evidence(conn, results, reference):
    """What the analysis found in the reference frame, for the director (``reference_frame.shows``); empty parts are left out.

    Per person: the state the vision model read and the body DWPose measured (keyframe analysis), the screen side and the
    share of the frame height their Target-frame box fills. Also the changes into and out of the frame, without times
    (the frame's own second is given beside it).
    """
    from . import prompt_context
    view = prompt_context.keyframe_view((results.get('keyframe_analysis') or {}).get('content'), 'beats') or {}
    content = (results.get('keyframe_analysis') or {}).get('content') or {}
    label = next((frame.get('label') for frame in content.get('frames') or [] if isinstance(frame, dict) and frame.get('ordinal') == reference['ordinal']), None)
    people = {}
    if label:
        for frame in view.get('frames') or []:
            if frame.get('frame') == label:
                for person in frame.get('characters') or []:
                    people[(person.get('name') or '').casefold()] = {key: value for key, value in person.items() if value}
    for box in boxes(conn, reference['cut_id'], reference['ordinal']):
        key = box['name'].casefold() if box['name'] != 'unnamed person' else f"unnamed {len(people)}"
        people.setdefault(key, {'name': box['name']}).update(position=box['position'], height_in_frame=box['height_in_frame'])
    into, after = [], []
    if label:
        for change in view.get('changes') or []:
            first, _, second = str(change.get('pair') or '').partition('→')
            item = {key: value for key, value in change.items() if key not in ('pair', 'at')}
            if second == label:
                into.append({'from': first, **item})
            elif first == label:
                after.append({'to': second, **item})
    shows = {'people': list(people.values()) or None, 'changes_into_this_frame': into or None, 'changes_after_this_frame': after or None}
    return {key: value for key, value in shows.items() if value}
