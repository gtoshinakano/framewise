"""What the consistency reviewer looks at besides the stored text: pictures of the cut and episode-wide context.

Both are built when the job starts (never snapshotted at queue time), so a chained job sees what the chain wrote.

- ``episode_context``: the neighbouring cuts (their own review, else how their opening frame read each character's
  expression, and the previous cut's end state), the transcript lines just before and after the cut, and for each
  voice label heard in the cut how its lines are linked to characters across the whole episode.
- ``plan``: references to the pictures (``{kind, label, caption, cut_id, at, crop, edge, tile}``), never pixels:
  the first frame, the keyframes F1…Fn at the times the keyframe result was written from, one strip of four frames
  per spoken line (cropped around the speaker when they have a box), a reference crop per character and the frames
  on each side of the cut. ``fit`` lowers the detail until the request fits one server slot, and ``render`` cuts the
  JPEGs from the original video (``assistant.images``) when the request is sent.

Only the references are stored with the result (``evidence.pictures``), the way Cut Assistance stores its pictures.
"""
import concurrent.futures
import json
import math
import tempfile
from pathlib import Path

from . import db, frame_cache, media, prompt_context
from .assistant import images

MAX_LINES = 6  # spoken lines that get a strip of four frames
STRIP_FRAMES = 4
MAX_REFERENCES = 4
FIRST_EDGE, KEYFRAME_EDGE, SMALL_EDGE, REFERENCE_EDGE = 1024, 768, 512, 384
FRAME_TILE, HEAD_TILE = (512, 288), (320, 320)
OUTPUT_RESERVE = 8192  # thinking budget plus the answer
LINE_GAP = 0.6  # transcript fragments of one speaker closer than this read as one line
BEFORE_SECONDS, AFTER_SECONDS, BEFORE_LINES, AFTER_LINES = 30.0, 15.0, 8, 6
RENDER_THREADS = 4
EXPRESSIVE_PARTS = ('expression', 'eyes', 'mouth', 'head')


def _present(**values):
    return {key: value for key, value in values.items() if value not in (None, '', [], {})}


# --- episode context -------------------------------------------------------------------------------------------

def _structured(conn, cut_id):
    from . import structured_results
    return {item['type']: item for item in structured_results.list_results(cut_id, conn=conn)}


def _expressions(frame):
    """How a first-frame result read each character's face, in a few words."""
    people = []
    for person in (frame.get('visible_characters') if isinstance(frame, dict) else None) or []:
        if isinstance(person, dict) and person.get('character'):
            people.append(_present(character=person.get('character'), expression=person.get('expression'), emotion=person.get('emotion')))
    return people


def _end_state(keyframes):
    """Each character's state at the last keyframe of a keyframe result."""
    frames = (keyframes or {}).get('frames') if isinstance(keyframes, dict) else None
    if not frames:
        return []
    return [_present(character=person.get('name'), state=person.get('state')) for person in frames[-1].get('characters') or []
            if isinstance(person, dict) and person.get('name') and person.get('state')]


def _neighbour(conn, cut, label):
    by_type = _structured(conn, cut['id'])
    entry = {'cut': cut['cut_index'] + 1}
    review = (by_type.get('consistency_summary') or {}).get('content')
    if isinstance(review, dict) and review.get('summary'):
        # Its own review, never its questions or picture checks: those are open points about that cut, not facts.
        entry.update(_present(summary=review.get('summary'), clear=review.get('clear')))
    else:
        entry.update(_present(opening_faces=_expressions((by_type.get('initial_frame') or {}).get('content')),
                                             beats_summary=((by_type.get('performance_beats') or {}).get('content') or {}).get('summary')))
    if label == 'previous':
        entry.update(_present(ends_with=_end_state((by_type.get('keyframe_analysis') or {}).get('content'))))
    return entry


def episode_context(conn, cut):
    """``{neighbours, dialogue_around, voices}`` of one cut row; empty parts are left out."""
    out = {}
    neighbours = {}
    if cut['active']:
        for label, index in (('previous', cut['cut_index'] - 1), ('next', cut['cut_index'] + 1)):
            row = conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?', (cut['episode_id'], index)).fetchone()
            if row:
                entry = _neighbour(conn, row, label)
                if len(entry) > 1:
                    neighbours[label] = entry
    if neighbours:
        out['neighbours'] = neighbours
    line = "SELECT t.start_time,t.end_time,t.text,t.speaker_label,c.title AS character_name FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND "
    before = conn.execute(line + 't.end_time<=? AND t.end_time>? ORDER BY t.start_time DESC LIMIT ?', (cut['episode_id'], cut['start_time'], cut['start_time'] - BEFORE_SECONDS, BEFORE_LINES)).fetchall()
    after = conn.execute(line + 't.start_time>=? AND t.start_time<? ORDER BY t.start_time LIMIT ?', (cut['episode_id'], cut['end_time'], cut['end_time'] + AFTER_SECONDS, AFTER_LINES)).fetchall()
    around = _present(before=[prompt_context.transcript_line(dict(row)) for row in reversed(before)],
                                     after=[prompt_context.transcript_line(dict(row)) for row in after])
    if around:
        out['dialogue_around'] = around
    labels = [row['speaker_label'] for row in conn.execute("""SELECT DISTINCT speaker_label FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0
                                                            AND start_time<? AND end_time>? AND coalesce(speaker_label,'')<>'' ORDER BY speaker_label""",
                                                         (cut['episode_id'], cut['end_time'], cut['start_time']))]
    voices = []
    for label in labels:
        linked, unlinked = {}, 0
        for row in conn.execute("""SELECT c.title AS name,count(*) AS lines FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id
                                   WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.speaker_label=? GROUP BY c.title ORDER BY lines DESC""", (cut['episode_id'], label)):
            if row['name']:
                linked[row['name']] = row['lines']
            else:
                unlinked += row['lines']
        voices.append(_present(speaker_label=label, lines_linked_in_episode=linked, unlinked_lines_in_episode=unlinked))
    if voices:
        out['voices'] = voices
    return out


# --- which pictures --------------------------------------------------------------------------------------------

def spoken_lines(segments, start, end):
    """Transcript lines overlapping ``[start, end)`` with fragments of one speaker merged: ``[{t0, t1, speaker, character_id, who}]``, episode seconds."""
    lines = []
    for segment in sorted(segments or [], key=lambda item: item.get('start_time') or 0):
        if segment.get('deleted') or not str(segment.get('text') or '').strip():
            continue
        t0, t1 = float(segment['start_time']), float(segment['end_time'])
        if t1 <= start or t0 >= end:
            continue
        speaker, character = segment.get('speaker_label') or '', segment.get('character_context_id')
        last = lines[-1] if lines else None
        if last and last['speaker'] == speaker and last['character_id'] == character and t0 - last['t1'] <= LINE_GAP:
            last['t1'] = max(last['t1'], t1)
            continue
        lines.append({'t0': t0, 't1': t1, 'speaker': speaker, 'character_id': character, 'who': segment.get('character_name') or speaker or 'unknown speaker'})
    return lines


def head_crop(box, size):
    """Fractions ``[x1, y1, x2, y2]`` around the head of a character box (pixels of a ``size`` frame): its top part with a margin."""
    width, height = size
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    head = min(h, max(w, h * .4)) if h > 1.3 * w else h  # a standing figure: its top; a close-up: the whole box
    side = max(w, head) * 1.3
    cx, cy = (x1 + x2) / 2, y1 + head / 2
    side = max(side, .1 * max(width, height))
    left, top = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
    right, bottom = min(float(width), left + side), min(float(height), top + side)
    return [round(left / width, 4), round(top / height, 4), round(right / width, 4), round(bottom / height, 4)]


def _box_near(boxes, character_id, t):
    """The speaker's box (``{t, box, size}``, clip seconds) nearest to clip time ``t``."""
    own = [item for item in boxes if character_id and item['character_id'] == character_id]
    return min(own, key=lambda item: abs(item['t'] - t)) if own else None


def _frame_size(edge, aspect):
    return edge, max(1, round(edge * aspect))


def plan(cut, keyframes, segments, boxes, characters=(), neighbours=None, aspect=9 / 16):
    """References to every picture the reviewer gets, best first within each kind (pure).

    ``cut`` has ``id``, ``start_time``, ``end_time``; ``keyframes`` is the cut's keyframe result content or None;
    ``boxes`` are ``{character_id, t, box, size}`` in clip seconds; ``characters`` are ``{name, cut_id, at, crop}``
    reference crops; ``neighbours`` maps ``previous``/``next`` to a cut id.
    """
    duration = max(0.0, cut['end_time'] - cut['start_time'])
    pictures = []
    frames = [frame for frame in (keyframes or {}).get('frames') or [] if isinstance(frame, dict) and frame.get('label')]
    if not frames or float(frames[0].get('t') or 0) > 0.05:
        pictures.append({'kind': 'first', 'label': 'First frame', 'caption': 'the opening frame the first-frame description was written from',
                         'cut_id': cut['id'], 'at': ['first'], 'edge': FIRST_EDGE, 'size': _frame_size(FIRST_EDGE, aspect)})
    for index, frame in enumerate(frames):
        t = min(max(0.0, float(frame.get('t') or 0)), duration)
        first = index == 0 and t <= 0.05
        edge = FIRST_EDGE if first else KEYFRAME_EDGE
        pictures.append({'kind': 'keyframe', 'label': frame['label'], 'caption': f"keyframe {frame['label']}, {t:.2f} s into the cut" + (' (also the first frame)' if first else ''),
                         'cut_id': cut['id'], 'at': ['first' if first else f'{t:.3f}s'], 't': round(t, 3), 'edge': edge, 'size': _frame_size(edge, aspect)})
    if not frames:
        pictures.append({'kind': 'keyframe', 'label': 'Last frame', 'caption': 'the last frame of the cut', 'cut_id': cut['id'], 'at': ['last'],
                         'edge': KEYFRAME_EDGE, 'size': _frame_size(KEYFRAME_EDGE, aspect)})
    for line in spoken_lines(segments, cut['start_time'], cut['end_time'])[:MAX_LINES]:
        a, b = max(0.0, line['t0'] - cut['start_time']), min(duration, line['t1'] - cut['start_time'])
        if b - a < 0.2:
            continue
        times = [round(a + (b - a) * (n + .5) / STRIP_FRAMES, 3) for n in range(STRIP_FRAMES)]
        near = _box_near(boxes, line['character_id'], (a + b) / 2)
        crop = head_crop(near['box'], near['size']) if near else None
        label = f"Line {line['t0']:.2f}-{line['t1']:.2f}s"
        caption = (f"{STRIP_FRAMES} frames spread over the line {line['t0']:.2f}-{line['t1']:.2f}s ({line['who']}), in reading order, "
                   + (f"cropped around {line['who']}'s box" if crop else 'whole frames (the speaker has no box)'))
        tile = HEAD_TILE if crop else FRAME_TILE
        pictures.append({'kind': 'line', 'label': label, 'caption': caption, 'cut_id': cut['id'], 'at': [f'{t:.3f}s' for t in times], 'crop': crop,
                         'tile': list(tile), 'edge': tile[0] * 2, 'size': (tile[0] * 2, tile[1] * 2), 'speaker': line['who']})
    for item in list(characters)[:MAX_REFERENCES]:
        pictures.append({'kind': 'reference', 'label': f"Reference: {item['name']}", 'caption': f"{item['name']} at another moment (their usual look)",
                         'cut_id': item['cut_id'], 'at': [item['at']], 'crop': item.get('crop'), 'edge': REFERENCE_EDGE, 'size': (REFERENCE_EDGE, REFERENCE_EDGE)})
    for side, name, label in (('previous', 'last', 'Previous cut, last frame'), ('next', 'first', 'Next cut, first frame')):
        if (neighbours or {}).get(side):
            pictures.append({'kind': 'neighbour', 'label': label, 'caption': label.lower() + ' (continuity only)', 'cut_id': neighbours[side], 'at': [name],
                             'edge': SMALL_EDGE, 'size': _frame_size(SMALL_EDGE, aspect)})
    return pictures


def picture_tokens(picture):
    """What the vision encoder makes of one picture: 32 px per token in each direction (Qwen3.6), plus its caption."""
    width, height = picture['size']
    return math.ceil(width / 32) * math.ceil(height / 32) + 24


def _changed_frames(keyframes):
    """Keyframe labels at either end of a pair where a face, eyes, mouth or head change was reported."""
    found = set()
    for segment in (keyframes or {}).get('segments') or []:
        for person in segment.get('characters') or [] if isinstance(segment, dict) else []:
            if isinstance(person, dict) and (person.get('speaking') == 'yes' or any(isinstance(change, dict) and change.get('part') in EXPRESSIVE_PARTS for change in person.get('changes') or [])):
                found.update((segment.get('from'), segment.get('to')))
    return found


def _resize(picture, edge, aspect):
    return {**picture, 'edge': edge, 'size': _frame_size(edge, aspect)}


def fit(pictures, evidence, system, slot_tokens, keyframes=None, aspect=9 / 16):
    """``(pictures, evidence, level)`` lowered step by step until the request fits ``slot_tokens``.

    Level 0 is everything. Then: neighbours shrink to one line each, keyframes after the first go to 512 px, only the keyframes that
    show a face or mouth change (and the first and last) stay, at most three line strips, no neighbour pictures, no
    reference crops, no dialogue around the cut, and finally only the first picture.
    """
    from .assistant.compact import estimate_tokens
    budget = slot_tokens - OUTPUT_RESERVE

    def total(pics, ev):
        return estimate_tokens(system) + estimate_tokens(json.dumps(ev, ensure_ascii=False)) + sum(picture_tokens(item) for item in pics)

    def shrink_neighbours(pics, ev):
        context = ev.get('episode') or {}
        if context.get('neighbours'):
            context = {**context, 'neighbours': {side: {'cut': item.get('cut'), 'summary': str(item.get('summary') or item.get('beats_summary') or '')[:200]}
                                                 for side, item in context['neighbours'].items()}}
            ev = {**ev, 'episode': context}
        return pics, ev

    def small_keyframes(pics, ev):  # the opening frame keeps its size: it decides the still
        return [_resize(item, SMALL_EDGE, aspect) if item['kind'] == 'keyframe' and item['at'] != ['first'] and item['edge'] > SMALL_EDGE else item for item in pics], ev

    def few_keyframes(pics, ev):
        frames = [item for item in pics if item['kind'] == 'keyframe']
        keep = _changed_frames(keyframes) | ({frames[0]['label'], frames[-1]['label']} if frames else set())
        return [item for item in pics if item['kind'] != 'keyframe' or item['label'] in keep], ev

    def few_lines(pics, ev):
        lines = [id(item) for item in pics if item['kind'] == 'line'][:3]
        return [item for item in pics if item['kind'] != 'line' or id(item) in lines], ev

    def without(kind):
        return lambda pics, ev: ([item for item in pics if item['kind'] != kind], ev)

    def no_dialogue_around(pics, ev):
        context = ev.get('episode') or {}
        return pics, ({**ev, 'episode': {key: value for key, value in context.items() if key != 'dialogue_around'}} if 'dialogue_around' in context else ev)

    def first_only(pics, ev):
        return pics[:1], ev

    level = 0
    for step in (shrink_neighbours, small_keyframes, few_keyframes, few_lines, without('neighbour'), without('reference'), no_dialogue_around, first_only):
        if total(pictures, evidence) <= budget:
            break
        pictures, evidence = step(pictures, evidence)
        level += 1
    return pictures, evidence, level


# --- gathering and rendering -----------------------------------------------------------------------------------

def _first_frame_boxes(conn, cut_id):
    """Labelled boxes on the first frame of the cut's latest detection run: ``[{character_id, t, box, size}]`` (clip seconds)."""
    run = conn.execute('SELECT id,frames_json FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut_id,)).fetchone()
    frame = next((item for item in json.loads(run['frames_json'] or '[]') if isinstance(item, dict) and item.get('position') == 'first'), None) if run else None
    if not frame or not frame.get('image_size'):
        return []
    out = []
    for row in conn.execute("SELECT box_json,character_context_id FROM character_detections WHERE run_id=? AND position='first' AND character_context_id IS NOT NULL", (run['id'],)):
        box = json.loads(row['box_json'] or '[]')
        if len(box) == 4:
            out.append({'character_id': row['character_context_id'], 't': 0.0, 'box': box, 'size': frame['image_size']})
    return out


def _target_boxes(conn, cut_id):
    """Boxes of the cut's latest Target frames run, labelled by their slot's character (clip seconds)."""
    run = conn.execute('SELECT id,frames_json FROM target_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1', (cut_id,)).fetchone()
    if not run:
        return []
    frames = {frame.get('ordinal'): frame for frame in json.loads(run['frames_json'] or '[]') if isinstance(frame, dict)}
    slots = {row['slot']: row['character_context_id'] for row in conn.execute('SELECT slot,character_context_id FROM target_slots WHERE run_id=?', (run['id'],))}
    out = []
    for row in conn.execute('SELECT frame_ordinal,slot,box_json FROM target_detections WHERE run_id=?', (run['id'],)):
        frame, box = frames.get(row['frame_ordinal']), json.loads(row['box_json'] or '[]')
        if slots.get(row['slot']) and frame and frame.get('image_size') and frame.get('relative_time') is not None and len(box) == 4:
            out.append({'character_id': slots[row['slot']], 't': float(frame['relative_time']), 'box': box, 'size': frame['image_size']})
    return out


def _references(conn, ids, cut_id):
    """One crop per character: a saved gallery reference (largest box) else their largest manual box in this cut."""
    from . import sequence_package  # never sequence_export: it imports the API (FastAPI), which the inference worker does not have
    found = sequence_package.character_candidates(conn, list(ids), [cut_id])
    names = {row['id']: row['title'] for row in conn.execute(f"SELECT id,title FROM context_items WHERE id IN ({','.join('?' * len(ids))})", list(ids))} if ids else {}
    out = []
    for cid in ids:
        for candidate in found.get(cid) or []:
            row = conn.execute('SELECT d.position,run.cut_id FROM character_detections d JOIN character_detection_runs run ON run.id=d.run_id WHERE d.id=?', (candidate['detection_id'],)).fetchone()
            if row and row['position'] in ('first', 'last') and cid in names:
                x1, y1, x2, y2 = sequence_package.crop_fractions(candidate)
                out.append({'name': names[cid], 'cut_id': row['cut_id'], 'at': row['position'],
                            'crop': [round(max(0.0, x1), 4), round(max(0.0, y1), 4), round(min(1.0, x2), 4), round(min(1.0, y2), 4)]})
                break
    return out


def gather(cut_id, by_type, conn=None):
    """``(pictures, keyframe content, aspect)``: every picture this cut could offer, before ``fit``."""
    if conn is None:
        with db.connect() as conn:
            return gather(cut_id, by_type, conn)
    cut = conn.execute('SELECT c.*,e.width,e.height FROM cuts c JOIN episodes e ON e.id=c.episode_id WHERE c.id=?', (cut_id,)).fetchone()
    if not cut:
        return [], None, 9 / 16
    segments = [dict(row) for row in conn.execute("""SELECT t.start_time,t.end_time,t.text,t.speaker_label,t.character_context_id,c.title AS character_name
                                                     FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id
                                                     WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>?
                                                     ORDER BY t.start_time""", (cut['episode_id'], cut['end_time'], cut['start_time']))]
    aspect = (cut['height'] / cut['width']) if cut['width'] and cut['height'] else 9 / 16
    keyframes = (by_type.get('keyframe_analysis') or {}).get('content')
    keyframes = keyframes if isinstance(keyframes, dict) and keyframes.get('contract') == 'keyframes-v1' else None
    boxes = _target_boxes(conn, cut_id) + _first_frame_boxes(conn, cut_id)
    ids = []
    for item in boxes + [{'character_id': segment['character_context_id']} for segment in segments]:
        if item.get('character_id') and item['character_id'] not in ids:
            ids.append(item['character_id'])
    characters = _references(conn, ids[:MAX_REFERENCES], cut_id) if ids else []
    neighbours = {}
    if cut['active']:
        for side, index in (('previous', cut['cut_index'] - 1), ('next', cut['cut_index'] + 1)):
            row = conn.execute('SELECT id FROM cuts WHERE episode_id=? AND active=1 AND cut_index=?', (cut['episode_id'], index)).fetchone()
            if row:
                neighbours[side] = row['id']
    return plan(dict(cut), keyframes, segments, boxes, characters, neighbours, aspect), keyframes, aspect


def _scaled(source, output, edge):
    media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(source), '-vf',
               f"scale=w='if(gte(iw,ih),min(iw,{edge}),-2)':h='if(gte(iw,ih),-2,min(ih,{edge}))'", '-frames:v', '1', '-q:v', '3', str(output)], 60)


def _frame(folder, picture, name, index, edge):
    """One frame of a picture as a JPEG file, cropped when the picture has a crop."""
    located = images.locate(picture['cut_id'], name)
    spec = dict(located['spec'])
    spec['width'] = 4096 if picture.get('crop') else edge
    path = folder / f'{index}.jpg'
    path.write_bytes(frame_cache.render(spec))
    if picture.get('crop'):
        cropped, scaled = folder / f'{index}-crop.jpg', folder / f'{index}-scaled.jpg'
        media.crop_image(path, cropped, tuple(picture['crop']))
        _scaled(cropped, scaled, edge)
        return scaled
    return path


def render_one(picture):
    """The JPEG bytes of one planned picture (a strip is tiled with each frame's time printed on it)."""
    with tempfile.TemporaryDirectory(prefix='framewise-consistency-') as temp:
        folder = Path(temp)
        if picture.get('tile'):
            width, height = picture['tile']
            paths = [(_frame(folder, picture, name, index, max(width, height)), f"+{float(name.rstrip('s')):.2f}s") for index, name in enumerate(picture['at'])]
            output = folder / 'strip.jpg'
            media.tile_images(paths, output, 2, (width, height))
            return output.read_bytes()
        return _frame(folder, picture, picture['at'][0], 0, picture['edge']).read_bytes()


def render(pictures, cancel_check=None):
    """``(rendered, missing)``: ``[(picture, jpeg bytes)]`` in plan order, and the labels that could not be cut (never replaced)."""
    rendered, missing = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=RENDER_THREADS, thread_name_prefix='consistency-picture') as pool:
        futures = {pool.submit(render_one, picture): index for index, picture in enumerate(pictures)}
        for future in concurrent.futures.as_completed(futures):
            if cancel_check:
                cancel_check()
            index = futures[future]
            try:
                rendered[index] = future.result()
            except (images.FrameError, RuntimeError, OSError, ValueError) as exc:
                missing.append({'label': pictures[index]['label'], 'error': str(exc)[:200]})
    return [(pictures[index], rendered[index]) for index in sorted(rendered)], missing


def reference(picture):
    """What the stored evidence keeps of one picture that was sent."""
    return {key: picture[key] for key in ('kind', 'label', 'cut_id', 'at', 'crop', 'edge') if picture.get(key) is not None}

