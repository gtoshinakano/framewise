"""Keyframe analysis: what changes between the keyframes of a cut (the ``keyframe_analysis`` stage of a visual job).

It replaced the Qwen3-VL video analysis on 2026-11-18, following ``multiframe_experiment/PIPELINE_HANDOFF.md``
(the "D hybrid": hand-label F1 61 against 31 for the video analysis):

1. The keyframe selector's frames (``keyframes.py``) at 1024 px, in windows of 5 frames sharing one frame.
2. CPU, before any model call: the camera between consecutive keyframes measured from pixels with every person
   masked out (``keyframe_camera``), DWPose body keypoints tracked across the keyframes (``keyframe_pose``), and
   the rules that turn them into per-frame states and per-pair body changes (``keyframe_rules``).
3. the vision model (Qwen3.6 by default; the prompt was measured on Qwen3.5-27B), one call per window with the measured ``observer`` prompt and the character roster
   (``keyframe_prompts``, ``keyframe_schema``): each frame's state first, then the changes per pair.
4. The merge (``keyframe_rules.merge``): DWPose's measured body changes replace Qwen's for that part, Qwen keeps
   head, eyes, mouth, expression and everything DWPose did not measure; the camera is always the pixel estimate.

Names come from the Target frames tab when its detection run is current for this selection (each keyframe's
boxes, one slot per person), else from the first frame's character boxes; YOLOX finds everyone else. Without the
DWPose weights the stage runs Qwen alone and says so (``method: 'qwen'``).

Rows: ``keyframe_analysis_runs`` (one per cut, selection and inputs; provenance), ``_frames`` (keypoints and states),
``_pairs`` (camera and merged changes) and ``_windows`` (each Qwen answer: the checkpoint a retried job resumes
from). The merged answer is saved as the ``keyframe_analysis`` result (contract ``keyframes-v1``) in the same
transaction that queues the ``performance_beats`` text job.
"""
import base64
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import db, frame_cache, keyframe_camera, keyframe_pose, keyframe_prompts, keyframe_rules, keyframe_schema, keyframes, visual_models, worker

log = logging.getLogger('framewise.keyframe_analysis')

CONTRACT = 'keyframes-v1'
RULES_VERSION = 'pose-rules-2026-11-18'
WINDOW = 5
IMAGE_WIDTH = 1024
MAX_ATTEMPTS = 3
MAX_TOKENS = 4096
RAW_LIMIT = 16000
CONTEXT_PER_SLOT = 16384  # five 1024 px images plus about 1,400 answer tokens
PARALLEL = max(1, int(os.environ.get('KEYFRAME_PARALLEL', '2')))


# -- inputs ----------------------------------------------------------------------------------------------------------

def frames_of(result):
    """The selection's frames in order, labelled F1…Fn with their time from the start of the cut."""
    out = []
    for k, item in enumerate(sorted(result['selected_frames'], key=lambda f: f['source_frame_index']), 1):
        out.append({'label': f'F{k}', 'ordinal': item['source_frame_index'], 't': round(item['relative_timestamp_seconds'], 3),
                    'time': item['timestamp_seconds'], 'reason': item.get('reason')})
    return out


def _grid(box, size):
    return [round(box[0] * 1000 / size[0]), round(box[1] * 1000 / size[1]), round(box[2] * 1000 / size[0]), round(box[3] * 1000 / size[1])]


def _side(grid):
    centre = (grid[0] + grid[2]) / 2
    return 'left' if centre < 380 else 'right' if centre > 620 else 'centre'


def target_roster(run, frames):
    """Roster and anchors from a current Target frames run. Anchors are per ordinal, in the run's frame pixels."""
    sizes = {item['ordinal']: item['image_size'] for item in run['frames']}
    labels = {frame['ordinal']: frame['label'] for frame in frames}
    order = {frame['ordinal']: number for number, frame in enumerate(frames)}
    anchors, first = {}, {}
    for box in sorted(run['detections'], key=lambda item: (order.get(item['frame_ordinal'], len(order)), item['slot'])):
        if box['frame_ordinal'] not in labels or not sizes.get(box['frame_ordinal']):
            continue
        slot = next((item for item in run['slots'] if item['slot'] == box['slot']), None)
        anchors.setdefault(box['frame_ordinal'], []).append({'box': box['box'], 'name': (slot or {}).get('character_name'), 'slot': box['slot'],
                                                            'character_context_id': (slot or {}).get('character_context_id'), 'source': f"target:{box['source']}"})
        if slot and slot.get('character_name') and slot['slot'] not in first:
            first[slot['slot']] = (box, slot)
    roster, seen = [], set()
    for box, slot in first.values():
        if slot['character_name'] in seen:
            continue
        seen.add(slot['character_name'])
        grid = _grid(box['box'], sizes[box['frame_ordinal']])
        roster.append({'name': slot['character_name'], 'character_context_id': slot['character_context_id'], 'box': grid, 'side': _side(grid),
                       'first_seen': labels[box['frame_ordinal']], 'source': 'target_slots'})
    roster.sort(key=lambda person: person['box'][0])
    return roster, anchors, sizes


def first_frame_roster(detections, frames):
    """Roster and anchors from the job's first-frame character boxes (the experiment's roster)."""
    detections = detections or {}
    size = next((frame.get('image_size') for frame in detections.get('frames', []) if frame.get('position') == 'first'), None)
    if not size or not frames or frames[0].get('reason') != 'initial_state':
        return [], {}, {}
    boxes = [d for d in detections.get('detections', []) if d.get('position') == 'first' and isinstance(d.get('box'), list) and len(d['box']) == 4]
    anchors = {frames[0]['ordinal']: [{'box': d['box'], 'name': d.get('character_name') if d.get('character_context_id') else None,
                                       'character_context_id': d.get('character_context_id'), 'source': f"first_frame:{d.get('label_source') or '?'}"}
                                      for d in boxes]}
    roster, seen = [], set()
    for d in boxes:
        name = d.get('character_name') if d.get('character_context_id') else None
        if not name or name in seen:
            continue
        seen.add(name)
        grid = _grid(d['box'], size)
        roster.append({'name': name, 'character_context_id': d['character_context_id'], 'box': grid, 'side': _side(grid), 'source': 'first_frame'})
    roster.sort(key=lambda person: person['box'][0])
    return roster, anchors, {frames[0]['ordinal']: size}


def target_run(cut_id, key, cut):
    """The cut's Target frames run for this selection with its slots and boxes, when it was made for these boundaries."""
    from . import target_characters
    run = target_characters.run_for(cut_id, key)
    if not run or run['cut_start'] != cut['start_time'] or run['cut_end'] != cut['end_time']:
        return None
    state = target_characters.state(cut_id)
    detail = (state or {}).get('run')
    if not detail or detail['id'] != run['id'] or detail['stale']:
        return None
    if not detail['detections'] and not detail.get('review_version'):
        return None
    return detail


def band_for(strip, result):
    """The subtitle strip to ignore: the extracted one, else the band the selection masked."""
    if strip and strip.get('top') is not None:
        return {'top': strip['top'], 'bottom': strip['bottom']}
    mask = (result or {}).get('subtitle_mask') or {}
    return {'top': mask['top'], 'bottom': mask['bottom']} if mask.get('top') is not None else None


# -- the Qwen call ---------------------------------------------------------------------------------------------------

def _clip(text):
    return text if len(text or '') <= RAW_LIMIT else text[:RAW_LIMIT] + ' …[clipped]'


def chat(server_url, messages, schema, cancel_check=None):
    """One chat call with the json_schema grammar, plain JSON when the server cannot build it. Returns the reply fields."""
    from .inference_worker import request_json

    def ask(fmt):
        body = {'messages': messages, 'temperature': 0, 'max_tokens': MAX_TOKENS, 'response_format': fmt, 'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(server_url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        return request_json(request, cancel_check, timeout=900)
    grammar = True
    try:
        payload = ask({'type': 'json_schema', 'json_schema': {'name': 'keyframes', 'strict': True, 'schema': schema}})
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 422, 500):
            raise
        grammar = False
        payload = ask({'type': 'json_object'})
    choice = payload['choices'][0]
    return {'content': choice['message'].get('content') or '', 'finish_reason': choice.get('finish_reason'), 'usage': payload.get('usage') or {},
            'timings': payload.get('timings') or {}, 'grammar': grammar}


def describe_window(ask, sample, window, images):
    """One window through the vision model, validated and retried with the validator's sentences. Returns the window record."""
    messages = keyframe_prompts.build_messages(sample, window, images)
    prompt = keyframe_prompts.prompt_text(messages)
    grammar = keyframe_schema.json_schema(window)
    attempts, answer = [], None
    for number in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            reply = ask(messages, grammar)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            attempts.append({'n': number, 'seconds': round(time.monotonic() - started, 2), 'errors': [f'request failed: {exc}'], 'raw': ''})
            continue
        raw = reply['content']
        if reply.get('finish_reason') == 'length':
            errors = ['the answer was cut off by the length limit; return a complete, shorter JSON object (shorter descriptions)']
        else:
            try:
                answer, errors = keyframe_schema.validate(keyframe_schema.parse_json(raw), window)
            except ValueError as exc:
                answer, errors = None, [f'the answer is not valid JSON ({exc})']
        usage, timings = reply.get('usage') or {}, reply.get('timings') or {}
        attempts.append({'n': number, 'seconds': round(time.monotonic() - started, 2), 'prompt_tokens': usage.get('prompt_tokens'),
                         'completion_tokens': usage.get('completion_tokens'), 'prompt_ms': timings.get('prompt_ms'), 'predicted_ms': timings.get('predicted_ms'),
                         'finish': reply.get('finish_reason'), 'grammar': reply.get('grammar'), 'errors': errors, 'raw': _clip(raw)})
        if not errors:
            break
        answer = None
        # A cut-off answer is usually a repetition loop: resend only its start so the retry does not carry it.
        shown = raw if reply.get('finish_reason') != 'length' else (raw or '')[:1500] + ' …'
        messages = [*messages[:2], {'role': 'assistant', 'content': shown or '{}'}, {'role': 'user', 'content': keyframe_schema.format_errors(errors)}]
    return {'answer': answer, 'attempts': attempts, 'prompt': prompt, 'seconds': round(sum(a['seconds'] for a in attempts), 2)}


# -- assembling the result (pure) ------------------------------------------------------------------------------------

def _link(name, roster):
    from .visual_contracts import _match_name
    person = _match_name(name, roster)
    return person.get('character_context_id') if person else None


def _tag_sources(merged, rules):
    """Mark each change of the merged answer with where it came from: ``dwpose`` (measured) or ``qwen``."""
    measured = {}
    for segment in (rules or {}).get('segments') or []:
        for person in segment['characters']:
            measured.setdefault((segment['from'], person['name']), []).extend(person['changes'])
    for segment in merged['segments']:
        for person in segment['characters']:
            source = (person.get('merged_from') or {}).get('dwpose')
            ours = measured.get((segment['from'], source), []) if source else []
            for change in person['changes']:
                change['source'] = 'dwpose' if any(change is item for item in ours) else 'qwen'
            person.pop('merged_from', None)
    return merged


def merge_window(answer, rules, states, pairs):
    """The window's final answer: the hybrid when DWPose ran, else Qwen's with the measured camera."""
    if rules is not None:
        return _tag_sources(keyframe_rules.merge(answer, rules, states, pairs), rules)
    measured = {(p['from'], p['to']): p for p in pairs}
    segments = []
    for segment in answer['segments']:
        camera = (measured.get((segment['from'], segment['to'])) or {}).get('camera') or {}
        people = [{**person, 'changes': [{**change, 'source': 'qwen'} for change in person['changes']]} for person in segment['characters']]
        segments.append({**segment, 'camera': {'move': camera.get('move') or 'unclear', 'detail': 'pixel estimate'}, 'characters': people})
    return {'frames': answer.get('frames') or [], 'segments': segments, 'summary': answer.get('summary', '')}


def assemble(frames, windows, merged, roster, dwpose_states, method, run_id, key):
    """The ``keyframes-v1`` content from the merged windows. A frame shared by two windows keeps its first window's state."""
    qwen_states = {}
    for result in merged:
        for frame in result.get('frames') or []:
            qwen_states.setdefault(frame['frame'], frame['characters'])
    out_frames = []
    for frame in frames:
        people = []
        names = [person['name'] for person in qwen_states.get(frame['label'], [])]
        poses = (dwpose_states or {}).get(frame['label'], {})
        for person in qwen_states.get(frame['label'], []):
            people.append({'name': person['name'], 'character_context_id': _link(person['name'], roster), 'state': person['state'],
                           'pose': poses.get(person['name'])})
        for name, text in poses.items():
            if name not in names:
                people.append({'name': name, 'character_context_id': _link(name, roster), 'state': None, 'pose': text})
        out_frames.append({'label': frame['label'], 'ordinal': frame['ordinal'], 't': frame['t'], 'reason': frame.get('reason'), 'characters': people})
    segments, notes = [], {'from_dwpose': 0, 'from_qwen': 0, 'dwpose_only_characters': 0}
    for result in merged:
        for key_name, value in (result.get('merge') or {}).items():
            notes[key_name] = notes.get(key_name, 0) + value
        for segment in result['segments']:
            people = [{**person, 'character_context_id': _link(person['name'], roster)} for person in segment['characters']]
            segments.append({**segment, 'characters': people})
    summaries = [{'index': window['index'], 'labels': keyframe_schema.labels(window), 'summary': result.get('summary') or ''}
                 for window, result in zip(windows, merged)]
    return {'contract': CONTRACT, 'method': method, 'run_id': run_id, 'selection_key': key,
            'roster': [{'name': p['name'], 'character_context_id': p.get('character_context_id'), 'source': p.get('source')} for p in roster],
            'frames': out_frames, 'segments': segments, 'windows': summaries,
            'summary': ' '.join(item['summary'] for item in summaries if item['summary']), 'merge': notes}


# -- pose on the CPU -------------------------------------------------------------------------------------------------

def load_pose():
    """(DWPose sessions or None, provenance): None when the weights or onnxruntime are missing; the stage then runs the vision model alone."""
    model = visual_models.MODELS['dwpose']
    info = visual_models.status('dwpose')
    provenance = {'model_id': model.model_id, 'revision': model.revision, 'files': [model.path(name).name for name in model.files],
                  'provider': 'CPUExecutionProvider', 'rules': RULES_VERSION}
    if not info['available']:
        return None, {**provenance, 'missing': info['detail']}
    try:
        threads = int(os.environ.get('DWPOSE_THREADS') or 0) or None
        sessions = keyframe_pose.DWPose(model.path('detector').parent, threads=threads)
        import onnxruntime
        provenance['onnxruntime'] = onnxruntime.__version__
        return sessions, provenance
    except Exception as exc:  # noqa: BLE001 - a broken install falls back to Qwen alone, recorded in provenance
        log.exception('DWPose could not be loaded; keyframe analysis runs without it')
        return None, {**provenance, 'missing': f'Could not load: {exc}'}


def _scale(box, factor):
    return [v * factor for v in box[:4]]


def prepare(images, frames, anchors, anchor_sizes, band, pose):
    """CPU evidence of one cut: per-frame people (with keypoints when ``pose``), and per-pair cameras.

    ``images``: BGR arrays of the keyframes. ``anchors``: ``{ordinal: [box dicts]}`` in the pixels of ``anchor_sizes[ordinal]``.
    Returns ``(people per frame, pairs)``; a pair's ``camera`` has the people masked out, ``camera_all`` does not (used for tracking).
    """
    sizes = [[image.shape[1], image.shape[0]] for image in images]
    grays = [keyframe_camera.gray(image) for image in images]
    camera_all = keyframe_camera.pair_cameras(grays, band)
    people = []
    for frame, image, size in zip(frames, images, sizes):
        mine = []
        for anchor in anchors.get(frame['ordinal'], []):
            factor = size[0] / anchor_sizes[frame['ordinal']][0]
            mine.append({**anchor, 'box': _scale(anchor['box'], factor)})
        if pose is not None:
            found, _ = keyframe_pose.people_for_frame(pose, image, mine)
            people.append(found)
        else:
            people.append([{**anchor, 'box': [round(v, 1) for v in anchor['box']]} for anchor in mine])
    if pose is not None and people:
        keyframe_pose.track(people, sizes[0][0], sizes[0][1], camera_all)
        keyframe_pose.describe_people(sizes, people)
    camera = keyframe_camera.pair_cameras(grays, band, [[person['box'] for person in found] for found in people], sizes)
    pairs = [{'from': a['label'], 'to': b['label'], 'start': a['t'], 'end': b['t'], 'camera': camera[i], 'camera_all': camera_all[i]}
             for i, (a, b) in enumerate(zip(frames, frames[1:]))]
    return people, pairs


def dwpose_states(frames, people, sizes):
    """``keyframe_rules.frame_states`` of the stored keypoints: {label: {name: measured state}}."""
    return keyframe_rules.frame_states({'frames': [{'label': frame['label'], 'size': size, 'persons': found}
                                                   for frame, found, size in zip(frames, people, sizes)]})


def state_texts(states):
    return {label: {name: keyframe_rules.state_text(state) for name, state in people.items()} for label, people in states.items()}


# -- the stage -------------------------------------------------------------------------------------------------------

def input_hash(parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _extract(source, cut, frames, folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        path = folder / f"{frame['label']}.jpg"
        spec = {'source': source, 'time': frame['time'], 'start': cut['start_time'], 'end': cut['end_time'],
                'width': IMAGE_WIDTH, 'last': False, 'ordinal': frame['ordinal']}
        path.write_bytes(frame_cache.render(spec))
        paths.append(path)
    return paths


def _load_run(run_id):
    run = db.one('SELECT * FROM keyframe_analysis_runs WHERE id=?', (run_id,))
    frames = db.all_rows('SELECT * FROM keyframe_analysis_frames WHERE run_id=? ORDER BY relative_time,ordinal', (run_id,))
    pairs = db.all_rows('SELECT * FROM keyframe_analysis_pairs WHERE run_id=? ORDER BY pair_index', (run_id,))
    windows = db.all_rows('SELECT * FROM keyframe_analysis_windows WHERE run_id=? ORDER BY window_index', (run_id,))
    return run, frames, pairs, windows


def analyze(cut_run, server_url, scratch, model, pose, pose_provenance, ask=None):
    """Run the keyframe analysis of one cut inside the job's ``keyframe_analysis`` stage and save its result.

    ``ask(messages, schema)`` is the Qwen call (``chat`` on ``server_url`` unless a test replaces it).
    """
    import cv2
    from . import performance_beats
    if not cut_run.needs('keyframe_analysis'):
        return
    with cut_run.stage('keyframe_analysis', 'Keyframe analysis'):
        started = time.monotonic()
        timings = {}
        ask = ask or (lambda messages, schema: chat(server_url, messages, schema, cut_run.cancel_check))
        cut = cut_run.cut
        _, source, key, result = keyframes.current_selection(cut_run.cut_id)
        if key is None:
            raise RuntimeError('This cut has no current keyframe selection (it failed, or the cut or its subtitles changed); '
                               'open Target frames and evaluate the cut, then run Keyframes again')
        frames = frames_of(result)
        windows = keyframe_schema.make_windows(frames, WINDOW)
        detail = target_run(cut_run.cut_id, key, cut) if not cut.get('no_characters') else None
        if cut.get('no_characters'):
            roster, anchors, anchor_sizes, roster_source = [], {}, {}, 'none'
        elif detail:
            roster, anchors, anchor_sizes = target_roster(detail, frames)
            roster_source = 'target_slots'
        else:
            roster, anchors, anchor_sizes = first_frame_roster(cut_run.context.get('character_detections'), frames)
            roster_source = 'first_frame' if anchors else 'none'
        band = band_for(cut_run.strip, result)
        method = 'hybrid' if pose is not None else 'qwen'
        sample = {'duration': cut['end_time'] - cut['start_time'], 'no_characters': bool(cut.get('no_characters')), 'roster': roster, 'subtitle_band': band}
        settings = {'window_size': WINDOW, 'image_width': IMAGE_WIDTH, 'max_attempts': MAX_ATTEMPTS, 'max_tokens': MAX_TOKENS,
                    'parallel': PARALLEL, 'context_per_slot': CONTEXT_PER_SLOT}
        digest = input_hash({'prompt': keyframe_prompts.PROMPT_VERSION, 'system': hashlib.sha256(keyframe_prompts.SYSTEM.encode()).hexdigest(),
                             'model': model.revision, 'rules': RULES_VERSION, 'method': method,
                             'settings': {key: settings[key] for key in ('window_size', 'image_width', 'max_attempts', 'max_tokens')},  # how many run at once changes no answer
                             'roster': keyframe_prompts.roster_lines(sample), 'band': band, 'frames': [f['ordinal'] for f in frames],
                             'anchors': anchors, 'cut': [cut['start_time'], cut['end_time']]})
        existing = db.one("SELECT id FROM keyframe_analysis_runs WHERE cut_id=? AND selection_key=? AND input_hash=? AND phase!='complete' ORDER BY rowid DESC LIMIT 1",
                          (cut_run.cut_id, key, digest))
        provenance = {'model': {'key': model.key, 'model_id': model.model_id, 'revision': model.revision}, 'dwpose': pose_provenance,
                      'prompt_version': keyframe_prompts.PROMPT_VERSION, 'system': keyframe_prompts.SYSTEM, 'settings': settings, 'band': band}
        if existing:
            run_id = existing['id']
        else:
            run_id = str(uuid.uuid4())
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                worker.require_lease(conn, cut_run.job['id'], cut_run.owner)
                cut_run.check_cut(conn)
                conn.execute('''INSERT INTO keyframe_analysis_runs(id,cut_id,selection_key,cut_start,cut_end,job_id,input_hash,method,roster_source,target_run_id,roster_json,provenance_json)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                             (run_id, cut_run.cut_id, key, cut['start_time'], cut['end_time'], cut_run.job['id'], digest, method, roster_source,
                              detail['id'] if detail else None, json.dumps(roster, ensure_ascii=False), json.dumps(provenance, ensure_ascii=False)))
        cut_run.progress(f'Extracting {len(frames)} keyframes')
        began = time.monotonic()
        paths = _extract(source, cut, frames, Path(scratch) / f'keyframes-{cut_run.cut_id}')
        images = [cv2.imread(str(path)) for path in paths]
        if any(image is None for image in images):
            raise RuntimeError('A keyframe could not be read')
        sizes = [[image.shape[1], image.shape[0]] for image in images]
        timings['extract'] = round(time.monotonic() - began, 2)
        run, frame_rows, pair_rows, window_rows = _load_run(run_id)
        if run['phase'] == 'prepare':
            cut_run.progress('Measuring the camera' + (' and body poses' if pose is not None else ''))
            began = time.monotonic()
            people, pairs = prepare(images, frames, anchors, anchor_sizes, band, pose)
            states = dwpose_states(frames, people, sizes) if pose is not None else {}
            rules = [keyframe_rules.answer_for_window(states, window, pairs) if pose is not None and len(window['frames']) > 1 else None for window in windows]
            texts = state_texts(states)
            timings['prepare'] = round(time.monotonic() - began, 2)
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                worker.require_lease(conn, cut_run.job['id'], cut_run.owner)
                cut_run.check_cut(conn)
                for frame, found, size in zip(frames, people, sizes):
                    conn.execute('''INSERT OR REPLACE INTO keyframe_analysis_frames(run_id,ordinal,label,frame_time,relative_time,reason,image_size_json,pose_json,states_json)
                                    VALUES (?,?,?,?,?,?,?,?,?)''',
                                 (run_id, frame['ordinal'], frame['label'], frame['time'], frame['t'], frame.get('reason'), json.dumps(size),
                                  json.dumps(found) if pose is not None else None, json.dumps({'dwpose': texts.get(frame['label'], {})}, ensure_ascii=False)))
                for index, (pair, a, b) in enumerate(zip(pairs, frames, frames[1:])):
                    conn.execute('INSERT OR REPLACE INTO keyframe_analysis_pairs(run_id,pair_index,from_ordinal,to_ordinal,start,"end",camera_json) VALUES (?,?,?,?,?,?,?)',
                                 (run_id, index, a['ordinal'], b['ordinal'], a['t'], b['t'], json.dumps({**pair['camera'], 'camera_all': pair['camera_all']})))
                for window, rule in zip(windows, rules):
                    conn.execute('''INSERT INTO keyframe_analysis_windows(run_id,window_index,labels_json,rules_json) VALUES (?,?,?,?)
                                    ON CONFLICT(run_id,window_index) DO UPDATE SET labels_json=excluded.labels_json,rules_json=excluded.rules_json''',
                                 (run_id, window['index'], json.dumps(keyframe_schema.labels(window)), json.dumps(rule, ensure_ascii=False) if rule else None))
                conn.execute("UPDATE keyframe_analysis_runs SET phase='describe',job_id=? WHERE id=?", (cut_run.job['id'], run_id))
            run, frame_rows, pair_rows, window_rows = _load_run(run_id)
        people = [json.loads(row['pose_json']) if row['pose_json'] else [] for row in frame_rows]
        states = dwpose_states(frames, people, sizes) if pose is not None and method == 'hybrid' else {}
        pairs = [{'from': a['label'], 'to': b['label'], 'start': a['t'], 'end': b['t'], 'camera': json.loads(row['camera_json'])}
                 for row, a, b in zip(pair_rows, frames, frames[1:])]
        saved = {row['window_index']: row for row in window_rows}
        todo = [window for window in windows if len(window['frames']) > 1 and not (saved.get(window['index']) or {}).get('answer_json')]
        encoded = {frame['label']: base64.b64encode(path.read_bytes()).decode('ascii') for frame, path in zip(frames, paths)}
        began = time.monotonic()
        done = [len(windows) - len(todo)]

        def describe(window):
            cut_run.cancel_check()
            record = describe_window(ask, sample, window, [encoded[frame['label']] for frame in window['frames']])
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                worker.require_lease(conn, cut_run.job['id'], cut_run.owner)
                cut_run.check_cut(conn)
                conn.execute('''INSERT INTO keyframe_analysis_windows(run_id,window_index,labels_json,answer_json,attempts_json,prompt,ok,seconds) VALUES (?,?,?,?,?,?,?,?)
                                ON CONFLICT(run_id,window_index) DO UPDATE SET answer_json=excluded.answer_json,attempts_json=excluded.attempts_json,
                                prompt=excluded.prompt,ok=excluded.ok,seconds=excluded.seconds''',
                             (run_id, window['index'], json.dumps(keyframe_schema.labels(window)), json.dumps(record['answer'], ensure_ascii=False) if record['answer'] else None,
                              json.dumps(record['attempts'], ensure_ascii=False), record['prompt'], int(record['answer'] is not None), record['seconds']))
            if record['answer'] is None:
                errors = (record['attempts'][-1]['errors'] if record['attempts'] else ['no attempt'])[:3]
                raise RuntimeError(f"Keyframes {window['frames'][0]['label']}–{window['frames'][-1]['label']} had no valid answer after {len(record['attempts'])} attempts: {'; '.join(errors)}")
            done[0] += 1
            cut_run.progress(f'Describing keyframes ({done[0]}/{len(windows)} windows)')

        if todo:
            cut_run.progress(f'Describing keyframes ({done[0]}/{len(windows)} windows)')
            with ThreadPoolExecutor(max_workers=min(PARALLEL, len(todo)), thread_name_prefix='keyframe-window') as pool:
                futures = [pool.submit(describe, window) for window in todo]
                errors = [future.exception() for future in futures]
            for error in errors:
                if error:
                    raise error
        timings['qwen'] = round(time.monotonic() - began, 2)
        window_rows = db.all_rows('SELECT * FROM keyframe_analysis_windows WHERE run_id=? ORDER BY window_index', (run_id,))
        stored = {row['window_index']: row for row in window_rows}
        merged = []
        for window in windows:
            if len(window['frames']) < 2:
                merged.append({'frames': [], 'segments': [], 'summary': ''})
                continue
            row = stored[window['index']]
            answer = json.loads(row['answer_json'])
            rules = json.loads(row['rules_json']) if row['rules_json'] and method == 'hybrid' else None
            merged.append(merge_window(answer, rules, states, [pair for pair in pairs if pair['from'] in keyframe_schema.labels(window)]))
        content = assemble(frames, windows, merged, roster, state_texts(states), method, run_id, key)
        content['cut'] = {'start_time': cut['start_time'], 'end_time': cut['end_time']}
        usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        retries = 0
        for row in window_rows:
            attempts = json.loads(row['attempts_json'] or '[]')
            retries += max(0, len(attempts) - 1)
            for attempt in attempts:
                for name in usage:
                    usage[name] += attempt.get(name) or 0
        timings['total'] = round(time.monotonic() - started, 2)
        extra = {'model_revision': model.revision, 'visual_model': model.key, 'keyframe_contract': CONTRACT, 'method': method, 'run_id': run_id,
                 'selection_key': key, 'roster_source': roster_source, 'prompt_version': keyframe_prompts.PROMPT_VERSION, **settings,
                 'windows': len([w for w in windows if len(w['frames']) > 1]), 'retries': retries, 'keyframes': len(frames),
                 'dwpose': pose_provenance, 'generation_seconds': timings['total'], 'timings': timings, 'usage': usage}

        def finish(conn, result_id):
            if not conn.execute('SELECT 1 FROM keyframe_results WHERE cache_key=?', (key,)).fetchone():
                raise RuntimeError('The keyframe selection changed while analysing; run Keyframes again')
            for frame in content['frames']:
                qwen = [{'name': p['name'], 'character_context_id': p['character_context_id'], 'state': p['state']} for p in frame['characters'] if p['state']]
                conn.execute("UPDATE keyframe_analysis_frames SET states_json=json_set(states_json,'$.qwen',json(?)) WHERE run_id=? AND ordinal=?",
                             (json.dumps(qwen, ensure_ascii=False), run_id, frame['ordinal']))
            for index, segment in enumerate(content['segments']):
                conn.execute('UPDATE keyframe_analysis_pairs SET changes_json=? WHERE run_id=? AND pair_index=?', (json.dumps(segment, ensure_ascii=False), run_id, index))
            conn.execute("UPDATE keyframe_analysis_runs SET phase='complete',result_id=?,timings_json=?,finished_at=CURRENT_TIMESTAMP WHERE id=?",
                         (result_id, json.dumps(timings), run_id))
            performance_beats.ensure_queued(conn, cut_run.episode['id'], cut_run.cut_id)

        cut_run.save('keyframe_analysis', content, model.model_id, keyframe_prompts.SYSTEM, extra, after=finish)
        log.info('keyframe analysis cut=%s method=%s frames=%d windows=%d took %.1fs', cut.get('cut_index'), method, len(frames), len(windows), timings['total'])


def stage(runs, scratch, model):
    """The ``keyframe_analysis`` stage of a visual job: one vision-model server and one DWPose load for every cut, cuts one at a time."""
    from . import frame_model, single_cut_pipeline
    todo = [cut_run for cut_run in runs if not cut_run.failure and cut_run.needs('keyframe_analysis')]
    if not todo:
        return
    pose, pose_provenance = load_pose()
    with frame_model.frame_server(*model.server_files(), cancel_check=todo[0].cancel_check, parallel=PARALLEL,
                                  context_per_slot=CONTEXT_PER_SLOT, extra_args=model.server_args) as server:
        single_cut_pipeline._for_each(todo, lambda cut_run: analyze(cut_run, server, scratch, model, pose, pose_provenance))


# -- reading -----------------------------------------------------------------------------------------------------------

def run_model(run):
    """Short name of the model that described the run's windows (the tab says which one it was)."""
    try:
        key = json.loads(run['provenance_json'])['model']['key']
    except (KeyError, TypeError, ValueError):
        return None
    return visual_models.MODELS[key].short_name if key in visual_models.MODELS else key


def state(cut_id):
    """What the Target frames tab shows: the current keyframe analysis result, its run, and the latest job covering the cut."""
    from . import structured_results
    _, _, key, _ = keyframes.current_selection(cut_id)
    current = next((item for item in structured_results.list_results(cut_id) if item['type'] == 'keyframe_analysis'), None)
    content = (current or {}).get('content') or {}
    run = db.one('SELECT id,phase,method,roster_source,roster_json,timings_json,selection_key,target_run_id,provenance_json FROM keyframe_analysis_runs WHERE id=?',
                 (content.get('run_id'),)) if content.get('run_id') else None
    job = db.one("""SELECT j.id,j.status,j.stage,j.error FROM jobs j WHERE j.pipeline='analyze'
                    AND EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.cuts') i WHERE json_extract(i.value,'$.cut_id')=?)
                    ORDER BY j.rowid DESC LIMIT 1""", (cut_id,))
    status = 'missing' if not current else 'current' if key and content.get('selection_key') == key else 'stale'
    return {'status': status, 'selection_key': key,
            'result': {'id': current['id'], 'origin': current['origin'], 'created_at': current['created_at'], **content} if current else None,
            'run': {'id': run['id'], 'phase': run['phase'], 'method': run['method'], 'roster_source': run['roster_source'], 'target_run_id': run['target_run_id'],
                    'model': run_model(run), 'roster': json.loads(run['roster_json']), 'timings': json.loads(run['timings_json'] or '{}')} if run else None,
            'job': job}
