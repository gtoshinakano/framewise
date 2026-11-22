"""Characters that matter: naming first-frame boxes, grouping unknown people and settling target-frame flags
(docs/autopilot/HARNESS.md, milestone M6).

Everything here reads what recognition and the target-frame tracker already stored (``character_detections.candidates_json``,
the CCIP vectors in ``character_detection_embeddings``, ``target_detection_runs.frames_json``) and the keyframe analysis's
per-frame states; nothing loads a model, so the job runs on the CPU in the API process.

**First-frame boxes** (decision kind ``box_names``), per box of a cut's current, complete detection run:

* an *unnamed* box is named from its gallery shortlist when the best candidate clears the bar with a margin: a lower bar
  when the same character is boxed in the cut before or after (the context the user uses for fragments, D13); a box
  whose character is already boxed in the cut is a duplicate box (removed) or a question; two look-alikes (profile)
  never decide alone; a character the profile calls *background* is unboxed.
* an *automatic* label (recognition's ``auto``) is never confirmed by a rule (the replay found the shortlist leader wrong
  too often even at high similarity); Claude looks at a sheet of up to ``SHEET_ROWS`` boxes per character, least sure
  first, while the run's budget lasts, and confirms (``agent``) or rejects them.
* an unnamed box nobody in the gallery resembles is an *unknown person*: the boxes of the episode are grouped by their
  CCIP vectors, and each group is weighed by the relevance rule (HARNESS.md section 5): one small appearance is
  background (unboxed), recurring or large people matter and go to the person as one card (name them once, every box of
  the group follows), the rest go to Claude first.

Agent labels are ``label_source='agent'``: never gallery examples (``anime_gallery`` reads only ``manual``, D10), and a
recognition re-run leaves them alone. Only the person's own choice becomes ``manual``.

**Target frames** (decision kind ``target_frames``): a flagged frame is marked reviewed when a rule explains its flag (the
headcount dropped and the keyframe states say exactly those people left the picture); every other flagged frame of a
cut is one question, for Claude with a sheet of the frames, then the person in the Target frames tab. A frame whose only flag
is an extra crop that fits no colour, with everyone from the opening frame still boxed, is marked reviewed too.

Each change is an ``agent_decisions`` row written in the transaction that applies it, with an undo in
``agent_log.UNDO_KINDS``. Cuts with analysis off or ``no_characters`` are left alone.
"""
import json
import math
import re
import uuid
from collections import Counter, defaultdict

from . import agent_log, anime_gallery, db, job_inputs, structured_results, worker

PIPELINE = 'character_decisions'
KIND_BOXES = 'box_names'          # autopilot_profile.DECISIONS and agent_decisions.kind
KIND_TARGETS = 'target_frames'
ESC_LABELS = 'box_label'          # unsure names, one per character (or per box without a shortlist leader)
ESC_PERSON = 'new_person'         # one group of unknown boxes
ESC_TARGET = 'target_frame'       # the flagged target frames of one cut
ESCALATIONS = (ESC_LABELS, ESC_PERSON, ESC_TARGET)
VERSION = 1

# Naming an unnamed box from its shortlist (tuned by autopilot/replay/box_names.py on the user's manual labels).
NAME_MIN, NAME_MARGIN = 0.72, 0.10              # no context
NAME_MIN_NEAR, NAME_MARGIN_NEAR = 0.58, 0.08    # the character is boxed in the cut before or after
ALWAYS_MIN = 0.48                               # a character the profile says always matters, boxed next door
CLOSE_MIN = 0.50                                # below this nobody in the gallery resembles the box: an unknown person
LOOK_ALIKE_MARGIN = 0.15                        # a look-alike pair closer than this is never decided alone
DUPLICATE_IOU = 0.7                             # a second box over the same person
# Grouping unknown people and weighing them.
GROUP_MIN = 0.82                                # cosine to a group's mean vector (CCIP track threshold is 0.85)
MATTERS_CUTS = 3                                # appears in this many cuts
MATTERS_FILL = 0.45                             # or one box this tall (share of the frame height)
RECURRING_FILL = 0.25                           # or two cuts with a box this tall
BACKGROUND_FILL = 0.20                          # one cut, all boxes smaller than this (or too small to identify): background
SHEET_ROWS = 12
NOT_VISIBLE = re.compile(r'not visible|no longer visible|not (in|on|within) (the )?(frame|picture|shot|view|screen)|out of (the )?(frame|picture|shot|screen)'
                         r'|off( the)? (screen|frame|picture)|disappear|left the (frame|picture|shot|screen)|leaves? the (frame|picture|shot|screen)'
                         r'|no longer (shown|present)|gone from the (frame|picture|screen)', re.I)


class Refused(Exception):
    """An answer that cannot be applied (the reviewer may not give it, or the boxes changed)."""


def _load(text, default=None):
    if text in (None, ''):
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --- what the episode holds ------------------------------------------------------------------------------------------

def profile_context(conn, episode_id, run_id=None):
    from . import autopilot_profile
    profile = autopilot_profile.profile_for(conn, episode_id, run_id, KIND_BOXES)
    relevance = {cid: (entry or {}).get('relevance', 'auto') for cid, entry in (profile.get('characters') or {}).items()}
    pairs = {frozenset(pair) for pair in profile.get('look_alikes') or [] if len(pair) == 2}
    claude = profile.get('claude') or {}
    return {'relevance': relevance, 'look_alikes': pairs, 'modes': profile.get('modes') or {},
            'claude': bool(claude.get('enabled', True)), 'claude_items': int(claude.get('max_items_per_run', 0) or 0)}


def policy_for(conn, episode_id, run_id=None):
    context = profile_context(conn, episode_id, run_id)
    return {'boxes': context['modes'].get(KIND_BOXES, 'apply'), 'targets': context['modes'].get(KIND_TARGETS, 'apply'),
            'claude': context['claude'], 'claude_items': context['claude_items'],
            'relevance': context['relevance'], 'look_alikes': [sorted(pair) for pair in context['look_alikes']]}


def episode_cuts(conn, episode_id, first_index=0, last_index=None):
    return [dict(row) for row in conn.execute("""SELECT id,cut_index,start_time,end_time,analysis_enabled,no_characters FROM cuts
        WHERE episode_id=? AND active=1 AND cut_index>=? AND (? IS NULL OR cut_index<=?) ORDER BY cut_index""",
                                              (episode_id, first_index, last_index, last_index))]


def episode_boxes(conn, episode_id, first_index=0, last_index=None):
    """``(cuts, boxes)``: every first-frame box of each active cut's latest detection run that still fits the cut.

    A box carries its cut, shortlist ``[(character_id, score)]``, ``small`` (too small to identify anyone), ``fill``
    (box height / frame height) and the run's frame time. Cuts whose run is not complete are listed with ``pending``."""
    cuts = episode_cuts(conn, episode_id, first_index, last_index)
    boxes = []
    for cut in cuts:
        run = conn.execute("""SELECT id,cut_start,cut_end,frames_json,phase FROM character_detection_runs WHERE cut_id=?
            ORDER BY rowid DESC LIMIT 1""", (cut['id'],)).fetchone()
        cut['run_id'] = run['id'] if run else None
        cut['pending'] = not run or run['phase'] != 'complete' or abs(run['cut_start'] - cut['start_time']) > 1e-6 or abs(run['cut_end'] - cut['end_time']) > 1e-6
        if not run:
            continue
        frame = next((item for item in _load(run['frames_json'], []) if item.get('position') == 'first'), None) or {}
        size = frame.get('image_size') or [1280, 720]
        for row in conn.execute("""SELECT id,box_json,character_context_id,label_source,candidates_json,recognition_use FROM character_detections
                WHERE run_id=? AND position='first' ORDER BY rowid""", (run['id'],)):
            box = _load(row['box_json'], [0, 0, 0, 0])
            ranked = [(item['character_id'], float(item['score'])) for item in _load(row['candidates_json'], []) or []
                      if isinstance(item, dict) and item.get('character_id')]
            boxes.append({'id': row['id'], 'cut_id': cut['id'], 'cut_index': cut['cut_index'], 'run_id': run['id'], 'box': box,
                          'image_size': size, 'time': frame.get('time'), 'character_context_id': row['character_context_id'],
                          'label_source': row['label_source'], 'candidates': ranked,
                          'small': anime_gallery.box_problem(box, size) == 'small',
                          'fill': round(max(0.0, box[3] - box[1]) / max(1, size[1]), 3), 'pending': cut['pending']})
    return cuts, boxes


def neighbour_characters(cuts, boxes):
    """{cut_id: characters labelled in the active cut just before or just after it}."""
    by_cut = defaultdict(set)
    for box in boxes:
        if box['character_context_id']:
            by_cut[box['cut_id']].add(box['character_context_id'])
    near = {}
    for index, cut in enumerate(cuts):
        found = set()
        for other in cuts[max(0, index - 1):index] + cuts[index + 1:index + 2]:
            found |= by_cut.get(other['id'], set())
        near[cut['id']] = found
    return near


# --- deciding one box -------------------------------------------------------------------------------------------------

def _look_alike(ranked, pairs):
    return len(ranked) > 1 and frozenset((ranked[0][0], ranked[1][0])) in pairs and ranked[0][1] - ranked[1][1] < LOOK_ALIKE_MARGIN


def decide_box(box, *, cut_boxes, near, relevance, look_alikes, has_vector=False):
    """What the agent does with one first-frame box. Pure: ``{action, character?, why, confidence?, rule}``.

    Actions: ``name`` (an unnamed box gets a character), ``review`` (an automatic label for Claude's sheet),
    ``remove`` (a duplicate box, or a character the profile calls background), ``claude`` (close: the second reviewer looks),
    ``unknown`` (nobody in the gallery resembles it: grouped with the episode's other unknown people), ``keep`` (nothing to do)
    and ``skip`` (no shortlist and no vector: recognition has not run on it)."""
    source = box['label_source']
    if source not in (None, 'auto'):
        return {'action': 'keep', 'why': f'labelled by {source}', 'rule': 'decided'}
    ranked = box['candidates']
    others = [item for item in cut_boxes if item['id'] != box['id'] and item['character_context_id']]
    if source == 'auto':
        # Rules never confirm an automatic label: on the boxes the user labelled by hand, the shortlist leader at similarity
        # >= 0.80 with a 0.06 lead was their label only 72% of the time (autopilot/replay/box_names.py). Claude looks at a
        # sheet instead, least sure first.
        label = box['character_context_id']
        top, score = ranked[0] if ranked else (None, 0.0)
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
        if top != label:
            why, rule = 'the shortlist puts another character first', 'shortlist_disagrees'
        elif _look_alike(ranked, look_alikes):
            why, rule = 'a look-alike pair is close', 'look_alike'
        elif relevance.get(label) == 'background':
            why, rule = 'the profile calls this character background', 'background_label'
        else:
            why, rule = f'similarity {score:.2f}, {margin:.2f} ahead of the next', 'review'
        return {'action': 'review', 'character': label, 'why': why, 'rule': rule, 'confidence': round(score if top == label else 0.0, 3)}
    if not ranked:
        return {'action': 'unknown' if has_vector else 'skip', 'why': 'no shortlist' if has_vector else 'recognition has not run on this box',
                'rule': 'no_shortlist'}
    top, score = ranked[0]
    margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    taken = [item for item in others if item['character_context_id'] == top]
    if taken:
        if any(iou(box['box'], item['box']) >= DUPLICATE_IOU for item in taken):
            return {'action': 'remove', 'character': top, 'why': 'a second box over a person already boxed and named', 'rule': 'duplicate', 'confidence': 1.0}
        return {'action': 'claude', 'character': top, 'why': 'its best match is already boxed in this cut', 'rule': 'already_present'}
    if _look_alike(ranked, look_alikes):
        return {'action': 'claude', 'character': top, 'why': 'a look-alike pair is close', 'rule': 'look_alike'}
    supported = top in near
    if relevance.get(top) == 'background' and score >= NAME_MIN_NEAR and margin >= NAME_MARGIN_NEAR:
        return {'action': 'remove', 'character': top, 'why': 'the profile calls this character background', 'rule': 'background', 'confidence': round(score, 3)}
    least, gap = (NAME_MIN_NEAR, NAME_MARGIN_NEAR) if supported else (NAME_MIN, NAME_MARGIN)
    if supported and relevance.get(top) == 'always':
        least = min(least, ALWAYS_MIN)
    if score >= least and margin >= gap:
        return {'action': 'name', 'character': top, 'confidence': round(score, 3), 'rule': 'near' if supported else 'shortlist',
                'why': f'similarity {score:.2f}, {margin:.2f} ahead of the next' + (', boxed in the cut next to it' if supported else '')}
    if score >= CLOSE_MIN:
        return {'action': 'claude', 'character': top, 'why': f'similarity {score:.2f}, {margin:.2f} ahead of the next', 'rule': 'close'}
    return {'action': 'unknown', 'why': f'nobody in the gallery resembles it (best {score:.2f})', 'rule': 'unknown'}


# --- unknown people ---------------------------------------------------------------------------------------------------

def stored_vectors(conn, detection_ids):
    """{detection_id: unit vector} from the cached CCIP (or DINOv3) embeddings, one model: the one most boxes have."""
    from . import anime_matching
    if not detection_ids:
        return {}
    rows = []
    for start in range(0, len(detection_ids), 500):
        chunk = detection_ids[start:start + 500]
        rows += conn.execute(f"""SELECT detection_id,model_id,model_revision,preprocessing,dimensions,embedding FROM character_detection_embeddings
            WHERE detection_id IN ({','.join('?' * len(chunk))})""", chunk).fetchall()
    if not rows:
        return {}
    model = Counter((row['model_id'], row['model_revision'], row['preprocessing']) for row in rows).most_common(1)[0][0]
    vectors = {}
    for row in rows:
        if (row['model_id'], row['model_revision'], row['preprocessing']) == model:
            try:
                vectors[row['detection_id']] = anime_matching.unpack_embedding(row['embedding'], row['dimensions'])
            except ValueError:
                continue
    return vectors


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def group_unknowns(boxes, vectors):
    """Groups of unknown boxes that look like one person: greedy, each box joins the group whose mean vector it resembles
    most (at least ``GROUP_MIN``), largest boxes first so a clear face seeds a group. Boxes without a vector stay alone."""
    groups = []
    for box in sorted(boxes, key=lambda item: (-item['fill'], item['id'])):
        vector = vectors.get(box['id'])
        best, best_score = None, GROUP_MIN
        if vector is not None:
            for group in groups:
                if group['mean'] is None:
                    continue
                score = _dot(vector, group['mean'])
                if score >= best_score:
                    best, best_score = group, score
        if best is None:
            groups.append({'boxes': [box], 'sum': list(vector) if vector is not None else None, 'mean': _unit(vector) if vector is not None else None})
        else:
            best['boxes'].append(box)
            best['sum'] = [x + y for x, y in zip(best['sum'], vector)]
            best['mean'] = _unit(best['sum'])
    return [{'boxes': group['boxes']} for group in groups]


def weigh(group):
    """The relevance rule for a group of unknown boxes: ``matters``, ``background`` or ``borderline``, with why."""
    cuts = {box['cut_id'] for box in group['boxes']}
    fill = max(box['fill'] for box in group['boxes'])
    tall = {box['cut_id'] for box in group['boxes'] if box['fill'] >= RECURRING_FILL}
    if len(cuts) >= MATTERS_CUTS:
        return 'matters', f'appears in {len(cuts)} cuts'
    if fill >= MATTERS_FILL:
        return 'matters', f'a large figure ({round(fill * 100)}% of the frame height)'
    if len(tall) >= 2:
        return 'matters', f'clearly visible in {len(tall)} cuts'
    if len(cuts) == 1 and all(box['small'] or box['fill'] < BACKGROUND_FILL for box in group['boxes']):
        return 'background', 'one small appearance (no usable face)'
    return 'borderline', f'{len(cuts)} cut(s), largest box {round(fill * 100)}% of the frame height'


# --- target frames ----------------------------------------------------------------------------------------------------

def keyframe_states(conn, cut_id):
    """{ordinal: {character_id: state text}} from the cut's current keyframe analysis (Qwen's per-frame states)."""
    states = defaultdict(dict)
    run = conn.execute("""SELECT k.id FROM keyframe_analysis_runs k JOIN cuts c ON c.id=k.cut_id WHERE k.cut_id=? AND k.phase='complete'
        AND k.cut_start=c.start_time AND k.cut_end=c.end_time ORDER BY k.rowid DESC LIMIT 1""", (cut_id,)).fetchone()
    if not run:
        return states
    for row in conn.execute('SELECT ordinal,states_json FROM keyframe_analysis_frames WHERE run_id=?', (run['id'],)):
        for item in (_load(row['states_json'], {}) or {}).get('qwen') or []:
            if isinstance(item, dict) and item.get('character_context_id'):
                states[row['ordinal']][item['character_context_id']] = str(item.get('state') or '')
    return states


def target_runs(conn, cuts):
    """The current target-frame run of each cut (its bounds still fit), with its slots and boxes per frame."""
    found = []
    for cut in cuts:
        run = conn.execute("""SELECT * FROM target_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=? ORDER BY rowid DESC LIMIT 1""",
                           (cut['id'], cut['start_time'], cut['end_time'])).fetchone()
        if not run:
            continue
        slots = {row['slot']: dict(row) for row in conn.execute('SELECT * FROM target_slots WHERE run_id=?', (run['id'],))}
        here = defaultdict(set)
        for row in conn.execute('SELECT frame_ordinal,slot FROM target_detections WHERE run_id=?', (run['id'],)):
            here[row['frame_ordinal']].add(row['slot'])
        found.append({'cut': cut, 'run': dict(run), 'frames': _load(run['frames_json'], []), 'slots': slots, 'here': here})
    return found


def decide_frame(frame, *, opening, here, slots, states, crops=()):
    """``{action: reviewed|ask, why, rule}`` for one flagged, unreviewed target frame (None when nothing is flagged).

    ``opening``/``here``: slot numbers on the opening frame and on this frame; ``slots``: slot rows (their character);
    ``states``: {character_id: keyframe state text} at this frame, empty without a keyframe analysis; ``crops``: the tracker's
    unassigned crops on this frame (``candidates_json`` entries)."""
    review = frame.get('review') if isinstance(frame, dict) else None
    if frame.get('mirror') or not isinstance(review, dict) or review.get('reviewed') or not review.get('flags'):
        return None
    flags = review['flags']
    keys = set(flags)
    if keys == {'count_changed'}:
        count = flags['count_changed'] or {}
        missing = [slot for slot in opening if slot not in here and not (slots.get(slot) or {}).get('ignored')]
        if count.get('here', 0) < count.get('first', 0) and missing and not (here - set(opening)):
            gone = []
            for slot in missing:
                character = (slots.get(slot) or {}).get('character_context_id')
                text = states.get(character) if character else None
                if not text or not NOT_VISIBLE.search(text):
                    return {'action': 'ask', 'why': 'fewer people than the opening frame, and the keyframe states do not say who left', 'rule': 'count_unexplained'}
                gone.append(character)
            return {'action': 'reviewed', 'why': 'the keyframe states say the missing people left the picture', 'rule': 'left_picture', 'characters': gone}
    if keys == {'unmatched'}:
        # Everyone from the opening frame is still boxed and the extra crop fits no colour: on the user's own reviews, every
        # frame of this shape (16/16) was left unchanged (autopilot/replay/target_frames.py). The keyframe states, when the
        # cut has them, must not name anyone the opening frame lacks.
        cast = {(slots.get(slot) or {}).get('character_context_id') for slot in opening}
        unmatched = [crop for crop in crops if crop.get('tag') == 'unmatched']
        if (set(opening) <= here and unmatched and not any(crop.get('suggested_slot') is not None for crop in unmatched)
                and not (set(states) - cast)):
            return {'action': 'reviewed', 'why': 'everyone from the opening frame is boxed; the extra crop fits no colour', 'rule': 'extra_crop'}
    reasons = {'count_changed': 'the headcount changed', 'unmatched': 'a crop nobody has a colour for', 'suspected_merge': 'two people may share one box',
               'low_match': 'a colour matches weakly', 'jump': 'a box jumped', 'duplicate_character': 'one character has two colours'}
    return {'action': 'ask', 'why': ', '.join(reasons.get(key, key) for key in sorted(keys)), 'rule': 'flags'}


# --- the job ----------------------------------------------------------------------------------------------------------

def queue(conn, episode_id, *, run_id=None, first_index=0, last_index=None, boxes=True, targets=True):
    """Queue the character agent inside the caller's transaction; None when the profile switched both parts off."""
    policy = policy_for(conn, episode_id, run_id)
    parts = {'boxes': boxes and policy['boxes'] != 'off', 'targets': targets and policy['targets'] != 'off'}
    if not any(parts.values()):
        return None
    payload = {'policy': policy, 'run_id': run_id, 'parts': parts, 'version': VERSION, 'first_index': first_index, 'last_index': last_index}
    waits = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts', 'missed_cuts', 'character_detect', 'keyframes'))
    if waits:
        payload['depends_on'] = waits
    job_id = worker.insert_job(conn, episode_id, PIPELINE, payload, priority_class='batch')
    worker.sync_episode_status(conn, episode_id)
    return job_id


def gather(conn, job):
    payload = _load(job.get('payload_json'), {}) or {}
    cuts, boxes = episode_boxes(conn, job['episode_id'], payload.get('first_index') or 0, payload.get('last_index'))
    runs = target_runs(conn, cuts)
    return {'boxes': [[box['id'], box['character_context_id'], box['label_source']] for box in boxes],
            'targets': [[item['run']['id'], item['run']['review_version'], len(item['run']['frames_json'])] for item in runs],
            'cuts': [[cut['id'], cut['start_time'], cut['end_time'], cut['analysis_enabled'], cut['no_characters']] for cut in cuts]}


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    """Each change marks its own cut's results stale (``opening_box_edited``); a reviewed frame changes no box."""


def claude_budget(conn, episode_id, run_id, policy):
    if not policy.get('claude'):
        return 0
    marks = ','.join('?' for _ in ESCALATIONS)
    if run_id:
        used = conn.execute(f"SELECT COUNT(*) FROM agent_escalations WHERE kind IN ({marks}) AND run_id=? AND group_key LIKE '%:claude'",
                            (*ESCALATIONS, run_id)).fetchone()[0]
    else:
        used = conn.execute(f"SELECT COUNT(*) FROM agent_escalations WHERE kind IN ({marks}) AND episode_id=? AND status='open' AND reviewer='claude'",
                            (*ESCALATIONS, episode_id)).fetchone()[0]
    return int(policy.get('claude_items', 0)) - used


def titles(conn, episode_id):
    project = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    return {row['id']: row['title'] for row in conn.execute("SELECT id,title FROM context_items WHERE project_id=? AND kind='character'", (project['project_id'],))}


def _box_state(conn, detection_id):
    row = conn.execute('SELECT id,run_id,character_context_id,label_source,match_reason,box_json FROM character_detections WHERE id=?', (detection_id,)).fetchone()
    return dict(row) if row else None


def set_label(conn, box, character_id, source, *, kind=KIND_BOXES, tier='rules', run_id=None, episode_id, why='', confidence=None, evidence=None, names=None):
    """Label (or unlabel) one box and log it. ``box`` is the row as read now; nothing happens when it changed meanwhile."""
    before = {'character_context_id': box['character_context_id'], 'label_source': box['label_source'], 'match_reason': box.get('match_reason')}
    reason = f'Agent ({tier}): {why}'[:2000] if character_id else None
    cursor = conn.execute("""UPDATE character_detections SET character_context_id=?,label_source=?,match_reason=?
        WHERE id=? AND character_context_id IS ? AND label_source IS ?""",
                          (character_id, source if character_id else None, reason, box['id'], box['character_context_id'], box['label_source']))
    if not cursor.rowcount:
        return None
    if before['character_context_id'] != character_id:
        structured_results.opening_box_edited(conn, box['run_id'], 'first')
    name = (names or {}).get(character_id, 'nobody') if character_id else 'nobody'
    verb = 'Confirmed' if before['character_context_id'] == character_id else ('Named' if character_id else 'Cleared the name of')
    cut = conn.execute('SELECT cut_id FROM character_detection_runs WHERE id=?', (box['run_id'],)).fetchone()
    return agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=cut['cut_id'] if cut else None, kind=kind, tier=tier,
                            target_table='character_detections', target_id=box['id'], before=before,
                            after={'character_context_id': character_id, 'label_source': source if character_id else None, 'match_reason': reason},
                            evidence=evidence or {}, confidence=confidence, summary=f'{verb} a box as {name}: {why}'[:500])


def remove_box(conn, box_id, *, tier='rules', run_id=None, episode_id, why='', evidence=None, confidence=None):
    """Delete one first-frame box (background or a duplicate) and log the whole row so undo can put it back."""
    if conn.execute('SELECT 1 FROM character_image_references WHERE detection_id=?', (box_id,)).fetchone():
        return None
    row = conn.execute('SELECT * FROM character_detections WHERE id=?', (box_id,)).fetchone()
    if row is None or row['label_source'] == 'manual':
        return None
    saved = dict(row)
    conn.execute('DELETE FROM character_detections WHERE id=?', (box_id,))
    structured_results.opening_box_edited(conn, saved['run_id'], saved['position'])
    cut = conn.execute('SELECT cut_id FROM character_detection_runs WHERE id=?', (saved['run_id'],)).fetchone()
    return agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=cut['cut_id'] if cut else None, kind=KIND_BOXES, tier=tier,
                            target_table='character_detections', target_id=box_id, before={'row': saved}, after={'removed': True},
                            evidence=evidence or {}, confidence=confidence, summary=f'Unboxed a person: {why}'[:500])


def mark_reviewed(conn, run, ordinal, *, tier='rules', run_id=None, episode_id, why='', evidence=None):
    """Mark one target frame reviewed (the flag is explained) and log it. No box changes, so nothing turns stale."""
    from . import target_reviews
    current = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (run['id'],)).fetchone()
    if current is None:
        return None
    frame = next((item for item in _load(current['frames_json'], []) if item.get('ordinal') == ordinal), None)
    if not frame or frame.get('mirror') or (frame.get('review') or {}).get('reviewed'):
        return None
    target_reviews.update_frames(conn, current, {ordinal}, reviewed=True)
    label = f"F{next((i for i, item in enumerate(_load(current['frames_json'], []), 1) if item.get('ordinal') == ordinal), '?')}"
    return agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=current['cut_id'], kind=KIND_TARGETS, tier=tier,
                            target_table='target_detection_runs', target_id=current['id'], before={'ordinal': ordinal, 'reviewed': False},
                            after={'ordinal': ordinal, 'reviewed': True}, evidence={'frame': label, **(evidence or {})},
                            summary=f'Marked target frame {label} reviewed: {why}'[:500])


def asked_boxes(conn, episode_id):
    """Box ids already waiting in an open question (a later run does not ask about them again)."""
    found = set()
    for row in conn.execute(f"SELECT evidence_json FROM agent_escalations WHERE episode_id=? AND status='open' AND kind IN (?,?)",
                            (episode_id, ESC_LABELS, ESC_PERSON)):
        found |= {box.get('id') for box in (_load(row['evidence_json'], {}) or {}).get('boxes') or []}
    return found


def ask_labels(conn, episode_id, character_id, chunk, reviewer, purpose, names, counts, run_id):
    """One ``box_label`` question for up to ``SHEET_ROWS`` boxes proposed as one character."""
    name = names.get(character_id, 'someone')
    key = f"{purpose}:{character_id}:{chunk[0]['id']}" + (':claude' if reviewer == 'claude' else '')
    question = f'Are these {len(chunk)} box(es) {name}?' if character_id != 'unknown' else f'Who are these {len(chunk)} box(es)?'
    agent_log.escalate(conn, episode_id=episode_id, kind=ESC_LABELS, group_key=key, run_id=run_id, reviewer=reviewer, question=question,
                       cut_ids=[item['cut_id'] for item in chunk], options=['confirm', 'reject'],
                       proposal={'character_context_id': None if character_id == 'unknown' else character_id, 'name': name, 'purpose': purpose},
                       evidence={'boxes': chunk, 'purpose': purpose})
    counts['asked_labels'] += 1


def publish_boxes(conn, episode_id, payload, counts, names):
    policy, run_id = payload.get('policy') or {}, payload.get('run_id')
    mode = policy.get('boxes', 'apply')
    cuts, boxes = episode_boxes(conn, episode_id, payload.get('first_index') or 0, payload.get('last_index'))
    eligible = {cut['id'] for cut in cuts if cut['analysis_enabled'] and not cut['no_characters'] and not cut['pending']}
    counts['cuts_pending'] = sum(1 for cut in cuts if cut['analysis_enabled'] and not cut['no_characters'] and cut['pending'])
    boxes = [box for box in boxes if box['cut_id'] in eligible]
    near = neighbour_characters([cut for cut in cuts if cut['analysis_enabled']], boxes)
    by_cut = defaultdict(list)
    for box in boxes:
        by_cut[box['cut_id']].append(box)
    open_boxes = [box for box in boxes if box['label_source'] is None]
    vectors = stored_vectors(conn, [box['id'] for box in open_boxes])
    relevance = policy.get('relevance') or {}
    pairs = {frozenset(pair) for pair in policy.get('look_alikes') or []}
    asks, reviews, unknowns = defaultdict(list), defaultdict(list), []
    budget = claude_budget(conn, episode_id, run_id, policy)
    waiting = asked_boxes(conn, episode_id)
    for box in boxes:
        if box['id'] in waiting:
            counts['boxes_waiting'] += 1
            continue
        decision = decide_box(box, cut_boxes=by_cut[box['cut_id']], near=near.get(box['cut_id'], set()), relevance=relevance,
                              look_alikes=pairs, has_vector=box['id'] in vectors)
        action = decision['action']
        if action in ('keep', 'skip'):
            counts['boxes_' + action] += 1
            continue
        if action == 'unknown':
            unknowns.append(box)
            continue
        brief = {**_box_brief(box, names), 'proposal': decision.get('character'), 'why': decision['why'], 'rule': decision['rule'],
                 'confidence': decision.get('confidence')}
        if action == 'review':
            reviews[decision['character']].append(brief)
            continue
        if action == 'claude' or mode == 'ask':
            asks[decision.get('character') or 'unknown'].append({**brief, 'proposed_action': action if action != 'claude' else 'name'})
            continue
        evidence = {'rule': decision['rule'], 'candidates': [[cid, score] for cid, score in box['candidates'][:3]], 'cut': box['cut_index'] + 1}
        if action == 'name':
            done = set_label(conn, box, decision['character'], agent_log.AGENT, run_id=run_id, episode_id=episode_id, why=decision['why'],
                             confidence=decision.get('confidence'), evidence=evidence, names=names)
            if done:
                box['character_context_id'], box['label_source'] = decision['character'], agent_log.AGENT
        else:
            done = remove_box(conn, box['id'], run_id=run_id, episode_id=episode_id, why=decision['why'], evidence=evidence, confidence=decision.get('confidence'))
        counts[f'boxes_{action}' + ('' if done else '_changed')] += 1

    # Unsure names of unnamed boxes (a gate for the person): Claude first while the budget lasts, else the person.
    for character_id, items in asks.items():
        for start in range(0, len(items), SHEET_ROWS):
            reviewer = 'claude' if budget > 0 and mode == 'apply' else 'human'
            budget -= reviewer == 'claude'
            ask_labels(conn, episode_id, character_id, items[start:start + SHEET_ROWS], reviewer, 'name', names, counts, run_id)
    # Unknown people: grouped across the episode, weighed by the relevance rule.
    for group in group_unknowns(unknowns, vectors):
        weight, why = weigh(group)
        members = group['boxes']
        representative = max(members, key=lambda item: (item['fill'], item['id']))
        if weight == 'background' and mode == 'apply':
            for box in members:
                done = remove_box(conn, box['id'], run_id=run_id, episode_id=episode_id, why=f'background person: {why}',
                                  evidence={'rule': 'background', 'cut': box['cut_index'] + 1, 'fill': box['fill']})
                counts['boxes_background' + ('' if done else '_changed')] += 1
            continue
        reviewer = 'claude' if weight == 'borderline' and budget > 0 and mode != 'ask' else 'human'
        if reviewer == 'claude':
            budget -= 1
        key = f"person:{representative['id']}" + (':claude' if reviewer == 'claude' else '')
        cuts_seen = sorted({box['cut_index'] + 1 for box in members})
        agent_log.escalate(conn, episode_id=episode_id, kind=ESC_PERSON, group_key=key, run_id=run_id, reviewer=reviewer,
                           question=(f'Unknown person in {len(cuts_seen)} cut(s) ({why}). Who is it?' if weight != 'background'
                                     else f'Unknown small figure ({why}). Background?'),
                           cut_ids=list(dict.fromkeys(box['cut_id'] for box in members)),
                           options=['character', 'new', 'background', 'leave'],
                           proposal={'weight': weight, 'representative': representative['id'],
                                     'candidates': [[cid, names.get(cid), score] for cid, score in representative['candidates'][:3]]},
                           evidence={'weight': weight, 'why': why, 'cuts': cuts_seen, 'boxes': [_box_brief(box, names) for box in members[:60]],
                                     'more': max(0, len(members) - 60)})
        counts['asked_people'] += 1
        counts['unknown_boxes'] += len(members)
    return reviews


def publish_reviews(conn, episode_id, payload, counts, names, reviews):
    """Automatic labels (not a gate): only to Claude, least sure first, with what the budget leaves after the gates;
    the rest wait for a later run."""
    policy, run_id = payload.get('policy') or {}, payload.get('run_id')
    budget = claude_budget(conn, episode_id, run_id, policy)
    for character_id, items in sorted(reviews.items(), key=lambda pair: min(item['confidence'] or 0 for item in pair[1])):
        items.sort(key=lambda item: (item['rule'] == 'review', item['confidence'] or 0))
        for start in range(0, len(items), SHEET_ROWS):
            if budget <= 0 or policy.get('boxes', 'apply') != 'apply':
                counts['auto_labels_left'] += len(items) - start
                break
            budget -= 1
            ask_labels(conn, episode_id, character_id, items[start:start + SHEET_ROWS], 'claude', 'auto', names, counts, run_id)


def _box_brief(box, names):
    return {'id': box['id'], 'cut_id': box['cut_id'], 'cut': box['cut_index'] + 1, 'box': [round(value, 1) for value in box['box']],
            'fill': box['fill'], 'small': box['small'], 'current': names.get(box['character_context_id']) if box['character_context_id'] else None,
            'label_source': box['label_source'], 'candidates': [[cid, names.get(cid), round(score, 3)] for cid, score in box['candidates'][:3]]}


def publish_targets(conn, episode_id, payload, counts, names):
    policy, run_id = payload.get('policy') or {}, payload.get('run_id')
    mode = policy.get('targets', 'apply')
    budget = claude_budget(conn, episode_id, run_id, policy)
    cuts = [cut for cut in episode_cuts(conn, episode_id, payload.get('first_index') or 0, payload.get('last_index'))
            if cut['analysis_enabled'] and not cut['no_characters']]
    for item in target_runs(conn, cuts):
        run, frames, slots = item['run'], item['frames'], item['slots']
        if not frames:
            continue
        opening = item['here'].get(frames[0].get('ordinal'), set())
        states = keyframe_states(conn, item['cut']['id']) if any((frame.get('review') or {}).get('flags') for frame in frames) else {}
        crops = defaultdict(list)
        for crop in _load(run['candidates_json'], []) or []:
            crops[crop.get('frame_ordinal')].append(crop)
        asks = []
        for number, frame in enumerate(frames, 1):
            decision = decide_frame(frame, opening=opening, here=item['here'].get(frame.get('ordinal'), set()), slots=slots,
                                    states=states.get(frame.get('ordinal'), {}), crops=crops.get(frame.get('ordinal'), []))
            if decision is None:
                continue
            if decision['action'] == 'reviewed' and mode == 'apply':
                done = mark_reviewed(conn, run, frame['ordinal'], run_id=run_id, episode_id=episode_id, why=decision['why'],
                                     evidence={'rule': decision['rule'], 'characters': [names.get(cid) for cid in decision.get('characters', [])]})
                counts['frames_reviewed' + ('' if done else '_changed')] += 1
                continue
            asks.append({'ordinal': frame['ordinal'], 'frame': f'F{number}', 'flags': sorted((frame.get('review') or {}).get('flags') or {}),
                         'score': (frame.get('review') or {}).get('score'), 'why': decision['why'],
                         'proposal': 'reviewed' if decision['action'] == 'reviewed' else None})
        unnamed = [number for number, slot in slots.items() if not slot['ignored'] and not slot['character_context_id']
                   and any(number in found for found in item['here'].values())]
        if not asks and not unnamed:
            continue
        reviewer = 'claude' if asks and not unnamed and budget > 0 and mode == 'apply' else 'human'
        if reviewer == 'claude':
            budget -= 1
        cut = item['cut']
        key = f"cut:{cut['id']}" + (':claude' if reviewer == 'claude' else '')
        parts = [f'{len(asks)} flagged frame(s)'] if asks else []
        if unnamed:
            parts.append(f'{len(unnamed)} unnamed colour(s)')
        agent_log.escalate(conn, episode_id=episode_id, kind=ESC_TARGET, group_key=key, run_id=run_id, reviewer=reviewer,
                           question=f"Target frames of cut {cut['cut_index'] + 1}: {' and '.join(parts)}. Are the boxes right?",
                           cut_ids=[cut['id']], options=['ok', 'person'],
                           evidence={'run_id': run['id'], 'frames': asks, 'unnamed_colours': [number + 1 for number in unnamed], 'cut': cut['cut_index'] + 1})
        counts['asked_target_cuts'] += 1


def close_settled(conn, episode_id):
    """Resolve open questions whose items were settled elsewhere: boxes labelled or removed, frames reviewed or replaced."""
    closed = 0
    for item in agent_log.escalations(conn, episode_id, 'open'):
        if item['kind'] in (ESC_LABELS, ESC_PERSON):
            ids = [box['id'] for box in (item['evidence'] or {}).get('boxes') or []]
            rows = {row['id']: row for row in conn.execute(f"SELECT id,label_source FROM character_detections WHERE id IN ({','.join('?' for _ in ids)})", ids)} if ids else {}
            purpose = (item['evidence'] or {}).get('purpose')
            # A box is settled once it is gone, the person or the agent labelled it, or (an automatic label under review) it was cleared;
            # an unnamed box counts as settled by any label, recognition's included.
            settled = all(box_id not in rows or rows[box_id]['label_source'] in ('manual', agent_log.AGENT)
                          or (purpose == 'auto' and rows[box_id]['label_source'] is None)
                          or (purpose != 'auto' and rows[box_id]['label_source'] == 'auto') for box_id in ids)
        elif item['kind'] == ESC_TARGET:
            evidence = item['evidence'] or {}
            run = conn.execute('SELECT frames_json FROM target_detection_runs WHERE id=?', (evidence.get('run_id'),)).fetchone()
            if run is None:
                settled = True
            else:
                frames = {frame.get('ordinal'): frame for frame in _load(run['frames_json'], [])}
                wanted = [frame['ordinal'] for frame in evidence.get('frames') or []]
                settled = all((frames.get(ordinal, {}).get('review') or {}).get('reviewed') or not (frames.get(ordinal, {}).get('review') or {}).get('flags')
                              for ordinal in wanted) and not evidence.get('unnamed_colours')
        else:
            continue
        if settled and agent_log.close_escalation(conn, item['id'], 'resolved', {'by': 'settled elsewhere'}):
            closed += 1
    return closed


def publish(conn, job, payload):
    episode_id = job['episode_id']
    names = titles(conn, episode_id)
    counts = Counter()
    counts['closed'] = close_settled(conn, episode_id)
    parts = payload.get('parts') or {'boxes': True, 'targets': True}
    reviews = publish_boxes(conn, episode_id, payload, counts, names) if parts.get('boxes') else {}
    if parts.get('targets'):
        publish_targets(conn, episode_id, payload, counts, names)
    # Gates (unnamed boxes, unknown people, flagged frames) took the Claude budget first; automatic labels get what is left.
    if reviews:
        publish_reviews(conn, episode_id, payload, counts, names, reviews)
    return dict(counts)


def process(job_id, owner):
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    with db.connect() as conn:
        conn.execute('BEGIN')
        snapshot = job_inputs.JobInputs(conn, job, owner)
    worker.stage(job_id, owner, job['episode_id'], 'Naming boxes and settling target frames')
    with snapshot.publish(invalidate=False) as conn:
        counts = publish(conn, job, payload)
        named = counts.get('boxes_name', 0)
        removed = counts.get('boxes_remove', 0) + counts.get('boxes_background', 0)
        asked = counts.get('asked_labels', 0) + counts.get('asked_people', 0) + counts.get('asked_target_cuts', 0)
        stage = f"Complete · {named} named, {removed} unboxed, {counts.get('frames_reviewed', 0)} frames settled, {asked} asked"
        payload['result'] = counts
        conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,payload_json=?,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (stage, json.dumps(payload), job_id))
        worker.sync_episode_status(conn, job['episode_id'])
    worker.wake_event.set()


# --- answers ------------------------------------------------------------------------------------------------------------

def _open(conn, escalation_id, kinds):
    row = conn.execute('SELECT * FROM agent_escalations WHERE id=?', (escalation_id,)).fetchone()
    if row is None or row['kind'] not in kinds:
        raise KeyError(escalation_id)
    item = agent_log.escalation_view(row)
    if item['status'] != 'open':
        raise Refused('This question is already closed')
    return item


def _check_reviewer(item, tier):
    if tier == 'claude' and item['reviewer'] != 'claude':
        raise Refused("This question is the person's; Claude cannot answer it")


def answer_labels(conn, escalation_id, *, confirm=(), reject=(), character_id=None, tier='human', run_id=None, note=None):
    """Answer a ``box_label`` question: ``confirm`` the proposed (or a chosen) character on these boxes, ``reject`` (clear) others.

    The person's confirmation is their own choice (``manual``, a gallery example, D10); Claude's is the agent's."""
    item = _open(conn, escalation_id, (ESC_LABELS,))
    _check_reviewer(item, tier)
    episode_id = item['episode_id']
    names = titles(conn, episode_id)
    proposed = character_id or (item['proposal'] or {}).get('character_context_id')
    if confirm and not proposed:
        raise Refused('Name the character these boxes show')
    if proposed and proposed not in names:
        raise Refused('Choose a character from this project')
    listed = {box['id'] for box in (item['evidence'] or {}).get('boxes') or []}
    if set(confirm) - listed or set(reject) - listed:
        raise Refused('Answer only the boxes this question lists')
    if not confirm and not reject:
        if tier == 'claude':
            agent_log.hand_to_person(conn, escalation_id, note)
            return {'handed_to_person': True}
        raise Refused('Confirm or reject at least one box')
    source = 'manual' if tier == 'human' else agent_log.AGENT
    done = []
    for box_id in confirm:
        box = _box_state(conn, box_id)
        if box and box['label_source'] not in ('manual',):
            decision = set_label(conn, box, proposed, source, tier=tier, run_id=run_id, episode_id=episode_id,
                                 why=f"{tier} confirmed it on the sheet" + (f': {note}' if note else ''), evidence={'escalation': escalation_id}, names=names)
            done.append(decision)
    for box_id in reject:
        box = _box_state(conn, box_id)
        if box and box['label_source'] in (None, 'auto', agent_log.AGENT) and box['character_context_id']:
            done.append(set_label(conn, box, None, None, tier=tier, run_id=run_id, episode_id=episode_id,
                                  why=f'{tier} said it is not {names.get(box["character_context_id"])}', evidence={'escalation': escalation_id}, names=names))
    rest = listed - set(confirm) - set(reject)
    if rest and tier == 'claude':
        agent_log.hand_to_person(conn, escalation_id, note or f'{len(rest)} box(es) left for the person')
        return {'decisions': [d for d in done if d], 'handed_to_person': True}
    agent_log.close_escalation(conn, escalation_id, 'resolved', {'tier': tier, 'confirm': list(confirm), 'reject': list(reject), 'note': note})
    return {'decisions': [d for d in done if d], 'resolved': True}


def answer_person(conn, escalation_id, *, choice, character_id=None, title=None, content='', is_main=False, episode_only=False,
                  boxes=None, tier='human', run_id=None, note=None):
    """Answer a ``new_person`` question. ``character``: an existing character (the representative box becomes the person's own
    ``manual`` label when the person answers, the rest are the agent's); ``new``: create the character (person only) from the
    representative box; ``background``: unbox the group; ``leave``: close it without changes; ``unsure`` (Claude): hand it on.
    ``boxes`` limits the answer to some of the group's boxes (the others stay open)."""
    item = _open(conn, escalation_id, (ESC_PERSON,))
    _check_reviewer(item, tier)
    episode_id = item['episode_id']
    names = titles(conn, episode_id)
    listed = [box['id'] for box in (item['evidence'] or {}).get('boxes') or []]
    chosen = [box_id for box_id in (boxes or listed) if box_id in listed]
    if tier == 'claude' and choice in ('new', 'leave', 'unsure'):
        agent_log.hand_to_person(conn, escalation_id, note or ('Claude thinks this is a new character' if choice == 'new' else None))
        return {'handed_to_person': True}
    if choice == 'unsure':
        raise Refused('Only Claude can answer unsure')
    done, created = [], None
    if choice in ('character', 'new'):
        representative = (item['proposal'] or {}).get('representative')
        if representative not in chosen:
            representative = chosen[0] if chosen else None
        if representative is None:
            raise Refused('None of the listed boxes is left')
        if choice == 'new':
            if not (title or '').strip():
                raise Refused('Enter a character name')
            created = create_character(conn, representative, {'title': title.strip(), 'content': content or '', 'is_main': bool(is_main),
                                                              'episode_only': bool(episode_only)})
            character_id = created['character']['id']
            names[character_id] = title.strip()
            done.append(agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=None, kind=KIND_BOXES, tier='human',
                                         target_table='context_items', target_id=character_id, before=None, after={'title': title.strip()},
                                         evidence={'escalation': escalation_id}, summary=f'Created the character {title.strip()} from an unknown person'))
        elif character_id not in names:
            raise Refused('Choose a character from this project')
        for box_id in chosen:
            box = _box_state(conn, box_id)
            if not box or box['label_source'] == 'manual' or (choice == 'new' and box_id == representative):
                continue
            source = 'manual' if tier == 'human' and box_id == representative else agent_log.AGENT
            done.append(set_label(conn, box, character_id, source, tier=tier, run_id=run_id, episode_id=episode_id,
                                  why=f'{tier} named the group' + (f': {note}' if note else ''), evidence={'escalation': escalation_id}, names=names))
    elif choice == 'background':
        for box_id in chosen:
            done.append(remove_box(conn, box_id, tier=tier, run_id=run_id, episode_id=episode_id, why=f'{tier} called the person background',
                                   evidence={'escalation': escalation_id}))
    elif choice != 'leave':
        raise Refused('Unknown answer')
    remaining = [box_id for box_id in listed if box_id not in chosen]
    if remaining:
        evidence = item['evidence'] or {}
        evidence['boxes'] = [box for box in evidence.get('boxes') or [] if box['id'] in remaining]
        conn.execute('UPDATE agent_escalations SET evidence_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (json.dumps(evidence), escalation_id))
    else:
        agent_log.close_escalation(conn, escalation_id, 'resolved', {'tier': tier, 'choice': choice, 'character_context_id': character_id, 'note': note})
    return {'decisions': [d for d in done if d], 'created': created, 'resolved': not remaining}


def answer_target(conn, escalation_id, *, frames, tier='human', run_id=None, note=None):
    """Answer a ``target_frame`` question: ``{ordinal: 'ok'|'person'}``. ``ok`` marks the frame reviewed; anything left
    (or ``person``) goes to the person when Claude answers."""
    item = _open(conn, escalation_id, (ESC_TARGET,))
    _check_reviewer(item, tier)
    evidence = item['evidence'] or {}
    run = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (evidence.get('run_id'),)).fetchone()
    if run is None:
        agent_log.close_escalation(conn, escalation_id, 'dismissed', {'why': 'the target run was replaced'})
        raise Refused('The target frames were detected again; this question is closed')
    listed = {frame['ordinal']: frame for frame in evidence.get('frames') or []}
    answers = {int(key): value for key, value in (frames or {}).items()}
    if set(answers) - set(listed):
        raise Refused('Answer only the frames this question lists')
    done = []
    for ordinal, value in answers.items():
        if value == 'ok':
            done.append(mark_reviewed(conn, run, ordinal, tier=tier, run_id=run_id, episode_id=item['episode_id'],
                                      why=f'{tier} checked it' + (f': {note}' if note else ''), evidence={'escalation': escalation_id}))
    left = [ordinal for ordinal in listed if answers.get(ordinal) != 'ok']
    if left or evidence.get('unnamed_colours'):
        evidence['frames'] = [listed[ordinal] for ordinal in left]
        conn.execute('UPDATE agent_escalations SET evidence_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (json.dumps(evidence), escalation_id))
        if tier == 'claude':
            agent_log.hand_to_person(conn, escalation_id, note)
        return {'decisions': [d for d in done if d], 'left': left, 'handed_to_person': tier == 'claude'}
    agent_log.close_escalation(conn, escalation_id, 'resolved', {'tier': tier, 'frames': answers, 'note': note})
    return {'decisions': [d for d in done if d], 'resolved': True}


def create_character(conn, detection_id, request):
    """Create a project character, label this box with it (the person's own label) and queue its profile description.

    Shared by ``POST /api/character-detections/{id}/new-character`` and the inbox's answer to an unknown person.
    Raises ``Refused`` with an HTTP status in ``args[1]``."""
    row = conn.execute('''SELECT d.id,d.run_id,d.position,r.cut_start,r.cut_end,c.id AS cut_id,c.active,c.start_time,c.end_time,
                                 c.episode_id,e.project_id FROM character_detections d
                          JOIN character_detection_runs r ON r.id=d.run_id JOIN cuts c ON c.id=r.cut_id
                          JOIN episodes e ON e.id=c.episode_id WHERE d.id=?''', (detection_id,)).fetchone()
    if not row:
        raise Refused('Detection not found', 404)
    if conn.execute('SELECT 1 FROM character_image_references WHERE detection_id=?', (detection_id,)).fetchone():
        raise Refused('Remove the saved reference before changing this label', 409)
    if not row['active'] or row['cut_start'] != row['start_time'] or row['cut_end'] != row['end_time']:
        raise Refused('Cut timing changed. Detect characters again before labelling this box', 409)
    title = request['title']
    existing = conn.execute("""SELECT j.id,json_extract(j.payload_json,'$.character_context_id') AS character_id
        FROM jobs j JOIN character_detections d ON d.id=? AND d.character_context_id=json_extract(j.payload_json,'$.character_context_id')
        WHERE j.episode_id=? AND j.pipeline='character_profile' AND j.status IN ('queued','running')
        AND json_extract(j.payload_json,'$.detection_id')=? AND json_extract(j.payload_json,'$.creation')=json(?) LIMIT 1""",
                            (detection_id, row['episode_id'], detection_id, json.dumps(request, ensure_ascii=False))).fetchone()
    if existing:
        character = dict(conn.execute('SELECT * FROM context_items WHERE id=?', (existing['character_id'],)).fetchone())
        return {'character': character, 'job_id': existing['id']}
    # Sequences link subjects to records by name, so two characters must not share one.
    if conn.execute("SELECT 1 FROM context_items WHERE project_id=? AND kind='character' AND lower(trim(title))=lower(?)", (row['project_id'], title)).fetchone():
        raise Refused(f'A character named “{title}” already exists in this project', 409)
    character_id = str(uuid.uuid4())
    conn.execute("INSERT INTO context_items (id,project_id,episode_id,kind,title,content,is_main) VALUES (?,?,?,'character',?,?,?)",
                 (character_id, row['project_id'], row['episode_id'] if request['episode_only'] else None, title, request['content'], int(request['is_main'])))
    conn.execute("UPDATE character_detections SET character_context_id=?,label_source='manual',match_reason=NULL,match_model_id=NULL,match_model_revision=NULL,gallery_status=NULL,gallery_note=NULL WHERE id=?", (character_id, detection_id))
    structured_results.opening_box_edited(conn, row['run_id'], row['position'])
    structured_results.invalidate_context(conn, row['project_id'], row['episode_id'] if request['episode_only'] else None)
    from . import visual_models
    payload = {'detection_id': detection_id, 'character_context_id': character_id, 'cut_id': row['cut_id'], 'creation': request,
               'models': visual_models.snapshot(conn, ('character_profile',))}
    job_id = worker.insert_job(conn, row['episode_id'], 'character_profile', payload)
    worker.sync_episode_status(conn, row['episode_id'])
    return {'character': dict(conn.execute('SELECT * FROM context_items WHERE id=?', (character_id,)).fetchone()), 'job_id': job_id}


# --- sheets for a reviewer ------------------------------------------------------------------------------------------------

def _frame_at(conn, episode_id, moment, width=640):
    from . import missed_cuts
    proxy = missed_cuts.proxy_path(conn, episode_id)
    if not proxy.is_file():
        raise RuntimeError('The preview video is missing')
    frames = missed_cuts.frames_near(proxy, moment, span=0.05, width=width)
    found = missed_cuts.nearest(frames, moment)
    if not found:
        raise RuntimeError(f'No frame near {moment:.2f}s')
    return found[1]


def _example_box(conn, character_id):
    """The user's largest manual box of a character (any episode of the project): the face Claude compares with."""
    return conn.execute("""SELECT d.box_json,r.frames_json,c.episode_id FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id
        JOIN cuts c ON c.id=r.cut_id WHERE d.character_context_id=? AND d.label_source='manual' AND d.position='first'
        ORDER BY ((json_extract(d.box_json,'$[2]')-json_extract(d.box_json,'$[0]'))*(json_extract(d.box_json,'$[3]')-json_extract(d.box_json,'$[1]'))) DESC
        LIMIT 1""", (character_id,)).fetchone()


def _crop(image, box, size, out=(200, 200)):
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    scale_x, scale_y = w / max(1, size[0]), h / max(1, size[1])
    x0, y0, x1, y1 = int(box[0] * scale_x), int(box[1] * scale_y), int(math.ceil(box[2] * scale_x)), int(math.ceil(box[3] * scale_y))
    crop = image[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    if crop.size == 0:
        return np.zeros((out[1], out[0], 3), dtype=np.uint8)
    ratio = min(out[0] / crop.shape[1], out[1] / crop.shape[0])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * ratio)), max(1, int(crop.shape[0] * ratio))))
    tile = np.zeros((out[1], out[0], 3), dtype=np.uint8)
    tile[:resized.shape[0], :resized.shape[1]] = resized
    return tile


def _label(tile, text):
    import cv2
    cv2.putText(tile, text[:34], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
    cv2.putText(tile, text[:34], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return tile


def box_sheet(conn, escalation_id):
    """JPEG bytes for a ``box_label`` or ``new_person`` question: per box (B1…) the frame with the box drawn and its crop,
    then the user's own example of each shortlisted character, so a reviewer compares faces."""
    import cv2
    import numpy as np
    item = _open(conn, escalation_id, (ESC_LABELS, ESC_PERSON))
    names = titles(conn, item['episode_id'])
    boxes = (item['evidence'] or {}).get('boxes') or []
    if not boxes:
        raise ValueError('This question lists no boxes')
    rows, examples = [], {}
    for number, brief in enumerate(boxes[:SHEET_ROWS], 1):
        run = conn.execute("""SELECT r.frames_json FROM character_detections d JOIN character_detection_runs r ON r.id=d.run_id WHERE d.id=?""",
                           (brief['id'],)).fetchone()
        frame = next((f for f in _load(run['frames_json'], []) if f.get('position') == 'first'), None) if run else None
        if not frame:
            continue
        image = _frame_at(conn, item['episode_id'], frame['time'], 640)
        size = frame.get('image_size') or [1280, 720]
        scale = image.shape[1] / size[0]
        full = image.copy()
        b = brief['box']
        cv2.rectangle(full, (int(b[0] * scale), int(b[1] * scale)), (int(b[2] * scale), int(b[3] * scale)), (0, 255, 255), 2)
        full = cv2.resize(full, (356, int(full.shape[0] * 356 / full.shape[1])))
        tiles = [_label(full, f"B{number} cut {brief['cut']}"), _label(_crop(image, b, size), f'B{number}')]
        for cid, name, score in (brief.get('candidates') or [])[:2]:
            if cid not in examples:
                example = _example_box(conn, cid)
                tile = None
                if example:
                    shot = next((f for f in _load(example['frames_json'], []) if f.get('position') == 'first'), None)
                    if shot:
                        try:
                            tile = _crop(_frame_at(conn, example['episode_id'], shot['time'], 640), _load(example['box_json']), shot.get('image_size') or [1280, 720])
                        except RuntimeError:
                            tile = None
                examples[cid] = tile
            tile = examples[cid].copy() if examples[cid] is not None else np.zeros((200, 200, 3), dtype=np.uint8)
            tiles.append(_label(tile, f"{name or names.get(cid, '?')} {score:.2f}"))
        height = max(tile.shape[0] for tile in tiles)
        rows.append(np.hstack([np.pad(tile, ((0, height - tile.shape[0]), (0, 4), (0, 0))) for tile in tiles]))
    if not rows:
        raise ValueError('None of the boxes has a frame')
    width = max(row.shape[1] for row in rows)
    sheet = np.vstack([np.pad(row, ((0, 4), (0, width - row.shape[1]), (0, 0))) for row in rows])
    ok, encoded = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('could not encode the sheet')
    return encoded.tobytes()


SLOT_COLOURS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (220, 0, 220), (0, 0, 255), (255, 255, 0), (128, 128, 255), (255, 255, 255)]


def target_sheet(conn, escalation_id):
    """JPEG bytes for a ``target_frame`` question: the opening frame (F1) and each flagged frame with every colour's box
    (numbered as in the Target frames tab) and the unassigned crops dashed in grey."""
    import cv2
    import numpy as np
    item = _open(conn, escalation_id, (ESC_TARGET,))
    evidence = item['evidence'] or {}
    run = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (evidence.get('run_id'),)).fetchone()
    if run is None:
        raise ValueError('The target frames were detected again')
    frames = _load(run['frames_json'], [])
    wanted = [frame['ordinal'] for frame in evidence.get('frames') or []]
    names = titles(conn, item['episode_id'])
    slots = {row['slot']: row['character_context_id'] for row in conn.execute('SELECT slot,character_context_id FROM target_slots WHERE run_id=?', (run['id'],))}
    boxes = defaultdict(list)
    for row in conn.execute('SELECT frame_ordinal,slot,box_json FROM target_detections WHERE run_id=?', (run['id'],)):
        boxes[row['frame_ordinal']].append((row['slot'], _load(row['box_json'])))
    candidates = defaultdict(list)
    for candidate in _load(run['candidates_json'], []) or []:
        candidates[candidate.get('frame_ordinal')].append(candidate.get('box'))
    tiles = []
    shown = [frames[0]['ordinal']] + [ordinal for ordinal in wanted if ordinal != frames[0]['ordinal']] if frames else wanted
    for ordinal in shown[:SHEET_ROWS]:
        index, frame = next(((i, f) for i, f in enumerate(frames, 1) if f.get('ordinal') == ordinal), (None, None))
        if frame is None:
            continue
        image = _frame_at(conn, item['episode_id'], frame['time'], 480)
        size = frame.get('image_size') or [1280, 720]
        scale = image.shape[1] / size[0]
        for slot, b in boxes.get(ordinal, []):
            colour = SLOT_COLOURS[slot % len(SLOT_COLOURS)]
            cv2.rectangle(image, (int(b[0] * scale), int(b[1] * scale)), (int(b[2] * scale), int(b[3] * scale)), colour, 2)
            cv2.putText(image, f"{slot + 1} {names.get(slots.get(slot), '?')}"[:22], (int(b[0] * scale) + 2, int(b[1] * scale) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        for b in candidates.get(ordinal, []):
            if b:
                cv2.rectangle(image, (int(b[0] * scale), int(b[1] * scale)), (int(b[2] * scale), int(b[3] * scale)), (160, 160, 160), 1)
        flags = next((f['flags'] for f in evidence.get('frames') or [] if f['ordinal'] == ordinal), [])
        tiles.append(_label(image, f"F{index}{' ' + ','.join(flags) if flags else ' (opening)'}"))
    if not tiles:
        raise ValueError('No frame to show')
    columns = 3
    height, width = tiles[0].shape[:2]
    tiles = [cv2.resize(tile, (width, height)) for tile in tiles]
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack([np.pad(tile, ((0, 4), (0, 4), (0, 0))) for tile in tiles[i:i + columns]]) for i in range(0, len(tiles), columns)]
    ok, encoded = cv2.imencode('.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('could not encode the sheet')
    return encoded.tobytes()


# --- undo -------------------------------------------------------------------------------------------------------------------

def undo_boxes(conn, decision):
    before, after = _load(decision['before_json'], {}) or {}, _load(decision['after_json'], {}) or {}
    if decision['target_table'] == 'context_items':
        raise agent_log.UndoConflict('A character the person created is not removed by undo; delete it in the Context library')
    if after.get('removed'):
        row = before.get('row') or {}
        if conn.execute('SELECT 1 FROM character_detections WHERE id=?', (decision['target_id'],)).fetchone():
            raise agent_log.UndoConflict('The box is back already')
        if not conn.execute('SELECT 1 FROM character_detection_runs WHERE id=?', (row.get('run_id'),)).fetchone():
            raise agent_log.UndoConflict('The detection run was replaced; detect again instead')
        columns = [key for key in row if key]
        conn.execute(f"INSERT INTO character_detections ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", [row[key] for key in columns])
        structured_results.opening_box_edited(conn, row['run_id'], row.get('position') or 'first')
        return
    current = conn.execute('SELECT run_id,character_context_id,label_source FROM character_detections WHERE id=?', (decision['target_id'],)).fetchone()
    if current is None:
        raise agent_log.UndoConflict('The box no longer exists')
    if current['character_context_id'] != after.get('character_context_id') or current['label_source'] != after.get('label_source'):
        raise agent_log.UndoConflict('Changed since the agent decided; undo would overwrite that change')
    conn.execute('UPDATE character_detections SET character_context_id=?,label_source=?,match_reason=? WHERE id=?',
                 (before.get('character_context_id'), before.get('label_source'), before.get('match_reason'), decision['target_id']))
    if before.get('character_context_id') != after.get('character_context_id'):
        structured_results.opening_box_edited(conn, current['run_id'], 'first')


def undo_target(conn, decision):
    from . import target_reviews
    after = _load(decision['after_json'], {}) or {}
    run = conn.execute('SELECT * FROM target_detection_runs WHERE id=?', (decision['target_id'],)).fetchone()
    if run is None:
        raise agent_log.UndoConflict('The target frames were detected again')
    frame = next((item for item in _load(run['frames_json'], []) if item.get('ordinal') == after.get('ordinal')), None)
    if not frame or not (frame.get('review') or {}).get('reviewed'):
        raise agent_log.UndoConflict('The frame is no longer marked reviewed')
    target_reviews.update_frames(conn, run, {after['ordinal']}, reviewed=False)


agent_log.UNDO_KINDS[KIND_BOXES] = undo_boxes
agent_log.UNDO_KINDS[KIND_TARGETS] = undo_target
