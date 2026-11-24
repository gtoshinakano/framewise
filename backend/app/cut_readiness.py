"""Whether a cut's inputs are settled enough for the next expensive step (docs/autopilot/HARNESS.md, D26 and D37).

Read-only and free of FastAPI, so the inference worker can ask it too (``keyframe_analysis`` queues beats only for a ready
cut when the job says ``beats: when_ready``). Evaluate (``batch_runs``) runs a step only for cuts whose inputs pass:

- visual analysis (first frame, keyframes) needs settled first-frame boxes and settled target frames;
- performance beats also need a current visual analysis;
- the consistency review also needs beats built from the current keyframes.

A box label is settled when a person or an agent set it, or when the gallery match is strong (the label is the top
candidate, its score at least ``STRONG_SCORE`` and ahead of the next character by ``STRONG_LEAD``). Speech holds a cut only
through a visible question: an open voice question in the inbox (raised once, by Prepare's voice resolver: HARNESS.md D37).
Line state alone never holds a cut, so a re-transcription or a later speaker attribution cannot hold or flag anything
again. An open inbox item (for the person or for Claude) holds its cut from the step it is about onward (``INBOX_HOLDS``,
HARNESS.md D32): a box, person or target-frame question visual analysis and after, a voice beats and after. Each failed check
gives a short reason that the Run Multiple dialog shows the person.

``speaker_open``/``open_speaker_sql`` say whether a line's speaker is undecided (no character unless the person left it so,
or a low-confidence character nobody decided: HARNESS.md D29). Only the voice resolver uses them, to choose the lines it
works on. Nothing here is an approval status, and nothing is saved.
"""
import json

STRONG_SCORE = 0.95
STRONG_LEAD = 0.06
LEFT_UNKNOWN = 'left_unknown'
DECIDED_SPEAKERS = ('manual', 'agent', LEFT_UNKNOWN)
MAX_REASONS = 3
STEPS = ('target_frames', 'visual', 'beats', 'consistency')
# Escalation kind -> (the first step it holds, how the hand-off names it)
INBOX_HOLDS = {'box_label': ('visual', 'box names'), 'new_person': ('visual', 'an unknown person'),
               'target_frame': ('visual', 'target frames'), 'voice': ('beats', 'an unknown voice')}


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def target_review(conn, cut, positions=False):
    """Flagged frames and unnamed people on the cut's current Target frames run (the TargetFrames tab's Needs review).
    ``positions``: also give each flagged frame its place among the run's frames (``position``, the tab's F1…Fn)."""
    run = conn.execute('''SELECT id,frames_json,candidates_json FROM target_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=?
                          ORDER BY rowid DESC LIMIT 1''', (cut['id'], cut['start_time'], cut['end_time'])).fetchone()
    if not run:
        return None
    from . import character_decisions  # the flags the tab shows: boxes the person removed no longer count
    shown, _ = character_decisions.visible_target_state(conn, run)
    frames = []
    for position, frame in enumerate(shown, 1):
        review = frame.get('review') if isinstance(frame, dict) else None
        # frontend/src/targetView.ts frameNeedsReview
        if isinstance(review, dict) and not review.get('reviewed') and review.get('flags'):
            frames.append({'ordinal': frame.get('ordinal'), 'flags': sorted(review['flags']), **({'position': position} if positions else {})})
    unnamed = conn.execute('''SELECT COUNT(DISTINCT s.slot) FROM target_slots s JOIN target_detections t ON t.run_id=s.run_id AND t.slot=s.slot
                              WHERE s.run_id=? AND s.ignored=0 AND s.character_context_id IS NULL AND COALESCE(s.label_source,'')<>'manual'
                              ''', (run['id'],)).fetchone()[0]
    return {'frames': frames, 'unnamed_people': unnamed} if frames or unnamed else None


def strong_match(character_id, candidates):
    """(settled, top score, lead) of an automatic label from its stored shortlist."""
    ranked = sorted(((float(item.get('score') or 0), item.get('character_id')) for item in candidates
                     if isinstance(item, dict) and item.get('character_id')), reverse=True)
    if not ranked:
        return False, None, None
    top, leader = ranked[0]
    lead = top - (ranked[1][0] if len(ranked) > 1 else 0.0)
    return leader == character_id and top >= STRONG_SCORE and lead >= STRONG_LEAD, round(top, 2), round(lead, 2)


def _boxes(conn, cut):
    run = conn.execute("""SELECT id,phase FROM character_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=?
                          ORDER BY rowid DESC LIMIT 1""", (cut['id'], cut['start_time'], cut['end_time'])).fetchone()
    if not run or run['phase'] != 'complete':
        return ['first-frame boxes not detected for the current cut'], False
    reasons, number = [], 0
    rows = conn.execute("""SELECT d.character_context_id,d.label_source,d.candidates_json,i.title FROM character_detections d
        LEFT JOIN context_items i ON i.id=d.character_context_id WHERE d.run_id=? AND d.position='first' ORDER BY d.rowid""", (run['id'],)).fetchall()
    for number, row in enumerate(rows, 1):
        if not row['character_context_id']:
            if row['label_source'] != 'manual':  # a box the person left unnamed is their decision (HARNESS.md D32)
                reasons.append(f'box {number} has no character')
        elif row['label_source'] == 'auto':
            settled, top, lead = strong_match(row['character_context_id'], _loads(row['candidates_json'], []))
            if not settled:
                detail = f' ({row["title"]} {top:.2f}, lead {lead:.2f})' if top is not None else f' ({row["title"]}, no shortlist)'
                reasons.append(f'box {number} label not confirmed{detail}')
    return reasons, bool(rows)


def _targets(conn, cut, has_boxes):
    if not has_boxes:
        return []
    if not conn.execute('SELECT 1 FROM target_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=? LIMIT 1',
                        (cut['id'], cut['start_time'], cut['end_time'])).fetchone():
        return ['target frames not detected']
    review = target_review(conn, cut, positions=True)
    if not review:
        return []
    reasons = [f"target frame F{frame['position']} flagged ({', '.join(frame['flags'])})" for frame in review['frames']]
    if review['unnamed_people']:
        reasons.append(f"{review['unnamed_people']} unnamed person(s) on the target frames")
    return reasons


def _current(conn, cut_id, structured_type):
    return conn.execute("""SELECT id,stale_reason,content FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL
        AND json_extract(settings_json,'$.structured_type')=? ORDER BY rowid DESC LIMIT 1""", (cut_id, structured_type)).fetchone()


def _visual(conn, cut):
    """(reasons, keyframes row or None, video row or None)."""
    reasons = []
    first = _current(conn, cut['id'], 'initial_frame')
    keyframes = _current(conn, cut['id'], 'keyframe_analysis')
    video = None if keyframes else _current(conn, cut['id'], 'video_analysis')
    if not first:
        reasons.append('first-frame analysis missing')
    elif first['stale_reason']:
        reasons.append(f"first-frame analysis stale ({first['stale_reason']})")
    motion = keyframes or video
    if not motion:
        reasons.append('keyframe analysis missing')
    elif motion['stale_reason']:
        reasons.append(f"keyframe analysis stale ({motion['stale_reason']})")
    return reasons, keyframes, video


def _beats(conn, cut, keyframes, video):
    if not keyframes:
        return [] if video else ['keyframe analysis missing']
    beats = _current(conn, cut['id'], 'performance_beats')
    if not beats:
        return ['performance beats missing']
    if beats['stale_reason']:
        return [f"performance beats stale ({beats['stale_reason']})"]
    if (_loads(beats['content'], {}).get('based_on') or {}).get('keyframe_analysis') != keyframes['id']:
        return ['performance beats older than the keyframe analysis']
    return []


def speaker_open(row):
    """Whether a transcript line still needs a speaker decision: no character (unless the person left it so), or a
    low-confidence character nobody decided."""
    def value(key):
        try:
            return row[key]
        except (KeyError, IndexError):
            return None
    source = value('speaker_source') or ''
    if not value('character_context_id'):
        return source != LEFT_UNKNOWN
    return value('confidence') == 'low' and source not in DECIDED_SPEAKERS


def open_speaker_sql(alias=''):
    """``speaker_open`` as an SQL condition over transcript_segments (``alias.`` prefixed when given)."""
    a = f'{alias}.' if alias else ''
    decided = ','.join(f"'{value}'" for value in DECIDED_SPEAKERS)
    return (f"(({a}character_context_id IS NULL AND COALESCE({a}speaker_source,'')<>'{LEFT_UNKNOWN}') OR "
            f"({a}character_context_id IS NOT NULL AND {a}confidence='low' AND COALESCE({a}speaker_source,'') NOT IN ({decided})))")


def escalation_cuts(conn, item, cuts):
    """The cuts an escalation is about: its own ``cut_ids``, and for a voice question every cut its lines touch (it lists only
    a few sample cuts, but one answer settles all of them, so none of them should flag the same voice again)."""
    found = list(item['cut_ids'])
    line_ids = (item.get('evidence') or {}).get('line_ids') if item['kind'] == 'voice' else None
    if line_ids:
        marks = ','.join('?' for _ in line_ids)
        for line in conn.execute(f'SELECT start_time,end_time FROM transcript_segments WHERE id IN ({marks})', list(line_ids)):
            for cut in cuts:
                if cut['id'] not in found and line['end_time'] > cut['start_time'] and line['start_time'] < cut['end_time']:
                    found.append(cut['id'])
    return found


def inbox(conn, episode_id, cuts):
    """{cut id: {escalation kind}} of the open inbox items that hold these cuts (``INBOX_HOLDS``)."""
    from . import agent_log
    found = {}
    for item in agent_log.escalations(conn, episode_id, 'open', limit=5000):
        if item['kind'] in INBOX_HOLDS:
            for cut_id in escalation_cuts(conn, item, cuts):
                found.setdefault(cut_id, set()).add(item['kind'])
    return found


def _inbox_reasons(kinds, step):
    return [f'inbox: {label}' for kind, (first, label) in sorted(INBOX_HOLDS.items())
            if kind in (kinds or ()) and STEPS.index(first) <= STEPS.index(step)]


def for_cut(conn, episode, cut, inbox_kinds=None):
    """Readiness of one active cut (a row with id, start_time, end_time, no_characters). ``inbox_kinds``: the open inbox items
    on the cut (``inbox``); None reads them."""
    if inbox_kinds is None:
        episode_id = cut.get('episode_id') or (episode or {}).get('id')
        inbox_kinds = inbox(conn, episode_id, [cut]).get(cut['id'], set()) if episode_id else set()
    if cut.get('no_characters'):
        boxes, targets = [], []
    else:
        boxes, has_boxes = _boxes(conn, cut)
        targets = _targets(conn, cut, has_boxes) if not boxes else []
    visual, keyframes, video = _visual(conn, cut)
    beats = _beats(conn, cut, keyframes, video)
    held_target = _inbox_reasons(inbox_kinds, 'target_frames') + boxes
    held_visual = _inbox_reasons(inbox_kinds, 'visual') + boxes + targets
    held_beats = _inbox_reasons(inbox_kinds, 'beats') + boxes + targets + visual
    held_consistency = _inbox_reasons(inbox_kinds, 'consistency') + boxes + targets + visual + beats
    return {'boxes': not boxes, 'target_frames': not targets, 'visual': not visual, 'beats_current': not beats,
            'inbox': sorted(inbox_kinds or ()), 'may_target_frames': not held_target,
            'may_visual': not held_visual, 'may_beats': not held_beats, 'may_consistency': not held_consistency,
            'held': {'target_frames': held_target[:MAX_REASONS], 'visual': held_visual[:MAX_REASONS], 'beats': held_beats[:MAX_REASONS],
                     'consistency': held_consistency[:MAX_REASONS]}}


def episode(conn, episode_id, cuts):
    """{cut id: readiness} for the given cut rows of one episode."""
    row = conn.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
    episode_row = dict(row) if row else None
    held = inbox(conn, episode_id, cuts)
    return {cut['id']: for_cut(conn, episode_row, cut, held.get(cut['id'], set())) for cut in cuts}


def cut(conn, cut_id):
    """Readiness of one cut by id, or None when it is not an active cut."""
    row = conn.execute('SELECT id,episode_id,start_time,end_time,no_characters FROM cuts WHERE id=? AND active=1', (cut_id,)).fetchone()
    if not row:
        return None
    episode_row = conn.execute('SELECT * FROM episodes WHERE id=?', (row['episode_id'],)).fetchone()
    return for_cut(conn, dict(episode_row) if episode_row else None, dict(row))
