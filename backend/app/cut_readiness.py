"""Whether a cut's inputs are settled enough for the next expensive step (docs/autopilot/HARNESS.md, D26).

Read-only and free of FastAPI, so the inference worker can ask it too (``keyframe_analysis`` queues beats only for a ready
cut when the job says ``beats: when_ready``). The autopilot runs a step only for cuts whose inputs pass:

- visual analysis (first frame, keyframes) needs settled first-frame boxes and settled target frames;
- performance beats also need a current visual analysis and settled speech;
- the consistency review also needs beats built from the current keyframes.

A box label is settled when a person or an agent set it, or when the gallery match is strong (the label is the top
candidate, its score at least ``STRONG_SCORE`` and ahead of the next character by ``STRONG_LEAD``). Speech is settled when
every line overlapping the cut has a character and none of them is low confidence unless a person or an agent set its
speaker, and the episode's transcript and speaker attribution are not stale. A line the person left as it is (``left_unknown``,
the voice question's "Leave unknown" or the line editor's "Unknown – don't ask again") is settled with or without a character:
an unknown speaker is asked once (HARNESS.md D29). ``speaker_open``/``open_speaker_sql`` are that one rule, shared by every
surface that flags a speaker (review report, cut needs, the voice resolver). Each failed check gives a short reason that
the hand-off shows the person. Nothing here is an approval status, and nothing is saved.
"""
import json

STRONG_SCORE = 0.95
STRONG_LEAD = 0.06
LEFT_UNKNOWN = 'left_unknown'
DECIDED_SPEAKERS = ('manual', 'agent', LEFT_UNKNOWN)
OVERLAP = 0.1  # seconds a line must share with the cut to count as its speech
# speaker_attribution's own marker for lines it could not attribute: those lines have no character, which is checked per line
UNATTRIBUTED = 'Some lines could not be attributed'
MAX_REASONS = 3


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def clock(seconds):
    minutes, rest = divmod(max(0.0, float(seconds)), 60)
    return f'{int(minutes)}:{rest:04.1f}'


def target_review(conn, cut, positions=False):
    """Flagged frames and unnamed people on the cut's current Target frames run (the TargetFrames tab's Needs review).
    ``positions``: also give each flagged frame its place among the run's frames (``position``, the tab's F1…Fn)."""
    run = conn.execute('''SELECT id,frames_json FROM target_detection_runs WHERE cut_id=? AND cut_start=? AND cut_end=?
                          ORDER BY rowid DESC LIMIT 1''', (cut['id'], cut['start_time'], cut['end_time'])).fetchone()
    if not run:
        return None
    frames = []
    for position, frame in enumerate(_loads(run['frames_json'], []), 1):
        review = frame.get('review') if isinstance(frame, dict) else None
        # frontend/src/targetView.ts frameNeedsReview
        if isinstance(review, dict) and not review.get('reviewed') and review.get('flags'):
            frames.append({'ordinal': frame.get('ordinal'), 'flags': sorted(review['flags']), **({'position': position} if positions else {})})
    unnamed = conn.execute('''SELECT COUNT(DISTINCT s.slot) FROM target_slots s JOIN target_detections t ON t.run_id=s.run_id AND t.slot=s.slot
                              WHERE s.run_id=? AND s.ignored=0 AND s.character_context_id IS NULL''', (run['id'],)).fetchone()[0]
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


def episode_speech(episode):
    if not episode or episode.get('has_audio') == 0:
        return None
    reasons = []
    if episode.get('transcript_stale_reason'):
        reasons.append(f"transcript stale ({episode['transcript_stale_reason']})")
    stale = episode.get('speaker_attribution_stale_reason')
    if stale and stale != UNATTRIBUTED:
        reasons.append(f'speaker attribution stale ({stale})')
    return reasons


def _speech(conn, episode, cut, episode_reasons):
    if episode_reasons is None:
        return []
    reasons = list(episode_reasons)
    for row in conn.execute("""SELECT start_time,character_context_id,confidence,speaker_source FROM transcript_segments
        WHERE episode_id=? AND superseded_by IS NULL AND deleted=0 AND MIN(end_time, ?) - MAX(start_time, ?) > ?
        ORDER BY start_time""", (episode['id'], cut['end_time'], cut['start_time'], OVERLAP)):
        if not speaker_open(row):
            continue
        if not row['character_context_id']:
            reasons.append(f"line at {clock(row['start_time'])} has no speaker")
        else:
            reasons.append(f"line at {clock(row['start_time'])} has a low-confidence speaker")
    return reasons


def for_cut(conn, episode, cut, episode_reasons=False):
    """Readiness of one active cut (a row with id, start_time, end_time, no_characters)."""
    if episode_reasons is False:
        episode_reasons = episode_speech(episode)
    if cut.get('no_characters'):
        boxes, targets = [], []
    else:
        boxes, has_boxes = _boxes(conn, cut)
        targets = _targets(conn, cut, has_boxes) if not boxes else []
    visual, keyframes, video = _visual(conn, cut)
    speech = _speech(conn, episode, cut, episode_reasons)
    beats = _beats(conn, cut, keyframes, video)
    held_visual = boxes + targets
    held_beats = held_visual + visual + speech
    held_consistency = held_beats + beats
    return {'boxes': not boxes, 'target_frames': not targets, 'visual': not visual, 'speech': not speech, 'beats_current': not beats,
            'may_visual': not held_visual, 'may_beats': not held_beats, 'may_consistency': not held_consistency,
            'held': {'target_frames': boxes[:MAX_REASONS], 'visual': held_visual[:MAX_REASONS], 'beats': held_beats[:MAX_REASONS], 'consistency': held_consistency[:MAX_REASONS]}}


def episode(conn, episode_id, cuts):
    """{cut id: readiness} for the given cut rows of one episode."""
    row = conn.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
    episode_row = dict(row) if row else None
    episode_reasons = episode_speech(episode_row)
    return {cut['id']: for_cut(conn, episode_row, cut, episode_reasons) for cut in cuts}


def cut(conn, cut_id):
    """Readiness of one cut by id, or None when it is not an active cut."""
    row = conn.execute('SELECT id,episode_id,start_time,end_time,no_characters FROM cuts WHERE id=? AND active=1', (cut_id,)).fetchone()
    if not row:
        return None
    episode_row = conn.execute('SELECT * FROM episodes WHERE id=?', (row['episode_id'],)).fetchone()
    return for_cut(conn, dict(episode_row) if episode_row else None, dict(row))
