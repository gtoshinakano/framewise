"""Transcript lines that run across a hard cut, divided at a pause (docs/autopilot/HARNESS.md, milestone M3).

A line *crosses* a hard cut when it lasts more than ``CROSS`` on both sides of it. ``plan_line`` decides what to do with
one line and one cut, from the line's aligned words (``transcript_parts.parts``: the line's own text spread over the
aligner's word slots) and the level of ``audio.wav`` around it (``sequence_timeline.audio_envelope``/``find_pauses``):

* **split**: words lie on both sides of the cut and a quiet stretch of at least ``MIN_PAUSE`` lies between two words
  within ``REACH`` of the cut, at a place a phrase can end (``natural``: never inside a Japanese word or before an
  auxiliary or particle, which the aligner gives as tokens of their own). The split goes on the cut itself when the cut is inside that quiet stretch, else on the
  quiet stretch's edge nearest the cut, ``PAD`` into the silence. The text divides between the words (never inside one)
  and both parts together must spell the line exactly, so an ``srt_edit`` line keeps every character.
* **trim**: only silence crosses, the line's edge moves to where the voice stops (or starts), never past the cut. The audio
  decides first: when the level stays quiet from the cut to the line's edge, the aligner's word times there are padding
  (the user trims these by ear; words left outside the new edges lose their timings, as a manual retime does). Otherwise
  every word must lie on one side (``speech_edge`` from the last word); a word that itself runs across the cut stays.
* **leave**: no pause near the cut (a phrase bleeding into the next cut is acceptable: the user moves it while building a
  sequence), no word timings, or text that does not follow them.

The job (pipeline ``line_splits``, CPU, media lane) plans every crossing line of the episode, applies the plans in its
publish transaction (a piece that still crosses another cut is planned again), and logs each change as an agent decision
of kind ``line_splits`` whose undo restores the original line while the pieces are unchanged. Pieces keep the source
line's ``origin`` (the words are still the model's or the user's: re-transcription and speaker attribution treat them as
before); provenance is the decision. Lines inside a sequence's range are left alone (editing them would make the user's
confirmed dialogue stale) and reported.
"""
import difflib
import json
import uuid

from . import agent_log, db, job_inputs, line_edit, media, sequence_timeline, structured_results, subtitle_store, transcript_parts, worker

KIND = 'line_splits'
CROSS = 0.15          # seconds a line must last on both sides of a cut to cross it (HARNESS.md section 3)
REACH = 1.5           # a pause farther than this from the cut is not used (tuned by autopilot/replay/line_splits.py)
MIN_PAUSE = sequence_timeline.MIN_PAUSE   # 0.3 s of quiet
PAD = sequence_timeline.EDGE_PAD          # a split on a pause's edge sits this far into the silence
WORD_OVER = 0.05      # a word ending this far past the cut runs across it: never trimmed away
CONTEXT = 1.0         # audio read on each side of a line, so the noise floor is the scene's, not the line's
WINDOW = sequence_timeline.WINDOW
SAME_TEXT = 0.6       # a line whose text resembles its aligned words less than this was rewritten: its word times do not fit
MAX_PASSES = 4        # a long line crossing several cuts is planned again after each change
# Aligner tokens are morphemes: a pause can fall inside a word (わかっ…た) or before an auxiliary (人な…んだろう). A right
# piece never starts with one of these tokens, and a left piece never ends on a small っ.
NO_START = frozenset(sequence_timeline.PARTICLES) | {
    'た', 'だ', 'ん', 'です', 'ます', 'ない', 'ず', 'う', 'よう', 'る', 'れる', 'られる', 'せる', 'させる', 'ちゃう', 'じゃ', 'でしょう',
    'だろう', 'でし', 'まし', 'ませ', 'なかっ', 'たい', 'そう', 'って', 'たら', 'なら', 'ちゃ', 'てる', 'ている', 'いる', 'さ', 'ぞ', 'わ'}


def crossings(row, cut_starts):
    """The hard cuts (start times of active cuts after the first) a line crosses, in time order."""
    return [moment for moment in cut_starts if row['start_time'] < moment - CROSS and row['end_time'] > moment + CROSS]


def _squashed(text):
    return ''.join((text or '').split())


def natural(left_parts, right_parts):
    """May a line be divided between these two words? Not inside a Japanese word or before an auxiliary or particle."""
    left, right = left_parts[-1]['text'].strip(), right_parts[0]['text'].strip()
    if left.endswith(('、', '。', '！', '？', '!', '?', '…', ',', '.')):
        return True
    return not left.endswith('っ') and right.rstrip('、。！？!?…,.') not in NO_START


def _join(parts):
    text = ''
    for part in parts:
        text = transcript_parts.join_text(text, part['text'])
    return text


def plan_line(row, cut, envelope, env_start):
    """What to do with ``row`` (a transcript line) crossing the hard cut at ``cut``.

    ``envelope`` is the audio level from ``env_start`` (``sequence_timeline.audio_envelope``). Returns
    ``{action: split|trim|leave, why, ...}``: a split has ``at``, ``left``, ``right``; a trim has ``start``/``end``."""
    row = dict(row)
    if not row.get('words_json'):
        return {'action': 'leave', 'why': 'the line has no word timings'}
    aligned = ''.join(word[2] for word in transcript_parts.aligned_words(row) or [])
    if difflib.SequenceMatcher(None, _squashed(aligned), _squashed(row['text']), autojunk=False).ratio() < SAME_TEXT:
        return {'action': 'leave', 'why': 'the text was rewritten since its words were aligned, so the word times no longer fit it'}
    parts = transcript_parts.parts(row)
    if len(parts) == 1 and parts[0].get('words_json') is not None:
        return {'action': 'leave', 'why': 'the line has no word timings'}
    middle = [(part['start_time'] + part['end_time']) / 2 for part in parts]
    before = [part for part, mid in zip(parts, middle) if mid < cut]
    after = [part for part, mid in zip(parts, middle) if mid >= cut]
    threshold = sequence_timeline.quiet_threshold(envelope)

    def levels(t0, t1):
        first, last = max(0, int((t0 - env_start) / WINDOW + 1e-6)), min(len(envelope), int((t1 - env_start) / WINDOW + 1e-6))
        return envelope[first:last]

    def voice_before(until):
        """Where the voice last sounds before ``until`` (the end of the last loud window), or None."""
        last = min(len(envelope), int((until - env_start) / WINDOW + 1e-6))
        for index in range(last - 1, -1, -1):
            if envelope[index] > threshold:
                return env_start + (index + 1) * WINDOW
        return None

    def voice_after(since):
        first = max(0, int((since - env_start) / WINDOW + 1e-6))
        for index in range(first, len(envelope)):
            if envelope[index] > threshold:
                return env_start + index * WINDOW
        return None

    # The audio decides first: a tail (or head) that is silent all the way is aligner padding, whatever the word times say.
    tail, head = levels(cut + WINDOW, row['end_time']), levels(row['start_time'], cut - WINDOW)
    if tail and max(tail) <= threshold and (spoken := voice_before(cut)) is not None and spoken > row['start_time']:
        end = round(min(cut, spoken + PAD), 3)
        if end < row['end_time'] - 0.01:
            return {'action': 'trim', 'start': row['start_time'], 'end': end, 'by': 'audio',
                    'why': f"the audio is silent from the cut to the line's end ({row['end_time'] - cut:.2f}s)"}
    if head and max(head) <= threshold and (spoken := voice_after(cut)) is not None and spoken < row['end_time']:
        start = round(max(cut, spoken - PAD), 3)
        if start > row['start_time'] + 0.01:
            return {'action': 'trim', 'start': start, 'end': row['end_time'], 'by': 'audio',
                    'why': f"the audio is silent from the line's start to the cut ({cut - row['start_time']:.2f}s)"}

    def edge(t, side, low, high):
        return round(env_start + sequence_timeline.speech_edge(envelope, t - env_start, side, low - env_start, high - env_start), 3)

    if not after:
        last = max(part['end_time'] for part in before)
        if last > cut + WORD_OVER:
            return {'action': 'leave', 'why': f'its last word runs across the cut (to {last:.2f}s)'}
        end = min(cut, edge(last, 'after', last, cut))
        if end >= row['end_time'] - 0.01 or end <= row['start_time']:
            return {'action': 'leave', 'why': 'nothing to trim'}
        return {'action': 'trim', 'start': row['start_time'], 'end': end,
                'why': f"every word ends before the cut; only silence ran {row['end_time'] - cut:.2f}s past it"}
    if not before:
        first = min(part['start_time'] for part in after)
        if first < cut - WORD_OVER:
            return {'action': 'leave', 'why': f'its first word runs across the cut (from {first:.2f}s)'}
        start = max(cut, edge(first, 'before', cut, first))
        if start <= row['start_time'] + 0.01 or start >= row['end_time']:
            return {'action': 'leave', 'why': 'nothing to trim'}
        return {'action': 'trim', 'start': start, 'end': row['end_time'],
                'why': f"every word starts after the cut; only silence ran {cut - row['start_time']:.2f}s before it"}
    pauses = [(env_start + a, env_start + b) for a, b in sequence_timeline.find_pauses(envelope, WINDOW, MIN_PAUSE)]
    best = None
    for index in range(len(parts) - 1):
        if not natural(parts[:index + 1], parts[index + 1:]):
            continue
        low, high = middle[index], middle[index + 1]
        for p0, p1 in pauses:
            q0, q1 = max(p0, low), min(p1, high)
            if q1 - q0 < MIN_PAUSE - 1e-6:
                continue
            if q0 <= cut <= q1:
                at = cut
            elif cut < q0:
                at = min(q0 + PAD, (q0 + q1) / 2)
            else:
                at = max(q1 - PAD, (q0 + q1) / 2)
            distance = abs(at - cut)
            if distance <= REACH and (best is None or distance < best[0]):
                best = (distance, round(at, 3), index, (round(q0, 3), round(q1, 3)))
    if best is None:
        return {'action': 'leave', 'why': f'no pause of {MIN_PAUSE}s between two words within {REACH}s of the cut: the phrase bleeds into the next cut'
                                          ' (a pause inside a word or before a particle does not count)'}
    _, at, index, pause = best
    left_parts = [part for part, mid in zip(parts, middle) if mid < at]
    right_parts = [part for part, mid in zip(parts, middle) if mid >= at]
    left, right = _join(left_parts), _join(right_parts)
    if not left or not right or _squashed(left + right) != _squashed(row['text']):
        return {'action': 'leave', 'why': 'the text does not follow the word timings, so it cannot be divided safely'}
    if not row['start_time'] < at < row['end_time']:
        return {'action': 'leave', 'why': 'the pause is outside the line'}
    return {'action': 'split', 'at': at, 'left': left, 'right': right, 'pause': list(pause),
            'why': ('on the cut, inside a pause' if at == round(cut, 3) else f'at the pause {abs(at - cut):.2f}s from the cut')}


# --- the job -----------------------------------------------------------------------------------------------------

def audio_path(conn, episode_id):
    episode = conn.execute('SELECT project_id FROM episodes WHERE id=?', (episode_id,)).fetchone()
    return media.episode_dir(episode['project_id'], episode_id) / 'audio.wav'


def sequence_ranges(conn, episode_id):
    return [(row['start_time'], row['end_time']) for row in conn.execute('SELECT start_time,end_time FROM sequences WHERE episode_id=?', (episode_id,))]


def policy_for(conn, episode_id, run_id=None):
    from . import autopilot_profile
    return autopilot_profile.policy(conn, episode_id, KIND, run_id)


def queue(conn, episode_id, *, run_id=None):
    """Queue the line splitter for the whole episode inside the caller's transaction; None when the profile says off."""
    policy = policy_for(conn, episode_id, run_id)
    if policy['mode'] == 'off':
        return None
    payload = {'policy': policy, 'run_id': run_id}
    waits = worker.pending_jobs(conn, episode_id, ('prepare', 'hardcuts', 'missed_cuts', 'transcribe'))
    if waits:
        payload['depends_on'] = waits
    job_id = worker.insert_job(conn, episode_id, KIND, payload, priority_class='batch')
    worker.sync_episode_status(conn, episode_id)
    return job_id


def current_lines(conn, episode_id):
    return [dict(row) for row in conn.execute("""SELECT * FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL AND deleted=0
        ORDER BY start_time""", (episode_id,))]


def cut_starts(conn, episode_id):
    return [row[0] for row in conn.execute('SELECT start_time FROM cuts WHERE episode_id=? AND active=1 AND cut_index>0 ORDER BY cut_index', (episode_id,))]


def gather(conn, job):
    episode_id = job['episode_id']
    audio = audio_path(conn, episode_id)
    try:
        stat = audio.stat()
        audio_key = [stat.st_size, stat.st_mtime_ns]
    except OSError:
        audio_key = None
    return {'audio': audio_key, 'cuts': [round(moment, 6) for moment in cut_starts(conn, episode_id)],
            'lines': [[row['id'], row['start_time'], row['end_time']] for row in current_lines(conn, episode_id)],
            'sequences': sequence_ranges(conn, episode_id)}


def inputs(conn, job):
    return job_inputs.fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    """Each change marks the cuts its line touches stale itself (``_invalidate``)."""


def _invalidate(conn, episode_id, start, end):
    structured_results.invalidate_transcript(conn, episode_id, start, end)
    subtitle_store.invalidate_probes(conn, episode_id)


def plan_episode(conn, episode_id, lines=None, envelope_for=None):
    """``[(row, cut, plan)]`` for every crossing of every current line (the first crossing of each line only)."""
    starts = cut_starts(conn, episode_id)
    audio = audio_path(conn, episode_id)
    sequences = sequence_ranges(conn, episode_id)
    plans = []
    for row in lines if lines is not None else current_lines(conn, episode_id):
        crossed = crossings(row, starts)
        if not crossed:
            continue
        if any(row['start_time'] < end and row['end_time'] > start for start, end in sequences):
            plans.append((row, crossed[0], {'action': 'leave', 'why': 'the line is inside a sequence (its dialogue is the user\'s)', 'sequence': True}))
            continue
        env_start = max(0.0, row['start_time'] - CONTEXT)
        envelope = envelope_for(env_start, row['end_time'] + CONTEXT) if envelope_for else sequence_timeline.audio_envelope(audio, env_start, row['end_time'] + CONTEXT)
        if not envelope:
            plans.append((row, crossed[0], {'action': 'leave', 'why': 'no usable audio'}))
            continue
        # The crossing nearest a pause is the one to settle first; plan each and keep the first that is not "leave".
        chosen = None
        for cut in crossed:
            plan = plan_line(row, cut, envelope, env_start)
            if plan['action'] != 'leave':
                chosen = (row, cut, plan)
                break
            chosen = chosen or (row, cut, plan)
        plans.append(chosen)
    return plans


def _trim(conn, row, start, end):
    """A revision of ``row`` with new edges and the same text. Its word times stay only while every word is inside."""
    words = transcript_parts.aligned_words(row)
    words_json = row['words_json'] if words and all(start - 0.02 <= word[0] and word[1] <= end + 0.02 for word in words) else None
    row = {**dict(row), 'words_json': words_json}
    new_id = str(uuid.uuid4())
    conn.execute(f"""INSERT INTO transcript_segments ({line_edit.COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (new_id, row['episode_id'], start, end, row['text'], row['speaker_label'], row['character_context_id'], row['origin'],
                  row['model_id'], row['job_id'], row['id'], row['words_json'], row['speaker_source'], row['confidence'], row['evidence_json']))
    conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, row['id']))
    _invalidate(conn, row['episode_id'], row['start_time'], row['end_time'])
    return new_id


def _cut_for(conn, episode_id, moment):
    row = conn.execute('SELECT id,cut_index FROM cuts WHERE episode_id=? AND active=1 AND start_time<=? AND end_time>? LIMIT 1',
                       (episode_id, moment, moment)).fetchone()
    return row


def apply_plan(conn, row, cut, plan, *, run_id=None, tier='rules'):
    """Carry out one plan and log it. Returns the ids of the new current rows."""
    episode_id = row['episode_id']
    before = {key: row[key] for key in ('id', 'start_time', 'end_time', 'text')}
    owner = _cut_for(conn, episode_id, cut)
    if plan['action'] == 'split':
        left_id, right_id = line_edit.split(conn, row['id'], plan['at'], plan['left'], plan['right'], origin=row['origin'])
        after = {'rows': [{'id': left_id, 'start_time': row['start_time'], 'end_time': plan['at'], 'text': plan['left']},
                          {'id': right_id, 'start_time': plan['at'], 'end_time': row['end_time'], 'text': plan['right']}]}
        summary = f"Split the line at {plan['at']:.2f}s ({plan['why']}): \"{plan['left'][-20:]}\" | \"{plan['right'][:20]}\""
        made = [left_id, right_id]
    else:
        new_id = _trim(conn, row, plan['start'], plan['end'])
        after = {'rows': [{'id': new_id, 'start_time': plan['start'], 'end_time': plan['end'], 'text': row['text']}]}
        summary = f"Trimmed the line to {plan['start']:.2f}-{plan['end']:.2f}s ({plan['why']})"
        made = [new_id]
    agent_log.record(conn, episode_id=episode_id, run_id=run_id, cut_id=owner['id'] if owner else None, kind=KIND, tier=tier,
                     target_table='transcript_segments', target_id=row['id'], before=before, after=after,
                     evidence={'cut': cut, **{key: plan.get(key) for key in ('action', 'why', 'at', 'pause', 'start', 'end')}}, summary=summary)
    return made


def publish(conn, episode_id, payload):
    """Plan and apply every crossing (pieces that still cross another cut are planned again). Returns the outcome counts."""
    policy, run_id = payload.get('policy') or {'mode': 'apply'}, payload.get('run_id')
    counts = {'split': 0, 'trim': 0, 'leave': 0, 'asked': 0, 'in_sequence': 0}
    reasons = {}
    lines, fresh = current_lines(conn, episode_id), None
    for _ in range(MAX_PASSES):
        made = []
        for row, cut, plan in plan_episode(conn, episode_id, lines):
            if plan['action'] == 'leave':
                if fresh is None:  # a piece left as it is was already counted with its line
                    counts['in_sequence' if plan.get('sequence') else 'leave'] += 1
                    reason = plan['why'].split(':')[0]
                    reasons[reason] = reasons.get(reason, 0) + 1
                continue
            if policy['mode'] == 'ask':
                owner = _cut_for(conn, episode_id, cut)
                agent_log.escalate(conn, episode_id=episode_id, kind='line_split', group_key=f"line:{row['id']}",
                                   question=f"Line \"{row['text'][:60]}\" ({row['start_time']:.2f}-{row['end_time']:.2f}s) runs across the hard cut at "
                                            f"{cut:.2f}s. {plan['action'].capitalize()} it? ({plan['why']})",
                                   cut_ids=[owner['id']] if owner else [], options=['Apply', 'Leave it'], proposal=plan, run_id=run_id)
                counts['asked'] += 1
                continue
            made += apply_plan(conn, row, cut, plan, run_id=run_id)
            counts[plan['action']] += 1
        if not made:
            break
        fresh = made
        lines = [dict(row) for row in conn.execute(f"SELECT * FROM transcript_segments WHERE id IN ({','.join('?' for _ in made)})", made)]
    return {**counts, 'left_because': reasons}


def process(job_id, owner):
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    with db.connect() as conn:
        conn.execute('BEGIN')
        snapshot = job_inputs.JobInputs(conn, job, owner)
    if snapshot.data['audio'] is None:
        raise RuntimeError('The episode has no prepared audio; run Prepare first')
    worker.stage(job_id, owner, job['episode_id'], 'Splitting lines that run across hard cuts')
    with snapshot.publish(invalidate=False) as conn:
        counts = publish(conn, job['episode_id'], payload)
        stage = f"Complete · {counts['split']} split, {counts['trim']} trimmed, {counts['leave']} left as bleed" + (
            f", {counts['asked']} asked" if counts['asked'] else '') + (f", {counts['in_sequence']} in sequences" if counts['in_sequence'] else '')
        payload['result'] = counts
        conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,payload_json=?,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (stage, json.dumps(payload), job_id))
        worker.sync_episode_status(conn, job['episode_id'])
    worker.wake_event.set()


# --- undo --------------------------------------------------------------------------------------------------------

def undo(conn, decision):
    """Put the original line back while the pieces (or the trimmed line) are current and unchanged."""
    before = json.loads(decision['before_json'] or '{}')
    after = json.loads(decision['after_json'] or '{}')
    pieces = after.get('rows') or []
    if not pieces:
        raise agent_log.UndoConflict('This decision changed no line')
    rows = {row['id']: row for row in conn.execute(f"SELECT * FROM transcript_segments WHERE id IN ({','.join('?' for _ in pieces)})",
                                                   [piece['id'] for piece in pieces])}
    for piece in pieces:
        row = rows.get(piece['id'])
        if (row is None or row['superseded_by'] or row['deleted'] or row['text'] != piece['text']
                or abs(row['start_time'] - piece['start_time']) > 1e-6 or abs(row['end_time'] - piece['end_time']) > 1e-6):
            raise agent_log.UndoConflict('The line changed since; undo would overwrite that')
    original = conn.execute('SELECT * FROM transcript_segments WHERE id=?', (before['id'],)).fetchone()
    if original is None:
        raise agent_log.UndoConflict('The original line no longer exists')
    restored = str(uuid.uuid4())
    conn.execute(f"""INSERT INTO transcript_segments ({line_edit.COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (restored, original['episode_id'], original['start_time'], original['end_time'], original['text'], original['speaker_label'],
                  original['character_context_id'], original['origin'], original['model_id'], original['job_id'], pieces[0]['id'],
                  original['words_json'], original['speaker_source'], original['confidence'], original['evidence_json']))
    for piece in pieces:
        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (restored, piece['id']))
    _invalidate(conn, original['episode_id'], original['start_time'], original['end_time'])


agent_log.UNDO_KINDS[KIND] = undo
