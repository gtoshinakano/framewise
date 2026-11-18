"""Performance beats: a text model compiles a cut's acting into timed beats (the ``performance_beats`` text job).

The keyframe analysis says what each character looks like at each keyframe and what changed between two of them;
the first frame says how the cut starts; the transcript says when someone speaks. This job turns that into the
timed acting plan a performance director would give: beats that tile the cut, each with the camera, every visible
character's body, face, gaze and mouth, and the beat's dramatic intent, plus how each spoken line is delivered
(``on_screen`` lip-synced, ``off_screen``, ``inner_voice``, ``narrative``). The words are never sent to the model and
never stored here: the dialogue belongs to the transcript and the sequence's dialogue step.

The app calculates and the model decides: the app gives the beat edges it may use (keyframe times and line edges),
each line's speaker, whether the speaker is in the picture and whether their mouth moves at the keyframes, and a
suggested delivery; the model chooses the beats, the acting words and the delivery. ``validate`` snaps and checks
the answer and sends its sentences back to the model.

The job is queued in the transaction that saves a keyframe analysis (``ensure_queued``), by hand
(``POST /api/cuts/{id}/performance-beats``) and as the first link of the text chain. It reads its inputs when it
starts; a keyframe analysis saved meanwhile makes it finish as superseded (a newer beats job is queued with it).
The result is the ``performance_beats`` analysis result (contract ``beats-v1``, ``based_on`` the keyframe and
first-frame results it read). It has no approval status.
"""
import json
import logging
import os
import time
import uuid

from . import db, keyframe_view, prompt_context, structured_results, worker

log = logging.getLogger('framewise.performance_beats')

CONTRACT = 'beats-v1'
PROMPT_VERSION = 'beats-v1'
DELIVERIES = ('on_screen', 'off_screen', 'inner_voice', 'narrative')
MOUTHS = ('speaking', 'closed', 'open', 'not_visible', 'unclear')
MAX_BEATS = 12
SNAP = 0.25          # seconds an edge may be off an anchor (or the cut's ends) and still be snapped onto it
MIN_BEAT = 0.2
ON_SCREEN_SLACK = 0.5  # a speaker counts as in the picture when a keyframe this close to the line shows them
FIELDS = ['summary', 'beats', 'speech']

SYSTEM = '''You are the performance director for one cut of an anime episode that is being re-created shot by shot with AI: a still image of the opening frame is generated first, then a video model animates that still over the cut's original audio. Write the timed acting plan the video model follows.

## EVIDENCE

- opening_frame: a large vision model's description of the cut's first frame. It decides how the cut starts: who is in the picture, where, in which pose, facing where and looking at what.
- keyframes: frames picked where the picture changes, labelled F1, F2 ... with seconds from the first frame. "frames" gives each visible character's state at each keyframe; "changes" lists what changed between two consecutive keyframes. A change happened somewhere inside its pair: nothing between two keyframes was seen. body_measured and changes marked "keypoints" were measured from body keypoints; "vision model" changes were read from the images. The camera of each pair was measured from the background pixels and is never a guess; "not measurable" means the picture had no usable background.
- speech: the transcript lines heard in the cut, by id, with seconds from the first frame and the speaker. The words are not given and must not be guessed. on_screen says whether the speaker is in the picture at a keyframe near the line; mouth says whether the keyframes show that speaker's mouth moving (yes, no or unclear); suggested is the app's delivery for the line, or null when it cannot tell.
- characters: the names you may use. series: background only, never evidence of what this cut shows.

## BEATS

- Beats tile the whole cut: the first starts at 0, each next one starts where the previous one ended, the last ends at the cut's duration. Every edge between two beats must be one of the anchors (keyframe times and line edges). Use at most max_beats beats.
- Start a new beat where the acting changes: a pose or gesture, a gaze shift, an expression, a line starting or ending, a camera move. Merge stretches where nothing changes; a cut where nothing changes is one beat.
- For every character visible in the beat, by exact name: body (pose and what the body, arms and legs do), face (expression and how it changes), gaze (whom or what the eyes and head are directed at; left and right are the viewer's), mouth (speaking, closed, open, not_visible or unclear). Write not visible for a part the picture does not show. Describe only what the evidence supports; never invent motion.
- camera: the measured move of the pairs inside the beat (static when they are all static), with a short detail.
- intent: a short phrase on what the beat does dramatically (a reaction, a decision, a reveal). Never the content of what is said.

## SPEECH

- Give every line exactly one delivery: on_screen (the speaker is visible and lip-synced), off_screen (the speaker is not in the picture), inner_voice (the speaker is visible but the mouth does not move: a thought, or the face turned away) or narrative (a voice over the picture, too far or too small for lip sync).
- Follow suggested unless the keyframes clearly contradict it; when it is null, decide from on_screen and mouth.
- A beat's mouth value agrees with its lines: a character with an on_screen line in the beat is speaking.
- Never quote, paraphrase or summarise speech, and never mention subtitles or on-screen text.

## OUTPUT

Return one JSON object: {"summary": "at most two sentences on the cut's performance, without the words", "beats": [{"start": seconds, "end": seconds, "camera": {"move": "...", "detail": "..."}, "intent": "...", "characters": [{"name": "...", "body": "...", "face": "...", "gaze": "...", "mouth": "..."}]}], "speech": [{"line": "S1", "delivery": "..."}]}.
Write each field as a short phrase, in English, with the character names. JSON only.'''


# -- inputs (read when the job starts) -----------------------------------------------------------------------------------

def _rows(conn, sql, args=()):
    return [dict(row) for row in conn.execute(sql, args).fetchall()]


def _speakers_in(frames, start, end, slack=ON_SCREEN_SLACK):
    """Character names and ids in the picture at the keyframes inside ``[start - slack, end + slack]`` (else the last one before)."""
    near = [frame for frame in frames if start - slack <= frame.get('t', 0) <= end + slack]
    if not near:
        before = [frame for frame in frames if frame.get('t', 0) <= start]
        near = before[-1:] if before else frames[:1]
    names, ids = set(), set()
    for frame in near:
        for person in frame.get('characters') or []:
            names.add(person.get('name'))
            if person.get('character_context_id'):
                ids.add(person['character_context_id'])
    return names, ids


def _mouth(segments, start, end, name, character_id):
    values = []
    for segment in segments:
        if segment.get('end', 0) <= start or segment.get('start', 0) >= end:
            continue
        for person in segment.get('characters') or []:
            if person.get('name') == name or (character_id and person.get('character_context_id') == character_id):
                values.append(person.get('speaking'))
    if 'yes' in values:
        return 'yes'
    return 'no' if values and all(value == 'no' for value in values) else 'unclear'


def suggested_delivery(on_screen, mouth, linked):
    """The app's delivery for one line: off screen without the speaker, lip-synced with a moving mouth, inner voice without; else None."""
    if not linked:
        return None
    if not on_screen:
        return 'off_screen'
    return {'yes': 'on_screen', 'no': 'inner_voice'}.get(mouth)


def speech_lines(transcript, cut, keyframes):
    """The cut's merged transcript lines as ``S1…`` with clip-relative times, speaker, picture and mouth evidence, never the words."""
    from .assistant.transcript_edit import merged_transcript
    duration = cut['end_time'] - cut['start_time']
    frames, segments = keyframes.get('frames') or [], keyframes.get('segments') or []
    lines = []
    for line in merged_transcript(transcript):
        start = round(max(0.0, line['start_time'] - cut['start_time']), 2)
        end = round(min(duration, line['end_time'] - cut['start_time']), 2)
        if end <= start:
            continue
        name, character_id = line.get('character_name'), line.get('character_context_id')
        names, ids = _speakers_in(frames, start, end)
        on_screen = bool(name and (name in names or (character_id and character_id in ids)))
        mouth = _mouth(segments, start, end, name, character_id) if on_screen else 'unclear'
        lines.append({'line': f'S{len(lines) + 1}', 'start': start, 'end': end, 'speaker': name or line.get('speaker_label') or 'unknown speaker',
                      'character_context_id': character_id, 'on_screen': on_screen, 'mouth': mouth,
                      'suggested': suggested_delivery(on_screen, mouth, bool(name))})
    return lines


def anchors_for(duration, keyframes, lines):
    times = {0.0, round(duration, 2)}
    times.update(round(frame['t'], 2) for frame in keyframes.get('frames') or [] if isinstance(frame.get('t'), (int, float)))
    for line in lines:
        times.update((line['start'], line['end']))
    return sorted(t for t in times if 0 <= t <= duration + 1e-6)


def gather(conn, cut_id):
    """``(model input, validation context, based_on)`` from the cut's current results; RuntimeError when it has no keyframe analysis."""
    cut = conn.execute('''SELECT c.*,e.project_id,e.title AS episode_title,e.subtitle_language FROM cuts c JOIN episodes e ON e.id=c.episode_id
                          WHERE c.id=? AND c.active=1''', (cut_id,)).fetchone()
    if not cut:
        raise RuntimeError('The cut is archived or was deleted')
    by_type = {item['type']: item for item in structured_results.list_results(cut_id, conn=conn)}
    keyframes = by_type.get('keyframe_analysis')
    if not keyframes or not keyframe_view.is_keyframes(keyframes.get('content')):
        raise RuntimeError('This cut has no keyframe analysis; run Keyframes first')
    first = by_type.get('initial_frame')
    transcript = _rows(conn, '''SELECT t.id,t.start_time,t.end_time,t.text,t.speaker_label,t.character_context_id,t.origin,c.title AS character_name
                               FROM transcript_segments t LEFT JOIN context_items c ON c.id=t.character_context_id
                               WHERE t.episode_id=? AND t.superseded_by IS NULL AND t.deleted=0 AND t.start_time<? AND t.end_time>? AND trim(t.text)!=''
                               ORDER BY t.start_time,t.id''', (cut['episode_id'], cut['end_time'], cut['start_time']))
    duration = round(cut['end_time'] - cut['start_time'], 2)
    lines = speech_lines(transcript, cut, keyframes['content'])
    anchors = anchors_for(duration, keyframes['content'], lines)
    roster = {person['name']: person.get('character_context_id') for person in keyframes['content'].get('roster') or [] if person.get('name')}
    for frame in keyframes['content'].get('frames') or []:
        for person in frame.get('characters') or []:
            roster.setdefault(person.get('name'), person.get('character_context_id'))
    if first and isinstance(first.get('content'), dict):
        for person in first['content'].get('visible_characters') or []:
            if isinstance(person, dict) and person.get('character'):
                roster.setdefault(person['character'], person.get('character_context_id'))
    roster.pop(None, None)
    project = conn.execute('SELECT synopsis FROM projects WHERE id=?', (cut['project_id'],)).fetchone()
    series = prompt_context.series_context({
        'project': {'synopsis': project['synopsis'] if project else ''}, 'episode': {'title': cut['episode_title']},
        'chunks': _rows(conn, 'SELECT title,notes FROM chunks WHERE episode_id=? AND start_time<? AND end_time>? ORDER BY chunk_index', (cut['episode_id'], cut['end_time'], cut['start_time'])),
        'context_items': _rows(conn, 'SELECT kind,title,content FROM context_items WHERE project_id=? AND (episode_id IS NULL OR episode_id=?)', (cut['project_id'], cut['episode_id']))})
    opening = None
    if first and isinstance(first.get('content'), dict):
        from .consistency_summary import _without_ids
        opening = _without_ids(prompt_context.frame_view(first['content']))
    max_beats = max(1, min(MAX_BEATS, len(anchors) - 1))
    data = {'cut': {'duration': duration}, 'opening_frame': opening or 'unavailable', 'keyframes': prompt_context.keyframe_view(keyframes['content'], 'beats'),
            'speech': [{key: line[key] for key in ('line', 'start', 'end', 'speaker', 'on_screen', 'mouth', 'suggested')} for line in lines] or 'nobody speaks in this cut',
            'characters': sorted(roster), 'anchors': anchors, 'max_beats': max_beats}
    if series:
        data['series'] = series
    if cut['subtitle_language']:
        data['dialogue_language'] = cut['subtitle_language']
    context = {'duration': duration, 'anchors': anchors, 'names': roster, 'lines': lines, 'max_beats': max_beats,
               'cut': {'cut_index': cut['cut_index'], 'start_time': cut['start_time'], 'end_time': cut['end_time']}}
    based_on = {'keyframe_analysis': keyframes['id'], 'initial_frame': first['id'] if first else None}
    return data, context, based_on


# -- the answer ----------------------------------------------------------------------------------------------------------

def answer_format(names, line_ids, max_beats):
    """``response_format`` for the beats: names and line ids are enums when there are any (an empty enum breaks llama-server)."""
    text = {'type': 'string', 'maxLength': 240}
    camera = {'type': 'object', 'additionalProperties': False, 'required': ['move', 'detail'],
              'properties': {'move': {'type': 'string', 'maxLength': 40}, 'detail': {'type': 'string', 'maxLength': 200}}}
    person = {'type': 'object', 'additionalProperties': False, 'required': ['name', 'body', 'face', 'gaze', 'mouth'],
              'properties': {'name': {'enum': list(names)} if names else {'type': 'string', 'maxLength': 80}, 'body': text, 'face': text, 'gaze': text,
                             'mouth': {'enum': list(MOUTHS)}}}
    beat = {'type': 'object', 'additionalProperties': False, 'required': ['start', 'end', 'camera', 'intent', 'characters'],
            'properties': {'start': {'type': 'number'}, 'end': {'type': 'number'}, 'camera': camera, 'intent': {'type': 'string', 'maxLength': 200},
                           'characters': {'type': 'array', 'maxItems': 6, 'items': person}}}
    speech = {'type': 'object', 'additionalProperties': False, 'required': ['line', 'delivery'],
              'properties': {'line': {'enum': list(line_ids)} if line_ids else {'type': 'string'}, 'delivery': {'enum': list(DELIVERIES)}}}
    schema = {'type': 'object', 'additionalProperties': False, 'required': FIELDS,
              'properties': {'summary': {'type': 'string', 'maxLength': 600}, 'beats': {'type': 'array', 'minItems': 1, 'maxItems': max_beats, 'items': beat},
                             'speech': {'type': 'array', 'minItems': len(line_ids), 'maxItems': len(line_ids), 'items': speech}}}
    return {'type': 'json_schema', 'json_schema': {'name': 'performance_beats', 'strict': True, 'schema': schema}}


def _text(value, limit=240):
    return ' '.join(value.split())[:limit] if isinstance(value, str) else ''


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _nearest(value, anchors):
    return min(anchors, key=lambda anchor: abs(anchor - value))


def validate(answer, ctx):
    """The answer as stored; a ValueError (sentences, one per problem) goes back to the model with its answer."""
    if not isinstance(answer, dict):
        raise ValueError('The answer must be one JSON object with summary, beats and speech.')
    duration, anchors, names = ctx['duration'], ctx['anchors'], ctx['names']
    raw = [beat for beat in answer.get('beats') or [] if isinstance(beat, dict)]
    errors = []
    if not raw:
        raise ValueError('beats must hold at least one beat covering the whole cut, from 0 to %.2f seconds.' % duration)
    if len(raw) > ctx['max_beats']:
        errors.append(f"Use at most {ctx['max_beats']} beats; merge beats where nothing changes.")
    beats = []
    for number, beat in enumerate(sorted(raw, key=lambda item: _number(item.get('start')) if _number(item.get('start')) is not None else 0), 1):
        start, end = _number(beat.get('start')), _number(beat.get('end'))
        if start is None or end is None:
            errors.append(f'Beat {number} needs numeric start and end seconds.')
            continue
        beats.append({'start': start, 'end': end, 'raw': beat})
    if errors:
        raise ValueError(' '.join(errors))
    first, last = beats[0], beats[-1]
    if abs(first['start']) > SNAP:
        errors.append(f"The first beat must start at 0 (it starts at {first['start']:.2f} s).")
    first['start'] = 0.0
    if abs(last['end'] - duration) > SNAP:
        errors.append(f"The last beat must end at the cut's duration, {duration:.2f} s (it ends at {last['end']:.2f} s).")
    last['end'] = duration
    for number, (before, after) in enumerate(zip(beats, beats[1:]), 2):
        if abs(after['start'] - before['end']) > 0.1:
            errors.append(f"Beat {number} starts at {after['start']:.2f} s but beat {number - 1} ends at {before['end']:.2f} s; beats must follow each other without gaps or overlaps.")
            continue
        edge = _nearest(before['end'], anchors)
        if abs(edge - before['end']) > SNAP:
            close = sorted(anchors, key=lambda anchor: abs(anchor - before['end']))[:3]
            errors.append(f"The edge at {before['end']:.2f} s between beats {number - 1} and {number} is not an anchor; use one of {', '.join(f'{a:.2f}' for a in sorted(close))}.")
            continue
        before['end'] = after['start'] = edge
    for number, beat in enumerate(beats, 1):
        if beat['end'] - beat['start'] < MIN_BEAT - 1e-6:
            errors.append(f"Beat {number} ({beat['start']:.2f}-{beat['end']:.2f} s) is shorter than {MIN_BEAT} s; merge it with a neighbour.")
    line_ids = [line['line'] for line in ctx['lines']]
    given = {}
    for item in answer.get('speech') or []:
        if not isinstance(item, dict):
            continue
        line, delivery = item.get('line'), item.get('delivery')
        if line not in line_ids:
            errors.append(f'speech names an unknown line {line!r}; use only {", ".join(line_ids) or "no lines (an empty list)"}.')
        elif line in given:
            errors.append(f'{line} has more than one delivery; give each line exactly one.')
        elif delivery not in DELIVERIES:
            errors.append(f'{line}: delivery must be one of {", ".join(DELIVERIES)}.')
        else:
            given[line] = delivery
    missing = [line for line in line_ids if line not in given]
    if missing:
        errors.append(f'Give a delivery for every line; missing: {", ".join(missing)}.')
    if errors:
        raise ValueError(' '.join(errors))
    lines = {line['line']: line for line in ctx['lines']}
    speech = [{'line': line_id, 'speaker': lines[line_id]['speaker'], 'character_context_id': lines[line_id]['character_context_id'],
               'start': lines[line_id]['start'], 'end': lines[line_id]['end'], 'delivery': given[line_id], 'suggested': lines[line_id]['suggested']}
              for line_id in line_ids]
    checks, out = [], []
    for number, beat in enumerate(beats, 1):
        raw = beat['raw']
        camera = raw.get('camera') if isinstance(raw.get('camera'), dict) else {}
        people, seen = [], set()
        for person in raw.get('characters') or []:
            if not isinstance(person, dict) or not _text(person.get('name')):
                continue
            name = _text(person['name'], 80)
            if name in seen:
                continue
            seen.add(name)
            mouth = person.get('mouth') if person.get('mouth') in MOUTHS else 'unclear'
            people.append({'character': name, 'character_context_id': names.get(name), 'body': _text(person.get('body')), 'face': _text(person.get('face')),
                           'gaze': _text(person.get('gaze')), 'mouth': mouth})
        inside = [item for item in speech if beat['start'] <= (item['start'] + item['end']) / 2 < beat['end'] or (beat is beats[-1] and (item['start'] + item['end']) / 2 == beat['end'])]
        for person in people:
            spoken = [item for item in inside if item['speaker'] == person['character'] and item['delivery'] == 'on_screen']
            if person['mouth'] == 'speaking' and not any(item['speaker'] == person['character'] for item in inside):
                checks.append({'kind': 'mouth_without_line', 'beat': number, 'character': person['character']})
            if spoken and person['mouth'] in ('closed', 'not_visible'):
                checks.append({'kind': 'lip_sync_without_mouth', 'beat': number, 'character': person['character']})
        out.append({'start': round(beat['start'], 2), 'end': round(beat['end'], 2), 'camera': {'move': _text(camera.get('move'), 40) or 'static', 'detail': _text(camera.get('detail'), 200)},
                    'intent': _text(raw.get('intent'), 200), 'characters': people, 'speech': inside})
    for item in speech:
        if item['delivery'] == 'on_screen' and not lines[item['line']]['on_screen']:
            checks.append({'kind': 'on_screen_speaker_not_seen', 'line': item['line'], 'speaker': item['speaker']})
        if item['suggested'] and item['delivery'] != item['suggested']:
            checks.append({'kind': 'delivery_differs_from_suggestion', 'line': item['line'], 'suggested': item['suggested'], 'chosen': item['delivery']})
    return {'contract': CONTRACT, 'duration': duration, 'summary': _text(answer.get('summary'), 600), 'beats': out, 'speech': speech, 'checks': checks}


# -- queueing ------------------------------------------------------------------------------------------------------------

def text_config(conn):
    """The text engine the user chose (``text_generation_settings``) as a job's ``config``."""
    row = conn.execute('SELECT text_provider,local_thinking,openrouter_model FROM text_generation_settings WHERE id=1').fetchone()
    if not row:
        return {'text_provider': 'openrouter', 'local_thinking': False, 'openrouter_model': 'openrouter/free'}
    return {'text_provider': row['text_provider'], 'local_thinking': bool(row['local_thinking']), 'openrouter_model': row['openrouter_model']}


def pending_job(conn, cut_id, statuses=('queued', 'running')):
    marks = ','.join('?' for _ in statuses)
    row = conn.execute(f"SELECT id FROM jobs WHERE pipeline='performance_beats' AND status IN ({marks}) AND json_extract(payload_json,'$.cut_id')=? ORDER BY rowid DESC LIMIT 1",
                       (*statuses, cut_id)).fetchone()
    return row['id'] if row else None


def insert(conn, episode_id, cut_id, config=None):
    job_id = str(uuid.uuid4())
    conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,?,'performance_beats',?)",
                 (job_id, episode_id, json.dumps({'cut_id': cut_id, 'config': config or text_config(conn)})))
    return job_id


def ensure_queued(conn, episode_id, cut_id, config=None):
    """A queued beats job for the cut: the waiting one, else a new one (also beside a running one, which reads older inputs)."""
    return pending_job(conn, cut_id, ('queued',)) or insert(conn, episode_id, cut_id, config)


# -- the job -------------------------------------------------------------------------------------------------------------

def generate(config, data, ctx, progress=None, cancel_check=None, usage_out=None):
    from .inference_worker import chat_json
    return chat_json(config.get('openrouter_model', 'openrouter/free'), SYSTEM, json.dumps(data, ensure_ascii=False), FIELDS, max_tokens=6144,
                     progress=progress, cancel_check=cancel_check, provider=config.get('text_provider', 'openrouter'), thinking=config.get('local_thinking', False),
                     usage_out=usage_out, validator=lambda answer: validate(answer, ctx),
                     response_format=answer_format(sorted(ctx['names']), [line['line'] for line in ctx['lines']], ctx['max_beats']), temperature=0.2)


def _done(conn, job_id, stage='Complete'):
    conn.execute("UPDATE jobs SET status='done',stage=?,error=NULL,lease_owner=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?", (stage, job_id))


def process(job_id, owner):
    from . import inference_worker, local_text_runtime
    job = db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND lease_owner=?", (job_id, owner))
    if not job:
        return
    payload = json.loads(job['payload_json'])
    cut_id, config = payload['cut_id'], payload.get('config') or {}
    try:
        # A crash can occur after committing the result and before marking the job done.
        if any(structured_results.structured_type(row['settings_json']) == 'performance_beats'
               for row in db.all_rows('SELECT settings_json FROM analysis_results WHERE job_id=? AND cut_id=?', (job_id, cut_id))):
            with db.connect() as conn:
                worker.require_lease(conn, job_id, owner)
                _done(conn, job_id)
                worker.sync_episode_status(conn, job['episode_id'])
            return
        with worker.analysis_stage(job_id, owner, cut_id, 'performance_beats', 'Performance beats'):
            worker.stage(job_id, owner, job['episode_id'], 'Writing performance beats')
            with db.connect() as conn:
                data, ctx, based_on = gather(conn, cut_id)
                base = next((item['id'] for item in structured_results.list_results(cut_id, conn=conn) if item['type'] == 'performance_beats'), None)
            telemetry = {}
            started = time.monotonic()
            content, actual_model = generate(config, data, ctx,
                progress=lambda attempt, total: worker.stage(job_id, owner, job['episode_id'], f'Performance beats · attempt {attempt}/{total}'),
                cancel_check=lambda: inference_worker.check_summary_lease(job_id, owner), usage_out=telemetry)
            elapsed = time.monotonic() - started
            content['based_on'] = based_on
            local = config.get('text_provider') == 'local'
            settings = {'provider': config.get('text_provider', 'openrouter'), 'prompt_version': PROMPT_VERSION, 'beats_contract': CONTRACT,
                        'model_revision': local_text_runtime.revision_for(telemetry.get('model_variant')) if local else None,
                        'thinking': config.get('local_thinking', False) if local else None,
                        'thinking_budget_tokens': int(os.environ.get('QWEN36_THINKING_BUDGET', '2048')) if local and config.get('local_thinking') else None,
                        'summary_generation_seconds': elapsed, 'usage': telemetry}
            with db.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                worker.require_lease(conn, job_id, owner)
                current = next((item for item in structured_results.list_results(cut_id, conn=conn) if item['type'] == 'keyframe_analysis'), None)
                if not current or current['id'] != based_on['keyframe_analysis']:
                    # A newer keyframe analysis queued its own beats job in the transaction that saved it.
                    _done(conn, job_id, 'Superseded by a newer keyframe analysis')
                    worker.sync_episode_status(conn, job['episode_id'])
                    return
                structured_results.save_generated(conn, cut_id, job_id, owner, 'performance_beats', content, actual_model, SYSTEM, settings,
                                                  {'cut': ctx['cut'], 'input': data}, expected_current=base)
        with db.connect() as conn:
            worker.require_lease(conn, job_id, owner)
            _done(conn, job_id)
            worker.sync_episode_status(conn, job['episode_id'])
    except inference_worker.SummaryCancelled:
        if config.get('text_provider') == 'local':
            local_text_runtime.cancel_local('Cancelled')
        log.info('Performance beats %s was cancelled', job_id)
    except Exception as exc:
        inference_worker.fail_job(job_id, owner, job['episode_id'], exc)
        log.exception('Performance beats failed for job %s', job_id)
