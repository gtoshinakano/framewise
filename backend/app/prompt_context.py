"""Reduce stored cut context to what a text prompt needs.

The analysis context is built for the API and the UI, so it carries database ids, model/job provenance,
revision links, timestamps and match diagnostics. None of that helps a model write about a cut, it costs
tokens and it gives the model things to over-interpret. Prompt builders call these helpers instead of
sending rows as they are. Every function accepts its own output, so queued jobs that already hold a
slimmed snapshot can be slimmed again safely.
"""

import json

from . import keyframe_view as _keyframes, pose_detail

CONSISTENCY_TITLE = 'Consistency summary'


def _present(**values):
    return {key: value for key, value in values.items() if value not in (None, '', [], {})}


def slim_transcript(segments):
    """Current spoken lines as {start_time, end_time, text, speaker_label, character_name}; deleted ones never appear."""
    lines = []
    for segment in segments or []:
        if segment.get('deleted') or not str(segment.get('text') or '').strip():
            continue
        lines.append(_present(start_time=segment.get('start_time'), end_time=segment.get('end_time'), text=segment['text'],
                              speaker_label=segment.get('speaker_label'), character_name=segment.get('character_name')))
    return lines


def transcript_line(segment):
    """One readable line, `12.000–13.500s; SPEAKER_00; Sakuta: text`."""
    speaker = segment.get('speaker_label') or 'Unknown speaker'
    character = segment.get('character_name')
    who = f'{speaker}; {character}' if character and character != speaker else speaker
    return f"{segment['start_time']:.3f}–{segment['end_time']:.3f}s; {who}: {segment['text']}"


def slim_cut(cut):
    if not cut:
        return None
    return _present(cut_index=cut.get('cut_index'), start_time=cut.get('start_time'), end_time=cut.get('end_time'),
                    title=cut.get('title'), description=cut.get('description'), notes=cut.get('notes'))


def slim_detections(detections):
    """Boxes with the character they carry; recognition scores, reasons, model names and row ids are left out."""
    if not detections:
        return None
    frames = [_present(position=frame.get('position'), time=frame.get('time')) for frame in detections.get('frames') or []]
    boxes = [_present(position=item.get('position'), box=item.get('box'), character_name=item.get('character_name'),
                      character_context_id=item.get('character_context_id'), label_source=item.get('label_source'))
             for item in detections.get('detections') or []]
    return _present(frames=frames, detections=boxes, labels_confirmed_by_queue=detections.get('labels_confirmed_by_queue'))


def slim_neighbors(context):
    """Only each neighbouring cut's Consistency summary, when it has one."""
    source = context.get('neighbor_consistency_summaries')
    if source is None:
        source = {}
        for label, rows in (context.get('neighbor_results') or {}).items():
            row = next((row for row in rows if row.get('title') == CONSISTENCY_TITLE and row.get('content')), None)
            if row:
                content = row['content']
                try:
                    content = json.loads(content) if isinstance(content, str) else content
                except ValueError:
                    pass
                source[label] = content
    # A neighbour's open questions and picture checks are not facts about it.
    return {label: ({key: value for key, value in content.items() if key not in ('questions', 'visual_check')} if isinstance(content, dict) else content) for label, content in source.items()}


def slim_context(context, dialogue=True):
    """The part of an analysis context that goes into a prompt.

    ``dialogue=False`` leaves out the transcript and the subtitle lines, for prompts that must not
    work with what is said (the Cut summary).
    """
    slim = {}
    project, episode = context.get('project') or {}, context.get('episode') or {}
    if project:
        slim['project'] = _present(title=project.get('title'), synopsis=project.get('synopsis'))
    if episode:
        slim['episode'] = _present(title=episode.get('title'))
    slim['cut'] = slim_cut(context.get('cut'))
    for label in ('previous_cut', 'next_cut'):
        if context.get(label):
            slim[label] = slim_cut(context[label])
    neighbors = slim_neighbors(context)
    if neighbors:
        slim['neighbor_consistency_summaries'] = neighbors
    chunks = [_present(start_time=chunk.get('start_time'), end_time=chunk.get('end_time'), title=chunk.get('title'), notes=chunk.get('notes'))
              for chunk in context.get('chunks') or []]
    chunks = [chunk for chunk in chunks if chunk.get('title') or chunk.get('notes')]
    if chunks:
        slim['chunks'] = chunks
    items = [_present(id=item.get('id'), kind=item.get('kind'), title=item.get('title'), content=item.get('content'))
             for item in context.get('context_items') or []]
    if items:
        slim['context_items'] = items
    if context.get('additional_context'):
        slim['additional_context'] = context['additional_context']
    if dialogue:
        slim['transcript_segments'] = slim_transcript(context.get('transcript_segments'))
        if context.get('subtitles') is not None:
            slim['subtitles'] = context['subtitles']
    detections = slim_detections(context.get('character_detections'))
    if detections:
        slim['character_detections'] = detections
    return slim


def series_context(context):
    """Background a reviewer may use: synopsis, scene notes and the series notes that are not about characters.

    Character notes are left out (they carry outfits that get copied into answers) and so are boxes and ids.
    Empty when there is nothing to say.
    """
    if not isinstance(context, dict):
        return {}
    project, episode = context.get('project') or {}, context.get('episode') or {}
    chunks = [_present(title=chunk.get('title'), notes=str(chunk.get('notes') or '')[:300]) for chunk in context.get('chunks') or [] if isinstance(chunk, dict)]
    items = [_present(kind=item.get('kind'), title=item.get('title'), content=str(item.get('content') or '')[:300])
             for item in context.get('context_items') or [] if isinstance(item, dict) and item.get('kind') != 'character' and (item.get('content') or item.get('title'))]
    return _present(synopsis=str(project.get('synopsis') or '')[:600], episode=episode.get('title'), scene_notes=[chunk for chunk in chunks if chunk], notes=items,
                    extra=str(context.get('additional_context') or '')[:600])


MOTION_VIEW = 'motion direction for the whole cut; event times are seconds from its first frame'


def is_direction(content):
    """True for a video analysis written as a motion direction (camera, per-character motion, end state)."""
    return isinstance(content, dict) and isinstance(content.get('direction'), dict)


def frame_view(content):
    """A first-frame result as a prompt reads it: the description without run bookkeeping.

    Each character's pose block becomes its compact form (state, torso, head, eyes, what is looked at,
    the app's flags); a character without one simply has no pose entry.
    """
    if not isinstance(content, dict):
        return content
    view = {key: value for key, value in content.items() if key not in ('mode', 'frame_time', 'subtitle_regions', 'contract')}
    if isinstance(view.get('visible_characters'), list):
        people = []
        for person in view['visible_characters']:
            if isinstance(person, dict) and 'pose_detail' in person:
                compact = pose_detail.compact(person['pose_detail'])
                person = {**{key: value for key, value in person.items() if key != 'pose_detail'}, **({'pose_detail': compact} if compact else {})}
            people.append(person)
        view['visible_characters'] = people
    return view


def video_view(content):
    """A motion-direction video analysis as a prompt reads it.

    One camera line, one entry per character and one event list for the whole cut, with event times in
    seconds from the cut's first frame. Window bookkeeping, token counts and model names are left out.
    A result that is not a motion direction (older analyses, or a view made earlier) is returned as it is.
    """
    if not is_direction(content) or not isinstance(content.get('windows'), list):
        return content
    direction = content['direction']
    windows = [window for window in content['windows'] if isinstance(window, dict)]
    origin = next((window['source']['source_start'] for window in windows
                   if isinstance(window.get('source'), dict) and isinstance(window['source'].get('source_start'), (int, float))), 0.0)
    events, limits = [], []
    for window in windows:
        source, analysis = window.get('source') or {}, window.get('analysis')
        if not isinstance(analysis, dict):
            continue
        offset = (source.get('source_start') if isinstance(source.get('source_start'), (int, float)) else origin) - origin
        for event in analysis.get('events') or []:
            if not isinstance(event, dict) or not str(event.get('visual_action') or '').strip():
                continue
            start, end = event.get('start_seconds'), event.get('end_seconds')
            timed = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (start, end))
            events.append(_present(seconds=f'{offset + start:.1f}-{offset + end:.1f}' if timed else 'unknown', subject=event.get('subject'), action=event['visual_action'].strip()))
        note = str(analysis.get('uncertainties') or '').strip()
        if note and note.lower().rstrip('.') != 'none' and note not in limits:
            limits.append(note)
    # A person without a character record was labelled by the video model, not marked by the user.
    people = [{**{key: value for key, value in person.items() if key != 'character_context_id'}, **({} if person.get('character_context_id') else {'identity': 'not a marked character; the label is the video model\'s'})}
              for person in direction.get('characters') or [] if isinstance(person, dict)]
    end_state = {key: value for key, value in (direction.get('end_state') or {}).items() if value and key in ('in_frame', 'source')}
    return _present(kind=MOTION_VIEW, summary=direction.get('summary'), camera=direction.get('camera'), characters=people, events=events,
                    end_of_cut=end_state, uncertainties='; '.join(limits))


FRAME_TITLES = ('Initial frame description', 'Last frame description')
VIDEO_TITLE = 'Video analysis'
KEYFRAME_TITLE = 'Keyframe analysis'
BEATS_TITLE = 'Performance beats'


# -- the motion of a cut: keyframes (since 2026-11-18), else an older video analysis ------------------------------------

def keyframe_view(content, level='review'):
    """A keyframe analysis as a prompt reads it (``keyframe_view.keyframe_view``)."""
    return _keyframes.keyframe_view(content, level)


def beats_view(content):
    """Performance beats as a prompt reads them (``keyframe_view.beats_view``)."""
    return _keyframes.beats_view(content)


def beats_current(by_type):
    """True when the cut's performance beats were built from its current keyframe analysis."""
    keyframes, beats = by_type.get('keyframe_analysis'), by_type.get('performance_beats')
    if not keyframes or not beats or not isinstance(beats.get('content'), dict):
        return False
    return (beats['content'].get('based_on') or {}).get('keyframe_analysis') == keyframes.get('id')


def motion_source(by_type):
    """``{type, id, content}`` of what tells a cut's motion: the keyframe analysis in the motion-direction shape when the
    cut has one (``keyframe_view.direction_from_keyframes``), else its older video analysis, else None.

    ``by_type`` maps result types to ``{id, content}`` (``structured_results.list_results`` items).
    """
    keyframes = by_type.get('keyframe_analysis')
    if keyframes and _keyframes.is_keyframes(keyframes.get('content')):
        beats = (by_type.get('performance_beats') or {}).get('content') if beats_current(by_type) else None
        return {'type': 'keyframe_analysis', 'id': keyframes.get('id'), 'content': _keyframes.direction_from_keyframes(keyframes['content'], beats)}
    video = by_type.get('video_analysis')
    if video and video.get('content') is not None:
        return {'type': 'video_analysis', 'id': video.get('id'), 'content': video['content']}
    return None


def with_motion(by_type):
    """A copy of ``by_type`` whose ``video_analysis`` is the cut's motion source, for readers written for the video analysis."""
    source = motion_source(by_type)
    if not source or source['type'] == 'video_analysis':
        return dict(by_type)
    return {**by_type, 'video_analysis': {'id': source['id'], 'content': source['content'], 'source_type': source['type']}}


def motion_view(by_type):
    """The cut's motion as a prompt reads a motion direction (``video_view``), or None."""
    source = motion_source(by_type)
    return video_view(source['content']) if source else None


def current_motion(conn, cut_id):
    """The current motion source of a cut read straight from the database (keyframes first, then video)."""
    rows = conn.execute("""SELECT id,content,json_extract(settings_json,'$.structured_type') AS type FROM analysis_results
                           WHERE cut_id=? AND superseded_by IS NULL AND json_extract(settings_json,'$.structured_type') IN ('keyframe_analysis','performance_beats','video_analysis')
                           ORDER BY rowid DESC""", (cut_id,)).fetchall()
    by_type = {}
    for row in rows:
        if row['type'] not in by_type:
            by_type[row['type']] = {'id': row['id'], 'content': _loaded(row['content'])}
    return motion_source(by_type)


def _loaded(content):
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except ValueError:
        return content


def slim_results(results, drop=()):
    """Results as title and content; origin only for text a person or the assistant wrote. Ids, kinds and model names stay out.

    Results whose title is in ``drop`` are left out.

    A first-frame description, a keyframe analysis, performance beats and a motion-direction video analysis are
    sent as compact objects (`frame_view`, `keyframe_view`, `beats_view`, `video_view`). When the cut has keyframes
    or a motion direction, a last-frame description kept from an older analysis is left out: it was written against
    an earlier first frame, and the newer motion already says how the cut ends. An older video analysis is left
    out once the cut has keyframes.
    """
    rows = list(results or [])
    video = next((_loaded(row.get('content')) for row in rows if row.get('title') == VIDEO_TITLE), None)
    keyframes = any(row.get('title') == KEYFRAME_TITLE for row in rows)
    current_flow = keyframes or is_direction(video) or (isinstance(video, dict) and video.get('kind') == MOTION_VIEW)
    slim = []
    for row in rows:
        title, content = row.get('title'), row.get('content')
        if title in drop or (current_flow and title == FRAME_TITLES[1]) or (keyframes and title == VIDEO_TITLE):
            continue
        if title in FRAME_TITLES or title in (VIDEO_TITLE, KEYFRAME_TITLE, BEATS_TITLE):
            loaded = _loaded(content)
            if isinstance(loaded, dict):
                content = (frame_view(loaded) if title in FRAME_TITLES else video_view(loaded) if title == VIDEO_TITLE
                           else keyframe_view(loaded, 'brief') if title == KEYFRAME_TITLE else beats_view(loaded))
        slim.append(_present(title=title, content=content, origin=row.get('origin') if row.get('origin') != 'model' else None))
    return slim
