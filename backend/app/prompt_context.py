"""Reduce stored cut context to what a text prompt needs.

The analysis context is built for the API and the UI, so it carries database ids, model/job provenance,
revision links, timestamps and match diagnostics. None of that helps a model write about a cut, it costs
tokens and it gives the model things to over-interpret. Prompt builders call these helpers instead of
sending rows as they are. Every function accepts its own output, so queued jobs that already hold a
slimmed snapshot can be slimmed again safely.
"""

import json

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
    return source


def slim_context(context):
    """The part of an analysis context that goes into a prompt."""
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
    slim['transcript_segments'] = slim_transcript(context.get('transcript_segments'))
    if context.get('subtitles') is not None:
        slim['subtitles'] = context['subtitles']
    detections = slim_detections(context.get('character_detections'))
    if detections:
        slim['character_detections'] = detections
    return slim


MOTION_VIEW = 'motion direction for the whole cut; event times are seconds from its first frame'


def is_direction(content):
    """True for a video analysis written as a motion direction (camera, per-character motion, end state)."""
    return isinstance(content, dict) and isinstance(content.get('direction'), dict)


def frame_view(content):
    """A first-frame result as a prompt reads it: the description without run bookkeeping."""
    if not isinstance(content, dict):
        return content
    return {key: value for key, value in content.items() if key not in ('mode', 'frame_time', 'subtitle_regions')}


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


def _loaded(content):
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except ValueError:
        return content


def slim_results(results):
    """Results as title and content; origin only for text a person or the assistant wrote. Ids, kinds and model names stay out.

    A first-frame description and a motion-direction video analysis are sent as compact objects
    (`frame_view`, `video_view`). When the cut has a motion direction, a last-frame description kept
    from an older analysis is left out: it was written against an earlier first frame, and the
    direction already says how the cut ends.
    """
    rows = list(results or [])
    video = next((_loaded(row.get('content')) for row in rows if row.get('title') == VIDEO_TITLE), None)
    current_flow = is_direction(video) or (isinstance(video, dict) and video.get('kind') == MOTION_VIEW)
    slim = []
    for row in rows:
        title, content = row.get('title'), row.get('content')
        if current_flow and title == FRAME_TITLES[1]:
            continue
        if title in FRAME_TITLES or title == VIDEO_TITLE:
            loaded = _loaded(content)
            if isinstance(loaded, dict):
                content = frame_view(loaded) if title in FRAME_TITLES else video_view(loaded)
        slim.append(_present(title=title, content=content, origin=row.get('origin') if row.get('origin') != 'model' else None))
    return slim
