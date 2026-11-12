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


def slim_results(results):
    """Results as title and content; origin only for text a person or the assistant wrote. Ids, kinds and model names stay out."""
    return [_present(title=row.get('title'), content=row.get('content'), origin=row.get('origin') if row.get('origin') != 'model' else None)
            for row in results or []]
