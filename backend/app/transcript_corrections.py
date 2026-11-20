"""Corrections of current transcript lines checked against an outside source (a real .srt), in one transaction.

A correction respells a line and, rarely, names another character. The line keeps its times and its
aligner words: `transcript_parts.parts` spreads a respelled line over the original word slots, so sequence
edges stay where the voice is. Revisions carry ``origin='srt_edit'``; like every origin but ``model`` they
survive a new transcription and speaker attribution leaves them alone.

The caller opens ``BEGIN IMMEDIATE``. A line that changed since the plan was made is skipped with its
reason and the others are still applied, so a long plan is not lost to one edit made meanwhile.
"""
import json
import uuid

from . import structured_results

ORIGIN = 'srt_edit'
MAX_CORRECTIONS = 2000
COLUMNS = 'id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,model_id,job_id,parent_id,words_json,speaker_source,confidence,evidence_json'


def _evidence(old, source, correction):
    try:
        evidence = json.loads(old or '{}')
    except ValueError:
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {'previous': evidence}
    evidence['srt'] = {key: value for key, value in {
        'file': source.get('file'), 'offset': source.get('offset'), 'cues': correction.get('cues'),
        'srt_text': correction.get('srt_text'), 'note': correction.get('note')}.items() if value not in (None, '', [])}
    return json.dumps(evidence, ensure_ascii=False)


def apply(conn, episode, source, corrections):
    """Revise each named line. Returns ``{'applied': [{segment_id, id}], 'skipped': [{segment_id, reason}]}``."""
    characters = {row['id'] for row in conn.execute(
        "SELECT id FROM context_items WHERE project_id=? AND kind='character'", (episode['project_id'],))}
    applied, skipped, seen = [], [], set()
    for correction in corrections:
        segment_id = correction['segment_id']
        if segment_id in seen:
            skipped.append({'segment_id': segment_id, 'reason': 'listed twice'})
            continue
        seen.add(segment_id)
        old = conn.execute('SELECT * FROM transcript_segments WHERE id=?', (segment_id,)).fetchone()
        if not old or old['episode_id'] != episode['id']:
            skipped.append({'segment_id': segment_id, 'reason': 'not a line of this episode'})
            continue
        if old['superseded_by'] or old['deleted']:
            skipped.append({'segment_id': segment_id, 'reason': 'the line was edited or deleted since the plan was made'})
            continue
        text = correction.get('text')
        text = old['text'] if text is None else text.strip()
        character = correction.get('character_context_id')
        if not text:
            skipped.append({'segment_id': segment_id, 'reason': 'empty text'})
            continue
        if character is not None and character not in characters:
            skipped.append({'segment_id': segment_id, 'reason': 'not a character of this project'})
            continue
        speaker_changed = character is not None and character != old['character_context_id']
        if text == old['text'] and not speaker_changed:
            skipped.append({'segment_id': segment_id, 'reason': 'nothing to change'})
            continue
        new_id = str(uuid.uuid4())
        conn.execute(f'INSERT INTO transcript_segments ({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                     (new_id, old['episode_id'], old['start_time'], old['end_time'], text, old['speaker_label'],
                      character if speaker_changed else old['character_context_id'], ORIGIN, old['model_id'], old['job_id'],
                      old['id'], old['words_json'], 'manual' if speaker_changed else old['speaker_source'],
                      None if speaker_changed else old['confidence'], _evidence(old['evidence_json'], source, correction)))
        conn.execute('UPDATE transcript_segments SET superseded_by=? WHERE id=?', (new_id, old['id']))
        structured_results.invalidate_transcript(conn, episode['id'], old['start_time'], old['end_time'])
        applied.append({'segment_id': segment_id, 'id': new_id})
    return {'applied': applied, 'skipped': skipped}
