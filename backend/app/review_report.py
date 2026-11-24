"""What in one episode waits for a person, read for scripts and agents (`scripts/fw review`).

Read-only. It reports facts that are already stored: missing steps, the checkpoints a person must settle before the next
automatic step (``gates``) and things worth a look (``checks``). Nothing here is an approval status, and nothing is saved.
Cut numbers are the editor's (1-based); ``from_cut``/``to_cut`` limit the per-cut lists and the voices to that range.
"""
import json
import time

from fastapi import APIRouter, HTTPException

from . import agent_log, anime_bulk, cut_readiness, db, media, review_decisions, runs, sequence_store, subtitle_store

router = APIRouter(prefix='/api')

# The cut flags of the editor's cut list (main.episode_response); `c` is the cut row.
CUT_FLAGS = """EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL) AS has_analysis_results,
        EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.title='Cut summary' AND a.superseded_by IS NULL) AS has_summary,
        EXISTS(SELECT 1 FROM character_detections d WHERE d.position='first' AND d.run_id=(SELECT r.id FROM character_detection_runs r WHERE r.cut_id=c.id ORDER BY r.rowid DESC LIMIT 1)) AS has_boxes,
        EXISTS(SELECT 1 FROM character_detections d WHERE d.position='first' AND d.character_context_id IS NOT NULL AND d.run_id=(SELECT r.id FROM character_detection_runs r WHERE r.cut_id=c.id ORDER BY r.rowid DESC LIMIT 1)) AS has_characters,
        EXISTS(SELECT 1 FROM character_detections d WHERE d.position='first' AND d.character_context_id IS NULL AND COALESCE(d.label_source,'')<>'manual' AND d.run_id=(SELECT r.id FROM character_detection_runs r WHERE r.cut_id=c.id ORDER BY r.rowid DESC LIMIT 1)) AS has_unlabeled_boxes,
        EXISTS(SELECT 1 FROM target_detections t JOIN target_slots s ON s.run_id=t.run_id AND s.slot=t.slot WHERE s.ignored=0
               AND t.run_id=(SELECT r.id FROM target_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time ORDER BY r.rowid DESC LIMIT 1)) AS has_target_boxes,
        EXISTS(SELECT 1 FROM target_detections t JOIN target_slots s ON s.run_id=t.run_id AND s.slot=t.slot WHERE s.character_context_id IS NULL AND COALESCE(s.label_source,'')<>'manual' AND s.ignored=0
               AND t.run_id=(SELECT r.id FROM target_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time ORDER BY r.rowid DESC LIMIT 1)) AS has_unlabeled_target_boxes,
        -- Ready for review: a first frame and the motion, i.e. performance beats built from the current keyframe analysis, or for a
        -- cut analysed before keyframes existed its video analysis (summary_queue.missing_evidence).
        (EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL AND json_extract(a.settings_json,'$.structured_type')='initial_frame')
         AND (EXISTS(SELECT 1 FROM analysis_results b JOIN analysis_results k ON k.cut_id=c.id AND k.superseded_by IS NULL AND json_extract(k.settings_json,'$.structured_type')='keyframe_analysis'
                     WHERE b.cut_id=c.id AND b.superseded_by IS NULL AND b.stale_reason IS NULL AND json_extract(b.settings_json,'$.structured_type')='performance_beats' AND json_extract(b.content,'$.based_on.keyframe_analysis')=k.id)
              OR (NOT EXISTS(SELECT 1 FROM analysis_results k WHERE k.cut_id=c.id AND k.superseded_by IS NULL AND json_extract(k.settings_json,'$.structured_type')='keyframe_analysis')
                  AND EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL AND json_extract(a.settings_json,'$.structured_type')='video_analysis')))) AS has_visual_results,
        EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL AND json_extract(a.settings_json,'$.structured_type')='consistency_summary') AS has_consistency_summary,
        -- an output an input changed after it was made (the titles the current flow makes again; stage_targets.STALE_TITLES)
        EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL AND a.stale_reason IS NOT NULL
               AND a.title IN ('Initial frame description','Keyframe analysis','Video analysis','Performance beats','Consistency summary','Cut summary')) AS has_stale_results,
        -- Newest job by rowid, which is creation order (created_at never goes backwards in rowid order). Ordering by
        -- created_at made SQLite read every candidate job's row past its payload_json, megabytes per cut.
        COALESCE((SELECT j.status='failed' FROM analysis_job_cuts jc JOIN jobs j ON j.id=jc.job_id WHERE jc.cut_id=c.id ORDER BY j.rowid DESC LIMIT 1),0) AS last_job_failed"""

def _has(structured_type):
    return f"""EXISTS(SELECT 1 FROM analysis_results a WHERE a.cut_id=c.id AND a.superseded_by IS NULL
               AND json_extract(a.settings_json,'$.structured_type')='{structured_type}')"""
ERROR_CHARS = 300


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def _setup(conn, episode):
    """Which setup steps are done. ``subtitles`` is information only: they are optional and nothing waits for them (D37)."""
    folder = media.episode_dir(episode['project_id'], episode['id'])
    subtitles = subtitle_store.current_run(episode['id'], conn)
    transcribed = episode['has_audio'] == 0 or bool(conn.execute("""SELECT 1 FROM jobs WHERE episode_id=? AND pipeline='transcribe'
        AND status='done' AND json_extract(payload_json,'$.range') IS NULL LIMIT 1""", (episode['id'],)).fetchone())
    return {'media': bool(episode['duration']) and (folder / 'proxy.mp4').is_file(),
            'cuts': bool(conn.execute('SELECT 1 FROM cuts WHERE episode_id=? AND active=1 LIMIT 1', (episode['id'],)).fetchone()),
            'transcript': transcribed,
            'subtitles': bool(subtitles),
            'boxes': not anime_bulk.eligible_cuts(conn, episode['id'], 'full')}


_target_review = cut_readiness.target_review


def _consistency(conn, cut_id):
    """The current consistency result, its open questions and contradicted picture checks, and whether answers were saved since."""
    row = conn.execute("""SELECT id,content,created_at FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL
        AND json_extract(settings_json,'$.structured_type')='consistency_summary' ORDER BY rowid DESC LIMIT 1""", (cut_id,)).fetchone()
    if not row:
        return None
    content = _loads(row['content'], {})
    questions = [item for item in content.get('questions') or [] if isinstance(item, dict)]
    answers = conn.execute('SELECT updated_at,content FROM cut_brief_instructions WHERE cut_id=?', (cut_id,)).fetchone()
    # A question the saved answers already cover (same topic and character, or one of its options) stays answered after the
    # review is written again: the person's answer is their last word (HARNESS.md D32). Corrections the reviewer settled
    # itself are never the person's to answer.
    text = answers['content'] if answers else ''
    open_questions = [item for item in review_decisions.asked(questions) if review_decisions.selection(text, item, questions) is None]
    # Answers saved after the review count as the person's reply to all of it, unless an agent wrote them (the settled
    # corrections are written in the same transaction as the review).
    saved_since = bool(answers and text.strip() and answers['updated_at'] >= row['created_at'] and not review_decisions.agent_wrote(conn, cut_id, text))
    answered = not open_questions or saved_since
    contradictions = [{'character': item.get('character'), 'picture': item.get('picture'), 'claim': item.get('claim')}
                      for item in content.get('visual_check') or [] if isinstance(item, dict) and item.get('verdict') == 'contradicts']
    draft = bool(conn.execute('SELECT 1 FROM cut_review_drafts WHERE cut_id=?', (cut_id,)).fetchone())
    return {'questions': len(open_questions) if not answered else 0, 'answered': answered, 'draft': draft, 'contradictions': contradictions}


def _voices(conn, episode_id, start, end):
    """The open voice questions, one per voice: raised once, by Prepare's voice resolver (HARNESS.md D29, D37). Line state
    alone is never a gate, so a re-transcription or a later attribution adds nothing here."""
    found = []
    for item in agent_log.escalations(conn, episode_id, 'open', kind='voice', limit=5000):
        evidence = item['evidence'] or {}
        ids = list(evidence.get('line_ids') or [])
        lines = conn.execute(f"SELECT start_time,end_time FROM transcript_segments WHERE id IN ({','.join('?' for _ in ids)})", ids).fetchall() if ids else []
        if start is not None and not any(line['end_time'] > start and line['start_time'] < end for line in lines):
            continue
        found.append({'speaker_label': evidence.get('voice_label'), 'lines': len(ids), 'reviewer': item['reviewer'], 'escalation_id': item['id'],
                      'first_time': round(min((line['start_time'] for line in lines), default=0.0), 2)})
    return sorted(found, key=lambda item: (-item['lines'], item['first_time']))


def _sequences(conn, episode_id):
    listed = []
    for row in conn.execute('SELECT * FROM sequences WHERE episode_id=? ORDER BY start_time', (episode_id,)):
        sequence = dict(row)
        if sequence_store.current(conn, sequence['id'], 'compilation') and not sequence_store.reviewed_compilation(conn, sequence):
            listed.append({'id': sequence['id'], 'title': sequence['title']})
    return listed


def _failures(conn, episode_id):
    return [{'id': run['id'], 'kind': run['kind'], 'label': run['label'], 'error': (run['error'] or '')[:ERROR_CHARS],
             'cut_span': run['cut_span'], 'finished_at': run['finished_at'], 'retry': run['actions']['retry']}
            for run in runs.describe(conn, runs.attention_rows(conn, episode_id), time.time())]


def report(conn, episode_id, from_cut=None, to_cut=None):
    episode = conn.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
    if not episode:
        raise HTTPException(404, 'Episode not found')
    if to_cut is not None and to_cut < (from_cut or 1):
        raise HTTPException(422, 'to_cut must not be before from_cut')
    episode = dict(episode)
    first, last = max(0, (from_cut or 1) - 1), (to_cut - 1 if to_cut is not None else None)
    cuts = [dict(row) for row in conn.execute(f"""SELECT c.id,c.cut_index,c.start_time,c.end_time,c.analysis_enabled,c.no_characters,
        {CUT_FLAGS},
        EXISTS(SELECT 1 FROM target_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time) AS has_target_run,
        {_has('initial_frame')} AS has_initial_frame, {_has('keyframe_analysis')} AS has_keyframes, {_has('video_analysis')} AS has_video,
        EXISTS(SELECT 1 FROM analysis_results b JOIN analysis_results k ON k.cut_id=c.id AND k.superseded_by IS NULL
               AND json_extract(k.settings_json,'$.structured_type')='keyframe_analysis'
               WHERE b.cut_id=c.id AND b.superseded_by IS NULL AND json_extract(b.settings_json,'$.structured_type')='performance_beats'
               AND json_extract(b.content,'$.based_on.keyframe_analysis')=k.id) AS has_beats,
        (SELECT r.phase FROM character_detection_runs r WHERE r.cut_id=c.id AND r.cut_start=c.start_time AND r.cut_end=c.end_time
         ORDER BY r.rowid DESC LIMIT 1) AS detection_phase
        FROM cuts c WHERE c.episode_id=? AND c.active=1 AND c.cut_index>=? AND (? IS NULL OR c.cut_index<=?) ORDER BY c.cut_index""",
        (episode_id, first, last, last))]
    gates = {'first_frame': [], 'target_frames': [], 'voices': [], 'decisions': []}
    checks = {'auto_labels': [], 'no_boxes': [], 'stale_results': [], 'picture_contradictions': []}
    # visual_missing: the visual job still has a stage to run. beats_missing: keyframes without beats built from them.
    # Stale beats are neither: re-running them is the user's call, so they are listed under checks.stale_results.
    # first_frame_missing / keyframes_missing split visual_missing into its two stages; analysis_off lists the cuts left out of it all.
    progress = {'cuts': len(cuts), 'enabled': 0, 'boxes_missing': [], 'target_missing': [], 'visual_missing': [],
                'first_frame_missing': [], 'keyframes_missing': [], 'beats_missing': [], 'consistency_missing': [],
                'summary_missing': [], 'analysis_off': []}
    for cut in cuts:
        number = cut['cut_index'] + 1
        if not cut['analysis_enabled']:
            progress['analysis_off'].append(number)
            continue
        progress['enabled'] += 1
        if not cut['no_characters']:
            if cut['detection_phase'] != 'complete':
                progress['boxes_missing'].append(number)
            elif not cut['has_boxes']:
                checks['no_boxes'].append(number)
            if cut['has_unlabeled_boxes']:
                unlabeled = conn.execute("""SELECT COUNT(*) FROM character_detections WHERE position='first' AND character_context_id IS NULL
                    AND COALESCE(label_source,'')<>'manual'
                    AND run_id=(SELECT id FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1)""", (cut['id'],)).fetchone()[0]
                gates['first_frame'].append({'cut': number, 'cut_id': cut['id'], 'unlabeled_boxes': unlabeled})
            auto = conn.execute("""SELECT COUNT(*) FROM character_detections WHERE position='first' AND label_source='auto'
                AND run_id=(SELECT id FROM character_detection_runs WHERE cut_id=? ORDER BY rowid DESC LIMIT 1)""", (cut['id'],)).fetchone()[0]
            if auto:
                checks['auto_labels'].append({'cut': number, 'cut_id': cut['id'], 'labels': auto})
            if cut['has_boxes'] and not cut['has_target_run']:
                progress['target_missing'].append(number)
            if target := _target_review(conn, cut):
                gates['target_frames'].append({'cut': number, 'cut_id': cut['id'], **target})
        if not cut['has_initial_frame']:
            progress['first_frame_missing'].append(number)
        if not (cut['has_keyframes'] or cut['has_video']):
            progress['keyframes_missing'].append(number)
        if not cut['has_initial_frame'] or not (cut['has_keyframes'] or cut['has_video']):
            progress['visual_missing'].append(number)
        elif cut['has_keyframes'] and not cut['has_beats']:
            progress['beats_missing'].append(number)
        if cut['has_visual_results'] and not cut['has_consistency_summary']:
            progress['consistency_missing'].append(number)
        if consistency := _consistency(conn, cut['id']):
            if consistency['questions'] and not consistency['answered']:
                gates['decisions'].append({'cut': number, 'cut_id': cut['id'], 'questions': consistency['questions'], 'draft': consistency['draft']})
            elif not cut['has_summary']:
                progress['summary_missing'].append(number)
            if consistency['contradictions']:
                checks['picture_contradictions'].append({'cut': number, 'cut_id': cut['id'], 'checks': consistency['contradictions']})
        for row in conn.execute("""SELECT title,stale_reason FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL
            AND stale_reason IS NOT NULL ORDER BY rowid""", (cut['id'],)):
            checks['stale_results'].append({'cut': number, 'cut_id': cut['id'], 'title': row['title'], 'reason': row['stale_reason']})
    # Cuts whose inputs are not settled enough for a step (cut_readiness, D26): the autopilot never queues the step for them.
    progress['held'] = {'target_frames': [], 'visual': [], 'beats': [], 'consistency': []}
    enabled = [cut for cut in cuts if cut['analysis_enabled']]
    inbox = cut_readiness.inbox(conn, episode_id, enabled)
    for cut in enabled:
        ready = cut_readiness.for_cut(conn, episode, cut, inbox.get(cut['id'], set()))
        for step, reasons in ready['held'].items():
            if reasons:
                progress['held'][step].append({'cut': cut['cut_index'] + 1, 'cut_id': cut['id'], 'reasons': reasons})
    ranged = from_cut is not None or to_cut is not None
    start, end = (cuts[0]['start_time'], cuts[-1]['end_time']) if ranged and cuts else (None, None)
    if not ranged or cuts:
        gates['voices'] = _voices(conn, episode_id, start, end)
    checks['sequences_unconfirmed'] = _sequences(conn, episode_id)
    return {'episode': {'id': episode['id'], 'project_id': episode['project_id'], 'title': episode['title'],
                        'subtitle_language': episode.get('subtitle_language'), 'has_audio': episode['has_audio']},
            'range': {'from_cut': first + 1, 'to_cut': to_cut} if ranged else None,
            'setup': _setup(conn, episode), 'gates': gates, 'checks': checks, 'progress': progress,
            'failures': _failures(conn, episode_id)}


@router.get('/episodes/{episode_id}/review', summary='What in this episode waits for a person: open checkpoints, checks and missing steps')
def episode_review(episode_id: str, from_cut: int | None = None, to_cut: int | None = None):
    with db.connect() as conn:
        conn.execute('BEGIN')
        return report(conn, episode_id, from_cut, to_cut)
