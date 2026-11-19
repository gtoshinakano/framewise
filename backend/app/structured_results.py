"""Revision-safe storage for structured single-cut pipeline outputs."""
import json
import uuid
from . import db, worker

KINDS = {
    'initial_frame': ('first_frame', 'Initial frame description'),
    'video_analysis': ('cut_description', 'Video analysis'),  # replaced by keyframe_analysis on 2026-11-18; older results stay readable
    'keyframe_analysis': ('cut_description', 'Keyframe analysis'),
    'performance_beats': ('cut_description', 'Performance beats'),
    'last_frame': ('last_frame', 'Last frame description'),
    'subtitle_ocr': ('other', 'Subtitle OCR'),
    'consistency_summary': ('cut_description', 'Consistency summary'),
}


def structured_type(settings):
    try:
        kind = json.loads(settings or '{}').get('structured_type')
        return kind if kind in KINDS else None
    except (ValueError, TypeError, AttributeError):
        return None


def list_results(cut_id, history=False, conn=None):
    from . import frame_source
    if conn is None:
        with db.connect() as conn:
            return list_results(cut_id, history, conn)
    where = '' if history else ' AND superseded_by IS NULL'
    rows = conn.execute(f'SELECT * FROM analysis_results WHERE cut_id=?{where} ORDER BY rowid DESC', (cut_id,)).fetchall()
    results = []
    for row in rows:
        kind = structured_type(row['settings_json'])
        if not kind:
            continue
        try:
            content = json.loads(row['content'])
        except (ValueError, TypeError):
            content = None
        if not isinstance(content, dict):
            content = None
        frame_url = f"/api/analysis-results/{row['id']}/frame" if kind in ('initial_frame', 'last_frame') and frame_source.result_frame_spec(row['id'], conn) else None
        results.append({'id': row['id'], 'cut_id': row['cut_id'], 'type': kind, 'content': content, 'frame_url': frame_url,
                        'origin': row['origin'], 'model_id': row['model_id'], 'job_id': row['job_id'],
                        'parent_id': row['parent_id'], 'superseded_by': row['superseded_by'], 'created_at': row['created_at'],
                        'stale_reason': row['stale_reason']})
    return results


def invalidate_text(conn, cut_id, evidence):
    """A change to current visual evidence makes its existing text derivatives stale."""
    reason = {'initial_frame': 'Initial frame changed', 'keyframe_analysis': 'Keyframe analysis changed'}.get(evidence)
    if not reason:
        return
    conn.execute("""UPDATE analysis_results SET stale_reason=? WHERE cut_id=? AND superseded_by IS NULL
        AND (json_extract(settings_json,'$.structured_type') IN ('performance_beats','consistency_summary') OR title='Cut summary')""",
        (reason, cut_id))


def invalidate_detections(conn, cut_id, *, target=False, reason=None):
    """Existing analyses of changed people remain available with a stale reason."""
    reason = reason or ('Target frame detections changed' if target else 'First-frame detections changed')
    types = ('keyframe_analysis', 'performance_beats', 'consistency_summary')
    if not target:
        types = ('initial_frame', *types)
    marks = ','.join('?' for _ in types)
    conn.execute(f"""UPDATE analysis_results SET stale_reason=? WHERE cut_id=? AND superseded_by IS NULL
        AND (json_extract(settings_json,'$.structured_type') IN ({marks}) OR title='Cut summary')""", (reason, cut_id, *types))


def opening_box_edited(conn, run_id, position):
    if position != 'first':
        return
    row = conn.execute("""SELECT c.id FROM character_detection_runs r JOIN cuts c ON c.id=r.cut_id
        WHERE r.id=? AND r.cut_start=c.start_time AND r.cut_end=c.end_time
          AND r.rowid=(SELECT MAX(latest.rowid) FROM character_detection_runs latest WHERE latest.cut_id=c.id)""", (run_id,)).fetchone()
    if row:
        invalidate_detections(conn, row['id'], reason='First-frame boxes or labels were edited')


def save_generated(conn, cut_id, job_id, owner, result_type, content, model_id, prompt, settings, context, *, expected_current=...):
    """Append a generated revision while holding the worker lease and transaction."""
    worker.require_lease(conn, job_id, owner)
    if result_type not in KINDS or not isinstance(content, dict):
        raise ValueError('Invalid structured result')
    body = json.dumps(content, ensure_ascii=False, allow_nan=False)
    if len(body) > 5000000:
        raise ValueError('Structured result exceeds 5000000 characters')
    current = next((row for row in conn.execute('SELECT id,settings_json FROM analysis_results WHERE cut_id=? AND superseded_by IS NULL ORDER BY rowid DESC', (cut_id,)) if structured_type(row['settings_json']) == result_type), None)
    if expected_current is not ... and (current['id'] if current else None) != expected_current:
        raise RuntimeError('The result changed while the job was queued; run it again')
    result_id = str(uuid.uuid4())
    kind, title = KINDS[result_type]
    conn.execute('''INSERT INTO analysis_results (id,cut_id,kind,title,content,origin,model_id,job_id,prompt,settings_json,context_json,parent_id)
                    VALUES (?,?,?,?,?,'model',?,?,?,?,?,?)''',
                 (result_id, cut_id, kind, title, body, model_id, job_id, prompt,
                  json.dumps({**settings, 'structured_type': result_type}, ensure_ascii=False),
                  json.dumps(context, ensure_ascii=False), current['id'] if current else None))
    if current:
        conn.execute('UPDATE analysis_results SET superseded_by=? WHERE id=?', (result_id, current['id']))
    return result_id
