"""Revision-safe storage for structured single-cut pipeline outputs."""
import json
import uuid
from . import db, worker

KINDS = {
    'initial_frame': ('first_frame', 'Initial frame description'),
    'video_analysis': ('cut_description', 'Video analysis'),
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


def list_results(cut_id, history=False):
    from . import frame_source
    where = '' if history else ' AND superseded_by IS NULL'
    rows = db.all_rows(f'SELECT * FROM analysis_results WHERE cut_id=?{where} ORDER BY rowid DESC', (cut_id,))
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
        frame_url = f"/api/analysis-results/{row['id']}/frame" if kind in ('initial_frame', 'last_frame') and frame_source.result_frame_spec(row['id']) else None
        results.append({'id': row['id'], 'cut_id': row['cut_id'], 'type': kind, 'content': content, 'frame_url': frame_url,
                        'origin': row['origin'], 'model_id': row['model_id'], 'job_id': row['job_id'],
                        'parent_id': row['parent_id'], 'superseded_by': row['superseded_by'], 'created_at': row['created_at']})
    return results


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
