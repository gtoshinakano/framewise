"""Queue and inspect episode keyframe selection."""
import json
import subprocess
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from . import db, media, worker

router = APIRouter()

class Request(BaseModel):
    settings: dict = Field(default_factory=dict)

@router.post('/api/episodes/{episode_id}/keyframe-selection')
def enqueue_selection(episode_id: str, body: Request = Request()):
    from . import keyframes
    if not db.one('SELECT id FROM episodes WHERE id=?', (episode_id,)): raise HTTPException(404, 'Episode not found')
    try:
        settings = keyframes.config(body.settings)
        band = keyframes.mask_for(episode_id, settings)
        cuts = [dict(c) for c in db.all_rows('SELECT id,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time', (episode_id,))]
        if not cuts: raise ValueError('Detect hard cuts first')
        if any(c['end_time'] <= c['start_time'] or (i and c['start_time'] < cuts[i-1]['end_time']) for i,c in enumerate(cuts)): raise ValueError('Cuts must have non-overlapping positive ranges')
        return {'job_id': worker.enqueue(episode_id, 'keyframes', {'keyframe_settings': settings, 'subtitle_mask': band, 'cuts': cuts})}
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc

@router.get('/api/episodes/{episode_id}/keyframe-selection')
def get_selection(episode_id: str):
    from . import keyframes
    episode = db.one('SELECT * FROM episodes WHERE id=?', (episode_id,))
    if not episode: raise HTTPException(404, 'Episode not found')
    job = db.one("SELECT id,status,error,payload_json FROM jobs WHERE episode_id=? AND pipeline='keyframes' ORDER BY rowid DESC LIMIT 1", (episode_id,))
    if not job: return {'status': 'missing', 'results': []}
    payload = json.loads(job['payload_json']); settings = payload['keyframe_settings']
    band = keyframes.mask_for(episode_id, settings)
    source = media.episode_dir(episode['project_id'], episode_id) / f"original{episode['original_ext']}"
    results = []
    for cut in db.all_rows('SELECT id,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time', (episode_id,)):
        try: key = keyframes.cache_key(source, cut, settings, band)
        except OSError: key = ''
        row = db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (key,))
        results.append({'cut_id': cut['id'], 'status': 'current' if row else 'stale', 'result': json.loads(row['result_json']) if row else None})
    return {'job_id': job['id'], 'status': job['status'], 'error': job['error'], 'results': results}

@router.post('/api/cuts/{cut_id}/keyframe-selection')
def enqueue_cut_selection(cut_id: str, body: Request = Request()):
    from . import keyframes
    cut = db.one('SELECT id,episode_id,start_time,end_time FROM cuts WHERE id=? AND active=1', (cut_id,))
    if not cut: raise HTTPException(404, 'Active cut not found')
    try:
        settings = keyframes.config(body.settings)
        band = keyframes.mask_for(cut['episode_id'], settings)
        snapshot = {key: cut[key] for key in ('id', 'start_time', 'end_time')}
        return {'job_id': worker.enqueue(cut['episode_id'], 'keyframes', {'keyframe_settings': settings, 'subtitle_mask': band, 'cuts': [snapshot]})}
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc


def cut_selection(cut_id):
    from . import keyframes
    cut = db.one('''SELECT c.*,e.project_id,e.original_ext FROM cuts c JOIN episodes e ON e.id=c.episode_id
                    WHERE c.id=? AND c.active=1''', (cut_id,))
    if not cut: raise HTTPException(404, 'Active cut not found')
    job = db.one('''SELECT j.id,j.status,j.error,j.payload_json FROM jobs j WHERE j.episode_id=? AND j.pipeline='keyframes'
                    AND EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.cuts') item WHERE json_extract(item.value,'$.id')=?)
                    ORDER BY j.rowid DESC LIMIT 1''', (cut['episode_id'], cut_id))
    if not job: return cut, None, {'status': 'missing', 'result': None}
    payload = json.loads(job['payload_json']); settings = payload['keyframe_settings']
    band = keyframes.mask_for(cut['episode_id'], settings)
    source = media.episode_dir(cut['project_id'], cut['episode_id']) / f"original{cut['original_ext']}"
    try: key = keyframes.cache_key(source, cut, settings, band)
    except OSError: key = ''
    row = db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (key,))
    result = {'status': job['status'], 'job_id': job['id'], 'error': job['error'],
              'result_status': 'current' if row else 'stale', 'revision': key if row else None, 'result': json.loads(row['result_json']) if row else None}
    return cut, source, result

@router.get('/api/cuts/{cut_id}/keyframe-selection')
def get_cut_selection(cut_id: str):
    return cut_selection(cut_id)[2]

@router.get('/api/cuts/{cut_id}/keyframe-selection/frames/{ordinal}')
async def get_target_frame(cut_id: str, ordinal: int, revision: str | None = None):
    from fastapi.responses import Response
    from . import frame_cache
    cut, source, state = cut_selection(cut_id)
    result = state.get('result')
    frame = next((frame for frame in result['selected_frames'] if frame['source_frame_index'] == ordinal), None) if result else None
    if not frame or (revision is not None and revision != state.get('revision')):
        raise HTTPException(404, 'Target frame unavailable or stale')
    spec = {'source': source, 'time': frame['timestamp_seconds'], 'start': cut['start_time'], 'end': cut['end_time'],
            'width': 640, 'last': False, 'ordinal': ordinal}
    try:
        image = await frame_cache.get(spec)
    except (RuntimeError, OSError, TimeoutError, subprocess.SubprocessError) as exc:
        raise HTTPException(503, 'Could not extract target frame') from exc
    return Response(content=image, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=300' if revision else 'no-store'})


@router.get('/api/cuts/{cut_id}/keyframe-analysis')
def get_keyframe_analysis(cut_id: str):
    """The cut's current keyframe analysis for the Target frames tab: per-frame states, the changes between frames, its run and job.

    ``status`` is ``current`` when it was made for the cut's current keyframe selection, ``stale`` when the selection
    changed since, ``missing`` before the first analysis.
    """
    from . import keyframe_analysis
    if not db.one('SELECT id FROM cuts WHERE id=? AND active=1', (cut_id,)):
        raise HTTPException(404, 'Active cut not found')
    return keyframe_analysis.state(cut_id)
