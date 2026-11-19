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
        return {'job_id': worker.enqueue(episode_id, 'keyframes', {'keyframe_settings': settings, 'all_cuts': True})}
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
    with db.connect() as conn:
        conn.execute('BEGIN')
        for cut in db.all_rows('SELECT id,start_time,end_time FROM cuts WHERE episode_id=? AND active=1 ORDER BY start_time', (episode_id,), conn=conn):
            try: key = keyframes.cache_key(source, cut, settings, band)
            except OSError: key = ''
            row = db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (key,), conn=conn)
            result = keyframes.apply_edits(key, json.loads(row['result_json']), keyframes.edits_for(cut, conn))[1] if row else None
            results.append({'cut_id': cut['id'], 'status': 'current' if row else 'stale', 'result': result})
    return {'job_id': job['id'], 'status': job['status'], 'error': job['error'], 'results': results}

@router.post('/api/cuts/{cut_id}/keyframe-selection')
def enqueue_cut_selection(cut_id: str, body: Request = Request()):
    from . import keyframes
    cut = db.one('SELECT id,episode_id,start_time,end_time FROM cuts WHERE id=? AND active=1', (cut_id,))
    if not cut: raise HTTPException(404, 'Active cut not found')
    try:
        settings = keyframes.config(body.settings)
        return {'job_id': worker.enqueue(cut['episode_id'], 'keyframes', {'keyframe_settings': settings, 'cut_ids': [cut_id]})}
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc


def cut_selection(cut_id):
    """``(cut, source, state)``: the cut's current selection with the user's added and removed frames."""
    from . import keyframes
    with db.connect() as conn:
        conn.execute('BEGIN')
        cut, source, key, result = keyframes.current_selection(cut_id, conn)
        if not cut: raise HTTPException(404, 'Active cut not found')
        job = keyframes.latest_job(conn, cut['episode_id'], cut_id)
    if not job: return cut, None, {'status': 'missing', 'result': None}
    state = {'status': job['status'], 'job_id': job['id'], 'error': job['error'], 'max_frames': keyframes.MAX_FRAMES,
             'result_status': 'current' if key else 'stale', 'revision': key, 'result': result}
    return cut, source, state

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


def _frames(state):
    return sorted((state.get('result') or {}).get('selected_frames') or [], key=lambda frame: frame['source_frame_index'])


@router.get('/api/cuts/{cut_id}/keyframe-selection/between')
def get_between(cut_id: str, after: int):
    """Up to four frames evenly spaced between the selected frame ``after`` and the next one, to add by hand."""
    from . import keyframes
    cut, source, state = cut_selection(cut_id)
    frames = _frames(state)
    start = next((frame for frame in frames if frame['source_frame_index'] == after), None)
    if not start: raise HTTPException(404, 'Target frame unavailable or stale')
    _, end = keyframes.neighbours(frames, after)
    try:
        found = keyframes.frames_between(source, start, end, cut)
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc
    except (RuntimeError, OSError) as exc: raise HTTPException(503, 'Could not read the frames of this cut') from exc
    return {'after': after, 'next': end['source_frame_index'] if end else None, 'revision': state['revision'], 'available': len(found),
            'frames': [{'ordinal': ordinal, 'timestamp_seconds': t, 'relative_timestamp_seconds': t - cut['start_time']}
                       for ordinal, t in keyframes.spread(found)]}


@router.get('/api/cuts/{cut_id}/keyframe-selection/candidates/{ordinal}')
async def get_candidate_frame(cut_id: str, ordinal: int, time: float):
    """A small picture of a frame offered between two selected frames."""
    from fastapi.responses import Response
    from . import frame_cache
    cut, source, state = cut_selection(cut_id)
    if not source or not cut['start_time'] <= time < cut['end_time']:
        raise HTTPException(404, 'Frame outside this cut')
    spec = {'source': source, 'time': time, 'start': cut['start_time'], 'end': cut['end_time'], 'width': 320, 'last': False, 'ordinal': ordinal}
    try:
        image = await frame_cache.get(spec)
    except (RuntimeError, OSError, TimeoutError, subprocess.SubprocessError) as exc:
        raise HTTPException(503, 'Could not extract the frame') from exc
    return Response(content=image, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=300'})


class FrameEdit(BaseModel):
    ordinal: int


def _edit(cut_id, **change):
    from . import keyframes
    try:
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            keyframes.edit_frames(conn, cut_id, **change)
    except LookupError as exc: raise HTTPException(404, str(exc)) from exc
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc
    except (RuntimeError, OSError) as exc: raise HTTPException(503, 'Could not read the frames of this cut') from exc
    return cut_selection(cut_id)[2]


@router.post('/api/cuts/{cut_id}/keyframe-selection/frames')
def add_frame(cut_id: str, body: FrameEdit):
    """Add a frame of the cut to its selection. The keyframe analysis and what is built on it turn stale."""
    return _edit(cut_id, add=body.ordinal)


@router.delete('/api/cuts/{cut_id}/keyframe-selection/frames/{ordinal}')
def remove_frame(cut_id: str, ordinal: int):
    """Remove a frame (never the first) from the cut's selection, whether the selector or the user chose it."""
    return _edit(cut_id, remove=ordinal)


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
