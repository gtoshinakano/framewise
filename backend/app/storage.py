"""Storage inventory, run provenance, and conservative, revalidated cleanup."""
import json
import os
import re
import stat
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db, frame_source, frame_cache, media

router = APIRouter(prefix='/api', tags=['Storage'])


def _json(value, fallback):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return fallback


def _inventory(conn):
    projects = {row['id']: dict(row) for row in conn.execute('SELECT id,title FROM projects')}
    episodes = {row['id']: dict(row) for row in conn.execute('SELECT id,project_id,title,original_ext,status,has_audio FROM episodes')}
    active = {row['episode_id'] for row in conn.execute("SELECT DISTINCT episode_id FROM jobs WHERE status IN ('queued','running')")}
    busy_projects = {episodes[eid]['project_id'] for eid in active if eid in episodes}
    cuts = [dict(row) for row in conn.execute('SELECT id,episode_id,cut_index,active FROM cuts')]
    results = {row['id']: dict(row) for row in conn.execute('SELECT a.id,a.cut_id,a.job_id,a.title,a.origin,c.episode_id FROM analysis_results a JOIN cuts c ON c.id=a.cut_id')}
    samples = {row['id']: dict(row) for row in conn.execute('SELECT v.id,c.project_id FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id')}
    evidence = {}
    warnings = []
    for row in conn.execute('SELECT a.job_id,a.cut_id,a.evidence_json,j.episode_id,j.payload_json FROM analysis_job_cuts a JOIN jobs j ON j.id=a.job_id'):
        frames = _json(row['evidence_json'], None)
        payload = _json(row['payload_json'], {})
        if isinstance(payload, dict) and payload.get('workflow') == 'single_cut_v2' and isinstance(frames, dict):
            # V2 stores IDs of committed result revisions, not media filenames.
            from .structured_results import KINDS
            valid = all(key in KINDS and isinstance(value, str) and conn.execute('SELECT 1 FROM analysis_results WHERE id=? AND cut_id=?', (value, row['cut_id'])).fetchone() for key, value in frames.items())
            if valid:
                continue
            warnings.append(f"Run {row['job_id']} has an invalid structured checkpoint; its files are protected.")
            active.add(row['episode_id'])
            continue
        if not isinstance(frames, list):
            warnings.append(f"Run {row['job_id']} has unreadable saved frame evidence; its files are protected.")
            # A malformed reference could point to any file in this episode.
            active.add(row['episode_id'])
            continue
        for frame in frames:
            # Older runs stored descriptions without saving frame files.
            if isinstance(frame, dict) and 'frame_file' not in frame:
                continue
            name = frame.get('frame_file') if isinstance(frame, dict) else None
            if not isinstance(name, str) or not name or Path(name).name != name:
                warnings.append(f"Run {row['job_id']} has incomplete saved frame evidence; its files are protected.")
                active.add(row['episode_id'])
                continue
            evidence.setdefault((row['episode_id'], name), []).append({'job_id': row['job_id'], 'cut_id': row['cut_id']})

    evidence_names = {name for _, name in evidence}
    for table, column in [('episodes', 'settings_json'), ('cuts', 'results_json'), ('analysis_results', 'settings_json'), ('analysis_results', 'context_json'), ('jobs', 'payload_json')]:
        for row in conn.execute(f'SELECT id,{column} FROM {table}'):
            if not isinstance(_json(row[column], None), dict):
                warnings.append(f'Unreadable {column} in {table} record {row["id"]}; stored data has been retained.')
    orphan_results = [dict(row) for row in conn.execute("SELECT a.id,a.cut_id,a.title,a.kind,a.content,a.origin,a.job_id,a.created_at FROM analysis_results a LEFT JOIN cuts c ON c.id=a.cut_id WHERE c.id IS NULL")]
    orphan_result_ids = {row['id'] for row in orphan_results}
    orphan_evidence = set()
    orphan_evidence_records = []
    for row in conn.execute('SELECT a.job_id,a.cut_id,a.evidence_json FROM analysis_job_cuts a LEFT JOIN jobs j ON j.id=a.job_id WHERE j.id IS NULL'):
        frames = _json(row['evidence_json'], None)
        orphan_evidence_records.append({'job_id': row['job_id'], 'cut_id': row['cut_id'], 'frame_count': len(frames) if isinstance(frames, list) else None})
        if not isinstance(frames, list) or any(not isinstance(frame, dict) or ('frame_file' in frame and (not isinstance(frame['frame_file'], str) or not frame['frame_file'] or Path(frame['frame_file']).name != frame['frame_file'])) for frame in frames):
            warnings.append(f"Orphan evidence for missing run {row['job_id']} is unreadable; episode files are protected.")
            active.update(episodes)
        else:
            orphan_evidence.update(frame['frame_file'] for frame in frames if 'frame_file' in frame)
    orphan_transcripts = [dict(row) for row in conn.execute("SELECT t.id,t.episode_id,t.text,t.start_time,t.end_time,t.job_id,t.created_at FROM transcript_segments t LEFT JOIN episodes e ON e.id=t.episode_id WHERE e.id IS NULL")]
    for table, rowid, parent, _fkid in conn.execute('PRAGMA foreign_key_check'):
        warnings.append(f'Broken database link: {table} row {rowid} references missing {parent}.')

    base = db.DATA_DIR / 'projects'
    files = []
    found = set()
    if base.is_dir() and not base.is_symlink():
        for root, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
            for name in names:
                path = Path(root) / name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                rel = path.relative_to(db.DATA_DIR).as_posix()
                parts = path.relative_to(base).parts
                project_id = parts[0] if parts else None
                project = projects.get(project_id)
                episode_id = parts[2] if len(parts) >= 4 and parts[1] == 'episodes' else None
                episode = episodes.get(episode_id)
                if episode and episode['project_id'] != project_id:
                    episode = None
                sub = '/'.join(parts[3:]) if episode_id else ''
                kind, reason, deletable, links = 'Unknown file', 'Not recognized; inspect before removing outside Framewise.', False, []
                cleanup_candidate = False
                if episode:
                    found.add((episode_id, sub))
                    if sub == f"original{episode['original_ext']}":
                        kind, reason = 'Original video', 'Uploaded original; protected.'
                    elif sub in ('proxy.partial.mp4', 'audio.partial.wav') or (sub.startswith(('frames/', 'thumbnails/')) and len(Path(sub).parts) == 2 and '.partial.' in name):
                        kind, reason, deletable = 'Incomplete generated file', 'Leftover temporary file from media preparation.', True
                    elif sub in ('proxy.mp4', 'audio.wav', 'waveform.json') or (sub.startswith('frames/') and (name == 'manifest.json' or re.fullmatch(r'frame_\d{6}\.jpg', name)) and len(Path(sub).parts) == 2):
                        kind, reason, deletable = 'Prepared media', 'Used for playback or analysis. Deleting it makes that feature unavailable until Prepare is run again.', True
                        if sub.startswith('frames/') and name != 'manifest.json':
                            source = db.DATA_DIR / 'projects' / project_id / 'episodes' / episode_id / 'proxy.mp4'
                            if source.is_file():
                                kind, reason = 'Recreatable sampled frame', 'Recreated from the preview video when requested.'
                    elif sub.startswith('thumbnails/') and len(Path(sub).parts) == 2 and (re.fullmatch(r'cut_\d{6}\.jpg', name) or re.fullmatch(r'[0-9a-f-]{36}_(?:first|last)\.jpg', name)):
                        episode_base = db.DATA_DIR / 'projects' / project_id / 'episodes' / episode_id
                        can_recreate = (episode_base / 'proxy.mp4').is_file() or (episode_base / f"original{episode['original_ext']}").is_file()
                        kind, reason, deletable = ('Recreatable cut preview', 'Recreated from the source video when requested.', True) if can_recreate else ('Cut preview', 'Keep this preview: the source video is unavailable.', False)
                    elif sub.startswith('analysis/') and len(Path(sub).parts) == 2:
                        refs = evidence.get((episode_id, name), [])
                        result = results.get(Path(name).stem) if name.endswith('.jpg') else None
                        if result and result['episode_id'] == episode_id:
                            spec = frame_source.result_frame_spec(result['id'], conn)
                            if spec and 'source' in spec:
                                kind, reason, deletable = 'Recreatable result frame', f"Linked to result {result['title'] or result['id']}; recreated from the original video when requested.", True
                            else:
                                kind, reason = 'Result frame', f"Linked to result {result['title'] or result['id']}; no reconstructible source was found."
                            links = [{'result_id': result['id'], 'cut_id': result['cut_id'], 'job_id': result['job_id']}]
                        elif result:
                            kind, reason, links = 'Misplaced result frame', 'A result in another episode references this filename; inspect before removing.', [{'result_id': result['id'], 'cut_id': result['cut_id'], 'job_id': result['job_id']}]
                        elif name in evidence_names and not refs:
                            kind, reason = 'Misplaced run frame', 'A run in another episode references this filename; inspect before removing.'
                        elif Path(name).stem in orphan_result_ids:
                            kind, reason, links = 'Orphan result frame', 'Linked to an orphan result; inspect that record before deleting its frame.', [{'result_id': Path(name).stem}]
                        elif name in orphan_evidence:
                            kind, reason = 'Orphan run evidence', 'Referenced by evidence whose run is missing; database repair is required before cleanup.'
                        elif refs:
                            original = db.DATA_DIR / 'projects' / project_id / 'episodes' / episode_id / f"original{episode['original_ext']}"
                            if original.is_file():
                                kind, reason, deletable = 'Recreatable run frame', 'Run descriptions and timestamps remain in the database; a new frame can be extracted from the original.', True
                            else:
                                kind, reason = 'Saved run evidence', 'Referenced by a run; the original video is unavailable.'
                            links = refs
                        elif re.fullmatch(r'(?:[0-9a-f-]{36}|[0-9a-f-]{36}-[0-9a-f-]{36}-[0-9]+)\.jpg', name):
                            kind, reason, deletable = 'Unreferenced analysis frame', 'No result or saved run evidence references this frame.', True
                    elif sub.startswith('voice-uploads/') and len(Path(sub).parts) == 2:
                        try:
                            upload_id = str(uuid.UUID(name))
                        except ValueError:
                            upload_id = None
                        if upload_id == name and time.time() - info.st_mtime > 86400:
                            kind, reason, deletable = 'Old voice upload', 'Temporary upload older than one day, with no active episode jobs.', True
                        else:
                            kind, reason = 'Voice upload', 'Temporary input; retained while recent or an episode job is active.'
                    cleanup_candidate = deletable and kind not in ('Prepared media', 'Cut preview')
                    if episode_id in active:
                        if deletable:
                            reason += ' Cleanup is unavailable while jobs are active or saved evidence is unreadable.'
                        deletable = cleanup_candidate = False
                elif project and len(parts) == 3 and parts[1] == 'voice-references':
                    sample = samples.get(Path(name).stem) if name.endswith('.wav') else None
                    if sample and sample['project_id'] == project_id:
                        kind, reason = 'Voice reference', 'Linked to a character voice sample.'
                        links = [{'sample_id': sample['id']}]
                    else:
                        try:
                            valid_name = str(uuid.UUID(Path(name).stem)) == Path(name).stem and name.endswith('.wav')
                        except ValueError:
                            valid_name = False
                        kind, reason, deletable = 'Unlinked voice reference', 'No voice sample references this generated file.', valid_name and project_id not in busy_projects
                        if project_id in busy_projects:
                            reason += ' Cleanup is unavailable while project jobs are active.'
                elif not episode:
                    kind, reason = 'Unlinked file', 'No matching episode record; originals and unknown files are protected.'
                    # An abandoned episode directory can contain an original upload.
                    # Only recognize fixed generated names, and allow time for imports.
                    if episode_id and len(parts) >= 4 and time.time() - info.st_mtime > 86400:
                        generated = sub in ('proxy.mp4', 'proxy.partial.mp4', 'audio.wav', 'audio.partial.wav', 'waveform.json')
                        if len(Path(sub).parts) == 2:
                            generated = generated or bool(sub.startswith('analysis/') and re.fullmatch(r'(?:[0-9a-f-]{36}|[0-9a-f-]{36}-[0-9a-f-]{36}-[0-9]+)\.jpg', name))
                            generated = generated or bool(sub.startswith('frames/') and (name == 'manifest.json' or re.fullmatch(r'frame_\d{6}\.jpg', name)))
                            generated = generated or bool(sub.startswith('thumbnails/') and (re.fullmatch(r'cut_\d{6}\.jpg', name) or re.fullmatch(r'[0-9a-f-]{36}_(?:first|last)\.jpg', name)))
                        if generated and project_id not in busy_projects:
                            kind, reason, deletable = 'Unlinked generated file', 'Generated file in an episode directory with no database record, older than one day.', True
                if kind in ('Unlinked generated file', 'Unlinked voice reference'):
                    cleanup_candidate = deletable
                if episode and sub in ('waveform.json', 'frames/manifest.json'):
                    try:
                        metadata = json.loads(path.read_text())
                        key = 'peaks' if sub == 'waveform.json' else 'frames'
                        if not isinstance(metadata, dict) or not isinstance(metadata.get(key), list):
                            raise ValueError('unexpected structure')
                    except (OSError, ValueError, UnicodeError):
                        warnings.append(f'Unreadable {sub} in episode {episode_id}; run Prepare to rebuild it.')
                files.append({'path': rel, 'size': info.st_size, 'modified_ns': str(info.st_mtime_ns), 'project_id': project_id if project else None,
                              'project_title': project['title'] if project else None, 'episode_id': episode_id if episode else None,
                              'episode_title': episode['title'] if episode else None, 'kind': kind, 'reason': reason,
                              'deletable': deletable, 'cleanup_candidate': cleanup_candidate, 'links': links})
    missing = []
    for ep in episodes.values():
        expected = [(f"original{ep['original_ext']}", 'Original video')]
        if ep['status'] == 'ready':
            expected.extend([('proxy.mp4', 'Playback proxy'), ('waveform.json', 'Waveform')])
            if ep['has_audio']:
                expected.append(('audio.wav', 'Prepared audio'))
        for name, label in expected:
            if (ep['id'], name) not in found:
                missing.append({'episode_id': ep['id'], 'episode_title': ep['title'], 'project_id': ep['project_id'], 'file': name, 'description': label + ' is missing.'})
    # JPEGs are disposable when their source and recorded timestamp exist.
    for (episode_id, name), refs in evidence.items():
        if (episode_id, f'analysis/{name}') not in found:
            ep = episodes.get(episode_id, {})
            original = db.DATA_DIR / 'projects' / str(ep.get('project_id', '')) / 'episodes' / episode_id / f"original{ep.get('original_ext', '')}"
            if not original.is_file():
                missing.append({'episode_id': episode_id, 'episode_title': ep.get('title', 'Unknown episode'), 'project_id': ep.get('project_id'), 'file': f'analysis/{name}', 'description': 'Saved run frame and original video are unavailable.', 'links': refs})
    for result in results.values():
        if result['origin'] == 'model' and (result['title'].startswith('Frame at ') or result['title'].startswith('Object detections at ')) and not frame_source.result_frame_spec(result['id'], conn):
            ep = episodes.get(result['episode_id'], {})
            missing.append({'episode_id': result['episode_id'], 'episode_title': ep.get('title', 'Unknown episode'), 'project_id': ep.get('project_id'), 'file': f"analysis/{result['id']}.jpg", 'description': 'This result has no available source frame or original video.', 'links': [{'job_id': result['job_id'], 'cut_id': result['cut_id'], 'result_id': result['id']}]})
    return {'files': sorted(files, key=lambda f: f['path']), 'missing': missing, 'warnings': warnings,
            'orphan_results': orphan_results, 'orphan_transcripts': orphan_transcripts, 'orphan_evidence': orphan_evidence_records,
            'total_bytes': sum(f['size'] for f in files), 'reclaimable_bytes': sum(f['size'] for f in files if f['cleanup_candidate'])}


@router.get('/storage')
def inventory():
    with db.connect() as conn:
        return _inventory(conn)


class CleanupFile(BaseModel):
    path: str
    size: int = Field(ge=0)
    modified_ns: str = Field(pattern=r"^[0-9]+$")


class CleanupRequest(BaseModel):
    files: list[CleanupFile] = Field(min_length=1, max_length=1000)


def _unlink_checked(relative, expected):
    # Open each directory without following links, including after the inventory scan.
    parts = Path(relative).parts
    fd = os.open(db.DATA_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected['size'] or str(info.st_mtime_ns) != expected['modified_ns']:
            raise OSError('File changed after the inventory scan')
        os.unlink(parts[-1], dir_fd=fd)
    finally:
        os.close(fd)


def _recreation_spec(conn, item):
    """Resolve a generated JPEG without accepting a client-supplied path."""
    path = db.DATA_DIR / item['path']
    name = path.name
    ep = conn.execute('SELECT * FROM episodes WHERE id=?', (item['episode_id'],)).fetchone() if item['episode_id'] else None
    if not ep:
        return None
    base = db.DATA_DIR / 'projects' / ep['project_id'] / 'episodes' / ep['id']
    if item['kind'] == 'Recreatable result frame':
        return frame_source.result_frame_spec(path.stem, conn)
    if item['kind'] == 'Recreatable run frame':
        original = base / f"original{ep['original_ext']}"
        for link in item['links']:
            row = conn.execute('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (link['job_id'], link['cut_id'])).fetchone()
            frames = _json(row['evidence_json'], []) if row else []
            for frame in frames if isinstance(frames, list) else []:
                if isinstance(frame, dict) and frame.get('frame_file') == name:
                    try:
                        timestamp = float(frame['time'])
                        end = max(float(ep['duration'] or 0), timestamp + 1)
                        if timestamp >= 0 and timestamp < end:
                            return {'source': original, 'time': timestamp, 'start': 0, 'end': end, 'width': 1280, 'last': False, 'seek_tolerance': .000002}
                    except (KeyError, ValueError, TypeError):
                        pass
        return None
    if item['kind'] == 'Recreatable cut preview':
        match = re.fullmatch(r'([0-9a-f-]{36})_(first|last)\.jpg', name)
        if not match:
            # Gallery copies are disposable; their index can change on redetection.
            return None
        cut = conn.execute('SELECT * FROM cuts WHERE id=? AND episode_id=?', (match.group(1), ep['id'])).fetchone()
        if not cut:
            return None
        source = base / 'proxy.mp4'
        if not source.is_file():
            source = base / f"original{ep['original_ext']}"
        position = match.group(2)
        saved = cut['first_frame_time'] if position == 'first' else cut['last_frame_time']
        first, last = media.cut_frame_times(cut['start_time'], cut['end_time'])
        return {'source': source, 'time': saved if saved is not None else first if position == 'first' else last,
                'start': cut['start_time'], 'end': cut['end_time'], 'width': 480, 'last': position == 'last' and saved is None, 'seek_tolerance': .000002, 'probe_padding': 1 if position == 'last' and saved is None else 0}
    if item['kind'] == 'Recreatable sampled frame':
        try:
            manifest = _json((base / 'frames' / 'manifest.json').read_text(), {})
        except OSError:
            return None
        if not isinstance(manifest, dict) or not any(frame.get('name') == name for frame in manifest.get('frames', []) if isinstance(frame, dict)):
            return None
        try:
            boundary = min(ep['duration'], (int(name[6:12]) - .5) * float(manifest['interval']))
            if boundary > 0:
                return {'source': base / 'proxy.mp4', 'time': boundary, 'start': max(0, boundary - 5), 'end': boundary,
                        'width': 480, 'last': True, 'quality': 4, 'seek_tolerance': .000002, 'probe_padding': 1}
        except (KeyError, ValueError, TypeError):
            return None
    return None


@router.post('/storage/cleanup')
def cleanup(body: CleanupRequest):
    # Verify linked JPEGs outside the SQLite write lock: only delete them when
    # the generated replacement matches the saved evidence byte-for-byte.
    verified = {}
    with db.connect() as conn:
        available = {item['path']: item for item in _inventory(conn)['files']}
        requests = []
        for requested in body.files:
            item = available.get(requested.path)
            if not item or not item['deletable'] or item['size'] != requested.size or item['modified_ns'] != requested.modified_ns:
                raise HTTPException(409, f'{requested.path} changed or is no longer safe to remove. Refresh the inventory.')
            if requested.path in verified:
                raise HTTPException(400, 'A file was selected more than once')
            verified[requested.path] = None
            if item['kind'] in ('Recreatable result frame', 'Recreatable run frame', 'Recreatable cut preview', 'Recreatable sampled frame'):
                spec = _recreation_spec(conn, item)
                if spec is None:
                    if item['kind'] == 'Recreatable cut preview' and re.fullmatch(r'cut_\d{6}\.jpg', Path(item['path']).name):
                        continue
                    raise HTTPException(409, f"Cannot verify a replacement for {requested.path}; it was retained.")
                try:
                    source = Path(spec['source'])
                    source_stat = source.stat()
                except OSError as exc:
                    raise HTTPException(409, f"Source unavailable for {requested.path}; it was retained.") from exc
                requests.append((requested.path, spec, (source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)))
        for index in range(0, len(requests), 32):
            batch = requests[index:index + 32]
            try:
                futures = [(relative, spec, source_stat, frame_cache.request(spec)) for relative, spec, source_stat in batch]
                for relative, spec, source_stat, future in futures:
                    data = future.result(timeout=120)
                    if (db.DATA_DIR / relative).read_bytes() != data:
                        raise HTTPException(409, f"Replacement for {relative} differs from the saved frame; it was retained.")
                    verified[relative] = (Path(spec['source']), source_stat)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(409, f'Could not verify a replacement; no selected files were deleted: {exc}') from exc
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        available = {item['path']: item for item in _inventory(conn)['files']}
        for requested in body.files:
            item = available.get(requested.path)
            if not item or not item['deletable'] or item['size'] != requested.size or item['modified_ns'] != requested.modified_ns:
                raise HTTPException(409, f'{requested.path} changed or is no longer safe to remove. Refresh the inventory.')
            source = verified[requested.path]
            if source:
                try:
                    stat = source[0].stat()
                    if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != source[1]:
                        raise OSError('source changed')
                except OSError as exc:
                    raise HTTPException(409, f'The source for {requested.path} changed; refresh the inventory.') from exc
        removed, errors = [], []
        for relative in verified:
            try:
                _unlink_checked(relative, available[relative])
                removed.append(relative)
            except OSError as exc:
                errors.append({'path': relative, 'error': str(exc)})
    return {'removed': removed, 'errors': errors}


@router.get('/jobs/{job_id}/outputs')
def job_outputs(job_id: str):
    with db.connect() as conn:
        job = conn.execute('SELECT id,episode_id,payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, 'Run not found')
        results = [dict(row) for row in conn.execute('SELECT a.id,a.cut_id,a.title,a.kind,a.origin,a.model_id,a.parent_id,a.superseded_by,c.cut_index,c.active FROM analysis_results a JOIN cuts c ON c.id=a.cut_id WHERE a.job_id=? ORDER BY a.rowid', (job_id,))]
        segments = [dict(row) for row in conn.execute('SELECT id,start_time,end_time,text,speaker_label,model_id,parent_id,superseded_by,deleted FROM transcript_segments WHERE job_id=? ORDER BY start_time', (job_id,))]
        voices = [dict(row) for row in conn.execute('SELECT v.id,v.start_time,v.end_time,v.model_id,c.title AS character_name FROM character_voice_samples v JOIN context_items c ON c.id=v.character_context_id WHERE v.id=?', (job_id,))]
        evidence = [dict(row) for row in conn.execute('SELECT cut_id,completed,evidence_json FROM analysis_job_cuts WHERE job_id=?', (job_id,))]
        for item in evidence:
            saved = _json(item.pop('evidence_json'), None)
            item['frame_count'] = len(saved) if isinstance(saved, list) else None
            payload = _json(job['payload_json'], {})
            item['checkpoint_count'] = len(saved) if isinstance(saved, dict) and isinstance(payload, dict) and payload.get('workflow') == 'single_cut_v2' else None
        rejected = [dict(row) for row in conn.execute('SELECT id,cut_id,window_index,start_time,end_time,error,response,created_at FROM rejected_video_responses WHERE job_id=? ORDER BY id', (job_id,))]
        return {'results': results, 'segments': segments, 'evidence': evidence, 'voices': voices, 'rejected_video_responses': rejected}


@router.delete('/jobs/{job_id}')
def delete_job(job_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        job = conn.execute('SELECT status,episode_id FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, 'Run not found')
        if job['status'] in ('queued', 'running'):
            raise HTTPException(409, 'A queued or running job cannot be deleted')
        if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (job['episode_id'],)).fetchone():
            raise HTTPException(409, 'Wait for episode jobs to finish before deleting run history')
        conn.execute('DELETE FROM jobs WHERE id=?', (job_id,))
    return {'ok': True}


@router.delete('/storage/orphans/{kind}/{record_id}')
def delete_orphan(kind: str, record_id: str):
    if kind not in ('result', 'transcript'):
        raise HTTPException(404, 'Unknown record type')
    table, owner_table, owner_column = ('analysis_results', 'cuts', 'cut_id') if kind == 'result' else ('transcript_segments', 'episodes', 'episode_id')
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute(f'SELECT a.id FROM {table} a LEFT JOIN {owner_table} o ON o.id=a.{owner_column} WHERE a.id=? AND o.id IS NULL', (record_id,)).fetchone()
        if not row:
            raise HTTPException(409, 'Record is no longer orphaned or does not exist')
        # Delete the connected orphan revision chain together. Never alter a
        # revision which still belongs to a valid cut or episode.
        pending, chain = [record_id], set()
        while pending:
            current = pending.pop()
            if current in chain:
                continue
            linked = conn.execute(f'SELECT a.id,a.parent_id,a.superseded_by,o.id AS owner FROM {table} a LEFT JOIN {owner_table} o ON o.id=a.{owner_column} WHERE a.id=? OR a.parent_id=? OR a.superseded_by=?', (current, current, current)).fetchall()
            for item in linked:
                if item['owner'] is not None:
                    raise HTTPException(409, 'An existing record links to this orphan chain; repair the database before deleting it')
                if item['id'] not in chain:
                    pending.append(item['id'])
                for ref in (item['parent_id'], item['superseded_by']):
                    if ref and ref not in chain:
                        pending.append(ref)
            chain.add(current)
        ids = list(chain)
        placeholders = ','.join('?' for _ in ids)
        if conn.execute(f"SELECT 1 FROM {table} a JOIN jobs j ON j.id=a.job_id WHERE a.id IN ({placeholders}) AND j.status IN ('queued','running')", ids).fetchone():
            raise HTTPException(409, 'A run linked to this data is still active')
        conn.execute(f'DELETE FROM {table} WHERE id IN ({placeholders})', ids)
    return {'ok': True}


@router.delete('/jobs/{job_id}/evidence/{cut_id}')
def delete_job_evidence(job_id: str, cut_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        job = conn.execute('SELECT episode_id,payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, 'Run not found')
        if conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (job['episode_id'],)).fetchone():
            raise HTTPException(409, 'Wait for episode jobs to finish before deleting saved evidence')
        checkpoint = conn.execute('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job_id, cut_id)).fetchone()
        payload = _json(job['payload_json'], {})
        if checkpoint and isinstance(payload, dict) and payload.get('workflow') == 'single_cut_v2' and isinstance(_json(checkpoint['evidence_json'], None), dict):
            raise HTTPException(409, 'Structured checkpoints are required to resume this run and cannot be deleted separately')
        if conn.execute('SELECT 1 FROM cuts WHERE id=?', (cut_id,)).fetchone():
            changed = conn.execute("UPDATE analysis_job_cuts SET evidence_json='[]' WHERE job_id=? AND cut_id=?", (job_id, cut_id)).rowcount
        else:
            changed = conn.execute('DELETE FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job_id, cut_id)).rowcount
        if not changed:
            raise HTTPException(404, 'Saved evidence not found')
    return {'ok': True}


@router.delete('/storage/orphan-evidence/{job_id}/{cut_id}')
def delete_orphan_evidence(job_id: str, cut_id: str):
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT a.job_id,c.episode_id FROM analysis_job_cuts a LEFT JOIN jobs j ON j.id=a.job_id LEFT JOIN cuts c ON c.id=a.cut_id WHERE a.job_id=? AND a.cut_id=? AND j.id IS NULL', (job_id, cut_id)).fetchone()
        if not row:
            raise HTTPException(409, 'Evidence is no longer orphaned or does not exist')
        if row['episode_id'] and conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running')", (row['episode_id'],)).fetchone():
            raise HTTPException(409, 'Wait for episode jobs to finish before deleting orphan evidence')
        conn.execute('DELETE FROM analysis_job_cuts WHERE job_id=? AND cut_id=?', (job_id, cut_id))
    return {'ok': True}
