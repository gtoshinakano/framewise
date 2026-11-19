"""Episode briefs and leased asset exports."""
import json
import math
import os
import tempfile
import uuid
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from . import db, media, frame_source, structured_results


def make_brief(episode_id, first, last, conn=None):
    if conn is None:
        with db.connect() as conn:
            conn.execute('BEGIN')
            return make_brief(episode_id, first, last, conn)
    episode = db.one('SELECT * FROM episodes WHERE id=?', (episode_id,), conn=conn)
    if not episode:
        raise HTTPException(404, 'Episode not found')
    if first < 1 or last < first or last - first >= 500:
        raise HTTPException(422, 'Choose an inclusive range of at most 500 cuts')
    cuts = db.all_rows('SELECT * FROM cuts WHERE episode_id=? AND active=1 AND cut_index>=? AND cut_index<=? ORDER BY cut_index', (episode_id, first - 1, last - 1), conn=conn)
    if len(cuts) != last - first + 1:
        raise HTTPException(404, 'One or more requested active cuts do not exist')
    audio = media.episode_dir(episode['project_id'], episode_id) / 'audio.wav'
    result = []
    for cut in cuts:
        row = db.one("SELECT * FROM analysis_results WHERE cut_id=? AND title='Cut summary' AND superseded_by IS NULL ORDER BY rowid DESC LIMIT 1", (cut['id'],), conn=conn)
        try:
            sections = json.loads(row['context_json'] or '{}').get('brief_sections', {}) if row and row['origin'] == 'model' else {}
        except (ValueError, AttributeError, TypeError):
            sections = {}
        def section(key):
            value = sections.get(key)
            return value if isinstance(value, dict) and isinstance(value.get('content'), str) else None
        folder = media.episode_dir(episode['project_id'], episode_id)
        source_available = (folder / 'proxy.mp4').is_file() or (folder / f"original{episode['original_ext']}").is_file()
        thumbnail_available = (folder / 'thumbnails' / f"{cut['id']}_first.jpg").is_file()
        frame_url = f"/api/cuts/{cut['id']}/frame/first" if source_available or thumbnail_available else None
        transcript = db.all_rows('''SELECT id,start_time,end_time,text,speaker_label,character_context_id FROM transcript_segments
            WHERE episode_id=? AND superseded_by IS NULL AND deleted=0 AND start_time<? AND end_time>? ORDER BY start_time,id''', (episode_id, cut['end_time'], cut['start_time']), conn=conn)
        initial = next((item for item in structured_results.list_results(cut['id'], conn=conn) if item['type'] == 'initial_frame'), None)
        result.append({'id': cut['id'], 'cut_number': cut['cut_index'] + 1, 'start_time': cut['start_time'], 'end_time': cut['end_time'],
            'production_brief': {key: row[key] for key in ('id', 'content', 'origin', 'model_id', 'job_id', 'created_at')} if row else None,
            'initial_frame': initial,
            'suggested_frame_url': frame_url, 'audio_url': f"/api/v1/cuts/{cut['id']}/audio" if episode['has_audio'] and audio.is_file() else None,
            'dialogue': transcript, 'visible_characters': {'main': section('main_characters'), 'secondary': section('secondary_characters')},
            'camera': {'position': section('camera_position'), 'movement': section('camera_movement')}, 'setting': section('setting'),
            'action_beats': section('actions'), 'lighting': section('lighting'), 'composition': section('opening_composition')})
    return {'episode_id': episode_id, 'from': first, 'to': last, 'cuts': result}


def export_path(episode, job_id):
    return media.episode_dir(episode['project_id'], episode['id']) / 'exports' / (str(uuid.UUID(job_id)) + '.zip')


def _file_stamp(path):
    try:
        info = path.stat()
        return info.st_ino, info.st_size, info.st_mtime_ns
    except FileNotFoundError:
        return None


def gather(conn, job):
    from . import composition_brief, sequence_package
    episode_id = job['episode_id']
    request = json.loads(job['payload_json'])
    episode = db.one('SELECT id,project_id,original_ext FROM episodes WHERE id=?', (episode_id,), conn=conn)
    if not episode:
        return {'error': 'Episode not found'}
    folder = media.episode_dir(episode['project_id'], episode_id)
    try:
        if request.get('kind') == 'sequence':
            if 'sequence' in request:  # Older requests carry an already-captured package.
                payload = request
            else:
                document = sequence_package.document(request['sequence_id'], conn)
                if not document:
                    return {'error': 'Sequence was removed'}
                full, assets = document
                if not full.get('shots'):
                    return {'error': 'This sequence has no shot plan yet'}
                queued_at = datetime.fromisoformat(job['created_at']).replace(tzinfo=timezone.utc).isoformat(timespec='seconds')
                payload = {**request, 'contract': sequence_package.CONTRACT, 'sequence': full, 'assets': assets, 'queued_at': queued_at}
            cuts = payload['sequence'].get('cuts', [])
        else:
            payload = request if 'brief' in request else _brief_snapshot(conn, episode_id, request)
            cuts = payload['brief']['cuts']
    except (HTTPException, composition_brief.CompositionError) as exc:
        return {'error': exc.detail}
    return {'data': payload, 'cuts': cuts,
            'source': [_file_stamp(folder / f"original{episode['original_ext']}"), _file_stamp(folder / 'audio.wav')]}


def _brief_snapshot(conn, episode_id, request):
    first, last = request['from'], request['to']
    options = request.get('composition')
    if options:
        from . import composition_brief
        snapshot = composition_brief._make_composition_brief(conn, episode_id, first, last, **options)
    else:
        snapshot = make_brief(episode_id, first, last, conn)
    payload = {'brief': snapshot, 'first_frame_times': {}}
    assets = []
    for item in snapshot['cuts']:
        cut = db.one('SELECT id,cut_index,start_time,end_time,first_frame_time,last_frame_time FROM cuts WHERE id=?', (item['id'],), conn=conn)
        payload['first_frame_times'][cut['id']] = cut['first_frame_time']
        if not options:
            continue
        cut['cut_number'] = cut['cut_index'] + 1
        for result in structured_results.list_results(cut['id'], conn=conn):
            if result['type'] not in ('initial_frame', 'last_frame'):
                continue
            key = 'analysis_' + result['type'] + '_time'
            if key in cut:
                continue
            spec = frame_source.result_frame_spec(result['id'], conn)
            if spec and spec['time'] is not None and cut['start_time'] <= spec['time'] < cut['end_time']:
                cut[key] = spec['time']
        cut['has_speech'] = bool(conn.execute('''SELECT 1 FROM transcript_segments WHERE episode_id=? AND superseded_by IS NULL
            AND deleted=0 AND start_time<? AND end_time>? LIMIT 1''', (episode_id, cut['end_time'], cut['start_time'])).fetchone())
        assets.append(cut)
    if options:
        payload.update(composition=True, asset_cuts=assets, include_last_frame=request['include_last_frame'])
    return payload


def inputs(conn, job):
    from .job_inputs import fingerprint
    return fingerprint(job.get('gathered') or gather(conn, job))


def invalidate(conn, job):
    """An export is a standalone download with no downstream results."""


def write_audio(source_path, target, start, end):
    with wave.open(str(source_path), 'rb') as source, wave.open(str(target), 'wb') as out:
        rate = source.getframerate()
        left = min(source.getnframes(), max(0, math.floor(start * rate)))
        right = min(source.getnframes(), math.ceil(end * rate))
        if right <= left:
            raise ValueError('No audio samples available for this cut')
        out.setparams(source.getparams()); source.setpos(left)
        remaining = right - left
        while remaining:
            n = min(remaining, 65536)
            data = source.readframes(n)
            if not data:
                raise ValueError('Prepared audio ended unexpectedly')
            out.writeframesraw(data)
            remaining -= len(data) // (source.getnchannels() * source.getsampwidth())


def process_export(job, owner, episode):
    from . import job_inputs, worker
    with db.connect() as conn:
        conn.execute('BEGIN')
        live = job_inputs.JobInputs(conn, job, owner)
    if live.data.get('error'):
        raise ValueError(live.data['error'])
    payload = live.data['data']

    def publish(archive, target):
        with live.publish() as conn:
            conn.execute("UPDATE jobs SET status='done',stage='Complete',error=NULL,input_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                         (json.dumps({'fingerprint': live.stamp, 'cut_ids': [cut['id'] for cut in live.data['cuts']]}), job['id']))
            os.replace(archive, target)

    if payload.get('kind') == 'sequence':
        # A directed sequence is exported by the same worker thread and served by the same endpoints.
        from .sequence_export import process_export as export_sequence
        return export_sequence(job, owner, episode, payload, publish)
    target = export_path(episode, job['id'])
    target.parent.mkdir(parents=True, exist_ok=True)
    brief = payload['brief']
    with tempfile.TemporaryDirectory(prefix='framewise-export-', dir=target.parent) as temp:
        temp = Path(temp)
        archive = temp / 'export.zip'
        manifest = []
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for cut in payload.get('asset_cuts', brief['cuts']):
                worker.stage(job['id'], owner, episode['id'], f"Exporting cut {cut['cut_number']}")
                entry = {'cut_id': cut['id'], 'cut_number': cut['cut_number'], 'audio': None, 'frame': None,
                         'audio_seconds': None, 'has_speech': cut.get('has_speech', bool(cut.get('dialogue'))), 'errors': {}}
                base = f"cut-{cut['cut_number']:04d}-{cut['id']}"
                try:
                    if not cut.get('audio_url') and not payload.get('composition'):
                        raise ValueError('Prepared audio unavailable')
                    wav = temp / 'cut.wav'
                    write_audio(media.episode_dir(episode['project_id'], episode['id']) / 'audio.wav', wav, cut['start_time'], cut['end_time'])
                    entry['audio'] = base + '.wav'; z.write(wav, entry['audio'])
                    with wave.open(str(wav), 'rb') as audio:
                        entry['audio_seconds'] = audio.getnframes() / audio.getframerate()
                except (OSError, ValueError, wave.Error, EOFError) as exc:
                    entry['errors']['audio'] = str(exc)
                if payload.get('composition'):
                    folder = media.episode_dir(episode['project_id'], episode['id'])
                    source = folder / f"original{episode['original_ext']}"
                    for position in (('initial', 'last') if payload.get('include_last_frame') else ('initial',)):
                        key = 'frame' if position == 'initial' else 'last_frame'
                        try:
                            if not source.is_file(): raise ValueError('Original video unavailable')
                            image = temp / f'{position}.jpg'
                            analyzed_stamp = cut.get('analysis_initial_frame_time' if position == 'initial' else 'analysis_last_frame_time')
                            stamp = analyzed_stamp if analyzed_stamp is not None else cut.get('first_frame_time' if position == 'initial' else 'last_frame_time')
                            if stamp is None: stamp = cut['start_time'] if position == 'initial' else cut['end_time']
                            media.extract_cut_frame(source, image, stamp, cut['start_time'], cut['end_time'], width=None,
                                                    last=position == 'last' and analyzed_stamp is None, seek_tolerance=.000002)
                            entry[key] = f'{base}-{position}.jpg'; z.write(image, entry[key])
                        except Exception as exc:
                            entry['errors'][key] = str(exc)[:500]
                    manifest.append(entry)
                    continue
                try:
                    if not cut['suggested_frame_url']:
                        raise ValueError('Suggested frame unavailable')
                    image = temp / 'frame.jpg'
                    if cut['suggested_frame_url'].startswith('/api/analysis-results/'):
                        # Exports queued before the first-frame default retain their snapshot.
                        frame_id = cut['suggested_frame_url'].split('/')[-2]
                        spec = frame_source.result_frame_spec(frame_id)
                        if not spec:
                            raise ValueError('Saved frame unavailable')
                        if spec['legacy'].is_file():
                            image.write_bytes(spec['legacy'].read_bytes())
                        else:
                            media.extract_cut_frame(spec['source'], image, spec['time'], spec['start'], spec['end'], width=spec['width'], seek_tolerance=spec['seek_tolerance'])
                    else:
                        folder = media.episode_dir(episode['project_id'], episode['id'])
                        source = folder / f"original{episode['original_ext']}"
                        if not source.is_file():
                            raise ValueError('Original video unavailable')
                        timestamp = payload.get('first_frame_times', {}).get(cut['id'])
                        timestamp = timestamp if timestamp is not None else cut['start_time']
                        media.extract_cut_frame(source, image, timestamp, cut['start_time'], cut['end_time'], width=None, seek_tolerance=.000002)
                    entry['frame'] = base + '.jpg'; z.write(image, entry['frame'])
                except Exception as exc:
                    entry['errors']['frame'] = str(exc)[:500]
                manifest.append(entry)
            z.writestr('brief.json', json.dumps(brief, ensure_ascii=False, **({'separators': (',', ':')} if payload.get('composition') else {'indent': 2})))
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        publish(archive, target)
