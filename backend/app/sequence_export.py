"""ZIP download of a directed sequence for AI Director (contract ``sequence-v2``).

Runs on the export worker thread like a brief export (pipeline ``brief_export`` with
``payload.kind == 'sequence'``), so the same status and download endpoints serve it. The package and the
asset specs are read when the export starts (``sequence_package.document``); audio and frames are
cut from the prepared audio and the original video, at the original resolution, when the job runs. Every
file is listed in ``manifest.json`` with its size and sha256; a file that cannot be made is listed under
``errors`` instead, never replaced by something else.

Always: ``sequence.json`` (lean view), ``overview.json`` (digest), ``manifest.json``, ``contact-sheet.jpg``,
``sequence.srt`` (when the range has burned-in subtitle lines), and per shot ``shot-NN.wav`` (exact),
``shot-NN-ref.wav`` (only for shots shorter than 5 s: the original audio from the shot's start for the generated length,
silence only where the episode audio ends), ``shot-NN-initial.jpg``,
``shot-NN-last.jpg``, ``shot-NN-strip.jpg``, ``preview/shot-NN.jpg``, ``shot-NN-prompt.txt``,
``shot-NN-prompt-template.txt`` and ``shot-NN-still.txt``. On request (``include``): ``sequence.wav``,
``shot-NN-clip.mp4`` and ``references/`` (character crops, scene frames, voice samples).
"""
import hashlib
import json
import math
import tempfile
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import media, sequence_package as packages
from .brief import export_path, write_audio
from .sequence_package import CROP_MARGIN, crop_fractions  # noqa: F401  (kept here for the export's readers)

FRAME_WORKERS = 4
STRIP_TILE = (320, 180)
STRIP_COLUMNS = 5
CONTACT_TILE = (384, 216)
CONTACT_COLUMNS = 5
PREVIEW_WIDTH = 640
AUDIO_ERRORS = (OSError, ValueError, wave.Error, EOFError)


def seconds_of(path):
    with wave.open(str(path), 'rb') as audio:
        return round(audio.getnframes() / audio.getframerate(), 3)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def name(index, suffix):
    return f'shot-{index:02d}{suffix}'


def fmt_clock(seconds):
    whole = int(seconds)
    return f'{whole // 60}:{seconds - whole // 60 * 60:04.1f}'


def srt_time(seconds):
    millis = max(0, round(seconds * 1000))
    return f'{millis // 3600000:02d}:{millis // 60000 % 60:02d}:{millis // 1000 % 60:02d},{millis % 1000:03d}'


def srt(lines, start, duration):
    """Burned-in subtitle lines of the range on the sequence clock."""
    rows = []
    for line in lines:
        t0, t1 = max(0.0, line['start_time'] - start), min(duration, line['end_time'] - start)
        if t1 > t0:
            rows.append(f"{len(rows) + 1}\n{srt_time(t0)} --> {srt_time(t1)}\n{line['text']}\n")
    return '\n'.join(rows)


def strip_times(picture):
    """About one frame per second of the shot's own picture, at the middle of each second."""
    start, end = picture
    count = max(1, math.ceil(end - start - 1e-6))
    step = (end - start) / count
    return [start + (number + 0.5) * step for number in range(count)]


class Archive:
    """The ZIP being written: every file with its size and sha256, and every asset that could not be made."""

    def __init__(self, z):
        self.z, self.files, self.errors = z, [], []

    def add_file(self, path, target, kind, **meta):
        digest, size = hashlib.sha256(), 0
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
                size += len(block)
        self.z.write(path, target)
        self.files.append(packages.pack({'path': target, 'bytes': size, 'sha256': digest.hexdigest(), 'kind': kind, **meta}))
        return target

    def add_bytes(self, data, target, kind, **meta):
        data = data.encode() if isinstance(data, str) else data
        self.z.writestr(target, data)
        self.files.append(packages.pack({'path': target, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'kind': kind, **meta}))
        return target

    def error(self, asset, exc, shot=None, path=None):
        self.errors.append(packages.pack({'asset': asset, 'shot': shot, 'path': path, 'error': str(exc)[:500]}))


def process_export(job, owner, episode, payload, publish):
    from . import worker
    if payload.get('contract') != packages.CONTRACT:
        raise ValueError('This export was queued before sequence-v2; queue the export again')
    full, assets, include = payload['sequence'], payload['assets'], set(payload.get('include') or [])
    target = export_path(episode, job['id'])
    target.parent.mkdir(parents=True, exist_ok=True)
    folder = media.episode_dir(episode['project_id'], episode['id'])
    audio, source = folder / 'audio.wav', folder / f"original{episode['original_ext']}"
    shots = {shot['index']: shot for shot in full.get('shots') or []}
    with tempfile.TemporaryDirectory(prefix='framewise-export-', dir=target.parent) as temp:
        temp = Path(temp)
        archive = temp / 'export.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            out = Archive(z)
            out.add_bytes(dumps(packages.lean(full)), 'sequence.json', 'document', source='sequence-v2 lean view')
            out.add_bytes(dumps(packages.digest(full)), 'overview.json', 'digest', source='sequence-v2 digest')
            if 'sequence_audio' in include:
                try:
                    wav = temp / 'sequence.wav'
                    write_audio(audio, wav, full['start_time'], full['start_time'] + full['duration'])
                    out.add_file(wav, 'sequence.wav', 'sequence_audio', seconds=seconds_of(wav))
                except AUDIO_ERRORS as exc:
                    out.error('sequence_audio', exc, path='sequence.wav')
            initials = write_shots(job, owner, episode, worker, out, temp, assets['shots'], shots, audio, source, include)
            try:
                worker.stage(job['id'], owner, episode['id'], 'Exporting the contact sheet')
                sheet = temp / 'contact-sheet.jpg'
                media.tile_images([(initials.get(item['index']), f"{item['index']} · cut {item['cut']}{' (%d/%d)' % tuple(item['part']) if item.get('part') else ''} · {fmt_clock(item['at'])}")
                                   for item in assets['shots']], sheet, CONTACT_COLUMNS, CONTACT_TILE)
                out.add_file(sheet, 'contact-sheet.jpg', 'contact_sheet', source='initial frames of every shot')
            except Exception as exc:
                out.error('contact_sheet', exc, path='contact-sheet.jpg')
            if assets.get('subtitles'):
                text = srt(assets['subtitles'], full['start_time'], full['duration'])
                if text:
                    out.add_bytes(text, 'sequence.srt', 'subtitles', source=f"burned-in subtitles ({(full.get('subtitles') or {}).get('language') or 'language not set'}), sequence clock")
            references = write_references(job, owner, episode, worker, out, temp, assets) if 'references' in include else None
            manifest = packages.pack({'contract': packages.CONTRACT, 'sequence_id': full['id'], 'job_id': job['id'], 'queued_at': payload.get('queued_at'),
                                      'exported_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'sequence_updated_at': full.get('updated_at'),
                                      'include': sorted(include), 'files': out.files, 'references': references, 'errors': out.errors})
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
        publish(archive, target)


def reference_audio(audio, folder, spec, exact):
    """The original audio from the shot's start for ``generate_seconds``: the shot itself, then what follows it in the episode.

    Silence is added only where the episode audio ends first. Returns the file and its manifest fields.
    """
    continued = folder / 'continued.wav'
    write_audio(audio, continued, spec['start_time'], spec['start_time'] + spec['generate_seconds'])
    heard = seconds_of(continued)
    ref = continued
    if spec['generate_seconds'] - heard > 0.001:
        ref = folder / 'ref.wav'
        media.pad_wav(continued, ref, spec['generate_seconds'])
    total = seconds_of(ref)
    silence = round(total - heard, 3)
    source = f"original audio from the shot's start for generate_seconds: {name(spec['index'], '.wav')}, then the audio that follows it"
    return ref, {'audio_seconds': exact, 'continued_seconds': round(heard - exact, 3), 'silence_seconds': silence if silence > 0.001 else None, 'padded_seconds': total,
                 'source': source + ('; silence where the episode audio ends' if silence > 0.001 else '')}


def write_shots(job, owner, episode, worker, out, temp, specs, shots, audio, source, include):
    """Per shot audio, frames, strip, preview, texts and clip; returns the initial JPEG of each shot for the contact sheet."""
    initials = {}
    with ThreadPoolExecutor(FRAME_WORKERS) as pool:
        frames = {}
        for spec in specs:
            index, (start, end) = spec['index'], spec['picture']
            folder = temp / f'shot-{index:02d}'
            folder.mkdir()
            if not source.is_file():
                continue
            frames[index] = {
                'initial': pool.submit(media.extract_cut_frame, source, folder / 'initial.jpg', spec['frame_time'], spec['cut_start'], spec['cut_end'], width=None, seek_tolerance=.000002),
                'last': pool.submit(media.extract_cut_frame, source, folder / 'last.jpg', end, start, end, width=None, last=True, seek_tolerance=.000002),
                'strip': [(time, pool.submit(media.extract_cut_frame, source, folder / f'strip-{number:02d}.jpg', time, start, end, width=STRIP_TILE[0], seek_tolerance=.000002))
                          for number, time in enumerate(strip_times(spec['picture']))]}
        for spec in specs:
            index = spec['index']
            worker.stage(job['id'], owner, episode['id'], f'Exporting shot {index}')
            shot, folder = shots.get(index) or {}, temp / f'shot-{index:02d}'
            try:
                wav = folder / 'exact.wav'
                write_audio(audio, wav, spec['start_time'], spec['end_time'])
                exact = seconds_of(wav)
                out.add_file(wav, name(index, '.wav'), 'shot_audio', shot=index, seconds=exact, source='original audio, exact shot interval')
                if spec['generate_seconds'] - exact > 0.001:
                    ref, meta = reference_audio(audio, folder, spec, exact)
                    out.add_file(ref, name(index, '-ref.wav'), 'shot_audio_padded', shot=index, **meta)
            except AUDIO_ERRORS as exc:
                out.error('shot_audio', exc, shot=index, path=name(index, '.wav'))
            jobs = frames.get(index)
            if jobs is None:
                for asset, suffix in (('initial_frame', '-initial.jpg'), ('last_frame', '-last.jpg'), ('strip', '-strip.jpg')):
                    out.error(asset, 'Original video unavailable', shot=index, path=name(index, suffix))
            else:
                try:
                    time = jobs['initial'].result()
                    initials[index] = folder / 'initial.jpg'
                    out.add_file(initials[index], name(index, '-initial.jpg'), 'initial_frame', shot=index, time=time,
                                 source=f"original video at {time:.3f} s ({spec['description_source']} description)")
                    try:
                        preview = folder / 'preview.jpg'
                        media.scale_image(initials[index], preview, PREVIEW_WIDTH)
                        out.add_file(preview, f'preview/shot-{index:02d}.jpg', 'preview', shot=index, source=f"{name(index, '-initial.jpg')} scaled to {PREVIEW_WIDTH} px wide")
                    except Exception as exc:
                        out.error('preview', exc, shot=index, path=f'preview/shot-{index:02d}.jpg')
                except Exception as exc:
                    out.error('initial_frame', exc, shot=index, path=name(index, '-initial.jpg'))
                try:
                    time = jobs['last'].result()
                    out.add_file(folder / 'last.jpg', name(index, '-last.jpg'), 'last_frame', shot=index, time=time,
                                 source=f"original video, last frame of the shot's own picture ({spec['picture'][0]:.3f}-{spec['picture'][1]:.3f} s)")
                except Exception as exc:
                    out.error('last_frame', exc, shot=index, path=name(index, '-last.jpg'))
                try:
                    tiles = []
                    for number, (time, future) in enumerate(jobs['strip']):
                        try:
                            future.result()
                            tiles.append((folder / f'strip-{number:02d}.jpg', f"{time - spec['start_time']:.1f} s"))
                        except Exception:
                            tiles.append((None, f"{time - spec['start_time']:.1f} s"))
                    strip = folder / 'strip.jpg'
                    media.tile_images(tiles, strip, STRIP_COLUMNS, STRIP_TILE)
                    out.add_file(strip, name(index, '-strip.jpg'), 'strip', shot=index, times=[round(time - spec['start_time'], 3) for time, _ in jobs['strip']],
                                 source="original video about once per second of the shot's own picture; labels are seconds of the shot")
                except Exception as exc:
                    out.error('strip', exc, shot=index, path=name(index, '-strip.jpg'))
            for key, suffix, kind in (('prompt', '-prompt.txt', 'prompt'), ('prompt_template', '-prompt-template.txt', 'prompt_template'), ('still_prompt', '-still.txt', 'still_prompt')):
                if shot.get(key):
                    out.add_bytes(shot[key], name(index, suffix), kind, shot=index)
            if 'clips' in include:
                try:
                    if not source.is_file():
                        raise ValueError('Original video unavailable')
                    clip = folder / 'clip.mp4'
                    media.cut_clip(source, clip, *spec['picture'])
                    out.add_file(clip, name(index, '-clip.mp4'), 'clip', shot=index, seconds=round(spec['picture'][1] - spec['picture'][0], 3),
                                 source="original video, the shot's own picture, silent, at most 360 lines")
                except Exception as exc:
                    out.error('clip', exc, shot=index, path=name(index, '-clip.mp4'))
    return initials


def original_of(project_id, episode_id, ext):
    return media.episode_dir(project_id, episode_id) / f'original{ext}'


def write_references(job, owner, episode, worker, out, temp, assets):
    """Character crops, scene frames and voice samples, with every candidate and the one chosen."""
    worker.stage(job['id'], owner, episode['id'], 'Exporting references')
    folder = temp / 'references'
    folder.mkdir()
    listing = {'characters': {}, 'scenes': {}, 'voices': {}}
    for cid, item in (assets.get('characters') or {}).items():
        path = f'references/characters/{cid}.jpg'
        entry = {'name': item.get('name'), 'path': None, 'candidates': [packages.pack({key: candidate.get(key) for key in ('source', 'detection_id', 'episode_id', 'cut_number', 'time', 'box', 'image_size', 'area')})
                                                                       for candidate in item.get('candidates') or []]}
        listing['characters'][cid] = entry
        if not item.get('candidates'):
            out.error('character_reference', 'Framewise has no saved reference or manual label of this character', path=path)
            continue
        chosen = item['candidates'][0]
        entry['chosen'] = 0
        entry['reason'] = 'largest saved gallery reference' if chosen['source'] == 'saved_reference' else 'no saved gallery reference; largest manually labelled box in these cuts'
        try:
            original = original_of(chosen['project_id'], chosen['episode_id'], chosen['original_ext'])
            if not original.is_file():
                raise ValueError('Original video of that episode unavailable')
            frame, crop = folder / f'{cid}-frame.jpg', folder / f'{cid}.jpg'
            media.extract_cut_frame(original, frame, chosen['time'], chosen['cut_start'], chosen['cut_end'], width=None, seek_tolerance=.000002)
            media.crop_image(frame, crop, crop_fractions(chosen))
            entry['path'] = out.add_file(crop, path, 'character_reference', source=f"{chosen['source']} {chosen['detection_id']} (episode {chosen['episode_id']}, {chosen['time']:.3f} s)")
        except Exception as exc:
            out.error('character_reference', exc, path=path)
    for scene_id, item in (assets.get('scenes') or {}).items():
        path = f'references/scenes/{scene_id}.jpg'
        listing['scenes'][scene_id] = {'cut': item['cut'], 'time': item['time'], 'chosen_by': item['chosen_by'], 'path': None}
        try:
            original = original_of(episode['project_id'], episode['id'], episode['original_ext'])
            if not original.is_file():
                raise ValueError('Original video unavailable')
            frame = folder / f'scene-{scene_id}.jpg'
            media.extract_cut_frame(original, frame, item['time'], item['cut_start'], item['cut_end'], width=None, seek_tolerance=.000002)
            listing['scenes'][scene_id]['path'] = out.add_file(frame, path, 'scene_reference', source=f"first frame of cut {item['cut']}: {item['chosen_by']}")
        except Exception as exc:
            out.error('scene_reference', exc, path=path)
    project_id = assets['episode']['project_id']
    for cid, item in (assets.get('voices') or {}).items():
        samples = []
        for sample in item.get('samples') or []:
            path = f'references/voices/{cid}.wav' if sample.get('chosen') else f"references/voices/{cid}-{sample['id'][:8]}.wav"
            entry = packages.pack({key: sample.get(key) for key in ('id', 'episode_id', 'ranges', 'source_filename', 'seconds', 'preview_available', 'chosen', 'reason')})
            entry['preview_available'] = bool(sample.get('preview_available'))
            samples.append(entry)
            try:
                saved = media.voice_reference_path(project_id, sample['id'])
                wav = folder / f"voice-{sample['id']}.wav"
                if saved.is_file():
                    origin = 'saved voice sample'
                    wav = saved
                elif sample.get('episode_id') and sample.get('ranges'):
                    origin = "rebuilt from the episode audio over the sample's ranges"
                    media.write_wav_ranges(media.episode_dir(project_id, sample['episode_id']) / 'audio.wav', sample['ranges'], wav)
                else:
                    raise ValueError('The sample audio is unavailable')
                entry['path'] = out.add_file(wav, path, 'voice_reference', seconds=seconds_of(wav), source=origin)
            except AUDIO_ERRORS as exc:
                out.error('voice_reference', exc, path=path)
        listing['voices'][cid] = {'name': item.get('name'), 'samples': samples}
    return packages.pack({group: {key: packages.pack(value) for key, value in items.items()} for group, items in listing.items()})
