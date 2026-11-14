"""ZIP download of a directed sequence for the video generation app.

Runs on the export worker thread like a brief export (pipeline ``brief_export`` with
``payload.kind == 'sequence'``), so the same status and download endpoints serve it. The package is
snapshotted when the export is queued; audio and frames are cut from the prepared audio and the
original video when the job runs. Files:

- ``sequence.json``: the ``sequence-v1`` package.
- ``manifest.json``: per shot, the files written and any extraction error.
- ``sequence.wav``: the original audio of the whole sequence.
- ``shot-NN.wav``, ``shot-NN-initial.jpg``, ``shot-NN-prompt.txt``, ``shot-NN-still.txt``.
"""
import json
import os
import tempfile
import wave
import zipfile
from pathlib import Path

from . import db, media
from .brief import export_path, write_audio


def seconds_of(path):
    with wave.open(str(path), 'rb') as audio:
        return audio.getnframes() / audio.getframerate()


def process_export(job, owner, episode, payload):
    from . import worker
    package = payload['sequence']
    target = export_path(episode, job['id'])
    target.parent.mkdir(parents=True, exist_ok=True)
    folder = media.episode_dir(episode['project_id'], episode['id'])
    audio, source = folder / 'audio.wav', folder / f"original{episode['original_ext']}"
    cuts = {cut['id']: cut for cut in package['cuts']}
    with tempfile.TemporaryDirectory(prefix='framewise-export-', dir=target.parent) as temp:
        temp = Path(temp)
        archive = temp / 'export.zip'
        manifest = {'sequence_id': package['id'], 'audio': None, 'audio_seconds': None, 'errors': {}, 'shots': []}
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            try:
                wav = temp / 'sequence.wav'
                write_audio(audio, wav, package['start_time'], package['end_time'])
                manifest.update(audio='sequence.wav', audio_seconds=seconds_of(wav))
                z.write(wav, 'sequence.wav')
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                manifest['errors']['audio'] = str(exc)[:500]
            for shot in package['shots']:
                worker.stage(job['id'], owner, episode['id'], f"Exporting shot {shot['index']}")
                base = f"shot-{shot['index']:02d}"
                entry = {'index': shot['index'], 'key': shot['key'], 'audio': None, 'audio_seconds': None, 'frame': None, 'prompt': None, 'still_prompt': None,
                         'has_speech': shot['audio']['has_speech'], 'errors': {}}
                try:
                    wav = temp / 'shot.wav'
                    write_audio(audio, wav, shot['start_time'], shot['end_time'])
                    entry.update(audio=base + '.wav', audio_seconds=seconds_of(wav))
                    z.write(wav, entry['audio'])
                except (OSError, ValueError, wave.Error, EOFError) as exc:
                    entry['errors']['audio'] = str(exc)[:500]
                try:
                    if not source.is_file():
                        raise ValueError('Original video unavailable')
                    cut = cuts[shot['initial_frame']['cut_id']]
                    image = temp / 'frame.jpg'
                    media.extract_cut_frame(source, image, shot['initial_frame']['time'], cut['start_time'], cut['end_time'], width=None, seek_tolerance=.000002)
                    entry['frame'] = base + '-initial.jpg'
                    z.write(image, entry['frame'])
                except Exception as exc:
                    entry['errors']['frame'] = str(exc)[:500]
                if shot.get('prompt'):
                    entry['prompt'] = base + '-prompt.txt'
                    z.writestr(entry['prompt'], shot['prompt'])
                if shot['initial_frame'].get('still_prompt'):
                    entry['still_prompt'] = base + '-still.txt'
                    z.writestr(entry['still_prompt'], shot['initial_frame']['still_prompt'])
                manifest['shots'].append(entry)
            z.writestr('sequence.json', json.dumps(package, ensure_ascii=False, indent=2))
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        with db.connect() as conn:
            worker.require_lease(conn, job['id'], owner)
            os.replace(archive, target)
