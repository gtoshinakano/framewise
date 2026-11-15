"""The download of a directed sequence: the sequence-v2 ZIP, its manifest and assets, the audio routes and the ffmpeg helpers."""
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import tests.test_sequence_pipeline as pipeline_tests
from app import db, media, sequence_export, worker


def write_audio(path, seconds=10, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(rate)
        out.writeframes(b'\x01\x00' * int(seconds * rate))


def wav_seconds(data):
    with wave.open(io.BytesIO(data)) as audio:
        return audio.getnframes() / audio.getframerate()


SHOT_FILES = ('.wav', '-initial.jpg', '-last.jpg', '-strip.jpg', '-prompt.txt', '-prompt-template.txt', '-still.txt')


class SequenceExportTest(pipeline_tests.SequenceBase):
    def setUp(self):
        super().setUp()
        self.folder = media.episode_dir('p', 'e')
        write_audio(self.folder / 'audio.wav')
        (self.folder / 'original.mp4').write_bytes(b'video')
        self.frames, self.tiles, self.crops, self.clips = [], [], [], []

    def subtitles(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,status,language) VALUES ('r','e','complete','Japanese')")
            conn.executemany("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES (?,'r','e',?,?,?,?,'high')",
                             [('s1', 0, 0.3, 1.5, 'こんにちは', ), ('s2', 1, 3.8, 4.6, 'またね')])

    def export(self, sequence_id, include='', expect='done'):
        queued = self.client.post(f'/api/v1/sequences/{sequence_id}/exports' + (f'?include={include}' if include else ''))
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id = queued.json()['job_id']

        def extract(source, target, timestamp, start, end, **kwargs):
            self.frames.append((source.name, round(timestamp, 3), start, end, kwargs.get('width'), kwargs.get('last', False)))
            target.write_bytes(f'jpeg {timestamp:.3f}'.encode())
            return timestamp

        def tile(items, output, columns, size=None):
            self.tiles.append(([label for _, label in items], [path is not None for path, _ in items], columns))
            output.write_bytes(b'tiles')

        def crop(source, output, box):
            self.crops.append(tuple(round(value, 4) for value in box))
            output.write_bytes(b'crop')

        def clip(source, output, start, end, height=360):
            self.clips.append((start, end))
            output.write_bytes(b'mp4')
        with patch.object(media, 'extract_cut_frame', side_effect=extract), patch.object(media, 'tile_images', side_effect=tile), \
                patch.object(media, 'scale_image', side_effect=lambda source, output, width: output.write_bytes(b'small')), \
                patch.object(media, 'crop_image', side_effect=crop), patch.object(media, 'cut_clip', side_effect=clip):
            worker.process(*worker.claim_job(('brief_export',)))
        status = self.client.get(f'/api/v1/brief-exports/{job_id}').json()
        self.assertEqual(status['status'], expect, status)
        if expect != 'done':
            return status
        download = self.client.get(status['download_url'])
        self.assertIn(f'framewise-sequence-{job_id}.zip', download.headers['content-disposition'])
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            files = {item: archive.read(item) for item in archive.namelist()}
        manifest = json.loads(files['manifest.json'])
        # Every file is listed with its true size and hash, and nothing else is in the ZIP.
        self.assertEqual({item['path'] for item in manifest['files']}, set(files) - {'manifest.json'})
        for item in manifest['files']:
            self.assertEqual((item['bytes'], item['sha256']), (len(files[item['path']]), hashlib.sha256(files[item['path']]).hexdigest()), item['path'])
        self.assertEqual((manifest['contract'], manifest['sequence_id'], manifest['job_id']), ('sequence-v2', sequence_id, job_id))
        self.assertTrue(manifest['queued_at'] and manifest['exported_at'] and manifest['sequence_updated_at'])
        return files, manifest

    def test_export_zip_holds_documents_audio_frames_strips_and_prompts(self):
        self.subtitles()
        sequence_id = self.written()
        files, manifest = self.export(sequence_id)
        expected = {'sequence.json', 'overview.json', 'manifest.json', 'contact-sheet.jpg', 'sequence.srt',
                    *(f'shot-0{n}{suffix}' for n in (1, 2, 3) for suffix in SHOT_FILES), *(f'preview/shot-0{n}.jpg' for n in (1, 2, 3)),
                    'shot-01-ref.wav', 'shot-02-ref.wav'}  # only shots shorter than 5 s get a padded copy
        self.assertEqual(set(files), expected)
        self.assertNotIn('errors', manifest)
        # The ZIP's documents are the lean view and the digest, snapshotted when the export was queued.
        lean, digest = json.loads(files['sequence.json']), json.loads(files['overview.json'])
        self.assertEqual((lean['contract'], lean['view'], digest['view']), ('sequence-v2', 'lean', 'digest'))
        self.assertNotIn('prompt', lean['shots'][0])
        self.assertEqual(lean, self.client.get(f'/api/v1/sequences/{sequence_id}').json() | {'updated_at': lean['updated_at']})
        self.assertNotIn(b'\n ', files['sequence.json'])  # compact JSON
        full = self.client.get(f'/api/v1/sequences/{sequence_id}?view=full').json()
        self.assertEqual(files['shot-01-prompt.txt'].decode(), full['shots'][0]['prompt'])
        self.assertEqual(files['shot-01-prompt-template.txt'].decode(), full['shots'][0]['prompt_template'])
        self.assertIn('<d>[{language}] {line 1.1}</d>', full['shots'][0]['prompt_template'])
        # Exact and padded audio, and which is which.
        by_path = {item['path']: item for item in manifest['files']}
        self.assertEqual((wav_seconds(files['shot-01.wav']), wav_seconds(files['shot-01-ref.wav']), wav_seconds(files['shot-03.wav'])), (2.0, 5.0, 6.0))
        self.assertEqual((by_path['shot-01.wav']['seconds'], by_path['shot-01-ref.wav']['audio_seconds'], by_path['shot-01-ref.wav']['padded_seconds']), (2.0, 2.0, 5.0))
        self.assertEqual(by_path['shot-02-initial.jpg']['kind'], 'initial_frame')
        # Initial frames at the original resolution, the last frame of each shot's own picture, strips about once a second.
        self.assertIn(('original.mp4', 0.0, 0.0, 2.0, None, False), self.frames)
        self.assertIn(('original.mp4', 2.0, 0.0, 2.0, None, True), self.frames)
        self.assertEqual([call[1] for call in self.frames if call[4] == 320 and call[2] == 4.0], [4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        strip = next(labels for labels, _, columns in self.tiles if labels[0] == '0.5 s' and len(labels) == 6)
        self.assertEqual(strip, ['0.5 s', '1.5 s', '2.5 s', '3.5 s', '4.5 s', '5.5 s'])
        sheet = self.tiles[-1]
        self.assertEqual((sheet[0], sheet[1]), (['1 · cut 1 · 0:00.0', '2 · cut 2 · 0:02.0', '3 · cut 3 · 0:04.0'], [True, True, True]))
        self.assertEqual(files['sequence.srt'].decode(), '1\n00:00:00,300 --> 00:00:01,500\nこんにちは\n\n2\n00:00:03,800 --> 00:00:04,600\nまたね\n')
        self.assertEqual(lean['shots'][1]['subtitles'], [{'t0': 1.8, 't1': 2.0, 'text': 'またね', 'crosses_shot': True}])
        self.assertEqual(lean['subtitles'], {'language': 'Japanese', 'extracted': True})
        # The export is a snapshot: sequence jobs and episode status are untouched by it.
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")['status'], 'ready')

    def test_a_shot_that_runs_past_its_cut_never_shows_the_next_cut(self):
        self.interrupt()  # L3 runs across the hard cut between cut 2 and cut 3
        sequence_id = self.directed()
        self.assertEqual(self.client.put(f'/api/sequences/{sequence_id}/boundaries', json={'after_cut_id': 'c2', 'choice': 'after_line'}).status_code, 200)
        files, manifest = self.export(sequence_id)
        lean = json.loads(files['sequence.json'])
        second, third = lean['shots'][1], lean['shots'][2]
        self.assertEqual((second['closes']['how'], second['closes']['boundary'], second['closes']['keeps']), ('after_line', 'K2', ['L3']))
        self.assertEqual(third['initial_frame']['description_source'], 'derived')
        end = 2.0 + second['duration']
        self.assertGreater(end, 4.0)
        # Shot 2 plays on after cut 2 ends: its last frame and strip stay inside cut 2's picture.
        self.assertIn(('original.mp4', 4.0, 2.0, 4.0, None, True), self.frames)
        self.assertTrue(all(call[3] <= 4.0 for call in self.frames if call[2] == 2.0))
        # Shot 3 opens later in cut 3 and its strip starts there; labels are seconds of the shot.
        self.assertIn(('original.mp4', round(end, 3), 4.0, 10.0, None, False), self.frames)
        self.assertTrue(all(call[2] == round(end, 6) or call[2] == end for call in self.frames if call[4] == 320 and call[3] == 10.0))
        self.assertEqual({item['path'] for item in manifest['files'] if item['kind'] == 'prompt'}, set())  # nothing written yet: no prompt files

    def test_includes_add_sequence_audio_clips_and_references(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run2','c2',2,4,?,'aniref')",
                         (json.dumps([{'position': 'first', 'time': 2.0, 'image_size': [1280, 720]}]),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d-small','run2','first','[100,100,200,300]','ctx','manual')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d-big','run2','first','[400,100,800,700]','ctx','manual')")
            conn.execute("INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES ('ref','ctx','d-small')")
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,ranges_json) VALUES ('aaaaaaaa-1','ctx','e',0,3,'[]','m',?)",
                         (json.dumps([{'start_time': 0, 'end_time': 3}]),))
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,ranges_json,created_at) VALUES ('bbbbbbbb-2','ctx','e',4,5,'[]','m',?,'2026-01-01')",
                         (json.dumps([{'start_time': 4, 'end_time': 5}]),))
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,source_filename) VALUES ('cccccccc-3','ctx',NULL,0,0,'[]','m','upload.wav')")
        write_audio(media.voice_reference_path('p', 'aaaaaaaa-1'), seconds=3)
        sequence_id = self.written()
        files, manifest = self.export(sequence_id, 'sequence_audio,references,clips')
        self.assertEqual(wav_seconds(files['sequence.wav']), 10.0)
        self.assertTrue({'shot-01-clip.mp4', 'shot-03-clip.mp4', 'references/characters/ctx.jpg', 'references/scenes/S1.jpg', 'references/voices/ctx.wav', 'references/voices/ctx-bbbbbbbb.wav'} <= set(files))
        self.assertEqual(self.clips, [(0.0, 2.0), (2.0, 4.0), (4.0, 10.0)])
        # The saved gallery reference wins over a larger manual box; the crop keeps a margin around the box.
        characters = manifest['references']['characters']['ctx']
        self.assertEqual((characters['chosen'], characters['reason'], [item['source'] for item in characters['candidates']]),
                         (0, 'largest saved gallery reference', ['saved_reference', 'manual_label']))
        self.assertEqual(self.crops, [tuple(round(value, 4) for value in ((100 - 15) / 1280, (100 - 30) / 720, (200 + 15) / 1280, (300 + 30) / 720))])
        self.assertEqual(manifest['references']['scenes']['S1']['cut'], 3)  # every cut is equally framed: the longest one
        # Every voice sample is listed; the 2-15 s one is chosen, the one without audio is an error.
        voices = manifest['references']['voices']['ctx']['samples']
        self.assertEqual([(item['id'], item.get('chosen', False), item['preview_available']) for item in voices],
                         [('bbbbbbbb-2', False, True), ('aaaaaaaa-1', True, True), ('cccccccc-3', False, False)])
        self.assertEqual(next(item for item in voices if item['id'] == 'aaaaaaaa-1')['path'], 'references/voices/ctx.wav')
        self.assertEqual(wav_seconds(files['references/voices/ctx-bbbbbbbb.wav']), 1.0)  # rebuilt from the episode audio
        self.assertEqual([(error['asset'], error['path']) for error in manifest['errors']], [('voice_reference', 'references/voices/ctx-cccccccc.wav')])

    def test_manual_label_is_the_fallback_and_a_missing_source_is_an_error_not_a_substitute(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run2','c2',2,4,?,'aniref')",
                         (json.dumps([{'position': 'first', 'time': 2.0, 'image_size': [1280, 720]}]),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d-auto','run2','first','[0,0,1280,720]','ctx','auto')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d-big','run2','first','[400,100,800,700]','ctx','manual')")
        sequence_id = self.written()
        (self.folder / 'original.mp4').unlink()
        files, manifest = self.export(sequence_id, 'references')
        characters = manifest['references']['characters']['ctx']
        self.assertEqual(([item['detection_id'] for item in characters['candidates']], characters['reason']), (['d-big'], 'no saved gallery reference; largest manually labelled box in these cuts'))
        self.assertFalse(any(name.endswith('.jpg') and name.startswith('shot-') for name in files))
        errors = {(error['asset'], error.get('shot')) for error in manifest['errors']}
        self.assertTrue({('initial_frame', 1), ('last_frame', 1), ('strip', 1), ('character_reference', None), ('scene_reference', None)} <= errors)
        self.assertEqual(self.tiles[-1][1], [False, False, False])  # the contact sheet marks the missing frames

    def test_audio_endpoints_cut_the_original_audio(self):
        sequence_id = self.written()
        whole = self.client.get(f'/api/v1/sequences/{sequence_id}/audio')
        self.assertEqual(whole.status_code, 200)
        self.assertEqual(wav_seconds(whole.content), 10.0)
        exact = self.client.get(f'/api/v1/sequences/{sequence_id}/shots/2/audio')
        self.assertEqual((exact.headers['X-Audio-Start-Seconds'], exact.headers['X-Audio-End-Seconds'], exact.headers['content-disposition']), ('2.0', '4.0', 'attachment; filename="shot-02.wav"'))
        self.assertEqual(wav_seconds(exact.content), 2.0)
        self.assertEqual(wav_seconds(self.client.get(f'/api/v1/sequences/{sequence_id}/shots/2/audio?pad=true').content), 5.0)  # silence up to the generated length
        self.assertEqual(self.client.get(f'/api/v1/sequences/{sequence_id}/shots/9/audio').status_code, 404)
        self.assertEqual(self.client.get('/api/v1/sequences/missing/audio').status_code, 404)

    def test_export_needs_a_plan_known_includes_and_a_v2_payload(self):
        sequence_id = self.create()
        self.assertEqual(self.client.post(f'/api/v1/sequences/{sequence_id}/exports').status_code, 409)
        sequence_id = self.directed()
        self.assertEqual(self.client.post(f'/api/v1/sequences/{sequence_id}/exports?include=prompts').status_code, 422)
        with db.connect() as conn:  # queued by a server that still wrote sequence-v1
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('11111111-1111-1111-1111-111111111111','e','brief_export',?)",
                         (json.dumps({'kind': 'sequence', 'sequence': {'contract': 'sequence-v1'}}),))
        worker.process(*worker.claim_job(('brief_export',)))
        job = db.one("SELECT status,error FROM jobs WHERE id='11111111-1111-1111-1111-111111111111'")
        self.assertEqual(job['status'], 'failed')
        self.assertIn('queued before sequence-v2', job['error'])


class ExportHelpersTest(unittest.TestCase):
    def test_strip_times_srt_and_crop_box(self):
        self.assertEqual(sequence_export.strip_times([4.0, 6.0]), [4.5, 5.5])
        self.assertEqual([round(time, 3) for time in sequence_export.strip_times([0.0, 1.839])], [0.46, 1.379])  # a short shot still gets two frames
        self.assertEqual(sequence_export.srt_time(3723.0456), '01:02:03,046')
        self.assertEqual(sequence_export.srt([{'start_time': 9.0, 'end_time': 12.0, 'text': 'a'}, {'start_time': 20.0, 'end_time': 21.0, 'text': 'b'}], 10.0, 5.0),
                         '1\n00:00:00,000 --> 00:00:02,000\na\n')
        self.assertEqual(sequence_export.crop_fractions({'image_size': [1000, 500], 'box': [100, 100, 300, 300]}), (0.07, 0.14, 0.33, 0.66))
        self.assertEqual(sequence_export.fmt_clock(83.46), '1:23.5')


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is not installed')
class MediaImageTest(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp)
        filters = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True).stdout
        if ' drawtext ' not in filters or ' xstack ' not in filters:
            self.skipTest('ffmpeg lacks drawtext or xstack')

    def image(self, name, size='1920x1080', color='red'):
        path = self.temp / name
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', f'color=c={color}:s={size}:d=1', '-frames:v', '1', str(path)], check=True)
        return path

    def size(self, path):
        info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)], capture_output=True, text=True, check=True)
        return tuple(int(value) for value in info.stdout.strip().split(','))

    def test_tiles_scale_crop_and_clip(self):
        first, second = self.image('a.jpg'), self.image('b.jpg', '640x480', 'blue')
        sheet = self.temp / 'sheet.jpg'
        media.tile_images([(first, "1 · cut 7 (2/2) · 0:25.3 'quoted': 100%"), (second, 'two'), (None, 'three')], sheet, 2, (320, 180))
        self.assertEqual(self.size(sheet), (640, 360))
        single = self.temp / 'single.jpg'
        media.tile_images([(first, 'only')], single, 5, (320, 180))
        self.assertEqual(self.size(single), (320, 180))
        small = self.temp / 'small.jpg'
        media.scale_image(first, small, 640)
        self.assertEqual(self.size(small), (640, 360))
        crop = self.temp / 'crop.jpg'
        media.crop_image(first, crop, (0.25, 0.5, 0.75, 1.2))  # clamped to the image
        self.assertEqual(self.size(crop), (960, 540))
        video, clip = self.temp / 'v.mp4', self.temp / 'clip.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=s=1280x720:d=3', '-pix_fmt', 'yuv420p', str(video)], check=True)
        media.cut_clip(video, clip, 1.0, 2.0)
        self.assertEqual(self.size(clip), (640, 360))
