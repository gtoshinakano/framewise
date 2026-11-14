"""The download of a directed sequence: the sequence-v1 package, its audio and frame endpoints and the ZIP."""
import io
import json
import wave
import zipfile
from unittest.mock import patch

import tests.test_sequence_pipeline as pipeline_tests
from app import db, media, worker


def write_audio(path, seconds=10, rate=16000):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(rate)
        out.writeframes(b'\x01\x00' * int(seconds * rate))


class SequenceExportTest(pipeline_tests.SequenceBase):
    def setUp(self):
        super().setUp()
        self.folder = media.episode_dir('p', 'e')
        write_audio(self.folder / 'audio.wav')
        (self.folder / 'original.mp4').write_bytes(b'video')

    def test_package_declares_references_dialogue_and_readiness(self):
        sequence_id = self.directed()
        package = self.client.get(f'/api/v1/sequences/{sequence_id}')
        self.assertEqual(package.status_code, 200, package.text)
        before = package.json()
        self.assertEqual((before['contract'], before['ready'], before['pending'], before['duration']), ('sequence-v1', False, [1, 2, 3], 10.0))
        self.assertEqual((before['shots'][0]['prompt'], before['shots'][0]['written'], before['shots'][0]['dialogue'][0]['delivery']), (None, False, None))
        self.assertEqual(before['shots'][1]['dialogue'][0]['attribution'], 'director')  # the bible's voice decision, before any shot is written
        self.run_step(sequence_id, 'shots')
        self.work()
        package = self.client.get(f'/api/v1/sequences/{sequence_id}').json()
        self.assertEqual((package['ready'], package['pending'], package['bible']['spoken_language'], package['review']['issues'][0]['shot']), (True, [], 'Japanese', 1))
        shot = package['shots'][0]
        self.assertEqual((shot['index'], shot['start_time'], shot['end_time'], shot['generate_seconds'], shot['trim_seconds'], shot['source_cuts'], shot['merged']), (1, 0.0, 2.0, 5.0, 2.0, [1], False))
        self.assertEqual(shot['initial_frame']['strategy'], 'cut_first_frame')
        self.assertEqual((shot['initial_frame']['file'], shot['audio']['file'], shot['audio']['has_speech']), ('shot-01-initial.jpg', 'shot-01.wav', True))
        self.assertTrue(shot['initial_frame']['still_prompt'].startswith('Shot: Medium shot at eye level'))
        self.assertEqual(shot['references'], [{'label': '<Picture 1>', 'role': 'initial_frame', 'character_context_id': None, 'context_id': None, 'name': None},
                                              {'label': '<Picture 2>', 'role': 'character_sheet', 'character_context_id': 'ctx', 'context_id': None, 'name': 'Hero'},
                                              {'label': '<Audio 1>', 'role': 'original_audio', 'character_context_id': None, 'context_id': None, 'name': None}])
        self.assertEqual(shot['subjects'][0], {'number': 1, 'label': '<Subject 1>', 'name': 'Hero', 'kind': 'character', 'character_context_id': 'ctx', 'context_id': None, 'picture': 2,
                                               'anchors': 'slim build, short dark hair, green wool coat over a white shirt'})
        line = shot['dialogue'][0]
        self.assertEqual((line['line'], line['t0'], line['t1'], line['text'], line['character_name'], line['attribution'], line['delivery'], line['subject']),
                         ('L1', 0.4, 1.2, 'こんにちは', 'Hero', 'transcript', 'lip_sync', 1))
        self.assertEqual([(beat['start'], beat['end'], beat.get('hold')) for beat in shot['beats']], [(0.0, 1.0, None), (1.0, 2.0, None), (2.0, 5.0, True)])
        self.assertIn('“shot-01.wav”', shot['prompt'])
        listed = self.client.get('/api/v1/episodes/e/sequences').json()
        self.assertEqual((listed[0]['id'], listed[0]['shots_planned'], listed[0]['shots_written'], listed[0]['has_review']), (sequence_id, 3, 3, True))
        self.assertEqual(self.client.get('/api/v1/sequences/missing').status_code, 404)

    def test_audio_endpoints_cut_the_original_audio(self):
        sequence_id = self.written()
        whole = self.client.get(f'/api/v1/sequences/{sequence_id}/audio')
        self.assertEqual(whole.status_code, 200)
        with wave.open(io.BytesIO(whole.content)) as audio:
            self.assertEqual(audio.getnframes() / audio.getframerate(), 10.0)
        exact = self.client.get(f'/api/v1/sequences/{sequence_id}/shots/2/audio')
        self.assertEqual((exact.headers['X-Audio-Start-Seconds'], exact.headers['X-Audio-End-Seconds']), ('2.0', '4.0'))
        with wave.open(io.BytesIO(exact.content)) as audio:
            self.assertEqual(audio.getnframes() / audio.getframerate(), 2.0)
        padded = self.client.get(f'/api/v1/sequences/{sequence_id}/shots/2/audio?pad=true')
        with wave.open(io.BytesIO(padded.content)) as audio:
            self.assertEqual(audio.getnframes() / audio.getframerate(), 5.0)  # silence up to the generated length
        self.assertEqual(self.client.get(f'/api/v1/sequences/{sequence_id}/shots/9/audio').status_code, 404)

    def test_export_zip_holds_package_audio_frames_and_prompts(self):
        sequence_id = self.written()
        queued = self.client.post(f'/api/v1/sequences/{sequence_id}/exports')
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id = queued.json()['job_id']
        calls = []

        def extract(source, target, timestamp, start, end, **kwargs):
            calls.append((source.name, timestamp, start, end, kwargs.get('width')))
            target.write_bytes(b'jpeg')
            return timestamp
        with patch.object(media, 'extract_cut_frame', side_effect=extract):
            worker.process(*worker.claim_job(('brief_export',)))
        status = self.client.get(f'/api/v1/brief-exports/{job_id}').json()
        self.assertEqual((status['status'], status['download_url']), ('done', f'/api/v1/brief-exports/{job_id}/download'), status)
        self.assertEqual(calls, [('original.mp4', 0.0, 0.0, 2.0, None), ('original.mp4', 2.0, 2.0, 4.0, None), ('original.mp4', 4.0, 4.0, 10.0, None)])
        download = self.client.get(status['download_url'])
        self.assertIn(f'framewise-sequence-{job_id}.zip', download.headers['content-disposition'])
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            names = set(archive.namelist())
            self.assertEqual(names, {'sequence.json', 'manifest.json', 'sequence.wav', *(f'shot-0{n}{suffix}' for n in (1, 2, 3) for suffix in ('.wav', '-initial.jpg', '-prompt.txt', '-still.txt'))})
            package = json.loads(archive.read('sequence.json'))
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual((package['contract'], package['id'], len(package['shots'])), ('sequence-v1', sequence_id, 3))
            self.assertEqual(archive.read('shot-01-prompt.txt').decode(), package['shots'][0]['prompt'])
            self.assertEqual((manifest['audio_seconds'], manifest['errors']), (10.0, {}))
            self.assertEqual([(shot['index'], shot['audio_seconds'], shot['frame'], shot['errors']) for shot in manifest['shots']],
                             [(1, 2.0, 'shot-01-initial.jpg', {}), (2, 2.0, 'shot-02-initial.jpg', {}), (3, 6.0, 'shot-03-initial.jpg', {})])
        # The export is a snapshot: sequence jobs and episode status are untouched by it.
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")['status'], 'ready')

    def test_export_needs_a_plan(self):
        sequence_id = self.create()
        self.assertEqual(self.client.post(f'/api/v1/sequences/{sequence_id}/exports').status_code, 409)
