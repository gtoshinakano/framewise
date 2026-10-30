import json
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import db, media, inference_worker
from app.main import app


class AudioApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'db.sqlite'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Project'),('other','Other')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,duration,has_audio,settings_json) VALUES('e','p','Episode','x.mp4','.mp4',10,1,'{}')")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES('c','p','character','Alice'),('x','other','character','Elsewhere')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES('cut','e',0,0,5)")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_voice_queue_validation_and_cancel(self):
        endpoint = '/api/episodes/e/voice-samples'
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET has_audio=0 WHERE id='e'")
        self.assertEqual(self.client.post('/api/episodes/e/transcribe').status_code, 400)
        self.assertEqual(self.client.post(endpoint, json={'character_context_id':'c', 'start_time':1, 'end_time':4}).status_code, 400)
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET has_audio=1 WHERE id='e'")
        self.assertEqual(self.client.post(endpoint, json={'character_context_id':'x', 'start_time':1, 'end_time':4}).status_code, 400)
        self.assertEqual(self.client.post(endpoint, json={'character_context_id':'c', 'start_time':9, 'end_time':12}).status_code, 400)
        response = self.client.post(endpoint, json={'character_context_id':'c', 'start_time':1, 'end_time':4})
        self.assertEqual(response.status_code, 200, response.text)
        job = db.one('SELECT * FROM jobs WHERE id=?', (response.json()['job_id'],))
        self.assertEqual(json.loads(job['payload_json'])['character_context_id'], 'c')
        self.assertEqual(self.client.post('/api/jobs/' + job['id'] + '/cancel').status_code, 200)
        self.assertEqual(self.client.post('/api/episodes/e/transcribe').status_code, 200)

    def test_multiple_ranges_and_upload_validation(self):
        endpoint = '/api/episodes/e/voice-samples'
        ranges = [{'start_time': 1, 'end_time': 2}, {'start_time': 4, 'end_time': 5.5}]
        self.assertEqual(self.client.post(endpoint, json={'character_context_id': 'c', 'ranges': [ranges[0]]}).status_code, 400)
        self.assertEqual(self.client.post(endpoint, json={'character_context_id': 'c', 'ranges': [ranges[0], {'start_time': 1.5, 'end_time': 4}]}).status_code, 400)
        response = self.client.post(endpoint, json={'character_context_id': 'c', 'ranges': ranges})
        self.assertEqual(response.status_code, 200, response.text)
        job = db.one('SELECT * FROM jobs WHERE id=?', (response.json()['job_id'],))
        self.assertEqual(json.loads(job['payload_json'])['ranges'], ranges)
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET has_audio=0 WHERE id='e'")
        self.assertEqual(self.client.post(endpoint + '/upload', data={'character_context_id':'x'}, files={'file':('voice.wav', b'audio', 'audio/wav')}).status_code, 400)
        self.assertEqual(self.client.post(endpoint + '/upload', data={'character_context_id':'c'}, files={'file':('voice.wav', b'', 'audio/wav')}).status_code, 400)
        response = self.client.post(endpoint + '/upload', data={'character_context_id':'c'}, files={'file':('voice.wav', b'audio', 'audio/wav')})
        self.assertEqual(response.status_code, 200, response.text)
        job = db.one('SELECT * FROM jobs WHERE id=?', (response.json()['job_id'],))
        upload = media.episode_dir('p', 'e') / 'voice-uploads' / json.loads(job['payload_json'])['upload_id']
        self.assertEqual(upload.read_bytes(), b'audio')
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        self.assertFalse(upload.exists())

    def test_interrupted_upload_is_kept_until_attempts_are_exhausted(self):
        from app import worker
        response = self.client.post('/api/episodes/e/voice-samples/upload', data={'character_context_id':'c'}, files={'file':('voice.wav', b'audio', 'audio/wav')})
        job_id = response.json()['job_id']
        job = db.one('SELECT * FROM jobs WHERE id=?', (job_id,))
        upload = media.episode_dir('p', 'e') / 'voice-uploads' / json.loads(job['payload_json'])['upload_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running',lease_until=0,attempt_count=? WHERE id=?", (worker.MAX_ATTEMPTS - 1, job_id))
        self.assertEqual(worker.claim_job(('voice_sample',))[0], job_id)
        self.assertTrue(upload.exists())
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job_id,))
        self.assertIsNone(worker.claim_job(('voice_sample',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')
        self.assertFalse(upload.exists())

    def test_voice_preview_combines_ranges_and_supports_legacy_references(self):
        path = media.episode_dir('p', 'e') / 'audio.wav'
        with wave.open(str(path), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            out.writeframes(b'\x01\x00' * 32000 + b'\x02\x00' * 32000 + b'\x03\x00' * 96000)
        ranges = [{'start_time': 0, 'end_time': 1}, {'start_time': 2, 'end_time': 3}]
        response = self.client.post('/api/episodes/e/voice-samples/preview', json={'ranges': ranges})
        self.assertEqual(response.status_code, 200, response.text)
        with wave.open(io.BytesIO(response.content), 'rb') as wav:
            self.assertEqual(wav.getnframes(), 32000)
            self.assertEqual(wav.readframes(1), b'\x01\x00')
            wav.setpos(16000)
            self.assertEqual(wav.readframes(1), b'\x02\x00')
        self.assertEqual(self.client.post('/api/episodes/e/voice-samples/preview', json={'ranges':[{'start_time':0,'end_time':2},{'start_time':1,'end_time':3}]}).status_code, 400)
        with db.connect() as conn:
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,ranges_json) VALUES('sample','c','e',0,3,'[1]','model',?)", (json.dumps(ranges),))
        self.assertTrue(self.client.get('/api/projects/p/voice-samples').json()[0]['preview_available'])
        self.assertEqual(self.client.get('/api/voice-samples/sample/audio').content, response.content)

    def test_saved_voice_audio_survives_episode_deletion_and_is_removed_with_reference(self):
        sample_id = 'a01df801-c519-4708-a259-ec849127482a'
        target = media.voice_reference_path('p', sample_id)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'prepared voice')
        with db.connect() as conn:
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id,source_filename) VALUES(?, 'c','e',0,3,'[1]','model','voice.wav')", (sample_id,))
        self.assertEqual(self.client.delete('/api/episodes/e').status_code, 200)
        self.assertEqual(self.client.get('/api/voice-samples/' + sample_id + '/audio').content, b'prepared voice')
        self.assertEqual(self.client.delete('/api/voice-samples/' + sample_id).status_code, 200)
        self.assertFalse(target.exists())

    def test_context_deletion_removes_saved_voice_audio(self):
        sample_id = 'a01df801-c519-4708-a259-ec849127482a'
        target = media.voice_reference_path('p', sample_id)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'voice')
        with db.connect() as conn:
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id) VALUES(?, 'c','e',0,3,'[1]','model')", (sample_id,))
        self.assertEqual(self.client.delete('/api/context/c').status_code, 200)
        self.assertFalse(target.exists())

    def test_transcription_is_in_context_without_approval_and_preserves_history(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,character_context_id,origin,model_id) VALUES('t','e',1,2,'Hello','c','model','model')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,deleted) VALUES('approved','e',2,2.5,'First',0),('rejected','e',2.5,3,'Second',0),('deleted','e',3,3.5,'Removed',1),('outside','e',6,7,'Outside',0)")
        history = self.client.get('/api/episodes/e/transcript?history=true').json()
        self.assertEqual(len(history), 5)
        self.assertNotIn('status', next(row for row in history if row['id'] == 't'))
        context = self.client.get('/api/cuts/cut/analysis-context').json()
        self.assertEqual([row['id'] for row in context['transcript_segments']], ['t', 'approved', 'rejected'])
        self.assertEqual(context['transcript_segments'][0]['character_name'], 'Alice')
        self.assertEqual(context['transcript_segments'][0]['text'], 'Hello')
        sections = {key: {'content': 'Evidence', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
        with patch.object(inference_worker, 'chat_json', side_effect=[({'visual_evidence': 'unknown', 'text_evidence': 'unknown'}, 'model'), (sections, 'model')]) as chat:
            inference_worker.synthesize({'openrouter_model': 'test'}, context, [])
        summary_context = json.loads(chat.call_args_list[1].args[2])['context']
        self.assertEqual(summary_context['transcript_segments'][0]['character_name'], 'Alice')
        self.assertEqual(summary_context['transcript_segments'][0]['text'], 'Hello')
        segment = context['transcript_segments'][0]
        edit = {key: segment[key] for key in ('start_time', 'end_time', 'text', 'speaker_label', 'character_context_id')}
        edit['end_time'] = 2.25
        updated = self.client.put('/api/transcript-segments/' + segment['id'], json=edit)
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()['model_id'], 'model')
        self.assertFalse(updated.json()['deleted'])
        context = self.client.get('/api/cuts/cut/analysis-context').json()
        self.assertEqual([row['id'] for row in context['transcript_segments']], [updated.json()['id'], 'approved', 'rejected'])
        rejected = self.client.put('/api/transcript-segments/rejected', json={'start_time': 2.5, 'end_time': 3, 'text': 'Corrected', 'speaker_label':'', 'character_context_id':None})
        self.assertFalse(rejected.json()['deleted'])
        self.assertIn(rejected.json()['id'], [row['id'] for row in self.client.get('/api/cuts/cut/analysis-context').json()['transcript_segments']])
        self.client.delete('/api/transcript-segments/' + updated.json()['id'])
        history = self.client.get('/api/episodes/e/transcript?history=true').json()
        self.assertEqual(next(row for row in history if row['deleted'] and row['model_id'] == 'model')['model_id'], 'model')

    def test_many_generated_words_are_queued_as_analysis_context_without_approval(self):
        with db.connect() as conn:
            conn.executemany("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin) VALUES(?, 'e',?,?,?,'model')",
                             [(f'word-{i}', i / 300, (i + 1) / 300, f'word{i}') for i in range(1200)])
        context = self.client.get('/api/cuts/cut/analysis-context').json()
        self.assertEqual(len(context['transcript_segments']), 1200)
        response = self.client.post('/api/cuts/cut/analyze')
        self.assertEqual(response.status_code, 200, response.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (response.json()['job_id'],))['payload_json'])
        self.assertEqual(len(payload['cuts'][0]['context']['transcript_segments']), 1200)
