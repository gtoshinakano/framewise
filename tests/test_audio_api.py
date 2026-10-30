import json
import tempfile
import unittest
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

    def test_approval_preserves_history_and_named_character_in_cut_context(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,character_context_id,origin,model_id) VALUES('t','e',1,2,'Hello','c','model','model')")
        response = self.client.post('/api/episodes/e/transcript/approve-drafts')
        self.assertEqual(response.json(), {'approved': 1})
        history = self.client.get('/api/episodes/e/transcript?history=true').json()
        self.assertEqual(len(history), 2)
        self.assertEqual(next(row for row in history if row['id'] == 't')['status'], 'draft')
        context = self.client.get('/api/cuts/cut/analysis-context').json()
        self.assertEqual(context['transcript_segments'][0]['character_name'], 'Alice')
        self.assertEqual(context['transcript_segments'][0]['text'], 'Hello')
        sections = {key: {'content': 'Evidence', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
        with patch.object(inference_worker, 'chat_json', side_effect=[({'visual_evidence': 'unknown', 'text_evidence': 'unknown'}, 'model'), (sections, 'model')]) as chat:
            inference_worker.synthesize({'openrouter_model': 'test'}, context, [])
        summary_context = json.loads(chat.call_args_list[1].args[2])['context']
        self.assertEqual(summary_context['transcript_segments'][0]['character_name'], 'Alice')
        self.assertEqual(summary_context['transcript_segments'][0]['text'], 'Hello')
        segment = context['transcript_segments'][0]
        edit = {key: segment[key] for key in ('start_time', 'end_time', 'text', 'speaker_label', 'character_context_id', 'status')}
        edit['end_time'] = 2.25
        updated = self.client.put('/api/transcript-segments/' + segment['id'], json=edit)
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()['model_id'], 'model')
        self.client.delete('/api/transcript-segments/' + updated.json()['id'])
        history = self.client.get('/api/episodes/e/transcript?history=true').json()
        self.assertEqual(next(row for row in history if row['status'] == 'deleted')['model_id'], 'model')
