import json
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from fastapi.testclient import TestClient
from app import db, media, worker
from app.main import app


class BriefTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'db.sqlite'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio) VALUES ('e','p','E','a.mp4','.mp4','ready','{}',3,1)")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c','e',0,0.25,1.0)")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label) VALUES ('t','e',0,0.5,'Hello','S1')")
        path = media.episode_dir('p', 'e') / 'audio.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as w:
            w.setparams((1, 2, 8000, 0, 'NONE', 'not compressed')); w.writeframes(b'\0\0' * 24000)
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_brief_and_export_short_audio(self):
        response = self.client.get('/api/v1/episodes/e/brief?from=1&to=1')
        self.assertEqual(response.status_code, 200, response.text)
        cut = response.json()['cuts'][0]
        self.assertEqual(cut['dialogue'][0]['speaker_label'], 'S1')
        self.assertIsNone(cut['action_beats'])
        self.assertEqual(self.client.get('/api/v1/episodes/e/brief?from=1&to=2').status_code, 404)
        job = self.client.post('/api/v1/episodes/e/brief/exports?from=1&to=1').json()['job_id']
        self.assertEqual(worker.claim_job(), (job, (owner := db.one('SELECT lease_owner FROM jobs WHERE id=?', (job,))['lease_owner'])))
        worker.process(job, owner)
        status = self.client.get('/api/v1/brief-exports/' + job).json()
        self.assertEqual(status['status'], 'done')
        archive = self.client.get(status['download_url'])
        import io
        with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
            manifest = json.loads(z.read('manifest.json'))
            self.assertIn('frame', manifest[0]['errors'])
            with wave.open(io.BytesIO(z.read(manifest[0]['audio']))) as w:
                self.assertEqual(w.getnframes(), 6000)

    def test_structured_sections_and_frame_link(self):
        folder = media.episode_dir('p', 'e') / 'analysis'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'frame.jpg').write_bytes(b'jpeg')
        sections = {'main_characters': {'content': 'A person in a coat', 'confidence': 'medium'},
                    'actions': {'content': '1. The person turns.', 'confidence': 'low'},
                    'suggested_composition_frame': {'content': 'A useful view', 'confidence': 'high', 'frame_id': 'frame'}}
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,context_json) VALUES ('frame','c','first_frame','Frame at 0.250s','Frame','model','{}')")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,context_json) VALUES ('s','c','cut_description','Cut summary','Summary','model',?)", (json.dumps({'brief_sections': sections}),))
        body = self.client.get('/api/v1/episodes/e/brief?from=1&to=1').json()['cuts'][0]
        self.assertEqual(body['suggested_frame_url'], '/api/analysis-results/frame/frame')
        self.assertEqual(body['visible_characters']['main']['content'], 'A person in a coat')
        self.assertEqual(body['action_beats']['confidence'], 'low')
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/api/v1/episodes/{episode_id}/brief', schema['paths'])
