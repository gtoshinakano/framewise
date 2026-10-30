"""External consumer contract and sample-accurate audio export."""
import io
import tempfile
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient
from app import db, media
from app.external_api import AnalysisContext, AnalysisResult
from app.main import app


class ExternalAPITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.temp.name)
        db.DB_PATH = db.DATA_DIR / "app.sqlite3"
        self.client = TestClient(app)
        self.client.__enter__()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Project','Synopsis')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,settings_json,status,duration,width,height,has_audio) VALUES ('e','p','Episode','original.mp4','.mp4','{}','ready',5,320,180,1)")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,settings_json) VALUES ('pending','p','Pending','original.mp4','.mp4','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,1,2),('c2','e',1,2,4),('old','e',-1,0,1)")
            conn.execute("UPDATE cuts SET active=0 WHERE id='old'")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content,superseded_by) VALUES ('r0','c1','cut_description','old','r1')")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content,parent_id) VALUES ('r1','c1','cut_description','new','r0')")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('character','p','character','A')")
            conn.execute("INSERT INTO context_items(id,project_id,episode_id,kind,title) VALUES ('other','p','pending','general','Other')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,character_context_id) VALUES ('t','e',1.5,2.5,'Hello','character'),('rejected','e',1,2,'No',NULL)")
            conn.execute("INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id) VALUES ('v','character','e',0,2,'[]','model')")
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES ('j','e','analyze','done','{\"example\":true}')")
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json,completed) VALUES ('j','c1','[{\"observation\":\"frame\"}]',1)")
        self.audio = media.episode_dir('p', 'e') / 'audio.wav'
        self.samples = b''.join(i.to_bytes(2, 'little', signed=True) for i in range(50))
        with wave.open(str(self.audio), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(10)
            audio.writeframes(self.samples)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.temp.cleanup()

    def test_discovery_detail_and_history(self):
        self.assertEqual(self.client.get('/api/v1/projects').json()[0]['id'], 'p')
        episodes = self.client.get('/api/v1/projects/p/episodes').json()
        self.assertIsNone(next(e for e in episodes if e['id'] == 'pending')['has_audio'])
        first = self.client.get('/api/v1/episodes/e/cuts?limit=1').json()
        self.assertEqual((first['total'], first['items'][0]['cut_index'], first['next_offset']), (2, 0, 1))
        self.assertEqual(self.client.get('/api/v1/episodes/e/cuts?offset=1&limit=1').json()['items'][0]['id'], 'c2')
        self.assertEqual(self.client.get('/api/v1/episodes/e/cuts?include_archived=true').json()['items'][-1]['id'], 'old')
        detail = self.client.get('/api/v1/cuts/c1')
        self.assertEqual(detail.status_code, 200, detail.text)
        data = detail.json()
        self.assertEqual([r['id'] for r in data['analysis_results']], ['r1'])
        self.assertEqual([t['id'] for t in data['transcript_segments']], ['rejected', 't'])
        self.assertEqual([c['id'] for c in data['context_items']], ['character'])
        self.assertEqual(data['voice_samples'][0]['character_context_id'], 'character')
        self.assertTrue(data['voice_samples'][0]['preview_available'])
        self.assertEqual(self.client.get('/api/v1/voice-samples/v').json()['embedding'], [])
        job = self.client.get('/api/v1/jobs/j').json()
        self.assertTrue(job['payload']['example'])
        self.assertEqual(job['cut_evidence'][0]['evidence'][0]['observation'], 'frame')
        self.assertEqual(data['next_cut']['id'], 'c2')
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results?history=true').json()), 2)
        voice = self.client.get('/api/voice-samples/v/audio')
        self.assertEqual(voice.status_code, 200)
        with wave.open(io.BytesIO(voice.content)) as audio:
            self.assertEqual(audio.readframes(20), self.samples[:40])
        self.assertEqual(self.client.get('/api/v1/cuts/missing').status_code, 404)

    def test_streamed_audio_margins_and_validation(self):
        response = self.client.get('/api/v1/cuts/c1/audio?before=0.5&after=1')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'audio/wav')
        self.assertEqual(float(response.headers['x-audio-start-seconds']), 0.5)
        self.assertEqual(float(response.headers['x-audio-end-seconds']), 3)
        self.assertEqual(int(response.headers['content-length']), len(response.content))
        with wave.open(io.BytesIO(response.content)) as audio:
            self.assertEqual(audio.getframerate(), 10)
            self.assertEqual(audio.getnframes(), 25)
            self.assertEqual(audio.readframes(25), self.samples[10:60])
        clipped = self.client.get('/api/v1/cuts/c1/audio?before=10&after=10')
        with wave.open(io.BytesIO(clipped.content)) as audio:
            self.assertEqual(audio.readframes(50), self.samples)
        self.assertEqual(float(clipped.headers['x-audio-start-seconds']), 0)
        self.assertEqual(float(clipped.headers['x-audio-end-seconds']), 5)
        for query in ['before=-1', 'after=3601', 'before=nan']:
            self.assertEqual(self.client.get('/api/v1/cuts/c1/audio?' + query).status_code, 422)
        self.audio.unlink()
        self.assertEqual(self.client.get('/api/v1/cuts/c1/audio').status_code, 404)

    def test_openapi_and_offline_swagger(self):
        spec = self.client.get('/openapi.json').json()
        self.assertIn('/api/v1/cuts/{cut_id}/audio', spec['paths'])
        self.assertIn('Recommended read sequence', spec['info']['description'])
        self.assertEqual(spec['info']['description'], self.client.get('/api/integration-guide').text)
        schemas = spec['components']['schemas']
        for name in ('ResultCreate', 'ResultUpdate', 'AnalysisResult', 'Transcript'):
            self.assertNotIn('status', schemas[name]['properties'])
        result_path = spec['paths']['/api/cuts/{cut_id}/analysis-results']
        self.assertEqual(result_path['post']['responses']['200']['content']['application/json']['schema']['$ref'], '#/components/schemas/AnalysisResult')
        self.assertEqual(spec['paths']['/api/analysis-results/{result_id}']['put']['responses']['200']['content']['application/json']['schema']['$ref'], '#/components/schemas/AnalysisResult')
        self.assertEqual(spec['paths']['/api/cuts/{cut_id}/analysis-context']['get']['responses']['200']['content']['application/json']['schema']['$ref'], '#/components/schemas/AnalysisContext')
        headers = spec['paths']['/api/v1/cuts/{cut_id}/audio']['get']['responses']['200']['headers']
        self.assertIn('X-Audio-Start-Seconds', headers)
        self.assertIn('X-Audio-End-Seconds', headers)
        AnalysisContext.model_validate(self.client.get('/api/cuts/c1/analysis-context').json())
        created = self.client.post('/api/cuts/c1/analysis-results', json={'kind': 'other', 'content': 'New finding'})
        self.assertEqual(created.status_code, 200, created.text)
        AnalysisResult.model_validate(created.json())
        revised = self.client.put('/api/analysis-results/' + created.json()['id'], json={'content': 'Revised finding'})
        self.assertEqual(revised.status_code, 200, revised.text)
        AnalysisResult.model_validate(revised.json())
        response = self.client.get('/docs')
        self.assertIn('/api/swagger-assets/swagger-ui-bundle.js', response.text)
        self.assertNotIn('cdn.jsdelivr.net', response.text)
        self.assertEqual(self.client.get('/api/swagger-assets/swagger-ui-bundle.js').status_code, 200)
        self.assertEqual(self.client.get('/api/integration-guide').status_code, 200)


if __name__ == '__main__':
    unittest.main()
