"""Contract tests for analysis snapshots, worker isolation, and revision-safe batch writes."""
import json
import os
import tempfile
import threading
import unittest
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import db, media, worker, inference_worker
from app.main import app


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','Snapshot synopsis')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,2),('c2','e',1,2,4)")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('ctx','p','character','Hero','Green coat'),('other','p','world','World','Rain')")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_batch_snapshots_and_resume(self):
        config = {'frame_mode': 'uniform', 'frame_count': 3, 'detection_queries': ['person', 'coat'], 'context_ids': ['ctx'], 'additional_context': 'At night', 'openrouter_model': 'openrouter/free'}
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json=config).status_code, 200)
        response = self.client.post('/api/episodes/e/analyze')
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])
        job_summary = self.client.get('/api/episodes/e').json()['jobs'][0]
        self.assertEqual(job_summary['id'], job_id)
        self.assertNotIn('payload_json', job_summary)
        self.assertNotIn('lease_owner', job_summary)
        self.assertEqual(payload['cuts'][0]['config'], config)
        self.assertEqual([item['id'] for item in payload['cuts'][0]['context']['context_items']], ['ctx'])
        self.assertEqual(payload['cuts'][1]['config']['frame_mode'], 'endpoints')
        with db.connect() as conn:
            conn.execute("UPDATE context_items SET content='Changed' WHERE id='ctx'")
        self.assertEqual(payload['cuts'][0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertIsNone(worker.claim_job())
        claimed = worker.claim_job(('analyze',))
        self.assertEqual(claimed[0], job_id)
        calls = []
        def descriptions(job, owner, episode, cut, config):
            calls.append(cut['id'])
            return [{'time': cut['start_time'] + .05, 'description': 'Visible figure'}]
        def summary(config, context, observations):
            return ({key: f'{key} draft' for _, _, key in inference_worker.FIELDS}, 'test/model')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=descriptions), patch.object(inference_worker, 'synthesize', side_effect=summary):
            inference_worker.process(*claimed)
        self.assertEqual(calls, ['c1', 'c2'])
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        results = self.client.get('/api/cuts/c1/analysis-results').json()
        self.assertEqual(len(results), 8)
        self.assertEqual(results[0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertEqual(results[0]['job_id'], job_id)
        compact = self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()
        self.assertEqual(len(compact), len(results))
        self.assertEqual(compact[0]['id'], results[0]['id'])
        self.assertEqual(compact[0]['context'], {})
        saved = self.client.put('/api/analysis-results/' + results[0]['id'], json={'title': 'Edited', 'content': 'Human revision', 'status': 'approved'})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results?history=true').json()), 9)

    def test_invalid_context_and_episode_exclusivity(self):
        response = self.client.put('/api/cuts/c1/analysis-config', json={'context_ids': ['missing']})
        self.assertEqual(response.status_code, 400)
        queued = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c2/analyze').status_code, 409)
        self.assertEqual(self.client.post(f"/api/jobs/{queued.json()['job_id']}/cancel").status_code, 200)
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")["status"], 'ready')
        self.assertEqual(self.client.post('/api/cuts/c2/analyze').status_code, 200)

    def test_queue_snapshot_excludes_concurrent_context_edits(self):
        from app import main
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json={'context_ids': ['ctx']}).status_code, 200)
        captured, release, write_finished = threading.Event(), threading.Event(), threading.Event()
        original = main.selected_analysis_context
        def hold_snapshot(cut_id, config):
            value = original(cut_id, config)
            captured.set()
            if not release.wait(5):
                raise RuntimeError('Test did not release the snapshot')
            return value
        def edit_context():
            with db.connect() as conn:
                conn.execute("UPDATE context_items SET content='Changed concurrently' WHERE id='ctx'")
            write_finished.set()
        with ThreadPoolExecutor(max_workers=2) as executor, patch.object(main, 'selected_analysis_context', side_effect=hold_snapshot):
            queued = executor.submit(self.client.post, '/api/cuts/c1/analyze')
            try:
                self.assertTrue(captured.wait(5))
                edited = executor.submit(edit_context)
                self.assertFalse(write_finished.wait(.2))
            finally:
                release.set()
            response = queued.result(timeout=5)
            edited.result(timeout=5)
        self.assertEqual(response.status_code, 200, response.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (response.json()['job_id'],))['payload_json'])
        self.assertEqual(payload['cuts'][0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertEqual(db.one("SELECT content FROM context_items WHERE id='ctx'")['content'], 'Changed concurrently')

    def test_checkpoint_reuse(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        observations = [{'time': 0.05, 'description': 'A figure'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', return_value=observations), patch.object(inference_worker, 'synthesize', side_effect=RuntimeError('network error')):
            inference_worker.process(*claimed)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')
        self.assertEqual(json.loads(db.one('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=?', (job_id,))['evidence_json']), observations)
        retry = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(retry.json()['job_id'], job_id)
        claimed = worker.claim_job(('analyze',))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=AssertionError('should reuse checkpoint')), patch.object(inference_worker, 'synthesize', return_value=({key: 'draft' for _, _, key in inference_worker.FIELDS}, 'test/model')):
            inference_worker.process(*claimed)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results').json()), 8)

    def test_retry_rejects_changed_unfinished_cut(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            inference_worker.process(*claimed)
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        response = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(response.status_code, 409)
        self.assertIn('changed', response.json()['detail'])
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')

    def test_retry_batch_skips_completed_cuts(self):
        job_id = self.client.post('/api/episodes/e/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        described, summarized = [], []
        def describe(job, owner, episode, cut, config):
            described.append(cut['id'])
            return [{'time': cut['start_time'], 'description': 'A figure'}]
        def summary(config, context, observations):
            summarized.append(context['cut']['id'])
            if context['cut']['id'] == 'c2' and summarized.count('c2') == 1:
                raise RuntimeError('Temporary provider error')
            return ({key: 'draft' for _, _, key in inference_worker.FIELDS}, 'test/model')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=describe), patch.object(inference_worker, 'synthesize', side_effect=summary):
            inference_worker.process(*claimed)
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')
            self.assertEqual(self.client.post(f'/api/jobs/{job_id}/retry').status_code, 200)
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(described, ['c1', 'c2'])
        self.assertEqual(summarized, ['c1', 'c2', 'c2'])
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results').json()), 8)
        self.assertEqual(len(self.client.get('/api/cuts/c2/analysis-results').json()), 8)

    def test_lost_lease_prevents_result_writes(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        def lose_lease(*args):
            with db.connect() as conn:
                conn.execute("UPDATE jobs SET lease_owner='another-worker' WHERE id=?", (job_id,))
            return [{'time': .05, 'description': 'A figure'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=lose_lease):
            inference_worker.process(*claimed)
        self.assertFalse(db.all_rows('SELECT * FROM analysis_results WHERE job_id=?', (job_id,)))
        self.assertIsNone(db.one('SELECT * FROM analysis_job_cuts WHERE job_id=?', (job_id,)))
        self.assertEqual(db.one('SELECT lease_owner FROM jobs WHERE id=?', (job_id,))['lease_owner'], 'another-worker')

    def test_sampling_stays_inside_cut(self):
        self.assertEqual(inference_worker.sample_times(2, 4, {'frame_mode': 'endpoints'}), [2, 3.999999])
        times = inference_worker.sample_times(2, 4, {'frame_mode': 'uniform', 'frame_count': 5})
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 2)
        self.assertEqual(times[-1], 3.999999)

    def test_openrouter_request_and_validation(self):
        fields = {key: f'{key} draft' for _, _, key in inference_worker.FIELDS}
        response = {'model': 'provider/concrete', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(fields)}}]}
        config = {'openrouter_model': 'openrouter/free'}
        observations = [{'time': .05, 'description': 'Visible figure'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result, model = inference_worker.synthesize(config, {'additional_context': 'night'}, observations)
        self.assertEqual(result, fields)
        self.assertEqual(model, 'provider/concrete')
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, 'https://openrouter.ai/api/v1/chat/completions')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'openrouter/free')
        self.assertEqual(json.loads(payload['messages'][1]['content'])['sampled_frames'], observations)
        self.assertNotIn('secret', request.data.decode())
        response['choices'][0]['finish_reason'] = 'length'
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                inference_worker.synthesize(config, {}, observations)

    def test_openrouter_repairs_missing_fields_once(self):
        fields = {key: f'{key} draft' for _, _, key in inference_worker.FIELDS}
        incomplete = dict(fields)
        del incomplete['character_sheets']
        first = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(incomplete)}}]}
        repaired = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(fields)}}]}
        replies = [io.BytesIO(json.dumps(reply).encode()) for reply in (first, repaired)]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=replies) as call:
            result, model = inference_worker.synthesize({'openrouter_model': 'openrouter/free'}, {}, [])
        self.assertEqual(result, fields)
        self.assertEqual(model, 'provider/free')
        repair_payload = json.loads(call.call_args_list[1].args[0].data)
        self.assertIn('character_sheets', repair_payload['messages'][-1]['content'])
        self.assertEqual(call.call_count, 2)

    def test_subtitles_default_and_prompt_rule(self):
        config = self.client.get('/api/cuts/c1/analysis-config').json()
        self.assertIn('subtitles', config['detection_queries'])
        self.assertIn('Subtitle rule', inference_worker.SYSTEM_PROMPT)
        self.assertIn('Subtitle rule', inference_worker.ANIMATION_SYSTEM)

    def test_animation_prompt_regeneration_creates_revision(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,status,origin) VALUES ('anim','c1','generation_prompt','MiniMax animation prompt','Old prompt','approved','model'),('shot','c1','cut_description','Shot description','A figure','draft','model')")
        self.assertEqual(self.client.post('/api/cuts/c1/regenerate-prompt', json={'mode': 'edit', 'instruction': ' '}).status_code, 400)
        queued = self.client.post('/api/cuts/c1/regenerate-prompt', json={'mode': 'edit', 'instruction': 'Slower camera', 'current_prompt': 'Unsaved prompt'})
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c2/regenerate-prompt', json={'mode': 'regenerate'}).status_code, 409)
        job_id = queued.json()['job_id']
        payload = json.loads(db.one("SELECT payload_json FROM jobs WHERE id=?", (job_id,))['payload_json'])
        self.assertEqual((payload['base_id'], payload['current_prompt']), ('anim', 'Unsaved prompt'))
        self.assertEqual([item['title'] for item in payload['results']], ['Shot description'])
        claimed = worker.claim_job(('regen',))
        response = {'model': 'provider/x', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'minimax_animation_prompt': 'New prompt'})}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'k'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            inference_worker.process_regen(*claimed)
        results = self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()
        current = next(item for item in results if item['title'] == 'MiniMax animation prompt')
        self.assertEqual((current['content'], current['parent_id'], current['origin']), ('New prompt', 'anim', 'model'))
        self.assertEqual(db.one("SELECT superseded_by FROM analysis_results WHERE id='anim'")['superseded_by'], current['id'])
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))['status'], 'done')

    def test_regeneration_fails_when_prompt_changed_meanwhile(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,status,origin) VALUES ('anim','c1','generation_prompt','MiniMax animation prompt','Old','draft','model')")
        job_id = self.client.post('/api/cuts/c1/regenerate-prompt', json={'mode': 'regenerate'}).json()['job_id']
        claimed = worker.claim_job(('regen',))
        self.client.put('/api/analysis-results/anim', json={'title': 'MiniMax animation prompt', 'content': 'Edited by hand', 'status': 'draft'})
        response = {'model': 'm', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'minimax_animation_prompt': 'New'})}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'k'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            inference_worker.process_regen(*claimed)
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))['status'], 'failed')

    def test_result_frame_follows_revision_chain(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,status,origin) VALUES ('det','c1','other','Object detections at 0.000s','{}','draft','model')")
        self.assertEqual(self.client.get('/api/analysis-results/det/frame').status_code, 404)
        folder = media.episode_dir('p', 'e') / 'analysis'
        folder.mkdir()
        (folder / 'det.jpg').write_bytes(b'jpeg')
        revised = self.client.put('/api/analysis-results/det', json={'title': 'x', 'content': '{}', 'status': 'draft'}).json()
        self.assertEqual(self.client.get(f"/api/analysis-results/{revised['id']}/frame").content, b'jpeg')


if __name__ == '__main__':
    unittest.main()
