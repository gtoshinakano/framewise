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

    def test_repeated_merges_refresh_only_changed_cuts(self):
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET duration=8 WHERE id='e'")
            conn.execute("UPDATE cuts SET first_frame_time=start_time,last_frame_time=end_time-0.1 WHERE episode_id='e'")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c3','e',2,4,6,4,5.9),('c4','e',3,6,8,6,7.9)")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('saved','c2','other','Note','Keep me')")
        (media.episode_dir('p', 'e') / 'proxy.mp4').touch()
        def extract(_proxy, target, timestamp, *_args, **_kwargs):
            target.write_bytes(b'frame')
            return timestamp
        for expected_archived in ('c2', 'c3'):
            response = self.client.post('/api/cuts/c1/merge-next')
            self.assertEqual(response.status_code, 200, response.text)
            job = db.one("SELECT * FROM jobs WHERE episode_id='e' ORDER BY rowid DESC LIMIT 1")
            self.assertEqual(json.loads(job['payload_json'])['cut_ids'], ['c1'])
            self.assertIsNotNone(db.one("SELECT first_frame_time FROM cuts WHERE id='c4'")['first_frame_time'])
            with patch.object(media, 'extract_cut_frame', side_effect=extract) as extract_frame:
                worker.process(*worker.claim_job(('cutframes',)))
            self.assertEqual(extract_frame.call_count, 2)
            self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job['id'],))['status'], 'done')
            self.assertEqual(db.one("SELECT active FROM cuts WHERE id=?", (expected_archived,))['active'], 0)
        self.assertEqual(db.one("SELECT content FROM analysis_results WHERE id='saved'")['content'], 'Keep me')
        cuts = db.all_rows("SELECT cut_index,start_time,end_time FROM cuts WHERE episode_id='e' AND active=1 ORDER BY cut_index")
        self.assertEqual([(c['cut_index'], c['start_time'], c['end_time']) for c in cuts], [(0, 0, 6), (1, 6, 8)])

    def test_cut_timing_refreshes_only_boundaries_that_changed(self):
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET duration=6 WHERE id='e'")
            conn.execute("UPDATE cuts SET first_frame_time=start_time,last_frame_time=end_time-0.1 WHERE episode_id='e'")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c3','e',2,4,6,4,5.9)")
        response = self.client.put('/api/cuts/c2/timing', json={'start_time': 1.5, 'end_time': 4})
        self.assertEqual(response.status_code, 200, response.text)
        job = db.one("SELECT payload_json FROM jobs WHERE episode_id='e' ORDER BY rowid DESC LIMIT 1")
        self.assertEqual(json.loads(job['payload_json'])['cut_ids'], ['c1', 'c2'])
        self.assertEqual(db.one("SELECT first_frame_time FROM cuts WHERE id='c3'")['first_frame_time'], 4)

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
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertEqual(results[0]['job_id'], job_id)
        compact = self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()
        self.assertEqual(len(compact), len(results))
        self.assertEqual(compact[0]['id'], results[0]['id'])
        self.assertEqual(compact[0]['context'], {})
        saved = self.client.put('/api/analysis-results/' + results[0]['id'], json={'title': 'Edited', 'content': 'Human revision', 'status': 'approved'})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results?history=true').json()), 5)

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
        observations = [{'time': 0.05, 'description': 'A figure', 'ocr_response': '<ref>OPEN</ref><box><0><0><100><100></box>', 'ocr_prompt': 'detect words'}]
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
        results = self.client.get('/api/cuts/c1/analysis-results').json()
        self.assertEqual(len(results), 5)
        raw = next(item for item in results if item['title'].startswith('Raw OCR'))
        self.assertIn('OPEN', raw['content'])
        self.assertEqual(raw['model_id'], inference_worker.MODEL_ID)
        summary = next(item for item in results if item['title'] == 'Cut summary')
        self.assertEqual(summary['context']['sampled_frames'][0]['ocr_response'], observations[0]['ocr_response'])
        self.assertEqual(summary['context']['assessments']['visual_evidence'], 'draft')

    def test_older_checkpoint_is_refreshed_for_ocr(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES (?,?,?)", (job_id, 'c1', json.dumps([{'time': 0.05, 'description': 'Old frame'}])))
        fresh = [{'time': 0.05, 'description': 'New frame', 'ocr_response': 'OPEN', 'ocr_prompt': 'detect words'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', return_value=fresh) as describe, patch.object(inference_worker, 'synthesize', return_value=({key: 'draft' for _, _, key in inference_worker.FIELDS}, 'test/model')):
            inference_worker.process(*worker.claim_job(('analyze',)))
        describe.assert_called_once()
        self.assertEqual(json.loads(db.one('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=?', (job_id,))['evidence_json']), fresh)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')

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
            return [{'time': cut['start_time'], 'description': 'A figure', 'ocr_response': '', 'ocr_prompt': 'detect words'}]
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
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results').json()), 5)
        self.assertEqual(len(self.client.get('/api/cuts/c2/analysis-results').json()), 5)

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
        sections = {key: {'content': f'{key} draft', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}
        fields['cut_summary'] = inference_worker.render_summary(sections)
        response = {'model': 'provider/concrete', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({**fields, **sections})}}]}
        config = {'openrouter_model': 'openrouter/free'}
        observations = [{'time': .05, 'description': 'Visible figure'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode())) as call:
            result, model = inference_worker.synthesize(config, {'additional_context': 'night'}, observations)
        self.assertEqual({key: result[key] for key in fields}, fields)
        self.assertEqual(model, 'provider/concrete')
        request = call.call_args_list[0].args[0]
        self.assertEqual(request.full_url, 'https://openrouter.ai/api/v1/chat/completions')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'openrouter/free')
        self.assertEqual(json.loads(payload['messages'][1]['content'])['sampled_frames'], observations)
        summary_payload = json.loads(call.call_args_list[1].args[0].data)
        self.assertEqual(json.loads(summary_payload['messages'][1]['content'])['assessments']['visual_evidence'], fields['visual_evidence'])
        self.assertNotIn('secret', request.data.decode())
        response['choices'][0]['finish_reason'] = 'length'
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                inference_worker.synthesize(config, {}, observations)

    def test_summary_sections_are_required_and_rendered_in_fixed_order(self):
        sections = {key: {'content': f'Body for {key}', 'confidence': 'medium'} for key, _ in reversed(inference_worker.SUMMARY_TOPICS)}
        incomplete = dict(sections)
        del incomplete['camera_movement']
        def reply(value):
            return io.BytesIO(json.dumps({'model': 'test/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(incomplete), reply(sections)]) as call:
            output, _ = inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', [key for key, _ in inference_worker.SUMMARY_TOPICS], summary_topics=True)
        rendered = inference_worker.render_summary(output)
        self.assertEqual(rendered, '\n\n'.join(f'{title} · Confidence: Medium\nBody for {key}' for key, title in inference_worker.SUMMARY_TOPICS))
        repair = json.loads(call.call_args_list[1].args[0].data)
        self.assertIn('camera_movement', repair['messages'][-1]['content'])
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(incomplete), reply(incomplete)]):
            with self.assertRaisesRegex(RuntimeError, 'camera_movement'):
                inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', [key for key, _ in inference_worker.SUMMARY_TOPICS], summary_topics=True)

    def test_summary_confidence_labels_are_validated_and_repaired(self):
        sections = {key: {'content': f'Body for {key}', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
        invalid = {**sections, 'lighting': {'content': 'Bright', 'confidence': '95%'}}
        def reply(value):
            return io.BytesIO(json.dumps({'model': 'test/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        keys = [key for key, _ in inference_worker.SUMMARY_TOPICS]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(invalid), reply(sections)]) as call:
            output, _ = inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', keys, summary_topics=True)
        self.assertIn('lighting confidence must be', json.loads(call.call_args_list[1].args[0].data)['messages'][-1]['content'])
        self.assertEqual(inference_worker.render_summary(output).count('Confidence: Unknown'), len(keys))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(invalid), reply(invalid)]):
            with self.assertRaisesRegex(RuntimeError, 'lighting confidence'):
                inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', keys, summary_topics=True)

    def test_openrouter_accepts_structured_evidence(self):
        payload = {"visual_evidence": [{"observation": "coat", "confidence": "low"}], "text_evidence": [{"reading": "OPEN", "category": "scene text"}]}
        response = {'model': 'provider/x', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(payload)}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            result, _ = inference_worker.chat_json('test', 'system', 'evidence', ['visual_evidence', 'text_evidence'])
        self.assertEqual(json.loads(result['visual_evidence'])[0]['confidence'], 'low')
        self.assertEqual(json.loads(result['text_evidence'])[0]['reading'], 'OPEN')

    def test_openrouter_repairs_missing_fields_once(self):
        fields = {key: f'{key} draft' for _, _, key in inference_worker.FIELDS}
        sections = {key: {'content': f'{key} draft', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}
        fields['cut_summary'] = inference_worker.render_summary(sections)
        incomplete = dict(fields)
        del incomplete['visual_evidence']
        first = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(incomplete)}}]}
        repaired = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({**fields, **sections})}}]}
        replies = [io.BytesIO(json.dumps(reply).encode()) for reply in (first, repaired, repaired)]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=replies) as call:
            result, model = inference_worker.synthesize({'openrouter_model': 'openrouter/free'}, {}, [])
        self.assertEqual({key: result[key] for key in fields}, fields)
        self.assertEqual(model, 'provider/free')
        repair_payload = json.loads(call.call_args_list[1].args[0].data)
        self.assertIn('visual_evidence', repair_payload['messages'][-1]['content'])
        self.assertEqual(call.call_count, 3)

    def test_subtitles_default_and_prompt_rule(self):
        config = self.client.get('/api/cuts/c1/analysis-config').json()
        self.assertIn('subtitles', config['detection_queries'])
        self.assertIn('Subtitle rule', inference_worker.SYSTEM_PROMPT)
        self.assertIn('Subtitle rule', inference_worker.ANIMATION_SYSTEM)

    def test_cut_summary_regeneration_creates_revision(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,status,origin) VALUES ('anim','c1','generation_prompt','Cut summary','Old prompt','approved','model'),('shot','c1','cut_description','Shot description','A figure','draft','model')")
        self.assertEqual(self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'edit', 'instruction': ' '}).status_code, 400)
        queued = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'edit', 'instruction': 'Slower camera', 'current_prompt': 'Unsaved prompt'})
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c2/regenerate-summary', json={'mode': 'regenerate'}).status_code, 409)
        job_id = queued.json()['job_id']
        payload = json.loads(db.one("SELECT payload_json FROM jobs WHERE id=?", (job_id,))['payload_json'])
        self.assertEqual((payload['base_id'], payload['current_prompt']), ('anim', 'Unsaved prompt'))
        self.assertEqual([item['title'] for item in payload['results']], ['Shot description'])
        claimed = worker.claim_job(('regen',))
        response = {'model': 'provider/x', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({key: {'content': 'New prompt', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS})}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'k'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            inference_worker.process_regen(*claimed)
        results = self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()
        current = next(item for item in results if item['title'] == 'Cut summary')
        self.assertEqual((current['content'], current['parent_id'], current['origin']), (inference_worker.render_summary({key: {'content': 'New prompt', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}), 'anim', 'model'))
        self.assertEqual(db.one("SELECT superseded_by FROM analysis_results WHERE id='anim'")['superseded_by'], current['id'])
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))['status'], 'done')

    def test_summary_regeneration_fails_when_changed_meanwhile(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,status,origin) VALUES ('anim','c1','generation_prompt','Cut summary','Old','draft','model')")
        job_id = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
        claimed = worker.claim_job(('regen',))
        self.client.put('/api/analysis-results/anim', json={'title': 'Cut summary', 'content': 'Edited by hand', 'status': 'draft'})
        response = {'model': 'm', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({key: {'content': 'New', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS})}}]}
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

    def test_delete_results_reverts_revision_and_discards_drafts(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,status,origin) VALUES ('a','c1','other','A','one','approved','model'),('d1','c1','other','D1','x','draft','model'),('d2','c1','other','D2','y','draft','model')")
        revised = self.client.put('/api/analysis-results/a', json={'title': 'A', 'content': 'two', 'status': 'draft'}).json()
        self.assertEqual(self.client.delete('/api/analysis-results/a').status_code, 409)
        self.assertEqual(self.client.delete(f"/api/analysis-results/{revised['id']}").status_code, 200)
        self.assertIsNone(db.one("SELECT superseded_by FROM analysis_results WHERE id='a'")['superseded_by'])
        self.assertEqual(self.client.delete('/api/cuts/c1/analysis-results').json(), {'deleted': 2})
        remaining = [item['id'] for item in self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()]
        self.assertEqual(remaining, ['a'])
        self.assertEqual(self.client.delete('/api/analysis-results/missing').status_code, 404)


if __name__ == '__main__':
    unittest.main()
