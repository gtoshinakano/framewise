import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, inference_worker, worker


class CharacterDetectionAPITest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown
    def test_database_upgrade_adds_detection_provenance_columns(self):
        with db.connect() as conn:
            conn.execute('ALTER TABLE character_detection_runs DROP COLUMN model_revision')
            conn.execute('ALTER TABLE character_detections DROP COLUMN match_model_id')
            conn.execute('ALTER TABLE character_detections DROP COLUMN match_model_revision')
        db.init_db()
        with db.connect() as conn:
            self.assertIn('model_revision', {row['name'] for row in conn.execute('PRAGMA table_info(character_detection_runs)')})
            self.assertTrue({'match_model_id', 'match_model_revision'} <= {row['name'] for row in conn.execute('PRAGMA table_info(character_detections)')})

    def test_detect_label_reference_and_snapshot(self):
        from app import character_detection, single_cut_pipeline
        queued = self.client.post('/api/cuts/c1/character-detection')
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id, owner = worker.claim_job(('character_detect',))
        frames = [{'position': position, 'time': time, 'image_size': [100, 80], 'boxes': [[5, 4, 40, 60]], 'raw_response': 'raw', 'prompt': 'detect'} for position, time in [('first', 0), ('last', 1.9)]]
        with patch.object(inference_worker, 'get_model', return_value=object()), patch.object(single_cut_pipeline, 'release_gpu_models'), patch.object(character_detection, 'detect_boundary_people', return_value=frames):
            inference_worker.process_character_detection(job_id, owner)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        run = self.client.get('/api/cuts/c1/character-detections').json()
        self.assertEqual(len(run['detections']), 2)
        first = run['detections'][0]['id']
        self.assertEqual(self.client.put(f'/api/character-detections/{first}', json={'character_context_id': 'other'}).status_code, 400)
        self.assertEqual(self.client.put(f'/api/character-detections/{first}', json={'character_context_id': 'ctx'}).status_code, 200)
        self.assertEqual(self.client.post(f'/api/character-detections/{first}/reference').status_code, 200)
        self.assertEqual(self.client.put(f'/api/character-detections/{first}', json={'character_context_id': None}).status_code, 409)
        self.assertEqual(len(self.client.get('/api/projects/p/character-references').json()), 1)
        queued_analysis = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        item = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (queued_analysis,))['payload_json'])['cuts'][0]
        self.assertEqual(item['context']['character_detections']['detections'][0]['character_name'], 'Hero')
        self.assertEqual(item['character_references'][0]['character_context_id'], 'ctx')
        self.assertEqual(self.client.delete(f'/api/character-detections/{first}/reference').status_code, 200)
        self.assertEqual(item['character_references'][0]['character_context_id'], 'ctx')
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        self.assertTrue(self.client.get('/api/cuts/c1/character-detections').json()['stale'])

    def test_automatic_matches_are_unverified_and_queued_labels_are_snapshotted(self):
        from app import character_detection, single_cut_pipeline, frame_model
        frames = [{'position': position, 'time': time, 'image_size': [100, 80], 'boxes': [[5, 4, 40, 60]], 'raw_response': 'raw', 'prompt': 'detect'} for position, time in [('first', 0), ('last', 1.9)]]
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('old','c1',0,2,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('box','old','first','[5,4,40,60]','ctx','manual')")
            conn.execute("INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES ('ref','ctx','box')")
        response = self.client.post('/api/cuts/c1/character-detection')
        self.assertEqual(response.status_code, 200, response.text)
        job_id, owner = worker.claim_job(('character_detect',))
        with patch.object(inference_worker, 'get_model', return_value=object()), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(character_detection, 'detect_boundary_people', return_value=frames), patch.object(character_detection, 'crop_frame', return_value=b'image'), \
             patch.object(frame_model, 'frame_server', return_value=nullcontext('server')), \
             patch.object(frame_model, 'match_character', return_value={'character_id':'ctx','confidence':'high','reason':'same hair'}):
            inference_worker.process_character_detection(job_id, owner)
        run = self.client.get('/api/cuts/c1/character-detections').json()
        self.assertEqual([d['label_source'] for d in run['detections']], ['auto', 'auto'])
        self.assertEqual(len(self.client.get('/api/projects/p/character-references').json()), 1)
        analysis_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        context = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (analysis_id,))['payload_json'])['cuts'][0]['context']
        self.assertEqual(context['character_detections']['detections'][0]['label_source'], 'auto')
        self.assertTrue(context['character_detections']['labels_confirmed_by_queue'])

    def test_analysis_receives_frame_labels_and_reference_images(self):
        from app import character_detection, single_cut_pipeline, frame_model
        frames = [{'position': pos, 'time': t, 'image_size': [100, 80], 'boxes': [[5,4,40,60]], 'raw_response':'raw','prompt':'detect'} for pos,t in [('first',0),('last',1.9)]]
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c1',0,2,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('box','run','first','[5,4,40,60]','ctx','manual')")
            conn.execute("INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES ('ref','ctx','box')")
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        _, owner = worker.claim_job(('analyze',))
        subtitles = {'samples': [], 'events': []}
        frame = {'composition':'','setting':'','lighting':'','camera':'','visible_characters':[], 'other_visual_details':'','uncertainties':''}
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', return_value=object()), patch.object(single_cut_pipeline.subtitle_model, 'inspect_subtitles', return_value=subtitles), \
             patch.object(single_cut_pipeline.media, 'extract_cut_frame', side_effect=lambda source,target,time,*a,**k: time), \
             patch.object(character_detection, 'crop_frame', return_value=b'reference image'), \
             patch.object(frame_model, 'frame_server', return_value=nullcontext('server')), \
             patch.object(frame_model, 'describe_frame', side_effect=lambda *a,**k: dict(frame)) as describe, \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', return_value={'windows': []}) as video:
            inference_worker.process(job_id, owner)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual([call.args[3]['target_frame'] for call in describe.call_args_list], ['first', 'last'])
        self.assertEqual(describe.call_args_list[0].kwargs['reference_images'][0]['image'], b'reference image')
        self.assertEqual(video.call_args.kwargs['context_hints']['character_detections']['detections'][0]['character_name'], 'Hero')
        self.assertTrue(video.call_args.kwargs['context_hints']['character_detections']['labels_confirmed_by_queue'])
        self.assertTrue(describe.call_args_list[0].args[3]['character_detections']['labels_confirmed_by_queue'])
        self.assertEqual(describe.call_args_list[0].args[3]['target_frame'], 'first')
        self.assertEqual(describe.call_args_list[1].args[3]['target_frame'], 'last')
        self.assertIsNone(db.one("SELECT id FROM jobs WHERE pipeline='consistency_summary'"))
        queued = self.client.post('/api/cuts/c1/consistency-summary')
        self.assertEqual(queued.status_code, 200, queued.text)
        summary = json.loads(db.one("SELECT payload_json FROM jobs WHERE pipeline='consistency_summary'")['payload_json'])
        self.assertEqual(summary['evidence']['character_detections']['detections'][0]['label_source'], 'manual')

    def test_batch_queue_does_not_confirm_labels(self):
        frames = [{'position':'first', 'time':0, 'image_size':[100,80]}]
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c1',0,2,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('box','run','first','[5,4,40,60]','ctx','auto')")
        response = self.client.post('/api/episodes/e/analyze')
        self.assertEqual(response.status_code, 200, response.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (response.json()['job_id'],))['payload_json'])
        context = next(item['context'] for item in payload['cuts'] if item['cut_id'] == 'c1')
        self.assertNotIn('labels_confirmed_by_queue', context['character_detections'])

    def test_wrong_cut_and_out_of_bounds_manual_box_rejected(self):
        self.assertEqual(self.client.post('/api/cuts/missing/character-detection').status_code, 404)
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c1',0,2,?,'test')", (json.dumps([{'position':'first','image_size':[100,80]}]),))
        self.assertEqual(self.client.post('/api/character-detection-runs/run/boxes', json={'position': 'first', 'box': [-1, 0, 20, 20]}).status_code, 400)
        self.assertEqual(self.client.post('/api/character-detection-runs/run/boxes', json={'position': 'first', 'box': [0, 0, 20, 20]}).status_code, 200)

    def test_bulk_detection_queues_missing_stale_and_failed_cuts_once(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,active,analysis_enabled) VALUES ('c3','e',2,4,6,1,0),('c4','e',-1,6,8,0,1),('c5','e',3,8,10,1,1)")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('done','c1',0,2,'[]','test'),('stale','c2',2,3,'[]','test')")
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status) VALUES ('failed','e','character_detect','failed')")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,job_id,cut_start,cut_end,frames_json,model_id) VALUES ('partial','c5','failed',8,10,'[]','test')")
        response = self.client.post('/api/episodes/e/character-detection')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['queued'], 2)
        jobs = db.all_rows("SELECT id,payload_json FROM jobs WHERE status='queued' ORDER BY rowid")
        self.assertEqual([json.loads(job['payload_json'])['cut_id'] for job in jobs], ['c2', 'c5'])
        self.assertEqual(self.client.post('/api/episodes/e/character-detection').json()['queued'], 0)
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET analysis_enabled=1 WHERE id='c3'")
        self.assertEqual(self.client.post('/api/episodes/e/character-detection').json()['queued'], 1)
        self.assertEqual(worker.claim_job(('character_detect',))[0], jobs[0]['id'])
        self.assertIsNone(worker.claim_job(('character_detect',)))
        response = self.client.get('/api/episodes/e')
        self.assertEqual({j['cuts'][0]['id'] for j in response.json()['jobs'] if j['status'] in ('queued', 'running')}, {'c2', 'c3', 'c5'})

    def test_bulk_detection_empty_missing_episode_and_conflicting_work(self):
        self.assertEqual(self.client.post('/api/episodes/missing/character-detection').status_code, 404)
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('r1','c1',0,2,'[]','test'),('r2','c2',2,4,'[]','test')")
        self.assertEqual(self.client.post('/api/episodes/e/character-detection').json(), {'queued': 0, 'job_ids': []})
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status) VALUES ('busy','e','hardcuts','queued')")
        self.assertEqual(self.client.post('/api/episodes/e/character-detection').status_code, 409)

    def test_bulk_detection_continues_after_failure_and_requeues_only_failed(self):
        from app import character_detection, single_cut_pipeline
        response = self.client.post('/api/episodes/e/character-detection')
        self.assertEqual(response.status_code, 200, response.text)
        first, second = response.json()['job_ids']
        frames = [{'position': pos, 'time': time, 'image_size': [100, 80], 'boxes': []} for pos, time in [('first', 2), ('last', 3.9)]]
        with patch.object(inference_worker, 'get_model', return_value=object()), patch.object(single_cut_pipeline, 'release_gpu_models'), patch.object(character_detection, 'detect_boundary_people', side_effect=[RuntimeError('detection failed'), frames]):
            claimed, owner = worker.claim_job(('character_detect',))
            self.assertEqual(claimed, first)
            inference_worker.process_character_detection(claimed, owner)
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (first,))['status'], 'failed')
            claimed, owner = worker.claim_job(('character_detect',))
            self.assertEqual(claimed, second)
            inference_worker.process_character_detection(claimed, owner)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (second,))['status'], 'done')
        retry = self.client.post('/api/episodes/e/character-detection').json()
        self.assertEqual(retry['queued'], 1)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (retry['job_ids'][0],))['payload_json'])
        self.assertEqual(payload['cut_id'], 'c1')

    def test_bulk_detection_concurrent_requests_do_not_duplicate_jobs(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: self.client.post('/api/episodes/e/character-detection'), range(2)))
        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(sorted(response.json()['queued'] for response in responses), [0, 2])
        jobs = db.all_rows("SELECT payload_json FROM jobs WHERE pipeline='character_detect' AND status='queued'")
        self.assertEqual(sorted(json.loads(job['payload_json'])['cut_id'] for job in jobs), ['c1', 'c2'])
