import json
import os
import unittest
from unittest.mock import patch, Mock

import tests.test_analysis as analysis_tests
from app import db, worker, inference_worker, local_text_runtime


class LocalTextTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def seed_visual_results(self):
        with db.connect() as conn:
            for cut in ('c1', 'c2'):
                for kind, title in [('initial_frame', 'Initial frame description'), ('video_analysis', 'Video analysis'), ('last_frame', 'Last frame description')]:
                    conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES (?,?,?,?,?,?)',
                        (cut+kind, cut, 'cut_description', title, json.dumps({'observation': kind}), json.dumps({'structured_type': kind})))

    def test_model_status_and_desired_state(self):
        import time
        with db.connect() as conn:
            conn.execute("UPDATE local_text_model_state SET heartbeat=?,state='loaded' WHERE id=1", (time.time(),))
        self.assertEqual(self.client.get('/api/local-text-model').json()['state'], 'loaded')
        self.assertTrue(self.client.put('/api/local-text-model', json={'loaded': True}).json()['desired_loaded'])
        with db.connect() as conn:
            conn.execute('UPDATE local_text_model_state SET heartbeat=0 WHERE id=1')
        self.assertEqual(self.client.get('/api/local-text-model').json()['state'], 'worker_offline')

    def test_busy_worker_heartbeat_preserves_generation_state(self):
        runtime = local_text_runtime.LocalTextRuntime()
        with db.connect() as conn:
            conn.execute("UPDATE local_text_model_state SET heartbeat=0,state='generating',detail='Processing' WHERE id=1")
        done = Mock()
        done.wait.side_effect = [False, True]
        runtime.heartbeat(done)
        status = self.client.get('/api/local-text-model').json()
        self.assertEqual(status['state'], 'generating')
        self.assertEqual(status['detail'], 'Processing')
        self.assertGreater(status['heartbeat'], 0)

    def test_worker_starts_and_joins_both_heartbeats(self):
        runtime, job_thread, model_thread = Mock(), Mock(), Mock()
        runtime.process = None
        with patch.object(db, 'init_db'), \
             patch.object(local_text_runtime, 'runtime', None), \
             patch.object(local_text_runtime, 'LocalTextRuntime', return_value=runtime), \
             patch.object(worker, 'claim_job', side_effect=[('job', 'owner'), KeyboardInterrupt]), \
             patch.object(db, 'one', side_effect=[{'pipeline': 'regen'}, {'payload_json': '{}'}]), \
             patch.object(inference_worker.threading, 'Thread', side_effect=[job_thread, model_thread]) as threads, \
             patch.object(inference_worker, 'process_regen'), \
             patch.object(inference_worker.log, 'exception') as log_error:
            with self.assertRaises(KeyboardInterrupt):
                inference_worker.main()
        self.assertIs(threads.call_args_list[1].kwargs['target'], runtime.heartbeat)
        job_thread.start.assert_called_once(); model_thread.start.assert_called_once()
        job_thread.join.assert_called_once(); model_thread.join.assert_called_once()
        log_error.assert_not_called()

    def test_visual_batch_is_separate_and_duplicate_safe(self):
        response = self.client.post('/api/episodes/e/visual-analysis')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['queued'], 2)
        self.assertEqual(self.client.post('/api/episodes/e/visual-analysis').json()['queued'], 0)
        jobs = db.all_rows("SELECT payload_json FROM jobs ORDER BY rowid")
        self.assertEqual([json.loads(j['payload_json'])['cuts'][0]['cut_id'] for j in jobs], ['c1','c2'])
        self.assertTrue(all(json.loads(j['payload_json'])['auto_consistency'] is False for j in jobs))

    def test_older_episode_analysis_route_queues_only_visual_jobs(self):
        response = self.client.post('/api/episodes/e/analyze')
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['queued'], 2)
        self.assertEqual(data['job_id'], data['job_ids'][0])
        jobs = db.all_rows('SELECT pipeline,payload_json FROM jobs ORDER BY rowid')
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            payload = json.loads(job['payload_json'])
            self.assertEqual((job['pipeline'], payload['workflow'], payload['auto_consistency']), ('analyze', 'single_cut_v2', False))
        repeated = self.client.post('/api/episodes/e/analyze').json()
        self.assertEqual((repeated['queued'], repeated['job_id']), (0, None))

    def test_single_cut_visual_queue_rejects_duplicate_and_allows_retry(self):
        first = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(first.status_code, 200, first.text)
        duplicate = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(self.client.post('/api/episodes/e/visual-analysis').json()['queued'], 1)
        self.assertEqual(self.client.post('/api/jobs/' + first.json()['job_id'] + '/cancel').status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c1/analyze').status_code, 200)

    def test_retry_rejects_an_already_queued_retry(self):
        first = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        self.client.post('/api/jobs/' + first + '/cancel')
        retry = self.client.post('/api/jobs/' + first + '/retry')
        self.assertEqual(retry.status_code, 200, retry.text)
        repeated = self.client.post('/api/jobs/' + first + '/retry')
        self.assertEqual(repeated.status_code, 409)
        self.assertEqual(len(db.all_rows("SELECT id FROM jobs WHERE pipeline='analyze' AND status='queued'")), 1)

    def test_local_text_batch_snapshots_and_uses_dependency_and_usage(self):
        self.seed_visual_results()
        response = self.client.post('/api/episodes/e/text-processing', json={'text_provider':'local','local_thinking': True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['cuts_queued'], 2)
        self.assertEqual(self.client.post('/api/episodes/e/text-processing', json={'text_provider':'local'}).json()['cuts_queued'], 0)
        job_ids = response.json()['job_ids']
        batch = [db.one('SELECT * FROM jobs WHERE id=?', (job_id,)) for job_id in job_ids]
        self.assertEqual([job['pipeline'] for job in batch], ['consistency_summary','regen','consistency_summary','regen'])
        self.assertEqual(json.loads(batch[1]['payload_json'])['depends_on_consistency'], job_ids[0])
        model = Mock(); model.ensure.return_value = 'http://127.0.0.1:1234'
        fields = {'summary':'summary','clear':'clear','confusing':'None identified.','human_review':'None required.','recommended_checks':'None required.'}
        replies = [{'choices':[{'message':{'content':json.dumps(fields)},'finish_reason':'stop'}], 'model':'path', 'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}},
                   {'choices':[{'message':{'content':analysis_tests.PRODUCTION_TEXT},'finish_reason':'stop'}], 'model':'path', 'usage':{'prompt_tokens':200,'completion_tokens':30,'total_tokens':230}}]
        requests=[]
        def respond(req, *args, **kwargs):
            requests.append(json.loads(req.data)); return replies.pop(0)
        with patch.object(local_text_runtime, 'runtime', model), patch.object(inference_worker, 'request_json', side_effect=respond), patch.dict(os.environ, {'OPENROUTER_API_KEY':''}):
            jid, owner = worker.claim_job(('consistency_summary','regen'))
            inference_worker.process_consistency_summary(jid, owner)
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?',(jid,))['status'], 'done')
            jid, owner = worker.claim_job(('consistency_summary','regen'))
            inference_worker.process_regen(jid, owner)
        row = db.one('SELECT * FROM analysis_results WHERE job_id=?', (job_ids[1],))
        self.assertEqual(row['model_id'], local_text_runtime.MODEL_ID)
        settings = json.loads(row['settings_json'])
        self.assertEqual(settings['provider'], 'local')
        self.assertTrue(settings['thinking'])
        self.assertEqual(settings['usage']['total_tokens'], 230)
        self.assertTrue(requests[0]['chat_template_kwargs']['enable_thinking'])
        self.assertIn('Consistency summary', requests[1]['messages'][1]['content'])
        consistency = db.one('SELECT settings_json FROM analysis_results WHERE job_id=?', (job_ids[0],))
        self.assertEqual(json.loads(consistency['settings_json'])['usage']['prompt_tokens'], 100)

    def test_failed_dependency_does_not_generate_a_brief(self):
        self.seed_visual_results()
        ids=self.client.post('/api/episodes/e/text-processing', json={}).json()['job_ids']
        self.client.post('/api/jobs/'+ids[0]+'/cancel')
        dependent = db.one('SELECT status,stage FROM jobs WHERE id=?',(ids[1],))
        self.assertEqual((dependent['status'], dependent['stage']), ('failed', 'Cancelled'))
        self.assertEqual(worker.claim_job(('regen',))[0], ids[3])

    def test_openrouter_usage_and_source_are_recorded(self):
        import io
        self.seed_visual_results()
        response = self.client.post('/api/cuts/c1/consistency-summary')
        self.assertEqual(response.status_code, 200, response.text)
        fields = {'summary':'summary','clear':'clear','confusing':'None identified.','human_review':'None required.','recommended_checks':'None required.'}
        payload = {'provider':'Example provider','model':'model/example','choices':[{'message':{'content':json.dumps(fields)},'finish_reason':'stop'}], 'usage':{'prompt_tokens':432,'completion_tokens':54,'total_tokens':486}}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY':'test'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
            inference_worker.process_consistency_summary(*worker.claim_job(('consistency_summary',)))
        row=db.one('SELECT * FROM analysis_results WHERE job_id=?',(response.json()['job_id'],))
        settings=json.loads(row['settings_json'])
        self.assertEqual((settings['provider'],row['model_id']),('openrouter','model/example'))
        self.assertEqual(settings['usage']['total_tokens'],486)
        self.assertEqual(settings['usage']['routed_provider'],'Example provider')
        self.assertGreaterEqual(settings['summary_generation_seconds'],0)

    def test_loading_wait_can_be_cancelled_during_download(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'model.gguf'
            model.with_name(model.name+'.partial').write_bytes(b'download-in-progress')
            with patch.dict(os.environ, {'QWEN36_MODEL_PATH': str(model)}):
                runtime=local_text_runtime.LocalTextRuntime()
                runtime.reconcile()
                self.assertEqual(db.one('SELECT state FROM local_text_model_state WHERE id=1')['state'],'downloading')
                def cancel(): raise inference_worker.SummaryCancelled()
                with self.assertRaises(inference_worker.SummaryCancelled):
                    runtime.ensure(cancel_check=cancel)
                self.assertIsNone(runtime.process)

    def test_stalled_partial_download_is_reported_when_idle(self):
        from pathlib import Path
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model.gguf'
            partial = model.with_name(model.name + '.partial')
            partial.write_bytes(b'incomplete')
            old = time.time() - 301
            os.utime(partial, (old, old))
            with patch.dict(os.environ, {'QWEN36_MODEL_PATH': str(model)}):
                runtime = local_text_runtime.LocalTextRuntime()
                runtime.reconcile()
                status = self.client.get('/api/local-text-model').json()
                self.assertEqual(status['state'], 'error')
                self.assertIn('stalled', status['detail'])

    def test_failed_legacy_analysis_can_retry_as_cancellable_local_text(self):
        old_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        with db.connect() as conn:
            old = conn.execute('SELECT payload_json FROM jobs WHERE id=?', (old_id,)).fetchone()
            data = json.loads(old['payload_json']); data.pop('workflow', None)
            conn.execute("UPDATE jobs SET status='failed',payload_json=? WHERE id=?", (json.dumps(data), old_id))
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json,completed) VALUES (?,'c1',?,0)", (old_id, json.dumps([{'time':.5,'description':'A room'}])))
        self.client.put('/api/text-generation-settings', json={'text_provider':'local', 'local_thinking': True})
        body = {'source_job_id':old_id, 'request_override':{'model':'openrouter/free','system':'system','summary_system':'summary','input':{'context':{},'sampled_frames':[]}}}
        response = self.client.post('/api/cuts/c1/retry-analysis-request', json=body)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.post('/api/cuts/c1/retry-analysis-request', json=body).status_code, 409)
        job_id=response.json()['job_id']
        payload=json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?',(job_id,))['payload_json'])
        self.assertTrue(payload['text_only'])
        self.assertEqual(payload['cuts'][0]['config']['text_provider'],'local')
        jid,owner=worker.claim_job(('analyze',))
        with patch.object(inference_worker,'describe_frames',side_effect=AssertionError('must reuse saved frames')), patch.object(inference_worker,'synthesize',return_value=({'visual_evidence':'a','text_evidence':'b','cut_summary':'c','_usage':{'total_tokens':8}},'local-model')) as synth, patch.dict(os.environ,{'OPENROUTER_API_KEY':''}):
            inference_worker.process(jid,owner)
        self.assertIn('cancel_check',synth.call_args.kwargs)
        result=db.one("SELECT settings_json FROM analysis_results WHERE job_id=? AND title='Cut summary'",(job_id,))
        settings=json.loads(result['settings_json'])
        self.assertEqual((settings['provider'],settings['usage']['total_tokens']),('local',8))

    def test_bulk_queues_skip_every_cut_in_older_multi_cut_jobs(self):
        self.seed_visual_results()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('legacy','e','analyze',?)", (json.dumps({'cuts':[{'cut_id':'c1'},{'cut_id':'c2'}]}),))
        self.assertEqual(self.client.post('/api/episodes/e/visual-analysis').json()['queued'],0)
        response=self.client.post('/api/episodes/e/text-processing',json={}).json()
        self.assertEqual((response['cuts_queued'],response['skipped']),(0,2))

    def test_concurrent_batch_requests_do_not_duplicate_work(self):
        from concurrent.futures import ThreadPoolExecutor
        self.seed_visual_results()
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses=list(pool.map(lambda _: self.client.post('/api/episodes/e/text-processing',json={'text_provider':'local'}), range(2)))
        self.assertEqual([r.status_code for r in responses],[200,200])
        self.assertEqual(sorted(r.json()['cuts_queued'] for r in responses),[0,2])
        self.assertEqual(len(db.all_rows("SELECT id FROM jobs WHERE status='queued'")),4)

    def test_idle_preload_yields_for_visual_or_remote_work_but_not_local_text(self):
        runtime=local_text_runtime.LocalTextRuntime()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('j','e','regen',?)",(json.dumps({'text_provider':'local'}),))
        self.assertFalse(runtime.pending_nonlocal_work())
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET payload_json=? WHERE id='j'",(json.dumps({'text_provider':'openrouter'}),))
        self.assertTrue(runtime.pending_nonlocal_work())
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET pipeline='analyze',payload_json=? WHERE id='j'",(json.dumps({'cuts':[{'cut_id':'c1'}]}),))
        self.assertTrue(runtime.pending_nonlocal_work())

    def test_cancelling_running_local_summary_stops_model_without_saving(self):
        self.client.put('/api/text-generation-settings', json={'text_provider':'local'})
        job_id=self.client.post('/api/cuts/c1/regenerate-summary',json={'mode':'regenerate'}).json()['job_id']
        jid,owner=worker.claim_job(('regen',))
        runtime=Mock(); runtime.ensure.return_value='http://127.0.0.1:1234'
        def cancel_during_request(request, cancel_check, **kwargs):
            self.assertEqual(self.client.post('/api/jobs/'+job_id+'/cancel').status_code,200)
            cancel_check()
        with patch.object(local_text_runtime,'runtime',runtime), patch.object(inference_worker,'request_json',side_effect=cancel_during_request), patch.dict(os.environ,{'OPENROUTER_API_KEY':''}):
            inference_worker.process_regen(jid,owner)
        runtime.stop.assert_called_once_with('Cancelled')
        self.assertEqual(db.one('SELECT stage FROM jobs WHERE id=?',(job_id,))['stage'],'Cancelled')
        self.assertIsNone(db.one('SELECT id FROM analysis_results WHERE job_id=?',(job_id,)))
