import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from app import db, inference_worker, job_inputs, keyframe_analysis, single_cut_pipeline, worker
import tests.test_analysis as analysis_tests
from tests.test_keyframe_analysis import answer_for, fake_models, seed_selection


class VisualStageJobsTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def queue_stage(self, stage):
        response = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': stage})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['job_id']

    def add_subtitle_run(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,language,language_code,status,band_json) VALUES ('run','e','Portuguese','pt','complete',?)", (json.dumps({'top': .8, 'bottom': .96}),))
            conn.execute("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES ('l','run','e',0,0.5,1.5,'Olá','high')")

    def test_subtitle_is_not_a_stage_cut_analysis_can_run(self):
        response = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'subtitle_ocr'})
        self.assertEqual(response.status_code, 422)

    def test_keyframe_stage_is_told_the_episode_strip_and_runs_alone(self):
        self.add_subtitle_run()
        seed_selection()
        job_id = self.queue_stage('keyframe_analysis')
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('subtitle model loaded')), \
             patch.object(single_cut_pipeline.frame_model, 'describe_frame', side_effect=AssertionError('first frame described')), \
             fake_models() as calls:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertIn('A burned-in subtitle strip covers 80-96% of the image height; ignore it.', keyframe_analysis.keyframe_prompts.prompt_text(calls[0]))
        self.assertEqual([row['type'] for row in self.client.get('/api/cuts/c1/structured-analysis').json()], ['keyframe_analysis'])
        provenance = json.loads(db.one('SELECT context_json FROM analysis_results WHERE job_id=?', (job_id,))['context_json'])
        self.assertEqual(provenance['subtitle_strip']['language'], 'Portuguese')

    def test_distinct_requests_queue_and_duplicates_reuse_the_job(self):
        first = self.queue_stage('keyframe_analysis')
        second = self.queue_stage('initial_frame')
        self.assertNotEqual(first, second)
        self.assertEqual(self.queue_stage('keyframe_analysis'), first)
        full = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(full.status_code, 200)
        self.assertNotIn(full.json()['job_id'], (first, second))
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage':'unknown'}).status_code, 422)
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage':'video_analysis'}).status_code, 410)

    def test_initial_frame_runs_without_subtitle_or_video_models(self):
        self.add_subtitle_run()
        job_id = self.queue_stage('initial_frame')
        frame = {'composition': 'wide', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'visible_characters': [], 'other_visual_details': '', 'uncertainties': ''}
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('subtitle model loaded')), \
             patch.object(keyframe_analysis, 'load_pose', side_effect=AssertionError('keyframe stage started')), \
             patch.object(single_cut_pipeline.media, 'extract_cut_frame', side_effect=lambda source, target, sample, *a, **kw: sample), \
             patch.object(single_cut_pipeline.frame_model, 'frame_server', return_value=nullcontext('server')), \
             patch.object(single_cut_pipeline.frame_model, 'describe_frame', return_value=frame) as describe:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual([row['type'] for row in self.client.get('/api/cuts/c1/structured-analysis').json()], ['initial_frame'])
        # The frame model is told the one subtitle language and where the strip sits on the 1280-wide frame.
        self.assertEqual(describe.call_args.kwargs['subtitle_language'], 'Portuguese')
        self.assertEqual(describe.call_args.args[2], [[0, 576, 1280, 691]])

    def test_selected_stage_starts_from_the_latest_result(self):
        seed_selection()
        job_id = self.queue_stage('keyframe_analysis')
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES ('edited','c1','cut_description','Keyframe analysis','{}',?)", (json.dumps({'structured_type':'keyframe_analysis'}),))
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), fake_models():
            inference_worker.process(*worker.claim_job(('analyze',)))
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'done')
        revised = db.one("SELECT parent_id FROM analysis_results WHERE job_id=?", (job_id,))
        self.assertEqual(revised['parent_id'], 'edited')

    def test_visual_request_waits_for_character_detection_without_capturing_boxes(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('detect','e','character_detect',?)",
                         (json.dumps({'cut_id': 'c1', 'workflow': 'anime_bulk_v1'}),))
        job_id = self.queue_stage('initial_frame')
        queued = db.one('SELECT priority_class,payload_json FROM jobs WHERE id=?', (job_id,))
        payload = json.loads(queued['payload_json'])
        self.assertEqual((queued['priority_class'], payload['depends_on']), ('interactive', ['detect']))
        self.assertNotIn('context', payload['cuts'][0])
        self.assertIsNone(worker.claim_job(('analyze',)))
        self.assertEqual(self.client.get('/api/episodes/e/queue').json()['waiting'][-1]['waiting_for'], ['Character detection'])

    def test_an_edit_during_analysis_restarts_with_current_detection_inputs(self):
        job_id = self.queue_stage('initial_frame')
        seen = []

        def describe(*args, **kwargs):
            seen.append(kwargs['people'])
            if len(seen) == 1:
                response = self.client.put('/api/cuts/c1/no-characters', json={'no_characters': True})
                self.assertEqual(response.status_code, 200, response.text)
            return {'composition': 'wide'}

        with db.connect() as conn:
            conn.execute("""INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,model_revision)
                VALUES ('detect','c1',0,2,?,'detector','revision')""", (json.dumps([{'position': 'first', 'time': 0, 'image_size': [100, 100]}]),))
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source)
                VALUES ('box','detect','first','[10,10,50,75]','ctx','manual')""")
        with patch('app.worker.time.time', return_value=100) as clock, \
             patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(single_cut_pipeline.media, 'extract_cut_frame', return_value=0), \
             patch.object(single_cut_pipeline.frame_model, 'frame_server', return_value=nullcontext('server')), \
             patch.object(single_cut_pipeline.frame_model, 'describe_frame', side_effect=describe):
            claim = worker.claim_job(('analyze',))
            inference_worker.process(*claim)
            job = db.one('SELECT status,stage,attempt_count FROM jobs WHERE id=?', (job_id,))
            self.assertEqual(job, {'status': 'queued', 'stage': job_inputs.RESTART_STAGE, 'attempt_count': 0})
            self.assertIsNone(db.one('SELECT id FROM analysis_results WHERE job_id=?', (job_id,)))
            worker.release_lease(*claim)
            self.assertIsNone(worker.claim_job(('analyze',)))
            clock.return_value += job_inputs.QUIET_SECONDS
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual([person['name'] for person in seen[0]], ['Hero'])
        self.assertEqual(seen[1], [])
        result = db.one('SELECT context_json FROM analysis_results WHERE job_id=?', (job_id,))
        provenance = json.loads(result['context_json'])
        self.assertTrue(provenance['cut']['no_characters'])
        self.assertEqual(len(provenance['input_fingerprint']), 64)

    def test_cancellation_keeps_the_described_windows_for_the_retry(self):
        seed_selection(count=9)
        job_id = self.queue_stage('keyframe_analysis')
        asked = []

        def cancel_after_first(messages, schema):
            asked.append(schema)
            if len(asked) == 2:
                self.assertEqual(self.client.post(f'/api/jobs/{job_id}/cancel').status_code, 200)
            return answer_for(schema)
        with patch.object(keyframe_analysis, 'PARALLEL', 1), patch.object(single_cut_pipeline, 'ensure_models_available'), \
             patch.object(single_cut_pipeline, 'release_gpu_models'), fake_models(ask=cancel_after_first):
            claim = worker.claim_job(('analyze',))
            inference_worker.process(*claim)
            worker.release_lease(*claim)
        job = db.one('SELECT status,stage FROM jobs WHERE id=?', (job_id,))
        self.assertEqual((job['status'], job['stage']), ('cancelled', 'Cancelled'))
        self.assertEqual(db.one('SELECT status FROM analysis_job_stages WHERE job_id=?', (job_id,))['status'], 'cancelled')
        self.assertEqual(self.client.get('/api/cuts/c1/structured-analysis').json(), [])
        saved = db.one('SELECT COUNT(*) AS n FROM keyframe_analysis_windows WHERE ok=1')['n']
        self.assertGreaterEqual(saved, 1)
        response = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(response.status_code, 200, response.text)
        retry_id = response.json()['job_id']
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), fake_models() as calls:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (retry_id,))['status'], 'done')
        self.assertEqual(len(calls), 2 - saved)
        self.assertEqual([r['type'] for r in self.client.get('/api/cuts/c1/structured-analysis').json()], ['keyframe_analysis'])
