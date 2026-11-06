import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from app import db, inference_worker, single_cut_pipeline, worker
import tests.test_analysis as analysis_tests


class VisualStageJobsTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def queue_stage(self, stage):
        response = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': stage})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['job_id']

    def test_subtitle_only_runs_without_frame_or_video_models(self):
        job_id = self.queue_stage('subtitle_ocr')
        subtitles = {'samples': [], 'events': [], 'ocr_performed': False}
        with patch.object(single_cut_pipeline, 'release_gpu_models'), patch.object(inference_worker, 'get_model', return_value=object()), \
             patch.object(single_cut_pipeline.subtitle_model, 'inspect_subtitles', return_value=subtitles) as ocr, \
             patch.object(single_cut_pipeline.frame_model, 'frame_server', side_effect=AssertionError('frame model loaded')), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=AssertionError('video model loaded')):
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual([row['type'] for row in self.client.get('/api/cuts/c1/structured-analysis').json()], ['subtitle_ocr'])
        self.assertEqual(ocr.call_count, 1)

    def test_video_only_uses_queued_subtitle_snapshot_and_leaves_other_results(self):
        old = {'samples': [{'time': 0, 'regions': [[1, 2, 3, 4]], 'image_size': [10, 10]}], 'events': []}
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,context_json) VALUES ('subs','c1','other','Subtitle OCR',?,?,?)", (json.dumps(old), json.dumps({'structured_type':'subtitle_ocr'}), json.dumps({'cut': {'start_time':0, 'end_time':2}})))
        job_id = self.queue_stage('video_analysis')
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET content=? WHERE id='subs'", (json.dumps({'samples': [], 'events': []}),))
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('subtitle model loaded')), \
             patch.object(single_cut_pipeline.frame_model, 'frame_server', side_effect=AssertionError('frame model loaded')), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', return_value={'windows': []}) as video:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(video.call_args.kwargs['subtitle_samples'], old['samples'])
        result = db.one("SELECT context_json FROM analysis_results WHERE job_id=?", (job_id,))
        self.assertEqual(json.loads(result['context_json'])['subtitle_source_result_id'], 'subs')
        self.assertEqual({row['type'] for row in self.client.get('/api/cuts/c1/structured-analysis').json()}, {'subtitle_ocr', 'video_analysis'})

    def test_distinct_stages_can_queue_but_duplicate_and_full_run_cannot(self):
        first = self.queue_stage('video_analysis')
        second = self.queue_stage('subtitle_ocr')
        self.assertNotEqual(first, second)
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage':'video_analysis'}).status_code, 409)
        self.assertEqual(self.client.post('/api/cuts/c1/analyze').status_code, 409)
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage':'unknown'}).status_code, 422)

    def test_initial_frame_runs_without_subtitle_or_video_models(self):
        job_id = self.queue_stage('initial_frame')
        frame = {'composition': 'wide', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'visible_characters': [], 'other_visual_details': '', 'uncertainties': ''}
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('subtitle model loaded')), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=AssertionError('video model loaded')), \
             patch.object(single_cut_pipeline.media, 'extract_cut_frame', side_effect=lambda source, target, sample, *a, **kw: sample), \
             patch.object(single_cut_pipeline.frame_model, 'frame_server', return_value=nullcontext('server')), \
             patch.object(single_cut_pipeline.frame_model, 'describe_frame', return_value=frame) as describe:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual([row['type'] for row in self.client.get('/api/cuts/c1/structured-analysis').json()], ['initial_frame'])
        self.assertEqual(describe.call_args.args[2], [])

    def test_old_subtitle_timing_is_not_reused(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,context_json) VALUES ('old-subs','c1','other','Subtitle OCR',?,?,?)", (json.dumps({'samples':[{'time':0}], 'events':[]}), json.dumps({'structured_type':'subtitle_ocr'}), json.dumps({'cut': {'start_time':0, 'end_time':3}})))
        job_id = self.queue_stage('video_analysis')
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])
        self.assertNotIn('subtitle_input', payload['cuts'][0])

    def test_selected_stage_fails_if_its_result_changes_while_queued(self):
        job_id = self.queue_stage('video_analysis')
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES ('edited','c1','cut_description','Video analysis','{}',?)", (json.dumps({'structured_type':'video_analysis'}),))
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', return_value={'windows': []}):
            inference_worker.process(*worker.claim_job(('analyze',)))
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('result changed', job['error'])
        self.assertEqual(db.one("SELECT superseded_by FROM analysis_results WHERE id='edited'")['superseded_by'], None)

    def test_stage_does_not_save_if_cut_timing_changes_during_processing(self):
        job_id = self.queue_stage('video_analysis')
        def video(*args, **kwargs):
            with db.connect() as conn:
                conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
            return {'windows': []}
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=video):
            inference_worker.process(*worker.claim_job(('analyze',)))
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('cut changed', job['error'])
        self.assertEqual(self.client.get('/api/cuts/c1/structured-analysis').json(), [])

    def test_cancellation_preserves_completed_video_windows_for_retry(self):
        job_id = self.queue_stage('video_analysis')
        window = {'source': {'source_start':0, 'source_end':2}, 'analysis': {'events':[]}}
        def cancel_video(*args, save_window, cancel_check, **kwargs):
            save_window(0, window)
            self.assertEqual(self.client.post(f'/api/jobs/{job_id}/cancel').status_code, 200)
            cancel_check()
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=cancel_video):
            inference_worker.process(*worker.claim_job(('analyze',)))
        job = db.one('SELECT status,stage FROM jobs WHERE id=?', (job_id,))
        self.assertEqual((job['status'],job['stage']), ('failed','Cancelled'))
        self.assertEqual(db.one('SELECT status FROM analysis_job_stages WHERE job_id=?', (job_id,))['status'], 'cancelled')
        self.assertEqual(len(self.client.get('/api/cuts/c1/structured-analysis').json()), 0)
        response = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(response.status_code, 200, response.text)
        retry_id = response.json()['job_id']
        def resume(*args, load_window, **kwargs):
            self.assertEqual(load_window(0, 0, 2), window)
            return {'windows':[window]}
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=resume):
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (retry_id,))['status'], 'done')
        self.assertEqual([r['type'] for r in self.client.get('/api/cuts/c1/structured-analysis').json()], ['video_analysis'])
