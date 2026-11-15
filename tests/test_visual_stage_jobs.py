import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from app import db, inference_worker, media, single_cut_pipeline, worker
import tests.test_analysis as analysis_tests


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

    def test_video_stage_is_told_the_episode_language_and_strip_and_runs_alone(self):
        self.add_subtitle_run()
        proxy = media.episode_dir('p', 'e') / 'proxy.mp4'
        proxy.touch()
        job_id = self.queue_stage('video_analysis')
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('subtitle model loaded')), \
             patch.object(single_cut_pipeline.frame_model, 'frame_server', side_effect=AssertionError('frame model loaded')), \
             patch.object(single_cut_pipeline.video_model, 'analyze_video', return_value={'windows': []}) as video:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(video.call_args.kwargs['subtitle_strip'], {'language': 'Portuguese', 'top': .8, 'bottom': .96})
        self.assertEqual(video.call_args.args[0], proxy)
        self.assertEqual([row['type'] for row in self.client.get('/api/cuts/c1/structured-analysis').json()], ['video_analysis'])
        provenance = json.loads(db.one('SELECT context_json FROM analysis_results WHERE job_id=?', (job_id,))['context_json'])
        self.assertEqual(provenance['subtitle_strip']['language'], 'Portuguese')

    def test_distinct_stages_can_queue_but_duplicate_and_full_run_cannot(self):
        first = self.queue_stage('video_analysis')
        second = self.queue_stage('initial_frame')
        self.assertNotEqual(first, second)
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage':'video_analysis'}).status_code, 409)
        self.assertEqual(self.client.post('/api/cuts/c1/analyze').status_code, 409)
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage':'unknown'}).status_code, 422)

    def test_initial_frame_runs_without_subtitle_or_video_models(self):
        self.add_subtitle_run()
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
        # The frame model is told the one subtitle language and where the strip sits on the 1280-wide frame.
        self.assertEqual(describe.call_args.kwargs['subtitle_language'], 'Portuguese')
        self.assertEqual(describe.call_args.args[2], [[0, 576, 1280, 691]])

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
        self.assertEqual((job['status'],job['stage']), ('cancelled','Cancelled'))
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
