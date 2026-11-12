"""Episode-level batch endpoints queue one job per step and the worker runs it model by model.

These call the endpoint functions directly, so they need no HTTP test client.
"""
import json
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from app import (cut_timeline, db, frame_model, inference_worker, main, media, single_cut_pipeline as pipeline,
                 video_model, worker)


class BatchQueueingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c1','e',0,0,2,0,1.9),('c2','e',1,2,4,2,3.9),('c3','e',2,4,6,4,5.9)")

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def jobs(self, pipeline_name):
        return db.all_rows('SELECT * FROM jobs WHERE pipeline=? ORDER BY rowid', (pipeline_name,))

    def test_episode_visual_analysis_queues_one_job_for_every_cut(self):
        result = main.queue_episode_visual_analysis('e')
        self.assertEqual(result['queued'], 3)
        self.assertEqual(len(result['job_ids']), 1)
        payload = json.loads(self.jobs('analyze')[0]['payload_json'])
        self.assertEqual([item['cut_id'] for item in payload['cuts']], ['c1', 'c2', 'c3'])
        self.assertEqual(payload['workflow'], 'single_cut_v2')
        self.assertEqual(main.queue_episode_visual_analysis('e'), {'queued': 0, 'skipped': 0, 'blocked': {'unknown_characters': [], 'not_detected': []}, 'job_ids': []})

    def test_visual_analysis_can_start_from_a_cut_number(self):
        result = main.queue_episode_visual_analysis('e', from_cut=2)  # the editor numbers cuts from 1
        self.assertEqual(result['queued'], 2)
        payload = json.loads(self.jobs('analyze')[0]['payload_json'])
        self.assertEqual([item['cut_id'] for item in payload['cuts']], ['c2', 'c3'])

    def seed_result(self, cut_id, stage, context=None):
        content = {'summary': 'x'}
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,context_json) VALUES (?,?,'other',?,?,'model',?,?)",
                         (str(uuid.uuid4()), cut_id, stage, json.dumps(content), json.dumps({'structured_type': stage}), json.dumps(context or {})))

    def test_only_missing_stages_are_queued_and_finished_cuts_are_skipped(self):
        self.seed_result('c1', 'initial_frame', {'cut': {'start_time': 0, 'end_time': 2}})  # c1 only lacks the video
        for stage in pipeline.ALL_STAGES:                    # c2 is finished
            self.seed_result('c2', stage, {'cut': {'start_time': 2, 'end_time': 4}})
        result = main.queue_episode_visual_analysis('e')     # c3 has nothing
        self.assertEqual((result['queued'], result['skipped']), (2, 1))
        items = {item['cut_id']: item for item in json.loads(self.jobs('analyze')[0]['payload_json'])['cuts']}
        self.assertEqual(sorted(items), ['c1', 'c3'])
        self.assertEqual(items['c1']['stages'], ['video_analysis'])  # the last frame is no longer a stage
        self.assertNotIn('stages', items['c3'])              # nothing done: every stage runs

    def test_subtitles_are_not_a_visual_stage_and_old_results_do_not_count(self):
        self.assertNotIn('subtitle_ocr', main.VISUAL_STAGES)
        self.assertNotIn('subtitle_ocr', pipeline.ALL_STAGES)
        self.seed_result('c1', 'subtitle_ocr', {'cut': {'start_time': 0, 'end_time': 2}})  # extracted per cut before the episode-wide step
        main.queue_episode_visual_analysis('e')
        item = next(i for i in json.loads(self.jobs('analyze')[0]['payload_json'])['cuts'] if i['cut_id'] == 'c1')
        self.assertNotIn('stages', item)
        self.assertNotIn('subtitle_input', item)

    def test_queued_cut_context_carries_the_episode_subtitles_that_overlap_it(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,language,language_code,status,band_json) VALUES ('run','e','Portuguese','pt','complete',?)", (json.dumps({'top': .8, 'bottom': .95}),))
            conn.executemany("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES (?,?,?,?,?,?,?,?)",
                             [('l1', 'run', 'e', 0, 1.5, 2.5, 'Spans the boundary', 'high'), ('l2', 'run', 'e', 1, 3, 3.5, 'Inside c2', 'low')])
        main.queue_episode_visual_analysis('e')
        items = {item['cut_id']: item for item in json.loads(self.jobs('analyze')[0]['payload_json'])['cuts']}
        first, second = items['c1']['context']['subtitles'], items['c2']['context']['subtitles']
        self.assertEqual((first['language'], first['strip'], first['extracted']), ('Portuguese', {'top': .8, 'bottom': .95}, True))
        self.assertEqual([(line['text'], line['crosses_cut']) for line in first['lines']], [('Spans the boundary', True)])
        self.assertEqual([(line['text'], line['crosses_cut']) for line in second['lines']], [('Spans the boundary', True), ('Inside c2', False)])
        self.assertEqual(items['c3']['context']['subtitles']['lines'], [])

    def test_only_missing_can_be_turned_off_to_redo_everything(self):
        for stage in pipeline.ALL_STAGES:
            self.seed_result('c2', stage, {'cut': {'start_time': 2, 'end_time': 4}})
        result = main.queue_episode_visual_analysis('e', only_missing=False)
        self.assertEqual((result['queued'], result['skipped']), (3, 0))

    def test_batch_runs_only_the_stages_each_cut_still_needs(self):
        self.seed_result('c1', 'initial_frame', {'cut': {'start_time': 0, 'end_time': 2}})
        main.queue_episode_visual_analysis('e')
        job_id, owner = worker.claim_job(('analyze',))
        events = []
        self.process_with_fake_models(job_id, owner, events)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(events.count('frame'), 2)  # one first frame per cut; c1 already had its own
        self.assertEqual(events.count('video-cut'), 3)

    def test_cuts_report_what_still_needs_human_action(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('hero','p','character','Hero')")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('r','c1',0,2,'[]','t')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id) VALUES ('d1','r','first','[1,1,2,2]','hero'),('d2','r','last','[1,1,2,2]',NULL)")
        for stage in ('video_analysis', 'consistency_summary'):
            self.seed_result('c1', stage)
        self.seed_result('c2', 'video_analysis')
        cuts = {cut['id']: cut for cut in main.episode_response('e')['cuts']}
        self.assertTrue(cuts['c1']['has_boxes'] and cuts['c1']['has_characters'] and cuts['c1']['has_unlabeled_boxes'])
        self.assertTrue(cuts['c1']['has_visual_results'] and cuts['c1']['has_consistency_summary'])
        self.assertFalse(cuts['c2']['has_boxes'] or cuts['c2']['has_unlabeled_boxes'] or cuts['c2']['has_consistency_summary'])
        self.assertTrue(cuts['c2']['has_visual_results'])
        self.assertFalse(cuts['c3']['has_visual_results'] or cuts['c3']['has_consistency_summary'])

    def test_characters_must_be_settled_before_cuts_are_queued_from_a_cut(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c4','e',3,6,8,6,7.9)")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('hero','p','character','Hero')")
            conn.execute("UPDATE cuts SET no_characters=1 WHERE id='c1'")                      # marked: settled
            for run, cut, start, end in (('r2', 'c2', 2, 4), ('r3', 'c3', 4, 6)):
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES (?,?,?,?,'[]','t')", (run, cut, start, end))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id) VALUES ('d2','r2','first','[1,1,2,2]','hero')")  # c2: all named
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id) VALUES ('d3','r3','first','[1,1,2,2]',NULL)")    # c3: unknown box
        result = main.queue_episode_visual_analysis('e', from_cut=2, characters_resolved=True)  # c4 was never detected
        self.assertEqual(result['blocked'], {'unknown_characters': [3], 'not_detected': [4]})
        self.assertEqual(result['queued'], 1)
        queued = [item['cut_id'] for item in json.loads(self.jobs('analyze')[0]['payload_json'])['cuts']]
        self.assertEqual(queued, ['c2'])                                                          # c1 is before from_cut

    def test_without_the_prerequisite_nothing_is_blocked(self):
        result = main.queue_episode_visual_analysis('e')
        self.assertEqual(result['blocked'], {'unknown_characters': [], 'not_detected': []})
        self.assertEqual(result['queued'], 3)

    def test_single_cut_trigger_still_queues_a_one_cut_job(self):
        main.analyze_cut('c2')
        payload = json.loads(self.jobs('analyze')[0]['payload_json'])
        self.assertEqual([item['cut_id'] for item in payload['cuts']], ['c2'])

    def test_batch_job_appears_in_every_cut_timeline(self):
        main.queue_episode_visual_analysis('e')
        for cut_id in ('c1', 'c3'):
            kinds = [event['kind'] for event in cut_timeline.build(cut_id)['events'] if event['status'] != 'pending']
            self.assertIn('visual_analysis', kinds, cut_id)

    def test_detect_all_characters_queues_full_phase_jobs_in_one_batch(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET no_characters=1 WHERE id='c2'")
        result = main.queue_all_character_detection('e')
        self.assertEqual(result['queued'], 2)
        payloads = [json.loads(job['payload_json']) for job in self.jobs('character_detect')]
        self.assertEqual({p['phase'] for p in payloads}, {'full'})
        self.assertEqual([p['cut_id'] for p in payloads], ['c1', 'c3'])
        self.assertEqual(len({p['batch_id'] for p in payloads}), 1)
        self.assertEqual(main.queue_all_character_detection('e')['queued'], 0)

    def process_with_fake_models(self, job_id, owner, events, describe=None):
        """Run a queued analysis job through the real worker with stand-ins for the GPU models."""
        @contextmanager
        def server(*_args, **_kwargs):
            events.append('server-start')
            yield 'server'
            events.append('server-stop')

        class Runtime(video_model.VideoRuntime):
            def load(self, _path):
                events.append('video-load')
                return None, None, None

            def close(self):
                events.append('video-close')

        def save_generated(conn, cut_id, job, _owner, result_type, data, *_args, **_kwargs):
            result_id = str(uuid.uuid4())
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,job_id) VALUES (?,?,'other',?,?,'model',?)",
                         (result_id, cut_id, result_type, json.dumps(data), job))
            events.append('save:' + result_type)
            return result_id

        def analyze(_original, _cut, _config, path, _scratch, runtime=None, **_kwargs):
            runtime.load(path)
            events.append('video-cut')
            return {'windows': [], 'usage': {}}

        def default_describe(*_args, **_kwargs):
            events.append('frame')
            return {'summary': 'x'}

        with patch.object(pipeline, 'ensure_models_available'), patch.object(pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('the WeVisOne model belongs to the subtitle step, not to cut analysis')), \
             patch.object(frame_model, 'frame_server', server), \
             patch.object(frame_model, 'describe_frame', side_effect=describe or default_describe), \
             patch.object(video_model, 'VideoRuntime', Runtime), patch.object(video_model, 'analyze_video', analyze), \
             patch.object(pipeline.character_detection, 'crop_frame', return_value=b''), \
             patch.object(pipeline.media, 'extract_cut_frame', return_value=1.0), \
             patch.object(pipeline.structured_results, 'save_generated', save_generated):
            inference_worker.process(job_id, owner)

    def test_worker_runs_each_model_once_across_the_queued_cuts(self):
        main.queue_episode_visual_analysis('e')
        job_id, owner = worker.claim_job(('analyze',))
        events = []
        self.process_with_fake_models(job_id, owner, events)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(db.one('SELECT SUM(completed) AS n FROM analysis_job_cuts WHERE job_id=?', (job_id,))['n'], 3)
        shape = [event for event in events if not event.startswith('save:') and event != 'video-load']
        self.assertEqual(shape, ['server-start'] + ['frame'] * 3 + ['server-stop'] + ['video-cut'] * 3 + ['video-close'])

    def test_one_failing_cut_does_not_stop_the_others_and_a_retry_finishes_only_it(self):
        main.queue_episode_visual_analysis('e')
        job_id, owner = worker.claim_job(('analyze',))
        events, calls = [], []

        def describe(_server, _image, _regions, hints, **kwargs):
            calls.append(kwargs['frame_label'])
            if len(calls) == 2:  # the second cut's first frame, like a truncated model answer
                raise ValueError('Response was truncated')
            events.append('frame')
            return {'summary': 'x'}

        with patch.object(pipeline, 'FRAME_PARALLEL', 1):  # the fake fails on the third call, so keep the order fixed
            self.process_with_fake_models(job_id, owner, events, describe)
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('1 of 3 cuts failed', job['error'])
        self.assertIn('cut 2: Response was truncated', job['error'])
        done = {row['cut_id'] for row in db.all_rows('SELECT cut_id FROM analysis_job_cuts WHERE job_id=? AND completed=1', (job_id,))}
        self.assertEqual(len(done), 2)
        self.assertEqual(events.count('video-cut'), 2)  # the failed cut is skipped by the later stages

        retry_id = main.retry_analysis(job_id)['job_id']
        retry_job, retry_owner = worker.claim_job(('analyze',))
        self.assertEqual(retry_job, retry_id)
        retry_events = []
        self.process_with_fake_models(retry_id, retry_owner, retry_events)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (retry_id,))['status'], 'done')
        self.assertEqual(retry_events.count('video-cut'), 1)  # only the cut that failed is redone
        self.assertEqual(retry_events.count('frame'), 1)


if __name__ == '__main__':
    unittest.main()
