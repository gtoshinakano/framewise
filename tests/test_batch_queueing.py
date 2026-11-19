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

from app import cut_timeline, db, frame_model, inference_worker, main, media, single_cut_pipeline as pipeline, worker
from tests.test_analysis import visual_inputs


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

    def test_episode_requests_wait_and_select_new_cuts_with_queued_model_choices(self):
        from app import target_characters_api
        with db.connect() as conn:
            conn.execute('UPDATE cuts SET active=0,cut_index=-1-cut_index')
        hardcuts = worker.enqueue('e', 'hardcuts')
        self.assertEqual(main.put_text_generation_settings(main.TextGenerationSettings(openrouter_model='test/original'))['openrouter_model'], 'test/original')
        requests = [main.queue_episode_consistency_summaries('e'), main.queue_episode_review_answers('e'), main.queue_episode_text_chain('e'),
                    main.queue_episode_cut_summaries('e'), main.queue_episode_text_processing('e', main.BatchTextInput(openrouter_model='test/explicit')),
                    target_characters_api.queue_episode('e')]
        for result in requests:
            self.assertTrue(result['deferred'], result)
            job = db.one('SELECT payload_json,priority_class FROM jobs WHERE id=?', (result['job_ids'][0],))
            self.assertEqual(json.loads(job['payload_json'])['depends_on'], [hardcuts])
            self.assertEqual(job['priority_class'], 'batch')
        self.assertEqual(main.queue_episode_consistency_summaries('e')['job_ids'], requests[0]['job_ids'])
        self.assertIsNone(worker.claim_job(('episode_request',)))
        main.put_text_generation_settings(main.TextGenerationSettings(openrouter_model='test/later'))
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('new','e',0,0,2)")
            visual = worker.insert_job(conn, 'e', 'analyze', {'workflow': 'single_cut_v2', 'cuts': [{'cut_id': 'new'}], 'depends_on': [hardcuts]})
            conn.execute("UPDATE jobs SET status='done' WHERE id=?", (hardcuts,))
        while lease := worker.claim_job(('episode_request',)):
            worker.process(*lease)
            worker.release_lease(*lease)
        jobs = [json.loads(row['payload_json']) for row in self.jobs('consistency_summary')]
        self.assertEqual({item['cut_id'] for item in jobs}, {'new'})
        self.assertEqual({item['config']['openrouter_model'] for item in jobs}, {'test/original', 'test/explicit'})
        self.assertTrue(all(visual in item['depends_on'] for item in jobs))
        targets = [json.loads(row['payload_json']) for row in self.jobs('character_detect')]
        self.assertEqual([item['cut_id'] for item in targets], ['new'])

    def test_episode_visual_analysis_queues_one_job_for_every_cut(self):
        result = main.queue_episode_visual_analysis('e')
        self.assertEqual(result['queued'], 3)
        self.assertEqual(len(result['job_ids']), 1)
        payload = json.loads(self.jobs('analyze')[0]['payload_json'])
        self.assertEqual([item['cut_id'] for item in payload['cuts']], ['c1', 'c2', 'c3'])
        self.assertEqual(payload['workflow'], 'single_cut_v2')
        self.assertEqual(main.queue_episode_visual_analysis('e')['job_ids'], result['job_ids'])

    def test_visual_analysis_can_start_from_a_cut_number(self):
        result = main.queue_episode_visual_analysis('e', from_cut=2)  # the editor numbers cuts from 1
        self.assertEqual(result['queued'], 2)
        payload = json.loads(self.jobs('analyze')[0]['payload_json'])
        self.assertEqual([item['cut_id'] for item in payload['cuts']], ['c2', 'c3'])

    def seed_result(self, cut_id, stage, context=None, content=None):
        result_id = str(uuid.uuid4())
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,context_json) VALUES (?,?,'other',?,?,'model',?,?)",
                         (result_id, cut_id, stage, json.dumps(content or {'summary': 'x'}), json.dumps({'structured_type': stage}), json.dumps(context or {})))
        return result_id

    def test_only_missing_stages_run_and_finished_cuts_are_skipped(self):
        self.seed_result('c1', 'initial_frame', {'cut': {'start_time': 0, 'end_time': 2}})  # c1 only lacks the keyframes
        for stage in pipeline.ALL_STAGES:                    # c2 is finished
            self.seed_result('c2', stage, {'cut': {'start_time': 2, 'end_time': 4}})
        result = main.queue_episode_visual_analysis('e')     # c3 has nothing
        self.assertEqual(result['queued'], 3)
        job_id = result['job_ids'][0]
        self.assertEqual(visual_inputs(job_id, 'c1')['stages'], ['keyframe_analysis'])
        self.assertEqual(visual_inputs(job_id, 'c2')['skip'], 'Requested stages are current')
        self.assertEqual(list(visual_inputs(job_id, 'c3')['stages']), list(pipeline.ALL_STAGES))

    def test_subtitles_are_not_a_visual_stage_and_old_results_do_not_count(self):
        self.assertNotIn('subtitle_ocr', main.VISUAL_STAGES)
        self.assertNotIn('subtitle_ocr', pipeline.ALL_STAGES)
        self.seed_result('c1', 'subtitle_ocr', {'cut': {'start_time': 0, 'end_time': 2}})  # extracted per cut before the episode-wide step
        main.queue_episode_visual_analysis('e')
        item = next(i for i in json.loads(self.jobs('analyze')[0]['payload_json'])['cuts'] if i['cut_id'] == 'c1')
        self.assertNotIn('stages', item)
        self.assertNotIn('subtitle_input', item)

    def test_cut_context_at_start_has_subtitle_overlap(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,language,language_code,status,band_json) VALUES ('run','e','Portuguese','pt','complete',?)", (json.dumps({'top': .8, 'bottom': .95}),))
            conn.executemany("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES (?,?,?,?,?,?,?,?)",
                             [('l1', 'run', 'e', 0, 1.5, 2.5, 'Spans the boundary', 'high'), ('l2', 'run', 'e', 1, 3, 3.5, 'Inside c2', 'low')])
        job_id, = main.queue_episode_visual_analysis('e')['job_ids']
        items = {cut: visual_inputs(job_id, cut) for cut in ('c1', 'c2', 'c3')}
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
        self.assertEqual(events.count('keyframe-cut'), 3)

    def test_cuts_report_what_still_needs_human_action(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('hero','p','character','Hero')")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('r','c1',0,2,'[]','t')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id) VALUES ('d1','r','first','[1,1,2,2]','hero'),('d2','r','first','[1,1,2,2]',NULL)")
        for stage in ('initial_frame', 'video_analysis', 'consistency_summary'):  # c1 was analysed before keyframes existed
            self.seed_result('c1', stage)
        keyframes = self.seed_result('c2', 'keyframe_analysis')                   # c2 has keyframes, its beats are not written yet
        self.seed_result('c2', 'initial_frame')
        self.seed_result('c2', 'video_analysis')                                   # an older video analysis does not stand in for the beats
        self.seed_result('c3', 'video_analysis')                                   # c3 lacks its first frame
        cuts = {cut['id']: cut for cut in main.episode_response('e')['cuts']}
        self.assertTrue(cuts['c1']['has_boxes'] and cuts['c1']['has_characters'] and cuts['c1']['has_unlabeled_boxes'])
        self.assertTrue(cuts['c1']['has_visual_results'] and cuts['c1']['has_consistency_summary'])
        self.assertFalse(cuts['c2']['has_boxes'] or cuts['c2']['has_unlabeled_boxes'] or cuts['c2']['has_consistency_summary'])
        self.assertFalse(cuts['c2']['has_visual_results'])
        self.assertFalse(cuts['c3']['has_visual_results'] or cuts['c3']['has_consistency_summary'])
        self.seed_result('c2', 'performance_beats', content={'beats': [], 'based_on': {'keyframe_analysis': 'older'}})
        self.assertFalse(next(cut for cut in main.episode_response('e')['cuts'] if cut['id'] == 'c2')['has_visual_results'])
        self.seed_result('c2', 'performance_beats', content={'beats': [], 'based_on': {'keyframe_analysis': keyframes}})
        self.assertTrue(next(cut for cut in main.episode_response('e')['cuts'] if cut['id'] == 'c2')['has_visual_results'])

    def test_character_readiness_is_checked_before_analysis_starts(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c4','e',3,6,8,6,7.9)")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('hero','p','character','Hero')")
            conn.execute("UPDATE cuts SET no_characters=1 WHERE id='c1'")                      # marked: settled
            for run, cut, start, end in (('r2', 'c2', 2, 4), ('r3', 'c3', 4, 6)):
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES (?,?,?,?,'[{\"position\":\"first\",\"time\":0,\"image_size\":[100,80]}]','t')", (run, cut, start, end))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id) VALUES ('d2','r2','first','[1,1,2,2]','hero')")  # c2: all named
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id) VALUES ('d3','r3','first','[1,1,2,2]',NULL)")    # c3: unknown box
        result = main.queue_episode_visual_analysis('e', from_cut=2, characters_resolved=True)  # c4 was never detected
        self.assertEqual(result['queued'], 3)
        job_id = result['job_ids'][0]
        queued = [item['cut_id'] for item in json.loads(self.jobs('analyze')[0]['payload_json'])['cuts']]
        self.assertEqual(queued, ['c2', 'c3', 'c4'])
        self.assertNotIn('skip', visual_inputs(job_id, 'c2'))
        self.assertEqual(visual_inputs(job_id, 'c3')['skip'], 'Character labels still need review')
        self.assertEqual(visual_inputs(job_id, 'c4')['skip'], 'Character boxes are missing')

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

    def test_timeline_reads_only_this_cuts_jobs_of_its_episode(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e2','p','Other','o.mp4','.mp4','ready','{}')")
            for job_id, episode, pipeline_name, payload in (
                    ('summary', 'e', 'regen', {'cut_id': 'c1', 'result_type': 'summary'}),
                    ('boxes', 'e', 'character_detect', {'cut_id': 'c1', 'phase': 'boxes'}),
                    ('neighbour', 'e', 'regen', {'cut_id': 'c2', 'result_type': 'summary', 'context': {'previous': 'c1'}}),
                    ('elsewhere', 'e2', 'regen', {'cut_id': 'c1', 'result_type': 'summary'})):
                conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,stage,payload_json) VALUES (?,?,?,'done','',?)", (job_id, episode, pipeline_name, json.dumps(payload)))
        main.queue_episode_visual_analysis('e')  # queued: listed in the payload, not yet in analysis_job_cuts
        events = [event for event in cut_timeline.build('c1')['events'] if event['status'] != 'pending']
        self.assertEqual(sorted((event['kind'], event['job_id']) for event in events if event['job_id']),
                         [('character_boxes', 'boxes'), ('cut_summary', 'summary'), ('visual_analysis', self.jobs('analyze')[0]['id'])])

    def test_detect_all_characters_queues_full_phase_jobs_in_one_batch(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET no_characters=1 WHERE id='c2'")
        result = main.queue_all_character_detection('e')
        self.assertEqual(result['queued'], 2)
        payloads = [json.loads(job['payload_json']) for job in self.jobs('character_detect')]
        self.assertEqual({p['phase'] for p in payloads}, {'full'})
        self.assertEqual([p['cut_id'] for p in payloads], ['c1', 'c3'])
        self.assertEqual(len({p['batch_id'] for p in payloads}), 1)
        self.assertEqual(main.queue_all_character_detection('e')['job_ids'], result['job_ids'])

    def process_with_fake_models(self, job_id, owner, events, describe=None):
        """Run a queued analysis job through the real worker with stand-ins for the GPU models."""
        @contextmanager
        def server(*_args, **_kwargs):
            events.append('server-start')
            yield 'server'
            events.append('server-stop')

        def load_pose():
            events.append('pose-load')
            return None, {'missing': 'Not installed'}

        def save_generated(conn, cut_id, job, _owner, result_type, data, *_args, **_kwargs):
            result_id = str(uuid.uuid4())
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,job_id) VALUES (?,?,'other',?,?,'model',?)",
                         (result_id, cut_id, result_type, json.dumps(data), job))
            events.append('save:' + result_type)
            return result_id

        def analyze(cut_run, _server, _scratch, model, _pose, _provenance):
            if not cut_run.needs('keyframe_analysis'):
                return
            with cut_run.stage('keyframe_analysis', 'Keyframe analysis'):
                events.append('keyframe-cut')
                cut_run.save('keyframe_analysis', {'contract': 'keyframes-v1', 'frames': [], 'segments': []}, model.model_id, 'prompt', {})

        def default_describe(*_args, **_kwargs):
            events.append('frame')
            return {'summary': 'x'}

        with patch.object(pipeline, 'ensure_models_available'), patch.object(pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=AssertionError('the WeVisOne model belongs to the subtitle step, not to cut analysis')), \
             patch.object(frame_model, 'frame_server', server), \
             patch.object(frame_model, 'describe_frame', side_effect=describe or default_describe), \
             patch.object(pipeline.keyframe_analysis, 'load_pose', load_pose), patch.object(pipeline.keyframe_analysis, 'analyze', analyze), \
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
        shape = [event for event in events if not event.startswith('save:')]
        # One server for the first frames, then DWPose once and one server for every cut's keyframes.
        self.assertEqual(shape, ['server-start'] + ['frame'] * 3 + ['server-stop'] + ['pose-load', 'server-start'] + ['keyframe-cut'] * 3 + ['server-stop'])

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
        self.assertEqual(events.count('keyframe-cut'), 2)  # the failed cut is skipped by the later stages

        retry_id = main.retry_analysis(job_id)['job_id']
        retry_job, retry_owner = worker.claim_job(('analyze',))
        self.assertEqual(retry_job, retry_id)
        retry_events = []
        self.process_with_fake_models(retry_id, retry_owner, retry_events)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (retry_id,))['status'], 'done')
        self.assertEqual(retry_events.count('keyframe-cut'), 1)  # only the cut that failed is redone
        self.assertEqual(retry_events.count('frame'), 1)


if __name__ == '__main__':
    unittest.main()
