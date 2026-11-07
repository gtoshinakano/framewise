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
                 subtitle_model, video_model, worker)


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
        self.assertEqual(main.queue_episode_visual_analysis('e'), {'queued': 0, 'job_ids': []})

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

    def test_worker_runs_each_model_once_across_the_queued_cuts(self):
        main.queue_episode_visual_analysis('e')
        job_id, owner = worker.claim_job(('analyze',))
        events = []

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

        with patch.object(pipeline, 'ensure_models_available'), patch.object(pipeline, 'release_gpu_models'), \
             patch.object(inference_worker, 'get_model', side_effect=lambda: events.append('subtitle-model') or object()), \
             patch.object(subtitle_model, 'inspect_subtitles', side_effect=lambda *a, **k: events.append('subtitle-cut') or {'samples': [], 'events': []}), \
             patch.object(frame_model, 'frame_server', server), \
             patch.object(frame_model, 'describe_frame', side_effect=lambda *a, **k: events.append('frame') or {'summary': 'x'}), \
             patch.object(frame_model, 'classify_subtitle_languages', return_value=[]), \
             patch.object(video_model, 'VideoRuntime', Runtime), patch.object(video_model, 'analyze_video', analyze), \
             patch.object(pipeline.character_detection, 'crop_frame', return_value=b''), \
             patch.object(pipeline.media, 'extract_cut_frame', return_value=1.0), \
             patch.object(pipeline.structured_results, 'save_generated', save_generated):
            inference_worker.process(job_id, owner)

        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(db.one('SELECT SUM(completed) AS n FROM analysis_job_cuts WHERE job_id=?', (job_id,))['n'], 3)
        shape = [event for event in events if not event.startswith('save:') and event != 'video-load']
        self.assertEqual(shape, ['subtitle-model'] + ['subtitle-cut'] * 3 + ['server-start'] + ['frame'] * 6 + ['server-stop']
                         + ['video-cut'] * 3 + ['video-close'])


if __name__ == '__main__':
    unittest.main()
