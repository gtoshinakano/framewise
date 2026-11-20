"""Episode batch endpoints accept a closed range of displayed cut numbers (from_cut..to_cut, 1-based, inclusive).

These call the endpoint functions directly, like test_batch_queueing.
"""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from app import db, episode_requests, main, media, worker


class CutRangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}')")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time)
                VALUES ('c1','e',0,0,2,0,1.9),('c2','e',1,2,4,2,3.9),('c3','e',2,4,6,4,5.9),('c4','e',3,6,8,6,7.9)""")

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def payloads(self, pipeline):
        return [json.loads(row['payload_json']) for row in db.all_rows('SELECT payload_json FROM jobs WHERE pipeline=? ORDER BY rowid', (pipeline,))]

    def test_cut_range_options(self):
        self.assertEqual(episode_requests.cut_range(None, None), {'first_index': 0})
        self.assertEqual(episode_requests.cut_range(3, None), {'first_index': 2})
        self.assertEqual(episode_requests.cut_range(2, 3), {'first_index': 1, 'last_index': 2})
        self.assertEqual(episode_requests.cut_range(None, 1), {'first_index': 0, 'last_index': 0})
        with self.assertRaises(HTTPException) as caught:
            episode_requests.cut_range(3, 2)
        self.assertEqual(caught.exception.status_code, 422)

    def test_selection_stops_at_the_last_index(self):
        with db.connect() as conn:
            self.assertEqual([cut['id'] for cut in episode_requests.cuts(conn, 'e', 1, last_index=2)], ['c2', 'c3'])
            self.assertEqual([cut['id'] for cut in episode_requests.cuts(conn, 'e', 1)], ['c2', 'c3', 'c4'])

    def test_visual_analysis_queues_only_the_range(self):
        result = main.queue_episode_visual_analysis('e', from_cut=2, to_cut=3)
        self.assertEqual(result['queued'], 2)
        self.assertEqual([item['cut_id'] for item in self.payloads('analyze')[0]['cuts']], ['c2', 'c3'])

    def test_character_requests_queue_only_the_range(self):
        result = main.queue_character_boxes('e', from_cut=2, to_cut=3)
        self.assertEqual(result['queued'], 2)
        self.assertEqual([item['cut_id'] for item in self.payloads('character_detect')], ['c2', 'c3'])
        main.queue_all_character_detection('e', to_cut=1)
        self.assertEqual([item['cut_id'] for item in self.payloads('character_detect')][2:], ['c1'])

    def test_consistency_range_and_only_missing(self):
        with db.connect() as conn:
            conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json)
                VALUES ('r2','c2','text','Consistency summary','{}','model','{"structured_type":"consistency_summary"}')""")
        result = main.queue_episode_consistency_summaries('e', from_cut=1, to_cut=3, only_missing=True)
        self.assertEqual((result['queued'], result['skipped']), (2, 1))
        self.assertEqual([item['cut_id'] for item in self.payloads('consistency_summary')], ['c1', 'c3'])

    def test_review_answers_can_stay_suggestions(self):
        main.queue_episode_consistency_summaries('e', from_cut=1, to_cut=2)
        suggestions = main.queue_episode_review_answers('e', from_cut=1, to_cut=1, auto_save=False)
        self.assertEqual(suggestions['queued'], 1)
        saved = main.queue_episode_review_answers('e', from_cut=2, to_cut=2)
        self.assertEqual(saved['queued'], 1)
        payloads = {item['cut_id']: item for item in self.payloads('review_answers')}
        self.assertFalse(payloads['c1'].get('auto_save'))
        self.assertTrue(payloads['c2']['auto_save'])

    def test_cut_summaries_consider_only_the_range(self):
        result = main.queue_episode_cut_summaries('e', from_cut=2, to_cut=3)
        self.assertEqual(result['blocked']['no_consistency'], [2, 3])

    def test_target_detection_queues_only_the_range(self):
        from app import target_characters_api
        target_characters_api.queue_episode('e', from_cut=3, to_cut=3)
        cut_ids = {item.get('cut_id') for item in self.payloads('character_detect') if item.get('workflow') == 'target_frames_v1'}
        self.assertEqual(cut_ids, {'c3'})

    def test_a_range_waits_for_final_cut_numbers(self):
        from app import target_characters_api
        worker.enqueue('e', 'hardcuts')
        for request in (lambda: main.queue_episode_visual_analysis('e', from_cut=1, to_cut=2),
                        lambda: main.queue_episode_cut_summaries('e', to_cut=2),
                        lambda: main.queue_character_boxes('e', from_cut=2),
                        lambda: target_characters_api.queue_episode('e', from_cut=1, to_cut=1)):
            with self.assertRaises(HTTPException) as caught:
                request()
            self.assertEqual(caught.exception.status_code, 409)
        # An open range keeps today's deferred behaviour.
        self.assertTrue(main.queue_episode_cut_summaries('e', from_cut=2)['deferred'])


if __name__ == '__main__':
    unittest.main()
