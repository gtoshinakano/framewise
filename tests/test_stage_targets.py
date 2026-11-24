"""stage_targets (D31, D37): the target cuts of each stage, and the cuts held for the person or Claude."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import agent_log, db, media
from app.main import app


def result(conn, rid, cut, structured_type, content=None, stale=None, title=None):
    conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,stale_reason)
        VALUES (?,?,'text',?,?,'model',?,?)""", (rid, cut, title or structured_type, json.dumps(content or {}),
                                                json.dumps({'structured_type': structured_type}), stale))


class StageTargetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        self.client = TestClient(app)
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio)
                VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}',10,1)""")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES
                ('c1','e',0,0,2),('c2','e',1,2,4),('c3','e',2,4,6),('c4','e',3,6,8),('c5','e',4,8,10)""")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('ann','p','character','Ann','')")
            for index, cut in enumerate(('c1', 'c2', 'c3', 'c4')):
                start = index * 2
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES (?,?,?,?,'[]','t','complete')",
                             (f'r{cut}', cut, start, start + 2))
                conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES (?,?,'k',?,?,'[]')",
                             (f't{cut}', cut, start, start + 2))
            # c2 has an automatic label the gallery is unsure of: its visual analysis waits for the person.
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES
                ('d1','rc1','first','{}','ann','manual'),('d2','rc2','first','{}','ann','auto'),('d3','rc3','first','{}','ann','manual'),
                ('d4','rc4','first','{}','ann','manual')""")
            # c1 is analysed through the review (no questions): only its Cut summary is missing.
            result(conn, 'f1', 'c1', 'initial_frame')
            result(conn, 'k1', 'c1', 'keyframe_analysis')
            result(conn, 'b1', 'c1', 'performance_beats', {'based_on': {'keyframe_analysis': 'k1'}})
            result(conn, 's1', 'c1', 'consistency_summary', {'questions': []}, title='Consistency summary')
            # c3: its first frame is stale. c4: visual done, beats missing.
            result(conn, 'f3', 'c3', 'initial_frame', stale='First-frame boxes or labels were edited', title='Initial frame description')
            result(conn, 'k3', 'c3', 'keyframe_analysis')
            result(conn, 'f4', 'c4', 'initial_frame')
            result(conn, 'k4', 'c4', 'keyframe_analysis')
            conn.execute("UPDATE cuts SET analysis_enabled=0 WHERE id='c5'")

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def get(self, stage=None):
        response = self.client.get('/api/episodes/e/targets', params={'stage': stage} if stage else {})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def cuts(step):
        return [item['cut'] for item in step['targets']]

    def test_evaluate_runs_only_on_settled_cuts_missing_or_stale(self):
        evaluate = self.get('evaluate')['evaluate']
        self.assertEqual(self.cuts(evaluate['visual']), [3])
        self.assertEqual(evaluate['visual']['targets'][0]['why'], 'stale')
        self.assertEqual([item['cut'] for item in evaluate['visual']['held']], [2])
        self.assertIn('box 1 label not confirmed', evaluate['visual']['held'][0]['reasons'][0])
        self.assertEqual(self.cuts(evaluate['beats']), [4])
        self.assertEqual(self.cuts(evaluate['consistency']), [])

    def test_an_open_inbox_item_holds_its_cut(self):
        with db.connect() as conn:  # a line across a cut is never a hold any more (D37), a voice question is
            agent_log.escalate(conn, episode_id='e', kind='line_split', group_key='line:x', question='?', cut_ids=['c4'])
        self.assertEqual(self.cuts(self.get('evaluate')['evaluate']['beats']), [4])
        with db.connect() as conn:
            agent_log.escalate(conn, episode_id='e', kind='voice', group_key='voice:S', question='?', cut_ids=['c4'], reviewer='claude')
        data = self.get('evaluate')
        self.assertEqual(self.cuts(data['evaluate']['beats']), [])
        self.assertEqual(data['evaluate']['beats']['held'][0]['reasons'], ['inbox: an unknown voice'])
        self.assertEqual(data['inbox']['voice'], {'human': 0, 'claude': 1})

    def test_results_list_cut_summaries_and_their_switch(self):
        results = self.get('results')['results']
        self.assertEqual(self.cuts(results['summary']), [1])
        self.assertEqual((results['summary']['mode'], results['answers']['mode']), ('off', 'off'))
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET content=? WHERE id='s1'", (json.dumps({'questions': [{'id': 'q', 'topic': 'pose', 'options': []}]}),))
        results = self.get('results')['results']
        self.assertEqual(self.cuts(results['answers']), [1])
        self.assertEqual(results['summary']['targets'], [])

    def test_prepare_is_done_once_its_run_finished(self):
        prepare = self.get('prepare')['prepare']
        self.assertIsNone(prepare['done'])
        self.assertEqual([step['step'] for step in prepare['steps']][:3], ['media', 'hardcuts', 'missed_cuts'])
        with db.connect() as conn:
            run = agent_log.start_run(conn, 'e', 'fw', 'prepare')
            agent_log.finish_run(conn, run)
        self.assertEqual(self.get('prepare')['prepare']['done']['id'], run)

    def test_prepare_cuts_finds_a_split_piece_and_stale_target_runs(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1b','e',5,1.5,2)")
        prepare = self.get('prepare_cuts')['prepare_cuts']
        self.assertIn(1, self.cuts(prepare['boxes']))
        # the target runs of this fixture have no keyframe selection behind them: all stale
        self.assertEqual({item['why'] for item in prepare['target_frames']['targets']}, {'stale'})

    def test_a_bad_stage_is_refused(self):
        self.assertEqual(self.client.get('/api/episodes/e/targets', params={'stage': 'nope'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
