"""GET /api/episodes/{id}/review: open checkpoints, checks and missing steps of one episode, read-only."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import db, media
from app.main import app


def result(conn, rid, cut, structured_type, content, created='2026-11-20 10:00:00', title='', stale=None):
    conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,created_at,stale_reason)
        VALUES (?,?,'text',?,?,'model',?,?,?)""", (rid, cut, title, json.dumps(content), json.dumps({'structured_type': structured_type}), created, stale))


class ReviewReportTest(unittest.TestCase):
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
                VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}',8,1)""")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time)
                VALUES ('c1','e',0,0,2,0,1.9),('c2','e',1,2,4,2,3.9),('c3','e',2,4,6,4,5.9),('c4','e',3,6,8,6,7.9)""")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('ann','p','character','Ann','')")
            for cut, start, end in (('c1', 0, 2), ('c2', 2, 4), ('c3', 4, 6)):
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES (?,?,?,?,'[]','test')",
                             (f'r{cut}', cut, start, end))
            # c1: one unlabeled box; c2: an automatic label; c3: a manual label. c4 was never detected.
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES
                ('d1','rc1','first','{}',NULL,NULL),('d2','rc2','first','{}','ann','auto'),('d3','rc3','first','{}','ann','manual')""")
            # c2 target frames: frame 2 flagged and not reviewed, frame 3 flagged but reviewed, one unnamed person.
            frames = [{'ordinal': 1, 'review': {'flags': {}}}, {'ordinal': 2, 'review': {'flags': {'low_match': [1]}}},
                      {'ordinal': 3, 'review': {'flags': {'jump': [1]}, 'reviewed': True}}]
            conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('t2','c2','k',2,4,?)",
                         (json.dumps(frames),))
            conn.execute("INSERT INTO target_slots(run_id,slot,origin,character_context_id) VALUES ('t2',1,'first',NULL)")
            conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json) VALUES ('td','t2',1,1,'{}')")
            # c3 is analysed and reviewed with two open questions and a contradicted picture check.
            result(conn, 'f3', 'c3', 'initial_frame', {})
            result(conn, 'v3', 'c3', 'video_analysis', {})
            result(conn, 'k3', 'c3', 'consistency_summary', {'questions': [{'id': 'q1'}, {'id': 'q2'}],
                   'visual_check': [{'character': 'Ann', 'picture': 'F1', 'claim': 'smiles', 'verdict': 'contradicts'},
                                    {'character': 'Ann', 'picture': 'F2', 'claim': 'sits', 'verdict': 'agrees'}]})
            result(conn, 's3', 'c3', 'performance_beats', {}, stale='Keyframes changed', title='Performance beats')
            # c2 has its first frame and keyframes but no beats yet; c4 has stale beats for its current keyframes.
            result(conn, 'f2', 'c2', 'initial_frame', {})
            result(conn, 'k2', 'c2', 'keyframe_analysis', {})
            result(conn, 'f4', 'c4', 'initial_frame', {})
            result(conn, 'k4', 'c4', 'keyframe_analysis', {})
            result(conn, 'b4', 'c4', 'performance_beats', {'based_on': {'keyframe_analysis': 'k4'}}, stale='Keyframes edited', title='Performance beats')
            conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,confidence) VALUES
                ('l1','e',0.5,1,'Hi','SPEAKER_00','low'),('l2','e',4.5,5,'Yes','SPEAKER_01',NULL),('l3','e',5,5.5,'No','SPEAKER_01',NULL)""")
            conn.execute("UPDATE episodes SET speaker_attribution_stale_reason='Name glossary changed' WHERE id='e'")

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def get(self, query=''):
        response = self.client.get(f'/api/episodes/e/review{query}')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_reports_gates_checks_and_progress(self):
        report = self.get()
        gates, checks, progress = report['gates'], report['checks'], report['progress']
        self.assertEqual(gates['first_frame'], [{'cut': 1, 'cut_id': 'c1', 'unlabeled_boxes': 1}])
        self.assertEqual(gates['target_frames'], [{'cut': 2, 'cut_id': 'c2', 'frames': [{'ordinal': 2, 'flags': ['low_match']}], 'unnamed_people': 1}])
        self.assertEqual([(voice['speaker_label'], voice['lines']) for voice in gates['voices']], [('SPEAKER_01', 2), ('SPEAKER_00', 1)])
        self.assertEqual(gates['decisions'], [{'cut': 3, 'cut_id': 'c3', 'questions': 2, 'draft': False}])
        self.assertEqual(checks['auto_labels'], [{'cut': 2, 'cut_id': 'c2', 'labels': 1}])
        self.assertEqual(checks['picture_contradictions'][0]['checks'], [{'character': 'Ann', 'picture': 'F1', 'claim': 'smiles'}])
        self.assertEqual([(item['cut'], item['reason']) for item in checks['stale_results']], [(3, 'Keyframes changed'), (4, 'Keyframes edited')])
        # l1 has no speaker: it is a voice, so it is not also a line to check (one line, one flag)
        self.assertEqual(checks['lines_to_check']['count'], 0)
        self.assertEqual(checks['speaker_attribution_stale'], 'Name glossary changed')
        self.assertEqual(progress['boxes_missing'], [4])
        self.assertEqual(progress['target_missing'], [1, 3])
        self.assertEqual(progress['visual_missing'], [1])
        self.assertEqual(progress['first_frame_missing'], [1])
        self.assertEqual(progress['keyframes_missing'], [1])
        self.assertEqual(progress['analysis_off'], [])
        self.assertEqual(progress['beats_missing'], [2])
        self.assertEqual(progress['summary_missing'], [])
        self.assertEqual(report['setup'], {'media': False, 'cuts': True, 'transcript': False, 'subtitles': False, 'boxes': False})

    def test_cuts_with_analysis_off_are_listed_and_not_missing(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET analysis_enabled=0 WHERE id='c1'")
        progress = self.get()['progress']
        self.assertEqual(progress['analysis_off'], [1])
        self.assertEqual(progress['first_frame_missing'], [])
        self.assertEqual(progress['keyframes_missing'], [])

    def test_gates_clear_once_settled(self):
        with db.connect() as conn:
            conn.execute("UPDATE character_detections SET character_context_id='ann',label_source='manual' WHERE id='d1'")
            conn.execute("UPDATE target_slots SET character_context_id='ann' WHERE run_id='t2'")
            conn.execute("UPDATE target_detection_runs SET frames_json='[]' WHERE id='t2'")
            conn.execute("UPDATE transcript_segments SET character_context_id='ann'")
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content,updated_at) VALUES ('c3','- Smile: yes','2026-11-20 11:00:00')")
        report = self.get()
        self.assertEqual(report['gates'], {'first_frame': [], 'target_frames': [], 'voices': [], 'decisions': []})
        self.assertEqual(report['progress']['summary_missing'], [3])

    def test_answers_saved_before_the_review_do_not_count(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content,updated_at) VALUES ('c3','old','2026-11-20 09:00:00')")
        self.assertEqual(self.get()['gates']['decisions'][0]['cut'], 3)

    def test_range_limits_cuts_and_voices(self):
        report = self.get('?from_cut=3&to_cut=4')
        self.assertEqual(report['gates']['first_frame'], [])
        self.assertEqual(report['gates']['decisions'][0]['cut'], 3)
        self.assertEqual([voice['speaker_label'] for voice in report['gates']['voices']], ['SPEAKER_01'])
        self.assertEqual(report['progress']['visual_missing'], [])
        self.assertEqual(self.client.get('/api/episodes/e/review?from_cut=3&to_cut=2').status_code, 422)
        self.assertEqual(self.client.get('/api/episodes/missing/review').status_code, 404)

    def test_episode_cut_list_flags_are_unchanged(self):
        cuts = {cut['id']: cut for cut in self.client.get('/api/episodes/e').json()['cuts']}
        self.assertTrue(cuts['c1']['has_unlabeled_boxes'])
        self.assertTrue(cuts['c2']['has_unlabeled_target_boxes'])
        self.assertTrue(cuts['c3']['has_visual_results'])
        self.assertTrue(cuts['c3']['has_consistency_summary'])


if __name__ == '__main__':
    unittest.main()
