"""cut_readiness (D26): a step runs for a cut only once its inputs are settled; the reasons go to the person."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import agent_log, autopilot_profile, cut_readiness, db, keyframe_analysis, media
from app.main import app


def result(conn, rid, cut, structured_type, content=None, stale=None):
    conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,stale_reason)
        VALUES (?,?,'text',?,?,'model',?,?)""", (rid, cut, structured_type, json.dumps(content or {}),
                                                json.dumps({'structured_type': structured_type}), stale))


class CutReadinessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio)
                VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}',8,1)""")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time)
                VALUES ('c1','e',0,0,2,0,1.9),('c2','e',1,2,4,2,3.9)""")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('ann','p','character','Ann',''),('bob','p','character','Bob','')")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES ('r1','c1',0,2,'[]','t','complete'),('r2','c2',2,4,'[]','t','complete')")
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source,candidates_json)
                VALUES ('d1','r1','first','{}','ann','auto',?)""", (json.dumps([{'character_id': 'ann', 'score': 0.97}, {'character_id': 'bob', 'score': 0.93}]),))
            conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('t1','c1','k',0,2,'[]')")
            for cut in ('c1', 'c2'):
                result(conn, f'f{cut}', cut, 'initial_frame')
                result(conn, f'k{cut}', cut, 'keyframe_analysis')
            result(conn, 'b1', 'c1', 'performance_beats', {'based_on': {'keyframe_analysis': 'kc1'}})
            conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,character_context_id,confidence,speaker_source)
                VALUES ('l1','e',0.5,1.5,'Hi','ann','high','voice'),('l2','e',2.2,2.5,'edge','bob','low','qwen')""")

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def ready(self, cut='c1'):
        with db.connect() as conn:
            return cut_readiness.cut(conn, cut)

    def sql(self, statement, args=()):
        with db.connect() as conn:
            conn.execute(statement, args)

    def test_a_weak_gallery_label_holds_everything_until_it_is_confirmed(self):
        ready = self.ready()
        self.assertFalse(ready['may_visual'])
        self.assertEqual(ready['held']['visual'], ['box 1 label not confirmed (Ann 0.97, lead 0.04)'])
        self.sql("UPDATE character_detections SET label_source='agent' WHERE id='d1'")
        ready = self.ready()
        self.assertTrue(ready['may_visual'] and ready['may_beats'] and ready['may_consistency'])

    def test_a_strong_gallery_match_counts_as_settled(self):
        self.sql("UPDATE character_detections SET candidates_json=? WHERE id='d1'", (json.dumps([{'character_id': 'ann', 'score': 0.96}, {'character_id': 'bob', 'score': 0.85}]),))
        self.assertTrue(self.ready()['may_beats'])
        self.assertEqual(cut_readiness.strong_match('bob', [{'character_id': 'ann', 'score': 0.99}]), (False, 0.99, 0.99))

    def test_unnamed_boxes_and_flagged_target_frames_hold_the_visual_step(self):
        self.sql("UPDATE character_detections SET label_source='manual' WHERE id='d1'")
        self.sql("INSERT INTO character_detections(id,run_id,position,box_json) VALUES ('d2','r1','first','{}')")
        self.assertEqual(self.ready()['held']['visual'], ['box 2 has no character'])
        self.sql("DELETE FROM character_detections WHERE id='d2'")
        self.sql("UPDATE target_detection_runs SET frames_json=? WHERE id='t1'", (json.dumps([{'ordinal': 1, 'review': {'flags': {}}}, {'ordinal': 40, 'review': {'flags': {'low_match': [1]}}}]),))
        ready = self.ready()
        self.assertEqual(ready['held']['visual'], ['target frame F2 flagged (low_match)'])
        self.assertEqual(ready['held']['target_frames'], [])   # target frames themselves may run: the boxes are settled

    def test_speech_must_be_named_and_not_low_unless_a_person_set_it(self):
        self.sql("UPDATE character_detections SET label_source='manual' WHERE id='d1'")
        self.assertTrue(self.ready()['may_beats'])   # c1's own line is high; the low line is c2's
        ready = self.ready('c2')                      # c2 has no boxes at all
        self.assertTrue(ready['may_visual'])
        self.assertEqual(ready['held']['beats'], ['line at 0:02.2 has a low-confidence speaker'])
        self.sql("UPDATE transcript_segments SET speaker_source='manual' WHERE id='l2'")
        self.assertTrue(self.ready('c2')['speech'])
        self.sql("UPDATE transcript_segments SET character_context_id=NULL WHERE id='l2'")
        self.assertEqual(self.ready('c2')['held']['beats'], ['line at 0:02.2 has no speaker'])
        self.sql("UPDATE transcript_segments SET character_context_id='bob' WHERE id='l2'")
        self.sql("UPDATE episodes SET speaker_attribution_stale_reason='Name glossary changed' WHERE id='e'")
        self.assertEqual(self.ready('c2')['held']['beats'], ['speaker attribution stale (Name glossary changed)'])
        self.sql(f"UPDATE episodes SET speaker_attribution_stale_reason='{cut_readiness.UNATTRIBUTED}' WHERE id='e'")
        self.assertTrue(self.ready('c2')['speech'])

    def test_beats_wait_for_current_visual_results_and_consistency_for_current_beats(self):
        self.sql("UPDATE character_detections SET label_source='manual' WHERE id='d1'")
        self.sql("UPDATE analysis_results SET stale_reason='First-frame boxes or labels were edited' WHERE id='kc1'")
        ready = self.ready()
        self.assertTrue(ready['may_visual'])
        self.assertFalse(ready['may_beats'])
        self.assertIn('keyframe analysis stale (First-frame boxes or labels were edited)', ready['held']['beats'])
        self.sql("UPDATE analysis_results SET stale_reason=NULL WHERE id='kc1'")
        self.sql("UPDATE analysis_results SET content=? WHERE id='b1'", (json.dumps({'based_on': {'keyframe_analysis': 'old'}}),))
        ready = self.ready()
        self.assertTrue(ready['may_beats'])
        self.assertEqual(ready['held']['consistency'], ['performance beats older than the keyframe analysis'])

    def test_a_cut_without_characters_skips_the_box_checks(self):
        self.sql("INSERT INTO character_detections(id,run_id,position,box_json) VALUES ('d2','r1','first','{}')")
        self.sql("UPDATE cuts SET no_characters=1 WHERE id='c1'")
        self.assertTrue(self.ready()['may_visual'])

    def test_a_split_cut_without_a_detection_of_its_boundaries_is_held(self):
        self.sql("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        self.assertEqual(self.ready()['held']['visual'], ['first-frame boxes not detected for the current cut'])

    def test_the_review_report_lists_the_held_cuts_per_step(self):
        held = TestClient(app).get('/api/episodes/e/review').json()['progress']['held']
        self.assertEqual([item['cut'] for item in held['visual']], [1])
        self.assertEqual([item['cut'] for item in held['beats']], [1, 2])
        self.assertEqual(held['target_frames'][0]['reasons'], ['box 1 label not confirmed (Ann 0.97, lead 0.04)'])

    def test_keyframes_queue_beats_only_as_the_job_asks(self):
        def cut_run(beats):
            payload = {} if beats is None else {'beats': beats}
            return SimpleNamespace(job={'payload_json': json.dumps(payload)}, cut_id='c2')
        with db.connect() as conn:
            self.assertTrue(keyframe_analysis.beats_wanted(conn, cut_run(None)))
            self.assertFalse(keyframe_analysis.beats_wanted(conn, cut_run('off')))
            self.assertFalse(keyframe_analysis.beats_wanted(conn, cut_run('when_ready')))   # c2's low line
            conn.execute("UPDATE transcript_segments SET confidence='medium' WHERE id='l2'")
            self.assertTrue(keyframe_analysis.beats_wanted(conn, cut_run('when_ready')))


class AutopilotFixesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_answers_and_summaries_are_off_by_default_and_switched_off_once_in_stored_profiles(self):
        defaults = autopilot_profile.defaults()['modes']
        self.assertEqual((defaults['review_answers'], defaults['cut_summaries'], defaults['box_names']), ('off', 'off', 'apply'))
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("INSERT INTO autopilot_profiles(project_id,profile_json) VALUES ('p',?)",
                         (json.dumps({'modes': {'review_answers': 'apply', 'missed_cuts': 'ask'}}),))
            autopilot_profile.answers_and_summaries_off(conn)
            profile, _ = autopilot_profile.load(conn, 'p')
            self.assertEqual((profile['modes']['review_answers'], profile['modes']['cut_summaries'], profile['modes']['missed_cuts']), ('off', 'off', 'ask'))
            # the user turns them on again: kept
            autopilot_profile.save(conn, 'p', {**profile, 'modes': {**profile['modes'], 'cut_summaries': 'apply'}})
            self.assertEqual(autopilot_profile.load(conn, 'p')[0]['modes']['cut_summaries'], 'apply')

    def test_reconfirming_the_agents_label_is_not_an_override(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json)
                VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}')""")
            for decision, target in (('a', 'd1'), ('b', 'd2')):
                conn.execute("""INSERT INTO agent_decisions(id,episode_id,kind,tier,target_table,target_id,after_json)
                    VALUES (?,'e','box_names','claude','character_detections',?,?)""", (decision, target, json.dumps({'character_context_id': 'ann', 'label_source': 'agent'})))
            agent_log.note_override(conn, 'character_detections', 'd1', now={'character_context_id': 'ann'})
            agent_log.note_override(conn, 'character_detections', 'd2', now={'character_context_id': 'bob'})
            marked = dict(conn.execute('SELECT id,overridden_at IS NOT NULL FROM agent_decisions').fetchall())
        self.assertEqual(marked, {'a': 0, 'b': 1})


if __name__ == '__main__':
    unittest.main()
