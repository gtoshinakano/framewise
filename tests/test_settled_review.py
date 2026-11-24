"""The consistency reviewer settles pose, gaze, state and position itself and asks the person about emotions and voices
(the user's rule, 2026-11-24): settled corrections, the gaze rule and how they reach the saved answers."""
import json
import tempfile
import unittest
from pathlib import Path

from app import agent_log, consistency_summary as cs, db, media, performance_beats, pose_detail, review_decisions as rd, review_report, visual_contracts
from tests.test_review_decisions import result

EMOTION = {'id': 'q1', 'topic': 'emotion', 'character': 'Mai', 'question': 'How does Mai feel at the start?', 'recommended': 'a', 'why': 'first frame',
           'options': [{'id': 'a', 'text': 'Brows drawn together, mouth flat: quiet irritation'}, {'id': 'b', 'text': 'Relaxed brows, slight smile: calm'}]}
SETTLED = {'topic': 'gaze', 'character': 'Mai', 'stored': 'looking at the camera', 'corrected': 'Mai looks just past the camera at Sakuta, off screen',
           'basis': 'gaze_rule', 'why': 'not a point-of-view shot; she talks to Sakuta'}


class ValidateTest(unittest.TestCase):
    def test_settled_entries_become_answered_questions_after_the_asked_ones(self):
        out = cs.validate({'summary': 's', 'questions': [EMOTION, {**EMOTION, 'topic': 'pose', 'question': 'Is Mai sitting?'}], 'settled': [SETTLED]})
        self.assertEqual([(q['id'], q['topic']) for q in out['questions']], [('q1', 'emotion'), ('q2', 'gaze')])  # the pose question is dropped
        settled = out['questions'][1]
        self.assertEqual((settled['settled'], settled['recommended']), ('gaze_rule', 'a'))
        self.assertEqual([option['text'] for option in settled['options']], [SETTLED['corrected'], SETTLED['stored']])
        self.assertNotIn('settled', out['questions'][0])

    def test_a_picture_settles_only_when_pictures_were_sent_and_each_reading_once(self):
        picture = {**SETTLED, 'basis': 'picture'}
        self.assertEqual(cs.validate({'summary': 's', 'questions': [], 'settled': [picture]})['questions'], [])
        seen = cs.validate({'summary': 's', 'questions': [], 'visual_check': [], 'settled': [picture, {**SETTLED, 'corrected': 'Mai looks down'}]}, labels=['F1'])
        self.assertEqual(len(seen['questions']), 1)  # one answer line per topic and character
        unchanged = cs.validate({'summary': 's', 'questions': [], 'settled': [{**SETTLED, 'corrected': 'Looking at the camera'}]})
        self.assertEqual(unchanged['questions'], [])  # corrects nothing
        self.assertEqual(cs.validate({'summary': 's', 'questions': []})['questions'], [])  # an older answer without settled


class GazeRuleTest(unittest.TestCase):
    def test_every_prompt_that_reads_a_gaze_carries_the_rule(self):
        for text in (cs.system(False), cs.system(True), visual_contracts.FRAME_INSTRUCTION, performance_beats.SYSTEM):
            self.assertIn(pose_detail.GAZE_RULE, text)
        self.assertNotIn('"at the camera"', visual_contracts.FRAME_INSTRUCTION)  # the old example taught the model to say it
        self.assertIn('## EMOTION QUESTIONS', cs.system(True))
        self.assertIn('a picture that agrees confirms nothing', cs.system(True))
        self.assertNotIn('a picture that agrees', cs.system(False))

    def test_readings_into_the_camera_become_leads(self):
        frame = {'visible_characters': [
            {'character': 'Mai', 'gaze': 'at the camera', 'pose_detail': {'gaze_target': 'camera', 'flags': [{'code': 'camera_gaze', 'text': 'x'}]}},
            {'character': 'Rio', 'gaze': 'down at the cup', 'pose_detail': {'gaze_target': 'object'}}]}
        keyframes = {'frames': [{'label': 'F2', 'characters': [{'name': 'Rio', 'state': 'Sitting. Eyes open, looking toward the viewer.'}]}]}
        beats = {'beats': [{'characters': [{'character': 'Mai', 'gaze': 'into the camera'}, {'character': 'Rio', 'gaze': 'at Mai'}]}]}
        leads = cs.camera_gaze(frame, keyframes, beats)
        self.assertEqual([(lead['character'], lead['where']) for lead in leads], [('Mai', 'first frame'), ('Rio', 'keyframe F2'), ('Mai', 'beats')])
        checks = cs.app_checks({'initial_frame': {'content': frame}}, [])
        self.assertEqual([check['kind'] for check in checks], ['camera_gaze'])  # the pose flag says the same and is not repeated


class AnswersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        self.questions = cs.validate({'summary': 's', 'questions': [EMOTION], 'settled': [SETTLED, {**SETTLED, 'topic': 'pose', 'character': 'Rio', 'stored': 'standing',
                                                                                                 'corrected': 'Rio sits on the bench', 'basis': 'keypoints'}]})['questions']
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','E','a.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,4),('c2','e',1,4,8)")
            for cut in ('c1', 'c2'):
                result(conn, f'r-{cut}', cut, {'summary': 's', 'questions': self.questions})
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content) VALUES ('c2','- Pose (Rio): Rio stands by the door')")  # the person's own line

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def apply(self, conn, cut_id):
        return rd.apply_settled(conn, episode_id='e', cut_id=cut_id, result_id=f'r-{cut_id}', questions=self.questions, model='m')

    def test_settled_readings_are_written_without_touching_the_persons_lines_and_undo(self):
        with db.connect() as conn:
            decision_id = self.apply(conn, 'c1')
            self.assertEqual(conn.execute("SELECT content FROM cut_brief_instructions WHERE cut_id='c1'").fetchone()[0],
                             f"- Gaze (Mai): {SETTLED['corrected']}\n- Pose (Rio): Rio sits on the bench")
            row = conn.execute('SELECT * FROM agent_decisions WHERE id=?', (decision_id,)).fetchone()
            self.assertTrue(json.loads(row['evidence_json'])['settled'])
            self.assertTrue(row['summary'].startswith('Settled 2 reading(s)'))
            self.assertEqual(rd.agent_held(conn, ['c1']), set())  # never counted as the agent answering for the person
            self.apply(conn, 'c2')
            self.assertEqual(conn.execute("SELECT content FROM cut_brief_instructions WHERE cut_id='c2'").fetchone()[0],
                             f"- Pose (Rio): Rio stands by the door\n- Gaze (Mai): {SETTLED['corrected']}")
            agent_log.undo_decision(conn, decision_id)
            self.assertEqual(conn.execute("SELECT content FROM cut_brief_instructions WHERE cut_id='c1'").fetchone()[0], '')

    def test_the_emotion_question_stays_open_for_the_person_after_the_settled_write(self):
        with db.connect() as conn:
            self.apply(conn, 'c1')
            gate = review_report._consistency(conn, 'c1')
            self.assertEqual((gate['questions'], gate['answered']), (1, False))
            self.assertEqual([q['id'] for q in rd.asked(self.questions)], ['q1'])
            text = conn.execute("SELECT content FROM cut_brief_instructions WHERE cut_id='c1'").fetchone()[0]
            conn.execute("UPDATE cut_brief_instructions SET content=? WHERE cut_id='c1'", (text + '\n- Emotion (Mai): Relaxed brows, slight smile: calm',))
            self.assertEqual(review_report._consistency(conn, 'c1')['questions'], 0)


if __name__ == '__main__':
    unittest.main()
