"""Autopilot M4: the agent answers the consistency review questions (review_decisions.py, docs/autopilot/HARNESS.md)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_log, autopilot_profile, db, job_kinds, media, review_decisions as rd, worker
from app.main import app

# The fixture of frontend/tests/consistencyView.test.mjs: the Python port must write the same lines.
QUESTIONS = [
    {'id': 'q1', 'topic': 'gaze', 'character': 'Mai', 'question': 'Where does Mai look?', 'recommended': 'a', 'why': 'first frame',
     'options': [{'id': 'a', 'text': 'At Rio'}, {'id': 'b', 'text': 'Down at the cup'}]},
    {'id': 'q2', 'topic': 'state', 'character': '', 'question': 'What is Rio doing?',
     'options': [{'id': 'a', 'text': 'Sitting'}, {'id': 'b', 'text': 'Standing'}, {'id': 'c', 'text': 'Walking'}]},
]


class AnswerLinesTest(unittest.TestCase):
    def test_choosing_writes_replaces_and_clears_the_line_of_that_question_only(self):
        text = rd.set_answer('', QUESTIONS[0], 'At Rio', QUESTIONS)
        self.assertEqual(text, '- Gaze (Mai): At Rio')
        text = rd.set_answer(text, QUESTIONS[1], 'Sitting', QUESTIONS)
        self.assertEqual(text, '- Gaze (Mai): At Rio\n- State: Sitting')
        text = rd.set_answer(text, QUESTIONS[0], 'Down at the cup', QUESTIONS)
        self.assertEqual(text, '- Gaze (Mai): Down at the cup\n- State: Sitting')
        self.assertEqual(rd.selection(text, QUESTIONS[0], QUESTIONS), {'option': 'b', 'custom': ''})
        text = rd.set_answer(text, QUESTIONS[0], None, QUESTIONS)
        self.assertEqual(text, '- State: Sitting')
        self.assertIsNone(rd.selection(text, QUESTIONS[0], QUESTIONS))

    def test_text_around_the_choices_is_kept_and_an_own_answer_is_custom(self):
        notes = 'Keep the coat green\n\n- Camera: static\n  but slightly handheld'
        with_own = rd.set_answer(notes, QUESTIONS[0], 'She looks at the window', QUESTIONS)
        self.assertEqual(with_own, f'{notes}\n- Gaze (Mai): She looks at the window')
        self.assertEqual(rd.selection(with_own, QUESTIONS[0], QUESTIONS), {'option': None, 'custom': 'She looks at the window'})
        self.assertEqual(rd.set_answer(with_own, QUESTIONS[0], None, QUESTIONS), notes)
        wrapped = '- Gaze (Mai): At Rio and then\n  down at the cup\n- State: Standing'
        self.assertEqual(rd.selection(wrapped, QUESTIONS[1], QUESTIONS), {'option': 'b', 'custom': ''})
        self.assertEqual(rd.set_answer(wrapped, QUESTIONS[0], 'Down at the cup', QUESTIONS), '- Gaze (Mai): Down at the cup\n- State: Standing')
        self.assertEqual(rd.selection('- Eyes: at rio.', QUESTIONS[0], QUESTIONS), {'option': 'a', 'custom': ''})
        self.assertEqual(rd.selection('- gaze (mai): Down at the cup', QUESTIONS[0], QUESTIONS), {'option': 'b', 'custom': ''})
        self.assertIsNone(rd.selection('- Camera: static', QUESTIONS[0], QUESTIONS))


def decision(choice, confidence='high', kind='fact', rule=False):
    return {'choice': choice, 'confidence': confidence, 'kind': kind, 'rule': rule, 'reason': 'test'}


class DeciderAndResolverTest(unittest.TestCase):
    def test_the_grammar_names_every_question_once_and_the_validator_sends_back_mistakes(self):
        schema = rd.decision_format(QUESTIONS)['json_schema']['schema']
        decisions = schema['properties']['decisions']
        self.assertEqual((decisions['minItems'], decisions['maxItems']), (2, 2))
        self.assertEqual(decisions['items']['properties']['question']['enum'], ['q1', 'q2'])
        found = rd.validate({'decisions': [{'question': 'q1', 'choice': 'a', 'confidence': 'high', 'kind': 'fact', 'rule': False, 'reason': ' first   frame '},
                                           {'question': 'q2', 'choice': None, 'confidence': 'low', 'kind': 'creative', 'rule': False, 'reason': ''}]}, QUESTIONS)
        self.assertEqual((found['q1']['choice'], found['q1']['reason'], found['q2']['kind']), ('a', 'first frame', 'creative'))
        with self.assertRaisesRegex(ValueError, 'q1 has no option d; choose one of a, b or null; no decision for q1, q2'):
            rd.validate({'decisions': [{'question': 'q1', 'choice': 'd', 'confidence': 'high'}]}, QUESTIONS)

    def test_resolve_applies_only_rules_and_trusted_agreement(self):
        q = QUESTIONS[0]
        rule = {'rule': 'Speakers on screen are lip-synced', 'when_unsure': 'evidence'}
        self.assertEqual(rd.resolve(q, decision('b', rule=True), rule)['action'], 'apply')                # a standing rule, even against the reviewer
        self.assertEqual(rd.resolve(q, decision('b', rule=True), {'rule': ''})['action'], 'claude')        # no rule written: the claim is ignored
        self.assertEqual(rd.resolve(q, decision('a'), {}, trust=0.95)['action'], 'apply')                 # agreement in a trusted topic
        self.assertEqual(rd.resolve(q, decision('a'), {}, trust=0.6)['action'], 'claude')                 # the user often disagrees here
        self.assertEqual(rd.resolve(q, decision('a'), {}, trust=None)['why'], 'this topic is not trusted yet')
        self.assertEqual(rd.resolve(q, decision('a', 'medium'), {}, trust=0.95)['action'], 'claude')
        disagree = rd.resolve(q, decision('b'), {}, trust=0.95)
        self.assertEqual((disagree['action'], disagree['option'], disagree['why']), ('claude', 'b', 'the decider and the reviewer disagree'))
        taken = rd.resolve(q, decision('b'), {'when_unsure': 'recommended'}, trust=0.2)
        self.assertEqual((taken['action'], taken['option'], taken['tier']), ('apply', 'a', 'rules'))
        self.assertEqual(rd.resolve(q, decision(None), {'when_unsure': 'ask'})['action'], 'human')
        self.assertEqual(rd.resolve(q, decision('a', kind='creative'), {}, trust=1.0)['action'], 'human')
        self.assertEqual(rd.resolve(q, decision('a'), {}, trust=1.0, mode='ask')['action'], 'human')     # profile: ask before answering


def result(conn, result_id, cut_id, content, title='Consistency summary', kind='consistency_summary', stale=None, context=None):
    conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,context_json,stale_reason)
        VALUES (?,?,'structured',?,?,'model',?,?,?)""", (result_id, cut_id, title, json.dumps(content), json.dumps({'structured_type': kind}),
                                                          json.dumps(context or {}), stale))


class ResolverJobTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json)
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}')""")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,4),('c2','e',1,4,8),('c3','e',2,8,12)""")
            for cut in ('c1', 'c2', 'c3'):
                result(conn, f'f-{cut}', cut, {'composition': 'two people'}, title='First frame', kind='initial_frame')
                result(conn, f'r-{cut}', cut, {'summary': 's', 'clear': 'c', 'confusing': 'x', 'questions': QUESTIONS},
                       context={'pictures': {'sent': [{'kind': 'first', 'label': 'First frame', 'cut_id': cut, 'at': ['first']}]}})
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content) VALUES ('c2','- State: Standing')")  # the user answered q2 of c2
            # off by default since D27; these tests exercise the resolver itself
            autopilot_profile.save(conn, 'p', {'modes': {'review_answers': 'apply'}})
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def run_jobs(self, decisions_by_cut):
        """Claim and run every queued decider job with the model replaced by ``decisions_by_cut``."""
        def fake(config, evidence, questions, **kwargs):
            return decisions_by_cut[self.current], 'test-model'
        ran = []
        while True:
            claimed = worker.claim_job(('review_decisions',))
            if not claimed:
                break
            job_id, owner = claimed
            self.current = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])['cut_id']
            with mock.patch.object(rd, 'generate', side_effect=fake):
                rd.process(job_id, owner)
            ran.append(db.one('SELECT status,stage,error FROM jobs WHERE id=?', (job_id,)))
        return ran

    def answers(self, cut_id):
        return (db.one('SELECT content FROM cut_brief_instructions WHERE cut_id=?', (cut_id,)) or {}).get('content', '')

    def test_the_kind_is_a_text_job_and_shadow_writes_nothing(self):
        self.assertIn('review_decisions', worker.TEXT_PIPELINES)
        self.assertEqual(job_kinds.KINDS['review_decisions'].scope({'cut_id': 'c1', 'shadow': True}).writes, frozenset())
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'review_answers': 'off'}}})
        off = self.client.post('/api/episodes/e/review-decisions', json={}).json()
        self.assertEqual(off['job_ids'], [])
        shadow = self.client.post('/api/episodes/e/review-decisions', json={'shadow': True}).json()
        self.assertEqual((len(shadow['job_ids']), shadow['skipped']), (1, {'no_user_answers': [1, 3]}))  # only c2 has the user's answers
        ran = self.run_jobs({'c2': {'q1': decision('a'), 'q2': decision('b', 'medium')}})
        self.assertEqual([item['status'] for item in ran], ['done'], ran)
        with db.connect() as conn:
            row = conn.execute('SELECT outcomes_json FROM review_decisions WHERE shadow=1').fetchone()
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM agent_decisions').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM agent_escalations').fetchone()[0], 0)
        self.assertEqual(set(json.loads(row[0])), {'q1', 'q2'})  # the user's answer to q2 was hidden, so q2 was decided too
        self.assertEqual(self.answers('c2'), '- State: Standing')
        again = self.client.post('/api/episodes/e/review-decisions', json={'shadow': True}).json()
        self.assertEqual((again['job_ids'], again['skipped']['decided']), ([], [2]))

    def test_a_run_applies_rules_leaves_the_rest_to_the_cut_dialog_and_keeps_the_users_answers(self):
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'review_answers': 'apply'},
            'topics': {'gaze': {'rule': 'Characters look at whoever speaks', 'when_unsure': 'evidence'}, 'state': {'rule': '', 'when_unsure': 'ask'}},
            'claude': {'enabled': True, 'max_items_per_run': 1}}})
        queued = self.client.post('/api/episodes/e/review-decisions', json={}).json()
        self.assertEqual(len(queued['job_ids']), 3)
        ran = self.run_jobs({'c1': {'q1': decision('b', rule=True), 'q2': decision('a', 'low')},   # rule applies q1; q2 stays open
                             'c2': {'q1': decision('b'), 'q2': decision('a')},                     # q1 disagrees: open; q2 is the user's
                             'c3': {'q1': decision(None, 'low'), 'q2': decision('a', 'low')}})
        self.assertEqual([item['status'] for item in ran], ['done'] * 3, ran)
        self.assertEqual(self.answers('c1'), '- Gaze (Mai): Down at the cup')
        self.assertEqual(self.answers('c2'), '- State: Standing')
        report = self.client.get('/api/episodes/e/review-decisions').json()
        self.assertEqual(report['counts'], {'apply': 1, 'human': 4, 'keep': 1})
        self.assertEqual(report['questions'], [])  # review questions never go to the inbox (D32): the cut dialog has them
        with db.connect() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM agent_escalations').fetchone()[0], 0)
        applied = report['applied'][0]
        self.assertEqual((applied['tier'], applied['after']['lines'][0]['question']), ('local', 'q1'))

        # A person's edit of an agent answer is an override; undo then refuses instead of overwriting it.
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': '- Gaze (Mai): At Rio'})
        decision_id = applied['id']
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision_id}/undo').status_code, 409)
        with db.connect() as conn:
            self.assertIsNotNone(conn.execute('SELECT overridden_at FROM agent_decisions WHERE id=?', (decision_id,)).fetchone()[0])

    def test_an_open_question_left_from_before_can_still_be_answered_and_undo_restores_the_answers(self):
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'review_answers': 'apply'}, 'topics': {'state': {'rule': '', 'when_unsure': 'recommended'}}}})
        self.client.post('/api/episodes/e/review-decisions', json={'from_cut': 1, 'to_cut': 1})
        self.run_jobs({'c1': {'q1': decision('b'), 'q2': decision(None, 'low')}})  # q1 disagrees, q2 has no recommendation: both left open
        self.assertEqual(self.answers('c1'), '')
        with db.connect() as conn:  # a question escalated before D32
            rd.escalate_questions(conn, 'e', 'c1', 'r-c1', [{**rd._question_view(QUESTIONS[1]), 'proposal': None, 'why_asked': 'older run'}], 'human')
            person = agent_log.escalations(conn, 'e', 'open', rd.ESCALATION)
        self.client.post(f"/api/escalations/{person[0]['id']}/review-answers", json={'answers': [{'question': 'q2', 'text': 'Kneeling by the bag'}]})
        self.assertEqual(self.answers('c1'), '- State: Kneeling by the bag')
        with db.connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM agent_decisions ORDER BY rowid DESC")]
            for decision_id in ids:
                agent_log.undo_decision(conn, decision_id)
        self.assertEqual(self.answers('c1'), '')

    def test_a_new_review_closes_the_old_questions_and_trust_counts_only_the_users_answers(self):
        with db.connect() as conn:  # a question escalated before D32
            rd.escalate_questions(conn, 'e', 'c1', 'r-c1', [{**rd._question_view(QUESTIONS[0]), 'proposal': None, 'why_asked': 'older run'}], 'human')
            conn.execute("UPDATE analysis_results SET superseded_by='new' WHERE id='r-c1'")
            result(conn, 'new', 'c1', {'summary': 's', 'questions': QUESTIONS[:1]})
            self.assertEqual(rd.close_settled(conn, 'e'), 1)
            # trust: ten cuts whose user took the recommended option for gaze; agent-written answers are ignored
            for index in range(10):
                conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES (?,?,?,?,?)', (f't{index}', 'e', 10 + index, 20 + index, 21 + index))
                result(conn, f'rt{index}', f't{index}', {'questions': QUESTIONS[:1]})
                conn.execute('INSERT INTO cut_brief_instructions(cut_id,content) VALUES (?,?)', (f't{index}', '- Gaze (Mai): At Rio'))
            table = rd.trust_table(conn, 'p')
            self.assertEqual((table['gaze']['rate'], table['gaze']['scored'], table['gaze']['source']), (1.0, 10, 'recommended'))
            agent_log.record(conn, episode_id='e', cut_id='t0', kind=rd.KIND, tier='local', target_table='cut_brief_instructions', target_id='t0',
                             before={'content': ''}, after={'content': '- Gaze (Mai): At Rio'})
            self.assertEqual(rd.trust_table(conn, 'p')['gaze']['source'], 'too few')
            # shadow measurements win once a topic has enough of them
            for index in range(1, 10):
                conn.execute("INSERT INTO review_decisions(id,cut_id,episode_id,result_id,shadow,decisions_json) VALUES (?,?,?,?,1,?)",
                             (f'sd{index}', f't{index}', 'e', f'rt{index}', json.dumps({'q1': decision('a')})))
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('t10','e',30,40,41)")
            result(conn, 'rt10', 't10', {'questions': QUESTIONS[:1]})
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content) VALUES ('t10','- Gaze (Mai): Down at the cup')")
            conn.execute("INSERT INTO review_decisions(id,cut_id,episode_id,result_id,shadow,decisions_json) VALUES ('sd10','t10','e','rt10',1,?)",
                         (json.dumps({'q1': decision('a')}),))
            gaze = rd.trust_table(conn, 'p')['gaze']
        self.assertEqual((gaze['source'], gaze['scored'], round(gaze['rate'], 2)), ('shadow', 10, 0.9))


if __name__ == '__main__':
    unittest.main()
