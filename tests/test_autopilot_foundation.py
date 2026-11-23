"""Autopilot M0: agent runs, decisions with undo, escalations and the per-project profile (docs/autopilot/HARNESS.md)."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import agent_log, autopilot_profile, db, media
from app.main import app


class AutopilotFoundationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P'),('other','O')")
            conn.execute("""INSERT INTO context_items(id,project_id,kind,title,content) VALUES
                ('mai','p','character','Mai',''),('kaede','p','character','Kaede',''),('nodoka','p','character','Nodoka',''),
                ('x','other','character','Stranger','')""")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json)
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}'),('e2','other','Other','b.mp4','.mp4','ready','{}')""")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,2),('c2','e',1,2,4),('o1','e2',0,0,1)")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c1',0,2,'[]','test')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json) VALUES ('d1','run','first','[0,0,1,1]'),('d2','run','first','[1,1,2,2]')")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def label(self, conn, detection_id, character, run_id, tier='local'):
        """What a resolver does: change the row and log the decision in the same transaction."""
        before = dict(conn.execute('SELECT character_context_id,label_source FROM character_detections WHERE id=?', (detection_id,)).fetchone())
        conn.execute('UPDATE character_detections SET character_context_id=?,label_source=? WHERE id=?', (character, agent_log.AGENT, detection_id))
        return agent_log.record(conn, episode_id='e', run_id=run_id, cut_id='c1', kind='box_names', tier=tier, target_table='character_detections',
                                target_id=detection_id, before=before, after={'character_context_id': character, 'label_source': agent_log.AGENT},
                                evidence={'gallery': 0.91}, confidence=0.91, summary=f'Named the box {character}')

    # --- runs, decisions and undo ----------------------------------------------------------------------------------

    def test_a_run_snapshots_the_profile_and_counts_its_decisions(self):
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'voices': 'ask'}}})
        started = self.client.post('/api/episodes/e/agent-runs', json={'agent': 'claude-code', 'purpose': 'M0 test'})
        self.assertEqual(started.status_code, 200, started.text)
        run = started.json()
        self.assertEqual((run['status'], run['policy']['modes']['voices']), ('running', 'ask'))
        with db.connect() as conn:
            self.assertEqual(json.loads(conn.execute('SELECT policy_json FROM agent_runs WHERE id=?', (run['id'],)).fetchone()[0])['modes']['voices'], 'ask')
            self.label(conn, 'd1', 'mai', run['id'])
        finished = self.client.post(f"/api/agent-runs/{run['id']}/finish", json={'summary': {'applied': 1}, 'claude_items': 3}).json()
        self.assertEqual((finished['status'], finished['claude_items'], finished['summary']), ('finished', 3, {'applied': 1}))
        listed = self.client.get('/api/episodes/e/agent-runs').json()
        self.assertEqual(listed[0]['decisions'], {'total': 1, 'undone': 0, 'overridden': 0})
        detail = self.client.get(f"/api/agent-runs/{run['id']}").json()
        self.assertEqual(detail['decisions'][0]['after'], {'character_context_id': 'mai', 'label_source': 'agent'})
        self.assertEqual(self.client.post('/api/episodes/missing/agent-runs', json={}).status_code, 404)

    def test_undo_restores_the_row_and_refuses_after_a_person_changed_it(self):
        with db.connect() as conn:
            run_id = agent_log.start_run(conn, 'e', 'test')
            first = self.label(conn, 'd1', 'mai', run_id)
            second = self.label(conn, 'd2', 'kaede', run_id)
            conn.execute("UPDATE character_detections SET character_context_id='nodoka',label_source='manual' WHERE id='d2'")  # the user corrects d2
            agent_log.note_override(conn, 'character_detections', 'd2')
        conflict = self.client.post(f'/api/agent-decisions/{second}/undo')
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.client.post(f'/api/agent-decisions/{first}/undo').json(), {'undone': True})
        self.assertEqual(self.client.post(f'/api/agent-decisions/{first}/undo').json(), {'undone': False})
        self.assertEqual(self.client.post('/api/agent-decisions/missing/undo').status_code, 404)
        with db.connect() as conn:
            rows = {row['id']: (row['character_context_id'], row['label_source']) for row in conn.execute('SELECT * FROM character_detections')}
            overridden = conn.execute('SELECT overridden_at FROM agent_decisions WHERE id=?', (second,)).fetchone()[0]
        self.assertEqual(rows, {'d1': (None, None), 'd2': ('nodoka', 'manual')})  # the user's edit is never overwritten
        self.assertIsNotNone(overridden)

    def test_undoing_a_run_skips_conflicts_and_keeps_the_run_open_for_them(self):
        with db.connect() as conn:
            run_id = agent_log.start_run(conn, 'e', 'test')
            self.label(conn, 'd1', 'mai', run_id)
            self.label(conn, 'd2', 'kaede', run_id)
            conn.execute("UPDATE character_detections SET character_context_id='nodoka' WHERE id='d1'")
        result = self.client.post(f'/api/agent-runs/{run_id}/undo').json()
        self.assertEqual((len(result['undone']), len(result['conflicts'])), (1, 1))
        self.assertIn('Changed since', result['conflicts'][0]['reason'])
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT character_context_id FROM character_detections WHERE id='d2'").fetchone()[0], None)
            self.assertEqual(conn.execute('SELECT status FROM agent_runs WHERE id=?', (run_id,)).fetchone()[0], 'running')
            conn.execute("UPDATE character_detections SET character_context_id='mai' WHERE id='d1'")  # back to what the agent wrote
        self.assertEqual(len(self.client.post(f'/api/agent-runs/{run_id}/undo').json()['undone']), 1)
        with db.connect() as conn:
            self.assertEqual(conn.execute('SELECT status FROM agent_runs WHERE id=?', (run_id,)).fetchone()[0], 'undone')

    def test_undo_only_restores_registered_columns(self):
        with db.connect() as conn:
            decision = agent_log.record(conn, episode_id='e', kind='x', target_table='cuts', target_id='c1',
                                        before={'start_time': 0}, after={'start_time': 1})
            other = agent_log.record(conn, episode_id='e', kind='x', target_table='jobs', target_id='j', before={}, after={'a': 1})
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').status_code, 409)
        self.assertEqual(self.client.post(f'/api/agent-decisions/{other}/undo').status_code, 409)
        with self.assertRaises(ValueError):
            with db.connect() as conn:
                agent_log.record(conn, episode_id='e', kind='x', tier='guess', target_table='cuts', target_id='c1', before={}, after={})

    def test_a_decision_never_blocks_removing_a_cut(self):
        with db.connect() as conn:
            agent_log.record(conn, episode_id='e', cut_id='c2', kind='x', target_table='cuts', target_id='c2', before={'no_characters': 0}, after={'no_characters': 1})
            conn.execute("DELETE FROM cuts WHERE id='c2'")  # what cut_history does when it undoes a split
            self.assertEqual(conn.execute("SELECT cut_id FROM agent_decisions").fetchone()[0], 'c2')

    # --- escalations -----------------------------------------------------------------------------------------------

    def test_one_open_escalation_per_group_collects_its_cuts(self):
        body = {'kind': 'new_character', 'group_key': 'face:cluster-1', 'question': 'Who is this?', 'cut_ids': ['c1'], 'options': ['Mai', 'New character']}
        first = self.client.post('/api/episodes/e/escalations', json=body)
        self.assertEqual(first.status_code, 200, first.text)
        second = self.client.post('/api/episodes/e/escalations', json={**body, 'cut_ids': ['c2', 'c1'], 'question': 'Who is this child?'}).json()
        self.assertEqual((second['id'], second['cut_ids'], second['question']), (first.json()['id'], ['c1', 'c2'], 'Who is this child?'))
        listed = self.client.get('/api/episodes/e/escalations').json()
        self.assertEqual((len(listed['items']), listed['counts']['open']), (1, {'new_character': 1}))
        closed = self.client.post(f"/api/escalations/{second['id']}/close", json={'status': 'resolved', 'resolution': {'character': 'kaede'}})
        self.assertEqual(closed.json()['resolution'], {'character': 'kaede'})
        self.assertEqual(self.client.post(f"/api/escalations/{second['id']}/close", json={'status': 'dismissed'}).status_code, 409)
        reopened = self.client.post('/api/episodes/e/escalations', json=body).json()
        self.assertNotEqual(reopened['id'], second['id'])  # a closed group can be asked again
        self.assertEqual(len(self.client.get('/api/episodes/e/escalations?status=all').json()['items']), 2)

    def test_escalations_only_name_cuts_and_runs_of_their_episode(self):
        body = {'kind': 'voice', 'group_key': 'SPEAKER_04', 'question': 'Whose voice?'}
        self.assertEqual(self.client.post('/api/episodes/e/escalations', json={**body, 'cut_ids': ['o1']}).status_code, 422)
        with db.connect() as conn:
            run_id = agent_log.start_run(conn, 'e2')
        self.assertEqual(self.client.post('/api/episodes/e/escalations', json={**body, 'run_id': run_id}).status_code, 422)
        self.assertEqual(self.client.post('/api/episodes/missing/escalations', json=body).status_code, 404)

    # --- profile ---------------------------------------------------------------------------------------------------

    def test_the_profile_starts_from_the_harness_defaults(self):
        response = self.client.get('/api/projects/p/autopilot-profile').json()
        profile = response['profile']
        self.assertEqual({key for key, mode in profile['modes'].items() if mode != 'apply'}, set(autopilot_profile.OFF_BY_DEFAULT))
        self.assertEqual({profile['modes'][key] for key in autopilot_profile.OFF_BY_DEFAULT}, {'off'})
        self.assertEqual(profile['claude'], {'enabled': True, 'max_items_per_run': autopilot_profile.DEFAULT_CLAUDE_ITEMS})
        self.assertEqual(set(profile['topics']), set(autopilot_profile.TOPICS))
        self.assertIsNone(response['updated_at'])
        self.assertEqual(set(response['decisions']), set(autopilot_profile.DECISIONS))
        self.assertEqual(self.client.get('/api/projects/missing/autopilot-profile').status_code, 404)

    def test_saving_keeps_only_valid_choices_of_this_project(self):
        saved = self.client.put('/api/projects/p/autopilot-profile', json={'profile': {
            'modes': {'voices': 'ask', 'unknown_kind': 'off'},
            'claude': {'enabled': False, 'max_items_per_run': 10},
            'topics': {'speaking': {'rule': ' Lines with the face turned away are inner voice. ', 'when_unsure': 'ask'}, 'nonsense': {}},
            'characters': {'mai': {'relevance': 'always'}, 'kaede': {'relevance': 'auto'}, 'x': {'relevance': 'background'}},
            'look_alikes': [['nodoka', 'mai'], ['mai', 'nodoka'], ['mai', 'x']],
            'notes': 'Flashbacks blur at the edges.'}})
        self.assertEqual(saved.status_code, 200, saved.text)
        profile = saved.json()['profile']
        self.assertEqual(profile['modes']['voices'], 'ask')
        self.assertNotIn('unknown_kind', profile['modes'])
        self.assertEqual(profile['claude'], {'enabled': False, 'max_items_per_run': 10})
        self.assertEqual(profile['topics']['speaking'], {'rule': 'Lines with the face turned away are inner voice.', 'when_unsure': 'ask'})
        self.assertEqual(profile['characters'], {'mai': {'relevance': 'always'}})  # auto is the default; other projects' characters are dropped
        self.assertEqual(profile['look_alikes'], [['mai', 'nodoka']])
        self.assertIsNotNone(saved.json()['updated_at'])
        self.assertEqual(self.client.get('/api/projects/p/autopilot-profile').json()['profile'], profile)
        with db.connect() as conn:
            conn.execute("DELETE FROM context_items WHERE id='nodoka'")
        self.assertEqual(self.client.get('/api/projects/p/autopilot-profile').json()['profile']['look_alikes'], [])

    def test_invalid_choices_are_refused_with_a_reason(self):
        for bad in ({'modes': {'voices': 'sometimes'}}, {'claude': {'max_items_per_run': 9999}}, {'claude': {'max_items_per_run': 'many'}},
                    {'topics': {'pose': {'when_unsure': 'guess'}}}, {'topics': {'pose': {'rule': 'x' * 700}}},
                    {'characters': {'mai': {'relevance': 'sometimes'}}}, {'look_alikes': [['mai']]}, {'look_alikes': [['mai', 'mai']]}):
            response = self.client.put('/api/projects/p/autopilot-profile', json={'profile': bad})
            self.assertEqual(response.status_code, 422, bad)
            self.assertTrue(response.json()['detail'])

    def test_topic_stats_count_the_users_answers_against_the_recommendation(self):
        review = {'questions': [
            {'id': 'q1', 'topic': 'speaking', 'question': 'Mouth moving?', 'recommended': 'a',
             'options': [{'id': 'a', 'text': 'Yes, her mouth moves.'}, {'id': 'b', 'text': 'No, it stays closed.'}]},
            {'id': 'q2', 'topic': 'gaze', 'question': 'Where?', 'recommended': 'a',
             'options': [{'id': 'a', 'text': 'At the box'}, {'id': 'b', 'text': 'Forward'}]},
            {'id': 'q3', 'topic': 'pose', 'question': 'Pose?', 'recommended': 'b',
             'options': [{'id': 'a', 'text': 'Sitting'}, {'id': 'b', 'text': 'Standing'}]}]}
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('r','c1','cut_description','Consistency summary',?,'model')", (json.dumps(review),))
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content) VALUES ('c1',?)",
                         ('- Speaking (Mai): Yes, her mouth moves.\n- Gaze (Mai): Forward\n',))
        stats = self.client.get('/api/projects/p/autopilot-profile').json()['topic_stats']
        self.assertEqual(stats['speaking'], {'asked': 1, 'answered': 1, 'recommended': 1})
        self.assertEqual(stats['gaze'], {'asked': 1, 'answered': 1, 'recommended': 0})
        self.assertEqual(stats['pose'], {'asked': 1, 'answered': 0, 'recommended': 0})


if __name__ == '__main__':
    unittest.main()
