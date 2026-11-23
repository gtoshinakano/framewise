"""GET /api/episodes/{id}/cut-needs: the review report regrouped by cut, plus open escalations and agent decisions."""
import unittest

from app import agent_log, db
from tests import test_review_report as fixture


class CutNeedsTest(unittest.TestCase):
    # The review report's episode: c1 unlabeled box, c2 target frames to review, c3 open questions, c4 never detected.
    setUp = fixture.ReviewReportTest.setUp
    tearDown = fixture.ReviewReportTest.tearDown

    def needs(self):
        response = self.client.get('/api/episodes/e/cut-needs')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['cuts']

    def people(self, cuts, cut_id):
        return {item['key']: item['count'] for item in cuts[cut_id]['people']}

    def test_a_low_confidence_speaker_nobody_confirmed_is_its_own_need(self):
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET confidence='low',speaker_source='qwen' WHERE episode_id='e' AND character_context_id IS NOT NULL")
        count = sum(self.people(self.needs(), cut).get('speaker_unsure', 0) for cut in ('c1', 'c2', 'c3', 'c4'))
        with db.connect() as conn:
            expected = conn.execute("SELECT COUNT(*) FROM transcript_segments WHERE episode_id='e' AND superseded_by IS NULL AND deleted=0 AND confidence='low' AND character_context_id IS NOT NULL").fetchone()[0]
            conn.execute("UPDATE transcript_segments SET speaker_source='manual' WHERE episode_id='e'")
        self.assertGreaterEqual(count, 1 if expected else 0)
        self.assertEqual(sum(self.people(self.needs(), cut).get('speaker_unsure', 0) for cut in ('c1', 'c2', 'c3', 'c4')), 0)

    def test_gates_checks_and_steps_by_cut(self):
        cuts = self.needs()
        self.assertEqual(self.people(cuts, 'c1'), {'box_names': 1, 'voice': 1, 'lines_to_check': 1})
        self.assertEqual(self.people(cuts, 'c2'), {'target_frames': 2})
        self.assertEqual(self.people(cuts, 'c3'), {'voice': 2, 'decisions': 2, 'picture_check': 1})
        self.assertEqual(self.people(cuts, 'c4'), {})
        self.assertIn('first_frame', cuts['c1']['steps'])
        self.assertIn('beats', cuts['c2']['steps'])
        self.assertIn('boxes', cuts['c4']['steps'])
        self.assertEqual(cuts['c1']['claude'], [])

    def test_escalations_and_answers(self):
        with db.connect() as conn:
            human = agent_log.escalate(conn, episode_id='e', kind='missed_cut', group_key='m', question='Cut?', cut_ids=['c4'])
            agent_log.escalate(conn, episode_id='e', kind='box_label', group_key='b', question='Ann?', cut_ids=['c1'])
            claude = agent_log.escalate(conn, episode_id='e', kind='review_question', group_key='q', question='Smile?', cut_ids=['c3'], reviewer='claude')
            agent_log.record(conn, episode_id='e', cut_id='c2', kind='box_names', target_table='character_detections', target_id='d2',
                             before={}, after={}, tier='claude')
        cuts = self.needs()
        self.assertEqual(cuts['c4']['people'], [{'key': 'missed_cut', 'count': 1, 'escalation_ids': [human]}])
        # The escalation restates the unlabeled box the report already counted.
        self.assertEqual(self.people(cuts, 'c1')['box_names'], 1)
        # Open questions that wait for Claude are Claude's, not the person's.
        self.assertNotIn('decisions', self.people(cuts, 'c3'))
        self.assertEqual(cuts['c3']['claude'], [{'kind': 'review_question', 'escalation_id': claude}])
        self.assertEqual(cuts['c2']['answered_by'], ['claude'])

    def test_analysis_off_has_no_needs_from_lines(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET analysis_enabled=0 WHERE id='c1'")
        cuts = self.needs()
        self.assertTrue(cuts['c1']['analysis_off'])
        self.assertNotIn('voice', self.people(cuts, 'c1'))
        self.assertEqual(self.client.get('/api/episodes/missing/cut-needs').status_code, 404)


if __name__ == '__main__':
    unittest.main()
