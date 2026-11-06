import json
import unittest

from app import db
import tests.test_analysis as analysis_tests


class BriefAnswerQueueTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_answers_can_be_saved_during_a_running_summary_for_the_next_job(self):
        self.assertEqual(self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Original answer'}).status_code, 200)
        first = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'})
        self.assertEqual(first.status_code, 200, first.text)
        first_id = first.json()['job_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running' WHERE id=?", (first_id,))

        saved = self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Answer for next time'})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], 'Answer for next time')
        self.assertEqual(json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (first_id,))['payload_json'])['human_instructions'], 'Original answer')

        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done' WHERE id=?", (first_id,))
        next_job = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'})
        self.assertEqual(next_job.status_code, 200, next_job.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (next_job.json()['job_id'],))['payload_json'])
        self.assertEqual(payload['human_instructions'], 'Answer for next time')
