import json
import unittest
from unittest.mock import patch

from app import db, inference_worker, worker
import tests.test_analysis as analysis_tests


class BriefAnswerQueueTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_answers_can_be_saved_during_a_running_summary_and_restart_it(self):
        self.assertEqual(self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Original answer'}).status_code, 200)
        first = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'})
        self.assertEqual(first.status_code, 200, first.text)
        first_id = first.json()['job_id']
        self.assertNotIn('human_instructions', json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (first_id,))['payload_json']))
        claim = worker.claim_job(('regen',))

        def edit_while_writing(model, system, request, *args, **kwargs):
            self.assertEqual(json.loads(request)['human_instructions'], 'Original answer')
            saved = self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Use the new decision'})
            self.assertEqual(saved.status_code, 200, saved.text)
            return analysis_tests.PRODUCTION_TEXT, 'test/model'

        with patch.object(inference_worker, 'chat_json', side_effect=edit_while_writing):
            inference_worker.process_regen(*claim)
        worker.release_lease(*claim)
        self.assertEqual(db.one('SELECT status,stage,attempt_count FROM jobs WHERE id=?', (first_id,)),
                         {'status': 'queued', 'stage': 'Inputs changed, restarting', 'attempt_count': 0})
        self.assertIsNone(db.one('SELECT id FROM analysis_results WHERE job_id=?', (first_id,)))

        with db.connect() as conn:
            conn.execute('UPDATE jobs SET not_before=0 WHERE id=?', (first_id,))
        with patch.object(inference_worker, 'chat_json', return_value=(analysis_tests.PRODUCTION_TEXT, 'test/model')) as chat:
            inference_worker.process_regen(*worker.claim_job(('regen',)))
        self.assertEqual(json.loads(chat.call_args.args[2])['human_instructions'], 'Use the new decision')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (first_id,))['status'], 'done')
