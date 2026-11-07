import json
import unittest
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, worker, inference_worker, review_answers


class ReviewAnswersTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def seed(self, cuts=('c1', 'c2'), consistency=True):
        with db.connect() as conn:
            for cut in cuts:
                kinds = [('initial_frame', 'Initial frame description'), ('video_analysis', 'Video analysis'), ('last_frame', 'Last frame description')]
                if consistency:
                    kinds.append(('consistency_summary', 'Consistency summary'))
                for kind, title in kinds:
                    content = {'human_review': 'Confirm coat colour.', 'recommended_checks': 'Check frame 2.'} if kind == 'consistency_summary' else {'observation': cut + kind}
                    conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES (?,?,?,?,?,?)',
                                 (cut + kind, cut, 'cut_description', title, json.dumps(content), json.dumps({'structured_type': kind})))

    def payload(self, job_id):
        return json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])

    def test_requires_consistency_summary(self):
        self.seed(consistency=False)
        self.assertEqual(self.client.post('/api/cuts/c1/review-answers').status_code, 409)

    def test_queue_snapshots_engine_answers_and_neighbour_evidence(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('s2','c2','cut_description','Cut summary','Hero keeps the green coat.')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin) VALUES ('t',  'e',0.2,1,'hello','manual')")
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Coat is green'})
        first = self.client.post('/api/cuts/c1/review-answers')
        self.assertEqual(first.status_code, 200, first.text)
        payload = self.payload(first.json()['job_id'])
        evidence = payload['evidence']
        self.assertEqual(evidence['existing_answers'], 'Coat is green')
        self.assertEqual(evidence['review']['human_review'], 'Confirm coat colour.')
        self.assertIn('hello', evidence['current_cut']['transcript'])
        self.assertEqual(evidence['next_cut']['cut_summary'], 'Hero keeps the green coat.')
        self.assertNotIn('video_analysis', evidence['next_cut'])
        self.assertNotIn('previous_cut', evidence)
        self.assertIn('text_provider', payload['config'])
        self.assertEqual(self.client.post('/api/cuts/c1/review-answers').status_code, 409)

    def test_neighbour_without_summary_falls_back_to_video_analysis_and_transcript(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin) VALUES ('t','e',2.5,3,'next line','manual')")
        job = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        neighbour = self.payload(job)['evidence']['next_cut']
        self.assertIsNone(neighbour['cut_summary'])
        self.assertIn('c2video_analysis', neighbour['video_analysis'])
        self.assertIn('next line', neighbour['transcript'])

    def test_worker_stores_draft(self):
        self.seed()
        job_id = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        with patch('app.inference_worker.chat_json', return_value=({'answers': '- Coat: green'}, 'model/x')):
            inference_worker.process_review_answers(*worker.claim_job(('review_answers',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(self.client.get('/api/cuts/c1/review-answers').json(), {'job_id': job_id, 'content': '- Coat: green'})

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            review_answers.validate_answers('  ')
