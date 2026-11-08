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

    def queue_from(self, number=None):
        return self.client.post('/api/episodes/e/review-answers' + (f'?from_cut={number}' if number else ''))

    def test_batch_queues_ready_cuts_from_here_and_lists_the_rest(self):
        self.seed(cuts=('c1',))  # c2 has no consistency summary yet
        body = self.queue_from(1).json()
        self.assertEqual((body['queued'], body['blocked']), (1, {'no_consistency': [2], 'busy': [], 'nothing_to_review': []}))
        payload = self.payload(body['job_ids'][0])
        self.assertEqual(payload['cut_id'], 'c1')
        self.assertTrue(payload['auto_save'])
        later = self.queue_from(2).json()  # earlier cuts are left out
        self.assertEqual((later['queued'], later['blocked']['no_consistency']), (0, [2]))

    def test_batch_skips_cuts_that_are_busy_or_have_nothing_to_decide(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET content=? WHERE id='c2consistency_summary'", (json.dumps({'human_review': 'None required.', 'recommended_checks': 'None required.'}),))
        self.assertEqual(self.client.post('/api/cuts/c1/review-answers').status_code, 200)  # a draft is already queued for c1
        body = self.queue_from().json()
        self.assertEqual((body['queued'], body['blocked']), (0, {'no_consistency': [], 'busy': [1], 'nothing_to_review': [2]}))
        self.client.put('/api/cuts/c2/brief-instructions', json={'content': 'Keep the coat green'})  # saved answers are worth keeping
        self.assertEqual(self.queue_from(2).json()['queued'], 1)

    def test_batch_waits_for_media_work(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json,status) VALUES ('prep','e','prepare','{}','running')")
        self.assertEqual(self.queue_from().status_code, 409)

    def run_draft(self, job_id):
        with patch('app.inference_worker.chat_json', return_value=({'answers': '- Coat: green'}, 'model/x')):
            inference_worker.process_review_answers(*worker.claim_job(('review_answers',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')

    def test_batch_job_saves_the_answers_it_drafted(self):
        self.seed(cuts=('c1',))
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Coat is green'})
        job_id = self.queue_from().json()['job_ids'][0]
        self.run_draft(job_id)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], '- Coat: green')
        self.assertEqual(self.client.get('/api/cuts/c1/review-answers').json()['content'], '- Coat: green')
        self.assertEqual(db.one('SELECT stage FROM jobs WHERE id=?', (job_id,))['stage'], 'Complete')
        self.assertEqual(self.payload(job_id)['existing_answers'], 'Coat is green')  # the replaced text stays in the job

    def test_batch_job_does_not_overwrite_answers_edited_while_it_waited(self):
        self.seed(cuts=('c1',))
        job_id = self.queue_from().json()['job_ids'][0]
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Typed meanwhile'})
        self.run_draft(job_id)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], 'Typed meanwhile')
        self.assertEqual(self.client.get('/api/cuts/c1/review-answers').json()['content'], '- Coat: green')  # still offered as a draft
        self.assertIn('not saved', db.one('SELECT stage FROM jobs WHERE id=?', (job_id,))['stage'])

    def test_single_cut_draft_still_only_stores_a_draft(self):
        self.seed(cuts=('c1',))
        job_id = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        self.run_draft(job_id)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], '')

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            review_answers.validate_answers('  ')
