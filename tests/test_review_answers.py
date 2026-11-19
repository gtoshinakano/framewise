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

    def evidence(self, job_id):
        with db.connect() as conn:
            job = db.one('SELECT * FROM jobs WHERE id=?', (job_id,), conn=conn)
            return review_answers.gather(conn, job)['evidence']

    def test_waits_for_consistency_summary(self):
        self.seed(consistency=False)
        parent = self.client.post('/api/cuts/c1/consistency-summary').json()['job_id']
        response = self.client.post('/api/cuts/c1/review-answers')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.payload(response.json()['job_id'])['depends_on'], [parent])
        self.assertIsNone(worker.claim_job(('review_answers',)))

    def test_queue_snapshots_engine_and_reads_live_answers_and_neighbour_evidence(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('s2','c2','cut_description','Cut summary','Hero keeps the green coat.')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin) VALUES ('t',  'e',0.2,1,'hello','manual')")
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Coat is green'})
        first = self.client.post('/api/cuts/c1/review-answers')
        self.assertEqual(first.status_code, 200, first.text)
        payload = self.payload(first.json()['job_id'])
        self.assertNotIn('evidence', payload)
        evidence = self.evidence(first.json()['job_id'])
        self.assertEqual(evidence['existing_answers'], 'Coat is green')
        self.assertEqual(evidence['review']['human_review'], 'Confirm coat colour.')
        self.assertIn('hello', evidence['current_cut']['transcript'])
        self.assertEqual(evidence['next_cut']['consistency_summary']['human_review'], 'Confirm coat colour.')
        self.assertNotIn('Hero keeps the green coat.', json.dumps(evidence))
        self.assertEqual(set(evidence['next_cut']), {'cut_index', 'start_time', 'end_time', 'consistency_summary'})
        self.assertNotIn('previous_cut', evidence)
        self.assertIn('text_provider', payload['config'])
        self.assertEqual(self.client.post('/api/cuts/c1/review-answers').json()['job_id'], first.json()['job_id'])

    def test_neighbour_without_consistency_summary_sends_only_its_boundaries(self):
        self.seed(cuts=('c1',))
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('v2','c2','cut_description','Video analysis','c2 video')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin) VALUES ('t','e',2.5,3,'next line','manual')")
        job = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        neighbour = self.evidence(job)['next_cut']
        self.assertEqual(set(neighbour), {'cut_index', 'start_time', 'end_time'})

    def test_worker_stores_draft(self):
        self.seed()
        job_id = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        with patch('app.inference_worker.chat_json', return_value=({'answers': '- Coat: green'}, 'model/x')):
            inference_worker.process_review_answers(*worker.claim_job(('review_answers',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertEqual(self.client.get('/api/cuts/c1/review-answers').json(), {'job_id': job_id, 'content': '- Coat: green', 'stale_reason': None})

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

    def test_batch_can_queue_beside_explicit_drafts_and_skips_cuts_with_nothing_to_decide(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET content=? WHERE id='c2consistency_summary'", (json.dumps({'human_review': 'None required.', 'recommended_checks': 'None required.'}),))
        self.assertEqual(self.client.post('/api/cuts/c1/review-answers').status_code, 200)  # a draft is already queued for c1
        body = self.queue_from().json()
        self.assertEqual((body['queued'], body['blocked']), (1, {'no_consistency': [], 'busy': [], 'nothing_to_review': [2]}))
        self.client.put('/api/cuts/c2/brief-instructions', json={'content': 'Keep the coat green'})  # saved answers are worth keeping
        self.assertEqual(self.queue_from(2).json()['queued'], 1)

    def test_questions_are_what_the_draft_answers(self):
        self.seed()
        questions = [{'id': 'q1', 'topic': 'gaze', 'character': 'Mai', 'question': 'Where does Mai look?', 'options': [{'id': 'a', 'text': 'At Rio'}, {'id': 'b', 'text': 'Down'}], 'recommended': 'a', 'why': 'first frame'}]
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET content=? WHERE id='c1consistency_summary'", (json.dumps({'summary': 's', 'clear': 'c', 'confusing': 'None identified.', 'questions': questions}),))
            conn.execute("UPDATE analysis_results SET content=? WHERE id='c2consistency_summary'", (json.dumps({'summary': 's', 'clear': 'c', 'confusing': 'None identified.', 'questions': questions}),))
        job_id = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        evidence = self.evidence(job_id)
        self.assertEqual(evidence['review']['questions'], questions)
        self.assertNotIn('human_review', evidence['review'])  # only older summaries carry the two text fields
        self.assertNotIn('questions', evidence['next_cut']['consistency_summary'])  # a neighbour's open questions are not facts about it
        self.assertTrue(review_answers.has_review_items(evidence))
        self.assertFalse(review_answers.has_review_items({'review': {'questions': [], 'confusing': 'None identified.'}}))
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': '- Gaze (Mai): At Rio'})
        self.assertEqual(self.evidence(job_id)['existing_answers'], '- Gaze (Mai): At Rio')
        for rule in ('copy that option\'s text word for word', '- Gaze (Name): option text'):
            self.assertIn(rule, review_answers.SYSTEM)

    def test_batch_waits_for_media_work(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json,status) VALUES ('prep','e','prepare','{}','running')")
        response = self.queue_from()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['job_ids'])
        self.assertEqual(self.payload(response.json()['job_ids'][0])['depends_on'], ['prep'])

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
        provenance = json.loads(db.one('SELECT input_json FROM jobs WHERE id=?', (job_id,))['input_json'])
        self.assertEqual(provenance['existing_answers'], 'Coat is green')

    def test_batch_job_does_not_overwrite_answers_edited_while_it_waited(self):
        self.seed(cuts=('c1',))
        job_id = self.queue_from().json()['job_ids'][0]
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Typed meanwhile'})
        self.run_draft(job_id)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], 'Typed meanwhile')
        self.assertEqual(self.client.get('/api/cuts/c1/review-answers').json()['content'], '- Coat: green')  # still offered as a draft
        self.assertIn('saved answers kept', db.one('SELECT stage FROM jobs WHERE id=?', (job_id,))['stage'])

    def test_editing_answers_during_drafting_restarts_with_the_new_text(self):
        self.seed(cuts=('c1',))
        job_id = self.queue_from().json()['job_ids'][0]
        claim = worker.claim_job(('review_answers',))

        def edit(*args, **kwargs):
            response = self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Keep my decision'})
            self.assertEqual(response.status_code, 200)
            return '- Coat: old', 'model/x'

        with patch.object(review_answers, 'generate', side_effect=edit):
            inference_worker.process_review_answers(*claim)
        worker.release_lease(*claim)
        self.assertEqual(db.one('SELECT status,stage FROM jobs WHERE id=?', (job_id,)),
                         {'status': 'queued', 'stage': 'Inputs changed, restarting'})
        self.assertIsNone(db.one('SELECT cut_id FROM cut_review_drafts'))
        with db.connect() as conn:
            conn.execute('UPDATE jobs SET not_before=0 WHERE id=?', (job_id,))
        with patch.object(review_answers, 'generate', return_value=('- Coat: new draft', 'model/x')) as generate:
            inference_worker.process_review_answers(*worker.claim_job(('review_answers',)))
        self.assertEqual(generate.call_args.args[1]['existing_answers'], 'Keep my decision')
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], 'Keep my decision')
        self.assertEqual(self.client.get('/api/cuts/c1/review-answers').json()['content'], '- Coat: new draft')

    def test_single_cut_draft_still_only_stores_a_draft(self):
        self.seed(cuts=('c1',))
        job_id = self.client.post('/api/cuts/c1/review-answers').json()['job_id']
        self.run_draft(job_id)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], '')

    def test_saved_answers_report_the_last_saved_time(self):
        self.seed(cuts=('c1',))
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['updated_at'], None)
        saved = self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Coat is green'})
        self.assertEqual(saved.status_code, 200, saved.text)
        stamp = saved.json()['updated_at']
        self.assertTrue(isinstance(stamp, str) and stamp)
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['updated_at'], stamp)

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            review_answers.validate_answers('  ')
