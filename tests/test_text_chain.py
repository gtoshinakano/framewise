import json
import unittest
from unittest.mock import patch

from app import db, worker, inference_worker
import tests.test_analysis as analysis_tests
import tests.test_review_answers as review_tests

TEXT = ('consistency_summary', 'review_answers', 'regen')


class TextChainTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown
    seed = review_tests.ReviewAnswersTest.seed
    payload = review_tests.ReviewAnswersTest.payload

    def chain(self, number=None, only_missing=None):
        query = [f'from_cut={number}'] if number else []
        if only_missing is not None:
            query.append(f'only_missing={str(only_missing).lower()}')
        return self.client.post('/api/episodes/e/text-chain' + ('?' + '&'.join(query) if query else ''))

    def jobs(self, pipeline):
        return {self.payload(row['id'])['cut_id']: row['id'] for row in db.all_rows("SELECT id FROM jobs WHERE pipeline=? ORDER BY rowid", (pipeline,))}

    def finish_consistency(self, cut, human_review='Confirm the NEW coat colour.'):
        """Stand in for the consistency job: store its result and mark it done."""
        job_id = self.jobs('consistency_summary')[cut]
        with db.connect() as conn:
            conn.execute("DELETE FROM analysis_results WHERE cut_id=? AND kind='cut_description' AND title='Consistency summary'", (cut,))
            conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,job_id) VALUES (?,?,?,?,?,?,?)',
                         (cut + 'new', cut, 'cut_description', 'Consistency summary', json.dumps({'human_review': human_review, 'recommended_checks': 'None required.'}),
                          json.dumps({'structured_type': 'consistency_summary'}), job_id))
            conn.execute("UPDATE jobs SET status='done' WHERE id=?", (job_id,))
        return job_id

    def test_queues_three_ordered_jobs_per_ready_cut(self):
        self.seed(consistency=False)
        body = self.chain().json()
        self.assertEqual((body['queued'], body['skipped'], len(body['job_ids'])), (2, 0, 6))
        consistency, answers, summary = self.jobs('consistency_summary'), self.jobs('review_answers'), self.jobs('regen')
        for cut in ('c1', 'c2'):
            a, r = self.payload(answers[cut]), self.payload(summary[cut])
            self.assertEqual(a['wait_for'], consistency[cut])
            self.assertTrue(a['auto_save'])
            self.assertEqual((r['wait_for'], r['depends_on_consistency'], r['refresh_answers']), (answers[cut], consistency[cut], True))

    def test_only_consistency_and_answers_when_a_cut_summary_is_kept(self):
        self.seed(consistency=False)
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('s1','c1','cut_description','Cut summary','Written by hand')")
        body = self.chain().json()
        self.assertEqual((body['queued'], body['skipped'], len(body['job_ids'])), (2, 1, 5))
        self.assertNotIn('c1', self.jobs('regen'))
        again = self.chain(only_missing=False).json()  # c1's chain has no summary step yet, c2's is complete
        self.assertEqual((again['attached'], again['blocked']['busy']), (1, [2]))

    def test_replacing_summaries_adds_the_step_to_chains_queued_without_it(self):
        self.seed(consistency=False)
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('s1','c1','cut_description','Cut summary','Old')")
        self.chain()  # c1 keeps its summary, so it has no Cut summary job
        self.assertNotIn('c1', self.jobs('regen'))
        body = self.chain(only_missing=False).json()
        self.assertEqual((body['queued'], body['attached'], len(body['job_ids']), body['blocked']['busy']), (0, 1, 1, [2]))  # c2's chain is complete
        brief = self.payload(self.jobs('regen')['c1'])
        self.assertEqual((brief['wait_for'], brief['depends_on_consistency'], brief['refresh_answers'], brief['base_id']),
                         (self.jobs('review_answers')['c1'], self.jobs('consistency_summary')['c1'], True, 's1'))
        self.assertEqual(self.chain(only_missing=False).json()['attached'], 0)  # not added twice

    def test_lists_cuts_that_are_busy_or_not_analysed(self):
        self.seed(cuts=('c1',), consistency=False)  # c2 has no visual results
        body = self.chain().json()
        self.assertEqual((body['queued'], body['blocked']), (1, {'missing_results': [2], 'busy': []}))
        again = self.chain().json()
        self.assertEqual((again['queued'], again['blocked']['busy']), (0, [1]))
        self.assertEqual(self.chain(2).json()['blocked']['missing_results'], [2])

    def test_later_jobs_wait_for_the_job_before_them(self):
        self.seed(consistency=False)
        self.chain()
        consistency = self.jobs('consistency_summary')
        first = worker.claim_job(TEXT)
        self.assertEqual(first[0], consistency['c1'])
        second = worker.claim_job(TEXT)  # a second lane: the answers for c1 must not start, so it takes c2's summary
        self.assertEqual(second[0], consistency['c2'])
        self.assertIsNone(worker.claim_job(TEXT))  # everything else waits
        self.finish_consistency('c1')
        self.assertEqual(worker.claim_job(TEXT)[0], self.jobs('review_answers')['c1'])
        self.assertIsNone(worker.claim_job(TEXT))  # c1's Cut summary still waits for its answers

    def test_a_failed_job_cancels_the_ones_waiting_for_it(self):
        self.seed(consistency=False)
        self.chain()
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (self.jobs('consistency_summary')['c1'],))
        claimed = worker.claim_job(TEXT)
        self.assertEqual(claimed[0], self.jobs('consistency_summary')['c2'])
        for pipeline in ('review_answers', 'regen'):
            row = db.one('SELECT status,stage FROM jobs WHERE id=?', (self.jobs(pipeline)['c1'],))
            self.assertEqual((row['status'], row['stage']), ('cancelled', 'Cancelled'))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (self.jobs('review_answers')['c2'],))['status'], 'queued')

    def test_answers_use_the_consistency_summary_written_by_the_chain_and_the_latest_saved_answers(self):
        self.seed(consistency=False)
        self.chain()
        self.client.put('/api/cuts/c1/brief-instructions', json={'content': 'Typed after queuing'})
        self.finish_consistency('c1')
        sent = {}

        def fake_chat(model, system, user, *args, **kwargs):
            sent.update(json.loads(user))
            return {'answers': '- Coat: green'}, 'model/x'
        answers_id = self.jobs('review_answers')['c1']
        with patch('app.inference_worker.chat_json', side_effect=fake_chat):
            inference_worker.process_review_answers(*worker.claim_job(('review_answers',)))
        self.assertEqual(sent['review']['human_review'], 'Confirm the NEW coat colour.')
        self.assertEqual(sent['existing_answers'], 'Typed after queuing')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (answers_id,))['status'], 'done')
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], '- Coat: green')

    def test_answers_are_skipped_when_the_new_summary_leaves_nothing_to_decide(self):
        self.seed(cuts=('c1',), consistency=False)
        self.chain()
        self.finish_consistency('c1', human_review='None required.')
        with patch('app.inference_worker.chat_json') as chat:
            inference_worker.process_review_answers(*worker.claim_job(('review_answers',)))
        chat.assert_not_called()
        row = db.one('SELECT status,stage FROM jobs WHERE id=?', (self.jobs('review_answers')['c1'],))
        self.assertEqual((row['status'], row['stage']), ('done', 'Nothing to decide'))
        self.assertEqual(worker.claim_job(TEXT)[0], self.jobs('regen')['c1'])  # the summary is still released


if __name__ == '__main__':
    unittest.main()
