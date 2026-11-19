import json
import unittest

from app import db, summary_jobs, worker
import tests.test_analysis as analysis_tests


class CutSummaryBatchTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def seed(self, cuts=('c1', 'c2'), summary=()):
        with db.connect() as conn:
            for cut in cuts:
                conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES (?,?,?,?,?,?)',
                             (cut + 'consistency', cut, 'cut_description', 'Consistency summary', json.dumps({'human_review': 'x'}), json.dumps({'structured_type': 'consistency_summary'})))
            for cut in summary:
                conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES (?,?,'cut_description','Cut summary','Existing','manual')", (cut + 'summary', cut))

    def queue_from(self, number=None, only_missing=None):
        query = '&'.join(part for part in (f'from_cut={number}' if number else '', f'only_missing={str(only_missing).lower()}' if only_missing is not None else '') if part)
        return self.client.post('/api/episodes/e/cut-summaries' + (f'?{query}' if query else ''))

    def payload(self, job_id):
        return json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])

    def test_queues_a_summary_job_for_each_ready_cut_from_here(self):
        self.seed()
        self.client.put('/api/cuts/c2/brief-instructions', json={'content': 'Coat is green'})
        everything = self.queue_from().json()
        self.assertEqual((everything['queued'], everything['skipped']), (2, 0))
        first = self.payload(everything['job_ids'][0])
        self.assertEqual((first['cut_id'], first['mode'], first['result_type']), ('c1', 'regenerate', 'summary'))
        self.assertNotIn('human_instructions', self.payload(everything['job_ids'][1]))
        with db.connect() as conn:
            job = db.one('SELECT * FROM jobs WHERE id=?', (everything['job_ids'][1],), conn=conn)
            self.assertEqual(summary_jobs.gather(conn, job)['human_instructions'], 'Coat is green')
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done'")
        self.assertEqual(self.queue_from(2).json()['queued'], 1)  # earlier cuts are left out

    def test_cuts_without_a_consistency_summary_are_listed_not_queued(self):
        self.seed(cuts=('c1',))
        body = self.queue_from().json()
        self.assertEqual((body['queued'], body['blocked']['no_consistency']), (1, [2]))

    def test_cuts_with_a_summary_are_kept_unless_asked_to_redo(self):
        self.seed(summary=('c1',))
        body = self.queue_from().json()
        self.assertEqual((body['queued'], body['skipped']), (1, 1))
        self.assertEqual(self.payload(body['job_ids'][0])['cut_id'], 'c2')
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done'")
        redo = self.queue_from(only_missing=False).json()
        self.assertEqual((redo['queued'], redo['skipped']), (2, 0))
        with db.connect() as conn:
            job = db.one('SELECT * FROM jobs WHERE id=?', (redo['job_ids'][0],), conn=conn)
            self.assertEqual(summary_jobs.gather(conn, job)['base']['id'], 'c1summary')

    def test_a_cut_whose_answers_are_still_being_drafted_waits(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('r','e','review_answers',?)", (json.dumps({'cut_id': 'c1'}),))
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('a','e','analyze',?)", (json.dumps({'cuts': [{'cut_id': 'c2'}]}),))
        body = self.queue_from().json()
        self.assertEqual(body['queued'], 2)
        for job_id, predecessor in zip(body['job_ids'], ('r', 'a')):
            self.assertIn(predecessor, self.payload(job_id)['depends_on'])
        self.assertIsNone(worker.claim_job(('regen',)))

    def test_a_summary_already_queued_is_not_queued_twice(self):
        self.seed()
        first = self.queue_from().json()
        again = self.queue_from().json()
        self.assertEqual(first['queued'], 2)
        self.assertEqual(again['job_ids'], first['job_ids'])

    def test_waits_for_media_work(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json,status) VALUES ('prep','e','prepare','{}','running')")
        response = self.queue_from()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.payload(response.json()['job_ids'][0])['depends_on'], ['prep'])

    def test_archived_and_disabled_cuts_are_left_out(self):
        self.seed()
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET analysis_enabled=0 WHERE id='c1'")
            conn.execute("UPDATE cuts SET active=0,cut_index=-1 WHERE id='c2'")
        self.assertEqual(self.queue_from().json(), {'queued': 0, 'skipped': 0, 'blocked': {'no_consistency': [], 'answers_pending': [], 'busy': []}, 'job_ids': []})
