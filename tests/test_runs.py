import json
import re
import threading
import time
import unittest
from pathlib import Path

import tests.test_analysis as analysis_tests
from app import db, job_kinds, runs, worker


def add_job(job_id, pipeline, status='queued', payload=None, episode='e', created=None, finished=None, **columns):
    names = ['id', 'episode_id', 'pipeline', 'status', 'payload_json', *columns]
    values = [job_id, episode, pipeline, status, json.dumps(payload or {}), *columns.values()]
    if created:
        names.append('created_at')
        values.append(created)
    if finished:
        names.append('finished_at')
        values.append(finished)
    with db.connect() as conn:
        conn.execute(f"INSERT INTO jobs({','.join(names)}) VALUES ({','.join('?' for _ in names)})", values)


def visual(*cut_ids, stages=None):
    return {'workflow': 'single_cut_v2', **({'stages': stages} if stages else {}),
            'cuts': [{'cut_id': cut_id, 'context': {'cut': {'cut_index': index, 'title': '', 'start_time': 0, 'end_time': 1}}} for index, cut_id in enumerate(cut_ids)]}


class RunsTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def queue(self):
        response = self.client.get('/api/episodes/e/queue')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def summary(self):
        with db.connect() as conn:
            return runs.summary(conn, 'e')

    def test_every_pipeline_a_worker_claims_has_a_kind(self):
        source = Path(job_kinds.__file__).parent
        claimed = set(worker.TEXT_PIPELINES)
        for name in ('inference_worker.py', 'audio_worker.py', 'worker.py'):
            text = (source / name).read_text()
            for group in re.findall(r"(?:claim_job\(|main_pipelines = |def work_loop\(pipelines=|def claim_job\(pipelines=)\(([^)]*)\)", text):
                claimed.update(re.findall(r"['\"]([a-z_]+)['\"]", group))
        self.assertTrue({'analyze', 'transcribe', 'prepare', 'brief_export', 'subtitles'} <= claimed, claimed)
        self.assertEqual(sorted(claimed - set(job_kinds.KINDS)), [])

    def test_cancelling_is_not_a_failure(self):
        add_job('summary', 'regen', payload={'cut_id': 'c1'})
        response = self.client.post('/api/jobs/summary/cancel')
        self.assertEqual(response.status_code, 200, response.text)
        job = db.one('SELECT status,stage,error,finished_at FROM jobs WHERE id=?', ('summary',))
        self.assertEqual((job['status'], job['stage'], job['error']), ('cancelled', 'Cancelled', 'Cancelled before processing'))
        self.assertIsNotNone(job['finished_at'])
        state = self.queue()
        self.assertEqual((state['summary']['failed'], state['summary']['attention'], state['attention']), (0, 0, []))
        history = self.client.get('/api/episodes/e/runs').json()
        self.assertEqual(history['runs'][0]['status'], 'cancelled')
        self.assertEqual(history['counts'], {'done': 0, 'failed': 0, 'cancelled': 1})
        self.assertEqual(self.client.post('/api/jobs/summary/cancel').status_code, 409)

    def test_old_cancels_stored_as_failures_become_cancelled(self):
        add_job('old', 'regen', 'failed', {'cut_id': 'c1'}, stage='Cancelled')
        add_job('broken', 'regen', 'failed', {'cut_id': 'c2'}, stage='Failed')
        db.init_db()
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='old'")['status'], 'cancelled')
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='broken'")['status'], 'failed')

    def test_visual_analysis_counts_one_step_per_cut_and_stage(self):
        add_job('batch', 'analyze', 'running', visual('c1', 'c2'))
        now = time.time()
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_job_stages(id,job_id,cut_id,stage_key,label,status,started_at,last_seen_at) VALUES ('s1','batch','c1','initial_frame','Initial frame','done',?,?),('s2','batch','c2','initial_frame','Initial frame','running',?,?)", (now, now, now, now))
        summary = self.summary()
        self.assertEqual({key: summary[key] for key in ('done', 'running', 'waiting', 'total', 'finished')},
                         {'done': 1, 'running': 1, 'waiting': 2, 'total': 4, 'finished': 1})
        run = self.queue()['running'][0]
        self.assertEqual((run['label'], run['units'], [cut['number'] for cut in run['cuts']]), ('Visual analysis', {'done': 1, 'total': 4}, [1, 2]))
        # A failed batch keeps its finished step, counts the failing one once and drops the rest.
        add_job('other', 'regen', payload={'cut_id': 'c2'})
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed',stage='Failed' WHERE id='batch'")
        summary = self.summary()
        self.assertEqual({key: summary[key] for key in ('done', 'failed', 'waiting', 'total')}, {'done': 1, 'failed': 1, 'waiting': 1, 'total': 3})

    def test_every_gpu_kind_counts_and_media_work_does_not(self):
        for job_id, pipeline, payload in [('a', 'review_answers', {'cut_id': 'c1'}), ('b', 'sequence', {'sequence_id': 'q', 'step': 'plan'}),
                                          ('c', 'subtitles', {'language': 'ja'}), ('d', 'transcribe', {}), ('v', 'voice_sample', {}),
                                          ('m', 'composition_compact', {'cut_id': 'c1'}), ('p', 'cutframes', {'cut_ids': ['c1']})]:
            add_job(job_id, pipeline, payload=payload)
        summary = self.summary()
        self.assertEqual((summary['waiting'], summary['total'], summary['other_active']), (6, 6, 1))

    def test_idle_episode_shows_no_counter_and_a_busy_period_reaches_back(self):
        add_job('long-ago', 'regen', 'done', {'cut_id': 'c1'}, created='2026-01-01 10:00:00', finished='2026-01-01 10:01:00')
        add_job('first', 'regen', 'done', {'cut_id': 'c1'}, created='2026-01-02 10:00:00', finished='2026-01-02 10:10:00')
        self.assertEqual((self.summary()['busy'], self.summary()['total']), (False, 0))
        add_job('second', 'regen', 'done', {'cut_id': 'c2'}, created='2026-01-02 10:05:00', finished='2026-01-02 10:20:00')
        add_job('now', 'consistency_summary', 'running', {'cut_id': 'c2'}, created='2026-01-02 10:15:00')
        summary = self.summary()
        self.assertEqual((summary['busy'], summary['since'], summary['done'], summary['running'], summary['total']), (True, '2026-01-02T10:00:00Z', 2, 1, 3))

    def test_every_cut_scoped_run_names_its_cut(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET title='Opening' WHERE id='c1'")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,active) VALUES ('gone','e',-1,4,5,0)")
        cases = {'regen': {'cut_id': 'c1'}, 'consistency_summary': {'cut_id': 'c1'}, 'review_answers': {'cut_id': 'c1'},
                 'composition_compact': {'cut_id': 'c1'}, 'character_detect': {'cut_id': 'c1', 'cut': {'cut_index': 0}, 'workflow': 'anime_bulk_v1', 'phase': 'boxes'},
                 'hardcuts': {'cut_id': 'c1'}, 'cutframes': {'cut_ids': ['c1', 'c2']}, 'analyze': visual('c1')}
        for pipeline, payload in cases.items():
            add_job(pipeline, pipeline, 'done', payload)
        add_job('archived', 'regen', 'done', {'cut_id': 'gone', 'context': {'cut': {'cut_index': 6}}})
        found = {run['id']: run for run in self.client.get('/api/episodes/e/runs').json()['runs']}
        for pipeline in cases:
            self.assertEqual((found[pipeline]['cuts'][0]['number'], found[pipeline]['cuts'][0]['title']), (1, 'Opening'), pipeline)
        self.assertEqual([cut['number'] for cut in found['cutframes']['cuts']], [1, 2])
        self.assertEqual(found['character_detect']['label'], 'Character boxes')
        self.assertEqual({key: found['archived']['cuts'][0][key] for key in ('number', 'archived')}, {'number': 7, 'archived': True})

    def test_sequence_shot_names_its_sequence_shot_and_cut(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO sequences(id,episode_id,title,first_cut_id,last_cut_id,start_time,end_time) VALUES ('q','e','Chase','c1','c2',0,4)")
            conn.execute("INSERT INTO sequence_results(id,sequence_id,type,content) VALUES ('plan','q','plan',?)",
                         (json.dumps({'shots': [{'start_time': 0, 'end_time': 2, 'frame_cut_id': 'c1'}, {'start_time': 2, 'end_time': 4, 'frame_cut_id': 'c2'}]}),))
        add_job('shot', 'sequence', 'done', {'sequence_id': 'q', 'step': 'shot', 'shot_key': '2000-4000'})
        add_job('replanned', 'sequence', 'done', {'sequence_id': 'q', 'step': 'shot', 'shot_key': '100-1900'})
        found = {run['id']: run for run in self.client.get('/api/episodes/e/runs').json()['runs']}
        self.assertEqual((found['shot']['label'], found['shot']['sequence']['title'], found['shot']['sequence']['shot_number'], found['shot']['cuts'][0]['number']),
                         ('Sequence shot', 'Chase', 2, 2))
        self.assertEqual((found['replanned']['sequence']['shot_number'], found['replanned']['cuts'][0]['number']), (None, 1))

    def test_attention_lists_failures_until_run_again_retried_or_dismissed(self):
        add_job('broken', 'regen', 'failed', {'cut_id': 'c1'}, stage='Failed', error='Model refused')
        add_job('export', 'brief_export', 'failed', {}, stage='Failed')
        state = self.queue()
        self.assertEqual(([run['id'] for run in state['attention']], state['summary']['attention']), (['broken'], 1))
        self.assertTrue(state['attention'][0]['actions']['dismiss'])
        add_job('again', 'regen', 'queued', {'cut_id': 'c1'})
        self.assertEqual(self.queue()['attention'], [])
        add_job('visual', 'analyze', 'failed', visual('c1', 'c2'), stage='Failed')
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json,completed) VALUES ('visual','c1','{}',1)")
        self.assertEqual([run['id'] for run in self.queue()['attention']], ['visual'])
        add_job('only-c2', 'analyze', 'done', visual('c2'))  # the unfinished cut ran again
        self.assertEqual(self.queue()['attention'], [])
        add_job('lost', 'transcribe', 'failed', {}, stage='Failed')
        self.assertEqual(self.client.post('/api/jobs/again/dismiss').status_code, 409)
        self.assertEqual(self.client.post('/api/jobs/lost/dismiss').status_code, 200)
        self.assertEqual(self.queue()['attention'], [])
        add_job('one', 'voice_sample', 'failed', {'character_context_id': 'ctx'}, stage='Failed')
        add_job('two', 'subtitles', 'failed', {'language': 'ja'}, stage='Failed')
        self.assertEqual(self.client.post('/api/episodes/e/queue/dismiss-failures').json(), {'dismissed': 2})
        self.assertEqual(self.summary()['attention'], 0)

    def test_a_retry_deals_with_the_failure(self):
        add_job('failed', 'analyze', 'failed', visual('c1'), stage='Failed')
        add_job('retry', 'analyze', 'queued', visual('c2'), retry_of='failed')
        self.assertEqual(self.queue()['attention'], [])

    def test_a_cancelled_running_job_is_stopping_until_its_worker_lets_go(self):
        add_job('voice', 'transcribe')
        job_id, owner = worker.claim_job(('transcribe',))
        self.assertEqual(self.client.post(f'/api/jobs/{job_id}/cancel').status_code, 200)
        state = self.queue()
        self.assertEqual(([run['status'] for run in state['running']], state['summary']['stopping'], state['summary']['failed']), (['stopping'], 1, 0))
        with self.assertRaises(worker.LeaseLost):
            with db.connect() as conn:
                worker.require_lease(conn, job_id, owner)
        done = threading.Event()
        done.set()
        worker.heartbeat(job_id, owner, done)  # what every worker loop does when the job returns
        state = self.queue()
        self.assertEqual((state['running'], state['summary']['busy']), ([], False))
        self.assertEqual(db.one('SELECT status,lease_owner FROM jobs WHERE id=?', (job_id,)), {'status': 'cancelled', 'lease_owner': None})

    def test_running_detection_and_voice_jobs_can_be_cancelled(self):
        from app import inference_worker
        add_job('boxes', 'character_detect', payload={'cut_id': 'c1', 'workflow': 'anime_bulk_v1', 'phase': 'boxes'})
        add_job('older', 'character_detect', episode='e', payload={'cut_id': 'c2'})
        boxes, owner = worker.claim_job(('character_detect',))
        self.assertEqual(boxes, 'boxes')
        self.assertTrue(self.queue()['running'][0]['actions']['cancel'])
        self.assertEqual(self.client.post('/api/jobs/boxes/cancel').status_code, 200)
        inference_worker.fail_job('boxes', owner, 'e', RuntimeError('lease lost'))  # a worker noticing too late writes nothing
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='boxes'")['status'], 'cancelled')
        older, _ = worker.claim_job(('character_detect',))
        self.assertEqual(older, 'older')
        self.assertEqual(self.client.post('/api/jobs/older/cancel').status_code, 409)  # the older workflow cannot stop midway
        sample = '00000000-0000-4000-8000-000000000001'
        add_job(sample, 'voice_sample', payload={})
        self.assertEqual(self.client.post(f'/api/jobs/{sample}/cancel').status_code, 200)

    def test_cancelling_a_job_cancels_the_chain_waiting_for_it(self):
        add_job('consistency', 'consistency_summary', payload={'cut_id': 'c1'})
        add_job('answers', 'review_answers', payload={'cut_id': 'c1', 'wait_for': 'consistency'})
        add_job('summary', 'regen', payload={'cut_id': 'c1', 'wait_for': 'answers'})
        add_job('unrelated', 'regen', payload={'cut_id': 'c2'})
        self.assertEqual(self.client.post('/api/jobs/consistency/cancel').status_code, 200)
        rows = {row['id']: (row['status'], row['error']) for row in db.all_rows('SELECT id,status,error FROM jobs')}
        self.assertEqual(rows['answers'], ('cancelled', worker.WAITED_FOR_FAILED))
        self.assertEqual(rows['summary'], ('cancelled', worker.WAITED_FOR_FAILED))
        self.assertEqual(rows['unrelated'][0], 'queued')

    def test_episode_queue_cancel_by_group_and_with_running(self):
        for index in range(3):
            add_job(f'box{index}', 'character_detect', payload={'cut_id': 'c1', 'workflow': 'anime_bulk_v1', 'phase': 'boxes'})
        add_job('answers', 'review_answers', payload={'cut_id': 'c2'})
        add_job('prepare', 'prepare')
        state = self.queue()
        groups = {group['label']: group for group in state['waiting_groups']}
        self.assertEqual((groups['Character boxes']['count'], groups['Prepare media']['cancel'], state['waiting_total']), (3, False, 5))
        response = self.client.post('/api/episodes/e/queue/cancel', json={'group': groups['Character boxes']['group']})
        self.assertEqual(response.json(), {'cancelled': 3})
        claimed = worker.claim_job(('review_answers',))
        self.assertEqual(self.client.post('/api/episodes/e/queue/cancel', json={}).json(), {'cancelled': 0})
        self.assertEqual(self.client.post('/api/episodes/e/queue/cancel', json={'include_running': True}).json(), {'cancelled': 1})
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (claimed[0],))['status'], 'cancelled')
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='prepare'")['status'], 'queued')

    def test_history_pages_and_filters_this_episode_only(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e2','p','Other','a.mp4','.mp4','ready','{}')")
        for index in range(5):
            add_job(f'done{index}', 'regen', 'done', {'cut_id': 'c1'})
        add_job('cancelled', 'regen', 'cancelled', {'cut_id': 'c1'})
        add_job('elsewhere', 'regen', 'done', {}, episode='e2')
        first = self.client.get('/api/episodes/e/runs?limit=4').json()
        self.assertEqual([run['id'] for run in first['runs']], ['cancelled', 'done4', 'done3', 'done2'])
        second = self.client.get(f"/api/episodes/e/runs?limit=4&before={first['next']}").json()
        self.assertEqual(([run['id'] for run in second['runs']], second['next']), (['done1', 'done0'], None))
        only = self.client.get('/api/episodes/e/runs?status=cancelled').json()
        self.assertEqual([run['id'] for run in only['runs']], ['cancelled'])
        self.assertEqual(self.client.get('/api/episodes/missing/runs').status_code, 404)

    def test_queue_names_gpu_work_of_other_episodes_and_chat(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e2','p','Other','a.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO chat_gateway_requests(id,status,stage,request_json,created_at) VALUES ('chat','running','Answering','{}',?)", (time.time(),))
        add_job('there', 'analyze', 'running', visual('c1'), episode='e2')
        add_job('there2', 'regen', 'queued', {'cut_id': 'c1'}, episode='e2')
        elsewhere = self.queue()['elsewhere']
        self.assertEqual([(item['title'], item['running'], item['waiting'], item['current']['label']) for item in elsewhere['episodes']], [('Other', 1, 1, 'Visual analysis')])
        self.assertEqual([item['id'] for item in elsewhere['chat_api']], ['chat'])
        self.assertEqual(self.summary()['total'], 0)
        self.assertEqual(self.client.post('/api/chat-gateway/requests/chat/cancel').status_code, 200)
        self.assertEqual(db.one("SELECT cancel_requested FROM chat_gateway_requests WHERE id='chat'")['cancel_requested'], 1)

    def test_status_carries_the_counter_and_jobs_carry_their_actions(self):
        add_job('waiting', 'regen', payload={'cut_id': 'c1'})
        status = self.client.get('/api/episodes/e/status').json()
        self.assertEqual((status['queue']['waiting'], status['queue']['total']), (1, 1))
        job = next(job for job in self.client.get('/api/episodes/e').json()['jobs'] if job['id'] == 'waiting')
        self.assertEqual(job['actions'], {'cancel': True, 'retry': False})


if __name__ == '__main__':
    unittest.main()
