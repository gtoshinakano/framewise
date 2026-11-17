import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import tests.test_analysis as analysis_tests
from app import db, worker, inference_worker, local_text_runtime, remote_text, sequence_evidence
from app.assistant import providers

AUTH = 'Bearer secret'


class FakeLlama(BaseHTTPRequestHandler):
    """An OpenAI-compatible llama.cpp router behind a Bearer token."""
    requests = []

    def log_message(self, *args):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if self.headers.get('Authorization') != AUTH:
            self.reply(401, {'error': 'unauthorized'})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == '/health':
            self.reply(200, {'status': 'ok'})
        elif self.path == '/v1/models':
            self.reply(200, {'data': [{'id': 'qwen3.6-mtp'}, {'id': 'bonsai2'}]})
        elif self.path.startswith('/props?model=qwen3.6-mtp'):
            self.reply(200, {'default_generation_settings': {'n_ctx': 49152}})
        else:
            self.reply(404, {})

    def do_POST(self):
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeLlama.requests.append(body)
        self.reply(200, {'model': body['model'], 'choices': [{'message': {'content': '{"answer": "ok"}'}, 'finish_reason': 'stop'}],
                         'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
                         'timings': {'predicted_n': 4, 'predicted_ms': 100, 'draft_n': 4, 'draft_n_accepted': 3}})


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


class RemoteTextTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def start_server(self):
        FakeLlama.requests = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), FakeLlama)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_address[1]}'

    def configure(self, base_url='http://127.0.0.1:9', routes=None, enabled=True, fallback=''):
        with db.connect() as conn:
            conn.execute('UPDATE remote_text_server SET enabled=?,base_url=?,token=?,parallel=2,routes_json=?,assistant_fallback=? WHERE id=1',
                         (int(enabled), base_url, 'secret', json.dumps(routes or {}), fallback))
        return remote_text.config()

    def job(self, job_id, pipeline, provider):
        payload = {'text_provider': provider} if pipeline == 'regen' else {'config': {'text_provider': provider}}
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,'e',?,?)", (job_id, pipeline, json.dumps(payload)))

    # ------------------------------------------------------------ claiming

    def test_routing_splits_local_provider_jobs_between_lanes(self):
        cfg = self.configure(routes={'review_answers': {'mode': 'remote', 'model': 'bonsai2'}, 'regen': {'mode': 'any', 'model': 'qwen3.6-mtp'}})
        self.assertEqual(remote_text.remote_only(cfg), ('review_answers',))
        self.assertEqual(set(remote_text.remote_pipelines(cfg)), {'review_answers', 'regen'})
        self.job('remote-local', 'review_answers', 'local')
        self.job('remote-router', 'review_answers', 'openrouter')
        # This host's lanes skip the local-provider job routed to the remote but keep its OpenRouter job.
        claimed = worker.claim_job(worker.TEXT_PIPELINES, remote_only=remote_text.remote_only(cfg))
        self.assertEqual(claimed[0], 'remote-router')
        self.assertIsNone(worker.claim_job(worker.TEXT_PIPELINES, remote_only=remote_text.remote_only(cfg)))
        # A remote lane takes only local-provider jobs of routed pipelines.
        self.job('any-local', 'regen', 'local')
        self.job('consistency-local', 'consistency_summary', 'local')
        lane = [worker.claim_job(remote_text.remote_pipelines(cfg), local_provider_only=True) for _ in range(3)]
        self.assertEqual({item[0] for item in lane if item}, {'remote-local', 'any-local'})
        self.assertIsNone(lane[2])
        self.assertIsNone(worker.claim_job((), local_provider_only=True))

    def running(self, job_id, pipeline, owner='other-worker'):
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,lease_owner,lease_until,payload_json) VALUES (?,'e',?,'running',?,?,'{}')",
                         (job_id, pipeline, owner, time.time() + 60))

    def test_remote_lane_runs_text_beside_visual_analysis_without_holding_it_back(self):
        self.running('visual', 'analyze')
        self.job('summary', 'consistency_summary', 'local')
        # This host's text lane waits for the episode's visual job (one GPU); a remote lane does not need it.
        self.assertIsNone(worker.claim_job(worker.TEXT_PIPELINES))
        job_id, owner = worker.claim_job(('consistency_summary',), local_provider_only=True, remote_lane=True)
        self.assertEqual(job_id, 'summary')
        self.assertTrue(owner.startswith(worker.REMOTE_OWNER))
        self.assertEqual(db.one("SELECT lease_owner FROM jobs WHERE id='summary'")['lease_owner'], owner)
        # The remote text job running now does not stop the next visual job of the episode from starting.
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done' WHERE id='visual'")
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('visual-2','e','analyze','{}')")
        self.assertEqual(worker.claim_job(('analyze',))[0], 'visual-2')

    def test_a_job_held_by_this_hosts_text_lane_still_holds_the_episode_from_visual_analysis(self):
        self.running('local-summary', 'consistency_summary')  # plain owner: this host's lane
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('visual','e','analyze','{}')")
        self.assertIsNone(worker.claim_job(('analyze',)))

    def test_other_running_jobs_still_hold_the_episode_from_remote_lanes(self):
        for pipeline in ('transcribe', 'subtitles', 'character_detect', 'hardcuts', 'prepare'):
            with self.subTest(pipeline):
                with db.connect() as conn:
                    conn.execute('DELETE FROM jobs')
                self.running('busy', pipeline)
                self.job('summary', 'consistency_summary', 'local')
                self.assertIsNone(worker.claim_job(('consistency_summary',), local_provider_only=True, remote_lane=True))
        # A remote text job also keeps holding the episode from work that is not visual analysis.
        with db.connect() as conn:
            conn.execute('DELETE FROM jobs')
        self.running('remote-summary', 'consistency_summary', owner=worker.REMOTE_OWNER + 'x')
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('prep','e','prepare','{}')")
        self.assertIsNone(worker.claim_job(('prepare',)))

    def test_disabled_server_routes_nothing(self):
        cfg = self.configure(routes={'review_answers': {'mode': 'remote', 'model': 'bonsai2'}}, enabled=False)
        self.assertEqual(remote_text.remote_only(cfg), ())
        self.assertEqual(remote_text.remote_pipelines(cfg), ())
        self.assertEqual(remote_text.route_for(cfg, 'review_answers'), ('local', ''))

    def test_routes_without_a_model_are_dropped(self):
        self.assertEqual(remote_text.parse_routes({'regen': {'mode': 'remote', 'model': ''}, 'sequence': {'mode': 'any', 'model': 'm'}, 'other': {'mode': 'remote', 'model': 'm'}}),
                         {'sequence': {'mode': 'any', 'model': 'm'}})

    def test_requeue_restores_the_attempt(self):
        self.job('j', 'review_answers', 'local')
        job_id, owner = worker.claim_job(('review_answers',))
        self.assertTrue(worker.requeue_job(job_id, owner))
        row = db.one('SELECT status,stage,attempt_count,lease_owner FROM jobs WHERE id=?', (job_id,))
        self.assertEqual((row['status'], row['attempt_count'], row['lease_owner']), ('queued', 0, None))
        self.assertFalse(worker.requeue_job(job_id, owner))  # no longer held

    # ------------------------------------------------------------ pool and chat priority

    def publish_models(self, state='online', models=('qwen3.6-mtp', 'bonsai2')):
        remote_text.publish({'state': state, 'detail': '', 'models': [{'id': name, 'n_ctx': 32768} for name in models]})

    def test_jobs_without_a_saved_route_share_the_work_with_the_remote(self):
        self.publish_models()
        cfg = self.configure(routes={'review_answers': {'mode': 'local', 'model': ''}})
        self.assertEqual(remote_text.route_for(cfg, 'regen'), ('any', 'qwen3.6-mtp'))
        self.assertEqual(remote_text.route_for(cfg, 'review_answers'), ('local', ''))  # a saved choice wins
        self.assertIn('sequence', remote_text.remote_pipelines(cfg))
        cfg = self.configure(fallback='bonsai2')
        self.assertEqual(remote_text.route_for(cfg, 'regen'), ('any', 'bonsai2'))
        self.assertEqual(remote_text.effective_routes(self.configure(enabled=False)), {})

    def test_prefer_and_chat_reserve_are_saved_and_validated(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        body = {'enabled': True, 'base_url': 'http://192.168.0.7:8088', 'parallel': 3, 'prefer': 'remote', 'chat_reserve': 2}
        saved = client.put('/api/remote-text-server', json=body).json()
        self.assertEqual((saved['prefer'], saved['chat_reserve'], saved['local_slots'] >= 1), ('remote', 2, True))
        self.assertEqual(remote_text.config()['prefer'], 'remote')
        self.assertEqual(client.put('/api/remote-text-server', json={**body, 'prefer': 'nowhere'}).status_code, 422)

    def test_jobs_leave_the_reserved_slots_to_chat_and_pause_while_chat_waits(self):
        remote_text.pool.__init__()
        cfg = self.configure()  # parallel 2, reserve 1
        with db.connect() as conn:
            conn.execute('UPDATE remote_text_server SET chat_reserve=1 WHERE id=1')
        cfg = remote_text.config()
        self.assertEqual(remote_text.pool.job_limit('remote', cfg), 1)
        self.assertTrue(remote_text.pool.try_job('remote', cfg))
        self.assertFalse(remote_text.pool.try_job('remote', cfg))  # the second slot belongs to chat
        remote_text.pool.release_job('remote')
        with remote_text.pool.chat_wait():
            self.assertFalse(remote_text.pool.try_job('remote', cfg))
        self.assertTrue(remote_text.pool.try_job('remote', cfg))
        remote_text.pool.release_job('remote')
        self.assertEqual(remote_text.pool.job_limit('remote', {**cfg, 'chat_reserve': 0}), 2)
        self.assertEqual(remote_text.pool.job_limit('remote', {**cfg, 'chat_reserve': 9}), 1)

    def test_pick_server_takes_the_preferred_free_server_and_skips_busy_ones(self):
        remote_text.pool.__init__()
        self.publish_models()
        self.configure(self.start_server())
        runtime = Mock(gpu_busy=False)
        runtime.switch_pending.return_value = False
        with patch.object(local_text_runtime, 'runtime', runtime), patch.object(local_text_runtime, 'text_job_concurrency', return_value=2):
            self.assertEqual(remote_text.pick_server(), ('local', ''))
            with db.connect() as conn:
                conn.execute("UPDATE remote_text_server SET prefer='remote' WHERE id=1")
            self.assertEqual(remote_text.pick_server(), ('remote', 'qwen3.6-mtp'))
            runtime.gpu_busy = True  # visual work holds the GPU
            self.assertEqual(remote_text.pick_server(), ('remote', 'qwen3.6-mtp'))
            with remote_text.pool.chat('remote'), remote_text.pool.chat('remote'):  # both remote slots are answering chats
                self.assertIsNone(remote_text.pick_server())
            self.publish_models(state='offline')
            self.assertIsNone(remote_text.pick_server())
            runtime.gpu_busy = False
            self.assertEqual(remote_text.pick_server(), ('local', ''))

    def test_auto_provider_waits_for_a_free_server_then_answers_on_it(self):
        remote_text.pool.__init__()
        self.publish_models()
        self.configure(self.start_server())
        picks = [None, None, ('remote', 'bonsai2')]
        stages = []
        with patch.object(remote_text, 'pick_server', side_effect=picks), patch.object(providers.time, 'sleep'):
            with providers.AutoServer().session('auto', False, lambda: None, stages.append) as endpoint:
                self.assertEqual((endpoint.model, endpoint.extra['_server']), ('bonsai2', 'remote'))
                self.assertEqual(remote_text.pool.busy_with_chat()['remote'], 1)
                self.assertEqual(remote_text.pool.waiting, 0)
        self.assertEqual(stages[0], 'Waiting for a free server')
        self.assertEqual(remote_text.pool.busy_with_chat()['remote'], 0)

    # ------------------------------------------------------------ requests

    def test_chat_json_uses_the_bound_remote_model(self):
        cfg = self.configure(self.start_server())
        runtime = Mock()
        usage = {}
        with patch.object(local_text_runtime, 'runtime', runtime), remote_text.bind(cfg, 'bonsai2'):
            output, model = inference_worker.chat_json('ignored', 'system', 'evidence', ['answer'], provider='local', usage_out=usage)
        self.assertEqual((output['answer'], model), ('ok', 'bonsai2'))
        self.assertEqual(FakeLlama.requests[0]['model'], 'bonsai2')
        runtime.ensure.assert_not_called()
        runtime.request.assert_not_called()
        self.assertEqual(usage['model_variant'], 'remote:bonsai2')
        self.assertEqual(usage['endpoint'], 'remote')
        self.assertEqual(usage['draft_acceptance'], 0.75)
        self.assertEqual(local_text_runtime.revision_for(usage['model_variant']), 'remote:bonsai2')

    def test_unreachable_server_interrupts_the_job_instead_of_failing_it(self):
        cfg = self.configure(f'http://127.0.0.1:{free_port()}')
        with remote_text.bind(cfg, 'bonsai2') as binding:
            with self.assertRaises(inference_worker.RemoteUnreachable):
                inference_worker.chat_json('m', 'system', 'evidence', ['answer'], provider='local')
        self.assertTrue(binding.unreachable)
        self.assertTrue(issubclass(inference_worker.RemoteUnreachable, inference_worker.SummaryCancelled))

    def test_remote_lane_requeues_a_job_whose_server_went_away(self):
        cfg = self.configure(routes={'review_answers': {'mode': 'remote', 'model': 'bonsai2'}})
        self.job('j', 'review_answers', 'local')

        def handler(job_id, owner):
            remote_text.bound().unreachable = True

        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            raise KeyboardInterrupt  # end the lane after its first idle wait

        with patch.object(remote_text, 'online', return_value=True), patch.object(inference_worker, 'process_review_answers', side_effect=handler), \
             patch.object(inference_worker.time, 'sleep', side_effect=sleep), patch.object(remote_text, 'publish') as publish:
            with self.assertRaises(KeyboardInterrupt):
                inference_worker.remote_lane(0)
        row = db.one("SELECT status,attempt_count,stage FROM jobs WHERE id='j'")
        self.assertEqual((row['status'], row['attempt_count']), ('queued', 0))
        self.assertEqual(publish.call_args.args[0]['state'], 'offline')
        self.assertEqual(cfg['parallel'], 2)
        # Lanes beyond ``parallel`` stay idle.
        with patch.object(remote_text, 'online', return_value=True), patch.object(worker, 'claim_job') as claim, \
             patch.object(inference_worker.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                inference_worker.remote_lane(2)
        claim.assert_not_called()

    def test_probe_reports_models_context_and_auth(self):
        cfg = self.configure(self.start_server())
        result = remote_text.probe(cfg)
        self.assertEqual(result['state'], 'online')
        self.assertEqual(result['models'], [{'id': 'qwen3.6-mtp', 'n_ctx': 49152}, {'id': 'bonsai2', 'n_ctx': None}])
        self.assertEqual(remote_text.probe({**cfg, 'token': 'wrong'})['state'], 'auth_failed')
        self.assertEqual(remote_text.probe({**cfg, 'base_url': f'http://127.0.0.1:{free_port()}'})['state'], 'offline')

    def test_bound_thread_sizes_inputs_by_the_remote_slot_and_never_stops_the_local_model(self):
        cfg = self.configure()
        remote_text.publish({'state': 'online', 'models': [{'id': 'qwen3.6-mtp', 'n_ctx': 49152}]})
        runtime = Mock()
        with patch.object(local_text_runtime, 'runtime', runtime):
            with remote_text.bind(cfg, 'qwen3.6-mtp'):
                self.assertEqual(sequence_evidence.slot_tokens('local'), 49152)
                local_text_runtime.cancel_local()
            runtime.stop.assert_not_called()
            with remote_text.bind(cfg, 'bonsai2'):
                self.assertEqual(sequence_evidence.slot_tokens('local'), remote_text.DEFAULT_SLOT_CONTEXT)
            local_text_runtime.cancel_local()
            runtime.stop.assert_called_once_with('Cancelled')
        self.assertEqual(local_text_runtime.known_variant('remote:x'), 'remote:x')
        self.assertEqual(local_text_runtime.known_variant('nope'), 'standard')

    # ------------------------------------------------------------ API

    def test_settings_api_never_returns_the_token(self):
        body = {'enabled': True, 'base_url': 'http://192.168.0.20:8080/', 'token': 'secret', 'parallel': 3,
                'routes': {'review_answers': {'mode': 'remote', 'model': 'bonsai2'}}, 'assistant_fallback': 'qwen3.6-mtp'}
        saved = self.client.put('/api/remote-text-server', json=body)
        self.assertEqual(saved.status_code, 200, saved.text)
        value = saved.json()
        self.assertNotIn('token', value)
        self.assertTrue(value['has_token'])
        self.assertEqual(value['base_url'], 'http://192.168.0.20:8080')
        self.assertEqual(value['routes'], {'review_answers': {'mode': 'remote', 'model': 'bonsai2'}})
        self.assertEqual(value['state']['state'], 'worker_offline')
        # An omitted token keeps the saved one, for saving and for testing.
        body.pop('token')
        self.client.put('/api/remote-text-server', json=body)
        self.assertEqual(remote_text.config()['token'], 'secret')
        with patch.object(remote_text, 'probe', return_value={'state': 'online', 'detail': '', 'models': []}) as probe:
            self.assertEqual(self.client.post('/api/remote-text-server/test', json=body).status_code, 200)
        self.assertEqual(probe.call_args.args[0]['token'], 'secret')
        for bad in ({'base_url': 'ftp://x'}, {'routes': {'review_answers': {'mode': 'any', 'model': ''}}}, {'routes': {'analyze': {'mode': 'local'}}}, {'base_url': ''}):
            self.assertEqual(self.client.put('/api/remote-text-server', json={**body, **bad}).status_code, 400, bad)
        self.assertEqual(self.client.put('/api/remote-text-server', json={**body, 'parallel': 9}).status_code, 422)

    # ------------------------------------------------------------ assistant

    def test_assistant_falls_back_to_the_remote_model_when_it_is_online(self):
        self.configure(fallback='qwen3.6-mtp')
        router = Mock(id='openrouter')
        router.available.return_value = (True, '')
        remote = providers.RemoteLlama()
        available = {'remote': remote, 'openrouter': router}
        self.assertEqual(providers.gpu_fallback(available), (router, providers.FALLBACK_MODEL))  # remote not online yet
        remote_text.publish({'state': 'online', 'detail': '', 'models': [{'id': 'qwen3.6-mtp', 'n_ctx': 49152}]})
        self.assertEqual(providers.gpu_fallback(available), (remote, 'qwen3.6-mtp'))
        info = remote.describe()
        self.assertTrue(info['available'])
        self.assertEqual(info['models'], [{'id': 'qwen3.6-mtp', 'label': 'qwen3.6-mtp', 'context': 49152}])
        endpoint = remote.endpoint('qwen3.6-mtp', True)
        self.assertEqual(endpoint.headers['Authorization'], AUTH)
        self.assertEqual(endpoint.url, 'http://127.0.0.1:9/v1/chat/completions')
        with db.connect() as conn:
            conn.execute('UPDATE remote_text_state SET heartbeat=? WHERE id=1', (time.time() - 60,))
        self.assertFalse(remote.available()[0])  # a silent worker means the state is stale


if __name__ == '__main__':
    unittest.main()
