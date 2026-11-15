"""OpenAI-compatible chat gateway: auth, request validation, queueing, worker streaming and cancellation."""
import json
import os
import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import chat_gateway, chat_gateway_worker, db
from app.assistant import providers

KEY = 'secret-key'
HEADERS = {'Authorization': f'Bearer {KEY}'}
BODY = {'model': 'anything', 'messages': [{'role': 'user', 'content': 'Hello'}]}


class FakeAuto:
    def available(self):
        return True, ''

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage):
        yield providers.Endpoint('http://127.0.0.1:1/v1/chat/completions', 'Qwen/test', context_limit=32768, fixed_model=True,
                                 extra={'chat_template_kwargs': {'enable_thinking': thinking}, '_server': 'local'})


class GatewayBase(unittest.TestCase):
    setUp_base = analysis_tests.AnalysisTest.setUp
    tearDown_base = analysis_tests.AnalysisTest.tearDown

    def setUp(self):
        self.setUp_base()
        self.env = patch.dict(os.environ, {'CHAT_API_KEY': KEY})
        self.env.start()
        self.seen = {}

    def tearDown(self):
        self.env.stop()
        self.tearDown_base()

    @contextmanager
    def worker(self, stream):
        stop = threading.Event()
        with patch.object(providers, 'AutoServer', FakeAuto), patch.object(providers, 'stream_chat', side_effect=stream):
            thread = threading.Thread(target=chat_gateway_worker.loop, args=(stop,), daemon=True)
            thread.start()
            try:
                yield
            finally:
                stop.set()
                thread.join(5)

    def answer(self, text, reasoning='', calls=None, finish='stop'):
        def stream(endpoint, messages, tools, *, max_tokens, temperature, on_delta, checkpoint, **kwargs):
            self.seen.update(endpoint=endpoint, messages=messages, tools=tools, max_tokens=max_tokens, temperature=temperature)
            for index in range(0, len(text), 4):
                on_delta(text[index:index + 4], reasoning if index == 0 else '')
                checkpoint()
            return providers.StepResult(content=text, reasoning=reasoning, tool_calls=calls or [], finish_reason=finish,
                                        usage={'prompt_tokens': 7, 'completion_tokens': 3, 'reasoning_tokens': 1}, model='Qwen/test', seconds=0.1)
        return stream

    def online(self):
        with db.connect() as conn:
            conn.execute('UPDATE chat_gateway_state SET heartbeat=?', (time.time(),))


class AuthAndValidation(GatewayBase):
    def test_disabled_without_a_key_and_rejects_wrong_keys(self):
        with patch.dict(os.environ, {'CHAT_API_KEY': ''}):
            response = self.client.post('/v1/chat/completions', json=BODY, headers=HEADERS)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()['error']['type'], 'server_error')
            self.assertEqual(self.client.get('/v1/models', headers=HEADERS).status_code, 503)
        self.assertEqual(self.client.post('/v1/chat/completions', json=BODY).status_code, 401)
        self.assertEqual(self.client.post('/v1/chat/completions', json=BODY, headers={'Authorization': 'Bearer nope'}).status_code, 401)
        self.assertEqual(self.client.get('/v1/models', headers={'Authorization': 'Bearer nope'}).status_code, 401)

    def test_models_lists_one_chat_model(self):
        data = self.client.get('/v1/models', headers=HEADERS).json()
        self.assertEqual([item['id'] for item in data['data']], [chat_gateway.PUBLIC_MODEL])

    def test_bad_requests_get_openai_errors_and_queue_nothing(self):
        self.online()
        for body in ({'messages': []}, {'messages': [{'role': 'wizard', 'content': 'x'}]}, {'messages': [{'role': 'user', 'content': 'x'}], 'temperature': 9},
                     {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}]}, {'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 0}):
            response = self.client.post('/v1/chat/completions', json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 400, body)
            self.assertIn('message', response.json()['error'])
        self.assertEqual(self.client.post('/v1/chat/completions', content=b'not json', headers=HEADERS).status_code, 400)
        self.assertIsNone(db.one('SELECT id FROM chat_gateway_requests'))

    def test_offline_worker_is_reported_instead_of_hanging(self):
        response = self.client.post('/v1/chat/completions', json=BODY, headers=HEADERS)
        self.assertEqual(response.status_code, 503)
        self.assertIn('inference worker', response.json()['error']['message'])
        self.assertIsNone(db.one('SELECT id FROM chat_gateway_requests'))

    def test_thinking_defaults_on_and_can_be_switched_off(self):
        self.assertTrue(chat_gateway.thinking_choice({}))
        self.assertFalse(chat_gateway.thinking_choice({'chat_template_kwargs': {'enable_thinking': False}}))
        self.assertFalse(chat_gateway.thinking_choice({'reasoning_effort': 'none'}))
        self.assertTrue(chat_gateway.thinking_choice({'reasoning_effort': 'high'}))


class Completion(GatewayBase):
    def test_non_streaming_reply_has_openai_shape(self):
        self.online()
        with self.worker(self.answer('Hello there, friend.', reasoning='thinking')):
            response = self.client.post('/v1/chat/completions', json={**BODY, 'max_tokens': 50, 'temperature': 0.2, 'top_p': 0.5, 'reasoning_effort': 'none'}, headers=HEADERS)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['object'], 'chat.completion')
        self.assertEqual(data['model'], 'anything')
        message = data['choices'][0]['message']
        self.assertEqual((message['role'], message['content'], message['reasoning_content']), ('assistant', 'Hello there, friend.', 'thinking'))
        self.assertEqual(data['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(data['usage'], {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10, 'completion_tokens_details': {'reasoning_tokens': 1}})
        self.assertEqual(self.seen['max_tokens'], 50)
        self.assertEqual(self.seen['temperature'], 0.2)
        self.assertEqual(self.seen['endpoint'].extra['top_p'], 0.5)
        self.assertFalse(self.seen['endpoint'].extra['chat_template_kwargs']['enable_thinking'])

    def test_streaming_sends_chunks_usage_and_done(self):
        self.online()
        with self.worker(self.answer('Streamed words here.')):
            with self.client.stream('POST', '/v1/chat/completions', json={**BODY, 'stream': True, 'stream_options': {'include_usage': True}}, headers=HEADERS) as response:
                lines = [line for line in response.iter_lines() if line.startswith('data:')]
        self.assertEqual(lines[-1], 'data: [DONE]')
        chunks = [json.loads(line[5:]) for line in lines[:-1]]
        self.assertEqual(chunks[0]['choices'][0]['delta'].get('role'), 'assistant')
        text = ''.join(part['choices'][0]['delta'].get('content', '') for part in chunks if part['choices'])
        self.assertEqual(text, 'Streamed words here.')
        self.assertEqual([part['choices'][0]['finish_reason'] for part in chunks if part['choices'] and part['choices'][0]['finish_reason']], ['stop'])
        self.assertEqual(chunks[-1]['choices'], [])
        self.assertEqual(chunks[-1]['usage']['total_tokens'], 10)

    def test_tool_calls_and_tools_are_passed_through(self):
        self.online()
        calls = [{'id': 'call_1', 'name': 'lookup', 'arguments': '{"q": 1}'}]
        tool = {'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}
        with self.worker(self.answer('', calls=calls, finish='tool_calls')):
            data = self.client.post('/v1/chat/completions', json={**BODY, 'tools': [tool], 'tool_choice': 'required'}, headers=HEADERS).json()
        message = data['choices'][0]['message']
        self.assertIsNone(message['content'])
        self.assertEqual(message['tool_calls'], [{'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"q": 1}'}}])
        self.assertEqual(data['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(self.seen['tools'], [tool])
        self.assertEqual(self.seen['endpoint'].extra['tool_choice'], 'required')

    def test_developer_role_becomes_system(self):
        self.online()
        body = {'messages': [{'role': 'developer', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}]}
        with self.worker(self.answer('ok')):
            self.client.post('/v1/chat/completions', json=body, headers=HEADERS)
        self.assertEqual(self.seen['messages'][0]['role'], 'system')

    def test_server_errors_keep_client_status_for_bad_requests(self):
        self.online()

        def failing(*args, **kwargs):
            raise providers.ProviderError('HTTP 400: request (70000 tokens) exceeds the available context size')
        with self.worker(failing):
            response = self.client.post('/v1/chat/completions', json=BODY, headers=HEADERS)
        self.assertEqual(response.status_code, 400)
        self.assertIn('exceeds', response.json()['error']['message'])

    def test_unreachable_server_is_a_502(self):
        self.online()

        def failing(*args, **kwargs):
            raise providers.ProviderError('Connection failed: refused')
        with self.worker(failing):
            self.assertEqual(self.client.post('/v1/chat/completions', json=BODY, headers=HEADERS).status_code, 502)

    def test_streaming_failure_ends_with_an_error_event(self):
        self.online()

        def failing(*args, **kwargs):
            raise providers.ProviderError('Local model could not start: out of memory')
        with self.worker(failing):
            with self.client.stream('POST', '/v1/chat/completions', json={**BODY, 'stream': True}, headers=HEADERS) as response:
                lines = [line for line in response.iter_lines() if line.startswith('data:')]
        self.assertIn('out of memory', json.loads(lines[-2][5:])['error']['message'])
        self.assertEqual(lines[-1], 'data: [DONE]')


class WorkerState(GatewayBase):
    def queue(self):
        return chat_gateway.queue_request(chat_gateway.clean_request(BODY), 'm', True)

    def test_interrupted_running_request_fails_and_is_not_restarted(self):
        request_id = self.queue()
        claimed = chat_gateway_worker.claim()
        self.assertEqual(claimed[0], request_id)
        with db.connect() as conn:
            conn.execute('UPDATE chat_gateway_requests SET lease_until=1 WHERE id=?', (request_id,))
        self.assertIsNone(chat_gateway_worker.claim())
        row = chat_gateway.snapshot(request_id)
        self.assertEqual((row['status'], row['error_status']), ('failed', 503))

    def test_cancelling_a_queued_request_is_final_and_never_claimed(self):
        request_id = self.queue()
        chat_gateway.request_cancel(request_id)
        self.assertEqual(chat_gateway.snapshot(request_id)['status'], 'cancelled')
        self.assertIsNone(chat_gateway_worker.claim())

    def test_cancel_stops_a_running_request(self):
        request_id = self.queue()
        claimed = chat_gateway_worker.claim()

        def stream(endpoint, messages, tools, *, on_delta, checkpoint, **kwargs):
            on_delta('partial', '')
            chat_gateway.request_cancel(request_id)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:  # the worker checks for a cancel at most every 0.4 s
                checkpoint()
                time.sleep(0.05)
            raise AssertionError('checkpoint must raise Cancelled')
        with patch.object(providers, 'AutoServer', FakeAuto), patch.object(providers, 'stream_chat', side_effect=stream):
            chat_gateway_worker.run_request(*claimed)
        row = chat_gateway.snapshot(request_id)
        self.assertEqual(row['status'], 'cancelled')
        self.assertEqual(row['content'], 'partial')

    def test_old_finished_rows_are_removed_when_new_ones_arrive(self):
        old = self.queue()
        with db.connect() as conn:
            conn.execute("UPDATE chat_gateway_requests SET status='done',finished_at=? WHERE id=?", (time.time() - 2 * chat_gateway.KEEP_FINISHED_SECONDS, old))
        self.queue()
        self.assertIsNone(chat_gateway.snapshot(old))


if __name__ == '__main__':
    unittest.main()
