"""Cut Assistance: tools, proposals, API, streaming parser, runner and GPU coordination."""
import http.server
import json
import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, local_text_runtime
from app.assistant import compact, providers, runner, skills, store, tools

VIDEO = {'model': 'Qwen/Qwen3-VL-8B-Instruct', 'windows': [{'source': {'source_start': 0.5, 'source_end': 2.0, 'fps': 30.0},
         'analysis': {'overview': 'A person in a green coat stands still.', 'uncertainties': 'none',
                      'events': [{'start_seconds': 0.2, 'end_seconds': 1.0, 'subject': 'a person', 'visual_action': 'a person raises a hand', 'camera': 'static', 'uncertainty': 'none',
                                  'episode_start_seconds': 0.7, 'episode_end_seconds': 1.5}]},
         'usage': {'prompt_tokens': 10}}]}
FRAME = {'composition': 'Medium shot of a person.', 'setting': 'Street', 'lighting': 'Day', 'camera': 'Static',
         'visible_characters': [{'appearance': 'Green coat', 'position': 'center', 'pose': 'standing', 'expression': 'calm', 'identity_evidence': 'clothing match'}],
         'other_visual_details': '', 'uncertainties': 'Identity unconfirmed.', 'frame_time': 0.52, 'subtitle_regions': []}


class AssistantBase(unittest.TestCase):
    setUp_base = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def setUp(self):
        self.setUp_base()
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET start_time=0.5,end_time=2.0 WHERE id='c1'")
            conn.execute("UPDATE cuts SET start_time=2.0 WHERE id='c2'")
            for result_id, stype, kind, title, content in (
                ('rf', 'initial_frame', 'first_frame', 'Initial frame description', FRAME),
                ('rv', 'video_analysis', 'cut_description', 'Video analysis', VIDEO),
                ('rc', 'consistency_summary', 'cut_description', 'Consistency summary', {'summary': 'A person raises a hand.', 'human_review': 'Confirm the person is Hero.'})):
                conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,model_id,settings_json,context_json) VALUES (?,?,?,?,?,'model','m',?,?)",
                             (result_id, 'c1', kind, title, json.dumps(content), json.dumps({'structured_type': stype, 'provider': 'local'}), json.dumps({'evidence': 1})))
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('rb','c1','cut_description','Cut summary','### Scene\nA person on a street.','model')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin) VALUES ('t1','e',0.4,1.2,'こんにちは','SPEAKER_01','ctx','model')")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c1',0.5,2.0,'[]','aniref')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d1','run','first','[0,0,1,1]','ctx','manual')")
        self.ctx = tools.ToolContext(cut_id='c1', session_id='', provider='fake', model='fake-model')

    def worker_online(self, providers_list=None):
        described = providers_list or [{'id': 'fake', 'label': 'Fake', 'available': True, 'models': [{'id': 'fake-model'}], 'free_model': False}]
        with db.connect() as conn:
            conn.execute('UPDATE assistant_worker_state SET providers_json=?,heartbeat=? WHERE id=1', (json.dumps(described), time.time()))

    def session(self):
        response = self.client.post('/api/cuts/c1/assistant/sessions', json={'provider': 'fake', 'model': 'fake-model'})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['session']['id']

    def run_tool(self, name, args):
        content, label, ok = tools.execute(self.ctx, name, json.dumps(args))
        return json.loads(content) if not tools.REGISTRY[name].raw else content, label, ok


class ToolTest(AssistantBase):
    def test_status_and_card_report_existing_and_missing_results(self):
        value, label, ok = self.run_tool('get_cut_status', {})
        self.assertTrue(ok)
        self.assertEqual(value['results']['production_brief']['origin'], 'model')
        self.assertEqual(value['results']['last_frame'], 'missing')
        self.assertEqual(value['characters']['named'], ['Hero'])
        self.assertEqual(value['transcript_segments'], 1)
        card = tools.cut_card('c1')
        self.assertIn('Cut 1 of episode "Episode"', card)
        self.assertIn('last_frame=missing', card)
        self.assertIn('Hero', card)

    def test_video_result_is_compact_and_clip_relative(self):
        value, _, ok = self.run_tool('get_result', {'type': 'video_analysis'})
        self.assertTrue(ok)
        event = value['content']['events'][0]
        self.assertEqual(event['t'], '0.20–1.00')  # window starts at the cut start
        self.assertEqual(event['action'], 'a person raises a hand')
        self.assertNotIn('uncertainty', event)  # "none" is dropped
        self.assertNotIn('usage', json.dumps(value))
        fields, _, _ = self.run_tool('get_result', {'type': 'initial_frame', 'fields': ['visible_characters']})
        self.assertEqual(list(fields['content']), ['visible_characters'])
        bad, _, ok = self.run_tool('get_result', {'type': 'initial_frame', 'fields': ['nope']})
        self.assertIn('Available', bad['error'])
        missing, label, _ = self.run_tool('get_result', {'type': 'h3_prompt'})
        self.assertTrue(missing['missing'])
        self.assertIn('missing', label)
        enum_error, _, ok = self.run_tool('get_result', {'type': 'everything'})
        self.assertFalse(ok)
        self.assertIn('must be one of', enum_error['error'])

    def test_transcript_characters_search_and_skill(self):
        transcript, _, _ = self.run_tool('get_transcript', {})
        self.assertEqual(transcript['segments'][0]['t'], '-0.10–0.70')
        self.assertEqual(transcript['segments'][0]['character'], 'Hero')
        characters, _, _ = self.run_tool('get_characters', {})
        self.assertEqual(characters['boxes_in_cut'][0]['name'], 'Hero')
        found, label, _ = self.run_tool('find_in_results', {'query': 'A PERSON'})
        self.assertEqual({item['type'] for item in found['matches']}, {'video_analysis', 'initial_frame', 'consistency_summary', 'production_brief'})
        self.assertIn('hits', label)
        body, label, ok = self.run_tool('load_skill', {'name': 'h3_animation_prompt'})
        self.assertTrue(ok)
        self.assertIn('<Picture 1>', body)
        self.assertEqual(label, 'Loaded skill: H3 animation prompt')

    def test_skill_files_parse_and_cover_the_planned_prompts(self):
        names = set(skills.names())
        self.assertTrue({'cut_status_summary', 'explain_result', 'h3_animation_prompt', 'still_image_prompt', 'correct_result'} <= names)
        examples = [example for skill in skills.all_skills() for example in skill['examples']]
        for prompt in ('Summarize the cut analysis so far', 'What was analyzed from first image?', 'Fix this H3 prompt based on results',
                       'Fix the still image generation prompt', 'Fix the character name in the keyframe analysis'):
            self.assertIn(prompt, examples)
        with self.assertRaises(ValueError):
            skills.parse('no front matter')

    def test_payload_compaction(self):
        text = compact.encode({'a': None, 'b': '', 'c': [], 'd': 1.23456, 'usage': {'x': 1}, 'e': 'x' * 2000}, limit=7000)
        data = json.loads(text)
        self.assertEqual(set(data), {'d', 'e'})
        self.assertEqual(data['d'], 1.235)
        self.assertIn('…[+800 chars', data['e'])
        self.assertNotIn(' ', compact.encode({'a': [1, 2]}))
        huge = compact.encode({'items': ['y' * 1000 for _ in range(50)]}, limit=3000)
        self.assertLessEqual(len(huge), 3000)


class ProposalTest(AssistantBase):
    def propose(self, args):
        value, _, ok = self.run_tool('propose_result_edit', args)
        return value, ok

    def test_json_replacement_keeps_structure_and_apply_creates_revision(self):
        session_id = self.session()
        self.ctx.session_id = session_id
        value, ok = self.propose({'type': 'video_analysis', 'replacements': [{'find': 'a person', 'replace': 'Hero'}], 'summary': 'Name the person'})
        self.assertTrue(ok, value)
        self.assertEqual(value['applied'], [{'find': 'a person', 'replaced': 2}])
        self.assertEqual(value['status'], 'pending_user_approval')
        proposal = db.one('SELECT * FROM assistant_proposals WHERE id=?', (value['proposal_id'],))
        data = json.loads(proposal['content'])
        self.assertEqual(data['windows'][0]['analysis']['events'][0]['subject'], 'Hero')
        self.assertEqual(data['windows'][0]['source']['source_start'], 0.5)
        # Nothing is saved until the user applies it.
        self.assertIsNone(db.one("SELECT superseded_by FROM analysis_results WHERE id='rv'")['superseded_by'])
        detail = self.client.get(f'/api/assistant/sessions/{session_id}').json()
        card = detail['proposals'][0]
        self.assertFalse(card['stale'])
        self.assertTrue(any(part['op'] == 'insert' and 'Hero' in part['text'] for part in card['diff']))
        applied = self.client.post(f"/api/assistant/proposals/{value['proposal_id']}/apply")
        self.assertEqual(applied.status_code, 200, applied.text)
        new_id = applied.json()['applied_result_id']
        row = db.one('SELECT * FROM analysis_results WHERE id=?', (new_id,))
        self.assertEqual((row['origin'], row['parent_id'], row['model_id']), ('assistant_edit', 'rv', 'fake-model'))
        settings = json.loads(row['settings_json'])
        self.assertEqual(settings['structured_type'], 'video_analysis')
        self.assertEqual(settings['assistant_edit']['proposal_id'], value['proposal_id'])
        self.assertEqual(json.loads(row['context_json']), {'evidence': 1})
        self.assertEqual(db.one("SELECT superseded_by FROM analysis_results WHERE id='rv'")['superseded_by'], new_id)
        self.assertEqual(self.client.post(f"/api/assistant/proposals/{value['proposal_id']}/apply").status_code, 409)

    def test_stale_proposal_cannot_overwrite_a_newer_revision(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose({'type': 'initial_frame', 'set': [{'path': 'visible_characters.0.identity_evidence', 'value': 'User-verified box: Hero'}], 'summary': 'Identity'})
        self.assertTrue(ok, value)
        self.client.put('/api/analysis-results/rf', json={'title': 'Initial frame description', 'content': json.dumps(FRAME)})
        response = self.client.post(f"/api/assistant/proposals/{value['proposal_id']}/apply")
        self.assertEqual(response.status_code, 409)
        self.assertIn('changed', response.json()['detail'])

    def test_errors_are_returned_to_the_model(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose({'type': 'video_analysis', 'replacements': [{'find': 'nobody', 'replace': 'Hero'}], 'summary': 'x'})
        self.assertFalse(ok)
        self.assertIn('Text not found', value['error'])
        value, ok = self.propose({'type': 'video_analysis', 'set': [{'path': 'windows.0.nope.x', 'value': 1}], 'summary': 'x'})
        self.assertIn("key 'nope' does not exist", value['error'])
        value, ok = self.propose({'type': 'last_frame', 'content': '{}', 'summary': 'x'})
        self.assertIn('no last frame result', value['error'])
        value, ok = self.propose({'type': 'production_brief', 'set': [{'path': 'a', 'value': 1}], 'summary': 'x'})
        self.assertIn('set only applies', value['error'])
        value, ok = self.propose({'type': 'production_brief', 'replacements': [{'find': 'street', 'replace': 'street'}], 'summary': 'x'})
        self.assertIn('does not change', value['error'])

    def test_new_prompt_and_review_answers(self):
        session_id = self.session()
        self.ctx.session_id = session_id
        value, ok = self.propose({'type': 'image_prompt', 'content': 'Anime style. A tight close-up of the exact opening frame.', 'summary': 'Draft'})
        self.assertTrue(ok, value)
        applied = self.client.post(f"/api/assistant/proposals/{value['proposal_id']}/apply").json()
        row = db.one('SELECT kind,title,origin,parent_id FROM analysis_results WHERE id=?', (applied['applied_result_id'],))
        self.assertEqual(dict(row), {'kind': 'generation_prompt', 'title': 'Still image prompt', 'origin': 'assistant_edit', 'parent_id': None})
        edited, ok = self.propose({'type': 'image_prompt', 'replacements': [{'find': 'exact', 'replace': 'same'}], 'summary': 'Edit'})
        self.assertTrue(ok, edited)
        answers, ok = self.propose({'type': 'review_answers', 'content': 'The hand belongs to Hero.', 'summary': 'Answer'})
        self.assertTrue(ok, answers)
        self.client.post(f"/api/assistant/proposals/{answers['proposal_id']}/apply")
        self.assertEqual(self.client.get('/api/cuts/c1/brief-instructions').json()['content'], 'The hand belongs to Hero.')
        discarded = self.client.post(f"/api/assistant/proposals/{edited['proposal_id']}/discard").json()
        self.assertEqual(discarded['status'], 'discarded')


class ApiTest(AssistantBase):
    def test_sessions_messages_cancel_retry_and_delete(self):
        session_id = self.session()
        self.assertEqual(self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'fake', 'model': 'fake-model'}).status_code, 503)
        self.worker_online()
        bad = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'fake', 'model': 'other'})
        self.assertEqual(bad.status_code, 400)
        sent = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'Summarize the cut\nplease', 'provider': 'fake', 'model': 'fake-model', 'skill': 'cut_status_summary'})
        self.assertEqual(sent.status_code, 200, sent.text)
        turn = sent.json()['turn']
        self.assertEqual((turn['status'], turn['skill']), ('queued', 'cut_status_summary'))
        sessions = self.client.get('/api/cuts/c1/assistant/sessions').json()
        self.assertEqual((sessions[0]['title'], sessions[0]['message_count'], sessions[0]['last_status']), ('Summarize the cut', 1, 'queued'))
        self.assertEqual(self.client.post(f"/api/assistant/turns/{turn['id']}/cancel").json()['status'], 'cancelled')
        retried = self.client.post(f"/api/assistant/turns/{turn['id']}/retry", json={})
        self.assertEqual(retried.json()['turns'][0]['status'], 'queued')
        follow_up = {'content': 'again', 'provider': 'fake', 'model': 'fake-model'}
        queued = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json=follow_up)
        self.assertEqual(queued.status_code, 200, queued.text)
        self.assertEqual(queued.json()['turn']['stage'], 'Queued after current reply')
        self.assertEqual(queued.json()['turn']['priority_class'], 'chat')
        self.assertEqual(self.client.post(f'/api/assistant/sessions/{session_id}/messages', json=follow_up).json(), queued.json())
        config = self.client.get('/api/assistant/config').json()
        self.assertTrue(config['worker_online'])
        self.assertIn('h3_animation_prompt', [skill['name'] for skill in config['skills']])
        self.assertNotIn('body', config['skills'][0])
        self.assertEqual(self.client.delete(f'/api/assistant/sessions/{session_id}').status_code, 200)
        self.assertIsNone(db.one('SELECT id FROM assistant_turns WHERE id=?', (turn['id'],)))

    def test_stream_sends_snapshot_and_done_for_finished_turn(self):
        session_id = self.session()
        self.worker_online()
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'fake', 'model': 'fake-model'}).json()['turn']
        with db.connect() as conn:
            store.bump(conn, turn['id'], status='done', stage='Complete')
        with self.client.stream('GET', f"/api/assistant/turns/{turn['id']}/stream") as response:
            body = ''.join(response.iter_text())
        self.assertIn('event: snapshot', body)
        self.assertIn('event: done', body)


class FakeProvider(providers.Provider):
    id, label = 'fake', 'Fake'

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage):
        yield providers.Endpoint('http://fake', model, context_limit=1000)


class RunnerTest(AssistantBase):
    def test_follow_ups_claim_in_order_and_use_only_earlier_turns(self):
        session_id = self.session()
        self.worker_online()

        def queue(content):
            return self.client.post(f'/api/assistant/sessions/{session_id}/messages',
                                    json={'content': content, 'provider': 'fake', 'model': 'fake-model'}).json()['turn']['id']

        first, second = queue('first'), queue('follow up')
        self.assertEqual(runner.claim_turn()[0], first)
        self.assertIsNone(runner.claim_turn())
        self.assertEqual([message['content'] for message in runner.history_messages(session_id, first, 1000)], ['first'])
        with db.connect() as conn:
            conn.execute("INSERT INTO assistant_messages(id,session_id,turn_id,seq,role,content) VALUES ('answer',?,?,?,'assistant','answer')",
                         (session_id, first, store.next_seq(conn, session_id)))
            store.bump(conn, first, status='done')
        self.assertEqual(runner.claim_turn()[0], second)
        self.assertEqual([message['content'] for message in runner.history_messages(session_id, second, 1000)], ['first', 'answer', 'follow up'])

    def test_active_reply_can_queue_a_retry_with_different_choices(self):
        session_id = self.session()
        self.worker_online()
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'describe', 'provider': 'fake', 'model': 'fake-model'}).json()['turn']
        self.assertEqual(runner.claim_turn()[0], turn['id'])
        path = f"/api/assistant/turns/{turn['id']}/retry"
        self.assertEqual(len(self.client.post(path, json={}).json()['turns']), 1)
        retry = self.client.post(path, json={'thinking': False}).json()['turns'][1]
        self.assertEqual(retry['status'], 'queued')
        self.assertEqual(self.client.post(path, json={'thinking': False}).json()['turns'][1]['id'], retry['id'])
        with db.connect() as conn:
            store.bump(conn, turn['id'], status='done')
        self.assertEqual(runner.claim_turn()[0], retry['id'])
        self.assertEqual([message['content'] for message in runner.history_messages(session_id, retry['id'], 1000)], ['describe'])

    def test_turn_runs_tools_then_answers_with_usage(self):
        session_id = self.session()
        self.worker_online()
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'Fix the name', 'provider': 'fake', 'model': 'fake-model', 'skill': 'correct_result'}).json()['turn']
        requests = []

        def fake_stream(endpoint, messages, tool_schemas, *, max_tokens, temperature, on_delta, checkpoint, **kwargs):
            requests.append({'messages': messages, 'tools': tool_schemas})
            checkpoint()
            if len(requests) == 1:
                on_delta('', 'I should read it.')
                return providers.StepResult(reasoning='I should read it.', tool_calls=[{'id': 'c1', 'name': 'find_in_results', 'arguments': '{"query":"a person"}'}],
                                            finish_reason='tool_calls', usage={'prompt_tokens': 500, 'completion_tokens': 20}, seconds=1.0, first_token_seconds=0.2)
            on_delta('Proposed **Hero**.', '')
            return providers.StepResult(content='Proposed **Hero**.', finish_reason='stop', usage={'prompt_tokens': 700, 'completion_tokens': 30, 'tokens_per_second': 42.0}, seconds=0.5)

        with patch.object(providers, 'registry', return_value={'fake': FakeProvider()}), patch.object(providers, 'stream_chat', side_effect=fake_stream):
            claimed = runner.claim_turn()
            self.assertEqual(claimed[0], turn['id'])
            runner.run_turn(*claimed)
        detail = self.client.get(f'/api/assistant/sessions/{session_id}').json()
        finished = detail['turns'][0]
        self.assertEqual(finished['status'], 'done', finished['error'])
        self.assertEqual(finished['usage']['prompt_tokens'], 1200)
        self.assertEqual(finished['usage']['context_tokens'], 730)
        self.assertEqual(finished['usage']['context_limit'], 1000)
        self.assertEqual((finished['usage']['steps'], finished['usage']['tool_calls']), (2, 1))
        roles = [(message['role'], message['tool_name']) for message in detail['messages']]
        self.assertEqual(roles, [('user', None), ('assistant', None), ('tool', 'load_skill'), ('assistant', None), ('tool', 'find_in_results'), ('assistant', None)])
        self.assertEqual(detail['messages'][3]['reasoning'], 'I should read it.')
        self.assertEqual(detail['messages'][-1]['content'], 'Proposed **Hero**.')
        first_request = requests[0]['messages']
        self.assertEqual(first_request[0]['role'], 'system')
        self.assertIn('## Cut card', first_request[0]['content'])
        self.assertIn('h3_animation_prompt:', first_request[0]['content'])
        self.assertEqual([m['role'] for m in first_request[1:]], ['user', 'assistant', 'tool'])
        self.assertIn('# Correcting a saved result', first_request[3]['content'])
        self.assertEqual(requests[1]['messages'][-1]['role'], 'tool')
        self.assertTrue(any(schema['function']['name'] == 'propose_result_edit' for schema in requests[0]['tools']))

    def test_failed_required_check_sends_the_model_back(self):
        session_id = self.session()
        self.worker_online()
        self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'Fix it', 'provider': 'fake', 'model': 'fake-model'})
        seen = []

        def fake_stream(endpoint, messages, tool_schemas, *, on_delta, checkpoint, **kwargs):
            seen.append(messages[-1]['content'] if messages[-1]['role'] == 'user' else messages[-1]['role'])
            if len(seen) == 1:
                return providers.StepResult(tool_calls=[{'id': 'k1', 'name': 'check_h3_prompt', 'arguments': json.dumps({'prompt': 'not a prompt'})}])
            if len(seen) == 2:
                on_delta('Here is my best guess.', '')
                return providers.StepResult(content='Here is my best guess.')
            on_delta('Fixed after the reminder.', '')
            return providers.StepResult(content='Fixed after the reminder.')

        with patch.object(providers, 'registry', return_value={'fake': FakeProvider()}), patch.object(providers, 'stream_chat', side_effect=fake_stream):
            runner.run_turn(*runner.claim_turn())
        self.assertIn('check_h3_prompt reported errors', seen[2])
        self.assertEqual(self.client.get(f'/api/assistant/sessions/{session_id}').json()['messages'][-1]['content'], 'Fixed after the reminder.')

    def test_cancel_and_provider_errors_finish_the_turn(self):
        session_id = self.session()
        self.worker_online()
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'fake', 'model': 'fake-model'}).json()['turn']

        def cancelled(endpoint, messages, tool_schemas, *, on_delta, checkpoint, **kwargs):
            on_delta('Partial', '')
            self.client.post(f"/api/assistant/turns/{turn['id']}/cancel")
            checkpoint(force=True)

        with patch.object(providers, 'registry', return_value={'fake': FakeProvider()}), patch.object(providers, 'stream_chat', side_effect=cancelled):
            runner.run_turn(*runner.claim_turn())
        detail = self.client.get(f'/api/assistant/sessions/{session_id}').json()
        self.assertEqual(detail['turns'][0]['status'], 'cancelled')
        self.assertEqual(detail['messages'][-1]['content'], 'Partial')
        self.assertFalse(detail['messages'][-1]['streaming'])

        self.client.post(f"/api/assistant/turns/{turn['id']}/retry", json={})
        with patch.object(providers, 'registry', return_value={'fake': FakeProvider()}), \
             patch.object(providers, 'stream_chat', side_effect=providers.ProviderError('HTTP 404: No endpoints support tool use')):
            runner.run_turn(*runner.claim_turn())
        failed = self.client.get(f'/api/assistant/sessions/{session_id}').json()['turns'][0]
        self.assertEqual((failed['status'], failed['error']), ('failed', 'HTTP 404: No endpoints support tool use'))

    def test_interrupted_turn_restarts_from_the_user_message(self):
        session_id = self.session()
        self.worker_online()
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'fake', 'model': 'fake-model'}).json()['turn']
        turn_id, _ = runner.claim_turn()
        with db.connect() as conn:
            conn.execute("INSERT INTO assistant_messages(id,session_id,turn_id,seq,role,content,streaming) VALUES ('half',?,?,2,'assistant','Half',1)", (session_id, turn_id))
            conn.execute('UPDATE assistant_turns SET lease_until=0 WHERE id=?', (turn_id,))
        self.assertEqual(runner.claim_turn()[0], turn['id'])
        self.assertIsNone(db.one("SELECT id FROM assistant_messages WHERE id='half'"))
        with db.connect() as conn:
            conn.execute('UPDATE assistant_turns SET lease_until=0 WHERE id=?', (turn_id,))
        self.assertIsNone(runner.claim_turn())
        self.assertEqual(db.one('SELECT status FROM assistant_turns WHERE id=?', (turn_id,))['status'], 'failed')

    def test_history_elides_old_tool_results_only_over_budget(self):
        session_id = self.session()
        with db.connect() as conn:
            for seq, (turn, role, extra) in enumerate([
                ('t-old', 'user', {'content': 'old question'}),
                ('t-old', 'assistant', {'tool_calls_json': json.dumps([{'id': 'x1', 'name': 'get_result', 'arguments': '{}'}])}),
                ('t-old', 'tool', {'content': 'z' * 4000, 'tool_call_id': 'x1', 'tool_name': 'get_result', 'tool_summary': 'Read video analysis'}),
                ('t-old', 'assistant', {'content': 'old answer'}),
                ('t-new', 'user', {'content': 'new question'}),
            ], start=1):
                if not conn.execute('SELECT 1 FROM assistant_turns WHERE id=?', (turn,)).fetchone():
                    conn.execute("INSERT INTO assistant_turns(id,session_id,status,provider,model) VALUES (?,?,'done','fake','m')", (turn, session_id))
                conn.execute('INSERT INTO assistant_messages(id,session_id,turn_id,seq,role,content,tool_calls_json,tool_call_id,tool_name,tool_summary) VALUES (?,?,?,?,?,?,?,?,?,?)',
                             (f'm{seq}', session_id, turn, seq, role, extra.get('content', ''), extra.get('tool_calls_json'), extra.get('tool_call_id'), extra.get('tool_name'), extra.get('tool_summary')))
        full = runner.history_messages(session_id, 't-new', 100000)
        self.assertEqual(full[2]['content'], 'z' * 4000)
        compacted = runner.history_messages(session_id, 't-new', 600)
        self.assertIn('elided', compacted[2]['content'])
        self.assertEqual(compacted[-1]['content'], 'new question')
        tiny = runner.history_messages(session_id, 't-new', 5)
        self.assertIn('omitted', tiny[0]['content'])
        self.assertEqual(tiny[-1]['content'], 'new question')


class FakeLocal(providers.Provider):
    """Stands in for LocalQwen: with ``busy`` set, a session that must not wait raises GpuBusy."""
    id, label = 'local', 'Local'

    def __init__(self, busy):
        self.busy, self.waits = busy, []

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage, wait_for_gpu=True):
        self.waits.append(wait_for_gpu)
        if self.busy and not wait_for_gpu:
            raise local_text_runtime.GpuBusy()
        yield providers.Endpoint('http://local', model, context_limit=1000, fixed_model=True)


class FakeOpenRouter(providers.Provider):
    id, label = 'openrouter', 'OpenRouter'

    def __init__(self, ok=True):
        self.ok, self.models = ok, []

    def available(self):
        return self.ok, '' if self.ok else 'OPENROUTER_API_KEY is not configured'

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage):
        self.models.append(model)
        yield providers.Endpoint('http://openrouter', model, context_limit=2000)


class GpuFallbackTest(AssistantBase):
    LOCAL = [{'id': 'local', 'label': 'Local', 'available': True, 'models': [{'id': 'qwen'}], 'free_model': False}]

    def ask(self, fallback, **extra):
        session_id = self.session()
        self.worker_online(self.LOCAL)
        body = {'content': 'Describe the cut', 'provider': 'local', 'model': 'qwen', 'gpu_fallback': fallback, **extra}
        response = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return session_id, response.json()['turn']

    def run_with(self, local, router, name='local'):
        endpoints = []

        def fake_stream(endpoint, messages, tool_schemas, *, on_delta, checkpoint, **kwargs):
            endpoints.append(endpoint)
            on_delta('Done.', '')
            return providers.StepResult(content='Done.', usage={'prompt_tokens': 10, 'completion_tokens': 2}, model='free/model:free')

        with patch.object(providers, 'registry', return_value={name: local, 'openrouter': router}), patch.object(providers, 'stream_chat', side_effect=fake_stream):
            runner.run_turn(*runner.claim_turn())
        return endpoints

    def finished(self, session_id):
        return self.client.get(f'/api/assistant/sessions/{session_id}').json()['turns'][0]

    def test_busy_gpu_answers_with_openrouter_free(self):
        session_id, turn = self.ask(True)
        self.assertTrue(turn['gpu_fallback'])
        local, router = FakeLocal(busy=True), FakeOpenRouter()
        endpoints = self.run_with(local, router)
        self.assertEqual([endpoint.url for endpoint in endpoints], ['http://openrouter'])
        self.assertEqual(router.models, ['openrouter/free'])
        self.assertEqual(local.waits, [False])  # It never queued behind the visual job.
        done = self.finished(session_id)
        self.assertEqual(done['status'], 'done', done['error'])
        self.assertEqual(done['usage']['fallback'], {'reason': 'gpu_busy', 'from': 'local', 'provider': 'openrouter', 'model': 'openrouter/free'})
        self.assertEqual(done['usage']['model'], 'free/model:free')
        # The chosen model is kept, so Retry and the next message start from the local model again.
        self.assertEqual((done['provider'], done['model']), ('local', 'qwen'))

    def test_free_gpu_uses_the_local_model(self):
        session_id, _ = self.ask(True)
        local, router = FakeLocal(busy=False), FakeOpenRouter()
        endpoints = self.run_with(local, router)
        self.assertEqual([endpoint.url for endpoint in endpoints], ['http://local'])
        self.assertEqual(router.models, [])
        self.assertNotIn('fallback', self.finished(session_id)['usage'])

    def test_without_the_option_a_busy_gpu_still_waits(self):
        session_id, _ = self.ask(False)
        local, router = FakeLocal(busy=True), FakeOpenRouter()
        endpoints = self.run_with(local, router)
        self.assertEqual(local.waits, [True])
        self.assertEqual([endpoint.url for endpoint in endpoints], ['http://local'])
        self.assertEqual(router.models, [])
        self.assertNotIn('fallback', self.finished(session_id)['usage'])

    def test_missing_openrouter_key_keeps_waiting_for_the_gpu(self):
        session_id, _ = self.ask(True)
        local, router = FakeLocal(busy=True), FakeOpenRouter(ok=False)
        endpoints = self.run_with(local, router)
        self.assertEqual(local.waits, [True])
        self.assertEqual([endpoint.url for endpoint in endpoints], ['http://local'])
        self.assertEqual(self.finished(session_id)['status'], 'done')

    def test_only_local_turns_fall_back(self):
        session_id = self.session()
        self.worker_online()
        self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'fake', 'model': 'fake-model', 'gpu_fallback': True})
        router = FakeOpenRouter()
        endpoints = self.run_with(FakeProvider(), router, name='fake')
        self.assertEqual([endpoint.url for endpoint in endpoints], ['http://fake'])
        self.assertEqual(router.models, [])

    def test_retry_keeps_or_overrides_the_option(self):
        session_id, turn = self.ask(True)
        self.client.post(f"/api/assistant/turns/{turn['id']}/cancel")
        kept = self.client.post(f"/api/assistant/turns/{turn['id']}/retry", json={}).json()['turns'][0]
        self.assertTrue(kept['gpu_fallback'])
        self.client.post(f"/api/assistant/turns/{turn['id']}/cancel")
        off = self.client.post(f"/api/assistant/turns/{turn['id']}/retry", json={'gpu_fallback': False}).json()['turns'][0]
        self.assertFalse(off['gpu_fallback'])

    def test_option_is_off_unless_requested(self):
        session_id = self.session()
        self.worker_online(self.LOCAL)
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'hi', 'provider': 'local', 'model': 'qwen'}).json()['turn']
        self.assertFalse(turn['gpu_fallback'])

    def test_restarted_turn_forgets_the_fallback_of_its_earlier_attempt(self):
        session_id, turn = self.ask(True)
        with db.connect() as conn:
            conn.execute("UPDATE assistant_turns SET status='running',lease_until=1,attempt_count=1,usage_json=? WHERE id=?", (json.dumps({'fallback': {'reason': 'gpu_busy'}}), turn['id']))
        runner.claim_turn()
        self.assertEqual(self.finished(session_id)['usage'], {})

    def test_worker_publishes_busy_state_and_fallback_model(self):
        self.assertEqual(providers.OpenRouter().describe()['fallback_model'], 'openrouter/free')
        with patch.object(local_text_runtime.LocalTextRuntime, 'publish'):
            runtime = local_text_runtime.LocalTextRuntime()
        runtime.publish = lambda *args: None
        previous, local_text_runtime.runtime = local_text_runtime.runtime, runtime
        try:
            self.assertFalse(providers.LocalQwen().describe()['busy'])
            runtime.begin_gpu_work()
            self.assertTrue(providers.LocalQwen().describe()['busy'])
        finally:
            local_text_runtime.runtime = previous


class StreamParserTest(unittest.TestCase):
    def test_think_splitter_handles_tags_across_chunks(self):
        splitter = providers.ThinkSplitter()
        pieces = [splitter.feed(chunk) for chunk in ('<thi', 'nk>reason', 'ing</th', 'ink>\nAnswer', ' done')]
        content = ''.join(piece[0] for piece in pieces) + splitter.flush()[0]
        reasoning = ''.join(piece[1] for piece in pieces)
        self.assertEqual((content, reasoning), ('Answer done', 'reasoning'))
        plain = providers.ThinkSplitter()
        self.assertEqual(plain.feed('Hello'), ('Hello', ''))

    def test_stream_chat_parses_reasoning_tool_calls_and_usage(self):
        chunks = [
            {'choices': [{'delta': {'reasoning_content': 'Need '}}]},
            {'choices': [{'delta': {'reasoning_content': 'data.'}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'get_result', 'arguments': '{"type":'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"video_analysis"}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 12, 'completion_tokens_details': {'reasoning_tokens': 5}}, 'timings': {'predicted_per_second': 55.5}, 'model': 'qwen'},
        ]
        received = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                received['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                    self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            deltas = []
            endpoint = providers.Endpoint(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', 'qwen', extra={'cache_prompt': True, '_load_seconds': 3})
            result = providers.stream_chat(endpoint, [{'role': 'user', 'content': 'hi'}], tools.schemas(), max_tokens=100, temperature=0.5,
                                           on_delta=lambda c, r: deltas.append((c, r)), checkpoint=lambda: None)
        finally:
            server.shutdown()
        self.assertEqual(result.reasoning, 'Need data.')
        self.assertEqual(result.tool_calls, [{'id': 'call_1', 'name': 'get_result', 'arguments': '{"type":"video_analysis"}'}])
        self.assertEqual(result.usage['reasoning_tokens'], 5)
        self.assertEqual(result.usage['total_tokens'], 112)
        self.assertEqual(result.usage['tokens_per_second'], 55.5)
        self.assertEqual(result.model, 'qwen')
        self.assertTrue(received['body']['stream'])
        self.assertTrue(received['body']['cache_prompt'])
        self.assertNotIn('_load_seconds', received['body'])
        self.assertEqual(deltas, [('', 'Need '), ('', 'data.')])

    def test_http_errors_become_provider_errors(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({'error': {'message': 'No endpoints found that support tool use'}}).encode())

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with self.assertRaisesRegex(providers.ProviderError, 'HTTP 404: No endpoints found'):
                providers.stream_chat(providers.Endpoint(f'http://127.0.0.1:{server.server_port}/x', 'm'), [], None, max_tokens=1, temperature=0,
                                      on_delta=lambda *a: None, checkpoint=lambda: None)
        finally:
            server.shutdown()


class RuntimeCoordinationTest(unittest.TestCase):
    def setUp(self):
        self.audio_requested = self.enterContext(patch('app.gpu_device.audio_requested', return_value=False))

    def test_chat_can_route_away_from_an_audio_job(self):
        runtime = self.runtime()
        self.audio_requested.return_value = True
        with self.assertRaises(local_text_runtime.GpuBusy), runtime.chat_session(wait=False):
            self.fail('Local chat entered while the GPU was reserved for audio')

    def runtime(self):
        with patch.object(local_text_runtime.LocalTextRuntime, 'publish'):
            runtime = local_text_runtime.LocalTextRuntime()
        runtime.publish = lambda *args: None
        return runtime

    def test_visual_work_waits_for_chat_and_chat_waits_for_visual_work(self):
        runtime = self.runtime()
        order = []
        with runtime.chat_session():
            runtime.begin_gpu_work()
            stopper = threading.Thread(target=lambda: (runtime.stop('Released'), order.append('stopped')))
            stopper.start()
            time.sleep(0.3)
            order.append('chat done')
        stopper.join(timeout=5)
        self.assertEqual(order, ['chat done', 'stopped'])
        waited, holder = [], {}

        def enter():
            holder['session'] = runtime.chat_session(on_wait=waited.append)
            holder['session'].__enter__()

        chat = threading.Thread(target=enter)
        chat.start()
        time.sleep(0.3)
        self.assertTrue(chat.is_alive())
        runtime.end_gpu_work()
        chat.join(timeout=5)
        self.assertFalse(chat.is_alive())
        self.assertEqual(runtime.chat_users, 1)
        self.assertIn('Waiting for the GPU', waited[0])
        holder['session'].__exit__(None, None, None)
        self.assertEqual(runtime.chat_users, 0)

    def test_chat_session_can_refuse_to_wait(self):
        runtime = self.runtime()
        with runtime.chat_session(wait=False):
            self.assertEqual(runtime.chat_users, 1)
        runtime.begin_gpu_work()
        with self.assertRaises(local_text_runtime.GpuBusy):
            with runtime.chat_session(wait=False):
                self.fail('chat must not start while visual work holds the GPU')
        self.assertEqual(runtime.chat_users, 0)
        runtime.end_gpu_work()
        with runtime.chat_session(wait=False):
            self.assertEqual(runtime.chat_users, 1)


if __name__ == '__main__':
    unittest.main()
