"""Cut Assistance looks at the video: frame names, the view_frames tool, pictures in the history and which models can see."""
import json
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from app import db, frame_cache, local_text_runtime, media, remote_text
from app.assistant import images, providers, runner, skills, store, tools
import tests.test_assistant as base

JPEG = b'\xff\xd8 fake jpeg \xff\xd9'


class VisionProvider(providers.Provider):
    id, label = 'fake', 'Fake'
    vision = True

    @contextmanager
    def session(self, model, thinking, cancel_check, on_stage):
        yield providers.Endpoint('http://fake', model, context_limit=100000, vision=self.vision)


class Blind(VisionProvider):
    vision = False


class VisionBase(base.AssistantBase):
    def setUp(self):
        super().setUp()
        (media.episode_dir('p', 'e') / 'original.mp4').touch()
        images._cache.clear()
        self.renders = []

        def render(spec):
            self.renders.append(dict(spec))
            return JPEG + str(spec['time']).encode()

        patcher = patch.object(frame_cache, 'render', side_effect=render)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(images._cache.clear)
        self.ctx.vision = True

    def selection(self, frames):
        """The keyframe selection of cut c1: ``frames`` is a list of (source frame index, seconds from the cut start)."""
        selected = [{'source_frame_index': index, 'timestamp_seconds': 0.5 + at, 'relative_timestamp_seconds': at} for index, at in frames]
        return patch('app.keyframes.current_selection', return_value=({}, None, 'key', {'selected_frames': selected}))


class FrameNameTest(unittest.TestCase):
    def test_names_are_first_last_keyframes_or_seconds(self):
        self.assertEqual(images.parse('first'), ('first', None))
        self.assertEqual(images.parse(' LAST '), ('last', None))
        self.assertEqual(images.parse('F3'), ('keyframe', 3))
        self.assertEqual(images.parse('f12'), ('keyframe', 12))
        self.assertEqual(images.parse('1.25'), ('time', 1.25))
        self.assertEqual(images.parse('0 s'), ('time', 0.0))
        self.assertEqual(images.parse('2s'), ('time', 2.0))
        for bad in ('', 'middle', 'F0', 'F', '-1', '1:30', 'F1.5'):
            with self.assertRaises(images.FrameError, msg=bad):
                images.parse(bad)

    def test_a_zoom_box_is_fractions_of_the_frame(self):
        self.assertIsNone(images.zoom_box(None))
        self.assertIsNone(images.zoom_box([]))
        self.assertEqual(images.zoom_box([0.5, 0, 1, 0.5]), [0.5, 0.0, 1.0, 0.5])
        for bad in ([0.5, 0, 1], [0, 0, 1, 1.2], [0.6, 0, 0.5, 1], [0, 0, 0.03, 1], ['a', 0, 1, 1], 'all', [True, 0, 1, 1]):
            with self.assertRaises(images.FrameError, msg=str(bad)):
                images.zoom_box(bad)


class LocateTest(VisionBase):
    def test_first_and_last_use_the_cut_s_own_boundary_frames(self):
        located = images.locate('c1', 'first')
        self.assertEqual((located['spec']['time'], located['spec']['last'], located['spec']['width']), (0.5, False, images.LONG_EDGE))
        self.assertEqual(located['label'], 'Cut 1 · first frame')
        last = images.locate('c1', 'last')
        self.assertTrue(last['spec']['last'])  # no saved last frame: the final frame is searched for
        self.assertEqual((last['clip_time'], last['frame']), (1.5, 'last'))
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET first_frame_time=0.58,last_frame_time=1.9 WHERE id='c1'")
        self.assertEqual(images.locate('c1', 'first')['spec']['time'], 0.58)  # the frame the first-frame analysis was made from
        saved = images.locate('c1', 'last')
        self.assertEqual((saved['spec']['time'], saved['spec']['last']), (1.9, False))

    def test_keyframes_are_numbered_in_frame_order_like_the_keyframe_analysis(self):
        with self.selection([(40, 1.0), (10, 0.2), (25, 0.6)]):
            second = images.locate('c1', 'F2')
            self.assertEqual((second['spec']['ordinal'], second['spec']['time'], second['clip_time'], second['label']), (25, 1.1, 0.6, 'Cut 1 · keyframe F2 (0.6 s)'))
            self.assertEqual(images.locate('c1', 'f1')['spec']['ordinal'], 10)
            with self.assertRaisesRegex(images.FrameError, 'F1 to F3; F4 does not exist'):
                images.locate('c1', 'F4')
        with patch('app.keyframes.current_selection', return_value=({}, None, None, None)):
            with self.assertRaisesRegex(images.FrameError, 'no current keyframes'):
                images.locate('c1', 'F1')

    def test_seconds_are_clip_relative_and_stay_inside_the_cut(self):
        located = images.locate('c1', '1.0')
        self.assertEqual((located['spec']['time'], located['clip_time'], located['frame']), (1.5, 1.0, '1s'))
        end = images.locate('c1', '1.5')
        self.assertLess(end['spec']['time'], 2.0)  # the cut's last frame, never the next cut's first
        with self.assertRaisesRegex(images.FrameError, 'lasts 1.50 s'):
            images.locate('c1', '2')

    def test_a_zoom_is_rendered_at_the_full_size_and_named(self):
        located = images.locate('c1', 'first', [0.5, 0, 1, 0.5])
        self.assertEqual((located['spec']['width'], located['crop']), (4096, [0.5, 0.0, 1.0, 0.5]))
        self.assertIn('zoom 0.5 0 1 0.5', located['label'])

    def test_missing_cut_or_video_is_explained(self):
        with self.assertRaisesRegex(images.FrameError, 'no longer exists'):
            images.locate('gone', 'first')
        (media.episode_dir('p', 'e') / 'original.mp4').unlink()
        with self.assertRaisesRegex(images.FrameError, 'original video'):
            images.locate('c1', 'first')


class PictureTest(VisionBase):
    def test_a_frame_is_cut_once_and_then_cached(self):
        first = images.picture('c1', 'first')
        self.assertEqual(images.picture('c1', 'first'), first)
        self.assertEqual(len(self.renders), 1)
        images.picture('c1', '1')
        self.assertEqual(len(self.renders), 2)
        self.assertTrue(images.data_url({'cut_id': 'c1', 'frame': 'first', 'crop': None}).startswith('data:image/jpeg;base64,'))

    def test_a_zoom_crops_the_full_size_frame(self):
        with patch.object(images, '_zoomed', return_value=b'zoomed') as zoomed:
            self.assertEqual(images.picture('c1', 'first', [0.5, 0, 1, 0.5]), b'zoomed')
        zoomed.assert_called_once_with(JPEG + b'0.5', [0.5, 0.0, 1.0, 0.5])

    def test_a_failing_extraction_is_reported_as_a_frame_error(self):
        with patch.object(frame_cache, 'render', side_effect=RuntimeError('ffmpeg exploded')):
            with self.assertRaisesRegex(images.FrameError, 'could not be cut from the video: ffmpeg exploded'):
                images.picture('c1', 'first')


class ViewFramesToolTest(VisionBase):
    def test_the_tool_is_offered_and_run_only_to_a_model_that_can_see(self):
        self.assertNotIn('view_frames', [item['function']['name'] for item in tools.schemas('cut')])
        self.assertIn('view_frames', [item['function']['name'] for item in tools.schemas('cut', vision=True)])
        self.assertIn('view_frames', [item['function']['name'] for item in tools.schemas('sequence', vision=True)])
        self.ctx.vision = False
        content, _, ok = tools.execute(self.ctx, 'view_frames', json.dumps({'frames': ['first']}))
        self.assertFalse(ok)
        self.assertIn('Unknown tool view_frames', content)
        self.assertEqual(self.ctx.images, [])

    def test_a_call_describes_the_pictures_and_queues_them_for_the_next_message(self):
        value, label, ok = self.run_tool('view_frames', {'frames': ['first', '1.0', 'first']})
        self.assertTrue(ok)
        self.assertEqual([item['shows'] for item in value['pictures']], ['Cut 1 · first frame', 'Cut 1 · frame at 1 s'])  # the repeated name is shown once
        self.assertEqual(label, 'Looked at Cut 1 · first frame, Cut 1 · frame at 1 s')
        self.assertIn('next message', value['note'])
        self.assertEqual([(ref['cut_id'], ref['frame']) for ref in self.ctx.images], [('c1', 'first'), ('c1', '1s')])
        self.assertNotIn('data:', json.dumps(value))  # the result text never carries pixels

    def test_one_bad_frame_is_reported_and_the_others_are_still_shown(self):
        value, _, ok = self.run_tool('view_frames', {'frames': ['first', 'F2', 'sideways']})
        self.assertTrue(ok)
        self.assertEqual(len(value['pictures']), 1)
        self.assertEqual({item['frame'] for item in value['errors']}, {'F2', 'sideways'})
        self.assertEqual(len(self.ctx.images), 1)

    def test_nothing_to_show_is_a_failed_call(self):
        value, _, ok = self.run_tool('view_frames', {'frames': ['F9']})
        self.assertFalse(ok)
        self.assertIn('no current keyframes', value['error'])
        self.assertEqual(self.ctx.images, [])
        self.assertFalse(self.run_tool('view_frames', {'frames': []})[2])
        self.assertIn('At most 4', self.run_tool('view_frames', {'frames': ['first', 'last', '0.1', '0.2', '0.3']})[0]['error'])

    def test_a_zoom_applies_to_every_frame_and_a_bad_zoom_is_explained(self):
        with patch.object(images, '_zoomed', return_value=b'zoomed'):
            value, _, ok = self.run_tool('view_frames', {'frames': ['first', 'last'], 'crop': [0, 0, 0.5, 0.5]})
        self.assertTrue(ok)
        self.assertEqual([ref['crop'] for ref in self.ctx.images], [[0.0, 0.0, 0.5, 0.5]] * 2)
        self.assertIn('crop needs', self.run_tool('view_frames', {'frames': ['first'], 'crop': [0, 0, 5, 5]})[0]['error'])

    def test_another_cut_is_chosen_by_its_number(self):
        value, _, ok = self.run_tool('view_frames', {'frames': ['last'], 'cut': 2})
        self.assertTrue(ok, value)
        self.assertEqual((self.ctx.images[0]['cut_id'], value['pictures'][0]['shows'].split(' · ')[0]), ('c2', 'Cut 2'))
        self.assertIn('no active cut 7', self.run_tool('view_frames', {'frames': ['first'], 'cut': 7})[0]['error'])

    def test_a_sequence_chat_must_say_which_cut(self):
        self.ctx.sequence_id = 'seq'
        value, _, ok = self.run_tool('view_frames', {'frames': ['first']})
        self.assertFalse(ok)
        self.assertIn('Say which cut', value['error'])


class FakeRows:
    """A finished turn in the database: a question, a view_frames call that showed ``pictures`` frames, and the answer."""

    def __init__(self, test, session_id):
        self.test, self.session_id, self.seq = test, session_id, 0

    def add(self, turn_id, role, **values):
        self.seq += 1
        with db.connect() as conn:
            conn.execute('''INSERT INTO assistant_messages(id,session_id,turn_id,seq,role,content,tool_calls_json,tool_call_id,tool_name,tool_summary,usage_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                         (f'm{self.seq}', self.session_id, turn_id, self.seq, role, values.get('content', ''), values.get('calls'), values.get('call_id'), values.get('tool'),
                          values.get('summary'), json.dumps(values.get('usage') or {})))

    def turn(self, number, frames, answer='It looks right.'):
        turn_id = f'turn{number}'
        with db.connect() as conn:
            conn.execute("INSERT INTO assistant_turns(id,session_id,status,provider,model) VALUES (?,?,'done','fake','fake-model')", (turn_id, self.session_id))
        refs = [{'cut_id': 'c1', 'frame': name, 'crop': None, 'label': f'Cut 1 · {name}', 't': 0.0} for name in frames]
        self.add(turn_id, 'user', content=f'Look {number}')
        self.add(turn_id, 'assistant', calls=json.dumps([{'id': f'call{number}', 'name': 'view_frames', 'arguments': '{}'}]))
        self.add(turn_id, 'tool', content='{"pictures":[]}', call_id=f'call{number}', tool='view_frames', summary='Looked', usage={'ok': True, 'images': refs})
        self.add(turn_id, 'assistant', content=answer)
        return turn_id


class HistoryTest(VisionBase):
    def history(self, vision, turns=(('first', 'F1'),), budget=100000):
        session_id = self.session()
        rows = FakeRows(self, session_id)
        for number, frames in enumerate(turns, 1):
            rows.turn(number, frames)
        return runner.history_messages(session_id, 'not-a-turn', budget, vision)

    def test_pictures_come_back_in_one_user_message_after_the_tool_results(self):
        messages = self.history(True, [('first', 'last')])
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'tool', 'user', 'assistant'])
        parts = messages[3]['content']
        self.assertEqual(parts[0]['type'], 'text')
        self.assertIn('1. Cut 1 · first', parts[0]['text'])
        self.assertIn('2. Cut 1 · last', parts[0]['text'])
        self.assertEqual([part['type'] for part in parts[1:]], ['image_url', 'image_url'])
        self.assertNotIn(runner.PICTURES_DROPPED, messages[2]['content'])
        # Nothing is cut from the video until the request is made.
        self.assertEqual(self.renders, [])
        runner.attach_pictures(messages)
        self.assertTrue(all(part['image_url']['url'].startswith('data:image/jpeg;base64,') for part in messages[3]['content'][1:]))
        self.assertEqual(len(self.renders), 2)

    def test_a_model_that_cannot_see_gets_the_text_only(self):
        messages = self.history(False, [('first', 'F1')])
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'tool', 'assistant'])
        self.assertTrue(all(isinstance(m['content'], str) for m in messages))
        self.assertTrue(messages[2]['content'].endswith(runner.PICTURES_UNSEEN))
        self.assertEqual(self.renders, [])

    def test_only_the_latest_pictures_stay_attached(self):
        frames = [f'F{n}' for n in range(1, 5)]
        messages = self.history(True, [frames, frames])  # eight pictures over two calls; the budget keeps six
        carried = [len([part for part in message['content'] if part['type'] == 'image_url']) for message in messages if isinstance(message['content'], list)]
        self.assertEqual((images.MAX_ATTACHED, carried), (6, [2, 4]))
        tool_results = [message['content'] for message in messages if message['role'] == 'tool']
        self.assertTrue(tool_results[0].endswith(runner.PICTURES_DROPPED))
        self.assertFalse(tool_results[1].endswith(runner.PICTURES_DROPPED))

    def test_a_picture_counts_for_what_the_model_makes_of_it_not_for_its_bytes(self):
        refs = [{'cut_id': 'c1', 'frame': 'first', 'crop': None, 'label': 'x', 't': 0}] * 3
        message = runner.picture_message(refs)
        before = runner.block_tokens(message)
        self.assertGreaterEqual(before, 3 * images.TOKENS_PER_PICTURE)
        self.assertLess(before, 3 * images.TOKENS_PER_PICTURE + 100)
        runner.attach_pictures([message])
        self.assertEqual(runner.block_tokens(message), before)  # the base64 now in the message is not counted as text

    def test_a_picture_that_cannot_be_made_any_more_becomes_a_line_of_text(self):
        messages = self.history(True, [('F5',)])
        with patch.object(images, 'data_url', side_effect=images.FrameError('the keyframes changed')):
            runner.attach_pictures(messages)
        text = messages[3]['content']
        self.assertTrue(all(part['type'] == 'text' for part in text))
        self.assertIn('the keyframes changed', text[-1]['text'])


class RunnerVisionTest(VisionBase):
    def run_turn(self, provider, answers):
        session_id = self.session()
        self.worker_online()
        self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'Is the first frame right?', 'provider': 'fake', 'model': 'fake-model'})
        requests = []

        def fake_stream(endpoint, messages, tool_schemas, *, on_delta, checkpoint, **kwargs):
            requests.append({'messages': json.loads(json.dumps(messages)), 'tools': tool_schemas, 'vision': endpoint.vision})
            result = answers[len(requests) - 1]
            if result.content:
                on_delta(result.content, '')
            return result

        with patch.object(providers, 'registry', return_value={'fake': provider}), patch.object(providers, 'stream_chat', side_effect=fake_stream):
            runner.run_turn(*runner.claim_turn())
        return session_id, requests

    def test_the_model_looks_then_answers_with_the_pictures_in_its_next_request(self):
        call = providers.StepResult(tool_calls=[{'id': 'v1', 'name': 'view_frames', 'arguments': json.dumps({'frames': ['first', '1.0']})}], finish_reason='tool_calls')
        session_id, requests = self.run_turn(VisionProvider(), [call, providers.StepResult(content='It matches.', finish_reason='stop')])
        first, second = requests
        self.assertTrue(first['vision'])
        self.assertIn('view_frames', [item['function']['name'] for item in first['tools']])
        system = first['messages'][0]['content']
        self.assertIn('You can look at the video with `view_frames`', system)
        self.assertIn('look_at_frames:', system)
        roles = [message['role'] for message in second['messages']]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'tool', 'user'])
        pictures = second['messages'][-1]['content']
        self.assertEqual([part['type'] for part in pictures], ['text', 'image_url', 'image_url'])
        self.assertTrue(all(part['image_url']['url'].startswith('data:image/jpeg;base64,') for part in pictures[1:]))
        # The database keeps references, never pixels.
        detail = self.client.get(f'/api/assistant/sessions/{session_id}').json()
        self.assertEqual(detail['turns'][0]['status'], 'done', detail['turns'][0]['error'])
        self.assertTrue(detail['turns'][0]['usage']['vision'])
        tool = next(message for message in detail['messages'] if message['role'] == 'tool')
        self.assertEqual([(item['cut_id'], item['frame']) for item in tool['usage']['images']], [('c1', 'first'), ('c1', '1s')])
        self.assertNotIn('base64', json.dumps(detail))
        # The chat page asks for the same pictures through the API.
        response = self.client.get(f"/api/assistant/messages/{tool['id']}/images/0")
        self.assertEqual((response.status_code, response.headers['content-type']), (200, 'image/jpeg'))
        self.assertTrue(response.content.startswith(JPEG))
        self.assertEqual(self.client.get(f"/api/assistant/messages/{tool['id']}/images/5").status_code, 404)
        self.assertEqual(self.client.get('/api/assistant/messages/nope/images/0').status_code, 404)

    def test_a_model_that_cannot_see_is_not_told_about_looking(self):
        session_id, requests = self.run_turn(Blind(), [providers.StepResult(content='From the results only.', finish_reason='stop')])
        self.assertFalse(requests[0]['vision'])
        self.assertNotIn('view_frames', [item['function']['name'] for item in requests[0]['tools']])
        system = requests[0]['messages'][0]['content']
        self.assertNotIn('view_frames', system)
        self.assertNotIn('look_at_frames', system)

    def test_the_skill_is_listed_only_to_a_model_that_can_see(self):
        self.assertNotIn('look_at_frames', skills.index_text('cut'))
        self.assertIn('look_at_frames', skills.index_text('cut', vision=True))
        self.assertIn('look_at_frames', skills.index_text('sequence', vision=True))
        skill = skills.get('look_at_frames')
        self.assertTrue(skill['vision'])
        self.assertIn('view_frames', skill['body'])
        for name in ('explain_result', 'correct_result', 'still_image_prompt', 'h3_animation_prompt', 'cut_status_summary', 'sequence_direction'):
            self.assertIn('view_frames', skills.get(name)['body'], name)  # each says when to look, and only "if view_frames is in your tool list"
            self.assertIn('tool list', skills.get(name)['body'], name)
        config = self.client.get('/api/assistant/config').json()
        self.assertTrue(next(item for item in config['skills'] if item['name'] == 'look_at_frames')['vision'])


class ProviderVisionTest(unittest.TestCase):
    def test_a_configured_endpoint_says_whether_its_model_sees(self):
        config = {'id': 'lmstudio', 'base_url': 'http://lan:1234/v1', 'models': ['qwen3-vl']}
        self.assertFalse(providers.OpenAICompatible(config).endpoint('qwen3-vl', False).vision)
        seeing = providers.OpenAICompatible({**config, 'vision': True})
        self.assertTrue(seeing.endpoint('qwen3-vl', False).vision)
        self.assertTrue(seeing.describe()['vision'])

    def test_openrouter_free_never_gets_the_tool(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'k'}), patch.object(providers, 'openrouter_context', return_value=None):
            self.assertFalse(providers.OpenRouter().endpoint('openrouter/free', False).vision)

    def test_the_remote_server_sees_when_it_reported_vision_for_that_model(self):
        cfg = {'enabled': True, 'base_url': 'http://remote:8080', 'token': ''}
        state = {'state': 'online', 'detail': '', 'models': [{'id': 'seer', 'n_ctx': 8192, 'vision': True}, {'id': 'blind', 'n_ctx': 8192}], 'heartbeat': 0, 'active_jobs': 0}
        with patch.object(remote_text, 'config', return_value=cfg), patch.object(remote_text, 'state', return_value=state), \
             patch.object(providers.RemoteLlama, 'available', return_value=(True, '')):
            remote = providers.RemoteLlama()
            self.assertTrue(remote.endpoint('seer', False).vision)
            self.assertFalse(remote.endpoint('blind', False).vision)
            self.assertEqual([item['vision'] for item in remote.describe()['models']], [True, False])

    def test_the_local_model_sees_while_its_projector_is_loaded(self):
        runtime = Mock()
        runtime.process.poll.return_value = None
        runtime.ensure.return_value = 'http://127.0.0.1:1'
        runtime.chat_session.side_effect = lambda *args, **kwargs: nullcontext()
        for loaded in (True, False):
            runtime.vision = loaded
            with patch.object(local_text_runtime, 'runtime', runtime), patch.object(remote_text.pool, 'chat', return_value=nullcontext()), patch('app.db.connect') as connect:
                connect.return_value.__enter__.return_value = Mock()
                with providers.LocalQwen().session('m', False, None, lambda text: None) as endpoint:
                    self.assertEqual(endpoint.vision, loaded)


class VisionStatusTest(unittest.TestCase):
    def runtime(self, **values):
        runtime = Mock(vision_failed={}, vision=True, variant='standard')
        runtime.process.poll.return_value = None
        for key, value in values.items():
            setattr(runtime, key, value)
        return runtime

    def test_reasons_chat_cannot_see(self):
        with tempfile.TemporaryDirectory() as folder:
            projector = Path(folder) / 'mmproj.gguf'
            with patch.dict('os.environ', {'QWEN36_PROJECTOR_PATH': str(projector)}):
                self.assertEqual(local_text_runtime.vision_status(None, 'standard'), (False, 'The vision projector is not installed (python3 scripts/install-model.py qwen36-vision)'))
                self.assertFalse(local_text_runtime.vision_enabled())
                projector.touch()
                self.assertEqual(local_text_runtime.vision_status(self.runtime(), 'standard'), (True, ''))
                self.assertEqual(local_text_runtime.vision_status(None, 'mtp'), (True, ''))
                failed = self.runtime(vision_failed={'mtp': 'exited'})
                self.assertFalse(local_text_runtime.vision_status(failed, 'mtp')[0])
                self.assertTrue(local_text_runtime.vision_status(failed, 'standard')[0])  # another variant is tried on its own
                self.assertIn('did not load', local_text_runtime.vision_status(self.runtime(vision=False), 'standard')[1])
                with patch.dict('os.environ', {'QWEN36_TEXT_VISION': '0'}):
                    self.assertEqual(local_text_runtime.vision_status(self.runtime(), 'standard'), (False, 'Turned off (QWEN36_TEXT_VISION=0)'))


if __name__ == '__main__':
    unittest.main()
