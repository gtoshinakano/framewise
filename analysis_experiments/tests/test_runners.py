import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import conclude  # noqa: E402
import fake_backend  # noqa: E402
import pose_schema as ps  # noqa: E402
import runners  # noqa: E402
import scoring  # noqa: E402


class Scripted:
    """A client that answers from a list and remembers what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, max_tokens, schema=None, hint=None, thinking=False, temperature=0):
        self.sent.append(list(messages))
        reply = self.replies.pop(0)
        return reply if isinstance(reply, dict) else {'content': reply, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}


def strict_entry(handle='B1', **changes):
    entry = {'character_id': handle, 'body_yaw': 0, 'body_pose': 'standing', 'head_yaw': 45, 'head_pitch': 0, 'head_roll': 'neutral', 'eye_gaze': 'right', 'eye_openness': 'normal',
             'gaze_target': 'camera', 'mouth_state': 'closed', 'eyebrow_state': 'neutral', 'facial_expression': 'neutral', 'camera_elevation': 'eye_level', 'shot_size': 'medium'}
    entry.update(changes)
    return entry


def answer(*entries):
    return json.dumps({'composition': 'x', 'pose_detail': list(entries)})


SAMPLE = {'id': 's01', 'marked': [{'handle': 'B1', 'name': 'Mai', 'box': [100, 100, 300, 900]}, {'handle': 'B2', 'name': 'Sakuta', 'box': [600, 100, 900, 900]}],
          'hints': None, 'subtitle': None, 'width': 1280, 'height': 720}


def jpeg(directory):
    path = Path(directory) / 'frame.jpg'
    path.write_bytes(b'\xff\xd8\xff\xd9')
    return path


class FrameCallTest(unittest.TestCase):
    def people(self):
        return [{'ref': 'B1', 'name': 'Mai', 'box': [100, 100, 300, 900]}, {'ref': 'B2', 'name': 'Sakuta', 'box': [600, 100, 900, 900]}]

    def test_a_rejected_answer_is_sent_back_with_the_errors_and_retried(self):
        client = Scripted([answer(strict_entry('B1', head_yaw=30), strict_entry('B2')), answer(strict_entry('B1'), strict_entry('B2'))])
        with tempfile.TemporaryDirectory() as directory:
            record = runners.frame_call(client, SAMPLE, 'strict_detailed', 'normal', jpeg(directory), self.people())
        self.assertTrue(record['ok'])
        self.assertFalse(record['first_try_ok'])
        self.assertEqual(len(record['attempts']), 2)
        self.assertIn('45-degree grid', record['attempts'][0]['errors'][0])
        retry = client.sent[1]
        self.assertEqual([m['role'] for m in retry], ['system', 'user', 'assistant', 'user'])
        self.assertIn('45-degree grid', retry[3]['content'])
        self.assertEqual(record['answer'][0]['head_yaw'], 45)
        self.assertIn('POSE DETAIL', client.sent[0][0]['content'])

    def test_truncation_and_garbage_count_as_failures_and_three_failures_are_hard(self):
        client = Scripted([{'content': '{"pose_detail": [', 'finish_reason': 'length', 'usage': {}}, 'not json', answer(strict_entry('B1'))])
        with tempfile.TemporaryDirectory() as directory:
            record = runners.frame_call(client, SAMPLE, 'strict_detailed', 'normal', jpeg(directory), self.people())
        self.assertFalse(record['ok'])
        self.assertEqual(len(record['attempts']), 3)
        self.assertIn('cut off', record['attempts'][0]['errors'][0])
        self.assertIn('not valid JSON', record['attempts'][1]['errors'][0])
        self.assertEqual(len(record['last_answer']), 1)  # the last parsed pose_detail is kept for the report
        self.assertIn('invalid after 3', record['error'])

    def test_nobody_marked_expects_an_empty_array(self):
        client = Scripted([answer()])
        with tempfile.TemporaryDirectory() as directory:
            record = runners.frame_call(client, {**SAMPLE, 'marked': []}, 'full_confidence', 'normal', jpeg(directory), [])
        self.assertTrue(record['ok'])
        self.assertEqual(record['answer'], [])

    def test_mirror_people_flips_boxes_and_renumbers(self):
        flipped, mapping = runners.mirror_people(self.people())
        self.assertEqual(mapping, {'B1': 'B2', 'B2': 'B1'})
        self.assertEqual([p['ref'] for p in flipped], ['B1', 'B2'])
        self.assertEqual(flipped[0]['box'], [100, 100, 400, 900])  # Sakuta, now at the left
        self.assertEqual(flipped[0]['name'], 'Sakuta')
        self.assertEqual(flipped[1]['box'], [700, 100, 900, 900])

    def test_subtitle_strip_scales_to_the_analysis_frame(self):
        self.assertEqual(runners.strip_regions({'top': 0.8, 'bottom': 0.95}, 1280, 720), [[0, 576, 1280, 684]])
        self.assertEqual(runners.strip_regions(None, 1280, 720), [])

    def test_the_fake_clients_exercise_retries_and_stay_valid_after_them(self):
        for approach in ps.APPROACHES:
            for number in range(8):
                sample = {**SAMPLE, 'id': f's{number}'}
                client = fake_backend.FakeFrameClient()
                with tempfile.TemporaryDirectory() as directory:
                    record = runners.frame_call(client, sample, approach, 'normal', jpeg(directory), self.people())
                self.assertTrue(record['ok'], (approach, number, record['error']))
        record_attempts = []
        for number in range(30):
            with tempfile.TemporaryDirectory() as directory:
                record_attempts.append(len(runners.frame_call(fake_backend.FakeFrameClient(), {**SAMPLE, 'id': f'x{number}'}, 'strict_detailed', 'normal', jpeg(directory), self.people())['attempts']))
        self.assertTrue(any(n == 2 for n in record_attempts))  # some first attempts are deliberately broken


class VideoGeneratorTest(unittest.TestCase):
    def test_the_generator_validates_pose_detail_and_keeps_every_attempt(self):
        roster = [{'name': 'Mai Sakurajima', 'character_context_id': 'mai'}]
        windows = []
        generate = runners.make_generator('strict_detailed', windows, roster, fake_backend.FakeVideoBackend())
        output, usage, attempts = generate(None, None, None, None, 4, 'prompt', 3.0, 10.0)
        self.assertEqual(attempts, len(windows[0]['attempts']))
        self.assertTrue(windows[0]['ok'])
        self.assertEqual(windows[0]['characters'], ['Mai Sakurajima'])
        self.assertEqual(windows[0]['answer'][0]['character'], 'Mai Sakurajima')
        self.assertEqual(usage['total_tokens'], usage['prompt_tokens'] + usage['completion_tokens'])
        self.assertEqual(ps.validate_video_answer(windows[0]['answer'], 'strict_detailed', ['Mai Sakurajima']), [])

    def test_a_window_that_never_validates_raises_after_three_attempts(self):
        class Broken:
            def generate(self, approach, prompt, attempt, window, names):
                return '{"characters": [], "pose_detail": 5}', 100, 20, 1.0

        windows = []
        generate = runners.make_generator('full_confidence', windows, [], Broken())
        with self.assertRaises(RuntimeError):
            generate(None, None, None, None, 4, 'prompt', 3.0, 0.0)
        self.assertEqual(len(windows[0]['attempts']), 3)
        self.assertFalse(windows[0]['ok'])


class ConcludeTest(unittest.TestCase):
    def test_validators(self):
        good = {'verdict': 'adopt', 'headline': 'h', 'strengths': ['a'], 'weaknesses': ['b'], 'reliable_fields': ['head_yaw'], 'unreliable_fields': [], 'recommendation': 'r'}
        self.assertEqual(conclude.validate_run(good), [])
        self.assertTrue(conclude.validate_run({**good, 'verdict': 'maybe'}))
        self.assertTrue(conclude.validate_run({**good, 'reliable_fields': ['head_yawn']}))
        self.assertTrue(conclude.validate_run({**good, 'extra': 1}))
        overall = {'best_approach': 'strict_detailed', 'headline': 'h', 'reasoning': 'r', 'fields_to_add': [], 'fields_to_avoid': [], 'next_steps': ['x'], 'caveats': ['y']}
        self.assertEqual(conclude.validate_overall(overall), [])
        self.assertTrue(conclude.validate_overall({**overall, 'best_approach': 'A1'}))

    def test_run_retries_with_the_validator_errors_and_keeps_finished_answers(self):
        sample = {'id': 's01', 'reason': 'r', 'marked': [], 'stored': {'baseline': {'frame_seconds': 20, 'video_seconds': 22}, 'first_frame': {}, 'video': {}}}
        scored = scoring.score_all([sample], [])
        seen = []
        good = fake_backend.fake_chat('', 'x')

        def chat(system, user):
            seen.append(user)
            return ('{"verdict": "nope"}' if len(seen) == 1 else fake_backend.fake_chat(system, user)), 2.0, {'completion_tokens': 5}

        result = conclude.run(chat, scored, [sample], [])
        self.assertTrue(all(result['runs'][a]['ok'] for a in ps.APPROACHES) and result['overall']['ok'])
        self.assertEqual(len(result['runs']['full_confidence']['attempts']), 2)
        self.assertIn('rejected', seen[1])
        self.assertIn('verdict must be one of', seen[1])
        calls_before = len(seen)
        conclude.run(chat, scored, [sample], [], previous=result)
        self.assertEqual(len(seen), calls_before)  # nothing is asked twice
        self.assertIn('Evidence for A1', seen[0])
        self.assertTrue(good)


if __name__ == '__main__':
    unittest.main()
