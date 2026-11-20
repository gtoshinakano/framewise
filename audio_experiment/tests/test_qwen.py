import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen  # noqa: E402

WINDOW = [{'id': 'L1', 'ja': 'おはよう。'}, {'id': 'L2', 'ja': 'サクタさん、眠れなかったの？'}]
NAMES = ['Mai Sakurajima', 'Sakuta Azusagawa']


def item(i, speaker='Mai Sakurajima', **kw):
    return {'id': i, 'speaker': speaker, 'confidence': 'high', 'reason': 'voice', 'text_fix': '', 'fix_reason': 'none', **kw}


class Validation(unittest.TestCase):
    def test_a_good_answer(self):
        answers, errors = qwen.validate({'lines': [item('L1'), item('L2', 'unknown')]}, WINDOW, NAMES)
        self.assertEqual(errors, [])
        self.assertEqual(answers['L2']['speaker'], 'unknown')

    def test_missing_line_and_unknown_speaker_are_sentences(self):
        _, errors = qwen.validate({'lines': [item('L1', 'Nobody')]}, WINDOW, NAMES)
        self.assertTrue(any('missing' in e for e in errors))
        self.assertTrue(any('speaker must be' in e for e in errors))

    def test_text_fix_is_only_for_small_changes_with_a_reason(self):
        _, errors = qwen.validate({'lines': [item('L1'), item('L2', text_fix='咲太さん、眠れなかったの？', fix_reason='name')]}, WINDOW, NAMES)
        self.assertEqual(errors, [])
        _, errors = qwen.validate({'lines': [item('L1', text_fix='まったく別の長い文章になってしまいました', fix_reason='name'), item('L2')]}, WINDOW, NAMES)
        self.assertTrue(any('changes too much' in e for e in errors))
        _, errors = qwen.validate({'lines': [item('L1', text_fix='おはよう', fix_reason='none'), item('L2')]}, WINDOW, NAMES)
        self.assertTrue(any('fix_reason' in e for e in errors))


class Schema(unittest.TestCase):
    def test_schema_pins_ids_and_count(self):
        s = qwen.answer_schema(['L1', 'L2'], NAMES)
        props = s['properties']['lines']
        self.assertEqual((props['minItems'], props['maxItems']), (2, 2))
        self.assertEqual(props['items']['properties']['speaker']['enum'], NAMES + ['other', 'unknown'])


class Windows(unittest.TestCase):
    def test_overlap_and_shift(self):
        self.assertEqual(qwen.make_windows(list(range(100)), 40, 4), [(0, 40), (36, 76), (72, 100)])
        self.assertEqual(qwen.make_windows(list(range(100)), 40, 4, offset=20)[0], (0, 20))

    def test_every_line_is_covered(self):
        for n in (1, 39, 40, 41, 80, 123):
            covered = set()
            for a, b in qwen.make_windows(list(range(n)), 40, 4, offset=20):
                covered.update(range(a, b))
            self.assertEqual(covered, set(range(n)))


class Profiles(unittest.TestCase):
    def test_profiles_need_confident_lines(self):
        lines = [{'speaker': 'Mai Sakurajima', 'confidence': 'high', 'text': '私は先輩よ。'}] * 9 + [{'speaker': 'Rio Futaba', 'confidence': 'low', 'text': '僕は'}] * 9
        rows = qwen.speech_profiles(lines)
        self.assertEqual(list(rows), ['Mai Sakurajima'])
        self.assertIn('watashi', rows['Mai Sakurajima']['text'])


class RetryLoop(unittest.TestCase):
    def test_a_rejected_answer_is_sent_back_and_corrected(self):
        calls = []

        class Client:
            def chat(self, messages, max_tokens, schema_=None, hint=None, temperature=.2):
                calls.append(messages)
                bad = len(calls) == 1
                out = '{"lines": [' + ('' if bad else '{"id":"L1","speaker":"Mai Sakurajima","confidence":"high","reason":"voice","text_fix":"","fix_reason":"none"},') + \
                      '{"id":"L2","speaker":"unknown","confidence":"low","reason":"other","text_fix":"","fix_reason":"none"}]}'
                return {'content': out, 'finish_reason': 'stop', 'usage': {}, 'timings': {}, 'grammar': True}
        result = qwen.call_window(Client(), WINDOW, NAMES)
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['attempts']), 2)
        self.assertIn('rejected', calls[1][-1]['content'])


if __name__ == '__main__':
    unittest.main()
