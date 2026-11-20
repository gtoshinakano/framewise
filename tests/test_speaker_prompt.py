import unittest

from app import speaker_prompt as qwen

WINDOW = [{'id': 'L1', 'ja': 'おはよう。'}, {'id': 'L2', 'ja': 'サクタさん、眠れなかったの？'}]
NAMES = ['Mai Sakurajima', 'Sakuta Azusagawa']


def item(i, speaker='Mai Sakurajima', **kw):
    return {'id': i, 'speaker': speaker, 'confidence': 'high', 'reason': 'voice', **kw}


class Validation(unittest.TestCase):
    def test_a_good_answer(self):
        answers, errors = qwen.validate({'lines': [item('L1'), item('L2', 'unknown')]}, WINDOW, NAMES)
        self.assertEqual(errors, [])
        self.assertEqual(answers['L2']['speaker'], 'unknown')

    def test_missing_line_and_unknown_speaker_are_sentences(self):
        _, errors = qwen.validate({'lines': [item('L1', 'Nobody')]}, WINDOW, NAMES)
        self.assertTrue(any('missing' in e for e in errors))
        self.assertTrue(any('speaker must be' in e for e in errors))

class Schema(unittest.TestCase):
    def test_schema_pins_ids_and_count(self):
        s = qwen.answer_schema(['L1', 'L2'], NAMES)
        props = s['properties']['lines']
        self.assertEqual((props['minItems'], props['maxItems']), (2, 2))
        self.assertEqual(props['items']['properties']['speaker']['enum'], NAMES + ['other', 'unknown'])


class Windows(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(qwen.make_windows(list(range(100)), 40, 4), [(0, 40), (36, 76), (72, 100)])

    def test_every_line_is_covered(self):
        for n in (1, 39, 40, 41, 80, 123):
            covered = set()
            for a, b in qwen.make_windows(list(range(n)), 40, 4):
                covered.update(range(a, b))
            self.assertEqual(covered, set(range(n)))
    def test_names_are_omitted_from_speaker_contract(self):
        item_schema = qwen.answer_schema(['L1'], NAMES)['properties']['lines']['items']
        self.assertEqual(set(item_schema['properties']), {'id', 'speaker', 'confidence', 'reason'})

    def test_only_open_lines_are_asked_and_the_interior_wins(self):
        items = [{'id': str(i), 'decided': 'Mai'} if i % 2 else {'id': str(i)} for i in range(42)]
        def ask(window, shown, number, total):
            self.assertTrue(all('decided' not in line for line in window))
            self.assertTrue(any('decided' in line for line in shown))
            self.assertEqual(total, 2)
            return {line['id']: number for line in window}
        result = qwen.attribute_open(items, ask)
        self.assertEqual(result['36'], 1)
        self.assertEqual(result['38'], 2)
        self.assertNotIn('37', result)
