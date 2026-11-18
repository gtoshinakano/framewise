"""The keyframe answer contract and the prompt (ported from ``multiframe_experiment/tests/test_core.py`` and ``test_prompts.py``)."""
import hashlib
import unittest

from app import keyframe_prompts as prompts
from app import keyframe_schema as schema


def frames(n, step=0.4):
    return [{'label': f'F{k}', 't': round((k - 1) * step, 3)} for k in range(1, n + 1)]


def window(n=5):
    return schema.make_windows(frames(n))[0]


def good_answer(win, name='Tomoe Koga'):
    names = schema.labels(win)
    return {'frames': [{'frame': label, 'characters': [{'name': name, 'state': 'standing'}]} for label in names],
            'segments': [{'from': a, 'to': b, 'camera': {'move': 'static', 'detail': ''},
                          'characters': [{'name': name, 'walking': 'no', 'speaking': 'yes', 'blinking': 'no',
                                          'changes': [{'part': 'head', 'description': 'turns left'}]}], 'other_changes': []}
                         for a, b in zip(names, names[1:])], 'summary': 'x'}


class Windows(unittest.TestCase):
    def test_overlap_by_one(self):
        windows = schema.make_windows(frames(9))
        self.assertEqual([[f['label'] for f in w['frames']] for w in windows], [['F1', 'F2', 'F3', 'F4', 'F5'], ['F5', 'F6', 'F7', 'F8', 'F9']])
        self.assertEqual([len(w['frames']) for w in schema.make_windows(frames(10))], [5, 5, 2])
        self.assertEqual(len(schema.make_windows(frames(1))), 1)
        self.assertEqual([len(w['frames']) for w in schema.make_windows(frames(4))], [4])


class Schema(unittest.TestCase):
    def test_valid_and_coerced(self):
        win = window()
        answer = good_answer(win)
        answer['segments'][0]['camera']['move'] = 'Pan Left'
        answer['segments'][0]['characters'][0]['walking'] = True
        clean, errors = schema.validate(answer, win)
        self.assertEqual(errors, [])
        self.assertEqual(clean['segments'][0]['camera']['move'], 'pan_left')
        self.assertEqual(clean['segments'][0]['characters'][0]['walking'], 'yes')
        self.assertEqual(clean['segments'][1]['start'], 0.4)
        self.assertEqual(clean['frames'][2], {'frame': 'F3', 't': 0.8, 'characters': [{'name': 'Tomoe Koga', 'state': 'standing'}]})

    def test_rejects_wrong_count_and_pairs(self):
        win = window()
        answer = good_answer(win)
        answer['segments'] = answer['segments'][:2]
        self.assertTrue(schema.validate(answer, win)[1])
        answer = good_answer(win)
        answer['segments'][1]['from'] = 'F3'
        self.assertTrue(schema.validate(answer, win)[1])

    def test_duplicate_character_is_merged(self):
        win = window()
        answer = good_answer(win)
        twin = dict(answer['segments'][0]['characters'][0], walking='yes', changes=[{'part': 'arms', 'description': 'raises the arm'}])
        answer['segments'][0]['characters'].append(twin)
        clean, errors = schema.validate(answer, win)
        self.assertEqual(errors, [])
        person = clean['segments'][0]['characters']
        self.assertEqual(len(person), 1)
        self.assertEqual(person[0]['walking'], 'yes')
        self.assertEqual(len(person[0]['changes']), 2)

    def test_frames_required(self):
        win = window()
        answer = good_answer(win)
        answer['frames'] = answer['frames'][:3]
        self.assertTrue(schema.validate(answer, win)[1])
        self.assertEqual(schema.json_schema(win)['required'][0], 'frames')

    def test_schema_fixes_count_and_labels(self):
        grammar = schema.json_schema(window())
        self.assertEqual(grammar['properties']['segments']['minItems'], 4)
        self.assertEqual(grammar['properties']['frames']['items']['properties']['frame']['enum'], ['F1', 'F2', 'F3', 'F4', 'F5'])

    def test_parse_json_with_prose(self):
        self.assertEqual(schema.parse_json('Sure: {"a": "}"} trailing'), {'a': '}'})
        self.assertEqual(schema.parse_json('```json\n{"a": 1}\n```'), {'a': 1})

    def test_no_change_wording(self):
        self.assertTrue(schema.is_no_change('arm remains extended toward the girl'))
        self.assertTrue(schema.is_no_change("Tomoe's green top and white skirt remain unchanged"))
        self.assertFalse(schema.is_no_change('raises the right arm'))

    def test_errors_go_back_as_one_message(self):
        message = schema.format_errors(['first', 'second'])
        self.assertIn('- first\n- second', message)
        self.assertTrue(message.endswith('JSON object only.'))


class Prompt(unittest.TestCase):
    def setUp(self):
        self.sample = {'duration': 1.8, 'no_characters': False, 'subtitle_band': {'top': 0.8, 'bottom': 1.0},
                       'roster': [{'name': 'Tomoe Koga', 'box': [500, 0, 900, 1000], 'side': 'right'}]}

    def text(self, sample, win=None):
        win = win or window()
        return prompts.prompt_text(prompts.build_messages(sample, win, ['' for _ in win['frames']]))

    def test_system_prompt_is_the_measured_one(self):
        # The experiment's numbers belong to this exact text (observer style); change it only with a new PROMPT_VERSION.
        self.assertEqual(hashlib.sha256(prompts.SYSTEM.encode()).hexdigest(), '40cd7b372269d31fcb40bbc4278b9d49c7d5cb86560c24de55a785bbbf499e3a')
        self.assertEqual(prompts.PROMPT_VERSION, 'observer-roster-v1')

    def test_roster_context(self):
        text = self.text(self.sample)
        self.assertIn('- Tomoe Koga: box [500, 0, 900, 1000] (right)', text)
        self.assertIn('covers 80-100% of the image height; ignore it.', text)
        self.assertIn('F3 · t=0.80 s\n<image>', text)
        self.assertIn('Answer with exactly 4 segments: F1→F2, F2→F3, F3→F4, F4→F5. JSON only.', text)
        self.assertNotIn('Transcript', text)

    def test_a_person_named_later_says_where_they_appear(self):
        sample = {**self.sample, 'roster': [{'name': 'Mai', 'box': [0, 0, 300, 1000], 'side': 'left', 'first_seen': 'F3'}]}
        self.assertIn('- Mai: box [0, 0, 300, 1000] (left, first seen in F3)', self.text(sample))

    def test_empty_roster_and_no_characters(self):
        self.assertIn('"unknown person 1"', self.text({**self.sample, 'roster': []}))
        self.assertIn('marked this shot as having no characters', self.text({**self.sample, 'no_characters': True}))

    def test_no_band_no_line(self):
        self.assertNotIn('subtitle strip covers', self.text({**self.sample, 'subtitle_band': None}))


if __name__ == '__main__':
    unittest.main()
