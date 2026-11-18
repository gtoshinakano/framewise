"""The 2026-11-18 prompt variants: walker (observer + a definition of walking) and roster_cam (roster + the measured camera)."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402
from tests.test_core import sample  # noqa: E402


def text(smp, window, style, context):
    return prompts.prompt_text(prompts.build_messages(smp, window, style, context, ['' for _ in window['frames']]))


class Variants(unittest.TestCase):
    def setUp(self):
        self.sample = sample(5)
        self.window = self.sample['windows'][0]

    def test_walker_differs_from_observer_only_in_the_walking_sentence(self):
        observer, walker = prompts.STYLES['observer']['system'], prompts.STYLES['walker']['system']
        self.assertNotEqual(observer, walker)
        self.assertEqual(walker.replace(prompts.WALK_NEW, prompts.WALK_OLD), observer)
        self.assertIn('legs are cropped out of the picture', walker)
        self.assertNotIn('legs are cropped', observer)

    def test_observer_roster_prompt_is_unchanged(self):
        # the stored baseline answers were written with this prompt: it must not drift
        full = text(self.sample, self.window, 'observer', 'roster')
        self.assertNotIn('Camera movement measured', full)
        self.assertNotIn('travels over the ground', full)

    def test_camera_context_adds_one_line_with_every_pair(self):
        plain, cam = text(self.sample, self.window, 'walker', 'roster'), text(self.sample, self.window, 'walker', 'roster_cam')
        extra = [line for line in cam.splitlines() if line not in plain.splitlines()]
        self.assertEqual(len(extra), 1)
        for a, b in zip(['F1', 'F2', 'F3', 'F4'], ['F2', 'F3', 'F4', 'F5']):
            self.assertIn(f'{a}→{b} static', extra[0])

    def test_an_unmeasured_pair_says_so(self):
        smp = sample(5)
        smp['pairs'][1]['camera'] = {'move': 'unclear'}
        smp['pairs'][2]['camera'] = {'move': 'tilt_up', 'dx': 0.0, 'dy': 0.04}
        line = prompts._camera_lines(smp, smp['windows'][0])[0]
        self.assertIn('F2→F3 not measurable', line)
        self.assertIn('F3→F4 tilt_up', line)

    def test_a_later_window_names_only_its_own_pairs(self):
        smp = sample(9)
        line = prompts._camera_lines(smp, smp['windows'][1])[0]
        self.assertIn('F5→F6 static', line)
        self.assertNotIn('F1→F2', line)

    def test_neither_variant_is_chained_or_told_the_speech(self):
        self.assertFalse(prompts.CONTEXTS['roster_cam']['chained'] or prompts.CONTEXTS['roster_cam']['tells_speech'])


if __name__ == '__main__':
    unittest.main()
