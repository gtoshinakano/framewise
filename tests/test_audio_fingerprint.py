import unittest

try:
    import numpy as np
    from app import audio_fingerprint as oped
except ImportError:  # the host has no numpy; these run inside the image
    np = None


@unittest.skipIf(np is None, 'needs numpy')
class RepeatedAudio(unittest.TestCase):
    def make(self, seed, shared, at, total=240):
        rng = np.random.default_rng(seed)
        audio = rng.normal(0, .05, total * 16000).astype(np.float32)
        audio[at * 16000:(at + len(shared) // 16000) * 16000] = shared[:(total - at) * 16000] * 1.0 + rng.normal(0, .005, min(len(shared), (total - at) * 16000)).astype(np.float32)
        return audio

    def test_shared_segment_is_found_at_different_places(self):
        rng = np.random.default_rng(7)
        # a "song": sum of drifting tones, the same in both episodes
        song = (np.random.default_rng(99).normal(0, .3, 80 * 16000) * (1 + .8 * np.sin(np.arange(80 * 16000) / 16000. * 3))).astype(np.float32)
        a, b = self.make(1, song, 20), self.make(2, song, 100)
        ha, hb = oped.fingerprint(a), oped.fingerprint(b)
        segments = oped.match_pair(ha, hb)
        self.assertTrue(segments)
        best = max(segments, key=lambda s: s['a'][1] - s['a'][0])
        self.assertAlmostEqual(best['a'][0], 20, delta=2)
        self.assertAlmostEqual(best['b'][0], 100, delta=2)
        self.assertGreater(best['a'][1] - best['a'][0], 60)

    def test_unrelated_noise_matches_nothing(self):
        a = np.random.default_rng(3).normal(0, .05, 120 * 16000).astype(np.float32)
        b = np.random.default_rng(4).normal(0, .05, 120 * 16000).astype(np.float32)
        self.assertEqual(oped.match_pair(oped.fingerprint(a), oped.fingerprint(b)), [])


class Consolidation(unittest.TestCase):
    def test_support_and_medians(self):
        if np is None:
            self.skipTest('needs numpy')
        pairs = {(0, 1): [{'a': (10, 100), 'b': (12, 102), 'score': 1}], (0, 2): [{'a': (10, 98), 'b': (30, 118), 'score': 1}],
                 (1, 2): [{'a': (12, 102), 'b': (30, 120), 'score': 1}]}
        result = oped.consolidate(pairs, 3)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]['support'], 2)
        self.assertAlmostEqual(result[0][0]['start'], 10)

    def test_single_partner_is_not_enough_in_a_big_set(self):
        if np is None:
            self.skipTest('needs numpy')
        pairs = {(0, 1): [{'a': (10, 100), 'b': (10, 100), 'score': 1}]}
        self.assertEqual(oped.consolidate(pairs, 7)[0], [])


if __name__ == '__main__':
    unittest.main()
