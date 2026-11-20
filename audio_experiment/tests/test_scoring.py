import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scoring as S  # noqa: E402
import voice_tables as VT  # noqa: E402


def line(i, s, e, text, speaker, **kw):
    return {'id': i, 'start_time': s, 'end_time': e, 'text': text, 'speaker': speaker, **kw}


ANCHORS = [{'episode': 1, 'kind': 'sample', 'character': 'Mai Sakurajima', 'start': 10, 'end': 20, 'range_id': 'x:0'},
           {'episode': 1, 'kind': 'manual', 'character': 'Sakuta Azusagawa', 'start': 50, 'end': 50.3, 'range_id': 'row:1'}]


class Anchors(unittest.TestCase):
    def test_strict_and_answered_accuracy(self):
        lines = [line('a', 11, 13, '...', 'Mai Sakurajima'), line('b', 14, 16, '...', 'Rio Futaba'), line('c', 17, 18, '...', None), line('d', 30, 31, '...', 'Rio Futaba')]
        r = S.anchor_accuracy(lines, ANCHORS, kind='sample')
        self.assertEqual((r['n'], r['answered'], r['correct']), (3, 2, 1))
        self.assertAlmostEqual(r['strict'], 1 / 3)
        self.assertAlmostEqual(r['answered_accuracy'], .5)

    def test_a_line_only_partly_in_a_sample_range_is_not_an_anchor(self):
        self.assertEqual(S.anchor_accuracy([line('a', 18, 30, '...', 'Mai Sakurajima')], ANCHORS)['n'], 0)

    def test_manual_rows_match_by_the_row_span(self):
        self.assertEqual(S.anchor_accuracy([line('a', 49, 52, '...', 'Sakuta Azusagawa')], ANCHORS, kind='manual')['correct'], 1)


class Contradictions(unittest.TestCase):
    def test_addressing_oneself(self):
        lines = [line('a', 0, 1, 'サクタさん、起きて', 'Sakuta Azusagawa'), line('b', 1, 2, 'サクタさん、起きて', 'Mai Sakurajima'), line('c', 2, 3, '咲太、起きて', 'Sakuta Azusagawa')]
        r = S.addressed_contradictions(lines)
        self.assertEqual((r['addressed_lines'], r['contradictions']), (2, 1))

    def test_unknown_speaker_is_not_counted(self):
        self.assertEqual(S.addressed_contradictions([line('a', 0, 1, '梓川さん', 'unknown'), line('b', 0, 1, '麻衣先輩', None)])['addressed_lines'], 0)


class Pronouns(unittest.TestCase):
    def test_orthographic_variants_are_one_class(self):
        self.assertEqual(S.pronoun_class('私は、わたしの'), {'watashi': 2})

    def test_entropy_is_lower_for_a_steady_speaker(self):
        steady = [line(str(i), i, i + 1, '僕は行く', 'A') for i in range(8)]
        mixed = [line(str(i), i, i + 1, '僕は' if i % 2 else '俺は', 'B') for i in range(8)]
        r = S.pronoun_consistency(steady + mixed)
        self.assertEqual(r['by_character']['A']['entropy'], 0.)
        self.assertAlmostEqual(r['by_character']['B']['entropy'], 1.)


class Agreement(unittest.TestCase):
    def test_only_lines_both_answered(self):
        a = [line('1', 0, 1, '', 'X'), line('2', 1, 2, '', 'Y'), line('3', 2, 3, '', None)]
        b = [line('1', 0, 1, '', 'X'), line('2', 1, 2, '', 'Z'), line('3', 2, 3, '', 'X')]
        r = S.agreement(a, b)
        self.assertEqual((r['both_answered'], r['same']), (2, 1))


class Ranking(unittest.TestCase):
    def test_anchor_first_then_contradictions(self):
        rows = [{'method': 'a', 'anchor_strict': .8, 'contradiction_rate': .1}, {'method': 'b', 'anchor_strict': .9, 'contradiction_rate': .5},
                {'method': 'c', 'anchor_strict': .9, 'contradiction_rate': .2}, {'method': 'd', 'anchor_strict': None}]
        self.assertEqual([r['method'] for r in S.rank_speaker_methods(rows)], ['c', 'b', 'a', 'd'])


class BaselineSwitches(unittest.TestCase):
    def test_sentence_with_two_labels(self):
        rows = [{'text': 'それ', 'speaker_label': 'A'}, {'text': 'って デート', 'speaker_label': 'B'}, {'text': '？', 'speaker_label': 'B'}, {'text': 'はい。', 'speaker_label': 'C'}]
        self.assertEqual(S.baseline_speaker_switches(rows), 1)


class VoiceTables(unittest.TestCase):
    refs = [{'range_id': 's1:0', 'character': 'Mai', 'episode': 1, 'start': 10, 'end': 14}, {'range_id': 's1:1', 'character': 'Mai', 'episode': 1, 'start': 20, 'end': 24},
            {'range_id': 's2:0', 'character': 'Rio', 'episode': 1, 'start': 30, 'end': 35}]

    def test_leave_one_out_only_drops_the_overlapping_range(self):
        exclude = VT.overlapping_ranges(self.refs, 1, 11, 13)
        self.assertEqual(exclude, {'s1:0'})
        cos = {'s1:0': .9, 's1:1': .5, 's2:0': .3}
        self.assertAlmostEqual(VT.scores(cos, self.refs, exclude)['Mai'], .5)
        self.assertAlmostEqual(VT.scores(cos, self.refs)['Mai'], .7)

    def test_a_character_with_no_other_range_disappears(self):
        cos = {'s1:0': .9, 's1:1': .5, 's2:0': .8}
        self.assertNotIn('Rio', VT.scores(cos, self.refs, {'s2:0'}))

    def test_decide_needs_threshold_and_margin(self):
        self.assertEqual(VT.decide({'a': .6, 'b': .3})[0], 'a')
        self.assertIsNone(VT.decide({'a': .6, 'b': .58})[0])
        self.assertIsNone(VT.decide({'a': .2, 'b': .0})[0])

    def test_other_episode_references_are_not_excluded(self):
        self.assertEqual(VT.overlapping_ranges(self.refs, 3, 11, 13), set())

    def test_app_rule(self):
        samples = [{'id': 'a', 'character': 'Mai'}, {'id': 'b', 'character': 'Rio'}]
        m = VT.match_clusters_app({'C0': {'a': .8, 'b': .5}, 'C1': {'a': .75, 'b': .72}, 'C2': {'a': .4, 'b': .3}}, samples)
        self.assertEqual(m, {'C0': 'Mai'})


if __name__ == '__main__':
    unittest.main()
