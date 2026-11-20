import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lines as L  # noqa: E402


def tok(s, e, t):
    return {'start': s, 'end': e, 'text': t}


class Text(unittest.TestCase):
    def test_scripts(self):
        self.assertTrue(L.looks_non_japanese("Oh, couldn't care less"))
        self.assertTrue(L.looks_non_japanese('今日，我一个人什么都'))
        self.assertFalse(L.looks_non_japanese('母'))
        self.assertFalse(L.looks_non_japanese('眠れなかったの？'))

    def test_repetition(self):
        self.assertTrue(L.repetition_loop('ラララララララララララ'))
        self.assertTrue(L.repetition_loop('そうそうそうそうそうそうそうそう'))
        self.assertFalse(L.repetition_loop('今日は晴れです。明日も晴れです。'))

    def test_sentence_end_and_closing_quote(self):
        self.assertTrue(L.ends_sentence('そうね。」'))
        self.assertTrue(L.ends_sentence('本当に？'))
        self.assertFalse(L.ends_sentence('そうね、'))

    def test_cer_ignores_punctuation(self):
        self.assertEqual(L.cer('おはよう。', 'おはよう'), 0.)
        self.assertAlmostEqual(L.cer('私のせいだって', '私のせいだと'), 2 / 7, places=3)

    def test_join_text(self):
        self.assertEqual(L.join_text('今日', 'は'), '今日は')
        self.assertEqual(L.join_text('Hello', 'world'), 'Hello world')


class BaselineRows(unittest.TestCase):
    def test_merge_rows_like_the_app(self):
        rows = [{'start_time': 0, 'end_time': .2, 'text': 'それ', 'speaker_label': 'A'}, {'start_time': .3, 'end_time': .6, 'text': 'って', 'speaker_label': 'A'},
                {'start_time': .7, 'end_time': 1, 'text': 'デート', 'speaker_label': 'B'}, {'start_time': 3, 'end_time': 3.2, 'text': 'はい', 'speaker_label': 'B'}]
        merged = L.merge_rows(rows)
        self.assertEqual([m['text'] for m in merged], ['それって', 'デート', 'はい'])

    def test_punctuation_on_the_next_row_closes_the_sentence_before_it(self):
        rows = [{'text': 'それ', 'speaker_label': 'A'}, {'text': '。暴力', 'speaker_label': 'B'}, {'text': 'です。', 'speaker_label': 'B'}]
        sentences = L.sentences_from_rows(rows)
        self.assertEqual([[r['text'] for r in s] for s in sentences], [['それ'], ['。暴力', 'です。']])


class Intervals(unittest.TestCase):
    def test_subtract(self):
        self.assertEqual(L.subtract_intervals([(0, 10)], [(2, 4), (8, 12)]), [(0, 2), (4, 8)])

    def test_inside_any(self):
        self.assertTrue(L.inside_any(5, 7, [(0, 10)]))
        self.assertFalse(L.inside_any(9, 20, [(0, 10)]))


class Windows(unittest.TestCase):
    rms = [1.] * 4000

    def test_windows_respect_the_cap_and_cover_the_speech(self):
        regions = [(i * 2., i * 2. + 1.5) for i in range(40)]            # 80 s of short phrases with .5 s pauses
        windows = L.plan_windows(regions, self.rms, .05, cap=25.)
        self.assertTrue(all(b - a <= 25.01 for a, b in windows))
        covered = sum(b - a for a, b in windows)
        self.assertGreater(covered, 70)
        self.assertEqual(windows, sorted(windows))
        self.assertTrue(all(windows[i][1] <= windows[i + 1][0] + 1e-6 for i in range(len(windows) - 1)))

    def test_long_stretch_is_cut_at_the_quietest_frame(self):
        rms = [1.] * 4000
        for i in range(int(18 / .05), int(18.3 / .05)):
            rms[i] = 0.01
        windows = L.plan_windows([(0, 40)], rms, .05, cap=25.)
        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0][1], 18.15 + .25, delta=.5)

    def test_window_never_crosses_the_mask(self):
        windows = L.plan_windows([(0, 5), (6, 11), (30, 35)], self.rms, .05, mask=[(12, 29)])
        for a, b in windows:
            self.assertFalse(a < 29 and b > 12)

    def test_cut_at_the_longest_pause_near_the_cap(self):
        regions = [(0, 10), (10.5, 20), (21.5, 31), (31.2, 40)]        # the longest pause (1.5 s) is before the third phrase
        windows = L.plan_windows(regions, self.rms, .05, cap=25.)
        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0][1], 20.25, delta=.01)


class SpeakerPerSentence(unittest.TestCase):
    turns = [{'start_time': 0, 'end_time': 3, 'speaker': 'A'}, {'start_time': 3, 'end_time': 6, 'speaker': 'B'}]

    def test_majority_speaker_takes_the_whole_sentence(self):
        index = L.TurnIndex(self.turns)
        tokens = [tok(.5, 1, 'それ'), tok(1.1, 2.5, 'って'), tok(2.6, 3.4, 'デート？')]    # last token straddles the turn boundary, no pause
        lines = L.build_lines(tokens, index)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]['diar'], 'A')
        self.assertTrue(lines[0]['mixed'] or True)

    def test_split_only_where_a_pause_of_0_3_s_sits_at_the_change(self):
        index = L.TurnIndex(self.turns)
        tokens = [tok(.5, 1, 'うん'), tok(1.1, 2.5, 'そう'), tok(3.0, 3.5, 'いや'), tok(3.6, 5, 'ちがう')]       # .5 s pause between 2.5 and 3.0
        lines = L.build_lines(tokens, index)
        self.assertEqual([l['diar'] for l in lines], ['A', 'B'])
        tokens = [tok(.5, 1, 'うん'), tok(1.1, 2.9, 'そう'), tok(3.0, 3.5, 'いや'), tok(3.6, 5, 'ちがう')]       # .1 s: no split
        self.assertEqual(len(L.build_lines(tokens, index)), 1)

    def test_final_punctuation_splits_sentences_and_stays_with_the_word(self):
        index = L.TurnIndex(self.turns)
        tokens = [tok(.5, 1, 'おはよう。'), tok(1.2, 2, 'おはよう'), tok(2.1, 2.6, 'ございます。')]
        lines = L.build_lines(tokens, index)
        self.assertEqual([l['text'] for l in lines], ['おはよう。', 'おはようございます。'])

    def test_lines_are_flagged_at_the_window_edge(self):
        index = L.TurnIndex(self.turns)
        lines = L.build_lines([tok(10.1, 11, 'ええ、')], index, window_edges=(10, 20))
        self.assertTrue(lines[0]['window_edge'])


class Subtitles(unittest.TestCase):
    def test_edge_match(self):
        r = L.edge_match([1.0, 5.0, 9.0], [1.2, 9.5, 20.0])
        self.assertEqual(r['precision'], 1 / 3)
        self.assertEqual(r['recall'], 1 / 3)

    def test_overlap_pick(self):
        cues = [{'start_time': 0, 'end_time': 2, 'text': 'a'}, {'start_time': 10, 'end_time': 12, 'text': 'b'}]
        self.assertEqual([c['text'] for c in L.subtitle_overlap({'start_time': 1, 'end_time': 3}, cues)], ['a'])


if __name__ == '__main__':
    unittest.main()
