import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import names as N  # noqa: E402
import round2_rules as R  # noqa: E402


def line(i, start, end, diar='A', text='x'):
    return {'id': i, 'start_time': start, 'end_time': end, 'diar': diar, 'text': text}


class Names(unittest.TestCase):
    def test_katakana_readings_become_the_verified_spelling(self):
        text, changes = N.apply_aliases('二年一組のアズサガワサクタです。サクタ、カエデは？')
        self.assertEqual(text, '二年一組の梓川咲太です。咲太、かえでは？')
        self.assertTrue(changes)

    def test_two_kana_names_only_with_an_honorific_or_the_family_name(self):
        self.assertEqual(N.apply_aliases('マイさん、桜島マイだよ。')[0], '麻衣さん、桜島麻衣だよ。')
        self.assertEqual(N.apply_aliases('マイペースだね。')[0], 'マイペースだね。')
        self.assertEqual(N.apply_aliases('リオデジャネイロ')[0], 'リオデジャネイロ')

    def test_a_name_inside_a_longer_katakana_word_stays(self):
        self.assertEqual(N.apply_aliases('サクタロウ')[0], 'サクタロウ')

    def test_a_fix_may_only_swap_names(self):
        self.assertTrue(N.name_only_change('安沢川サクタです。', '梓川サクタです。'))
        self.assertTrue(N.name_only_change('牧之原証拠以外', '牧之原翔子以外'))
        self.assertFalse(N.name_only_change('安沢川です。', '梓川です！'))                # punctuation changed
        self.assertFalse(N.name_only_change('安沢川です。', '相坂です。'))                 # not a cast spelling
        self.assertFalse(N.name_only_change('おはよう。', 'おはよう、梓川。'))              # a name added
        self.assertFalse(N.name_only_change('梓川です。', '梓川です。'))

    def test_validate_keeps_good_fixes_and_explains_bad_ones(self):
        window = [{'id': 'L1', 'ja': '安沢川です。'}, {'id': 'L2', 'ja': 'おはよう。'}]
        fixes, errors = N.validate({'fixes': [{'id': 'L1', 'ja': '梓川です。'}, {'id': 'L2', 'ja': 'おはよう、咲太。'}]}, window)
        self.assertEqual(fixes, {'L1': '梓川です。'})
        self.assertEqual(len(errors), 1)
        self.assertIn('L2', errors[0])


class Voice(unittest.TestCase):
    REFS = [{'range_id': 'a:0', 'character': 'Mai', 'episode': 1, 'start': 100., 'end': 104.},
            {'range_id': 'a:1', 'character': 'Mai', 'episode': 1, 'start': 500., 'end': 504.},
            {'range_id': 'b:0', 'character': 'Sakuta', 'episode': 1, 'start': 900., 'end': 904.}]

    def test_guard_leaves_out_references_within_a_minute(self):
        cos = {'a:0': .9, 'a:1': .5, 'b:0': .3}
        self.assertAlmostEqual(R.guarded_scores(cos, self.REFS, 1, 110., 111.)['Mai'], .5)
        self.assertAlmostEqual(R.guarded_scores(cos, self.REFS, 2, 110., 111.)['Mai'], .7)

    def test_cascade_takes_the_first_decisive_source_and_keeps_evidence_otherwise(self):
        weak = {'Mai': .40, 'Sakuta': .38}
        strong = {'Mai': .60, 'Sakuta': .20}
        d = R.cascade({'line': weak, 'pad': strong})
        self.assertEqual((d['speaker'], d['source']), ('Mai', 'pad'))
        d = R.cascade({'line': weak})
        self.assertIsNone(d['speaker'])
        self.assertEqual(d['evidence'][0][0], 'Mai')
        self.assertEqual(R.cascade({'line': strong}, top_min=.7)['speaker'], None)

    def test_pad_grows_into_silence_but_not_into_neighbours(self):
        ordered = [line('L1', 10., 11.), line('L2', 11.4, 11.6), line('L3', 14., 15.)]
        self.assertEqual(R.pad_span(ordered, 1), (11.0, 12.6))
        self.assertEqual(R.pad_span(ordered, 0), (9.0, 11.4))

    def test_turn_stretch_follows_the_speaker_across_short_gaps(self):
        turns = R.Turns([{'start_time': 9., 'end_time': 10.5, 'speaker': 'A'}, {'start_time': 10.6, 'end_time': 12., 'speaker': 'A'},
                         {'start_time': 12., 'end_time': 14., 'speaker': 'B'}, {'start_time': 14.5, 'end_time': 20., 'speaker': 'A'}])
        self.assertEqual(turns.stretch(10.4, 10.7, 'A'), (9.0, 12.0))
        self.assertIsNone(turns.stretch(30., 31., 'A'))
        self.assertEqual(turns.stretch(15., 15.5, 'A', reach=1.), (14.5, 16.5))

    def test_borrow_needs_the_same_label_close_by(self):
        ordered = [line('L1', 0., 1., 'A'), line('L2', 1.5, 1.7, 'A'), line('L3', 2., 2.2, 'B'), line('L4', 6., 6.2, 'A')]
        got = R.borrow(ordered, {'L1': 'Mai'})
        self.assertEqual(got, {'L2': 'Mai'})

    def test_confidence_comes_from_agreement(self):
        self.assertEqual(R.confidence('Mai', 'Mai', 'Mai'), 'high')
        self.assertEqual(R.confidence('Mai', 'Mai', 'Sakuta'), 'medium')
        self.assertEqual(R.confidence(None, 'Mai', 'Mai', 'Mai'), 'medium')
        self.assertEqual(R.confidence(None, 'Mai', 'Mai', 'Sakuta'), 'low')


class Music(unittest.TestCase):
    def grid(self, labels, start=0.):
        return [{'start': start + 5 * i, 'end': start + 5 * i + 5, 'label': x} for i, x in enumerate(labels)]

    def test_ranges_grow_over_adjacent_music_and_stop_at_speech(self):
        windows = self.grid(['speech', 'speech', 'singing', 'music', 'singing', 'singing', 'singing', 'speech'])
        out = R.extend_ranges([{'start': 22., 'end': 33., 'kind': 'ED'}], windows)
        self.assertEqual((out[0]['start'], out[0]['end']), (10., 35.))
        self.assertEqual(out[0]['grown'], [12., 2.])

    def test_music_at_the_far_edge_is_not_taken(self):
        windows = self.grid(['speech', 'music', 'music', 'singing', 'music', 'music', 'speech'])
        out = R.extend_ranges([{'start': 20., 'end': 21., 'kind': 'OP'}], windows)
        self.assertEqual((out[0]['start'], out[0]['end']), (15., 21.))

    def test_a_long_singing_run_becomes_a_range(self):
        windows = self.grid(['speech'] + ['singing'] * 5 + ['speech'] + ['singing'] * 2)
        out = R.extend_ranges([], windows)
        self.assertEqual([(r['start'], r['end'], r['source']) for r in out], [(5., 30., 'lfm3')])

    def test_lines_mostly_inside_the_mask_are_dropped(self):
        kept, dropped = R.split_by_mask([line('L1', 0., 2.), line('L2', 9., 12.)], [(10., 20.)])
        self.assertEqual(([l['id'] for l in kept], [l['id'] for l in dropped]), (['L1'], ['L2']))


class Labels(unittest.TestCase):
    def test_timed_labels_map_to_lines_and_conflicts_are_left_out(self):
        base = {1: [line('L1', 0., 1.), line('L2', 2., 3.), line('L3', 5., 6.)]}
        timed = [{'episode': 1, 'start': 0., 'end': 3.2, 'speaker': 'Mai', 'set': 'typical'},
                 {'episode': 1, 'start': 2., 'end': 3., 'speaker': 'Sakuta', 'set': 'hard'},
                 {'episode': 1, 'start': 5., 'end': 6., 'speaker': 'Rio', 'set': 'hard'}]
        got, conflicts = R.label_lines(timed, base)
        self.assertEqual(got[1], {'L1': ('Mai', 'typical'), 'L3': ('Rio', 'hard')})
        self.assertEqual(conflicts[0]['id'], 'L2')

    def test_id_labels_become_timed(self):
        base = {2: [line('L7', 3., 4.)]}
        out = R.timed_from_ids({'ep2': {'L7': 'Mai', 'L9': 'Rio'}}, base, 'round2', 'x.json')
        self.assertEqual([(o['episode'], o['start'], o['speaker']) for o in out], [(2, 3., 'Mai')])


if __name__ == '__main__':
    unittest.main()
