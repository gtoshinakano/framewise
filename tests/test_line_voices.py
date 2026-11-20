import unittest

from app import line_voices as R

def line(i, start, end, diar='A', text='x'):
    return {'id': i, 'start_time': start, 'end_time': end, 'diar': diar, 'text': text}


class Voice(unittest.TestCase):
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

class Music(unittest.TestCase):
    def test_lines_mostly_inside_the_mask_are_dropped(self):
        kept, dropped = R.split_by_mask([line('L1', 0., 2.), line('L2', 9., 12.)], [(10., 20.)])
        self.assertEqual(([l['id'] for l in kept], [l['id'] for l in dropped]), (['L1'], ['L2']))
