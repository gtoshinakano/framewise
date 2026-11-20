import json
import unittest

from app.transcript_parts import parts, within


class TranscriptPartsTests(unittest.TestCase):
    def test_crossing_line_uses_alignments_and_published_spelling(self):
        row = {'id': 'line', 'start_time': 1, 'end_time': 3, 'text': '梓川です。',
               'words_json': json.dumps([[1, 2, 'アズサガワ'], [2, 3, 'です。']])}
        self.assertEqual([p['text'] for p in parts(row)], ['梓川', 'です。'])
        self.assertEqual(within(row, 2, 3)['text'], 'です。')
        self.assertEqual(within(row, 1, 2)['text'], '梓川')

    def test_legacy_rows_keep_their_shape_and_id(self):
        row = {'id': 'word', 'start_time': 1, 'end_time': 2, 'text': 'はい。'}
        self.assertEqual(parts(row), [row])
