import unittest

from app.audio_matching import assign_words, match_voices, restore_aligned_text


class AudioMatchingTests(unittest.TestCase):
    def test_restores_punctuation_and_spacing_or_reports_mismatch(self):
        words = [{'text': 'Hello'}, {'text': 'world'}]
        result = restore_aligned_text('Hello, world!', words)
        self.assertEqual(''.join(word['text'] for word in result), 'Hello, world!')
        self.assertEqual([word['text'] for word in result], ['Hello,', ' world!'])
        self.assertIsNone(restore_aligned_text('Different', words))

    def test_punctuation_stays_with_its_word_and_text_is_preserved(self):
        words = [{'text': word, 'start_time': index, 'end_time': index + .5}
                 for index, word in enumerate(['Hello', 'world', 'Next', 'yes'])]
        text = '“Hello,” (world)! Next… ¿yes?'
        result = restore_aligned_text(text, words)
        self.assertEqual([word['text'] for word in result], ['“Hello,”', ' (world)!', ' Next…', ' ¿yes?'])
        self.assertEqual(''.join(word['text'] for word in result), text)
        self.assertEqual([(word['start_time'], word['end_time']) for word in result],
                         [(word['start_time'], word['end_time']) for word in words])

    def test_assign_words_uses_overlap_and_preserves_unattributed_speech(self):
        words = [{'start_time': 1., 'end_time': 2., 'text': 'hello'},
                 {'start_time': 3., 'end_time': 4., 'text': 'there'}]
        turns = [{'start_time': .8, 'end_time': 1.2, 'speaker': 'A'},
                 {'start_time': 1.2, 'end_time': 2.5, 'speaker': 'B'}]
        self.assertEqual([w['speaker'] for w in assign_words(words, turns)], ['B', None])
        with self.assertRaises(ValueError):
            assign_words([{'start_time': 2., 'end_time': 1., 'text': 'bad'}], turns)

    def test_voice_matching_requires_confidence_and_distinct_character(self):
        refs = [{'character_id': 'alice', 'embedding': [1., 0.]},
                {'character_id': 'alice', 'embedding': [0.99, 0.01]},
                {'character_id': 'bob', 'embedding': [0., 1.]}]
        result = match_voices({'A': [1., 0.], 'B': [1., 1.], 'C': [0., 0.]}, refs)
        self.assertEqual(result['A']['character_id'], 'alice')
        self.assertNotIn('B', result)
        self.assertNotIn('C', result)
