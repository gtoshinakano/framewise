import unittest
from app.consistency_summary import SYSTEM, build_input


class SummaryInputTest(unittest.TestCase):
    def test_preserves_transcript_speaker_and_subtitles_as_separate_evidence(self):
        results = [{'type': 'initial_frame', 'content': {'composition': 'wide'}}]
        transcript = [{'start_time': 1, 'end_time': 2, 'speaker_label': 'S1', 'character_context_id': 'ch', 'text': 'Hello', 'origin': 'model'}]
        subtitles = {'extracted': True, 'language': 'Portuguese', 'strip': {'top': .8, 'bottom': .95},
                     'lines': [{'id': 'l1', 'start_time': 0.5, 'end_time': 2.5, 'text': 'Olá', 'confidence': 'high', 'crosses_cut': True}]}
        data = build_input(results, transcript, [{'id': 'ch', 'title': 'Hero', 'kind': 'character'}], subtitles)
        self.assertIn('composition: wide', data['visual_evidence_text'])
        self.assertIn('S1; linked character Hero (ch)', data['transcript_text'])
        self.assertEqual(data['subtitles'], {'extracted': True, 'language': 'Portuguese', 'lines': [
            {'start_time': 0.5, 'end_time': 2.5, 'text': 'Olá', 'confidence': 'high', 'crosses_cut': True}]})
        self.assertNotIn('Olá', data['visual_evidence_text'])
        self.assertNotIn('Olá', data['transcript_text'])

    def test_missing_extraction_is_reported_as_unavailable_not_absent(self):
        data = build_input([], [], [])
        self.assertEqual(data['subtitles'], {'extracted': False, 'language': None, 'lines': []})
        self.assertIn('subtitles.extracted is false', SYSTEM)
        self.assertIn('never a contradiction', SYSTEM)
