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
        self.assertEqual('1.000–2.000s; S1; Hero: Hello', data['transcript_text'])
        self.assertEqual(data['subtitles'], {'extracted': True, 'language': 'Portuguese', 'lines': [
            {'start_time': 0.5, 'end_time': 2.5, 'text': 'Olá', 'confidence': 'high', 'crosses_cut': True}]})
        self.assertNotIn('Olá', data['visual_evidence_text'])
        self.assertNotIn('Olá', data['transcript_text'])

    def test_missing_extraction_is_reported_as_unavailable_not_absent(self):
        data = build_input([], [], [])
        self.assertEqual(data['subtitles'], {'extracted': False, 'language': None, 'lines': []})
        self.assertIn('subtitles.extracted is false', SYSTEM)
        self.assertIn('never a contradiction', SYSTEM)


class PromptContextTest(unittest.TestCase):
    def test_context_keeps_only_what_a_prompt_needs(self):
        from app import prompt_context
        context = {'project': {'id': 'p', 'title': 'T', 'synopsis': 'S'}, 'episode': {'id': 'e', 'title': 'E', 'duration': 9},
                   'cut': {'id': 'c', 'cut_index': 1, 'start_time': 0, 'end_time': 2, 'title': '', 'results': [], 'has_boxes': True},
                   'neighbor_results': {'previous': [{'kind': 'x', 'title': 'Cut summary', 'content': 'long', 'model_id': 'm'},
                                                       {'kind': 'x', 'title': 'Consistency summary', 'content': '{"summary": "ok"}', 'model_id': 'm'}], 'next': []},
                   'transcript_segments': [{'id': 't', 'start_time': 0, 'end_time': 1, 'text': 'hi', 'speaker_label': 'S', 'character_name': 'Hero',
                                            'model_id': 'm', 'origin': 'model', 'deleted': False},
                                           {'id': 'd', 'start_time': 1, 'end_time': 2, 'text': 'gone', 'deleted': True}],
                   'character_detections': {'run_id': 'r', 'model_id': 'ccip', 'frames': [{'position': 'first', 'time': 0, 'image_size': [1, 1]}],
                                            'detections': [{'id': 'd1', 'position': 'first', 'box': [1, 2, 3, 4], 'character_name': 'Hero', 'character_context_id': 'h',
                                                            'label_source': 'auto', 'match_reason': 'CCIP similarity 0.84', 'match_model_id': 'ccip'}]}}
        slim = prompt_context.slim_context(context)
        self.assertEqual(slim['neighbor_consistency_summaries'], {'previous': {'summary': 'ok'}})
        self.assertEqual(slim['transcript_segments'], [{'start_time': 0, 'end_time': 1, 'text': 'hi', 'speaker_label': 'S', 'character_name': 'Hero'}])
        self.assertEqual(slim['cut'], {'cut_index': 1, 'start_time': 0, 'end_time': 2})
        text = repr(slim)
        for leaked in ('match_reason', 'CCIP', 'model_id', 'run_id', 'long', 'image_size'):
            self.assertNotIn(leaked, text)
        self.assertEqual(prompt_context.slim_context(slim), slim)
