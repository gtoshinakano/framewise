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


FRAME = {'composition': 'Medium shot, Mai left', 'mood': 'calm', 'props': 'none', 'mode': 'full', 'frame_time': 12.02, 'subtitle_regions': [[0, 540, 1280, 700]],
         'visible_characters': [{'character': 'Mai', 'character_context_id': 'mai-id', 'appearance': 'white shirt', 'emotion': 'calm', 'gaze': 'at Sakuta', 'action': 'standing'}]}
WINDOW = lambda start, end, analysis: {'source': {'source_start': start, 'source_end': end, 'frame_count': 30, 'fps': 7.5, 'contract': 'direction-v1'}, 'analysis': analysis, 'usage': {'prompt_tokens': 4000}}
VIDEO = {'model': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': 'abc', 'contract': 'direction-v1', 'usage': {'prompt_tokens': 8000}, 'limitations': 'Window boundaries may split events.',
         'direction': {'summary': 'Mai turns and walks off.', 'camera': {'movement': 'static', 'detail': 'none'},
                       'characters': [{'character': 'Mai', 'character_context_id': 'mai-id', 'activity': 'standing → walking', 'travel': 'exits right', 'motion': 'Turns, then walks right.',
                                       'speaking': 'yes', 'emotion': 'calm → annoyed', 'end': 'left frame'},
                                      {'character': 'Sakuta', 'character_context_id': 'sakuta-id', 'activity': 'standing', 'travel': 'none', 'motion': 'holds still',
                                       'speaking': 'no', 'emotion': 'calm', 'end': 'right, facing left', 'in_last_frame': True}],
                       'end_state': {'in_frame': ['Sakuta'], 'source': 'character boxes of the last frame'}},
         'windows': [WINDOW(12.0, 16.0, {'overview': 'Mai turns.', 'events': [{'start_seconds': 0.5, 'end_seconds': 1.5, 'subject': 'Mai', 'visual_action': 'Turns her head right.'}], 'uncertainties': 'none'}),
                     WINDOW(16.0, 19.0, {'overview': 'Mai walks off.', 'events': [{'start_seconds': 1.0, 'end_seconds': 2.5, 'subject': 'Mai', 'visual_action': 'Walks out to the right.'},
                                                                                {'start_seconds': None, 'end_seconds': None, 'subject': 'camera', 'visual_action': 'Stays fixed.'}], 'uncertainties': 'Face is small.'})]}
LAST = {'composition': 'unchanged', 'mode': 'changes', 'visible_characters': []}


class MotionDirectionEvidenceTest(unittest.TestCase):
    def results(self, *extra):
        return [{'id': 'f', 'type': 'initial_frame', 'content': FRAME}, {'id': 'v', 'type': 'video_analysis', 'content': VIDEO}, *extra]

    def test_reviewer_gets_the_still_and_one_motion_direction_without_bookkeeping(self):
        text = build_input(self.results({'id': 'l', 'type': 'last_frame', 'content': LAST}), [], [])['visual_evidence_text']
        self.assertTrue(text.startswith('First frame:'))
        for expected in ('emotion: calm', 'gaze: at Sakuta', 'movement: static', 'activity: standing → walking', 'travel: exits right', 'speaking: yes',
                         'seconds: 0.5-1.5', 'seconds: 5.0-6.5', 'seconds: unknown', 'end: left frame', 'in last frame: True', 'in frame: \n    - Sakuta', 'uncertainties: Face is small.'):
            self.assertIn(expected, text)
        for noise in ('Last frame', 'unchanged', 'frame_count', 'prompt_tokens', 'Qwen', 'subtitle regions', 'frame time', 'mai-id', 'limitations', 'source start'):
            self.assertNotIn(noise, text)  # a leftover last frame, run bookkeeping and ids are not evidence

    def test_older_analyses_are_still_sent_with_their_last_frame(self):
        old_video = {'windows': [WINDOW(12.0, 19.0, {'overview': 'Static camera.', 'events': []})]}
        text = build_input([{'type': 'initial_frame', 'content': FRAME}, {'type': 'video_analysis', 'content': old_video}, {'type': 'last_frame', 'content': LAST}], [], [])['visual_evidence_text']
        self.assertIn('Initial frame:', text)
        self.assertIn('Last frame:\ncomposition: unchanged', text)

    def test_queueing_an_older_cut_needs_a_first_frame_and_a_video_analysis_only(self):
        from app import summary_queue
        context = {'cut': {'cut_index': 0, 'start_time': 12, 'end_time': 19}, 'transcript_segments': [], 'context_items': []}
        payload = summary_queue.payload_for('c1', {}, context, self.results({'id': 'l', 'type': 'last_frame', 'content': LAST}))
        self.assertEqual(payload['source_result_ids'], ['f', 'v'])  # the leftover last frame is not a source
        with self.assertRaisesRegex(ValueError, 'Initial frame and keyframe analysis'):
            summary_queue.payload_for('c1', {}, context, self.results()[:1])
        old = [{'id': 'f', 'type': 'initial_frame', 'content': FRAME}, {'id': 'v', 'type': 'video_analysis', 'content': {'windows': []}}]
        with self.assertRaisesRegex(ValueError, 'without a last frame; run Keyframes'):
            summary_queue.payload_for('c1', {}, context, old)
        legacy = summary_queue.payload_for('c1', {}, context, [*old, {'id': 'l', 'type': 'last_frame', 'content': LAST}])
        self.assertEqual(legacy['source_result_ids'], ['f', 'v', 'l'])

    def test_reviewer_prompt_is_about_the_three_generation_steps(self):
        for expected in ('a still image of the opening frame', 'the dialogue audio', 'animates the still', 'speaking', 'inner voice', 'NOT A QUESTION',
                         'is not thereby gone', 'Never reproduce subtitle text', 'state_correction', 'at most 5', 'Colour shades'):
            self.assertIn(expected, SYSTEM)
        self.assertNotIn('ending-frame descriptions', SYSTEM)

    def test_cut_summary_inputs_are_compact_and_stable_when_slimmed_twice(self):
        import json
        from app import prompt_context
        rows = [{'id': '1', 'kind': 'first_frame', 'title': 'Initial frame description', 'content': json.dumps(FRAME), 'origin': 'model'},
                {'id': '2', 'kind': 'cut_description', 'title': 'Video analysis', 'content': json.dumps(VIDEO), 'origin': 'model'},
                {'id': '3', 'kind': 'last_frame', 'title': 'Last frame description', 'content': json.dumps(LAST), 'origin': 'model'},
                {'id': '4', 'kind': 'cut_description', 'title': 'Consistency summary', 'content': '{"summary": "ok"}', 'origin': 'model'},
                {'id': '5', 'kind': 'other', 'title': 'Note', 'content': 'hand written', 'origin': 'manual'}]
        slim = prompt_context.slim_results(rows)
        self.assertEqual([row['title'] for row in slim], ['Initial frame description', 'Video analysis', 'Consistency summary', 'Note'])
        frame, video = slim[0]['content'], slim[1]['content']
        self.assertEqual(frame['visible_characters'][0]['emotion'], 'calm')
        self.assertNotIn('subtitle_regions', frame)
        self.assertEqual(set(video), {'kind', 'summary', 'camera', 'characters', 'events', 'end_of_cut', 'uncertainties'})
        self.assertEqual([event['seconds'] for event in video['events']], ['0.5-1.5', '5.0-6.5', 'unknown'])
        self.assertEqual(video['end_of_cut'], {'in_frame': ['Sakuta'], 'source': 'character boxes of the last frame'})
        self.assertEqual([person.get('in_last_frame') for person in video['characters']], [None, True])
        self.assertNotIn('identity', video['characters'][0])  # marked characters need no caveat
        guessed = prompt_context.video_view({**VIDEO, 'direction': {**VIDEO['direction'], 'characters': [{'character': 'girl with red umbrella', 'character_context_id': None, 'speaking': 'yes'}]}})
        self.assertIn('not a marked character', guessed['characters'][0]['identity'])
        self.assertEqual(slim[2]['content'], '{"summary": "ok"}')
        self.assertEqual(slim[3], {'title': 'Note', 'content': 'hand written', 'origin': 'manual'})
        self.assertEqual(prompt_context.slim_results(slim), slim)  # the worker slims a queued snapshot again
        # A cut analyzed before motion directions keeps its last frame and its video analysis as stored.
        old = [rows[0], {**rows[1], 'content': json.dumps({'windows': [{'analysis': {'overview': 'x'}}]})}, rows[2]]
        kept = prompt_context.slim_results(old)
        self.assertEqual([row['title'] for row in kept], ['Initial frame description', 'Video analysis', 'Last frame description'])
        self.assertEqual(kept[1]['content'], {'windows': [{'analysis': {'overview': 'x'}}]})


class QuestionsTest(unittest.TestCase):
    def test_questions_are_normalised_without_rejecting_the_answer(self):
        from app.consistency_summary import MAX_QUESTIONS, normalize_questions, validate
        raw = [{'id': 'x', 'topic': 'Facing', 'character': ' Mai ', 'question': 'Where does Mai look?', 'recommended': 'B',
                'options': [{'id': 'A', 'text': 'At Rio'}, {'id': 'B', 'text': 'At the table'}, {'id': 'C', 'text': 'at rio'}, 'At the camera', {'label': 'Down'}, 'Fifth']},
               {'question': 'One option only', 'options': ['alone']}, {'question': '', 'options': ['a', 'b']}, 'junk', None,
               {'question': 'Is Rio speaking?', 'topic': 'nonsense', 'recommended': 1, 'options': ['Yes', 'No']},
               {'question': 'By text', 'recommended': 'no', 'options': ['Yes', 'No']}, {'question': 'Unknown rec', 'recommended': 'zzz', 'options': ['Yes', 'No']}]
        found = normalize_questions(raw + [{'question': f'extra {n}', 'options': ['a', 'b']} for n in range(6)])
        self.assertEqual(len(found), MAX_QUESTIONS)
        first = found[0]
        self.assertEqual((first['id'], first['topic'], first['character'], first['recommended']), ('q1', 'pose', 'Mai', 'b'))
        self.assertEqual([option['text'] for option in first['options']], ['At Rio', 'At the table', 'At the camera', 'Down'])  # duplicates dropped, capped at four
        self.assertEqual([option['id'] for option in first['options']], ['a', 'b', 'c', 'd'])
        self.assertEqual((found[1]['topic'], found[1]['recommended']), ('other', 'b'))  # a position counts from zero
        self.assertEqual(found[2]['recommended'], 'b')
        self.assertIsNone(found[3]['recommended'])
        self.assertEqual(normalize_questions('nothing'), [])
        output = validate({'summary': ' Mai sits. ', 'clear': 'Still: x', 'confusing': '', 'questions': []})
        self.assertEqual(output, {'summary': 'Mai sits.', 'clear': 'Still: x', 'confusing': 'None identified.', 'questions': []})
        with self.assertRaises(ValueError): validate({'summary': 'x'})
        with self.assertRaises(ValueError): validate({'summary': '', 'questions': []})
        self.assertNotIn('human_review', output)

    def test_the_grammar_allows_five_questions_with_two_to_four_options(self):
        from app.consistency_summary import question_format
        schema = question_format()['json_schema']['schema']
        question = schema['properties']['questions']
        self.assertEqual(question['maxItems'], 5)
        self.assertEqual((question['items']['properties']['options']['minItems'], question['items']['properties']['options']['maxItems']), (2, 4))
        self.assertNotIn('gaze', question['items']['properties']['topic']['enum'])  # settled by the reviewer, never asked
        self.assertIn('emotion', question['items']['properties']['topic']['enum'])
        settled = schema['properties']['settled']['items']['properties']
        self.assertEqual(settled['topic']['enum'], ['pose', 'gaze', 'state', 'position'])
        self.assertEqual(settled['basis']['enum'], ['keypoints', 'gaze_rule'])  # no picture without pictures

    def test_the_app_hands_the_reviewer_its_own_leads(self):
        frame = {'composition': 'MS', 'visible_characters': [{'character': 'Mai', 'pose_detail': {'body_pose': 'standing', 'body_yaw': 0, 'head_yaw': -60, 'eye_gaze': 'left', 'gaze_target': 'Rio',
                                                                                              'flags': [{'code': 'gaze_side', 'text': 'Mai looks at Rio, who is on the image right, but the head points left'}]}}]}
        direction = {'summary': 'x', 'camera': {}, 'characters': [{'character': 'Mai', 'speaking': 'no', 'state_correction': {'first_frame': 'standing', 'clip': 'walking'}},
                                                               {'character': 'Rio', 'speaking': 'yes'}]}
        results = [{'type': 'initial_frame', 'content': frame}, {'type': 'video_analysis', 'content': {'direction': direction, 'windows': []}}]
        lines = [{'start_time': 1, 'end_time': 2, 'text': 'a', 'character_name': 'Mai'}, {'start_time': 1.5, 'end_time': 3, 'text': 'b', 'character_name': 'Mai'},
                 {'start_time': 4, 'end_time': 5, 'text': 'c', 'speaker_label': 'SPEAKER_01'}, {'start_time': 6, 'end_time': 7, 'text': 'd', 'character_name': 'Kaede'}]
        data = build_input(results, lines, [], None, {'synopsis': 'A school story'})
        kinds = [check['kind'] for check in data['app_checks']]
        self.assertEqual(kinds, ['pose_gaze_side', 'state_differs', 'line_without_visible_speaking', 'unlinked_speaker_may_be'])  # one per character; an off-screen speaker is no lead
        self.assertEqual(data['app_checks'][3]['character'], 'Rio')
        self.assertEqual(data['series'], {'synopsis': 'A school story'})
        self.assertIn('pose detail', data['visual_evidence_text'])
        self.assertNotIn('app_checks', build_input([{'type': 'initial_frame', 'content': {'composition': 'x'}}], [], []))

    def test_series_context_leaves_out_characters_and_boxes(self):
        from app import prompt_context
        context = {'project': {'synopsis': 'A school story'}, 'episode': {'title': 'Ep 1'}, 'chunks': [{'title': 'Classroom scene', 'notes': 'rainy day', 'start_time': 0}],
                   'context_items': [{'id': 'c', 'kind': 'character', 'title': 'Mai', 'content': 'red tie'}, {'id': 'w', 'kind': 'world', 'title': 'School', 'content': 'a calm school'}],
                   'additional_context': ''}
        self.assertEqual(prompt_context.series_context(context), {'synopsis': 'A school story', 'episode': 'Ep 1', 'scene_notes': [{'title': 'Classroom scene', 'notes': 'rainy day'}],
                                                                  'notes': [{'kind': 'world', 'title': 'School', 'content': 'a calm school'}]})
        self.assertEqual(prompt_context.series_context({}), {})
