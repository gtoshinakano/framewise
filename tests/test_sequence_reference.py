"""The reference keyframe of a shot (2026-11-22): chosen on the Shots tab, kept by later rewrites, sent to the director with
its analysis (and as a picture when the engine can see), added to the H3 prompt for staging only, and exported in the ZIP."""
import json
import re
import unittest
from unittest.mock import patch

import tests.test_sequence_export as export_tests
import tests.test_sequence_pipeline as pipeline_tests
import tests.test_sequence_prompt as prompt_tests
from app import db, media, sequence_direction as direction, sequence_pipeline, sequence_prompt, sequence_reference
from app.assistant import h3_prompt, images

REF = {'cut_id': 'c1', 'ordinal': 72, 'time': 3.0, 'at': 3.0, 'frame': 'F2'}
LINE = 'Mai Sakurajima stands screen-left in the foreground facing Sakuta Azusagawa, who stands screen-right in the background with a bag in hand.'
# Shot 3 of the pipeline fixtures is cut c3 (4-10 s). F4 lies past its end, so no shot shows it.
FRAMES = {'c1': [('F1', 0, 0.0), ('F2', 24, 1.0)], 'c2': [('F1', 48, 2.0)], 'c3': [('F1', 96, 4.0), ('F2', 144, 6.0), ('F3', 216, 9.0), ('F4', 300, 12.5)]}
SAID = 'Hero stands at screen-centre in the midground with the right hand raised at shoulder height, looking at the camera.'


def fake_selection(conn, cut_id):
    cut = db.one('SELECT id,start_time,end_time FROM cuts WHERE id=? AND active=1', (cut_id,), conn=conn)
    if cut is None:
        return None, None, []
    return cut, f'key-{cut_id}', [{'frame': label, 'ordinal': ordinal, 'time': time} for label, ordinal, time in FRAMES.get(cut_id, [])]


class ReferenceRenderTest(unittest.TestCase):
    def build(self, reference=REF, **change):
        content = {**prompt_tests.answer(), 'reference_line': LINE, **change}
        normal = direction.validate_shot(content, prompt_tests.shot(), prompt_tests.BIBLE, prompt_tests.ROSTER, prompt_tests.SCENARIO, prompt_tests.DIALOGUE, reference=reference)
        return normal, *sequence_prompt.build(prompt_tests.shot(), normal, prompt_tests.BIBLE, prompt_tests.ROSTER, prompt_tests.SCENARIO, prompt_tests.DIALOGUE)

    def test_the_keyframe_is_the_last_picture_and_lends_only_its_staging(self):
        normal, prompt, verdict, derived = self.build()
        self.assertTrue(verdict['ok'], verdict['errors'])
        self.assertEqual(verdict['stats']['pictures'], [1, 2, 3, 4, 5])  # initial frame, two sheets, the scenario, then the reference
        self.assertEqual([(item['label'], item['role']) for item in derived['references']][-2:], [('<Picture 5>', 'reference_frame'), ('<Audio 1>', 'original_audio')])
        self.assertEqual({key: derived['references'][-2][key] for key in ('ordinal', 'time', 'at', 'frame')}, {'ordinal': 72, 'time': 3.0, 'at': 3.0, 'frame': 'F2'})
        self.assertEqual(normal['reference_frame'], REF)
        sections = h3_prompt.parse(prompt)['sections']
        definitions = sections['subject_definitions'].splitlines()
        self.assertEqual(definitions[-2], '<Picture 5> is a frame of this same scene at 00:03.000, a staging reference only: <Subject 1> stands screen-left in the foreground facing <Subject 2>, '
                                          'who stands screen-right in the background with a bag in hand. ' + sequence_prompt.REFERENCE_USE)
        self.assertTrue(definitions[-1].startswith('<Audio 1>'))
        self.assertEqual(sections['retention_analysis'].splitlines()[-1], '<Picture 5> (staging reference at 00:03.000): partially_preserved - only the placement, poses, facing, '
                                                                          'expressions and the objects present at that moment; none of its look is carried over.')
        first, second = [row for row in sections['detailed_description'].splitlines() if row.startswith('00:')]
        self.assertIn('The staging at 00:03.000 matches <Picture 5>', first)  # second 3 lies in the first beat (0-4 s)
        self.assertNotIn('<Picture 5>', second)
        self.assertEqual(h3_prompt.style_words(prompt), [])
        self.assertIsNone(re.search(r'\bcuts?\b', prompt, re.IGNORECASE))
        self.assertEqual(h3_prompt.misplaced_times(sections['detailed_description']), [])  # "at" is not a spoken line's "At"
        template, _ = sequence_prompt.template(prompt_tests.shot(), normal, prompt_tests.BIBLE, derived['subjects'], prompt_tests.DIALOGUE, prompt)
        self.assertIsNotNone(template)  # the template still gives the prompt back
        self.assertEqual(sequence_prompt.still_prompt(normal, prompt_tests.BIBLE), sequence_prompt.still_prompt(prompt_tests.answer(), prompt_tests.BIBLE))  # the still is the opening frame

    def test_a_reference_on_the_last_instant_goes_to_the_last_acted_beat(self):
        _, prompt, verdict, _ = self.build({**REF, 'at': 8.0})
        self.assertTrue(verdict['ok'], verdict['errors'])
        last = [row for row in h3_prompt.parse(prompt)['sections']['detailed_description'].splitlines() if row.startswith('00:')][-1]
        self.assertIn('The staging at 00:08.000 matches <Picture 5>', last)

    def test_without_a_reference_nothing_changes_and_the_json_cannot_set_one(self):
        plain = direction.validate_shot(prompt_tests.answer(), prompt_tests.shot(), prompt_tests.BIBLE, prompt_tests.ROSTER, prompt_tests.SCENARIO, prompt_tests.DIALOGUE)
        typed, prompt, _, derived = self.build(reference=None, reference_frame=REF)
        self.assertNotIn('reference_frame', typed)
        self.assertNotIn('reference_line', typed)
        self.assertEqual(typed, plain)
        self.assertEqual(prompt, sequence_prompt.build(prompt_tests.shot(), plain, prompt_tests.BIBLE, prompt_tests.ROSTER, prompt_tests.SCENARIO, prompt_tests.DIALOGUE)[0])
        self.assertNotIn('reference_frame', [item['role'] for item in derived['references']])

    def test_a_model_answer_with_a_reference_describes_it_without_labels_times_or_style(self):
        def errors(line):
            content = {**prompt_tests.answer(), 'reference_line': line}
            with self.assertRaises(ValueError) as raised:
                direction.validate_shot(content, prompt_tests.shot(), prompt_tests.BIBLE, prompt_tests.ROSTER, prompt_tests.SCENARIO, prompt_tests.DIALOGUE, generated=True, reference=REF)
            return str(raised.exception)
        self.assertIn('"reference_line" is empty', errors(''))
        self.assertIn('reference_line contains a timecode', errors('At 00:03.000 Mai stands screen-left.'))
        self.assertIn('reference_line contains a reference label', errors('<Subject 1> stands screen-left.'))
        self.assertIn('reference_line names a visual style or medium (anime)', errors('Mai stands screen-left in anime proportions.'))
        self.assertIn('reference_line contains the word "cut"', errors('Mai stands screen-left as in the cut.'))
        schema = direction.shot_format(prompt_tests.shot(), prompt_tests.ROSTER, prompt_tests.DIALOGUE, reference=True)['json_schema']['schema']
        self.assertIn('reference_line', schema['required'])
        self.assertNotIn('reference_line', direction.shot_format(prompt_tests.shot(), prompt_tests.ROSTER, prompt_tests.DIALOGUE)['json_schema']['schema']['properties'])

    def test_the_wording_names_no_style_and_keeps_the_look_on_the_opening_still(self):
        texts = (sequence_prompt.REFERENCE_USE, sequence_prompt.reference_definition(5, REF, '', []), sequence_prompt.reference_retention(5, REF),
                 sequence_prompt.reference_beat(5, REF), direction.REFERENCE_PICTURE_NOTE)
        for text in texts:
            self.assertEqual(h3_prompt.style_words(text), [], text)
            self.assertIsNone(re.search(r'\bcuts?\b', text, re.IGNORECASE), text)
        self.assertIsNone(h3_prompt.SPOKEN_AT.search(sequence_prompt.reference_beat(5, REF)))
        self.assertIn('the look of <Picture 1>', sequence_prompt.REFERENCE_USE)
        for words in ('poses', 'expressions', 'layout of the setting', 'objects present', 'proportions'):
            self.assertIn(words, sequence_prompt.REFERENCE_USE)
        self.assertEqual(sequence_prompt.reference_definition(5, REF, '', []).split(': ')[0], '<Picture 5> is a frame of this same scene at 00:03.000, a staging reference only. Use it only for where each person stands, their poses, facing, gestures and expressions, the layout of the setting and the objects present. None of its look carries over (rendering, outlines, colours, shading, light, proportions)')
        for words in ('## REFERENCE FRAME', 'reference_line', 'reference_frame'):
            self.assertIn(words, direction.SHOT_SYSTEM)


class ReferenceFlowTest(pipeline_tests.SequenceBase):
    def setUp(self):
        super().setUp()
        patcher = patch('app.sequence_reference.selection', side_effect=fake_selection)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.answers['shot'] = lambda data: {**pipeline_tests.shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds']),
                                             **({'reference_line': SAID} if 'reference_frame' in data else {})}

    def rewrite(self, sequence_id, key, expected=200, **body):
        self.calls.clear()
        response = self.run_step(sequence_id, 'shots', expected, shot_keys=[key], **body)
        if expected == 200:
            self.assertEqual(self.work(), 1)
        return response

    def third(self, sequence_id):
        return self.detail(sequence_id)['shots'][2]

    def test_a_chosen_keyframe_stays_with_the_shot_until_it_is_cleared(self):
        sequence_id = self.written()
        shots = self.detail(sequence_id)['shots']
        key = shots[2]['key']
        listing = self.client.get(f'/api/sequences/{sequence_id}/shots/{key}/keyframes').json()
        self.assertEqual([(item['frame'], item['ordinal'], item['at'], item['analysed']) for item in listing['frames']],
                         [('F1', 96, 0.0, False), ('F2', 144, 2.0, False), ('F3', 216, 5.0, False)])  # F4 lies past the shot
        self.assertEqual((listing['cut_id'], listing['revision'], listing['status'], listing['current']), ('c3', 'key-c3', 'current', None))
        # Refused: another frame, a frame the shot does not show, more than one shot, or no shot at all.
        self.assertIn('is not one of the current keyframes of cut 3', self.rewrite(sequence_id, key, 422, reference_frame={'ordinal': 999})['detail'])
        self.assertIn('F4 of cut 3 is outside shot 3', self.rewrite(sequence_id, key, 422, reference_frame={'ordinal': 300})['detail'])
        self.assertEqual(self.run_step(sequence_id, 'shots', 422, shot_keys=[key, shots[0]['key']], reference_frame={'ordinal': 144})['detail'], 'A reference keyframe is chosen for one shot at a time')
        self.run_step(sequence_id, 'shots', 422, reference_frame={'ordinal': 144})
        # Chosen: the director gets its moment and analysis, the prompt gets one more picture.
        self.rewrite(sequence_id, key, reference_frame={'ordinal': 144})
        sent = self.calls[0][1]
        self.assertEqual(sent['reference_frame'], {'frame': 'F2', 'at': 2.0, 'chosen_by': 'user', 'shows': 'not analysed: no keyframe states or boxes for this frame', 'picture_attached': False})
        result = self.third(sequence_id)['result']
        self.assertEqual(result['content']['reference_frame'], {'cut_id': 'c3', 'ordinal': 144, 'time': 6.0, 'at': 2.0, 'frame': 'F2'})
        self.assertEqual(result['content']['reference_line'], SAID)
        self.assertEqual((result['reference_frame_source'], result['reference_picture']), ('user', 'the text engine cannot read images'))
        roles = [item['role'] for item in result['derived']['references']]
        self.assertEqual(roles[-2:], ['reference_frame', 'original_audio'])
        number = len(roles) - 1
        self.assertIn(f'<Picture {number}> is a frame of this same scene at 00:02.000, a staging reference only: <Subject 1> stands at screen-centre', result['prompt'])
        self.assertIn(f'The staging at 00:02.000 matches <Picture {number}>', result['prompt'])
        self.assertTrue(result['check']['ok'], result['check'])
        self.assertEqual(h3_prompt.style_words(result['prompt']), [])
        self.assertEqual(self.client.get(f'/api/sequences/{sequence_id}/shots/{key}/keyframes').json()['current']['ordinal'], 144)
        # A plain rewrite and a batch rewrite keep it.
        self.rewrite(sequence_id, key)
        self.assertEqual(self.calls[0][1]['reference_frame']['frame'], 'F2')
        self.assertEqual(self.third(sequence_id)['result']['reference_frame_source'], 'kept')
        self.calls.clear()
        self.run_step(sequence_id, 'shots', only_missing=False)
        self.work()
        sent = {data['shot']['number']: data for step, data in self.calls if step == 'shot'}
        self.assertEqual(sent[3]['reference_frame']['frame'], 'F2')
        self.assertNotIn('reference_frame', sent[1])  # other shots have none
        self.assertEqual(self.third(sequence_id)['result']['content']['reference_frame']['ordinal'], 144)
        # Cleared: the next version has no reference, and later rewrites stay without one.
        self.rewrite(sequence_id, key, reference_frame=None)
        self.assertNotIn('reference_frame', self.calls[0][1])
        result = self.third(sequence_id)['result']
        self.assertNotIn('reference_frame', result['content'])
        self.assertNotIn('reference_line', result['content'])
        self.assertEqual(result['reference_frame_source'], 'cleared')
        self.assertNotIn('reference_frame', [item['role'] for item in result['derived']['references']])
        self.rewrite(sequence_id, key)
        self.assertNotIn('reference_frame', self.calls[0][1])
        history = self.client.get(f'/api/sequences/{sequence_id}/shots/{key}/history').json()
        self.assertEqual([(row['reference_source'], (row['reference'] or {}).get('frame')) for row in history['revisions']][:5],
                         [(None, None), ('cleared', None), ('kept', 'F2'), ('kept', 'F2'), ('user', 'F2')])

    def test_the_director_sees_the_keyframe_when_the_engine_can_read_images(self):
        sequence_id = self.written()
        key = self.third(sequence_id)['key']
        seen, original = [], self.fake_chat

        def chat(model, system, evidence, fields, **kwargs):
            seen.append((system, kwargs.get('images'), json.loads(evidence)))
            return original(model, system.replace(direction.REFERENCE_PICTURE_NOTE, ''), evidence, fields, **kwargs)

        with patch.object(self, 'fake_chat', chat), patch('app.inference_worker.endpoint_vision', return_value=True), \
                patch('app.assistant.images.keyframe_picture', return_value=b'jpeg') as cut:
            self.rewrite(sequence_id, key, reference_frame={'ordinal': 144})
            system, pictures, evidence = seen[-1]
            self.assertTrue(system.endswith(direction.REFERENCE_PICTURE_NOTE))
            self.assertEqual(pictures, [(sequence_pipeline.REFERENCE_CAPTION, b'jpeg')])
            self.assertTrue(evidence['reference_frame']['picture_attached'])
            cut.assert_called_with('c3', 144, 6.0)
            self.assertEqual(self.third(sequence_id)['result']['reference_picture'], 'attached')
            # A frame that cannot be cut never fails the shot: it is written from the text alone.
            cut.side_effect = images.FrameError('The original video of this cut is not available')
            self.rewrite(sequence_id, key)
            system, pictures, evidence = seen[-1]
            self.assertEqual((pictures, evidence['reference_frame']['picture_attached'], system), (None, False, direction.SHOT_SYSTEM))
            result = self.third(sequence_id)['result']
            self.assertTrue(result['reference_picture'].startswith('the frame could not be cut'))
            self.assertEqual(result['content']['reference_frame']['ordinal'], 144)
        # Without a reference no picture goes along, whatever the engine.
        self.rewrite(sequence_id, key, reference_frame=None)
        self.assertNotIn('reference_frame', self.calls[0][1])

    def test_hand_and_assistant_edits_keep_the_reference_the_user_chose(self):
        sequence_id = self.written()
        key = self.third(sequence_id)['key']
        self.rewrite(sequence_id, key, reference_frame={'ordinal': 144})
        result = self.third(sequence_id)['result']
        moved = {**result['content'], 'reference_frame': {'cut_id': 'c3', 'ordinal': 216, 'time': 9.0, 'at': 5.0, 'frame': 'F3'}, 'reference_line': 'Hero stands alone at screen-centre.'}
        saved = self.client.put(f"/api/sequence-results/{result['id']}", json={'content': moved})
        self.assertEqual(saved.status_code, 200, saved.text)
        edited = saved.json()['shots'][2]['result']
        self.assertEqual((edited['content']['reference_frame']['ordinal'], edited['content']['reference_line']), (144, 'Hero stands alone at screen-centre.'))
        dropped = {name: value for name, value in edited['content'].items() if name != 'reference_frame'}
        saved = self.client.put(f"/api/sequence-results/{edited['id']}", json={'content': dropped})
        self.assertEqual(saved.json()['shots'][2]['result']['content']['reference_frame']['ordinal'], 144)
        styled = {**dropped, 'reference_line': 'Hero stands at screen-centre, drawn in a cartoon look.'}
        refused = self.client.put(f"/api/sequence-results/{saved.json()['shots'][2]['result']['id']}", json={'content': styled})
        self.assertEqual(refused.status_code, 422)
        self.assertIn('reference_line names a visual style or medium (cartoon)', refused.json()['detail'])

    def test_the_director_reads_what_the_analysis_found_in_that_frame(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('t','c3','key-c3',4,10,?)",
                         (json.dumps([{'ordinal': 144, 'image_size': [1000, 500], 'relative_time': 2.0}]),))
            conn.execute("INSERT INTO target_slots(run_id,slot,origin,character_context_id) VALUES ('t',0,'first_frame','ctx'),('t',1,'appeared',NULL),('t',2,'appeared',NULL)")
            conn.execute("UPDATE target_slots SET ignored=1 WHERE run_id='t' AND slot=2")
            conn.executemany("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json) VALUES (?,'t',?,?,?)",
                             [('d1', 144, 0, '[100,50,300,450]'), ('d2', 144, 1, '[700,200,800,300]'), ('d3', 144, 2, '[0,0,10,10]'), ('d4', 96, 0, '[0,0,500,500]')])
            content = {'contract': 'keyframes-v1', 'method': 'hybrid',
                       'frames': [{'label': 'F1', 'ordinal': 96, 't': 0.0, 'characters': []},
                                  {'label': 'F2', 'ordinal': 144, 't': 2.0, 'characters': [{'name': 'Hero', 'state': 'standing, right hand raised', 'pose': 'right arm raised above the shoulder'}]}],
                       'segments': [{'from': 'F1', 'to': 'F2', 'start': 0.0, 'end': 2.0, 'camera': {'move': 'static'}, 'other_changes': [],
                                     'characters': [{'name': 'Hero', 'changes': [{'part': 'right arm', 'description': 'rises to shoulder height', 'source': 'dwpose'}]}]},
                                    {'from': 'F2', 'to': 'F3', 'start': 2.0, 'end': 5.0, 'camera': {'move': 'static'}, 'characters': [],
                                     'other_changes': [{'kind': 'object', 'description': 'a door closes'}]}], 'summary': ''}
            shows = sequence_reference.evidence(conn, {'keyframe_analysis': {'id': 'k', 'content': content}}, {'cut_id': 'c3', 'ordinal': 144})
        self.assertEqual(shows['people'], [{'name': 'Hero', 'state': 'standing, right hand raised', 'body_measured': 'right arm raised above the shoulder', 'position': 'screen-left', 'height_in_frame': 0.8},
                                           {'name': 'unnamed person', 'position': 'screen-right', 'height_in_frame': 0.2}])  # an ignored colour is left out
        self.assertEqual(shows['changes_into_this_frame'][0]['from'], 'F1')
        self.assertEqual(shows['changes_into_this_frame'][0]['characters'][0]['changes'], ['right arm: rises to shoulder height (keypoints)'])
        self.assertEqual((shows['changes_after_this_frame'][0]['to'], shows['changes_after_this_frame'][0]['other']), ('F3', ['object: a door closes']))
        self.assertEqual(sequence_reference.analysed_ordinals({'keyframe_analysis': {'content': content}}), {96, 144})

    def test_a_kept_reference_is_dropped_once_the_shot_no_longer_shows_it(self):
        shot = {'cut_id': 'c3', 'cut': 3, 'index': 3, 'start_time': 4.0, 'end_time': 10.0}
        with db.connect() as conn:
            kept, source = sequence_reference.resolve(conn, shot, None, {'reference_frame': {'cut_id': 'c3', 'ordinal': 144, 'time': 6.0, 'at': 2.0, 'frame': 'F9'}})
            self.assertEqual((kept['frame'], kept['at'], source), ('F2', 2.0, 'kept'))  # the label follows the current selection
            self.assertEqual(sequence_reference.resolve(conn, {**shot, 'start_time': 7.0}, None, {'reference_frame': kept}), (None, 'dropped: the reference keyframe is no longer inside this shot'))
            self.assertEqual(sequence_reference.resolve(conn, shot, None, {'reference_frame': {**kept, 'cut_id': 'c2'}}), (None, 'dropped: the reference keyframe belongs to another cut'))
            self.assertEqual(sequence_reference.resolve(conn, shot, sequence_reference.CLEAR, {'reference_frame': kept}), (None, 'cleared'))
            self.assertEqual(sequence_reference.resolve(conn, shot, None, {}), (None, None))


class ReferenceExportTest(pipeline_tests.SequenceBase):
    # The export helper only; subclassing SequenceExportTest would run all of its tests again here.
    export = export_tests.SequenceExportTest.export

    def setUp(self):
        super().setUp()
        folder = media.episode_dir('p', 'e')
        export_tests.write_audio(folder / 'audio.wav')
        (folder / 'original.mp4').write_bytes(b'video')
        self.frames, self.tiles, self.crops, self.clips = [], [], [], []

    def test_the_zip_holds_the_exact_reference_frame_and_names_its_picture(self):
        with patch('app.sequence_reference.selection', side_effect=fake_selection):
            sequence_id = self.written()
            key = self.detail(sequence_id)['shots'][2]['key']
            self.answers['shot'] = lambda data: {**pipeline_tests.shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds']), 'reference_line': SAID}
            self.run_step(sequence_id, 'shots', shot_keys=[key], reference_frame={'ordinal': 144})
            self.assertEqual(self.work(), 1)
        rendered = []

        def render(spec):
            rendered.append({name: spec[name] for name in ('time', 'start', 'end', 'width', 'ordinal')})
            return b'exact frame'

        with patch('app.frame_cache.render', side_effect=render):
            files, manifest = self.export(sequence_id)
        self.assertEqual(files['shot-03-reference.jpg'], b'exact frame')
        self.assertNotIn('shot-01-reference.jpg', files)
        self.assertEqual(rendered, [{'time': 6.0, 'start': 4, 'end': 10, 'width': 8192, 'ordinal': 144}])  # by frame number, at the original size
        item = next(entry for entry in manifest['files'] if entry['path'] == 'shot-03-reference.jpg')
        self.assertEqual((item['kind'], item['at'], item['frame'], item['time']), ('reference_frame', 2.0, 'F2', 6.0))
        lean = json.loads(files['sequence.json'])
        pictures = lean['shots'][2]['pictures']
        self.assertEqual(pictures[-1], {'role': 'reference_frame', 'at': 2.0, 'frame': 'F2'})
        self.assertNotIn('reference_frame', [picture['role'] for picture in lean['shots'][0]['pictures']])
        refs = json.loads(files['overview.json'])['shots'][2]['refs']
        self.assertIn(f'P{len(pictures)} staging reference F2 @2.00s', refs)
        unanalysed = [issue for issue in lean['issues'] if issue['kind'] == 'reference_unanalysed']
        self.assertEqual((len(unanalysed), unanalysed[0]['severity'], unanalysed[0].get('shots')), (1, 'info', [3]))

    def test_a_reference_frame_that_cannot_be_cut_is_an_error_not_a_substitute(self):
        with patch('app.sequence_reference.selection', side_effect=fake_selection):
            sequence_id = self.written()
            key = self.detail(sequence_id)['shots'][2]['key']
            self.answers['shot'] = lambda data: {**pipeline_tests.shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds']), 'reference_line': SAID}
            self.run_step(sequence_id, 'shots', shot_keys=[key], reference_frame={'ordinal': 144})
            self.work()
        with patch('app.frame_cache.render', side_effect=RuntimeError('That frame is not inside this cut')):
            files, manifest = self.export(sequence_id)
        self.assertNotIn('shot-03-reference.jpg', files)
        self.assertIn({'asset': 'reference_frame', 'shot': 3, 'path': 'shot-03-reference.jpg', 'error': 'That frame is not inside this cut'}, manifest['errors'])


if __name__ == '__main__':
    unittest.main()
