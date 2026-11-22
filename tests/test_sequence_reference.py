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


def directed_answer(data):
    """The fake director with a reference: it describes the frame and, like a model, ends a beat at reference_frame.beat_ends_at rounded to 0.1 s."""
    content = pipeline_tests.shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds'])
    reference = data.get('reference_frame')
    if reference:
        content['reference_line'] = SAID
        if 'beat_ends_at' in reference:
            content['beats'][0]['end'] = round(reference['beat_ends_at'], 1)
    return content


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


HERO = [{'name': 'Hero', 'character_context_id': 'hero'}]
# Replays 2026-11-22: shot 3 of "Sakuta's Excitement Before His Date", 2.044 s, keyframes at 0, 0.084, 0.209 (the chosen F3)
# and 0.334, speech from 0.42 s. Every rewrite failed on beats of 0.1-0.2 s, so the reference never reached the prompt.
LIVE = {'cut_id': 'c1', 'ordinal': 12169, 'time': 507.632, 'at': 0.209, 'frame': 'F3'}
SPEECH = [prompt_tests.line('L1', 0.42, 2.0, 'お兄ちゃん', 'Hero')]


def short(seconds=2.044):
    return prompt_tests.shot(duration=seconds)


class ReferenceTimingTest(unittest.TestCase):
    """The reference keyframe's second is a beat edge when the shot's beats allow one there, exactly on the second; never a reason to fail."""

    def answer(self, *ends):
        content = {**pipeline_tests.shot_answer(['Hero'], 2.044), 'reference_line': SAID}
        actions = [beat['action'] for beat in content['beats']]
        content['beats'] = [{'end': end, 'action': actions[index % 2]} for index, end in enumerate(ends)]
        return content

    def validate(self, ends, reference=LIVE, generated=True, seconds=2.044):
        return direction.validate_shot(self.answer(*ends), short(seconds), prompt_tests.BIBLE, HERO, None, SPEECH, generated=generated, reference=reference)

    def errors(self, ends, **kwargs):
        with self.assertRaises(ValueError) as raised:
            self.validate(ends, **kwargs)
        return str(raised.exception)

    def rows(self, normal, seconds=2.044):
        prompt, verdict, _ = sequence_prompt.build(short(seconds), normal, prompt_tests.BIBLE, HERO, None, SPEECH)
        self.assertTrue(verdict['ok'], verdict['errors'])
        return {row.split(': ', 1)[0]: row for row in h3_prompt.parse(prompt)['sections']['detailed_description'].splitlines() if row.startswith('00:')}

    def test_the_live_failure_now_lands_the_staging_exactly_on_its_second(self):
        self.assertEqual(sequence_prompt.reference_landing(0.209, 2.044), 'edge')
        # Attempt 1 and 3 of the failed job, with the sentences the model now gets back.
        first = self.errors([0.1, 0.2, 0.42, 2.04])
        self.assertIn('beat 1 ends at 0.1 s, but the shot starts at 0 s and a beat lasts more than 0.2 s: end beat 1 after 0.2 s, or fold this moment into a neighbouring beat', first)
        self.assertNotIn('beat 2', first)  # 0.2 is the reference's second, rounded: the app puts it on 0.209
        third = self.errors([0.2, 0.4, 0.5, 2.04])
        self.assertIn('beat 2 ends at 0.4 s, but the previous beat ends at 0.209 s on the reference frame and a beat lasts more than 0.2 s: end beat 2 after 0.409 s', third)
        self.assertNotIn('beat 1', third)
        self.assertNotIn('beat 3', third)
        # What the model writes once it follows them: three beats, one more than a 2-s shot has, for the reference's edge.
        normal = self.validate([0.2, 0.42, 2.04])
        self.assertEqual([beat['end'] for beat in normal['beats']], [0.209, 0.42, 2.044])
        rows = self.rows(normal)
        self.assertEqual(list(rows), ['00:00.000-00:00.209', '00:00.209-00:00.420', '00:00.420-00:02.044', '00:02.044-00:05.000'])
        self.assertIn('The staging at 00:00.209 matches <Picture 3>', rows['00:00.000-00:00.209'])  # the beat that arrives there
        self.assertNotIn('<Picture 3>', rows['00:00.209-00:00.420'] + rows['00:00.420-00:02.044'] + rows['00:02.044-00:05.000'])
        self.assertIn('At 00:00.420, <Subject 1> (S1) says', rows['00:00.420-00:02.044'])  # speech keeps its own second
        self.assertIn('3 beats are too many for 2.044 s; write at most 2', self.errors([0.21, 0.42, 2.04], reference=None))

    def test_only_a_rounded_end_is_moved_and_a_missing_edge_sends_the_director_back(self):
        self.assertEqual([beat['end'] for beat in self.validate([0.25, 2.04])['beats']], [0.209, 2.044])  # 41 ms: about one frame
        self.assertIn('end one beat exactly at 0.209 s (reference_frame.beat_ends_at)', self.errors([0.3, 2.04]))
        self.assertIn('end one beat exactly at 0.209 s', self.errors([2.04]))  # the last beat ends at the shot's length, never on the reference
        # The user's own timing is never moved: a hand edit keeps 0.3, and the sentence goes to the beat the second falls in.
        edited = self.validate([0.3, 2.04], generated=False)
        self.assertEqual([beat['end'] for beat in edited['beats']], [0.3, 2.044])
        self.assertIn('The staging at 00:00.209 matches <Picture 3>', self.rows(edited)['00:00.000-00:00.300'])

    def test_a_second_too_close_to_the_start_or_the_end_is_reached_by_the_first_or_last_beat(self):
        opening = {**LIVE, 'at': 0.15}
        self.assertEqual(sequence_prompt.reference_landing(0.15, 2.044), 'opening')
        rows = self.rows(self.validate([0.42, 2.04], reference=opening))  # no edge demanded, no extra beat
        self.assertIn('The staging at 00:00.150 matches <Picture 3>', rows['00:00.000-00:00.420'])
        self.assertIn('3 beats are too many for 2.044 s; write at most 2', self.errors([0.42, 1.0, 2.04], reference=opening))
        end = {**LIVE, 'at': 1.95}
        self.assertEqual(sequence_prompt.reference_landing(1.95, 2.044), 'end')
        rows = self.rows(self.validate([0.42, 2.04], reference=end))
        self.assertIn('The staging at 00:01.950 matches <Picture 3>', rows['00:00.420-00:02.044'])
        self.assertNotIn('<Picture 3>', rows['00:02.044-00:05.000'])  # never the generated hold

    def test_an_edge_landing_is_exactly_a_beat_end_the_validator_accepts(self):
        for seconds in (0.3, 1.0, 2.044, 8.0):
            gap = sequence_prompt.beat_gap(seconds)
            for at in (gap - 0.001, gap, gap + 0.001, seconds - gap - 0.001, seconds - gap, seconds - gap + 0.001):
                with self.subTest(seconds=seconds, at=at):
                    try:
                        self.validate([at, seconds], reference={**LIVE, 'at': at}, generated=False, seconds=seconds)
                        refused = False
                    except ValueError as exc:
                        refused = 'a beat lasts more than' in str(exc)
                    self.assertEqual(sequence_prompt.reference_landing(at, seconds) == 'edge', not refused)


class ReferenceFlowTest(pipeline_tests.SequenceBase):
    def setUp(self):
        super().setUp()
        patcher = patch('app.sequence_reference.selection', side_effect=fake_selection)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.answers['shot'] = directed_answer

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
        self.assertEqual(sent['reference_frame'], {'frame': 'F2', 'at': 2.0, 'beat_ends_at': 2.0, 'chosen_by': 'user', 'shows': 'not analysed: no keyframe states or boxes for this frame',
                                                   'picture_attached': False})
        self.assertEqual(sent['shot']['beats'], {'at_most': 6, 'each_longer_than': 0.2})  # 5 for 6 s, and one more for the reference's edge
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

    def test_the_reference_second_follows_the_shot_clock_when_the_shot_starts_before_or_after_its_cut(self):
        keyframes = {'contract': 'keyframes-v1', 'cut': {'start_time': 2, 'end_time': 4}, 'summary': '',
                     'frames': [{'label': 'F1', 'ordinal': 48, 't': 0.0, 'characters': [{'name': 'Hero', 'state': 'standing, arms down'}]},
                                {'label': 'F2', 'ordinal': 77, 't': 1.23, 'characters': [{'name': 'Hero', 'state': 'standing, right hand raised'}]}],
                     'segments': [{'from': 'F1', 'to': 'F2', 'start': 0.0, 'end': 1.23, 'camera': {'move': 'static'}, 'other_changes': [],
                                   'characters': [{'name': 'Hero', 'changes': [{'part': 'right arm', 'description': 'rises', 'source': 'dwpose'}]}]}]}
        beats = {'contract': 'beats-v1', 'based_on': {'keyframe_analysis': 'kf'},  # a cut with keyframes is ready once its beats are current
                 'beats': [{'start': 0, 'end': 1.23, 'intent': 'Greets.', 'characters': [{'character': 'Hero', 'body': 'The right arm rises.', 'face': 'Calm.', 'gaze': 'Ahead.', 'mouth': 'Closed.'}]}]}
        with db.connect() as conn:
            for identifier, kind, content in (('kf', 'keyframe_analysis', keyframes), ('pb', 'performance_beats', beats)):
                conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json)
                    VALUES (?,'c2','cut_description',?,?,'model',?)""", (identifier, kind, json.dumps(content), json.dumps({'structured_type': kind})))
        sequence_id = self.directed()
        # Cut 2 runs 2-4 s and F2 is 3.23 s of the episode. A boundary placed at 2.5 s starts its shot 0.5 s into the cut;
        # one at 1.6 s starts it 0.4 s before the cut, on cut 1's audio. Either way the staging lands where its sound plays.
        with patch.dict(FRAMES, {'c2': [('F1', 48, 2.0), ('F2', 77, 3.23)]}):
            for point, offered, at, plays_at in ((2.5, [('F2', 0.73)], 0.73, [0.0, 1.5]), (1.6, [('F1', 0.4), ('F2', 1.63)], 1.63, [0.4, 2.4])):
                with self.subTest(point=point):
                    response = self.client.put(f'/api/sequences/{sequence_id}/shot-points', json={'points': [{'after_cut_id': 'c1', 'time': point}]})
                    self.assertEqual(response.status_code, 200, response.text)
                    shot = response.json()['shots'][1]
                    self.assertEqual((shot['cut'], shot['start_time'], shot['end_time']), (2, point, 4.0))
                    listing = self.client.get(f"/api/sequences/{sequence_id}/shots/{shot['key']}/keyframes").json()
                    self.assertEqual([(item['frame'], item['at']) for item in listing['frames']], offered)  # only the shot's own picture
                    self.assertEqual({item['lands'] for item in listing['frames']}, {'edge'})
                    self.rewrite(sequence_id, shot['key'], reference_frame={'ordinal': 77})
                    sent = self.calls[0][1]
                    own = next(cut for cut in sent['cuts'] if cut['cut'] == 2)
                    state = next(item for item in own['keyframes']['states'] if item['frame'] == 'F2')
                    # One clock for the reference, the keyframe state of the same frame, the cut's picture and the beats.
                    self.assertEqual((sent['reference_frame']['at'], sent['reference_frame']['beat_ends_at'], state['t'], own['plays_at']), (at, at, at, plays_at))
                    self.assertEqual(sent['shot']['beats'], {'at_most': 3, 'each_longer_than': 0.2})
                    result = self.detail(sequence_id)['shots'][1]['result']
                    self.assertEqual((result['content']['reference_frame']['time'], result['content']['reference_frame']['at']), (3.23, at))
                    self.assertEqual(result['content']['beats'][0]['end'], at)  # the director wrote it rounded; the app put it on the frame's second
                    row = next(line for line in result['prompt'].splitlines() if line.startswith(f'00:00.000-{h3_prompt.fmt(at)}:'))
                    self.assertIn(f'The staging at {h3_prompt.fmt(at)} matches <Picture', row)
                    self.assertTrue(result['check']['ok'], result['check'])

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
            self.answers['shot'] = directed_answer
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
            self.answers['shot'] = directed_answer
            self.run_step(sequence_id, 'shots', shot_keys=[key], reference_frame={'ordinal': 144})
            self.work()
        with patch('app.frame_cache.render', side_effect=RuntimeError('That frame is not inside this cut')):
            files, manifest = self.export(sequence_id)
        self.assertNotIn('shot-03-reference.jpg', files)
        self.assertIn({'asset': 'reference_frame', 'shot': 3, 'path': 'shot-03-reference.jpg', 'error': 'That frame is not inside this cut'}, manifest['errors'])


if __name__ == '__main__':
    unittest.main()
