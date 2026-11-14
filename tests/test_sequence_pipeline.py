"""Sequence direction end to end: API, queued steps, revisions, staleness and hand edits (the model is faked)."""
import json
import unittest
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, sequence_direction as direction, sequence_pipeline, worker

FRAME = {'mode': 'full', 'composition': 'Medium shot at eye level of Hero in a street.', 'setting': 'A narrow shopping street at dusk', 'lighting': 'Warm low sun from the left',
         'camera': 'Static medium shot at eye level', 'mood': 'calm', 'props': 'none', 'background_people': 'none', 'notes': '',
         'visible_characters': [{'character': 'Hero', 'character_context_id': 'ctx', 'appearance': 'Green coat, short dark hair', 'position': 'center', 'pose': 'standing',
                                 'expression': 'calm', 'emotion': 'calm', 'gaze': 'ahead', 'action': 'standing still'}]}


def video(start, end):
    return {'contract': 'direction-v2',
            'direction': {'summary': 'Hero raises a hand and speaks.', 'camera': {'movement': 'static', 'detail': 'none'}, 'end_state': {},
                          'characters': [{'character': 'Hero', 'character_context_id': 'ctx', 'activity': 'standing', 'travel': 'none', 'motion': 'raises a hand',
                                          'speaking': 'yes', 'emotion': 'calm', 'end': 'center, hand raised'}]},
            'windows': [{'source': {'source_start': start, 'source_end': end},
                         'analysis': {'events': [{'start_seconds': 0.2, 'end_seconds': 1.0, 'subject': 'Hero', 'visual_action': 'raises a hand'}], 'uncertainties': 'none'}}]}


BIBLE = {'logline': 'Hero greets someone in the street.', 'synopsis': 'Hero stands in a street, greets an unseen friend and leaves.',
         'style': 'The target video is a 2D cel-shaded anime with clean ink lines and warm flat colours.', 'spoken_language': 'Japanese',
         'scenes': [{'id': 'A', 'from_cut': 1, 'to_cut': 3, 'location': 'Shopping street', 'time_of_day': 'dusk', 'lighting': 'warm low sun', 'mood': 'calm', 'scenario': None}],
         'characters': [{'name': 'hero', 'role': 'the lead', 'anchors': 'slim build, short dark hair, green wool coat', 'outfit': 'green coat over a white shirt', 'arc': 'decides to leave'}],
         'voices': [{'speaker_label': 'SPEAKER_01', 'character': 'Somebody else', 'kind': 'character', 'confidence': 'low', 'reason': 'guess'},
                    {'speaker_label': 'SPEAKER_02', 'character': 'Hero', 'kind': 'character', 'confidence': 'medium', 'reason': 'Visibly speaking while it is heard.'}],
         'arc': 'From greeting to parting.', 'pacing': 'Even and calm.', 'continuity': ['The green coat stays on.'], 'open_questions': []}


def shot_answer(lines=(), seconds=2.0):
    return {
        'subjects': [{'name': 'Hero', 'anchors': 'slim build, short dark hair, green wool coat over a white shirt', 'keep': 'face, hair and green coat follow the sheet exactly'}],
        'picture_line': 'Hero stands in the centre of a narrow street at dusk, facing the camera.',
        'keep_picture': 'composition, cel-shaded style, warm palette and low sun of the opening frame',
        'still': {'shot': 'Medium shot at eye level, cropped at the waist.', 'subjects': 'Hero in the centre, body square to the camera, arms at the sides, calm face, eyes ahead, green coat buttoned.',
                  'layout': 'Shop fronts on both sides recede to a bright gap in the background.', 'setting': 'A narrow shopping street.', 'lighting': 'Low warm sun from the left, long soft shadows.'},
        'summary': 'Hero raises a hand in greeting and speaks while the camera holds still.',
        'shot_line': 'A medium shot at eye level with Hero standing in the centre of the narrow street, shop fronts receding on both sides toward a bright gap behind.',
        'framing_lock': 'Waist to just above the head, never wider.',
        'light': 'Low warm sunlight comes from the left and lays long soft shadows across the paving while the far end of the street glows.',
        'camera': {'move': 'static', 'amplitude': 'small', 'speed': 'slow', 'detail': ''},
        'beats': [{'end': round(seconds / 2, 3), 'action': 'Hero stands with both arms at the sides and looks straight ahead, the green coat hanging still. The right hand lifts from the hip to shoulder height '
                                                           'with the palm open, and the sleeve slides back a little from the wrist. The hand stops beside the face with the fingers relaxed and slightly spread.'},
                  {'end': seconds, 'action': 'Hero keeps the hand raised and the lips move in short, even shapes as the eyes stay fixed ahead. The head dips once in a small nod and a strand of dark hair '
                                             'shifts across the forehead. The hand lowers halfway and settles at chest height, the shoulders loosen and the mouth closes in a faint smile.'}],
        'speech': [{'line': line, 'delivery': 'lip_sync', 'subject': 'hero'} for line in lines],
        'scenario_anchors': '', 'scenario_keep': '',
        'soundscape': 'Quiet street ambience with a light breeze and distant footsteps. The soft rustle of a coat sleeve as the arm lifts.',
        'continuity': {'from_previous': 'Hero stands in the same spot as before.', 'to_next': 'Hero ends with the hand at chest height, facing the camera.'},
        'open_questions': []}


class SequenceBase(unittest.TestCase):
    setUp_base = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def setUp(self):
        self.setUp_base()
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c3','e',2,4,10)")
            conn.execute("UPDATE cuts SET first_frame_time=start_time")
            conn.execute("UPDATE episodes SET duration=10,has_audio=1")
            for cut, start, end in (('c1', 0, 2), ('c2', 2, 4), ('c3', 4, 10)):
                self.analyse(conn, cut, start, end)
            for row in (('t1', 0.4, 1.2, 'こんにちは', 'SPEAKER_01', 'ctx'), ('t2', 2.5, 3.5, 'はい', 'SPEAKER_02', None), ('t3', 5.0, 6.0, 'じゃあね', 'SPEAKER_02', None)):
                conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin) VALUES (?,'e',?,?,?,?,?,'model')", row)
        self.calls, self.answers = [], {}

    def analyse(self, conn, cut, start, end):
        for kind, row_kind, title, content in (('initial_frame', 'first_frame', 'Initial frame description', FRAME), ('video_analysis', 'cut_description', 'Video analysis', video(start, end))):
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,model_id,settings_json) VALUES (?,?,?,?,?,'model','m',?)",
                         (f'{cut}-{kind}', cut, row_kind, title, json.dumps(content), json.dumps({'structured_type': kind})))

    # ------------------------------------------------------------ fake model

    def answer(self, system, data):
        """What the director 'writes' for a step; tests replace single steps through ``self.answers``."""
        step = {direction.BIBLE_SYSTEM: 'bible', direction.PLAN_SYSTEM: 'plan', direction.SHOT_SYSTEM: 'shot', direction.REVIEW_SYSTEM: 'review'}[system]
        self.calls.append((step, data))
        if step in self.answers:
            custom = self.answers[step]
            return custom(data) if callable(custom) else custom
        if step == 'bible':
            return BIBLE
        if step == 'plan':
            ids = [item['id'] for item in data['candidates'] if item['kind'] in ('start', 'cut', 'end')]
            return {'shots': [{'from': a, 'to': b, 'frame_cut': None, 'scene': 'S1', 'intent': 'Follow the source.', 'staging': ''} for a, b in zip(ids, ids[1:])], 'notes': 'One shot per cut.'}
        if step == 'shot':
            lines = [item['line'] for item in data['dialogue']] if isinstance(data['dialogue'], list) else []
            return shot_answer(lines, data['shot']['seconds'])
        return {'verdict': 'The sequence holds together.', 'issues': [{'shot': 1, 'severity': 'check', 'problem': 'The hand ends lower than shot 2 starts.', 'suggestion': 'End with the hand at shoulder height.'},
                                                                      {'shot': 99, 'severity': 'fix', 'problem': 'No such shot.', 'suggestion': ''}]}

    def fake_chat(self, model, system, evidence, fields, **kwargs):
        try:
            return kwargs['validator'](self.answer(system, json.loads(evidence))), 'fake-model'
        except ValueError as exc:
            raise RuntimeError(f'OpenRouter returned an invalid analysis after 3 attempts: {exc}') from exc

    def work(self, limit=40):
        """Run queued sequence jobs until none is left; returns how many ran."""
        ran = 0
        with patch('app.inference_worker.chat_json', side_effect=self.fake_chat):
            while ran < limit:
                claimed = worker.claim_job(('sequence',))
                if not claimed:
                    break
                sequence_pipeline.process(*claimed)
                ran += 1
        return ran

    # --------------------------------------------------------------- helpers

    def create(self, first=1, last=3, **extra):
        response = self.client.post('/api/episodes/e/sequences', json={'from_cut': first, 'to_cut': last, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['sequence']['id']

    def detail(self, sequence_id):
        response = self.client.get(f'/api/sequences/{sequence_id}')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def run_step(self, sequence_id, step, expected=200, **body):
        response = self.client.post(f'/api/sequences/{sequence_id}/run', json={'step': step, **body})
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def directed(self, **extra):
        """A sequence whose bible and plan are written."""
        sequence_id = self.create(**extra)
        self.run_step(sequence_id, 'direct')
        self.work()
        return sequence_id

    def written(self):
        """A sequence with every shot written and reviewed."""
        sequence_id = self.directed()
        self.run_step(sequence_id, 'shots')
        self.work()
        return sequence_id

    def jobs(self, sequence_id=None):
        return db.all_rows("SELECT status,stage,error,json_extract(payload_json,'$.step') AS step FROM jobs WHERE pipeline='sequence' ORDER BY rowid")


class SequenceFlowTest(SequenceBase):
    def test_create_checks_the_range_and_the_three_minute_limit(self):
        detail = self.detail(self.create(title='  Greeting  ', instruction='Keep it calm.'))
        self.assertEqual((detail['sequence']['title'], detail['sequence']['first_cut'], detail['sequence']['last_cut'], detail['sequence']['duration']), ('Greeting', 1, 3, 10.0))
        self.assertEqual([cut['number'] for cut in detail['cuts']], [1, 2, 3])
        self.assertEqual([line['id'] for line in detail['timeline']['lines']], ['L1', 'L2', 'L3'])
        self.assertEqual((detail['bible'], detail['plan'], detail['shots'], detail['ready'], detail['missing']), (None, None, [], False, []))
        self.assertEqual(self.client.post('/api/episodes/e/sequences', json={'from_cut': 2, 'to_cut': 9}).status_code, 404)
        self.assertEqual(self.client.post('/api/episodes/e/sequences', json={'from_cut': 3, 'to_cut': 1}).status_code, 422)
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=400 WHERE id='c3'")
        too_long = self.client.post('/api/episodes/e/sequences', json={'from_cut': 1, 'to_cut': 3})
        self.assertEqual(too_long.status_code, 422)
        self.assertIn('at most 180 s', too_long.json()['detail'])
        self.assertEqual(self.client.post('/api/episodes/e/sequences', json={'from_cut': 1, 'to_cut': 2}).json()['sequence']['title'], 'Cuts 1–2')
        self.assertEqual(len(self.client.get('/api/episodes/e/sequences').json()), 2)

    def test_direct_writes_bible_then_plan_and_stops(self):
        sequence_id = self.create()
        self.assertEqual(self.run_step(sequence_id, 'direct')['queued'], 1)
        self.run_step(sequence_id, 'direct', expected=409)  # already running
        self.assertEqual(self.work(), 2)  # the bible queued the plan; nothing follows the plan
        self.assertEqual([(job['step'], job['status']) for job in self.jobs()], [('bible', 'done'), ('plan', 'done')])
        detail = self.detail(sequence_id)
        bible = detail['bible']['content']
        self.assertEqual((bible['characters'][0]['name'], bible['characters'][0]['character_context_id'], bible['scenes'][0]['id']), ('Hero', 'ctx', 'S1'))
        # The user's transcript link decides a voice; the model's guess for it is discarded.
        voices = {voice['speaker_label']: voice for voice in bible['voices']}
        self.assertEqual((voices['SPEAKER_01']['character'], voices['SPEAKER_01']['source']), ('Hero', 'transcript'))
        self.assertEqual((voices['SPEAKER_02']['character'], voices['SPEAKER_02']['source'], voices['SPEAKER_02']['confidence']), ('Hero', 'director', 'medium'))
        self.assertEqual([(shot['index'], shot['source_cuts'], shot['duration'], shot['result'], shot['scene']) for shot in detail['shots']],
                         [(1, [1], 2.0, None, 'S1'), (2, [2], 2.0, None, 'S1'), (3, [3], 6.0, None, 'S1')])
        self.assertEqual(detail['shots'][0]['lines'][0]['text'], 'こんにちは')
        self.assertEqual((detail['ready'], detail['plan_problems']), (False, []))
        # The plan step was given the bible and candidates on the sequence clock.
        plan_input = next(data for step, data in self.calls if step == 'plan')
        self.assertEqual([item['id'] for item in plan_input['candidates']], ['B0', 'B1', 'B2', 'B3'])
        self.assertIn('L2 2.50-3.50 SPEAKER_02 / Hero: はい', plan_input['transcript'])
        # The bible was given what the opening frames show of each character, not only the record notes.
        bible_input = next(data for step, data in self.calls if step == 'bible')
        self.assertEqual(bible_input['characters'], [{'name': 'Hero', 'notes': 'Green coat', 'seen_in_cuts': [1, 2, 3], 'appearance_in_first_frames': ['cut 1: Green coat, short dark hair']}])
        self.assertEqual([voice['speaker_label'] for voice in bible_input['voices']], ['SPEAKER_01', 'SPEAKER_02'])

    def test_shots_are_written_reviewed_and_ready_for_download(self):
        sequence_id = self.directed()
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 3)
        self.assertEqual(self.work(), 4)  # three shots, then the review queued by the last one
        self.assertEqual([job['step'] for job in self.jobs()][-4:], ['shot', 'shot', 'shot', 'review'])
        detail = self.detail(sequence_id)
        self.assertTrue(detail['ready'])
        first = detail['shots'][0]
        self.assertEqual((first['stale'], first['result']['check']['ok'], first['result']['origin'], first['generate_seconds'], first['trim_seconds']), ([], True, 'model', 5.0, 2.0))
        prompt = first['result']['prompt']
        self.assertTrue(prompt.startswith('subject_definitions:\n<Picture 1> is the first frame of [Shot 1]: Hero stands'))
        self.assertIn('<Subject 1> is Hero, whose face, hair and clothing come from <Picture 2>:', prompt)
        self.assertIn('At 00:00.400, <Subject 1> (S1) says: <d>[Japanese] こんにちは</d>', prompt)
        self.assertIn('00:02.000-00:05.000: Everyone holds the final pose', prompt)  # shorter than H3's minimum
        self.assertIn('<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:00.400.', prompt)
        self.assertNotIn('Hero stands with both arms', prompt.split('detailed_description:')[1])  # names became labels
        self.assertEqual([item['label'] for item in first['result']['derived']['references']], ['<Picture 1>', '<Picture 2>', '<Audio 1>'])
        self.assertEqual(detail['review']['content']['issues'], [{'shot': 1, 'severity': 'check', 'problem': 'The hand ends lower than shot 2 starts.',
                                                                  'suggestion': 'End with the hand at shoulder height.', 'shot_key': first['key']}])
        # Each shot was written from its own cuts, on its own clock.
        third = [data for step, data in self.calls if step == 'shot'][2]
        self.assertEqual((third['shot']['number'], third['shot']['seconds'], third['cuts'][0]['plays_at'], third['dialogue'][0]['t']), (3, 6.0, [0.0, 6.0], [1.0, 2.0]))
        self.assertEqual(third['dialogue'][0]['speaker_known_from'], 'sequence bible (medium)')
        self.assertEqual(third['previous_shot']['hands_over'], 'Hero ends with the hand at chest height, facing the camera.')
        # Shots are written side by side, so each one also gets its neighbours' states from the analysis itself.
        second = [data for step, data in self.calls if step == 'shot'][1]
        self.assertEqual(third['previous_shot']['ends_with'], [{'name': 'Hero', 'ends': 'center, hand raised'}])
        self.assertEqual(second['next_shot']['opens_with'], [{'name': 'Hero', 'position': 'center', 'pose': 'standing'}])
        self.assertNotIn('next_shot', third)
        self.run_step(sequence_id, 'shots', expected=409)  # nothing left to write

    def test_an_invalid_plan_is_completed_by_the_app(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=24 WHERE id='c3'")  # 20 s: too long for one shot
            conn.execute("UPDATE episodes SET duration=24")
        sequence_id = self.directed()
        detail = self.detail(sequence_id)
        self.assertEqual(detail['plan']['completed_by_app'][:45], 'OpenRouter returned an invalid analysis after')
        self.assertEqual([shot['source_cuts'] for shot in detail['shots']], [[1], [2], [3], [3]])
        self.assertTrue(all(shot['duration'] <= 15 for shot in detail['shots']))
        self.assertEqual((detail['shots'][3]['auto_fixed'], detail['shots'][3]['frame']['strategy'], detail['shots'][3]['starts_mid_cut']), (True, 'frame_at_time', True))

    def test_lines_without_a_voice_label_get_no_owner_from_the_bible(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES ('t4','e',7.0,8.0,'えっ','','model')")
        sequence_id = self.written()
        detail = self.detail(sequence_id)
        self.assertEqual([voice['speaker_label'] for voice in detail['bible']['content']['voices']], ['SPEAKER_01', 'SPEAKER_02'])
        bible_input = next(data for step, data in self.calls if step == 'bible')
        self.assertIn('L4 7.00-8.00 Unknown speaker: えっ', bible_input['transcript'])
        third = [data for step, data in self.calls if step == 'shot'][2]
        self.assertEqual((third['dialogue'][1]['speaker'], third['dialogue'][1]['speaker_known_from']), ('Unknown speaker', 'nobody: unconfirmed voice'))
        line = detail['shots'][2]['result']['derived']['dialogue'][1]
        self.assertEqual((line['character_name'], line['attribution']), (None, 'unknown'))

    def test_direction_needs_analysed_cuts(self):
        with db.connect() as conn:
            conn.execute("DELETE FROM analysis_results WHERE id='c2-video_analysis'")
        sequence_id = self.create()
        self.assertEqual(self.detail(sequence_id)['missing'], [2])
        self.assertIn('first-frame and a video analysis first: 2', self.run_step(sequence_id, 'direct', expected=409)['detail'])
        self.run_step(sequence_id, 'shots', expected=409)
        self.run_step(sequence_id, 'review', expected=409)

    def test_changed_inputs_mark_shots_stale_and_one_shot_can_be_rewritten(self):
        sequence_id = self.written()
        key = self.detail(sequence_id)['shots'][1]['key']
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET text='いいえ' WHERE id='t2'")
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content) VALUES ('c3','- Speaking: the voice is off screen.')")
        detail = self.detail(sequence_id)
        self.assertEqual([shot['stale'] for shot in detail['shots']], [[], ['plan'], ['evidence']])
        self.assertFalse(detail['ready'])
        self.calls.clear()
        self.assertEqual(self.run_step(sequence_id, 'shots', shot_keys=[key], instruction='Keep the mouth closed.')['queued'], 1)
        self.assertEqual(self.work(), 1)  # a single shot does not trigger a review
        shot = self.detail(sequence_id)['shots'][1]
        self.assertEqual((shot['stale'], shot['result']['instruction'], shot['result']['parent_id'] is not None), ([], 'Keep the mouth closed.', True))
        self.assertEqual(self.calls[0][1]['instruction_for_this_shot'], 'Keep the mouth closed.')
        self.assertIn('<d>[Japanese] いいえ</d>', shot['result']['prompt'])
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 1)  # only the remaining stale shot
        self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/run', json={'step': 'shots', 'shot_keys': ['1-2']}).status_code, 404)

    def test_a_result_edited_while_its_job_ran_is_not_overwritten(self):
        sequence_id = self.directed()
        self.run_step(sequence_id, 'bible')
        bible = self.detail(sequence_id)['bible']
        edited = {**bible['content'], 'logline': 'Edited by hand.'}
        self.assertEqual(self.client.put(f"/api/sequence-results/{bible['id']}", json={'content': edited}).status_code, 200)
        self.work()
        job = self.jobs()[-1]
        self.assertEqual((job['step'], job['status']), ('bible', 'failed'))
        self.assertIn('changed while', job['error'])
        self.assertEqual(self.detail(sequence_id)['bible']['content']['logline'], 'Edited by hand.')

    def test_hand_edits_are_validated_like_model_output(self):
        sequence_id = self.written()
        detail = self.detail(sequence_id)
        shot = detail['shots'][0]
        content = shot['result']['content']
        bad = {**content, 'summary': 'Hero waves, then a cut to black.'}
        refused = self.client.put(f"/api/sequence-results/{shot['result']['id']}", json={'content': bad})
        self.assertEqual(refused.status_code, 422)
        self.assertIn('summary contains the word "cut"', refused.json()['detail'])
        silent = {**content, 'speech': [{'line': 'L1', 'delivery': 'off_screen', 'subject': None}]}
        saved = self.client.put(f"/api/sequence-results/{shot['result']['id']}", json={'content': silent})
        self.assertEqual(saved.status_code, 200, saved.text)
        result = saved.json()['shots'][0]['result']
        self.assertEqual((result['origin'], result['parent_id']), ('manual_edit', shot['result']['id']))
        self.assertIn('At 00:00.400, an off-screen voice is heard: <d>[Japanese] こんにちは</d>', result['prompt'])
        self.assertIn('<Subject 1> does not speak; the lips stay closed.', result['prompt'])
        self.assertEqual(result['derived']['dialogue'][0]['delivery'], 'off_screen')
        self.assertEqual(self.client.put(f"/api/sequence-results/{shot['result']['id']}", json={'content': silent}).status_code, 409)  # superseded
        history = self.client.get(f"/api/sequence-results/{result['id']}/history").json()
        self.assertEqual([row['origin'] for row in history], ['manual_edit', 'model'])
        bible = detail['bible']
        wrong = self.client.put(f"/api/sequence-results/{bible['id']}", json={'content': {**bible['content'], 'style': 'Anime.'}})
        self.assertEqual(wrong.status_code, 422)
        self.assertIn('The target video is', wrong.json()['detail'])
        self.assertEqual(self.client.put(f"/api/sequence-results/{detail['review']['id']}", json={'content': {}}).status_code, 409)

    def test_merge_and_split_rewrite_the_plan(self):
        sequence_id = self.written()
        shots = self.detail(sequence_id)['shots']
        merged = self.client.post(f'/api/sequences/{sequence_id}/plan/merge', json={'shot_key': shots[0]['key']})
        self.assertEqual(merged.status_code, 200, merged.text)
        after = merged.json()
        self.assertEqual([(shot['source_cuts'], shot['merged'], shot['result'] is None) for shot in after['shots']], [([1, 2], True, True), ([3], False, False)])
        self.assertEqual((after['plan']['origin'], after['ready']), ('manual_edit', False))
        split = self.client.post(f'/api/sequences/{sequence_id}/plan/split', json={'shot_key': after['shots'][1]['key'], 'time': 7.0}).json()
        self.assertEqual([(shot['start_time'], shot['end_time']) for shot in split['shots']], [(0.0, 4.0), (4.0, 7.0), (7.0, 10.0)])
        self.assertEqual(split['shots'][2]['frame']['strategy'], 'frame_at_time')
        self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/plan/split', json={'shot_key': split['shots'][1]['key'], 'time': 4.1}).status_code, 422)
        self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/plan/merge', json={'shot_key': 'missing'}).status_code, 404)
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 3)

    def test_cut_changes_block_shots_until_the_sequence_is_directed_again(self):
        sequence_id = self.directed()
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=3 WHERE id='c2'")
            conn.execute("UPDATE cuts SET start_time=3 WHERE id='c3'")
        detail = self.detail(sequence_id)
        self.assertTrue(detail['sequence']['cuts_changed'])
        # The old plan still tiles the range in time, but its shots no longer follow the hard cuts.
        self.assertEqual((detail['plan_problems'], detail['shots'][1]['source_cuts']), ([], [2, 3]))
        self.assertIn('direct the sequence again', self.run_step(sequence_id, 'shots', expected=409)['detail'])
        self.run_step(sequence_id, 'direct')
        self.work()
        detail = self.detail(sequence_id)
        self.assertEqual((detail['sequence']['cuts_changed'], detail['plan_problems'], [shot['duration'] for shot in detail['shots']]), (False, [], [2.0, 1.0, 7.0]))

    def test_cancel_and_delete(self):
        sequence_id = self.directed()
        self.run_step(sequence_id, 'shots')
        self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/cancel').json(), {'cancelled': 3})
        self.assertEqual(self.work(), 0)
        self.run_step(sequence_id, 'shots')
        job = db.one("SELECT id FROM jobs WHERE pipeline='sequence' AND status='queued' LIMIT 1")
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/cancel").status_code, 200)
        self.assertEqual(self.client.delete(f'/api/sequences/{sequence_id}').status_code, 200)
        self.assertEqual(self.client.get(f'/api/sequences/{sequence_id}').status_code, 404)
        self.assertEqual(db.one('SELECT COUNT(*) AS n FROM sequence_results')['n'], 0)
        self.assertEqual(db.one("SELECT COUNT(*) AS n FROM jobs WHERE pipeline='sequence' AND status IN ('queued','running')")['n'], 0)

    def test_sequence_jobs_share_their_episode_with_text_jobs_only(self):
        sequence_id = self.directed()
        self.run_step(sequence_id, 'shots')
        first = worker.claim_job(worker.TEXT_PIPELINES)
        second = worker.claim_job(worker.TEXT_PIPELINES)
        self.assertTrue(first and second and first[0] != second[0])  # two shots write side by side
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='queued',lease_owner=NULL,lease_until=NULL WHERE id IN (?,?)", (first[0], second[0]))
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,lease_owner,lease_until) VALUES ('visual','e','analyze','running','x',?)", (9e12,))
        self.assertIsNone(worker.claim_job(worker.TEXT_PIPELINES))  # visual analysis holds the episode
        episode = self.client.get('/api/episodes/e').json()
        job = next(job for job in episode['jobs'] if job['pipeline'] == 'sequence' and job['sequence']['step'] == 'shot')
        self.assertEqual((job['sequence']['id'], job['cuts']), (sequence_id, []))


if __name__ == '__main__':
    unittest.main()
