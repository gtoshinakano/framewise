"""Sequence direction end to end: API, queued steps, revisions, staleness and hand edits (the model is faked)."""
import json
import re
import unittest
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, sequence_direction as direction, sequence_edit, sequence_pipeline, worker

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
         'spoken_language': 'Japanese',
         'scenes': [{'id': 'A', 'from_cut': 1, 'to_cut': 3, 'location': 'Shopping street', 'time_of_day': 'dusk', 'lighting': 'warm low sun', 'mood': 'calm', 'scenario': None}],
         'characters': [{'name': 'hero', 'role': 'the lead', 'anchors': 'slim build, short dark hair, green wool coat', 'outfit': 'green coat over a white shirt', 'arc': 'decides to leave'}],
         'voices': [{'speaker_label': 'SPEAKER_01', 'character': 'Somebody else', 'kind': 'character', 'confidence': 'low', 'reason': 'guess'},
                    {'speaker_label': 'SPEAKER_02', 'character': 'Hero', 'kind': 'character', 'confidence': 'medium', 'reason': 'Visibly speaking while it is heard.'}],
         'arc': 'From greeting to parting.', 'pacing': 'Even and calm.', 'continuity': ['The green coat stays on.'], 'open_questions': []}


def shot_answer(speakers=(), seconds=2.0):
    return {
        'subjects': [{'name': 'Hero', 'anchors': 'slim build, short dark hair, green wool coat over a white shirt', 'keep': 'face, hair and green coat follow the sheet exactly'}],
        'picture_line': 'Hero stands in the centre of a narrow street at dusk, facing the camera.',
        'keep_picture': 'composition, cel-shaded style, warm palette and low sun of the opening frame',
        'still': {'shot': 'Medium shot at eye level, cropped at the waist.', 'placement': 'Hero stands in the centre midground, torso and head facing the camera, eyes looking straight into the lens.',
                  'subjects': 'Hero with arms at the sides, calm face, green coat buttoned.',
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
        'voices': [{'speaker': speaker, 'from': 0, 'delivery': 'on_screen' if speaker == 'Hero' else 'off_screen', 'subject': 'hero' if speaker == 'Hero' else None} for speaker in speakers],
        'scenario_anchors': '', 'scenario_keep': '',
        'continuity': {'from_previous': 'Hero stands in the same spot as before.', 'to_next': 'Hero ends with the hand at chest height, facing the camera.'},
        'open_questions': []}


def sides(diff):
    """The texts before and after a diff."""
    return ''.join(part['text'] for part in diff if part['op'] != 'insert'), ''.join(part['text'] for part in diff if part['op'] != 'delete')


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
        step = {direction.COMPILE_SYSTEM: 'compile', direction.TIMING_SYSTEM: 'timing', direction.BIBLE_SYSTEM: 'bible', direction.PLAN_SYSTEM: 'plan', direction.SHOT_SYSTEM: 'shot', direction.REVIEW_SYSTEM: 'review'}[system]
        self.calls.append((step, data))
        if step in self.answers:
            custom = self.answers[step]
            return custom(data) if callable(custom) else custom
        if step == 'compile':
            rows = re.findall(r'^(L\d+) [\d.]+-[\d.]+ (\S+)(?: / (.+?))?: (.*)$', data['lines'], re.M)
            return {'lines': [{'ids': [line], 'text': text, 'speaker': name or None, 'subtitle_ids': [], 'confidence': 'high', 'note': '', 'continues_next': False}
                              for line, _, name, text in rows], 'voices': [], 'open_questions': []}
        if step == 'timing':
            return {'decisions': [{'id': item['id'], 'choice': next((choice for choice in ('after_line', 'before_line') if choice in item['options']), 'keep_cut'),
                                   'reason': 'The phrase stays with the speaking character.'} for item in data['decisions']]}
        if step == 'bible':
            return BIBLE
        if step == 'plan':
            decisions = data['decisions'] if isinstance(data['decisions'], list) else []
            return {'decisions': [{'id': item['id'], 'choice': 'keep_cut', 'reason': 'The reply cuts in.'} for item in decisions],
                    'cuts': [{'cut': item['cut'], 'intent': 'Follow the source.'} for item in data['cuts']], 'notes': 'One shot per cut.'}
        if step == 'shot':
            return shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds'])
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
                worker.release_lease(*claimed)
                ran += 1
        return ran

    # --------------------------------------------------------------- helpers

    def interrupt(self):
        """Hero's line L3 runs across the hard cut between cut 2 and cut 3 (3.6-4.6 s)."""
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin) VALUES ('t4','e',3.6,4.6,'ちょっと待って','SPEAKER_01','ctx','model')")

    def create(self, first=1, last=3, review='source', **extra):
        """A new sequence; its dialogue is confirmed as the original transcript unless ``review`` is None (the compile job is queued either way)."""
        response = self.client.post('/api/episodes/e/sequences', json={'from_cut': first, 'to_cut': last, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        sequence_id = response.json()['sequence']['id']
        if review == 'source':
            with db.connect() as conn:
                conn.execute("UPDATE jobs SET status='failed',error='skipped in tests' WHERE pipeline='sequence' AND status='queued' AND json_extract(payload_json,'$.step')='compile'")
            self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/compilation/use-source').status_code, 200)
        return sequence_id

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
        self.run_step(sequence_id, 'review')
        self.work()
        return sequence_id

    def jobs(self, sequence_id=None):
        return db.all_rows("SELECT status,stage,error,json_extract(payload_json,'$.step') AS step FROM jobs WHERE pipeline='sequence' AND json_extract(payload_json,'$.step')!='compile' ORDER BY rowid")


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
        first = self.run_step(sequence_id, 'direct')
        self.assertEqual(first['queued'], 2)
        self.assertEqual(self.run_step(sequence_id, 'direct'), first)
        self.assertEqual(self.work(), 2)
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
        # The plan step was given the bible, one digest per cut and the transcript; no hard cut falls inside speech here.
        plan_input = next(data for step, data in self.calls if step == 'plan')
        self.assertEqual(([item['cut'] for item in plan_input['cuts']], plan_input['decisions']), ([1, 2, 3], 'No hard cut falls inside speech: every boundary stays on its hard cut.'))
        self.assertIn('L2 2.50-3.50 SPEAKER_02 / Hero: はい', plan_input['transcript'])
        self.assertEqual([(shot['cut'], shot['opens']['how'], shot['closes']['how']) for shot in detail['shots']], [(1, 'start', 'keep_cut'), (2, 'keep_cut', 'keep_cut'), (3, 'keep_cut', 'end')])
        # The bible was given what the opening frames show of each character, not only the record notes.
        bible_input = next(data for step, data in self.calls if step == 'bible')
        self.assertEqual(bible_input['characters'], [{'name': 'Hero', 'notes': 'Green coat', 'seen_in_cuts': [1, 2, 3], 'appearance_in_first_frames': ['cut 1: Green coat, short dark hair']}])
        self.assertEqual([voice['speaker_label'] for voice in bible_input['voices']], ['SPEAKER_01', 'SPEAKER_02'])

    def test_shots_are_written_reviewed_and_ready_for_download(self):
        sequence_id = self.directed()
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 3)
        self.assertEqual(self.work(), 3)
        self.assertEqual([job['step'] for job in self.jobs()][-3:], ['shot', 'shot', 'shot'])
        detail = self.detail(sequence_id)
        self.assertTrue(detail['ready'])
        first = detail['shots'][0]
        self.assertEqual((first['stale'], first['result']['check']['ok'], first['result']['origin'], first['generate_seconds'], first['trim_seconds']), ([], True, 'model', 5.0, 2.0))
        prompt = first['result']['prompt']
        self.assertTrue(prompt.startswith('subject_definitions:\n<Picture 1> is the first frame of [Shot 1]: Hero stands'))
        self.assertIn('<Subject 1> is Hero, whose face, hair and clothing come from <Picture 2>:', prompt)
        self.assertIn('At 00:00.400, <Subject 1> (S1) says: <d>[Japanese] こんにちは</d>', prompt)
        self.assertIn('00:02.000-00:05.000: Everyone holds the final pose', prompt)  # shorter than H3's minimum
        self.assertIn('<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:00.400 to 00:01.200.', prompt)
        self.assertNotIn('Hero stands with both arms', prompt.split('detailed_description:')[1])  # names became labels
        self.assertEqual([item['label'] for item in first['result']['derived']['references']], ['<Picture 1>', '<Picture 2>', '<Audio 1>'])
        self.assertIsNone(detail['review'])
        self.run_step(sequence_id, 'review')
        self.assertEqual(self.work(), 1)
        detail = self.detail(sequence_id)
        self.assertTrue(detail['ready'])
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
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 0)


    def test_shot_writer_uses_current_keyframes_beats_and_summary_camera(self):
        keyframes = {'contract': 'keyframes-v1', 'cut': {'start_time': 2, 'end_time': 4}, 'summary': 'Hero looks sideways.',
                     'frames': [{'label': 'F1', 't': 0, 'characters': [{'name': 'Hero', 'state': {'gaze': 'straight ahead'}}]},
                                {'label': 'F2', 't': 2, 'characters': [{'name': 'Hero', 'state': {'gaze': 'screen-right'}}]}],
                     'segments': [{'from': 'F1', 'to': 'F2', 'start': 0, 'end': 2, 'camera': {'move': 'static'},
                                   'characters': [{'name': 'Hero', 'changes': [{'part': 'eyes', 'description': 'Eyes flick screen-right', 'source': 'qwen'}]}]}]}
        beats = {'contract': 'beats-v1', 'based_on': {'keyframe_analysis': 'keyframes'},
                 'beats': [{'start': 0, 'end': 1.3, 'intent': 'Suppresses a laugh.', 'characters': [
                     {'character': 'Hero', 'body': 'Shoulders tighten.', 'face': 'Cheeks lift.', 'gaze': 'Eyes flick right.', 'mouth': 'Lips press together.'}]}]}
        with db.connect() as conn:
            for identifier, kind, content in [('keyframes', 'keyframe_analysis', keyframes), ('performance', 'performance_beats', beats)]:
                conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json)
                    VALUES (?,'c2','cut_description',?,?,'model',?)""", (identifier, kind, json.dumps(content), json.dumps({'structured_type': kind})))
            conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin)
                VALUES ('summary','c2','cut_description','Cut summary',?,'model')""", (analysis_tests.PRODUCTION_TEXT,))
        sequence_id = self.directed()
        plan = next(data for step, data in self.calls if step == 'plan')
        self.assertEqual(plan['cuts'][1]['performance'][0]['intent'], 'Suppresses a laugh.')
        points_url = f'/api/sequences/{sequence_id}/shot-points'
        response = self.client.put(points_url, json={'points': [{'after_cut_id': 'c1', 'time': 2.5}]})
        self.assertEqual(response.status_code, 200, response.text)
        shot_key = response.json()['shots'][1]['key']
        self.run_step(sequence_id, 'shots', shot_keys=[shot_key])
        self.assertEqual(self.work(), 1)
        shot_input = next(data for step, data in self.calls if step == 'shot')
        own = next(cut for cut in shot_input['cuts'] if cut['cut'] == 2)
        self.assertEqual(own['performance_beats'][0]['t'], [0, .8])
        self.assertEqual(own['performance_beats'][0]['characters'][0]['face'], 'Cheeks lift.')
        self.assertEqual(own['keyframes']['states'][0]['t'], -.5)
        self.assertNotIn('Cinematic Camera', own['cut_summary'])
        shot = self.detail(sequence_id)['shots'][1]['result']
        self.assertIn('ARRI Alexa 35', shot['derived']['still_prompt'])
        self.assertIn('Cooke S4/i', shot['derived']['still_prompt'])
        self.assertNotIn('ARRI Alexa 35', shot['prompt'])

    def test_review_suggestions_can_be_dismissed_without_affecting_readiness(self):
        sequence_id = self.written()
        detail = self.detail(sequence_id)
        review = detail['review']
        self.assertTrue(detail['ready'])
        dismissed = self.client.post(f"/api/sequence-results/{review['id']}/review-issues/0/dismiss")
        self.assertEqual(dismissed.status_code, 200, dismissed.text)
        self.assertTrue(dismissed.json()['ready'])
        self.assertTrue(dismissed.json()['review']['content']['issues'][0]['dismissed'])


    def test_an_invalid_plan_is_completed_by_the_app(self):
        self.interrupt()
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=24 WHERE id='c3'")  # 20 s: too long for one shot
            conn.execute("UPDATE episodes SET duration=24")
        # The director divides L3 without a reason and forgets the intents.
        self.answers['plan'] = {'decisions': [{'id': 'K2', 'choice': 'keep_cut', 'reason': ''}], 'cuts': [], 'notes': 'Quick.'}
        sequence_id = self.directed()
        detail = self.detail(sequence_id)
        self.assertEqual(detail['plan']['completed_by_app'][:45], 'OpenRouter returned an invalid analysis after')
        plan_input = next(data for step, data in self.calls if step == 'plan')
        decision = plan_input['decisions'][0]
        self.assertEqual((decision['id'], decision['cuts'], list(decision['options'])), ('K2', [2, 3], ['keep_cut', 'before_line', 'after_line']))
        self.assertEqual(decision['options']['after_line'], 'moves 0.75 s later to 4.75: the shot of cut 2 runs on until L3 has ended and carries it whole; the first 0.75 s of cut 3 are hidden; the hidden picture shows 0.2-1.0 s Hero: raises a hand')
        self.assertEqual(plan_input['cuts'][2]['parts'], 2)
        # One shot per cut, the long cut in two parts; the app kept L3 whole with its visibly speaking speaker in cut 2.
        self.assertEqual([(shot['cut'], shot['part']) for shot in detail['shots']], [(1, None), (2, None), (3, [1, 2]), (3, [2, 2])])
        self.assertTrue(all(shot['duration'] <= 15 for shot in detail['shots']))
        boundary = next(item for item in detail['timeline']['boundaries'] if item['id'] == 'K2')
        self.assertEqual((boundary['choice'], boundary['by'], boundary['reason']), ('after_line', 'app', 'Chosen by the app: keeps the phrase whole.'))
        self.assertEqual((detail['shots'][1]['end_time'], detail['shots'][1]['closes']['how'], detail['shots'][2]['frame']['strategy']), (4.75, 'after_line', 'frame_at_time'))
        self.assertEqual((detail['shots'][3]['frame']['strategy'], detail['shots'][3]['frame']['continues_previous_shot'], detail['shots'][3]['opens']['how']), ('frame_at_time', True, 'continuation'))
        self.assertEqual(detail['shots'][0]['intent'], 'Follow the source.')

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
        bible, plan = self.run_step(sequence_id, 'direct')['job_ids']
        self.run_step(sequence_id, 'shots')
        self.run_step(sequence_id, 'review')
        self.work()
        self.assertIn('first-frame analysis, keyframes and performance beats for cuts: 2', db.one('SELECT error FROM jobs WHERE id=?', (bible,))['error'])
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (plan,))['status'], 'cancelled')

    def test_changed_inputs_mark_shots_stale_and_one_shot_can_be_rewritten(self):
        sequence_id = self.written()
        key = self.detail(sequence_id)['shots'][1]['key']
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET text='いいえ' WHERE id='t2'")
            conn.execute("INSERT INTO cut_brief_instructions(cut_id,content) VALUES ('c3','- Speaking: the voice is off screen.')")
        detail = self.detail(sequence_id)
        # The reviewed dialogue is frozen: a changed transcript does not move the shots, it makes the compilation stale.
        self.assertEqual([shot['stale'] for shot in detail['shots']], [[], [], ['evidence']])
        self.assertTrue(detail['compilation']['stale'])
        self.assertFalse(detail['ready'])
        self.calls.clear()
        self.assertEqual(self.run_step(sequence_id, 'shots', shot_keys=[key], instruction='Keep the mouth closed.')['queued'], 1)
        self.assertEqual(self.work(), 1)  # a single shot does not trigger a review
        shot = self.detail(sequence_id)['shots'][1]
        self.assertEqual((shot['stale'], shot['result']['instruction'], shot['result']['parent_id'] is not None), ([], 'Keep the mouth closed.', True))
        self.assertEqual(self.calls[0][1]['instruction_for_this_shot'], 'Keep the mouth closed.')
        self.assertIn('<d>[Japanese] はい</d>', shot['result']['prompt'])  # the reviewed dialogue, not the edited transcript
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 1)  # only the remaining stale shot
        self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/run', json={'step': 'shots', 'shot_keys': ['1-2']}).status_code, 404)

    def test_the_users_instructions_are_remembered_by_every_later_rewrite_of_that_shot(self):
        sequence_id = self.written()
        first, second = (shot['key'] for shot in self.detail(sequence_id)['shots'][:2])

        def rewrite(key, instruction='', **extra):
            self.calls.clear()
            self.run_step(sequence_id, 'shots', shot_keys=[key], instruction=instruction, **extra)
            self.assertEqual(self.work(), 1)
            return self.calls[0][1]

        sent = rewrite(first, 'Keep the camera still.')
        self.assertEqual(sent['instruction_for_this_shot'], 'Keep the camera still.')
        self.assertNotIn('earlier_instructions_for_this_shot', sent)  # nothing came before it
        sent = rewrite(first, 'She smiles at the end.')
        self.assertEqual((sent['instruction_for_this_shot'], sent['earlier_instructions_for_this_shot']), ('She smiles at the end.', ['Keep the camera still.']))
        # A plain rewrite, and a rewrite after a hand edit, still follow everything the user said.
        shot = self.detail(sequence_id)['shots'][0]
        edited = {**shot['result']['content'], 'summary': 'Hero raises a hand and the camera holds still.'}
        self.assertEqual(self.client.put(f"/api/sequence-results/{shot['result']['id']}", json={'content': edited}).status_code, 200)
        sent = rewrite(first)
        self.assertNotIn('instruction_for_this_shot', sent)
        self.assertEqual(sent['earlier_instructions_for_this_shot'], ['Keep the camera still.', 'She smiles at the end.'])
        # Saying one again does not repeat it; another shot has its own.
        sent = rewrite(first, 'keep the camera still.')
        self.assertEqual((sent['instruction_for_this_shot'], sent['earlier_instructions_for_this_shot']), ('keep the camera still.', ['She smiles at the end.']))
        self.assertNotIn('earlier_instructions_for_this_shot', rewrite(second))
        # A suggestion of the automatic review is used once: it ranks below the user's words and is never remembered.
        sent = rewrite(first, 'End with the hand at shoulder height.', instruction_source='review')
        self.assertNotIn('instruction_for_this_shot', sent)
        self.assertEqual((sent['review_suggestion'], sent['earlier_instructions_for_this_shot']), ('End with the hand at shoulder height.', ['She smiles at the end.', 'keep the camera still.']))
        sent = rewrite(first)
        self.assertEqual(sent['earlier_instructions_for_this_shot'], ['She smiles at the end.', 'keep the camera still.'])
        self.assertNotIn('review_suggestion', sent)
        history = self.client.get(f'/api/sequences/{sequence_id}/shots/{first}/history').json()
        self.assertEqual(history['remembered'], ['She smiles at the end.', 'keep the camera still.'])
        by_source = [(row['instruction'], row['instruction_source']) for row in history['revisions'] if row['instruction']]
        self.assertIn(('End with the hand at shoulder height.', 'review'), by_source)
        self.assertEqual(history['revisions'][0]['earlier_instructions'], 2)
        # The review is told what the user asked for, so it does not suggest undoing it.
        self.calls.clear()
        self.run_step(sequence_id, 'review')
        self.work()
        rows = next(data for step, data in self.calls if step == 'review')['shots']
        self.assertEqual(rows[0]['user_instructions'], ['She smiles at the end.', 'keep the camera still.'])
        self.assertNotIn('user_instructions', rows[1])
        self.assertIn('user_instructions', direction.REVIEW_SYSTEM)

    def test_prompts_make_the_users_dialogue_the_only_truth_and_a_summary_holds_no_speech(self):
        for system in (direction.BIBLE_SYSTEM, direction.PLAN_SYSTEM, direction.SHOT_SYSTEM, direction.REVIEW_SYSTEM):
            self.assertIn("The user's own spoken words and timings are absolute", system)
            self.assertIn('supporting material only', system)
        self.assertIn('earlier_instructions_for_this_shot', direction.SHOT_SYSTEM)
        with db.connect() as conn:
            # A summary written before dialogue moved to the sequence step still holds a Dialogue section with other words and times.
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json) VALUES ('c1-summary','c1','production_brief','Cut summary',?,'model','{}')",
                         ('### Scene\nHero in a street.\n### Opening Frame\nCentre.\n### Animation Timeline\n[0-2s] Hero raises a hand.\n'
                          '### Dialogue\n[1.9s] Hero: 違う言葉\n### Continuity & Constraints\nThe coat stays green.',))
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,status,language) VALUES ('r','e','complete','Japanese')")
            conn.execute("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES ('s1','r','e',0,0.4,1.2,'字幕の言葉','high')")
        self.written()
        shot = next(data for step, data in self.calls if step == 'shot')
        summary = shot['cuts'][0]['cut_summary']
        self.assertNotIn('違う言葉', summary)
        self.assertNotIn('Dialogue', summary)
        self.assertIn('Hero raises a hand', summary)
        self.assertIn('The coat stays green.', summary)  # the sections around it stay
        self.assertEqual([(line['text'], line['t']) for line in shot['dialogue']], [('こんにちは', [0.4, 1.2])])
        # The transcript was used as it is (nobody reconciled it with the subtitles), so they stay available to understand the words.
        self.assertNotIn('dialogue_reviewed', shot)
        self.assertEqual(shot['subtitles_for_meaning_only']['lines'], [{'t': 0.4, 'text': '字幕の言葉'}])

    def test_a_summary_loses_only_its_dialogue_section(self):
        from app import sequence_evidence
        text = '### Scene\nA.\n### Dialogue\n[0.4s] Hero: words\n### Continuity & Constraints\nB.'
        self.assertEqual(sequence_evidence.without_dialogue(text), '### Scene\nA.\n### Continuity & Constraints\nB.')
        self.assertEqual(sequence_evidence.without_dialogue('### Scene\nA.\n### Dialogue\n[0.4s] Hero: words'), '### Scene\nA.')
        self.assertEqual(sequence_evidence.without_dialogue('### Scene\nHe gives a Dialogue.\n### Continuity & Constraints\nB.'), '### Scene\nHe gives a Dialogue.\n### Continuity & Constraints\nB.')
        self.assertEqual(sequence_evidence.without_dialogue(None), '')

    def test_a_result_edited_while_its_job_ran_is_not_overwritten(self):
        sequence_id = self.directed()
        [job_id] = self.run_step(sequence_id, 'bible')['job_ids']
        bible = self.detail(sequence_id)['bible']
        edited = {**bible['content'], 'logline': 'Edited by hand.'}

        def edit_while_writing(_data):
            self.assertEqual(self.client.put(f"/api/sequence-results/{bible['id']}", json={'content': edited}).status_code, 200)
            return BIBLE

        self.answers['bible'] = edit_while_writing
        self.work()
        self.assertEqual(db.one('SELECT status,stage FROM jobs WHERE id=?', (job_id,)),
                         {'status': 'queued', 'stage': 'Inputs changed, restarting'})
        with db.connect() as conn:
            conn.execute('UPDATE jobs SET not_before=0 WHERE id=?', (job_id,))
        self.work()
        self.assertIn('edited by hand', db.one('SELECT stage FROM jobs WHERE id=?', (job_id,))['stage'])
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
        silent = {**content, 'voices': [{'speaker': 'Hero', 'from': 0, 'delivery': 'off_screen', 'subject': None}]}
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
        wrong = self.client.put(f"/api/sequence-results/{bible['id']}", json={'content': {**bible['content'], 'spoken_language': '日本語'}})
        self.assertEqual(wrong.status_code, 422)
        self.assertIn('spoken_language', wrong.json()['detail'])
        self.assertEqual(self.client.put(f"/api/sequence-results/{detail['review']['id']}", json={'content': {}}).status_code, 409)

    def test_a_shot_written_before_placement_can_still_be_edited_but_a_model_answer_needs_it(self):
        sequence_id = self.written()
        shot = self.detail(sequence_id)['shots'][0]
        content = shot['result']['content']
        self.assertIn('Hero stands in the centre midground', shot['result']['derived']['still_prompt'])
        legacy = {**content, 'still': {key: value for key, value in content['still'].items() if key != 'placement'}}
        saved = self.client.put(f"/api/sequence-results/{shot['result']['id']}", json={'content': legacy})
        self.assertEqual(saved.status_code, 200, saved.text)  # the editor of an older shot is not asked to invent a placement
        self.assertNotIn('Hero stands in the centre midground', saved.json()['shots'][0]['result']['derived']['still_prompt'])
        from app import sequence_direction as direction
        for generated, expected in ((True, True), (False, False)):
            try:
                direction.validate_shot({}, {'duration': 2.0}, {}, [], None, [], generated=generated)
            except ValueError as error:
                self.assertEqual('"still.placement" is empty' in str(error), expected)
        required = direction.shot_format({'duration': 2.0}, [], [])['json_schema']['schema']['properties']['still']['required']
        self.assertEqual(required, list(direction.STILL_KEYS))
        self.assertIn('placement', required)

    def test_shot_history_lists_instructions_and_prompt_changes(self):
        sequence_id = self.written()
        key = self.detail(sequence_id)['shots'][2]['key']
        url = f'/api/sequences/{sequence_id}/shots/{key}/history'
        first = self.client.get(url).json()
        self.assertEqual(([item['number'] for item in first['revisions']], first['revisions'][0]['diff'], first['pending'], first['proposals']), ([1], None, None, []))
        # A rewrite with an instruction: the history shows it while it waits, then diffs the new prompt against the old one.
        self.run_step(sequence_id, 'shots', shot_keys=[key], instruction='Make the light cooler.')
        self.assertEqual(self.client.get(url).json()['pending'], {'status': 'queued', 'instruction': 'Make the light cooler.', 'trigger': None})
        cooler = 'Cool blue evening light falls from the right while the shop windows glow.'
        self.answers['shot'] = lambda data: {**shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds']), 'light': cooler}
        self.assertEqual(self.work(), 1)
        history = self.client.get(url).json()
        newest, oldest = history['revisions']
        self.assertEqual((newest['number'], newest['origin'], newest['instruction'], newest['trigger'], newest['check_ok'], history['pending']), (2, 'model', 'Make the light cooler.', None, True, None))
        self.assertIsNone(oldest['diff'])
        self.assertEqual(sides(newest['diff']), (oldest['prompt'], newest['prompt']))
        self.assertIn(cooler, newest['prompt'])
        self.assertIn('Cool', ''.join(part['text'] for part in newest['diff'] if part['op'] == 'insert'))
        self.assertIn('warm', ''.join(part['text'] for part in newest['diff'] if part['op'] == 'delete'))
        # A hand edit is a revision of its own.
        result = self.detail(sequence_id)['shots'][2]['result']
        edited = self.client.put(f"/api/sequence-results/{result['id']}", json={'content': {**result['content'], 'framing_lock': 'Head and shoulders only.'}})
        self.assertEqual(edited.status_code, 200, edited.text)
        newest = self.client.get(url).json()['revisions'][0]
        self.assertEqual((newest['number'], newest['origin'], newest['instruction']), (3, 'manual_edit', None))
        self.assertIn('Head and shoulders only', ''.join(part['text'] for part in newest['diff'] if part['op'] == 'insert'))
        # A voice change rewrites the shot without an instruction; the history says why.
        saved = self.client.put(f'/api/sequences/{sequence_id}/voices', json={'shot_key': key, 'speaker': 'Hero', 'segments': [{'from': 0, 'delivery': 'narrative'}]})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertIsNone(self.client.get(url).json()['pending'])
        self.run_step(sequence_id, 'shots', shot_keys=[key])
        self.assertEqual(self.work(), 1)
        newest = self.client.get(url).json()['revisions'][0]
        self.assertEqual((newest['number'], newest['instruction'], newest['trigger']), (4, None, None))
        self.assertIsNone(self.detail(sequence_id)['shots'][2]['result']['trigger'])
        self.assertEqual(self.client.get(f'/api/sequences/missing/shots/{key}/history').status_code, 404)

    def test_a_boundary_inside_speech_can_be_moved_by_hand(self):
        self.interrupt()
        sequence_id = self.written()
        detail = self.detail(sequence_id)
        boundary = next(item for item in detail['timeline']['boundaries'] if item['id'] == 'K2')
        self.assertEqual((boundary['choice'], boundary['by'], boundary['reason'], boundary['speech']['lines']), ('keep_cut', 'director', 'The reply cuts in.', ['L3']))
        self.assertEqual((detail['shots'][1]['closes']['split'], detail['shots'][1]['closes']['reason']), ({'lines': ['L3'], 'before': '', 'after': 'ちょっと待って'}, 'The reply cuts in.'))
        url = f'/api/sequences/{sequence_id}/boundaries'
        moved = self.client.put(url, json={'after_cut_id': 'c2', 'choice': 'after_line'})
        self.assertEqual(moved.status_code, 200, moved.text)
        after = moved.json()
        self.assertEqual([(shot['start_time'], shot['end_time'], shot['result'] is None) for shot in after['shots']], [(0.0, 2.0, False), (2.0, 4.75, True), (4.75, 10.0, True)])
        self.assertEqual((after['plan']['origin'], after['plan']['content']['boundaries'], after['ready']), ('manual_edit', [{'after_cut_id': 'c2', 'choice': 'after_line', 'reason': '', 'by': 'user'}], False))
        self.assertEqual(next(item for item in after['timeline']['boundaries'] if item['id'] == 'K2')['by'], 'user')
        # Positions that are not offered, a hard cut without speech and a cut with no hard cut after it are refused.
        self.assertEqual(self.client.put(url, json={'after_cut_id': 'c1', 'choice': 'after_line'}).status_code, 422)
        self.assertEqual(self.client.put(url, json={'after_cut_id': 'c3', 'choice': 'keep_cut'}).status_code, 404)
        self.assertEqual(self.client.put(url, json={'after_cut_id': 'c2', 'choice': 'sideways'}).status_code, 422)
        self.assertEqual(self.client.post(f'/api/sequences/{sequence_id}/plan/merge', json={'shot_key': after['shots'][0]['key']}).status_code, 404)
        self.calls.clear()
        self.assertEqual(self.run_step(sequence_id, 'shots')['queued'], 2)
        self.work()
        second, third = [data for step, data in self.calls if step == 'shot']
        self.assertIn('runs 0.75 s past the end of its own cut to finish L3: from 2.00 s to the end it keeps this framing', second['shot']['closes']['how'])
        self.assertEqual((second['shot']['closes']['decided_by'], second['shot']['audio_from_cuts'], second['cuts'][1]), ('the user', [2, 3], {'cut': 3, 'plays_at': [2.0, 2.75], 'picture': 'not shown: only its audio plays in this shot'}))
        # The borrowed seconds keep this shot's own picture: the neighbour's on-screen facts are not offered as if it were shown.
        self.assertEqual(second['speakers'][0]['per_cut'][1], {'cut': 3, 'plays_at': [2.0, 2.75], 'picture': "not shown: only its audio is borrowed, and this shot's own picture goes on over these seconds"})
        self.assertIn('starts 0.75 s into its own cut, after L3, which the previous shot finished', third['shot']['opens']['how'])
        self.assertEqual(third['shot']['initial_frame']['is'][:30], 'the frame 0.8 s into that cut ')
        # Hand-edited plans follow the same rules: a merge is refused.
        plan = self.detail(sequence_id)['plan']
        merged = {**plan['content'], 'shots': [plan['content']['shots'][0], {**plan['content']['shots'][1], 'end_time': 10.0}]}
        refused = self.client.put(f"/api/sequence-results/{plan['id']}", json={'content': merged})
        self.assertEqual(refused.status_code, 422)
        self.assertIn('merges, drops or reorders cuts', refused.json()['detail'])

    def test_boundaries_placed_on_the_dialogue_timeline_are_final(self):
        self.interrupt()  # L3 (3.6-4.6 s) runs across the hard cut between cut 2 and cut 3
        sequence_id = self.create()
        url = f'/api/sequences/{sequence_id}/shot-points'
        # Before direction: shot 3 starts just before L3, so the whole phrase is in its shot.
        placed = self.client.put(url, json={'points': [{'after_cut_id': 'c2', 'time': 3.55}]})
        self.assertEqual(placed.status_code, 200, placed.text)
        boundary = next(item for item in placed.json()['timeline']['boundaries'] if item['id'] == 'K2')
        self.assertEqual((placed.json()['sequence']['shot_points'], boundary['fixed'], boundary['by']),
                         ([{'after_cut_id': 'c2', 'time': 3.55}], {'t': 3.55, 'time': 3.55, 'shift': -0.45}, 'user'))
        # The hard rules: no shot without its own picture, inside the sequence, only hard cuts of the sequence.
        refused = self.client.put(url, json={'points': [{'after_cut_id': 'c2', 'time': 1.0}]})
        self.assertEqual(refused.status_code, 422)
        self.assertIn('would show 0.00 s of its own picture', refused.json()['detail'])
        self.assertEqual(self.client.put(url, json={'points': [{'after_cut_id': 'c2', 'time': 11.0}]}).status_code, 422)
        self.assertEqual(self.client.put(url, json={'points': [{'after_cut_id': 'c3', 'time': 5.0}]}).status_code, 422)
        # The director does not decide a placed boundary; it is told about it.
        self.run_step(sequence_id, 'direct')
        self.work()
        plan_input = next(data for step, data in self.calls if step == 'plan')
        self.assertEqual(plan_input['decisions'], 'Nothing to decide: the user placed the boundary at every hard cut inside speech.')
        self.assertEqual(plan_input['fixed_by_user'][0]['id'], 'K2')
        self.assertIn('0.45 s before the hard cut, at 3.55', plan_input['fixed_by_user'][0]['boundary'])
        self.assertEqual((plan_input['fixed_by_user'][0]['speech'], plan_input['fixed_by_user'][0]['opening_frame'][:28]), ('keeps L3 whole in the shot of cut 3', 'the first frame of cut 3, he'))
        detail = self.detail(sequence_id)
        self.assertEqual([(shot['start_time'], shot['end_time']) for shot in detail['shots']], [(0.0, 2.0), (2.0, 3.55), (3.55, 10.0)])
        self.assertEqual((detail['plan']['content']['boundaries'], detail['shots'][2]['opens']['placed'], detail['shots'][2]['opens']['by']), ([], True, 'user'))
        # The Shots tab picker and the assistant cannot move it.
        moved = self.client.put(f'/api/sequences/{sequence_id}/boundaries', json={'after_cut_id': 'c2', 'choice': 'after_line'})
        self.assertEqual(moved.status_code, 422)
        self.assertIn('placed by you on the Dialogue timeline', moved.json()['detail'])
        self.run_step(sequence_id, 'shots')
        self.work()
        third = [data for step, data in self.calls if step == 'shot'][-1]
        self.assertIn('starts 0.45 s before its own cut, where the user placed the boundary, and carries L3 whole', third['shot']['opens']['how'])
        self.assertEqual(third['shot']['opens']['decided_by'], 'the user')
        # Moved after the plan exists: the plan follows in the same request and the two shots beside it must be written again.
        later = self.client.put(url, json={'points': [{'after_cut_id': 'c2', 'time': 4.7}]})
        self.assertEqual(later.status_code, 200, later.text)
        after = later.json()
        self.assertEqual([(shot['start_time'], shot['end_time'], shot['result'] is None) for shot in after['shots']], [(0.0, 2.0, False), (2.0, 4.7, True), (4.7, 10.0, True)])
        self.assertEqual((after['plan']['origin'], after['plan_problems'], after['ready']), ('manual_edit', [], False))
        # Handed back to the director: the app keeps the phrase with its visibly speaking speaker until the shots are planned again.
        released = self.client.put(url, json={'points': []}).json()
        self.assertEqual([(shot['start_time'], shot['end_time']) for shot in released['shots']], [(0.0, 2.0), (2.0, 4.75), (4.75, 10.0)])
        self.assertEqual(released['plan']['content']['boundaries'], [{'after_cut_id': 'c2', 'choice': 'after_line', 'reason': sequence_edit.RELEASED, 'by': 'app'}])
        self.assertIsNone(next(item for item in released['timeline']['boundaries'] if item['id'] == 'K2')['fixed'])
        self.run_step(sequence_id, 'plan')
        self.assertEqual(self.client.put(url, json={'points': [{'after_cut_id': 'c2', 'time': 3.55}]}).status_code, 200)
        self.work()
        self.assertEqual(self.detail(sequence_id)['sequence']['shot_points'], [{'after_cut_id': 'c2', 'time': 3.55}])

    def test_voices_are_decided_per_shot_and_a_rewrite_reads_them(self):
        sequence_id = self.written()
        shot = self.detail(sequence_id)['shots'][2]
        self.assertEqual(shot['speakers'], ['Hero'])
        self.assertEqual([(voice['speaker'], voice['delivery'], voice['fixed_by_user']) for voice in shot['result']['derived']['voices']], [('Hero', 'on_screen', False)])
        url = f'/api/sequences/{sequence_id}/voices'
        self.assertEqual(self.client.put(url, json={'shot_key': shot['key'], 'speaker': 'Ghost', 'segments': None}).status_code, 422)
        self.assertEqual(self.client.put(url, json={'shot_key': shot['key'], 'speaker': 'Hero', 'segments': [{'from': 0, 'delivery': 'lip_sync'}]}).status_code, 422)
        self.assertEqual(self.client.put(url, json={'shot_key': shot['key'], 'speaker': 'Hero', 'segments': [{'from': 9, 'delivery': 'narrative'}]}).status_code, 422)
        saved = self.client.put(url, json={'shot_key': shot['key'], 'speaker': 'Hero', 'segments': [{'from': 0, 'delivery': 'narrative'}, {'from': 1.5, 'delivery': 'inner_voice'}]})
        self.assertEqual(saved.status_code, 200, saved.text)
        after = saved.json()
        self.assertEqual(after['plan']['content']['voices'], [{'speaker': 'Hero', 'from': 4.0, 'to': 5.5, 'delivery': 'narrative'}, {'speaker': 'Hero', 'from': 5.5, 'to': 10.0, 'delivery': 'inner_voice'}])
        target = after['shots'][2]
        self.assertEqual((target['stale'], target['job']), (['plan'], None))
        self.assertEqual(target['fixed_voices'], [{'speaker': 'Hero', 'from': 0.0, 'to': 1.5, 'delivery': 'narrative'}, {'speaker': 'Hero', 'from': 1.5, 'to': 6.0, 'delivery': 'inner_voice'}])
        self.assertEqual([shot['stale'] for shot in after['shots'][:2]], [[], []])  # nothing leaks into the neighbouring shots
        self.run_step(sequence_id, 'shots', shot_keys=[target['key']])
        # The queued rewrite reads the next voice edit when it starts.
        again = self.client.put(url, json={'shot_key': shot['key'], 'speaker': 'Hero', 'segments': [{'from': 0, 'delivery': 'narrative'}, {'from': 1.5, 'delivery': 'off_screen'}]})
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual([job['step'] for job in self.jobs() if job['status'] == 'queued'], ['shot'])
        self.assertEqual(self.work(), 1)
        shot_call = [data for step, data in self.calls if step == 'shot'][-1]
        self.assertEqual(shot_call['speakers'][0]['fixed_by_user'], [{'from': 0.0, 'to': 1.5, 'delivery': 'narrative'}, {'from': 1.5, 'to': 6.0, 'delivery': 'off_screen'}])
        self.assertEqual(shot_call['speakers'][0]['per_cut'][0]['picture'], "this shot's picture")
        rewritten = self.detail(sequence_id)['shots'][2]
        self.assertEqual(rewritten['stale'], [])
        self.assertEqual([(voice['from'], voice['delivery'], voice['fixed_by_user']) for voice in rewritten['result']['derived']['voices']], [(0.0, 'narrative', True), (1.5, 'off_screen', True)])
        self.assertIn('At 00:01.000, an off-screen voice is heard: <d>[Japanese] じゃあね</d>', rewritten['result']['prompt'])  # L3 is heard 1.0-2.0 s into the shot
        # The review sees each shot's voices as one line, with the user's ranges marked.
        self.run_step(sequence_id, 'review')
        self.work()
        review_call = [data for step, data in self.calls if step == 'review'][-1]
        self.assertEqual(review_call['shots'][2]['voices'], 'Hero narrative 0-1.5 (user), off_screen 1.5-6 (user)')
        self.assertEqual(review_call['shots'][0]['voices'], 'Hero on_screen')
        # Planning again keeps the ranges, because they are stored in episode time.
        self.run_step(sequence_id, 'plan')
        self.work()
        replanned = self.detail(sequence_id)
        self.assertEqual((replanned['plan']['origin'], len(replanned['plan']['content']['voices'])), ('model', 2))
        cleared = self.client.put(url, json={'shot_key': replanned['shots'][2]['key'], 'speaker': 'Hero', 'segments': None})
        self.assertEqual(cleared.status_code, 200, cleared.text)
        self.assertNotIn('voices', cleared.json()['plan']['content'])

    def test_cut_changes_block_shots_until_the_sequence_is_directed_again(self):
        sequence_id = self.directed()
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=3 WHERE id='c2'")
            conn.execute("UPDATE cuts SET start_time=3 WHERE id='c3'")
        detail = self.detail(sequence_id)
        self.assertTrue(detail['sequence']['cuts_changed'])
        # The old plan still tiles the range in time, but its shots no longer follow the hard cuts.
        self.assertEqual((detail['plan_problems'], detail['shots'][1]['source_cuts']), ([], [2, 3]))
        direction_jobs = self.run_step(sequence_id, 'direct')['job_ids']
        [waiting] = self.run_step(sequence_id, 'shots')['job_ids']
        request = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (waiting,))['payload_json'])
        self.assertIn(direction_jobs[-1], request['depends_on'])
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
