import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pose_schema as ps  # noqa: E402
import scoring  # noqa: E402

SAMPLE = {'id': 's01', 'marked': [{'handle': 'B1', 'name': 'Mai Sakurajima', 'box': [50, 100, 300, 900]}, {'handle': 'B2', 'name': 'Sakuta Azusagawa', 'box': [600, 100, 900, 900]}],
          'stored': {'first_frame': {'characters': [
              {'character': 'Mai Sakurajima', 'action': 'sitting, talking to Sakuta Azusagawa', 'gaze': 'at Sakuta Azusagawa', 'emotion': 'mild worry', 'expression': 'eyes wide, mouth slightly open'},
              {'character': 'Sakuta Azusagawa', 'action': 'standing, looking at the camera', 'gaze': 'at the camera', 'emotion': 'calm, neutral', 'expression': 'mouth closed'}]},
                     'video': {'characters': [{'character': 'Mai Sakurajima', 'activity': 'standing → walking', 'speaking': 'yes', 'motion': 'holds still'}]}}}


def pose(**changes):
    value = {'body_yaw': 0, 'body_pose': 'sitting', 'head_yaw': 45, 'head_pitch': 0, 'head_roll': 'neutral', 'eye_gaze': 'right', 'eye_openness': 'wide', 'gaze_target': 'B2',
             'mouth_state': 'open', 'eyebrow_state': 'raised', 'facial_expression': 'worried', 'camera_elevation': 'eye_level', 'shot_size': 'medium'}
    value.update(changes)
    return value


def frame_call(approach='strict_detailed', variant='normal', answer=None, ok=True, **extra):
    return {'key': f'frame|{approach}|s01|{variant}', 'stage': 'frame', 'approach': approach, 'sample': 's01', 'variant': variant, 'ok': ok, 'first_try_ok': ok,
            'seconds': 40.0, 'attempts': [{'n': 1, 'seconds': 40.0, 'prompt_tokens': 3000, 'completion_tokens': 700, 'errors': [] if ok else ['bad']}], 'answer': answer, **extra}


class MirrorTest(unittest.TestCase):
    def test_a_consistent_flip_passes_and_a_wrong_side_fails(self):
        one = {'character_id': 'B1', **pose(head_yaw=45, eye_gaze='right', head_roll='tilt_left', gaze_target='B2')}
        two = {'character_id': 'B2', **pose(head_yaw=-90, eye_gaze='left', gaze_target='B1')}
        normal = frame_call(answer=[one, two])
        # flipped: B1 is now the old B2 and vice versa
        flipped_answer = [{'character_id': 'B1', **pose(head_yaw=90, eye_gaze='right', gaze_target='B2')},
                          {'character_id': 'B2', **pose(head_yaw=-45, eye_gaze='left', head_roll='tilt_right', gaze_target='B1')}]
        mirror = frame_call(variant='mirror', answer=flipped_answer, mirror_map={'B1': 'B2', 'B2': 'B1'})
        checks = scoring.mirror_checks(normal, mirror)
        self.assertTrue(checks and all(c['status'] == 'pass' for c in checks), [c for c in checks if c['status'] != 'pass'])
        flipped_answer[1]['head_yaw'] = 45  # the model failed to negate
        failed = [c for c in scoring.mirror_checks(normal, mirror) if c['status'] == 'fail']
        self.assertEqual([c['detail'].split(':')[0] for c in failed], ['head_yaw'])

    def test_abstentions_and_failed_calls_are_skipped(self):
        normal = frame_call(answer=[{'character_id': 'B1', **pose(head_yaw=None, eye_gaze='unclear')}])
        mirror = frame_call(variant='mirror', answer=[{'character_id': 'B1', **pose(head_yaw=None, eye_gaze='unclear')}], mirror_map={'B1': 'B1'})
        names = {c['detail'].split(':')[0] for c in scoring.mirror_checks(normal, mirror)}
        self.assertNotIn('head_yaw', names)
        self.assertNotIn('eye_gaze', names)
        self.assertEqual(scoring.mirror_checks(normal, frame_call(variant='mirror', ok=False)), [])

    def test_shot_size_may_move_one_step(self):
        self.assertTrue(scoring._consistent('shot_size', 'medium', 'medium_long'))
        self.assertFalse(scoring._consistent('shot_size', 'medium', 'long'))
        self.assertTrue(scoring._consistent('head_yaw', 180, -180))


class GeometryTest(unittest.TestCase):
    def test_gaze_side_uses_box_positions(self):
        good = scoring.geometry_checks(SAMPLE, [{'character_id': 'B1', **pose(head_yaw=45, eye_gaze='right', gaze_target='B2')}])
        self.assertEqual([c['status'] for c in good if c['name'] == 'gaze_side'], ['pass'])
        bad = scoring.geometry_checks(SAMPLE, [{'character_id': 'B1', **pose(head_yaw=-45, eye_gaze='left', gaze_target='B2')}])
        self.assertEqual([c['status'] for c in bad if c['name'] == 'gaze_side'], ['fail'])
        mixed = scoring.geometry_checks(SAMPLE, [{'character_id': 'B1', **pose(head_yaw=45, eye_gaze='left', gaze_target='B2')}])
        self.assertEqual([c['status'] for c in mixed if c['name'] == 'gaze_side'], ['fail'])
        unknown = scoring.geometry_checks(SAMPLE, [{'character_id': 'B1', **pose(head_yaw=None, eye_gaze='unclear', gaze_target='B2')}])
        self.assertEqual([c for c in unknown if c['name'] == 'gaze_side'], [])

    def test_camera_gaze_and_pitch_and_body(self):
        names = lambda entry: {c['name']: c['status'] for c in scoring.pose_checks('B1', entry)}
        self.assertEqual(names(pose(gaze_target='camera', head_yaw=0, eye_gaze='center'))['camera_gaze'], 'pass')
        self.assertEqual(names(pose(gaze_target='camera', head_yaw=90, eye_gaze='right'))['camera_gaze'], 'fail')
        self.assertEqual(names(pose(head_pitch=-45, eye_gaze='up'))['pitch_gaze'], 'fail')
        self.assertEqual(names(pose(head_pitch=-45, eye_gaze='down'))['pitch_gaze'], 'pass')
        self.assertNotIn('pitch_gaze', names(pose(head_pitch=0, eye_gaze='up')))
        self.assertEqual(names(pose(head_yaw=135, body_yaw=-45))['head_body'], 'fail')

    def test_shot_size_against_box_and_shared_elevation(self):
        sample = {**SAMPLE, 'marked': [{'handle': 'B1', 'name': 'Mai Sakurajima', 'box': [50, 40, 300, 960]}, SAMPLE['marked'][1]]}
        tall = scoring.geometry_checks(sample, [{'character_id': 'B1', **pose(shot_size='extreme_long', gaze_target='none')}])
        self.assertEqual([c['status'] for c in tall if c['name'] == 'shot_size_box'], ['fail'])
        fine = scoring.geometry_checks(sample, [{'character_id': 'B1', **pose(shot_size='medium', gaze_target='none')}])
        self.assertEqual([c['status'] for c in fine if c['name'] == 'shot_size_box'], ['pass'])
        middle = scoring.geometry_checks(SAMPLE, [{'character_id': 'B1', **pose(shot_size='extreme_long', gaze_target='none')}])
        self.assertEqual([c for c in middle if c['name'] == 'shot_size_box'], [])  # a box 80% tall says nothing
        shared = scoring.geometry_checks(SAMPLE, [{'character_id': 'B1', **pose(camera_elevation='low_angle', gaze_target='none')}, {'character_id': 'B2', **pose(camera_elevation='eye_level', gaze_target='none')}])
        self.assertEqual({c['status'] for c in shared if c['name'] == 'elevation_shared'}, {'fail'})


class AgreementTest(unittest.TestCase):
    def test_words_and_families(self):
        self.assertEqual(scoring.pose_word('sitting, talking'), 'sitting')
        self.assertEqual(scoring.pose_word('Crouching low'), 'other')
        self.assertIsNone(scoring.pose_word('talking to Mai'))
        self.assertEqual(scoring.expression_families('mild worry', 'smile'), {'worried', 'happy'})
        self.assertEqual(scoring.expression_families('not readable', ''), set())
        self.assertEqual(scoring.mouth_expected('Mouth slightly open, blush'), {'slightly_open', 'open', 'wide_open'})
        self.assertIsNone(scoring.mouth_expected('mouth open then closed mouth'))
        self.assertEqual(scoring.eyes_expected('eyes wide, brows up'), {'wide'})

    def test_stored_target_needs_exactly_one_other_character(self):
        names = {'B1': 'Mai Sakurajima', 'B2': 'Sakuta Azusagawa'}
        self.assertEqual(scoring.stored_target('at Sakuta Azusagawa', 'Mai Sakurajima', names), 'B2')
        self.assertEqual(scoring.stored_target('looks at Sakuta', 'Mai Sakurajima', names), 'B2')
        self.assertEqual(scoring.stored_target('at the camera', 'Mai Sakurajima', names), 'camera')
        self.assertIsNone(scoring.stored_target('down at the desk', 'Mai Sakurajima', names))
        self.assertIsNone(scoring.stored_target('at Mai Sakurajima', 'Mai Sakurajima', names))

    def test_frame_agreement_against_the_stored_result(self):
        answer = [{'character_id': 'B1', **pose(body_pose='sitting', gaze_target='B2', facial_expression='worried', eye_openness='wide', mouth_state='slightly_open')},
                  {'character_id': 'B2', **pose(body_pose='sitting', gaze_target='B1', facial_expression='happy', mouth_state='closed')}]
        statuses = {(c['character'], c['name']): c['status'] for c in scoring.frame_agreement(SAMPLE, answer)}
        self.assertEqual(statuses[('B1', 'body_pose')], 'pass')
        self.assertEqual(statuses[('B1', 'gaze_target')], 'pass')
        self.assertEqual(statuses[('B1', 'facial_expression')], 'pass')
        self.assertEqual(statuses[('B1', 'eye_openness')], 'pass')
        self.assertEqual(statuses[('B2', 'body_pose')], 'fail')
        self.assertEqual(statuses[('B2', 'gaze_target')], 'fail')
        self.assertEqual(statuses[('B2', 'facial_expression')], 'fail')

    def test_video_agreement_and_stability(self):
        start, end = pose(body_pose='standing', head_yaw=0, mouth_state='closed'), pose(body_pose='walking', head_yaw=90, mouth_state='closed')
        window = {'index': 0, 'ok': True, 'answer': [{'character': 'Mai Sakurajima', 'start': start, 'end': end}]}
        by_name = {c['name']: c['status'] for c in scoring.video_checks(SAMPLE, [window])}
        self.assertEqual(by_name['body_pose'], 'pass')
        self.assertEqual(by_name['speaking'], 'fail')  # stored says speaking, both mouths closed
        self.assertEqual(by_name['still_stable'], 'fail')  # stored says it holds still, the head turned 90 degrees


class ScoreTest(unittest.TestCase):
    def test_scores_rank_a_reliable_fast_approach_above_a_failing_slow_one(self):
        good = [frame_call('strict_detailed', answer=[{'character_id': 'B1', **pose(gaze_target='B2')}, {'character_id': 'B2', **pose(gaze_target='B1', head_yaw=-45, eye_gaze='left')}])]
        bad = [frame_call('full_confidence', ok=False, answer=None, seconds=120.0)]
        bad[0]['attempts'] = [{'n': i, 'seconds': 40.0, 'errors': ['x'], 'completion_tokens': 5} for i in (1, 2, 3)]
        result = scoring.score_all([SAMPLE], good + bad)
        a3, a1 = result['approaches']['strict_detailed'], result['approaches']['full_confidence']
        self.assertGreater(a3['final'], a1['final'])
        self.assertEqual(a1['summary']['frame']['hard_failures'], 1)
        self.assertEqual(a1['summary']['frame']['validator_failures'], 3)
        self.assertEqual(a1['summary']['frame']['retries'], 2)
        self.assertEqual(a3['summary']['frame']['first_try_rate'], 1.0)
        self.assertIsNone(a3['stage_scores']['video'])  # nothing ran in the video stage yet: it is not scored as zero
        self.assertEqual(a3['final'], a3['stage_scores']['frame']['score'])
        self.assertAlmostEqual(sum(scoring.WEIGHTS.values()), 1.0)

    def test_coverage_counts_abstentions_and_omissions(self):
        entry = {'character_id': 'B1', **pose(head_yaw=None, eye_gaze='unclear')}
        call = frame_call(answer=[entry])
        sample = {**SAMPLE, 'marked': SAMPLE['marked'][:1]}
        summary = scoring.score_all([sample], [call])['approaches']['strict_detailed']['summary']['frame']
        self.assertAlmostEqual(summary['coverage'], 11 / 13)
        self.assertEqual(summary['fields']['head_yaw']['abstained'], 1)

    def test_top5_choices_and_calibration(self):
        entry = {'character_id': 'B1', 'head_yaw': 45, 'eye_gaze': 'right', 'gaze_target': 'B2', 'body_pose': 'sitting', 'facial_expression': 'worried',
                 'confidence': {'head_yaw': 0.9, 'eye_gaze': 0.8, 'gaze_target': 0.7, 'body_pose': 0.6, 'facial_expression': 0.5}}
        call = frame_call('top5_confidence', answer=[entry])
        result = scoring.score_all([{**SAMPLE, 'marked': SAMPLE['marked']}], [call])['approaches']['top5_confidence']
        self.assertEqual(result['a2_choices']['head_yaw'], 1)
        self.assertNotIn('shot_size', result['a2_choices'])
        self.assertGreater(result['summary']['frame']['calibration']['n_pass'], 0)

    def test_agreement_between_approaches(self):
        one = frame_call('full_confidence', answer=[{'character_id': 'B1', **pose(head_yaw=45)}])
        two = frame_call('strict_detailed', answer=[{'character_id': 'B1', **pose(head_yaw=45, body_pose='walking')}])
        matrix = scoring.agreement_between([SAMPLE], {'full_confidence': [one], 'strict_detailed': [two]})
        rate = matrix['full_confidence']['strict_detailed']
        self.assertEqual(rate['n'], 13)
        self.assertAlmostEqual(rate['rate'], 12 / 13)


if __name__ == '__main__':
    unittest.main()
