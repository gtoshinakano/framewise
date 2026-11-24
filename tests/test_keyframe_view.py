"""The keyframe analysis as prompts read it, and as the readers written for the video analysis see it."""
import unittest

from app import composition_brief, keyframe_view, prompt_context, sequence_evidence
from tests.test_performance_beats import KEYFRAMES

BEATS = {'contract': 'beats-v1', 'duration': 2.0, 'summary': 'Hero greets Mia.', 'based_on': {'keyframe_analysis': 'kf'},
         'beats': [{'start': 0.0, 'end': 2.0, 'camera': {'move': 'static', 'detail': ''}, 'intent': 'greets',
                    'characters': [{'character': 'Hero', 'character_context_id': 'ctx', 'body': 'stands', 'face': 'smiles', 'gaze': 'at Mia', 'mouth': 'speaking'}],
                    'speech': [{'line': 'S1', 'speaker': 'Hero', 'character_context_id': 'ctx', 'start': 0.2, 'end': 0.9, 'delivery': 'on_screen', 'suggested': 'on_screen'}]}],
         'speech': [], 'checks': []}


class ViewTest(unittest.TestCase):
    def test_review_view_has_states_and_tagged_changes_without_ids(self):
        view = prompt_context.keyframe_view(KEYFRAMES, 'review')
        self.assertEqual(view['frames'][0], {'frame': 'F1', 't': 0.0, 'characters': [{'name': 'Hero', 'state': 'standing, mouth closed', 'body_measured': 'standing upright'}]})
        first = view['changes'][0]
        self.assertEqual((first['pair'], first['at'], first['camera']), ('F1→F2', '0.00-0.80s', 'static'))
        self.assertEqual(first['characters'][0]['changes'], ['head: turns toward screen-right (vision model)', 'arms: right arm raised to shoulder height (keypoints)'])
        self.assertEqual(first['characters'][0]['visible'], 'speaking')
        self.assertNotIn('character_context_id', str(view))

    def test_beats_view_adds_the_measured_body_and_brief_view_only_the_changes(self):
        self.assertEqual(prompt_context.keyframe_view(KEYFRAMES, 'beats')['frames'][0]['characters'][0]['body_measured'], 'standing upright')
        brief = prompt_context.keyframe_view(KEYFRAMES, 'brief')
        self.assertNotIn('frames', brief)
        self.assertEqual(brief['changes'][0]['characters'][0]['changes'][0], 'head: turns toward screen-right')

    def test_beats_view(self):
        view = prompt_context.beats_view(BEATS)
        beat = view['beats'][0]
        self.assertEqual((beat['at'], beat['intent'], beat['characters'][0]['name'], beat['speech'][0]['delivery']), ('0.00-2.00s', 'greets', 'Hero', 'on_screen'))
        self.assertNotIn('character_context_id', str(view))

    def test_other_content_passes_through(self):
        self.assertEqual(prompt_context.keyframe_view({'windows': []}), {'windows': []})
        self.assertIsNone(keyframe_view.direction_from_keyframes({'windows': []}))


class AdapterTest(unittest.TestCase):
    def test_the_adapter_is_a_motion_direction(self):
        direction = keyframe_view.direction_from_keyframes(KEYFRAMES, BEATS)
        self.assertTrue(prompt_context.is_direction(direction))
        self.assertEqual(direction['direction']['summary'], 'Hero greets Mia.')
        self.assertEqual(direction['direction']['camera']['movement'], 'static → pan left')
        hero, = direction['direction']['characters']
        self.assertEqual((hero['character'], hero['character_context_id'], hero['speaking'], hero['activity']), ('Hero', 'ctx', 'yes', 'standing, mouth closed'))
        self.assertEqual(hero['motion'], 'right arm raised to shoulder height')
        self.assertEqual(hero['end'], 'standing, mouth open')
        view = prompt_context.video_view(direction)
        self.assertEqual(view['events'][0], {'seconds': '0.0-0.8', 'subject': 'Hero', 'action': 'mouth moves (speaking)'})
        self.assertIn({'seconds': '0.8-1.6', 'subject': 'camera', 'action': 'camera: pan left'}, view['events'])

    def test_older_readers_work_through_with_motion(self):
        by_type = {'keyframe_analysis': {'id': 'kf', 'content': KEYFRAMES}, 'performance_beats': {'id': 'b', 'content': BEATS},
                   'video_analysis': {'id': 'old', 'content': {'windows': []}}, 'initial_frame': {'id': 'f', 'content': {'mode': 'full'}}}
        motion = prompt_context.with_motion(by_type)
        self.assertEqual((motion['video_analysis']['id'], motion['video_analysis']['source_type']), ('kf', 'keyframe_analysis'))
        direction = motion['video_analysis']['content']['direction']
        shot = composition_brief.direction_camera(direction)
        self.assertEqual((shot['camera'], shot['direction']), ('pan', 'left'))  # the first real move of 'static → pan left'
        self.assertTrue(composition_brief.motion_fields(direction['characters'][0]))
        beats, partial = composition_brief.video_beats(motion['video_analysis']['content'], 0, 2)
        self.assertFalse(partial)
        self.assertTrue(beats)
        self.assertTrue(sequence_evidence.is_ready(motion))

    def test_beats_are_current_only_for_their_keyframes(self):
        current = {'keyframe_analysis': {'id': 'kf'}, 'performance_beats': {'content': BEATS}}
        self.assertTrue(prompt_context.beats_current(current))
        self.assertFalse(prompt_context.beats_current({**current, 'keyframe_analysis': {'id': 'kf2'}}))
        older = prompt_context.motion_source({'keyframe_analysis': {'id': 'kf2', 'content': KEYFRAMES}, 'performance_beats': {'content': BEATS}})
        self.assertEqual(older['content']['direction']['summary'], 'Hero turns and talks.')  # stale beats give no summary

    def test_a_cut_without_keyframes_keeps_its_video_analysis(self):
        video = {'direction': {'summary': 'old'}, 'windows': []}
        self.assertEqual(prompt_context.motion_source({'video_analysis': {'id': 'v', 'content': video}}), {'type': 'video_analysis', 'id': 'v', 'content': video})
        self.assertIsNone(prompt_context.motion_source({}))

    def test_slim_results_send_the_new_results_and_drop_the_old_motion(self):
        rows = [{'title': 'Keyframe analysis', 'content': KEYFRAMES}, {'title': 'Performance beats', 'content': BEATS},
                {'title': 'Video analysis', 'content': {'windows': []}}, {'title': 'Last frame description', 'content': {'a': 1}}]
        slim = prompt_context.slim_results(rows)
        self.assertEqual([row['title'] for row in slim], ['Keyframe analysis', 'Performance beats'])
        self.assertEqual(slim[0]['content']['kind'], keyframe_view.KEYFRAME_VIEW)
        self.assertEqual(prompt_context.slim_results(slim), slim)  # slimming twice is stable


if __name__ == '__main__':
    unittest.main()
