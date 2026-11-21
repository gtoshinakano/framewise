import unittest

from app import pose_detail, prompt_context, sequence_direction as direction, sequence_evidence as evidence, sequence_prompt

MAI = {'ref': 'B1', 'character_context_id': 'mai', 'name': 'Mai', 'box': [10, 100, 300, 900]}
RIO = {'ref': 'B2', 'character_context_id': 'rio', 'name': 'Rio', 'box': [600, 100, 900, 900]}


def pose(**changes):
    return pose_detail.resolve({'body_pose': 'sitting', 'body_yaw': 45, 'head_yaw': 45, 'head_pitch': 0, 'eye_gaze': 'right', 'gaze_target': 'B2', 'gaze_object': '', **changes}, MAI, [MAI, RIO])


FRAME = {'composition': 'MS', 'visible_characters': [{'character': 'Mai', 'character_context_id': 'mai', 'position': 'left', 'pose_detail': pose()},
                                                     {'character': 'Rio', 'character_context_id': 'rio', 'position': 'right'}]}


class SequencePosesTest(unittest.TestCase):
    def test_the_still_prompt_places_and_aims_the_characters_when_the_shot_says_where(self):
        content = {'still': {'shot': 'Medium shot', 'placement': 'Mai left foreground, torso in three-quarter view toward the image right, eyes on Rio.', 'subjects': 'Mai: calm', 'setting': 'A room', 'lighting': 'Soft'}}
        text = sequence_prompt.still_prompt(content, {})
        self.assertIn('Mai left foreground, torso in three-quarter view toward the image right, eyes on Rio.', text)
        self.assertLess(text.index('Mai: calm'), text.index('A room'))
        older = sequence_prompt.still_prompt({'still': {key: value for key, value in content['still'].items() if key != 'placement'}}, {})
        self.assertNotIn('eyes on Rio', older)  # a shot written before the key still renders

    def test_the_shot_writer_is_told_to_use_the_facts_and_to_build_beats_on_the_clock(self):
        for needed in ('pose_detail', 'state_correction', 'placement:', 'never as degrees', 'Anchor every beat to the seconds', 'Position precisely', 'Name every object'):
            self.assertIn(needed, direction.SHOT_SYSTEM)
        self.assertEqual(direction.STILL_KEYS[:2], ('shot', 'placement'))

    def test_the_review_cares_about_state_facing_and_gaze_and_not_about_shades(self):
        for needed in ('opens_with and ends_with', 'a cut is a jump', 'brown-red against grey-red', 'never an issue unless it is a different garment'):
            self.assertIn(needed, direction.REVIEW_SYSTEM)

    def test_pose_facts_shrink_with_the_detail_level(self):
        view = prompt_context.frame_view(FRAME)
        full = evidence._without_ids(view, 0)['visible_characters'][0]['pose_detail']
        self.assertEqual(set(full), {'state', 'torso', 'head', 'eyes', 'looking_at'})
        self.assertEqual(set(evidence._without_ids(view, 2)['visible_characters'][0]['pose_detail']), {'state', 'looking_at'})
        self.assertNotIn('pose_detail', evidence._without_ids(view, 3)['visible_characters'][0])
        self.assertNotIn('character_context_id', evidence._without_ids(view, 0)['visible_characters'][0])
        self.assertNotIn('pose_detail', evidence._without_ids(view, 0)['visible_characters'][1])  # a character without a block adds nothing

    def test_the_review_gets_how_each_shot_opens_and_ends(self):
        direction_part = {'characters': [{'character': 'Mai', 'end': 'stands at the door, facing left, hand on the handle'}, {'character': 'Rio', 'end': ''}]}
        ctx = {'results': {'c1': {'initial_frame': {'content': FRAME}, 'video_analysis': {'content': {'direction': direction_part}}}}}
        shots = [{'index': 1, 'cut_id': 'c1', 'frame': {'strategy': 'cut_first_frame'}, 'part': None},
                 {'index': 2, 'cut_id': 'c1', 'frame': {'strategy': 'later_moment'}, 'part': (1, 2)},
                 {'index': 3, 'cut_id': 'c1', 'frame': {'strategy': 'later_moment'}, 'part': (2, 2)}]
        states = evidence.shot_states(ctx, shots)
        self.assertEqual(states[1]['opens_with'], ['Mai: sitting; torso three-quarter view toward screen-right; head three-quarter view toward screen-right; looking at Rio'])
        self.assertEqual(states[1]['ends_with'], ['Mai: stands at the door, facing left, hand on the handle'])  # Rio has no end text
        self.assertNotIn(2, states)  # neither the opening frame nor the end of its cut is this shot's
        self.assertNotIn('opens_with', states[3])
        self.assertIn('ends_with', states[3])
        bible = {'voices': [], 'characters': []}
        row = [{'index': 1, 'key': 'k', 't0': 0, 't1': 2, 'cut': 1, 'scene': None, 'intent': 'x', 'opens': {'how': 'start'}, 'closes': {'how': 'end'}, 'lines': []}]
        self.assertEqual(evidence.review_input(bible, row, {}, 0, states)['shots'][0]['opens_with'], states[1]['opens_with'])
        self.assertNotIn('opens_with', evidence.review_input(bible, row, {}, 3, states)['shots'][0])  # the barest level drops them

    def test_the_next_shots_opening_carries_the_pose_reading(self):
        brief = pose_detail.brief(FRAME['visible_characters'][0]['pose_detail'])
        self.assertEqual(brief, 'sitting; torso three-quarter view toward screen-right; head three-quarter view toward screen-right; looking at Rio')
        self.assertEqual(pose_detail.brief(None), '')
        self.assertEqual(pose_detail.brief({'state': 'sitting'}), '')


if __name__ == '__main__':
    unittest.main()
