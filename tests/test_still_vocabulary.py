import unittest
from unittest.mock import patch

from app import pose_detail, sequence_direction as direction, still_vocabulary as vocabulary
from app.assistant import skills


class StillVocabularyTest(unittest.TestCase):
    def test_the_dictionary_is_kept_as_written(self):
        data = vocabulary.load()
        self.assertEqual((data['version'], data['coordinate_system']), ('1.0.0', 'viewer_relative'))
        self.assertIn('three-quarter view toward screen-left', vocabulary.values('body.orientation'))
        self.assertEqual(vocabulary.values('body.nothing'), [])
        self.assertEqual(vocabulary.values('nothing.nothing'), [])

    def test_sections_that_must_never_reach_a_prompt_cannot_be_read(self):
        for path in ('visual_style.medium', 'visual_style.color_palette', 'fallback.x', 'evidence.visibility', 'reference.role', 'output.layout'):
            self.assertEqual(vocabulary.values(path), [], path)
        self.assertIn('anime', vocabulary.load()['visual_style']['medium'])  # still in the file for people who read it

    def test_every_listed_path_exists_and_nothing_else_is_offered(self):
        for part, paths in (*vocabulary.STILL_GUIDE, *vocabulary.BEATS_GUIDE):
            for path in paths:
                self.assertTrue(vocabulary.values(path), f'{part}: {path} is not in the dictionary')
                self.assertNotIn(path.split('.')[0], vocabulary.EXCLUDED)
        offered = ' '.join(vocabulary.shot_block().split('\n'))
        for style in ('photorealistic', 'anime', 'film grain', 'teal and orange'):
            self.assertNotIn(style, offered)
        for spoken in ('action.communication', 'mouth.activity', 'whispering', 'shouting'):  # who speaks is decided by the voices alone
            self.assertNotIn(spoken, offered)

    def test_no_suggested_word_trips_a_validator_of_the_shot_answer(self):
        for part, paths in (*vocabulary.STILL_GUIDE, *vocabulary.BEATS_GUIDE):
            for path in paths:
                for word in vocabulary.values(path):
                    for pattern, problem in direction.BANNED:
                        self.assertIsNone(pattern.search(word), f'{path}: "{word}" {problem}')

    def test_the_shot_writer_gets_the_keywords_per_part_of_its_answer(self):
        text = direction.SHOT_SYSTEM
        for needed in ('## KEYWORDS FOR THE STILL', '## KEYWORDS FOR THE BEATS', 'still.placement', '- body.orientation: facing the camera; slightly turned toward screen-left',
                       '- eyes.gaze_direction: directly into the camera', '- lighting.direction: frontal lighting', '- hand.gesture: open palm', 'Prefer these words when one of them fits',
                       'never take a medium, an art style or a rendering technique from a list'):
            self.assertIn(needed, text)
        self.assertLess(text.index('## KEYWORDS FOR THE STILL'), text.index('## OUTPUT'))
        self.assertNotIn('image right, head in profile', text)  # the prompt gives rules, not a sentence to copy

    def test_the_prompt_works_without_the_dictionary(self):
        with patch.object(vocabulary, 'load', return_value={}):
            self.assertEqual((vocabulary.still_block(), vocabulary.beats_block(), vocabulary.shot_block()), ('', '', ''))
            self.assertEqual(vocabulary.values('body.orientation'), [])

    def test_the_pose_block_is_described_in_the_dictionarys_own_words(self):
        orientation, yaw, pitch = vocabulary.values('body.orientation'), vocabulary.values('head.yaw'), vocabulary.values('head.pitch')
        gaze = [*vocabulary.values('eyes.gaze_direction'), *vocabulary.values('eyes.openness')]
        for degrees in range(-180, 181, 15):
            self.assertIn(pose_detail.torso_phrase(degrees), orientation, degrees)
            self.assertIn(pose_detail.head_phrase(degrees), yaw, degrees)
        for degrees in range(-90, 91, 15):
            phrase = pose_detail.pitch_phrase(degrees)
            self.assertTrue(phrase is None or phrase in pitch, degrees)
        for sector in (*pose_detail.GAZE_SECTORS, 'closed'):
            self.assertIn(pose_detail.gaze_phrase(sector), gaze, sector)
        self.assertIn(pose_detail.gaze_phrase('center', 'camera'), gaze)
        targets = vocabulary.values('eyes.gaze_target')
        for kind in ('camera', 'off_screen', 'none'):
            self.assertIn(pose_detail.target_phrase({'gaze_target': kind}), targets, kind)
        self.assertIsNone(pose_detail.torso_phrase(None))
        self.assertIsNone(pose_detail.gaze_phrase('not_visible'))
        self.assertEqual(pose_detail.pitch_phrase(-30), 'chin lowered')
        self.assertEqual(pose_detail.torso_phrase(-45), 'three-quarter view toward screen-left')
        self.assertEqual(pose_detail.head_phrase(90), 'side profile facing screen-right')

    def test_the_still_skill_carries_the_keywords_for_the_assistant(self):
        skill = skills.get('still_image_prompt')
        self.assertNotIn('{{still_keywords}}', skill['body'])
        self.assertIn('- eyes.gaze_direction:', skill['body'])
        self.assertIn('Prefer these words', skill['body'])


if __name__ == '__main__':
    unittest.main()
