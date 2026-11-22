"""Rendering a directed shot as an H3 prompt, and the validators that guard the director's answers."""
import copy
import unittest

import tests.test_sequence_pipeline as pipeline_tests
from app import sequence_direction as direction, sequence_evidence, sequence_prompt, sequence_timeline
from app.assistant import h3_prompt

BIBLE = {'spoken_language': 'Japanese'}
ROSTER = [{'name': 'Mai Sakurajima', 'character_context_id': 'mai'}, {'name': 'Sakuta Azusagawa', 'character_context_id': 'sakuta'}]
SCENARIO = {'id': 'station', 'title': 'Fujisawa Station', 'content': 'A platform.'}


def shot(duration=8.0, index=3):
    return {'index': index, 'key': 'k', 'duration': duration, 'generate_seconds': max(duration, 5.0), 'trim_seconds': duration if duration < 5 else None,
            'frame': {'cut': 1, 'cut_id': 'c1', 'time': 0.0, 'strategy': 'cut_first_frame', 'continues_previous_shot': False}}


def line(line_id, t0, t1, text, name=None):
    return {'line': line_id, 't0': t0, 't1': t1, 'text': text, 'speaker_label': 'SPEAKER_01', 'character_context_id': None, 'character_name': name,
            'attribution': 'transcript' if name else 'unknown', 'crosses_start': False, 'crosses_end': False}


DIALOGUE = [line('L4', 0.5, 2.0, 'おはよう', 'Mai Sakurajima'), line('L5', 4.5, 6.0, '遅いよ'), line('L6', 6.5, 7.5, '……', 'Sakuta Azusagawa')]


def answer():
    content = pipeline_tests.shot_answer(seconds=8.0)
    content['subjects'] = [{'name': 'mai sakurajima', 'anchors': 'long black hair, school blazer', 'keep': 'face and uniform'},
                           {'name': 'Sakuta Azusagawa', 'anchors': 'short brown hair, school blazer', 'keep': 'face and uniform'},
                           {'name': 'the station attendant', 'anchors': 'navy cap and uniform', 'keep': 'cap and uniform'}]
    content['picture_line'] = 'Mai Sakurajima stands on the platform of Fujisawa Station facing Sakuta Azusagawa.'
    content['summary'] = 'Mai greets Sakuta on the platform while the camera pushes in.'
    content['shot_line'] = 'A medium two-shot at eye level: Mai Sakurajima on the left and Sakuta Azusagawa on the right of the platform at Fujisawa station, the station attendant far behind.'
    content['beats'] = [{'end': 4.0, 'action': content['beats'][0]['action'].replace('Hero', 'Mai')}, {'end': 8.0, 'action': content['beats'][1]['action'].replace('Hero', 'Sakuta')}]
    content['camera'] = {'move': 'push_in', 'amplitude': 'small', 'speed': 'slow', 'detail': "It ends on a medium close-up of Mai's face."}
    content['voices'] = [{'speaker': 'Mai Sakurajima', 'from': 0, 'delivery': 'on_screen', 'subject': 'Mai Sakurajima'}, {'speaker': 'SPEAKER_01', 'from': 0, 'delivery': 'off_screen', 'subject': None},
                         {'speaker': 'sakuta azusagawa', 'from': 0, 'delivery': 'inner_voice', 'subject': None}]
    content['scenario_anchors'] = 'grey platform, yellow tactile strip, overhead signs left unmarked'
    content['scenario_keep'] = 'platform layout'
    return content


class RenderTest(unittest.TestCase):
    def build(self, content=None, scenario=SCENARIO, target=None, dialogue=DIALOGUE):
        normal = direction.validate_shot(content or answer(), target or shot(), BIBLE, ROSTER, scenario, dialogue)
        return normal, *sequence_prompt.build(target or shot(), normal, BIBLE, ROSTER, scenario, dialogue)

    def test_prompt_follows_the_h3_contract_with_app_written_labels_times_and_words(self):
        normal, prompt, verdict, derived = self.build()
        self.assertTrue(verdict['ok'], verdict['errors'])
        self.assertEqual((verdict['stats']['mode'], verdict['stats']['seconds'], verdict['stats']['pictures'], verdict['stats']['audio'], verdict['stats']['subjects']), ('omni_reference', 8.0, [1, 2, 3, 4], [1], [1, 2, 3, 4]))
        self.assertEqual(h3_prompt.parse(prompt)['order'], list(h3_prompt.SECTIONS))
        self.assertEqual([subject['name'] for subject in normal['subjects']], ['Mai Sakurajima', 'Sakuta Azusagawa', 'the station attendant'])  # record names are canonical
        definitions = h3_prompt.parse(prompt)['sections']['subject_definitions'].splitlines()
        self.assertEqual(definitions[0], '<Picture 1> is the first frame of [Shot 1]: Mai Sakurajima stands on the platform of Fujisawa Station facing Sakuta Azusagawa.')
        self.assertEqual(definitions[1], '<Subject 1> is Mai Sakurajima, whose face, hair and clothing come from <Picture 2>: long black hair, school blazer.')
        self.assertEqual(definitions[3], '<Subject 3> is the station attendant, seen in <Picture 1>: navy cap and uniform.')  # nobody without a record gets a sheet
        self.assertEqual(definitions[4], '<Subject 4> is Fujisawa Station, the setting from <Picture 4>: grey platform, yellow tactile strip, overhead signs left unmarked.')
        self.assertEqual(definitions[5], '<Audio 1> is the original sound of this shot, “shot-03.wav”: its voices, sound and timing to follow.')
        self.assertEqual(h3_prompt.subject_names(prompt), {1: 'Mai Sakurajima', 2: 'Sakuta Azusagawa', 3: 'station attendant', 4: 'Fujisawa Station'})  # the checker reads names without a leading "the"
        description = h3_prompt.parse(prompt)['sections']['detailed_description']
        self.assertTrue(description.startswith('The shot keeps the look of <Picture 1> unchanged'))
        self.assertNotRegex(prompt, r'(?i)anime|2D |target video')
        # Names in prose become labels: full names, single name words, people without a record and the place.
        self.assertIn('[Shot 1] The shot begins from <Picture 1>: A medium two-shot at eye level: <Subject 1> on the left and <Subject 2> on the right of the platform at <Subject 4>, <Subject 3> far behind.', description)
        self.assertNotRegex(description.replace('Mai Sakurajima', ''), r'\b(?:Mai|Sakuta|Hero)\b')
        self.assertIn("The camera pushes in with small amplitude at slow speed. It ends on a medium close-up of <Subject 1>'s face.", description)
        self.assertIn('00:00.000-00:04.000:', description)
        self.assertIn('00:04.000-00:08.000:', description)
        first, second = [row for row in description.splitlines() if row.startswith('00:')]
        self.assertIn('At 00:00.500, <Subject 1> (S1) says: <d>[Japanese] おはよう</d>', first)
        self.assertIn('At 00:04.500, an off-screen voice is heard: <d>[Japanese] 遅いよ</d>', second)
        self.assertIn('At 00:06.500, the voice of <Subject 2> is heard while its lips stay closed: <d>[Japanese] ……</d>', second)
        self.assertEqual(h3_prompt.speaking_subjects(prompt), {1})
        self.assertIn('<Subject 2> keeps the lips closed while its voice is heard. <Subject 3> does not speak.', description)  # the lips clause is for named characters
        self.assertNotIn('<Subject 4> does not speak', description)
        soundscape = h3_prompt.parse(prompt)['sections']['overall_soundscape']
        self.assertIn('<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:00.500 to 00:02.000.', soundscape)
        self.assertIn('An off-screen voice speaks the words of <Audio 1> from 00:04.500 to 00:06.000; no lips on screen move for them.', soundscape)
        self.assertIn('The words of <Audio 1> from 00:06.500 to 00:07.500 are the voice of <Subject 2>; its lips stay closed.', soundscape)
        self.assertNotIn('<d>', soundscape)
        # Nobody heard the audio: the sound section only says to follow it, never what it sounds like.
        self.assertTrue(soundscape.startswith(sequence_prompt.SOUND_SENTENCE + ' '))
        self.assertNotIn('soundscape', normal)
        self.assertEqual(h3_prompt.parse(prompt)['sections']['non_diegetic_music'], 'N/A')
        self.assertEqual([(item['label'], item['role']) for item in derived['references']],
                         [('<Picture 1>', 'initial_frame'), ('<Picture 2>', 'character_sheet'), ('<Picture 3>', 'character_sheet'), ('<Picture 4>', 'scenario'), ('<Audio 1>', 'original_audio')])
        self.assertEqual([(item['line'], item['delivery'], item['subject']) for item in derived['dialogue']], [('L4', 'on_screen', 1), ('L5', 'off_screen', None), ('L6', 'inner_voice', 2)])
        self.assertEqual([(item['speaker'], item['from'], item['to'], item['delivery'], item['subject']) for item in normal['voices']],
                         [('Mai Sakurajima', 0.0, 8.0, 'on_screen', 'Mai Sakurajima'), ('SPEAKER_01', 0.0, 8.0, 'off_screen', None), ('Sakuta Azusagawa', 0.0, 8.0, 'inner_voice', 'Sakuta Azusagawa')])
        self.assertIn('Low warm sun from the left, long soft shadows.', derived['still_prompt'])
        self.assertIn('A single frozen frame with no text, subtitles, captions, logos, overlays or watermarks.', derived['still_prompt'])

    def test_a_speaker_can_change_delivery_inside_a_shot_and_the_user_ranges_win(self):
        content = answer()
        content['voices'][0:1] = [{'speaker': 'Mai Sakurajima', 'from': 0, 'delivery': 'narrative', 'subject': 'Mai Sakurajima'},
                                  {'speaker': 'Mai Sakurajima', 'from': 3.0, 'delivery': 'on_screen', 'subject': 'Mai Sakurajima'}]
        dialogue = [*DIALOGUE, line('L7', 3.2, 3.9, 'またね', 'Mai Sakurajima')]
        normal, prompt, verdict, derived = self.build(content, dialogue=dialogue)
        self.assertTrue(verdict['ok'], verdict['errors'])
        description = h3_prompt.parse(prompt)['sections']['detailed_description']
        self.assertIn('At 00:00.500, the voice of <Subject 1> plays over the picture without lip sync: <d>[Japanese] おはよう</d>', description)
        self.assertIn('At 00:03.200, <Subject 1> (S1) says: <d>[Japanese] またね</d>', description)
        self.assertIn('<Subject 1> moves the lips only while its own on-screen words are heard.', description)
        soundscape = h3_prompt.parse(prompt)['sections']['overall_soundscape']
        self.assertIn('The words of <Audio 1> from 00:00.500 to 00:02.000 are the voice of <Subject 1>, too far from the camera for lip sync', soundscape)
        self.assertIn('<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:03.200 to 00:03.900.', soundscape)
        self.assertEqual([(item['from'], item['to'], item['delivery']) for item in derived['voices'] if item['speaker'] == 'Mai Sakurajima'], [(0.0, 3.0, 'narrative'), (3.0, 8.0, 'on_screen')])
        # The user decided Mai is off screen from 2 s: that range replaces the director's, whatever the model wrote.
        fixed = {**shot(), 'fixed_voices': [{'speaker': 'Mai Sakurajima', 'from': 2.0, 'to': 8.0, 'delivery': 'off_screen'}]}
        normal, prompt, verdict, derived = self.build(content, target=fixed, dialogue=dialogue)
        self.assertEqual([(item['from'], item['delivery'], bool(item.get('fixed'))) for item in normal['voices'] if item['speaker'] == 'Mai Sakurajima'], [(0.0, 'narrative', False), (2.0, 'off_screen', True)])
        self.assertIn('At 00:03.200, an off-screen voice is heard: <d>[Japanese] またね</d>', prompt)
        self.assertEqual(derived['dialogue'][-1]['fixed_by_user'], True)
        # A user range that needs a visible speaker sends the director back to put that speaker in the picture.
        visible = {**shot(), 'fixed_voices': [{'speaker': 'SPEAKER_01', 'from': 0.0, 'to': 8.0, 'delivery': 'on_screen'}]}
        self.assertIn('the user set SPEAKER_01 on_screen from 0 s to 8 s', self.errors(lambda c: None, target=visible))
        narrator = [line('L8', 1.0, 2.0, '昔々', 'Narrator')]
        self.assertIn('a narrator is off_screen or narrative', self.errors(lambda c: c.update(voices=[{'speaker': 'Narrator', 'from': 0, 'delivery': 'on_screen', 'subject': 'Mai Sakurajima'}]), dialogue=narrator))

    def test_a_guessed_ambience_never_reaches_the_prompt(self):
        content = answer()
        content['soundscape'] = 'Quiet classroom room tone with the hum of a study area.'
        normal, prompt, verdict, derived = self.build(content)
        self.assertTrue(verdict['ok'], verdict['errors'])
        self.assertNotIn('soundscape', normal)
        self.assertNotIn('classroom', prompt)
        legacy = {**normal, 'soundscape': 'Quiet classroom room tone with the hum of a study area.'}  # a direction stored before the change
        self.assertNotIn('classroom', sequence_prompt.build(shot(), legacy, BIBLE, ROSTER, SCENARIO, DIALOGUE)[0])
        self.assertNotIn('soundscape', direction.shot_format(shot(), ROSTER, DIALOGUE)['json_schema']['schema']['properties'])

    def test_shots_written_with_per_line_speech_still_render(self):
        content = answer()
        del content['voices']
        content['speech'] = [{'line': 'L4', 'delivery': 'lip_sync', 'subject': 'Mai Sakurajima'}, {'line': 'L6', 'delivery': 'narration', 'subject': None}]
        content['subjects'][0]['name'] = 'Mai Sakurajima'
        subjects, _ = sequence_prompt.assign(content, ROSTER, SCENARIO)
        lines = sequence_prompt.spoken(DIALOGUE, content, subjects, 8.0)
        self.assertEqual([(item['line'], item['delivery'], item['subject']) for item in lines], [('L4', 'on_screen', 1), ('L5', 'off_screen', None), ('L6', 'narrative', None)])

    def test_a_silent_shot_without_people_or_scenario(self):
        content = answer()
        content.update(subjects=[], voices=[], picture_line='An empty platform at dusk.', shot_line='A wide shot of the empty platform, tracks running to the right.')
        content['beats'] = [{'end': 8.0, 'action': ' '.join(['Wind moves a paper cup along the yellow strip and the light slowly dims on the far wall.'] * 9)}]
        content['camera'] = {'move': 'static', 'amplitude': 'large', 'speed': 'fast', 'detail': 'ignored for a static shot'}
        normal, prompt, verdict, derived = self.build(content, scenario=None, dialogue=[])
        self.assertTrue(verdict['ok'], verdict['errors'])
        self.assertEqual((normal['camera']['detail'], normal['scenario_anchors']), ('', ''))
        self.assertIn('The camera holds a static shot.', prompt)
        self.assertEqual(h3_prompt.parse(prompt)['sections']['overall_soundscape'], sequence_prompt.SOUND_SENTENCE + ' ' + sequence_prompt.NO_SPEECH)
        self.assertNotIn('carries no speech', prompt)  # untranscribed background voices are kept, never declared absent
        self.assertIn('its sound and timing to follow', prompt)
        self.assertEqual([item['label'] for item in derived['references']], ['<Picture 1>', '<Audio 1>'])

    def template(self, content=None, dialogue=DIALOGUE, scenario=SCENARIO, bible=BIBLE):
        normal, prompt, verdict, derived = self.build(content, scenario=scenario, dialogue=dialogue)
        return prompt, *sequence_prompt.template(shot(), normal, bible, derived['subjects'], derived['dialogue'], prompt)

    def test_the_sound_section_keeps_every_sound_of_the_original_and_times_actions_to_it(self):
        _, prompt, verdict, _ = self.build()
        soundscape = h3_prompt.parse(prompt)['sections']['overall_soundscape']
        for kept in ('completely, exactly as recorded', 'secondary voices', 'indistinct background talk', 'Nothing in it is removed, muted, cleaned up or replaced',
                     'Each visible action that makes a sound lands exactly on that sound in <Audio 1>.'):
            self.assertIn(kept, soundscape)
        self.assertTrue(verdict['ok'], verdict['errors'])
        # The director is told to put an action's own contact sound in its beat, and only that.
        self.assertIn('Sound of actions:', direction.SHOT_SYSTEM)
        self.assertIn('the only sound a beat names is the sound of its own action', direction.SHOT_SYSTEM)

    def test_a_prompt_written_with_the_earlier_sound_wording_keeps_its_template(self):
        normal, prompt, verdict, derived = self.build()
        older = sequence_prompt.fill(sequence_prompt.render(shot(), normal, BIBLE, derived['subjects'], derived['dialogue'], sequence_prompt.line_slots(derived['dialogue']),
                                                            sequence_prompt.LEGACY_SOUND),
                                     sequence_prompt.slot_values(derived['dialogue'], sequence_prompt.line_slots(derived['dialogue']), BIBLE))
        self.assertIn('Follow <Audio 1> exactly for all sound', older)
        text, _ = sequence_prompt.template(shot(), normal, BIBLE, derived['subjects'], derived['dialogue'], older)
        self.assertIn('Follow <Audio 1> exactly for all sound', text)
        self.assertIsNotNone(sequence_prompt.template(shot(), normal, BIBLE, derived['subjects'], derived['dialogue'], prompt)[0])
        self.assertIsNone(sequence_prompt.template(shot(), normal, BIBLE, derived['subjects'], derived['dialogue'], prompt.replace('pushes in', 'pulls out'))[0])

    def test_prompt_template_replaces_language_words_and_line_times_only(self):
        prompt, text, slots = self.template()
        self.assertEqual(slots, {'L4': '1.1', 'L5': '1.2', 'L6': '1.3'})
        self.assertIsNotNone(text)
        self.assertIn('At {start 1.1}, <Subject 1> (S1) says: <d>[{language}] {line 1.1}</d>', text)
        self.assertIn('At {start 1.3}, the voice of <Subject 2> is heard while its lips stay closed: <d>[{language}] {line 1.3}</d>', text)
        self.assertIn('lips in sync, from {start 1.1} to {end 1.1}.', text)
        self.assertIn('00:00.000-00:04.000:', text)  # beat times are not line times
        self.assertNotIn('おはよう', text)
        self.assertNotIn('Japanese', text)
        self.assertEqual(sequence_prompt.fill(text, sequence_prompt.slot_values(DIALOGUE, slots, BIBLE)), prompt)
        dubbed = sequence_prompt.fill(text, {'language': 'English', 'line 1.1': 'Morning.', 'start 1.1': '00:00.700'})
        self.assertIn('At 00:00.700, <Subject 1> (S1) says: <d>[English] Morning.</d>', dubbed)
        self.assertIn('{line 1.2}', dubbed)  # a placeholder without a value stays

    def test_template_numbers_lines_by_start_and_joins_runs_of_one_voice(self):
        dialogue = [line('L9', 3.0, 3.8, 'まって', 'Mai Sakurajima'), line('L8', 0.5, 2.0, 'おはよう', 'Mai Sakurajima')]
        dialogue[1]['crosses_start'] = True  # the end of a line the previous shot began still gets its own slot here
        content = answer()
        content['voices'] = content['voices'][:1]
        prompt, text, slots = self.template(content, dialogue=dialogue)
        self.assertEqual(slots, {'L8': '1.1', 'L9': '1.2'})
        self.assertIn('lips in sync, from {start 1.1} to {end 1.2}.', text)  # one run of Mai's voice range
        self.assertEqual(sequence_prompt.fill(text, sequence_prompt.slot_values(dialogue, slots, BIBLE)), prompt)

    def test_silent_shot_template_is_the_prompt_and_a_mismatch_gives_none(self):
        content = answer()
        content.update(subjects=[], voices=[], picture_line='An empty platform at dusk.', shot_line='A wide shot of the empty platform.')
        prompt, text, slots = self.template(content, dialogue=[], scenario=None)
        self.assertEqual((text, slots), (prompt, {}))
        self.assertNotIn('{', text)
        # A prompt that the stored direction no longer renders (another spoken language in the bible now) has no template.
        prompt, text, slots = self.template(bible={'spoken_language': 'Korean'})
        self.assertIsNone(text)
        self.assertEqual(slots, {'L4': '1.1', 'L5': '1.2', 'L6': '1.3'})

    def test_only_four_character_sheets_are_attached(self):
        roster = [{'name': f'Person{number}', 'character_context_id': f'id{number}'} for number in range(1, 6)]
        content = {'subjects': [{'name': person['name'], 'anchors': 'a', 'keep': 'b'} for person in roster]}
        subjects, references = sequence_prompt.assign(content, roster, SCENARIO)
        self.assertEqual([subject['picture'] for subject in subjects], [2, 3, 4, 5, None, 6])
        self.assertEqual([item['role'] for item in references].count('character_sheet'), sequence_prompt.MAX_SHEETS)

    def errors(self, change, **kwargs):
        content = answer()
        change(content)
        with self.assertRaises(ValueError) as raised:
            direction.validate_shot(content, kwargs.get('target') or shot(), BIBLE, ROSTER, kwargs.get('scenario', SCENARIO), kwargs.get('dialogue', DIALOGUE))
        return str(raised.exception)

    def test_validator_messages_tell_the_director_what_to_fix(self):
        self.assertIn('beat 2 contains the word "cut"', self.errors(lambda c: c['beats'][1].update(action='A quick cut to her hands.')))
        self.assertIn('summary contains a reference label', self.errors(lambda c: c.update(summary='<Subject 1> waves.')))
        self.assertIn('shot_line quotes dialogue', self.errors(lambda c: c.update(shot_line='She says <d>hello</d>.')))
        self.assertIn('beat 1 contains a timecode', self.errors(lambda c: c['beats'][0].update(action='At 00:01.500 she turns.')))
        self.assertIn('voices needs an entry from second 0 for SPEAKER_01, who speaks L5 in this shot', self.errors(lambda c: c['voices'].pop(1)))
        self.assertIn('voices Mai Sakurajima: on_screen needs "subject" to be one of your subjects (Mai Sakurajima, Sakuta Azusagawa, the station attendant)', self.errors(lambda c: c['voices'][0].update(subject='Kaede')))
        self.assertIn('voices: "Kaede" is not heard in this shot', self.errors(lambda c: c['voices'].append({'speaker': 'Kaede', 'from': 0, 'delivery': 'off_screen', 'subject': None})))
        self.assertIn('"from" 12 s is outside the shot', self.errors(lambda c: c['voices'].append({'speaker': 'SPEAKER_01', 'from': 12, 'delivery': 'narrative', 'subject': None})))
        self.assertIn('voices SPEAKER_01: inner_voice needs "subject"', self.errors(lambda c: c['voices'][1].update(delivery='inner_voice')))
        self.assertIn('camera.move must be one of: static, push_in', self.errors(lambda c: c['camera'].update(move='zoom')))
        self.assertIn('beat 2 ends at 20 s but the shot lasts 8 s', self.errors(lambda c: c['beats'][1].update(end=20)))
        self.assertIn('beat 2 ends at 3 s, which is not after the previous beat (4 s)', self.errors(lambda c: c['beats'][1].update(end=3)))
        self.assertIn('"scenario_anchors" is empty', self.errors(lambda c: c.update(scenario_anchors='')))
        self.assertIn('"picture_line" is empty', self.errors(lambda c: c.update(picture_line=' ')))
        self.assertIn('subject "the station attendant" needs anchors', self.errors(lambda c: c['subjects'][2].update(anchors='')))
        thin = self.errors(lambda c: c.update(beats=[{'end': 8.0, 'action': 'She waves.'}], shot_line='Two people.', framing_lock='', light=''))
        self.assertIn('shot_line, framing_lock, light and beats together have', thin)
        several = self.errors(lambda c: (c.update(summary='A cut.'), c['camera'].update(move='zoom')))
        self.assertTrue(several.startswith('(1) ') and '(2) ' in several)

    def test_no_field_names_a_style_or_medium(self):
        self.assertIn('keep_picture names a visual style or medium (animation)', self.errors(lambda c: c.update(keep_picture='composition and palette, identical throughout the animation')))
        self.assertIn('still.lighting names a visual style or medium (animation)', self.errors(lambda c: c['still'].update(lighting="Soft daylight typical of the series' animation style.")))
        self.assertIn('(anime, cel-shaded)', self.errors(lambda c: c['still'].update(subjects='Mai, not anime and not cel-shaded.')))
        self.assertIn('subject Mai Sakurajima names a visual style or medium (illustrated)', self.errors(lambda c: c['subjects'][0].update(anchors='long black hair, illustrated look')))
        acting = answer()
        acting['beats'][0]['action'] += ' Her brows are drawn together beside the painted wall.'
        direction.validate_shot(acting, shot(), BIBLE, ROSTER, SCENARIO, DIALOGUE)  # everyday words for acting and settings stay allowed
        self.assertIn('not even to rule it out', direction.SHOT_SYSTEM)

    def test_a_generated_still_names_the_depth_layer_and_screen_side_of_each_person(self):
        def validate(placement, generated=True):
            content = answer()
            content['still']['placement'] = placement
            return direction.validate_shot(content, shot(), BIBLE, ROSTER, SCENARIO, DIALOGUE, generated=generated)
        flat = 'Mai stands screen-left facing Sakuta, who stands screen-right with his head turned toward her.'
        with self.assertRaisesRegex(ValueError, r'still.placement places Mai Sakurajima, Sakuta Azusagawa without their depth'):
            validate(flat)
        validate(flat, generated=False)  # a hand edit of an older shot is not held to it
        validate('Mai stands in the foreground screen-left; Sakuta stands in the background screen-right, head turned toward her.')
        validate('Mai and Sakuta share the midground, Mai screen-left and Sakuta screen-right.')
        with self.assertRaisesRegex(ValueError, 'gives no screen position'):
            validate('Mai Sakurajima stands in the foreground facing the camera.')
        validate('The platform is empty.')  # nobody placed, nothing to require
        for rule in ('who is in the foreground, who is in the midground (only when three or more people stand at different depths) and who is in the background',
                     'the outfit only as a very short reference of a few words', 'The character sheet sent with the still shows the full outfit'):
            self.assertIn(rule, direction.SHOT_SYSTEM)

    def test_bible_and_review_validators(self):
        ctx = {'characters': [{'id': 'mai', 'name': 'Mai Sakurajima', 'notes': ''}], 'scenarios': [{'id': 'station', 'title': 'Fujisawa Station', 'content': ''}], 'results': {}}
        timeline = {'cuts': [{'id': 'c1', 'number': 7, 't0': 0, 't1': 4}, {'id': 'c2', 'number': 8, 't0': 4, 't1': 9}],
                    'lines': [{'id': 'L1', 't0': 1, 't1': 2, 'text': 'a', 'speaker_label': 'SPEAKER_00', 'character_name': None, 'character_context_id': None}]}
        bible = copy.deepcopy(pipeline_tests.BIBLE)
        bible['scenes'] = [{'from_cut': 7, 'to_cut': 7, 'location': 'Platform', 'scenario': 'fujisawa station'}, {'from_cut': 8, 'to_cut': 8, 'location': 'Train', 'scenario': 'null'}]
        bible['characters'] = [{'name': 'MAI SAKURAJIMA', 'anchors': 'long black hair'}]
        bible['outfits'] = [{**bible['outfits'][0], 'character': 'mai sakurajima'}]
        bible['voices'] = [{'speaker_label': 'SPEAKER_00', 'character': None, 'kind': 'narrator', 'confidence': 'high', 'reason': 'Narration.'}]
        normal = direction.validate_bible(bible, ctx, timeline)
        self.assertEqual([(scene['id'], scene['scenario'], scene['scenario_context_id']) for scene in normal['scenes']], [('S1', 'Fujisawa Station', 'station'), ('S2', None, None)])
        self.assertEqual((normal['characters'][0]['name'], normal['characters'][0]['character_context_id']), ('Mai Sakurajima', 'mai'))
        self.assertEqual([(item['character'], item['character_context_id'], item['status']) for item in normal['outfits']], [('Mai Sakurajima', 'mai', 'new')])
        self.assertEqual((normal['voices'][0]['kind'], normal['voices'][0]['character'], normal['voices'][0]['source']), ('narrator', None, 'director'))

        def problems(change):
            wrong = copy.deepcopy(bible)
            change(wrong)
            with self.assertRaises(ValueError) as raised:
                direction.validate_bible(wrong, ctx, timeline)
            return str(raised.exception)
        self.assertIn('scenes must cover cuts 7 to 8', problems(lambda b: b['scenes'][1].update(from_cut=9, to_cut=9)))
        self.assertIn('they must cover every cut up to 8', problems(lambda b: b['scenes'].pop()))
        self.assertIn('is not a scenario note; use one of: Fujisawa Station, or null', problems(lambda b: b['scenes'][0].update(scenario='Enoshima')))
        self.assertIn('has no character record; use exact names from: Mai Sakurajima', problems(lambda b: b['characters'][0].update(name='Kaede')))
        self.assertIn('voice SPEAKER_00: "Kaede" has no character record', problems(lambda b: b['voices'][0].update(character='Kaede')))
        self.assertIn('"spoken_language" must be the language name in English letters', problems(lambda b: b.update(spoken_language='日本語')))
        review = direction.validate_review({'verdict': 'Fine.', 'issues': [{'shot': 2, 'severity': 'urgent', 'problem': 'Coat changes.'}, {'shot': 5, 'problem': 'x'}, 'junk']}, 3)
        self.assertEqual(review, {'verdict': 'Fine.', 'issues': [{'shot': 2, 'severity': 'check', 'problem': 'Coat changes.', 'suggestion': ''}]})

    def test_the_review_sees_where_speech_is_and_where_there_is_none(self):
        from app import sequence_evidence
        edge = {'how': 'keep_cut'}
        base = {'t0': 0.0, 't1': 4.0, 'cut': 7, 'scene': 'S1', 'intent': 'x', 'opens': {'how': 'start'}, 'closes': edge, 'duration': 4.0}
        shots = [{**base, 'index': 1, 'key': 'a', 'lines': [line('L1', 0.5, 2.0, 'おはよう', 'Mai Sakurajima')]}, {**base, 'index': 2, 'key': 'b', 'lines': []}]
        rows = sequence_evidence.review_input({'voices': []}, shots, {})['shots']
        self.assertEqual(rows[0]['lines'], ['L1 0.5-2.0 SPEAKER_01: おはよう'])
        self.assertEqual(rows[1]['lines'], 'none')
        for rule in ('has no voice to add', 'hairpins', 'Never "check", "verify" or "consider"'):
            self.assertIn(rule, direction.REVIEW_SYSTEM)

    def test_answer_formats_constrain_ids_to_what_the_app_offers(self):
        timeline = {'boundaries': [{'id': 'K1', 'speech': None}, {'id': 'K2', 'speech': {'lines': ['L3']}}], 'cuts': [{'number': 7}, {'number': 8}, {'number': 9}]}
        plan = direction.plan_format(timeline, {'scenes': [{'id': 'S1'}]})['json_schema']['schema']['properties']
        decision = plan['decisions']['items']['properties']
        self.assertEqual((decision['id']['enum'], decision['choice']['enum'], plan['decisions']['minItems'], plan['decisions']['maxItems']), (['K2'], ['keep_cut', 'before_line', 'after_line'], 1, 1))
        self.assertEqual((plan['cuts']['items']['properties']['cut']['enum'], plan['cuts']['minItems'], plan['cuts']['maxItems']), ([7, 8, 9], 3, 3))
        silent = direction.plan_format({'boundaries': [{'id': 'K1', 'speech': None}], 'cuts': [{'number': 1}, {'number': 2}]}, {})['json_schema']['schema']['properties']['decisions']
        self.assertEqual(silent['maxItems'], 0)
        voices = direction.shot_format(shot(), ROSTER, DIALOGUE)['json_schema']['schema']['properties']['voices']['items']['properties']
        self.assertEqual((voices['speaker']['enum'], voices['delivery']['enum']), (['Mai Sakurajima', 'SPEAKER_01', 'Sakuta Azusagawa'], ['on_screen', 'off_screen', 'inner_voice', 'narrative']))
        for step in direction.STEPS.values():
            self.assertIn('Return JSON only.', step['system'])


def plan_timeline(points=None):
    """Cuts 1-3 (0-5, 5-6.6, 6.6-12 s): A's line crosses the first hard cut, B's the second."""
    cut = lambda cut_id, index, start, end: {'id': cut_id, 'cut_index': index, 'start_time': start, 'end_time': end, 'first_frame_time': start}
    line = lambda start, end, text, speaker: {'start_time': start, 'end_time': end, 'text': text, 'speaker_label': speaker, 'character_context_id': None, 'character_name': None,
                                               'parts': [{'start_time': start, 'end_time': end, 'text': text}]}
    return sequence_timeline.build([cut('a', 0, 0.0, 5.0), cut('b', 1, 5.0, 6.6), cut('c', 2, 6.6, 12.0)], [line(3.6, 5.3, 'あのね', 'A'), line(6.2, 7.0, 'うん', 'B')], [], points=points)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.timeline = plan_timeline()
        self.cuts = [{'cut': number, 'intent': 'Show it.'} for number in (1, 2, 3)]

    def problems(self, decisions, cuts=None):
        with self.assertRaises(ValueError) as caught:
            direction.validate_plan({'decisions': decisions, 'cuts': self.cuts if cuts is None else cuts, 'notes': ''}, self.timeline, {})
        return str(caught.exception)

    def test_plan_answers_are_checked_with_actionable_messages(self):
        good = direction.validate_plan({'decisions': [{'id': 'K1', 'choice': 'after_line', 'reason': ''}, {'id': 'K2', 'choice': 'keep_cut', 'reason': 'B answers over the picture of A.'}],
                                        'cuts': list(reversed(self.cuts)), 'notes': 'Even.'}, self.timeline, {})
        self.assertEqual(good['decisions'], [{'id': 'K1', 'choice': 'after_line', 'reason': '', 'by': 'director'}, {'id': 'K2', 'choice': 'keep_cut', 'reason': 'B answers over the picture of A.', 'by': 'director'}])
        self.assertEqual([item['cut'] for item in good['cuts']], [1, 2, 3])
        content = direction.plan_content(good, self.timeline, {'scenes': [{'id': 'S1', 'from_cut': 1, 'to_cut': 3}]})
        self.assertEqual([(span['start_time'], span['end_time'], span['frame_cut_id'], span['scene']) for span in content['shots']], [(0.0, 5.45, 'a', 'S1'), (5.45, 6.6, 'b', 'S1'), (6.6, 12.0, 'c', 'S1')])
        self.assertEqual(content['boundaries'][0], {'after_cut_id': 'a', 'choice': 'after_line', 'reason': '', 'by': 'director'})
        self.assertIn('answer every decision: K2 is missing', self.problems([{'id': 'K1', 'choice': 'keep_cut', 'reason': 'x'}]))
        self.assertIn('K1: keep_cut divides L1; give the reason the scene needs it divided at the hard cut, or keep the phrase whole with before_line or after_line',
                      self.problems([{'id': 'K1', 'choice': 'keep_cut', 'reason': ''}, {'id': 'K2', 'choice': 'keep_cut', 'reason': 'x'}]))
        self.assertIn('decision "K7" is not one of the listed decisions (K1, K2)', self.problems([{'id': 'K7', 'choice': 'keep_cut', 'reason': 'x'}]))
        self.assertIn('cut 2 keeps only 0.60 s of its 1.60 s picture (K1 after_line, K2 before_line)',
                      self.problems([{'id': 'K1', 'choice': 'after_line', 'reason': ''}, {'id': 'K2', 'choice': 'before_line', 'reason': ''}]))
        self.assertIn('give every cut an intent: cut 3 has none', self.problems([{'id': 'K1', 'choice': 'keep_cut', 'reason': 'x'}, {'id': 'K2', 'choice': 'keep_cut', 'reason': 'x'}], self.cuts[:2]))

    def test_the_app_completes_a_plan_the_director_could_not_answer(self):
        # Both moves together would empty cut 2: the first choice stays, and the app keeps the second phrase whole on its other side.
        answer = {'decisions': [{'id': 'K1', 'choice': 'after_line', 'reason': ''}, {'id': 'K2', 'choice': 'before_line', 'reason': ''}], 'cuts': self.cuts[:1]}
        completed = direction.complete_plan(answer, self.timeline, {}, {'results': {}})
        self.assertEqual([(item['id'], item['choice'], item['by']) for item in completed['decisions']], [('K1', 'after_line', 'director'), ('K2', 'after_line', 'app')])
        self.assertEqual([item['intent'] for item in completed['cuts']], ['Show it.', 'Follow the source.', 'Follow the source.'])
        self.assertEqual(sequence_timeline.check_spans(direction.plan_content(completed, self.timeline, {})['shots'], self.timeline), [])
        # Nothing usable: the app keeps each phrase whole with the smaller move where it fits.
        fallback = direction.complete_plan('nonsense', self.timeline, {}, {'results': {}})
        self.assertEqual([(item['id'], item['choice'], item['reason']) for item in fallback['decisions']],
                         [('K1', 'after_line', 'Chosen by the app: keeps the phrase whole.'), ('K2', 'after_line', 'Chosen by the app: keeps the phrase whole.')])
        # ... or on the side where its speaker is visibly speaking.
        speaking = {'results': {'b': {'video_analysis': {'content': {'direction': {'characters': [{'character': 'B', 'character_context_id': 'b-id', 'speaking': 'yes'}]}}}}}}
        bible = {'voices': [{'speaker_label': 'B', 'character': 'B', 'character_context_id': 'b-id'}]}
        self.assertEqual(direction.default_choice(self.timeline['boundaries'][1], speaking, self.timeline, bible), 'after_line')


    def test_the_director_leaves_a_placed_boundary_alone(self):
        # The user put K1 1.5 s before its hard cut, just before A's phrase.
        timeline = plan_timeline([{'after_cut_id': 'a', 'time': 3.5}])
        self.assertEqual([item['id'] for item in direction.open_boundaries(timeline)], ['K2'])
        self.assertEqual(direction.plan_format(timeline, {})['json_schema']['schema']['properties']['decisions']['items']['properties']['id']['enum'], ['K2'])
        # An answer for K1 is ignored rather than refused; K2 is still the director's.
        good = direction.validate_plan({'decisions': [{'id': 'K1', 'choice': 'keep_cut', 'reason': 'x'}, {'id': 'K2', 'choice': 'after_line', 'reason': ''}], 'cuts': self.cuts, 'notes': ''}, timeline, {})
        self.assertEqual([item['id'] for item in good['decisions']], ['K2'])
        content = direction.plan_content(good, timeline, {})
        self.assertEqual([(span['start_time'], span['end_time']) for span in content['shots']], [(0.0, 3.5), (3.5, 7.15), (7.15, 12.0)])
        self.assertEqual([item['after_cut_id'] for item in content['boundaries']], ['b'])
        completed = direction.complete_plan('nonsense', timeline, {}, {'results': {}})
        self.assertEqual([item['id'] for item in completed['decisions']], ['K2'])
        edges = sequence_timeline.expand(content['shots'], timeline)[1]['opens']
        self.assertEqual((edges['placed'], edges['by'], edges['keeps']), (True, 'user', ['L1']))
        self.assertEqual(sequence_evidence.edge_brief(edges), 'placed by user -1.50 s keeping L1')
        self.assertEqual(sequence_evidence.edge_brief({**edges, 'how': 'keep_cut', 'shift': 0.0, 'keeps': None}), 'placed by user on the hard cut')
        for rule in ('fixed_by_user', 'Never answer a hard cut listed in fixed_by_user'):
            self.assertIn(rule, direction.PLAN_SYSTEM)
        self.assertIn('An edge the user placed (decided_by the user) is exact and final', direction.SHOT_SYSTEM)
        self.assertIn('placed by user', direction.REVIEW_SYSTEM)


if __name__ == '__main__':
    unittest.main()
