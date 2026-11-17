import json
import unittest
from app.visual_contracts import (FRAME_INSTRUCTION, SUBTITLE_INSTRUCTION, VIDEO_INSTRUCTION, cut_direction, earlier_note, frame_schema, full_last_frame,
                                  link_video, person_context, resolve_frame, roster_note, speech_note, subtitle_note, validate_video, video_roster)

PEOPLE = [{'ref': 'B1', 'detection_id': 'd-mai', 'character_context_id': 'mai', 'name': 'Mai', 'box': [10, 20, 300, 900]},
          {'ref': 'B2', 'detection_id': 'd-sakuta', 'character_context_id': 'sakuta', 'name': 'Sakuta', 'box': [500, 20, 800, 900]}]


def frame(**changes):
    return {'composition': 'MS, eye level', 'setting': 'classroom', 'lighting': 'soft daylight', 'camera': 'level', 'visible_characters': [],
            'background_people': 'none', 'notes': 'none', **changes}


def entry(box, **fields):
    return {'box': box, 'appearance': 'a', 'position': 'left', 'pose': 'sitting', 'expression': 'calm', 'emotion': 'content', 'gaze': 'at the book', 'action': 'sitting, reading', **fields}


class VisualContractsTest(unittest.TestCase):
    def test_frame_answers_are_resolved_to_real_records(self):
        raw = frame(visible_characters=[entry('B2', pose='standing'), entry('B1')])
        result = resolve_frame('```json\n' + json.dumps(raw) + '\n```', PEOPLE)
        self.assertEqual([item['character'] for item in result['visible_characters']], ['Mai', 'Sakuta'])  # box order, not the model's order
        mai = result['visible_characters'][0]
        self.assertEqual((mai['character'], mai['character_context_id']), ('Mai', 'mai'))
        self.assertNotIn('identity_evidence', mai)  # identity is certain, so no box ID or verification text is stored
        self.assertNotIn('detection_id', mai)
        self.assertNotIn('box', mai)
        self.assertEqual((mai['emotion'], mai['gaze'], mai['action']), ('content', 'at the book', 'sitting, reading'))
        self.assertEqual(result['mode'], 'full')
        self.assertEqual((result['mood'], result['props']), ('', 'none'))  # a model that skipped the new keys still yields a full record

    def test_frame_keeps_mood_and_props(self):
        result = resolve_frame(json.dumps(frame(mood='tense, quiet', props='red umbrella held by Mai', visible_characters=[entry('B1'), entry('B2')])), PEOPLE)
        self.assertEqual((result['mood'], result['props']), ('tense, quiet', 'red umbrella held by Mai'))

    def test_orphaned_references_are_repaired(self):
        raw = frame(visible_characters=[entry('B1'), entry('b1', pose='lying'), entry('B9'), 'invented'])
        result = resolve_frame(json.dumps(raw), PEOPLE)
        self.assertEqual([item['character'] for item in result['visible_characters']], ['Mai', 'Sakuta'])  # B9 dropped, skipped B2 added
        self.assertEqual(result['visible_characters'][0]['pose'], 'sitting')  # a repeated handle keeps the first answer
        self.assertEqual(result['visible_characters'][1]['pose'], '')
        self.assertEqual(resolve_frame(json.dumps(frame(visible_characters=[entry('B1')])), [])['visible_characters'], [])
        with self.assertRaises(ValueError): resolve_frame(json.dumps(frame(visible_characters='none')), PEOPLE)
        with self.assertRaises(ValueError): resolve_frame('[]', PEOPLE)

    def test_older_last_frame_in_changes_mode_still_reads_back_as_full_states(self):
        # Last frames are no longer analyzed, but the ones stored earlier must stay readable.
        first = {'composition': 'MS, eye level', 'setting': 'classroom', 'lighting': 'soft', 'camera': 'level', 'mode': 'full', 'background_people': 'none', 'notes': 'none',
                 'visible_characters': [{'character': 'Mai', 'character_context_id': 'mai', 'appearance': 'red tie', 'position': 'left', 'pose': 'sitting', 'expression': 'calm', 'action': 'reading'},
                                        {'character': 'Sakuta', 'character_context_id': 'sakuta', 'appearance': 'a', 'position': 'right', 'pose': 'standing', 'expression': 'calm', 'action': 'waiting'}]}
        last = {'composition': 'unchanged', 'setting': 'unchanged', 'lighting': 'unchanged', 'camera': 'unchanged', 'mode': 'changes', 'background_people': 'unchanged', 'notes': 'none',
                'visible_characters': [{'character': 'Mai', 'character_context_id': 'mai', 'appearance': 'unchanged', 'position': 'unchanged', 'pose': 'standing', 'expression': 'unchanged', 'action': 'unchanged', 'new_in_frame': False}]}
        full = full_last_frame(first, last)
        self.assertEqual(full['composition'], 'MS, eye level')
        self.assertEqual([(item['character'], item['appearance'], item['pose']) for item in full['visible_characters']], [('Mai', 'red tie', 'standing')])
        self.assertNotIn('emotion', full['visible_characters'][0])  # nothing is invented for fields older results never had
        self.assertIs(full_last_frame(first, first), first)  # full results pass through

    def test_schema_offers_only_the_listed_handles(self):
        schema = frame_schema(['B1', 'B2'])
        people = schema['properties']['visible_characters']
        self.assertEqual(people['items']['properties']['box']['enum'], ['B1', 'B2'])
        self.assertEqual((people['minItems'], people['maxItems']), (2, 2))
        self.assertEqual(frame_schema([])['properties']['visible_characters'], {'type': 'array', 'maxItems': 0})
        # No empty enum anywhere: llama-server refuses "enum": [] as a grammar (HTTP 500).
        self.assertNotIn('"enum": []', json.dumps(frame_schema([])))
        self.assertEqual(list(schema['properties'])[:5], ['composition', 'setting', 'lighting', 'camera', 'mood'])
        self.assertEqual(people['items']['required'], ['box', 'appearance', 'position', 'pose', 'expression', 'emotion', 'gaze', 'action', 'pose_detail'])
        pose = people['items']['properties']['pose_detail']
        self.assertEqual(pose['properties']['gaze_target']['enum'][:2], ['B1', 'B2'])  # a handle or a fixed kind, never free text
        self.assertEqual(list(schema['properties'])[-3:], ['background_people', 'props', 'notes'])

    def test_prompts_are_short_and_do_not_ask_for_boilerplate(self):
        self.assertNotIn('walking is unconfirmed', FRAME_INSTRUCTION.lower())
        self.assertIn('one single language', FRAME_INSTRUCTION)
        self.assertIn('certain', FRAME_INSTRUCTION)
        for needed in ('emotion:', 'gaze:', 'mood:', 'props:', 'by the exact name', 'EVERY person who is not in the marked list'):
            self.assertIn(needed, FRAME_INSTRUCTION)
        self.assertIn('none', person_context([], 'first frame'))
        self.assertIn('B1: Mai', person_context(PEOPLE, 'first frame'))
        self.assertIn('one single language', SUBTITLE_INSTRUCTION)
        self.assertNotIn('Portuguese or English', SUBTITLE_INSTRUCTION)

    def test_subtitle_note_states_the_language_as_certain_and_names_the_strip(self):
        note = subtitle_note('Japanese', [[0, 540, 1280, 700]], 'image pixels')
        self.assertIn('every burned-in subtitle in this video is in Japanese; that is certain', note)
        self.assertIn('[[0, 540, 1280, 700]] in image pixels', note)
        self.assertIn('never read, describe, translate or reproduce it', note)
        self.assertEqual(subtitle_note(None, [], 'image pixels'), '')
        self.assertNotIn('certain', subtitle_note(None, [[0, 1, 2, 3]], 'image pixels'))

    def test_video_accepts_overlapping_subject_events_without_reordering(self):
        events = [
            {'start_seconds': 0, 'end_seconds': .6, 'subject': 'Rio'},
            {'start_seconds': .6, 'end_seconds': 1.6, 'subject': 'Rio'},
            {'start_seconds': 1.6, 'end_seconds': 3.6, 'subject': 'Rio'},
            {'start_seconds': 0, 'end_seconds': 3.6, 'subject': 'Sakuta'},
        ]
        result = validate_video(json.dumps({'overview': 'scene', 'events': events}), 3.6, 10)
        self.assertEqual([event['subject'] for event in result['events']], ['Rio', 'Rio', 'Rio', 'Sakuta'])
        self.assertEqual(result['events'][-1]['episode_start_seconds'], 10)
        self.assertEqual(result['events'][-1]['episode_end_seconds'], 13.6)

    def test_video_preserves_unusual_json_without_inventing_times(self):
        payload = {'events': [{'start_seconds': 4, 'end_seconds': 5}, {'start_seconds': 'unknown'}, 'unstructured'], 'other': [1, 2]}
        self.assertEqual(validate_video(json.dumps(payload), 3, 10), payload)
        self.assertEqual(validate_video('{"events": null, "extra": true}', 3, 10), {'events': None, 'extra': True})
        for valid, expected in [('[]', []), ('null', None), ('"text"', 'text'), ('42', 42)]:
            self.assertEqual(validate_video(valid, 3, 10), expected)
        for invalid in ('not json', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_video(invalid, 3, 10)

    def test_video_preserves_original_rounded_timestamp(self):
        start, end = 161.035458, 167.792208
        event = {'start_seconds': 2.1, 'end_seconds': 6.75675}
        result = validate_video(json.dumps({'events': [event]}), end - start, start)
        self.assertEqual(result['events'][0]['end_seconds'], 6.75675)
        self.assertEqual(result['events'][0]['episode_end_seconds'], end)


INITIAL = {'composition': 'Medium shot, Mai left, Sakuta right', 'background_people': 'one girl with a red umbrella, right edge', 'mode': 'full',
           'visible_characters': [{'character': 'Mai', 'character_context_id': 'mai', 'appearance': 'white shirt, dark blue pleated skirt, long black hair', 'position': 'left, facing right',
                                   'pose': 'standing upright', 'expression': 'calm', 'emotion': 'calm', 'gaze': 'at Sakuta', 'action': 'standing',
                                   'pose_detail': {'body_pose': 'standing', 'body_yaw': 45, 'head_yaw': 45, 'head_pitch': 0, 'eye_gaze': 'right', 'gaze_target': 'Sakuta', 'gaze_object': '',
                                                   'body_view': 'three_quarter_right', 'head_view': 'three_quarter_right'}}]}


class VideoDirectionTest(unittest.TestCase):
    def roster(self):
        return video_roster(PEOPLE, [PEOPLE[1], {'ref': 'B2', 'character_context_id': 'kaede', 'name': 'Kaede', 'box': [700, 100, 900, 900]}], INITIAL)

    def test_roster_says_who_is_in_frame_at_each_end_and_how_the_first_frame_described_them(self):
        mai, sakuta, kaede = self.roster()
        # A last-frame box proves presence; Mai has none, which proves nothing, so her end is unknown rather than "left".
        self.assertEqual((mai['at_start'], mai['at_end'], sakuta['at_start'], sakuta['at_end'], kaede['at_start'], kaede['at_end']), (True, None, True, True, False, True))
        self.assertNotIn('appearance', mai)  # outfits are the first frame's job; the video model is not told them
        note = roster_note(self.roster(), INITIAL)
        self.assertIn('- Mai: first frame: left, facing right, box [10, 20, 300, 900]; standing upright. first-frame reading (from one still, may miss motion): state standing, '
                      'torso three-quarter view toward screen-right (+45°), head three-quarter view toward screen-right (+45°), eyes toward screen-right, looking at Sakuta. emotion in the first frame: calm.', note)
        self.assertNotIn('white shirt', note)
        self.assertNotIn('not in frame', note)  # the detector often misses people who are still there
        self.assertIn('- Sakuta: first frame: centre, box [500, 20, 800, 900]. last frame of the cut: in frame (centre, box [500, 20, 800, 900]).', note)
        self.assertIn('- Kaede: not marked in the first frame. last frame of the cut: in frame (right, box [700, 100, 900, 900]), so this character probably enters during the cut.', note)
        self.assertIn('Unnamed people in the first frame: one girl with a red umbrella', note)
        self.assertIn('Opening shot: Medium shot', note)
        self.assertNotIn('mai', note.replace('Mai', ''))  # names, never database ids

    def test_roster_without_last_frame_boxes_leaves_the_end_unknown(self):
        roster = video_roster(PEOPLE, None)
        self.assertEqual([person['at_end'] for person in roster], [None, None])
        self.assertNotIn('last frame of the cut', roster_note(roster))
        self.assertNotIn('end_state', cut_direction([], roster))
        self.assertIn('nobody was identified', roster_note([]))

    def test_speech_is_given_as_clip_relative_spans_without_the_words(self):
        segments = [{'start_time': 10.2, 'end_time': 10.6, 'text': 'えっ', 'character_name': 'Mai'}, {'start_time': 10.7, 'end_time': 11.5, 'text': '何', 'character_name': 'Mai'},
                    {'start_time': 12.5, 'end_time': 14, 'text': 'うん', 'speaker_label': 'SPEAKER_03'}, {'start_time': 30, 'end_time': 31, 'text': 'later', 'character_name': 'Mai'},
                    {'start_time': 10, 'end_time': 11, 'text': 'gone', 'character_name': 'Kaede', 'deleted': 1}]
        note = speech_note(segments, 10, 13, [{'name': 'Mai'}])
        self.assertIn('0.2-1.5 Mai; 2.5-3.0 another voice.', note)  # word fragments of one speaker are one span, clipped to the clip
        # A speaker who is not a marked character of this cut is never named, so the model cannot pin that name on someone it sees.
        self.assertIn('0.2-1.5 another voice', speech_note(segments, 10, 13, [{'name': 'Sakuta'}]))
        self.assertNotIn('えっ', note)
        self.assertNotIn('Kaede', note)
        self.assertIn('none in the transcript', speech_note([], 10, 13))

    def test_answers_are_linked_to_the_character_records(self):
        raw = {'overview': 'Mai turns.', 'camera': {'movement': 'Pan Left', 'detail': ''}, 'uncertainties': 'none',
               'characters': [{'name': 'mai', 'activity': 'standing', 'travel': 'none', 'motion': 'Turns her head.', 'speaking': 'Yes, briefly', 'emotion': 'calm', 'end': 'left, facing right'},
                              {'name': 'girl with red umbrella', 'activity': 'walking', 'travel': 'exits right', 'motion': 'Walks off.', 'speaking': 'maybe', 'emotion': 'not readable', 'end': 'left frame'}, 'junk'],
               'events': [{'start_seconds': 0, 'end_seconds': 1, 'subject': 'Sakuta', 'visual_action': 'Nods.'}, {'subject': 'camera', 'visual_action': 'Pans left.'}]}
        linked = link_video(raw, self.roster())
        mai, stranger = linked['characters']
        self.assertEqual((mai['character'], mai['character_context_id'], mai['speaking']), ('Mai', 'mai', 'yes'))
        self.assertEqual((stranger['character'], stranger['character_context_id'], stranger['speaking']), ('girl with red umbrella', None, 'unclear'))
        self.assertEqual(linked['camera'], {'movement': 'pan left', 'detail': 'none'})
        self.assertEqual(linked['events'][0]['character_context_id'], 'sakuta')
        group = link_video({'events': [{'subject': 'Mai, Sakuta and a teacher', 'visual_action': 'Walk left.'}]}, self.roster())['events'][0]
        self.assertEqual((group['subject'], group['character_context_ids']), ('Mai, Sakuta and a teacher', ['mai', 'sakuta']))  # one event for several people links each of them
        self.assertNotIn('character_context_id', linked['events'][1])
        self.assertEqual(link_video('free text', self.roster()), 'free text')  # unusual answers are kept as they are
        self.assertEqual(link_video({'overview': 'x', 'characters': 'nobody'}, [])['characters'], 'nobody')

    def test_gaze_objects_and_event_kinds_are_coerced_never_rejected(self):
        raw = {'characters': [{'name': 'Mai', 'activity': 'standing', 'gaze': 'Looks at Sakuta, then down at the cup.', 'objects': ['cup: lifted to her mouth', 'none', {'object': 'door'}, 3, 'cup: lifted to her mouth']},
                              {'name': 'Sakuta', 'objects': 'a red umbrella: closed and held', 'gaze': ''}, {'name': 'Kaede', 'objects': 'none'}],
               'events': [{'subject': 'Mai', 'kind': 'Gaze', 'visual_action': 'a'}, {'subject': 'Mai', 'kind': 'speech', 'visual_action': 'b'}, {'subject': 'Mai', 'kind': 'dance', 'visual_action': 'c'},
                          {'subject': 'Mai', 'kind': None, 'visual_action': 'd'}, {'subject': 'Mai', 'visual_action': 'older event without a kind'}]}
        linked = link_video(raw, self.roster())
        mai, sakuta, kaede = linked['characters']
        self.assertEqual(mai['gaze'], 'Looks at Sakuta, then down at the cup.')
        self.assertEqual(mai['objects'], ['cup: lifted to her mouth', 'door'])  # junk, duplicates and "none" are dropped
        self.assertEqual(sakuta['objects'], ['a red umbrella: closed and held'])
        self.assertNotIn('gaze', sakuta)  # an empty answer is not stored
        self.assertNotIn('objects', kaede)
        self.assertEqual([event.get('kind') for event in linked['events']], ['gaze', 'mouth', 'other', 'other', None])
        self.assertNotIn('kind', linked['events'][4])  # results from before the field keep their shape

    def test_a_clip_that_contradicts_the_still_is_a_state_correction(self):
        def window(**fields):
            return {'analysis': link_video({'overview': 'x', 'camera': {'movement': 'static', 'detail': 'none'}, 'events': [],
                                            'characters': [{'name': 'Mai', 'activity': 'standing', 'gaze': 'holds', 'objects': [], **fields}]}, self.roster())}
        roster = video_roster(PEOPLE, None, INITIAL)  # the first frame read Mai as standing
        mai = cut_direction([window(activity='walking', travel='right', gaze='Looks ahead.', objects=['bag: swung'])], roster)['characters'][0]
        self.assertEqual(mai['state_correction'], {'first_frame': 'standing', 'clip': 'walking'})
        self.assertEqual((mai['gaze'], mai['objects']), ('Looks ahead.', ['bag: swung']))
        for activity in ('standing', 'standing, then walking', 'turning', 'writing', ''):  # same state, a later transition, or no body state at all
            self.assertNotIn('state_correction', cut_direction([window(activity=activity)], roster)['characters'][0], activity)
        merged = cut_direction([window(gaze='Looks at Sakuta.'), window(gaze='holds'), window(gaze='Looks down.', objects=['cup: set down'])], roster)['characters'][0]
        self.assertEqual((merged['gaze'], merged['objects']), ('Looks at Sakuta. Then: Looks down.', ['cup: set down']))
        self.assertNotIn('gaze', cut_direction([window(gaze='holds')], roster)['characters'][0])
        self.assertNotIn('state_correction', cut_direction([window(activity='walking')], video_roster(PEOPLE, None))['characters'][0])  # no first-frame reading, nothing to compare

    def test_a_first_name_links_only_when_it_is_unambiguous(self):
        roster = [{'name': 'Mai Sakurajima', 'character_context_id': 'mai'}, {'name': 'Nodoka Toyohama', 'character_context_id': 'nodoka'}, {'name': 'Mai Clone', 'character_context_id': 'clone'}]
        linked = link_video({'characters': [{'name': 'Nodoka'}, {'name': 'Mai'}, {'name': 'mai sakurajima'}]}, roster)
        self.assertEqual([person['character_context_id'] for person in linked['characters']], ['nodoka', None, 'mai'])

    def test_windows_merge_into_one_direction_for_the_cut(self):
        def window(overview, movement, people):
            return {'analysis': link_video({'overview': overview, 'camera': {'movement': movement, 'detail': 'none'}, 'characters': people, 'events': []}, self.roster())}
        person = lambda name, **fields: {'name': name, 'activity': 'standing', 'travel': 'none', 'motion': 'holds still', 'speaking': 'no', 'emotion': 'calm', 'end': 'left', **fields}
        windows = [window('Mai waits.', 'static', [person('Mai'), person('Sakuta', speaking='yes', emotion='amused')]),
                   window('Mai walks off.', 'pan right', [person('Mai', activity='walking', travel='exits right', motion='Walks to the right.', emotion='annoyed', end='left frame'), person('Kaede', end='right, facing left')])]
        direction = cut_direction(windows, self.roster())
        self.assertEqual(direction['summary'], 'Mai waits. Then: Mai walks off.')
        self.assertEqual(direction['camera'], {'movement': 'static → pan right', 'detail': 'none'})
        mai, sakuta, kaede = direction['characters']
        self.assertEqual((mai['activity'], mai['travel'], mai['motion'], mai['speaking'], mai['emotion'], mai['end']),
                         ('standing → walking', 'exits right', 'Walks to the right.', 'no', 'calm → annoyed', 'left frame'))
        self.assertEqual((sakuta['speaking'], sakuta['end']), ('yes', 'not seen in the final part of the cut'))
        self.assertEqual(kaede['character_context_id'], 'kaede')
        self.assertEqual(direction['end_state'], {'in_frame': ['Sakuta', 'Kaede'], 'source': 'character boxes of the last frame'})
        self.assertEqual([person.get('in_last_frame') for person in direction['characters']], [None, True, True])  # Mai left according to the video; no box says otherwise
        note = earlier_note(windows[0])
        self.assertIn('Just before it: Mai waits', note)
        self.assertIn('- Mai: left', note)
        self.assertIn('not to be repeated in your answer', note)  # a window once copied this note into its own summary
        self.assertEqual(earlier_note({'analysis': 'text'}), '')

    def test_video_prompt_asks_for_direction_not_description(self):
        for needed in ('camera:', 'activity:', 'travel:', 'speaking:', 'emotion:', 'end:', 'exact name', 'Do not describe outfits', 'never give that voice', 'start from it', 'gaze:', 'objects:', 'kind (exactly one of'):
            self.assertIn(needed, VIDEO_INSTRUCTION)
        json.loads(VIDEO_INSTRUCTION.split('Example shape: ', 1)[1])  # the example the model copies must be valid JSON
