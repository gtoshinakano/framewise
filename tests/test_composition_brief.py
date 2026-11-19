import json
import re
import tempfile
import unittest
import wave
import zipfile
import io
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import db, media, worker, composition_brief, composition_compaction, external_api, inference_worker, structured_results
from app.main import app

FIXTURES = Path(__file__).parent / 'fixtures' / 'composition'


def fixture(number):
    return json.loads((FIXTURES / f'{number}.json').read_text())


class CompositionProjectionTest(unittest.TestCase):
    def test_consistency_issues_describe_only_explicit_discrepancies(self):
        def issues(number):
            value = next(r['content'] for r in fixture(number)['results'] if r['type']=='consistency_summary')
            return composition_brief.issue_phrases(value['confusing'])[0]
        self.assertIn('Camera descriptions differ', issues(44))
        self.assertIn('Dialogue and video mouth movements disagree', issues(47))
        self.assertIn('Character positions differ across analyses', issues(48))
        self.assertIn('Character identity is unconfirmed', issues(51))
        self.assertEqual(composition_brief.issue_phrases('No significant issues.'), ([],False))
        self.assertNotIn('Character identity is unconfirmed', composition_brief.issue_phrases('Identity is confirmed.')[0])
        self.assertNotIn('Camera descriptions differ', composition_brief.issue_phrases('Camera angle has no inconsistencies.')[0])

    def test_schema_rejects_invalid_utterance_times_and_long_end_state(self):
        from pydantic import ValidationError
        utterance = {'speaker_label':'s', 'character_context_id':None, 'character_name':'unknown',
                     'text':'spoken', 't0':0, 't1':2, 'crosses_cut':False}
        with self.assertRaises(ValidationError):
            external_api.CompositionUtterance.model_validate({**utterance, 't0':2, 't1':1})
        with self.assertRaises(ValidationError):
            external_api.CompositionEndSubject.model_validate({'subject':'s','pose':' '.join(['word'] * 13)})
        with self.assertRaises(ValidationError):
            external_api.CompositionCut.model_validate({'id':'c','cut_number':1,'ready':True,'duration':1,'dialogue':[utterance]})
        with self.assertRaises(ValidationError):
            external_api.CompositionCut.model_validate({'id':'c','cut_number':1,'ready':True,'duration':1,
                                                        'beats':[{'t0':0,'t1':1,'subject':0,'action':'turns'}]})
        base = {'id':'c','cut_number':1,'ready':True,'duration':2}
        with self.assertRaises(ValidationError):
            external_api.CompositionCut.model_validate({**base,'beats':[{'t0':0,'t1':1,'subject':'A','action':'turns'},
                                                                        {'t0':1,'t1':2,'subject':'A','action':'stops'}]})

    def test_subject_references_require_a_unique_identified_person(self):
        record = {'subjects':[{'character_context_id':'a','name':'Mai'},
                              {'character_context_id':'b','name':'Mai'},
                              {'character_context_id':None,'name':'unknown'}],
                  'beats':[{'subject':'Mai','action':'turns'}, {'subject':'unknown','action':'stands'}],
                  'end_state':{'subjects':[{'subject':'Mai','pose':'sitting'}]}}
        composition_brief.reference_subjects(record)
        self.assertEqual([item['subject'] for item in record['beats']],['Mai','unknown'])
        self.assertEqual(record['end_state']['subjects'][0]['subject'],'Mai')

    def test_event_uncertainty_does_not_erase_following_motion(self):
        action = composition_brief.event_action('She turns, possibly toward the door and raises her left arm.')
        self.assertIn('possibly toward the door', action)
        self.assertIn('raises her left arm', action)
        action = composition_brief.event_action('His head and upper body are visible, but his face is not clearly seen.')
        self.assertIn('head and upper body are visible', action)
        self.assertIn('face is not clearly seen', action)

    def test_compaction_can_only_rewrite_existing_visual_text(self):
        original = {'setting':'A grocery store with shelves', 'shot':{'camera':'static'},
                    'subjects':[{'character_context_id':'character-id','name':'Mai','pose':'standing upright', 'facing':'right'}],
                    'beats':[{'t0':0,'t1':2,'subject':'Mai','action':'turns right toward Sakuta'}],
                    'dialogue':[{'text':'Original spoken words'}], 'links':[{'first':[[0,1,2,3]]}]}
        fields = composition_compaction.text_fields(original)
        self.assertNotIn('dialogue.0.text', fields)
        changed = dict(fields, setting='grocery store; shelves', **{'subjects.0.pose':'standing upright', 'beats.0.action':'turns right toward Sakuta'})
        raw = json.dumps({'values':changed})
        values = composition_compaction.validate_response(raw,fields,names=['Sakuta'])
        self.assertEqual(composition_compaction.validate_response('<analysis>private</analysis>'+raw,fields,names=['Sakuta']), values)
        with self.assertRaises(ValueError):
            composition_compaction.validate_response('<reasoning>unfinished'+raw,fields,names=['Sakuta'])
        result = composition_compaction.apply_values(original,values)
        self.assertEqual(result['subjects'][0]['character_context_id'],'character-id')
        self.assertEqual(result['dialogue'],original['dialogue'])
        self.assertEqual(result['beats'][0]['t1'],2)
        self.assertEqual(result['subjects'][0]['pose'],'standing upright')
        for invalid in ({**changed,'subjects.0.facing':'left'}, {k:v for k,v in changed.items() if k!='setting'}, {**changed,'beats.0.action':'turns right'}):
            with self.assertRaises(ValueError):
                composition_compaction.validate_response(json.dumps({'values':invalid}),fields,names=['Sakuta'])
        source = {'setting':'likely a hallway or corridor', 'beats.0.action':'holds a bunch of bananas in both hands'}
        for changed in ({'setting':'hallway', 'beats.0.action':source['beats.0.action']},
                        {'setting':source['setting'], 'beats.0.action':'holds bananas in both hands'}):
            with self.assertRaises(ValueError):
                composition_compaction.validate_response(json.dumps({'values':changed}),source)
        source = {'beats.0.action':'bends forward at the waist and lowers his head'}
        with self.assertRaisesRegex(ValueError,'waist'):
            composition_compaction.validate_response(json.dumps({'values':{'beats.0.action':'bends forward and lowers head'}}),source)
        source = {'subjects.0.hands':'hand in pocket', 'subjects.0.expression':'neutral'}
        with self.assertRaises(ValueError):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'hand on pocket','subjects.0.expression':'neutral'}}),source)
        with self.assertRaises(ValueError):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'hand pocket','subjects.0.expression':'neutral'}}),source)
        source = {'subjects.0.hands':'hand on table under lamp'}
        with self.assertRaises(ValueError):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'hand on lamp under table'}}),source)
        source = {'subjects.0.hands':'hand rests on his arm'}
        with self.assertRaises(ValueError):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'hand rests on arm'}}),source)
        with self.assertRaises(ValueError):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'hand in pocket'}}),{'subjects.0.hands':'hand pocket'})
        source = {'subjects.0.hands':'hand in pocket', 'subjects.0.expression':'neutral'}
        with self.assertRaisesRegex(ValueError,'unsupported'):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'hand in pocket','subjects.0.expression':'neutral and angry'}}),source)
        source = {'beats.0.action':'turns left; Bob: holds a red ball'}
        with self.assertRaises(ValueError):
            composition_compaction.validate_response(json.dumps({'values':{'beats.0.action':'holds a red ball; Bob: turns left'}}),source)
        source = {'beats.0.action':'closes eyes then opens eyes'}
        with self.assertRaisesRegex(ValueError,'action order'):
            composition_compaction.validate_response(json.dumps({'values':{'beats.0.action':'opens eyes then closes eyes'}}),source)
        source = {'subjects.0.hands':'left hand holds carrot and right hand holds banana'}
        with self.assertRaisesRegex(ValueError,'left and right'):
            composition_compaction.validate_response(json.dumps({'values':{'subjects.0.hands':'left hand holds banana and right hand holds carrot'}}),source)

    def test_real_cut_51_compaction_retains_motion_and_uncertainty(self):
        item = json.loads((FIXTURES/'compaction-51.json').read_text())
        # The real response omitted participant separators, so it must not be
        # accepted until those associations are restored.
        with self.assertRaisesRegex(ValueError,'participant label'):
            composition_compaction.validate_response(json.dumps({'values':item['values']},ensure_ascii=False),item['input']['values'],item['input']['names'])
        fixed = dict(item['values'])
        fixed['beats.0.action'] = fixed['beats.0.action'].replace('Man in white shirt bends','Man in white shirt: bends').replace('Sakuta Azusagawa stands','Sakuta Azusagawa: stands')
        with self.assertRaisesRegex(ValueError,'relation'):
            composition_compaction.validate_response(json.dumps({'values':fixed},ensure_ascii=False),item['input']['values'],item['input']['names'])
        # Retain the source wording where the model omitted relations or
        # possessives; safe compaction of other fields remains eligible.
        for key, source in item['input']['values'].items():
            if key.endswith(('.hands', '.action')): fixed[key] = source
        values = composition_compaction.validate_response(json.dumps({'values':fixed},ensure_ascii=False),item['input']['values'],item['input']['names'])
        action = values['beats.0.action'].lower()
        self.assertIn('waist',action)
        self.assertIn('then',action)
        self.assertIn('both',action)
        self.assertIn('bunch',values['subjects.0.hands'].lower())

    def test_position_does_not_confuse_gaze_with_frame_location(self):
        item = fixture(57)
        frame = next(r['content'] for r in item['results'] if r['type'] == 'initial_frame')
        positions = [s.get('frame_pos') for s in composition_brief.subjects(frame, [])]
        self.assertEqual(positions, ['left fg', 'right fg', 'left bg'])
        frame = next(r['content'] for r in fixture(51)['results'] if r['type']=='initial_frame')
        self.assertEqual(composition_brief.subjects(frame, [])[1]['frame_pos'],'left mid')
        last = next(r['content'] for r in fixture(49)['results'] if r['type']=='last_frame')
        facing = composition_brief.subjects(last, [])[0]['facing']
        self.assertIn('right',facing)
        self.assertIn('three-quarter',facing)
        for number in (48,50,55,58,61,63,66):
            frame = next(r['content'] for r in fixture(number)['results'] if r['type']=='initial_frame')
            self.assertTrue(composition_brief.subjects(frame,[])[0]['frame_pos'].startswith('center '),number)

    def test_facing_retains_qualified_direction_and_visible_target(self):
        for number, index, expected in ((47,1,'slightly to the right'),
                                        (51,1,'slightly towards the right'),
                                        (54,1,'directly toward the camera'),
                                        (60,0,'male character')):
            frame = next(r['content'] for r in fixture(number)['results'] if r['type']=='initial_frame')
            person = composition_brief.subjects(frame,[])[index]
            self.assertIn(expected,person['facing'].lower(),(number,person))

    def test_named_verified_box_links_without_embedded_box_id(self):
        item = fixture(47)
        frame = next(r['content'] for r in item['results'] if r['type'] == 'initial_frame')
        links = [{'id':'manual-box','position':'first','box':[326,120,745,720], 'frame_width':1280,
                  'character_context_id':'mai','name':'Mai Sakurajima','label_source':'manual','stale':False,'verified':False}]
        result = composition_brief.subjects(frame, links)
        self.assertEqual(result[1]['character_context_id'], 'mai')
        self.assertEqual(result[1]['identity'], 'confirmed')
        self.assertEqual(result[1]['frame_pos'], 'center mid')
        links[0]['stale'] = True
        self.assertIsNone(composition_brief.subjects(frame, links)[1]['character_context_id'])

    def test_new_format_frames_link_by_character_and_changes_read_back_as_states(self):
        from app.visual_contracts import full_last_frame, resolve_frame
        mai = {'ref': 'B1', 'detection_id': 'first-box', 'character_context_id': 'mai', 'name': 'Mai', 'box': None}
        base = {'composition': 'MS', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'background_people': 'none', 'notes': 'none'}
        initial = resolve_frame(json.dumps({**base, 'visible_characters': [{'box': 'B1', 'appearance': 'red blazer', 'position': 'left foreground', 'pose': 'sitting at a desk', 'expression': 'calm', 'action': 'reading'}]}), [mai])
        # A last frame as stored before last frames stopped being analyzed: only the changes from the first.
        last = {**base, 'composition': 'unchanged', 'mode': 'changes', 'visible_characters': [{'character': 'Mai', 'character_context_id': 'mai', 'appearance': 'unchanged', 'position': 'unchanged',
                                                                                              'pose': 'standing', 'expression': 'unchanged', 'action': 'unchanged', 'new_in_frame': False}]}
        links = lambda pos, id_: [{'id': id_, 'position': pos, 'box': [0, 0, 300, 700], 'frame_width': 1280, 'character_context_id': 'mai', 'name': 'Mai', 'label_source': 'manual', 'stale': False, 'verified': True}]
        first = composition_brief.subjects(initial, links('first', 'first-box'))[0]
        self.assertEqual((first['name'], first['identity'], first['outfit_visible']), ('Mai', 'confirmed', 'red blazer'))
        final = composition_brief.subjects(full_last_frame(initial, last), links('last', 'last-box'), boundary='last')[0]
        self.assertEqual((final['name'], final['pose'], final['outfit_visible']), ('Mai', 'standing', 'red blazer'))
        self.assertFalse(composition_brief.unchanged(first['pose'], final['pose'], 'pose'))

    def test_detection_citation_is_not_reused_across_boundary_frames(self):
        frame = {'visible_characters':[{'position':'center','pose':'standing',
                                        'identity_evidence':'Matches first-frame box first-id.'}]}
        links = [{'id':'first-id', 'position':'first', 'box':[300,0,900,720], 'frame_width':1280,
                  'character_context_id':'mai','name':'Mai','label_source':'manual','stale':False}]
        self.assertEqual(composition_brief.subjects(frame,links,boundary='first')[0]['character_context_id'], 'mai')
        self.assertIsNone(composition_brief.subjects(frame,links,boundary='last')[0]['character_context_id'])

    def test_unique_spatial_detection_links_automatic_label_as_likely(self):
        item = fixture(45)
        frame = next(r['content'] for r in item['results'] if r['type']=='initial_frame')
        links = [{'id':d['id'], 'position':d['position'], 'box':json.loads(d['box_json']), 'frame_width':1280,
                  'character_context_id':d['character_context_id'], 'name':d['name'] or 'unknown',
                  'label_source':d['label_source'] or 'unknown', 'stale':False, 'verified':False}
                 for d in item['detections']['detections']]
        people = composition_brief.subjects(frame,links)
        self.assertEqual(people[2]['name'],'Sakuta Azusagawa')
        self.assertEqual(people[2]['identity'],'likely')
        self.assertIsNone(people[1]['character_context_id'])
        # A second box in that zone makes the association ambiguous.
        links.append({**links[2], 'id':'other', 'character_context_id':None})
        self.assertIsNone(composition_brief.subjects(frame,links)[2]['character_context_id'])

    def test_unseen_expressions_and_duplicate_pose_are_not_hand_facts(self):
        item = fixture(47)
        frame = next(r['content'] for r in item['results'] if r['type']=='initial_frame')
        people = composition_brief.subjects(frame, [])
        self.assertNotIn('expression',people[0])
        self.assertNotEqual(people[1].get('hands'), 'Standing')

    def test_cut_50_expression_and_direct_gaze_are_not_omitted(self):
        frame = next(r['content'] for r in fixture(50)['results'] if r['type']=='initial_frame')
        person = composition_brief.subjects(frame,[])[0]
        self.assertIn('slightly serious',person['expression'].lower())
        self.assertIn('closed mouth',person['expression'].lower())
        self.assertEqual(person['gaze'],'direct eye contact')
        self.assertLessEqual(len(person['expression'].split()),12)

    def test_unrepresentable_subject_detail_is_reported(self):
        frame = {'visible_characters':[{'position':'center',
                 'pose':'A particularly elaborate posture with many words that cannot safely fit within the short phrase limit',
                 'expression':'A subtle and complicated expression described using many words that cannot safely fit in one short phrase'}]}
        missing = set()
        composition_brief.subjects(frame,[],missing=missing)
        self.assertTrue({'pose','expression'} <= missing)
        missing = set()
        composition_brief.subjects({'visible_characters':[{'position':'center','pose':'Not visible.',
                                                            'expression':'Unknown, face not visible.'}]},[],missing=missing)
        self.assertFalse(missing)

    def test_expression_preserves_negation_and_mixed_observations(self):
        frame = {'visible_characters':[{'position':'center','pose':'standing','appearance':'',
                                        'expression':'Not smiling, mouth closed.','identity_evidence':''},
                                       {'position':'right','pose':'standing','appearance':'',
                                        'expression':'Calm and composed, with a slight smile.','identity_evidence':''}]}
        people = composition_brief.subjects(frame, [])
        self.assertIn('not smiling',people[0]['expression'].lower())
        self.assertIn('calm',people[1]['expression'].lower())
        self.assertIn('smile',people[1]['expression'].lower())

    def test_cut_48_initial_eye_mouth_and_upright_state_are_available(self):
        frame = next(r['content'] for r in fixture(48)['results'] if r['type']=='initial_frame')
        missing = set()
        person = composition_brief.subjects(frame, [], missing=missing)[0]
        self.assertIn('eyes wide open',person['expression'].lower())
        self.assertIn('mouth closed',person['expression'].lower())
        self.assertIn('upright',person['pose'].lower())
        self.assertNotIn('expression',missing)
        value, incomplete = composition_brief.expression_details('Neutral. Eyes are open or closed.', 'neutral')
        self.assertEqual(value,'neutral; eyes open or closed')
        self.assertFalse(incomplete)

    def test_setting_preserves_cut_48_architecture_and_flags_omitted_details(self):
        frame = next(r['content'] for r in fixture(48)['results'] if r['type']=='initial_frame')
        missing = set()
        value = composition_brief.setting_phrase(frame['setting'],20,missing).lower()
        for detail in ('likely', 'hallway', 'teal', 'wall', 'pillar', 'left', 'window', 'glass door', 'right'):
            self.assertIn(detail,value)
        self.assertLessEqual(len(value.split()),20)
        self.assertFalse(missing)
        long = 'A room. A distinct red cabinet with a large blue box on top.'
        missing = set()
        self.assertEqual(composition_brief.setting_phrase(long,3,missing),'A room')
        self.assertEqual(missing,{'detail'})

    def test_multiple_visible_hand_observations_are_not_dropped(self):
        for number, index, facts in ((51,0,('bananas','right hand','left hand','cart')),
                                     (57,0,('left hand','palm open','right hand','cart')),
                                     (64,0,('left arm','right hand','rests on his arm'))):
            frame = next(r['content'] for r in fixture(number)['results'] if r['type']=='initial_frame')
            missing = set()
            person = composition_brief.subjects(frame, [], missing=missing)[index]
            text = (person.get('pose','')+'; '+person.get('hands','')).lower()
            for fact in facts:
                self.assertIn(fact,text,(number,person,missing))
            self.assertNotIn('hands',missing,(number,person))
            self.assertLessEqual(len(person.get('pose','').split()),12)
            self.assertLessEqual(len(person.get('hands','').split()),12)
        frame = next(r['content'] for r in fixture(57)['results'] if r['type']=='initial_frame')
        mai = composition_brief.subjects(frame, [])[1]
        self.assertEqual(mai['pose'].lower(),'standing')
        self.assertIn('arms at',mai['hands'].lower())

    def test_delta_preserves_opposite_directions_and_poses(self):
        self.assertFalse(composition_brief.unchanged('looking left', 'looking right'))
        self.assertFalse(composition_brief.unchanged('standing with arms raised', 'standing with arms lowered'))
        self.assertFalse(composition_brief.unchanged('eyes open', 'eyes closed'))
        self.assertFalse(composition_brief.unchanged('hand in pocket', 'hand on pocket', 'hands'))
        self.assertTrue(composition_brief.unchanged('standing', 'standing upright', 'pose'))
        self.assertTrue(composition_brief.unchanged('holding the handle of a shopping cart with both hands', 'both hands gripping the handle of the shopping cart', 'hands'))
        self.assertTrue(composition_brief.unchanged('holding the handle of a shopping cart with both hands', 'holding the handle of a shopping cart', 'hands'))
        self.assertTrue(composition_brief.unchanged('Standing upright', 'Standing upright, legs and feet not visible', 'pose'))
        self.assertTrue(composition_brief.unchanged('away; profile', 'away', 'facing'))
        self.assertTrue(composition_brief.unchanged('towards the male', 'at the male', 'gaze'))
        self.assertFalse(composition_brief.unchanged('left; profile', 'right', 'facing'))
        self.assertFalse(composition_brief.unchanged('towards the male', 'at the female', 'gaze'))

    def test_depth_delta_requires_explicit_depth_at_both_boundaries(self):
        compare = composition_brief.position_delta
        self.assertEqual(compare('center mid','center fg','Centered in the frame','Centered in the foreground'),(False,True))
        self.assertEqual(compare('right fg','right mid','Right foreground','Right side of the frame'),(False,True))
        self.assertEqual(compare('right fg','right bg','Right foreground','Right background'),(True,False))
        self.assertEqual(compare('right fg','left mid','Right foreground','Left side of the frame'),(True,True))
        self.assertEqual(compare('center mid','center mid','Centered','Centered'),(False,False))

    def test_static_dedup_requires_same_known_state_and_keeps_motion(self):
        state = {'pose':'standing', 'facing':'camera', 'gaze':'forward', 'expression':'neutral'}
        self.assertTrue(composition_brief.static_action_is_in_state('Stands still, facing camera and looking forward with a neutral expression',state))
        self.assertFalse(composition_brief.static_action_is_in_state('Stands still, looking left with a neutral expression',state))
        self.assertFalse(composition_brief.static_action_is_in_state('Stands and smiles',state))
        self.assertFalse(composition_brief.static_action_is_in_state('Stands holding a carrot',state))
        video = {'windows':[{'source':{'source_start':0,'source_end':2},'analysis':{'events':[
            {'start_seconds':0,'end_seconds':2,'subject':'A',
             'visual_action':'Stands still, facing camera and looking forward with a neutral expression. Turns left.'}]}}]}
        beats, partial = composition_brief.video_beats(video,0,2,{'A':{'first':state}})
        self.assertFalse(partial)
        self.assertEqual(beats[0]['action'],'Turns left')

    def test_anonymous_matching_requires_unique_distinctive_visual_evidence(self):
        subject = {'character_context_id':None,'name':'unknown','frame_pos':'left mid'}
        frame = {'visible_characters':[{'appearance':'black hair, glasses, white shirt and dark trousers'}]}
        self.assertEqual(composition_brief.anonymous_pairs([subject],[subject],frame,frame,[0],[0]),[(0,0)])
        two = {'visible_characters':frame['visible_characters']*2}
        self.assertEqual(composition_brief.anonymous_pairs([subject,subject],[subject],two,frame,[0,1],[0]),[])
        shoulder = {'visible_characters':[{'appearance':'white garment, possibly a shirt'}]}
        self.assertEqual(composition_brief.anonymous_pairs([subject],[subject],shoulder,frame,[0],[0]),[])

    def test_window_times_clamped_and_zero_events_ignored(self):
        for number in (49, 51):
            item = fixture(number)
            cut = item['cut']
            video = next(r['content'] for r in item['results'] if r['type'] == 'video_analysis')
            beats, _ = composition_brief.video_beats(video, cut['start_time'], cut['end_time'])
            self.assertTrue(beats, number)
            self.assertTrue(all(0 <= b['t0'] < b['t1'] <= round(cut['end_time']-cut['start_time'], 3) for b in beats), number)
            self.assertTrue(all(a['t1'] <= b['t0'] for a,b in zip(beats,beats[1:])), number)
        self.assertGreater(beats[-1]['t1'], 0)
        self.assertIn('bends forward', ' '.join(b['action'] for b in beats))

    def test_cut_48_preserves_lip_motion(self):
        item = fixture(48)
        cut = item['cut']
        video = next(r['content'] for r in item['results'] if r['type'] == 'video_analysis')
        beats, _ = composition_brief.video_beats(video, cut['start_time'], cut['end_time'])
        self.assertIn('opens and closes', ' '.join(b['action'] for b in beats))

    def test_cut_47_keeps_head_and_mouth_changes_in_the_same_beat(self):
        item = fixture(47)
        cut = item['cut']
        video = next(r['content'] for r in item['results'] if r['type']=='video_analysis')
        beats, _ = composition_brief.video_beats(video,cut['start_time'],cut['end_time'])
        self.assertIn('Turns her head to her left',beats[1]['action'])
        self.assertIn('mouth closes and then opens',beats[1]['action'])
        self.assertNotIn('Mai Sakurajima: ',beats[0]['action'])

    def test_one_millisecond_overlap_does_not_leak_into_the_contract(self):
        video = {'windows':[{'source':{'source_start':0,'source_end':3}, 'analysis':{'events':[
            {'start_seconds':0,'end_seconds':1.5,'subject':'A','visual_action':'Turns left.'},
            {'start_seconds':1.499,'end_seconds':3,'subject':'B','visual_action':'Turns right.'}]}}]}
        beats, _ = composition_brief.video_beats(video,0,3)
        self.assertTrue(all(a['t1']<=b['t0'] for a,b in zip(beats,beats[1:])))
        self.assertIn('Turns left',beats[0]['action'])
        self.assertIn('Turns right',beats[0]['action'])

    def test_continuous_event_split_at_window_boundary_merges(self):
        def window(start,end):
            return {'source':{'source_start':start,'source_end':end}, 'analysis':{'events':[
                {'start_seconds':0,'end_seconds':end-start+.016,'subject':'A','visual_action':'Raises left hand.'}]}}
        beats, partial = composition_brief.video_beats({'windows':[window(10,12),window(12,14)]},10,14)
        self.assertFalse(partial)
        self.assertEqual(len(beats),1)
        self.assertEqual((beats[0]['t0'],beats[0]['t1']),(0,4))

    def test_simultaneous_events_merge_across_windows_only_when_all_continue(self):
        def window(start,end,include_b=True):
            events = [{'start_seconds':0,'end_seconds':end-start,'subject':'A','visual_action':'Raises left hand.'}]
            if include_b: events.append({'start_seconds':0,'end_seconds':end-start,'subject':'B','visual_action':'Turns right.'})
            return {'source':{'source_start':start,'source_end':end},'analysis':{'events':events}}
        beats, _ = composition_brief.video_beats({'windows':[window(0,2),window(2,4)]},0,4)
        self.assertEqual(len(beats),1)
        self.assertEqual((beats[0]['t0'],beats[0]['t1']),(0,4))
        beats, _ = composition_brief.video_beats({'windows':[window(0,2),window(2,4,False)]},0,4)
        self.assertEqual(len(beats),2)

    def test_empty_window_is_reported(self):
        item = fixture(64)
        cut = item['cut']
        video = next(r['content'] for r in item['results'] if r['type'] == 'video_analysis')
        _, partial = composition_brief.video_beats(video, cut['start_time'], cut['end_time'])
        self.assertTrue(partial)
        beats, _ = composition_brief.video_beats(video, cut['start_time'], cut['end_time'])
        self.assertIn('gesturing', ' '.join(b['action'] for b in beats))
        self.assertIn('gaze is directed forward', ' '.join(b['action'] for b in beats))

    def test_ocr_variants_collapse_but_separate_repetitions_remain(self):
        for number, expected in ((51, 1), (57, 2), (64, 2)):
            item = fixture(number)
            cut = item['cut']
            ocr = next(r['content'] for r in item['results'] if r['type'] == 'subtitle_ocr')
            lines = composition_brief.subtitle_lines(ocr['events'], cut['start_time'], cut['end_time'])
            self.assertEqual(len(lines), expected, (number, lines))
            self.assertTrue(all(set(line) == {'text','lang','t'} for line in lines))
        events = [{'text':'A repeated sentence', 'language':'en', 'first_observed':0},
                  {'text':'A repeated sentence', 'language':'en', 'first_observed':8}]
        self.assertEqual(len(composition_brief.subtitle_lines(events,0,10)),2)
        for first, second in (('You should definitely go now', 'You should definitely not go now'),
                              ('There are 10 people at the store', 'There are 11 people at the store'),
                              ('There are 2 people left and 3 people right', 'There are 3 people left and 2 people right')):
            events = [{'text':first, 'language':'en', 'first_observed':0},
                      {'text':second, 'language':'en', 'first_observed':1}]
            self.assertEqual([line['text'] for line in composition_brief.subtitle_lines(events,0,2)], [first,second])
        events = [{'text':'A different subtitle', 'language':'en', 'first_observed':2},
                  {'text':'First subtitle', 'language':'en', 'first_observed':1}]
        self.assertEqual([line['t'] for line in composition_brief.subtitle_lines(events,0,3)], [1,2])
        missing = set()
        self.assertEqual(composition_brief.subtitle_lines([{'text':'late','language':'en','first_observed':11}],0,10,missing),[])
        self.assertEqual(missing,{'time'})

    def test_video_camera_overrides_still_frame_in_cut_44(self):
        item = fixture(44)
        video = next(r['content'] for r in item['results'] if r['type'] == 'video_analysis')
        self.assertEqual(composition_brief.video_camera(video)['camera'], 'pan')
        self.assertEqual(composition_brief.video_camera(video)['direction'], 'right')

    def test_camera_qualifiers_do_not_use_subject_position_or_lighting(self):
        result = composition_brief.shot_props({'composition':'A person on the left. The camera pans right slowly under overhead lighting.'})
        self.assertEqual(result, {'camera':'pan','direction':'right','speed':'slow'})
        result = composition_brief.shot_props({'composition':'The person on the left moves quickly. The camera pans.'})
        self.assertEqual(result, {'camera':'pan'})
        result = composition_brief.shot_props({'camera':'A slow left pan from an overhead shot.'})
        self.assertEqual(result, {'angle':'overhead','camera':'pan','direction':'left','speed':'slow'})
        result = composition_brief.shot_props({'camera':'A static camera with no panning or tilting.'})
        self.assertEqual(result, {'camera':'static'})
        result = composition_brief.shot_props({'camera':'No panning initially. The camera then pans right.'})
        self.assertEqual(result, {'camera':'pan','direction':'right'})
        video = {'windows':[{'analysis':{'events':[{'camera':'No panning initially. The camera then pans right.'}]}}]}
        self.assertEqual(composition_brief.video_camera(video),result)
        self.assertEqual(composition_brief.shot_props({'camera':'No significant camera movement.'}), {'camera':'static'})

    def test_utterances_merge_and_mark_cut_crossing(self):
        segments = [{'start_time': .8, 'end_time': 1., 'text': '人', 'speaker_label': 's', 'character_context_id': 'a', 'character_name': 'A'},
                    {'start_time': 1., 'end_time': 1.2, 'text': 'には', 'speaker_label': 's', 'character_context_id': 'a', 'character_name': 'A'},
                    {'start_time': 1.2, 'end_time': 1.4, 'text': '見える', 'speaker_label': 's', 'character_context_id': 'a', 'character_name': 'A'},
                    {'start_time': 1.4, 'end_time': 1.5, 'text': 'Yes', 'speaker_label': 'other', 'character_context_id': None}]
        result = composition_brief.utterances(segments, 1, 2)
        self.assertEqual(result[0]['text'], 'には見える')
        self.assertTrue(result[0]['crosses_cut'])
        self.assertEqual(result[0]['t0'], 0)
        self.assertEqual(result[1]['character_name'], 'unknown')

    def test_utterances_preserve_word_boundaries_across_scripts(self):
        def merged(fragments):
            segments = [{'start_time':i, 'end_time':i+0.5, 'text':text, 'speaker_label':'s'}
                        for i,text in enumerate(fragments)]
            return composition_brief.utterances(segments,0,len(fragments))[0]['text']
        self.assertEqual(merged(['Olá', 'mundo']), 'Olá mundo')
        self.assertEqual(merged(['مرحبا', 'بالعالم']), 'مرحبا بالعالم')
        self.assertEqual(merged(['人', 'には', '見える']), '人には見える')
        self.assertEqual(merged(['Hello, ', 'world']), 'Hello, world')

    def test_summary_validation(self):
        good = '### Scene\nA store.\n### Opening Frame\nA person.\n### Animation Timeline\n[0–1] Holds still.\n### Dialogue\nNo dialogue timing supplied.\n### Continuity & Constraints\nStatic camera.'
        self.assertEqual(inference_worker.validate_production_brief('<think>private</think>\n' + good), good)
        roomy = good + ' ' + 'x' * 3500  # a brief with activity, emotion and speaking per character needs more than the old 3 KB
        self.assertEqual(inference_worker.validate_production_brief(roomy), roomy)
        self.assertEqual(inference_worker.validate_production_brief('<analysis>private</analysis><reasoning>private</reasoning>\n' + good), good)
        for bad in (fixture(48)['results'][0]['type'], good + ('x'*7169), good + "\n[The model's response was cut off and may be incomplete.]", '<think>unfinished', '<analysis>unfinished'):
            with self.assertRaises(ValueError): inference_worker.validate_production_brief(bad)
        for timeline in ('Holds still.', '[1–0] Holds still.', '[0–3] Holds still.', '[0–1]', '[1–2] Turns.\n[0–1] Stands.'):
            with self.assertRaises(ValueError):
                inference_worker.validate_production_brief(good.replace('[0–1] Holds still.',timeline),duration=2)
        self.assertEqual(inference_worker.validate_production_brief(good.replace('[0–1]', '[0–2.04s]'),duration=2),good.replace('[0–1]', '[0–2.04s]'))

    def test_summary_sections_no_longer_include_dialogue(self):
        new = '### Scene\nA store.\n### Opening Frame\nA person.\n### Animation Timeline\n[0–1] Holds still.\n### Continuity & Constraints\nStatic camera.'
        self.assertEqual(inference_worker.validate_production_brief(new, duration=2), new)
        legacy = new.replace('### Continuity', '### Dialogue\nNo dialogue timing supplied.\n### Continuity')
        self.assertEqual(inference_worker.validate_production_brief(legacy, duration=2), legacy)  # an old summary that is edited still saves
        for bad in (new.replace('### Scene', '### Overview'), new.replace('### Opening Frame\nA person.\n', ''), new.replace('Static camera.', ''), new + '\n### Dialogue\nx'):
            with self.assertRaises(ValueError): inference_worker.validate_production_brief(bad)
        self.assertEqual(inference_worker.BRIEF_SECTIONS, ['Scene', 'Opening Frame', 'Performance Beats', 'Acting Notes', 'Continuity & Constraints'])
        for rule in ('Dialogue is not part of this brief', 'pose_detail', 'state_correction', 'Return all six headings', 'Performance Director',
                     'the keyframes win', '[start–end seconds] Name: body; face; gaze; mouth (delivery). Intent:', 'at most 7168 UTF-8 bytes'):
            self.assertIn(rule, inference_worker.PRODUCTION_BRIEF_SYSTEM)
        self.assertNotIn('### Dialogue', inference_worker.PRODUCTION_BRIEF_SYSTEM)
        self.assertNotIn('### Animation Timeline', inference_worker.PRODUCTION_BRIEF_SYSTEM)

    def test_the_performance_director_brief_has_five_sections_and_six_kilobytes(self):
        brief = ('### Scene\nA store.\n### Opening Frame\nHero stands left.\n### Performance Beats\n[0.0–2.0] Camera: static.\n'
                 '[0.0–1.0] Hero: stands; calm; looks at Mia; mouth moves as in speech (lip-synced). Intent: greets her.\n[2.0] End state: Hero stands.\n'
                 '### Acting Notes\nHero stays warm and steady.\n### Continuity & Constraints\nStatic camera.')
        self.assertEqual(inference_worker.validate_production_brief(brief, duration=2), brief)
        with self.assertRaisesRegex(ValueError, 'Performance Beats requires time-coded lines'):
            inference_worker.validate_production_brief(brief.replace('[0.0–1.0] Hero', 'Hero'), duration=2)
        with self.assertRaisesRegex(ValueError, 'Acting Notes'):  # the error names the required sections
            inference_worker.validate_production_brief(brief.replace('### Acting Notes\nHero stays warm and steady.\n', ''), duration=2)
        long = brief.replace('Hero stays warm and steady.', 'Hero stays warm. ' * 300)
        self.assertGreater(len(long.encode()), 4096)
        self.assertLess(len(long.encode()), 7168)
        self.assertEqual(inference_worker.validate_production_brief(long, duration=2), long)
        with self.assertRaisesRegex(ValueError, 'exceeds 7168 bytes'):
            inference_worker.validate_production_brief(brief.replace('Hero stays warm and steady.', 'x ' * 3700), duration=2)

    CAMERA_SECTION = ('### Cinematic Camera\n- Camera body: ARRI Alexa 35\n- Lens: Cooke S4/i spherical primes\n- Focal length: 35 mm\n- Aperture: T2.8\n'
                      '- Lighting: soft window key from camera left over warm practicals\n- Color grading: muted teal shadows with warm highlights\n'
                      '- Image texture: fine grain with gentle halation')
    FIVE_SECTIONS = ('### Scene\nA store.\n### Opening Frame\nHero stands left.\n### Performance Beats\n[0.0–2.0] Camera: static.\n'
                     '[0.0–1.0] Hero: stands; calm; looks at Mia; mouth stays closed. Intent: waits.\n[2.0] End state: Hero stands.\n'
                     '### Acting Notes\nHero stays calm.\n### Continuity & Constraints\nStatic camera.')

    def test_a_model_answer_must_end_with_the_cinematic_camera_section(self):
        validate = inference_worker.validate_production_brief
        self.assertEqual(inference_worker.CAMERA_BRIEF_SECTIONS, [*inference_worker.BRIEF_SECTIONS, 'Cinematic Camera'])
        full = self.FIVE_SECTIONS + '\n' + self.CAMERA_SECTION
        self.assertEqual(validate(full, duration=2, camera=True), full)
        self.assertEqual(validate(full, duration=2), full)  # an edit of a summary that has the section saves
        self.assertEqual(validate(self.FIVE_SECTIONS, duration=2), self.FIVE_SECTIONS)  # and so does an older summary without it
        with self.assertRaisesRegex(ValueError, 'Cinematic Camera'):  # the sentence that goes back to the model names the section
            validate(self.FIVE_SECTIONS, duration=2, camera=True)
        with self.assertRaises(ValueError):  # it is the last section, after the constraints
            validate(self.FIVE_SECTIONS.replace('### Acting Notes', self.CAMERA_SECTION + '\n### Acting Notes'), duration=2, camera=True)
        camera_lines = self.CAMERA_SECTION.splitlines()
        for index, label in ((1, 'Camera body'), (2, 'Lens'), (3, 'Focal length'), (4, 'Aperture'), (5, 'Lighting'), (6, 'Color grading'), (7, 'Image texture')):
            missing = '\n'.join(line for i, line in enumerate(camera_lines) if i != index)
            with self.assertRaisesRegex(ValueError, f'{label} is missing'):
                validate(self.FIVE_SECTIONS + '\n' + missing, duration=2, camera=True)
            for answer in ('unknown', 'N/A', 'Not established.', ''):
                blank = camera_lines[index].split(':')[0] + ': ' + answer
                with self.assertRaisesRegex(ValueError, f'{label} (?:has no decision|is missing)|leaked'):  # an empty last line also reads as a cut-off answer
                    validate(self.FIVE_SECTIONS + '\n' + '\n'.join([*camera_lines[:index], blank, *camera_lines[index + 1:]]), duration=2, camera=True)
        # The labels may be written in bold, in another order, or with the British spelling.
        loose = ['### Cinematic Camera', '- **Lens:** Cooke S4/i', '**Camera body**: ARRI Alexa Mini LF', '* Focal length: 50 mm', '- Aperture: T2', '- Lighting: hard sun',
                 '- Colour grading: bleach bypass', '- Image texture: coarse grain']
        validate(self.FIVE_SECTIONS + '\n' + '\n'.join(loose), duration=2, camera=True)
        self.assertLess(len(self.CAMERA_SECTION.encode()), 800)  # the section fits the 1 KB the byte cap grew by

    def test_the_prompt_asks_for_the_camera_look_and_prefers_arri_alexa(self):
        system = inference_worker.PRODUCTION_BRIEF_SYSTEM
        for rule in ('### Cinematic Camera', 'Return all six headings', 'prefer the ARRI Alexa family', 'Alexa 35', 'Alexa Mini LF', 'Camera body, Lens, Focal length, Aperture, Lighting, Color grading, Image texture',
                     'Never mention anime', 'Camera movement stays in Performance Beats', 'The one deliberate exception is the Cinematic Camera section'):
            self.assertIn(rule, system)
        for field, _ in inference_worker.CAMERA_FIELDS:
            self.assertIn(field, system)
        self.assertEqual([heading for heading in re.findall(r'^### (.+)$', system, re.M)], inference_worker.CAMERA_BRIEF_SECTIONS)

    def test_summary_generation_retries_truncation_and_strips_thinking(self):
        good = '### Scene\nA store.\n### Opening Frame\nA person.\n### Animation Timeline\n[0–1] Holds still.\n### Dialogue\nNo dialogue timing supplied.\n### Continuity & Constraints\nStatic camera.'
        replies = [{'choices':[{'finish_reason':'length','message':{'content':'untrusted incomplete output'}}]},
                   {'choices':[{'finish_reason':'stop','message':{'content':'<think>private</think>'+good}}]}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY':'test'}), patch.object(inference_worker, 'request_json', side_effect=replies) as call:
            result, _ = inference_worker.chat_json('test/model','system','{}',[],plain_text=True,text_validator=inference_worker.validate_production_brief)
        self.assertEqual(result,good)
        self.assertEqual(call.call_count,2)
        repair = json.loads(call.call_args_list[1].args[0].data)
        self.assertNotIn('untrusted incomplete output',json.dumps(repair))

    def test_filtered_summary_is_retried_and_never_accepted_as_complete(self):
        good = '### Scene\nA store.\n### Opening Frame\nA person.\n### Animation Timeline\n[0–1] Holds still.\n### Dialogue\nNo dialogue timing supplied.\n### Continuity & Constraints\nStatic camera.'
        reply = {'choices':[{'finish_reason':'content_filter','message':{'content':good}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY':'test'}), patch.object(inference_worker, 'request_json', return_value=reply) as call:
            with self.assertRaisesRegex(RuntimeError,'without completing'):
                inference_worker.chat_json('test/model','system','{}',[],plain_text=True,text_validator=inference_worker.validate_production_brief)
        self.assertEqual(call.call_count,3)


class CompositionApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.temp.name)
        db.DB_PATH = Path(self.temp.name)/'app.sqlite3'
        db.init_db()
        self.data = fixture(49)
        cut = self.data['cut']
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Project')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio) VALUES('e','p','Episode','v.mp4','.mp4','ready','{}',300,1)")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES(?, 'e',0,?,?)", (cut['id'],cut['start_time'],cut['end_time']))
            names = {'1ab61da1-6b7b-44af-89cb-9ea73ebb83a9':'Mai Sakurajima', '290d19bd-b44d-4500-ad8e-f1b45fd6057a':'Sakuta Azusagawa'}
            for id, name in names.items():
                conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES(?,'p','character',?)", (id,name))
            for item in self.data['results']:
                kind,title = structured_results.KINDS[item['type']]
                context = {'cut': {'start_time':cut['start_time'], 'end_time':cut['end_time']}}
                if item['type']=='initial_frame': context['model_context_hints'] = {'character_detections': self.data['verified_detections']}
                conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,context_json) VALUES(?,?,?,?,?,?,?)", (item['id'],cut['id'],kind,title,json.dumps(item['content']),json.dumps({'structured_type':item['type']}),json.dumps(context)))
            for s in cut['dialogue']:
                conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id) VALUES(?,'e',?,?,?,?,?)", (s['id'],s['start_time'],s['end_time'],s['text'],s['speaker_label'],s['character_context_id']))
            run = self.data['detections']
            if run:
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES(?,?,?,?,?,?)", (run['id'],cut['id'],run['cut_start'],run['cut_end'],run['frames_json'],run['model_id']))
                for d in run['detections']:
                    conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES(?,?,?,?,?,?)", (d['id'],run['id'],d['position'],d['box_json'],d['character_context_id'],d['label_source']))
        folder = media.episode_dir('p','e')
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'original.mp4').write_bytes(b'video')
        with wave.open(str(folder/'audio.wav'),'wb') as w:
            w.setparams((1,2,100,0,'NONE','not compressed')); w.writeframes(b'\0\0'*30000)
        self.client = TestClient(app)
        self.path = '/api/v1/episodes/e/brief?from=1&to=1&view=composition'

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.temp.cleanup()

    def test_projection_and_selection(self):
        response = self.client.get(self.path)
        self.assertEqual(response.status_code,200,response.text)
        result=response.json(); cut=result['cuts'][0]
        self.assertTrue(cut['ready'])
        self.assertEqual(cut['shot'], {'size':'MCU','angle':'eye','camera':'static'})
        self.assertAlmostEqual(cut['duration'],self.data['cut']['end_time']-self.data['cut']['start_time'],places=6)
        self.assertNotIn('start_time',cut)
        self.assertEqual(cut['subjects'][0]['name'],'Mai Sakurajima')
        self.assertEqual(cut['subjects'][0]['identity'],'confirmed')
        self.assertIn('carrot',cut['subjects'][0]['hands'])
        self.assertIn('blazer',cut['subjects'][0]['outfit_visible'])
        self.assertIn('turns her head',cut['beats'][0]['action'])
        self.assertIn('head and upper body are visible',cut['beats'][0]['action'])
        self.assertIn('face is not clearly seen',cut['beats'][0]['action'])
        self.assertEqual(cut['beats'][0]['subject'],0)
        self.assertEqual(cut['dialogue'][0]['text'],'人にはどう見えてるんですか？')
        self.assertNotIn('production_brief',cut)
        self.assertNotIn('initial_frame',cut)
        self.assertNotIn('audio_url',cut)
        removed = {'updated_at', 'analysis_rev', 'confidence', 'unavailable', 'links', 'issues'}
        self.assertTrue(removed.isdisjoint(cut))
        self.assertNotIn('unavailable_reasons', self.client.get(self.path).json())
        self.assertTrue(any(b['t1'] > 2 for b in cut['beats']))
        selected=self.client.get(self.path+'&fields=shot&include=dialogue_segments').json()['cuts'][0]
        self.assertIn('shot',selected); self.assertNotIn('setting',selected)
        self.assertGreater(len(selected['dialogue_segments']),2)
        self.assertEqual(self.client.get(self.path+'&include=nonsense').status_code,422)
        for name in removed:
            self.assertEqual(self.client.get(self.path+'&fields='+name).status_code, 422)
        actions = self.client.get(self.path+'&fields=beats,end_state').json()['cuts'][0]
        self.assertNotIn('subjects',actions)
        self.assertEqual(actions['beats'][0]['subject'],'Mai Sakurajima')
        self.assertTrue(any(s['subject']=='Mai Sakurajima' for s in actions['end_state']['subjects']))

    def test_subtitles_come_from_the_episode_track_and_are_not_needed_for_readiness(self):
        cut = self.data['cut']
        start = cut['start_time']
        with db.connect() as conn:
            conn.execute("DELETE FROM analysis_results WHERE cut_id=? AND json_extract(settings_json,'$.structured_type')='subtitle_ocr'", (cut['id'],))
        brief = self.client.get(self.path).json()['cuts'][0]
        self.assertTrue(brief['ready'])                      # a per-cut subtitle result is no longer required
        self.assertNotIn('subtitles', brief)
        with db.connect() as conn:
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,language,language_code,status) VALUES ('run','e','Japanese','ja','complete')")
            conn.executemany("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES (?,?,?,?,?,?,?,?)",
                             [('before', 'run', 'e', 0, start - 5, start - 1, 'Before the cut', 'high'),
                              ('begins', 'run', 'e', 1, start - 1, start + .5, 'Begins earlier', 'high'),
                              ('inside', 'run', 'e', 2, start + 1.5, start + 3, 'Inside the cut', 'low')])
        brief = self.client.get(self.path).json()['cuts'][0]
        self.assertEqual(brief['subtitles'], [{'text': 'Begins earlier', 'lang': 'ja', 't': 0}, {'text': 'Inside the cut', 'lang': 'ja', 't': 1.5}])

    def test_older_per_cut_subtitles_are_used_until_the_episode_is_extracted(self):
        brief = self.client.get(self.path).json()['cuts'][0]
        legacy = next(r['content'] for r in self.data['results'] if r['type'] == 'subtitle_ocr')
        cut = self.data['cut']
        expected = composition_brief.subtitle_lines(legacy['events'], cut['start_time'], cut['end_time'])
        self.assertEqual(brief.get('subtitles', []), expected)

    def test_openapi_and_guide_describe_composition_contract(self):
        spec = self.client.get('/openapi.json').json()
        path = spec['paths']['/api/v1/episodes/{episode_id}/brief']['get']
        names = {parameter['name'] for parameter in path['parameters']}
        self.assertTrue({'from','to','view','include','fields','only_ready','updated_since'} <= names)
        models = spec['components']['schemas']
        self.assertTrue({'CompositionBrief','CompositionCut','CompositionShot','CompositionUtterance'} <= models.keys())
        description = models['CompositionUtterance']['properties']['text']['description']
        self.assertIn('never truncated',description)
        guide = self.client.get('/api/integration-guide')
        self.assertEqual(guide.status_code,200)
        self.assertIn('view=composition',guide.text)
        self.assertTrue({'updated_at', 'analysis_rev', 'confidence', 'unavailable', 'links', 'issues'}.isdisjoint(models['CompositionCut']['properties']))
        self.assertNotIn('unavailable_reasons', models['CompositionBrief']['properties'])
        self.assertNotIn('unavailable', models['BriefCut']['properties'])

    def test_oversized_dialogue_is_preserved_in_full(self):
        text = 'An unusually long transcript. ' * 650
        start = self.data['cut']['start_time']
        with db.connect() as conn:
            conn.execute("DELETE FROM transcript_segments WHERE episode_id='e'")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label) VALUES('long','e',?,?,?,'speaker')",
                         (start, start + 1, text))
        response = self.client.get(self.path)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(len(response.content), 2048)
        self.assertEqual(response.json()['cuts'][0]['dialogue'][0]['text'], text.strip())

    def test_invalid_summary_timing_never_replaces_current_revision(self):
        cut_id = self.data['cut']['id']
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES('prior',?,'cut_description','Cut summary','Prior revision','model')",(cut_id,))
        response = self.client.post(f'/api/cuts/{cut_id}/regenerate-summary',json={'mode':'regenerate'})
        self.assertEqual(response.status_code,200,response.text)
        job_id = response.json()['job_id']
        bad = ('### Scene\nA store.\n### Opening Frame\nA person.\n### Performance Beats\n[0–99] Camera: a person turns.\n### Acting Notes\nWaits.\n'
               '### Continuity & Constraints\nStatic camera.\n### Cinematic Camera\n- Camera body: ARRI Alexa 35\n- Lens: Cooke S4/i\n- Focal length: 35 mm\n'
               '- Aperture: T2.8\n- Lighting: soft key\n- Color grading: muted\n- Image texture: fine grain')
        result = {'choices':[{'finish_reason':'stop','message':{'content':bad}}]}
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}), patch.object(inference_worker,'request_json',return_value=result) as call:
            inference_worker.process_regen(*worker.claim_job(('regen',)))
        self.assertEqual(call.call_count,3)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?',(job_id,))['status'],'failed')
        current = db.all_rows("SELECT id,content FROM analysis_results WHERE cut_id=? AND title='Cut summary' AND superseded_by IS NULL",(cut_id,))
        self.assertEqual(current,[{'id':'prior','content':'Prior revision'}])
        cut = self.client.get(self.path).json()['cuts'][0]
        self.assertTrue(cut['ready'])
        absolute = self.client.get(self.path+'&fields=start_time,end_time').json()['cuts'][0]
        self.assertEqual(absolute['start_time'],self.data['cut']['start_time'])
        self.assertEqual(absolute['end_time'],self.data['cut']['end_time'])

    def test_pending_never_contains_partial_data(self):
        cut=self.data['cut']
        with db.connect() as conn:
            conn.execute("DELETE FROM analysis_results WHERE cut_id=? AND json_extract(settings_json,'$.structured_type')='video_analysis'",(cut['id'],))
        result=self.client.get(self.path).json()
        self.assertEqual(result['pending'],[1]); self.assertFalse(result['cuts'][0]['ready'])
        self.assertNotIn('shot',result['cuts'][0])
        self.assertTrue({'updated_at', 'analysis_rev', 'confidence', 'unavailable', 'links', 'issues'}.isdisjoint(result['cuts'][0]))
        selected = self.client.get(self.path+'&fields=shot').json()['cuts'][0]
        self.assertEqual(set(selected), {'id', 'cut_number', 'ready'})
        self.assertEqual(self.client.get(self.path+'&only_ready=true').json()['cuts'],[])

    def test_changed_cut_boundaries_make_visual_snapshot_unready(self):
        with db.connect() as conn:
            conn.execute('UPDATE cuts SET end_time=end_time+.1 WHERE id=?',(self.data['cut']['id'],))
        cut=self.client.get(self.path).json()['cuts'][0]
        self.assertFalse(cut['ready'])
        self.assertNotIn('unavailable',cut)
        self.assertNotIn('subjects',cut)

    def test_incomplete_active_analysis_does_not_mix_revisions(self):
        cut_id = self.data['cut']['id']
        with db.connect() as conn:
            payload = json.dumps({'cuts':[{'cut_id':cut_id}]})
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES('running','e','analyze','running',?)",(payload,))
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json,completed) VALUES('running',?,'{}',0)",(cut_id,))
        cut = self.client.get(self.path).json()['cuts'][0]
        self.assertFalse(cut['ready'])
        self.assertNotIn('shot',cut)

    def test_incremental_reads_include_character_relabels(self):
        original = self.client.get(self.path).json()['cuts'][0]
        timestamp = db.one('SELECT updated_at FROM composition_changes WHERE cut_id=?', (original['id'],))['updated_at']
        with db.connect() as conn:
            conn.execute("UPDATE context_items SET title='Renamed' WHERE id='1ab61da1-6b7b-44af-89cb-9ea73ebb83a9'")
        changed = self.client.get(self.path+'&updated_since='+timestamp).json()['cuts'][0]
        self.assertEqual(changed['subjects'][0]['name'],'Renamed')
        self.assertEqual(self.client.get(self.path+'&updated_since=2099-01-01T00:00:00Z').json()['cuts'],[])

    def test_projection_upgrade_advances_incremental_watermark_once(self):
        with patch.object(composition_brief, 'projection_fingerprint', return_value='version-one'):
            composition_brief.register_projection()
        with db.connect() as conn:
            conn.execute("UPDATE composition_changes SET updated_at='2000-01-01T00:00:00.000Z'")
        with patch.object(composition_brief, 'projection_fingerprint', return_value='version-one'):
            composition_brief.register_projection()
        self.assertEqual(self.client.get(self.path+'&updated_since=2001-01-01T00:00:00Z').json()['cuts'],[])
        with patch.object(composition_brief, 'projection_fingerprint', return_value='version-two'):
            composition_brief.register_projection()
        updated = self.client.get(self.path+'&updated_since=2001-01-01T00:00:00Z').json()['cuts']
        self.assertEqual(len(updated),1)
        self.assertEqual(updated[0]['id'],self.data['cut']['id'])

    def use_motion_direction(self, drop_last_frame=True, boxed_at_end=False, end='left frame'):
        """Turn the fixture into a cut analyzed by the current flow: emotion in the first frame, a motion direction, no last frame."""
        cut_id = self.data['cut']['id']
        initial = next(r for r in self.data['results'] if r['type'] == 'initial_frame')
        video = next(r for r in self.data['results'] if r['type'] == 'video_analysis')
        frame = json.loads(json.dumps(initial['content']))
        frame.update(mood='light, teasing', props='orange carrot held by Mai Sakurajima')
        frame['visible_characters'][0].update(emotion='amused, relaxed', gaze='at Sakuta Azusagawa')
        mai = '1ab61da1-6b7b-44af-89cb-9ea73ebb83a9'
        content = {**video['content'], 'contract': 'direction-v1', 'direction': {
            'summary': 'Mai Sakurajima turns to Sakuta Azusagawa and speaks while the camera follows her.',
            'camera': {'movement': 'static → pan left', 'detail': 'slow, follows Mai Sakurajima'},
            'characters': [{'character': 'Mai Sakurajima', 'character_context_id': mai, 'activity': 'standing → walking', 'travel': 'left', 'motion': 'Turns her head, then steps left.',
                            'speaking': 'yes', 'emotion': 'amused → serious', 'end': end},
                           {'character': 'boy with a basket', 'character_context_id': None, 'activity': 'walking', 'travel': 'enters from right', 'motion': 'Walks in.', 'speaking': 'no', 'emotion': 'not readable', 'end': 'right, facing left'}],
            **({'end_state': {'in_frame': ['Mai Sakurajima'], 'source': 'character boxes of the last frame'}} if boxed_at_end else {})}}
        if boxed_at_end:
            content['direction']['characters'][0]['in_last_frame'] = True
        with db.connect() as conn:
            conn.execute('UPDATE analysis_results SET content=? WHERE id=?', (json.dumps(frame), initial['id']))
            conn.execute('UPDATE analysis_results SET content=? WHERE id=?', (json.dumps(content), video['id']))
            if drop_last_frame:
                conn.execute("DELETE FROM analysis_results WHERE cut_id=? AND json_extract(settings_json,'$.structured_type')='last_frame'", (cut_id,))

    def test_motion_direction_makes_a_cut_ready_without_a_last_frame(self):
        self.use_motion_direction()
        response = self.client.get(self.path)
        self.assertEqual(response.status_code, 200, response.text)  # the response model accepts the new fields
        cut = response.json()['cuts'][0]
        self.assertTrue(cut['ready'])
        self.assertNotIn('last_frame_id', cut)
        self.assertEqual((cut['shot']['camera'], cut['shot']['direction'], cut['shot']['camera_detail']), ('pan', 'left', 'slow, follows Mai Sakurajima'))
        self.assertEqual((cut['mood'], cut['props']), ('light, teasing', 'orange carrot held by Mai Sakurajima'))
        self.assertIn('turns to Sakuta', cut['action'])
        mai = cut['subjects'][0]
        self.assertEqual((mai['name'], mai['emotion'], mai['gaze']), ('Mai Sakurajima', 'amused, relaxed', 'at Sakuta Azusagawa'))
        self.assertEqual(mai['motion'], {'activity': 'standing → walking', 'travel': 'left', 'motion': 'Turns her head, then steps left.', 'speaking': 'yes',
                                         'emotion': 'amused → serious', 'end': 'left frame'})
        # The video analysis says she left and no last-frame box contradicts it; someone who is not an opening subject is listed with their motion.
        self.assertEqual(cut['end_state']['subjects'][0], {'subject': 0, 'change': 'left frame'})
        stranger = cut['end_state']['subjects'][1]
        self.assertEqual((stranger['subject'], stranger['change'], stranger['motion']['travel'], stranger['motion']['speaking']), ('boy with a basket', 'entered frame', 'enters from right', 'no'))
        self.assertNotIn('emotion', stranger['motion'])  # "not readable" is not a fact
        self.assertIn('turns her head', cut['beats'][0]['action'])  # timed events still become beats
        selected = self.client.get(self.path + '&fields=mood,action').json()['cuts'][0]
        self.assertEqual(set(selected), {'id', 'cut_number', 'ready', 'mood', 'action'})

    def test_a_leftover_last_frame_is_ignored_once_the_cut_has_a_motion_direction(self):
        self.use_motion_direction(drop_last_frame=False)
        cut = self.client.get(self.path).json()['cuts'][0]
        self.assertTrue(cut['ready'])
        self.assertNotIn('last_frame_id', cut)
        self.assertEqual([item.get('change') for item in cut['end_state']['subjects']], ['left frame', 'entered frame'])
        included = self.client.get(self.path + '&include=last_frame').json()['cuts'][0]
        self.assertEqual(included['last_frame']['type'], 'last_frame')  # still retrievable on request

    def test_nobody_is_reported_as_gone_without_the_video_saying_so_or_against_a_last_frame_box(self):
        # No last-frame box and the video says she is still there: a missing box proves nothing.
        self.use_motion_direction(end='left edge, facing away')
        subjects = self.client.get(self.path).json()['cuts'][0]['end_state']['subjects']
        self.assertEqual([item.get('change') for item in subjects], ['entered frame'])
        # The video says she left but a named box on the last frame shows her: the box wins.
        self.use_motion_direction(boxed_at_end=True)
        subjects = self.client.get(self.path).json()['cuts'][0]['end_state']['subjects']
        self.assertEqual([item.get('change') for item in subjects], ['entered frame'])

    def test_an_older_video_analysis_still_needs_its_last_frame(self):
        with db.connect() as conn:
            conn.execute("DELETE FROM analysis_results WHERE cut_id=? AND json_extract(settings_json,'$.structured_type')='last_frame'", (self.data['cut']['id'],))
        result = self.client.get(self.path).json()
        self.assertEqual((result['cuts'][0]['ready'], result['pending']), (False, [1]))

    def test_stale_detection_is_reported_and_not_used_as_identity(self):
        with db.connect() as conn:
            conn.execute('UPDATE character_detection_runs SET cut_end=cut_end+.1 WHERE id=?',(self.data['detections']['id'],))
        cut = self.client.get(self.path).json()['cuts'][0]
        self.assertIsNone(cut['subjects'][0]['character_context_id'])

    def test_end_state_matches_character_ids_without_assuming_list_order(self):
        item = next(r for r in self.data['results'] if r['type'] == 'last_frame')
        frame = dict(item['content'])
        frame['visible_characters'] = list(reversed(frame['visible_characters']))
        with db.connect() as conn:
            conn.execute('UPDATE analysis_results SET content=? WHERE id=?',(json.dumps(frame),item['id']))
        cut = self.client.get(self.path).json()['cuts'][0]
        mai = next(s for s in cut['end_state']['subjects'] if s['subject']==0)
        self.assertEqual(mai['expression'],'slight smile')

    def test_export_waits_for_analysis_that_writes_its_inputs(self):
        cut = self.data['cut']
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json,lease_owner,lease_until) VALUES('running','e','analyze','running','{}','owner',9999999999)")
            for kind, stamp in (('initial_frame',cut['start_time']+.1),('last_frame',cut['end_time']-.1)):
                row=conn.execute("SELECT id,context_json FROM analysis_results WHERE json_extract(settings_json,'$.structured_type')=?",(kind,)).fetchone()
                context=json.loads(row['context_json']); context['frame_time']=stamp
                conn.execute('UPDATE analysis_results SET context_json=? WHERE id=?',(json.dumps(context),row['id']))
        job=self.client.post('/api/v1/episodes/e/brief/exports?from=1&to=1&view=composition&include=last_frame').json()['job_id']
        self.client.post('/api/jobs/cancel-queued')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?',(job,))['status'],'queued')
        self.assertIsNone(worker.claim_job(('brief_export',)))
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done' WHERE id='running'")
        worker.release_lease('running', 'owner')
        claim=worker.claim_job(('brief_export',)); self.assertEqual(claim[0],job)
        samples=[]
        def extract(source, output, sample, start, end, **kwargs):
            self.assertIsNone(kwargs['width']); self.assertNotIn('exact_size', kwargs); output.write_bytes(b'jpeg')
            samples.append((sample,kwargs['last']))
        with patch('app.brief.media.extract_cut_frame',side_effect=extract): worker.process(*claim)
        self.assertEqual(samples,[(cut['start_time']+.1,False),(cut['end_time']-.1,False)])
        result=self.client.get('/api/v1/brief-exports/'+job).json()
        self.assertEqual(result['status'],'done')
        with zipfile.ZipFile(io.BytesIO(self.client.get(result['download_url']).content)) as z:
            manifest=json.loads(z.read('manifest.json'))[0]
            self.assertTrue(manifest['frame'].endswith('-initial.jpg'))
            self.assertTrue(manifest['last_frame'].endswith('-last.jpg'))
            self.assertTrue(manifest['has_speech']); self.assertGreater(manifest['audio_seconds'],0)
            exported = json.loads(z.read('brief.json'))
            self.assertEqual(exported['cuts'][0]['cut_number'],1)
            self.assertNotIn('unavailable_reasons', exported)
            self.assertTrue({'updated_at', 'analysis_rev', 'confidence', 'unavailable', 'links', 'issues'}.isdisjoint(exported['cuts'][0]))

    def test_compaction_job_is_revision_safe_and_records_provenance(self):
        record = self.client.get(self.path).json()['cuts'][0]
        values = composition_compaction.text_fields(record)
        names = composition_compaction.names_for(record)
        request = {'episode_id':'e','cut_id':record['id'],'values':values,'names':names,
                   'input_hash':composition_compaction.input_hash(values,names)}
        job = composition_compaction.enqueue([request])[0]
        self.assertEqual(composition_compaction.enqueue([request]),[job])
        claim = worker.claim_job(('composition_compact',))
        output = json.dumps({'values':values},ensure_ascii=False)
        with patch.object(inference_worker,'chat_json',return_value=(output,'test/model')):
            composition_compaction.process(*claim)
        worker.release_lease(*claim)
        cached = db.one('SELECT * FROM composition_compactions WHERE cut_id=?',(record['id'],))
        self.assertEqual((cached['input_hash'],cached['model_id'],cached['job_id']),(request['input_hash'],'test/model',job))
        self.assertIn('condense visual facts',cached['prompt'])
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")['status'],'ready')
        shortened = dict(values)
        shortened['subjects.0.hands'] = values['subjects.0.hands'].replace(', ', ' ').replace(' an ', ' ')
        composition_compaction.validate_response(json.dumps({'values':shortened}),values,names)
        with db.connect() as conn:
            conn.execute('UPDATE composition_compactions SET values_json=? WHERE cut_id=?',(json.dumps(shortened),record['id']))
        compact = self.client.get(self.path).json()['cuts'][0]
        self.assertEqual(compact['subjects'][0]['hands'],shortened['subjects.0.hands'])
        self.assertEqual(compact['subjects'][0]['character_context_id'],record['subjects'][0]['character_context_id'])

        frame_id = next(item['id'] for item in self.data['results'] if item['type']=='initial_frame')

        def change_pose(object_name):
            with db.connect() as conn:
                frame = json.loads(conn.execute('SELECT content FROM analysis_results WHERE id=?',(frame_id,)).fetchone()['content'])
                frame['visible_characters'][0]['pose'] = f'Sitting. Her right hand holds a {object_name}.'
                conn.execute('UPDATE analysis_results SET content=? WHERE id=?',(json.dumps(frame),frame_id))

        restarted = composition_compaction.enqueue([request])[0]
        change_pose('green pear')

        def answer(model, system, body, *args, **kwargs):
            self.assertIn('pear', body)
            change_pose('green apple')
            return body, 'test/model'

        claim = worker.claim_job(('composition_compact',))
        with patch.object(inference_worker, 'chat_json', side_effect=answer):
            composition_compaction.process(*claim)
        worker.release_lease(*claim)
        self.assertEqual(db.one('SELECT stage FROM jobs WHERE id=?', (restarted,))['stage'], 'Inputs changed, restarting')
        self.assertEqual(db.one('SELECT input_hash FROM composition_compactions WHERE cut_id=?',(record['id'],))['input_hash'],request['input_hash'])
        refreshed = self.client.get(self.path).json()['cuts'][0]
        self.assertIn('apple',refreshed['subjects'][0]['hands'])
        self.assertNotIn('carrot',refreshed['subjects'][0]['hands'])

    def test_compaction_yields_queue_priority_to_user_inference(self):
        record = self.client.get(self.path).json()['cuts'][0]
        values, names = composition_compaction.text_fields(record), composition_compaction.names_for(record)
        request = {'episode_id':'e','cut_id':record['id'],'values':values,'names':names,
                   'input_hash':composition_compaction.input_hash(values,names)}
        compact_job = composition_compaction.enqueue([request])[0]
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES('user-job','e','regen','{}')")
        self.assertEqual(worker.claim_job(('composition_compact','regen'))[0],'user-job')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?',(compact_job,))['status'],'queued')

    def test_cancelled_compaction_can_retry_without_changing_episode_status(self):
        record = self.client.get(self.path).json()['cuts'][0]
        values, names = composition_compaction.text_fields(record), composition_compaction.names_for(record)
        request = {'episode_id':'e','cut_id':record['id'],'values':values,'names':names,
                   'input_hash':composition_compaction.input_hash(values,names)}
        job = composition_compaction.enqueue([request])[0]
        self.assertEqual(self.client.post('/api/jobs/'+job+'/cancel').status_code,200)
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")['status'],'ready')
        retry = self.client.post('/api/jobs/'+job+'/retry')
        self.assertEqual(retry.status_code,200,retry.text)
        self.assertEqual(db.one('SELECT retry_of FROM jobs WHERE id=?',(retry.json()['job_id'],))['retry_of'],job)
        self.assertEqual(self.client.post('/api/jobs/'+job+'/retry').json(), retry.json())

    def test_completed_job_can_be_regenerated_if_its_cache_disappears(self):
        record = self.client.get(self.path).json()['cuts'][0]
        values, names = composition_compaction.text_fields(record), composition_compaction.names_for(record)
        request = {'episode_id':'e','cut_id':record['id'],'values':values,'names':names,
                   'input_hash':composition_compaction.input_hash(values,names)}
        first = composition_compaction.enqueue([request])[0]
        output = json.dumps({'values':values},ensure_ascii=False)
        with patch.object(inference_worker,'chat_json',return_value=(output,'test/model')):
            composition_compaction.process(*worker.claim_job(('composition_compact',)))
        with db.connect() as conn:
            conn.execute('DELETE FROM composition_compactions WHERE cut_id=?',(record['id'],))
        second = composition_compaction.enqueue([request])[0]
        self.assertNotEqual(first,second)


class CompositionRangeSizeTest(unittest.TestCase):
    """Range fixture preserves the user-provided episode evidence, not model summaries."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.temp.name)
        db.DB_PATH = Path(self.temp.name)/'app.sqlite3'
        db.init_db()
        data = [fixture(n) for n in range(44,69)]
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Project')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio) VALUES('e','p','Episode','v.mp4','.mp4','ready','{}',1000,1)")
            for item in data:
                cut=item['cut']
                conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES(?,\'e\',?,?,?)',(cut['id'],cut['cut_number']-1,cut['start_time'],cut['end_time']))
                for result in item['results']:
                    kind,title=structured_results.KINDS[result['type']]
                    conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,context_json) VALUES(?,?,?,?,?,?,?)',(result['id'],cut['id'],kind,title,json.dumps(result['content']),json.dumps({'structured_type':result['type']}),json.dumps({'cut':{'start_time':cut['start_time'],'end_time':cut['end_time']}})))
                run=item['detections']
                if run:
                    conn.execute('INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES(?,?,?,?,?,?)',(run['id'],cut['id'],run['cut_start'],run['cut_end'],run['frames_json'],run['model_id']))
                    for box in run['detections']:
                        if box['character_context_id']:
                            conn.execute("INSERT OR IGNORE INTO context_items(id,project_id,kind,title) VALUES(?,'p','character',?)",(box['character_context_id'],box['name'] or 'unknown'))
                        conn.execute('INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES(?,?,?,?,?,?)',(box['id'],run['id'],box['position'],box['box_json'],box['character_context_id'],box['label_source']))
                for seg in cut['dialogue']:
                    if seg['character_context_id']:
                        conn.execute("INSERT OR IGNORE INTO context_items(id,project_id,kind,title) VALUES(?,'p','character','unknown')",(seg['character_context_id'],))
                    conn.execute("INSERT OR IGNORE INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id) VALUES(?,'e',?,?,?,?,?)",(seg['id'],seg['start_time'],seg['end_time'],seg['text'],seg['speaker_label'],seg['character_context_id']))
        folder=media.episode_dir('p','e'); folder.mkdir(parents=True,exist_ok=True)
        (folder/'original.mp4').write_bytes(b'video')
        (folder/'audio.wav').write_bytes(b'prepared')
        self.client=TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.temp.cleanup()

    def test_range_contract_and_size_budget(self):
        response=self.client.get('/api/v1/episodes/e/brief?from=44&to=68&view=composition')
        self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        self.assertEqual((result['ready_count'],result['pending']),(25,[]))
        cut48 = result['cuts'][48-44]
        self.assertNotIn('frame_pos',cut48['end_state']['subjects'][0])
        self.assertNotIn('unavailable_reasons', result)
        for cut in result['cuts']:
            self.assertTrue({'updated_at', 'analysis_rev', 'confidence', 'unavailable', 'links', 'issues'}.isdisjoint(cut))
            self.assertNotIn('production_brief',cut)
            self.assertNotIn('initial_frame',cut)
            self.assertNotIn('audio_url',cut)
        sizes=[len(json.dumps(c,ensure_ascii=False,separators=(',',':')).encode('utf-8')) for c in result['cuts']]
        total=len(response.content)
        self.assertLessEqual(sum(sizes)/len(sizes),2560, f'average={sum(sizes)/len(sizes):.1f}, total={total}, per_cut={sizes}')
        self.assertLessEqual(total,64*1024, f'total={total}, per_cut={sizes}')
