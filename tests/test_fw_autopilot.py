"""scripts/fw autopilot commands (agent runs, inbox, escalate, close) against a fake API."""
import contextlib
import io
import unittest

from tests.test_fw_cli import CUTS, EPISODE, IDLE, FakeApi, fw, report


def fw_cuts():
    return CUTS


class AutopilotApi(FakeApi):
    def __init__(self, inbox=None):
        super().__init__()
        self.inbox = inbox

    def get(self, path, params=None):
        if path.endswith('/escalations'):
            self.calls.append((path, params or {}, None))
            return self.inbox
        if path.endswith('/agent-runs'):
            return [{'id': 'run1', 'status': 'finished', 'started_at': '2026-11-22 10:00:00', 'agent': 'claude-code', 'purpose': 'splits',
                     'decisions': {'total': 4, 'undone': 1, 'overridden': 0}}]
        return super().get(path, params)

    def post(self, path, params=None, body=None):
        self.calls.append((path, params or {}, body))
        if path.endswith('/agent-runs'):
            return {'id': 'run1', 'policy': {}}
        if path.endswith('/undo'):
            return {'undone': ['a', 'b'], 'conflicts': [{'decision_id': 'c', 'summary': 'Named the box Mai', 'reason': 'Changed since the agent decided'}]}
        return {'id': 'esc1', 'status': 'resolved', 'question': 'Who?'}


def run(api, *argv):
    args = fw.parser().parse_args(list(argv))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = args.func(api, args)
    return code, out.getvalue()


class FwAutopilotTest(unittest.TestCase):
    def test_escalate_names_cuts_by_their_editor_numbers(self):
        api = AutopilotApi()
        code, out = run(api, 'escalate', EPISODE['id'], '--kind', 'new_character', '--group', 'face:1', '--question', 'Who is the crying child?',
                        '--cuts', '2,4-5', '--option', 'New character', '--option', 'Background', '--proposal', '{"relevance": "always"}')
        self.assertEqual((code, out.strip()), (fw.OK, 'esc1'))
        path, _, body = api.calls[-1]
        self.assertEqual(path, f"/api/episodes/{EPISODE['id']}/escalations")
        self.assertEqual((body['cut_ids'], body['options'], body['proposal']), (['c2', 'c4', 'c5'], ['New character', 'Background'], {'relevance': 'always'}))
        with self.assertRaises(fw.Failure):
            run(api, 'escalate', EPISODE['id'], '--kind', 'x', '--group', 'g', '--question', 'q', '--cuts', '9')
        with self.assertRaises(fw.Failure):
            run(api, 'escalate', EPISODE['id'], '--kind', 'x', '--group', 'g', '--question', 'q', '--proposal', '{bad')

    def test_inbox_lists_questions_with_cut_links(self):
        inbox = {'counts': {'open_total': 2, 'applied_total': 7, 'open': {'voice': 1}, 'applied': {'line_splits': 7}, 'for_claude': 1, 'for_person': 1},
                 'items': [{'id': 'esc1', 'kind': 'voice', 'question': 'Whose voice is SPEAKER_04?', 'status': 'open', 'options': ['Mai', 'Kaede'],
                            'proposal': {'character': 'Mai'}, 'cut_ids': ['c4', 'c2']}]}
        code, out = run(AutopilotApi(inbox), 'inbox', EPISODE['id'])
        self.assertEqual(code, fw.OK)
        self.assertIn('1 open question(s) for you, 1 for Claude (the second reviewer), 7 decision(s) applied by agents', out)
        self.assertIn('applied: line_splits 7', out)
        self.assertIn('[voice] Whose voice is SPEAKER_04?', out)
        self.assertIn('cuts 2, 4', out)
        self.assertIn(f"- cut 2: http://framewise/#/episode/{EPISODE['id']}/cut/c2/analysis", out)

    def test_agent_runs_start_list_and_undo(self):
        api = AutopilotApi()
        self.assertEqual(run(api, 'agent', 'start', EPISODE['id'], '--agent', 'claude-code', '--purpose', 'splits'), (fw.OK, 'run1\n'))
        self.assertEqual(api.calls[-1][2], {'agent': 'claude-code', 'purpose': 'splits'})
        _, listed = run(api, 'agent', 'runs', EPISODE['id'])
        self.assertIn('4 decision(s), 1 undone', listed)
        _, undone = run(api, 'agent', 'undo', 'run1')
        self.assertIn('undid 2 decision(s)', undone)
        self.assertIn('kept (changed since): Named the box Mai', undone)
        self.assertEqual(api.calls[-1][0], '/api/agent-runs/run1/undo')
        run(api, 'agent', 'undo', 'dec1', '--decision')
        self.assertEqual(api.calls[-1][0], '/api/agent-decisions/dec1/undo')
        with self.assertRaises(fw.Failure):
            run(api, 'agent', 'finish')

    def test_diagnose_lists_cuts_worst_first_with_links_and_runnable_commands(self):
        health = {'checked': 5, 'counts': {'failed': 1, 'missing': 1}, 'episode_problems': [{'kind': 'voice', 'text': 'Voice SPEAKER_04: 3 line(s)', 'first_time': 4.5}],
                  'cuts': [{'cut': 2, 'cut_id': 'c2', 'busy': [], 'problems': [
                      {'kind': 'failed', 'text': 'Performance beats failed: a step it needs had not run', 'fix': 'queue the step it needs', 'place': 'analysis'},
                      {'kind': 'missing', 'text': 'Needs keyframe analysis', 'fix': {'text': 'queue keyframe analysis', 'command': 'fw visual EP 2'}, 'place': 'run'}]},
                           {'cut': 4, 'cut_id': 'c4', 'busy': ['summary'], 'problems': [{'kind': 'line_crossing', 'text': '1 transcript line(s) run across', 'place': 'analysis'}]}]}
        api = AutopilotApi()
        api.get = lambda path, params=None: health if path.endswith('/health') else AutopilotApi.get(api, path, params)
        code, out = run(api, 'diagnose', EPISODE['id'], '--kind', 'failed')
        self.assertEqual(code, fw.OK)
        self.assertIn(f"cut 2: http://framewise/#/episode/{EPISODE['id']}/cut/c2/analysis", out)
        self.assertIn('→ queue keyframe analysis  [fw visual ep-1111- 2]', out)
        self.assertNotIn('cut 4', out)  # filtered by --kind
        self.assertIn('Voice SPEAKER_04', out)

    def test_triage_plans_applies_and_exits_by_outcome(self):
        item = {'job_id': 'j1', 'label': 'Cut summary', 'cuts': [2], 'cause_text': 'validation failed', 'action': 'requeue', 'attempts': 0, 'limit': 1,
                'error': 'timeline lines', 'why': ''}
        api = AutopilotApi()
        api.get = lambda path, params=None: {'items': [item], 'counts': {'requeue': 1}, 'mode': 'apply', 'gpu_busy': False} \
            if path.endswith('/triage') else AutopilotApi.get(api, path, params)
        code, out = run(api, 'triage', EPISODE['id'])
        self.assertEqual(code, fw.OK)
        self.assertIn('WHAT TRIAGE WOULD DO', out)
        self.assertIn('queue again with the current settings (attempt 1 of 1)', out)
        escalated = {**item, 'action': 'escalate', 'why': 'the autopilot already fixed it 1 time(s)'}
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {
            'applied': [{**item, 'job_ids': ['new1']}], 'escalated': [escalated], 'waiting': [], 'left': [], 'mode': 'apply', 'closed_escalations': 1})[1]
        code, out = run(api, 'triage', EPISODE['id'], '--go', '--run', 'run1')
        self.assertEqual(code, fw.BLOCKED)
        self.assertEqual(api.calls[-1], (f"/api/episodes/{EPISODE['id']}/triage", None, {'run_id': 'run1'}))
        self.assertIn('queued: new1', out)
        self.assertIn('ASKED A PERSON', out)
        self.assertIn('closed 1 question(s)', out)

    def test_missed_cuts_check_report_and_answer(self):
        report = {'counts': {'clean': 3, 'apply': 1, 'grey': 1, 'unchecked': 0, 'changed_since': 0, 'analysis_off': 1},
                  'splits': [{'id': 'd1', 'summary': 'Split cut 2 at 0:03.000: a hard cut the detection missed', 'tier': 'rules', 'undone_at': None,
                              'overridden_at': None, 'after': {'left': {'id': 'c2'}}}],
                  'questions': [{'id': 'q1', 'reviewer': 'claude', 'question': 'Is there a hard cut in cut 4 at 0:09.000?',
                                 'evidence': {'strip': '/api/cuts/c4/missed-cut-strip?time=9.000000', 'cut_id': 'c4', 'time': 9.0}}]}
        api = AutopilotApi()
        api.get = lambda path, params=None: report if path.endswith('/missed-cuts') else AutopilotApi.get(api, path, params)
        code, out = run(api, 'missed-cuts', 'report', EPISODE['id'])
        self.assertEqual(code, fw.BLOCKED)
        self.assertIn('fw agent undo ID --decision', out)
        self.assertIn('[Claude] Is there a hard cut in cut 4', out)
        self.assertIn('picture: http://framewise/api/cuts/c4/missed-cut-strip?time=9.000000', out)
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'job_id': 'j1', 'job_ids': ['j1'], 'status': 'resolved', 'decision_id': 'd2'})[1]
        run(api, 'missed-cuts', 'check', EPISODE['id'], '3-9', '--redo')
        self.assertEqual(api.calls[-1], (f"/api/episodes/{EPISODE['id']}/missed-cuts", None, {'from_cut': 3, 'to_cut': 9, 'redo': True, 'run_id': None}))
        code, out = run(api, 'missed-cuts', 'answer', 'q1', 'no-cut', '--tier', 'claude', '--note', 'same shot')
        self.assertEqual(api.calls[-1][2], {'answer': 'no_cut', 'tier': 'claude', 'run_id': None, 'note': 'same shot'})
        with self.assertRaises(fw.Failure):
            run(api, 'missed-cuts', 'answer', 'q1', 'maybe')

    def test_line_splits_preview_and_run(self):
        preview = {'counts': {'split': 1, 'leave': 1}, 'applied': [],
                   'lines': [{'line_id': 'l1', 'start_time': 1.0, 'end_time': 3.2, 'text': 'あの人、どういう人なんだろう。', 'hard_cut': 1.9, 'cut': 2,
                              'cut_id': 'c2', 'action': 'split', 'why': 'on the cut, inside a pause', 'at': 1.9, 'left': 'あの人、', 'right': 'どういう人なんだろう。'},
                             {'line_id': 'l2', 'start_time': 5.0, 'end_time': 6.0, 'text': 'ねえ', 'hard_cut': 5.5, 'cut': 3, 'cut_id': 'c3',
                              'action': 'leave', 'why': 'no pause'}]}
        api = AutopilotApi()
        api.get = lambda path, params=None: preview if path.endswith('/line-splits') else AutopilotApi.get(api, path, params)
        code, out = run(api, 'line-splits', EPISODE['id'])
        self.assertEqual(code, fw.OK)
        self.assertIn('split at 1.90s: “あの人、” | “どういう人なんだろう。”', out)
        self.assertNotIn('ねえ', out)  # left lines only with --all
        self.assertIn('leave', run(api, 'line-splits', EPISODE['id'], '--all')[1])
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'job_id': 'j1', 'job_ids': ['j1']})[1]
        run(api, 'line-splits', EPISODE['id'], 'run', '--run', 'run1')
        self.assertEqual(api.calls[-1], (f"/api/episodes/{EPISODE['id']}/line-splits", None, {'run_id': 'run1'}))

    def test_review_decisions_run_report_and_answer(self):
        report = {'counts': {'apply': 1, 'claude': 1}, 'trust_needed': 0.9,
                  'trust': {'gaze': {'rate': 0.9, 'scored': 10, 'source': 'recommended'}, 'speaking': {'rate': None, 'scored': 3, 'source': 'too few'}},
                  'applied': [{'id': 'd1', 'summary': 'Answered 1 review question(s) (gaze; 1 recommended) [local]', 'undone_at': None, 'overridden_at': None,
                               'cut_id': 'c2', 'after': {'lines': [{'question': 'q1', 'topic': 'gaze', 'character': 'Mai', 'text': 'At Rio'}]}}],
                  'questions': [{'id': 'esc1', 'reviewer': 'claude', 'question': 'Cut 4: 1 review question(s) to decide (speaking · Mai)',
                                 'evidence': {'cut_id': 'c4', 'sheet': '/api/cuts/c4/review-sheet?labels=F2',
                                              'questions': [{'id': 'q2', 'question': 'Is Mai lip-synced?', 'recommended': 'b', 'proposal': 'a',
                                                             'why_asked': 'the decider and the reviewer disagree',
                                                             'options': [{'id': 'a', 'text': 'Yes'}, {'id': 'b', 'text': 'No'}]}]}}]}
        api = AutopilotApi()
        api.get = lambda path, params=None: report if path.endswith('/review-decisions') else AutopilotApi.get(api, path, params)
        code, out = run(api, 'review-decisions', 'report', EPISODE['id'])
        self.assertEqual(code, fw.BLOCKED)
        self.assertIn('gaze 90% (10, recommended)', out)
        self.assertIn('speaking — (3, too few)', out)
        self.assertIn('q1 gaze · Mai: At Rio', out)
        self.assertIn('[Claude] Cut 4', out)
        self.assertIn('a) Yes  b*) No', out)
        self.assertIn('pictures: http://framewise/api/cuts/c4/review-sheet?labels=F2', out)
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'job_ids': ['j1', 'j2'], 'skipped': {'answered': [3]}})[1]
        _, out = run(api, 'review-decisions', 'run', EPISODE['id'], '2-9', '--shadow')
        self.assertEqual(api.calls[-1][2], {'from_cut': 2, 'to_cut': 9, 'run_id': None, 'shadow': True, 'redo': False})
        self.assertIn('queued 2 shadow decision job(s)', out)
        api.get = lambda path, params=None: {'kind': 'review_question', 'evidence': {}} if path.startswith('/api/escalations/') else AutopilotApi.get(api, path, params)
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'status': 'handed_to_person', 'decision_id': 'd9', 'open': []})[1]
        _, out = run(api, 'review-decisions', 'answer', 'esc1', 'q2=a', 'q3=Kneeling by the bag', '--unsure', 'q4', '--tier', 'claude')
        self.assertEqual(api.calls[-1][0], '/api/escalations/esc1/review-answers')
        self.assertEqual(api.calls[-1][2]['answers'], [{'question': 'q2', 'option': 'a'}, {'question': 'q3', 'text': 'Kneeling by the bag'}])
        self.assertEqual((api.calls[-1][2]['unsure'], api.calls[-1][2]['tier']), (['q4'], 'claude'))
        self.assertIn('the rest goes to the person (decision d9)', out)
        with self.assertRaises(fw.Failure):
            run(api, 'review-decisions', 'answer', 'esc1', 'q2')

    def test_voices_run_report_and_answer(self):
        report = {'open_voices': [{'label': 'SPEAKER_04', 'lines': 69, 'first_time': 25.0}],
                  'last_run': {'status': 'done', 'stage': 'Complete · 140 lines given a character, 1 voices for Claude, 0 for you'},
                  'characters': [{'id': 'mai', 'title': 'Mai Sakurajima'}, {'id': 'sakuta', 'title': 'Sakuta Azusagawa'}],
                  'applied': [{'id': 'd1', 'summary': 'Gave voice SPEAKER_05 (12 lines, 3.0s) to Tomoe Koga: picture + voice', 'undone_at': None,
                               'overridden_at': None, 'cut_id': 'c2'}],
                  'questions': [{'id': 'esc1', 'reviewer': 'claude', 'question': 'Who speaks SPEAKER_04 (69 lines in cuts 3, 9)? the signals disagree.',
                                 'evidence': {'phrases': [{'start': 25.0, 'end': 26.5, 'text': 'ええちゃん', 'cut': 3, 'cut_id': 'c3',
                                                           'signals': {'voice': 'Sakuta Azusagawa', 'picture': 'Mai Sakurajima'},
                                                           'voice_top': [['Sakuta Azusagawa', 0.41]], 'addressed': []}]}}]}
        api = AutopilotApi()
        api.get = lambda path, params=None: report if path.endswith('/voice-decisions') else AutopilotApi.get(api, path, params)
        code, out = run(api, 'voices', 'report', EPISODE['id'])
        self.assertEqual(code, fw.BLOCKED)
        self.assertIn('69 line(s) without a character · SPEAKER_04 69', out)
        self.assertIn('to Tomoe Koga: picture + voice  id d1', out)
        self.assertIn('P1 cut 3 0:25.00-0:26.50 “ええちゃん” · voice Sakuta Azusagawa, picture Mai Sakurajima · voice Sakuta Azusagawa 0.41', out)
        self.assertIn('frames: http://framewise/api/escalations/esc1/voice-sheet', out)
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'job_id': 'j1'})[1]
        run(api, 'voices', 'run', EPISODE['id'], '--no-attribution')
        self.assertEqual(api.calls[-1][2], {'run_id': None, 'attribution': False})
        question = {'kind': 'voice', 'episode_id': EPISODE['id'], 'evidence': {'candidates': [{'id': 'mai', 'name': 'Mai Sakurajima'}]}}
        api.get = lambda path, params=None: question if path.startswith('/api/escalations/') else report
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'status': 'resolved', 'lines': 69, 'decision_id': 'd2'})[1]
        _, out = run(api, 'voices', 'answer', 'esc1', 'sakuta', '--tier', 'claude', '--note', 'his mouth moves')
        self.assertEqual(api.calls[-1], ('/api/escalations/esc1/voice', None, {'tier': 'claude', 'run_id': None, 'note': 'his mouth moves',
                                                                             'choice': 'character', 'character_context_id': 'sakuta'}))
        self.assertIn('69 line(s)', out)
        run(api, 'voices', 'answer', 'esc1', 'unsure', '--tier', 'claude')
        self.assertEqual(api.calls[-1][2]['choice'], 'unsure')
        with self.assertRaises(fw.Failure):
            run(api, 'voices', 'answer', 'esc1', 'Rio')

    def test_close_sends_the_resolution(self):
        api = AutopilotApi()
        run(api, 'close', 'esc1', '--resolution', '{"character": "mai"}')
        self.assertEqual(api.calls[-1], ('/api/escalations/esc1/close', {}, {'status': 'resolved', 'resolution': {'character': 'mai'}}))


def targets(**changes):
    """A /targets response: every stage, nothing to do, unless ``changes`` says so (stage__step=[cut numbers])."""
    def step(cuts=(), held=(), mode=None):
        return {'targets': [{'cut': n, 'cut_id': f'c{n}', 'why': 'missing'} for n in cuts],
                'held': [{'cut': n, 'cut_id': f'c{n}', 'reasons': ['box 1 has no character']} for n in held], **({'mode': mode} if mode else {})}
    names = ('media', 'hardcuts', 'missed_cuts', 'transcribe', 'subtitles', 'line_splits', 'voice_resolver', 'boxes', 'target_frames', 'character_decisions')
    data = {'episode': {'id': EPISODE['id'], 'subtitle_language': 'Japanese'}, 'range': None, 'inbox': {}, 'failures': 0,
            'prepare': {'done': None, 'steps': [{'step': name, 'done': True, 'targets': []} for name in names]},
            'prepare_cuts': {'boxes': step(), 'target_frames': step()},
            'lines': {'targets': [], 'mode': 'apply'},
            'evaluate': {name: step() for name in ('visual', 'beats', 'consistency')},
            'results': {'answers': step(mode='off'), 'summary': step(mode='off')}}
    for key, value in changes.items():
        stage, name = key.split('__')
        if stage == 'prepare':
            entry = next(item for item in data['prepare']['steps'] if item['step'] == name)
            entry.update(value)
        else:
            data[stage][name] = value
    return data


class StageApi(AutopilotApi):
    def __init__(self, data):
        super().__init__({'items': []})
        self.data = data

    def get(self, path, params=None):
        if path.endswith('/targets'):
            self.calls.append((path, params or {}, None))
            return self.data
        if path.endswith('/triage'):
            return {'items': []}
        if path.endswith('/agent-decisions'):
            return []
        if path.endswith('/sequences'):
            return []
        return super().get(path, params)


class StagePlanTest(unittest.TestCase):
    """The stages (D31): each step once, on its target cuts only (pure, no API)."""

    def test_prepare_order_and_resolvers_switched_off(self):
        data = targets(prepare__media={'done': False}, prepare__missed_cuts={'done': False}, prepare__voice_resolver={'done': False, 'mode': 'off'},
                       prepare__target_frames={'done': False, 'targets': [{'cut': 2}, {'cut': 3}]}, prepare__character_decisions={'done': False})
        steps, done = [], set()
        while action := fw.stage_next('prepare', data, done):
            steps.append(action['step'])
            done.add(action['step'])
        self.assertEqual(steps, ['setup', 'missed_cuts', 'target-frames', 'cast'])

    def test_evaluate_runs_each_step_once_on_its_targets(self):
        data = targets(evaluate__visual={'targets': [{'cut': 1}, {'cut': 2}], 'held': [{'cut': 4}]}, evaluate__beats={'targets': [{'cut': 5}]})
        first = fw.stage_next('evaluate', data, set())
        self.assertEqual((first['step'], first['cuts']), ('visual', [1, 2]))
        self.assertEqual(fw.stage_next('evaluate', data, {'visual'})['step'], 'beats')
        self.assertIsNone(fw.stage_next('evaluate', data, {'visual', 'beats'}))
        self.assertEqual(fw.stage_next('evaluate', data, set(), only='beats')['cuts'], [5])

    def test_prepare_cuts_runs_the_cast_resolver_only_after_it_changed_something(self):
        self.assertIsNone(fw.stage_next('prepare_cuts', targets(), set()))
        data = targets(prepare_cuts__target_frames={'targets': [{'cut': 3}]})
        self.assertEqual(fw.stage_next('prepare_cuts', data, set())['step'], 'target-frames')
        self.assertEqual(fw.stage_next('prepare_cuts', data, {'target-frames'})['step'], 'cast')

    def test_results_follow_the_profile(self):
        data = targets(results__answers={'targets': [{'cut': 2}], 'mode': 'off'}, results__summary={'targets': [{'cut': 1}], 'mode': 'off'})
        self.assertIsNone(fw.stage_next('results', data, set()))
        data['results']['summary']['mode'] = 'apply'
        self.assertEqual(fw.stage_next('results', data, set())['cuts'], [1])
        data['results']['answers']['mode'] = 'apply'
        self.assertEqual(fw.stage_next('results', data, set())['step'], 'review_decisions')

    def test_visual_is_queued_in_chunks_that_queue_beats_only_when_ready(self):
        api = AutopilotApi()
        api.post = lambda path, params=None, body=None: (api.calls.append((path, params, body)), {'job_ids': ['j']})[1]
        ids = fw.stage_execute(api, EPISODE, {'step': 'visual', 'cuts': list(range(1, 41)) + [45]}, 'run1')
        self.assertEqual(len(ids), 4)
        spans = [(params['from_cut'], params['to_cut']) for _, params, _ in api.calls]
        self.assertEqual(spans, [(1, 16), (17, 32), (33, 40), (45, 45)])
        self.assertTrue(all(params['beats'] == 'when_ready' and params['only_missing'] for _, params, _ in api.calls))

    def test_a_held_cut_links_to_where_its_reason_is_fixed(self):
        self.assertEqual(fw.held_place(['inbox: a line across the hard cut']), 'analysis')
        self.assertEqual(fw.held_place(['inbox: box names']), 'characters')
        self.assertEqual(fw.held_place(['target frame F2 flagged (low_match)']), 'target-frames')
        self.assertEqual(fw.held_place(['2 unnamed person(s) on the target frames']), 'target-frames')
        self.assertEqual(fw.held_place(['box 1 has no character']), 'characters')

    def test_runs_of_consecutive_cuts(self):
        self.assertEqual(fw.runs_of([5, 1, 2, 3, 7, 8]), [(1, 3), (5, 5), (7, 8)])


class StageCommandTest(unittest.TestCase):
    def test_dry_run_prints_targets_and_held_cuts_and_queues_nothing(self):
        api = StageApi(targets(evaluate__visual={'targets': [{'cut': 1, 'why': 'stale'}], 'held': [{'cut': 4, 'reasons': ['inbox: a line across the hard cut']}]}))
        code, out = run(api, 'evaluate', EPISODE['id'], '--dry-run')
        self.assertEqual(code, fw.OK)
        self.assertIn('next: visual · cuts 1', out)
        self.assertIn('visual: 1 target cut(s): 1 (1 stale); 1 held for you', out)
        self.assertIn('cut 4 (before visual): inbox: a line across the hard cut', out)
        self.assertIn('/cut/c4/analysis\n', out)
        self.assertFalse([call for call in api.calls if call[2] is not None])

    def test_prepare_never_runs_twice(self):
        data = targets()
        data['prepare']['done'] = {'id': 'run-prep-1', 'finished_at': '2026-11-24 10:00:00'}
        data['inbox'] = {'voice': {'human': 1, 'claude': 0}}
        api = StageApi(data)
        code, out = run(api, 'prepare', EPISODE['id'])
        self.assertEqual(code, fw.BLOCKED)
        self.assertIn('never runs again', out)
        self.assertIn('inbox: voice 1 for you', out)
        self.assertFalse([call for call in api.calls if call[2] is not None])

    def test_evaluate_queues_only_target_cuts_then_finishes_its_run(self):
        data = targets(evaluate__beats={'targets': [{'cut': 2}, {'cut': 4}]})
        api = StageApi(data)
        posted = []

        def post(path, params=None, body=None):
            posted.append(path)
            if path.endswith('/performance-beats'):
                data['evaluate']['beats'] = {'targets': [], 'held': []}
                return {'job_ids': ['j']}
            return AutopilotApi.post(api, path, params, body)
        api.post = post
        code, out = run(api, 'evaluate', EPISODE['id'])
        self.assertEqual(code, fw.OK, out)
        self.assertEqual([path for path in posted if 'performance-beats' in path], ['/api/cuts/c2/performance-beats', '/api/cuts/c4/performance-beats'])
        self.assertTrue(any(path.endswith('/finish') for path in posted))

    def test_autopilot_points_to_the_stages(self):
        args = fw.parser().parse_args(['autopilot', 'Episode 1', '--go'])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(args.func(None, args), fw.ERROR)
        self.assertIn('fw evaluate EP', err.getvalue())


if __name__ == '__main__':
    unittest.main()
