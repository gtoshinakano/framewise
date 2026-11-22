"""scripts/fw autopilot commands (agent runs, inbox, escalate, close) against a fake API."""
import contextlib
import io
import unittest

from tests.test_fw_cli import EPISODE, FakeApi, fw


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

    def test_close_sends_the_resolution(self):
        api = AutopilotApi()
        run(api, 'close', 'esc1', '--resolution', '{"character": "mai"}')
        self.assertEqual(api.calls[-1], ('/api/escalations/esc1/close', {}, {'status': 'resolved', 'resolution': {'character': 'mai'}}))


if __name__ == '__main__':
    unittest.main()
