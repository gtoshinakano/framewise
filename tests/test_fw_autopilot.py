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
        inbox = {'counts': {'open_total': 1, 'applied_total': 7, 'open': {'voice': 1}, 'applied': {'line_splits': 7}},
                 'items': [{'id': 'esc1', 'kind': 'voice', 'question': 'Whose voice is SPEAKER_04?', 'status': 'open', 'options': ['Mai', 'Kaede'],
                            'proposal': {'character': 'Mai'}, 'cut_ids': ['c4', 'c2']}]}
        code, out = run(AutopilotApi(inbox), 'inbox', EPISODE['id'])
        self.assertEqual(code, fw.OK)
        self.assertIn('1 open question(s) for you, 7 decision(s) applied by agents', out)
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

    def test_close_sends_the_resolution(self):
        api = AutopilotApi()
        run(api, 'close', 'esc1', '--resolution', '{"character": "mai"}')
        self.assertEqual(api.calls[-1], ('/api/escalations/esc1/close', {}, {'status': 'resolved', 'resolution': {'character': 'mai'}}))


if __name__ == '__main__':
    unittest.main()
