import unittest
from unittest.mock import patch
from types import SimpleNamespace
from app import keyframes_api


class CutKeyframesApiTest(unittest.IsolatedAsyncioTestCase):
    def test_queue_records_only_the_cut_id_and_settings(self):
        cut = {'id': 'c', 'episode_id': 'e', 'start_time': 1.0, 'end_time': 2.0}
        selector = SimpleNamespace(config=lambda value: {'budget': 5})
        with patch.object(keyframes_api.db, 'one', return_value=cut), patch('app.keyframes', selector, create=True), patch.object(keyframes_api.worker, 'enqueue', return_value='job') as enqueue:
            result = keyframes_api.enqueue_cut_selection('c', keyframes_api.Request())
        self.assertEqual(result['job_id'], 'job')
        self.assertEqual(enqueue.call_args.args[0:2], ('e', 'keyframes'))
        self.assertEqual(enqueue.call_args.args[2], {'cut_ids': ['c'], 'keyframe_settings': {'budget': 5}})

    async def test_reject_unselected_frame(self):
        from fastapi import HTTPException
        state = {'result': {'selected_frames': [{'source_frame_index': 7}]}}
        with patch.object(keyframes_api, 'cut_selection', return_value=({}, None, state)):
            with self.assertRaises(HTTPException) as error:
                await keyframes_api.get_target_frame('c', 8, None)
        self.assertEqual(error.exception.status_code, 404)

if __name__ == '__main__': unittest.main()
