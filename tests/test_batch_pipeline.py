"""Batch visual analysis loads each model once and runs stages in order across cuts."""
import contextlib
import unittest
from unittest.mock import MagicMock, patch

from app import anime_bulk, inference_worker, single_cut_pipeline as pipeline, worker


class FakeCutRun:
    """Minimal stand-in for CutRun that records which stages were executed."""

    def __init__(self, name, log, stages=pipeline.ALL_STAGES):
        self.name, self.log, self.stages = name, log, set(stages)
        self.cut_id, self.cut = name, {'cut_index': 0}
        self.completed = False
        self.failure = None
        self.item = {}
        self.references = []

    def needs(self, kind):
        return kind in self.stages

    @contextlib.contextmanager
    def stage(self, key, label):
        self.log.append((self.name, key))
        yield

    def cancel_check(self):
        pass

    def mark_complete(self):
        self.completed = True


class BatchPipelineTest(unittest.TestCase):
    def test_frame_model_server_starts_once_for_all_cuts(self):
        log = []
        runs = [FakeCutRun(name, log) for name in ('a', 'b', 'c')]
        server = MagicMock()
        server.return_value.__enter__.return_value = 'server'
        with patch.object(pipeline.frame_model, 'frame_server', server), \
             patch.object(pipeline, '_frame_stage') as frame_stage:
            pipeline._first_frame_stage(runs, '/scratch')
        self.assertEqual(server.call_count, 1)
        # One description per cut: the first frame. The last frame is no longer analyzed.
        self.assertEqual(sorted(call.args[0].name for call in frame_stage.call_args_list), ['a', 'b', 'c'])

    def test_frame_server_is_skipped_when_no_frame_stage_is_needed(self):
        runs = [FakeCutRun('a', [], stages=('video_analysis',))]
        with patch.object(pipeline.frame_model, 'frame_server') as server:
            pipeline._first_frame_stage(runs, '/scratch')
        server.assert_not_called()

    def test_a_job_queued_with_the_removed_last_frame_stage_runs_nothing_for_it(self):
        runs = [FakeCutRun('a', [], stages=('last_frame',))]
        with patch.object(pipeline.frame_model, 'frame_server') as server, patch.object(pipeline.video_model, 'VideoRuntime') as runtime:
            pipeline._first_frame_stage(runs, '/scratch')
            pipeline._video_stage(runs, '/scratch')
        server.assert_not_called()
        runtime.assert_not_called()
        self.assertEqual(pipeline.ALL_STAGES, ('initial_frame', 'video_analysis'))

    def test_video_model_is_shared_by_every_cut_and_closed_once(self):
        runs = [FakeCutRun(name, []) for name in ('a', 'b')]
        runtime = MagicMock()
        with patch.object(pipeline.video_model, 'VideoRuntime', return_value=runtime), \
             patch.object(pipeline, '_video_analysis') as analysis:
            pipeline._video_stage(runs, '/scratch')
        self.assertEqual([call.args[2] for call in analysis.call_args_list], [runtime, runtime])
        runtime.close.assert_called_once()

    def test_run_batch_runs_stages_in_model_order_for_all_cuts(self):
        order = []
        runs = [FakeCutRun('a', []), FakeCutRun('b', [])]
        job = {'id': 'job', 'payload_json': '{}'}
        with patch.object(pipeline, 'CutRun', side_effect=runs), \
             patch.object(pipeline, 'ensure_models_available'), patch.object(pipeline, 'release_gpu_models'), \
             patch.object(pipeline, '_first_frame_stage', side_effect=lambda *_: order.append('frames')), \
             patch.object(pipeline, '_video_stage', side_effect=lambda *_: order.append('video')):
            for run in runs:
                run.validate = lambda: None
            pipeline.run_batch(job, 'owner', {}, [{}, {}])
        self.assertEqual(order, ['frames', 'video'])  # the video stage builds on the first frame; subtitles are an episode-wide step of their own
        self.assertTrue(all(run.completed for run in runs))


class PerCutFailureTest(unittest.TestCase):
    def test_for_each_records_a_failure_and_keeps_going(self):
        runs = [FakeCutRun(name, []) for name in ('a', 'b', 'c')]
        seen = []

        def work(cut_run):
            seen.append(cut_run.name)
            if cut_run.name == 'a':
                raise ValueError('boom')

        pipeline._for_each(runs, work)
        self.assertEqual(seen, ['a', 'b', 'c'])
        self.assertIsInstance(runs[0].failure, ValueError)
        self.assertIsNone(runs[1].failure)
        pipeline._for_each(runs, work)  # a failed cut is not run again
        self.assertEqual(seen, ['a', 'b', 'c', 'b', 'c'])

    def test_lease_loss_and_cancellation_still_stop_the_whole_job(self):
        for error in (worker.LeaseLost('Job lease was lost'), inference_worker.SummaryCancelled()):
            run = FakeCutRun('a', [])
            with self.assertRaises(type(error)):
                pipeline._for_each([run], lambda _run, error=error: (_ for _ in ()).throw(error))


class ParallelForEachTest(unittest.TestCase):
    def test_cuts_run_at_the_same_time_and_a_failure_stays_on_its_cut(self):
        import threading
        runs = [FakeCutRun(name, []) for name in 'abcd']
        barrier = threading.Barrier(4, timeout=5)

        def work(run):
            barrier.wait()  # only passes when all four cuts are running together
            if run.name == 'b':
                raise ValueError('truncated')
        pipeline._for_each(runs, work, parallel=4)
        self.assertEqual([bool(run.failure) for run in runs], [False, True, False, False])

    def test_lease_loss_in_a_parallel_batch_still_ends_the_job(self):
        runs = [FakeCutRun(name, []) for name in 'abc']

        def work(run):
            if run.name == 'a':
                raise worker.LeaseLost('Job lease was lost')
        with self.assertRaises(worker.LeaseLost):
            pipeline._for_each(runs, work, parallel=3)


class CharacterBatchOrderTest(unittest.TestCase):
    def test_each_model_runs_over_every_cut_before_the_next_model(self):
        calls = []
        claimed = [('j1', 'o1'), ('j2', 'o2'), ('j3', 'o3')]

        def process(job_id, owner, runtime):
            calls.append(('reid', job_id))

        with patch.object(anime_bulk, '_claim_batch', return_value=claimed), \
             patch.object(anime_bulk, '_prepare_frames', side_effect=lambda *_: calls.append(('frames', 'all'))), \
             patch.object(anime_bulk, '_detect_batch', side_effect=lambda *_: calls.append(('aniref', 'all'))), \
             patch.object(anime_bulk, 'process', side_effect=process):
            anime_bulk.process_batch({'id': 'j1'}, 'o1', anime_bulk.BulkRuntime())
        self.assertEqual(calls, [('frames', 'all'), ('aniref', 'all'), ('reid', 'j1'), ('reid', 'j2'), ('reid', 'j3')])


if __name__ == '__main__':
    unittest.main()
