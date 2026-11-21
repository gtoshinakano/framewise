"""A picture asked for a cut never comes from the next cut (the review's F5 of cut 176 showed cut 177's first shot)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import consistency_evidence, frame_cache, media
from app.assistant import images

CUT = {'id': 'c1', 'start_time': 10.0, 'end_time': 14.0}
KEYFRAMES = {'contract': 'keyframes-v1', 'run_id': 'r1', 'frames': [{'label': 'F1', 'ordinal': 240, 't': 0.0}, {'label': 'F2', 'ordinal': 339, 't': 3.96}]}


class PlanByFrameTest(unittest.TestCase):
    def test_a_keyframe_picture_carries_its_frame_number_and_exact_time(self):
        pictures = consistency_evidence.plan(CUT, KEYFRAMES, [], [], exact={240: 10.000123, 339: 13.9601234})
        self.assertEqual([(p['label'], p['ordinal'], p['time']) for p in pictures], [('F1', 240, 10.000123), ('F2', 339, 13.9601234)])
        # Without exact times the plan is unchanged: pictures are cut by clip time, as before.
        self.assertNotIn('ordinal', consistency_evidence.plan(CUT, KEYFRAMES, [], [])[0])

    def test_the_picture_is_cut_by_frame_number(self):
        picture = consistency_evidence.plan(CUT, KEYFRAMES, [], [], exact={240: 10.000123, 339: 13.9601234})[1]
        located = {'spec': {'source': Path('x.mp4'), 'start': 10.0, 'end': 14.0, 'width': 1024, 'last': False, 'time': 13.96}, 'clip_time': 3.96, 'label': 'x', 'crop': None, 'frame': 'F2'}
        seen = []
        with tempfile.TemporaryDirectory() as folder, patch.object(images, 'locate', return_value=located), \
                patch.object(frame_cache, 'render', side_effect=lambda spec: seen.append(dict(spec)) or b'jpeg'):
            consistency_evidence._frame(Path(folder), picture, picture['at'][0], 0, 768)
        self.assertEqual((seen[0]['ordinal'], seen[0]['time']), (339, 13.9601234))

    def test_a_frame_outside_the_cut_is_refused(self):
        picture = {**consistency_evidence.plan(CUT, KEYFRAMES, [], [], exact={240: 10.0, 339: 14.0004})[1]}
        located = {'spec': {'source': Path('x.mp4'), 'start': 10.0, 'end': 14.0, 'width': 1024, 'last': False, 'time': 13.96}, 'clip_time': 3.96, 'label': 'x', 'crop': None, 'frame': 'F2'}
        with tempfile.TemporaryDirectory() as folder, patch.object(images, 'locate', return_value=located), \
                patch.object(frame_cache, 'render', return_value=b'jpeg'):
            with self.assertRaises(images.FrameError):
                consistency_evidence._frame(Path(folder), picture, picture['at'][0], 0, 768)

    def test_the_shared_renderer_refuses_a_frame_past_the_cut(self):
        spec = {'source': Path('x.mp4'), 'time': 14.0004, 'start': 10.0, 'end': 14.0, 'width': 320, 'last': False, 'ordinal': 340}
        with self.assertRaises(RuntimeError):
            frame_cache.render(spec)


class SeekPastTheLastFrameTest(unittest.TestCase):
    """The live case: the cut ends 0.4 ms before the next cut's first frame, and a rounded-up time seeks onto that frame."""

    @staticmethod
    def fake_run(calls, first_frame_pts, in_cut_frames):
        class Result:
            def __init__(self, stdout='', stderr=''): self.stdout, self.stderr = stdout, stderr

        def run(args, timeout=None):
            calls.append(args)
            if args[0] == 'ffprobe' and 'format=start_time' in args:
                return Result('0.000000\n')
            if args[0] == 'ffprobe':
                return Result(''.join(f'{t:.6f}\n' for t in in_cut_frames + [first_frame_pts]))
            seek = float(args[args.index('-ss') + 1])
            Path(args[-1]).write_bytes(b'jpeg')
            # showinfo prints the frame's pts in a 1/1000000 time base, relative to the seek point.
            pts = round(((first_frame_pts if seek > in_cut_frames[-1] else min(t for t in in_cut_frames if t >= seek)) - seek) * 1000000)
            return Result(stderr=f'[Parsed_showinfo_0 @ 0x0] config in time_base: 1/1000000, frame_rate: 24000/1001\n[Parsed_showinfo_0 @ 0x0] n:   0 pts:{pts:>8} pts_time:0\n')
        return run

    def test_a_time_that_lands_on_the_next_cuts_first_frame_gives_the_last_frame_of_this_cut(self):
        calls = []
        start, end = 666.540458, 667.958542
        in_cut = [667.875559, 667.917267]  # 16011, 16012: the last frame of the cut is 16012
        next_cut_first = 667.958958  # 16013: after `end`, but within the 0.4 ms the original runs ahead of the proxy
        with tempfile.TemporaryDirectory() as folder, patch.object(media, 'run', self.fake_run(calls, next_cut_first, in_cut)):
            actual = media.extract_cut_frame(Path('x.mp4'), Path(folder) / 'frame.jpg', 667.917458, start, end, width=64)
        self.assertLess(actual, end)
        self.assertAlmostEqual(actual, 667.917267, places=3)

    def test_a_time_inside_the_cut_is_unchanged(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder, patch.object(media, 'run', self.fake_run(calls, 667.958958, [667.875559, 667.917267])):
            actual = media.extract_cut_frame(Path('x.mp4'), Path(folder) / 'frame.jpg', 667.8755, 666.540458, 667.958542, width=64)
        self.assertAlmostEqual(actual, 667.875559, places=5)
        self.assertEqual(sum(1 for call in calls if call[0] == 'ffmpeg'), 1)


if __name__ == '__main__':
    unittest.main()
