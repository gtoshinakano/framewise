"""The subtitle scan finds the strip and when its text changes, without any model."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the API image has no numpy; the scan only runs in the inference image
    np = None

from app import subtitle_scan as scan

FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')


class ProbeTimesTest(unittest.TestCase):
    def test_probes_follow_speech_when_there_is_a_transcript(self):
        speech = [(10 + index * 60, 12 + index * 60) for index in range(30)]
        times = scan.choose_probe_times(1900, speech, 8)
        self.assertEqual(len(times), 8)
        self.assertTrue(all(any(start <= time <= end for start, end in speech) for time in times), times)
        self.assertEqual(times, sorted(times))
        self.assertGreater(times[-1] - times[0], 1000)  # spread over the episode, not clustered

    def test_probes_fall_back_to_an_even_grid_without_speech_and_avoid_the_ends(self):
        times = scan.choose_probe_times(100, [], 4)
        self.assertEqual(len(times), 4)
        self.assertGreaterEqual(times[0], 3)
        self.assertLessEqual(times[-1], 97)
        self.assertEqual(scan.choose_probe_times(0, [(1, 2)]), [])

    def test_a_short_transcript_is_topped_up_with_the_grid(self):
        times = scan.choose_probe_times(600, [(100, 104), (300, 304)], 6)
        self.assertEqual(len(times), 6)
        self.assertIn(102., times)
        self.assertIn(302., times)


class BandTest(unittest.TestCase):
    def probe(self, *boxes, size=(1280, 720)):
        return {'width': size[0], 'height': size[1], 'boxes': [list(box) for box in boxes]}

    def test_bottom_subtitles_give_a_strip_that_grows_upward_for_wrapped_lines(self):
        probes = [self.probe((300, 600, 980, 650)), self.probe((250, 610, 1030, 665)), self.probe((400, 598, 880, 648))]
        band = scan.band_from_probes(probes)
        self.assertEqual(band['source'], 'detector')
        self.assertEqual(band['support'], 3)
        self.assertLess(band['top'], 598 / 720 - .05)       # room for a second line above
        self.assertGreaterEqual(band['bottom'], 665 / 720)  # and the strip reaches the lowest line
        self.assertLessEqual(band['bottom'], 1.0)

    def test_top_subtitles_grow_downward(self):
        band = scan.band_from_probes([self.probe((300, 40, 980, 90)), self.probe((320, 44, 960, 92))])
        self.assertLessEqual(band['top'], 40 / 720)
        self.assertGreater(band['bottom'], 92 / 720 + .05)

    def test_the_position_seen_in_most_frames_wins_and_non_text_boxes_are_ignored(self):
        bottom = [self.probe((300, 600, 980, 650)) for _ in range(4)]
        occasional_sign = self.probe((100, 100, 400, 140))                       # one frame only
        tall_box = self.probe((0, 0, 1280, 500))                                 # not a line of text
        square = self.probe((600, 300, 640, 340))                                # not wide enough to be a line
        band = scan.band_from_probes([*bottom, occasional_sign, tall_box, square])
        self.assertGreater(band['top'], .7)
        self.assertEqual(band['support'], 4)

    def test_without_usable_detections_the_default_bottom_strip_is_used(self):
        band = scan.band_from_probes([self.probe(), self.probe((0, 0, 1280, 700))])
        self.assertEqual((band['top'], band['bottom'], band['source']), (.70, 1.0, 'default'))
        self.assertEqual(scan.band_from_probes([])['source'], 'default')
        invalid = scan.band_from_probes([{'width': 0, 'height': 0, 'boxes': [[1, 2, 3, 4]]}, self.probe((1, 2, float('nan'), 4))])
        self.assertEqual(invalid['source'], 'default')

    def test_geometry_is_even_and_inside_the_frame(self):
        geometry = scan.strip_geometry(1920, 1080, {'top': .77, 'bottom': .98})
        self.assertEqual(geometry['width'], 640)
        self.assertEqual(geometry['frame_height'], 360)
        self.assertEqual(geometry['height'] % 2, 0)
        self.assertLessEqual(geometry['y'] + geometry['height'], geometry['frame_height'])
        self.assertEqual(scan.canvas_box({'top': .5, 'bottom': .75}, 1280, 720), [0, 360, 1280, 540])


@unittest.skipIf(np is None, 'numpy is required')
class EventTest(unittest.TestCase):
    H, W = 48, 200

    def background(self, count, seed=1):
        rng = np.random.default_rng(seed)
        return (rng.integers(20, 70, size=(count, self.H, self.W))).astype(np.uint8)

    def draw(self, frames, indexes, text):
        """Bright thin strokes in a pattern that depends on ``text``, like glyphs of a subtitle."""
        rng = np.random.default_rng(sum(map(ord, text)))
        strokes = []
        for _ in range(26):
            row, col = int(rng.integers(14, 34)), int(rng.integers(20, 170))
            length = int(rng.integers(6, 16))
            strokes.append((row, col, length, bool(rng.integers(0, 2))))
        for index in indexes:
            for row, col, length, vertical in strokes:
                if vertical:
                    frames[index, row:row + length, col:col + 2] = 245
                else:
                    frames[index, row:row + 2, col:col + length] = 245
        return frames

    def run_events(self, frames, static=None):
        masks = scan.stroke_mask(frames)
        return scan.find_events(list(masks), static, min_pixels=20)

    def test_two_subtitles_back_to_back_are_two_events_with_exact_frames(self):
        frames = self.background(40)
        self.draw(frames, range(5, 15), 'first line')
        self.draw(frames, range(15, 27), 'a different line')
        events, counts = self.run_events(frames)
        self.assertEqual([(event['start'], event['end']) for event in events], [(5, 14), (15, 26)])
        self.assertTrue(all(counts[event['rep']] >= counts[event['start']] for event in events))

    def test_empty_background_and_moving_highlights_are_not_subtitles(self):
        frames = self.background(40)
        for index in range(40):                       # a bright thin streak that moves 9 px per frame
            column = 10 + 9 * (index % 18)
            frames[index, 5:44, column:column + 2] = 250
        self.assertEqual(self.run_events(frames)[0], [])

    def test_a_short_dropout_inside_one_line_does_not_split_it(self):
        frames = self.background(40)
        self.draw(frames, [*range(5, 14), *range(16, 26)], 'one line')   # two frames missing in the middle
        events, _ = self.run_events(frames)
        self.assertEqual([(event['start'], event['end']) for event in events], [(5, 25)])

    def test_a_longer_gap_ends_the_line(self):
        frames = self.background(60)
        self.draw(frames, [*range(5, 14), *range(20, 30)], 'same text twice')
        events, _ = self.run_events(frames)
        self.assertEqual([(event['start'], event['end']) for event in events], [(5, 13), (20, 29)])

    def test_a_static_overlay_is_removed_and_a_single_frame_flash_is_ignored(self):
        frames = self.background(40)
        self.draw(frames, range(40), 'always on logo')
        self.draw(frames, range(10, 20), 'subtitle over the logo')
        self.draw(frames, [30], 'one frame flash')
        masks = list(scan.stroke_mask(frames))
        static = np.mean(masks, axis=0) > scan.STATIC_FREQUENCY
        self.assertGreater(int(static.sum()), 20)
        events, _ = scan.find_events(masks, static, min_pixels=20)
        self.assertEqual([(event['start'], event['end']) for event in events], [(10, 19)])

    def test_representative_frame_is_the_most_complete_one_not_a_fade(self):
        counts = np.array([0, 5, 20, 80, 90, 92, 88, 60, 15, 4, 0])
        self.assertEqual(scan.pick_representative(counts, 1, 9), 5)
        self.assertEqual(scan.pick_representative(counts, 3, 4), 4)        # too short to trim: the larger of the two

    def test_long_lines_are_read_again_every_few_seconds(self):
        counts = np.full(200, 50, dtype=np.int32)
        event = {'start': 8, 'end': 8 + 8 * 11 - 1, 'rep': 40, 'peak': 50}   # 11 s at 8 fps
        parts = scan.split_long([event], counts, 8.)
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0]['start'], event['start'])
        self.assertEqual(parts[-1]['end'], event['end'])
        self.assertTrue(all(left['end'] + 1 == right['start'] for left, right in zip(parts, parts[1:])))
        self.assertTrue(all((part['end'] - part['start'] + 1) / 8. <= scan.MAX_SEGMENT_SECONDS for part in parts))
        short = {'start': 0, 'end': 23, 'rep': 10, 'peak': 5}
        self.assertEqual(scan.split_long([short], counts, 8.), [short])

    def test_event_times_are_seconds_with_an_exclusive_end(self):
        event = {'start': 8, 'end': 15, 'rep': 11, 'peak': 99}
        self.assertEqual(scan.to_seconds(event, 8.), {'start_time': 1.0, 'end_time': 2.0, 'rep_time': 1.375, 'peak_pixels': 99})

    def test_numpy_top_hat_agrees_with_opencv(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('OpenCV is not installed')
        frames = self.draw(self.background(3), [0, 1, 2], 'compare')
        self.assertTrue(np.array_equal(scan._tophat(frames), scan._tophat_numpy(frames, scan.TOPHAT_SIZE)))


@unittest.skipIf(np is None, 'numpy is required')
class OutlineTest(unittest.TestCase):
    """Real still-anime backgrounds contain bright thin structure (rails, straps) that is not subtitle text.
    Subtitles are drawn with a dark outline, which that clutter lacks."""
    H, W = 60, 220

    def scene(self, count=3):
        frames = np.full((count, self.H, self.W), 90, dtype=np.uint8)
        for column in (6, 18, 30, 200, 212):        # pale rails on a mid-grey scene, no dark pixels near them
            frames[:, 6:54, column:column + 2] = 235
        return frames

    def add_outlined_text(self, frames):
        for row, col, length in ((26, 40, 30), (26, 80, 14), (26, 110, 36), (26, 160, 20)):
            frames[:, row - 3:row + 5, col - 3:col + length + 3] = 5           # dark outline
            frames[:, row:row + 2, col:col + length] = 250                      # white stroke
        return frames

    def test_the_outline_test_removes_unoutlined_clutter_and_keeps_subtitles(self):
        frames = self.add_outlined_text(self.scene())
        loose, strict = scan.stroke_mask(frames), scan.stroke_mask(frames, outlined=True)
        self.assertGreater(int(loose[0].sum()), 3 * int(strict[0].sum()))
        self.assertTrue(strict[0][26:28, 45:60].all())                          # the text strokes are still found
        self.assertFalse(strict[0][8:20, 10:12].any())                          # the rails are gone

    def test_a_scene_without_subtitles_has_no_events_when_the_outline_test_is_used(self):
        frames = self.scene(12)
        self.assertEqual(scan.find_events(list(scan.stroke_mask(frames, outlined=True)), None, min_pixels=40)[0], [])
        self.assertTrue(scan.find_events(list(scan.stroke_mask(frames)), None, min_pixels=40)[0])   # without it the rails look like text

    def test_changes_of_text_over_a_cluttered_still_background_are_separate_events(self):
        frames = self.scene(24)
        for index in range(4, 12):
            frames[index] = self.add_outlined_text(frames[index:index + 1].copy())[0]
        for index in range(12, 20):
            frames[index:index + 1] = self.add_outlined_text(frames[index:index + 1].copy())
            frames[index, 24:34, 40:70] = 90                                     # a different line of text
            frames[index, 24:34, 150:200] = 5
            frames[index, 28:30, 155:195] = 250
        events, _ = scan.find_events(list(scan.stroke_mask(frames, outlined=True)), None, min_pixels=40)
        self.assertEqual([(event['start'], event['end']) for event in events], [(4, 11), (12, 19)])

    def test_outline_share_tells_outlined_subtitles_from_plain_ones(self):
        outlined = self.add_outlined_text(self.scene(1))[0]
        plain = self.scene(1)[0]
        for row, col, length in ((26, 40, 30), (26, 80, 14), (26, 110, 36), (26, 160, 20)):
            plain[row:row + 2, col:col + length] = 250                          # white text on mid-grey, no outline at all
        boxes = [[36, 20, 190, 36]]                                            # tight around the text, as a detector's box is
        self.assertGreater(scan.outline_share(outlined, boxes), .9)
        self.assertLess(scan.outline_share(plain, boxes), .3)
        self.assertIsNone(scan.outline_share(self.scene(1)[0] * 0, boxes))      # not enough strokes to judge

    def test_the_style_comes_from_the_median_probe_and_unknown_means_off(self):
        self.assertEqual(scan.outlined_style([.95, .9, None, .2, .92]), (True, .92))   # the middle of the usable shares
        self.assertEqual(scan.outlined_style([.1, .2, .3]), (False, .2))
        self.assertEqual(scan.outlined_style([None, None]), (False, None))
        self.assertEqual(scan.outlined_style([]), (False, None))

    def test_a_pil_frame_is_measured_at_the_scan_width(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        frame = np.full((360, 640), 90, dtype=np.uint8)
        for col in (200, 250, 300, 350):
            frame[300:324, col - 4:col + 40] = 5
            frame[310:313, col:col + 32] = 250
        image = Image.fromarray(frame).resize((1280, 720))
        share = scan.outline_share_of(image, [[360, 590, 800, 650]])
        self.assertIsNotNone(share)
        self.assertGreater(share, .5)


def ffmpeg_can_draw_text():
    if not shutil.which('ffmpeg') or not FONT.is_file():
        return False
    result = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True)
    return ' drawtext ' in result.stdout


@unittest.skipUnless(np is not None and ffmpeg_can_draw_text(), 'needs numpy, ffmpeg with drawtext, and DejaVu Sans Bold')
class VideoScanTest(unittest.TestCase):
    """End to end on a real encoded video: busy moving background, a watermark inside the strip, timed subtitles."""

    LINES = [(1.0, 3.0, 'Olá, tudo bem?'), (3.0, 5.0, 'Estou ótimo, obrigado!'), (6.0, 6.75, 'Não.'), (8.0, 15.0, 'Esta frase fica bastante tempo')]

    def make_video(self, folder, **style):
        color, size = style.get('color', 'white'), style.get('size', 40)
        texts = ','.join(f"drawtext=fontfile={FONT}:text='{text}':fontcolor={color}:borderw=3:bordercolor=black:fontsize={size}:x=(w-text_w)/2:y=h-90:enable='between(t,{start},{end})'"
                         for start, end, text in self.LINES)
        watermark = f"drawtext=fontfile={FONT}:text='TV7':fontcolor=white:borderw=2:bordercolor=black:fontsize=30:x=w-120:y=h-60"
        path = Path(folder) / 'episode.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=960x540:rate=24:duration=17',
                        '-vf', f'{watermark},{texts}', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30', '-pix_fmt', 'yuv420p', str(path)], check=True)
        return path

    def scan_video(self, band, **style):
        with tempfile.TemporaryDirectory() as folder:
            return scan.scan_video(self.make_video(folder, **style), 960, 540, 17., band)

    def test_events_match_when_each_subtitle_was_shown_and_the_watermark_is_ignored(self):
        result = self.scan_video({'top': .74, 'bottom': .97})
        events = result['events']
        self.assertEqual(result['fps'], 8.)
        # three short lines are one event each; the 7 s line is read again halfway as a safeguard
        self.assertEqual(len(events), 5, events)
        for event, (want_start, want_end, _) in zip(events[:3], self.LINES[:3]):
            self.assertAlmostEqual(event['start_time'], want_start, delta=.26)
            self.assertAlmostEqual(event['end_time'], want_end, delta=.26)
        first, second = events[3:]
        self.assertAlmostEqual(first['start_time'], 8., delta=.26)
        self.assertAlmostEqual(second['end_time'], 15., delta=.26)
        self.assertEqual(first['end_time'], second['start_time'])
        self.assertTrue(all(event['start_time'] <= event['rep_time'] < event['end_time'] for event in events))
        self.assertGreater(result['diagnostics']['static_pixels'], 0)

    def test_small_yellow_subtitles_on_a_noisy_encode_are_still_found(self):
        events = self.scan_video({'top': .74, 'bottom': .97}, color='yellow', size=28)['events']
        self.assertEqual(len(events), 5, events)

    def test_a_strip_without_subtitles_finds_nothing(self):
        self.assertEqual(self.scan_video({'top': .35, 'bottom': .6})['events'], [])

    def test_the_outline_test_keeps_every_outlined_subtitle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_video(folder)
            plain = scan.scan_video(path, 960, 540, 17., {'top': .74, 'bottom': .97})
            gated = scan.scan_video(path, 960, 540, 17., {'top': .74, 'bottom': .97}, outlined=True)
        self.assertEqual([(e['start_time'], e['end_time']) for e in gated['events']], [(e['start_time'], e['end_time']) for e in plain['events']])
        self.assertTrue(gated['diagnostics']['outline_test'])
        self.assertFalse(plain['diagnostics']['outline_test'])

    def test_a_missing_file_is_reported_not_treated_as_an_empty_video(self):
        with self.assertRaises(RuntimeError) as context:
            scan.scan_video('/nonexistent/video.mp4', 960, 540, 10., {'top': .7, 'bottom': .95})
        self.assertIn('Could not decode', str(context.exception))


if __name__ == '__main__':
    unittest.main()
