"""The camera between two keyframes from pixels: a pan, a zoom, and a moving person that must not read as a camera move."""
import unittest

try:
    import cv2
    import numpy as np
except ImportError:  # the camera needs opencv; the API image has it, a bare host may not
    cv2 = None

from app import keyframe_camera as camera


def texture(width=900, height=500, seed=3):
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.uint8)
    for _ in range(500):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(4, 30)), y + int(rng.integers(4, 30))), int(rng.integers(40, 255)), -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


def shifted(image, dx=0, scale=1.0):
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    matrix[0, 2] += dx
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


@unittest.skipIf(cv2 is None, 'opencv is not installed')
class CameraTest(unittest.TestCase):
    def setUp(self):
        self.big = texture()
        self.a = self.big[:, 100:740]

    def test_content_moving_left_is_a_pan_right(self):
        b = self.big[:, 120:760]  # the picture slides 20 px left
        estimate = camera.camera_estimate(self.a, b, camera.band_mask(self.a.shape, None))
        self.assertEqual(estimate['move'], 'pan_right')

    def test_zoom_in(self):
        b = shifted(self.a, scale=1.08)
        self.assertEqual(camera.camera_estimate(self.a, b, camera.band_mask(self.a.shape, None))['move'], 'zoom_in')

    def test_a_moving_person_is_masked_out(self):
        a, b = self.a.copy(), self.a.copy()
        person = texture(200, 400, seed=9)
        a[50:450, 100:300] = person
        b[50:450, 380:580] = person  # walks across a still backdrop
        boxes = [[[100, 50, 300, 450]], [[380, 50, 580, 450]]]
        sizes = [[640, 500], [640, 500]]
        masked = camera.pair_cameras([a, b], None, boxes, sizes)[0]
        self.assertEqual(masked['move'], 'static')

    def test_subtitle_band_is_left_out(self):
        mask = camera.band_mask((100, 50), {'top': 0.8, 'bottom': 1.0})
        self.assertFalse(mask[85].any())
        self.assertTrue(mask[10].all())

    def test_a_blank_picture_is_unclear(self):
        blank = np.zeros((300, 640), dtype=np.uint8)
        self.assertEqual(camera.camera_estimate(blank, blank, camera.band_mask(blank.shape, None))['move'], 'unclear')

    def test_gray_scales_to_the_working_width(self):
        image = np.zeros((360, 1280, 3), dtype=np.uint8)
        self.assertEqual(camera.gray(image).shape, (180, 640))


if __name__ == '__main__':
    unittest.main()
