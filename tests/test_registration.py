"""Tests for the multiscale phase-correlation registration."""

import unittest

import cv2
import numpy as np

from froth.motion.registration import multiscale_phase_correlation, preprocess


class RegistrationTests(unittest.TestCase):
    def test_recovers_known_translation(self):
        rng = np.random.default_rng(0)
        base = cv2.GaussianBlur((rng.random((256, 256)) * 255).astype(np.uint8), (0, 0), 3)
        moved = cv2.warpAffine(
            base, np.float32([[1, 0, 7], [0, 1, -4]]), (256, 256), borderMode=cv2.BORDER_REFLECT
        )
        shift, _ = multiscale_phase_correlation(preprocess(base), preprocess(moved))
        self.assertLess(abs(shift[0] - 7), 0.5)
        self.assertLess(abs(shift[1] + 4), 0.5)


if __name__ == "__main__":
    unittest.main()
