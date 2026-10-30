"""Tests for the dark-impurity detector."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from froth.impurity import FilterParams, SegmentParams, filter_contours, load_config, threshold_mask
from froth.impurity.pipeline import ImpurityConfig, run

ROOT = Path(__file__).resolve().parents[1]


class ImpurityTests(unittest.TestCase):
    def test_dynamic_threshold_shape_and_type(self):
        gray = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
        mask = threshold_mask(
            gray, SegmentParams(k_dynamic=0.8, morph_open_kernel=0, morph_open_iterations=0)
        )
        self.assertEqual(mask.shape, gray.shape)
        self.assertEqual(mask.dtype, np.uint8)

    def test_counts_small_dark_spots_only(self):
        img = np.full((60, 60), 200, np.uint8)
        img[10:14, 10:14] = 0  # 4x4 spot: contour area 9, within [5, 11]
        img[30:40, 30:40] = 0  # large dark region: rejected by the area filter
        mask = threshold_mask(img, SegmentParams(morph_open_kernel=0, morph_open_iterations=0))
        self.assertEqual(len(filter_contours(mask, FilterParams())), 1)

    def test_config_and_batch_run(self):
        cfg = load_config(ROOT / "configs" / "impurity.yaml")
        self.assertIsInstance(cfg, ImpurityConfig)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            cv2.imwrite(str(path), np.full((32, 32, 3), 180, np.uint8))
            csv_path = run([path], Path(tmp) / "out", cfg)
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
