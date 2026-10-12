"""Regression checks for distinct regions and known synthetic displacements."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from froth.motion.estimators import block_matching, motion_fusion, optical_flow, phase_correlation
from froth.postprocessing.log_voronoi import partition_probability
from froth.postprocessing.roi_watershed import process_image_roi_watershed
from froth.postprocessing.thresholds import dynamic_threshold


class PostprocessingTests(unittest.TestCase):
    def test_sparse_or_constant_probabilities_keep_zero_background(self):
        probability = np.zeros((20, 20), np.float32); probability[5, 5] = 1
        threshold = dynamic_threshold(probability, 0.736)
        self.assertEqual(int((probability > threshold).sum()), 1)
        self.assertEqual(dynamic_threshold(np.ones((5, 5)), 0.736), 0.5)

    def test_adjacent_voronoi_regions_do_not_merge_during_relabeling(self):
        probability = np.zeros((32, 32), np.float32); probability[5:27, 5:27] = 1
        scales = {"r10": 2, "r50": 4, "r90": 8}
        with patch("froth.postprocessing.log_voronoi.detect_log_seeds", return_value=[(12, 10), (12, 22)]):
            binary, labels, seeds = partition_probability(probability, scales)
        self.assertEqual(seeds, 2)
        self.assertEqual(labels.max(), 2)
        np.testing.assert_equal(binary > 0, probability > 0)

    def test_watershed_runs_without_creating_figures(self):
        import matplotlib.pyplot as plt
        probability = np.zeros((32, 32), np.float32)
        probability[4:12, 4:12] = 1; probability[20:28, 20:28] = 1
        before = plt.get_fignums()
        binary, count, _, _ = process_image_roi_watershed(probability)
        self.assertEqual(count, 2)
        self.assertTrue(binary.any())
        self.assertEqual(before, plt.get_fignums())


class MotionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.first = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        self.second = cv2.warpAffine(self.first, np.float32([[1, 0, 3], [0, 1, -2]]),
                                     (128, 128), borderMode=cv2.BORDER_REFLECT)

    def test_phase_translation_has_correct_sign_and_scale(self):
        np.testing.assert_allclose(phase_correlation(self.first, self.second), [3, -2], atol=0.8)

    def test_optical_flow_translation(self):
        flow = optical_flow(self.first, self.second)
        np.testing.assert_allclose(np.median(flow[20:-20, 20:-20], axis=(0, 1)), [3, -2], atol=0.7)

    def test_ncc_blocks_recover_translation(self):
        vectors = block_matching(self.first, self.second, 32, 6)
        np.testing.assert_allclose(np.median(vectors[:, 2:4], axis=0), [3, -2], atol=0.5)

    def test_fusion_does_not_attenuate_sparse_displacement(self):
        vectors = motion_fusion(self.first, self.second, 32, 16)
        np.testing.assert_allclose(np.median(vectors[:, 2:4], axis=0), [3, -2], atol=0.8)


if __name__ == "__main__":
    unittest.main()
