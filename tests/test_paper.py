"""Tests for the paper reproduction code on synthetic data."""

import unittest

import numpy as np
import pandas as pd

from paper.evaluation import metrics as M
from paper.figures import data_tables, review_tables


class InstanceMetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((40, 40), np.int32)
        self.gt[2:18, 2:18], self.gt[22:38, 22:38] = 1, 2

    def test_perfect_prediction(self):
        r = M.inst(self.gt.copy(), self.gt)
        self.assertEqual((r["inst_tp"], r["inst_f1"], r["count_rel_err"]), (2, 1.0, 0.0))
        self.assertAlmostEqual(r["pq"], 1.0)

    def test_merged_prediction_matches_nothing(self):
        merged = np.where(self.gt > 0, 1, 0)
        merged[18:22, 2:38] = 1  # bridge the two bubbles into one region
        r = M.inst(merged, self.gt)
        self.assertEqual((r["n_pred"], r["inst_tp"], r["count_rel_err"]), (1, 0, -0.5))

    def test_all_foreground_reference(self):
        gt = np.zeros((10, 10), bool)
        gt[:5] = True
        r = M.sem(np.ones_like(gt), gt)
        self.assertAlmostEqual(r["dice"], r["ref_dice_allfg"])


class TableTests(unittest.TestCase):
    def test_tag_counts(self):
        tags = pd.DataFrame(
            {
                "split": ["train", "train", "test"],
                "image_name": ["a", "b", "c"],
                "domain_yellow_copper": [1, 0, 1],
            }
        )
        for col, _ in data_tables.CONDITIONS:
            tags[col] = [1, 0, 0]
        rows = data_tables.tag_counts(tags).splitlines()
        self.assertEqual(rows[0], "Colour domain & 2 & 1 & 1 & 66.7 \\\\")
        self.assertTrue(rows[-1].startswith("At least 6 conditions & 1 & 1 & 0"))

    def test_geometry_uses_valid_shapes_only(self):
        cols = [
            "area_px2",
            "area_pct_image",
            "equivalent_diameter_px",
            "perimeter_px",
            "compactness",
            "bbox_aspect_ratio",
        ]
        m = pd.DataFrame({c: [1.0, 3.0, 1000.0] for c in cols})
        m["geometry_valid"] = [1, 1, 0]
        self.assertTrue(
            data_tables.geometry(m).startswith("Area (px$^2$) & 2.00 & 1.41 & 2.00 & 1.50--2.50")
        )

    def test_review_table_escaping(self):
        self.assertEqual(review_tables.esc("5% & ~2 – 3"), "5\\% \\& $\\sim$2 -- 3")
        self.assertTrue(review_tables.short("word " * 50, 20).endswith("\\ldots"))


if __name__ == "__main__":
    unittest.main()
