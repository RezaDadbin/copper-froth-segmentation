"""Analytic geometry/metric tests, independent of training data and checkpoints."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from froth.data.labelme import load_sample, rasterize, read_samples, restore_mask, transform_pair
from froth.metrics import binary_metrics


class DataTests(unittest.TestCase):
    def test_polygon_and_circle_are_both_rasterized(self):
        document = {"shapes": [
            {"label": "froth", "shape_type": "polygon", "points": [[1, 1], [3, 1], [3, 3], [1, 3]]},
            {"label": "froth", "shape_type": "circle", "points": [[7, 7], [8, 7]]}]}
        mask = rasterize(document, 10, 10)
        self.assertEqual(int(mask.sum()), 14)  # inclusive 3x3 square plus five circle pixels
        self.assertEqual(mask[7, 7], 1)

    def test_nearest_resize_preserves_binary_labels(self):
        image = np.zeros((10, 12, 3), np.uint8)
        mask = np.zeros((10, 12), np.uint8); mask[2:5, 4:8] = 1
        _, resized, _ = transform_pair(image, mask, {"image_size": [7, 9]})
        self.assertEqual(set(np.unique(resized)), {0, 1})

    def test_cropped_predictions_have_explicit_native_coverage(self):
        image, mask = np.zeros((6, 10, 3), np.uint8), np.ones((6, 10), np.uint8)
        _, transformed, geometry = transform_pair(image, mask, {"center_crop": 6, "image_size": [3, 3]})
        restored, covered = restore_mask(transformed, geometry)
        self.assertEqual(int(covered.sum()), 36)
        self.assertFalse(covered[:, :2].any())
        self.assertTrue(restored[covered].all())

    def test_padding_crop_restores_only_original_pixels(self):
        _, transformed, geometry = transform_pair(np.zeros((4, 6, 3), np.uint8), np.ones((4, 6), np.uint8),
                                                   {"center_crop": 8, "image_size": [8, 8]})
        restored, covered = restore_mask(transformed, geometry)
        self.assertTrue(covered.all()); self.assertTrue(restored.all())

    def test_manifest_ignores_unlisted_json_and_stale_image_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "train1").mkdir()
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "train1/a.tif")
            doc = {"imageWidth": 8, "imageHeight": 8, "imagePath": "incorrect.tif", "shapes": []}
            (root / "train1/a.json").write_text(json.dumps(doc))
            (root / "train1/unlisted.json").write_text("{}")
            manifest = root / "tags.csv"
            manifest.write_text("image_name,split\na.tif,train\n")
            samples = read_samples(root, manifest)
            self.assertEqual(len(samples), 1)
            self.assertEqual(load_sample(samples[0])[1].shape, (8, 8))
            manifest.write_text("image_name,split\na.tif,train\na.tif,train\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_samples(root, manifest)

    def test_invalid_shape_is_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            rasterize({"shapes": [{"label": "froth", "shape_type": "point", "points": [[0, 0]]}]}, 4, 4)


class MetricTests(unittest.TestCase):
    def test_hand_computed_confusion_and_distances(self):
        result = binary_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]])
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["dice"], 0.5)
        self.assertEqual(result["pixel_acc"], 0.5)
        self.assertEqual(result["HD"], 1)

    def test_empty_mask_policy(self):
        empty = np.zeros((5, 5), bool)
        both = binary_metrics(empty, empty)
        self.assertEqual(both["iou"], 1); self.assertEqual(both["HD"], 0)
        one = binary_metrics(empty, ~empty)
        self.assertIsNone(one["HD"]); self.assertIsNone(one["precision"])
        self.assertEqual(one["recall"], 0)

    def test_geometry_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics(np.zeros((2, 2)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
