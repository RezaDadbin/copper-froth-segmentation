"""Foreground semantic metrics; boundary distances are measured in grid pixels."""

from __future__ import annotations

import numpy as np


def binary_metrics(prediction, target, distances=True):
    pred, gt = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"Expected matching 2D masks, got {pred.shape} and {gt.shape}")
    tp, fp = int((pred & gt).sum()), int((pred & ~gt).sum())
    fn, tn = int((~pred & gt).sum()), int((~pred & ~gt).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    background_iou = tn / (tn + fp + fn) if tn + fp + fn else 1.0
    result = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "pixel_acc": (tp + tn) / pred.size,
              "iou": iou, "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
              "precision": tp / (tp + fp) if tp + fp else None,
              "recall": tp / (tp + fn) if tp + fn else None,
              "background_iou": background_iou, "miou": (iou + background_iou) / 2}
    if distances:
        from scipy.ndimage import binary_erosion, distance_transform_edt
        if not pred.any() and not gt.any():
            hd, hd95, status = 0.0, 0.0, "both_empty"
        elif not pred.any() or not gt.any():
            hd, hd95, status = None, None, "one_empty"
        else:
            boundary_pred = pred & ~binary_erosion(pred, border_value=0)
            boundary_gt = gt & ~binary_erosion(gt, border_value=0)
            values = np.concatenate([distance_transform_edt(~boundary_gt)[boundary_pred],
                                     distance_transform_edt(~boundary_pred)[boundary_gt]])
            hd, hd95, status = float(values.max()), float(np.percentile(values, 95)), "defined"
        result.update(HD=hd, HD95=hd95, boundary_distance_status=status)
    return result
