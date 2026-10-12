"""Constant-foreground reference scores from saved per-image confusion counts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from froth.io import sha256, write_json


def check(models, output):
    baselines, checks = [], []
    for source in sorted(Path(models).glob("*.csv")):
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]
        if not rows:
            continue
        fractions, constant_dice, dice_errors, iou_errors = [], [], [], []
        for row in rows:
            tp, tn, fp, fn = (float(row[key]) for key in ("tp", "tn", "fp", "fn"))
            total, positive = tp + tn + fp + fn, tp + fn
            if min(tp, tn, fp, fn) < 0 or min(total, tp+fp+fn, 2*tp+fp+fn) <= 0:
                raise ValueError(f"Undefined/invalid confusion counts: {source}")
            fractions.append(positive / total)
            constant_dice.append(2 * positive / (total + positive))
            dice_errors.append(abs(float(row["dice"]) - 2 * tp / (2 * tp + fp + fn)))
            iou_errors.append(abs(float(row["iou"]) - tp / (tp + fp + fn)))
        baselines.append({"model": source.stem, "images": len(rows),
                          "mean_foreground_fraction": statistics.fmean(fractions),
                          "mean_all_foreground_dice": statistics.fmean(constant_dice),
                          "recorded_mean_iou": statistics.fmean(float(row["iou"]) for row in rows),
                          "recorded_mean_dice": statistics.fmean(float(row["dice"]) for row in rows),
                          "source_sha256": sha256(source)})
        checks.append({"model": source.stem, "n": len(rows),
                       "max_dice_error": max(dice_errors), "max_iou_error": max(iou_errors)})
    if not baselines or any(max(row["max_dice_error"], row["max_iou_error"]) >= 1e-6 for row in checks):
        raise ValueError("Missing model files or metrics inconsistent with confusion counts")
    write_json(Path(output) / "foreground_baseline_check.json", baselines)
    write_json(Path(output) / "confusion_metric_consistency.json", checks)
    print(f"Checked {len(baselines)} model CSVs; outputs saved to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    check(args.models, args.output)


if __name__ == "__main__":
    main()
