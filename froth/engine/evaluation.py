"""Evaluate saved numeric predictions independently of model execution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from froth.data.labelme import load_sample, read_samples, transform_pair
from froth.io import sha256, write_csv, write_json
from froth.metrics import binary_metrics


def load_run(path):
    path = Path(path)
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    with (path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records or len({row["image_name"] for row in records}) != len(records):
        raise ValueError(f"Empty or duplicate predictions index: {path}")
    if metadata["images"] != len(records):
        raise ValueError("Prediction index and run metadata image counts disagree")
    return metadata, records


def evaluate(predictions, dataset_root, manifest, output, *, distances=True):
    predictions, output = Path(predictions), Path(output)
    metadata, records = load_run(predictions)
    if sha256(manifest) != metadata["manifest_sha256"]:
        raise ValueError("The supplied manifest differs from the prediction manifest")
    samples = {sample.image_name: sample for sample in read_samples(dataset_root, manifest)}
    rows = []
    for index, record in enumerate(records, 1):
        sample = samples[record["image_name"]]
        if sha256(sample.image_path) != record["image_sha256"] or sha256(sample.annotation_path) != record["annotation_sha256"]:
            raise ValueError(f"Image or annotations changed after prediction: {sample.image_name}")
        image, target = load_sample(sample)
        _, target, geometry = transform_pair(image, target, metadata["config"]["data"])
        if geometry != metadata["geometry"][sample.image_name]:
            raise ValueError("Saved and current evaluation geometry disagree")
        mask = np.asarray(Image.open(predictions / record["mask_file"]))
        if not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"Evaluation requires a binary mask, not a colormap: {record['mask_file']}")
        metrics = binary_metrics(mask > 0, target, distances=distances)
        rows.append({"image_name": sample.image_name, "source_split": sample.split,
                     "evaluation_role": "training" if sample.split == "train" else "reused_validation",
                     "evaluation_height": mask.shape[0], "evaluation_width": mask.shape[1], **metrics})
        print(f"Evaluated {index}/{len(records)}: {sample.image_name}", flush=True)
    summary = []
    metric_names = ("pixel_acc", "iou", "dice", "precision", "recall", "background_iou", "miou")
    if distances:
        metric_names += ("HD", "HD95")
    for split in sorted({row["source_split"] for row in rows}):
        selected = [row for row in rows if row["source_split"] == split]
        for metric in metric_names:
            values = [row[metric] for row in selected if row[metric] is not None]
            summary.append({"source_split": split, "metric": metric, "n_images": len(selected),
                            "n_defined": len(values), "mean": statistics.fmean(values) if values else None,
                            "std": statistics.stdev(values) if len(values) > 1 else None,
                            "median": statistics.median(values) if values else None})
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "evaluation.json", {
        "prediction_run": str(predictions.resolve()), "prediction_run_sha256": sha256(predictions / "run.json"),
        "images": len(rows), "aggregation": "unweighted per-image macro means within source split",
        "distance_units": "evaluation_grid_pixels", "ground_truth_assisted": metadata["ground_truth_assisted"],
        "empty_mask_policy": "IoU/Dice=1 when both empty; undefined precision/recall and one-empty HD/HD95 are blank",
        "HD95_policy": "95th percentile of pooled bidirectional foreground-boundary nearest distances",
        "inputs_unchanged": True})
    return output
