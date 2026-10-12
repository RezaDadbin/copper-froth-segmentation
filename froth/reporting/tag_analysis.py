#!/usr/bin/env python3
"""Summarize model metrics by tag and split using only the Python standard library."""

import argparse
import csv
import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = None
OUTPUT = None
METRICS = ("pixel_acc", "iou", "dice", "precision", "recall", "HD", "HD95")
STATISTICS = ("mean", "std", "median")
MODELS = {
    "deeplabv3_resnet101": "metrics_deeplabv3_resnet101_all.csv",
    "deeplabv3plus_r152_os8": "metrics_deeplabv3plus_r152_os8_all.csv",
    "hqsam": "metrics_hqsam_hqamg_all.csv",
    "medsam": "metrics_medsam_all.csv",
    "medsam_tiled": "metrics_medsam_all_tiled.csv",
    "robustsam": "metrics_robustsam_all.csv",
    "sam": "metrics_sam_amg_all.csv",
    "sbsnet": "metrics_sbsnet_all.csv",
    "unet": "metrics_unet_all.csv",
}
SPLITS = ("train", "test")
# The user confirmed that the source "test" images were used for both model
# selection and final reporting. Retain the source label and state their role.
EVALUATION_ROLES = {
    "train": "training",
    "test": "validation_used_for_selection_and_reporting",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if None in reader.fieldnames or any(None in row for row in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return reader.fieldnames, rows


def write_csv(filename, columns, rows):
    path = OUTPUT / filename
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        saved_columns, saved_rows = read_csv(temporary)
        expected_rows = [
            {column: str(row[column]) for column in columns} for row in rows
        ]
        if saved_columns != columns or saved_rows != expected_rows:
            raise ValueError(f"Output verification failed: {filename}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def format_number(value):
    return "" if value is None else format(value, ".10g")


def summarize(names, results):
    scored = [results[name] for name in names if name in results]
    summary = {
        "n_images": len(names),
        "n_scored": len(scored),
        "n_missing": len(names) - len(scored),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in scored]
        summary[f"{metric}_mean"] = format_number(
            statistics.mean(values) if values else None
        )
        summary[f"{metric}_std"] = format_number(
            statistics.stdev(values) if len(values) >= 2 else None
        )
        summary[f"{metric}_median"] = format_number(
            statistics.median(values) if values else None
        )
    return summary


def main(argv=None):
    global ROOT, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    ROOT, OUTPUT = args.project_root.resolve(), args.output_dir.resolve()
    if OUTPUT == ROOT:
        parser.error("Output must be separate from the project input directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_paths = [
        ROOT / "tags.csv",
        *sorted((ROOT / "tags").glob("*.csv")),
        *sorted((ROOT / "models").glob("*.csv")),
    ]
    original_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
    }
    columns, dataset = read_csv(ROOT / "tags.csv")
    by_name = {row["image_name"]: row for row in dataset}
    if len(by_name) != len(dataset):
        raise ValueError("Dataset image names must be unique.")
    if any(row["split"] not in SPLITS for row in dataset):
        raise ValueError("Unexpected dataset split.")
    individual_tags = [key for key in columns if key not in {"image_name", "split"}]
    condition_tags = [key for key in individual_tags if key != "domain_yellow_copper"]
    if len(condition_tags) != 9:
        raise ValueError("Expected nine condition tags excluding domain_yellow_copper.")
    if any(row[tag] not in {"0", "1"} for row in dataset for tag in individual_tags):
        raise ValueError("Tag values must be binary.")

    tag_groups = {}
    for path in sorted((ROOT / "tags").glob("*.csv")):
        tag_columns, rows = read_csv(path)
        names = [row["image_name"] for row in rows]
        if tag_columns != ["image_name"] or len(set(names)) != len(names):
            raise ValueError(f"Invalid tag CSV: {path.name}")
        tag = path.stem
        if tag in individual_tags:
            expected = [row["image_name"] for row in dataset if row[tag] == "1"]
        elif tag.startswith("no_") and tag[3:] in individual_tags:
            expected = [row["image_name"] for row in dataset if row[tag[3:]] == "0"]
        elif tag == "at_most_2_condition_tags":
            expected = [row["image_name"] for row in dataset if sum(int(row[key]) for key in condition_tags) <= 2]
        elif tag == "at_least_6_condition_tags":
            expected = [row["image_name"] for row in dataset if sum(int(row[key]) for key in condition_tags) >= 6]
        else:
            raise ValueError(f"Unknown tag group: {tag}")
        if names != expected:
            raise ValueError(f"Tag group differs from tags.csv: {tag}")
        tag_groups[tag] = names

    if {path.name for path in (ROOT / "models").glob("*.csv")} != set(MODELS.values()):
        raise ValueError("Unexpected or missing source model files.")
    models = {}
    for model, filename in MODELS.items():
        model_columns, rows = read_csv(ROOT / "models" / filename)
        if not set(METRICS).issubset(model_columns):
            raise ValueError(f"Required metrics missing: {filename}")
        results = {row["image_file"]: row for row in rows}
        if len(results) != len(rows) or set(results) - set(by_name):
            raise ValueError(f"Duplicate or unknown model image names: {filename}")
        for name, result in results.items():
            expected_split = {"train": "train", "test": "val"}[by_name[name]["split"]]
            if result["split"] != expected_split:
                raise ValueError(f"Model/dataset split mismatch: {model}/{name}")
            if any(not math.isfinite(float(result[key])) for key in METRICS):
                raise ValueError(f"Missing or nonfinite source metric: {model}/{name}")
        models[model] = results

    summaries = []
    overall = []
    comparisons = []
    for model, results in models.items():
        for split in SPLITS:
            names = [row["image_name"] for row in dataset if row["split"] == split]
            identifiers = {"model": model, "split": split, "evaluation_role": EVALUATION_ROLES[split]}
            overall.append({**identifiers, **summarize(names, results)})
            for tag, group_names in tag_groups.items():
                names = [name for name in group_names if by_name[name]["split"] == split]
                summaries.append({"tag_group": tag, **identifiers, **summarize(names, results)})
            for tag in individual_tags:
                row = {"tag": tag, **identifiers}
                parts = {}
                for value in ("0", "1"):
                    names = [item["image_name"] for item in dataset if item["split"] == split and item[tag] == value]
                    parts[value] = summarize(names, results)
                    for key, value_stat in parts[value].items():
                        row[f"tag{value}_{key}"] = value_stat
                for metric in METRICS:
                    # Subtract unrounded means computed from original per-image scores.
                    values0 = [float(results[item["image_name"]][metric]) for item in dataset if item["split"] == split and item[tag] == "0" and item["image_name"] in results]
                    values1 = [float(results[item["image_name"]][metric]) for item in dataset if item["split"] == split and item[tag] == "1" and item["image_name"] in results]
                    delta = statistics.mean(values1) - statistics.mean(values0) if values0 and values1 else None
                    row[f"{metric}_mean_difference_tag1_minus_tag0"] = format_number(delta)
                comparisons.append(row)

    stats_columns = ["n_images", "n_scored", "n_missing"] + [
        f"{metric}_{statistic}" for metric in METRICS for statistic in STATISTICS
    ]
    comparison_columns = ["tag", "model", "split", "evaluation_role"] + [
        f"tag{value}_{column}" for value in ("0", "1") for column in stats_columns
    ] + [f"{metric}_mean_difference_tag1_minus_tag0" for metric in METRICS]
    write_csv("overall_model_summary.csv", ["model", "split", "evaluation_role", *stats_columns], overall)
    write_csv("group_model_summary.csv", ["tag_group", "model", "split", "evaluation_role", *stats_columns], summaries)
    write_csv("tag_presence_comparison.csv", comparison_columns, comparisons)

    for path, before in original_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ValueError(f"Input file was modified: {path}")

    print(f"Dataset: {len(dataset)} images; splits: {dict(Counter(row['split'] for row in dataset))}")
    print(f"Models: {len(models)}; tag groups: {len(tag_groups)}; individual tags: {len(individual_tags)}")
    print(f"Overall summaries: {len(overall)} rows; group summaries: {len(summaries)} rows; tag-presence comparisons: {len(comparisons)} rows")
    print("All outputs verified; source files preserved. Statistics exclude missing results.")
    print("REUSED VALIDATION SET (source split=test): MODEL | N | MEAN IOU | MEAN DICE")
    for row in overall:
        if row["split"] == "test":
            print(row["model"], row["n_scored"], row["iou_mean"], row["dice_mean"], sep=" | ")


if __name__ == "__main__":
    main()
