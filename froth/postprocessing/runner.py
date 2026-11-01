"""Postprocess numeric predictions and retain the run layout used by render/eval."""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from froth.engine.evaluation import load_run
from froth.io import sha256, write_csv, write_json
from .thresholds import dynamic_threshold


def scale_priors(reference_run, k_factor=0.736):
    """Fit LoG size priors on a prediction run over training images."""
    _, records = load_run(reference_run)
    if any(row["source_split"] != "train" for row in records):
        raise ValueError("Scale priors must be fitted on training predictions only")
    areas = []
    for row in records:
        with np.load(Path(reference_run) / row["score_file"], allow_pickle=False) as data:
            probability = data["probability"]
        threshold = dynamic_threshold(probability, k_factor)
        mask = (probability > threshold).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas.extend(int(a) for a in stats[1:, cv2.CC_STAT_AREA] if a < 0.5 * mask.size)
    if not areas:
        raise ValueError("Training predictions contain no components for LoG scale estimation")
    values = np.percentile(areas, [10, 50, 90])
    return {**{f"p{p}": float(v) for p, v in zip((10, 50, 90), values)},
            **{f"r{p}": float(np.sqrt(max(v, 1) / np.pi)) for p, v in zip((10, 50, 90), values)}}


def postprocess(predictions, output, *, method="watershed", reference_run=None, k_factor=0.736, min_area=1):
    predictions, output = Path(predictions), Path(output)
    metadata, records = load_run(predictions)
    if (output / "run.json").exists():
        raise FileExistsError(output)
    scales = None
    if method == "log_voronoi":
        if reference_run is None:
            raise ValueError("--reference-run is required for training-only LoG size priors")
        reference_metadata, _ = load_run(reference_run)
        if reference_metadata["config"]["data"] != metadata["config"]["data"]:
            raise ValueError("Scale priors must use the same prediction grid and crop")
        scales = scale_priors(reference_run, k_factor)
        from .log_voronoi import partition_probability
    elif method == "watershed":
        from .roi_watershed import process_image_roi_watershed
    else:
        raise ValueError(method)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    (output / "scores").mkdir(exist_ok=True)
    counts = []
    for index, row in enumerate(records, 1):
        with np.load(predictions / row["score_file"], allow_pickle=False) as data:
            probability = data["probability"]
        if method == "watershed":
            binary, count, before, _ = process_image_roi_watershed(probability, k_factor=k_factor)
            _, labels = cv2.connectedComponents(binary, connectivity=8)
            seeds = None
        else:
            binary, labels, seeds = partition_probability(probability, scales, k_factor, min_area)
            count, before = int(labels.max()), None
        Image.fromarray(binary).save(output / row["mask_file"])
        np.savez_compressed(output / row["score_file"], probability=(binary > 0).astype(np.float32), instances=labels)
        counts.append({"image_name": row["image_name"], "source_split": row["source_split"],
                       "regions_after": count, "components_before": before, "seeds": seeds})
        print(f"Postprocessed {index}/{len(records)}: {row['image_name']} ({count} regions)", flush=True)
    metadata = copy.deepcopy(metadata)
    metadata.update(inference_mode="postprocessed_" + method, score_kind="binary_postprocessed_mask",
                    parent_run_sha256=sha256(predictions / "run.json"),
                    postprocessing={"method": method, "k_factor": k_factor, "min_area_log_voronoi": min_area,
                                    "scale_priors": scales, "scale_reference_run": str(reference_run) if reference_run else None,
                                    "region_counts_are_exploratory": True})
    write_json(output / "run.json", metadata)
    write_csv(output / "predictions.csv", records)
    write_csv(output / "region_counts.csv", counts)
    return output
