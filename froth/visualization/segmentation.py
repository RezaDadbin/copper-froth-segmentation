"""Linked segmentation panels, contours, error maps, probabilities and comparisons."""

from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from froth.data.labelme import load_sample, read_samples, restore_mask, transform_pair
from froth.engine.evaluation import load_run
from froth.io import sha256, write_json
from .common import paper_style, plt, save_figure


def contour_overlay(image, mask, covered=None):
    result = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 240, 210), max(1, image.shape[0] // 500))
    if covered is not None:
        result[~covered] = (result[~covered].astype(float) * 0.35 + 150 * 0.65).astype(np.uint8)
    return result


def error_map(prediction, target):
    pred, gt = prediction.astype(bool), target.astype(bool)
    image = np.ones((*gt.shape, 3), dtype=np.float32)
    image[pred & gt] = (0.65, 0.65, 0.65)
    image[pred & ~gt] = (0.85, 0.18, 0.20)
    image[~pred & gt] = (0.15, 0.35, 0.85)
    return image


def render_results(runs, dataset_root, manifest, output, *, image_names=None, limit=5):
    """runs is {display_label: prediction_run_path}; no model is loaded here."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = {s.image_name: s for s in read_samples(dataset_root, manifest)}
    loaded = {}
    for label, path in runs.items():
        metadata, rows = load_run(path)
        if metadata["manifest_sha256"] != sha256(manifest):
            raise ValueError("Render manifest differs from the model's prediction manifest")
        loaded[label] = (Path(path), metadata, {r["image_name"]: r for r in rows})
    common = set.intersection(*(set(value[2]) for value in loaded.values()))
    names = list(image_names) if image_names else [name for name in samples if name in common][:limit]
    if not names or any(name not in common for name in names):
        raise ValueError("Requested images must have predictions in every selected run")
    paper_style()
    written = []
    for name in names:
        raw_image, raw_gt = load_sample(samples[name])
        safe_name = re.sub(r"[^\w().-]+", "_", name)
        comparison, axs = plt.subplots(1, 2 + len(loaded), figsize=(2.8 * (2 + len(loaded)), 2.8), layout="constrained", squeeze=False)
        axes = axs[0]
        axes[0].imshow(raw_image); axes[0].set_title("Input")
        axes[1].imshow(raw_gt, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Annotation union")
        for column, (label, (path, metadata, index)) in enumerate(loaded.items(), 2):
            record = index[name]
            if sha256(samples[name].image_path) != record["image_sha256"] or sha256(samples[name].annotation_path) != record["annotation_sha256"]:
                raise ValueError(f"Rendering input changed after prediction: {name}")
            image, gt, geometry = transform_pair(raw_image, raw_gt, metadata["config"]["data"])
            pred = np.asarray(Image.open(path / record["mask_file"])) > 0
            with np.load(path / record["score_file"], allow_pickle=False) as scores:
                probability, instances = scores["probability"], scores["instances"]
            native, covered = restore_mask(pred, geometry)
            axes[column].imshow(contour_overlay(raw_image, native, covered))
            title = label + ("\nGT box prompt" if metadata["ground_truth_assisted"] else "")
            if not covered.all():
                title += "\ngray: outside crop"
            axes[column].set_title(title)
            panels, ax = plt.subplots(1, 6, figsize=(16, 2.8), layout="constrained")
            views = [image, gt, pred, contour_overlay(image, pred), error_map(pred, gt), probability]
            titles = ["Input on evaluation grid", "Annotation union", "Prediction", "Predicted contours",
                      "Red: FP / blue: FN", "Proposal union" if metadata["inference_mode"] == "amg" else "Foreground probability"]
            for i, (axis, view, title) in enumerate(zip(ax, views, titles)):
                axis.imshow(view, cmap="viridis" if i == 5 else "gray", vmin=0, vmax=1 if view.ndim == 2 else None)
                axis.set_title(title); axis.axis("off")
            safe_label = re.sub(r"[^\w.-]+", "_", label)
            stem = output / f"{safe_label}__{safe_name}"
            save_figure(panels, stem)
            written.append(stem.name)
            if instances.max():
                fig, axis = plt.subplots(figsize=(4, 3), layout="constrained")
                axis.imshow(instances, cmap="nipy_spectral", interpolation="nearest")
                label_title = ("region labels" if metadata["inference_mode"].startswith("postprocessed_")
                               else "proposal ownership (overlaps assigned by score)")
                axis.set_title(f"{label}: {label_title}"); axis.axis("off")
                save_figure(fig, Path(str(stem) + "__proposals"))
        for axis in axes:
            axis.axis("off")
        comparison.suptitle(name, fontsize=10)
        save_figure(comparison, output / f"comparison__{safe_name}")
    write_json(output / "render_manifest.json", {"images": names, "panel_stems": written,
               "runs": {key: str(Path(value).resolve()) for key, value in runs.items()},
               "formats": ["PNG (300 dpi)", "PDF"], "native_comparison_crops_marked": True})
    return output
