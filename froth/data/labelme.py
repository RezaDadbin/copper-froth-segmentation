"""Read paired images/LabelMe labels without trusting stale JSON imagePath fields."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    image_name: str
    split: str
    image_path: Path
    annotation_path: Path


def read_samples(dataset_root, manifest, split="all"):
    """Select only listed samples; reject duplicates, ambiguity and missing pairs."""
    root = Path(dataset_root).resolve()
    if split not in {"all", "train", "test"}:
        raise ValueError(f"Unexpected source split: {split}")
    with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    seen, samples = set(), []
    for row in rows:
        name, source_split = row["image_name"], row["split"]
        if name in seen or Path(name).name != name or source_split not in {"train", "test"}:
            raise ValueError(f"Duplicate or invalid manifest entry: {name!r}")
        seen.add(name)
        if split != "all" and source_split != split:
            continue
        folder = root / {"train": "train1", "test": "test1"}[source_split]
        image = folder / name
        if not image.suffix:
            candidates = [folder / (name + ext) for ext in (".tif", ".tiff", ".png", ".jpg")]
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous/missing image: {folder / name}")
            image = candidates[0]
        annotation = image.with_suffix(".json")
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Missing image/JSON pair: {image}")
        samples.append(Sample(name, source_split, image, annotation))
    if not samples:
        raise ValueError(f"No samples selected for {split!r}")
    return samples


def rasterize(document, height, width, label="froth"):
    """Union polygons/circles on the native grid, using OpenCV integer pixels.

    Self-intersecting polygons retain OpenCV's fill rule; this is a raster target,
    not a repair of invalid continuous annotation geometry.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in document.get("shapes", []):
        if shape.get("label", "").strip().lower() != label:
            continue
        points = np.asarray(shape.get("points", []), dtype=float)
        kind = shape.get("shape_type") or "polygon"
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Malformed or non-finite froth annotation points")
        if kind == "polygon" and len(points) >= 3:
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        elif kind == "circle" and len(points) == 2:
            center = tuple(np.rint(points[0]).astype(int))
            radius = int(round(math.hypot(*(points[1] - points[0]))))
            cv2.circle(mask, center, radius, 1, thickness=-1)
        else:
            raise ValueError(f"Unsupported froth shape: {kind!r} ({len(points)} points)")
    return mask


def load_sample(sample):
    with Image.open(sample.image_path) as source:
        image = np.asarray(source.convert("RGB"))
    document = json.loads(sample.annotation_path.read_text(encoding="utf-8"))
    height, width = image.shape[:2]
    if (document.get("imageHeight"), document.get("imageWidth")) != (height, width):
        raise ValueError(f"Annotation dimensions do not match image: {sample.image_name}")
    return image, rasterize(document, height, width)


def transform_pair(image, mask, data_config):
    """Deterministic evaluation transform; nearest-neighbor masks stay binary."""
    height, width = image.shape[:2]
    crop = data_config.get("center_crop")
    box = (0, 0, width, height)
    if crop:
        size = (1080 if height > 480 else 480) if crop == "legacy_unet" else int(crop)
        left, top = int(round((width - size) / 2)), int(round((height - size) / 2))
        box = (left, top, left + size, top + size)
        # PIL pads requested crop coordinates outside an image with zero.
        image = np.asarray(Image.fromarray(image).crop(box))
        mask = np.asarray(Image.fromarray(mask).crop(box))
    target = data_config.get("image_size")
    if target:
        out_h, out_w = map(int, target)
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return image, mask, {"native_size": [height, width], "crop_xyxy": list(box),
                         "evaluation_size": list(mask.shape)}


def restore_mask(mask, geometry):
    """Return native-grid prediction plus coverage; cropped-away pixels are unknown."""
    height, width = geometry["native_size"]
    x0, y0, x1, y1 = geometry["crop_xyxy"]
    resized = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    result, covered = np.zeros((height, width), bool), np.zeros((height, width), bool)
    left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    result[top:bottom, left:right] = resized[top-y0:bottom-y0, left-x0:right-x0] > 0
    covered[top:bottom, left:right] = True
    return result, covered


def prediction_name(image_name):
    # Retain the original extension so that two stems with different extensions
    # cannot silently overwrite one another.
    return image_name + ".png"
