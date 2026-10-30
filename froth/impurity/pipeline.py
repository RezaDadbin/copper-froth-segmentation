"""Configuration, per-image processing and batch runs for impurity detection."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .detection import (
    CLAHEParams,
    FilterParams,
    SegmentParams,
    VizParams,
    draw_overlay,
    filter_contours,
    preprocess_gray,
    threshold_mask,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImpurityConfig:
    clahe: CLAHEParams = field(default_factory=CLAHEParams)
    segment: SegmentParams = field(default_factory=SegmentParams)
    filter: FilterParams = field(default_factory=FilterParams)
    viz: VizParams = field(default_factory=VizParams)
    save_mask: bool = True
    save_overlay: bool = True


def load_config(path: str | Path) -> ImpurityConfig:
    """Read a YAML configuration (see ``configs/impurity.yaml``)."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text()) or {}
    viz = dict(cfg.get("visualize", {}))
    for key in ("overlay_color", "label_color"):
        if key in viz:
            viz[key] = tuple(viz[key])
    export = cfg.get("export", {})
    return ImpurityConfig(
        CLAHEParams(**cfg.get("preprocess", {}).get("clahe", {})),
        SegmentParams(**cfg.get("segment", {})),
        FilterParams(**cfg.get("filter", {})),
        VizParams(**viz),
        save_mask=bool(export.get("save_mask", True)),
        save_overlay=bool(export.get("save_overlay", True)),
    )


def process_image(path: str | Path, cfg: ImpurityConfig) -> dict:
    """Detect impurities on one image; returns the mask, the overlay and the count."""
    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = preprocess_gray(color, cfg.clahe)
    mask = threshold_mask(gray, cfg.segment)
    contours = filter_contours(mask, cfg.filter)
    return {
        "mask": mask,
        "overlay": draw_overlay(color, contours, len(contours), cfg.viz),
        "count": len(contours),
    }


def run(inputs: list[Path], output_dir: str | Path, cfg: ImpurityConfig) -> Path:
    """Process images, write masks/overlays and ``results.csv``; returns the CSV path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "count"])
        for path in inputs:
            res = process_image(path, cfg)
            if cfg.save_mask:
                cv2.imwrite(str(out / f"{path.stem}_mask.png"), res["mask"])
            if cfg.save_overlay:
                cv2.imwrite(str(out / f"{path.stem}_overlay.png"), res["overlay"])
            writer.writerow([path.name, res["count"]])
            print(f"{path.name}: {res['count']} impurities")
    return csv_path


def list_images(folder: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
