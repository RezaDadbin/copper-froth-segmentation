"""Preprocessing, thresholding, contour filtering and overlays for dark-impurity detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CLAHEParams:
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: int = 8


@dataclass
class SegmentParams:
    method: str = "dynamic"  # "dynamic" | "otsu" | "fixed" | "adaptive"
    k_dynamic: float = 0.834
    fixed_thresh: int = 140
    invert: bool = True
    morph_open_kernel: int = 2
    morph_open_iterations: int = 1
    adaptive_block_size: int = 35
    adaptive_C: int = -5


@dataclass
class FilterParams:
    min_area_px: int = 5
    max_area_px: int = 11


@dataclass
class VizParams:
    draw_contours: bool = True
    overlay_color: tuple[int, int, int] = (255, 0, 0)  # BGR
    overlay_thickness: int = 1
    label_color: tuple[int, int, int] = (0, 255, 0)  # BGR


def preprocess_gray(image: np.ndarray, params: CLAHEParams) -> np.ndarray:
    """Convert to grey levels and optionally apply CLAHE."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not params.enabled:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=float(params.clip_limit),
        tileGridSize=(int(params.tile_grid_size), int(params.tile_grid_size)),
    )
    return clahe.apply(gray)


def threshold_mask(gray: np.ndarray, p: SegmentParams) -> np.ndarray:
    """Binary mask of dark regions (255 = candidate impurity when ``invert`` is set)."""
    if p.method == "dynamic":
        t = int(max(0, min(255, round(float(np.mean(gray)) - p.k_dynamic * float(np.std(gray))))))
        _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    elif p.method == "otsu":
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif p.method == "fixed":
        _, m = cv2.threshold(gray, int(p.fixed_thresh), 255, cv2.THRESH_BINARY)
    elif p.method == "adaptive":
        block = max(3, p.adaptive_block_size | 1)
        m = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, p.adaptive_C
        )
    else:
        raise ValueError(f"Unknown method: {p.method}")
    if p.morph_open_kernel > 0 and p.morph_open_iterations > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.morph_open_kernel, p.morph_open_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=p.morph_open_iterations)
    return cv2.bitwise_not(m) if p.invert else m


def filter_contours(mask: np.ndarray, f: FilterParams) -> list[np.ndarray]:
    """External contours whose area lies within ``[min_area_px, max_area_px]``."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < f.min_area_px or (f.max_area_px and a > f.max_area_px):
            continue
        out.append(c)
    return out


def draw_overlay(
    image: np.ndarray, contours: list[np.ndarray], count: int, viz: VizParams
) -> np.ndarray:
    """Image with the detected contours and the impurity count."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if viz.draw_contours and contours:
        cv2.drawContours(out, contours, -1, viz.overlay_color, viz.overlay_thickness)
    cv2.putText(
        out,
        f"impurities: {count}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        viz.label_color,
        2,
        cv2.LINE_AA,
    )
    return out
