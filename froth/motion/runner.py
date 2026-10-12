"""Motion numeric results and linked headless visualization for an explicit pair."""

from pathlib import Path

import cv2
import numpy as np

from froth.io import sha256, write_csv, write_json
from .estimators import block_matching, motion_fusion, optical_flow, phase_correlation, stability, validate_frames


def run_motion(first_path, second_path, output, *, method="fusion", block_size=64, search_range=16,
               spatial_scale=None, seconds_per_frame=None):
    if (spatial_scale is None) != (seconds_per_frame is None):
        raise ValueError("Physical speed needs both --spatial-scale (mm/px) and --seconds-per-frame")
    if spatial_scale is not None and (spatial_scale <= 0 or seconds_per_frame <= 0):
        raise ValueError("Calibration values must be positive")
    first, second = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (first_path, second_path)]
    validate_frames(first, second)
    if method == "phase_correlation":
        dx, dy = phase_correlation(first, second)
        vectors = np.array([[first.shape[1]//2, first.shape[0]//2, dx, dy, 1]], np.float32)
    elif method == "optical_flow":
        flow = optical_flow(first, second)
        yy, xx = np.mgrid[0:first.shape[0]:16, 0:first.shape[1]:16]
        vectors = np.column_stack((xx.ravel(), yy.ravel(), flow[yy, xx].reshape(-1, 2), np.ones(xx.size)))
    elif method in {"block_matching", "block_matching_sad"}:
        vectors = block_matching(first, second, block_size, search_range,
                                 method="sad" if method.endswith("_sad") else "ncc")
    elif method == "fusion":
        vectors = motion_fusion(first, second, block_size, max(1, block_size//2))
    else:
        raise ValueError(method)
    if not len(vectors):
        raise ValueError("No textured motion vectors found; inspect the frame pair/parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "vectors.csv", [dict(zip(("x", "y", "dx_px", "dy_px", "confidence"), map(float, row))) for row in vectors])
    magnitude = np.hypot(vectors[:, 2], vectors[:, 3])
    summary = {"method": method, "vectors": len(vectors), "mean_displacement_px_per_frame": float(magnitude.mean()),
               "mean_dx_px": float(vectors[:, 2].mean()), "mean_dy_px": float(vectors[:, 3].mean()),
               "spatial_scale_mm_per_px": spatial_scale, "seconds_per_frame": seconds_per_frame,
               "physical_mean_speed_mm_per_s": float(magnitude.mean()*spatial_scale/seconds_per_frame) if spatial_scale else None,
               **stability(vectors), "first_sha256": sha256(first_path), "second_sha256": sha256(second_path),
               "exploratory_motion": True}
    write_json(output / "motion.json", summary)
    from froth.visualization.motion import render_motion
    render_motion(first, second, vectors, output)
    return output
