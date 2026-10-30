"""Detection and counting of dark impurities on froth images (classical OpenCV pipeline)."""

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
from .pipeline import ImpurityConfig, load_config, process_image, run

__all__ = [
    "CLAHEParams",
    "FilterParams",
    "SegmentParams",
    "VizParams",
    "ImpurityConfig",
    "draw_overlay",
    "filter_contours",
    "load_config",
    "preprocess_gray",
    "process_image",
    "run",
    "threshold_mask",
]
