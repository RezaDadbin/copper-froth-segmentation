"""Post-processing kernels: ROI watershed and LoG-seeded geodesic Voronoi, with optional intermediate stages.

Both routes start from a froth probability map and use the original post-processing code in ``froth.postprocessing``.
"""

import math

import cv2
import numpy as np
from skimage.segmentation import relabel_sequential

from froth.postprocessing import roi_watershed as RW
from froth.postprocessing.log_voronoi import (
    build_mask_dynamic,
    detect_log_seeds,
    geodesic_voronoi_partition,
)
from froth.postprocessing.thresholds import dynamic_threshold

RW.print = lambda *a, **k: None  # silence the historical per-image print
SEMANTIC = ["dlv3", "dlv3p", "sam_ft", "hq_ft", "medsam_ft", "medsam_ft_fix", "sam_zs", "robust_zs"]


def roi_watershed_staged(prob, k_factor=0.736, keep_stages=False):
    """Same algorithm as froth.postprocessing.roi_watershed.process_image_roi_watershed, plus stage canvases."""
    raw = np.rint(np.clip(prob, 0, 1) * 255).astype(np.uint8)
    img = raw.astype(np.float32) / 255.0
    thr = dynamic_threshold(img, k_factor)
    init = RW.simple_threshold_mask(img, thr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(init, connectivity=8)
    final = np.zeros_like(init)
    H, W = init.shape
    st = (
        dict(
            threshold=thr,
            init=init,
            dist=np.zeros((H, W), np.float32),
            seeds=np.zeros((H, W), np.uint8),
        )
        if keep_stages
        else None
    )
    for lab in range(1, n):
        x, y, w, h = (
            int(stats[lab, c])
            for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
        )
        x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(W, x + w + 2), min(H, y + h + 2)
        roi = cv2.bitwise_and(
            init[y0:y1, x0:x1].copy(), (labels[y0:y1, x0:x1] == lab).astype(np.uint8) * 255
        )
        cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cs, -1, 255, cv2.FILLED)
        seed, ns, dist = RW.find_seed_components(roi)
        if keep_stages:
            blob = labels[y0:y1, x0:x1] == lab
            st["dist"][y0:y1, x0:x1][blob] = np.asarray(dist)[blob]
            st["seeds"][y0:y1, x0:x1][blob] = seed[blob]
        if ns >= RW.MERGED_PEAKS_MIN:
            sep, _ = RW.local_watershed_on_roi(img[y0:y1, x0:x1].copy(), roi)
            if sep is None:
                final[y0:y1, x0:x1] |= roi
            else:
                final[y0:y1, x0:x1] |= RW.filter_small_components(sep, RW.MIN_AREA)[0]
        else:
            final[y0:y1, x0:x1] |= roi
    final, _ = RW.filter_small_components(final, RW.MIN_AREA)
    inst = cv2.connectedComponents(final, connectivity=8)[1].astype(np.int32)
    if keep_stages:
        st["final"] = final
    return inst, st


def log_voronoi_staged(prob, scales, k_factor=0.736, keep_stages=False):
    p = prob.astype(np.float32)
    mask = build_mask_dynamic(p, k_factor)
    seeds = detect_log_seeds(p, mask, scales)
    lab = relabel_sequential(geodesic_voronoi_partition(mask, seeds))[0].astype(np.int32)
    return lab, (dict(mask=mask, seeds=np.array(seeds)) if keep_stages else None)


def scale_priors(probs, k_factor=0.736):
    """Historical compute_percentiles_for_scales(), applied to one model's own probability maps (no labels used)."""
    areas = []
    for p in probs:
        raw = np.rint(p * 255).astype(np.uint8)
        th = float(raw.mean()) - k_factor * float(raw.std())
        m = cv2.threshold(raw, th, 255, cv2.THRESH_BINARY)[1]
        n, _, s, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            a = s[1:, cv2.CC_STAT_AREA]
            areas += a[a < 0.5 * raw.size].tolist()
    q = np.percentile(areas, [10, 50, 90]) if areas else np.array([50.0, 200.0, 800.0])
    r = [math.sqrt(max(v, 1.0) / math.pi) for v in q]
    return dict(p10=q[0], p50=q[1], p90=q[2], r10=r[0], r50=r[1], r90=r[2])


# ------------------------------------------------------------------ metrics
