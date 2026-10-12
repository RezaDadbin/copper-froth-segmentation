"""LoG/partition kernels extracted from POST/froth-postprocess/testing-gpt.ipynb.

The runner retains adjacent Voronoi label identities after filtering tiny labels;
converting the label map back to connected components would merge them again.
"""

from typing import Dict, Tuple, List
import os, math
import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.measure import label as sk_label
from skimage.segmentation import watershed, relabel_sequential
from scipy.ndimage import distance_transform_edt as edt
from .thresholds import dynamic_threshold

K_FACTOR = 0.736
MIN_FROTH_AREA = 1
LV_LOG_NUM_SCALES = 8
LV_LOG_THRESHOLD = 0.02
LV_MIN_DISTANCE_FACTOR = 0.50
LV_EDGE_SEED_DT_MIN_FACTOR = 0.30
LV_FORCE_ONE_SEED_PER_CC = True
LV_WATERSHED_LINE = False


def compute_percentiles_for_scales(image_folder: str,
                                   k_factor: float = 0.736) -> Dict[str, float]:
    """
    Apply your dynamic threshold per image, collect CC areas, compute p10/p50/p90.
    Excludes components >50% of image area.
    """
    files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
             if f.lower().endswith(('.png','.jpg','.jpeg','.tiff'))]
    areas = []
    H = W = None
    for p in files:
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        H, W = raw.shape[:2]
        mean_val = float(np.mean(raw))
        std_val  = float(np.std(raw))
        th255 = mean_val - k_factor * std_val
        _, init_mask = cv2.threshold(raw, th255, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(init_mask, connectivity=8)
        if n > 1:
            comp = stats[1:, cv2.CC_STAT_AREA]
            valid = comp[comp < 0.5 * H * W]
            areas.extend(valid.tolist())
    if not areas:
        p10 = 50.0; p50 = 200.0; p90 = 800.0  # fallbacks
    else:
        areas = np.asarray(areas, dtype=np.float64)
        p10 = float(np.percentile(areas, 10))
        p50 = float(np.percentile(areas, 50))
        p90 = float(np.percentile(areas, 90))

    def area_to_r(a): 
        return math.sqrt(max(a, 1.0) / math.pi)

    r10, r50, r90 = area_to_r(p10), area_to_r(p50), area_to_r(p90)
    return dict(p10=p10, p50=p50, p90=p90, r10=r10, r50=r50, r90=r90)


def build_mask_dynamic(img01: np.ndarray, k_factor: float) -> np.ndarray:
    raw_like = (img01 * 255.0).astype(np.float32)
    mean_val = float(np.mean(raw_like))
    std_val  = float(np.std(raw_like))
    dynamic_thresh_255 = mean_val - k_factor * std_val
    dynamic_thresh_norm = dynamic_threshold(img01, k_factor)
    mask_u8 = ((img01 > dynamic_thresh_norm).astype(np.uint8) * 255)
    return mask_u8


def _radius_to_sigma(radius_px: float) -> float:
    # For LoG, blob radius ≈ sqrt(2) * sigma
    return max(1.0, float(radius_px) / math.sqrt(2.0))


def detect_log_seeds(img01: np.ndarray,
                     mask_u8: np.ndarray,
                     scales: Dict[str, float]) -> List[Tuple[int,int]]:
    """
    Return list of seed (y,x) using multi-scale LoG inside the mask, with spacing and edge filters.
    """
    M = (mask_u8 > 0)
    if M.sum() == 0:
        return []

    imgM = img01.copy()
    imgM[~M] = 0.0

    # sigma range from priors
    min_sigma = _radius_to_sigma(0.8 * scales['r10'])
    max_sigma = _radius_to_sigma(1.2 * scales['r90'])
    if max_sigma < min_sigma:
        max_sigma = min_sigma + 1.0

    blobs = blob_log(imgM, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=LV_LOG_NUM_SCALES, threshold=LV_LOG_THRESHOLD, overlap=0.3)
    if blobs is None or len(blobs) == 0:
        blobs = np.zeros((0,3), dtype=float)

    pts = blobs[:, :2]  # (y,x)
    # distance to background for ranking / edge filter
    dt = cv2.distanceTransform((M.astype(np.uint8))*255, cv2.DIST_L2, 3)

    # Greedy NMS by descending DT score (favor true centers)
    min_dist = max(1.0, LV_MIN_DISTANCE_FACTOR * scales['r50'])
    kept = []
    used = np.zeros(len(pts), dtype=bool)
    if len(pts) > 0:
        scores = dt[np.clip(pts[:,0].astype(int), 0, dt.shape[0]-1),
                    np.clip(pts[:,1].astype(int), 0, dt.shape[1]-1)]
        order = np.argsort(-scores)
        for idx in order:
            if used[idx]:
                continue
            y, x = pts[idx]
            # edge filter: drop seeds too close to mask boundary
            if dt[int(round(y)), int(round(x))] < LV_EDGE_SEED_DT_MIN_FACTOR * scales['r10']:
                continue
            kept.append((int(round(y)), int(round(x))))
            # suppress neighbors
            d2 = (pts[:,0]-y)**2 + (pts[:,1]-x)**2
            used = used | (d2 <= (min_dist**2))

    # Fallback: ensure >=1 seed per connected component
    if LV_FORCE_ONE_SEED_PER_CC:
        lbl = sk_label(M)
        n_cc = lbl.max()
        if n_cc > 0:
            have = set()
            for (yy,xx) in kept:
                have.add(lbl[yy,xx])
            for comp_id in range(1, n_cc+1):
                if comp_id in have:
                    continue
                ys, xs = np.where(lbl == comp_id)
                if len(ys) == 0:
                    continue
                y0, x0, y1, x1 = ys.min(), xs.min(), ys.max()+1, xs.max()+1
                dt_sub = cv2.distanceTransform((M[y0:y1, x0:x1].astype(np.uint8))*255, cv2.DIST_L2, 3)
                my, mx = np.unravel_index(np.argmax(dt_sub), dt_sub.shape)
                kept.append((int(my+y0), int(mx+x0)))

    return kept


def geodesic_voronoi_partition(mask_u8: np.ndarray,
                               seeds_xy: List[Tuple[int,int]]) -> np.ndarray:
    """
    Returns labeled regions (int32) partitioning the mask into one cell per seed.
    """
    M = (mask_u8 > 0)
    if M.sum() == 0:
        return np.zeros_like(mask_u8, dtype=np.int32)

    markers = np.zeros(M.shape, dtype=np.int32)
    for i, (y, x) in enumerate(seeds_xy, start=1):
        if 0 <= y < M.shape[0] and 0 <= x < M.shape[1] and M[y, x]:
            markers[y, x] = i

    # distance inside mask; negative DT promotes boundaries between centers
    dist = edt(M)
    labels = watershed(-dist, markers=markers, mask=M, watershed_line=LV_WATERSHED_LINE)
    return labels.astype(np.int32)


def partition_probability(probability, scales, k_factor=0.736, min_area=1):
    mask = build_mask_dynamic(probability.astype(np.float32), k_factor)
    seeds = detect_log_seeds(probability.astype(np.float32), mask, scales)
    labels = geodesic_voronoi_partition(mask, seeds)
    areas = np.bincount(labels.ravel())
    for label in np.where(areas < min_area)[0]:
        if label:
            labels[labels == label] = 0
    labels = relabel_sequential(labels)[0].astype(np.int32)
    return (labels > 0).astype(np.uint8) * 255, labels, len(seeds)
