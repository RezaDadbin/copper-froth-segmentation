"""Explicit two-frame displacement estimators derived from the motion notebooks."""

from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.ndimage import median_filter


def validate_frames(first, second):
    if first is None or second is None or first.shape != second.shape or first.ndim != 2:
        raise ValueError("Motion needs two readable grayscale frames of equal dimensions")


def preprocess(image):
    return cv2.equalizeHist(cv2.GaussianBlur(image, (5, 5), 0)).astype(np.float32)


def phase_correlation(first, second, levels=4):
    """Coarse-to-fine phase correlation; returns (dx, dy) in native pixels."""
    validate_frames(first, second)
    a, b = [preprocess(first)], [preprocess(second)]
    for _ in range(1, levels):
        if min(a[-1].shape) < 16:
            break
        a.append(cv2.pyrDown(a[-1])); b.append(cv2.pyrDown(b[-1]))
    shift = np.zeros(2)
    for level in reversed(range(len(a))):
        local = shift / 2 ** level
        matrix = np.array([[1, 0, local[0]], [0, 1, local[1]]], np.float32)
        warped = cv2.warpAffine(a[level], matrix, (a[level].shape[1], a[level].shape[0]), borderMode=cv2.BORDER_REFLECT)
        window = np.outer(np.hanning(warped.shape[0]), np.hanning(warped.shape[1])).astype(np.float32)
        residual, _ = cv2.phaseCorrelate(warped.copy(), b[level].copy(), window)
        shift += np.asarray(residual) * 2 ** level
    return shift


def optical_flow(first, second):
    """Farneback plus median filtering from optical flow.ipynb; units px/frame."""
    validate_frames(first, second)
    flow = cv2.calcOpticalFlowFarneback(first, second, None, pyr_scale=0.5, levels=3,
                                     winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow[..., 0] = median_filter(flow[..., 0], size=5)
    flow[..., 1] = median_filter(flow[..., 1], size=5)
    return flow


def block_matching(first, second, block_size=64, search_range=16, texture_threshold=8, method="ncc"):
    """NCC from the fusion source or corrected signed-SAD from block matching.ipynb.

    Rows contain center x/y, dx/dy in pixels, and match confidence. SAD is slower
    because it preserves the notebook's exhaustive search.
    """
    validate_frames(first, second)
    if block_size < 2 or search_range < 0 or block_size > min(first.shape):
        raise ValueError("Invalid motion block/search size")
    height, width = first.shape
    a, b = preprocess(first), preprocess(second)
    rows = []
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            block = a[y:y+block_size, x:x+block_size]
            if block.std() < texture_threshold:
                continue
            x0, x1 = max(0, x-search_range), min(width, x+block_size+search_range)
            y0, y1 = max(0, y-search_range), min(height, y+block_size+search_range)
            if method == "sad":
                best, bx, by = float("inf"), x, y
                for yy in range(y0, y1-block_size+1):
                    for xx in range(x0, x1-block_size+1):
                        # Both operands are float32: no unsigned subtraction wrap.
                        score = float(np.abs(block - b[yy:yy+block_size, xx:xx+block_size]).sum())
                        if score < best:
                            best, bx, by = score, xx, yy
                confidence = 1 / (1 + best / block.size)
            else:
                response = cv2.matchTemplate(b[y0:y1, x0:x1], block, cv2.TM_CCORR_NORMED)
                _, confidence, _, location = cv2.minMaxLoc(response)
                bx, by = x0+location[0], y0+location[1]
            rows.append((x+block_size//2, y+block_size//2, bx-x, by-y, confidence))
    return np.asarray(rows, np.float32).reshape(-1, 5)


def motion_fusion(first, second, tile_size=64, step=32):
    """Notebook OF/phase/NCC confidence fusion with normalized sparse smoothing."""
    validate_frames(first, second)
    if tile_size < 4 or tile_size > min(first.shape) or step < 1:
        raise ValueError("Invalid fusion tile/step size")
    a, b = preprocess(first), preprocess(second)
    flow = cv2.calcOpticalFlowFarneback(a.astype(np.uint8), b.astype(np.uint8), None,
                                     pyr_scale=0.5, levels=3, winsize=9, iterations=7,
                                     poly_n=7, poly_sigma=1.5, flags=0)
    height, width = first.shape
    rows = []
    for y in range(0, height-tile_size+1, step):
        for x in range(0, width-tile_size+1, step):
            cx, cy = x+tile_size//2, y+tile_size//2
            vectors = []
            dx, dy = flow[cy, cx]
            if math.hypot(dx, dy) >= 0.2:
                vectors.append((dx, dy, 1.0))
            window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
            shift, quality = cv2.phaseCorrelate(a[y:y+tile_size, x:x+tile_size].copy(),
                                              b[y:y+tile_size, x:x+tile_size].copy(), window)
            if quality >= 0.02:
                vectors.append((*shift, max(quality, 0.001)*1.4))
            radius = int(tile_size*0.6)
            x0, x1 = max(0, x-radius), min(width, x+tile_size+radius)
            y0, y1 = max(0, y-radius), min(height, y+tile_size+radius)
            block = a[y:y+tile_size, x:x+tile_size]
            if block.std() >= 8:
                response = cv2.matchTemplate(b[y0:y1, x0:x1], block, cv2.TM_CCORR_NORMED)
                _, quality, _, where = cv2.minMaxLoc(response)
                if quality >= 0.3:
                    vectors.append((x0+where[0]-x, y0+where[1]-y, quality*1.8))
            if vectors:
                values = np.asarray(vectors)
                dx, dy = np.average(values[:, :2], axis=0, weights=values[:, 2])
                rows.append((cx, cy, dx, dy, float(values[:, 2].mean())))
    result = np.asarray(rows, np.float32).reshape(-1, 5)
    # Sparse fields must be normalized by their support; Gaussian-blurring
    # inserted zeros alone attenuated the historical displacement magnitudes.
    support = np.zeros(first.shape, np.float32)
    u, v = support.copy(), support.copy()
    for x, y, dx, dy, _ in result:
        support[int(y), int(x)] = 1
        u[int(y), int(x)], v[int(y), int(x)] = dx, dy
    normalizer = cv2.GaussianBlur(support, (7, 7), 0)
    u = cv2.GaussianBlur(u, (7, 7), 0) / np.maximum(normalizer, 1e-8)
    v = cv2.GaussianBlur(v, (7, 7), 0) / np.maximum(normalizer, 1e-8)
    for row in result:
        row[2:4] = u[int(row[1]), int(row[0])], v[int(row[1]), int(row[0])]
    return result


def stability(vectors):
    if not len(vectors):
        return {key: None for key in ("motion_variance", "smoothness_ratio", "directional_coherence")}
    u, v = vectors[:, 2], vectors[:, 3]
    magnitude = np.hypot(u, v)
    moving = magnitude > 1e-6
    coherence = float(abs(np.exp(1j*np.arctan2(v[moving], u[moving])).mean())) if moving.any() else None
    return {"motion_variance": float(np.var(u)+np.var(v)),
            "smoothness_ratio": float(magnitude.mean()/(magnitude.std()+1e-5)),
            "directional_coherence": coherence}
