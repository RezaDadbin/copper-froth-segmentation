"""ROI watershed: splits a froth probability map into individual bubbles."""

import cv2
import numpy as np
from .thresholds import dynamic_threshold

K_FACTOR = 0.736
MIN_FROTH_AREA = 7
MAX_FROTH_AREA = 1193
MAX_SEED_RATIO = 0.7
MIN_SEED_RATIO = 0.1
SEED_K_FACTOR = -1
THRESH = None
MERGED_PEAKS_MIN = 2
MORPH_KERNEL = (3, 3)
MORPH_OPEN_ITERS = 1
MORPH_CLOSE_ITERS = 1
MIN_AREA = 2
DEBUG_VISUALIZE = False
SAVE_DEBUG = False


def simple_threshold_mask(norm_img, thresh=THRESH):
    """Return binary uint8 mask (0/255) from normalized image."""
    return (norm_img > thresh).astype(np.uint8) * 255


def find_seed_components(roi_mask):
    """
    Compute distance transform and find seeds using a robust STATISTICAL threshold.
    Returns seeds_uint8 (0/255) and number_of_seed_components.
    """
    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L1, 3)
    if dist.max() <= 0:
        return np.zeros_like(roi_mask), 0, dist

    internal_distances = dist[roi_mask == 255]
    if len(internal_distances) < 1:
        return np.zeros_like(roi_mask), 0, dist

    mu = np.mean(internal_distances)
    sigma = np.std(internal_distances)
    thresh_val = mu + SEED_K_FACTOR * sigma
    
    _, seed_mask = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    seed_mask = seed_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, _ = cv2.connectedComponents(seed_mask)
    return seed_mask, n_labels, dist


def local_watershed_on_roi(roi_img_norm, roi_mask):
    """
    roi_img_norm: normalized gray ROI (0..1) float32
    roi_mask: binary uint8 (0/255) same shape
    returns: separated_mask uint8 (0/255), markers_int or None
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERS)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERS)
    if cleaned.max() == 0:
        return np.zeros_like(cleaned), None

    seed_mask, n_seeds, dist = find_seed_components(cleaned)

    if n_seeds <= 1:
        return cleaned, None

    ret, markers = cv2.connectedComponents(seed_mask)
    markers = markers.astype(np.int32)
    markers = markers + 1   
    unknown = cv2.subtract(cleaned, (seed_mask > 0).astype(np.uint8)*255)
    markers[unknown == 255] = 0

    roi_for_watershed = (roi_img_norm * 255).astype(np.uint8)
    roi_color = cv2.cvtColor(roi_for_watershed, cv2.COLOR_GRAY2BGR)

    try:
        cv2.watershed(roi_color, markers)
    except Exception as e:
        print("Watershed failed on ROI:", e)
        return cleaned, None

    separated_mask = (markers > 1).astype(np.uint8) * 255
    return separated_mask, markers


def filter_small_components(binary_mask_uint8, min_area=MIN_AREA):
    """Remove connected components smaller than min_area. Return cleaned mask and count removed."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)
    out = np.zeros_like(binary_mask_uint8)
    removed = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area >= min_area:
            out[labels == i] = 255
        else:
            removed += 1
    return out, removed


def process_image_roi_watershed(img_path, visualize=DEBUG_VISUALIZE, save_debug=SAVE_DEBUG, k_factor=0.736):
    """
    img_path: path to grayscale segmentation map (values 0..255 or normalized)
    returns: final_binary_mask (0/255), count_after (int), count_before (int), debug_info
    """
    raw = (np.rint(np.clip(img_path, 0, 1) * 255).astype(np.uint8)
           if isinstance(img_path, np.ndarray) else cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE))
    if raw is None:
        raise FileNotFoundError(img_path)
    img = raw.astype(np.float32) / 255.0

    k = k_factor
    mean_val = np.mean(raw)           
    std_val = np.std(raw)             
    dynamic_thresh_255 = mean_val - k * std_val  # 0..255
    dynamic_thresh_norm = dynamic_threshold(img, k)
    print(f"Dynamic threshold (normalized): {dynamic_thresh_norm:.4f}")


    init_mask = simple_threshold_mask(img, dynamic_thresh_norm)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(init_mask, connectivity=8)
    count_before = n_labels - 1

    final_mask = np.zeros_like(init_mask)

    debug_rois = []

    for lab in range(1, n_labels):
        x, y, w, h, area = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP]), \
                           int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT]), \
                           int(stats[lab, cv2.CC_STAT_AREA])

        pad = 2
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(init_mask.shape[1], x + w + pad); y1 = min(init_mask.shape[0], y + h + pad)

        roi_mask = init_mask[y0:y1, x0:x1].copy()
        roi_img = img[y0:y1, x0:x1].copy()

        main_blob_mask = (labels[y0:y1, x0:x1] == lab).astype(np.uint8) * 255

        roi_mask = cv2.bitwise_and(roi_mask, main_blob_mask)


        roi_mask = init_mask[y0:y1, x0:x1].copy()
        roi_img = img[y0:y1, x0:x1].copy()
        main_blob_mask = (labels[y0:y1, x0:x1] == lab).astype(np.uint8) * 255
        roi_mask = cv2.bitwise_and(roi_mask, main_blob_mask)

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(roi_mask, contours, -1, (255), thickness=cv2.FILLED)

        seed_mask, n_seeds, dist = find_seed_components(roi_mask)

        run_watershed = (n_seeds >= MERGED_PEAKS_MIN)

        if run_watershed:
            separated_mask, markers = local_watershed_on_roi(roi_img, roi_mask)
            if separated_mask is None:
                final_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_mask[y0:y1, x0:x1], roi_mask)
            else:
                separated_mask_filtered, removed_small = filter_small_components(separated_mask, MIN_AREA)
                final_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_mask[y0:y1, x0:x1], separated_mask_filtered)

                if visualize or save_debug:
                    debug_rois.append({
                        'bbox': (x0, y0, x1, y1),
                        'roi_img': roi_img,
                        'roi_mask': roi_mask,
                        'dist': dist,
                        'seed_mask': seed_mask,
                        'separated_mask': separated_mask,
                        'separated_filtered': separated_mask_filtered,
                        'markers': markers
                    })
        else:
            final_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_mask[y0:y1, x0:x1], roi_mask)
            if visualize or save_debug:
                debug_rois.append({
                    'bbox': (x0, y0, x1, y1),
                    'roi_img': roi_img,
                    'roi_mask': roi_mask,
                    'dist': dist,
                    'seed_mask': seed_mask,
                    'separated_mask': None,
                    'separated_filtered': None,
                    'markers': None
                })

    final_mask_clean, removed_total = filter_small_components(final_mask, MIN_AREA)

    n_final, labels_final = cv2.connectedComponents(final_mask_clean, connectivity=8)[:2]
    count_after = n_final - 1

    debug_info = {'count_before': count_before, 'count_after': count_after,
                  'removed_small_objects': removed_total}
    return final_mask_clean, count_after, count_before, debug_info

