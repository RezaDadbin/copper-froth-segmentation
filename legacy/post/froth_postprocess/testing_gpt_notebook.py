"""Historical source export; maintained entry points live in scripts/.

Original: codes/POST/froth-postprocess/testing-gpt.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    # ===========================
    # Copper Froth Post-Processing: LoG Centers → Geodesic Voronoi (Area-Preserving)
    # Keeps your dynamic threshold (mean - K*std) and computes the undercount % metric.
    # ===========================

    # If needed (first time), uncomment and run:
    # !pip install numpy opencv-python scikit-image scipy matplotlib

    import os
    import json
    import math
    from typing import Dict, Tuple, List

    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    from skimage.feature import blob_log
    from skimage.segmentation import watershed
    from skimage.measure import label as sk_label
    from scipy.ndimage import distance_transform_edt as edt

    # -----------------------------
    # USER SETTINGS
    # -----------------------------
    IMAGE_FOLDER = './output_images_viridis'  # folder with your model's output images
    LABEL_FOLDER = './ground_truth_jsons'                  # folder with LabelMe JSONs matching image basenames
    K_FACTOR = 0.736                           # your dynamic threshold factor (mean - K*std)
    # K_FACTOR = 1
    GT_LABEL_NAME_FILTER = None                # e.g., "froth" to count only that label, else None
    MIN_FROTH_AREA = 1                        # min area (in px^2) to keep any region after splitting

    # Visualization / debug
    VISUALIZE = True                           # show plots for each processed image
    SAVE_DEBUG = False                         # also save debug images
    DEBUG_SAVE_DIR = "./debug_log_voronoi"     # where to save if SAVE_DEBUG=True

    # LoG + Voronoi hyperparameters
    LV_LOG_NUM_SCALES = 8            # number of sigma samples for blob_log
    LV_LOG_THRESHOLD = 0.02          # LoG response threshold (on normalized img [0,1])
    LV_MIN_DISTANCE_FACTOR = 0.50    # seed NMS spacing = factor * r50
    LV_EDGE_SEED_DT_MIN_FACTOR = 0.30# drop seeds too close to mask edge (< factor * r10)
    LV_FORCE_ONE_SEED_PER_CC = True  # ensure >=1 seed per connected component
    LV_WATERSHED_LINE = False        # assign boundary pixels to regions (no area loss)

    # -----------------------------
    # UTILITIES
    # -----------------------------
    def ensure_dir(p: str):
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    def imread_gray01(path: str) -> np.ndarray:
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(path)
        return raw.astype(np.float32) / 255.0

    # -----------------------------
    # SCALE PRIORS (p10, p50, p90) USING YOUR DYNAMIC THRESHOLD
    # -----------------------------
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

    # -----------------------------
    # MASK: YOUR dynamic threshold (mean - K*std) on normalized img
    # -----------------------------
    def build_mask_dynamic(img01: np.ndarray, k_factor: float) -> np.ndarray:
        raw_like = (img01 * 255.0).astype(np.float32)
        mean_val = float(np.mean(raw_like))
        std_val  = float(np.std(raw_like))
        dynamic_thresh_255 = mean_val - k_factor * std_val
        dynamic_thresh_norm = dynamic_thresh_255 / 255.0
        mask_u8 = ((img01 > dynamic_thresh_norm).astype(np.uint8) * 255)
        return mask_u8

    # -----------------------------
    # SEEDING: Multi-scale LoG + spacing + edge filter + per-CC fallback
    # -----------------------------
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

    # -----------------------------
    # GEODESIC VORONOI: watershed on -distance (masked)
    # -----------------------------
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

    # -----------------------------
    # METRIC HELPERS (LabelMe JSON)
    # -----------------------------
    def find_label_json_for_image(img_path, label_folder=LABEL_FOLDER):
        base = os.path.splitext(os.path.basename(img_path))[0]
        for p in (os.path.join(label_folder, base + '.json'),
                  os.path.join(os.path.dirname(img_path), base + '.json')):
            if os.path.exists(p):
                return p
        return None

    def count_gt_froths(label_json_path, label_filter=GT_LABEL_NAME_FILTER):
        with open(label_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])
        if label_filter is None:
            return len(shapes)
        lf = str(label_filter).strip().lower()
        return sum(1 for s in shapes if str(s.get('label','')).strip().lower() == lf)

    def undercount_percent(gt_count, pred_count):
        if gt_count is None:
            return None
        if gt_count == 0:
            return 0.0 if pred_count == 0 else -100.0
        return ((gt_count - pred_count) / gt_count) * 100.0

    # -----------------------------
    # MAIN PER-IMAGE PROCESSOR
    # -----------------------------
    def process_image_log_voronoi(img_path: str,
                                  scales: Dict[str, float],
                                  visualize: bool = False,
                                  save_debug: bool = False):
        """
        Returns:
          final_mask_u8, pred_count, seed_count, debug dict
        """
        img01 = imread_gray01(img_path)

        # 1) mask via YOUR dynamic threshold (mean - K*std)
        mask_u8 = build_mask_dynamic(img01, K_FACTOR)

        # 2) detect seeds via LoG (multi-scale)
        seeds = detect_log_seeds(img01, mask_u8, scales)

        # 3) geodesic Voronoi partition (area-preserving split of the mask)
        labels = geodesic_voronoi_partition(mask_u8, seeds)

        # 4) remove tiny regions per your MIN_FROTH_AREA, then relabel
        lab = labels.copy()
        max_lab = int(lab.max())
        if max_lab > 0 and MIN_FROTH_AREA is not None and MIN_FROTH_AREA > 0:
            areas = np.bincount(lab.ravel(), minlength=max_lab+1)
            tiny = np.where(areas < MIN_FROTH_AREA)[0]
            for t in tiny:
                if t == 0:
                    continue
                lab[lab == t] = 0
            if lab.max() > 0:
                lab = sk_label(lab > 0, connectivity=2)

        final_mask = ((lab > 0).astype(np.uint8) * 255)
        pred_count = int(lab.max())
        seed_count = len(seeds)

        # Visualization
        if visualize:
            fig, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(img01, cmap='gray'); ax[0].set_title('Input'); ax[0].axis('off')
            ax[1].imshow(mask_u8, cmap='gray'); ax[1].set_title(f'Mask (K={K_FACTOR:.3f})'); ax[1].axis('off')
            ax[2].imshow(lab, cmap='nipy_spectral'); ax[2].set_title(f'Voronoi labels (count={pred_count})'); ax[2].axis('off')
            for (y,x) in seeds:
                ax[1].plot([x],[y],'o')
                ax[2].plot([x],[y],'o')
            plt.tight_layout(); plt.show()

        # Optional saving
        if save_debug:
            ensure_dir(DEBUG_SAVE_DIR)
            base = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_mask.png"), mask_u8)
            # Colorize labels for saving
            if pred_count > 0:
                # random color map for labels
                rng = np.random.default_rng(123)
                colors = rng.integers(0, 255, size=(lab.max()+1, 3), dtype=np.uint8)
                colors[0] = (0,0,0)
                color_lab = colors[lab]
                cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_labels.png"), cv2.cvtColor(color_lab, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_final_mask.png"), final_mask)

        dbg = {'seeds': seeds, 'seed_count': seed_count}
        return final_mask, pred_count, seed_count, dbg

    # -----------------------------
    # RUN
    # -----------------------------
    if __name__ == "__main__":
        # 0) Compute dataset priors for scales using YOUR dynamic threshold
        print("Computing dataset scale priors from:", IMAGE_FOLDER)
        scales = compute_percentiles_for_scales(IMAGE_FOLDER, k_factor=K_FACTOR)
        print("Scale priors:",
              {k: (round(v,2) if isinstance(v,(int,float)) else v) for k,v in scales.items()})

        # 1) Collect images
        image_list = [os.path.join(IMAGE_FOLDER, f) for f in sorted(os.listdir(IMAGE_FOLDER))
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        print(f"Found {len(image_list)} images.")

        if not image_list:
            raise SystemExit(f"No images in {IMAGE_FOLDER}")

        # 2) Evaluate
        per_image_metrics = []
        for im_path in image_list:
            print("\nProcessing:", os.path.basename(im_path))
            try:
                final_mask, pred_cnt, seed_cnt, dbg = process_image_log_voronoi(
                    im_path, scales, visualize=VISUALIZE, save_debug=SAVE_DEBUG
                )
                # metric (LabelMe JSON)
                label_path = find_label_json_for_image(im_path, LABEL_FOLDER)
                if label_path is None:
                    print("⚠️  Label JSON not found; skipping metric.")
                    continue
                gt = count_gt_froths(label_path, GT_LABEL_NAME_FILTER)
                m = undercount_percent(gt, pred_cnt)
                per_image_metrics.append(m)
                print(f"GT={gt} | Pred={pred_cnt} | Seeds={seed_cnt} | Undercount % = {m:.2f}")
            except Exception as e:
                print("Failed:", e)

        # 3) Summary
        valid = [x for x in per_image_metrics if x is not None]
        if valid:
            mean_bias = float(np.mean(valid))
            mape = float(np.mean(np.abs(valid)))
            print("\n=== LoG+Voronoi EVALUATION SUMMARY ===")
            print(f"Images evaluated: {len(valid)}")
            print(f"Mean undercount % (bias): {mean_bias:.2f}")
            print(f"Mean absolute percent error (MAPE): {mape:.2f}")
        else:
            print("\nNo valid metrics computed (missing labels?).")

    # %% Original cell 1 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
