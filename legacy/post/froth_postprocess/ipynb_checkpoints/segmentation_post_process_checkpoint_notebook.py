"""Historical source export; maintained entry points live in scripts/.

Original: codes/POST/froth-postprocess/.ipynb_checkpoints/segmentation_post_process-checkpoint.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (markdown)
    # #### froth size distribution over 10 sample SBS-Net output segmentation map 
    # #### post processed initial masks using watershed algorithm
    # #### area of the recognized froths by the model

    # %% Original cell 1 (code)
    import cv2
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from skimage.measure import label, regionprops
    import matplotlib.patches as patches

    os.getcwd()

    # %% Original cell 2 (code)
    # --- USER SETTINGS ---
    # Point this to the folder with your model's output images
    IMAGE_FOLDER = '../new-dataset/part-I'
    # Use the same K-factor from your main script for consistent thresholding
    K_FACTOR = 0.736
    # --- END SETTINGS ---

    def get_initial_mask(img_path, k):
        """Reads an image and applies the initial dynamic threshold."""
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return None, 0
        mean_val = np.mean(raw)
        std_val = np.std(raw)
        dynamic_thresh_255 = mean_val - k * std_val
        _, init_mask = cv2.threshold(raw, dynamic_thresh_255, 255, cv2.THRESH_BINARY)
        return init_mask, raw.shape[0] * raw.shape[1]

    def analyze_froth_sizes():
        """Analyzes all images in a folder to find the distribution of froth sizes."""
        image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    
        if not image_files:
            print(f"Error: No images found in folder: {IMAGE_FOLDER}")
            return

        all_areas = []
        print(f"Analyzing {len(image_files)} images...")

        for img_path in image_files:
            mask, total_pixels = get_initial_mask(img_path, K_FACTOR)
            if mask is None:
                continue
        
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # stats[0] is the background, so we loop from 1 to n_labels
            if n_labels > 1:
                # We filter out froths that are extremely large (e.g., >50% of image), as they are likely bad segmentations
                areas = stats[1:, cv2.CC_STAT_AREA]
                valid_areas = [area for area in areas if area < total_pixels * 0.5]
                all_areas.extend(valid_areas)

        if not all_areas:
            print("Error: No valid froths were found. Check your K_FACTOR or image folder.")
            return

        all_areas = np.array(all_areas)
    
        # --- ANALYSIS RESULTS ---
        print("\n--- Froth Size Analysis ---")
        print(f"Found a total of {len(all_areas)} valid froths.")
    
        # Calculate percentiles to get robust estimates, ignoring extreme outliers
        p10 = np.percentile(all_areas, 10)
        p90 = np.percentile(all_areas, 90)
    
        print("\n--- Recommended Starting Values ---")
        print("Use these percentile values in your main script. They are robust to outliers.")
        print(f"10th Percentile: {int(p10)}  <-- Use this for MIN_FROTH_AREA")
        print(f"90th Percentile: {int(p90)}  <-- Use this for MAX_FROTH_AREA")
    
        # Plot a histogram for visual confirmation
        plt.figure(figsize=(10, 6))
        plt.hist(all_areas, bins=100, range=(0, np.percentile(all_areas, 99))) # Cap range for better visualization
        plt.title('Histogram of Froth Sizes')
        plt.xlabel('Froth Area (pixels)')
        plt.ylabel('Frequency')
        plt.axvline(p10, color='g', linestyle='--', label=f'10th Percentile ({int(p10)})')
        plt.axvline(p90, color='r', linestyle='--', label=f'90th Percentile ({int(p90)})')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()

    # %% Original cell 3 (code)
    from skimage.measure import label, regionprops
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def plot_froth_regions(final_clean_mask):
        """
        Detect connected froth regions in a binary mask, print their properties,
        and plot bounding boxes, centroids, and the froth area at the centroid.
        """

        # Step 1: Label connected components
        label_img = label(final_clean_mask)

        # Step 2: Extract region properties
        regions = regionprops(label_img)
        print(f"Total detected froth regions: {len(regions)}\n")

        # for i, region in enumerate(regions):
        #     print(f"Froth #{i+1}")
        #     print(f" - Area: {region.area} pixels")
        #     print(f" - Centroid: (y={region.centroid[0]:.2f}, x={region.centroid[1]:.2f})")
        #     print(f" - Bounding Box: {region.bbox}\n")

        # Step 3: Visualize
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(label_img, cmap='gray')

        for region in regions:
            # Draw bounding box (optional for reference)
            minr, minc, maxr, maxc = region.bbox
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     edgecolor='red', facecolor='none', linewidth=1.5)
            ax.add_patch(rect)

            # Place the area text exactly at the froth centroid
            cy, cx = region.centroid
            ax.text(cx, cy, str(region.area),
                    color='yellow', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        ax.set_title(f"Froth Regions: {len(regions)} (with areas labeled)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return fig, ax

    # %% Original cell 4 (code)
    # -----------------------------
    # Parameters (tweak these)
    # -----------------------------

    MIN_FROTH_AREA = 7  
    MAX_FROTH_AREA = 1193  


    MAX_SEED_RATIO = 0.7
    MIN_SEED_RATIO = 0.1

    SEED_K_FACTOR = -1

    #------------------------------

    THRESH = None                # global simple threshold on normalized segmap (0..1)
    # SEED_RATIO = 0.09         # dist transform peak threshold as fraction of dist.max() -> seeds for watershed
    MERGED_PEAKS_MIN = 2       # if number of peaks >= this: treat as merged and run watershed
    MORPH_KERNEL = (3, 3)       # kernel used for morphological open/close on ROIs
    MORPH_OPEN_ITERS = 1
    MORPH_CLOSE_ITERS = 1
    MIN_AREA = 2                # remove blobs smaller than this area (pixels)
    DEBUG_VISUALIZE = False    # show matplotlib figures for each processed image (set False in large batches)
    SAVE_DEBUG = False          
    DEBUG_SAVE_DIR = "./debug_roi_watershed"  

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p):
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

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

    # -----------------------------
    # Main processing per image
    # -----------------------------
    def process_image_roi_watershed(img_path, visualize=DEBUG_VISUALIZE, save_debug=SAVE_DEBUG ):
        """
        img_path: path to grayscale segmentation map (values 0..255 or normalized)
        returns: final_binary_mask (0/255), count_after (int), count_before (int), debug_info
        """
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(img_path)
        img = raw.astype(np.float32) / 255.0

        k = 0.736
        mean_val = np.mean(raw)           
        std_val = np.std(raw)             
        dynamic_thresh_255 = mean_val - k * std_val  # 0..255
        dynamic_thresh_norm = dynamic_thresh_255 / 255.0  # 0..1
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

        # Visualization / Debug
    
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1); plt.title(f"Input (normalized)"); plt.imshow(img, cmap='gray'); plt.axis('off')
        plt.subplot(1, 3, 2); plt.title(f"Initial Mask (th={THRESH}) count={count_before}"); plt.imshow(init_mask, cmap='gray'); plt.axis('off')
        plt.subplot(1, 3, 3); plt.title(f"Final Mask count={count_after}"); plt.imshow(final_mask_clean, cmap='gray'); plt.axis('off')
        plt.tight_layout()
        plt.show()

        fig, ax = plot_froth_regions(final_mask_clean)

    
        if visualize:
            # Show ROI-level debug panels (if any)
            for idx, r in enumerate(debug_rois):
                x0, y0, x1, y1 = r['bbox']
                plt.figure(figsize=(12, 6))

                plt.subplot(2, 4, 1); plt.title("ROI img"); plt.imshow(r['roi_img'], cmap='gray'); plt.axis('off')
                plt.subplot(2, 4, 2); plt.title("ROI initial mask"); plt.imshow(r['roi_mask'], cmap='gray'); plt.axis('off')
                plt.subplot(2, 4, 3); plt.title("Distance transform"); plt.imshow(r['dist'], cmap='magma'); plt.axis('off')
                plt.subplot(2, 4, 4); plt.title("Seed mask"); plt.imshow(r['seed_mask'], cmap='gray'); plt.axis('off')

                plt.subplot(2, 4, 5); plt.title("Separated (raw)"); 
                if r['separated_mask'] is None:
                    plt.text(0.2, 0.5, 'No watershed run', fontsize=12)
                    plt.axis('off')
                else:
                    plt.imshow(r['separated_mask'], cmap='gray'); plt.axis('off')

                plt.subplot(2, 4, 6); plt.title("Separated (filtered)");
                if r['separated_filtered'] is None:
                    plt.text(0.2, 0.5, 'No watershed run', fontsize=12)
                    plt.axis('off')
                else:
                    plt.imshow(r['separated_filtered'], cmap='gray'); plt.axis('off')

                plt.subplot(2, 4, 7); plt.title("Markers (labels)");
                if r['markers'] is None:
                    plt.text(0.2, 0.5, 'No markers', fontsize=12)
                    plt.axis('off')
                else:
                    markers_vis = r['markers'].copy()
                    markers_vis[markers_vis < 0] = 0
                    plt.imshow(markers_vis, cmap='nipy_spectral'); plt.axis('off')

                plt.subplot(2, 4, 8); plt.title("Overview in global img");
                context = np.zeros_like(final_mask_clean)
                context[y0:y1, x0:x1] = r['separated_filtered'] if r['separated_filtered'] is not None else r['roi_mask']
                plt.imshow(context, cmap='gray'); plt.axis('off')

                plt.suptitle(f"ROI #{idx+1} bbox={r['bbox']}")
                plt.tight_layout()
                plt.show()

        if save_debug:
            ensure_dir(DEBUG_SAVE_DIR)
            base = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_init_mask.png"), init_mask)
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_final_mask.png"), final_mask_clean)
            for idx, r in enumerate(debug_rois):
                x0, y0, x1, y1 = r['bbox']
                cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_img.png"), (r['roi_img']*255).astype(np.uint8))
                cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_initmask.png"), r['roi_mask'])
                if r['seed_mask'] is not None and r['seed_mask'].max() > 0:
                    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_seeds.png"), r['seed_mask'])
                if r['separated_mask'] is not None:
                    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_sep_raw.png"), r['separated_mask'])
                if r['separated_filtered'] is not None:
                    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_sep_filtered.png"), r['separated_filtered'])

        debug_info = {
            'count_before': count_before,
            'count_after': count_after,
            'removed_small_objects': removed_total,
            'num_rois_examined': len(debug_rois)
        }

        return final_mask_clean, count_after, count_before, debug_info

    # -----------------------------
    # Batch runner example
    # -----------------------------
    if __name__ == "__main__":
        folder = '../new-dataset/part-I'
        image_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    
        print("Found", len(image_list), "images.")
        analyze_froth_sizes()
        for im_path in image_list:
            print("\nProcessing:", im_path)
            try:
                final_mask, cnt_after, cnt_before, dbg = process_image_roi_watershed(im_path, visualize=DEBUG_VISUALIZE, save_debug=SAVE_DEBUG)
                print(f"Counts -> before: {cnt_before}, after: {cnt_after}, removed_small_objects: {dbg['removed_small_objects']}")
            except Exception as e:
                print("Failed:", e)

    # %% Original cell 5 (code)
    import os

    folder = 'F:/dataset/new-dataset/part-I'

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {os.path.abspath(folder)}")

    image_list = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]

    print("Total images found:", len(image_list))
    print("First few:", image_list[:5])

    # %% Original cell 6 (code)
    import os

    folder = os.path.join("..", "new-dataset", "part-I")
    print("Checking folder:", os.path.abspath(folder))

    if not os.path.exists(folder):
        print("Folder does NOT exist!")
    else:
        files = os.listdir(folder)
        print("Found", len(files), "entries in folder")
        print("First 20 entries:", files[:20])

    # %% Original cell 7 (code)

    # %% Original cell 8 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
