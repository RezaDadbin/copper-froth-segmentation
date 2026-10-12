"""Historical source export; maintained entry points live in scripts/.

Original: codes/froth_speed/froth_speed/.ipynb_checkpoints/froth_speed-checkpoint.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (markdown)
    # # Froth Motion and Stability Analysis for Two Consecutive Images
    # 
    # This code estimates the motion and stability of froth in a flotation process using **two consecutive images**.
    # 
    # ## Overview
    # Given two grayscale images representing the froth at times `t1` and `t2`, the algorithm computes:
    # 
    # 1. **Local motion vectors** for small regions (blocks) of the froth.
    # 2. **Tile-wise stability metrics** to assess uniformity and coherence of motion.
    # 
    # ## Methodology
    # 
    # 1. **Block Matching**
    #    - The first image is divided into blocks of size `block_size` (e.g., 64 pixels).  
    #    - For each block, the most similar block in the second image is found within a specified `search_range` using **sum of absolute differences (SAD)**.  
    #    - The displacement `(dx, dy)` for each block represents the local motion vector.
    # 
    # 2. **Motion Vector Aggregation**
    #    - All local motion vectors are collected.  
    #    - Average horizontal and vertical velocities (`avg_vx`, `avg_vy`) and overall average speed are computed.  
    #    - Motion vectors can be visualized by drawing arrows over the first image.
    # 
    # 3. **Tile-wise Stability Analysis**
    #    - The image is divided into larger tiles (`tile_size`, e.g., 128 pixels).  
    #    - For each tile, the motion vectors within that tile are used to compute stability metrics:
    #      - **Motion variance**: spread of displacement magnitudes (high = unstable).  
    #      - **Smoothness ratio**: mean / standard deviation of displacement (high = smooth, uniform motion).  
    #      - **Directional stability**: inverse of variance of vector directions (high = coherent flow).  
    #    - A heatmap is generated to visualize stability across the froth surface.
    # 
    # 4. **Outputs**
    #    - **Motion vectors image**: arrows showing direction and magnitude of froth motion.  
    #    - **CSV of vectors**: `(x, y, dx, dy)` for each block.  
    #    - **Tile stability heatmap**: colored visualization of directional stability.  
    #    - **CSV of tile stability**: per-tile metrics for motion variance, smoothness, and directional stability.
    # 
    # ## Interpretation
    # - **Stable froth**: low motion variance, high smoothness ratio, high directional stability.  
    # - **Unstable froth**: high motion variance, low smoothness ratio, low directional stability.  
    # - Arrows point in the direction of froth flow, with length representing speed.

    # %% Original cell 1 (code)
    # Multiscale phase-correlation registration with visualization
    import cv2, numpy as np, matplotlib.pyplot as plt
    from pathlib import Path

    p1 = "./mnt/data/images/image1.tiff"
    p2 = "./mnt/data/images/image2.tiff"
    img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
    assert img1 is not None and img2 is not None

    h, w = img1.shape
    print(f"Image size: {w} x {h}")

    # Preprocess: mild blur + equalize
    def preprocess(im):
        im = cv2.GaussianBlur(im, (5,5), 0)
        im = cv2.equalizeHist(im)
        return im

    pimg1 = preprocess(img1)
    pimg2 = preprocess(img2)

    # Build pyramid (level 0 = original, level 1 = pyrDown(original), ...)
    def build_pyramid(im, levels):
        pyr = [im]
        for i in range(1, levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    levels = 4  # try 4 levels (down to ~1/8 in each dim)
    pyr1 = build_pyramid(pimg1, levels)
    pyr2 = build_pyramid(pimg2, levels)

    # Multiscale phase correlation
    current_shift = np.array([0.0, 0.0])  # in original-image pixels
    responses = []

    for L in reversed(range(levels)):  # from smallest (coarsest) to largest (finest)
        im1_l = pyr1[L].astype(np.float32)
        im2_l = pyr2[L].astype(np.float32)
        scale_factor = 1.0 / (2**L)
        # apply current shift (converted to level coords) to im1_l to reduce residual
        shift_level = current_shift * scale_factor
        # warp im1_l by shift_level (use cv2.warpAffine)
        M = np.array([[1.0, 0.0, shift_level[0]], [0.0, 1.0, shift_level[1]]], dtype=np.float32)
        im1_w = cv2.warpAffine(im1_l, M, (im1_l.shape[1], im1_l.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # create Hanning window to reduce edge effects
        win_y = np.hanning(im1_w.shape[0])
        win_x = np.hanning(im1_w.shape[1])
        window = np.outer(win_y, win_x).astype(np.float32)
        # phase correlate
        shift_res, resp = cv2.phaseCorrelate(im1_w, im2_l, window)
        shift_res = np.array(shift_res)  # (dx, dy) at this level
        # update current_shift in original-pixel units
        current_shift += shift_res * (2**L)
        responses.append((L, float(resp), float(shift_res[0]), float(shift_res[1])))
        print(f"Level {L}: shift_res(level) = ({shift_res[0]:.4f}, {shift_res[1]:.4f}), resp={resp:.4f}, current_shift(orig)={current_shift[0]:.4f},{current_shift[1]:.4f}")

    # Final shift (current_shift) is in original-image pixels
    final_dx, final_dy = current_shift[0], current_shift[1]
    print("\nFINAL estimated shift (pixels, image1 -> image2): dx = {:.4f}, dy = {:.4f}".format(final_dx, final_dy))

    # Warp original image1 by final shift to align with image2
    M_final = np.array([[1.0, 0.0, final_dx], [0.0, 1.0, final_dy]], dtype=np.float32)
    img1_warp = cv2.warpAffine(img1.astype(np.float32), M_final, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img1_warp_u8 = np.clip(img1_warp, 0, 255).astype(np.uint8)

    # Compute residual image and stats
    residual = cv2.absdiff(img2, img1_warp_u8)
    mean_resid = float(np.mean(residual))
    median_resid = float(np.median(residual))
    max_resid = int(np.max(residual))
    print(f"Residual stats after alignment: mean={mean_resid:.2f}, median={median_resid:.2f}, max={max_resid}")

    # Save visualizations
    out_dir = Path("./mnt/data/phase-correlation")
    os.makedirs(out_dir, exist_ok=True)
    aligned_path = out_dir / "phasecor_aligned.png"
    residual_path = out_dir / "phasecor_residual.png"
    side_path = out_dir / "phasecor_side_by_side.png"

    # create overlay with arrow showing shift on image2
    vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    start = (w//2, h//8)
    end = (int(start[0] + final_dx*2.0), int(start[1] + final_dy*2.0))  # scale for visibility
    cv2.arrowedLine(vis, start, end, (255,0,0), 3, tipLength=0.3)
    cv2.putText(vis, f"dx={final_dx:.2f}, dy={final_dy:.2f}", (start[0]-120, start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # save aligned and residual images
    cv2.imwrite(str(aligned_path), img1_warp_u8)
    # residual as heatmap
    res_col = cv2.applyColorMap(np.clip((residual*2).astype(np.uint8),0,255), cv2.COLORMAP_JET)
    cv2.imwrite(str(residual_path), res_col)
    # side by side: original2, aligned1, residual
    side = np.concatenate([cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img1_warp_u8, cv2.COLOR_GRAY2BGR), res_col], axis=1)
    cv2.imwrite(str(side_path), side)

    # Display inline
    plt.figure(figsize=(14,6))
    plt.imshow(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Left: image2 (target) | Middle: image1 warped by estimated shift | Right: residual heatmap")
    plt.show()

    # Print outputs and file paths
    print("\nOutputs saved:")
    print(" - Aligned image1:", aligned_path)
    print(" - Residual heatmap:", residual_path)
    print(" - Side-by-side:", side_path)

    # Provide numeric results in a dictionary for user
    result = {
        "dx_pixels": float(final_dx),
        "dy_pixels": float(final_dy),
        "mean_residual": mean_resid,
        "median_residual": median_resid,
        "max_residual": max_resid,
        "responses": responses,
        "aligned_path": str(aligned_path),
        "residual_path": str(residual_path),
        "side_path": str(side_path)
    }
    result

    # %% Original cell 2 (code)

    # %% Original cell 3 (code)
    # Notebook-ready tuned froth motion fusion + tile stability heatmap
    # Paste into one Jupyter cell and run.
    import cv2
    import numpy as np
    import math, random, csv, os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd


    # ---------- Set your paths here ----------
    IMG1_PATH = "./mnt/data/images/F61 (19).tiff"   # change if running locally
    IMG2_PATH = "./mnt/data/images/F61 (20).tiff"
    OUT_DIR = "./mnt/data/froth-motion-fusion"
    os.makedirs(OUT_DIR, exist_ok=True)
    # -----------------------------------------

    # ---------- Tuned parameters (adjust if needed) ----------
    TILE_SIZE = 64
    TILE_STEP = 32
    OF_MAG_THRESHOLD = 0.2
    PC_RESP_THRESHOLD = 0.02
    BM_CORR_THRESHOLD = 0.3
    GAUSS_K = (7,7)
    ARROW_SCALE = 8.0
    RANSAC_ITERS = 2500
    RANSAC_THRESH = 12.0
    PYR_LEVELS = 4
    FARB_NEIGH = 9
    # --------------------------------------------------------

    def calculate_stability_from_block_matching(vx_list, vy_list):
        vx_array = np.array(vx_list)
        vy_array = np.array(vy_list)
        if vx_array.size == 0:
            return {"motion_variance": float('nan'), "smoothness_ratio": float('nan'), "directional_stability": float('nan')}
        motion_variance = np.var(vx_array) + np.var(vy_array)
        displacement = np.sqrt(vx_array**2 + vy_array**2)
        smoothness = np.mean(displacement) / (np.std(displacement) + 1e-5)
        angles = np.arctan2(vy_array, vx_array)
        direction_variance = np.var(angles)
        directional_stability = 1.0 / (direction_variance + 1e-5)
        return {"motion_variance": motion_variance, "smoothness_ratio": smoothness, "directional_stability": directional_stability}

    def preprocess(im):
        im = cv2.GaussianBlur(im, (5,5), 0)
        im = cv2.equalizeHist(im)
        return im.astype(np.float32)

    def multiscale_phase_correlation(a, b, levels=4):
        pyr_a = [a]; pyr_b = [b]
        for i in range(1, levels):
            pyr_a.append(cv2.pyrDown(pyr_a[-1])); pyr_b.append(cv2.pyrDown(pyr_b[-1]))
        current_shift = np.array([0.0, 0.0])
        for L in reversed(range(levels)):
            A = pyr_a[L]; B = pyr_b[L]
            scale = 1.0 / (2**L)
            shift_level = current_shift * scale
            M = np.array([[1.0, 0.0, shift_level[0]], [0.0, 1.0, shift_level[1]]], dtype=np.float32)
            Aw = cv2.warpAffine(A, M, (A.shape[1], A.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            win_y = np.hanning(Aw.shape[0]); win_x = np.hanning(Aw.shape[1])
            window = np.outer(win_y, win_x).astype(np.float32)
            try:
                shift_res, resp = cv2.phaseCorrelate(Aw, B, window)
            except:
                shift_res = (0.0, 0.0)
            shift_res = np.array(shift_res)
            current_shift += shift_res * (2**L)
        return current_shift

    # create out dir
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load images
    img1 = cv2.imread(IMG1_PATH, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(IMG2_PATH, cv2.IMREAD_GRAYSCALE)
    assert img1 is not None and img2 is not None, "Failed to load images — check IMG1_PATH/IMG2_PATH"
    h, w = img1.shape

    # preprocess
    p1 = preprocess(img1); p2 = preprocess(img2)

    # global multiscale phase-correlation
    global_shift = multiscale_phase_correlation(p1, p2, levels=PYR_LEVELS)
    print(f"Global phase-correlation shift (px): dx={global_shift[0]:.3f}, dy={global_shift[1]:.3f}")

    # dense Farneback (tuned)
    flow = cv2.calcOpticalFlowFarneback(p1.astype(np.uint8), p2.astype(np.uint8), None,
                                        pyr_scale=0.5, levels=3, winsize=FARB_NEIGH, iterations=7,
                                        poly_n=7, poly_sigma=1.5, flags=0)
    u = flow[...,0]; v = flow[...,1]

    # local tiles: phase-correlation + normalized cross-correlation block matching
    pc_vectors = []; bm_vectors = []
    for y in range(0, h - TILE_SIZE + 1, TILE_STEP):
        for x in range(0, w - TILE_SIZE + 1, TILE_STEP):
            a = p1[y:y+TILE_SIZE, x:x+TILE_SIZE]; b = p2[y:y+TILE_SIZE, x:x+TILE_SIZE]
            win_y = np.hanning(a.shape[0]); win_x = np.hanning(a.shape[1]); window = np.outer(win_y, win_x).astype(np.float32)
            try:
                sres, resp = cv2.phaseCorrelate(a, b, window)
            except:
                continue
            dx_pc, dy_pc = float(sres[0]), float(sres[1])
            cx, cy = x + TILE_SIZE//2, y + TILE_SIZE//2
            pc_vectors.append((cx, cy, dx_pc, dy_pc, float(resp)))
            # block-match (normed cross-correlation)
            sr = int(TILE_SIZE * 0.6)
            x0 = max(0, x - sr); x1 = min(w, x + TILE_SIZE + sr)
            y0 = max(0, y - sr); y1 = min(h, y + TILE_SIZE + sr)
            window_big = p2[y0:y1, x0:x1].astype(np.uint8)
            block = p1[y:y+TILE_SIZE, x:x+TILE_SIZE].astype(np.uint8)
            if block.std() < 8:
                continue
            res = cv2.matchTemplate(window_big, block, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            best_x = x0 + max_loc[0]; best_y = y0 + max_loc[1]
            dx_bm = best_x - x; dy_bm = best_y - y
            score = float(max_val)
            bm_vectors.append((cx, cy, dx_bm, dy_bm, score))

    # aggregate per center
    center_to_data = {}
    for cy in range(TILE_SIZE//2, h - TILE_SIZE//2 + 1, TILE_STEP):
        for cx in range(TILE_SIZE//2, w - TILE_SIZE//2 + 1, TILE_STEP):
            center_to_data[(cx,cy)] = {"of": None, "pc": None, "bm": None}

    for (cx,cy) in list(center_to_data.keys()):
        fx = int(np.clip(cx, 0, w-1)); fy = int(np.clip(cy, 0, h-1))
        dx_of = float(u[fy, fx]); dy_of = float(v[fy, fx])
        center_to_data[(cx,cy)]["of"] = (dx_of, dy_of, math.hypot(dx_of, dy_of))

    for cx,cy,dx,dy,resp in pc_vectors:
        key = (int(cx), int(cy))
        if key in center_to_data:
            center_to_data[key]["pc"] = (dx, dy, resp)
    for cx,cy,dx,dy,score in bm_vectors:
        key = (int(cx), int(cy))
        if key in center_to_data:
            center_to_data[key]["bm"] = (dx, dy, score)

    # fuse with weights (tuned)
    fused_pts = []
    for (cx,cy), d in center_to_data.items():
        of = d["of"]; pc = d["pc"]; bm = d["bm"]
        vecs = []
        if of is not None and of[2] >= OF_MAG_THRESHOLD:
            w_of = (of[2] + 0.1) * 0.8
            vecs.append((of[0], of[1], w_of))
        if pc is not None and pc[2] >= PC_RESP_THRESHOLD:
            w_pc = max(pc[2], 0.001) * 1.4
            vecs.append((pc[0], pc[1], w_pc))
        if bm is not None and bm[2] >= BM_CORR_THRESHOLD:
            w_bm = max(0.001, bm[2]) * 1.8
            vecs.append((bm[0], bm[1], w_bm))
        if len(vecs) == 0: continue
        ws = np.array([v[2] for v in vecs], dtype=np.float32)
        vals_dx = np.array([v[0] for v in vecs], dtype=np.float32)
        vals_dy = np.array([v[1] for v in vecs], dtype=np.float32)
        fused_dx = float((vals_dx * ws).sum() / (ws.sum() + 1e-8))
        fused_dy = float((vals_dy * ws).sum() / (ws.sum() + 1e-8))
        conf = float(np.mean(ws))
        fused_pts.append((cx, cy, fused_dx, fused_dy, conf))

    # smoothing
    dx_map = np.full((h, w), np.nan, dtype=np.float32); dy_map = np.full((h, w), np.nan, dtype=np.float32)
    for cx,cy,dx,dy,conf in fused_pts:
        dx_map[cy, cx] = dx; dy_map[cy, cx] = dy
    dx_map_f = np.copy(dx_map); dy_map_f = np.copy(dy_map)
    nan_mask = np.isnan(dx_map_f)
    dx_map_f[nan_mask] = 0.0; dy_map_f[nan_mask] = 0.0
    dx_s = cv2.GaussianBlur(dx_map_f, GAUSS_K, 0); dy_s = cv2.GaussianBlur(dy_map_f, GAUSS_K, 0)
    smoothed_vectors = []
    for cx,cy,dx,dy,conf in fused_pts:
        sdx = float(dx_s[cy, cx]); sdy = float(dy_s[cy, cx])
        smoothed_vectors.append((cx, cy, sdx, sdy, conf))

    # save CSV
    csv_path = out_dir / "fused_vectors_tuned.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["x","y","dx_px","dy_px","conf"])
        for cx,cy,dx,dy,conf in smoothed_vectors:
            writer.writerow([int(cx), int(cy), float(dx), float(dy), float(conf)])
    # print("Saved fused vectors CSV:", csv_path)

    # RANSAC translation-only
    best_tx = best_ty = 0.0; best_mask = None; best_in = 0
    if len(fused_pts) >= 3:
        src = np.array([[p[0], p[1]] for p in fused_pts], dtype=np.float32)
        dst = np.array([[p[0]+p[2], p[1]+p[3]] for p in fused_pts], dtype=np.float32)
        N = len(src)
        for _ in range(RANSAC_ITERS):
            i = random.randrange(N)
            tx_try = dst[i,0] - src[i,0]; ty_try = dst[i,1] - src[i,1]
            preds = src + np.array([tx_try, ty_try], dtype=np.float32)
            dists = np.linalg.norm(dst - preds, axis=1); mask = dists < RANSAC_THRESH
            cnt = int(mask.sum())
            if cnt > best_in:
                best_in = cnt; best_tx = tx_try; best_ty = ty_try; best_mask = mask.copy()
        inlier_src = src[best_mask]; inlier_dst = dst[best_mask]; in_disp = inlier_dst - inlier_src
        mean_dx = float(np.mean(in_disp[:,0])); mean_dy = float(np.mean(in_disp[:,1]))
        print(f"RANSAC translation: tx={best_tx:.3f}, ty={best_ty:.3f}, inliers={best_in}/{N}")
    else:
        print("Not enough fused points for RANSAC")

    # high-visibility arrow overlay
    vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for cx,cy,dx,dy,conf in smoothed_vectors:
        start=(int(cx),int(cy)); end=(int(cx+dx*ARROW_SCALE), int(cy+dy*ARROW_SCALE))
        mag=math.hypot(dx,dy); ang=math.atan2(dy,dx); hue=int(((ang+math.pi)/(2*math.pi))*179)
        val=min(255, int(60 + mag*12))
        color_hsv = np.uint8([[[hue,200,val]]]); color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        cv2.arrowedLine(vis, start, end, color_bgr, 2, tipLength=0.35)
    mid=(w//2, h//10); glob_end=(int(mid[0]+best_tx*ARROW_SCALE), int(mid[1]+best_ty*ARROW_SCALE))
    cv2.arrowedLine(vis, mid, glob_end, (255,0,0), 3, tipLength=0.35)
    vis_path = out_dir / "fused_vectors_tuned_visible.png"
    cv2.imwrite(str(vis_path), vis)
    # print("Saved visible arrow overlay:", vis_path)

    # quiver
    plt.figure(figsize=(10,8)); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    qx=[p[0] for p in smoothed_vectors]; qy=[p[1] for p in smoothed_vectors]
    qu=[p[2]*ARROW_SCALE for p in smoothed_vectors]; qv=[p[3]*ARROW_SCALE for p in smoothed_vectors]
    plt.quiver(qx, qy, qu, qv, angles='xy', scale_units='xy', scale=1, width=0.005)
    plt.gca().invert_yaxis(); plt.axis('off')
    qpath = out_dir / "fused_vectors_tuned_quiver.png"; plt.savefig(str(qpath), bbox_inches='tight', dpi=200); plt.close()
    # print("Saved quiver:", qpath)

    # tile-wise stability heatmap (directional_stability)
    tiles = []; stability_vals=[]
    for y in range(0, h - TILE_SIZE + 1, TILE_STEP):
        for x in range(0, w - TILE_SIZE + 1, TILE_STEP):
            vx_list=[]; vy_list=[]
            for sx,sy,sdx,sdy,conf in smoothed_vectors:
                if (x <= sx < x + TILE_SIZE) and (y <= sy < y + TILE_SIZE):
                    vx_list.append(sdx); vy_list.append(sdy)
            st = calculate_stability_from_block_matching(vx_list, vy_list)
            tiles.append(((x + TILE_SIZE//2, y + TILE_SIZE//2), st))
            stability_vals.append(st["directional_stability"] if not np.isnan(st["directional_stability"]) else 0.0)

    arr_ds = np.array(stability_vals); arr_ds = np.nan_to_num(arr_ds)
    lo = np.percentile(arr_ds, 5) if arr_ds.size>0 else 0.0; hi = np.percentile(arr_ds,95) if arr_ds.size>0 else 1.0
    if hi - lo < 1e-6: hi = lo + 1.0
    norm_ds = np.clip((arr_ds - lo) / (hi - lo), 0.0, 1.0)

    rows = (h - TILE_SIZE)//TILE_STEP + 1; cols = (w - TILE_SIZE)//TILE_STEP + 1
    grid = np.zeros((rows, cols), dtype=np.float32)
    idx = 0
    for ry in range(rows):
        for rx in range(cols):
            grid[ry, rx] = norm_ds[idx] if idx < len(norm_ds) else 0.0
            idx += 1
    heat_up = cv2.resize(grid, (w,h), interpolation=cv2.INTER_CUBIC)
    heat_up = cv2.GaussianBlur(heat_up, (31,31), 0)
    heat_u8 = (np.clip(heat_up,0,1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), 0.6, heat_color, 0.4, 0)
    heat_path = out_dir / "tile_stability_heatmap_tuned.png"; overlay_path = out_dir / "tile_stability_overlay_tuned.png"
    cv2.imwrite(str(heat_path), heat_color); cv2.imwrite(str(overlay_path), overlay)
    # print("Saved heatmap and overlay:", heat_path, overlay_path)

    # combined overlay with arrows
    combo = cv2.addWeighted(vis, 0.7, heat_color, 0.3, 0)
    combo_path = out_dir / "fused_vectors_tuned_with_stability.png"
    cv2.imwrite(str(combo_path), combo)
    # print("Saved combined overlay:", combo_path)

    # save tile csv
    tile_csv = out_dir / "tile_stability_values_tuned.csv"
    with open(tile_csv, "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["tile_cx","tile_cy","directional_stability","motion_variance","smoothness_ratio"])
        for ((cx,cy), st) in tiles:
            wcsv.writerow([int(cx), int(cy),
                           float(st["directional_stability"] if not np.isnan(st["directional_stability"]) else 0.0),
                           float(st["motion_variance"] if not np.isnan(st["motion_variance"]) else 0.0),
                           float(st["smoothness_ratio"] if not np.isnan(st["smoothness_ratio"]) else 0.0)])
    # print("Saved tile CSV:", tile_csv)

    # display final overlay inline
    # plt.figure(figsize=(12,8)); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.title("Tuned stability overlay"); plt.show()


    df = pd.read_csv("./mnt/data/froth-motion-fusion/tile_stability_values_tuned.csv")
    avg_metrics = df[["motion_variance", "smoothness_ratio", "directional_stability"]].mean()
    print("Average stability metrics:")
    print(avg_metrics)

    # %% Original cell 4 (markdown)
    # If the motion is mostly global and large (many tens of px), increase RANSAC_THRESH to 15–40 px.
    # 
    # If you prefer more local detail (finer vectors), reduce TILE_SIZE to 64 and TILE_STEP to 32 — but expect more noise (smooth more strongly).
    # 
    # If you want to favor phase-correlation strongly, raise PC_RESP_THRESHOLD. If block-matching is noisy, lower BM_NORM_THRESHOLD to accept only very good matches.

    # %% Original cell 5 (markdown)
    # vx_real = dx_pixels * spatial_scale / temporal_scale
    # vy_real = dy_pixels * spatial_scale / temporal_scale
    # speed_real = sqrt(vx_real**2 + vy_real**2)

    # %% Original cell 6 (code)
    import math

    # Phase correlation shift
    dx, dy = global_shift[0], global_shift[1]  # in pixels (or units)

    # Time interval between frames in seconds
    dt = 1.5  # example: 30 FPS => 1/30 sec

    # Calculate displacement magnitude
    displacement = math.sqrt(dx**2 + dy**2)

    # Speed in units per second
    speed = displacement / dt

    # Optional: separate x and y components
    speed_x = dx / dt
    speed_y = dy / dt

    print(f"Displacement magnitude: {displacement:.3f} units")
    print(f"Speed magnitude: {speed:.3f} units/sec")
    print(f"Speed components: x = {speed_x:.3f}, y = {speed_y:.3f} units/sec")

    # %% Original cell 7 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
