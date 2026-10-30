"""Post-processing outputs only (no metrics). For every semantic prediction writes the ROI-watershed and
LoG-Voronoi instance maps (uint16 PNG) and, for the figure images, every intermediate stage (npz)."""

import json
from multiprocessing import Pool

import cv2
import numpy as np

from paper.inference.kernels import (
    OUT,
    SEMANTIC,
    log_voronoi_staged,
    roi_watershed_staged,
    scale_priors,
)

DBG = {"B50 (10)", "S36 (48)", "F61 (9)", "B50 (22)", "S40 (24)", "B50 (52)", "B50 (4)"}


def job(a):
    model, split, f, scales = a
    stem = f.stem
    prob = cv2.imread(str(f), 0).astype(np.float32) / 255.0
    keep = stem in DBG
    ws, st_ws = roi_watershed_staged(prob, keep_stages=keep)
    lv, st_lv = log_voronoi_staged(prob, scales, keep_stages=keep)
    for rt, lbl in (("ws", ws), ("lv", lv)):
        d = OUT / f"{model}__{rt}" / split
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / (stem + ".png")), lbl.astype(np.uint16))
    if keep:
        d = OUT / "stages" / split
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / f"{model}__{stem}.npz",
            prob=(prob * 255).astype(np.uint8),
            ws_init=st_ws["init"],
            ws_dist=st_ws["dist"].astype(np.float16),
            ws_seeds=st_ws["seeds"],
            ws_final=st_ws["final"],
            ws_lbl=ws.astype(np.uint16),
            lv_mask=st_lv["mask"],
            lv_seeds=st_lv["seeds"],
            lv_lbl=lv.astype(np.uint16),
            ws_thr=st_ws["threshold"],
        )
    return model, split, stem


def main():
    tasks = []
    for m in SEMANTIC:
        if not (OUT / m / "test1").exists():
            continue
        sp = OUT / "scales" / f"{m}.json"
        if sp.exists():
            sc = json.load(open(sp))
        else:  # size priors from the model's own predictions (no labels), as in the original runner
            sc = scale_priors(
                [
                    cv2.imread(str(f), 0).astype(np.float32) / 255
                    for f in sorted((OUT / m / "test1").glob("*.png"))
                ]
            )
            sp.parent.mkdir(parents=True, exist_ok=True)
            json.dump(sc, open(sp, "w"))
        for split in ("test1", "train1"):
            for f in sorted((OUT / m / split).glob("*.png")):
                if not (OUT / f"{m}__lv" / split / f.name).exists():
                    tasks.append((m, split, f, sc))
    print(len(tasks), "post-processing jobs", flush=True)
    with Pool(12) as p:
        for i, r in enumerate(p.imap_unordered(job, tasks)):
            if i % 25 == 0:
                print(i + 1, r, flush=True)
    print("POST_DONE", flush=True)


if __name__ == "__main__":
    main()
