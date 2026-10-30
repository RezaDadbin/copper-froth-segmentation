"""Metrics (numpy/cv2 only) for every model output against the LabelMe annotations.
Routes: semantic models -> cc (connected components of prob>=0.5), ws, lv (post-processing outputs);
automatic models -> amg. Writes $FROTH_WORK/metrics.csv."""

import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from paper.paths import DATA
from paper.paths import RUN as RR

SEM = ["dlv3", "dlv3p", "sam_ft", "hq_ft", "medsam_ft", "medsam_ft_fix", "sam_zs", "robust_zs"]
AMG = [
    "amg_sam_zs",
    "amg_sam_ft",
    "amg_sam_ft_raw",
    "amg_hq_ft",
    "amg_robust_zs",
    "amg_sam_h_zs",
    "amg_hq_h_zs",
    "amg_sam2_zs",
    "cellpose_zs",
]
MIN_INST = 10


def gt_maps(jp, H, W):
    d = json.load(open(jp))
    u = np.zeros((H, W), np.uint8)
    inst = np.zeros((H, W), np.int32)
    k = 0
    for s in d.get("shapes", []):
        if s.get("label") != "froth":
            continue
        pts = np.array(s["points"], np.float64)
        k += 1
        if s.get("shape_type") == "circle" and len(pts) == 2:
            c = tuple(np.int32(pts[0]))
            r = int(round(np.hypot(*(pts[1] - pts[0]))))
            cv2.circle(u, c, r, 1, -1)
            cv2.circle(inst, c, r, k, -1)
        else:
            p = pts.astype(np.int32)
            cv2.fillPoly(u, [p], 1)
            cv2.fillPoly(inst, [p], k)
    return u.astype(bool), inst


def relabel(l):
    u, inv = np.unique(l, return_inverse=True)
    out = inv.reshape(l.shape)
    return out if u[0] == 0 else out + 1


def boundary(l):
    b = np.zeros(l.shape, bool)
    d1 = l[:-1] != l[1:]
    d2 = l[:, :-1] != l[:, 1:]
    b[:-1] |= d1
    b[1:] |= d1
    b[:, :-1] |= d2
    b[:, 1:] |= d2
    return b


def bf(pl, gl, tol=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    bp, bg = boundary(pl), boundary(gl)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    P = (bp & dg).sum() / max(1, bp.sum())
    R = (bg & dp).sum() / max(1, bg.sum())
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def sem(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    N = pred.size
    P = tp + fn
    return dict(
        dice=2 * tp / max(1, 2 * tp + fp + fn),
        iou=tp / max(1, tp + fp + fn),
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, P),
        fg_gt=P / N,
        fg_pred=(tp + fp) / N,
        ref_dice_allfg=2 * P / (N + P),
    )


def ks(a, b):
    if not len(a) or not len(b):
        return np.nan
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(
        np.max(
            np.abs(
                np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)
            )
        )
    )


def inst(pl, gl):
    pl = pl.astype(np.int64).copy()
    a = np.bincount(pl.ravel())
    small = np.where(a < MIN_INST)[0]
    small = small[small > 0]
    if len(small):
        pl[np.isin(pl, small)] = 0
    pl = relabel(pl)
    gl = gl.astype(np.int64)
    ng, npd = int(gl.max()), int(pl.max())
    ag = np.bincount(gl.ravel(), minlength=ng + 1)
    ap = np.bincount(pl.ravel(), minlength=npd + 1)
    both = (gl > 0) & (pl > 0)
    pair, cnt = np.unique(gl[both] * (npd + 1) + pl[both], return_counts=True)
    g, p = pair // (npd + 1), pair % (npd + 1)
    iou = cnt / (ag[g] + ap[p] - cnt)
    m = iou > 0.5
    tp = int(m.sum())
    fp = npd - tp
    fn = ng - tp
    dg = np.sqrt(4 * ag[1:][ag[1:] > 0] / np.pi)
    dp = np.sqrt(4 * ap[1:][ap[1:] > 0] / np.pi)
    return dict(
        n_gt=ng,
        n_pred=npd,
        inst_tp=tp,
        inst_precision=tp / max(1, npd),
        inst_recall=tp / max(1, ng),
        inst_f1=2 * tp / max(1, 2 * tp + fp + fn),
        mean_matched_iou=float(iou[m].mean()) if tp else 0.0,
        pq=float(iou[m].sum()) / max(1e-9, tp + 0.5 * fp + 0.5 * fn),
        count_rel_err=(npd - ng) / max(1, ng),
        d50_gt=float(np.median(dg)) if len(dg) else np.nan,
        d50_pred=float(np.median(dp)) if len(dp) else np.nan,
        d_ks=ks(dg, dp),
    )


def job(args):
    split, stem = args
    rows = []
    jp = DATA / split / (stem + ".json")
    gtcache = {}

    def G(H, W):
        if (H, W) not in gtcache:
            gtcache[(H, W)] = gt_maps(jp, H, W)
        return gtcache[(H, W)]

    for m in SEM + AMG:
        f = RR / m / split / (stem + ".png")
        if not f.exists():
            continue
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        H, W = a.shape[:2]
        g, gi = G(H, W)
        base = dict(split=split, image=stem, model=m)
        if m in AMG:
            l = a.astype(np.int32)
            r = sem(l > 0, g)
            r.update(inst(l, gi))
            r.update(zip(("bf_p", "bf_r", "bf_f"), bf(l, gi), strict=False))
            rows.append({**base, "route": "amg", **r})
            continue
        pred = a >= 128
        s = sem(pred, g)
        routes = {"cc": cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)[1]}
        for rt in ("ws", "lv"):
            f2 = RR / f"{m}__{rt}" / split / (stem + ".png")
            if f2.exists():
                routes[rt] = cv2.imread(str(f2), cv2.IMREAD_UNCHANGED).astype(np.int32)
        for rt, l in routes.items():
            r = dict(s) if rt == "cc" else sem(l > 0, g)
            r.update(inst(l, gi))
            r.update(zip(("bf_p", "bf_r", "bf_f"), bf(l, gi), strict=False))
            rows.append({**base, "route": rt, **r})
    return rows


def main():
    tasks = sorted(
        {
            (p.parent.name, p.stem)
            for p in RR.glob("*/*/*.png")
            if p.parent.name in ("test1", "train1")
        }
    )
    with Pool(max(2, os.cpu_count() - 2)) as pool:
        rows = [r for rs in pool.imap_unordered(job, tasks) for r in rs]
    df = (
        pd.DataFrame(rows).sort_values(["split", "image", "model", "route"]).reset_index(drop=True)
    )  # deterministic order for the bootstrap
    df.to_csv(RR.parent / "metrics.csv", index=False)
    print(len(tasks), "images,", len(df), "rows ->", RR.parent / "metrics.csv")


if __name__ == "__main__":
    main()
