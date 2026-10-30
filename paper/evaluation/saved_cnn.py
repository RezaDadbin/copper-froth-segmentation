"""Instance-level scores of the saved U-Net and SBSNet maps (ten colour evaluation images, central 1080 x 1080 crop),
raw and after both post-processing routes, next to the other models on exactly the same crops."""

import cv2
import numpy as np
import pandas as pd

from paper.evaluation import metrics as M
from paper.figures.common import saved_map
from paper.inference import kernels as P
from paper.paths import DATA, TABLES, WORK
from paper.paths import RUN as O

NAMES = [f"B50 ({k})" for k in range(1, 11)]
OTHER = [
    ("dlv3p", "cc"),
    ("dlv3p", "ws"),
    ("dlv3p", "lv"),
    ("amg_sam_zs", "amg"),
    ("amg_hq_h_zs", "amg"),
    ("amg_sam_ft", "amg"),
    ("cellpose_zs", "amg"),
]


def crop(a, c0):
    return a[:, c0 : c0 + 1080]


def score(lbl, g, gi):
    r = M.sem(lbl > 0, g)
    r.update(M.inst(lbl.astype(np.int32), gi))
    r["bf_f"] = M.bf(lbl, gi)[2]
    return r


rows = []
for n in NAMES:
    img = cv2.imread(str(O / "dlv3p/test1" / f"{n}.png"), 0)
    H, W = img.shape
    c0 = (W - 1080) // 2
    g, gi = M.gt_maps(DATA / "test1" / f"{n}.json", H, W)
    g, gi = crop(g, c0), crop(gi, c0)
    # gt instances cut by the crop are kept; fragments < 10 px are ignored by M.inst on the prediction side only
    gi = M.relabel(gi)
    for mdl in ("unet", "sbsnet"):
        m = saved_map(mdl, n + ".tiff").astype(np.float32) / 255.0
        prob = m if mdl == "unet" else m  # both maps: bright = bubble interior, dark = boundary
        t = cv2.threshold((prob * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
            1
        ]
        rows.append(
            dict(
                image=n,
                model=mdl,
                route="cc",
                **score(cv2.connectedComponents(t, connectivity=4)[1], g, gi),
            )
        )
        ws, _ = P.roi_watershed_staged(prob)
        rows.append(dict(image=n, model=mdl, route="ws", **score(ws, g, gi)))
        sc = P.scale_priors([prob])
        lv, _ = P.log_voronoi_staged(prob, sc)
        rows.append(dict(image=n, model=mdl, route="lv", **score(lv, g, gi)))
    for m, rt in OTHER:
        f = O / (m if rt in ("cc", "amg") else f"{m}__{rt}") / "test1" / f"{n}.png"
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        lbl = (
            cv2.connectedComponents((a >= 128).astype(np.uint8), connectivity=8)[1]
            if rt == "cc"
            else a.astype(np.int32)
        )
        rows.append(dict(image=n, model=m, route=rt, **score(M.relabel(crop(lbl, c0)), g, gi)))
d = pd.DataFrame(rows)
d.to_csv(WORK / "cnn_saved_eval.csv", index=False)
print(
    d.groupby(["model", "route"], sort=False)[
        ["dice", "bf_f", "inst_f1", "pq", "count_rel_err", "d_ks"]
    ]
    .mean()
    .round(3)
    .to_string()
)

ROWS = [
    ("unet", "cc", "U-Net, connected components"),
    ("unet", "ws", "U-Net + ROI watershed"),
    ("unet", "lv", "U-Net + LoG-Voronoi"),
    ("sbsnet", "cc", "SBSNet, connected components"),
    ("sbsnet", "ws", "SBSNet + ROI watershed"),
    ("sbsnet", "lv", "SBSNet + LoG-Voronoi"),
    ("dlv3p", "cc", "DeepLabV3+, connected components"),
    ("dlv3p", "ws", "DeepLabV3+ + ROI watershed"),
    ("dlv3p", "lv", "DeepLabV3+ + LoG-Voronoi"),
    ("amg_sam_ft", "amg", "SAM (ft), automatic"),
    ("amg_sam_zs", "amg", "SAM ViT-B (zs), automatic"),
    ("amg_hq_h_zs", "amg", "HQ-SAM ViT-H (zs), automatic"),
    ("cellpose_zs", "amg", "Cellpose-SAM (zs)"),
]
fmt = lambda v, f: f.format(v).replace("-", "$-$")
out = []
for i, (m, r, lab) in enumerate(ROWS):
    x = d[(d.model == m) & (d.route == r)]
    if i in (6, 9):
        out.append("\\addlinespace")
    out.append(
        f"{lab} & {fmt(100 * x.dice.mean(), '{:.1f}')} & {fmt(x.inst_f1.mean(), '{:.2f}')} & {fmt(x.pq.mean(), '{:.2f}')} & "
        f"{fmt(100 * x.count_rel_err.mean(), '{:+.0f}')} & {fmt(x.d_ks.mean(), '{:.2f}')}\\\\"
    )
TABLES.mkdir(parents=True, exist_ok=True)
(TABLES / "saved_cnn.tex").write_text("\n".join(out) + "\n")
