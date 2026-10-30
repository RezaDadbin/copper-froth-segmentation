"""Shared helpers for the figures: image and annotation loading, model outputs and the saved U-Net/SBSNet maps."""

import json

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.paths import DATA, FIGURES, RUN, SAVED_CNN, SUPP
from paper.paths import TAGS as TAGS_CSV

# names used by the figure scripts
FIG, RR, SAVED, SUP = FIGURES, RUN, SAVED_CNN, SUPP

TAGS = pd.read_csv(TAGS_CSV)
SEM = [
    ("dlv3", "DeepLabV3"),
    ("dlv3p", "DeepLabV3+"),
    ("sam_ft", "SAM (ft)"),
    ("hq_ft", "HQ-SAM (ft)"),
    ("medsam_ft", "MedSAM (ft)"),
    ("sam_zs", "SAM (zero-shot)"),
    ("robust_zs", "RobustSAM (zero-shot)"),
]
AMG = [
    ("amg_sam_zs", "SAM auto (zero-shot)"),
    ("amg_sam_ft", "SAM auto (ft)"),
    ("amg_hq_ft", "HQ-SAM auto (ft)"),
    ("amg_robust_zs", "RobustSAM auto"),
]
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
    }
)
FP_C, FN_C = (
    np.array([230, 60, 40]),
    np.array([40, 110, 230]),
)  # red = false froth, blue = missed froth


def split_of(name):
    return TAGS.set_index("image_name").loc[name, "split"] + "1"


def stem(name):
    return name.rsplit(".", 1)[0]


def load_img(name):
    import tifffile

    a = np.asarray(tifffile.imread(str(DATA / split_of(name) / name)))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return np.ascontiguousarray(a[:, :, :3])


def gt(name, H, W):
    d = json.load(open(DATA / split_of(name) / (stem(name) + ".json")))
    u = np.zeros((H, W), np.uint8)
    polys = []
    for s in d["shapes"]:
        if s.get("label") != "froth":
            continue
        p = np.array(s["points"], np.float64)
        if s.get("shape_type") == "circle" and len(p) == 2:
            r = np.hypot(*(p[1] - p[0]))
            t = np.linspace(0, 2 * np.pi, 40)
            p = np.stack([p[0][0] + r * np.cos(t), p[0][1] + r * np.sin(t)], 1)
        p = p.astype(np.int32)
        cv2.fillPoly(u, [p], 1)
        polys.append(p)
    return u.astype(bool), polys


def pred(model, name):
    f = RR / model / split_of(name) / (stem(name) + ".png")
    return cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.exists() else None


def err_overlay(img, p, g, a=0.55):
    o = img.astype(np.float32).copy()
    o[p & ~g] = (1 - a) * o[p & ~g] + a * FP_C
    o[~p & g] = (1 - a) * o[~p & g] + a * FN_C
    return o.clip(0, 255).astype(np.uint8)


def boundaries(lbl):
    b = np.zeros(lbl.shape, bool)
    d1 = lbl[:-1] != lbl[1:]
    d2 = lbl[:, :-1] != lbl[:, 1:]
    b[:-1] |= d1
    b[1:] |= d1
    b[:, :-1] |= d2
    b[:, 1:] |= d2
    return b


def inst_overlay(img, lbl, color=(255, 230, 0)):
    o = (img * 0.85).astype(np.uint8)
    b = boundaries(lbl.astype(np.int32))
    b = cv2.dilate(b.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    o[b] = color
    return o


def poly_overlay(img, polys, color=(0, 255, 255)):
    o = (img * 0.85).astype(np.uint8)
    cv2.polylines(o, polys, True, color, 2, cv2.LINE_AA)
    return o


def crop_box(name, img, size=540):
    H, W = img.shape[:2]
    s = min(size, H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    return y0, x0, s


def metric(df, name, model, route, col):
    r = df[(df.image == stem(name)) & (df.model == model) & (df.route == route)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def panel(ax, im, title, sub=None):
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    ax.set_title(title, pad=2)
    if sub:
        ax.text(
            0.02,
            0.03,
            sub,
            transform=ax.transAxes,
            color="white",
            fontsize=6,
            bbox=dict(facecolor="black", alpha=0.6, pad=1.2, lw=0),
        )


def saved_map(model, name):
    """Saved 256x256 U-Net/SBSNet map for B50 (1)-(10), mapped back onto the centre 1080 crop."""
    order = [f"B50 ({i})" for i in (1, 10, 2, 3, 4, 5, 6, 7, 8, 9)]
    if stem(name) not in order:
        return None
    k = order.index(stem(name))
    f = SAVED / (
        "model_output_U_Net/output_images_grey/output_%d.png" % k
        if model == "unet"
        else "model_ouptut_SBS_Net/output_images_grey/output_%d_%d.png" % (k // 2, k % 2)
    )
    m = cv2.imread(str(f), 0)
    return cv2.resize(m, (1080, 1080), interpolation=cv2.INTER_LINEAR)
