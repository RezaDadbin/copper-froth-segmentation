"""Version 2 of the output figures: one visual language (yellow = model's bubble borders, cyan = annotation), no empty panels.
Main: model_outputs.pdf (colour images with U-Net/SBSNet), postprocessing.pdf (3x3). Supplement: one page per condition,
plus a page with the saved U-Net/SBSNet maps."""

import json
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.figures.common import FIG, RR, SUP, TAGS, gt, load_img, pred, saved_map, split_of, stem
from paper.paths import SELECTION

df = pd.read_csv(RR.parent / "metrics.csv")
plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
    }
)
YEL, CYA = (255, 225, 0), (0, 240, 255)


def M(n, m, r, c):
    x = df[(df.image == stem(n)) & (df.model == m) & (df.route == r)][c]
    return float(x.iloc[0]) if len(x) else np.nan


def borders(lbl, th=2):
    l = lbl.astype(np.int32)
    b = np.zeros(l.shape, bool)
    d1 = l[:-1] != l[1:]
    d2 = l[:, :-1] != l[:, 1:]
    b[:-1] |= d1
    b[1:] |= d1
    b[:, :-1] |= d2
    b[:, 1:] |= d2
    return cv2.dilate(b.astype(np.uint8), np.ones((th, th), np.uint8)).astype(bool)


def draw(img, lbl, color=YEL):
    o = (img.astype(np.float32) * 0.9).astype(np.uint8)
    o[borders(lbl)] = color
    return o


def annot(img, polys):
    o = (img.astype(np.float32) * 0.9).astype(np.uint8)
    cv2.polylines(o, polys, True, CYA, 2, cv2.LINE_AA)
    return o


def label_of(model, route, n):
    if route == "amg":
        return pred(model, n).astype(np.int32)
    if route == "cc":
        return cv2.connectedComponents((pred(model, n) >= 128).astype(np.uint8), connectivity=8)[1]
    return cv2.imread(
        str(RR / f"{model}__{route}" / split_of(n) / (stem(n) + ".png")), cv2.IMREAD_UNCHANGED
    ).astype(np.int32)


def tile(ax, im, title=None, note=None):
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    if title:
        ax.set_title(title, fontsize=6.6, pad=2)
    if note:
        ax.text(
            0.03,
            0.04,
            note,
            transform=ax.transAxes,
            fontsize=5.6,
            color="white",
            bbox=dict(facecolor="black", alpha=0.62, pad=1.0, lw=0),
        )


def crop(img, size=560):
    H, W = img.shape[:2]
    s = min(size, H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    return (lambda a: a[y0 : y0 + s, x0 : x0 + s]), (y0, x0, s)


SEMS = [
    ("dlv3", "DeepLabV3"),
    ("dlv3p", "DeepLabV3+"),
    ("sam_ft", "SAM (ft)"),
    ("hq_ft", "HQ-SAM (ft)"),
    ("medsam_ft", "MedSAM (ft)"),
    ("robust_zs", "RobustSAM (zs)"),
]
INST = [
    ("amg_sam_zs", "amg", "SAM auto (zs)"),
    ("amg_sam_ft", "amg", "SAM auto (ft)"),
    ("amg_hq_ft", "amg", "HQ-SAM grid (ft)"),
    ("amg_robust_zs", "amg", "RobustSAM grid"),
    ("dlv3p", "ws", "DeepLabV3+ + watershed"),
    ("dlv3p", "lv", "DeepLabV3+ + LoG-Voronoi"),
]


def note_sem(n, m):
    return (
        f"Dice {100 * M(n, m, 'cc', 'dice'):.1f} | ref {100 * M(n, m, 'cc', 'ref_dice_allfg'):.1f}"
    )


def note_inst(n, m, r):
    return f"F1 {M(n, m, r, 'inst_f1'):.2f} | count {100 * M(n, m, r, 'count_rel_err'):+.0f}%"


def main_figure():
    rows = ["B50 (4).tiff", "B50 (8).tiff", "B50 (10).tiff"]
    cols = [
        "Annotation",
        "U-Net$^{\\dagger}$",
        "SBSNet$^{\\dagger}$",
        "DeepLabV3+",
        "HQ-SAM (ft)",
        "MedSAM (ft)",
        "SAM auto (zs)",
        "DLV3+ + LoG-V",
    ]
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(7.2, 2.95))
    for i, n in enumerate(rows):
        img = load_img(n)
        H, W = img.shape[:2]
        g, polys = gt(n, H, W)
        C, (y0, x0, s) = crop(img)
        a = axs[i]
        tile(a[0], C(annot(img, polys)), cols[0] if i == 0 else None, f"{len(polys)} bubbles")
        a[0].set_ylabel(f"Image {i + 1}", fontsize=6.6)
        for j, mdl in enumerate(("unet", "sbsnet")):
            mp = saved_map(mdl, n)
            c0 = (W - 1080) // 2
            tile(
                a[1 + j],
                np.dstack([mp[y0 : y0 + s, x0 - c0 : x0 - c0 + s]] * 3),
                cols[1 + j] if i == 0 else None,
            )
        for j, m in enumerate(("dlv3p", "hq_ft", "medsam_ft")):
            tile(
                a[3 + j],
                C(draw(img, label_of(m, "cc", n))),
                cols[3 + j] if i == 0 else None,
                note_sem(n, m),
            )
        tile(
            a[6],
            C(draw(img, label_of("amg_sam_zs", "amg", n))),
            cols[6] if i == 0 else None,
            f"F1 {M(n, 'amg_sam_zs', 'amg', 'inst_f1'):.2f}",
        )
        tile(
            a[7],
            C(draw(img, label_of("dlv3p", "lv", n))),
            cols[7] if i == 0 else None,
            f"F1 {M(n, 'dlv3p', 'lv', 'inst_f1'):.2f}",
        )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04)
    fig.savefig(FIG / "model_outputs.pdf", dpi=250)
    fig.savefig(FIG / "model_outputs.png", dpi=180)
    plt.close(fig)


def post_figure(n="B50 (4).tiff", m="dlv3p"):
    z = np.load(RR / "stages" / split_of(n) / f"{m}__{stem(n)}.npz", allow_pickle=True)
    img = load_img(n)
    H, W = img.shape[:2]
    _, polys = gt(n, H, W)
    C, _ = crop(img, 520)
    sd = (img * 0.9).astype(np.uint8)
    for yy, xx in z["lv_seeds"]:
        cv2.circle(sd, (int(xx), int(yy)), 7, (255, 50, 50), -1)
    P = [
        (C(img), "(a) Image"),
        (C(np.dstack([z["prob"]] * 3)), "(b) Froth probability (DeepLabV3+)"),
        (C(np.dstack([z["ws_init"]] * 3)), f"(c) Adaptive threshold, t = {float(z['ws_thr']):.2f}"),
        (C(z["ws_dist"].astype(np.float32)), "(d) L1 distance within each region"),
        (C(np.dstack([z["ws_seeds"]] * 3)), "(e) Seeds: distance > $\\mu-\\sigma$"),
        (C(draw(img, z["ws_lbl"])), "(f) ROI watershed"),
        (C(sd), "(g) LoG seeds"),
        (C(draw(img, z["lv_lbl"])), "(h) Geodesic Voronoi"),
        (C(annot(img, polys)), "(i) Annotation"),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(6.2, 6.5))
    for ax, (im, t) in zip(axs.ravel(), P, strict=False):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t, fontsize=7, loc="left", pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01, wspace=0.04, hspace=0.12)
    fig.savefig(FIG / "postprocessing.pdf", dpi=250)
    fig.savefig(FIG / "postprocessing.png", dpi=170)
    plt.close(fig)


def supp_page(names, title, out):
    ncol = 6
    fig, axs = plt.subplots(2 * len(names), ncol, figsize=(7.4, 2 * len(names) * 1.36 + 0.5))
    for i, n in enumerate(names):
        img = load_img(n)
        H, W = img.shape[:2]
        _, polys = gt(n, H, W)
        C, _ = crop(img)
        tg = TAGS.set_index("image_name").loc[n]
        dom = "colour" if tg.domain_yellow_copper else "greyscale"
        r1, r2 = axs[2 * i], axs[2 * i + 1]
        tile(
            r1[0],
            C(annot(img, polys)),
            f"{stem(n)}, {dom}, {'eval.' if tg.split == 'test' else 'train'}",
            f"annotation: {len(polys)} bubbles",
        )
        for j, (m, lab) in enumerate(SEMS[:5]):
            tile(r1[1 + j], C(draw(img, label_of(m, "cc", n))), lab, note_sem(n, m))
        tile(
            r2[0],
            C(draw(img, label_of("robust_zs", "cc", n))),
            SEMS[5][1],
            note_sem(n, "robust_zs"),
        )
        for j, (m, r, lab) in enumerate(INST[:5]):
            tile(r2[1 + j], C(draw(img, label_of(m, r, n))), lab, note_inst(n, m, r))
    fig.suptitle(title, fontsize=9, y=0.997)
    fig.text(
        0.5,
        0.004,
        "Yellow: borders predicted by each model; cyan: annotation. Semantic models: region borders of the predicted froth mask (no borders = "
        'everything predicted as froth). "ref" = Dice of labelling every pixel as froth. Central 560 px crop; scores are for the whole image.',
        ha="center",
        fontsize=5.6,
        wrap=True,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.03, wspace=0.03, hspace=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def supp_saved():
    names = [f"B50 ({k}).tiff" for k in range(1, 11)]
    for part, chunk in enumerate((names[:5], names[5:]), 1):
        fig, axs = plt.subplots(len(chunk), 3, figsize=(6.6, len(chunk) * 2.2 + 0.4))
        for i, n in enumerate(chunk):
            img = load_img(n)
            H, W = img.shape[:2]
            _, polys = gt(n, H, W)
            c0 = (W - 1080) // 2
            tile(
                axs[i, 0],
                annot(img, polys)[:, c0 : c0 + 1080],
                "Image and annotation" if i == 0 else None,
            )
            axs[i, 0].set_ylabel(f"Image {5 * (part - 1) + i + 1}", fontsize=6.6)
            for j, mdl in enumerate(("unet", "sbsnet")):
                tile(
                    axs[i, 1 + j],
                    np.dstack([saved_map(mdl, n)] * 3),
                    ("U-Net output (bright = bubble)", "SBSNet output (dark = boundary)")[j]
                    if i == 0
                    else None,
                )
        fig.suptitle(
            f"S{12 + part - 1}. Saved U-Net and SBSNet outputs for ten colour evaluation images, part {part} of 2",
            fontsize=9,
            y=0.995,
        )
        fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.06)
        fig.savefig(SUP / f"S{12 + part - 1:02d}_saved_unet_sbsnet_{part}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["main", "post", "supp", "saved"]
    if "main" in what:
        main_figure()
        print("main")
    if "post" in what:
        post_figure()
        print("post")
    if "supp" in what:
        sel = json.load(open(SELECTION))
        titles = {
            "structured_light_reflection": "Regular light reflections",
            "uneven_illumination": "Uneven illumination",
            "surface_artifacts_contaminants": "Surface artifacts and contaminants",
            "polydispersity_multiscale_bubbles": "Polydisperse (multiscale) bubbles",
            "dense_small_bubble_region": "Dense small-bubble regions",
            "elongated_irregular_bubbles": "Elongated or irregular bubbles",
            "poor_focus_defocus_blur": "Defocus blur",
            "localized_motion_or_vibration_blur": "Localized motion or vibration blur",
            "collapsed": "Collapse",
            "at_most_2_conditions": "Images with at most two conditions",
            "at_least_6_conditions": "Images with six or more conditions",
        }
        for f in SUP.glob("S*.png"):
            f.unlink()
        for i, (k, t) in enumerate(titles.items(), 1):
            supp_page(sel[k], f"S{i}. {t}", SUP / f"S{i:02d}_{k}.png")
            print("page", i)
        oz = sel["output_zip"]
        for j in range(0, len(oz), 3):
            supp_page(
                oz[j : j + 3],
                f"S{12 + j // 3}. Further examples ({j // 3 + 1})",
                SUP / f"S{12 + j // 3:02d}_examples_{j // 3 + 1}.png",
            )
            print("ex", j)
    if "saved" in what:
        supp_saved()
        print("saved")
