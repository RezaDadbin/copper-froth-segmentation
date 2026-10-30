"""Figures that need no re-run outputs: pipeline schematic, model-input preprocessing, recorded runs vs reference."""

import glob

import cv2
import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.paths import DATA, RECORDED, TAGS
from paper.paths import FIGURES as FIG

plt.rcParams.update(
    {
        "font.size": 7.5,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
    }
)
MEAN, STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])


def load(split, name):
    a = np.asarray(tifffile.imread(str(DATA / split / name)))
    return np.ascontiguousarray((np.stack([a] * 3, -1) if a.ndim == 2 else a)[:, :, :3])


def rls(img, L=1024):  # segment_anything ResizeLongestSide geometry
    h, w = img.shape[:2]
    s = L / max(h, w)
    nh, nw = int(h * s + 0.5), int(w * s + 0.5)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.zeros((L, L, 3), r.dtype)
    c[:nh, :nw] = r
    return c, (nh, nw)


def show(ax, im, t, cmap=None):
    ax.imshow(im, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(t, fontsize=5.6, pad=2)


def fig_preproc():
    rows = [
        ("test1", "B50 (4).tiff", "colour 1440$\\times$1080"),
        ("test1", "S36 (48).tif", "grey 640$\\times$480"),
    ]
    fig, axs = plt.subplots(2, 6, figsize=(7.2, 3.15))
    for r, (sp, n, lab) in enumerate(rows):
        img = load(sp, n)
        H, W = img.shape[:2]
        a = axs[r]
        show(a[0], img, f"Original\n({lab})")
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        show(
            a[1],
            cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (256, 256)),
            "U-Net, SBSNet:\ncentre crop, 256$^2$",
        )
        d = cv2.resize(img, (1024, 1024)).astype(np.float32) / 255
        dn = (d - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        show(
            a[2],
            ((dn - dn.min()) / (dn.max() - dn.min())),
            "DeepLab: stretched\nto 1024$^2$, ImageNet norm.",
        )
        c, (nh, nw) = rls(img)
        sam = c.copy()
        bx = (0, 0, int((W - 1) * 1024 / W), int((H - 1) * 1024 / H))
        cv2.rectangle(sam, (bx[0] + 10, bx[1] + 10), (bx[2] - 10, bx[3] - 10), (255, 40, 40), 14)
        show(a[3], sam, "SAM, HQ-SAM: long side\n1024, padded; box (red)")
        m = (c.astype(np.float32) - MEAN) / STD
        show(a[4], (m - m.min()) / (m.max() - m.min()), "MedSAM: pad, then\nSAM mean/std")
        g = c.copy()
        ys, xs = np.linspace(0, nh - 1, 32), np.linspace(0, nw - 1, 32)
        for y in ys:
            for x in xs:
                cv2.circle(g, (int(x), int(y)), 5, (255, 255, 0), -1)
        show(a[5], g, "Automatic mode:\n32$\\times$32 point grid")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.04, hspace=0.32)
    fig.savefig(FIG / "preprocessing.pdf")
    fig.savefig(FIG / "preprocessing.png", dpi=200)
    plt.close(fig)


def fig_recorded():
    tags = pd.read_csv(TAGS)
    tags["stem"] = tags.image_name.str.rsplit(".", n=1).str[0]
    dom = dict(zip(tags.stem, tags.domain_yellow_copper, strict=False))
    runs = [
        ("unet", "U-Net"),
        ("sbsnet", "SBSNet"),
        ("deeplabv3_resnet101", "DeepLabV3"),
        ("deeplabv3plus_r152_os8", "DeepLabV3+"),
        ("sam_amg", "SAM auto"),
        ("hqsam_hqamg", "HQ-SAM auto"),
        ("medsam_all.csv", "MedSAM"),
        ("medsam_all_tiled", "MedSAM tiled"),
        ("robustsam", "RobustSAM"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for k, (dv, dname) in enumerate(
        [(1, "Colour evaluation images (n = 60)"), (0, "Greyscale evaluation images (n = 50)")]
    ):
        ax = axs[k]
        for i, (key, _name) in enumerate(runs):
            f = [
                g
                for g in glob.glob(str(RECORDED / "metrics_*.csv"))
                if key in g and not (key == "medsam_all.csv" and "tiled" in g)
            ][0]
            d = pd.read_csv(f)
            d = d[d.split == "val"]
            d = d[d.image_file.str.rsplit(".", n=1).str[0].map(dom) == dv]
            N = d.tp + d.tn + d.fp + d.fn
            P = d.tp + d.fn
            ref = 100 * (2 * P / (N + P))
            ax.scatter(
                100 * d.dice,
                np.full(len(d), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(d)),
                s=4,
                color="#3b6ea5",
                alpha=0.45,
                lw=0,
            )
            ax.plot([ref.mean()] * 2, [i - 0.35, i + 0.35], color="#c0392b", lw=1.6)
            ax.plot(100 * d.dice.mean(), i, "D", color="black", ms=3.5)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([r[1] for r in runs])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Per-image Dice (%)")
        ax.set_title(dname, fontsize=8)
        ax.grid(axis="x", lw=0.3)
    axs[1].plot([], [], color="#c0392b", lw=1.6, label="all-froth reference (mean)")
    axs[1].plot([], [], "D", color="black", ms=3.5, label="model mean")
    axs[1].legend(loc="lower left", fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "recorded_runs.pdf")
    fig.savefig(FIG / "recorded_runs.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_preproc()
    fig_recorded()
    print("preprocessing and recorded-runs figures done")
