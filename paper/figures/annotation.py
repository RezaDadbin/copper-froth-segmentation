"""Annotation figure: a colour and a greyscale image with their polygons, plus 4x zoomed insets."""

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.figures.common import FIG, gt, load_img

rows = [
    ("B50 (52).tiff", (930, 700, 260)),
    ("S36 (48).tif", (230, 160, 110)),
]  # (x, y, size) of the zoom window
fig, axs = plt.subplots(2, 3, figsize=(10, 5.1), gridspec_kw=dict(width_ratios=[1.33, 1, 1]))
for i, (n, (x, y, s)) in enumerate(rows):
    img = load_img(n)
    H, W = img.shape[:2]
    _, polys = gt(n, H, W)
    ov = img.copy()
    cv2.polylines(ov, polys, True, (0, 255, 255), max(1, W // 700), cv2.LINE_AA)
    full = ov.copy()
    cv2.rectangle(full, (x, y), (x + s, y + s), (255, 230, 0), max(2, W // 300))
    z_img = cv2.resize(img[y : y + s, x : x + s], (600, 600), interpolation=cv2.INTER_NEAREST)
    z_ov = img[y : y + s, x : x + s].copy().astype(np.uint8)
    k = 600 / s
    z_ov = cv2.resize(z_ov, (600, 600), interpolation=cv2.INTER_NEAREST)
    for p in polys:
        q = ((p - [x, y]) * k).astype(np.int32)
        cv2.polylines(z_ov, [q], True, (0, 255, 255), 2, cv2.LINE_AA)
        for v in q:
            if 0 <= v[0] < 600 and 0 <= v[1] < 600:
                cv2.circle(z_ov, tuple(int(c) for c in v), 3, (255, 60, 60), -1)
    dom = "colour, 1440$\\times$1080" if W > 1000 else "greyscale, 640$\\times$480"
    for ax, im, t in zip(
        axs[i],
        (full, z_img, z_ov),
        (
            f"{dom.capitalize()} image, {len(polys)} bubbles",
            "Zoomed region",
            "Annotated polygons and vertices",
        ),
        strict=False,
    ):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t, fontsize=8, pad=3)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig(FIG / "annotation.pdf", dpi=220)
fig.savefig(FIG / "annotation.png", dpi=170)
print("done")
