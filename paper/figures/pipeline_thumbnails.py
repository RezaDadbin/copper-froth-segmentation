"""Figure 4: the comparative study as two routes from image to bubbles, with thumbnails from one colour evaluation image."""

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.figures.common import FIG, gt, load_img, pred
from paper.figures.outputs import annot, draw, label_of

N = "B50 (4).tiff"
S = 760
plt.rcParams.update({"font.size": 6.3, "font.family": "DejaVu Sans"})
INK, EDGE, BLUE, GREY, GREEN, ACC = "#222222", "#6b6b6b", "#eef3f8", "#f6f6f4", "#e7f1e7", "#2a5d8f"
img = load_img(N)
H, W = img.shape[:2]
_, polys = gt(N, H, W)
y0, x0 = (H - S) // 2, (W - S) // 2
C = lambda a: a[y0 : y0 + S, x0 : x0 + S]
th = {
    "img": C(img),
    "prob": C(np.dstack([pred("dlv3p", N)] * 3)),
    "post": C(draw(img, label_of("dlv3p", "lv", N))),
    "auto": C(draw(img, label_of("amg_hq_h_zs", "amg", N))),
    "gt": C(annot(img, polys)),
}

out = FIG
for k, a in th.items():  # thumbnails for the TikZ figure (figures/pipeline_tikz.tex)
    cv2.imwrite(
        str(out / f"pipe_{k}.jpg"),
        cv2.cvtColor(cv2.resize(a, (480, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
print("thumbnails done")
