"""Motion arrows, displacement distributions and alignment/residual panels."""

import cv2
import numpy as np

from .common import paper_style, plt, save_figure


def render_motion(first, second, vectors, output):
    paper_style()
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].imshow(second, cmap="gray")
    quiver = axes[0].quiver(vectors[:, 0], vectors[:, 1], vectors[:, 2], vectors[:, 3],
                          np.hypot(vectors[:, 2], vectors[:, 3]), angles="xy", scale_units="xy", scale=1,
                          cmap="viridis", width=0.004)
    figure.colorbar(quiver, ax=axes[0], label="Displacement (px/frame)")
    axes[0].set_title("Motion vectors"); axes[0].axis("off")
    axes[1].hist(np.hypot(vectors[:, 2], vectors[:, 3]), bins=30, color="#287a9b")
    axes[1].set(xlabel="Displacement (px/frame)", ylabel="Vectors")
    save_figure(figure, output / "motion_vectors")
    dx, dy = np.median(vectors[:, 2:4], axis=0)
    matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    aligned = cv2.warpAffine(first, matrix, (first.shape[1], first.shape[0]), borderMode=cv2.BORDER_REFLECT)
    residual = cv2.absdiff(second, aligned)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for axis, image, title in zip(axes, (second, aligned, residual), ("Target frame", "Aligned first frame", "Absolute residual")):
        axis.imshow(image, cmap="magma" if title == "Absolute residual" else "gray", vmin=0, vmax=255)
        axis.set_title(title); axis.axis("off")
    save_figure(figure, output / "alignment_residual")
