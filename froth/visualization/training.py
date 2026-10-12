"""Render training/validation histories written by the shared trainer."""

import csv
from pathlib import Path

from .common import paper_style, plt, save_figure


def render_history(history_csv, output):
    with Path(history_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training history is empty")
    paper_style()
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), layout="constrained")
    for ax, metric, label in zip(axes, ("loss", "iou", "pixel_acc"), ("Loss", "Foreground IoU", "Pixel accuracy")):
        ax.plot(epochs, [float(row[f"train_{metric}"]) for row in rows], label="Training")
        ax.plot(epochs, [float(row[f"validation_{metric}"]) for row in rows], label="Reused validation")
        ax.set(xlabel="Epoch", ylabel=label)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)
