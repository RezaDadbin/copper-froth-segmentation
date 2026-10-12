"""Headless publication exports with consistent figure dimensions and fonts."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def paper_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False})


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(str(output) + suffix, bbox_inches="tight", facecolor="white")
    plt.close(fig)
