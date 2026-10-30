"""Overview of the 42 reviewed segmentation studies: method families over time, dataset sizes, evaluation practice."""

import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.paths import FIGURES as FIG
from paper.paths import REVIEW

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
    }
)
ALL = json.load(open(REVIEW / "records.json"))
S = [
    r
    for r in ALL
    if r["category"] in ("deep_segmentation", "classical_segmentation", "foundation_model")
]


def family(r):
    if r["category"] == "foundation_model":
        return "Foundation (SAM)"
    if re.search(r"Mask R-CNN|YOLO|instance seg", str(r.get("method", "")), re.I):
        return "Instance network"
    return "Classical" if r["category"] == "classical_segmentation" else "CNN (edges/regions)"


FAM = ["Classical", "CNN (edges/regions)", "Instance network", "Foundation (SAM)"]
COL = {
    "Classical": "#9c8a6b",
    "CNN (edges/regions)": "#3b6ea5",
    "Instance network": "#d08b2c",
    "Foundation (SAM)": "#4f9d69",
}
PER = [
    ("≤2010", 1995, 2010),
    ("2011–19", 2011, 2019),
    ("2020–22", 2020, 2022),
    ("2023–24", 2023, 2024),
    ("2025–26", 2025, 2026),
]


def answer(v):
    v = str(v).lower()
    return (
        "yes"
        if v.startswith("yes") or v.startswith("likely yes")
        else ("no" if v.startswith("no") else ("partial" if v.startswith("partial") else "unclear"))
    )


# dataset sizes stated in the sources (images used for training/evaluation); pseudo-labelled sets hollow
SIZES = [
    ("Li 2013", 2013, 56, False),
    ("Zhao 2015", 2015, 4, False),
    ("Xie 2019", 2019, 92, False),
    ("da Silva 2021", 2021, 228, True),
    ("Ogonowski 2025", 2025, 1000, True),
    ("Wang 2015", 2015, 36, False),
    ("Zhang 2020", 2020, 35, False),
    ("Zhang 2022", 2022, 80, False),
    ("Gharehchobogh 2023\n(Sungun)", 2023, 1000, False),
    ("Liao 2025", 2025, 912, False),
    ("Zhu 2025", 2025, 310, False),
    ("Prokopov 2025", 2025, 15000, True),
    ("Mohammadi-V. 2026", 2026, 320, False),
    ("Yang 2026", 2026, 50, False),
    ("Vasilyeva 2026", 2026, 13613, True),
]

fig = plt.figure(figsize=(7.0, 4.75))
gs = fig.add_gridspec(
    2, 2, height_ratios=[1, 1.25], width_ratios=[1, 1.15], hspace=0.55, wspace=0.42
)
COL = {
    "Classical": "#2a78d6",
    "CNN (edges/regions)": "#eb6834",
    "Instance network": "#1baf7a",
    "Foundation (SAM)": "#eda100",
}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e3df"


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9a9a96")
    ax.tick_params(colors=INK, length=2.5)


# (a) method families over time
ax = fig.add_subplot(gs[0, 0])
bottom = np.zeros(len(PER))
for f in FAM:
    v = np.array([sum(1 for r in S if family(r) == f and a <= r["year"] <= b) for _, a, b in PER])
    ax.bar(
        range(len(PER)),
        v,
        bottom=bottom,
        color=COL[f],
        label=f,
        width=0.62,
        edgecolor="white",
        linewidth=0.8,
    )
    bottom += v
for k, t in enumerate(bottom):
    ax.text(k, t + 0.35, str(int(t)), ha="center", va="bottom", fontsize=7, color=INK)
ax.set_xticks(range(len(PER)))
ax.set_xticklabels([p[0] for p in PER])
ax.set_ylabel("Studies")
ax.set_ylim(0, 20)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
clean(ax)
ax.set_title(
    f"(a) Method families of the {len(S)} segmentation studies", loc="left", fontsize=8.5, pad=34
)
ax.legend(
    ncol=2,
    fontsize=6.8,
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0, 1.0),
    handlelength=0.9,
    columnspacing=1.0,
    borderaxespad=0.2,
)
# (b) evaluation practice
ax = fig.add_subplot(gs[0, 1])
cats = [
    ("Held-out\ntest set", "held_out_test"),
    ("Instance-level\nmetrics", "instance_level_metrics"),
]
cc = {"yes": "#1c5cab", "partial": "#86b6ef", "no": "#e34948", "unclear": "#d9d8d4"}
lab_k = {
    "yes": "reported",
    "partial": "partly",
    "no": "absent",
    "unclear": "not stated or not accessible",
}
for i, (_lab, key) in enumerate(cats):
    left = 0
    for k in ("yes", "partial", "no", "unclear"):
        n = sum(1 for r in S if answer(r.get(key)) == k)
        ax.barh(
            i,
            n,
            left=left,
            color=cc[k],
            height=0.56,
            edgecolor="white",
            linewidth=1.0,
            label=lab_k[k] if i == 0 else None,
        )
        if n:
            ax.text(
                left + n / 2,
                i,
                str(n),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if k in ("yes", "no") else INK,
            )
        left += n
ax.set_yticks(range(len(cats)))
ax.set_yticklabels([c[0] for c in cats])
ax.invert_yaxis()
ax.set_xlabel("Studies")
ax.set_xlim(0, len(S))
ax.tick_params(axis="y", length=0)
clean(ax)
ax.spines["left"].set_visible(False)
ax.set_title("(b) How the studies were evaluated", loc="left", fontsize=8.5, pad=34)
ax.legend(
    ncol=4,
    fontsize=6.8,
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(-0.02, 1.0),
    handlelength=0.9,
    columnspacing=0.9,
    borderaxespad=0.2,
)
# (c) dataset sizes as a dot plot
ax = fig.add_subplot(gs[1, :])
rows = sorted(SIZES + [("This study (Sungun)", 2026, 595, False)], key=lambda x: x[2])
for k, (lab, _year, n, pseudo) in enumerate(rows):
    me = lab.startswith("This study")
    c = "#c8642a" if me else "#2a5d8f"
    ax.plot([2, n], [k, k], color="#d6d5d1", lw=1.0, zorder=1)
    ax.scatter(
        [n],
        [k],
        s=26 if me else 18,
        facecolor="white" if pseudo else c,
        edgecolor=c,
        linewidth=1.1,
        zorder=3,
    )
    note = (
        f"{n:,}"
        + ("  pseudo-labels" if pseudo else "")
        + ("  manual polygons, 9 condition tags" if me else "")
    )
    ax.text(
        n * 1.18,
        k,
        note,
        va="center",
        fontsize=6.8,
        color=c if me else MUTED,
        fontweight="bold" if me else "normal",
    )
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0].replace(chr(10), " ") for r in rows], fontsize=7)
ax.set_xscale("log")
ax.set_xlim(2, 2e5)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("Images used for training and evaluation (log scale)")
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
clean(ax)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title(
    "(c) Dataset sizes stated in the studies (hollow: machine-generated pseudo-labels)",
    loc="left",
    fontsize=8.5,
)
fig.savefig(FIG / "review_overview.pdf", bbox_inches="tight")
fig.savefig(FIG / "review_overview.png", dpi=200, bbox_inches="tight")
for lab, key in cats:
    print(
        lab.replace(chr(10), " "),
        {
            k: sum(1 for r in S if answer(r.get(key)) == k)
            for k in ("yes", "partial", "no", "unclear")
        },
    )
print({f: sum(1 for r in S if family(r) == f) for f in FAM})
