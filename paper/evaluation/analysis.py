"""Bootstrap confidence intervals, per-domain and per-condition results, and the recorded-run reconciliation.

Writes eval110*.tex to $FROTH_TABLES, condition_instance.pdf to $FROTH_FIGURES and summary/reconcile CSVs to $FROTH_WORK.
"""

import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.paths import FIGURES as FIG
from paper.paths import RECORDED, TAGS
from paper.paths import TABLES as TAB
from paper.paths import WORK as RR

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
    }
)
rng = np.random.default_rng(2026)
B = 2000
tags = pd.read_csv(TAGS)
tags["image"] = tags.image_name.str.rsplit(".", n=1).str[0]
d = pd.read_csv(RR / "metrics.csv").merge(tags.drop(columns=["split", "image_name"]), on="image")

ROWS = [  # (model, route, label, group)
    ("dlv3", "cc", "DeepLabV3", "Fine-tuned, semantic"),
    ("dlv3", "ws", "+ ROI watershed", "Fine-tuned, semantic"),
    ("dlv3", "lv", "+ LoG-Voronoi", "Fine-tuned, semantic"),
    ("dlv3p", "cc", "DeepLabV3+", "Fine-tuned, semantic"),
    ("dlv3p", "ws", "+ ROI watershed", "Fine-tuned, semantic"),
    ("dlv3p", "lv", "+ LoG-Voronoi", "Fine-tuned, semantic"),
    ("sam_ft", "cc", "SAM (ft), box", "Fine-tuned, semantic"),
    ("sam_ft", "lv", "+ LoG-Voronoi", "Fine-tuned, semantic"),
    ("hq_ft", "cc", "HQ-SAM (ft), box", "Fine-tuned, semantic"),
    ("hq_ft", "lv", "+ LoG-Voronoi", "Fine-tuned, semantic"),
    ("medsam_ft", "cc", "MedSAM (ft), box, as trained", "Fine-tuned, semantic"),
    ("medsam_ft_fix", "cc", "MedSAM (ft), box, corrected up-sampling", "Fine-tuned, semantic"),
    ("amg_sam_ft", "amg", "SAM (ft), automatic, generator normalisation", "Fine-tuned, instance"),
    (
        "amg_sam_ft_raw",
        "amg",
        "SAM (ft), automatic, training normalisation",
        "Fine-tuned, instance",
    ),
    ("amg_hq_ft", "amg", "HQ-SAM (ft), point grid", "Fine-tuned, instance"),
    ("sam_zs", "cc", "SAM ViT-B, box", "Zero-shot"),
    ("robust_zs", "cc", "RobustSAM, box", "Zero-shot"),
    ("amg_sam_zs", "amg", "SAM ViT-B, automatic", "Zero-shot"),
    ("amg_sam_h_zs", "amg", "SAM ViT-H, automatic", "Zero-shot"),
    ("amg_hq_h_zs", "amg", "HQ-SAM ViT-H, automatic", "Zero-shot"),
    ("amg_sam2_zs", "amg", "SAM 2.1-L, automatic", "Zero-shot"),
    ("amg_robust_zs", "amg", "RobustSAM, point grid", "Zero-shot"),
    ("cellpose_zs", "amg", "Cellpose-SAM", "Zero-shot"),
]
MET = [
    ("dice", 100, "{:.1f}"),
    ("bf_f", 1, "{:.2f}"),
    ("inst_f1", 1, "{:.2f}"),
    ("pq", 1, "{:.2f}"),
    ("count_rel_err", 100, "{:+.0f}"),
    ("d_ks", 1, "{:.2f}"),
]


def F(f, v):
    o = f.format(v)
    if o.lstrip("+-").strip("0.") == "":
        o = o.lstrip("+-")
    return o.replace("-", "$-$")


def ci(x, scale=1):
    x = np.asarray(x, float) * scale
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = rng.choice(x, (B, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def table(sub, fn, cols=MET, ci_cols=None):
    out, rows, grp = [], [], None
    for m, r, lab, g in ROWS:
        x = sub[(sub.model == m) & (sub.route == r)]
        if not len(x):
            continue
        if g != grp:
            if grp is not None:
                out.append("\\addlinespace")
            out.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{g}}}}}\\\\")
            grp = g
        cells = []
        for c, sc, f in cols:
            mu, lo, hi = ci(x[c], sc)
            cells.append(
                f"{F(f, mu)} {{\\scriptsize[{F(f, lo)}, {F(f, hi)}]}}"
                if ci_cols is None or c in ci_cols
                else F(f, mu)
            )
            rows.append(
                dict(
                    model=m, route=r, label=lab, group=g, metric=c, n=len(x), mean=mu, lo=lo, hi=hi
                )
            )
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    (TAB / fn).write_text("\n".join(out) + "\n")
    return pd.DataFrame(rows)


def split_ci_table():
    """Split eval110_ci.tex into two four-column tables (pixel/boundary/instance F1; PQ/count/KS) for the supplement."""
    a, b = [], []
    for r in (TAB / "eval110_ci.tex").read_text().strip().split("\n"):
        if r.startswith("\\multicolumn"):
            a.append(r.replace("{7}", "{4}"))
            b.append(r.replace("{7}", "{4}"))
            continue
        if r.startswith("\\addlinespace"):
            a.append(r)
            b.append(r)
            continue
        c = r.rstrip(" \\").rstrip("\\").split(" & ")
        a.append(" & ".join(c[:4]) + " \\\\")
        b.append(" & ".join([c[0]] + c[4:7]) + " \\\\")
    (TAB / "eval110_ci_a.tex").write_text("\n".join(a) + "\n")
    (TAB / "eval110_ci_b.tex").write_text("\n".join(b) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    t = d[d.split == "test1"]
    print(
        "images",
        t.image.nunique(),
        "colour",
        t[t.domain_yellow_copper == 1].image.nunique(),
        "grey",
        t[t.domain_yellow_copper == 0].image.nunique(),
    )
    allr = table(t, "eval110_ci.tex")
    table(t, "eval110.tex", ci_cols=("inst_f1", "count_rel_err"))
    table(t[t.domain_yellow_copper == 1], "eval110_colour.tex", [MET[0], MET[2], MET[4]])
    table(t[t.domain_yellow_copper == 0], "eval110_grey.tex", [MET[0], MET[2], MET[4]])
    split_ci_table()
    ref = ci(t[(t.model == "dlv3p") & (t.route == "cc")].ref_dice_allfg, 100)
    print("all-froth reference Dice", ref)
    allr.to_csv(RR / "summary.csv", index=False)
    # paired difference: best zero-shot instance vs best fine-tuned route
    a = t[(t.model == "amg_sam_zs") & (t.route == "amg")].set_index("image").inst_f1
    for m, r in [
        ("dlv3p", "ws"),
        ("dlv3p", "lv"),
        ("amg_sam_ft", "amg"),
        ("amg_sam_h_zs", "amg"),
        ("amg_sam2_zs", "amg"),
        ("cellpose_zs", "amg"),
    ]:
        b = t[(t.model == m) & (t.route == r)].set_index("image").inst_f1
        j = a.index.intersection(b.index)
        if len(j):
            print(
                f"paired F1 diff SAM-B auto minus {m}/{r}:", [round(v, 3) for v in ci(a[j] - b[j])]
            )
    # per-condition instance F1 figure (evaluation images)
    T = [
        "structured_light_reflection",
        "uneven_illumination",
        "surface_artifacts_contaminants",
        "polydispersity_multiscale_bubbles",
        "dense_small_bubble_region",
        "elongated_irregular_bubbles",
        "poor_focus_defocus_blur",
        "localized_motion_or_vibration_blur",
        "collapsed",
    ]
    TN = [
        "Reflections",
        "Uneven illum.",
        "Surface artifacts",
        "Polydisperse",
        "Dense small bubbles",
        "Elongated",
        "Defocus blur",
        "Motion blur",
        "Collapse",
    ]
    sel = [
        ("amg_hq_h_zs", "amg", "HQ-SAM\nViT-H"),
        ("amg_sam_zs", "amg", "SAM\nViT-B"),
        ("amg_sam_h_zs", "amg", "SAM\nViT-H"),
        ("amg_sam2_zs", "amg", "SAM 2"),
        ("cellpose_zs", "amg", "Cellpose\n-SAM"),
        ("amg_sam_ft", "amg", "SAM\nViT-B"),
        ("dlv3p", "ws", "DLV3+\nWS"),
        ("dlv3p", "lv", "DLV3+\nLoG-V"),
    ]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    seq = LinearSegmentedColormap.from_list(
        "seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    div = LinearSegmentedColormap.from_list(
        "div", ["#104281", "#6da7ec", "#f0efec", "#e66767", "#a32c2c"]
    )
    rows = [("All images", None)] + list(zip(TN, T, strict=False))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for k, (col, title, sc, cmap, norm, fmt) in enumerate(
        [
            (
                "inst_f1",
                "(a) Instance F1 (higher is better)",
                1,
                seq,
                plt.Normalize(0, 1),
                "{:.2f}",
            ),
            (
                "count_rel_err",
                "(b) Bubble-count error, % (0 is best)",
                100,
                div,
                TwoSlopeNorm(0, -100, 100),
                "{:+.0f}",
            ),
        ]
    ):
        M = np.full((len(rows), len(sel)), np.nan)
        for c, (m, r, _) in enumerate(sel):
            x = t[(t.model == m) & (t.route == r)]
            for q, (_, tg) in enumerate(rows):
                v = x if tg is None else x[x[tg] == 1]
                if len(v):
                    M[q, c] = v[col].mean() * sc
        ax = axs[k]
        ax.imshow(np.clip(M, -100, 100) if sc == 100 else M, cmap=cmap, norm=norm, aspect="auto")
        for q in range(M.shape[0]):
            for c in range(M.shape[1]):
                val = np.clip(M[q, c], -100, 100) if sc == 100 else M[q, c]
                rgb = cmap(norm(val))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(
                    c,
                    q,
                    ""
                    if np.isnan(M[q, c])
                    else fmt.format(
                        M[q, c] + 0.0 if abs(M[q, c]) >= 0.5 or sc == 1 else 0.0
                    ).replace("+0", "0")
                    if sc == 100
                    else fmt.format(M[q, c]),
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if lum < 0.45 else "#222",
                )
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels([n for _, _, n in sel], fontsize=5.6)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        n = [len(t[(t.model == "amg_sam_zs") & (t.route == "amg")])] + [
            int((t[(t.model == "amg_sam_zs") & (t.route == "amg")][tg] == 1).sum())
            for _, tg in rows[1:]
        ]
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{a} (n={b})" for (a, _), b in zip(rows, n, strict=False)] if k == 0 else []
        )
        ax.set_xticks(np.arange(-0.5, len(sel)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.set_xlim(-0.5, len(sel) - 0.5)
        ax.tick_params(which="minor", length=0)
        ax.axhline(0.5, color="white", lw=4)
        for x0, x1, lab in [(-0.4, 4.4, "zero-shot, automatic"), (4.6, 7.4, "fine-tuned")]:
            ax.annotate(
                "",
                (x0, -1.55),
                (x1, -1.55),
                xycoords="data",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#555"),
            )
            ax.text(
                (x0 + x1) / 2,
                -1.75,
                lab,
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#444",
                clip_on=False,
            )
        ax.set_title(title, fontsize=7, pad=46, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "condition_instance.pdf")
    fig.savefig(FIG / "condition_instance.png", dpi=200)
    # reconciliation: recorded per-image Dice vs re-run Dice on the same evaluation images
    rec = {
        "deeplabv3_resnet101": "dlv3",
        "deeplabv3plus_r152_os8": "dlv3p",
        "sam_amg": "amg_sam_zs",
        "hqsam_hqamg": "amg_hq_ft",
        "medsam_all.csv": "medsam_ft",
        "robustsam": "amg_robust_zs",
    }
    out = []
    for key, m in rec.items():
        f = [
            g
            for g in glob.glob(str(RECORDED / "metrics_*.csv"))
            if key in g and not (key == "medsam_all.csv" and "tiled" in g)
        ][0]
        R = pd.read_csv(f)
        R = R[R.split == "val"]
        R["image"] = R.image_file.str.rsplit(".", n=1).str[0]
        route = "amg" if m.startswith("amg") else "cc"
        X = t[(t.model == m) & (t.route == route)][["image", "dice"]].merge(
            R[["image", "dice"]], on="image", suffixes=("_rerun", "_recorded")
        )
        if len(X):
            out.append(
                dict(
                    run=key,
                    rerun=m,
                    n=len(X),
                    recorded=100 * X.dice_recorded.mean(),
                    rerun_mean=100 * X.dice_rerun.mean(),
                    mean_abs_diff=100 * (X.dice_rerun - X.dice_recorded).abs().mean(),
                    r=X.dice_rerun.corr(X.dice_recorded),
                )
            )
    o = pd.DataFrame(out)
    o.to_csv(RR / "reconcile.csv", index=False)
    print(o.round(3).to_string())


if __name__ == "__main__":
    main()
