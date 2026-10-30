"""Which bubble measurements can be trusted. On the 110 evaluation images and the saved outputs:
 1) outcome of every annotated bubble by size quartile (matched / merged / split / missed),
 2) instance F1 as a function of the matching IoU threshold (greedy one-to-one matching),
 3) absolute per-image bubble-count errors,
 4) pooled equivalent-diameter distributions.
Size quartiles are fixed beforehand from the TRAINING annotations, per domain (pixel scales differ between domains)."""

import json
import os
from multiprocessing import Pool

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.evaluation import metrics as M
from paper.paths import DATA
from paper.paths import FIGURES as FIG
from paper.paths import RUN as O
from paper.paths import TABLES as TAB
from paper.paths import TAGS as TAGS_CSV
from paper.paths import WORK as RES

ROUTES = [
    ("amg_hq_h_zs", "amg", "HQ-SAM ViT-H (zs)"),
    ("amg_sam_zs", "amg", "SAM ViT-B (zs)"),
    ("amg_sam_h_zs", "amg", "SAM ViT-H (zs)"),
    ("amg_sam2_zs", "amg", "SAM 2 (zs)"),
    ("cellpose_zs", "amg", "Cellpose-SAM (zs)"),
    ("amg_sam_ft", "amg", "SAM ViT-B (ft), automatic"),
    ("dlv3p", "lv", "DeepLabV3+ + LoG-Voronoi"),
    ("dlv3p", "ws", "DeepLabV3+ + watershed"),
]
THR = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
TAGS = pd.read_csv(TAGS_CSV)
TAGS["stem"] = TAGS.image_name.str.rsplit(".", n=1).str[0]
DOM = dict(zip(TAGS.stem, TAGS.domain_yellow_copper, strict=False))
plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def size_edges():
    """Quartiles of equivalent diameter of the training-set polygons, per domain."""
    d = {0: [], 1: []}
    for jp in (DATA / "train1").glob("*.json"):
        st = jp.stem
        if st not in DOM:
            continue
        for s in json.load(open(jp)).get("shapes", []):
            if s.get("label") != "froth" or s.get("shape_type") == "circle":
                continue
            p = np.array(s["points"], float)
            a = 0.5 * abs(
                np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))
            )
            if a > 0:
                d[DOM[st]].append(np.sqrt(4 * a / np.pi))
    return {k: np.percentile(v, [25, 50, 75]).tolist() for k, v in d.items()}


def clean(pl):
    pl = pl.astype(np.int64)
    a = np.bincount(pl.ravel())
    small = np.where(a < M.MIN_INST)[0]
    small = small[small > 0]
    if len(small):
        pl[np.isin(pl, small)] = 0
    return M.relabel(pl)


def job(args):
    stem, edges = args
    H = W = None
    out = dict(outcomes=[], f1=[], counts=[], diam=[], fp=[])
    for m, rt, lab in ROUTES:
        f = O / (m if rt == "amg" else f"{m}__{rt}") / "test1" / f"{stem}.png"
        pl = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if H is None:
            H, W = pl.shape[:2]
            _, gi = M.gt_maps(DATA / "test1" / f"{stem}.json", H, W)
            gi = M.relabel(gi.astype(np.int64))
            ng = int(gi.max())
            ag = np.bincount(gi.ravel(), minlength=ng + 1)
            dg = np.sqrt(4 * ag[1:] / np.pi)
            dom = DOM[stem]
            e = edges[dom]
            gbin = np.digitize(dg, e)  # 0..3 = size quartile of each annotated bubble
            out["diam"].append(dict(route="annotation", dom=dom, d=dg[ag[1:] > 0].tolist()))
        pl = clean(pl)
        npd = int(pl.max())
        ap = np.bincount(pl.ravel(), minlength=npd + 1)
        both = (gi > 0) & (pl > 0)
        pair, cnt = np.unique(gi[both] * (npd + 1) + pl[both], return_counts=True)
        g, p = pair // (npd + 1), pair % (npd + 1)
        iou = cnt / (ag[g] + ap[p] - cnt)
        cov_g = cnt / ag[g]
        cov_p = cnt / ap[p]
        # 1) outcome per annotated bubble
        matched = np.zeros(ng + 1, bool)
        matched[g[iou > 0.5]] = True
        best = np.zeros(ng + 1)
        bestp = np.zeros(ng + 1, int)
        for gg, pp, c in zip(g, p, cov_g, strict=False):
            if c > best[gg]:
                best[gg], bestp[gg] = c, pp
        multi = np.bincount(
            p[cov_g >= 0.5], minlength=npd + 1
        )  # annotated bubbles covered >= 50 % by this prediction
        nsplit = np.bincount(
            g[cov_g >= 0.2], minlength=ng + 1
        )  # predictions covering >= 20 % of an annotated bubble
        for k in range(1, ng + 1):
            if ag[k] == 0:
                continue
            if matched[k]:
                o = "matched"
            elif best[k] >= 0.5 and multi[bestp[k]] >= 2:
                o = "merged"
            elif nsplit[k] >= 2:
                o = "split"
            else:
                o = "missed"
            out["outcomes"].append(dict(route=lab, dom=dom, bin=int(gbin[k - 1]), outcome=o))
        # 2) greedy one-to-one matching; TP(t) = accepted matches with IoU >= t
        order = np.argsort(-iou)
        ug, up, acc = set(), set(), []
        for i in order:
            if iou[i] < THR[0]:
                break
            if g[i] in ug or p[i] in up:
                continue
            ug.add(g[i])
            up.add(p[i])
            acc.append(iou[i])
        acc = np.array(acc)
        for t in THR:
            tp = int((acc >= t).sum())
            out["f1"].append(dict(route=lab, image=stem, thr=t, f1=2 * tp / max(1, ng + npd)))
        out["counts"].append(dict(route=lab, image=stem, dom=dom, n_gt=ng, n_pred=npd))
        mp = np.zeros(npd + 1, bool)
        mp[p[iou > 0.5]] = True
        dp = np.sqrt(4 * ap / np.pi)
        un = [k for k in range(1, npd + 1) if ap[k] > 0 and not mp[k]]
        out["fp"].append(
            dict(
                route=lab,
                dom=dom,
                n_pred=npd,
                n_unmatched=len(un),
                n_unmatched_small=int(sum(dp[k] < e[0] for k in un)),
                n_unmatched_no_overlap=int(sum(1 for k in un if not np.any(p[cov_p >= 0.5] == k))),
            )
        )
        out["diam"].append(
            dict(route=lab, dom=dom, d=np.sqrt(4 * ap[1:][ap[1:] > 0] / np.pi).tolist())
        )
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    edges = size_edges()
    json.dump(edges, open(RES / "size_edges.json", "w"))
    print("size quartile edges (px, eq. diameter):", edges)
    stems = sorted(TAGS[TAGS.split == "test"].stem)
    with Pool(max(2, os.cpu_count() - 2)) as pool:
        res = pool.map(job, [(s, edges) for s in stems])
    oc = pd.DataFrame([r for x in res for r in x["outcomes"]])
    f1 = pd.DataFrame([r for x in res for r in x["f1"]])
    ct = pd.DataFrame([r for x in res for r in x["counts"]])
    oc.to_csv(RES / "size_outcomes.csv", index=False)
    f1.to_csv(RES / "iou_sensitivity.csv", index=False)
    ct.to_csv(RES / "counts.csv", index=False)
    diam = {}
    for x in res:
        for r in x["diam"]:
            diam.setdefault((r["route"], r["dom"]), []).extend(r["d"])
    labs = [l for _, _, l in ROUTES]
    fp = (
        pd.DataFrame([r for x in res for r in x["fp"]])
        .groupby("route")[["n_pred", "n_unmatched", "n_unmatched_small", "n_unmatched_no_overlap"]]
        .sum()
    )
    fp["unmatched_%"] = 100 * fp.n_unmatched / fp.n_pred
    fp["small_%_of_unmatched"] = 100 * fp.n_unmatched_small / fp.n_unmatched
    fp["mostly_outside_annotation_%"] = 100 * fp.n_unmatched_no_overlap / fp.n_unmatched
    fp.to_csv(RES / "unmatched_predictions.csv")
    print(fp.round(1).to_string())
    # ---- size outcomes (pooled over domains; quartiles are within-domain)
    tab = (
        oc.groupby(["route", "bin"]).outcome.value_counts(normalize=True).unstack(fill_value=0)
        * 100
    )
    tab.to_csv(RES / "size_outcomes_summary.csv")
    print(tab.round(1).to_string())
    tabd = (
        oc.groupby(["route", "dom", "bin"])
        .outcome.value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    tabd.to_csv(RES / "size_outcomes_by_domain.csv")
    out = []
    for dom, nm in [(1, "Colour"), (0, "Greyscale")]:
        out.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{nm} images}}}}\\\\")
        for lab in labs:
            cells = [
                f"{tabd.loc[(lab, dom, b), 'matched']:.0f}/{tabd.loc[(lab, dom, b), 'merged']:.0f}/{tabd.loc[(lab, dom, b), 'missed']:.0f}"
                if (lab, dom, b) in tabd.index
                else "--"
                for b in range(4)
            ]
            out.append(lab + " & " + " & ".join(cells) + "\\\\")
    (TAB / "size_by_domain.tex").write_text("\n".join(out) + "\n")
    BN = ["Smallest quarter", "Second quarter", "Third quarter", "Largest quarter"]
    OC = [
        ("matched", "#2a78d6"),
        ("merged", "#eb6834"),
        ("split", "#1baf7a"),
        ("missed", "#c8c7c2"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    for b, ax in enumerate(axs):
        left = np.zeros(len(labs))
        for o, c in OC:
            v = np.array(
                [
                    tab.loc[(l, b), o] if (l, b) in tab.index and o in tab.columns else 0
                    for l in labs
                ]
            )
            ax.barh(
                range(len(labs)),
                v,
                left=left,
                color=c,
                height=0.72,
                edgecolor="white",
                linewidth=0.8,
                label=o,
            )
            left += v
        ax.set_title(BN[b], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"], fontsize=6)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs)
        ax.invert_yaxis()
        ax.tick_params(length=0)
    axs[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.13), frameon=False, fontsize=6.5)
    fig.supxlabel("Share of annotated bubbles (%)", fontsize=7, y=0.02)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.8, bottom=0.17, wspace=0.14)
    fig.savefig(FIG / "size_outcomes.pdf")
    fig.savefig(FIG / "size_outcomes.png", dpi=200)
    plt.close(fig)
    # ---- absolute count errors
    ct["err"] = (ct.n_pred - ct.n_gt) / ct.n_gt.clip(lower=1)
    ct["abs"] = ct.err.abs() * 100
    rows = []
    for l in labs:
        x = ct[ct.route == l]
        rows.append(
            dict(
                route=l,
                signed=100 * x.err.mean(),
                mean_abs=x["abs"].mean(),
                median_abs=x["abs"].median(),
                p90=np.percentile(x["abs"], 90),
                within10=100 * (x["abs"] <= 10).mean(),
                within20=100 * (x["abs"] <= 20).mean(),
                within20_col=100 * (x[x.dom == 1]["abs"] <= 20).mean(),
                within20_grey=100 * (x[x.dom == 0]["abs"] <= 20).mean(),
            )
        )
    cs = pd.DataFrame(rows)
    cs.to_csv(RES / "count_abs.csv", index=False)
    print(cs.round(1).to_string())
    fm = lambda v, f="{:.0f}": f.format(v).replace("-", "$-$")
    (TAB / "count_abs.tex").write_text(
        "\n".join(
            f"{r.route} & {fm(r.signed, '{:+.0f}')} & {fm(r.mean_abs)} & {fm(r.median_abs)} & {fm(r.p90)} & {fm(r.within10)} & {fm(r.within20)} & {fm(r.within20_col)} & {fm(r.within20_grey)}\\\\"
            for r in cs.itertuples()
        )
        + "\n"
    )
    # ---- IoU sensitivity
    fs = f1.groupby(["route", "thr"]).f1.mean().unstack()
    fs = fs.loc[labs]
    fs.to_csv(RES / "iou_sensitivity_summary.csv")
    print(fs.round(3).to_string())
    (TAB / "iou_sens.tex").write_text(
        "\n".join(
            lab + " & " + " & ".join(f"{v:.2f}" for v in fs.loc[lab].values) + "\\\\"
            for lab in labs
        )
        + "\n"
    )
    rank = fs.rank(ascending=False, axis=0).astype(int)
    print(rank.to_string())
    # ---- reliability figure: (a) F1 vs IoU threshold, (b) counts, (c, d) diameter CDFs
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.9))
    axs = axs.ravel()
    ax = axs[0]
    for i, l in enumerate(labs):
        ax.plot(THR, fs.loc[l].values, color=CAT[i], lw=1.4, marker="o", ms=2.4, label=l)
    ax.set_xlabel("Matching IoU threshold")
    ax.set_ylabel("Instance F1")
    ax.set_ylim(0, 1)
    ax.set_title("(a) Matching threshold", fontsize=7, loc="left")
    ax.legend(
        fontsize=5.3, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.3
    )
    S3 = [labs[0], labs[1], labs[6]]
    ax = axs[1]
    for i, l in enumerate(S3):
        x = ct[ct.route == l]
        ax.scatter(x.n_gt, x.n_pred, s=6, color=CAT[[0, 1, 6][i]], alpha=0.8, lw=0, label=l)
    mx = ct[ct.route.isin(S3)][["n_gt", "n_pred"]].max().max() * 1.05
    ax.plot([0, mx], [0, mx], color="#555", lw=0.6)
    ax.fill_between([0, mx], [0, 0.8 * mx], [0, 1.2 * mx], color="#888", alpha=0.12, lw=0)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Annotated bubbles per image")
    ax.set_ylabel("Predicted bubbles per image")
    ax.set_title("(b) Counts (band: ±20%)", fontsize=7, loc="left")
    ax.legend(fontsize=4.9, frameon=False, loc="upper left", handlelength=1)
    for k, (dom, nm) in enumerate([(1, "colour"), (0, "greyscale")]):
        ax = axs[2 + k]
        d = np.sort(diam[("annotation", dom)])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), color="black", lw=1.6, label="annotation")
        for i, l in enumerate(S3):
            d = np.sort(diam[(l, dom)])
            ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=CAT[[0, 1, 6][i]], lw=1.1, label=l)
        ax.set_xscale("log")
        ax.set_xlabel("Equivalent diameter (px)")
        ax.set_ylim(0, 1)
        ax.set_title(f"({'cd'[k]}) Size distribution, {nm}", fontsize=7, loc="left")
        if k == 0:
            ax.set_ylabel("Cumulative fraction of bubbles")
    axs[3].legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=1.2)
    for a in axs:
        a.grid(lw=0.3, color="#ddd")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.09, wspace=0.78, hspace=0.42)
    fig.savefig(FIG / "reliability.pdf")
    fig.savefig(FIG / "reliability.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
