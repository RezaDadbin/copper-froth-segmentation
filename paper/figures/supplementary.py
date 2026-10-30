"""Supplementary pages from the 110-image re-run: two evaluation images per condition tag (one colour, one greyscale where
available), every model and route on each image (3 rows x 6 panels). Same visual language as make_figs_v2.py."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.figures.common import SUP, TAGS, gt, load_img
from paper.figures.outputs import annot, crop, draw, label_of, note_inst, note_sem, tile
from paper.paths import SELECTION, WORK

PANELS = [  # (model, route, title, kind)
    [
        None,
        ("dlv3", "cc", "DeepLabV3", "s"),
        ("dlv3p", "cc", "DeepLabV3+", "s"),
        ("sam_ft", "cc", "SAM (ft), box", "s"),
        ("hq_ft", "cc", "HQ-SAM (ft), box", "s"),
        ("medsam_ft", "cc", "MedSAM (ft), box", "s"),
    ],
    [
        ("sam_zs", "cc", "SAM (zs), box", "s"),
        ("robust_zs", "cc", "RobustSAM (zs), box", "s"),
        ("dlv3p", "ws", "DLV3+ + watershed", "i"),
        ("dlv3p", "lv", "DLV3+ + LoG-Voronoi", "i"),
        ("amg_sam_ft", "amg", "SAM (ft), automatic", "i"),
        ("amg_hq_ft", "amg", "HQ-SAM (ft), point grid", "i"),
    ],
    [
        ("amg_sam_zs", "amg", "SAM ViT-B, automatic", "i"),
        ("amg_sam_h_zs", "amg", "SAM ViT-H, automatic", "i"),
        ("amg_hq_h_zs", "amg", "HQ-SAM ViT-H, automatic", "i"),
        ("amg_sam2_zs", "amg", "SAM 2, automatic", "i"),
        ("cellpose_zs", "amg", "Cellpose-SAM", "i"),
        ("amg_robust_zs", "amg", "RobustSAM, point grid", "i"),
    ],
]
TITLES = {
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
    "at_least_6_conditions": "Evaluation images with the most conditions (five tags)",
}


def select():
    t = TAGS[TAGS.split == "test"].copy()
    prev = set(sum(json.load(open(SELECTION)).values(), []))
    tc = [c for c in TITLES if c in t.columns]
    t["ntags"] = t[tc].sum(1)
    used, sel = set(), {}
    for k in TITLES:
        m = (
            t["ntags"] <= 2
            if k == "at_most_2_conditions"
            else t["ntags"] >= t["ntags"].nlargest(2).min()
            if k == "at_least_6_conditions"
            else t[k] == 1
        )
        c = t[m & ~t.image_name.isin(used)].copy()
        c["prev"] = c.image_name.isin(prev)
        c = c.sort_values(["prev", "image_name"], ascending=[False, True])
        pick = []
        for dom in (1, 0):
            d = c[c.domain_yellow_copper == dom]
            if len(d):
                pick.append(d.image_name.iloc[0])
        for n in c.image_name:
            if len(pick) >= 2:
                break
            if n not in pick:
                pick.append(n)
        sel[k] = pick
        used.update(pick)
    return sel


def page(names, title, out):
    fig, axs = plt.subplots(3 * len(names), 6, figsize=(7.4, 3 * len(names) * 1.3 + 0.55))
    for i, n in enumerate(names):
        img = load_img(n)
        H, W = img.shape[:2]
        _, polys = gt(n, H, W)
        C, _ = crop(img)
        dom = "colour" if TAGS.set_index("image_name").loc[n].domain_yellow_copper else "greyscale"
        for r, row in enumerate(PANELS):
            for j, p in enumerate(row):
                ax = axs[3 * i + r, j]
                if p is None:
                    tile(
                        ax,
                        C(annot(img, polys)),
                        f"{dom.capitalize()} image",
                        f"annotation: {len(polys)} bubbles",
                    )
                    continue
                m, rt, lab, kind = p
                tile(
                    ax,
                    C(draw(img, label_of(m, rt, n))),
                    lab,
                    note_sem(n, m) if kind == "s" else note_inst(n, m, rt),
                )
    fig.suptitle(title, fontsize=9, y=0.997)
    fig.text(
        0.5,
        0.004,
        "Yellow: borders predicted by each model; cyan: annotation. Box-prompted and semantic models: borders of the predicted froth regions "
        '(no borders = everything predicted as froth), with Dice and "ref" = Dice of labelling every pixel as froth. Other panels: individual bubbles, '
        "with instance F1 and bubble-count error. (ft) fine-tuned, (zs) zero-shot. Central 560 px crop; scores are for the whole image.",
        ha="center",
        fontsize=5.4,
        wrap=True,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.962, bottom=0.035, wspace=0.03, hspace=0.17)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sel = select()
    SUP.mkdir(parents=True, exist_ok=True)
    json.dump(sel, open(WORK / "selection110.json", "w"), indent=1)
    for f in SUP.glob("S*.png"):
        if "saved_unet_sbsnet" not in f.name:
            f.unlink()
    for i, (k, t) in enumerate(TITLES.items(), 1):
        page(sel[k], f"S{i}. {t}", SUP / f"S{i:02d}_{k}.png")
        print("page", i, sel[k], flush=True)
