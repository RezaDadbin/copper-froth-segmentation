"""Data tables of the paper: condition-tag frequencies, bubble geometry and the recorded runs.

The bubble measurements come from ``scripts/dataset_statistics.py`` (froth_measurements.csv); the recorded runs are
the per-image metric CSVs saved by the original training scripts.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from paper.paths import RECORDED, TABLES, TAGS, WORK

CONDITIONS = [
    ("structured_light_reflection", "Regular light reflections"),
    ("uneven_illumination", "Uneven illumination"),
    ("surface_artifacts_contaminants", "Surface artifacts / contaminants"),
    ("polydispersity_multiscale_bubbles", "Polydispersity / multiscale"),
    ("dense_small_bubble_region", "Dense small-bubble regions"),
    ("elongated_irregular_bubbles", "Elongated / irregular bubbles"),
    ("poor_focus_defocus_blur", "Poor focus / defocus blur"),
    ("localized_motion_or_vibration_blur", "Localized motion / vibration blur"),
    ("collapsed", "Collapse appearance"),
]
RECORDED_RUNS = [
    ("unet_all", "U-Net"),
    ("sbsnet_all", "SBSNet"),
    ("deeplabv3_resnet101", "DeepLabV3"),
    ("deeplabv3plus_r152_os8", "DeepLabV3+"),
    ("sam_amg", "SAM (automatic)"),
    ("hqsam_hqamg", "HQ-SAM (automatic)"),
    ("medsam_all.csv", "MedSAM"),
    ("medsam_all_tiled", "MedSAM tiled"),
    ("robustsam", "RobustSAM"),
]


def tag_counts(tags: pd.DataFrame) -> str:
    n_tags = tags[[c for c, _ in CONDITIONS]].sum(axis=1)
    groups = (
        [("Colour domain", tags.domain_yellow_copper == 1)]
        + [(lab, tags[c] == 1) for c, lab in CONDITIONS]
        + [
            ("No regular light reflections", tags.structured_light_reflection == 0),
            ("No surface artifacts / contaminants", tags.surface_artifacts_contaminants == 0),
            ("At most 2 conditions", n_tags <= 2),
            ("At least 6 conditions", n_tags >= 6),
        ]
    )
    rows = []
    for lab, m in groups:
        rows.append(
            f"{lab} & {int(m.sum())} & {int((m & (tags.split == 'train')).sum())} & {int((m & (tags.split == 'test')).sum())} "
            f"& {100 * m.mean():.1f} \\\\"
        )
    return "\n".join(rows) + "\n"


def geometry(measurements: pd.DataFrame) -> str:
    valid = measurements[measurements.geometry_valid == 1]
    cols = [
        ("area_px2", "Area (px$^2$)"),
        ("area_pct_image", "Area (\\% of frame)"),
        ("equivalent_diameter_px", "Equal-area diameter (px)"),
        ("perimeter_px", "Perimeter (px)"),
        ("compactness", "Compactness"),
        ("bbox_aspect_ratio", "Box aspect ratio"),
    ]
    rows = []
    for col, lab in cols:
        x = valid[col].astype(float)
        q1, q3 = np.percentile(x, [25, 75])
        rows.append(
            f"{lab} & {x.mean():,.2f} & {x.std():,.2f} & {x.median():,.2f} & {q1:,.2f}--{q3:,.2f} \\\\"
        )
    return "\n".join(rows) + "\n"


def recorded_runs(tags: pd.DataFrame) -> str:
    dom = dict(zip(tags.image_name, tags.domain_yellow_copper, strict=False))
    rows = []
    for key, lab in RECORDED_RUNS:
        f = [
            g
            for g in glob.glob(str(RECORDED / "metrics_*.csv"))
            if key in g and not (key == "medsam_all.csv" and "tiled" in g)
        ][0]
        d = pd.read_csv(f)
        d = d[d.split == "val"].copy()
        n = d.tp + d.tn + d.fp + d.fn
        p = d.tp + d.fn
        d["ref"] = 2 * p / (n + p)
        d["share"] = (d.tp + d.fp) / n
        d["colour"] = d.image_file.map(dom)
        cells = []
        for c in (1, 0):
            x = d[d.colour == c]
            cells += [
                f"{100 * x.dice.mean():.2f}",
                f"{100 * x.ref.mean():.2f}",
                f"{100 * x.share.mean():.1f}",
            ]
        rows.append(
            f"{lab} & "
            + " & ".join(cells)
            + f" & {100 * d.dice.mean():.2f} & {int((d.dice > d.ref).sum())} \\\\"
        )
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=WORK / "statistics" / "froth_measurements.csv",
        help="froth_measurements.csv written by scripts/dataset_statistics.py",
    )
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    tags = pd.read_csv(TAGS)
    (TABLES / "tag_counts.tex").write_text(tag_counts(tags))
    (TABLES / "geometry.tex").write_text(geometry(pd.read_csv(args.measurements)))
    (TABLES / "recorded_runs.tex").write_text(recorded_runs(tags))
    print(
        "written:", TABLES / "tag_counts.tex", TABLES / "geometry.tex", TABLES / "recorded_runs.tex"
    )


if __name__ == "__main__":
    main()
