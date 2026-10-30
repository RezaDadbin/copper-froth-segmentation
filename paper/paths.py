"""Locations used by the paper scripts.

The images, annotations and fine-tuned checkpoints are private and are not distributed. Point the environment
variables below at your own copies; the defaults assume the private folders sit next to this repository.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


DATA = _path(
    "FROTH_DATA", REPO.parent / "Dataset"
)  # train1/ and test1/: images (.tif/.tiff/.png) + LabelMe JSON
TAGS = _path(
    "FROTH_TAGS", DATA.parent / "tags.csv"
)  # split, image_name, domain flag and the nine condition tags
RECORDED = _path(
    "FROTH_RECORDED", DATA.parent / "models"
)  # per-image metric CSVs saved by the original training runs
SAVED_CNN = _path(
    "FROTH_SAVED_CNN", DATA.parent / "saved_cnn"
)  # saved U-Net and SBSNet output maps (ten colour images)
WORK = _path("FROTH_WORK", REPO / "outputs")  # inference outputs, metrics and analysis results
RUN = WORK / "out"  # one folder per model/route, written by paper.inference
CKPT = _path("FROTH_CKPT", WORK / "ckpt")  # fine-tuned and public model weights
ROBUSTSAM = _path(
    "FROTH_ROBUSTSAM", WORK / "RobustSAM"
)  # clone of the official RobustSAM repository
FIGURES = _path("FROTH_FIGURES", WORK / "figures")
SUPP = FIGURES / "supplementary"
TABLES = _path("FROTH_TABLES", WORK / "tables")
REVIEW = REPO / "paper" / "review"  # cleaned literature-review records and search protocol
SELECTION = (
    REPO / "paper" / "figures" / "selection.json"
)  # image selection for the earlier illustration pages
