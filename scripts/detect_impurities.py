#!/usr/bin/env python3
"""Detect and count dark impurities on one froth image or on every image in a folder."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from froth.impurity import load_config, run
from froth.impurity.pipeline import list_images

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single image")
    source.add_argument("--folder", type=Path, help="Folder of images (.png, .jpg, .bmp, .tif)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "impurity.yaml")
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for masks, overlays and results.csv"
    )
    args = parser.parse_args()
    images = [args.image] if args.image else list_images(args.folder)
    if not images:
        parser.error(f"No images found in {args.folder}")
    print("Results:", run(images, args.output, load_config(args.config)))


if __name__ == "__main__":
    main()
