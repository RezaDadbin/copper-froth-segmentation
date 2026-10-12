"""Small shared file helpers and run provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and not fields:
        raise ValueError("Empty CSV needs explicit column names")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def provenance(config, manifest, **extra):
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "Pillow",
                 "matplotlib", "scipy", "segment-anything", "segment-anything-hq",
                 "segmentation-models-pytorch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    code_root = Path(__file__).resolve().parent
    code_hashes = {p.relative_to(code_root).as_posix(): sha256(p)
                   for p in sorted(code_root.rglob("*.py"))}
    external_hashes = {}
    external_root = config["model"].get("external_root")
    if external_root:
        external = Path(external_root)
        external_hashes = {p.relative_to(external).as_posix(): sha256(p)
                           for p in sorted((external / "net/sbsnet").rglob("*.py"))}
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "packages": packages,
            "config": config, "manifest_sha256": sha256(manifest),
            "implementation_sha256": code_hashes, "external_implementation_sha256": external_hashes,
            "source_test_role": "validation_used_for_selection_and_reporting",
            "historical_csv_reproduction_claimed": False, **extra}
