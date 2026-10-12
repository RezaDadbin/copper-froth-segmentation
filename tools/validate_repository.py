#!/usr/bin/env python3
"""Validate Python conversion, configs, documentation and preserved originals.

Uses the standard library only. Functional tests are a separate command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from froth.config import load_config


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def public_files():
    excluded = {".git", "local_archive", "__pycache__", ".venv", ".pytest_cache"}
    return [p for p in ROOT.rglob("*") if p.is_file()
            and not excluded.intersection(p.relative_to(ROOT).parts)
            and not p.is_relative_to(ROOT / "results/generated")
            and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "codes")
    parser.add_argument("--require-archive", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    files = public_files()
    notebooks = [p for p in files if p.suffix == ".ipynb"]
    if notebooks:
        raise AssertionError(f"Public notebook files remain: {notebooks}")
    python = [p for p in files if p.suffix == ".py"]
    for p in python:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
    conversions = csv_rows(ROOT / "docs/python_conversion.csv")
    if len(conversions) != 38 or len({r["source_path"] for r in conversions}) != 38:
        raise AssertionError("Every one of the 38 original code sources must have a unique conversion row")
    for row in conversions:
        target = ROOT / row["python_path"]
        assert target.suffix == ".py" and digest(target) == row["python_sha256"], row
        namespace = runpy.run_path(str(target), run_name="source_validation")
        assert callable(namespace["historical_experiment"]), target
    configs = sorted((ROOT / "configs").glob("*.json"))
    for config in configs:
        load_config(config)
    assert len(configs) == 8
    scripts = sorted(p for p in (ROOT / "scripts").glob("*.py") if not p.name.startswith("_"))
    for script in scripts:
        result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT.parent,
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0 and "usage:" in result.stdout, (script.name, result.stderr)
    links = 0
    for path in (p for p in files if p.suffix == ".md"):
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            assert (path.parent / target).exists(), f"Broken link in {path.relative_to(ROOT)}: {target}"
            links += 1
    originals = csv_rows(ROOT / "docs/source_manifest.csv")
    archive = ROOT / "local_archive/original_codes"
    if args.require_archive and not archive.is_dir():
        raise FileNotFoundError(archive)
    counts = {"archive_files_verified": 0, "original_files_verified": 0}
    for row in originals:
        for label, folder in (("archive", archive), ("original", args.source_root)):
            if folder.is_dir():
                assert digest(folder / row["source_path"]) == row["source_sha256"], (label, row["source_path"])
                counts[f"{label}_files_verified"] += 1
        if row["public_path"]:
            assert (ROOT / row["public_path"]).is_file(), row["public_path"]
    for requirements in (ROOT / "requirements").glob("*.txt"):
        for line in requirements.read_text().splitlines():
            if line.startswith("-r "):
                assert (requirements.parent / line[3:].strip()).is_file()
    result = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "public_python_files_compiled": len(python), "public_notebooks": len(notebooks),
              "historical_sources_exported_and_import_checked": len(conversions),
              "original_notebook_sources": sum(r["source_path"].endswith(".ipynb") for r in conversions),
              "configs_checked": len(configs), "script_help_checked": len(scripts),
              "relative_document_links_checked": links, "source_files_accounted_for": len(originals), **counts,
              "functional_tests_are_separate": True}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
