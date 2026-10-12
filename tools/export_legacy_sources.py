#!/usr/bin/env python3
"""Export every original notebook/script as an import-safe historical .py file.

Historical transcripts preserve all cells, including alternate drivers and
widget rendering. Shell/IPython commands are comments; execution is deliberately
separate from the maintained scripts/ entry points.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import textwrap


def export_sources(source: Path, output: Path, manifest_path: Path):
    rows = []
    for path in sorted(source.rglob("*")):
        if path.suffix not in {".ipynb", ".py"} or not path.is_file():
            continue
        relative = path.relative_to(source)
        parts = [re.sub(r"[^A-Za-z0-9_]+", "_", p.replace("+", "_plus")).strip("_").lower()
                 for p in relative.with_suffix("").parts]
        if path.suffix == ".ipynb":
            parts[-1] += "_notebook"
        target = output.joinpath(*parts).with_suffix(".py")
        if target.exists():
            raise FileExistsError(target)
        if path.suffix == ".ipynb":
            cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
        else:
            cells = [{"cell_type": "code", "source": path.read_text(encoding="utf-8")}]
        body, disabled = [], []
        for index, cell in enumerate(cells):
            kind = cell.get("cell_type", "code")
            content = cell.get("source", "")
            content = "".join(content) if isinstance(content, list) else content
            body.append(f"# %% Original cell {index} ({kind})\n")
            if kind != "code":
                body.extend("# " + line + "\n" for line in content.splitlines())
            else:
                for line_number, line in enumerate(content.splitlines(), 1):
                    stripped = line.lstrip()
                    if stripped.startswith(("!", "%", "pip install ", "pip uninstall ",
                                            "get_ipython(", "from __future__ import")):
                        disabled.append(f"cell {index}, line {line_number}")
                        line = line[:len(line) - len(stripped)] + "pass  # IPython/setup disabled: " + stripped
                    elif re.match(r"from\s+[\w.]+\s+import\s+\*", stripped):
                        # Python forbids import-star inside functions. Keep it
                        # explicit and delayed in the historical transcript.
                        line = line[:len(line) - len(stripped)] + f"exec({stripped!r}, globals())"
                    body.append(line + "\n")
            body.append("\n")
        # An empty/comment-only notebook still produces a valid Python file.
        body.append("pass\n")
        text = ('"""Historical source export; maintained entry points live in scripts/.\n\n'
                f'Original: codes/{relative.as_posix()}\n'
                'Importing this file does not train, install packages, rename files or render.\n'
                'historical_experiment() is a manual research transcript, not a supported pipeline.\n'
                '"""\n\n\ndef historical_experiment():\n' +
                textwrap.indent("".join(body), "    ") +
                '\n\nif __name__ == "__main__":\n'
                '    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")\n')
        compile(text, str(target), "exec")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        rows.append({"source_path": relative.as_posix(),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "python_path": target.relative_to(manifest_path.parent.parent).as_posix(),
                     "python_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                     "original_cells": len(cells),
                     "commented_setup_lines": "; ".join(disabled)})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "legacy")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "docs/python_conversion.csv")
    args = parser.parse_args()
    rows = export_sources(args.source.resolve(), args.output.resolve(), args.manifest.resolve())
    print(f"Exported {len(rows)} source files; syntax checked; original files unchanged.")


if __name__ == "__main__":
    main()
