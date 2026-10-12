#!/usr/bin/env python3
"""Paper reporting entry point with explicit inputs and output directory."""

import _bootstrap  # noqa: F401
from froth.reporting.dataset_statistics import main


if __name__ == "__main__":
    raise SystemExit(main())
