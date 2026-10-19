#!/usr/bin/env python3
"""Convenience wrapper for local source-tree execution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xray_3pin_detector.cli import main


if __name__ == "__main__":
    main()
