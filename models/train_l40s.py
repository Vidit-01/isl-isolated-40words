"""Deprecated alias — use models/train_t4.py (NVIDIA T4 presets)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    print("NOTE: train_l40s.py is an alias; training uses T4 presets (train_t4.py).")
    target = Path(__file__).resolve().with_name("train_t4.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
