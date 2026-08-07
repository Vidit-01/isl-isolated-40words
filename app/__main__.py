"""CLI entry: python -m app"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import SUPPORTED_MODELS, WEIGHTS_DIR, AppConfig
from .pipeline import run_live


def main() -> None:
    ap = argparse.ArgumentParser(description="Live ISL word recognition (camera + TTS)")
    ap.add_argument(
        "--model",
        default="mediapipe_transformer",
        choices=list(SUPPORTED_MODELS),
        help="Which weights folder / zip to load",
    )
    ap.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--buffer-seconds", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.35, help="Min confidence to auto-speak")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig(
        model=args.model,
        weights_dir=args.weights_dir,
        camera_index=args.camera,
        buffer_seconds=args.buffer_seconds,
        conf_threshold=args.conf,
        device=args.device,
        mirror=not args.no_mirror,
    )
    run_live(cfg)


if __name__ == "__main__":
    main()
