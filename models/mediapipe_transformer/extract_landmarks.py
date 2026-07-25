"""Extract MediaPipe landmarks for all ISL videos (cached .npy)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from common import CACHE_DIR, load_metadata  # noqa: E402
from common.landmarks import load_or_extract  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=30, choices=[30, 60])
    ap.add_argument("--min-clips", type=int, default=1)
    args = ap.parse_args()

    df = load_metadata(min_clips=args.min_clips)
    cache = CACHE_DIR / f"landmarks_T{args.num_frames}"
    print(f"Extracting {len(df)} videos -> {cache}")
    ok, fail = 0, 0
    for path in tqdm(df["abs_path"].tolist(), desc="landmarks"):
        try:
            load_or_extract(path, cache, num_frames=args.num_frames)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAIL {path}: {e}")
    print(f"done ok={ok} fail={fail} feat cache={cache}")


if __name__ == "__main__":
    main()
