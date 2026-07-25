"""Evaluate saved weights on the held-out test split and rewrite test_metrics.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from common import (  # noqa: E402
    CACHE_DIR,
    WEIGHTS_DIR,
    build_label_maps,
    configure_t4,
    load_metadata,
    save_json,
    set_seed,
    stratified_split,
)
from common.engine import evaluate  # noqa: E402
from common.landmarks import FEAT_DIM  # noqa: E402
from landmark_tcn.model import LandmarkTCN  # noqa: E402
from mediapipe_transformer.dataset import LandmarkDataset, collate  # noqa: E402
from mediapipe_transformer.model import LandmarkTransformer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ISL models (T4-friendly defaults)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["landmark_tcn", "mediapipe_transformer", "videomae_finetune"],
    )
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else configure_t4()
    df = load_metadata(min_clips=args.min_clips)
    w2i, _ = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    _, _, test_df = stratified_split(df, seed=args.seed)

    results = {}
    for name in args.models:
        wdir = WEIGHTS_DIR / name
        ckpt = wdir / "model.pt"
        if not ckpt.exists() and name != "videomae_finetune":
            print(f"SKIP {name}: missing {ckpt}")
            continue

        if name in ("landmark_tcn", "mediapipe_transformer"):
            meta_frames = args.num_frames
            if ckpt.exists():
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
                meta = blob.get("meta", {})
                meta_frames = int(meta.get("num_frames", args.num_frames))
                cache = CACHE_DIR / f"landmarks_T{meta_frames}"
                num_classes = int(meta["num_classes"])
                if name == "landmark_tcn":
                    model = LandmarkTCN(FEAT_DIM, num_classes)
                else:
                    model = LandmarkTransformer(
                        FEAT_DIM,
                        num_classes,
                        d_model=int(meta.get("d_model", 128)),
                        nhead=int(meta.get("nhead", 4)),
                        num_layers=int(meta.get("layers", 3)),
                        max_len=meta_frames,
                    )
                model.load_state_dict(blob["state_dict"])
            else:
                print(f"SKIP {name}")
                continue
            model.to(device).eval()
            ds = LandmarkDataset(
                test_df["abs_path"].tolist(), test_df["y"].tolist(), cache, meta_frames, False
            )
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
            criterion = nn.CrossEntropyLoss()
            metrics = evaluate(model, loader, criterion, device)
        else:
            from transformers import AutoModelForVideoClassification, AutoImageProcessor
            from videomae_finetune.data import VideoClipDataset, make_collate

            hf_dir = wdir / "hf"
            if not hf_dir.exists():
                print(f"SKIP videomae: missing {hf_dir}")
                continue
            processor = AutoImageProcessor.from_pretrained(hf_dir)
            model = AutoModelForVideoClassification.from_pretrained(hf_dir).to(device)
            collate_fn = make_collate(processor)
            ds = VideoClipDataset(test_df["abs_path"].tolist(), test_df["y"].tolist(), 16, 224)
            # T4: keep VideoMAE eval batch small
            loader = DataLoader(
                ds, batch_size=min(2, args.batch_size), shuffle=False, collate_fn=collate_fn
            )

            def forward_fn(m, batch, criterion, dev, train=True):
                inputs, y = batch
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                y = y.to(dev)
                logits = m(**inputs).logits
                return logits, y, criterion(logits, y)

            metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device, forward_fn)

        save_json(
            {k: v for k, v in metrics.items()},
            wdir / "test_metrics.json",
        )
        results[name] = {"test_acc": metrics["acc"], "test_loss": metrics["loss"], "n": metrics["n"]}
        print(f"{name}: test_acc={metrics['acc']:.3f} n={metrics['n']}")

    save_json(results, WEIGHTS_DIR / "eval_summary.json")
    print("Wrote", WEIGHTS_DIR / "eval_summary.json")


if __name__ == "__main__":
    main()
