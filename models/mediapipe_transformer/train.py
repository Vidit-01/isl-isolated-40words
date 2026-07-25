"""Train MediaPipe landmark Transformer on ISL isolated words."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from common import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    accuracy,
    build_label_maps,
    load_metadata,
    save_json,
    set_seed,
    stratified_split,
)
from common.landmarks import FEAT_DIM  # noqa: E402
from mediapipe_transformer.dataset import LandmarkDataset, collate  # noqa: E402
from mediapipe_transformer.model import LandmarkTransformer  # noqa: E402


def run_epoch(model, loader, opt, criterion, device, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=30, choices=[30, 60])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--min-clips", type=int, default=2, help="Skip ultra-rare classes for stable training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(min_clips=args.min_clips)
    w2i, i2w = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    train_df, val_df = stratified_split(df, seed=args.seed)

    cache = CACHE_DIR / f"landmarks_T{args.num_frames}"
    train_ds = LandmarkDataset(
        train_df["abs_path"].tolist(),
        train_df["y"].tolist(),
        cache,
        num_frames=args.num_frames,
        augment=True,
    )
    val_ds = LandmarkDataset(
        val_df["abs_path"].tolist(),
        val_df["y"].tolist(),
        cache,
        num_frames=args.num_frames,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    model = LandmarkTransformer(
        feat_dim=FEAT_DIM,
        num_classes=len(w2i),
        d_model=args.d_model,
        num_layers=args.layers,
        max_len=args.num_frames,
    ).to(device)

    # class weights for imbalance
    counts = train_df["y"].value_counts().reindex(range(len(w2i)), fill_value=1)
    weights = 1.0 / torch.tensor(counts.values, dtype=torch.float32)
    weights = weights / weights.sum() * len(w2i)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = CHECKPOINT_DIR / "mediapipe_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"word_to_id": w2i, "id_to_word": {str(k): v for k, v in i2w.items()}}, out_dir / "labels.json")

    best = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, criterion, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, opt, criterion, device, False)
        sched.step()
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        )
        print(
            f"epoch {epoch:03d}  train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}"
        )
        if va_acc >= best:
            best = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "feat_dim": FEAT_DIM,
                    "num_classes": len(w2i),
                    "best_val_acc": best,
                },
                out_dir / "best.pt",
            )
    save_json(history, out_dir / "history.json")
    print(f"best val acc={best:.3f}  ckpt={out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
