"""Train / fine-tune VideoMAE for ISL word classification (low-compute defaults)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForVideoClassification, AutoImageProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from common import (  # noqa: E402
    CHECKPOINT_DIR,
    accuracy,
    build_label_maps,
    load_metadata,
    save_json,
    set_seed,
    stratified_split,
)
from videomae_finetune.data import VideoClipDataset, make_collate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        default="MCG-NJU/videomae-base",
        help="HF VideoMAE checkpoint (base is fine; use -small if available for lower compute)",
    )
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--unfreeze", action="store_true", help="Full fine-tune (more compute)")
    ap.add_argument("--min-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.unfreeze:
        args.freeze_backbone = False

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(min_clips=args.min_clips)
    w2i, i2w = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    train_df, val_df, _test_df = stratified_split(df, seed=args.seed)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForVideoClassification.from_pretrained(
        args.model_name,
        num_labels=len(w2i),
        ignore_mismatched_sizes=True,
    )

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False
        print("Frozen VideoMAE backbone; training classifier head only (low compute).")

    model.to(device)

    train_ds = VideoClipDataset(
        train_df["abs_path"].tolist(), train_df["y"].tolist(), args.num_frames, args.size
    )
    val_ds = VideoClipDataset(
        val_df["abs_path"].tolist(), val_df["y"].tolist(), args.num_frames, args.size
    )
    collate = make_collate(processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    counts = train_df["y"].value_counts().reindex(range(len(w2i)), fill_value=1)
    weights = 1.0 / torch.tensor(counts.values, dtype=torch.float32)
    weights = (weights / weights.sum() * len(w2i)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    out_dir = CHECKPOINT_DIR / "videomae_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"word_to_id": w2i, "id_to_word": {str(k): v for k, v in i2w.items()}}, out_dir / "labels.json")

    best = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        for inputs, y in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(**inputs)
            logits = out.logits
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc += accuracy(logits, y) * bs
            n += bs
        tr_loss /= max(n, 1)
        tr_acc /= max(n, 1)

        model.eval()
        va_loss, va_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, y in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                y = y.to(device)
                out = model(**inputs)
                logits = out.logits
                loss = criterion(logits, y)
                bs = y.size(0)
                va_loss += loss.item() * bs
                va_acc += accuracy(logits, y) * bs
                n += bs
        va_loss /= max(n, 1)
        va_acc /= max(n, 1)
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        )
        print(f"epoch {epoch:03d}  train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc >= best:
            best = va_acc
            model.save_pretrained(out_dir / "best_hf")
            processor.save_pretrained(out_dir / "best_hf")
            torch.save({"best_val_acc": best, "args": vars(args)}, out_dir / "best_meta.pt")

    save_json(history, out_dir / "history.json")
    print(f"best val acc={best:.3f}  dir={out_dir / 'best_hf'}")


if __name__ == "__main__":
    main()
