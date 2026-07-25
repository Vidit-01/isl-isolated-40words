"""
T4-optimized training for all three ISL models.

Presets tuned for NVIDIA T4 (~16GB): FP16 AMP, moderate batches, pin_memory.
VideoMAE defaults to frozen backbone to fit VRAM. Saves best val weights and
test metrics under models/_weights/<model_name>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from common import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    WEIGHTS_DIR,
    build_label_maps,
    configure_t4,
    load_metadata,
    save_json,
    set_seed,
    stratified_split,
)
from common.engine import evaluate, save_weights, train_one_epoch  # noqa: E402
from common.landmarks import FEAT_DIM  # noqa: E402
from landmark_tcn.model import LandmarkTCN  # noqa: E402
from mediapipe_transformer.dataset import LandmarkDataset, collate  # noqa: E402
from mediapipe_transformer.model import LandmarkTransformer  # noqa: E402


# --------------- T4 hyperparameter presets (~16GB) ---------------
PRESETS = {
    "landmark_tcn": {
        "epochs": 80,
        "batch_size": 64,
        "lr": 1e-3,
        "num_frames": 30,
        "num_workers": 4,
    },
    "mediapipe_transformer": {
        "epochs": 60,
        "batch_size": 32,
        "lr": 1e-3,
        "num_frames": 30,
        "d_model": 128,
        "layers": 3,
        "nhead": 4,
        "num_workers": 4,
    },
    "videomae_finetune": {
        "epochs": 20,
        "batch_size": 2,
        "lr": 5e-5,
        "num_frames": 16,
        "size": 224,
        "num_workers": 4,
        "model_name": "MCG-NJU/videomae-base",
        "freeze_backbone": True,  # T4: head-only by default to fit 16GB
    },
}


def class_weights(train_y, num_classes: int, device) -> torch.Tensor:
    import pandas as pd

    counts = pd.Series(train_y).value_counts().reindex(range(num_classes), fill_value=1)
    w = 1.0 / torch.tensor(counts.values, dtype=torch.float32)
    return (w / w.sum() * num_classes).to(device)


def make_landmark_loaders(train_df, val_df, test_df, num_frames, batch_size, num_workers):
    cache = CACHE_DIR / f"landmarks_T{num_frames}"
    train_ds = LandmarkDataset(
        train_df["abs_path"].tolist(), train_df["y"].tolist(), cache, num_frames, True
    )
    val_ds = LandmarkDataset(
        val_df["abs_path"].tolist(), val_df["y"].tolist(), cache, num_frames, False
    )
    test_ds = LandmarkDataset(
        test_df["abs_path"].tolist(), test_df["y"].tolist(), cache, num_frames, False
    )
    kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )


def train_landmark_model(name: str, model: nn.Module, loaders, args, labels, device, extra_meta=None):
    train_loader, val_loader, test_loader = loaders
    ys = train_loader.dataset.labels
    criterion = nn.CrossEntropyLoss(weight=class_weights(ys, labels["num_classes"], device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, criterion, device, scaler)
        va = evaluate(model, val_loader, criterion, device)
        sched.step()
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
        }
        history.append(row)
        print(
            f"[{name}] epoch {epoch:03d}  "
            f"train {tr['loss']:.4f}/{tr['acc']:.3f}  val {va['loss']:.4f}/{va['acc']:.3f}"
        )
        if va["acc"] >= best_acc:
            best_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, criterion, device)
    print(f"[{name}] TEST acc={test['acc']:.3f} loss={test['loss']:.4f} n={test['n']}")

    ckpt_dir = CHECKPOINT_DIR / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "best_val_acc": best_acc}, ckpt_dir / "best.pt")

    meta = {
        "model": name,
        "feat_dim": FEAT_DIM,
        "num_classes": labels["num_classes"],
        "num_frames": args.num_frames,
        "best_val_acc": best_acc,
        "preset": "t4",
    }
    if extra_meta:
        meta.update(extra_meta)
    weights_dir = WEIGHTS_DIR / name
    save_weights(
        model,
        weights_dir,
        meta=meta,
        labels=labels,
        history=history,
        test_metrics=test,
    )
    return test


def train_videomae(train_df, val_df, test_df, labels, args, device):
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
    from videomae_finetune.data import VideoClipDataset, make_collate

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForVideoClassification.from_pretrained(
        args.model_name,
        num_labels=labels["num_classes"],
        ignore_mismatched_sizes=True,
    )
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False
        print("[videomae] frozen backbone (T4 default)")
    else:
        print("[videomae] full fine-tune (watch VRAM on T4)")

    model.to(device)
    collate_fn = make_collate(processor)
    kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(
        VideoClipDataset(train_df["abs_path"].tolist(), train_df["y"].tolist(), args.num_frames, args.size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kw,
    )
    val_loader = DataLoader(
        VideoClipDataset(val_df["abs_path"].tolist(), val_df["y"].tolist(), args.num_frames, args.size),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **kw,
    )
    test_loader = DataLoader(
        VideoClipDataset(test_df["abs_path"].tolist(), test_df["y"].tolist(), args.num_frames, args.size),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **kw,
    )

    def forward_fn(m, batch, criterion, dev, train=True):
        inputs, y = batch
        inputs = {k: v.to(dev, non_blocking=True) for k, v in inputs.items()}
        y = y.to(dev, non_blocking=True)
        out = m(**inputs)
        logits = out.logits
        loss = criterion(logits, y)
        return logits, y, loss

    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["y"].tolist(), labels["num_classes"], device))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, criterion, device, scaler, forward_fn)
        va = evaluate(model, val_loader, criterion, device, forward_fn)
        sched.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
            }
        )
        print(
            f"[videomae] epoch {epoch:03d}  "
            f"train {tr['loss']:.4f}/{tr['acc']:.3f}  val {va['loss']:.4f}/{va['acc']:.3f}"
        )
        if va["acc"] >= best_acc:
            best_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, criterion, device, forward_fn)
    print(f"[videomae] TEST acc={test['acc']:.3f} loss={test['loss']:.4f} n={test['n']}")

    out_hf = WEIGHTS_DIR / "videomae_finetune" / "hf"
    out_hf.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_hf)
    processor.save_pretrained(out_hf)
    save_weights(
        model,
        WEIGHTS_DIR / "videomae_finetune",
        meta={
            "model": "videomae_finetune",
            "hf_name": args.model_name,
            "num_classes": labels["num_classes"],
            "num_frames": args.num_frames,
            "best_val_acc": best_acc,
            "freeze_backbone": args.freeze_backbone,
            "preset": "t4",
            "hf_dir": str(out_hf),
        },
        labels=labels,
        history=history,
        test_metrics=test,
    )
    return test


def parse_args():
    ap = argparse.ArgumentParser(description="Train ISL models with NVIDIA T4 presets")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["landmark_tcn", "mediapipe_transformer", "videomae_finetune"],
        choices=["landmark_tcn", "mediapipe_transformer", "videomae_finetune"],
    )
    ap.add_argument("--min-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--freeze-backbone", action="store_true", help="Force freeze VideoMAE backbone")
    ap.add_argument("--unfreeze", action="store_true", help="Full VideoMAE fine-tune (tight on T4 16GB)")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


def apply_preset(name: str, cli) -> argparse.Namespace:
    p = dict(PRESETS[name])
    ns = argparse.Namespace(**p)
    if cli.epochs is not None:
        ns.epochs = cli.epochs
    if cli.batch_size is not None:
        ns.batch_size = cli.batch_size
    if cli.lr is not None:
        ns.lr = cli.lr
    if cli.num_frames is not None:
        ns.num_frames = cli.num_frames
    if cli.num_workers is not None:
        ns.num_workers = cli.num_workers
    if name == "videomae_finetune":
        if cli.unfreeze:
            ns.freeze_backbone = False
        elif cli.freeze_backbone:
            ns.freeze_backbone = True
    return ns


def main() -> None:
    cli = parse_args()
    set_seed(cli.seed)
    device = torch.device("cpu") if cli.cpu else configure_t4()

    df = load_metadata(min_clips=cli.min_clips)
    w2i, i2w = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    train_df, val_df, test_df = stratified_split(df, seed=cli.seed)
    labels = {
        "word_to_id": w2i,
        "id_to_word": {str(k): v for k, v in i2w.items()},
        "num_classes": len(w2i),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(labels, WEIGHTS_DIR / "labels.json")
    print(
        f"split train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={len(w2i)}"
    )

    results = {}
    for name in cli.models:
        args = apply_preset(name, cli)
        print(f"\n=== Training {name} (T4 preset) ===")
        print(args)

        if name == "landmark_tcn":
            loaders = make_landmark_loaders(
                train_df, val_df, test_df, args.num_frames, args.batch_size, args.num_workers
            )
            model = LandmarkTCN(FEAT_DIM, len(w2i)).to(device)
            results[name] = train_landmark_model(name, model, loaders, args, labels, device)

        elif name == "mediapipe_transformer":
            loaders = make_landmark_loaders(
                train_df, val_df, test_df, args.num_frames, args.batch_size, args.num_workers
            )
            model = LandmarkTransformer(
                feat_dim=FEAT_DIM,
                num_classes=len(w2i),
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.layers,
                max_len=args.num_frames,
            ).to(device)
            results[name] = train_landmark_model(
                name,
                model,
                loaders,
                args,
                labels,
                device,
                extra_meta={"d_model": args.d_model, "layers": args.layers, "nhead": args.nhead},
            )

        elif name == "videomae_finetune":
            results[name] = train_videomae(train_df, val_df, test_df, labels, args, device)

    summary = {k: {"test_acc": v["acc"], "test_loss": v["loss"], "n": v["n"]} for k, v in results.items()}
    save_json(summary, WEIGHTS_DIR / "summary.json")
    print("\n=== DONE ===")
    print(summary)
    print(f"Weights: {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
