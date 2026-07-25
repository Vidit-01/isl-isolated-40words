"""Shared helpers for ISL recognition models."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "ISL_DATASET"
METADATA = DATASET_DIR / "metadata.csv"
CACHE_DIR = ROOT / "models" / "_cache"
CHECKPOINT_DIR = ROOT / "models" / "_checkpoints"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(min_clips: int = 1) -> pd.DataFrame:
    df = pd.read_csv(METADATA)
    df["video_path"] = (
        df["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace(r"^.*?ISL_DATASET/", "", regex=True)
    )
    counts = df.groupby("word").size()
    keep = counts[counts >= min_clips].index
    df = df[df["word"].isin(keep)].copy()
    df["abs_path"] = df["video_path"].map(lambda p: str(DATASET_DIR / p))
    return df.reset_index(drop=True)


def build_label_maps(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    words = sorted(set(words))
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_val_per_class: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ~val_ratio per class; if class has 1 clip, put it in train only."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for word, g in df.groupby("word"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            train_idx.extend(idx.tolist())
            continue
        n_val = max(min_val_per_class, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())
