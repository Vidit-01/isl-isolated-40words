"""Dataset + landmark cache builder for MediaPipe models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.landmarks import FEAT_DIM, load_or_extract  # noqa: E402


class LandmarkDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        cache_dir: Path,
        num_frames: int = 30,
        augment: bool = False,
        require_cache: bool = True,
    ):
        self.paths = paths
        self.labels = labels
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.augment = augment
        self.require_cache = require_cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr = load_or_extract(
            self.paths[idx],
            self.cache_dir,
            self.num_frames,
            require_cache=self.require_cache,
        )
        x = arr.astype(np.float32)
        if self.augment:
            # light noise + random temporal shift (circular)
            if np.random.rand() < 0.5:
                x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            if np.random.rand() < 0.5:
                shift = np.random.randint(0, self.num_frames)
                x = np.roll(x, shift, axis=0)
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)
