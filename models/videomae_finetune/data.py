"""Fine-tune a pretrained VideoMAE (Hugging Face) on ISL isolated words."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def sample_video_frames(path: str, num_frames: int = 16, size: int = 224) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB frames, uniformly sampled."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # read all
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            return np.zeros((num_frames, size, size, 3), dtype=np.uint8)
        total = len(frames)
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        out = []
        for i in idxs:
            rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(rgb, (size, size)))
        return np.stack(out, 0)

    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    wanted = set(idxs.tolist())
    got: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            got[i] = cv2.resize(rgb, (size, size))
        i += 1
    cap.release()
    seq = []
    last = np.zeros((size, size, 3), dtype=np.uint8)
    for i in idxs:
        if i in got:
            last = got[i]
        seq.append(last)
    return np.stack(seq, 0)


class VideoClipDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], num_frames: int = 16, size: int = 224):
        self.paths = paths
        self.labels = labels
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = sample_video_frames(self.paths[idx], self.num_frames, self.size)
        # list of PIL-like arrays; processor expects list of frames or video
        return frames, self.labels[idx]


def make_collate(processor):
    def _collate(batch):
        videos, labels = zip(*batch)
        # each video: (T,H,W,3) -> list of frames for VideoMAEImageProcessor
        videos_as_lists = [list(v) for v in videos]
        inputs = processor(videos_as_lists, return_tensors="pt")
        labels_t = torch.tensor(labels, dtype=torch.long)
        return inputs, labels_t

    return _collate
