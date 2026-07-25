"""Lightweight Temporal CNN (TCN) over MediaPipe landmarks.

Rationale (for ~640 clips / 40 classes / low compute):
- Reuses cheap landmark cache (no RGB backbone).
- TCNs often beat Transformers on *small* sequential datasets (less overfit).
- ~100k–300k params, trains on CPU in minutes after landmark extraction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = self.conv1(x)
        if self.pad:
            y = y[:, :, : -self.pad]
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        if self.pad:
            y = y[:, :, : -self.pad]
        y = F.gelu(y)
        y = self.dropout(y)
        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(y + res)


class LandmarkTCN(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        channels: tuple[int, ...] = (64, 64, 128),
        kernel: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = feat_dim
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel=kernel, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        h = x.transpose(1, 2)
        h = self.tcn(h)
        return self.head(h)
