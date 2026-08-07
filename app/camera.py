"""Webcam capture helpers."""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import platform

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
        else:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()


class FrameBuffer:
    """Sliding window of recent BGR frames for one prediction window."""

    def __init__(self, maxlen: int):
        self.frames: Deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def clear(self) -> None:
        self.frames.clear()

    def __len__(self) -> int:
        return len(self.frames)

    def as_list(self) -> list[np.ndarray]:
        return list(self.frames)

    @property
    def full(self) -> bool:
        return len(self.frames) >= self.frames.maxlen
