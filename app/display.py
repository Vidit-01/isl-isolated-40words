"""OpenCV overlay UI."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .predictor import Prediction


def draw_overlay(
    frame: np.ndarray,
    *,
    model_name: str,
    buffer_fill: float,
    pred: Optional[Prediction],
    status: str,
    conf_threshold: float,
    sentence: str = "",
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # top banner
    cv2.rectangle(out, (0, 0), (w, 78), (20, 20, 20), -1)
    cv2.putText(
        out,
        f"ISL Live  |  model: {model_name}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        status,
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    if sentence:
        cv2.rectangle(out, (0, 78), (w, 112), (30, 30, 30), -1)
        shown = sentence if len(sentence) <= 52 else f"{sentence[:49]}..."
        cv2.putText(
            out,
            f"Sentence: {shown}",
            (16, 101),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 230, 180),
            1,
            cv2.LINE_AA,
        )

    # buffer bar
    bar_w = int((w - 32) * min(max(buffer_fill, 0.0), 1.0))
    cv2.rectangle(out, (16, h - 28), (w - 16, h - 14), (60, 60, 60), -1)
    cv2.rectangle(out, (16, h - 28), (16 + bar_w, h - 14), (80, 200, 120), -1)
    cv2.putText(
        out,
        "buffer",
        (16, h - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if pred is not None:
        color = (80, 220, 120) if pred.confidence >= conf_threshold else (80, 160, 255)
        label = f"{pred.word.upper()}"
        conf = f"{pred.confidence * 100:.0f}%"
        cv2.putText(out, label, (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)
        cv2.putText(out, conf, (16, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # help
    help_lines = "SPACE predict  |  S speak  |  M switch model  |  C clear  |  Q quit"
    cv2.putText(
        out,
        help_lines,
        (16, h - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return out
