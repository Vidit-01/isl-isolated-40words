"""Paths and runtime defaults for the live app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
MODELS_DIR = ROOT / "models"


@dataclass
class AppConfig:
    model: str = "mediapipe_transformer"  # or landmark_tcn
    weights_dir: Path = WEIGHTS_DIR
    camera_index: int = 0
    buffer_seconds: float = 2.0
    target_fps: float = 15.0
    conf_threshold: float = 0.22
    speak_cooldown_s: float = 2.0
    mirror: bool = True
    device: str = "auto"  # auto | cpu | cuda
    window_name: str = "ISL Live"


SUPPORTED_MODELS = ("mediapipe_transformer", "landmark_tcn")
