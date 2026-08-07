"""Load trained landmark models from weights/."""
from __future__ import annotations

import json
import sys
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from common.landmarks import FEAT_DIM, N_HAND, N_POSE, landmarks_from_frames  # noqa: E402
from landmark_tcn.model import LandmarkTCN  # noqa: E402
from mediapipe_transformer.model import LandmarkTransformer  # noqa: E402


@dataclass
class Prediction:
    word: str
    confidence: float
    probs: Optional[np.ndarray] = None
    hands_detected: bool = True


class Predictor(ABC):
    name: str

    @abstractmethod
    def predict_frames(self, frames_bgr: list[np.ndarray]) -> Prediction:
        ...


def _ensure_extracted(weights_dir: Path, model_name: str) -> Path:
    """Return folder with model.pt; unzip from weights/<name>.zip if needed."""
    out = weights_dir / model_name
    ckpt = out / "model.pt"
    if ckpt.exists():
        return out
    zpath = weights_dir / f"{model_name}.zip"
    if zpath.exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out)
        # handle zip that contains a nested single folder
        if not ckpt.exists():
            subs = [p for p in out.iterdir() if p.is_dir()]
            for sub in subs:
                if (sub / "model.pt").exists():
                    for item in sub.iterdir():
                        target = out / item.name
                        if not target.exists():
                            item.rename(target)
                    break
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing {ckpt}. Place model.pt under weights/{model_name}/ "
            f"or provide weights/{model_name}.zip"
        )
    return out


def _load_labels(folder: Path, weights_dir: Path) -> dict[int, str]:
    for candidate in (folder / "labels.json", weights_dir / "labels.json"):
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            raw = data.get("id_to_word") or {str(v): k for k, v in data.get("word_to_id", {}).items()}
            return {int(k): str(v) for k, v in raw.items()}
    raise FileNotFoundError("labels.json not found next to weights")


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandmarkPredictor(Predictor):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        id_to_word: dict[int, str],
        num_frames: int,
        device: torch.device,
        holistic=None,
    ):
        self.name = name
        self.model = model.eval()
        self.id_to_word = id_to_word
        self.num_frames = num_frames
        self.device = device
        self.holistic = holistic

    @torch.inference_mode()
    def predict_frames(self, frames_bgr: list[np.ndarray]) -> Prediction:
        seq = landmarks_from_frames(
            frames_bgr,
            num_frames=self.num_frames,
            holistic=self.holistic,
        )
        # Hand landmarks are already part of the model input. Reading their
        # presence here adds pause metadata without changing inference data.
        hand_start = N_POSE * 3
        hand_end = (N_POSE + (2 * N_HAND)) * 3
        # A pause is based on the newest quarter of the rolling window. Older
        # frames may still contain the sign that was just completed.
        recent_frame_count = max(3, self.num_frames // 4)
        recent_hands = seq[-recent_frame_count:, hand_start:hand_end]
        hands_detected = bool(np.any(recent_hands != 0))
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, F)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        idx = int(probs.argmax())
        word = self.id_to_word.get(idx, str(idx))
        return Prediction(
            word=word,
            confidence=float(probs[idx]),
            probs=probs,
            hands_detected=hands_detected,
        )


def load_predictor(
    model_name: str,
    weights_dir: Path,
    device: str = "auto",
    holistic=None,
) -> LandmarkPredictor:
    folder = _ensure_extracted(weights_dir, model_name)
    blob = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    meta = blob.get("meta") or {}
    num_classes = int(meta.get("num_classes", 39))
    num_frames = int(meta.get("num_frames", 30))
    id_to_word = _load_labels(folder, weights_dir)
    dev = _resolve_device(device)

    if model_name == "landmark_tcn":
        model = LandmarkTCN(FEAT_DIM, num_classes)
    elif model_name == "mediapipe_transformer":
        model = LandmarkTransformer(
            feat_dim=FEAT_DIM,
            num_classes=num_classes,
            d_model=int(meta.get("d_model", 128)),
            nhead=int(meta.get("nhead", 4)),
            num_layers=int(meta.get("layers", 3)),
            max_len=num_frames,
        )
    else:
        raise ValueError(f"Unsupported live model: {model_name}")

    model.load_state_dict(blob["state_dict"])
    model.to(dev)
    return LandmarkPredictor(model_name, model, id_to_word, num_frames, dev, holistic=holistic)
