"""MediaPipe Holistic wrapper (classic solutions OR Tasks API)."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK_MODEL = ROOT / "cache" / "holistic_landmarker.task"
TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def _ensure_task_model() -> Path:
    TASK_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if TASK_MODEL.exists() and TASK_MODEL.stat().st_size > 1_000_000:
        return TASK_MODEL
    print(f"[holistic] downloading Tasks model -> {TASK_MODEL}")
    urllib.request.urlretrieve(TASK_URL, TASK_MODEL)
    return TASK_MODEL


def _list_to_landmark_obj(landmarks) -> Optional[Any]:
    if not landmarks:
        return None
    return SimpleNamespace(landmark=list(landmarks))


class HolisticSession:
    """Unified .process(rgb) -> pose/left_hand/right_hand/face landmarks."""

    def __init__(self, model_complexity: int = 0):
        self._mode = None
        self._impl = None
        self._timestamp_ms = 0
        self._open(model_complexity)

    def _open(self, model_complexity: int) -> None:
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self._impl = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._mode = "solutions"
                return
        except Exception as e:
            print(f"[holistic] solutions unavailable ({e}); trying Tasks API")

        from mediapipe.tasks.python.core import base_options as base_options_module
        from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode

        model_path = str(_ensure_task_model())
        options = HolisticLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode.VisionTaskRunningMode.VIDEO,
        )
        self._impl = HolisticLandmarker.create_from_options(options)
        self._mode = "tasks"

    def process(self, rgb: np.ndarray) -> Any:
        if self._mode == "solutions":
            return self._impl.process(rgb)

        import mediapipe as mp

        # Tasks expects mp.Image
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self._timestamp_ms += 33
        result = self._impl.detect_for_video(img, self._timestamp_ms)
        return SimpleNamespace(
            pose_landmarks=_list_to_landmark_obj(result.pose_landmarks),
            left_hand_landmarks=_list_to_landmark_obj(result.left_hand_landmarks),
            right_hand_landmarks=_list_to_landmark_obj(result.right_hand_landmarks),
            face_landmarks=_list_to_landmark_obj(result.face_landmarks),
        )

    def close(self) -> None:
        if self._impl is None:
            return
        try:
            self._impl.close()
        except Exception:
            pass
        self._impl = None
