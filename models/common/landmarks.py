"""MediaPipe Holistic landmark extraction + positional normalization."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Holistic landmark counts
N_POSE = 33
N_HAND = 21
N_FACE = 468
FEAT_DIM = (N_POSE + 2 * N_HAND + N_FACE) * 3  # xyz only


def _lm_to_array(landmarks, n: int) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def sample_frame_indices(n_frames: int, target: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(target, dtype=np.int64)
    if n_frames >= target:
        return np.linspace(0, n_frames - 1, target).astype(np.int64)
    # pad by repeating last
    idx = np.arange(n_frames)
    pad = np.full(target - n_frames, n_frames - 1, dtype=np.int64)
    return np.concatenate([idx, pad])


def normalize_landmarks(frame_xyz: np.ndarray) -> np.ndarray:
    """Positional independence: center on mid-hip, scale by shoulder width.

    frame_xyz: (N_LM, 3) concatenated pose|lh|rh|face in Holistic order.
    Pose indices: 11 L-shoulder, 12 R-shoulder, 23 L-hip, 24 R-hip.
    """
    pose = frame_xyz[:N_POSE]
    lh = frame_xyz[N_POSE : N_POSE + N_HAND]
    rh = frame_xyz[N_POSE + N_HAND : N_POSE + 2 * N_HAND]
    face = frame_xyz[N_POSE + 2 * N_HAND :]

    # Mid hip as origin (fallback to mid-shoulder)
    l_hip, r_hip = pose[23], pose[24]
    l_sh, r_sh = pose[11], pose[12]
    if np.any(l_hip) or np.any(r_hip):
        origin = 0.5 * (l_hip + r_hip)
    else:
        origin = 0.5 * (l_sh + r_sh)

    scale = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    if scale < 1e-3:
        scale = 1.0

    def norm_block(block: np.ndarray) -> np.ndarray:
        b = block.copy()
        present = np.any(b != 0, axis=1)
        b[present] = (b[present] - origin) / scale
        return b

    return np.concatenate(
        [norm_block(pose), norm_block(lh), norm_block(rh), norm_block(face)],
        axis=0,
    ).astype(np.float32)


def extract_video_landmarks(
    video_path: str | Path,
    num_frames: int = 30,
    max_side: int = 480,
) -> np.ndarray:
    """Return (T, FEAT_DIM) normalized landmark sequence."""
    import mediapipe as mp

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = set(sample_frame_indices(total if total > 0 else num_frames, num_frames).tolist())
    wanted = sample_frame_indices(total if total > 0 else num_frames, num_frames)

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    by_idx: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in indices:
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            pose = _lm_to_array(res.pose_landmarks, N_POSE)
            lh = _lm_to_array(res.left_hand_landmarks, N_HAND)
            rh = _lm_to_array(res.right_hand_landmarks, N_HAND)
            face = _lm_to_array(res.face_landmarks, N_FACE)
            raw = np.concatenate([pose, lh, rh, face], axis=0)
            by_idx[i] = normalize_landmarks(raw).reshape(-1)
        i += 1
    cap.release()
    holistic.close()

    seq = np.zeros((num_frames, FEAT_DIM), dtype=np.float32)
    last = np.zeros(FEAT_DIM, dtype=np.float32)
    for t, fi in enumerate(wanted.tolist()):
        if fi in by_idx:
            last = by_idx[fi]
        seq[t] = last
    return seq


def cache_key(video_path: str | Path, num_frames: int) -> str:
    p = Path(video_path).resolve().as_posix()
    h = hashlib.sha1(f"{p}|{num_frames}|v1".encode()).hexdigest()[:16]
    return f"{Path(video_path).stem}__T{num_frames}__{h}.npy"


def load_or_extract(
    video_path: str | Path,
    cache_dir: Path,
    num_frames: int = 30,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / cache_key(video_path, num_frames)
    if out.exists():
        return np.load(out)
    arr = extract_video_landmarks(video_path, num_frames=num_frames)
    np.save(out, arr)
    return arr
