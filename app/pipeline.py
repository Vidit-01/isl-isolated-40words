"""Main live loop: camera → landmarks → model → overlay + speech."""
from __future__ import annotations

import os
import time

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2

from .camera import Camera, FrameBuffer
from .config import SUPPORTED_MODELS, AppConfig
from .display import draw_overlay
from .holistic import HolisticSession
from .predictor import Prediction, load_predictor
from .speech import Speaker


def run_live(cfg: AppConfig) -> None:
    print(f"Loading model '{cfg.model}' from {cfg.weights_dir} ...")
    holistic = HolisticSession(model_complexity=0)
    predictor = load_predictor(cfg.model, cfg.weights_dir, device=cfg.device, holistic=holistic)
    speaker = Speaker(cooldown_s=cfg.speak_cooldown_s, enabled=True)

    maxlen = max(8, int(cfg.buffer_seconds * cfg.target_fps))
    buf = FrameBuffer(maxlen=maxlen)
    cam = Camera(cfg.camera_index)

    pred: Prediction | None = None
    status = "Sign clearly — auto predict + speak when confident"
    last_infer_t = 0.0
    auto_interval = max(1.2, cfg.buffer_seconds * 0.75)
    model_idx = list(SUPPORTED_MODELS).index(cfg.model) if cfg.model in SUPPORTED_MODELS else 0
    # require the same word twice in a row before speaking (stability)
    pending_word: str | None = None
    pending_count = 0

    print("Auto-speak ON. Controls: SPACE=predict now  M=switch model  C=clear  Q=quit")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                status = "Camera read failed"
                time.sleep(0.05)
                continue

            view = cv2.flip(frame, 1) if cfg.mirror else frame
            buf.push(view)

            now = time.time()
            if buf.full and (now - last_infer_t) >= auto_interval:
                pred = predictor.predict_frames(buf.as_list())
                last_infer_t = now
                status = f"{pred.word}  ({pred.confidence:.0%})"
                if pred.confidence >= cfg.conf_threshold:
                    if pred.word == pending_word:
                        pending_count += 1
                    else:
                        pending_word = pred.word
                        pending_count = 1
                    if pending_count >= 2:
                        if speaker.maybe_speak(pred.word):
                            status = f"Speaking: {pred.word}  ({pred.confidence:.0%})"
                        pending_count = 0  # reset so cooldown owns repeat timing
                else:
                    pending_word = None
                    pending_count = 0
                    status = f"low conf: {pred.word} ({pred.confidence:.0%}) — keep signing"

            overlay = draw_overlay(
                view,
                model_name=predictor.name,
                buffer_fill=len(buf) / maxlen,
                pred=pred,
                status=status,
                conf_threshold=cfg.conf_threshold,
            )
            cv2.imshow(cfg.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                buf.clear()
                pred = None
                pending_word = None
                pending_count = 0
                status = "Buffer cleared"
            if key == ord(" "):
                if len(buf) < 8:
                    status = "Need more frames in buffer"
                else:
                    pred = predictor.predict_frames(buf.as_list())
                    last_infer_t = time.time()
                    status = f"{pred.word}  ({pred.confidence:.0%})"
                    # SPACE always speaks if above a soft floor
                    if pred.confidence >= min(0.15, cfg.conf_threshold):
                        speaker.maybe_speak(pred.word, force=True)
                        status = f"Speaking: {pred.word}"
            if key in (ord("s"), ord("S")):
                if pred is not None:
                    speaker.maybe_speak(pred.word, force=True)
                    status = f"Speaking: {pred.word}"
            if key in (ord("m"), ord("M")):
                model_idx = (model_idx + 1) % len(SUPPORTED_MODELS)
                new_name = SUPPORTED_MODELS[model_idx]
                status = f"Loading {new_name}..."
                cv2.imshow(
                    cfg.window_name,
                    draw_overlay(
                        view,
                        model_name=new_name,
                        buffer_fill=len(buf) / maxlen,
                        pred=pred,
                        status=status,
                        conf_threshold=cfg.conf_threshold,
                    ),
                )
                cv2.waitKey(1)
                predictor = load_predictor(
                    new_name, cfg.weights_dir, device=cfg.device, holistic=holistic
                )
                status = f"Switched to {new_name}"
                print(status)
    finally:
        speaker.close()
        cam.release()
        holistic.close()
        cv2.destroyAllWindows()
