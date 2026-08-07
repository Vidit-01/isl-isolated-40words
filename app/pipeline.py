"""Main live loop: camera → landmarks → model → overlay + speech."""
from __future__ import annotations

import os
import threading
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
from .sentence_service import SentenceFormationError, formulate_english_sentence
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
    unique_signs: list[str] = []
    current_sentence = ""
    sentence_thread: threading.Thread | None = None
    sentence_generation = 0

    def process_sentence_task(words: list[str], generation: int) -> None:
        nonlocal current_sentence
        try:
            result = formulate_english_sentence(words)
            print(f"[sentence] {result}")
        except SentenceFormationError as exc:
            result = f"Sentence error: {exc}"
            print(f"[sentence] {exc}")
        if generation == sentence_generation:
            current_sentence = result

    def start_sentence_processing() -> None:
        nonlocal current_sentence, sentence_generation, sentence_thread, unique_signs
        if not unique_signs or (sentence_thread and sentence_thread.is_alive()):
            return
        words = list(unique_signs)
        unique_signs = []
        current_sentence = "Processing sentence..."
        print(f"[sentence] pause detected; recognized words: {', '.join(words)}")
        sentence_generation += 1
        sentence_thread = threading.Thread(
            target=process_sentence_task,
            args=(words, sentence_generation),
            name="sentence-worker",
            daemon=True,
        )
        sentence_thread.start()

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
                if not pred.hands_detected:
                    status = "No movement detected"
                    pending_word = None
                    pending_count = 0
                    start_sentence_processing()
                elif pred.confidence >= cfg.conf_threshold:
                    status = f"{pred.word}  ({pred.confidence:.0%})"
                    # Sentence collection does not require the extra speech
                    # stability match; one confident recognition is enough.
                    if pred.word not in unique_signs:
                        unique_signs.append(pred.word)
                        print(f"[sentence] buffered word: {pred.word}")
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
                sentence=current_sentence,
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
                unique_signs = []
                current_sentence = ""
                sentence_generation += 1
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
                        sentence=current_sentence,
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
