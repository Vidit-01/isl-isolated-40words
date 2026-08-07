"""Reliable TTS: dedicated worker thread + Windows SAPI fallback."""
from __future__ import annotations

import platform
import queue
import subprocess
import threading
import time
from typing import Optional


class Speaker:
    """Queue-based speaker. Auto-speak with cooldown; never blocks the camera loop."""

    def __init__(self, cooldown_s: float = 2.0, enabled: bool = True):
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self._last_word: Optional[str] = None
        self._last_t = 0.0
        self._q: queue.Queue[Optional[str]] = queue.Queue(maxsize=4)
        self._backend = "none"
        self._thread: Optional[threading.Thread] = None
        if enabled:
            self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
            self._thread.start()

    def maybe_speak(self, word: str, force: bool = False) -> bool:
        if not self.enabled or not word:
            return False
        word = word.strip()
        if not word:
            return False
        now = time.time()
        if not force:
            # don't spam the same word
            if word == self._last_word and (now - self._last_t) < self.cooldown_s:
                return False
            # brief gap between any utterances
            if (now - self._last_t) < 0.8:
                return False
        self._last_word = word
        self._last_t = now
        # drop oldest if queue is full so we speak the latest word
        try:
            self._q.put_nowait(word)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(word)
            except queue.Full:
                return False
        print(f"[speech] speaking: {word}  (backend={self._backend})")
        return True

    def _worker(self) -> None:
        engine = None
        # Prefer Windows SAPI — most reliable with OpenCV loops
        if platform.system() == "Windows":
            self._backend = "sapi"
            while True:
                word = self._q.get()
                if word is None:
                    return
                self._speak_sapi(word)
            return

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            self._backend = "pyttsx3"
        except Exception as e:
            print(f"[speech] pyttsx3 init failed: {e}")
            self._backend = "none"
            self.enabled = False
            return

        while True:
            word = self._q.get()
            if word is None:
                return
            try:
                engine.say(word)
                engine.runAndWait()
            except Exception as e:
                print(f"[speech] pyttsx3 failed: {e}")

    @staticmethod
    def _speak_sapi(word: str) -> None:
        # Escape single quotes for PowerShell string
        safe = word.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 0; "
            f"$s.Speak('{safe}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                check=False,
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as e:
            print(f"[speech] SAPI failed: {e}")

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
