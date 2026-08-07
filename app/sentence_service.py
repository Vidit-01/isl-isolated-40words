"""English sentence formation for recognized ISL words."""
from __future__ import annotations

import os
from collections.abc import Sequence


class SentenceFormationError(RuntimeError):
    """Raised when Gemini cannot formulate a sentence."""


def formulate_english_sentence(words: Sequence[str]) -> str:
    """Use Gemini to turn recognized ISL words into one English sentence."""
    cleaned_words = [str(word).strip() for word in words if str(word).strip()]
    if not cleaned_words:
        return ""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SentenceFormationError(
            "Set GEMINI_API_KEY before starting the app"
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            contents=(
                "Turn the following recognized Indian Sign Language words into "
                "one natural, grammatically correct English sentence. Preserve "
                "the intended meaning and word order where possible. Do not add "
                "new facts. Return only the sentence, with no explanation or "
                f"formatting.\n\nRecognized words: {', '.join(cleaned_words)}"
            ),
        )
        sentence = (response.text or "").strip().strip('"')
    except SentenceFormationError:
        raise
    except Exception as exc:
        raise SentenceFormationError(f"Gemini request failed: {exc}") from exc

    if not sentence:
        raise SentenceFormationError("Gemini returned an empty sentence")
    return sentence
