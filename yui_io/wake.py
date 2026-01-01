from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config.settings import Settings
from yui_io.stt import listen


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass(frozen=True)
class WakeResult:
    heard: bool
    transcript: Optional[str] = None


def wait_for_wake_word(settings: Settings) -> WakeResult:
    """
    Listens in short bursts; returns when wake word is heard.
    Works with SpeechRecognition (mic) or text fallback.
    """
    wake = _normalize(settings.wake_word)
    if not wake:
        return WakeResult(heard=True, transcript=None)

    transcript = listen(
        language=settings.stt_language,
        timeout_s=settings.wake_listen_timeout_s,
        phrase_time_limit_s=settings.wake_phrase_time_limit_s,
        backend=settings.stt_backend,
        microphone_index=settings.stt_microphone_index,
        sounddevice_index=settings.sounddevice_input_index,
    )
    if not transcript:
        return WakeResult(heard=False, transcript=None)

    norm = _normalize(transcript)
    # Accept "yui" alone or inside phrase: "yui ...", "oye yui ..."
    if wake in norm.split() or wake in norm:
        return WakeResult(heard=True, transcript=transcript)

    return WakeResult(heard=False, transcript=transcript)


def strip_wake_word(text: str, settings: Settings) -> str:
    """
    Removes wake word from a transcript like "YUI dime la hora".
    """
    wake = re.escape(_normalize(settings.wake_word))
    t = _normalize(text)
    t = re.sub(rf"\b{wake}\b[:,]?\s*", "", t).strip()
    return t
