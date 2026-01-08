from __future__ import annotations

import re
import threading
import time
from typing import Optional

from config.settings import Settings
from yui_io.stt import listen, microphone_available_speech_recognition, microphone_available_sounddevice
from yui_io.tts import hablar
from yui_io.wake import strip_wake_word, wait_for_wake_word
from ui.ws_events import UiEventBus


class VoiceAssistant:
    def __init__(self, settings: Settings, *, ui_bus: UiEventBus | None = None):
        self.settings = settings
        self._speak_lock = threading.Lock()
        self._ui_bus = ui_bus
        self.voice_enabled = True
        self._mic_available = (
            microphone_available_speech_recognition(microphone_index=self.settings.stt_microphone_index)
            or microphone_available_sounddevice(device_index=self.settings.sounddevice_input_index)
        )
        self._wake_followup_until = 0.0
        self._wake_misses = 0

    def listen(self, *, stop_event: threading.Event | None = None, allow_text_fallback: bool = True) -> Optional[str]:
        if not self.voice_enabled:
            return None
        if stop_event is not None and stop_event.is_set():
            return None

        # Refresh mic availability (devices can disappear/reappear on Windows).
        self._mic_available = (
            microphone_available_speech_recognition(microphone_index=self.settings.stt_microphone_index)
            or microphone_available_sounddevice(device_index=self.settings.sounddevice_input_index)
        )

        if self.settings.wake_word_enabled and not self._mic_available:
            # In text fallback mode, wake word would require typing "yui ..." and appears as "not responding".
            if allow_text_fallback:
                print("[YUI] Micrófono no disponible; desactivando wake word para modo texto.")
                return listen(
                    language=self.settings.stt_language,
                    timeout_s=self.settings.listen_timeout_s,
                    phrase_time_limit_s=self.settings.phrase_time_limit_s,
                    backend="text",
                    microphone_index=self.settings.stt_microphone_index,
                    sounddevice_index=self.settings.sounddevice_input_index,
                )
            return None

        now = time.time()
        # Conversation follow-up: after a wake/interaction, allow a few seconds without repeating the wake word.
        if not self.settings.wake_word_enabled or now < self._wake_followup_until:
            if stop_event is not None and stop_event.is_set():
                return None
            text = listen(
                language=self.settings.stt_language,
                timeout_s=self.settings.listen_timeout_s,
                phrase_time_limit_s=self.settings.phrase_time_limit_s,
                backend=self.settings.stt_backend,
                microphone_index=self.settings.stt_microphone_index,
                sounddevice_index=self.settings.sounddevice_input_index,
            )
            if text:
                remainder = strip_wake_word(text, self.settings)
                return remainder or text
            return None

        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            if not self.voice_enabled:
                return None
            wr = wait_for_wake_word(self.settings)
            if not wr.heard:
                if self.settings.debug_stt and wr.transcript:
                    print(f"[YUI] Wake miss. Oí: {wr.transcript!r}")
                self._wake_misses += 1
                miss_limit = int(self.settings.wake_miss_to_hotmic or 0)
                if miss_limit > 0 and self._wake_misses >= miss_limit:
                    self._wake_misses = 0
                    self._wake_followup_until = time.time() + float(self.settings.wake_hotmic_s or 0.0)
                    if self.settings.debug_stt:
                        print("[YUI] Wake fallback: modo conversación temporal (sin wake word).")
                    # Immediately accept a command without wake word.
                    text = listen(
                        language=self.settings.stt_language,
                        timeout_s=self.settings.listen_timeout_s,
                        phrase_time_limit_s=self.settings.phrase_time_limit_s,
                        backend=self.settings.stt_backend,
                        microphone_index=self.settings.stt_microphone_index,
                        sounddevice_index=self.settings.sounddevice_input_index,
                    )
                    if text:
                        remainder = strip_wake_word(text, self.settings)
                        return remainder or text
                    return None
                continue

            self._wake_misses = 0

            # If user said "YUI ..." in one go, use the remainder as the command.
            if wr.transcript:
                remainder = strip_wake_word(wr.transcript, self.settings)
                if remainder:
                    self._wake_followup_until = time.time() + float(self.settings.wake_followup_s or 0.0)
                    return remainder

            # Otherwise, listen for the actual command after wake word.
            text = listen(
                language=self.settings.stt_language,
                timeout_s=self.settings.listen_timeout_s,
                phrase_time_limit_s=self.settings.phrase_time_limit_s,
                backend=self.settings.stt_backend,
                microphone_index=self.settings.stt_microphone_index,
                sounddevice_index=self.settings.sounddevice_input_index,
            )
            if text:
                self._wake_followup_until = time.time() + float(self.settings.wake_followup_s or 0.0)
            else:
                self._wake_followup_until = time.time() + min(3.0, float(self.settings.wake_followup_s or 0.0))
            return text

    def speak(self, text: str, *, mood: Optional[str] = None) -> None:
        text = (text or "").strip()
        if not text:
            return

        edge_rate = self.settings.tts_edge_rate
        edge_volume = self.settings.tts_edge_volume
        edge_pitch = self.settings.tts_edge_pitch

        if self.settings.tts_contextual and mood:
            m = (mood or "").strip().lower()
            # Subtle prosody adjustments; keep it natural and not "acting".
            if m in {"enojo", "angustia"}:
                edge_rate = _add_percent(edge_rate, -4)
                edge_volume = _add_percent(edge_volume, -6)
                edge_pitch = _add_hz(edge_pitch, -10)
            elif m in {"tristeza"}:
                edge_rate = _add_percent(edge_rate, -6)
                edge_volume = _add_percent(edge_volume, -8)
                edge_pitch = _add_hz(edge_pitch, -12)
            elif m in {"felicidad"}:
                edge_rate = _add_percent(edge_rate, +2)
                edge_pitch = _add_hz(edge_pitch, +6)

        with self._speak_lock:
            tts_path = hablar(
                text,
                engine=self.settings.tts_engine,
                voice=self.settings.tts_voice,
                edge_rate=edge_rate,
                edge_volume=edge_volume,
                edge_pitch=edge_pitch,
                use_ssml=self.settings.tts_use_ssml,
                rate=self.settings.tts_rate,
                volume=self.settings.tts_volume,
            )
        if self.settings.wake_word_enabled:
            # After speaking, keep a follow-up window so the user can respond without repeating wake word.
            self._wake_followup_until = time.time() + float(self.settings.wake_followup_s or 0.0)
        if self._ui_bus is not None:
            self._ui_bus.publish("tts", {"text": text, "audio_path": str(tts_path) if tts_path else None})


def _add_percent(base: str, delta: int) -> str:
    s = (base or "").strip()
    m = re.search(r"([+-]?\d+)", s)
    val = int(m.group(1)) if m else 0
    out = val + int(delta)
    return f"{out:+d}%"


def _add_hz(base: str, delta: int) -> str:
    s = (base or "").strip()
    m = re.search(r"([+-]?\d+)", s)
    val = int(m.group(1)) if m else 0
    out = val + int(delta)
    return f"{out:+d}Hz"
