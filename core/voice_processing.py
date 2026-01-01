from __future__ import annotations

import threading
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

    def listen(self) -> Optional[str]:
        if not self.voice_enabled:
            return None

        if self.settings.wake_word_enabled and not self._mic_available:
            # In text fallback mode, wake word would require typing "yui ..." and appears as "not responding".
            print("[YUI] Micrófono no disponible; desactivando wake word para modo texto.")
            return listen(
                language=self.settings.stt_language,
                timeout_s=self.settings.listen_timeout_s,
                phrase_time_limit_s=self.settings.phrase_time_limit_s,
                backend="text",
                microphone_index=self.settings.stt_microphone_index,
                sounddevice_index=self.settings.sounddevice_input_index,
            )

        if not self.settings.wake_word_enabled:
            return listen(
                language=self.settings.stt_language,
                timeout_s=self.settings.listen_timeout_s,
                phrase_time_limit_s=self.settings.phrase_time_limit_s,
                backend=self.settings.stt_backend,
                microphone_index=self.settings.stt_microphone_index,
                sounddevice_index=self.settings.sounddevice_input_index,
            )

        while True:
            wr = wait_for_wake_word(self.settings)
            if not wr.heard:
                continue

            # If user said "YUI ..." in one go, use the remainder as the command.
            if wr.transcript:
                remainder = strip_wake_word(wr.transcript, self.settings)
                if remainder:
                    return remainder

            # Otherwise, listen for the actual command after wake word.
            return listen(
                language=self.settings.stt_language,
                timeout_s=self.settings.listen_timeout_s,
                phrase_time_limit_s=self.settings.phrase_time_limit_s,
                backend=self.settings.stt_backend,
                microphone_index=self.settings.stt_microphone_index,
                sounddevice_index=self.settings.sounddevice_input_index,
            )

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._speak_lock:
            tts_path = hablar(
                text,
                engine=self.settings.tts_engine,
                voice=self.settings.tts_voice,
                edge_rate=self.settings.tts_edge_rate,
                edge_volume=self.settings.tts_edge_volume,
                rate=self.settings.tts_rate,
                volume=self.settings.tts_volume,
            )
        if self._ui_bus is not None:
            self._ui_bus.publish("tts", {"text": text, "audio_path": str(tts_path) if tts_path else None})
