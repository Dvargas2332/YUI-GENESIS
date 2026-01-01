from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SttDevices:
    # Indices are what you should set in env vars.
    speech_recognition: list[tuple[int, str]]
    sounddevice: list[tuple[int, str]]


def list_devices() -> SttDevices:
    sr_names: list[tuple[int, str]] = []
    try:
        import speech_recognition as sr

        try:
            names = list(sr.Microphone.list_microphone_names())
            sr_names = [(i, str(n)) for i, n in enumerate(names)]
        except Exception:
            sr_names = []
    except Exception:
        sr_names = []

    sd_names: list[tuple[int, str]] = []
    try:
        import sounddevice as sd  # type: ignore

        try:
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                name = str(d.get("name", ""))
                max_in = int(d.get("max_input_channels", 0))
                if max_in > 0:
                    # `idx` is the actual device index for sounddevice.
                    sd_names.append((idx, name))
        except Exception:
            sd_names = []
    except Exception:
        sd_names = []

    return SttDevices(speech_recognition=sr_names, sounddevice=sd_names)


def _stdin_text_fallback() -> Optional[str]:
    if os.getenv("YUI_ALLOW_TEXT_INPUT", "1") in {"0", "false", "False"}:
        time.sleep(0.2)
        return None
    if not sys.stdin or not sys.stdin.isatty():
        time.sleep(0.2)
        return None
    wake_enabled = os.getenv("YUI_WAKE_WORD_ENABLED", "1") not in {"0", "false", "False"}
    wake_word = (os.getenv("YUI_WAKE_WORD", "yui") or "yui").strip()
    prompt = "Tú (texto): "
    if wake_enabled:
        prompt = f"Tú (texto, escribe '{wake_word} ...'): "

    try:
        text = input(prompt).strip()
    except EOFError:
        return None
    if not text:
        time.sleep(0.2)
        return None
    return text


def microphone_available_speech_recognition(*, microphone_index: int = -1) -> bool:
    try:
        import speech_recognition as sr
    except Exception:
        return False

    try:
        if microphone_index is not None and int(microphone_index) >= 0:
            _ = sr.Microphone(device_index=int(microphone_index))
        else:
            _ = sr.Microphone()
        return True
    except Exception:
        return False


def microphone_available_sounddevice(*, device_index: int = -1) -> bool:
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        return False

    try:
        if device_index is not None and int(device_index) >= 0:
            d = sd.query_devices(int(device_index))
            return int(d.get("max_input_channels", 0)) > 0
        d = sd.query_devices(kind="input")
        return int(d.get("max_input_channels", 0)) > 0
    except Exception:
        return False


def listen(
    *,
    language: str = "es-ES",
    timeout_s: float = 5,
    phrase_time_limit_s: float = 8,
    backend: str = "auto",
    microphone_index: int = -1,
    sounddevice_index: int = -1,
) -> Optional[str]:
    backend_norm = (backend or "").strip().lower()

    if backend_norm == "text":
        return _stdin_text_fallback()

    if backend_norm in {"auto", "speech_recognition"} and microphone_available_speech_recognition(microphone_index=microphone_index):
        text = _listen_speech_recognition(
            language=language,
            timeout_s=timeout_s,
            phrase_time_limit_s=phrase_time_limit_s,
            microphone_index=microphone_index,
        )
        if text:
            return text
        if backend_norm == "speech_recognition":
            return None

    if backend_norm in {"auto", "sounddevice"} and microphone_available_sounddevice(device_index=sounddevice_index):
        text = _listen_sounddevice_google(
            language=language,
            duration_s=phrase_time_limit_s,
            device_index=sounddevice_index,
        )
        if text:
            return text
        if backend_norm == "sounddevice":
            return None

    return _stdin_text_fallback()


def _listen_speech_recognition(
    *,
    language: str,
    timeout_s: float,
    phrase_time_limit_s: float,
    microphone_index: int,
) -> Optional[str]:
    try:
        import speech_recognition as sr
    except Exception:
        return None

    recognizer = sr.Recognizer()

    try:
        if microphone_index is not None and int(microphone_index) >= 0:
            mic = sr.Microphone(device_index=int(microphone_index))
        else:
            mic = sr.Microphone()
    except Exception:
        return None

    with mic as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception:
            pass

        print("Escuchando...")
        try:
            audio = recognizer.listen(source, timeout=timeout_s, phrase_time_limit=phrase_time_limit_s)
        except Exception:
            return None

    try:
        return (recognizer.recognize_google(audio, language=language) or "").strip() or None
    except Exception:
        return None


def _listen_sounddevice_google(*, language: str, duration_s: float, device_index: int) -> Optional[str]:
    """
    Fallback de mic sin PyAudio:
    graba con `sounddevice` y transcribe con `speech_recognition` (Google).
    """
    try:
        import numpy as np
        import sounddevice as sd  # type: ignore
        import speech_recognition as sr
    except Exception:
        return None

    samplerate = 16000
    channels = 1
    max_duration_s = max(1.0, float(duration_s))

    # Simple VAD: stop early after speech ends (reduces latency).
    block_s = 0.1
    block_frames = int(samplerate * block_s)
    max_blocks = int(max_duration_s / block_s)
    noise_blocks = 3  # first ~0.3s used to estimate ambient noise
    min_speech_blocks = 2  # must detect speech for >= ~0.2s
    trailing_silence_blocks = 7  # ~0.7s silence to stop

    blocks: list[np.ndarray] = []
    noise_rms: float = 0.0
    speech_started = False
    speech_blocks = 0
    silence_after_speech = 0

    def rms(x: np.ndarray) -> float:
        # x is int16
        xf = x.astype(np.float32)
        return float(np.sqrt(np.mean(np.square(xf))) + 1e-6)

    try:
        stream_kwargs = {"samplerate": samplerate, "channels": channels, "dtype": "int16", "blocksize": block_frames}
        if device_index is not None and int(device_index) >= 0:
            stream_kwargs["device"] = int(device_index)

        print("Escuchando...")
        with sd.InputStream(**stream_kwargs) as stream:
            for i in range(max_blocks):
                chunk, _overflowed = stream.read(block_frames)
                chunk = np.asarray(chunk)
                if chunk.ndim == 2:
                    chunk_mono = chunk[:, 0]
                else:
                    chunk_mono = chunk

                blocks.append(chunk_mono.copy())

                r = rms(chunk_mono)
                if i < noise_blocks:
                    noise_rms = max(noise_rms, r)
                    continue

                # thresholds relative to ambient noise
                speech_thr = max(300.0, noise_rms * 3.0)
                silence_thr = max(200.0, noise_rms * 1.6)

                if not speech_started:
                    if r >= speech_thr:
                        speech_blocks += 1
                        if speech_blocks >= min_speech_blocks:
                            speech_started = True
                    else:
                        speech_blocks = 0
                else:
                    if r < silence_thr:
                        silence_after_speech += 1
                    else:
                        silence_after_speech = 0

                    if silence_after_speech >= trailing_silence_blocks:
                        break
    except Exception:
        return None

    if not speech_started:
        return None

    try:
        audio_np = np.concatenate(blocks).astype(np.int16)
        audio_bytes = audio_np.tobytes()
        audio = sr.AudioData(audio_bytes, samplerate, 2)
        recognizer = sr.Recognizer()
        return (recognizer.recognize_google(audio, language=language) or "").strip() or None
    except Exception:
        return None
