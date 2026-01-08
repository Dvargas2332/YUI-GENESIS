from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional


_PYGAME_READY = False


def _debug_tts() -> bool:
    return os.getenv("YUI_DEBUG_TTS", "0") not in {"0", "false", "False"}


def _play_mp3(path: Path) -> bool:
    global _PYGAME_READY
    # Prefer pygame on Windows; playsound can be flaky depending on codecs/devices.
    try:
        import pygame  # type: ignore

        if not _PYGAME_READY:
            pygame.mixer.init()
            _PYGAME_READY = True

        try:
            pygame.mixer.music.set_volume(float(os.getenv("YUI_TTS_PYGAME_VOLUME", "1.0")))
        except Exception:
            pass

        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
        try:
            # Release the file handle (important on Windows when overwriting last_tts.mp3).
            pygame.mixer.music.unload()
        except Exception:
            pass
        return True
    except Exception as e:
        if _debug_tts():
            print(f"[YUI] pygame TTS playback failed: {type(e).__name__}: {e}")

    try:
        from playsound import playsound
    except Exception:
        return False
    try:
        playsound(str(path))
        return True
    except Exception as e:
        if _debug_tts():
            print(f"[YUI] playsound TTS playback failed: {type(e).__name__}: {e}")
        return False


def _speak_pyttsx3(text: str, rate: int = 165, volume: float = 1.0) -> bool:
    try:
        import pyttsx3
    except Exception:
        return False

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()
    return True


async def _edge_tts_to_file(*, text: str, voice: str, rate: str, volume: str, pitch: str, out_path: Path) -> None:
    import edge_tts  # type: ignore

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(str(out_path))

def hablar(
    texto: str,
    *,
    engine: str = "edge",
    voice: str = "es-MX-DaliaNeural",
    edge_rate: str = "+0%",
    edge_volume: str = "+0%",
    edge_pitch: str = "+0Hz",
    use_ssml: bool = True,
    rate: int = 165,
    volume: float = 1.0,
) -> Optional[Path]:
    texto = (texto or "").strip()
    if not texto:
        return None

    engine_norm = (engine or "").strip().lower()

    if engine_norm in {"edge", "auto"}:
        path = _speak_edge_to_path(texto, voice=voice, rate=edge_rate, volume=edge_volume, pitch=edge_pitch, use_ssml=use_ssml)
        if path is not None:
            ok = _play_mp3(path)
            if not ok and _debug_tts():
                print(f"[YUI] TTS audio generado pero no se pudo reproducir: {path}")
            return path

    if engine_norm in {"pyttsx3", "auto"}:
        if _speak_pyttsx3(texto, rate=rate, volume=volume):
            return None

    if engine_norm in {"gtts", "auto"}:
        path = _speak_gtts_to_path(texto)
        if path is not None:
            ok = _play_mp3(path)
            if not ok and _debug_tts():
                print(f"[YUI] TTS gTTS generado pero no se pudo reproducir: {path}")
            return path

    print(f"YUI: {texto}")
    return None


def _persist_last_tts(mp3_path: Path) -> Path:
    out = os.getenv("YUI_TTS_LAST_PATH", "")
    if out:
        out_path = Path(out)
    else:
        out_path = Path("data") / "last_tts.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_bytes(mp3_path.read_bytes())
        return out_path
    except Exception:
        return mp3_path


def _speak_edge_to_path(text: str, *, voice: str, rate: str, volume: str, pitch: str, use_ssml: bool) -> Optional[Path]:
    try:
        import edge_tts  # noqa: F401
    except Exception:
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="yui_edge_tts_"))
    mp3_path = tmp_dir / "tts.mp3"
    try:
        asyncio.run(_edge_tts_to_file(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch, out_path=mp3_path))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_edge_tts_to_file(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch, out_path=mp3_path))
        finally:
            loop.close()
    except Exception as e:
        # Retry with a known-good Spanish female voice.
        fallback_voice = os.getenv("YUI_TTS_VOICE_FALLBACK", "es-MX-DaliaNeural").strip() or "es-MX-DaliaNeural"
        if fallback_voice != voice:
            try:
                asyncio.run(_edge_tts_to_file(text=text, voice=fallback_voice, rate=rate, volume=volume, pitch=pitch, out_path=mp3_path))
            except Exception as e2:
                print(f"[YUI] edge-tts failed ({voice!r}): {type(e).__name__}: {e}")
                print(f"[YUI] edge-tts fallback failed ({fallback_voice!r}): {type(e2).__name__}: {e2}")
                return None
        else:
            print(f"[YUI] edge-tts failed ({voice!r}): {type(e).__name__}: {e}")
            return None

    if os.getenv("YUI_TTS_SAVE_LAST", "1") in {"1", "true", "True"}:
        return _persist_last_tts(mp3_path)
    return mp3_path


def _speak_gtts_to_path(text: str) -> Optional[Path]:
    try:
        from gtts import gTTS
    except Exception:
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="yui_gtts_"))
    mp3_path = tmp_dir / "tts.mp3"
    gTTS(text=text, lang="es").save(str(mp3_path))
    if os.getenv("YUI_TTS_SAVE_LAST", "1") in {"1", "true", "True"}:
        return _persist_last_tts(mp3_path)
    return mp3_path
