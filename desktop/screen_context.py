from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() not in {"0", "false", "False", "no", "NO"}


@dataclass(frozen=True)
class ActiveWindow:
    ts: float
    title: str
    pid: int


def get_active_window() -> Optional[ActiveWindow]:
    """
    Best-effort active window title (Windows).
    Returns None on unsupported platforms.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(int(length) + 1)
        user32.GetWindowTextW(hwnd, buf, int(length) + 1)
        title = (buf.value or "").strip()

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return ActiveWindow(ts=time.time(), title=title, pid=int(pid.value or 0))
    except Exception:
        return None


def active_window_summary() -> str:
    """
    Short, privacy-aware summary string for LLM context.
    """
    if _env_bool("YUI_SCREEN_CONTEXT_ENABLED", True) is False:
        return ""
    aw = get_active_window()
    if aw is None:
        return ""
    title = (aw.title or "").strip()
    if not title:
        return ""
    max_len = max(20, _env_int("YUI_SCREEN_CONTEXT_MAX_TITLE", 80))
    if len(title) > max_len:
        title = title[: max_len - 1].rstrip() + "…"
    return f"Ventana activa: {title}"


def capture_screen_embedding() -> Optional[list[float]]:
    """
    Captures a screenshot and converts it into a compact grayscale embedding vector.
    Stores no images; only returns a numeric vector.
    """
    if _env_bool("YUI_SCREEN_EMBEDDINGS_ENABLED", True) is False:
        return None

    try:
        from PIL import ImageGrab  # type: ignore
    except Exception:
        return None

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    w = max(8, _env_int("YUI_SCREEN_EMBED_W", 32))
    h = max(6, _env_int("YUI_SCREEN_EMBED_H", 18))

    try:
        img = ImageGrab.grab()
    except Exception:
        return None

    try:
        img = img.convert("L").resize((int(w), int(h)))
    except Exception:
        return None

    try:
        if np is not None:
            arr = np.asarray(img, dtype="float32") / 255.0
            return [float(x) for x in arr.reshape(-1).tolist()]
        # Pure-Python fallback (slower)
        px = list(img.getdata())
        return [float(int(v)) / 255.0 for v in px]
    except Exception:
        return None

