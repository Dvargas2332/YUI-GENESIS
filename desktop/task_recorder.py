from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() not in {"0", "false", "False", "no", "NO"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class RecordingResult:
    name: str
    steps: List[str]
    duration_s: float


class TaskRecorder:
    """
    Records high-level user actions (mouse clicks/scroll + hotkeys) to create a replayable task.

    Privacy-by-default:
    - Does NOT record typed text (alphanumeric keys) unless you explicitly enable it.
    - Intended for personal automation on your own PC.
    """

    def __init__(self) -> None:
        self._running = False
        self._name = ""
        self._events: List[Tuple[str, Any]] = []
        self._t0 = 0.0
        self._last_event_ts = 0.0
        self._mods: Set[str] = set()
        self._capture_text = _env_bool("YUI_RECORDER_CAPTURE_TEXT", False)

        self._kb_listener = None
        self._mouse_listener = None

    @property
    def running(self) -> bool:
        return bool(self._running)

    @property
    def name(self) -> str:
        return self._name

    def start(self, name: str) -> None:
        if self._running:
            return
        name = (name or "").strip().lower()
        if not name:
            raise ValueError("name required")

        # Optional dependency.
        try:
            from pynput import keyboard, mouse  # type: ignore
        except Exception as e:
            raise RuntimeError("missing_pynput") from e

        self._running = True
        self._name = name
        self._events = []
        self._mods = set()
        self._t0 = time.time()
        self._last_event_ts = self._t0

        def on_click(x, y, button, pressed):  # noqa: ANN001
            if not self._running:
                return False
            # Record on release to avoid duplicates.
            if pressed:
                return True
            now = time.time()
            self._maybe_wait(now)
            bname = str(getattr(button, "name", "") or "").lower() or str(button).lower()
            self._events.append(("click", (int(x), int(y), bname)))
            return True

        def on_scroll(x, y, dx, dy):  # noqa: ANN001
            if not self._running:
                return False
            now = time.time()
            self._maybe_wait(now)
            # dy is in "scroll steps"; convert into a pixel-ish value for pyautogui
            amt = int(dy) * 120
            self._events.append(("scroll", amt))
            return True

        def on_press(key):  # noqa: ANN001
            if not self._running:
                return False
            k = self._key_token(key)
            if not k:
                return True

            if k in {"ctrl", "alt", "shift"}:
                self._mods.add(k)
                return True

            now = time.time()
            self._maybe_wait(now)

            # Hotkeys
            if self._mods:
                combo = " ".join(sorted(self._mods)) + " " + k
                self._events.append(("hotkey", combo.strip()))
                return True

            # Single special keys (no text)
            if k in {"enter", "tab", "esc", "escape"}:
                self._events.append(("key", "enter" if k == "enter" else ("esc" if k in {"esc", "escape"} else k)))
                return True

            # Typed text (disabled by default)
            if self._capture_text and len(k) == 1 and re.match(r"[a-z0-9]", k):
                self._events.append(("text", k))
            return True

        def on_release(key):  # noqa: ANN001
            if not self._running:
                return False
            k = self._key_token(key)
            if k in {"ctrl", "alt", "shift"}:
                self._mods.discard(k)
            return True

        self._kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        self._kb_listener.start()
        self._mouse_listener.start()

    def stop(self) -> RecordingResult:
        if not self._running:
            return RecordingResult(name=self._name or "", steps=[], duration_s=0.0)

        self._running = False
        t1 = time.time()
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
        except Exception:
            pass
        try:
            if self._mouse_listener is not None:
                self._mouse_listener.stop()
        except Exception:
            pass

        steps = self._events_to_steps()
        name = self._name
        self._name = ""
        self._events = []
        self._mods = set()
        return RecordingResult(name=name, steps=steps, duration_s=max(0.0, float(t1 - self._t0)))

    def cancel(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
        except Exception:
            pass
        try:
            if self._mouse_listener is not None:
                self._mouse_listener.stop()
        except Exception:
            pass
        self._name = ""
        self._events = []
        self._mods = set()

    def _maybe_wait(self, now: float) -> None:
        min_wait = max(0.2, _env_float("YUI_RECORDER_WAIT_MIN_S", 0.45))
        dt = float(now - self._last_event_ts)
        self._last_event_ts = now
        if dt >= min_wait:
            # Round for stability.
            self._events.append(("wait", round(dt, 2)))

    def _events_to_steps(self) -> List[str]:
        steps: List[str] = []
        text_buf: List[str] = []

        def flush_text():
            nonlocal text_buf
            if not text_buf:
                return
            # Keep short; avoid capturing long strings unintentionally.
            s = "".join(text_buf)[:120]
            steps.append(f"escribe {s}")
            text_buf = []

        for kind, payload in self._events:
            if kind != "text":
                flush_text()

            if kind == "wait":
                steps.append(f"espera {payload}")
            elif kind == "click":
                x, y, bname = payload
                if "right" in bname:
                    steps.append(f"click derecho {x} {y}")
                else:
                    steps.append(f"click {x} {y}")
            elif kind == "scroll":
                steps.append(f"scroll {int(payload)}")
            elif kind == "hotkey":
                steps.append(f"atajo {payload}")
            elif kind == "key":
                steps.append(f"tecla {payload}")
            elif kind == "text":
                text_buf.append(str(payload))

        flush_text()

        # Remove leading waits (noise)
        while steps and steps[0].startswith("espera "):
            steps.pop(0)

        return steps

    def _key_token(self, key) -> Optional[str]:  # noqa: ANN001
        try:
            from pynput import keyboard  # type: ignore
        except Exception:
            return None

        # Modifiers
        try:
            if key in {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
                return "ctrl"
            if key in {keyboard.Key.alt_l, keyboard.Key.alt_r}:
                return "alt"
            if key in {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}:
                return "shift"
        except Exception:
            pass

        # Special keys
        try:
            if isinstance(key, keyboard.Key):
                name = str(getattr(key, "name", "") or "").lower()
                return name or None
        except Exception:
            pass

        # Characters
        try:
            ch = getattr(key, "char", None)
            if ch is None:
                return None
            ch = str(ch).strip().lower()
            if not ch:
                return None
            # Keep only simple chars
            if len(ch) == 1:
                return ch
        except Exception:
            return None
        return None

