from __future__ import annotations

import atexit
import faulthandler
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, TextIO


_LOG_FH: Optional[TextIO] = None


def _log_path() -> Path:
    p = (os.getenv("YUI_LOG_PATH") or "").strip()
    if p:
        return Path(p)
    return Path("data") / "yui.log"


def install_crash_logging() -> None:
    """
    Best-effort crash logging:
    - Writes unhandled exceptions (main + threads) to a log file.
    - Enables faulthandler (helps with hard crashes in native libs like cv2/mediapipe).
    """
    global _LOG_FH
    if _LOG_FH is not None:
        return

    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(path, "a", encoding="utf-8", errors="replace")
    except Exception:
        _LOG_FH = None

    def write_header() -> None:
        fh = _LOG_FH
        if fh is None:
            return
        try:
            fh.write("\n" + ("-" * 60) + "\n")
            fh.write(f"[YUI] start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.flush()
        except Exception:
            pass

    write_header()

    try:
        if _LOG_FH is not None:
            faulthandler.enable(file=_LOG_FH, all_threads=True)
        else:
            faulthandler.enable()
    except Exception:
        pass

    def _dump(prefix: str, exc_type, exc, tb) -> None:
        if _LOG_FH is None:
            return
        try:
            _LOG_FH.write(prefix + "\n")
            traceback.print_exception(exc_type, exc, tb, file=_LOG_FH)
            _LOG_FH.flush()
        except Exception:
            pass

    def _sys_excepthook(exc_type, exc, tb) -> None:
        _dump("[YUI] Unhandled exception (main thread)", exc_type, exc, tb)
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = _sys_excepthook

    def _thread_excepthook(args) -> None:
        prefix = f"[YUI] Unhandled exception (thread={getattr(args.thread, 'name', '?')})"
        _dump(prefix, args.exc_type, args.exc_value, args.exc_traceback)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

    try:
        threading.excepthook = _thread_excepthook  # type: ignore[attr-defined]
    except Exception:
        pass

    @atexit.register
    def _close_log() -> None:
        global _LOG_FH
        fh = _LOG_FH
        _LOG_FH = None
        if fh is None:
            return
        try:
            fh.write(f"[YUI] exit {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.flush()
            fh.close()
        except Exception:
            pass

