from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Optional, TextIO


_LOCK_FH: Optional[TextIO] = None


def acquire_single_instance_lock() -> bool:
    """
    Prevent multiple YUI instances running at the same time (avoids double TTS, camera/mic conflicts).
    Set `YUI_SINGLE_INSTANCE=0` to allow multiple instances.
    """
    enabled = os.getenv("YUI_SINGLE_INSTANCE", "1") not in {"0", "false", "False"}
    if not enabled:
        return True

    global _LOCK_FH
    if _LOCK_FH is not None:
        return True

    lock_path = (os.getenv("YUI_LOCK_PATH") or "").strip()
    path = Path(lock_path) if lock_path else (Path("data") / "yui.lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fh = open(path, "a+", encoding="utf-8", errors="replace")
    except Exception:
        return True

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # type: ignore

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        try:
            fh.close()
        except Exception:
            pass
        return False

    _LOCK_FH = fh

    @atexit.register
    def _release() -> None:
        global _LOCK_FH
        cur = _LOCK_FH
        _LOCK_FH = None
        if cur is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                cur.seek(0)
                msvcrt.locking(cur.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # type: ignore

                fcntl.flock(cur.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            cur.close()
        except Exception:
            pass

    return True

