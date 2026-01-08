from __future__ import annotations

import sys
from typing import List, Optional


def privacy_in_use_keys() -> List[str]:
    """
    Returns keys like: "camera|<app>" and "microphone|<app>" for currently-in-use devices.
    Best-effort on Windows using ConsentStore registry (stores no audio/video).
    """
    if not sys.platform.startswith("win"):
        return []

    try:
        import winreg  # type: ignore
    except Exception:
        return []

    base = r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore"
    caps = {"camera": "webcam", "microphone": "microphone"}

    out: list[str] = []

    def decode_app(name: str) -> str:
        # NonPackaged keys encode paths using '#'
        s = (name or "").strip()
        if "#" in s and (":#" in s or s.lower().startswith("c:#") or s.lower().startswith("d:#")):
            s = s.replace("#", "\\")
        return s

    def qword(k, name: str) -> Optional[int]:
        try:
            v, _t = winreg.QueryValueEx(k, name)
            return int(v)
        except Exception:
            return None

    def scan_key(cap_name: str, key_path: str, *, depth: int) -> None:
        try:
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        except Exception:
            return

        start = qword(k, "LastUsedTimeStart")
        stop = qword(k, "LastUsedTimeStop")
        if start and isinstance(stop, int) and stop == 0:
            leaf = key_path.rsplit("\\", 1)[-1]
            out.append(f"{cap_name}|{decode_app(leaf)}")

        if depth <= 0:
            return
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(k, i)
            except OSError:
                break
            i += 1
            scan_key(cap_name, key_path + "\\" + sub, depth=depth - 1)

    for cap, store in caps.items():
        scan_key(cap, base + "\\" + store, depth=2)

    return sorted(set(out))


def privacy_in_use_summary() -> str:
    keys = privacy_in_use_keys()
    if not keys:
        return "No veo la cámara ni el micrófono en uso ahora mismo."
    cams = [k.split("|", 1)[1] for k in keys if k.startswith("camera|")]
    mics = [k.split("|", 1)[1] for k in keys if k.startswith("microphone|")]
    parts: list[str] = []
    if cams:
        parts.append("cámara: " + ", ".join(cams[:3]))
    if mics:
        parts.append("micrófono: " + ", ".join(mics[:3]))
    return "En uso: " + " | ".join(parts)

