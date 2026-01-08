from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in {"0", "false", "False", "no", "NO"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


def _default_cache_path() -> Path:
    return Path(os.getenv("YUI_PROFILE_CACHE_PATH", "") or (Path("data") / "system_profile.json"))


def _get_total_memory_gb() -> Optional[float]:
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(float(stat.ullTotalPhys) / (1024.0**3), 2)

        # POSIX fallback
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024.0**3), 2)
    except Exception:
        return None
    return None


def _run_powershell(script: str, *, timeout_s: float = 4.0) -> Optional[str]:
    if not sys.platform.startswith("win"):
        return None
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s), check=False)
    except Exception:
        return None
    out = (res.stdout or "").strip()
    return out or None


def _which_any(names: list[str]) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


@dataclass(frozen=True)
class RuntimeTuning:
    tier: str
    vision_every_n_frames: int
    preview_width: int
    preview_height: int
    security_watch_interval_s: float


def collect_system_profile(*, extended: bool = False) -> Dict[str, Any]:
    """
    Collects a lightweight system snapshot to help tune runtime behavior.
    Does not access the network and avoids sensitive data (no usernames, no file lists).
    """
    prof: Dict[str, Any] = {}
    prof["ts"] = time.time()
    prof["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    prof["python"] = {"version": platform.python_version(), "implementation": platform.python_implementation()}
    prof["cpu"] = {"logical_cores": int(os.cpu_count() or 0)}
    mem_gb = _get_total_memory_gb()
    if mem_gb is not None:
        prof["memory_gb"] = mem_gb

    # Installed tooling (best-effort; used only to tailor instructions).
    tools = {
        "nmap": _which_any(["nmap", "nmap.exe"]),
        "wireshark": _which_any(["wireshark", "wireshark.exe"]),
        "tshark": _which_any(["tshark", "tshark.exe"]),
        "burp": _which_any(["burpsuite", "BurpSuite", "burp", "burp.exe"]),
    }
    prof["tools"] = {k: bool(v) for k, v in tools.items()}

    # Audio devices summary (names only).
    try:
        import sounddevice as sd  # type: ignore

        default_dev = getattr(sd, "default", None)
        prof["audio"] = {"default_device": getattr(default_dev, "device", None)}
    except Exception:
        pass

    if extended and sys.platform.startswith("win"):
        cpu_name = _run_powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)", timeout_s=4.0)
        if cpu_name:
            prof.setdefault("cpu", {})["name"] = cpu_name.strip()
        gpu_name = _run_powershell("(Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)", timeout_s=4.0)
        if gpu_name:
            prof["gpu"] = {"name": gpu_name.strip()}

    return prof


def load_cached_profile(*, max_age_s: float) -> Optional[Dict[str, Any]]:
    path = _default_cache_path()
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict):
            return None
        ts = float(data.get("ts", 0.0) or 0.0)
        if ts <= 0:
            return None
        if max_age_s > 0 and (time.time() - ts) > float(max_age_s):
            return None
        return data
    except Exception:
        return None


def save_profile(profile: Dict[str, Any]) -> None:
    path = _default_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_or_collect_profile() -> Dict[str, Any]:
    max_age = _env_float("YUI_PROFILE_MAX_AGE_S", 24 * 3600.0)
    cached = load_cached_profile(max_age_s=max_age)
    if cached is not None:
        return cached

    extended = _env_bool("YUI_PROFILE_EXTENDED", False)
    prof = collect_system_profile(extended=extended)
    save_profile(prof)
    return prof


def refresh_profile() -> Dict[str, Any]:
    """
    Forces a new profile collection and overwrites the cached profile.
    """
    extended = _env_bool("YUI_PROFILE_EXTENDED", False)
    prof = collect_system_profile(extended=extended)
    save_profile(prof)
    return prof


def summarize_profile(profile: Dict[str, Any]) -> str:
    cpu = profile.get("cpu", {}) if isinstance(profile.get("cpu"), dict) else {}
    cores = cpu.get("logical_cores")
    mem = profile.get("memory_gb")
    plat = profile.get("platform", {}) if isinstance(profile.get("platform"), dict) else {}
    sysname = plat.get("system") or ""
    rel = plat.get("release") or ""
    parts = []
    if sysname or rel:
        parts.append(f"{sysname} {rel}".strip())
    if cores:
        parts.append(f"{cores} cores")
    if mem:
        parts.append(f"{mem} GB RAM")
    return " | ".join(parts) if parts else "Entorno detectado."


def derive_tuning(profile: Dict[str, Any], *, default_preview: tuple[int, int], default_every_n: int, default_sec_interval_s: float) -> RuntimeTuning:
    cpu = profile.get("cpu", {}) if isinstance(profile.get("cpu"), dict) else {}
    cores = int(cpu.get("logical_cores") or 0)
    mem_gb = float(profile.get("memory_gb") or 0.0)

    if (cores and cores <= 3) or (mem_gb and mem_gb < 6.5):
        tier = "low"
    elif (cores and cores <= 6) or (mem_gb and mem_gb < 12.5):
        tier = "mid"
    else:
        tier = "high"

    # Vision tuning
    if tier == "low":
        every_n = max(int(default_every_n), 4)
        pw, ph = 320, 180
        sec = max(float(default_sec_interval_s), 60.0)
    elif tier == "mid":
        every_n = max(1, int(default_every_n))
        pw, ph = default_preview
        sec = max(20.0, float(default_sec_interval_s))
    else:
        every_n = 1
        pw, ph = max(default_preview[0], 640), max(default_preview[1], 360)
        sec = min(20.0, float(default_sec_interval_s))

    return RuntimeTuning(
        tier=tier,
        vision_every_n_frames=int(every_n),
        preview_width=int(pw),
        preview_height=int(ph),
        security_watch_interval_s=float(sec),
    )
