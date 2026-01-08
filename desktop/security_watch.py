from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ui.ws_events import UiEventBus
from desktop.privacy_devices import privacy_in_use_keys


@dataclass(frozen=True)
class PortListener:
    addr: str
    port: int
    pid: int
    process: str = ""


@dataclass(frozen=True)
class SuspiciousProcess:
    name: str
    pid: int
    path: str
    reason: str
    command_line: str = ""


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in {"0", "false", "False", "no", "NO"}


def _run_powershell_json(script: str, *, timeout_s: float = 12.0):
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception:
        return None
    out = (res.stdout or "").strip()
    if not out:
        return None
    try:
        import json

        return json.loads(out)
    except Exception:
        return None


def _defender_firewall_health() -> tuple[Optional[bool], Optional[bool], str]:
    """
    Returns (defender_ok, firewall_ok, detail).
    """
    defender = _run_powershell_json("Get-MpComputerStatus | ConvertTo-Json -Compress", timeout_s=10.0)
    firewall = _run_powershell_json("Get-NetFirewallProfile | Select-Object Name,Enabled | ConvertTo-Json -Compress", timeout_s=10.0)

    d_ok: Optional[bool] = None
    f_ok: Optional[bool] = None
    lines: list[str] = []

    if isinstance(defender, dict):
        rt = bool(defender.get("RealTimeProtectionEnabled", False))
        av = bool(defender.get("AntivirusEnabled", False))
        d_ok = bool(rt and av)
        lines.append(f"Defender: rt={rt} av={av}")
    else:
        lines.append("Defender: unavailable")

    if isinstance(firewall, dict):
        firewall = [firewall]
    if isinstance(firewall, list):
        enabled_all = True
        enabled_any = False
        for item in firewall:
            if not isinstance(item, dict):
                continue
            enabled = bool(item.get("Enabled", False))
            enabled_any = enabled_any or enabled
            enabled_all = enabled_all and enabled
        f_ok = bool(enabled_all)
        lines.append(f"Firewall: all={enabled_all} any={enabled_any}")
    else:
        lines.append("Firewall: unavailable")

    return d_ok, f_ok, "; ".join(lines)


def _listening_ports() -> List[Tuple[str, int, int]]:
    """
    Returns (addr, port, pid) for TCP listening sockets.
    """
    conns = _run_powershell_json(
        "Get-NetTCPConnection -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess | ConvertTo-Json -Compress",
        timeout_s=10.0,
    )
    if isinstance(conns, dict):
        conns = [conns]
    if isinstance(conns, list):
        out: list[tuple[str, int, int]] = []
        for c in conns:
            if not isinstance(c, dict):
                continue
            addr = str(c.get("LocalAddress") or "").strip() or "?"
            try:
                port = int(c.get("LocalPort") or 0)
                pid = int(c.get("OwningProcess") or 0)
            except Exception:
                continue
            if port > 0 and pid > 0:
                out.append((addr, port, pid))
        return out

    # Fallback: netstat (older systems / missing cmdlets)
    try:
        res = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10.0, check=False)
    except Exception:
        return []
    listens: list[tuple[str, int, int]] = []
    for line in (res.stdout or "").splitlines():
        if "LISTENING" not in line.upper():
            continue
        parts = [p for p in line.split() if p]
        if len(parts) < 5:
            continue
        local = parts[1]
        pid_s = parts[-1]
        try:
            pid = int(pid_s)
        except Exception:
            continue
        try:
            addr, port_s = local.rsplit(":", 1)
            port = int(port_s)
        except Exception:
            continue
        listens.append((addr or "?", port, pid))
    return listens


def _process_name_map() -> Dict[int, str]:
    procs = _run_powershell_json("Get-Process | Select-Object Id,ProcessName | ConvertTo-Json -Compress", timeout_s=8.0)
    if isinstance(procs, dict):
        procs = [procs]
    if not isinstance(procs, list):
        return {}
    out: dict[int, str] = {}
    for it in procs:
        if not isinstance(it, dict):
            continue
        try:
            pid = int(it.get("Id") or 0)
        except Exception:
            continue
        name = str(it.get("ProcessName") or "").strip()
        if pid > 0 and name:
            out[pid] = name
    return out


def _suspicious_processes() -> List[SuspiciousProcess]:
    procs = _run_powershell_json(
        "Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress",
        timeout_s=15.0,
    )
    if isinstance(procs, dict):
        procs = [procs]
    if not isinstance(procs, list):
        return []

    system_names = {
        "svchost.exe",
        "lsass.exe",
        "csrss.exe",
        "winlogon.exe",
        "services.exe",
        "explorer.exe",
        "spoolsv.exe",
        "smss.exe",
    }

    findings: list[SuspiciousProcess] = []
    for p in procs:
        if not isinstance(p, dict):
            continue
        name = str(p.get("Name") or "").strip()
        try:
            pid = int(p.get("ProcessId") or 0)
        except Exception:
            pid = 0
        path = str(p.get("ExecutablePath") or "").strip()
        cmd = str(p.get("CommandLine") or "").strip()
        if not name or pid <= 0:
            continue

        reason = ""
        path_low = path.lower()
        if path:
            if "\\appdata\\local\\temp\\" in path_low or "\\windows\\temp\\" in path_low:
                reason = "corre desde Temp"
            elif "\\downloads\\" in path_low:
                reason = "corre desde Descargas"
            elif name.lower() in system_names and not path_low.startswith(("c:\\windows\\system32\\", "c:\\windows\\")):
                reason = "nombre de sistema pero ruta rara"

        if not reason and cmd:
            low = cmd.lower()
            if "powershell" in low and re.search(r"(?i)(?:^|\\s)-(?:enc|encodedcommand)\\b", cmd):
                reason = "PowerShell con comando codificado"
            elif re.search(r"(?i)\\b(?:iwr|invoke-webrequest|irm|invoke-restmethod)\\b", cmd) and re.search(
                r"(?i)\\b(?:iex|invoke-expression)\\b", cmd
            ):
                reason = "descarga y ejecución en PowerShell"

        if reason:
            findings.append(SuspiciousProcess(name=name, pid=pid, path=path or "(sin ruta)", reason=reason, command_line=cmd))

    return findings


class SecurityWatch:
    """
    Defensive monitoring loop (read-only):
    - new listening ports
    - suspicious process heuristics
    - camera/microphone usage (privacy registry on Windows)

    No remediation is performed automatically.
    """

    def __init__(self, *, ui_bus: UiEventBus | None = None, speak: Callable[[str], None] | None = None):
        self.ui_bus = ui_bus
        self.speak = speak
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.interval_s = float(os.getenv("YUI_SECURITY_WATCH_INTERVAL_S", "30") or "30")
        self.speak_enabled = _env_bool("YUI_SECURITY_WATCH_SPEAK", True)
        self.enabled = _env_bool("YUI_SECURITY_WATCH_ENABLED", True)

        self._ports_prev: set[tuple[str, int, int]] = set()
        self._alerted: dict[str, float] = {}

    def start(self) -> None:
        self.enabled = True
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="yui-security-watch")
        self._thread.start()

    def stop(self) -> None:
        self.enabled = False
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        self._thread = None

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def _publish(self, severity: str, title: str, detail: str = "") -> None:
        if self.ui_bus is not None:
            self.ui_bus.publish("security_alert", {"severity": severity, "title": title, "detail": detail})

    def _should_speak(self) -> bool:
        if not self.speak_enabled or self.speak is None:
            return False
        if _env_bool("YUI_SECURITY_WATCH_SPEAK", True) is False:
            return False
        return True

    def _seen_recently(self, key: str, *, ttl_s: float) -> bool:
        now = time.time()
        ts = float(self._alerted.get(key, 0.0))
        if ts and (now - ts) < ttl_s:
            return True
        self._alerted[key] = now
        # cleanup
        if len(self._alerted) > 800:
            for k, v in list(self._alerted.items())[:200]:
                if (now - float(v)) > ttl_s:
                    self._alerted.pop(k, None)
        return False

    def _run(self) -> None:
        # Baseline: current listeners.
        try:
            base_ports = _listening_ports()
            self._ports_prev = set(base_ports)
        except Exception:
            self._ports_prev = set()

        # Baseline for hosts file
        hosts_path = Path(os.getenv("SystemRoot", r"C:\Windows")) / "System32" / "drivers" / "etc" / "hosts"
        try:
            hosts_mtime = hosts_path.stat().st_mtime if hosts_path.exists() else 0.0
        except Exception:
            hosts_mtime = 0.0

        last_health_ts = 0.0
        last_hosts_ts = 0.0
        privacy_prev: set[str] = set()
        try:
            privacy_prev = set(privacy_in_use_keys())
        except Exception:
            privacy_prev = set()

        if _env_bool("YUI_SECURITY_PRIVACY_NOTIFY_ON_START", True) and privacy_prev:
            try:
                ignore_raw = (os.getenv("YUI_SECURITY_PRIVACY_IGNORE", "python.exe,pythonw.exe") or "").strip()
                ignore = [x.strip().lower() for x in ignore_raw.split(",") if x.strip()]
                for item in sorted(list(privacy_prev))[:12]:
                    low_item = item.lower()
                    if any(x in low_item for x in ignore):
                        continue
                    try:
                        cap, app = item.split("|", 1)
                    except Exception:
                        cap, app = "device", item
                    title = f"Dispositivo en uso al iniciar: {cap} ({app})"
                    self._publish("medium", title, detail="Revisa si es esperado.")
                    print(f"[YUI][SEC] {title}")
            except Exception:
                pass

        if _env_bool("YUI_SECURITY_WATCH_ANNOUNCE", False) and self._should_speak():
            try:
                self.speak("Listo. Activo modo vigilancia de seguridad.")
            except Exception:
                pass

        while not self._stop.is_set():
            if not self.enabled and not _env_bool("YUI_SECURITY_WATCH_ENABLED", True):
                time.sleep(0.25)
                continue

            interval = float(self.interval_s or 30.0)
            interval = max(5.0, min(600.0, interval))

            alerts_voice: list[str] = []
            now = time.time()

            # New listening ports
            try:
                name_map = _process_name_map()
                ports = _listening_ports()
                cur = set(ports)
                new = sorted(list(cur - self._ports_prev), key=lambda t: t[1])
                self._ports_prev = cur

                for addr, port, pid in new[:12]:
                    pname = name_map.get(pid, "")
                    who = f"{pname} (pid={pid})" if pname else f"pid={pid}"
                    title = f"Nuevo puerto en escucha: {port} ({who})"
                    detail = f"{addr}:{port} -> {who}"
                    self._publish("medium", title, detail=detail)
                    print(f"[YUI][SEC] {title} [{detail}]")
                    key = f"port:{addr}:{port}:{pid}"
                    if not self._seen_recently(key, ttl_s=10 * 60):
                        alerts_voice.append(f"Ojo: se abrió un puerto {port} en tu equipo.")
            except Exception:
                pass

            # Suspicious processes
            try:
                suspects = _suspicious_processes()
                for s in suspects[:10]:
                    title = f"Proceso sospechoso (heurístico): {s.name} pid={s.pid} ({s.reason})"
                    detail = f"{s.path}\n{s.command_line}".strip()
                    self._publish("high", title, detail=detail)
                    print(f"[YUI][SEC] {title}\n{s.path}")
                    key = f"proc:{s.pid}:{s.reason}"
                    if not self._seen_recently(key, ttl_s=20 * 60):
                        alerts_voice.append("Ojo: detecté un proceso con señales raras. Revisa la consola.")
            except Exception:
                pass

            # Camera / microphone privacy watch (Windows)
            if _env_bool("YUI_SECURITY_PRIVACY_WATCH_ENABLED", True):
                try:
                    cur = set(privacy_in_use_keys())
                    new_uses = sorted(list(cur - privacy_prev))
                    privacy_prev = cur

                    ignore_raw = (os.getenv("YUI_SECURITY_PRIVACY_IGNORE", "python.exe,pythonw.exe") or "").strip()
                    ignore = [x.strip().lower() for x in ignore_raw.split(",") if x.strip()]

                    for item in new_uses[:12]:
                        low_item = item.lower()
                        if any(x in low_item for x in ignore):
                            continue
                        # item format: "camera|app" or "microphone|app"
                        try:
                            cap, app = item.split("|", 1)
                        except Exception:
                            cap, app = "device", item
                        title = f"Acceso a {cap} detectado: {app}"
                        self._publish("high", title, detail="Si no lo esperabas: cierra la app y revisa permisos de privacidad.")
                        print(f"[YUI][SEC] {title}")
                        key = f"privacy:{cap}:{app}"
                        if not self._seen_recently(key, ttl_s=10 * 60):
                            if cap == "camera":
                                alerts_voice.append("Ojo: otra aplicación encendió la cámara.")
                            elif cap == "microphone":
                                alerts_voice.append("Ojo: otra aplicación activó el micrófono.")
                            else:
                                alerts_voice.append("Ojo: una aplicación activó un dispositivo sensible.")
                except Exception:
                    pass

            # Defender / Firewall health (low frequency)
            health_every_s = float(os.getenv("YUI_SECURITY_WATCH_HEALTH_EVERY_S", "300") or "300")
            if health_every_s > 0 and (now - last_health_ts) >= max(30.0, health_every_s):
                last_health_ts = now
                try:
                    d_ok, f_ok, detail = _defender_firewall_health()
                    if d_ok is False:
                        title = "Defender parece desactivado (o protección incompleta)."
                        self._publish("high", title, detail=detail)
                        print(f"[YUI][SEC] {title} [{detail}]")
                        if not self._seen_recently("health:defender", ttl_s=30 * 60):
                            alerts_voice.append("Ojo: tu antivirus parece desactivado.")
                    if f_ok is False:
                        title = "Firewall parece desactivado (o solo activo en algunos perfiles)."
                        self._publish("high", title, detail=detail)
                        print(f"[YUI][SEC] {title} [{detail}]")
                        if not self._seen_recently("health:firewall", ttl_s=30 * 60):
                            alerts_voice.append("Ojo: tu firewall parece apagado.")
                except Exception:
                    pass

            # Hosts file changes (low frequency)
            hosts_every_s = float(os.getenv("YUI_SECURITY_WATCH_HOSTS_EVERY_S", "180") or "180")
            if hosts_every_s > 0 and (now - last_hosts_ts) >= max(30.0, hosts_every_s):
                last_hosts_ts = now
                try:
                    m = hosts_path.stat().st_mtime if hosts_path.exists() else 0.0
                except Exception:
                    m = hosts_mtime
                if m and hosts_mtime and m != hosts_mtime:
                    hosts_mtime = m
                    title = "Cambió el archivo hosts."
                    detail = str(hosts_path)
                    self._publish("medium", title, detail=detail)
                    print(f"[YUI][SEC] {title} [{detail}]")
                    if not self._seen_recently("hosts:changed", ttl_s=60 * 60):
                        alerts_voice.append("Ojo: cambió el archivo hosts. Revisa si fue intencional.")
                else:
                    hosts_mtime = m

            if alerts_voice and self._should_speak():
                try:
                    # Keep it short to avoid being annoying.
                    msg = alerts_voice[0]
                    if len(alerts_voice) >= 2:
                        msg = "Ojo: detecté cambios de seguridad. Te dejé el detalle en pantalla."
                    self.speak(msg)
                except Exception:
                    pass

            # Sleep with stop responsiveness.
            t_end = time.time() + interval
            while not self._stop.is_set() and time.time() < t_end:
                time.sleep(0.2)
