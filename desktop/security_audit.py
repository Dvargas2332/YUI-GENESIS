from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class SecurityAudit:
    voice: str
    detail: str = ""


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _run_powershell_json(script: str, *, timeout_s: float = 12.0) -> Optional[Any]:
    if not _is_windows():
        return None
    ps = os.getenv("COMSPEC", "")  # unused, but keep environment explicit
    _ = ps
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
        return json.loads(out)
    except Exception:
        return None


def quick_audit() -> SecurityAudit:
    """
    Lightweight security snapshot:
    - Windows Defender state (if available)
    - Windows Firewall profiles
    """
    if not _is_windows():
        return SecurityAudit(voice="Estoy en un sistema no Windows; la auditoría rápida no está disponible.")

    defender = _run_powershell_json("Get-MpComputerStatus | ConvertTo-Json -Compress", timeout_s=10.0)
    firewall = _run_powershell_json("Get-NetFirewallProfile | Select-Object Name,Enabled | ConvertTo-Json -Compress", timeout_s=10.0)

    lines: list[str] = []
    voice_bits: list[str] = []

    if isinstance(defender, dict):
        rt = bool(defender.get("RealTimeProtectionEnabled", False))
        av = bool(defender.get("AntivirusEnabled", False))
        am = bool(defender.get("AMServiceEnabled", False))
        sig = defender.get("AntispywareSignatureLastUpdated") or defender.get("AntivirusSignatureLastUpdated")
        threats = defender.get("ThreatsDetected")
        lines.append("[Defender]")
        lines.append(f"- RealTimeProtectionEnabled: {rt}")
        lines.append(f"- AntivirusEnabled: {av}")
        lines.append(f"- AMServiceEnabled: {am}")
        if sig:
            lines.append(f"- SignatureLastUpdated: {sig}")
        if threats is not None:
            lines.append(f"- ThreatsDetected: {threats}")

        if rt and av:
            voice_bits.append("Defender está activo.")
        else:
            voice_bits.append("Ojo: Defender parece estar desactivado o incompleto.")
    else:
        lines.append("[Defender]")
        lines.append("- No disponible (Get-MpComputerStatus falló o no existe).")
        voice_bits.append("No pude leer el estado de Defender.")

    fw_ok = None
    if isinstance(firewall, list):
        lines.append("[Firewall]")
        enabled_any = False
        enabled_all = True
        for item in firewall:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name", "")).strip()
            enabled = bool(item.get("Enabled", False))
            enabled_any = enabled_any or enabled
            enabled_all = enabled_all and enabled
            lines.append(f"- {name or 'Profile'}: {enabled}")
        if enabled_all:
            fw_ok = True
            voice_bits.append("El firewall está activo.")
        elif enabled_any:
            fw_ok = False
            voice_bits.append("El firewall está activo solo en algunos perfiles.")
        else:
            fw_ok = False
            voice_bits.append("Ojo: el firewall parece estar apagado.")
    else:
        lines.append("[Firewall]")
        lines.append("- No disponible (Get-NetFirewallProfile falló o no existe).")
        voice_bits.append("No pude leer el estado del firewall.")

    voice = " ".join(voice_bits).strip() or "Listo. Hice una auditoría rápida."
    detail = "\n".join(lines).strip()
    return SecurityAudit(voice=voice, detail=detail)


def system_audit() -> SecurityAudit:
    """
    Lightweight system snapshot (read-only):
    - OS / boot time
    - CPU
    - RAM usage (approx)
    - Filesystem drives (free/total)
    - Top processes by CPU time
    """
    if not _is_windows():
        return SecurityAudit(voice="La auditoría del sistema solo está disponible en Windows.")

    os_info = _run_powershell_json(
        "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture,LastBootUpTime,TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress",
        timeout_s=12.0,
    )
    cpu_info = _run_powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress",
        timeout_s=12.0,
    )
    drives = _run_powershell_json(
        "Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | ConvertTo-Json -Compress",
        timeout_s=12.0,
    )
    procs = _run_powershell_json(
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 ProcessName,Id,CPU,WorkingSet | ConvertTo-Json -Compress",
        timeout_s=12.0,
    )

    lines: list[str] = []
    lines.append("[Sistema]")

    if isinstance(os_info, dict):
        caption = str(os_info.get("Caption") or "").strip()
        version = str(os_info.get("Version") or "").strip()
        arch = str(os_info.get("OSArchitecture") or "").strip()
        boot = str(os_info.get("LastBootUpTime") or "").strip()
        if caption or version or arch:
            lines.append(f"- OS: {(caption or 'Windows').strip()} {version} ({arch})".strip())
        if boot:
            lines.append(f"- Boot: {boot}")

        try:
            total_kb = float(os_info.get("TotalVisibleMemorySize") or 0.0)
            free_kb = float(os_info.get("FreePhysicalMemory") or 0.0)
            if total_kb > 0:
                used_kb = max(0.0, total_kb - free_kb)
                used_pct = (used_kb / total_kb) * 100.0
                lines.append(f"- RAM: {used_pct:.0f}% usada (~{used_kb/1024/1024:.1f} GB / {total_kb/1024/1024:.1f} GB)")
        except Exception:
            pass
    else:
        lines.append("- OS: no disponible")

    if isinstance(cpu_info, dict):
        cpu_info = [cpu_info]
    if isinstance(cpu_info, list) and cpu_info:
        c0 = cpu_info[0] if isinstance(cpu_info[0], dict) else {}
        name = str(c0.get("Name") or "").strip()
        cores = c0.get("NumberOfCores")
        threads = c0.get("NumberOfLogicalProcessors")
        mhz = c0.get("MaxClockSpeed")
        bits = [b for b in [name, f"{cores}C" if cores else "", f"{threads}T" if threads else "", f"{mhz}MHz" if mhz else ""] if b]
        if bits:
            lines.append(f"- CPU: {' '.join(str(x) for x in bits)}")

    lines.append("")
    lines.append("[Discos]")
    if isinstance(drives, dict):
        drives = [drives]
    if isinstance(drives, list):
        for d in drives:
            if not isinstance(d, dict):
                continue
            name = str(d.get("Name") or "").strip()
            try:
                used = float(d.get("Used") or 0.0)
                free = float(d.get("Free") or 0.0)
            except Exception:
                used = 0.0
                free = 0.0
            total = used + free
            if name and total > 0:
                lines.append(f"- {name}: libre {free/1024/1024/1024:.1f} GB / {total/1024/1024/1024:.1f} GB")
    else:
        lines.append("- No disponible")

    lines.append("")
    lines.append("[Top procesos (CPU acumulado)]")
    if isinstance(procs, dict):
        procs = [procs]
    if isinstance(procs, list):
        for p in procs:
            if not isinstance(p, dict):
                continue
            name = str(p.get("ProcessName") or "").strip()
            pid = p.get("Id")
            cpu = p.get("CPU")
            ws = p.get("WorkingSet")
            try:
                ws_mb = float(ws or 0.0) / (1024.0 * 1024.0)
            except Exception:
                ws_mb = 0.0
            if name:
                lines.append(f"- {name} (pid={pid}) cpu={cpu} ws={ws_mb:.0f}MB")
    else:
        lines.append("- No disponible")

    voice = "Listo. Hice un autodiagnostico del sistema y te deje el detalle en consola."
    return SecurityAudit(voice=voice, detail="\n".join(lines).strip())


def _copy_sqlite_db(src: Path) -> Optional[Path]:
    try:
        import shutil
        import tempfile
    except Exception:
        return None
    try:
        if not src.exists():
            return None
        tmp_dir = Path(tempfile.mkdtemp(prefix="yui_cookie_audit_"))
        dst = tmp_dir / "Cookies.sqlite"
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def cookies_audit() -> SecurityAudit:
    """
    Privacy-oriented cookie snapshot (Chrome/Edge). Does NOT extract cookie values.
    """
    local = os.getenv("LOCALAPPDATA", "")
    if not local:
        return SecurityAudit(voice="No pude ubicar tu perfil de navegador (LOCALAPPDATA).")

    candidates: list[tuple[str, Path]] = []
    chrome = Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "Network" / "Cookies"
    chrome_old = Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "Cookies"
    edge = Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "Network" / "Cookies"
    edge_old = Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "Cookies"
    for name, p in [
        ("Chrome", chrome),
        ("Chrome", chrome_old),
        ("Edge", edge),
        ("Edge", edge_old),
    ]:
        if p.exists():
            candidates.append((name, p))

    if not candidates:
        return SecurityAudit(voice="No encontré la base de cookies de Chrome/Edge.")

    import sqlite3

    details: list[str] = []
    voice_bits: list[str] = []

    tracking_hints = ("doubleclick", "google-analytics", "googletagmanager", "facebook", "tiktok", "bing", "ads", "adservice")

    for browser, src in candidates[:2]:
        copy = _copy_sqlite_db(src)
        if copy is None:
            continue
        try:
            conn = sqlite3.connect(str(copy))
        except Exception:
            continue
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cookies'")
            if not cur.fetchone():
                continue
            cur.execute("SELECT host_key, name, is_secure, is_httponly, expires_utc FROM cookies")
            rows = cur.fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass

        total = len(rows)
        if total <= 0:
            details.append(f"[{browser}] sin datos o base bloqueada.")
            continue

        per_host: dict[str, int] = {}
        insecure = 0
        not_httponly = 0
        trackers = 0
        for host, cname, is_secure, is_httponly, _expires in rows:
            h = str(host or "").strip().lower()
            per_host[h] = per_host.get(h, 0) + 1
            if not bool(is_secure):
                insecure += 1
            if not bool(is_httponly):
                not_httponly += 1
            if any(t in h for t in tracking_hints):
                trackers += 1

        top = sorted(per_host.items(), key=lambda kv: kv[1], reverse=True)[:8]
        details.append(f"[{browser}] total_cookies={total} insecure={insecure} not_httponly={not_httponly} trackers_hint={trackers}")
        for h, c in top:
            details.append(f"  - {h or '(sin host)'}: {c}")

        voice_bits.append(f"En {browser}, vi {total} cookies.")
        if trackers >= max(5, int(total * 0.2)):
            voice_bits.append("Hay bastantes cookies de rastreo (heurístico).")

    voice = " ".join(voice_bits).strip() or "Listo. Revisé cookies (sin leer valores)."
    detail = "\n".join(details).strip()
    return SecurityAudit(voice=voice, detail=detail)


def _find_mpcmdrun() -> Optional[Path]:
    """
    Best-effort location of Windows Defender CLI scanner.
    """
    if not _is_windows():
        return None
    candidates: list[Path] = []
    pf = os.getenv("ProgramFiles", r"C:\Program Files")
    candidates.append(Path(pf) / "Windows Defender" / "MpCmdRun.exe")

    platform_root = Path(os.getenv("ProgramData", r"C:\ProgramData")) / "Microsoft" / "Windows Defender" / "Platform"
    try:
        if platform_root.exists():
            # Pick newest folder lexicographically (usually versioned).
            newest = sorted([p for p in platform_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)[:3]
            for d in newest:
                candidates.append(d / "MpCmdRun.exe")
    except Exception:
        pass

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def defender_custom_scan(path: Path, *, timeout_s: float = 3600.0) -> SecurityAudit:
    """
    Runs a custom scan on a file/folder using Windows Defender (no remediation).
    """
    if not _is_windows():
        return SecurityAudit(voice="Solo puedo usar el escáner de Defender en Windows.")
    exe = _find_mpcmdrun()
    if exe is None:
        return SecurityAudit(voice="No encontré MpCmdRun.exe (Windows Defender).")

    p = Path(path)
    if not p.exists():
        return SecurityAudit(voice="No encontré esa ruta para escanear.")

    cmd = [str(exe), "-Scan", "-ScanType", "3", "-File", str(p), "-DisableRemediation"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s), check=False)
    except subprocess.TimeoutExpired:
        return SecurityAudit(voice="El escaneo tardó demasiado y lo detuve.", detail="Timeout.")
    except Exception as e:
        return SecurityAudit(voice="No pude ejecutar el escaneo de Defender.", detail=f"{type(e).__name__}: {e}")

    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    out_low = out.lower()
    if "found no threats" in out_low or "no threats" in out_low:
        return SecurityAudit(voice="Listo. No encontré amenazas en ese escaneo.", detail=out)
    if "threat" in out_low or "threats" in out_low or "malware" in out_low:
        return SecurityAudit(voice="Ojo: el escaneo reportó amenazas. Revisa la consola y tu Seguridad de Windows.", detail=out)
    return SecurityAudit(voice="Listo. Terminé el escaneo, pero no pude interpretar el resultado.", detail=out)


def processes_audit() -> SecurityAudit:
    """
    Heuristic scan of running processes (read-only).
    Flags unusual executable paths (Temp/Downloads) and system-name impostors.
    """
    if not _is_windows():
        return SecurityAudit(voice="La auditoría de procesos solo está disponible en Windows.")

    procs = _run_powershell_json(
        "Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress",
        timeout_s=15.0,
    )
    if not isinstance(procs, list):
        return SecurityAudit(voice="No pude leer la lista de procesos.")

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

    suspects: list[tuple[str, int, str, str]] = []
    for p in procs:
        if not isinstance(p, dict):
            continue
        name = str(p.get("Name") or "").strip()
        pid = int(p.get("ProcessId") or 0)
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
        else:
            # No path is common for protected/system processes; don't flag by default.
            reason = ""

        if not reason and cmd:
            if re.search(r"(?i)(?:^|\s)-(?:enc|encodedcommand)\b", cmd) and "powershell" in cmd.lower():
                reason = "PowerShell con comando codificado"
            elif re.search(r"(?i)\b(?:iwr|invoke-webrequest|irm|invoke-restmethod)\b", cmd) and re.search(r"(?i)\b(?:iex|invoke-expression)\b", cmd):
                reason = "descarga y ejecución en PowerShell"

        if reason:
            suspects.append((name, pid, path or "(sin ruta)", reason))

    details: list[str] = []
    details.append("[Procesos sospechosos (heurístico)]")
    if not suspects:
        details.append("- No vi señales claras.")
        return SecurityAudit(voice="No vi procesos claramente sospechosos.", detail="\n".join(details))

    # Show first N suspects; keep stable ordering by reason then name.
    suspects.sort(key=lambda t: (t[3], t[0].lower()))
    for name, pid, path, reason in suspects[:15]:
        details.append(f"- {name} (pid={pid}) [{reason}] {path}")

    voice = f"Encontré {len(suspects)} procesos con señales raras. Te dejo el detalle en consola."
    return SecurityAudit(voice=voice, detail="\n".join(details))


def ports_audit() -> SecurityAudit:
    """
    Lists listening TCP ports and owning processes (read-only).
    """
    if not _is_windows():
        return SecurityAudit(voice="La auditoría de puertos solo está disponible en Windows.")

    conns = _run_powershell_json(
        "Get-NetTCPConnection -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess | ConvertTo-Json -Compress",
        timeout_s=10.0,
    )
    if conns is None:
        # Fallback to netstat (older systems)
        try:
            res = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10.0, check=False)
        except Exception:
            return SecurityAudit(voice="No pude listar puertos en escucha.")
        out = (res.stdout or "").splitlines()
        listens: list[tuple[str, int, int]] = []
        for line in out:
            if "LISTENING" not in line.upper():
                continue
            parts = [p for p in line.split() if p]
            # Proto LocalAddr ForeignAddr State PID
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
            listens.append((addr, port, pid))
        conns = [{"LocalAddress": a, "LocalPort": p, "OwningProcess": pid} for a, p, pid in listens]

    if isinstance(conns, dict):
        conns = [conns]
    if not isinstance(conns, list):
        return SecurityAudit(voice="No pude leer las conexiones en escucha.")

    proc_map: dict[int, str] = {}
    ps_procs = _run_powershell_json("Get-Process | Select-Object Id,ProcessName | ConvertTo-Json -Compress", timeout_s=8.0)
    if isinstance(ps_procs, list):
        for it in ps_procs:
            if not isinstance(it, dict):
                continue
            try:
                pid = int(it.get("Id") or 0)
                name = str(it.get("ProcessName") or "").strip()
                if pid > 0 and name:
                    proc_map[pid] = name
            except Exception:
                continue

    rows: list[tuple[str, int, int, str]] = []
    for c in conns:
        if not isinstance(c, dict):
            continue
        addr = str(c.get("LocalAddress") or "").strip()
        try:
            port = int(c.get("LocalPort") or 0)
            pid = int(c.get("OwningProcess") or 0)
        except Exception:
            continue
        if port <= 0:
            continue
        pname = proc_map.get(pid, "")
        rows.append((addr or "?", port, pid, pname))

    if not rows:
        return SecurityAudit(voice="No vi puertos en escucha.")

    rows.sort(key=lambda t: (t[1], t[0]))
    details: list[str] = []
    details.append("[Puertos en escucha]")
    for addr, port, pid, pname in rows[:25]:
        bind = "all" if addr in {"0.0.0.0", "::", "*"} else addr
        who = f"{pname} (pid={pid})" if pname else f"pid={pid}"
        details.append(f"- {bind}:{port} -> {who}")

    voice = f"Veo {len(rows)} puertos TCP en escucha. Te muestro los primeros en consola."
    return SecurityAudit(voice=voice, detail="\n".join(details))


def extensions_audit() -> SecurityAudit:
    """
    Chrome/Edge extensions permission snapshot (read-only).
    Flags extensions with broad permissions (heuristic).
    """
    local = os.getenv("LOCALAPPDATA", "")
    if not local:
        return SecurityAudit(voice="No pude ubicar tu perfil (LOCALAPPDATA).")

    bases: list[tuple[str, Path]] = []
    bases.append(("Chrome", Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "Extensions"))
    bases.append(("Edge", Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "Extensions"))

    high_risk = {
        "<all_urls>",
        "webRequest",
        "webRequestBlocking",
        "downloads",
        "history",
        "management",
        "proxy",
        "nativeMessaging",
        "debugger",
        "clipboardRead",
    }

    findings: list[str] = []
    total_ext = 0
    risky_ext = 0

    for browser, root in bases:
        if not root.exists():
            continue
        try:
            ids = [p for p in root.iterdir() if p.is_dir()]
        except Exception:
            continue

        for ext_dir in ids:
            total_ext += 1
            try:
                versions = [p for p in ext_dir.iterdir() if p.is_dir()]
            except Exception:
                continue
            if not versions:
                continue
            versions.sort(key=lambda p: p.name, reverse=True)
            manifest = versions[0] / "manifest.json"
            if not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            name = str(data.get("name") or "").strip()
            perms = data.get("permissions") or []
            host_perms = data.get("host_permissions") or []
            perms_list = []
            if isinstance(perms, list):
                perms_list.extend([str(x) for x in perms if isinstance(x, (str, int, float))])
            if isinstance(host_perms, list):
                perms_list.extend([str(x) for x in host_perms if isinstance(x, (str, int, float))])
            perms_norm = {p.strip() for p in perms_list if str(p).strip()}

            hit = sorted([p for p in perms_norm if p in high_risk])
            all_urls = any("<all_urls>" in p for p in perms_norm) or any("*://*/*" in p for p in perms_norm)
            if all_urls and "<all_urls>" not in hit:
                hit.append("<all_urls>")

            if hit:
                risky_ext += 1
                who = name if (name and not name.startswith("__MSG_")) else ext_dir.name
                findings.append(f"[{browser}] {who} perms={','.join(hit)}")

    detail = ""
    if findings:
        detail = "[Extensiones con permisos amplios (heurístico)]\n" + "\n".join(findings[:30])

    if total_ext <= 0:
        return SecurityAudit(voice="No encontré extensiones en Chrome/Edge.", detail=detail)
    if risky_ext <= 0:
        return SecurityAudit(voice=f"Encontré {total_ext} extensiones y no vi permisos claramente riesgosos.", detail=detail)

    return SecurityAudit(
        voice=f"Encontré {total_ext} extensiones; {risky_ext} tienen permisos amplios. Te dejo el detalle en consola.",
        detail=detail,
    )
