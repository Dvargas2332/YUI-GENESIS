from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse


def guard_enabled() -> bool:
    return os.getenv("YUI_SECURITY_GUARD", "1") not in {"0", "false", "False", "no", "NO"}


_HIGH_RISK_EXTS = {
    ".exe",
    ".msi",
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".js",
    ".jar",
    ".scr",
    ".lnk",
    ".reg",
    ".hta",
    ".com",
}

_MACRO_DOC_EXTS = {".docm", ".xlsm", ".pptm"}


def is_high_risk_path(path: Path) -> tuple[bool, str]:
    """
    True when opening the path is likely to execute code or carry macros.
    """
    try:
        name = path.name.lower()
        ext = path.suffix.lower()
    except Exception:
        return False, ""

    if ext in _HIGH_RISK_EXTS:
        return True, f"Es un archivo ejecutable ({ext})."
    if ext in _MACRO_DOC_EXTS:
        return True, f"Es un documento con macros ({ext})."

    # Common double-extension trick: "factura.pdf.exe"
    if re.search(r"\.(pdf|docx?|xlsx?|pptx?|jpg|png|txt)\.(exe|scr|bat|cmd|ps1|vbs|js|msi)$", name):
        return True, "Parece un archivo con doble extensión."

    return False, ""


def normalize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        return "https://" + raw
    return raw


def is_suspicious_url(raw: str) -> tuple[bool, str]:
    """
    Heuristics only (no reputation lookup). Avoids opening obvious phishing / unsafe URLs by accident.
    """
    url = normalize_url(raw)
    if not url:
        return False, ""

    try:
        p = urlparse(url)
    except Exception:
        return True, "No pude interpretar la URL."

    scheme = (p.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return True, f"Esquema no soportado: {scheme!r}."

    netloc = p.netloc or ""
    if "@" in netloc:
        return True, "La URL contiene '@' (patrón típico de phishing)."

    host = (p.hostname or "").strip().lower()
    if not host:
        return True, "No se detectó host en la URL."

    # Prefer HTTPS unless user explicitly wants HTTP.
    if scheme == "http" and os.getenv("YUI_SECURITY_REQUIRE_HTTPS", "1") not in {"0", "false", "False"}:
        return True, "No usa HTTPS."

    # Very long / deeply nested hostnames are often suspicious.
    if len(host) >= 70 and host.count(".") >= 4:
        return True, "Host demasiado largo/complicado."

    # Rare / abused TLDs (heuristic).
    tlds = {".zip", ".mov", ".xyz", ".top", ".click", ".work", ".rest", ".cam", ".gq", ".tk", ".ml", ".cf"}
    if any(host.endswith(tld) for tld in tlds):
        return True, "Dominio con TLD de alto riesgo (heurístico)."

    # Optional user-provided blocklist.
    block = (os.getenv("YUI_SECURITY_URL_BLOCKLIST", "") or "").strip()
    if block:
        for item in [x.strip().lower() for x in block.split(",") if x.strip()]:
            if host == item or host.endswith("." + item):
                return True, "Dominio en lista de bloqueo."

    return False, ""


def is_suspicious_command(cmd: str) -> tuple[bool, str]:
    """
    Heuristic flags for potentially dangerous command lines.
    This is defensive-only: it doesn't block, it just adds extra warning/confirmation.
    """
    c = (cmd or "").strip()
    if not c:
        return False, ""

    low = c.lower()

    # Common "download + execute" patterns
    if "powershell" in low:
        if re.search(r"(?i)(?:^|\s)-(?:enc|encodedcommand)\b", c):
            return True, "PowerShell con comando codificado."
        if re.search(r"(?i)\b(?:iex|invoke-expression)\b", c):
            return True, "PowerShell ejecutando texto (IEX)."
        if re.search(r"(?i)\b(?:iwr|invoke-webrequest|irm|invoke-restmethod)\b", c) and re.search(r"(?i)\b(?:iex|invoke-expression)\b", c):
            return True, "PowerShell descargando y ejecutando contenido."

    if re.search(r"(?i)\b(?:curl|wget)\b", c) and re.search(r"(?i)\|\s*(?:sh|bash|powershell|cmd)\b", c):
        return True, "Descarga y ejecución encadenada (pipe a intérprete)."

    # LOLBins frequently abused by malware
    if re.search(r"(?i)\b(?:certutil|bitsadmin|mshta|rundll32|regsvr32|wmic)\b", c):
        return True, "Usa utilidades del sistema comúnmente abusadas (LOLBins)."

    # Persistence mechanisms
    if re.search(r"(?i)\b(?:schtasks)\b.*\b/create\b", c) or re.search(r"(?i)\bNew-ScheduledTask\b", c):
        return True, "Crea tareas programadas (persistencia)."
    if re.search(r"(?i)\\software\\microsoft\\windows\\currentversion\\run\b", c):
        return True, "Modifica claves Run (persistencia)."

    # Clear log / hide traces
    if re.search(r"(?i)\bwevtutil\b.*\bcl\b", c):
        return True, "Limpia eventos (puede ocultar actividad)."

    return False, ""
