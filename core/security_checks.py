from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SecurityFinding:
    kind: str
    severity: str  # low|medium|high
    message: str


_RE_SQLI = re.compile(
    r"(?is)\b(select|insert|update|delete|drop|alter|union|exec|execute)\b.*(\bor\b|\band\b).*(=|like)|--|/\*|\*/"
)
_RE_SQLI_TAUT = re.compile(r"(?is)(?:'|\")\s*or\s*(?:'|\")?\s*1\s*=\s*1")
_RE_XSS = re.compile(r"(?is)<\s*script\b|on\w+\s*=\s*|javascript:")
_RE_CMD_INJ = re.compile(r"(?is)(?:;|&&|\|\|)\s*(?:curl|wget|powershell|cmd|sh)\b")
_RE_PATH_TRAV = re.compile(r"(?s)\.\./|\.\.\\")


def analyze_text(text: str) -> List[SecurityFinding]:
    t = (text or "").strip()
    if not t:
        return []

    findings: list[SecurityFinding] = []

    if _RE_SQLI_TAUT.search(t) or _RE_SQLI.search(t):
        findings.append(SecurityFinding(kind="sqli", severity="high", message="El texto tiene señales típicas de inyección SQL."))
    if _RE_XSS.search(t):
        findings.append(SecurityFinding(kind="xss", severity="high", message="El texto tiene señales típicas de XSS/JS inyectado."))
    if _RE_CMD_INJ.search(t):
        findings.append(SecurityFinding(kind="cmd_injection", severity="medium", message="El texto parece mezclar separadores de comandos (posible command injection)."))
    if _RE_PATH_TRAV.search(t):
        findings.append(SecurityFinding(kind="path_traversal", severity="medium", message="El texto tiene patrón de path traversal ('../' o '..\\')."))

    return findings


def findings_to_voice(findings: List[SecurityFinding]) -> Optional[str]:
    if not findings:
        return None
    kinds = {f.kind for f in findings}
    if "sqli" in kinds:
        return "Ojo: esto suena a inyección SQL. Si es tu app, usa queries parametrizadas y valida entradas."
    if "xss" in kinds:
        return "Ojo: esto suena a XSS. Escapa/encodea salida y evita inyectar HTML sin sanitizar."
    if "cmd_injection" in kinds:
        return "Ojo: posible inyección de comandos. Evita construir comandos con texto del usuario."
    if "path_traversal" in kinds:
        return "Ojo: posible path traversal. Normaliza rutas y restringe a un directorio permitido."
    return "Detecté patrones de riesgo en ese texto."

