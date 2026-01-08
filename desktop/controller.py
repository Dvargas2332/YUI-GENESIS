from __future__ import annotations

import os
import re
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from desktop.confirm import ConfirmGate
from desktop.security_guard import guard_enabled, is_high_risk_path, is_suspicious_command, is_suspicious_url, normalize_url


@dataclass(frozen=True)
class DesktopResult:
    handled: bool
    reply: Optional[str] = None


class DesktopController:
    def __init__(
        self,
        *,
        confirm_timeout_s: float = 60.0,
        confirm_on_request=None,  # noqa: ANN001
        confirm_on_clear=None,  # noqa: ANN001
    ):
        self.confirm = ConfirmGate(timeout_s=confirm_timeout_s, on_request=confirm_on_request, on_clear=confirm_on_clear)
        self._pointer_ok_until = 0.0

    def maybe_handle(self, text: str, *, source: str = "text") -> DesktopResult:
        """
        Parses simple Spanish desktop commands.
        Destructive actions require confirmation via ConfirmGate.
        """
        t = (text or "").strip()
        if not t:
            return DesktopResult(handled=False)

        # First: confirmation flow
        msg = self.confirm.handle(t, source=source)
        if msg is not None:
            return DesktopResult(handled=True, reply=msg)
        if self.confirm.pending is not None:
            return DesktopResult(
                handled=True,
                reply=f"Tengo una confirmación pendiente. Di: confirmar código {self.confirm.pending.code} (o 'cancelar').",
            )

        low = t.lower()

        if low.startswith("abre "):
            target = t[5:].strip()
            return self._open(target)

        if low.startswith("busca "):
            q = t[6:].strip()
            if q:
                url = "https://www.google.com/search?q=" + webbrowser.quote(q)  # type: ignore[attr-defined]
                webbrowser.open(url)
                return DesktopResult(handled=True, reply="Abro la búsqueda.")
            return DesktopResult(handled=True, reply="¿Qué quieres que busque?")

        if low.startswith("url "):
            url = t[4:].strip()
            if url:
                url_norm = normalize_url(url)
                if guard_enabled():
                    sus, reason = is_suspicious_url(url_norm)
                    if sus:
                        return self._confirm_open_url(url_norm, reason)
                webbrowser.open(url_norm)
                return DesktopResult(handled=True, reply="Abriendo.")
            return DesktopResult(handled=True, reply="Dime la URL.")

        if low.startswith("escribe "):
            payload = t[8:]
            return self._type_text(payload)

        if low in {"enter", "intro"}:
            return self._press("enter")
        if low in {"tab"}:
            return self._press("tab")
        if low in {"esc", "escape"}:
            return self._press("esc")

        if low.startswith("tecla "):
            key = t[6:].strip()
            return self._press(key)

        if low.startswith("atajo "):
            combo = t[6:].strip()
            return self._hotkey(combo)

        if low.startswith("click derecho "):
            rest = t[len("click derecho ") :].strip()
            return self._click(rest, button="right", double=False)

        if low.startswith("doble click "):
            rest = t[len("doble click ") :].strip()
            return self._click(rest, button="left", double=True)

        if low.startswith("click "):
            rest = t[len("click ") :].strip()
            return self._click(rest, button="left", double=False)

        if low.startswith("mueve mouse ") or low.startswith("mueve el mouse "):
            rest = t.split("mouse", 1)[-1].strip()
            return self._move_mouse(rest)

        if low.startswith("scroll "):
            rest = t[len("scroll ") :].strip()
            return self._scroll(rest)

        if low.startswith("espera "):
            rest = t[len("espera ") :].strip()
            return self._wait(rest)

        if low.startswith("elimina "):
            path = t[8:].strip().strip('"')
            return self._confirm_delete(path)

        if low.startswith("crea archivo "):
            rest = t[len("crea archivo ") :].strip()
            if " con " in rest:
                path, content = rest.split(" con ", 1)
                return self._confirm_write_file(path.strip().strip('"'), content)
            return DesktopResult(handled=True, reply="Usa: crea archivo RUTA con CONTENIDO")

        if low.startswith("ejecuta "):
            cmd = t[len("ejecuta ") :].strip()
            return self._confirm_run_command(cmd)

        return DesktopResult(handled=False)

    def _pointer_confirmation_needed(self) -> bool:
        require = os.getenv("YUI_DESKTOP_CONFIRM_POINTER", "1") not in {"0", "false", "False"}
        if not require:
            return False
        return time.time() >= float(self._pointer_ok_until or 0.0)

    def _confirm_pointer_action(self, desc: str, run) -> DesktopResult:  # noqa: ANN001
        grace = float(os.getenv("YUI_DESKTOP_POINTER_GRACE_S", "60") or "60")

        def wrapped():
            self._pointer_ok_until = time.time() + max(5.0, grace)
            run()

        pending = self.confirm.request(desc, wrapped)
        return DesktopResult(
            handled=True,
            reply=f"Eso controla el mouse/scroll. Para confirmar di: confirmar código {pending.code} (o 'cancelar').",
        )

    def _open(self, target: str) -> DesktopResult:
        target = (target or "").strip().strip('"')
        if not target:
            return DesktopResult(handled=True, reply="¿Qué quieres que abra?")

        # Try path
        p = Path(target)
        if p.exists():
            if guard_enabled():
                risky, reason = is_high_risk_path(p)
                if risky:
                    return self._confirm_open_path(p, reason)
            os.startfile(str(p))  # nosec - local user requested
            return DesktopResult(handled=True, reply="Abriendo.")

        # Common apps shortcuts
        app = target.lower()
        mapping = {
            "chrome": "chrome.exe",
            "navegador": "msedge.exe",
            "edge": "msedge.exe",
            "bloc de notas": "notepad.exe",
            "notepad": "notepad.exe",
            "calculadora": "calc.exe",
            "explorador": "explorer.exe",
        }
        exe = mapping.get(app)
        try:
            subprocess.Popen(exe or target, shell=False)  # nosec - local user requested
            return DesktopResult(handled=True, reply="Abriendo.")
        except Exception:
            return DesktopResult(handled=True, reply="No pude abrir eso.")

    def _confirm_open_path(self, path: Path, reason: str) -> DesktopResult:
        desc = f"Abrir '{path}'. Motivo: {reason}"

        def run():
            os.startfile(str(path))  # nosec - local user requested

        pending = self.confirm.request(desc, run)
        return DesktopResult(
            handled=True,
            reply=f"Eso podría ejecutar código. Para confirmar di: confirmar código {pending.code} (o 'cancelar').",
        )

    def _confirm_open_url(self, url: str, reason: str) -> DesktopResult:
        desc = f"Abrir URL '{url}'. Motivo: {reason}"

        def run():
            webbrowser.open(url)

        pending = self.confirm.request(desc, run)
        return DesktopResult(
            handled=True,
            reply=f"Esa URL parece sospechosa. Para confirmar di: confirmar código {pending.code} (o 'cancelar').",
        )

    def _type_text(self, text: str) -> DesktopResult:
        try:
            import pyautogui
        except Exception:
            return DesktopResult(handled=True, reply="Falta instalar pyautogui para escribir en pantalla.")
        pyautogui.typewrite(text, interval=0.01)
        return DesktopResult(handled=True, reply="Hecho.")

    def _press(self, key: str) -> DesktopResult:
        try:
            import pyautogui
        except Exception:
            return DesktopResult(handled=True, reply="Falta instalar pyautogui.")
        pyautogui.press(key)
        return DesktopResult(handled=True, reply="Ok.")

    def _hotkey(self, combo: str) -> DesktopResult:
        try:
            import pyautogui
        except Exception:
            return DesktopResult(handled=True, reply="Falta instalar pyautogui.")
        keys = [k.strip().lower() for k in combo.replace("+", " ").split() if k.strip()]
        if not keys:
            return DesktopResult(handled=True, reply="Dime el atajo, por ejemplo: atajo ctrl l")
        pyautogui.hotkey(*keys)
        return DesktopResult(handled=True, reply="Ok.")

    def _parse_xy(self, rest: str) -> Optional[Tuple[int, int]]:
        s = (rest or "").strip()
        if not s:
            return None
        m = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", s)
        if not m:
            return None
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None

    def _click(self, rest: str, *, button: str, double: bool) -> DesktopResult:
        xy = self._parse_xy(rest)
        if not xy:
            return DesktopResult(handled=True, reply="Usa: click 100 200 (o 'click derecho 100 200').")
        x, y = xy
        try:
            import pyautogui
        except Exception:
            return DesktopResult(handled=True, reply="Falta instalar pyautogui.")

        def run():
            if double:
                pyautogui.doubleClick(x=x, y=y, button=button)
            else:
                pyautogui.click(x=x, y=y, button=button)

        if self._pointer_confirmation_needed():
            kind = "doble click" if double else "click"
            return self._confirm_pointer_action(f"{kind} {button} en ({x},{y})", run)
        run()
        return DesktopResult(handled=True, reply="Hecho.")

    def _move_mouse(self, rest: str) -> DesktopResult:
        xy = self._parse_xy(rest)
        if not xy:
            return DesktopResult(handled=True, reply="Usa: mueve mouse 100 200")
        x, y = xy
        try:
            import pyautogui
        except Exception:
            return DesktopResult(handled=True, reply="Falta instalar pyautogui.")

        def run():
            pyautogui.moveTo(x=x, y=y, duration=0.0)

        if self._pointer_confirmation_needed():
            return self._confirm_pointer_action(f"mover mouse a ({x},{y})", run)
        run()
        return DesktopResult(handled=True, reply="Ok.")

    def _scroll(self, rest: str) -> DesktopResult:
        try:
            amt = int(float((rest or "").strip()))
        except Exception:
            return DesktopResult(handled=True, reply="Usa: scroll -300 o scroll 300")
        try:
            import pyautogui
        except Exception:
            return DesktopResult(handled=True, reply="Falta instalar pyautogui.")

        def run():
            pyautogui.scroll(int(amt))

        if self._pointer_confirmation_needed():
            return self._confirm_pointer_action(f"scroll {amt}", run)
        run()
        return DesktopResult(handled=True, reply="Ok.")

    def _wait(self, rest: str) -> DesktopResult:
        try:
            s = float((rest or "").strip())
        except Exception:
            return DesktopResult(handled=True, reply="Usa: espera 1.5")
        s = max(0.0, min(600.0, float(s)))
        time.sleep(s)
        return DesktopResult(handled=True, reply="Ok.")

    def _confirm_delete(self, path: str) -> DesktopResult:
        p = Path(path)
        desc = f"Eliminar '{p}'."

        def run():
            if p.is_dir():
                for child in p.rglob("*"):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                # attempt to remove dirs bottom-up
                for child in sorted(p.rglob("*"), reverse=True):
                    if child.is_dir():
                        try:
                            child.rmdir()
                        except Exception:
                            pass
                try:
                    p.rmdir()
                except Exception:
                    pass
            else:
                p.unlink(missing_ok=True)

        pending = self.confirm.request(desc, run)
        return DesktopResult(
            handled=True,
            reply=f"Eso elimina/modifica archivos. Para confirmar di: confirmar código {pending.code} (o 'cancelar').",
        )

    def _confirm_write_file(self, path: str, content: str) -> DesktopResult:
        p = Path(path)
        desc = f"Crear/modificar archivo '{p}'."

        def run():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        pending = self.confirm.request(desc, run)
        return DesktopResult(
            handled=True,
            reply=f"Eso crea/modifica archivos. Para confirmar di: confirmar código {pending.code} (o 'cancelar').",
        )

    def _confirm_run_command(self, cmd: str) -> DesktopResult:
        cmd = (cmd or "").strip()
        if not cmd:
            return DesktopResult(handled=True, reply="Dime el comando.")

        warn = ""
        if guard_enabled():
            sus, reason = is_suspicious_command(cmd)
            if sus:
                warn = f" Aviso: {reason}"

        desc = f"Ejecutar comando: {cmd!r}.{warn}".strip()

        def run():
            subprocess.Popen(cmd, shell=True)  # nosec - local user requested

        pending = self.confirm.request(desc, run)
        return DesktopResult(
            handled=True,
            reply=f"Eso puede modificar el sistema.{warn} Para confirmar di: confirmar código {pending.code} (o 'cancelar').",
        )
