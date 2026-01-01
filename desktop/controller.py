from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from desktop.confirm import ConfirmGate


@dataclass(frozen=True)
class DesktopResult:
    handled: bool
    reply: Optional[str] = None


class DesktopController:
    def __init__(self, *, confirm_timeout_s: float = 60.0):
        self.confirm = ConfirmGate(timeout_s=confirm_timeout_s)

    def maybe_handle(self, text: str) -> DesktopResult:
        """
        Parses simple Spanish desktop commands.
        Destructive actions require confirmation via ConfirmGate.
        """
        t = (text or "").strip()
        if not t:
            return DesktopResult(handled=False)

        # First: confirmation flow
        if self.confirm.try_cancel(t):
            return DesktopResult(handled=True, reply="Cancelado.")
        if self.confirm.try_confirm(t):
            return DesktopResult(handled=True, reply="Listo.")
        if self.confirm.is_expired():
            self.confirm.clear()

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
                webbrowser.open(url)
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

    def _open(self, target: str) -> DesktopResult:
        target = (target or "").strip().strip('"')
        if not target:
            return DesktopResult(handled=True, reply="¿Qué quieres que abra?")

        # Try path
        p = Path(target)
        if p.exists():
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
            reply=f"Eso elimina/modifica archivos. Para confirmar escribe: confirmar {pending.nonce} (o 'cancelar').",
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
            reply=f"Eso crea/modifica archivos. Para confirmar escribe: confirmar {pending.nonce} (o 'cancelar').",
        )

    def _confirm_run_command(self, cmd: str) -> DesktopResult:
        cmd = (cmd or "").strip()
        if not cmd:
            return DesktopResult(handled=True, reply="Dime el comando.")

        desc = f"Ejecutar comando: {cmd!r}"

        def run():
            subprocess.Popen(cmd, shell=True)  # nosec - local user requested

        pending = self.confirm.request(desc, run)
        return DesktopResult(
            handled=True,
            reply=f"Eso puede modificar el sistema. Para confirmar escribe: confirmar {pending.nonce} (o 'cancelar').",
        )

