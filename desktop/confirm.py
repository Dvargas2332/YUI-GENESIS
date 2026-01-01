from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PendingAction:
    nonce: str
    description: str
    created_at: float
    run: Callable[[], None]


class ConfirmGate:
    def __init__(self, *, timeout_s: float = 60.0):
        self.timeout_s = float(timeout_s)
        self.pending: Optional[PendingAction] = None

    def request(self, description: str, run: Callable[[], None]) -> PendingAction:
        nonce = secrets.token_hex(2)  # short code like "a3f2"
        self.pending = PendingAction(nonce=nonce, description=description, created_at=time.time(), run=run)
        return self.pending

    def clear(self) -> None:
        self.pending = None

    def is_expired(self) -> bool:
        if not self.pending:
            return False
        return (time.time() - self.pending.created_at) > self.timeout_s

    def try_confirm(self, text: str) -> bool:
        if not self.pending:
            return False
        if self.is_expired():
            self.clear()
            return False

        t = (text or "").strip().lower()
        if not t:
            return False

        # Accept: "confirmar a3f2", "confirmo a3f2", "sí a3f2"
        ok_words = {"confirmar", "confirmo", "confirmado", "si", "sí", "ok"}
        parts = t.split()
        if len(parts) < 2:
            return False
        if parts[0] not in ok_words:
            return False
        if parts[1] != self.pending.nonce:
            return False

        run = self.pending.run
        self.clear()
        run()
        return True

    def try_cancel(self, text: str) -> bool:
        if not self.pending:
            return False
        t = (text or "").strip().lower()
        if t in {"cancelar", "cancela", "no", "negativo"}:
            self.clear()
            return True
        return False

