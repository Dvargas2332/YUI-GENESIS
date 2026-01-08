from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Callable, Optional


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() not in {"0", "false", "False", "no", "NO"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


@dataclass
class PendingAction:
    nonce: str
    code: str
    description: str
    created_at: float
    run: Callable[[], None]


class ConfirmGate:
    def __init__(
        self,
        *,
        timeout_s: float = 60.0,
        on_request: Callable[[PendingAction], None] | None = None,
        on_clear: Callable[[PendingAction], None] | None = None,
    ):
        self.timeout_s = float(timeout_s)
        self.pending: Optional[PendingAction] = None
        self._text_arm_until = 0.0
        self._text_arm_nonce: Optional[str] = None
        self._on_request = on_request
        self._on_clear = on_clear

    def request(self, description: str, run: Callable[[], None]) -> PendingAction:
        nonce = secrets.token_hex(2)  # internal id like "a3f2"
        digits = _env_int("YUI_CONFIRM_CODE_DIGITS", 6)
        digits = max(4, min(10, int(digits)))
        code = f"{secrets.randbelow(10**digits):0{digits}d}"
        self.pending = PendingAction(nonce=nonce, code=code, description=description, created_at=time.time(), run=run)
        self._text_arm_until = 0.0
        self._text_arm_nonce = None
        if self._on_request is not None:
            try:
                self._on_request(self.pending)
            except Exception:
                pass
        return self.pending

    def clear(self) -> None:
        if self.pending is None:
            return
        old = self.pending
        self.pending = None
        self._text_arm_until = 0.0
        self._text_arm_nonce = None
        if self._on_clear is not None:
            try:
                self._on_clear(old)
            except Exception:
                pass

    def is_expired(self) -> bool:
        if not self.pending:
            return False
        return (time.time() - self.pending.created_at) > self.timeout_s

    def _text_arm_active(self) -> bool:
        if self.pending is None:
            return False
        if self._text_arm_nonce != self.pending.nonce:
            return False
        return time.time() < float(self._text_arm_until or 0.0)

    def _arm_text(self) -> bool:
        if not _env_bool("YUI_CONFIRM_TEXT_ARM_ENABLED", True):
            return False
        window_s = _env_int("YUI_CONFIRM_TEXT_ARM_WINDOW_S", 30)
        window_s = max(5, min(300, int(window_s)))
        if self.pending is None:
            return False
        self._text_arm_nonce = self.pending.nonce
        self._text_arm_until = time.time() + float(window_s)
        return True

    def handle(self, text: str, *, source: str = "text") -> Optional[str]:
        """
        Handles confirmation/cancel commands while a pending action exists.
        Returns a message if handled, otherwise None.

        Default: confirmation is voice-only and uses a numeric code.
        """
        if not self.pending:
            return None

        raw = (text or "").strip()
        if not raw:
            return None

        if self.is_expired():
            self.clear()
            return "La confirmación expiró."

        t = raw.strip().lower()
        parts = _tokenize(t)

        cancel_words = {"cancelar", "cancela", "no", "negativo"}
        if t in cancel_words or (parts and parts[0] in cancel_words):
            cancel_voice_only = _env_bool("YUI_CONFIRM_CANCEL_VOICE_ONLY", False)
            if cancel_voice_only and source != "voice":
                return "Para cancelar, dilo por voz."
            self.clear()
            return "Cancelado."

        digits = len(self.pending.code)

        # Accept: "confirmar 123456", "confirmar codigo 123456", "si 123456", or typing just "123456".
        ok_words = {"confirmar", "confirmo", "confirmado", "sí", "si", "ok", "dale"}
        looks_like_code_only = _looks_like_code_only(parts, digits=digits)
        if not looks_like_code_only and (not parts or parts[0] not in ok_words):
            return None

        voice_only = _env_bool("YUI_CONFIRM_VOICE_ONLY", True)
        if voice_only and source != "voice" and not self._text_arm_active():
            return (
                "Para confirmar, dilo por voz. Si el reconocimiento de números falla: "
                "di 'confirmar por texto' y luego escribe el código."
            )

        # Optional 2FA fallback: arm typed confirmation with a voice phrase.
        if parts and parts[0] in ok_words and any(p in {"texto", "teclado"} for p in parts):
            if source != "voice" and voice_only:
                return "Para habilitar confirmación por texto, dilo por voz."
            if self._arm_text():
                return "Listo. Escribe el código ahora."
            return "Confirmación por texto desactivada."

        code = _extract_confirmation_code(parts, expected_digits=digits, expected_code=self.pending.code)
        if not code:
            return f"Di: confirmar código {self.pending.code}."

        allow_nonce = _env_bool("YUI_CONFIRM_ALLOW_NONCE", False)
        if code != self.pending.code and not (allow_nonce and code == self.pending.nonce):
            return f"Código incorrecto. Di: confirmar código {self.pending.code}."

        run = self.pending.run
        self.clear()
        run()
        return "Listo."


_DIGIT_WORDS: dict[str, str] = {
    "cero": "0",
    "zero": "0",
    "o": "0",
    "uno": "1",
    "una": "1",
    "un": "1",
    "one": "1",
    "dos": "2",
    "two": "2",
    "tres": "3",
    "three": "3",
    "cuatro": "4",
    "four": "4",
    "cinco": "5",
    "five": "5",
    "seis": "6",
    "six": "6",
    "siete": "7",
    "seven": "7",
    "ocho": "8",
    "eight": "8",
    "nueve": "9",
    "nine": "9",
}


def _tokenize(text: str) -> list[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    text = text.replace(":", " ").replace("-", " ").replace("_", " ").replace("/", " ")
    return [t for t in re.split(r"\s+", text) if t]


def _looks_like_code_only(tokens: list[str], *, digits: int) -> bool:
    if not tokens:
        return False
    if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) == digits:
        return True
    if len(tokens) == digits and all(_token_to_digit(t) is not None for t in tokens):
        return True
    return False


def _token_to_digit(token: str) -> str | None:
    token = (token or "").strip().lower()
    if not token:
        return None
    if token.isdigit() and len(token) == 1:
        return token
    m = re.fullmatch(r"\D*(\d)\D*", token)
    if m:
        return m.group(1)
    return _DIGIT_WORDS.get(token)


def _extract_confirmation_code(tokens: list[str], *, expected_digits: int, expected_code: str) -> Optional[str]:
    """
    Tries to extract a numeric confirmation code.
    Supports: digits, spaced digits, and spoken digits (ES/EN).
    """
    if not tokens:
        return None

    joined_digits = "".join(ch for ch in " ".join(tokens) if ch.isdigit())
    if expected_code and expected_code in joined_digits:
        return expected_code
    if joined_digits and len(joined_digits) == expected_digits:
        return joined_digits

    # Try spoken digits after "codigo"/"código" if present.
    start = 0
    for i, tok in enumerate(tokens):
        if tok in {"codigo", "código"}:
            start = i + 1
            break

    digit_tokens: list[str] = []
    for tok in tokens[start:]:
        d = _token_to_digit(tok)
        if d is None:
            continue
        digit_tokens.append(d)
    if digit_tokens:
        spoken = "".join(digit_tokens)
        if expected_code and expected_code in spoken:
            return expected_code
        if len(spoken) == expected_digits:
            return spoken

    if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) == expected_digits:
        return tokens[0]

    return None
