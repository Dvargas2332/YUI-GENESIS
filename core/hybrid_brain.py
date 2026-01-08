from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from config.settings import Settings
from core.brain import Brain, VisualContext
from core.memory import MemoryStore


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in {"0", "false", "False", "no", "NO"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


STYLE_BANNED_KEY = "style.banned_phrases"
STYLE_MAX_SENTENCES_KEY = "style.max_sentences"

MACROS_ENABLED_ENV = "YUI_MACROS_ENABLED"

TEACHING_MODE_KEY = "mode.teaching"


@dataclass
class StyleProfile:
    banned_phrases: List[str]
    max_sentences: int

    def normalize(self) -> "StyleProfile":
        banned: List[str] = []
        seen = set()
        for p in self.banned_phrases:
            s = (p or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            banned.append(s)
        max_sent = int(self.max_sentences or 0)
        if max_sent < 0:
            max_sent = 0
        return StyleProfile(banned_phrases=banned, max_sentences=max_sent)


class StylePolicy:
    """
    Local "style brain": learns preferences and enforces them even if the LLM misbehaves.
    Stores preferences in MemoryStore.preferences.
    """

    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.enabled = _env_bool("YUI_STYLE_ENABLED", True)
        self.profile = self._load().normalize()

    def _load(self) -> StyleProfile:
        banned_raw = self.memory.get_preference(STYLE_BANNED_KEY)
        banned: List[str] = []
        if banned_raw:
            try:
                obj = json.loads(banned_raw)
                if isinstance(obj, list):
                    banned = [str(x) for x in obj if isinstance(x, (str, int, float)) and str(x).strip()]
            except Exception:
                banned = []
        if not banned:
            env_banned = (os.getenv("YUI_STYLE_BANNED_PHRASES", "") or "").strip()
            if env_banned:
                parts = re.split(r"[,\|]\s*", env_banned)
                banned = [p.strip() for p in parts if p.strip()]

        max_sent_raw = self.memory.get_preference(STYLE_MAX_SENTENCES_KEY)
        if max_sent_raw and max_sent_raw.strip().isdigit():
            max_sent = int(max_sent_raw.strip())
        else:
            max_sent = _env_int("YUI_STYLE_MAX_SENTENCES", 3)

        return StyleProfile(banned_phrases=banned, max_sentences=max_sent)

    def _save(self) -> None:
        prof = self.profile.normalize()
        try:
            self.memory.set_preference(STYLE_BANNED_KEY, json.dumps(prof.banned_phrases, ensure_ascii=False))
            self.memory.set_preference(STYLE_MAX_SENTENCES_KEY, str(int(prof.max_sentences)))
        except Exception:
            pass

    def system_prompt(self) -> str:
        if not self.enabled:
            return ""
        prof = self.profile.normalize()
        lines: List[str] = []
        if prof.max_sentences > 0:
            lines.append(f"- Por voz: máximo {prof.max_sentences} frases.")
        if prof.banned_phrases:
            limited = prof.banned_phrases[:12]
            joined = " | ".join([f"'{p}'" for p in limited])
            lines.append(f"- Nunca uses estas frases exactas: {joined}.")
        if not lines:
            return ""
        return "Estilo personalizado del usuario (cumplimiento obligatorio):\n" + "\n".join(lines)

    def postprocess(self, text: str) -> str:
        return self._postprocess(text, enforce_max_sentences=True)

    def postprocess_display(self, text: str) -> str:
        return self._postprocess(text, enforce_max_sentences=False)

    def _postprocess(self, text: str, *, enforce_max_sentences: bool) -> str:
        out = (text or "").strip()
        if not out or not self.enabled:
            return out
        prof = self.profile.normalize()

        # Remove banned phrases (best-effort, case-insensitive).
        for phrase in prof.banned_phrases:
            p = (phrase or "").strip()
            if not p:
                continue
            try:
                out = re.sub(re.escape(p), "", out, flags=re.IGNORECASE)
            except Exception:
                out = out.replace(p, "")

        out = re.sub(r"\s{2,}", " ", out).strip()
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)

        # Enforce max sentences for spoken output (keeps responses snappy).
        if enforce_max_sentences and prof.max_sentences > 0:
            parts = re.split(r"(?<=[\.\!\?])\s+", out)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > prof.max_sentences:
                out = " ".join(parts[: prof.max_sentences]).strip()

        return out.strip()

    def _clean_phrase(self, raw: str) -> str:
        s = (raw or "").strip()
        s = s.strip(" \t\r\n\"'“”‘’")
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def _try_extract_ban(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        low = t.lower()
        patterns = [
            r"^no\s+digas\s+(.+)$",
            r"^no\s+vuelvas\s+a\s+decir\s+(.+)$",
            r"^deja\s+de\s+decir\s+(.+)$",
            r"^prohibido\s+decir\s+(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, low, flags=re.IGNORECASE)
            if not m:
                continue
            # Use original text slice for better casing.
            phrase = t[m.start(1) :].strip()
            phrase = self._clean_phrase(phrase)
            # Avoid accidental training on one-word phrases like "no digas nada".
            if len(phrase) < 3:
                return None
            if len(phrase.split()) < 2 and len(phrase) < 8:
                return None
            return phrase
        return None

    def _try_extract_unban(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        low = t.lower()
        patterns = [
            r"^puedes\s+decir\s+(.+)$",
            r"^permite\s+decir\s+(.+)$",
            r"^vuelve\s+a\s+decir\s+(.+)$",
            r"^quita\s+la\s+prohibici[oó]n\s+de\s+(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, low, flags=re.IGNORECASE)
            if not m:
                continue
            phrase = t[m.start(1) :].strip()
            phrase = self._clean_phrase(phrase)
            if len(phrase) < 3:
                return None
            if len(phrase.split()) < 2 and len(phrase) < 8:
                return None
            return phrase
        return None

    def _try_extract_max_sentences(self, text: str) -> Optional[int]:
        t = (text or "").strip().lower()
        if t in {"modo corto", "responde corto", "responde más corto", "responde mas corto"}:
            return 2
        if t in {"modo normal", "responde normal"}:
            return 3
        if t in {"modo largo", "responde largo"}:
            return 0
        m = re.search(r"(?:m[aá]ximo|maximo)\s+(\d+)\s+(?:frases|oraciones)", t)
        if m:
            try:
                return max(0, int(m.group(1)))
            except Exception:
                return None
        m2 = re.search(r"responde\s+en\s+(\d+)\s+(?:frases|oraciones)", t)
        if m2:
            try:
                return max(0, int(m2.group(1)))
            except Exception:
                return None
        return None

    def maybe_handle_training(self, user_text: str) -> Optional[str]:
        if not self.enabled:
            return None

        t = (user_text or "").strip()
        if not t:
            return None
        low = t.lower().strip()

        # Explicit style prefix
        if low.startswith("estilo:") or low.startswith("estilo "):
            inner = t.split(":", 1)[-1].strip() if ":" in t else t.split(" ", 1)[-1].strip()
            inner_low = inner.lower().strip()
            if inner_low in {"", "ver", "mostrar", "estado"}:
                return self.describe()
            # Allow nested commands like "estilo: no digas ..."
            t = inner
            low = inner_low

        if low in {"mi estilo", "ver estilo", "mostrar estilo", "estilo"}:
            return self.describe()

        if low in {"reinicia estilo", "reset estilo", "restablece estilo", "borra estilo"}:
            self.profile = StyleProfile(banned_phrases=[], max_sentences=_env_int("YUI_STYLE_MAX_SENTENCES", 3)).normalize()
            self._save()
            return "Listo. Reinicié tu estilo."

        max_sent = self._try_extract_max_sentences(t)
        if max_sent is not None:
            self.profile = StyleProfile(banned_phrases=self.profile.banned_phrases, max_sentences=max_sent).normalize()
            self._save()
            if max_sent <= 0:
                return "Listo. Sin límite de frases."
            return f"Listo. Máximo {max_sent} frases."

        ban = self._try_extract_ban(t)
        if ban:
            prof = self.profile.normalize()
            prof.banned_phrases.append(ban)
            self.profile = prof.normalize()
            self._save()
            return "Listo."

        unban = self._try_extract_unban(t)
        if unban:
            prof = self.profile.normalize()
            prof.banned_phrases = [p for p in prof.banned_phrases if p.strip().lower() != unban.strip().lower()]
            self.profile = prof.normalize()
            self._save()
            return "Ok."

        return None

    def describe(self) -> str:
        prof = self.profile.normalize()
        parts: List[str] = []
        if prof.max_sentences > 0:
            parts.append(f"máximo {prof.max_sentences} frases")
        else:
            parts.append("sin límite de frases")
        if prof.banned_phrases:
            parts.append(f"{len(prof.banned_phrases)} frases prohibidas")
        return "Estilo: " + ", ".join(parts) + "."


class HybridBrain:
    """
    Hybrid brain:
    - Local policy brain learns preferences (fast, deterministic, persistent)
    - LLM brain handles language/reasoning when needed
    """

    def __init__(self, settings: Settings, memory: MemoryStore):
        self.settings = settings
        self.memory = memory
        self.llm = Brain(settings, memory)
        self.style = StylePolicy(memory)
        self.macros = MacroPolicy(memory)
        self.teaching = TeachingModePolicy(memory)
        self.last_display_text: Optional[str] = None

    @property
    def last_mode(self) -> str:
        return getattr(self.llm, "last_mode", "fast")

    def route(self, user_text: str) -> Tuple[str, float, str]:
        return self.llm.route(user_text)

    def health_check(self) -> bool:
        return self.llm.health_check()

    def reply(
        self,
        user_text: str,
        *,
        visual: VisualContext,
        screen_context: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_mode: Optional[str] = None,
    ) -> str:
        # Reset last display (used by UI/console).
        self.last_display_text = None

        # 0) Teaching mode toggles
        teach = self.teaching.maybe_handle(user_text)
        if teach is not None:
            try:
                self.memory.add("user", (user_text or "").strip())
                self.memory.add("assistant", teach)
            except Exception:
                pass
            return teach

        # 1) Local learning (style)
        local = self.style.maybe_handle_training(user_text)
        if local is not None:
            try:
                self.memory.add("user", (user_text or "").strip())
                self.memory.add("assistant", local)
            except Exception:
                pass
            return local

        # 2) Macro training / shortcuts (tasks brain)
        macro_train = self.macros.maybe_handle_training(user_text)
        if macro_train is not None:
            try:
                self.memory.add("user", (user_text or "").strip())
                self.memory.add("assistant", macro_train)
            except Exception:
                pass
            return macro_train

        # 3) LLM (with local policy prompt + postprocess enforcement)
        extra_parts = []
        style_prompt = self.style.system_prompt()
        if style_prompt:
            extra_parts.append(style_prompt)
        teach_prompt = self.teaching.system_prompt()
        if teach_prompt:
            extra_parts.append(teach_prompt)
        if screen_context:
            extra_parts.append("Contexto de pantalla (no lo repitas literalmente; úsalo para ser más útil):\n" + screen_context.strip())
        extra = "\n\n".join(extra_parts).strip() if extra_parts else None

        if self.teaching.enabled and self.teaching.print_full:
            # Keep full text for display, but keep voice short.
            full = self.llm.reply(
                user_text,
                visual=visual,
                llm_model=llm_model,
                llm_temperature=llm_temperature,
                llm_mode=llm_mode,
                extra_system_prompt=extra,
                postprocess=self.style.postprocess_display if self.style.enabled else None,
            )
            voice = self.style.postprocess(full) if self.style.enabled else full
            if full and voice and full.strip() != voice.strip():
                self.last_display_text = full.strip()
            return voice.strip() if voice else full

        post: Optional[Callable[[str], str]] = self.style.postprocess if self.style.enabled else None
        return self.llm.reply(
            user_text,
            visual=visual,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_mode=llm_mode,
            extra_system_prompt=extra,
            postprocess=post,
        )

    def preprocess(self, user_text: str) -> str:
        """
        Runs before command routing (desktop/local skills/LLM).
        Expands learned macro triggers into actions.
        """
        return self.macros.expand_or_original(user_text)


def _normalize_trigger(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


class MacroPolicy:
    """
    Local task brain: learn simple "when I say X, do Y" shortcuts.

    Stored in MemoryStore.macros as: trigger -> action
    Action is a normal command string already understood by YUI,
    e.g. "ejecuta tarea limpiar temp" or "abre chrome" or "auditoria seguridad".
    """

    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.enabled = _env_bool(MACROS_ENABLED_ENV, True)

    def expand_or_original(self, user_text: str) -> str:
        if not self.enabled:
            return user_text
        t = (user_text or "").strip()
        if not t:
            return user_text

        low = t.lower().strip()
        # Never expand confirmations/cancel (safety) or training commands.
        if low.startswith("confirmar") or low.startswith("confirm ") or low in {"cancelar", "cancela"}:
            return user_text
        if low.startswith("cuando diga") or low.startswith("aprende comando") or low.startswith("aprende atajo"):
            return user_text

        trigger = _normalize_trigger(t)
        action = self.memory.get_macro(trigger=trigger)
        if not action:
            return user_text
        return str(action).strip() or user_text

    def maybe_handle_training(self, user_text: str) -> Optional[str]:
        if not self.enabled:
            return None
        t = (user_text or "").strip()
        if not t:
            return None
        low = t.lower().strip()

        if low in {"lista comandos", "mis comandos", "comandos", "atajos", "lista atajos"}:
            pairs = self.memory.list_macros(limit=30)
            if not pairs:
                return "No hay comandos aprendidos."
            names = [tr for tr, _ in pairs[:15]]
            return "Comandos: " + ", ".join(names)

        if low.startswith("borra comando ") or low.startswith("borra atajo "):
            trig = t.split(" ", 2)[-1].strip()
            trig_n = _normalize_trigger(trig)
            if not trig_n:
                return "Dime cuál: borra comando FRASE"
            self.memory.delete_macro(trigger=trig_n)
            return "Listo."

        # "cuando diga X, ejecuta Y"
        m = re.match(r"^cuando\s+diga\s+(.+?)(?:,)?\s+ejecuta\s+(.+)$", t, flags=re.IGNORECASE)
        if m:
            trigger_raw = m.group(1)
            action_raw = m.group(2)
            return self._save_macro(trigger_raw, action_raw)

        # "aprende comando X => Y"
        m2 = re.match(r"^aprende\s+(?:comando|atajo)\s+(.+?)\s*(?:=>|=|->)\s*(.+)$", t, flags=re.IGNORECASE)
        if m2:
            trigger_raw = m2.group(1)
            action_raw = m2.group(2)
            return self._save_macro(trigger_raw, action_raw)

        return None

    def _save_macro(self, trigger_raw: str, action_raw: str) -> str:
        trigger = _normalize_trigger(trigger_raw)
        action = (action_raw or "").strip()
        if not trigger or len(trigger) < 2:
            return "Dime una frase para activar el comando."
        if not action:
            return "Dime qué acción hago."

        action_low = action.lower().strip()
        if action_low.startswith("confirmar") or action_low.startswith("confirm ") or action_low in {"cancelar", "cancela"}:
            return "Por seguridad, no guardo comandos que confirmen o cancelen acciones."

        # Convenience: allow "tarea X" as shorthand.
        if action_low.startswith("tarea "):
            name = action[6:].strip().lower()
            if name:
                action = f"ejecuta tarea {name}"

        self.memory.upsert_macro(trigger=trigger, action=action)
        return "Listo. Aprendí ese comando."


class TeachingModePolicy:
    """
    Special teaching mode:
    - Lets YUI be more detailed in text output when enabled
    - Keeps voice concise by speaking a shortened version (enforced by StylePolicy)
    - Does NOT remove safety boundaries (confirmations, defensive cybersecurity-only guidance, etc.)
    """

    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.enabled = self._load_enabled()
        self.print_full = _env_bool("YUI_TEACHING_PRINT_FULL", True)

    def _load_enabled(self) -> bool:
        raw = self.memory.get_preference(TEACHING_MODE_KEY)
        if raw is None or not str(raw).strip():
            return _env_bool("YUI_TEACHING_MODE", False)
        return str(raw).strip() not in {"0", "false", "False", "no", "NO"}

    def _save(self) -> None:
        try:
            self.memory.set_preference(TEACHING_MODE_KEY, "1" if self.enabled else "0")
        except Exception:
            pass

    def maybe_handle(self, user_text: str) -> Optional[str]:
        t = (user_text or "").strip()
        if not t:
            return None
        low = t.lower().strip()

        if low in {"modo enseñanza", "modo ensenanza", "activa enseñanza", "activar enseñanza", "activa ensenanza", "activar ensenanza", "enseñanza on", "ensenanza on"}:
            self.enabled = True
            self._save()
            return "Listo. Modo enseñanza activado."

        if low in {"modo normal", "desactiva enseñanza", "desactivar enseñanza", "desactiva ensenanza", "desactivar ensenanza", "enseñanza off", "ensenanza off"}:
            self.enabled = False
            self._save()
            return "Listo. Modo enseñanza desactivado."

        if low in {"enseñanza", "ensenanza", "estado enseñanza", "estado ensenanza", "enseñanza estado", "ensenanza estado"}:
            return "Modo enseñanza: " + ("activo." if self.enabled else "apagado.")

        return None

    def system_prompt(self) -> str:
        if not self.enabled:
            return ""
        return (
            "Modo enseñanza especial:\n"
            "- Puedes explicar con más detalle, dar ejemplos y pasos claros.\n"
            "- Mantén el tono natural y directo.\n"
            "- No ignores confirmaciones ni barreras de seguridad del sistema.\n"
        )
