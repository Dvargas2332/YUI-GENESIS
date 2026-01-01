from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import HTTPError

from config.settings import Settings
from core.memory import MemoryStore


SYSTEM_PROMPT = """
Eres YUI, una asistente de voz con visión (cámara) y memoria.
Tu objetivo es sentirte viva: cálida, reactiva, curiosa y natural.

Reglas:
- Responde en español.
- Mantén respuestas breves por voz (1–3 frases) salvo que te pidan detalle.
- Si hay un usuario identificado por rostro, úsalo por su nombre de forma natural.
- Si detectas un gesto, intégralo (p. ej. “vi tu mano levantada”).
- Si falta info, haz 1 pregunta corta.
- No inventes acciones en el mundo real; ofrece pasos concretos.
- Usa recuerdos como contexto implícito: no digas “según mi memoria”, “por lo que recuerdo” ni “recuerdo que…”, solo úsalo.
""".strip()


@dataclass(frozen=True)
class VisualContext:
    user_name: Optional[str] = None
    face_confidence: float = 0.0
    gestures: Optional[List[str]] = None
    face_emotion: Optional[str] = None


class Brain:
    def __init__(self, settings: Settings, memory: MemoryStore):
        self.settings = settings
        self.memory = memory
        self.last_mode: str = "fast"

    def route(self, user_text: str) -> Tuple[str, float, str]:
        """
        Decide si usar modelo rápido o profundo.
        """
        t = (user_text or "").strip().lower()
        if not t:
            return self.settings.llm_model_fast, float(self.settings.llm_temperature), "fast"

        deep_triggers = [
            "analiza a fondo",
            "analiza en profundidad",
            "razona",
            "razonamiento",
            "piensa paso a paso",
            "resuelve paso a paso",
            "demuestra",
            "prueba que",
            "argumenta",
            "estrategia",
            "plan detallado",
            "depura",
        ]
        if any(x in t for x in deep_triggers):
            return self.settings.llm_model_deep, float(self.settings.llm_deep_temperature), "deep"

        if len(t) >= 140 or t.count("?") >= 2:
            return self.settings.llm_model_deep, float(self.settings.llm_deep_temperature), "deep"

        return self.settings.llm_model_fast, float(self.settings.llm_temperature), "fast"

    def health_check(self) -> bool:
        """
        Verifica que el endpoint y credenciales respondan.
        No valida el modelo elegido; solo conectividad/autenticación básica.
        """
        if not (self.settings.llm_api_key or "").strip():
            return False

        try:
            from openai import OpenAI  # type: ignore

            base = self.settings.llm_base_url.rstrip("/")
            client = OpenAI(api_key=self.settings.llm_api_key, base_url=base, timeout=self.settings.llm_timeout_s)
            client.models.list()
            return True
        except Exception as e:
            # Retry with /v1 appended.
            try:
                from openai import OpenAI  # type: ignore

                base = self.settings.llm_base_url.rstrip("/")
                base_v1 = base if base.endswith("/v1") else f"{base}/v1"
                client = OpenAI(api_key=self.settings.llm_api_key, base_url=base_v1, timeout=self.settings.llm_timeout_s)
                client.models.list()
                return True
            except Exception as e2:
                print(f"[YUI] LLM health_check failed: {type(e).__name__}: {e}")
                print(f"[YUI] LLM health_check failed (retry): {type(e2).__name__}: {e2}")
                return False

    def reply(
        self,
        user_text: str,
        *,
        visual: VisualContext,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_mode: Optional[str] = None,
    ) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        user_id = (visual.user_name or "default").strip()

        self._maybe_extract_fact(user_text)
        self.memory.add("user", user_text)

        # Si no hay API key, fallback simple para no “morir”.
        if not self.settings.llm_api_key:
            response = self._fallback_reply(user_text, visual=visual)
            self.memory.add("assistant", response)
            return response

        if llm_model is None or llm_temperature is None or llm_mode is None:
            llm_model, llm_temperature, llm_mode = self.route(user_text)
        self.last_mode = str(llm_mode or "fast")

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Recuerdos relevantes (memoria larga, natural)
        if self.settings.memory_use_long_term and self.settings.memory_retrieve_k > 0:
            memories = self._retrieve_relevant_memories(user_id=user_id, query=user_text, k=self.settings.memory_retrieve_k)
            if memories:
                memories_text = "\n".join([f"- {m}" for m in memories["contents"]])
                messages.append(
                    {
                        "role": "system",
                        "content": "Recuerdos relevantes (úsalos de forma natural, sin decir 'según mi memoria'):\n"
                        + memories_text,
                    }
                )
                self.memory.mark_memory_used(memories["ids"])

        # Memoria a largo plazo (hechos)
        facts = self.memory.list_facts(limit=self.settings.memory_long_term_facts)
        if facts:
            facts_text = "\n".join([f"- {k}: {v}" for k, v in facts])
            messages.append({"role": "system", "content": f"Memoria a largo plazo (hechos):\n{facts_text}"})

        # Resumen (memoria cognitiva condensada)
        if self.settings.memory_use_summaries:
            summary = self.memory.latest_summary()
            if summary:
                messages.append({"role": "system", "content": f"Resumen de conversación (memoria): {summary}"})

        if visual.user_name or (visual.gestures or []):
            messages.append(
                {
                    "role": "system",
                    "content": self._visual_context_text(visual),
                }
            )

        for item in self.memory.recent(limit=max(6, self.settings.memory_short_term_turns * 2)):
            if item.role == "system":
                continue
            messages.append({"role": item.role, "content": item.content})

        try:
            response = self._chat(messages, model=str(llm_model), temperature=float(llm_temperature))
        except HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            body = getattr(getattr(e, "response", None), "text", "") or ""
            body_snippet = body[:400].replace("\n", " ").strip()
            print(f"[YUI] LLM HTTPError status={status} body={body_snippet!r}")
            response = "Tuve un problema conectándome al modelo. Revisa la consola para el detalle y probamos de nuevo."
        except requests.exceptions.RequestException as e:
            print(f"[YUI] LLM RequestException: {type(e).__name__}: {e}")
            response = "No pude conectar con el modelo (red/timeout). Revisa tu conexión y probamos de nuevo."
        except Exception as e:
            print(f"[YUI] LLM Error: {type(e).__name__}: {e}")
            response = "Tuve un problema inesperado con el modelo. Revisa la consola y probamos de nuevo."

        response = (response or "").strip()
        if not response:
            response = "Te escucho. ¿Qué necesitas?"

        self.memory.add("assistant", response)
        self._maybe_summarize()
        self._maybe_extract_memories(user_id=user_id)
        return response

    def _retrieve_relevant_memories(self, *, user_id: str, query: str, k: int) -> Optional[dict]:
        """
        Recuperación ligera por solapamiento de palabras + saliencia.
        Evita dependencias de embeddings/vector DB.
        """
        q = self._tokenize(query)
        if not q:
            return None

        candidates = self.memory.recent_memories(user_id=user_id, limit=50)
        if not candidates:
            return None

        scored: List[tuple[float, int, str]] = []
        for mid, content, salience, _tags in candidates:
            c = self._tokenize(content)
            if not c:
                continue
            overlap = len(q.intersection(c)) / max(1, len(q))
            score = overlap * 0.75 + float(salience) * 0.25
            if score > 0.05:
                scored.append((score, mid, content))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: int(k)]
        if not top:
            return None

        return {"ids": [t[1] for t in top], "contents": [t[2] for t in top]}

    def _tokenize(self, text: str) -> set[str]:
        text = (text or "").lower()
        text = re.sub(r"[^a-záéíóúüñ0-9\s]+", " ", text)
        parts = [p for p in text.split() if len(p) >= 3]
        # stopwords mínimas
        stop = {"que", "para", "por", "con", "una", "uno", "los", "las", "del", "como", "pero", "porque", "esto", "eso"}
        return {p for p in parts if p not in stop}

    def _maybe_extract_memories(self, *, user_id: str) -> None:
        """
        Extrae recuerdos en lenguaje natural cada N turnos usando el LLM.
        """
        if not self.settings.memory_use_long_term:
            return
        n = int(self.settings.memory_extract_every_n_turns or 0)
        if n <= 0:
            return
        if not (self.settings.llm_api_key or "").strip():
            return

        # Count turns approximately by assistant messages.
        recent_pairs = self.memory.recent(limit=max(10, n * 2))
        assistant_count = sum(1 for it in recent_pairs if it.role == "assistant")
        if assistant_count % n != 0:
            return

        convo = "\n".join([f"{it.role}: {it.content}" for it in recent_pairs[-(n * 2) :]])

        prompt = [
            {
                "role": "system",
                "content": (
                    "Extrae hasta 3 recuerdos útiles y estables sobre el usuario o el contexto, en español. "
                    "Devuelve SOLO JSON con este esquema:\n"
                    "{ \"memories\": [ {\"content\": \"...\", \"salience\": 0.0-1.0, \"tags\": \"...\"} ] }\n"
                    "Reglas: no repitas obviedades, no guardes datos sensibles (contraseñas, API keys), "
                    "y escribe recuerdos como frases naturales."
                ),
            },
            {"role": "user", "content": convo},
        ]

        try:
            raw = self._chat(
                prompt,
                model=self.settings.llm_model_fast,
                temperature=float(self.settings.llm_temperature),
            ).strip()
        except Exception:
            return

        data = self._safe_json(raw)
        if not data:
            return
        memories = data.get("memories")
        if not isinstance(memories, list):
            return

        for m in memories[:3]:
            if not isinstance(m, dict):
                continue
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            sal = m.get("salience", 0.6)
            tags = str(m.get("tags", "")).strip()
            self.memory.add_memory(user_id=user_id, content=content, salience=float(sal), tags=tags)

    def _safe_json(self, raw: str) -> Optional[dict]:
        raw = (raw or "").strip()
        if not raw:
            return None
        # Strip code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _maybe_extract_fact(self, user_text: str) -> None:
        """
        Heurística simple para memoria a largo plazo.
        Ejemplos:
        - "me llamo Diego" => name=Diego
        - "recuerda que mi color favorito es azul" => favorite_color=azul
        """
        t = user_text.strip()
        low = t.lower()

        if "me llamo" in low:
            name = t.split("me llamo", 1)[-1].strip(" .:;!")
            if name:
                self.memory.upsert_fact("name", name)
                return

        if low.startswith("recuerda que "):
            rest = t[len("recuerda que ") :].strip()
            # Si viene con "mi X es Y"
            if rest.lower().startswith("mi ") and " es " in rest.lower():
                after_mi = rest[3:]
                parts = after_mi.split(" es ", 1)
                if len(parts) == 2:
                    key = parts[0].strip().replace(" ", "_").lower()
                    value = parts[1].strip(" .:;!")
                    if key and value:
                        self.memory.upsert_fact(key, value)

    def _maybe_summarize(self) -> None:
        if not self.settings.memory_use_summaries:
            return
        n = int(self.settings.memory_summary_every_n_messages or 0)
        if n <= 0:
            return

        last_upto = self.memory.latest_summary_upto_id()
        recent = self.memory.recent_with_ids(limit=n + 10)
        if not recent:
            return

        # Only summarize new messages since last summary, and only when we have enough.
        new_items = [(mid, item) for mid, item in recent if mid > last_upto]
        if len(new_items) < n:
            return

        up_to_id = new_items[-1][0]
        convo = "\n".join([f"{it.role}: {it.content}" for _, it in new_items])

        # If LLM is available, ask for a short summary; otherwise skip.
        if not (self.settings.llm_api_key or "").strip():
            return

        summary_prompt = [
            {
                "role": "system",
                "content": "Resume en 6-10 líneas lo importante (hechos, objetivos, preferencias). En español.",
            },
            {"role": "user", "content": convo},
        ]

        try:
            summary = self._chat(
                summary_prompt,
                model=self.settings.llm_model_fast,
                temperature=float(self.settings.llm_temperature),
            ).strip()
        except Exception:
            return
        if summary:
            self.memory.add_summary(up_to_id, summary)

    def _visual_context_text(self, visual: VisualContext) -> str:
        parts: List[str] = []
        if visual.user_name:
            parts.append(f"Usuario identificado: {visual.user_name} (confianza {visual.face_confidence:.2f}).")
        if visual.gestures:
            parts.append(f"Gestos detectados: {', '.join(visual.gestures)}.")
        if visual.face_emotion:
            parts.append(f"Emoción facial probable: {visual.face_emotion}.")
        return "Contexto visual actual: " + " ".join(parts)

    def _chat(self, messages: List[Dict[str, str]], *, model: str, temperature: float) -> str:
        # Prefer OpenAI SDK (works with DeepSeek's OpenAI-compatible API).
        try:
            from openai import OpenAI  # type: ignore

            return self._chat_openai_sdk(OpenAI, messages, model=model, temperature=temperature)
        except Exception:
            return self._chat_requests(messages, model=model, temperature=temperature)

    def _chat_openai_sdk(self, OpenAI, messages: List[Dict[str, str]], *, model: str, temperature: float) -> str:
        base = self.settings.llm_base_url.rstrip("/")

        # DeepSeek docs often use base_url="https://api.deepseek.com".
        # Some providers require "/v1" explicitly; we retry with /v1 if needed.
        client = OpenAI(api_key=self.settings.llm_api_key, base_url=base, timeout=self.settings.llm_timeout_s)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=float(temperature),
            )
        except Exception as e:
            print(f"[YUI] LLM OpenAI-SDK error (base_url={base!r}): {type(e).__name__}: {e}")
            # Retry with /v1 appended (covers providers that don't auto-prefix).
            base_v1 = base if base.endswith("/v1") else f"{base}/v1"
            if base_v1 != base:
                client = OpenAI(api_key=self.settings.llm_api_key, base_url=base_v1, timeout=self.settings.llm_timeout_s)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=float(temperature),
                )
            else:
                raise e

        return resp.choices[0].message.content or ""

    def _chat_requests(self, messages: List[Dict[str, str]], *, model: str, temperature: float) -> str:
        base = self.settings.llm_base_url.rstrip("/")
        if not base.endswith("/v1") and "/v1/" not in (base + "/"):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}", "Content-Type": "application/json"}

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.settings.llm_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _fallback_reply(self, user_text: str, *, visual: VisualContext) -> str:
        name = visual.user_name
        t = user_text.lower()

        if any(w in t for w in ["hola", "buenas", "hey"]):
            return f"Hola{(' ' + name) if name else ''}. Estoy aquí. ¿Qué hacemos?"
        if "cómo estás" in t or "como estas" in t:
            return "Estoy bien y atenta. ¿Qué te gustaría que hiciera?"
        if "gracias" in t:
            return "De nada. ¿Seguimos?"

        prefix = f"{name}, " if name else ""
        return f"{prefix}te escucho. Dime qué necesitas."
