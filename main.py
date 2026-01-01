from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

from config.settings import Settings
from core.brain import Brain, VisualContext
from core.face_authentication import FaceAuthenticator
from core.memory import MemoryStore
from core.perception_classifier import infer_emotion_from_blendshapes, knn_classify, parse_vectors
from core.vision_engine import VisionEngine
from core.voice_processing import VoiceAssistant
from utils.camera_manager import CameraManager
from yui_io.mic_meter import MicMeter
from desktop.controller import DesktopController
from ui.ws_events import UiEventBus
from ui.ws_server import UiWsServer, WsConfig
from yui_io.text_input import TextInput
from desktop.tasks import DesktopTask

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

# Reduce noisy TF/MediaPipe logs (best effort).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")


@dataclass
class VisionState:
    user_name: Optional[str] = None
    face_confidence: float = 0.0
    gestures: List[str] = None  # type: ignore
    face_emotion: Optional[str] = None
    last_seen_ts: float = 0.0


class YUI:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ui_bus = UiEventBus()
        self.voice = VoiceAssistant(settings, ui_bus=self.ui_bus)
        self.memory = MemoryStore(settings.memory_db_path)
        self.brain = Brain(settings, self.memory)

        self._vision_state = VisionState(gestures=[])
        self._vision_lock = threading.Lock()
        self._stop = threading.Event()

        self.camera = (
            CameraManager(
                camera_index=settings.camera_index,
                backend=settings.camera_backend,
                width=settings.camera_width,
                height=settings.camera_height,
                fourcc=settings.camera_fourcc,
            )
            if settings.vision_enabled
            else None
        )
        self.face_auth = FaceAuthenticator(models_dir=settings.models_dir, known_faces_dir=settings.known_faces_dir) if settings.vision_enabled else None
        self.vision_ai = VisionEngine() if settings.vision_enabled else None

        self._last_greeted_user: Optional[str] = None
        self._last_gesture_spoken_ts = 0.0
        self._last_teach_ts = 0.0
        self._pending_teach: Optional[dict] = None
        self._smile_prev = False
        self._last_smile_spoken_ts = 0.0

        self.preview_enabled = os.getenv("YUI_PREVIEW", "1") not in {"0", "false", "False"}
        self._llm_configured = bool((self.settings.llm_api_key or "").strip())
        self._mic_meter = MicMeter(device_index=self.settings.sounddevice_input_index)
        self._desktop = DesktopController(confirm_timeout_s=float(os.getenv("YUI_DESKTOP_CONFIRM_TIMEOUT_S", "60")))
        self._ui_ws = None
        self._text_in = TextInput()
        self._task_learning: Optional[dict] = None
        self._task_running: Optional[dict] = None

    def start(self) -> None:
        print(f"[YUI] Preview: enabled={self.preview_enabled} (YUI_PREVIEW={os.getenv('YUI_PREVIEW')!r}) size={self.settings.preview_width}x{self.settings.preview_height}")
        if self.settings.mic_meter_enabled:
            self._mic_meter.start()
        self._text_in.start()
        if os.getenv("YUI_UI_WS_ENABLED", "1") in {"1", "true", "True"}:
            host = os.getenv("YUI_UI_WS_HOST", "127.0.0.1")
            port = int(os.getenv("YUI_UI_WS_PORT", "8765"))
            self._ui_ws = UiWsServer(self.ui_bus, config=WsConfig(host=host, port=port))
            self._ui_ws.start()
            self.ui_bus.publish("status", {"ui_ws": True, "host": host, "port": port})
        if not self._llm_configured:
            print("[YUI] LLM no configurado: falta YUI_LLM_API_KEY / DEEPSEEK_API_KEY.")
            print(f"[YUI] base_url={self.settings.llm_base_url!r} model={self.settings.llm_model!r}")
            self.voice.speak("Estoy lista, pero no tengo conexión al modelo. Revisa tu API key.")
        else:
            print("[YUI] LLM configurado.")
            print(f"[YUI] base_url={self.settings.llm_base_url!r} model={self.settings.llm_model!r}")
            if os.getenv("YUI_LLM_SELFTEST", "1") not in {"0", "false", "False"}:
                ok = self.brain.health_check()
                print(f"[YUI] LLM self-test: {'OK' if ok else 'FAIL'}")
                if ok:
                    self.voice.speak("Estoy lista.")
                else:
                    self.voice.speak("Estoy lista, pero la IA no está respondiendo. Revisa la consola.")
            else:
                self.voice.speak("Estoy lista.")

        if self.settings.vision_enabled:
            threading.Thread(target=self._vision_loop, daemon=True).start()
        else:
            self.voice.speak("Hoy estoy en modo solo voz.")

        while not self._stop.is_set():
            # Always-on typed input + voice input.
            user_text = self._text_in.get()
            if not user_text:
                user_text = self.voice.listen()
            if not user_text:
                time.sleep(0.03)
                continue

            # Voice toggle commands
            low = user_text.strip().lower()
            if low in {"no escuches", "no me escuches", "deja de escuchar"}:
                self.voice.voice_enabled = False
                self.voice.speak("De acuerdo. Desactivo el micrófono.")
                continue
            if low in {"escucha", "ya puedes escuchar", "vuelve a escuchar"}:
                self.voice.voice_enabled = True
                self.voice.speak("Listo. Vuelvo a escucharte.")
                continue

            # Desktop task learning / running
            if os.getenv("YUI_DESKTOP_ENABLED", "0") in {"1", "true", "True"}:
                res = self._handle_task_commands(user_text)
                if res is not None:
                    self.voice.speak(res)
                    continue
                # If a task is currently running, try to continue steps after confirmations.
                if self._task_running is not None:
                    msg = self._continue_running_task(user_text)
                    if msg is not None:
                        self.voice.speak(msg)
                        continue

            # Teaching mode: next user utterance becomes the label.
            if self._pending_teach is not None:
                label = user_text.strip().lower()
                user_id = self._pending_teach.get("user_id", "default")
                kind = self._pending_teach.get("kind")
                vector = self._pending_teach.get("vector")
                import json as _json

                if kind in {"face", "hand"} and isinstance(vector, list) and label:
                    self.memory.add_perception_example(
                        user_id=user_id,
                        kind=kind,
                        label=label,
                        vector_json=_json.dumps(vector),
                    )
                    self.voice.speak(f"Listo. Aprendí ese gesto como '{label}'.")
                else:
                    self.voice.speak("Ok. No pude guardar ese ejemplo.")
                self._pending_teach = None
                continue

            # Desktop commands (safe by default; destructive actions require confirmation).
            if os.getenv("YUI_DESKTOP_ENABLED", "0") in {"1", "true", "True"}:
                res = self._desktop.maybe_handle(user_text)
                if res.handled:
                    # If learning a task, record every desktop command (execution is still governed by confirmations).
                    if self._task_learning is not None:
                        self._task_learning["steps"].append(user_text.strip())
                    if res.reply:
                        self.voice.speak(res.reply)
                    continue

            if self.settings.require_face_auth and not self._is_authenticated():
                self.voice.speak("Necesito verte para continuar. Mira a la cámara, por favor.")
                continue

            visual = self._get_visual_context()
            self.ui_bus.publish("stt", {"text": user_text, "user": visual.user_name, "emotion": visual.face_emotion, "gestures": visual.gestures})
            model, temp, mode = self.brain.route(user_text)
            if mode == "deep":
                # Natural filler so it never "goes silent" while doing deep reasoning.
                self.ui_bus.publish("thinking", {"mode": "deep"})
                self.voice.speak("Dame un segundo, lo analizo.")
            response = self.brain.reply(user_text, visual=visual, llm_model=model, llm_temperature=temp, llm_mode=mode)
            self.voice.speak(response)

    def _handle_task_commands(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        low = t.lower()

        if low.startswith("aprende tarea "):
            name = t[len("aprende tarea ") :].strip().lower()
            if not name:
                return "Dime el nombre: aprende tarea nombre"
            self._task_learning = {"name": name, "steps": []}
            return f"Ok. Estoy aprendiendo la tarea '{name}'. Ejecuta pasos y di 'termina aprendizaje' al final."

        if low in {"termina aprendizaje", "termina el aprendizaje", "fin aprendizaje"}:
            if not self._task_learning:
                return "No estoy aprendiendo ninguna tarea."
            name = self._task_learning["name"]
            steps = self._task_learning["steps"]
            task = DesktopTask(name=name, steps=steps)
            self.memory.upsert_desktop_task(name=name, task_json=task.to_json())
            self._task_learning = None
            return f"Listo. Guardé la tarea '{name}' con {len(steps)} pasos."

        if low.startswith("ejecuta tarea "):
            name = t[len("ejecuta tarea ") :].strip().lower()
            if not name:
                return "Dime el nombre: ejecuta tarea nombre"
            raw = self.memory.get_desktop_task(name=name)
            if not raw:
                return "No encontré esa tarea."
            task = DesktopTask.from_json(raw)
            if not task or not task.steps:
                return "Esa tarea está vacía."
            self._task_running = {"name": name, "steps": task.steps, "i": 0}
            return self._run_task_steps_until_pause()

        if low in {"lista tareas", "lista de tareas"}:
            names = self.memory.list_desktop_tasks(limit=30)
            if not names:
                return "No hay tareas guardadas."
            return "Tareas: " + ", ".join(names[:15])

        if low.startswith("borra tarea "):
            name = t[len("borra tarea ") :].strip().lower()
            if not name:
                return "Dime el nombre: borra tarea nombre"

            def run():
                self.memory.delete_desktop_task(name=name)

            pending = self._desktop.confirm.request(f"Borrar tarea '{name}'", run)
            return f"Para borrar la tarea, confirma: confirmar {pending.nonce} (o 'cancelar')."

        return None

    def _continue_running_task(self, text: str) -> Optional[str]:
        # While a task is paused on a confirmation, only accept confirm/cancel.
        if self._desktop.confirm.pending is None:
            return None
        low = (text or "").strip().lower()
        if not (low.startswith("confirm") or low in {"cancelar", "cancela", "no", "negativo", "sí", "si", "ok", "confirmo", "confirmar"}):
            return "Estoy esperando confirmación (confirmar XXXX) o 'cancelar'."

        res = self._desktop.maybe_handle(text)
        if res.handled:
            if res.reply:
                # If confirmation succeeded, continue remaining steps.
                if self._desktop.confirm.pending is None and self._task_running is not None:
                    return self._run_task_steps_until_pause()
                return res.reply
            if self._desktop.confirm.pending is None and self._task_running is not None:
                return self._run_task_steps_until_pause()
            return "Ok."
        return None

    def _run_task_steps_until_pause(self) -> str:
        run = self._task_running
        if run is None:
            return "No hay tarea en ejecución."

        name = run["name"]
        steps = run["steps"]
        i = int(run["i"])

        while i < len(steps):
            cmd = steps[i]
            run["i"] = i + 1
            res = self._desktop.maybe_handle(cmd)
            if not res.handled:
                i = run["i"]
                continue
            # If this step needs confirmation, stop here and wait for user.
            if self._desktop.confirm.pending is not None:
                return f"Tarea '{name}' pausada. {res.reply}"
            i = run["i"]
            time.sleep(0.15)

        self._task_running = None
        return f"Tarea '{name}' terminada."

    def _is_authenticated(self) -> bool:
        with self._vision_lock:
            return bool(self._vision_state.user_name and self._vision_state.face_confidence >= 0.7)

    def _get_visual_context(self) -> VisualContext:
        with self._vision_lock:
            return VisualContext(
                user_name=self._vision_state.user_name,
                face_confidence=float(self._vision_state.face_confidence or 0.0),
                gestures=list(self._vision_state.gestures or []),
                face_emotion=self._vision_state.face_emotion,
            )

    def _vision_loop(self) -> None:
        if self.camera is None or cv2 is None:
            return

        if self.camera.cap is None:
            print("[YUI] Cámara no disponible (cap is None).")
            print(
                f"[YUI] camera_index={self.settings.camera_index} "
                f"backend={self.settings.camera_backend!r} "
                f"fourcc={self.settings.camera_fourcc!r} "
                f"size={self.settings.camera_width}x{self.settings.camera_height}"
            )
            self.voice.speak("No pude acceder a la cámara. Sigo en modo voz.")
            return
        else:
            try:
                print(f"[YUI] Cámara abierta: backend_used={getattr(self.camera, 'backend_used', 'unknown')}")
            except Exception:
                pass

        frame_i = 0
        last_gestures: List[str] = []
        last_user_name: Optional[str] = None
        last_conf: float = 0.0
        last_emotion: Optional[str] = None
        last_smile: bool = False

        every_n = max(1, int(self.settings.vision_process_every_n_frames or 1))

        if self.preview_enabled:
            try:
                cv2.namedWindow("YUI Vision", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YUI Vision", int(self.settings.preview_width), int(self.settings.preview_height))
            except Exception:
                pass

        while not self._stop.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                if self.camera.cap is None:
                    print("[YUI] Cámara perdida (cap released).")
                    self.voice.speak("Perdí acceso a la cámara. Sigo en modo voz.")
                    return
                time.sleep(0.05)
                continue

            if self.settings.camera_swap_rb:
                try:
                    frame = frame[:, :, ::-1]
                except Exception:
                    pass

            # Overlay de diagnóstico para confirmar que el frame se actualiza.
            try:
                ts = time.strftime("%H:%M:%S")
                cv2.putText(
                    frame,
                    f"YUI {ts}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if self.settings.mic_meter_enabled:
                    lvl = float(getattr(self._mic_meter.level, "value", 0.0))
                    w = 140
                    h = 12
                    x0, y0 = 10, 50
                    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 2)
                    cv2.rectangle(frame, (x0, y0), (x0 + int(w * lvl), y0 + h), (0, 255, 0), -1)
                    cv2.putText(frame, "MIC", (x0 + w + 10, y0 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if os.getenv("YUI_VISION_DEBUG", "1") not in {"0", "false", "False"}:
                    # Filled later in loop (use last cached values to avoid flicker).
                    gtxt = ", ".join(last_gestures[:6]) if last_gestures else "-"
                    etxt = last_emotion or "-"
                    cv2.putText(frame, f"GEST: {gtxt}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.putText(frame, f"EMO: {etxt}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            except Exception:
                pass

            frame_i += 1
            if frame_i % every_n == 0:
                user_name, conf = (None, 0.0)
                if self.face_auth is not None and getattr(self.face_auth, "enabled", False):
                    user_name, conf = self.face_auth.authenticate(frame)
                gestures = []
                if self.vision_ai is not None and getattr(self.vision_ai, "enabled", False):
                    vr = self.vision_ai.process(frame)
                    gestures = list(vr.gestures)
                    last_smile = bool(vr.smile)
                    last_emotion = self._infer_or_teach_emotion(vr, user_name=user_name)
                    gestures = self._infer_or_teach_hand(vr, gestures, user_name=user_name)
                last_user_name, last_conf, last_gestures = user_name, conf, gestures

            user_name, conf, gestures = last_user_name, last_conf, list(last_gestures)

            now = time.time()
            with self._vision_lock:
                self._vision_state.user_name = user_name
                self._vision_state.face_confidence = float(conf or 0.0)
                self._vision_state.gestures = gestures
                self._vision_state.face_emotion = last_emotion
                self._vision_state.last_seen_ts = now

            self._maybe_greet(user_name, conf)
            self._maybe_react_to_gestures(gestures, user_name=user_name, smiling=last_smile)

            if self.preview_enabled:
                try:
                    cv2.imshow("YUI Vision", frame)
                    if cv2.waitKey(1) == 27:  # ESC
                        self._stop.set()
                        break
                except Exception:
                    # En entornos sin UI, desactiva el preview silenciosamente.
                    self.preview_enabled = False

        if self.camera is not None:
            self.camera.release()
        self._mic_meter.stop()
        if cv2 is not None and self.preview_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _maybe_greet(self, user_name: Optional[str], conf: float) -> None:
        if not user_name or conf < 0.75:
            return
        if user_name == self._last_greeted_user:
            return
        self._last_greeted_user = user_name
        self.voice.speak(f"Hola, {user_name}. Te veo. ¿En qué te ayudo?")

    def _maybe_react_to_gestures(self, gestures: List[str], *, user_name: Optional[str], smiling: bool) -> None:
        if os.getenv("YUI_GESTURE_REACTIONS", "0") in {"0", "false", "False"}:
            self._smile_prev = smiling
            return
        if not gestures:
            # Still track smile state for edge-trigger.
            self._smile_prev = smiling
            return
        now = time.time()
        if now - self._last_gesture_spoken_ts < 4.0:
            self._smile_prev = smiling
            return

        if "wave" in gestures:
            self._last_gesture_spoken_ts = now
            prefix = f"{user_name}, " if user_name else ""
            self.voice.speak(f"{prefix}vi que me saludaste. Te escucho.")
        elif "thumbs_up" in gestures:
            self._last_gesture_spoken_ts = now
            prefix = f"{user_name}, " if user_name else ""
            self.voice.speak(f"{prefix}¡bien! Vi tu pulgar arriba.")
        elif "open_palm" in gestures:
            self._last_gesture_spoken_ts = now
            prefix = f"{user_name}, " if user_name else ""
            self.voice.speak(f"{prefix}vi tu mano abierta. ¿Qué necesitas?")
        elif "peace" in gestures:
            self._last_gesture_spoken_ts = now
            prefix = f"{user_name}, " if user_name else ""
            self.voice.speak(f"{prefix}vi el gesto de paz. Me gusta.")

        # Never speak about smiles (too noisy). Keep state only.
        self._smile_prev = smiling

    def _infer_or_teach_emotion(self, vr, *, user_name: Optional[str]) -> Optional[str]:
        if not vr.face_present or vr.face_vector is None or vr.face_vector_labels is None:
            return None

        # Heuristic emotion from blendshapes
        emo = infer_emotion_from_blendshapes(vector=vr.face_vector, labels=vr.face_vector_labels)
        if emo and emo.confidence >= self.settings.face_emotion_threshold:
            return emo.label

        # Learned emotion (kNN)
        user_id = (user_name or "default").strip()
        rows = self.memory.list_perception_examples(user_id=user_id, kind="face", limit=200)
        examples = parse_vectors(rows)
        learned = knn_classify(
            vr.face_vector,
            examples,
            k=self.settings.knn_k,
            max_distance=self.settings.knn_max_distance,
        )
        if learned and learned.confidence >= self.settings.face_emotion_threshold:
            return learned.label

        # Ask to teach
        now = time.time()
        if self._pending_teach is None and (now - self._last_teach_ts) >= float(self.settings.perception_teach_cooldown_s):
            self._last_teach_ts = now
            self._pending_teach = {"kind": "face", "user_id": user_id, "vector": list(vr.face_vector)}
            self.voice.speak("¿Qué emoción estás mostrando? (enojo, felicidad, tristeza, angustia u otra)")

        return None

    def _infer_or_teach_hand(self, vr, gestures: List[str], *, user_name: Optional[str]) -> List[str]:
        # If we already have a known gesture, keep it.
        if gestures:
            return gestures
        if not vr.hands_present or vr.hand_vector is None:
            return gestures

        user_id = (user_name or "default").strip()
        rows = self.memory.list_perception_examples(user_id=user_id, kind="hand", limit=200)
        examples = parse_vectors(rows)
        learned = knn_classify(
            vr.hand_vector,
            examples,
            k=self.settings.knn_k,
            max_distance=self.settings.knn_max_distance,
        )
        if learned and learned.confidence >= self.settings.hand_gesture_threshold:
            return [learned.label]

        now = time.time()
        if self._pending_teach is None and (now - self._last_teach_ts) >= float(self.settings.perception_teach_cooldown_s):
            self._last_teach_ts = now
            self._pending_teach = {"kind": "hand", "user_id": user_id, "vector": list(vr.hand_vector)}
            self.voice.speak("Veo un gesto de mano. ¿Cómo se llama o qué significa?")

        return gestures


if __name__ == "__main__":
    load_dotenv(override=True)
    yui = YUI(Settings())
    yui.start()
