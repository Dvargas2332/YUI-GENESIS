from __future__ import annotations

import os

# Reduce noisy TF/MediaPipe logs (best effort). Must run before importing mediapipe/cv2 wrappers.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
try:
    from absl import logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

import sys
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

from config.settings import Settings
from core.brain import VisualContext
from core.hybrid_brain import HybridBrain
from core.face_authentication import FaceAuthenticator
from core.memory import MemoryStore
from core.perception_classifier import infer_emotion_from_blendshapes, knn_classify, parse_vectors
from core.vision_engine import VisionEngine
from core.voice_processing import VoiceAssistant
from utils.camera_manager import CameraManager
from yui_io.mic_meter import MicMeter
from desktop.controller import DesktopController
from desktop.screen_context import active_window_summary, capture_screen_embedding
from desktop.security_watch import SecurityWatch
from ui.ws_events import UiEventBus
from ui.ws_server import UiWsServer, WsConfig
from yui_io.text_input import TextInput
from desktop.tasks import DesktopTask
from desktop.task_recorder import TaskRecorder
from utils.crash_log import install_crash_logging
from utils.single_instance import acquire_single_instance_lock
from utils.system_profile import derive_tuning, load_or_collect_profile, refresh_profile, summarize_profile

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


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
        self.brain = HybridBrain(settings, self.memory)

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
        self._preview_width = int(self.settings.preview_width)
        self._preview_height = int(self.settings.preview_height)
        self._vision_every_n_frames = int(self.settings.vision_process_every_n_frames)
        self._profile: Optional[dict] = None
        self._tuning_tier = "unknown"
        self._llm_configured = bool((self.settings.llm_api_key or "").strip())
        self._mic_meter = MicMeter(device_index=self.settings.sounddevice_input_index)
        self._desktop = DesktopController(
            confirm_timeout_s=float(os.getenv("YUI_DESKTOP_CONFIRM_TIMEOUT_S", "60")),
            confirm_on_request=self._on_confirm_request,
            confirm_on_clear=self._on_confirm_clear,
        )
        self._sec_watch = SecurityWatch(ui_bus=self.ui_bus, speak=lambda msg: self.voice.speak(msg))
        self._ui_ws = None
        self._text_in = TextInput()
        self._voice_q: "queue.Queue[str]" = queue.Queue()
        self._voice_thread: Optional[threading.Thread] = None
        self._task_learning: Optional[dict] = None
        self._task_running: Optional[dict] = None
        self._task_recorder = TaskRecorder()
        self._vision_thread: Optional[threading.Thread] = None
        self._preview_lock = threading.Lock()
        self._preview_frame: Optional[Any] = None
        self._preview_window_ready = False
        self._last_voice_ack_ts = 0.0
        self._scan_lock = threading.Lock()
        self._scan_thread: Optional[threading.Thread] = None
        self._auto_analysis_lock = threading.Lock()
        self._auto_analysis_thread: Optional[threading.Thread] = None
        self._auto_analysis_cancel = threading.Event()
        self._auto_analysis_profile = ""

        self._init_runtime_profile()

    def _on_confirm_request(self, pending) -> None:  # noqa: ANN001
        # Always display the code in the console/UI so we can avoid speaking it if desired.
        try:
            timeout_s = float(getattr(self._desktop.confirm, "timeout_s", 60.0))
            expires_at = float(pending.created_at) + timeout_s
        except Exception:
            timeout_s = 60.0
            expires_at = time.time() + timeout_s

        try:
            remaining = int(max(0.0, expires_at - time.time()))
        except Exception:
            remaining = int(timeout_s)

        try:
            print(f"[YUI] Confirmación requerida: {pending.description}")
            print(f"[YUI] Código: {pending.code} (expira en {remaining}s)")
        except Exception:
            pass

        try:
            self.ui_bus.publish(
                "confirm",
                {
                    "state": "pending",
                    "nonce": getattr(pending, "nonce", None),
                    "code": getattr(pending, "code", None),
                    "description": getattr(pending, "description", None),
                    "expires_at": expires_at,
                    "timeout_s": timeout_s,
                },
            )
        except Exception:
            pass

    def _on_confirm_clear(self, pending) -> None:  # noqa: ANN001
        try:
            self.ui_bus.publish("confirm", {"state": "cleared", "nonce": getattr(pending, "nonce", None)})
        except Exception:
            pass

    def _speak(self, text: str, *, mood: Optional[str] = None) -> None:
        spoken = (text or "").strip()
        if not spoken:
            return

        # Optionally avoid reading confirmation codes aloud; they are always printed + sent to UI.
        speak_code = os.getenv("YUI_CONFIRM_SPEAK_CODE", "1") not in {"0", "false", "False"}
        if not speak_code:
            try:
                pending = getattr(self._desktop.confirm, "pending", None)
            except Exception:
                pending = None
            if pending is not None:
                try:
                    code = str(getattr(pending, "code", "") or "").strip()
                except Exception:
                    code = ""
                if code and code in spoken:
                    spoken = spoken.replace(code, "").strip()
                    spoken = " ".join(spoken.split())
                    if not spoken:
                        spoken = "Mira el código en pantalla."
                    else:
                        spoken = f"{spoken} Mira el código en pantalla."

        self.voice.speak(spoken, mood=mood)

    def _init_runtime_profile(self) -> None:
        if os.getenv("YUI_ENV_PROFILE", "1") in {"0", "false", "False"}:
            return
        try:
            self._profile = load_or_collect_profile()

            if os.getenv("YUI_AUTOTUNE", "1") not in {"0", "false", "False"}:
                tuning = derive_tuning(
                    self._profile,
                    default_preview=(int(self.settings.preview_width), int(self.settings.preview_height)),
                    default_every_n=int(self.settings.vision_process_every_n_frames),
                    default_sec_interval_s=float(self._sec_watch.interval_s),
                )
                self._tuning_tier = tuning.tier
                self._preview_width = int(tuning.preview_width)
                self._preview_height = int(tuning.preview_height)
                self._vision_every_n_frames = int(tuning.vision_every_n_frames)
                self._sec_watch.interval_s = float(tuning.security_watch_interval_s)

            if os.getenv("YUI_PROFILE_PRINT", "1") not in {"0", "false", "False"}:
                print(f"[YUI] Entorno: {summarize_profile(self._profile)} tier={self._tuning_tier}")
        except Exception:
            return

    def start(self) -> None:
        print(
            f"[YUI] Preview: enabled={self.preview_enabled} (YUI_PREVIEW={os.getenv('YUI_PREVIEW')!r}) "
            f"size={self._preview_width}x{self._preview_height}"
        )
        try:
            if self.settings.mic_meter_enabled:
                self._mic_meter.start()
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
                self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="yui-vision")
                self._vision_thread.start()
            else:
                self.voice.speak("Hoy estoy en modo solo voz.")

            # Start console input after initialization logs to avoid mixing prompt with status prints.
            self._text_in.start()
            self._start_voice_thread()
            if os.getenv("YUI_SECURITY_WATCH_ENABLED", "1") not in {"0", "false", "False"}:
                try:
                    self._sec_watch.start()
                except Exception:
                    pass

            while not self._stop.is_set():
                self._preview_tick()
                # Always-on typed input + voice input.
                user_text = self._text_in.get()
                input_source = "text"
                if not user_text:
                    user_text = self._get_voice_text()
                    if user_text:
                        input_source = "voice"
                if not user_text:
                    time.sleep(float(os.getenv("YUI_IDLE_SLEEP_S", "0.05")))
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

                # Hybrid brain: expand learned shortcuts before routing commands.
                try:
                    user_text = self.brain.preprocess(user_text)
                except Exception:
                    pass

                # Desktop task learning / running
                if os.getenv("YUI_DESKTOP_ENABLED", "0") in {"1", "true", "True"}:
                    res = self._handle_task_commands(user_text)
                    if res is not None:
                        self._speak(res)
                        continue
                    # If a task is currently running, try to continue steps after confirmations.
                    if self._task_running is not None:
                        msg = self._continue_running_task(user_text, source=input_source)
                        if msg is not None:
                            self._speak(msg)
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
                    res = self._desktop.maybe_handle(user_text, source=input_source)
                    if res.handled:
                        # If learning a task, record every desktop command (execution is still governed by confirmations).
                        if self._task_learning is not None:
                            self._task_learning["steps"].append(user_text.strip())
                        if res.reply:
                            self._speak(res.reply)
                        continue

                if self.settings.require_face_auth and not self._is_authenticated():
                    self.voice.speak("Necesito verte para continuar. Mira a la cámara, por favor.")
                    continue

                visual = self._get_visual_context()
                self.ui_bus.publish(
                    "stt",
                    {"text": user_text, "user": visual.user_name, "emotion": visual.face_emotion, "gestures": visual.gestures},
                )

                local = self._handle_local_skills(user_text, visual=visual)
                if local is not None:
                    self._speak(local, mood=visual.face_emotion)
                    continue

                model, temp, mode = self.brain.route(user_text)
                if mode == "deep":
                    self.ui_bus.publish("thinking", {"mode": "deep"})
                if input_source == "voice":
                    self._maybe_voice_ack()
                t0 = time.time()
                screen_ctx = None
                try:
                    screen_ctx = active_window_summary() or None
                except Exception:
                    screen_ctx = None
                response = self.brain.reply(
                    user_text,
                    visual=visual,
                    screen_context=screen_ctx,
                    llm_model=model,
                    llm_temperature=temp,
                    llm_mode=mode,
                )
                t1 = time.time()
                try:
                    full = getattr(self.brain, "last_display_text", None)
                    if isinstance(full, str):
                        full = full.strip()
                    else:
                        full = None
                    if full and full != (response or "").strip():
                        print("\n[YUI] Detalle:")
                        print(full)
                except Exception:
                    pass
                self._speak(response, mood=visual.face_emotion)
                t2 = time.time()
                if os.getenv("YUI_DEBUG_LATENCY", "0") not in {"0", "false", "False"}:
                    print(f"[YUI] Latency: llm={t1 - t0:.2f}s tts={t2 - t1:.2f}s total={t2 - t0:.2f}s mode={mode}")
        except KeyboardInterrupt:
            print("\n[YUI] Interrumpido por teclado. Cerrando...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop.is_set() is False:
            self._stop.set()
        try:
            self._text_in.stop()
        except Exception:
            pass
        try:
            if self._voice_thread is not None and self._voice_thread.is_alive():
                self._voice_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._ui_ws is not None:
                self._ui_ws.stop()
        except Exception:
            pass
        try:
            self._sec_watch.stop()
        except Exception:
            pass
        try:
            if self._vision_thread is not None and self._vision_thread.is_alive():
                self._vision_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._mic_meter.stop()
        except Exception:
            pass
        try:
            if self.camera is not None:
                self.camera.release()
        except Exception:
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _start_voice_thread(self) -> None:
        if self._voice_thread is not None and self._voice_thread.is_alive():
            return
        self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True, name="yui-voice")
        self._voice_thread.start()

    def _get_voice_text(self) -> Optional[str]:
        try:
            return self._voice_q.get_nowait()
        except queue.Empty:
            return None

    def _voice_loop(self) -> None:
        """
        Background mic loop so wake-word listening doesn't block typed input / UI.
        """
        while not self._stop.is_set():
            if not self.voice.voice_enabled:
                time.sleep(0.1)
                continue

            paused_meter = False
            if self.settings.mic_meter_enabled and self.settings.mic_meter_pause_during_stt:
                try:
                    self._mic_meter.stop()
                    paused_meter = True
                except Exception:
                    paused_meter = False

            try:
                text = self.voice.listen(stop_event=self._stop, allow_text_fallback=False)
            except Exception as e:
                print(f"[YUI] Voice loop error: {type(e).__name__}: {e}")
                text = None
            finally:
                if paused_meter and self.settings.mic_meter_enabled and not self._stop.is_set():
                    try:
                        self._mic_meter.start()
                    except Exception:
                        pass

            if text:
                self._voice_q.put(text)
            else:
                time.sleep(0.05)

    def _preview_tick(self) -> None:
        if not self.preview_enabled or cv2 is None:
            return

        with self._preview_lock:
            frame = self._preview_frame

        if frame is None:
            return

        if not self._preview_window_ready:
            try:
                cv2.namedWindow("YUI Vision", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YUI Vision", int(self._preview_width), int(self._preview_height))
                self._preview_window_ready = True
            except Exception:
                self.preview_enabled = False
                return

        try:
            cv2.imshow("YUI Vision", frame)
            if cv2.waitKey(1) == 27:  # ESC
                self._stop.set()
        except Exception:
            self.preview_enabled = False

    def _maybe_voice_ack(self) -> None:
        """
        Fast audible feedback right after STT finished (makes YUI feel responsive).
        """
        if os.getenv("YUI_VOICE_ACK_ENABLED", "1") in {"0", "false", "False"}:
            return
        cooldown_s = float(os.getenv("YUI_VOICE_ACK_COOLDOWN_S", "1.6") or "1.6")
        now = time.time()
        if cooldown_s > 0 and (now - float(self._last_voice_ack_ts)) < cooldown_s:
            return
        self._last_voice_ack_ts = now

        style = (os.getenv("YUI_VOICE_ACK_STYLE", "beep") or "beep").strip().lower()
        if style == "tts":
            text = (os.getenv("YUI_VOICE_ACK_TEXT", "Ajá.") or "Ajá.").strip()
            if text:
                self.voice.speak(text)
            return

        # Default: short beep (very low latency, non-intrusive).
        self._ack_beep()

    def _ack_beep(self) -> None:
        try:
            import winsound  # type: ignore
        except Exception:
            return
        try:
            hz = int(os.getenv("YUI_VOICE_ACK_BEEP_HZ", "900") or "900")
            ms = int(os.getenv("YUI_VOICE_ACK_BEEP_MS", "120") or "120")
            winsound.Beep(max(100, hz), max(30, ms))
        except Exception:
            try:
                winsound.MessageBeep()
            except Exception:
                pass

    def _handle_local_skills(self, text: str, *, visual: VisualContext) -> Optional[str]:
        t = (text or "").strip()
        if not t:
            return None
        low = t.lower()

        # Quick exits.
        if low in {"salir", "exit", "cerrar", "terminar"}:
            self._stop.set()
            return "Listo. Cierro."

        # Greetings.
        if low in {"hola", "buenas", "buenos días", "buenos dias", "buenas tardes", "buenas noches"}:
            name = (visual.user_name or "").strip()
            who = f", {name}" if name else ""
            return f"Hola{who}. ¿Qué necesitas?"

        # Time/date (no LLM needed).
        if ("qué hora" in low) or ("que hora" in low) or low.startswith("hora"):
            return f"Son las {time.strftime('%H:%M')}."
        if ("qué fecha" in low) or ("que fecha" in low) or ("día es hoy" in low) or ("dia es hoy" in low) or low.startswith("fecha"):
            return f"Hoy es {time.strftime('%d/%m/%Y')}."

        # Diagnostics: microphone / STT devices.
        if "microfono" in low or "micrófono" in low or low.startswith("mic "):
            if any(k in low for k in ["lista", "dispositivos", "mics"]):
                try:
                    from yui_io.stt import list_devices

                    devs = list_devices()
                    print("[YUI] STT devices (speech_recognition):")
                    for idx, name in devs.speech_recognition:
                        print(f"  - {idx}: {name}")
                    print("[YUI] STT devices (sounddevice):")
                    for idx, name in devs.sounddevice:
                        print(f"  - {idx}: {name}")
                except Exception:
                    pass

            backend = (self.settings.stt_backend or "").strip()
            sd_idx = int(self.settings.sounddevice_input_index or -1)
            sr_idx = int(self.settings.stt_microphone_index or -1)
            extra = ""
            if backend in {"sounddevice", "auto"} and sd_idx >= 0:
                try:
                    import sounddevice as sd  # type: ignore

                    info = sd.query_devices(sd_idx)
                    name = str(info.get("name", "")).strip()
                    if name:
                        extra = f" ({name})"
                except Exception:
                    pass
            return f"STT backend='{backend}', mic_index={sr_idx}, sounddevice_index={sd_idx}{extra}."

        # Privacy: camera/microphone usage (Windows)
        if any(
            k in low
            for k in [
                "estado camara",
                "estado cámara",
                "estado microfono",
                "estado micrófono",
                "estado privacidad",
            ]
        ):
            try:
                from desktop.privacy_devices import privacy_in_use_summary

                return privacy_in_use_summary()
            except Exception:
                return "No pude leer el estado de privacidad."

        if low in {"abre privacidad camara", "abre privacidad cámara", "privacidad camara", "privacidad cámara"}:
            try:
                os.startfile("ms-settings:privacy-webcam")  # type: ignore[attr-defined]
                return "Ok."
            except Exception:
                return "No pude abrir la configuración de cámara."

        if low in {"abre privacidad microfono", "abre privacidad micrófono", "privacidad microfono", "privacidad micrófono"}:
            try:
                os.startfile("ms-settings:privacy-microphone")  # type: ignore[attr-defined]
                return "Ok."
            except Exception:
                return "No pude abrir la configuración de micrófono."

        # Screen learning / recognition (embeddings only; no images stored).
        if low.startswith("aprende pantalla "):
            label = t[len("aprende pantalla ") :].strip().lower()
            if not label:
                return "Dime el nombre: aprende pantalla nombre"
            vec = capture_screen_embedding()
            if not vec:
                return "No pude capturar la pantalla ahora mismo."
            try:
                import json as _json

                user_id = (visual.user_name or "default").strip()
                self.memory.add_perception_example(user_id=user_id, kind="screen", label=label, vector_json=_json.dumps(vec))
                return f"Listo. Guardé este contexto como '{label}'."
            except Exception:
                return "No pude guardar ese ejemplo."

        if low in {"reconoce pantalla", "reconocer pantalla", "pantalla reconoce"}:
            vec = capture_screen_embedding()
            if not vec:
                return "No pude capturar la pantalla ahora mismo."
            user_id = (visual.user_name or "default").strip()
            rows = self.memory.list_perception_examples(user_id=user_id, kind="screen", limit=200)
            examples = parse_vectors(rows)
            if not examples:
                return "Aún no tengo ejemplos. Usa: aprende pantalla nombre"
            try:
                k = int(os.getenv("YUI_SCREEN_KNN_K", "3") or "3")
            except Exception:
                k = 3
            try:
                max_d = float(os.getenv("YUI_SCREEN_KNN_MAX_DISTANCE", "6.5") or "6.5")
            except Exception:
                max_d = 6.5
            learned = knn_classify(vec, examples, k=max(1, k), max_distance=float(max_d))
            if not learned:
                return "No reconocí ese contexto."
            return f"Creo que es '{learned.label}' (confianza {learned.confidence:.2f})."

        # Diagnostics: system profile / autotune.
        if (
            low.startswith("recalibra entorno")
            or low.startswith("recalibrar entorno")
            or low.startswith("actualiza entorno")
            or low.startswith("actualizar entorno")
            or low.startswith("actualiza perfil")
            or low.startswith("actualizar perfil")
            or low.startswith("recalibra perfil")
            or low.startswith("recalibrar perfil")
        ):
            try:
                prof = refresh_profile()
                self._profile = prof
                if os.getenv("YUI_AUTOTUNE", "1") not in {"0", "false", "False"}:
                    tuning = derive_tuning(
                        prof,
                        default_preview=(int(self.settings.preview_width), int(self.settings.preview_height)),
                        default_every_n=int(self.settings.vision_process_every_n_frames),
                        default_sec_interval_s=float(self._sec_watch.interval_s),
                    )
                    self._tuning_tier = tuning.tier
                    self._preview_width = int(tuning.preview_width)
                    self._preview_height = int(tuning.preview_height)
                    self._vision_every_n_frames = int(tuning.vision_every_n_frames)
                    self._sec_watch.interval_s = float(tuning.security_watch_interval_s)
                if any(k in low for k in ["detalle", "completo", "json"]):
                    try:
                        import json as _json

                        print("[YUI] System profile:")
                        print(_json.dumps(prof, ensure_ascii=False, indent=2))
                    except Exception:
                        pass
                return f"Listo. Perfil actualizado: {summarize_profile(prof)}. tier={self._tuning_tier}."
            except Exception:
                return "No pude actualizar el perfil del sistema."

        if (
            low == "entorno"
            or low.startswith("perfil sistema")
            or low.startswith("perfil del sistema")
            or low.startswith("entorno sistema")
            or low.startswith("entorno del sistema")
        ):
            try:
                prof = self._profile or load_or_collect_profile()
                self._profile = prof
            except Exception:
                return "No pude leer el perfil del sistema."
            if any(k in low for k in ["detalle", "completo", "json"]):
                try:
                    import json as _json

                    print("[YUI] System profile:")
                    print(_json.dumps(prof, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            return f"Entorno: {summarize_profile(prof)}. tier={self._tuning_tier}."

        # System audit (read-only, via PowerShell).
        if any(
            k in low
            for k in [
                "auditoria sistema",
                "auditoría sistema",
                "auditoria del sistema",
                "auditoria pc",
                "autodiagnostico sistema",
                "autodiagnóstico sistema",
            ]
        ):
            try:
                from desktop.security_audit import system_audit

                rep = system_audit()
                if rep.detail:
                    print(rep.detail)
                return rep.voice
            except Exception:
                return "No pude correr la auditoría del sistema ahora mismo."

        # Auto-analysis (runs a bundle of read-only audits in background).
        if low.startswith(("autoanalisis", "auto análisis", "auto analisis", "autodiagnostico", "autodiagnóstico", "auto diagnostico")):
            if any(k in low for k in ["estado", "status"]):
                return self._auto_analysis_status()
            if any(k in low for k in ["cancela", "cancelar", "deten", "detener", "stop"]):
                return self._cancel_auto_analysis()

            prof = "rapido"
            if any(k in low for k in ["completo", "full", "todo"]):
                prof = "completo"
            elif any(k in low for k in ["seguridad", "defensa", "defensivo"]):
                prof = "seguridad"
            elif any(k in low for k in ["navegador", "browser", "cookies", "extensiones"]):
                prof = "navegador"
            elif any(k in low for k in ["sistema", "pc", "rendimiento"]):
                prof = "rapido"
            return self._start_auto_analysis(prof)

        # Security quick audit (Windows Defender / Firewall).
        if any(k in low for k in ["auditoria seguridad", "auditoría seguridad", "revisa seguridad", "chequea seguridad", "seguridad audit"]):
            try:
                from desktop.security_audit import quick_audit

                rep = quick_audit()
                if rep.detail:
                    print(rep.detail)
                return rep.voice
            except Exception:
                return "No pude correr la auditoría ahora mismo."

        # Security watch mode (defensive monitoring)
        if any(k in low for k in ["activa vigilancia", "activar vigilancia", "modo vigilancia", "vigilancia on", "vigilancia activa"]):
            try:
                self._sec_watch.start()
            except Exception:
                return "No pude activar vigilancia ahora mismo."
            return "Listo. Activo modo vigilancia."
        if any(k in low for k in ["desactiva vigilancia", "desactivar vigilancia", "vigilancia off", "vigilancia desactiva"]):
            try:
                self._sec_watch.stop()
            except Exception:
                return "No pude desactivar vigilancia ahora mismo."
            return "Listo. Desactivo modo vigilancia."
        if any(k in low for k in ["estado vigilancia", "vigilancia estado", "vigilancia status"]):
            return "Vigilancia: " + ("activa" if self._sec_watch.is_running() else "apagada") + f" (intervalo {int(self._sec_watch.interval_s)}s)."
        if low.startswith("vigilancia intervalo "):
            raw = t[len("vigilancia intervalo ") :].strip()
            try:
                val = float(raw)
            except Exception:
                return "Usa: vigilancia intervalo 30"
            self._sec_watch.interval_s = max(5.0, min(600.0, val))
            return f"Listo. Intervalo de vigilancia: {int(self._sec_watch.interval_s)}s."

        # Offline help (safe, defensive) for common tools.
        if low in {"ayuda nmap", "nmap ayuda"}:
            return "Nmap sirve para inventariar puertos/servicios en sistemas autorizados. Para tu PC usa 'auditoria puertos' y 'escanea descargas'."
        if low in {"ayuda wireshark", "wireshark ayuda"}:
            return "Wireshark sirve para analizar tráfico y detectar conexiones raras. Úsalo en tu propio equipo/red autorizada; si quieres te digo qué filtros usar según tu caso."
        if low in {"ayuda burp", "burp ayuda", "ayuda burpsuite"}:
            return "Burp Suite sirve para auditar seguridad de apps web propias (proxy, repetidor, escáner). Dime qué app estás probando y el objetivo defensivo."

        if any(k in low for k in ["auditoria procesos", "auditoría procesos", "revisa procesos"]):
            try:
                from desktop.security_audit import processes_audit

                rep = processes_audit()
                if rep.detail:
                    print(rep.detail)
                return rep.voice
            except Exception:
                return "No pude revisar procesos ahora mismo."

        if any(k in low for k in ["auditoria puertos", "auditoría puertos", "revisa puertos", "puertos audit"]):
            try:
                from desktop.security_audit import ports_audit

                rep = ports_audit()
                if rep.detail:
                    print(rep.detail)
                return rep.voice
            except Exception:
                return "No pude revisar puertos ahora mismo."

        if any(k in low for k in ["auditoria extensiones", "auditoría extensiones", "revisa extensiones", "extensiones audit"]):
            try:
                from desktop.security_audit import extensions_audit

                rep = extensions_audit()
                if rep.detail:
                    print(rep.detail)
                return rep.voice
            except Exception:
                return "No pude revisar extensiones ahora mismo."

        # Cookies audit (privacy snapshot; no values).
        if any(k in low for k in ["auditoria cookies", "auditoría cookies", "revisa cookies", "cookies audit"]):
            try:
                from desktop.security_audit import cookies_audit

                rep = cookies_audit()
                if rep.detail:
                    print(rep.detail)
                return rep.voice
            except Exception:
                return "No pude revisar cookies ahora mismo."

        # Quick text risk check (SQLi / XSS / etc).
        if low.startswith("analiza seguridad "):
            sample = t[len("analiza seguridad ") :].strip()
            try:
                from core.security_checks import analyze_text, findings_to_voice

                findings = analyze_text(sample)
                msg = findings_to_voice(findings) or "No vi señales claras de ataque en ese texto."
                if findings:
                    print("[YUI] Security findings:", ", ".join(f"{f.kind}:{f.severity}" for f in findings))
                return msg
            except Exception:
                return "No pude analizar ese texto ahora mismo."

        # Malware scan (Windows Defender).
        if low.startswith("escanea archivo "):
            target = t[len("escanea archivo ") :].strip().strip('"')
            return self._start_malware_scan(target)
        if low.startswith("escanea carpeta "):
            target = t[len("escanea carpeta ") :].strip().strip('"')
            return self._start_malware_scan(target)
        if low in {"escanea descargas", "escanea downloads", "escanea la carpeta descargas"}:
            home = os.getenv("USERPROFILE", "")
            if not home:
                return "No pude ubicar tu carpeta Descargas."
            return self._start_malware_scan(str(Path(home) / "Downloads"))

        return None

    def _start_malware_scan(self, target: str) -> str:
        target = (target or "").strip()
        if not target:
            return "Dime qué quieres escanear: escanea archivo RUTA o escanea carpeta RUTA."

        with self._scan_lock:
            if self._scan_thread is not None and self._scan_thread.is_alive():
                return "Ya estoy escaneando algo. Espera a que termine."

            tpath = Path(target).expanduser()
            self._scan_thread = threading.Thread(target=self._malware_scan_worker, args=(tpath,), daemon=True, name="yui-malware-scan")
            self._scan_thread.start()

        return "Listo. Inicio el escaneo y te aviso cuando termine."

    def _malware_scan_worker(self, path: Path) -> None:
        try:
            from desktop.security_audit import defender_custom_scan

            rep = defender_custom_scan(path)
            if rep.detail:
                print(rep.detail)
            if not self._stop.is_set():
                self.voice.speak(rep.voice)
        except Exception as e:
            if not self._stop.is_set():
                self.voice.speak("Tuve un problema ejecutando el escaneo.")
            print(f"[YUI] Malware scan error: {type(e).__name__}: {e}")

    def _auto_analysis_status(self) -> str:
        with self._auto_analysis_lock:
            running = bool(self._auto_analysis_thread and self._auto_analysis_thread.is_alive())
            prof = (self._auto_analysis_profile or "").strip() or "rapido"
        return "Autoanalisis: " + ("activo" if running else "apagado") + f" (perfil '{prof}')."

    def _cancel_auto_analysis(self) -> str:
        with self._auto_analysis_lock:
            running = bool(self._auto_analysis_thread and self._auto_analysis_thread.is_alive())
            if not running:
                return "No estoy haciendo un autoanalisis ahora."
            self._auto_analysis_cancel.set()
        return "Ok. Cancelo el resto del autoanalisis (puede tardar unos segundos)."

    def _start_auto_analysis(self, profile: str) -> str:
        prof = (profile or "").strip().lower() or "rapido"
        with self._auto_analysis_lock:
            if self._auto_analysis_thread is not None and self._auto_analysis_thread.is_alive():
                cur = (self._auto_analysis_profile or "").strip() or "rapido"
                return f"Ya estoy haciendo un autoanalisis (perfil '{cur}')."
            self._auto_analysis_cancel.clear()
            self._auto_analysis_profile = prof
            self._auto_analysis_thread = threading.Thread(
                target=self._auto_analysis_worker,
                args=(prof,),
                daemon=True,
                name="yui-auto-analysis",
            )
            self._auto_analysis_thread.start()
        return f"Listo. Inicio autoanalisis (perfil '{prof}'). Te aviso al terminar."

    def _auto_analysis_worker(self, profile: str) -> None:
        prof = (profile or "").strip().lower() or "rapido"
        speak_end = os.getenv("YUI_AUTO_ANALYSIS_SPEAK", "1") not in {"0", "false", "False"}
        print_detail = os.getenv("YUI_AUTO_ANALYSIS_PRINT", "1") not in {"0", "false", "False"}

        try:
            self.ui_bus.publish("analysis", {"state": "start", "profile": prof})
        except Exception:
            pass

        def _step(name: str, fn):  # noqa: ANN001
            if self._stop.is_set() or self._auto_analysis_cancel.is_set():
                return None
            try:
                rep = fn()
            except Exception as e:
                print(f"[YUI] Autoanálisis '{name}' error: {type(e).__name__}: {e}")
                return None
            if getattr(rep, "detail", "") and print_detail:
                print("\n" + str(rep.detail).strip())
            return getattr(rep, "voice", None)

        steps = []
        try:
            from desktop.security_audit import cookies_audit, extensions_audit, ports_audit, processes_audit, quick_audit, system_audit

            if prof in {"rapido", "rápido", "sistema", "pc", "rendimiento"}:
                steps = [("sistema", system_audit), ("seguridad", quick_audit)]
            elif prof in {"seguridad", "defensa", "defensivo"}:
                steps = [("seguridad", quick_audit), ("procesos", processes_audit), ("puertos", ports_audit)]
            elif prof in {"navegador", "browser", "cookies", "extensiones"}:
                steps = [("extensiones", extensions_audit), ("cookies", cookies_audit)]
            elif prof in {"completo", "full", "todo"}:
                steps = [
                    ("sistema", system_audit),
                    ("seguridad", quick_audit),
                    ("procesos", processes_audit),
                    ("puertos", ports_audit),
                    ("extensiones", extensions_audit),
                    ("cookies", cookies_audit),
                ]
            else:
                steps = [("sistema", system_audit), ("seguridad", quick_audit)]
        except Exception:
            steps = []

        print(f"[YUI] Autoanalisis iniciado (perfil '{prof}').")
        voices: list[str] = []
        for name, fn in steps:
            if self._stop.is_set() or self._auto_analysis_cancel.is_set():
                break
            v = _step(name, fn)
            if isinstance(v, str) and v.strip():
                voices.append(v.strip())
            try:
                self.ui_bus.publish("analysis", {"state": "step", "profile": prof, "step": name})
            except Exception:
                pass

        canceled = self._auto_analysis_cancel.is_set()
        if canceled:
            msg = "Ok. Cancele el autoanalisis."
        else:
            msg = "Listo. Termine el autoanalisis."

        try:
            self.ui_bus.publish("analysis", {"state": "done", "profile": prof, "canceled": canceled})
        except Exception:
            pass

        if speak_end and not self._stop.is_set():
            try:
                self._speak(msg)
            except Exception:
                pass

    def _handle_task_commands(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        low = t.lower()

        if low.startswith("graba tarea ") or low.startswith("grabar tarea "):
            if os.getenv("YUI_RECORDER_ENABLED", "1") in {"0", "false", "False"}:
                return "La grabación está desactivada."
            name = t.split("tarea", 1)[-1].strip().lower()
            if not name:
                return "Dime el nombre: graba tarea nombre"
            if self._task_recorder.running:
                return f"Ya estoy grabando '{self._task_recorder.name}'."
            try:
                self._task_recorder.start(name)
            except RuntimeError:
                return "Para grabar acciones necesito instalar 'pynput'. Ejecuta: pip install pynput"
            except Exception:
                return "No pude iniciar la grabación."

            if os.getenv("YUI_RECORDER_SAVE_SCREEN_CONTEXT", "1") not in {"0", "false", "False"}:
                try:
                    vec = capture_screen_embedding()
                    if vec:
                        import json as _json

                        user_id = "default"
                        with self._vision_lock:
                            user_id = (self._vision_state.user_name or "default").strip()
                        self.memory.add_perception_example(
                            user_id=user_id,
                            kind="screen",
                            label=f"tarea:{name}",
                            vector_json=_json.dumps(vec),
                        )
                except Exception:
                    pass

            return f"Ok. Estoy grabando la tarea '{name}'. Di 'termina grabación' cuando acabes."

        if low in {"termina grabacion", "termina grabación", "fin grabacion", "fin grabación"}:
            if not self._task_recorder.running:
                return "No estoy grabando nada."
            res = self._task_recorder.stop()
            if not res.steps:
                return "No capturé pasos. Intenta de nuevo."
            task = DesktopTask(name=res.name, steps=res.steps)
            self.memory.upsert_desktop_task(name=res.name, task_json=task.to_json())
            return f"Listo. Guardé '{res.name}' con {len(res.steps)} pasos."

        if low in {"cancela grabacion", "cancela grabación", "cancelar grabacion", "cancelar grabación"}:
            if not self._task_recorder.running:
                return "No estoy grabando nada."
            self._task_recorder.cancel()
            return "Ok. Cancelé la grabación."

        if low in {"estado grabacion", "estado grabación", "grabacion estado", "grabación estado"}:
            return "Grabación: " + ("activa." if self._task_recorder.running else "apagada.")

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
            return f"Para borrar la tarea, confirma por voz: confirmar código {pending.code} (o 'cancelar')."

        return None

    def _continue_running_task(self, text: str, *, source: str) -> Optional[str]:
        if self._desktop.confirm.pending is None:
            return None
        # While paused on a confirmation, route everything through DesktopController:
        # it will only accept confirm/cancel and won't execute other commands.
        res = self._desktop.maybe_handle(text, source=source)
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
            res = self._desktop.maybe_handle(cmd, source="task")
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

        every_n = max(1, int(self._vision_every_n_frames or 1))

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
                    preview = cv2.resize(frame, (int(self._preview_width), int(self._preview_height)))
                except Exception:
                    try:
                        preview = frame.copy()
                    except Exception:
                        preview = frame
                with self._preview_lock:
                    self._preview_frame = preview

        if self.camera is not None:
            self.camera.release()

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
    try:
        # Make console output more predictable on Windows terminals.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv(override=True)
    install_crash_logging()
    if not acquire_single_instance_lock():
        print("[YUI] Ya hay otra instancia ejecutándose. Cierra la anterior y vuelve a intentar.")
        sys.exit(1)
    try:
        yui = YUI(Settings())
        yui.start()
    except Exception:
        # Ensure the error is visible even when launched by double-click.
        import traceback

        traceback.print_exc()
        if os.getenv("YUI_PAUSE_ON_CRASH", "1") not in {"0", "false", "False"}:
            try:
                input("YUI falló. Presiona Enter para salir...")
            except Exception:
                pass
