from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    # Be robust to Windows env / dotenv values that may include trailing CR or spaces.
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() not in {"0", "false", "False", "no", "NO"}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return default if value is None else Path(value.strip())


@dataclass(frozen=True)
class Settings:
    # NOTE: use default_factory so env vars are read at instantiation time
    # (after `load_dotenv()`), not at import time.
    camera_index: int = field(default_factory=lambda: _env_int("YUI_CAMERA_INDEX", 0))  # -1 = auto-scan
    camera_backend: str = field(default_factory=lambda: _env_str("YUI_CAMERA_BACKEND", "dshow"))  # dshow|msmf|any
    camera_fourcc: str = field(default_factory=lambda: _env_str("YUI_CAMERA_FOURCC", ""))  # e.g. MJPG, YUY2
    camera_width: int = field(default_factory=lambda: _env_int("YUI_CAMERA_WIDTH", 1280))
    camera_height: int = field(default_factory=lambda: _env_int("YUI_CAMERA_HEIGHT", 720))
    camera_swap_rb: bool = field(default_factory=lambda: _env_bool("YUI_CAMERA_SWAP_RB", False))
    known_faces_dir: Path = field(default_factory=lambda: _env_path("YUI_KNOWN_FACES_DIR", PROJECT_ROOT / "known_faces"))
    models_dir: Path = field(default_factory=lambda: _env_path("YUI_MODELS_DIR", PROJECT_ROOT / "models"))

    vision_enabled: bool = field(default_factory=lambda: _env_bool("YUI_VISION_ENABLED", True))
    require_face_auth: bool = field(default_factory=lambda: _env_bool("YUI_REQUIRE_FACE_AUTH", False))
    vision_process_every_n_frames: int = field(default_factory=lambda: _env_int("YUI_VISION_EVERY_N_FRAMES", 2))
    preview_width: int = field(default_factory=lambda: _env_int("YUI_PREVIEW_WIDTH", 480))
    preview_height: int = field(default_factory=lambda: _env_int("YUI_PREVIEW_HEIGHT", 270))
    mic_meter_enabled: bool = field(default_factory=lambda: _env_bool("YUI_MIC_METER_ENABLED", True))
    perception_teach_cooldown_s: float = field(default_factory=lambda: _env_float("YUI_TEACH_COOLDOWN_S", 25.0))
    face_emotion_threshold: float = field(default_factory=lambda: _env_float("YUI_FACE_EMOTION_THRESHOLD", 0.55))
    hand_gesture_threshold: float = field(default_factory=lambda: _env_float("YUI_HAND_GESTURE_THRESHOLD", 0.55))
    knn_k: int = field(default_factory=lambda: _env_int("YUI_KNN_K", 5))
    knn_max_distance: float = field(default_factory=lambda: _env_float("YUI_KNN_MAX_DISTANCE", 2.5))

    # Speech recognition
    stt_language: str = field(default_factory=lambda: _env_str("YUI_STT_LANGUAGE", "es-ES"))
    listen_timeout_s: float = field(default_factory=lambda: _env_float("YUI_LISTEN_TIMEOUT_S", 5.0))
    phrase_time_limit_s: float = field(default_factory=lambda: _env_float("YUI_PHRASE_TIME_LIMIT_S", 8.0))
    stt_microphone_index: int = field(default_factory=lambda: _env_int("YUI_MIC_INDEX", -1))  # -1 = default
    stt_backend: str = field(default_factory=lambda: _env_str("YUI_STT_BACKEND", "auto"))  # auto|speech_recognition|sounddevice|text
    sounddevice_input_index: int = field(default_factory=lambda: _env_int("YUI_SOUNDDEVICE_INDEX", -1))  # -1 = default
    wake_word_enabled: bool = field(default_factory=lambda: _env_bool("YUI_WAKE_WORD_ENABLED", True))
    wake_word: str = field(default_factory=lambda: _env_str("YUI_WAKE_WORD", "yui"))
    wake_listen_timeout_s: float = field(default_factory=lambda: _env_float("YUI_WAKE_LISTEN_TIMEOUT_S", 2.0))
    wake_phrase_time_limit_s: float = field(default_factory=lambda: _env_float("YUI_WAKE_PHRASE_TIME_LIMIT_S", 2.5))

    # TTS (pyttsx3 is offline; gTTS fallback uses network)
    tts_rate: int = field(default_factory=lambda: _env_int("YUI_TTS_RATE", 165))
    tts_volume: float = field(default_factory=lambda: _env_float("YUI_TTS_VOLUME", 1.0))
    tts_engine: str = field(default_factory=lambda: _env_str("YUI_TTS_ENGINE", "edge"))  # edge|pyttsx3|gtts
    tts_voice: str = field(default_factory=lambda: _env_str("YUI_TTS_VOICE", "es-MX-DaliaNeural"))
    tts_edge_rate: str = field(default_factory=lambda: _env_str("YUI_TTS_EDGE_RATE", "+0%"))
    tts_edge_volume: str = field(default_factory=lambda: _env_str("YUI_TTS_EDGE_VOLUME", "+0%"))

    # OpenAI-compatible chat endpoint (DeepSeek by default)
    llm_base_url: str = field(default_factory=lambda: _env_str("YUI_LLM_BASE_URL", _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")))
    llm_api_key: str = field(default_factory=lambda: _env_str("YUI_LLM_API_KEY", _env_str("DEEPSEEK_API_KEY", "")))
    llm_model: str = field(default_factory=lambda: _env_str("YUI_LLM_MODEL", "deepseek-chat"))  # default fast
    llm_model_fast: str = field(default_factory=lambda: _env_str("YUI_LLM_MODEL_FAST", _env_str("YUI_LLM_MODEL", "deepseek-chat")))
    llm_model_deep: str = field(default_factory=lambda: _env_str("YUI_LLM_MODEL_DEEP", "deepseek-reasoner"))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("YUI_LLM_TIMEOUT_S", 30.0))
    llm_temperature: float = field(default_factory=lambda: _env_float("YUI_LLM_TEMPERATURE", 0.4))
    llm_deep_temperature: float = field(default_factory=lambda: _env_float("YUI_LLM_DEEP_TEMPERATURE", 0.2))

    # Memory
    memory_db_path: Path = field(default_factory=lambda: _env_path("YUI_MEMORY_DB_PATH", PROJECT_ROOT / "data" / "user_data.db"))
    memory_short_term_turns: int = field(default_factory=lambda: _env_int("YUI_MEMORY_SHORT_TURNS", 12))
    memory_long_term_facts: int = field(default_factory=lambda: _env_int("YUI_MEMORY_LONG_FACTS", 20))
    memory_use_summaries: bool = field(default_factory=lambda: _env_bool("YUI_MEMORY_USE_SUMMARIES", True))
    memory_summary_every_n_messages: int = field(default_factory=lambda: _env_int("YUI_MEMORY_SUMMARY_EVERY_N", 30))
    memory_use_long_term: bool = field(default_factory=lambda: _env_bool("YUI_MEMORY_USE_LONG_TERM", True))
    memory_retrieve_k: int = field(default_factory=lambda: _env_int("YUI_MEMORY_RETRIEVE_K", 6))
    memory_extract_every_n_turns: int = field(default_factory=lambda: _env_int("YUI_MEMORY_EXTRACT_EVERY_N", 8))
