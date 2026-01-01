# YUI (voz + visión)

Copyright (c) 2026 Diego Alonso Vargas Almengor

Asistente en Python con:
- Voz: STT (SpeechRecognition) + TTS (pyttsx3 offline, gTTS fallback).
- Voz (mejor): TTS con `edge-tts` (voz femenina natural) + visualizador de mic.
- Visión: cámara + IA (MediaPipe) para rostro/manos/gestos + (opcional) reconocimiento por `known_faces/`.
- “Cerebro”: endpoint tipo OpenAI-compatible (`/v1/chat/completions`) + memoria en SQLite.

## Ejecutar
1) Instala dependencias:
`pip install -r requirements.txt`

2) Configura un LLM (recomendado) en `.env` (DeepSeek):
- `YUI_LLM_API_KEY=...`
- `YUI_LLM_BASE_URL=https://api.deepseek.com`
- `YUI_LLM_MODEL=deepseek-chat` (rápido)
- `YUI_LLM_MODEL_DEEP=deepseek-reasoner` (análisis profundo automático)

Nota: el endpoint es tipo OpenAI-compatible; aquí lo dejamos configurado para DeepSeek.

3) Ejecuta:
`python main.py`

## Rostros conocidos (opcional)
- Crea `known_faces/` en la raíz del proyecto.
- Añade imágenes como `known_faces/Ana.jpg`, `known_faces/Carlos.png`.
- YUI saludará por nombre cuando reconozca el rostro.

## Controles útiles (env vars)
- `YUI_VISION_ENABLED=1` (0 para solo voz)
- `YUI_REQUIRE_FACE_AUTH=1` (exige rostro para responder)
- `YUI_PREVIEW=1` (0 desactiva ventana de cámara)
- `YUI_CAMERA_BACKEND=auto` (auto/dshow/msmf)
- `YUI_VISION_EVERY_N_FRAMES=2` (más alto = menos CPU, más rápido)
- `YUI_FACE_EMOTION_THRESHOLD=0.55` (umbral para emociones)
- `YUI_HAND_GESTURE_THRESHOLD=0.55` (umbral para gestos aprendidos)
- `YUI_TEACH_COOLDOWN_S=25` (cada cuánto puede preguntar para enseñar)
- `YUI_KNN_K=5` y `YUI_KNN_MAX_DISTANCE=2.5` (clasificador simple)
- `YUI_WAKE_WORD_ENABLED=1` (0 desactiva activación por “YUI”)
- `YUI_WAKE_WORD=yui`
- `YUI_STT_BACKEND=auto` (auto/speech_recognition/sounddevice/text)
- `YUI_MIC_INDEX=-1` (SpeechRecognition; usa `scripts/list_microphones.py`)
- `YUI_SOUNDDEVICE_INDEX=-1` (SoundDevice; usa `scripts/list_microphones.py`)
- `YUI_LISTEN_TIMEOUT_S=5` y `YUI_PHRASE_TIME_LIMIT_S=8` (bájalos para responder más rápido)
- `YUI_TTS_ENGINE=edge` y `YUI_TTS_VOICE=es-MX-DaliaNeural` (voz femenina natural)
- `YUI_PREVIEW_WIDTH=480` y `YUI_PREVIEW_HEIGHT=270` (ventana pequeña de cámara)
- `YUI_MIC_METER_ENABLED=1` (barra de nivel del mic en la ventana)
- `YUI_DESKTOP_ENABLED=1` (habilita comandos de escritorio; borrar/modificar requiere confirmación)
- Entrada siempre activa: voz + texto (escribe en consola en cualquier momento). Di `no escuches` para apagar micrófono y `escucha` para reactivarlo.
- `YUI_UI_WS_ENABLED=1` (emite eventos por WebSocket para UI 3D)
- `YUI_UI_WS_HOST=127.0.0.1` y `YUI_UI_WS_PORT=8765`
- `YUI_TTS_SAVE_LAST=1` y `YUI_TTS_LAST_PATH=data/last_tts.mp3` (para lip-sync por fonemas en el avatar)
- `YUI_MEMORY_SHORT_TURNS=12` (memoria corta)
- `YUI_MEMORY_LONG_FACTS=20` (memoria larga: hechos)
- `YUI_MEMORY_USE_SUMMARIES=1` (resúmenes automáticos)
- `YUI_MEMORY_USE_LONG_TERM=1` (recuerdos naturales)
- `YUI_MEMORY_RETRIEVE_K=6` (cuántos recuerdos inyectar por respuesta)
- `YUI_MEMORY_EXTRACT_EVERY_N=8` (cada cuántas respuestas crear nuevos recuerdos)

## Notas
- En Windows, si el micrófono no funciona por dependencias del sistema, YUI cae a entrada por texto.
- No subas `.env` a git (hay `.gitignore`).
