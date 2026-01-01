# YUI (voz + visión)

Asistente virtual en Python con voz, visión, memoria y control de escritorio (con confirmaciones para acciones destructivas).

Copyright (c) 2026 Diego Alonso Vargas Almengor

## Funciones
- Voz: STT + TTS (`edge-tts` con voz femenina natural) + visualizador de mic.
- Visión: cámara + IA (MediaPipe Tasks) para rostro/manos/gestos + (opcional) reconocimiento por `known_faces/`.
- LLM: endpoint tipo OpenAI-compatible (`/v1/chat/completions`), configurado para DeepSeek por defecto.
- Memoria: SQLite (corto/largo plazo + “recuerdos” naturales).
- UI 3D: eventos por WebSocket para integración con Unity.

## Ejecutar
1) Instala dependencias:
`pip install -r requirements.txt`

2) Crea tu `.env` (puedes copiar `.env.example`) y configura el LLM:
- `YUI_LLM_API_KEY=...`
- `YUI_LLM_BASE_URL=https://api.deepseek.com`
- `YUI_LLM_MODEL=deepseek-chat` (rápido)
- `YUI_LLM_MODEL_DEEP=deepseek-reasoner` (análisis profundo automático)

3) Ejecuta:
`python main.py`

## Rostros conocidos (opcional)
- Crea `known_faces/` en la raíz del proyecto.
- Añade imágenes como `known_faces/Ana.jpg`, `known_faces/Carlos.png`.
- YUI saludará por nombre cuando reconozca el rostro.

## Controles útiles (env vars)
- `YUI_PREVIEW=1` (0 desactiva la ventana de cámara)
- `YUI_VISION_EVERY_N_FRAMES=2` (más alto = menos CPU)
- `YUI_STT_BACKEND=auto` (auto/speech_recognition/sounddevice/text)
- `YUI_TTS_ENGINE=edge` y `YUI_TTS_VOICE=es-MX-DaliaNeural`
- `YUI_DESKTOP_ENABLED=1` (comandos de escritorio; borrar/modificar requiere confirmación)
- Entrada siempre activa: voz + texto. Di `no escuches` para apagar micrófono y `escucha` para reactivarlo.

## Notas
- No subas `.env` a git (hay `.gitignore`).
