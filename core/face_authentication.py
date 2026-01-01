from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


class FaceAuthenticator:
    """
    Autenticación facial ligera con OpenCV (Haar + LBPH).

    - Detección: Haar cascade (incluido con OpenCV).
    - Reconocimiento: LBPH si está disponible (requiere opencv-contrib-python).

    Para registrar usuarios: coloca imágenes en `known_faces/` con nombre `usuario.jpg/png`.
    """

    def __init__(self, models_dir: Path | None = None, known_faces_dir: Path | None = None):
        self._project_root = Path(__file__).resolve().parents[1]
        self._known_faces_dir = known_faces_dir or (self._project_root / "known_faces")

        self.enabled = cv2 is not None
        self._face_cascade = None
        self._recognizer = None
        self._label_to_name: Dict[int, str] = {}

        if not self.enabled:
            return

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._face_cascade.empty():
            self.enabled = False
            return

        # LBPH (si existe en esta build de OpenCV)
        recognizer_factory = getattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create", None)
        if recognizer_factory is not None:
            self._recognizer = recognizer_factory()

        self.load_known_faces(self._known_faces_dir)

    def load_known_faces(self, directory: str | Path) -> None:
        if not self.enabled or cv2 is None:
            return

        if self._recognizer is None:
            return
        if np is None:
            return

        directory_path = Path(directory)
        if not directory_path.is_absolute():
            directory_path = self._project_root / directory_path
        if not directory_path.exists():
            return

        images: List = []
        labels: List[int] = []
        self._label_to_name = {}

        label = 0
        for filename in sorted(os.listdir(directory_path)):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            name = os.path.splitext(filename)[0]
            path = directory_path / filename
            img = cv2.imread(str(path))
            if img is None:
                continue

            face = self._extract_face_gray(img)
            if face is None:
                continue

            images.append(face)
            labels.append(label)
            self._label_to_name[label] = name
            label += 1

        if not images:
            return

        labels_np = np.array(labels, dtype=np.int32)
        self._recognizer.train(images, labels_np)

    def authenticate(self, frame) -> Tuple[Optional[str], float]:
        """
        Retorna: (nombre_usuario, confianza) o (None, 0) si no coincide.
        """
        if not self.enabled or cv2 is None or self._face_cascade is None:
            return None, 0.0

        face = self._extract_face_gray(frame)
        if face is None:
            return None, 0.0

        if self._recognizer is None or not self._label_to_name:
            # Al menos detecta presencia de rostro.
            return None, 0.2

        try:
            label, distance = self._recognizer.predict(face)
        except Exception:
            return None, 0.0

        # LBPH: distancia menor es mejor. Umbral orientativo (ajustable).
        if distance < 60 and label in self._label_to_name:
            # Convertimos distancia a “confianza” 0..1 (aprox).
            confidence = max(0.0, min(1.0, 1.0 - (distance / 80.0)))
            return self._label_to_name[label], confidence

        return None, 0.0

    def register_new_face(self, frame, user_id: str) -> None:
        if not self.enabled or cv2 is None:
            return

        face = self._extract_face_bgr(frame)
        if face is None:
            return

        self._known_faces_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._known_faces_dir / f"{user_id}.jpg"
        cv2.imwrite(str(out_path), face)
        self.load_known_faces(self._known_faces_dir)

    def _extract_face_bgr(self, frame):
        if cv2 is None or self._face_cascade is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None

        # toma el rostro más grande
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_bgr = frame[y : y + h, x : x + w]
        return face_bgr

    def _extract_face_gray(self, frame):
        if cv2 is None:
            return None
        face_bgr = self._extract_face_bgr(frame)
        if face_bgr is None:
            return None
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))
        return gray
