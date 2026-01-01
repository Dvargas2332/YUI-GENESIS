from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


class GestureController:
    """
    Detector de gestos basado en pose.

    Nota: este módulo se desactiva automáticamente si faltan dependencias o pesos.
    """

    def __init__(self, models_dir: Path | None = None):
        self._project_root = Path(__file__).resolve().parents[1]
        self._models_dir = models_dir or (self._project_root / "models")
        self.enabled = True

        self.model = None
        self._face_cascade = None
        self._smile_cascade = None

        # Prefer YOLO pose if available.
        if YOLO is not None and np is not None:
            weights_path = self._models_dir / "yolov8n-pose.pt"
            if weights_path.exists():
                try:
                    self.model = YOLO(str(weights_path))
                    return
                except Exception:
                    self.model = None

        # Fallback: simple smile detection via Haar cascades (OpenCV only).
        if cv2 is None:
            self.enabled = False
            return

        face_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        smile_path = Path(cv2.data.haarcascades) / "haarcascade_smile.xml"
        self._face_cascade = cv2.CascadeClassifier(str(face_path))
        self._smile_cascade = cv2.CascadeClassifier(str(smile_path))
        if self._face_cascade.empty() or self._smile_cascade.empty():
            self.enabled = False

    def detect_gestures(self, frame) -> List[str]:
        if not self.enabled:
            return []

        if self.model is None:
            return self._detect_smile(frame)

        results = self.model(frame, stream=True, verbose=False)
        gestures: List[str] = []

        for r in results:
            if r.keypoints is None:
                continue
            try:
                keypoints = r.keypoints.xy[0].cpu().numpy()
            except Exception:
                continue

            if keypoints.shape[0] >= 10 and self._is_hand_raised(keypoints):
                gestures.append("hand_raised")

        return gestures

    def _detect_smile(self, frame) -> List[str]:
        if cv2 is None or self._face_cascade is None or self._smile_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            return []

        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        roi_gray = gray[y : y + h, x : x + w]
        smiles = self._smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        return ["smile"] if len(smiles) > 0 else []

    def _is_hand_raised(self, keypoints) -> bool:
        # Heurística simple: muñeca por encima del hombro (coordenada Y más pequeña).
        wrist = keypoints[9]  # muñeca
        shoulder = keypoints[5]  # hombro
        return bool(wrist[1] < shoulder[1])
