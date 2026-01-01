from __future__ import annotations

from typing import Optional

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

class CameraManager:
    def __init__(
        self,
        camera_index: int = 0,
        *,
        backend: str = "dshow",
        width: int = 1280,
        height: int = 720,
        fourcc: str = "",
    ):
        self.camera_index = camera_index
        self.backend = backend
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.cap = None
        self.backend_used = "none"
        self._consecutive_failures = 0
        self._open()

    def _open(self) -> None:
        if cv2 is None:
            self.cap = None
            self.backend_used = "none"
            return

        backend_norm = (self.backend or "").strip().lower()
        if backend_norm in {"dshow", "directshow"}:
            backends_to_try = [("dshow", getattr(cv2, "CAP_DSHOW", None))]
        elif backend_norm in {"msmf"}:
            backends_to_try = [("msmf", getattr(cv2, "CAP_MSMF", None))]
        elif backend_norm in {"any", "auto"}:
            backends_to_try = [
                ("dshow", getattr(cv2, "CAP_DSHOW", None)),
                ("msmf", getattr(cv2, "CAP_MSMF", None)),
                ("any", None),
            ]
        else:
            backends_to_try = [("any", None)]

        self.cap = None
        self.backend_used = "none"

        indices_to_try = [self.camera_index]
        if int(self.camera_index) < 0:
            indices_to_try = list(range(0, 6))

        for idx in indices_to_try:
            for backend_name, api_preference in backends_to_try:
                cap = None
                try:
                    cap = cv2.VideoCapture(idx) if api_preference is None else cv2.VideoCapture(idx, api_preference)
                except Exception:
                    cap = None

                if cap is None:
                    continue

                try:
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                except Exception:
                    pass

                try:
                    if self.width and self.height:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
                except Exception:
                    pass

                requested_fourcc = (self.fourcc or "").strip().upper()
                candidates = [requested_fourcc] if requested_fourcc else ["MJPG", "YUY2"]
                for code in candidates:
                    if not code:
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
                    except Exception:
                        continue

                try:
                    opened = bool(cap.isOpened())
                except Exception:
                    opened = False

                if not opened:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    continue

                self.cap = cap
                self.camera_index = idx
                self.backend_used = backend_name
                self._consecutive_failures = 0
                return

        try:
            if self.cap is not None and not self.cap.isOpened():
                self.cap.release()
                self.cap = None
        except Exception:
            self.cap = None
        
    def get_frame(self) -> Optional[tuple]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 10:
                self.restart()
            return None

        self._consecutive_failures = 0
        return frame
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.backend_used = "none"
        self._consecutive_failures = 0

    def restart(self) -> None:
        self.release()
        self._open()
