from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class MicLevel:
    value: float = 0.0  # 0..1


class MicMeter:
    def __init__(self, *, device_index: int = -1):
        self.device_index = int(device_index)
        self.level = MicLevel(0.0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd  # type: ignore
        except Exception:
            return

        samplerate = 16000
        block_ms = 60
        block = int(samplerate * (block_ms / 1000.0))

        alpha = 0.25  # smoothing

        kwargs = {"samplerate": samplerate, "channels": 1, "dtype": "int16", "blocksize": block}
        if self.device_index >= 0:
            kwargs["device"] = self.device_index

        try:
            with sd.InputStream(**kwargs) as stream:
                while not self._stop.is_set():
                    data, _overflowed = stream.read(block)
                    x = np.asarray(data).astype(np.float32)
                    if x.ndim == 2:
                        x = x[:, 0]
                    rms = float(np.sqrt(np.mean(x * x)) + 1e-6)
                    # Map to 0..1 (empirical)
                    v = max(0.0, min(1.0, rms / 8000.0))
                    self.level.value = (1.0 - alpha) * self.level.value + alpha * v
                    time.sleep(0.01)
        except Exception:
            return
