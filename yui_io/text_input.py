from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextInput:
    prompt: str = "Tú (texto): "

    def __post_init__(self):
        self._q: "queue.Queue[str]" = queue.Queue()
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

    def get(self) -> Optional[str]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return
        while not self._stop.is_set():
            try:
                line = input(self.prompt)
            except EOFError:
                return
            except Exception:
                continue
            line = (line or "").strip()
            if line:
                self._q.put(line)

