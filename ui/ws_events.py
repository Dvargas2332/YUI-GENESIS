from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class UiEvent:
    type: str
    data: Dict[str, Any]
    ts: float


class UiEventBus:
    def __init__(self):
        self._q: "queue.Queue[UiEvent]" = queue.Queue()

    def publish(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._q.put(UiEvent(type=event_type, data=data or {}, ts=time.time()))

    def get(self, timeout_s: float = 0.25) -> Optional[UiEvent]:
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            return None


def event_to_json(event: UiEvent) -> str:
    payload = {"type": event.type, "ts": event.ts, "data": event.data}
    return json.dumps(payload, ensure_ascii=False)

