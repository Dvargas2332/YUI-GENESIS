from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional, Set

from ui.ws_events import UiEventBus, event_to_json


@dataclass(frozen=True)
class WsConfig:
    host: str = "127.0.0.1"
    port: int = 8765


class UiWsServer:
    def __init__(self, bus: UiEventBus, *, config: WsConfig):
        self.bus = bus
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        try:
            import websockets  # type: ignore
        except Exception:
            print("[YUI] UI WS: falta instalar websockets.")
            return

        clients: Set = set()

        async def handler(ws):
            clients.add(ws)
            try:
                await ws.send(event_to_json(self.bus.get(timeout_s=0.0) or self._hello_event()))
                await ws.wait_closed()
            finally:
                clients.discard(ws)

        async def broadcaster():
            while not self._stop.is_set():
                ev = self.bus.get(timeout_s=0.1)
                if ev is None:
                    await asyncio.sleep(0.01)
                    continue
                msg = event_to_json(ev)
                if not clients:
                    continue
                dead = []
                for c in list(clients):
                    try:
                        await c.send(msg)
                    except Exception:
                        dead.append(c)
                for d in dead:
                    clients.discard(d)

        port = int(self.config.port)
        max_tries = 20
        last_error: Optional[Exception] = None
        for _ in range(max_tries):
            try:
                async with websockets.serve(handler, self.config.host, port):
                    print(f"[YUI] UI WS: ws://{self.config.host}:{port}")
                    await broadcaster()
                    return
            except OSError as e:
                # Windows: 10048 = address already in use
                last_error = e
                port += 1
                continue

        if last_error is not None:
            print(f"[YUI] UI WS: no pude iniciar servidor (último error: {last_error})")

    def _hello_event(self):
        from ui.ws_events import UiEvent
        import time

        return UiEvent(type="hello", data={"app": "YUI"}, ts=time.time())
