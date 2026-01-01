from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DesktopTask:
    name: str
    steps: List[str]

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "steps": self.steps}, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> Optional["DesktopTask"]:
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        name = str(obj.get("name", "")).strip()
        steps = obj.get("steps", [])
        if not name or not isinstance(steps, list):
            return None
        out_steps: List[str] = []
        for st in steps:
            if isinstance(st, str) and st.strip():
                out_steps.append(st.strip())
        return DesktopTask(name=name, steps=out_steps)

