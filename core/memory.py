from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Tuple


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class MemoryItem:
    role: Role
    content: str
    created_at: str


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    salience REAL NOT NULL DEFAULT 0.5,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS perception_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL, -- 'face' | 'hand'
                    label TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS desktop_tasks (
                    name TEXT PRIMARY KEY,
                    task_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    up_to_message_id INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS macros (
                    trigger TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add(self, role: Role, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, created_at),
            )
            conn.commit()

    def recent(self, limit: int = 20) -> List[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute("SELECT role, content, created_at FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        rows.reverse()
        return [MemoryItem(role=row[0], content=row[1], created_at=row[2]) for row in rows]

    def recent_with_ids(self, limit: int = 50) -> List[tuple[int, MemoryItem]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM messages ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        rows.reverse()
        return [(int(r[0]), MemoryItem(role=r[1], content=r[2], created_at=r[3])) for r in rows]

    def upsert_fact(self, key: str, value: str) -> None:
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, updated_at),
            )
            conn.commit()

    def list_facts(self, limit: int = 20) -> List[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def add_memory(self, *, user_id: str, content: str, salience: float = 0.6, tags: str = "") -> None:
        user_id = (user_id or "default").strip()
        content = (content or "").strip()
        if not content:
            return
        sal = float(salience)
        sal = max(0.0, min(1.0, sal))
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (user_id, content, salience, tags, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, content, sal, (tags or "").strip(), created_at),
            )
            conn.commit()

    def recent_memories(self, *, user_id: str, limit: int = 20) -> List[tuple[int, str, float, str]]:
        user_id = (user_id or "default").strip()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content, salience, tags FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, int(limit)),
            ).fetchall()
        return [(int(r[0]), str(r[1]), float(r[2]), str(r[3] or "")) for r in rows]

    def mark_memory_used(self, memory_ids: List[int]) -> None:
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany("UPDATE memories SET last_used_at=? WHERE id=?", [(now, int(mid)) for mid in memory_ids])
            conn.commit()

    def add_perception_example(self, *, user_id: str, kind: str, label: str, vector_json: str) -> None:
        user_id = (user_id or "default").strip()
        kind = (kind or "").strip().lower()
        label = (label or "").strip().lower()
        vector_json = (vector_json or "").strip()
        if kind not in {"face", "hand", "screen"}:
            return
        if not label or not vector_json:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO perception_examples (user_id, kind, label, vector_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, kind, label, vector_json, created_at),
            )
            conn.commit()

    def list_perception_examples(self, *, user_id: str, kind: str, limit: int = 200) -> List[tuple[str, str]]:
        user_id = (user_id or "default").strip()
        kind = (kind or "").strip().lower()
        if kind not in {"face", "hand", "screen"}:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT label, vector_json FROM perception_examples WHERE user_id=? AND kind=? ORDER BY id DESC LIMIT ?",
                (user_id, kind, int(limit)),
            ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def upsert_desktop_task(self, *, name: str, task_json: str) -> None:
        name = (name or "").strip().lower()
        task_json = (task_json or "").strip()
        if not name or not task_json:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO desktop_tasks (name, task_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET task_json=excluded.task_json, updated_at=excluded.updated_at",
                (name, task_json, updated_at),
            )
            conn.commit()

    def get_desktop_task(self, *, name: str) -> Optional[str]:
        name = (name or "").strip().lower()
        if not name:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT task_json FROM desktop_tasks WHERE name=? LIMIT 1", (name,)).fetchone()
        return str(row[0]) if row else None

    def list_desktop_tasks(self, limit: int = 50) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM desktop_tasks ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [str(r[0]) for r in rows]

    def set_preference(self, key: str, value: str) -> None:
        key = (key or "").strip().lower()
        value = (value or "").strip()
        if not key:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, updated_at),
            )
            conn.commit()

    def get_preference(self, key: str) -> Optional[str]:
        key = (key or "").strip().lower()
        if not key:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key=? LIMIT 1", (key,)).fetchone()
        if not row:
            return None
        return str(row[0])

    def delete_preference(self, key: str) -> None:
        key = (key or "").strip().lower()
        if not key:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM preferences WHERE key=?", (key,))
            conn.commit()

    def list_preferences(self, limit: int = 50) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        out: List[Tuple[str, str]] = []
        for r in rows:
            out.append((str(r[0]), str(r[1])))
        return out

    def upsert_macro(self, *, trigger: str, action: str) -> None:
        trigger = (trigger or "").strip().lower()
        action = (action or "").strip()
        if not trigger or not action:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO macros (trigger, action, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(trigger) DO UPDATE SET action=excluded.action, updated_at=excluded.updated_at",
                (trigger, action, updated_at),
            )
            conn.commit()

    def get_macro(self, *, trigger: str) -> Optional[str]:
        trigger = (trigger or "").strip().lower()
        if not trigger:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT action FROM macros WHERE trigger=? LIMIT 1", (trigger,)).fetchone()
        return str(row[0]) if row else None

    def list_macros(self, limit: int = 50) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT trigger, action FROM macros ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
        out: List[Tuple[str, str]] = []
        for r in rows:
            out.append((str(r[0]), str(r[1])))
        return out

    def delete_macro(self, *, trigger: str) -> None:
        trigger = (trigger or "").strip().lower()
        if not trigger:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM macros WHERE trigger=?", (trigger,))
            conn.commit()

    def delete_desktop_task(self, *, name: str) -> None:
        name = (name or "").strip().lower()
        if not name:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM desktop_tasks WHERE name=?", (name,))
            conn.commit()

    def add_summary(self, up_to_message_id: int, summary: str) -> None:
        summary = (summary or "").strip()
        if not summary:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (up_to_message_id, summary, created_at) VALUES (?, ?, ?)",
                (int(up_to_message_id), summary, created_at),
            )
            conn.commit()

    def latest_summary(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT summary FROM summaries ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return str(row[0])

    def latest_summary_upto_id(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT up_to_message_id FROM summaries ORDER BY id DESC LIMIT 1").fetchone()
        return int(row[0]) if row else 0
