"""Task-specific agent identities — Devon as a *somebody*.

Google CC/Carly pattern: the agent can own task-scoped identities
(name, email, handle) for delegated work — e.g. "Devon Research" with
its own handle when it books, mails, or posts on the owner's behalf.

Every method never raises.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_db() -> str:
    d = Path.home() / ".nomorals" / "agents"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(d / "identities.db")


@dataclass
class AgentIdentity:
    """One task-scoped identity Devon can act under."""

    id: str = ""
    name: str = ""          # e.g. "Devon Research"
    email: str = ""         # e.g. "devon.research@…"
    handle: str = ""        # e.g. "@devon_research"
    purpose: str = ""       # what this identity is for
    created_at: float = 0.0

    def label(self) -> str:
        return self.name or self.handle or self.id

    def signature(self) -> str:
        bits = [self.name]
        if self.handle:
            bits.append(self.handle)
        return " ".join(b for b in bits if b)


class IdentityStore:
    """SQLite persistence for agent identities. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        try:
            path = db_path or _default_db()
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS identities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL DEFAULT '',
                    handle TEXT NOT NULL DEFAULT '',
                    purpose TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )"""
            )
            self._db.commit()
        except Exception:
            self._db = None

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(self, name: str, *, email: str = "", handle: str = "",
               purpose: str = "") -> AgentIdentity | None:
        """Mint a new task identity. Never raises."""
        try:
            name = (name or "").strip()
            if not name or self._db is None:
                return None
            ident = AgentIdentity(
                id="ident_" + uuid.uuid4().hex[:10],
                name=name,
                email=(email or "").strip(),
                handle=(handle or "").strip(),
                purpose=(purpose or "").strip(),
                created_at=time.time(),
            )
            with self._lock:
                self._db.execute(
                    "INSERT INTO identities VALUES (?, ?, ?, ?, ?, ?)",
                    (ident.id, ident.name, ident.email, ident.handle,
                     ident.purpose, ident.created_at),
                )
                self._db.commit()
            return ident
        except Exception:
            return None

    def get(self, identity_id: str) -> AgentIdentity | None:
        try:
            if not identity_id or self._db is None:
                return None
            with self._lock:
                row = self._db.execute(
                    "SELECT * FROM identities WHERE id = ?", (identity_id,)
                ).fetchone()
            return self._row_to_identity(row) if row else None
        except Exception:
            return None

    def list(self) -> list[AgentIdentity]:
        try:
            if self._db is None:
                return []
            with self._lock:
                rows = self._db.execute(
                    "SELECT * FROM identities ORDER BY created_at"
                ).fetchall()
            return [self._row_to_identity(r) for r in rows]
        except Exception:
            return []

    def remove(self, identity_id: str) -> bool:
        try:
            if not identity_id or self._db is None:
                return False
            with self._lock:
                cur = self._db.execute(
                    "DELETE FROM identities WHERE id = ?", (identity_id,))
                self._db.commit()
                return cur.rowcount > 0
        except Exception:
            return False

    @staticmethod
    def _row_to_identity(row: Any) -> AgentIdentity:
        return AgentIdentity(
            id=row["id"], name=row["name"], email=row["email"],
            handle=row["handle"], purpose=row["purpose"],
            created_at=row["created_at"],
        )
