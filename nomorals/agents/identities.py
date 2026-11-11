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
from .identities_failures import record_failure


def _default_db() -> str:
    d = Path.home() / ".nomorals" / "agents"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        record_failure("_default_db", e)
    return str(d / "identities.db")


#: The store's own DDL — the native path executes it and the seam
#: mapping is derived from it, so the two can never drift.
_DDL = """CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL DEFAULT '',
    handle TEXT NOT NULL DEFAULT '',
    purpose TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
)"""


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
    """Persistence for agent identities. Never raises.

    Native path: a SQLite file at ``db_path``. When a document store
    is resolved (Mongo program P5 — an explicit ``store`` or the boot
    ambient), the same API runs over the seam in the
    ``agent-identities__identities`` collection instead (registered
    in storage.sidecars), and the file's existing rows are imported
    once, through the same mapping, at first construction.
    """

    def __init__(self, db_path: str = "", *, store: Any = None) -> None:
        self._db: sqlite3.Connection | None = None
        self._port: Any = None
        self._lock = threading.Lock()
        try:
            path = db_path or _default_db()
            from ..storage.sidecars import sidecar_port

            port = sidecar_port("agent-identities", _DDL, ("identities",),
                                path, store=store)
            if port is not None:
                table = port.tables["identities"]

                def write_identities(rows: list[dict[str, Any]]) -> int:
                    for row in rows:
                        table.put(row)
                    return len(rows)

                port.ensure_imported({"identities": write_identities})
                self._port = port
                return
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(_DDL)
            self._db.commit()
        except Exception as e:
            record_failure("__init__", e)
            self._db = None
            self._port = None

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(self, name: str, *, email: str = "", handle: str = "",
               purpose: str = "") -> AgentIdentity | None:
        """Mint a new task identity. Never raises."""
        try:
            name = (name or "").strip()
            if not name or (self._db is None and self._port is None):
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
                if self._port is not None:
                    self._port.tables["identities"].put({
                        "id": ident.id, "name": ident.name,
                        "email": ident.email, "handle": ident.handle,
                        "purpose": ident.purpose,
                        "created_at": ident.created_at,
                    })
                    return ident
                self._db.execute(
                    "INSERT INTO identities VALUES (?, ?, ?, ?, ?, ?)",
                    (ident.id, ident.name, ident.email, ident.handle,
                     ident.purpose, ident.created_at),
                )
                self._db.commit()
            return ident
        except Exception as e:
            record_failure("create", e)
            return None

    def get(self, identity_id: str) -> AgentIdentity | None:
        try:
            if not identity_id or (self._db is None and self._port is None):
                return None
            if self._port is not None:
                row = self._port.tables["identities"].get(identity_id)
                return self._row_to_identity(row) if row else None
            with self._lock:
                row = self._db.execute(
                    "SELECT * FROM identities WHERE id = ?", (identity_id,)
                ).fetchone()
            return self._row_to_identity(row) if row else None
        except Exception as e:
            record_failure("get", e)
            return None

    def list(self) -> list[AgentIdentity]:
        try:
            if self._db is None and self._port is None:
                return []
            if self._port is not None:
                rows = self._port.tables["identities"].all()
                rows.sort(key=lambda r: r["created_at"])
                return [self._row_to_identity(r) for r in rows]
            with self._lock:
                rows = self._db.execute(
                    "SELECT * FROM identities ORDER BY created_at"
                ).fetchall()
            return [self._row_to_identity(r) for r in rows]
        except Exception as e:
            record_failure("list", e)
            return []

    def remove(self, identity_id: str) -> bool:
        try:
            if not identity_id or (self._db is None and self._port is None):
                return False
            with self._lock:
                if self._port is not None:
                    return self._port.tables["identities"].delete(identity_id)
                cur = self._db.execute(
                    "DELETE FROM identities WHERE id = ?", (identity_id,))
                self._db.commit()
                return cur.rowcount > 0
        except Exception as e:
            record_failure("remove", e)
            return False

    @staticmethod
    def _row_to_identity(row: Any) -> AgentIdentity:
        return AgentIdentity(
            id=row["id"], name=row["name"], email=row["email"],
            handle=row["handle"], purpose=row["purpose"],
            created_at=row["created_at"],
        )
