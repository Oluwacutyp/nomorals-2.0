"""First-class OS sessions.

An OS :class:`Session` is the control plane's view of "who is talking to
Devon, through what, about what": a principal (``owner``/``guest``/...), a
frontend (``cli`` | ``tui`` | ``telegram`` | ``whatsapp`` | ``api`` |
``app`` | ``voice``), the project it is scoped to, the conversation it
continues, the artifact ids it may touch, and an opaque ``state`` dict for
frontend-specific baggage.

:class:`SessionStore` persists sessions in a SQLite ``os_sessions`` table
(``CREATE TABLE IF NOT EXISTS``) and publishes ``session.created`` /
``session.ended`` on :data:`nomorals.core.events.global_bus`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.events import global_bus
from ..core.ids import new_short_id
from ..storage.db import Database

_log = logging.getLogger(__name__)

__all__ = ["Session", "SessionStore", "FRONTENDS"]

FRONTENDS = ("cli", "tui", "telegram", "whatsapp", "api", "app", "voice")

_OS_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS os_sessions (
    id              TEXT PRIMARY KEY,
    principal       TEXT NOT NULL DEFAULT 'owner',
    frontend        TEXT NOT NULL DEFAULT 'cli',
    project_id      TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    artifact_scope  TEXT NOT NULL DEFAULT '[]',
    state           TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    ended_at        REAL
);
CREATE INDEX IF NOT EXISTS idx_os_sessions_active
    ON os_sessions(ended_at);
CREATE INDEX IF NOT EXISTS idx_os_sessions_project
    ON os_sessions(project_id);
"""


@dataclass
class Session:
    """One live interaction channel with Devon."""

    id: str
    principal: str = "owner"
    frontend: str = "cli"
    project_id: str = ""
    conversation_id: str = ""
    artifact_scope: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "principal": self.principal,
            "frontend": self.frontend,
            "project_id": self.project_id,
            "conversation_id": self.conversation_id,
            "artifact_scope": list(self.artifact_scope),
            "state": dict(self.state),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            id=str(data["id"]),
            principal=str(data.get("principal", "owner")),
            frontend=str(data.get("frontend", "cli")),
            project_id=str(data.get("project_id", "")),
            conversation_id=str(data.get("conversation_id", "")),
            artifact_scope=list(data.get("artifact_scope", []) or []),
            state=dict(data.get("state", {}) or {}),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            ended_at=data.get("ended_at"),
        )

    @classmethod
    def _from_row(cls, row: dict[str, Any]) -> "Session":
        def _json(text: Any, fallback: Any) -> Any:
            try:
                return json.loads(text) if text else fallback
            except (TypeError, ValueError):
                return fallback

        return cls(
            id=row["id"],
            principal=row.get("principal", "owner") or "owner",
            frontend=row.get("frontend", "cli") or "cli",
            project_id=row.get("project_id", "") or "",
            conversation_id=row.get("conversation_id", "") or "",
            artifact_scope=list(_json(row.get("artifact_scope"), []) or []),
            state=dict(_json(row.get("state"), {}) or {}),
            created_at=float(row.get("created_at", time.time())),
            updated_at=float(row.get("updated_at", time.time())),
            ended_at=row.get("ended_at"),
        )


class SessionStore:
    """Create / fetch / list / end OS sessions, persisted in SQLite."""

    def __init__(self, db: Database | str | Path | None = None) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_OS_SESSIONS_DDL)
        self._lock = threading.RLock()

    # ── CRUD ─────────────────────────────────────────────────────────────
    def create(self, *, frontend: str = "cli", principal: str = "owner",
               project_id: str = "", conversation_id: str = "",
               artifact_scope: list[str] | None = None,
               state: dict[str, Any] | None = None) -> Session:
        session = Session(
            id=new_short_id("sess_"),
            principal=principal or "owner",
            frontend=frontend or "cli",
            project_id=project_id or "",
            conversation_id=conversation_id or "",
            artifact_scope=list(artifact_scope or []),
            state=dict(state or {}),
        )
        with self._lock:
            self.db.execute(
                "INSERT INTO os_sessions (id, principal, frontend, project_id,"
                " conversation_id, artifact_scope, state, created_at,"
                " updated_at, ended_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (session.id, session.principal, session.frontend,
                 session.project_id, session.conversation_id,
                 json.dumps(session.artifact_scope),
                 json.dumps(session.state, default=str),
                 session.created_at, session.updated_at),
            )
        _log.info("os session %s created (frontend=%s principal=%s)",
                  session.id, session.frontend, session.principal)
        try:
            global_bus.publish("session.created", {
                "session_id": session.id,
                "frontend": session.frontend,
                "principal": session.principal,
            }, source="nomorals.os.session")
        except Exception:  # noqa: BLE001 — events must never break persistence
            _log.exception("failed to publish session.created")
        return session

    def get(self, session_id: str) -> Session | None:
        rows = self.db.query("SELECT * FROM os_sessions WHERE id = ?",
                             (session_id,))
        return Session._from_row(rows[0]) if rows else None

    def list_active(self) -> list[Session]:
        rows = self.db.query(
            "SELECT * FROM os_sessions WHERE ended_at IS NULL"
            " ORDER BY created_at")
        return [Session._from_row(r) for r in rows]

    def list_for_project(self, project_id: str) -> list[Session]:
        rows = self.db.query(
            "SELECT * FROM os_sessions WHERE project_id = ?"
            " ORDER BY created_at", (project_id,))
        return [Session._from_row(r) for r in rows]

    def update(self, session: Session) -> None:
        """Persist in-memory changes (state, artifact_scope, ...)."""
        session.touch()
        with self._lock:
            self.db.execute(
                "UPDATE os_sessions SET principal = ?, frontend = ?,"
                " project_id = ?, conversation_id = ?, artifact_scope = ?,"
                " state = ?, updated_at = ?, ended_at = ? WHERE id = ?",
                (session.principal, session.frontend, session.project_id,
                 session.conversation_id,
                 json.dumps(session.artifact_scope),
                 json.dumps(session.state, default=str),
                 session.updated_at, session.ended_at, session.id),
            )

    def end(self, session_id: str) -> bool:
        """Mark a session ended. Returns False when it did not exist."""
        with self._lock:
            cur = self.db.execute(
                "UPDATE os_sessions SET ended_at = ?, updated_at = ?"
                " WHERE id = ? AND ended_at IS NULL",
                (time.time(), time.time(), session_id),
            )
            ended = cur.rowcount > 0
        if ended:
            _log.info("os session %s ended", session_id)
            try:
                global_bus.publish("session.ended",
                                   {"session_id": session_id},
                                   source="nomorals.os.session")
            except Exception:  # noqa: BLE001 — events must never break persistence
                _log.exception("failed to publish session.ended")
        return ended
