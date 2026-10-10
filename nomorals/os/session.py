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
    ended_at        REAL,
    name            TEXT NOT NULL DEFAULT '',
    attach_count    INTEGER NOT NULL DEFAULT 0,
    last_activity_at REAL
);
CREATE INDEX IF NOT EXISTS idx_os_sessions_active
    ON os_sessions(ended_at);
CREATE INDEX IF NOT EXISTS idx_os_sessions_project
    ON os_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_os_sessions_conversation
    ON os_sessions(conversation_id);
"""


def _ensure_session_columns(db: Any) -> None:
    """ALTER older os_sessions tables up to the current schema."""
    try:
        cols = {row["name"] for row in db.query("PRAGMA table_info(os_sessions)")}
    except Exception:  # noqa: BLE001 — table may not exist yet
        return
    for column, ddl in (
        ("name", "TEXT NOT NULL DEFAULT ''"),
        ("attach_count", "INTEGER NOT NULL DEFAULT 0"),
        ("last_activity_at", "REAL"),
    ):
        if column not in cols:
            try:
                db.execute(f"ALTER TABLE os_sessions ADD COLUMN {column} {ddl}")
            except Exception:  # noqa: BLE001 — best effort
                pass


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
    #: Human label (renameable); defaults to the conversation id.
    name: str = ""
    #: How many frontends are currently attached (tmux-style presence).
    attach_count: int = 0
    #: Wall-clock of the last inbound user message (vs updated_at which
    #: also moves on system touches).
    last_activity_at: float = field(default_factory=time.time)

    @property
    def active(self) -> bool:
        return self.ended_at is None

    @property
    def attached(self) -> bool:
        return self.attach_count > 0

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last user activity."""
        return max(0.0, time.time() - (self.last_activity_at or self.created_at))

    @property
    def display_name(self) -> str:
        return self.name or self.conversation_id or self.id

    def touch(self, *, activity: bool = False) -> None:
        now = time.time()
        self.updated_at = now
        if activity:
            self.last_activity_at = now

    def attach(self) -> int:
        """A frontend attached (tmux-style). Returns the new count."""
        self.attach_count += 1
        self.touch()
        return self.attach_count

    def detach(self) -> int:
        """A frontend detached. Returns the new count (floors at 0)."""
        self.attach_count = max(0, self.attach_count - 1)
        self.touch()
        return self.attach_count

    def rename(self, name: str) -> None:
        self.name = str(name or "")
        self.touch()

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
            "name": self.name,
            "attach_count": self.attach_count,
            "last_activity_at": self.last_activity_at,
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
            name=str(data.get("name", "") or ""),
            attach_count=int(data.get("attach_count", 0) or 0),
            last_activity_at=float(data.get("last_activity_at")
                                   or data.get("updated_at", time.time())),
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
            name=str(row.get("name", "") or ""),
            attach_count=int(row.get("attach_count", 0) or 0),
            last_activity_at=float(row.get("last_activity_at")
                                   or row.get("updated_at", time.time())),
        )


class SessionStore:
    """Create / fetch / list / end OS sessions, persisted in SQLite."""

    def __init__(self, db: Database | str | Path | None = None) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_OS_SESSIONS_DDL)
        _ensure_session_columns(self.db)
        self._lock = threading.RLock()

    # ── CRUD ─────────────────────────────────────────────────────────────
    def create(self, *, frontend: str = "cli", principal: str = "owner",
               project_id: str = "", conversation_id: str = "",
               artifact_scope: list[str] | None = None,
               state: dict[str, Any] | None = None,
               name: str = "") -> Session:
        session = Session(
            id=new_short_id("sess_"),
            principal=principal or "owner",
            frontend=frontend or "cli",
            project_id=project_id or "",
            conversation_id=conversation_id or "",
            artifact_scope=list(artifact_scope or []),
            state=dict(state or {}),
            name=name or "",
        )
        with self._lock:
            self.db.execute(
                "INSERT INTO os_sessions (id, principal, frontend, project_id,"
                " conversation_id, artifact_scope, state, created_at,"
                " updated_at, ended_at, name, attach_count, last_activity_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?)",
                (session.id, session.principal, session.frontend,
                 session.project_id, session.conversation_id,
                 json.dumps(session.artifact_scope),
                 json.dumps(session.state, default=str),
                 session.created_at, session.updated_at,
                 session.name, session.last_activity_at),
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
                " state = ?, updated_at = ?, ended_at = ?, name = ?,"
                " attach_count = ?, last_activity_at = ? WHERE id = ?",
                (session.principal, session.frontend, session.project_id,
                 session.conversation_id,
                 json.dumps(session.artifact_scope),
                 json.dumps(session.state, default=str),
                 session.updated_at, session.ended_at, session.name,
                 session.attach_count, session.last_activity_at, session.id),
            )

    # ── presence / lifecycle extras ──────────────────────────────────────
    def rename(self, session_id: str, name: str) -> bool:
        """Give a session a human label. False when unknown."""
        session = self.get(session_id)
        if session is None:
            return False
        session.rename(name)
        self.update(session)
        return True

    def activity(self, session_id: str) -> bool:
        """Record user activity on a session (resets idle time)."""
        session = self.get(session_id)
        if session is None:
            return False
        session.touch(activity=True)
        self.update(session)
        return True

    def attach(self, session_id: str) -> int | None:
        """Mark a frontend attached; returns the attach count (None=unknown)."""
        session = self.get(session_id)
        if session is None:
            return None
        count = session.attach()
        self.update(session)
        return count

    def detach(self, session_id: str) -> int | None:
        """Mark a frontend detached; returns the attach count (None=unknown)."""
        session = self.get(session_id)
        if session is None:
            return None
        count = session.detach()
        self.update(session)
        return count

    def purge_expired(self, max_idle_s: float,
                      *, only_detached: bool = True) -> list[str]:
        """End sessions idle longer than ``max_idle_s``.

        Returns the ended session ids.  With ``only_detached=True`` (the
        default) sessions with a frontend still attached are spared even
        when idle — a quiet but present user is not reaped.
        """
        cutoff = time.time() - max(0.0, float(max_idle_s))
        victims = [s for s in self.list_active()
                   if (s.last_activity_at or s.created_at) < cutoff
                   and not (only_detached and s.attached)]
        ended = [s.id for s in victims if self.end(s.id)]
        if ended:
            _log.info("purged %d expired os sessions", len(ended))
        return ended

    def counts_by_frontend(self) -> dict[str, int]:
        """Active session counts per frontend."""
        counts: dict[str, int] = {}
        for session in self.list_active():
            counts[session.frontend] = counts.get(session.frontend, 0) + 1
        return counts

    def render(self) -> str:
        """Plain-text session table (tmux-ls style)."""
        sessions = self.list_active()
        lines = [f"sessions ({len(sessions)} active)"]
        for s in sessions:
            idle = s.idle_seconds
            if idle < 60:
                idle_s = f"{idle:.0f}s"
            elif idle < 3600:
                idle_s = f"{idle / 60:.0f}m"
            else:
                idle_s = f"{idle / 3600:.1f}h"
            presence = f"●{s.attach_count}" if s.attached else "○"
            lines.append(
                f"  {presence} {s.display_name} [{s.frontend}]"
                f" {s.principal} — idle {idle_s}")
        return "\n".join(lines)

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
