"""Human-in-the-loop checkpoints for account and signup flows.

When Devon's automation reaches a step only a human can do — a CAPTCHA, an
email-verification click, a 2FA code, accepting terms — it does NOT try to
bypass it. It persists a checkpoint, pings the owner through the owner-only
delivery channel, pauses, and resumes after the owner personally completes
the step. A human solving the check IS the check being satisfied; nothing
is bypassed.

HARD BOUNDARY: Devon never auto-solves CAPTCHAs — no solver services, no
AI-based bypass, no verification dodging. The human checkpoint is the only
path through human verification. (Structurally enforced: the connectors
package contains no CAPTCHA-solving code, and the test suite asserts it.)

Design rules for account flows:
* one account per service — a second account flow refuses while a
  credential exists;
* the owner's own identity — flows never invent fake identities;
* credentials are handed to the owner and vault-saved on completion.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..core.errors import NotFound
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .base import ConnectorError

__all__ = [
    "CheckpointKind",
    "CheckpointState",
    "CheckpointStore",
    "HumanCheckpoint",
    "HumanCheckpointPending",
    "request_human_action",
]

_log = get_logger(__name__)


class CheckpointKind(StrEnum):
    """What kind of human-only step this checkpoint waits on."""

    CAPTCHA = "captcha"
    EMAIL_VERIFY = "email_verify"
    PHONE_2FA = "phone_2fa"
    TOS_ACCEPT = "tos_accept"
    MANUAL_STEP = "manual_step"  # any other human-only step


class CheckpointState(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class HumanCheckpoint:
    """A paused-for-human step. ``resume_state`` carries the flow's
    pick-up state so a later process can continue after the owner acts."""

    id: str
    connector_id: str
    kind: CheckpointKind
    title: str
    instructions: str
    state: CheckpointState = CheckpointState.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0
    resolved_at: float | None = None
    resume_state: dict[str, Any] = field(default_factory=dict)
    result_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "connector_id": self.connector_id,
            "kind": self.kind.value,
            "title": self.title,
            "instructions": self.instructions,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "resume_state": dict(self.resume_state),
            "result_note": self.result_note,
        }


class HumanCheckpointPending(ConnectorError):
    """Raised to pause a flow at a human checkpoint (non-interactive mode).

    A ConnectorError so provision/connect flows surface it as a clean,
    actionable message. Carries the checkpoint id so the operator (or a
    later run) can resume with
    ``nm connectors checkpoint resolve --id <id>``.
    """

    def __init__(self, checkpoint: HumanCheckpoint) -> None:
        self.checkpoint = checkpoint
        super().__init__(
            f"paused for human action [{checkpoint.id}]: {checkpoint.title} — "
            f"resolve with `nm connectors checkpoint resolve --id {checkpoint.id}` "
            "after completing the step"
        )


class CheckpointStore:
    """Persisted checkpoints, backed by the app database."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS connector_checkpoints (
                    id TEXT PRIMARY KEY,
                    connector_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    resolved_at REAL,
                    resume_state TEXT NOT NULL DEFAULT '{}',
                    result_note TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self.db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_state
                ON connector_checkpoints(state)
                """
            )

    def create(
        self,
        connector_id: str,
        kind: CheckpointKind,
        title: str,
        instructions: str,
        *,
        resume_state: dict[str, Any] | None = None,
        ttl_seconds: float = 86400.0,
    ) -> HumanCheckpoint:
        now = time.time()
        cp = HumanCheckpoint(
            id=new_id("chk"),
            connector_id=connector_id,
            kind=kind,
            title=title,
            instructions=instructions,
            state=CheckpointState.PENDING,
            created_at=now,
            updated_at=now,
            resume_state=dict(resume_state or {}),
        )
        # ttl rides in resume_state metadata; expiry is evaluated on read.
        cp.resume_state.setdefault("_expires_at", now + ttl_seconds)
        with self.db.transaction():
            self.db.execute(
                """
                INSERT INTO connector_checkpoints
                (id, connector_id, kind, title, instructions, state,
                 created_at, updated_at, resolved_at, resume_state, result_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cp.id, cp.connector_id, cp.kind.value, cp.title,
                    cp.instructions, cp.state.value, cp.created_at,
                    cp.updated_at, cp.resolved_at,
                    json.dumps(cp.resume_state), cp.result_note,
                ),
            )
        _log.info("checkpoint %s created for %s: %s", cp.id, connector_id, title)
        return cp

    def get(self, checkpoint_id: str) -> HumanCheckpoint:
        self.expire_stale()
        row = self.db.query_one(
            "SELECT * FROM connector_checkpoints WHERE id = ?",
            (checkpoint_id,),
        )
        if not row:
            raise NotFound(f"no checkpoint {checkpoint_id!r}")
        return self._from_row(row)

    def list_pending(
        self, connector_id: str | None = None
    ) -> list[HumanCheckpoint]:
        self.expire_stale()
        if connector_id:
            rows = self.db.query(
                "SELECT * FROM connector_checkpoints WHERE state = 'pending' "
                "AND connector_id = ? ORDER BY created_at",
                (connector_id,),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM connector_checkpoints WHERE state = 'pending' "
                "ORDER BY created_at"
            )
        return [self._from_row(r) for r in rows]

    def resolve(
        self, checkpoint_id: str, note: str = ""
    ) -> HumanCheckpoint:
        cp = self.get(checkpoint_id)
        if cp.state != CheckpointState.PENDING:
            raise ConnectorError(
                f"checkpoint {checkpoint_id} is {cp.state.value}, "
                "not pending — nothing to resolve"
            )
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE connector_checkpoints SET state = 'resolved', "
                "resolved_at = ?, updated_at = ?, result_note = ? "
                "WHERE id = ?",
                (now, now, note, checkpoint_id),
            )
        cp.state = CheckpointState.RESOLVED
        cp.resolved_at = now
        cp.updated_at = now
        cp.result_note = note
        _log.info("checkpoint %s resolved", checkpoint_id)
        return cp

    def cancel(self, checkpoint_id: str, note: str = "") -> HumanCheckpoint:
        cp = self.get(checkpoint_id)
        if cp.state != CheckpointState.PENDING:
            raise ConnectorError(
                f"checkpoint {checkpoint_id} is {cp.state.value}, "
                "not pending — nothing to cancel"
            )
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE connector_checkpoints SET state = 'cancelled', "
                "updated_at = ?, result_note = ? WHERE id = ?",
                (now, note, checkpoint_id),
            )
        cp.state = CheckpointState.CANCELLED
        cp.updated_at = now
        cp.result_note = note
        return cp

    def expire_stale(self, now: float | None = None) -> int:
        """Mark pending checkpoints past their ttl as expired. Returns count."""
        now = now or time.time()
        rows = self.db.query(
            "SELECT id, resume_state FROM connector_checkpoints "
            "WHERE state = 'pending'"
        )
        expired = 0
        for row in rows:
            try:
                ttl_at = float(
                    json.loads(row["resume_state"] or "{}").get(
                        "_expires_at", 0
                    )
                )
            except (ValueError, TypeError):
                ttl_at = 0
            if ttl_at and ttl_at < now:
                with self.db.transaction():
                    self.db.execute(
                        "UPDATE connector_checkpoints SET state = 'expired', "
                        "updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                expired += 1
        return expired

    @staticmethod
    def _from_row(row: dict[str, Any]) -> HumanCheckpoint:
        try:
            resume_state = json.loads(row.get("resume_state") or "{}")
        except (ValueError, TypeError):
            resume_state = {}
        return HumanCheckpoint(
            id=row["id"],
            connector_id=row["connector_id"],
            kind=CheckpointKind(row["kind"]),
            title=row["title"],
            instructions=row.get("instructions") or "",
            state=CheckpointState(row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resolved_at=row.get("resolved_at"),
            resume_state=resume_state,
            result_note=row.get("result_note") or "",
        )


def request_human_action(
    connector_id: str,
    kind: CheckpointKind,
    title: str,
    instructions: str,
    *,
    db: Database,
    context: Any = None,
    resume_state: dict[str, Any] | None = None,
    ttl_seconds: float = 86400.0,
) -> HumanCheckpoint:
    """Pause for a human-only step: persist, ping the owner, then wait or stop.

    * The checkpoint is persisted first — a later process can always resume.
    * The owner is pinged through the owner-only delivery channel
      (``critical=True``: like an alarm, it bypasses quiet hours).
    * Interactive TTY: the instructions print and we wait for Enter — the
      owner is present, so Enter means "done", and we resolve.
    * Otherwise: raise :class:`HumanCheckpointPending` so the flow pauses
      cleanly instead of hanging. Resume with
      ``nm connectors checkpoint resolve --id <id>``.
    """
    store = CheckpointStore(db)
    cp = store.create(
        connector_id, kind, title, instructions,
        resume_state=resume_state, ttl_seconds=ttl_seconds,
    )
    if context is not None:
        from ..agents.notifier import notify

        notify(
            context,
            "checkpoint",
            f"Action needed: {title}",
            f"{instructions}\n\nResolve with: "
            f"nm connectors checkpoint resolve --id {cp.id}",
            critical=True,
        )
    if sys.stdin.isatty():
        print(f"\n{title}\n{instructions}\n")
        try:
            input("Press Enter when you have completed this step...")
        except EOFError:
            raise HumanCheckpointPending(cp) from None
        return store.resolve(cp.id, note="confirmed interactively")
    raise HumanCheckpointPending(cp)
