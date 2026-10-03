"""Durable trigger storage.

Follows the scheduler's persistence pattern: a ``storage.db.Database``
(SQLite, WAL) with two tables — ``trigger_defs`` for definitions and
``trigger_history`` for every evaluation outcome.  Nothing here is
in-memory-only: a restart loses no definitions and no history.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.db import Database
from .models import OUTCOME_FIRED, Trigger

_log = get_logger(__name__)

#: history rows older than this are purged on each engine tick
HISTORY_TTL_S = 7 * 24 * 3600


class TriggerStore:
    """SQLite-backed store for trigger definitions and fire history."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS trigger_defs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    source TEXT NOT NULL,
                    condition TEXT NOT NULL DEFAULT '{}',
                    action TEXT NOT NULL,
                    action_params TEXT NOT NULL DEFAULT '{}',
                    cooldown_s REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_fired REAL,
                    last_outcome TEXT,
                    fire_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS trigger_history (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id TEXT NOT NULL,
                    at REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '{}',
                    error TEXT
                )
            """)
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trigger_history_id
                ON trigger_history(trigger_id, seq)
            """)

    # ── definitions ──────────────────────────────────────────────────────

    def save(self, trigger: Trigger) -> None:
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO trigger_defs
                   (id, name, enabled, source, condition, action, action_params,
                    cooldown_s, created_at, last_fired, last_outcome, fire_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, enabled=excluded.enabled,
                     source=excluded.source, condition=excluded.condition,
                     action=excluded.action, action_params=excluded.action_params,
                     cooldown_s=excluded.cooldown_s,
                     last_fired=excluded.last_fired,
                     last_outcome=excluded.last_outcome,
                     fire_count=excluded.fire_count""",
                (trigger.id, trigger.name, int(trigger.enabled),
                 trigger.source, json.dumps(trigger.condition),
                 trigger.action, json.dumps(trigger.action_params),
                 trigger.cooldown_s, trigger.created_at, trigger.last_fired,
                 trigger.last_outcome, trigger.fire_count),
            )

    def _row_to_trigger(self, row: dict[str, Any]) -> Trigger:
        return Trigger(
            id=row["id"],
            name=row["name"],
            enabled=bool(row["enabled"]),
            source=row["source"],
            condition=json.loads(row["condition"] or "{}"),
            action=row["action"],
            action_params=json.loads(row["action_params"] or "{}"),
            cooldown_s=float(row["cooldown_s"] or 0.0),
            created_at=float(row["created_at"]),
            last_fired=row["last_fired"],
            last_outcome=row["last_outcome"],
            fire_count=int(row["fire_count"] or 0),
        )

    def get(self, trigger_id: str) -> Trigger | None:
        row = self.db.query_one(
            "SELECT * FROM trigger_defs WHERE id = ?", (trigger_id,))
        return self._row_to_trigger(row) if row else None

    def list(self, *, enabled_only: bool = False,
             source: str | None = None) -> list[Trigger]:
        sql = "SELECT * FROM trigger_defs"
        clauses, params = [], []
        if enabled_only:
            clauses.append("enabled = 1")
        if source:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC"
        return [self._row_to_trigger(r)
                for r in self.db.query(sql, params)]

    def delete(self, trigger_id: str) -> bool:
        with self.db.transaction():
            cur = self.db.execute(
                "DELETE FROM trigger_defs WHERE id = ?", (trigger_id,))
            self.db.execute(
                "DELETE FROM trigger_history WHERE trigger_id = ?",
                (trigger_id,))
        return (cur.rowcount or 0) > 0

    def set_enabled(self, trigger_id: str, enabled: bool) -> bool:
        with self.db.transaction():
            cur = self.db.execute(
                "UPDATE trigger_defs SET enabled = ? WHERE id = ?",
                (int(enabled), trigger_id))
        return (cur.rowcount or 0) > 0

    # ── history ──────────────────────────────────────────────────────────

    def record(self, trigger_id: str, outcome: str,
               detail: dict[str, Any] | None = None,
               error: str | None = None,
               *, fired: bool = False) -> None:
        """Record one evaluation outcome.  ``fired=True`` also bumps the
        definition's ``fire_count``/``last_fired``/``last_outcome``."""
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO trigger_history
                   (trigger_id, at, outcome, detail, error)
                   VALUES (?, ?, ?, ?, ?)""",
                (trigger_id, now, outcome, json.dumps(detail or {}), error),
            )
            if fired:
                self.db.execute(
                    """UPDATE trigger_defs
                       SET fire_count = fire_count + 1, last_fired = ?,
                           last_outcome = ?
                       WHERE id = ?""",
                    (now, outcome, trigger_id),
                )
            else:
                self.db.execute(
                    "UPDATE trigger_defs SET last_outcome = ? WHERE id = ?",
                    (outcome, trigger_id),
                )

    def history(self, trigger_id: str | None = None,
                *, limit: int = 100) -> list[dict[str, Any]]:
        sql = ("SELECT seq, trigger_id, at, outcome, detail, error "
               "FROM trigger_history")
        params: list[Any] = []
        if trigger_id:
            sql += " WHERE trigger_id = ?"
            params.append(trigger_id)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(max(1, limit))
        rows = []
        for r in self.db.query(sql, params):
            rows.append({
                "seq": r["seq"],
                "trigger_id": r["trigger_id"],
                "at": r["at"],
                "outcome": r["outcome"],
                "detail": json.loads(r["detail"] or "{}"),
                "error": r["error"],
            })
        return rows

    def purge_old(self, ttl_s: float = HISTORY_TTL_S) -> int:
        """Delete history rows older than ``ttl_s``; returns rows removed."""
        with self.db.transaction():
            cur = self.db.execute(
                "DELETE FROM trigger_history WHERE at < ?",
                (time.time() - ttl_s,))
        return cur.rowcount or 0

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM trigger_defs")
        return int(row["n"]) if row else 0
