"""The autonomy ledger: one durable journal for every autonomous subsystem.

The user's standing demand is that the system works as a whole without
manual tripping — which means "what ran, why, what it cost, what it
learned" must be answerable in one place instead of five silos
(cognition_log, proactive_log, trigger_history, mission checkpoints,
scheduler last_result strings).

Every writer here is best-effort: a broken ledger must never break the
system it observes.  All failures are swallowed into the module logger.

Systems: ``scheduler`` | ``mission`` | ``trigger`` | ``cognition`` |
``pulse`` | ``autonomy`` (partner proactive) | ``watcher``.

Kinds are free-form but conventional: ``run``, ``step``, ``tick``,
``fire``, ``terminal``, ``deferred``, ``skipped``, ``escalated``.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = ["AutonomyLedger", "record_ledger"]

_log = get_logger(__name__)

_LEDGER_SYSTEMS = frozenset(
    {"scheduler", "mission", "trigger", "cognition", "pulse",
     "autonomy", "watcher", "other"}
)


class AutonomyLedger:
    """Durable cross-system autonomy journal (``autonomy_ledger`` table).

    Created by migration 85; the table is created idempotently here too
    so exotic contexts (tests with hand-built tables) still work.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            with self.db.transaction():
                self.db.execute(
                    """CREATE TABLE IF NOT EXISTS autonomy_ledger (
                        id TEXT PRIMARY KEY,
                        ts REAL NOT NULL,
                        system TEXT NOT NULL DEFAULT '',
                        kind TEXT NOT NULL DEFAULT '',
                        ref_id TEXT NOT NULL DEFAULT '',
                        summary TEXT NOT NULL DEFAULT '',
                        cost_seconds REAL NOT NULL DEFAULT 0,
                        cost_tokens INTEGER NOT NULL DEFAULT 0,
                        ok INTEGER NOT NULL DEFAULT 1,
                        learned TEXT NOT NULL DEFAULT '',
                        metadata TEXT NOT NULL DEFAULT '{}'
                    )"""
                )
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_ts "
                    "ON autonomy_ledger(ts)"
                )
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_system "
                    "ON autonomy_ledger(system, ts)"
                )
        except Exception:  # noqa: BLE001 - ledger must never break boot
            _log.debug("autonomy_ledger table ensure failed", exc_info=True)

    # ── writes ───────────────────────────────────────────────────────────

    def record(
        self,
        system: str,
        kind: str,
        ref_id: str = "",
        summary: str = "",
        *,
        cost_seconds: float = 0.0,
        cost_tokens: int = 0,
        ok: bool = True,
        learned: str = "",
        metadata: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> str:
        """Append one ledger entry. Never raises — returns the entry id
        (empty string when the write failed)."""
        entry_id = new_id("ledger")
        system = (system or "other").strip().lower()
        if system not in _LEDGER_SYSTEMS:
            system = "other"
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO autonomy_ledger
                       (id, ts, system, kind, ref_id, summary, cost_seconds,
                        cost_tokens, ok, learned, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry_id,
                        ts if ts is not None else time.time(),
                        system,
                        str(kind or "")[:64],
                        str(ref_id or "")[:128],
                        str(summary or "")[:2000],
                        max(0.0, float(cost_seconds or 0.0)),
                        max(0, int(cost_tokens or 0)),
                        1 if ok else 0,
                        str(learned or "")[:2000],
                        json.dumps(metadata or {}, default=str)[:4000],
                    ),
                )
            return entry_id
        except Exception:  # noqa: BLE001 - the ledger is an observer
            _log.debug("autonomy_ledger record failed", exc_info=True)
            return ""

    # ── reads ────────────────────────────────────────────────────────────

    def recent(
        self,
        *,
        limit: int = 50,
        system: str | None = None,
        kind: str | None = None,
        ok: bool | None = None,
        since_hours: float | None = None,
    ) -> list[dict[str, Any]]:
        """Newest-first entries, filtered. Never raises (empty on error)."""
        clauses = ["1 = 1"]
        params: list[Any] = []
        if system:
            clauses.append("system = ?")
            params.append(system.strip().lower())
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if ok is not None:
            clauses.append("ok = ?")
            params.append(1 if ok else 0)
        if since_hours is not None:
            clauses.append("ts >= ?")
            params.append(time.time() - max(0.0, float(since_hours)) * 3600)
        try:
            rows = self.db.query(
                "SELECT * FROM autonomy_ledger WHERE " + " AND ".join(clauses)
                + " ORDER BY ts DESC LIMIT ?",
                (*params, max(1, int(limit))),
            )
        except Exception:  # noqa: BLE001
            _log.debug("autonomy_ledger recent failed", exc_info=True)
            return []
        out = []
        for row in rows:
            out.append({
                "id": row["id"],
                "ts": row["ts"],
                "system": row["system"],
                "kind": row["kind"],
                "ref_id": row["ref_id"],
                "summary": row["summary"],
                "cost_seconds": row["cost_seconds"],
                "cost_tokens": row["cost_tokens"],
                "ok": bool(row["ok"]),
                "learned": row["learned"],
                "metadata": _json(row.get("metadata")),
            })
        return out

    def summary(self, *, window_hours: float = 24.0) -> dict[str, Any]:
        """Per-system rollup over the window: counts, failures, cost.

        The owner-visible answer to "what has the system been doing on its
        own". Never raises.
        """
        cutoff = time.time() - max(0.0, float(window_hours)) * 3600
        out: dict[str, Any] = {
            "window_hours": window_hours,
            "systems": {},
            "totals": {"runs": 0, "failures": 0,
                       "cost_seconds": 0.0, "cost_tokens": 0},
        }
        try:
            rows = self.db.query(
                """SELECT system,
                          COUNT(*) AS runs,
                          SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failures,
                          COALESCE(SUM(cost_seconds), 0) AS cost_seconds,
                          COALESCE(SUM(cost_tokens), 0) AS cost_tokens,
                          MAX(ts) AS last_ts
                   FROM autonomy_ledger WHERE ts >= ?
                   GROUP BY system""",
                (cutoff,),
            )
        except Exception:  # noqa: BLE001
            _log.debug("autonomy_ledger summary failed", exc_info=True)
            return out
        for row in rows:
            name = row["system"] or "other"
            runs = int(row["runs"] or 0)
            failures = int(row["failures"] or 0)
            cost_s = float(row["cost_seconds"] or 0.0)
            cost_t = int(row["cost_tokens"] or 0)
            out["systems"][name] = {
                "runs": runs,
                "failures": failures,
                "cost_seconds": round(cost_s, 1),
                "cost_tokens": cost_t,
                "last_ts": row["last_ts"],
            }
            out["totals"]["runs"] += runs
            out["totals"]["failures"] += failures
            out["totals"]["cost_seconds"] = round(
                out["totals"]["cost_seconds"] + cost_s, 1)
            out["totals"]["cost_tokens"] += cost_t
        return out

    def purge(self, *, older_than_days: float = 90.0) -> int:
        """Drop entries older than the window. Returns rows deleted."""
        try:
            cutoff = time.time() - max(1.0, float(older_than_days)) * 86400
            with self.db.transaction():
                cur = self.db.execute(
                    "DELETE FROM autonomy_ledger WHERE ts < ?", (cutoff,))
            return cur.rowcount if cur is not None else 0
        except Exception:  # noqa: BLE001
            _log.debug("autonomy_ledger purge failed", exc_info=True)
            return 0


def _json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def record_ledger(
    db_or_context: Any,
    system: str,
    kind: str,
    ref_id: str = "",
    summary: str = "",
    **kwargs: Any,
) -> str:
    """One-call ledger write from any subsystem.

    Accepts a ``Database`` or a context carrying ``.db``.  Never raises.
    """
    try:
        db = getattr(db_or_context, "db", None)
        if db is None:
            db = db_or_context
        return AutonomyLedger(db).record(
            system, kind, ref_id, summary, **kwargs)
    except Exception:  # noqa: BLE001 - ledger is always best-effort
        _log.debug("record_ledger failed", exc_info=True)
        return ""
