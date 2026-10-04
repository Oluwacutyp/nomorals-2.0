"""SQLite-backed store of per-execution outcome trajectories.

The broker (and any caller) records one row per attempt at a task and
later asks:

- :meth:`TrajectoryStore.success_rate` — recency-weighted success
  probability for a (task_kind, capability, model/skill) slice.
- :meth:`TrajectoryStore.rank` — candidates ordered best-first by that
  probability (ties broken deterministically by name).
- :meth:`TrajectoryStore.failure_clusters` — failures grouped by a
  normalized error signature, so recurring breakage is visible.

Recency uses exponential decay with a configurable half-life
(default 14 days): outcomes from yesterday count far more than
outcomes from two months ago.  With no data the prior is 0.5 —
unknown is not failure.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from pathlib import Path

from ..core.ids import new_short_id
from ..storage.db import Database

_log = logging.getLogger(__name__)

__all__ = ["TrajectoryStore", "normalize_error", "cluster_key_for"]

_DEFAULT_HALF_LIFE_DAYS = 14.0
_EMPTY_PRIOR = 0.5
_SECONDS_PER_DAY = 86_400.0
#: Rows older than this are dropped by the retention prune.  With the
#: default 14-day half-life a 90-day-old outcome carries ~1% weight, so
#: pruning it does not move scoring — it just bounds table growth.
DEFAULT_RETENTION_DAYS = 90.0
#: At most one opportunistic prune per record() call chain per hour.
_PRUNE_INTERVAL_S = 3600.0

_TRAJECTORIES_DDL = """
CREATE TABLE IF NOT EXISTS cog_trajectories (
    id          TEXT PRIMARY KEY,
    task_kind   TEXT NOT NULL DEFAULT '',
    capability  TEXT NOT NULL DEFAULT '',
    model_id    TEXT NOT NULL DEFAULT '',
    skill_id    TEXT NOT NULL DEFAULT '',
    tool        TEXT NOT NULL DEFAULT '',
    success     INTEGER NOT NULL DEFAULT 0,
    latency_s   REAL NOT NULL DEFAULT 0.0,
    cost        REAL NOT NULL DEFAULT 0.0,
    error       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cog_traj_kind_cap
    ON cog_trajectories(task_kind, capability, created_at);
CREATE INDEX IF NOT EXISTS idx_cog_traj_model
    ON cog_trajectories(model_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cog_traj_skill
    ON cog_trajectories(skill_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cog_traj_created
    ON cog_trajectories(created_at);
"""

# ── error normalization ──────────────────────────────────────────────────
# Goal: identical failures cluster together while volatile text (memory
# addresses, timestamps, uuid-like request ids) does not split clusters.

_RE_HEX_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b")
_RE_ISO_TS = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
)
_RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_RE_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b")
_RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_RE_HEX_BLOB = re.compile(r"\b(?:[0-9a-f]{16,}|[0-9A-F]{16,})\b")
_RE_WS = re.compile(r"\s+")

_MAX_SIGNATURE_LEN = 220


def normalize_error(error: str) -> str:
    """Normalize an error message into a stable cluster signature.

    Takes the first non-empty line (exception class + message), strips
    volatile text (hex addresses, timestamps, uuid/request ids) and
    collapses whitespace.  Distinct exception classes and messages keep
    distinct signatures; two runs of the same bug collapse to one.
    """
    if not error:
        return "(no error message)"
    first_line = ""
    for line in str(error).splitlines():
        line = line.strip()
        if line:
            first_line = line
            break
    if not first_line:
        return "(no error message)"
    sig = _RE_HEX_ADDR.sub("<addr>", first_line)
    sig = _RE_ISO_TS.sub("<ts>", sig)
    sig = _RE_DATE.sub("<date>", sig)
    sig = _RE_TIME.sub("<ts>", sig)
    sig = _RE_UUID.sub("<id>", sig)
    sig = _RE_HEX_BLOB.sub("<id>", sig)
    sig = _RE_WS.sub(" ", sig).strip()
    if len(sig) > _MAX_SIGNATURE_LEN:
        sig = sig[: _MAX_SIGNATURE_LEN - 1] + "…"
    return sig or "(no error message)"


def cluster_key_for(task_kind: str, error_signature: str) -> str:
    """Deterministic key identifying one failure cluster."""
    return f"{task_kind}::{error_signature}"


def _decay_weight(age_s: float, half_life_s: float) -> float:
    if age_s <= 0:
        return 1.0
    if half_life_s <= 0:
        return 1.0 if age_s == 0 else 0.0
    return math.pow(2.0, -(age_s / half_life_s))


class TrajectoryStore:
    """Record and score execution outcomes, persisted in SQLite."""

    def __init__(self, db: Database | str | Path | None = None) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_TRAJECTORIES_DDL)
        self._lock = threading.RLock()
        # -inf so the first record() always runs the retention prune,
        # regardless of process uptime (monotonic clocks start near 0).
        self._last_prune = float("-inf")

    # ── recording ────────────────────────────────────────────────────────
    def record(
        self,
        *,
        task_kind: str,
        capability: str,
        model_id: str = "",
        skill_id: str = "",
        tool: str = "",
        success: bool,
        latency_s: float = 0.0,
        cost: float = 0.0,
        error: str = "",
    ) -> None:
        """Record one execution outcome."""
        self._maybe_prune()
        self._add(
            task_kind=task_kind,
            capability=capability,
            model_id=model_id,
            skill_id=skill_id,
            tool=tool,
            success=success,
            latency_s=latency_s,
            cost=cost,
            error=error,
            created_at=time.time(),
        )

    def _add(
        self,
        *,
        task_kind: str,
        capability: str,
        model_id: str = "",
        skill_id: str = "",
        tool: str = "",
        success: bool,
        latency_s: float = 0.0,
        cost: float = 0.0,
        error: str = "",
        created_at: float | None = None,
    ) -> str:
        """Internal insert honoring an explicit timestamp (used by tests)."""
        row_id = new_short_id("traj_")
        with self._lock:
            self.db.execute(
                "INSERT INTO cog_trajectories (id, task_kind, capability,"
                " model_id, skill_id, tool, success, latency_s, cost,"
                " error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row_id,
                    task_kind or "",
                    capability or "",
                    model_id or "",
                    skill_id or "",
                    tool or "",
                    1 if success else 0,
                    float(latency_s),
                    float(cost),
                    error or "",
                    float(created_at if created_at is not None else time.time()),
                ),
            )
        return row_id

    # ── retention ────────────────────────────────────────────────────────
    def _maybe_prune(self) -> None:
        """Opportunistic retention prune — at most once an hour, never raises."""
        now_m = time.monotonic()
        if now_m - self._last_prune < _PRUNE_INTERVAL_S:
            return
        self._last_prune = now_m
        try:
            dropped = self.prune()
        except Exception:  # noqa: BLE001 - pruning is hygiene, not load-bearing
            _log.debug("trajectory auto-prune failed", exc_info=True)
        else:
            if dropped:
                _log.info("trajectory prune dropped %d stale rows", dropped)

    def prune(self, older_than_days: float = DEFAULT_RETENTION_DAYS) -> int:
        """Delete trajectories older than ``older_than_days``.

        Returns the number of rows deleted.  The store is otherwise
        append-only, so without this the table — and the full-scan in
        :meth:`failure_clusters` — grows without bound on a long-running
        bot.  Rows past the retention window carry ~zero scoring weight
        (14-day half-life → a 90-day-old outcome weighs ~1%), so scoring
        is unaffected; failure-cluster *counts* reflect the window.
        """
        cutoff = time.time() - float(older_than_days) * _SECONDS_PER_DAY
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM cog_trajectories WHERE created_at < ?",
                (cutoff,),
            )
            return int(cur.rowcount or 0)

    # ── scoring ──────────────────────────────────────────────────────────
    def _slice(self, task_kind: str, capability: str, model_id: str,
               skill_id: str) -> list[dict]:
        """Rows for a slice; empty model_id/skill_id act as wildcards."""
        return self.db.query(
            "SELECT success, created_at FROM cog_trajectories"
            " WHERE task_kind = ? AND capability = ?"
            " AND (model_id = ? OR ? = '')"
            " AND (skill_id = ? OR ? = '')",
            (
                task_kind or "",
                capability or "",
                model_id or "",
                model_id or "",
                skill_id or "",
                skill_id or "",
            ),
        )

    @staticmethod
    def _weighted_rate(rows: list[dict], half_life_days: float,
                       now: float) -> float:
        half_life_s = float(half_life_days) * _SECONDS_PER_DAY
        num = 0.0
        den = 0.0
        for row in rows:
            w = _decay_weight(now - float(row["created_at"]), half_life_s)
            if w <= 0.0:
                continue
            num += w * float(row["success"])
            den += w
        return (num / den) if den > 0.0 else _EMPTY_PRIOR

    def success_rate(
        self,
        task_kind: str,
        capability: str,
        model_id: str = "",
        skill_id: str = "",
        half_life_days: float = 14.0,
    ) -> float:
        """Recency-weighted success probability for a slice.

        Exponential decay with ``half_life_days`` half-life; recent
        outcomes dominate.  Returns the 0.5 prior when no data exists.
        """
        rows = self._slice(task_kind, capability, model_id, skill_id)
        if not rows:
            return _EMPTY_PRIOR
        return self._weighted_rate(rows, half_life_days, time.time())

    def rank(
        self,
        task_kind: str,
        capability: str,
        candidates: list[str],
        kind: str = "model",
    ) -> list[str]:
        """Order candidates best-first by recency-weighted success rate.

        ``kind`` is ``"model"`` (candidates are model ids) or
        ``"skill"`` (candidates are skill ids).  Unknown candidates get
        the 0.5 prior and sort after any measured candidate above 0.5.
        Ties break deterministically by name so results are stable.
        """
        scored: list[tuple[float, str]] = []
        for cand in candidates:
            if kind == "skill":
                rate = self.success_rate(
                    task_kind, capability, skill_id=cand
                )
            else:
                rate = self.success_rate(
                    task_kind, capability, model_id=cand
                )
            scored.append((rate, cand))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [cand for _, cand in scored]

    # ── failure clustering ───────────────────────────────────────────────
    def failure_clusters(
        self, task_kind: str | None = None, limit: int = 10
    ) -> list[dict]:
        """Group failures by normalized error signature, most-frequent first.

        Each dict: ``task_kind``, ``error_signature``, ``count``,
        ``last_seen`` (unix time), ``example_ids`` (up to 5 trajectory
        ids).  ``task_kind=None`` aggregates across all task kinds.
        """
        params: list = []
        where = "success = 0"
        if task_kind is not None:
            where += " AND task_kind = ?"
            params.append(task_kind)
        rows = self.db.query(
            "SELECT id, task_kind, error, created_at FROM cog_trajectories"
            f" WHERE {where}",
            tuple(params),
        )
        clusters: dict[tuple[str, str], dict] = {}
        for row in rows:
            signature = normalize_error(row.get("error") or "")
            key = (row["task_kind"] or "", signature)
            cluster = clusters.get(key)
            if cluster is None:
                cluster = {
                    "task_kind": row["task_kind"] or "",
                    "error_signature": signature,
                    "count": 0,
                    "last_seen": 0.0,
                    "example_ids": [],
                }
                clusters[key] = cluster
            cluster["count"] += 1
            created = float(row["created_at"])
            if created > cluster["last_seen"]:
                cluster["last_seen"] = created
            if len(cluster["example_ids"]) < 5:
                cluster["example_ids"].append(row["id"])
        ordered = sorted(
            clusters.values(),
            key=lambda c: (-c["count"], -c["last_seen"]),
        )
        return ordered[: max(0, int(limit))]
