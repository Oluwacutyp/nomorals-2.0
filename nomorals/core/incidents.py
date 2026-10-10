"""Incident journal: the error catcher's memory.

Every failure is recorded with a *structural* signature — (exception type,
subsystem, code location) — never the message text. Two incidents with the
same signature are the same incident wearing different clothes.

Three mined rules govern this module (see ERROR_MINING.md):

1. Index failures by structure, not vocabulary.
2. Filter memory at recall, not at write — record everything, rank on read.
3. Never store a fix before an independent verifier confirms it held.

SQLite-backed so the memory survives restarts. Pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from .logging_setup import get_logger

__all__ = [
    "IncidentJournal",
    "Incident",
    "RecoveryRecord",
    "signature_of",
]

_log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signature     TEXT NOT NULL,
    exc_type      TEXT NOT NULL,
    subsystem     TEXT NOT NULL,
    location      TEXT NOT NULL,
    message       TEXT NOT NULL,
    category      TEXT NOT NULL,
    severity      TEXT NOT NULL,
    context_json  TEXT NOT NULL DEFAULT '{}',
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_sig_ts ON incidents (signature, ts);
CREATE INDEX IF NOT EXISTS idx_incidents_subsys_ts ON incidents (subsystem, ts);

CREATE TABLE IF NOT EXISTS recoveries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id   INTEGER NOT NULL REFERENCES incidents(id),
    signature     TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT '',
    verified      INTEGER NOT NULL DEFAULT 0,
    verify_note   TEXT NOT NULL DEFAULT '',
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recoveries_sig ON recoveries (signature);

CREATE TABLE IF NOT EXISTS verified_fixes (
    signature     TEXT PRIMARY KEY,
    strategy      TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT '',
    successes     INTEGER NOT NULL DEFAULT 1,
    failures      INTEGER NOT NULL DEFAULT 0,
    first_seen    REAL NOT NULL,
    last_used     REAL NOT NULL
);
"""


def _innermost_app_frame(exc: BaseException) -> str:
    """Code location of the innermost frame: file:function:line.

    Walks to the deepest frame (where the error actually fired) and trims
    the path to the repo-relative tail so signatures are stable across
    installs.
    """
    tb = getattr(exc, "__traceback__", None)
    last: tuple[str, str, int] | None = None
    seen = 0
    while tb is not None and seen < 64:
        seen += 1
        code = tb.tb_frame.f_code
        filename = code.co_filename or "<unknown>"
        # repo-relative tail: keep last 3 path components
        parts = filename.replace("\\", "/").split("/")
        tail = "/".join(parts[-3:])
        last = (tail, code.co_name, tb.tb_lineno or 0)
        tb = tb.tb_next
    if last is None:
        return "<no-traceback>:<unknown>:0"
    return f"{last[0]}:{last[1]}:{last[2]}"


def signature_of(exc: BaseException, subsystem: str = "") -> str:
    """Structural signature: sha1(exc_type | subsystem | code location).

    Deliberately excludes the message — "connection refused by 10.0.0.5"
    and "connection refused by 10.0.0.9" are the same incident.
    """
    loc = _innermost_app_frame(exc)
    raw = f"{type(exc).__name__}|{subsystem or '?'}|{loc}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Incident:
    """One recorded failure."""

    id: int
    signature: str
    exc_type: str
    subsystem: str
    location: str
    message: str
    category: str
    severity: str
    context: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0
    # Filled on read:
    repeat_count_1h: int = 0
    repeat_count_24h: int = 0
    verified_fix: dict[str, Any] | None = None


@dataclass
class RecoveryRecord:
    """One recovery attempt against an incident."""

    incident_id: int
    signature: str
    strategy: str
    detail: str = ""
    verified: bool = False
    verify_note: str = ""


class IncidentJournal:
    """SQLite-backed incident + recovery memory.

    Thread-safe. Designed for a t3.small: one small SQLite file, indexed
    reads, no background threads of its own.
    """

    # Recurrence thresholds that turn an incident into a pattern.
    CHRONIC_1H = 3      # 3+ same-signature in an hour -> chronic
    CHRONIC_24H = 10    # 10+ same-signature in a day -> chronic

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._lock = threading.RLock()
        # check_same_thread=False + our own lock: safe for the bot's threads.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # ── recording ──────────────────────────────────────────────────────

    def record_incident(
        self,
        exc: BaseException,
        *,
        subsystem: str = "",
        category: str = "unknown",
        severity: str = "medium",
        context: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> Incident:
        """Record a failure. Returns the incident with recurrence stats."""
        sig = signature_of(exc, subsystem)
        loc = _innermost_app_frame(exc)
        now = ts if ts is not None else time.time()
        ctx_json = json.dumps(context or {}, default=str)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO incidents (signature, exc_type, subsystem, location,"
                " message, category, severity, context_json, ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sig, type(exc).__name__, subsystem or "?",
                 loc, str(exc)[:2000], category, severity, ctx_json, now),
            )
            self._db.commit()
            inc_id = cur.lastrowid
        return self.get_incident(inc_id)

    def record_recovery(self, rec: RecoveryRecord) -> int:
        """Record a recovery attempt. Returns the recovery row id.

        If ``verified`` is True, the fix is promoted to the verified-fix
        table (mined rule 3: only verified fixes are remembered as fixes).
        """
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO recoveries (incident_id, signature, strategy,"
                " detail, verified, verify_note, ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rec.incident_id, rec.signature, rec.strategy, rec.detail,
                 1 if rec.verified else 0, rec.verify_note, now),
            )
            rid = cur.lastrowid
            if rec.verified:
                self._promote_fix(rec.signature, rec.strategy, rec.detail, now)
            else:
                # A failed fix still teaches: count it against the fix.
                self._db.execute(
                    "UPDATE verified_fixes SET failures = failures + 1,"
                    " last_used = ? WHERE signature = ? AND strategy = ?",
                    (now, rec.signature, rec.strategy),
                )
            self._db.commit()
        return rid

    def _promote_fix(self, signature: str, strategy: str, detail: str,
                     now: float) -> None:
        row = self._db.execute(
            "SELECT successes FROM verified_fixes"
            " WHERE signature = ? AND strategy = ?",
            (signature, strategy),
        ).fetchone()
        if row:
            self._db.execute(
                "UPDATE verified_fixes SET successes = successes + 1,"
                " detail = ?, last_used = ?"
                " WHERE signature = ? AND strategy = ?",
                (detail, now, signature, strategy),
            )
        else:
            self._db.execute(
                "INSERT INTO verified_fixes (signature, strategy, detail,"
                " successes, failures, first_seen, last_used)"
                " VALUES (?, ?, ?, 1, 0, ?, ?)",
                (signature, strategy, detail, now, now),
            )

    # ── recall (filtered at read time — mined rule 2) ──────────────────

    def get_incident(self, incident_id: int) -> Incident:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no incident {incident_id}")
        return self._hydrate(row)

    def _hydrate(self, row: sqlite3.Row) -> Incident:
        sig = row["signature"]
        now = time.time()
        with self._lock:
            c1h = self._db.execute(
                "SELECT COUNT(*) AS n FROM incidents"
                " WHERE signature = ? AND ts >= ?",
                (sig, now - 3600),
            ).fetchone()["n"]
            c24h = self._db.execute(
                "SELECT COUNT(*) AS n FROM incidents"
                " WHERE signature = ? AND ts >= ?",
                (sig, now - 86400),
            ).fetchone()["n"]
            fix = self._db.execute(
                "SELECT strategy, detail, successes, failures"
                " FROM verified_fixes WHERE signature = ?"
                " ORDER BY successes DESC LIMIT 1",
                (sig,),
            ).fetchone()
        return Incident(
            id=row["id"], signature=sig, exc_type=row["exc_type"],
            subsystem=row["subsystem"], location=row["location"],
            message=row["message"], category=row["category"],
            severity=row["severity"],
            context=json.loads(row["context_json"] or "{}"),
            ts=row["ts"],
            repeat_count_1h=c1h, repeat_count_24h=c24h,
            verified_fix=dict(fix) if fix else None,
        )

    def find_similar(self, exc: BaseException, subsystem: str = "",
                     limit: int = 5) -> list[Incident]:
        """Past incidents with the same structural signature, newest first."""
        sig = signature_of(exc, subsystem)
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM incidents WHERE signature = ?"
                " ORDER BY ts DESC LIMIT ?",
                (sig, limit),
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def find_similar_fuzzy(self, exc: BaseException, subsystem: str = "",
                           limit: int = 5,
                           location_threshold: float = 0.6) -> list[Incident]:
        """Fuzzy match: same exception type + subsystem, similar code location.

        Survives refactors — when a function moves files or line numbers
        shift, the exact signature breaks but the fuzzy match still finds
        the incident family. Ranked by (location similarity, recency).
        """
        import difflib
        exc_type = type(exc).__name__
        loc = _innermost_app_frame(exc)
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM incidents"
                " WHERE exc_type = ? AND subsystem = ?"
                " AND signature NOT LIKE 'heartbeat:%'"
                " AND ts >= ?"
                " ORDER BY ts DESC LIMIT 200",
                (exc_type, subsystem or "?", now - 30 * 86400),
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            sim = difflib.SequenceMatcher(None, loc, r["location"]).ratio()
            if sim >= location_threshold:
                # Recency boost: newer incidents rank higher at equal similarity.
                recency = 1.0 - min(1.0, (now - r["ts"]) / (30 * 86400))
                scored.append((sim * 0.7 + recency * 0.3, r))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [self._hydrate(r) for _, r in scored[:limit]]

    def causal_chain(self, incident: Incident,
                     window_s: float = 30.0,
                     limit: int = 5) -> list[Incident]:
        """Possible *causes*: incidents in OTHER subsystems that fired just
        before this one.

        If the database subsystem fails and 2 seconds later telegram fails,
        the telegram failure's root cause is probably the database — not
        telegram. Returns candidate causes, newest first.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM incidents"
                " WHERE subsystem != ? AND ts < ? AND ts >= ?"
                " AND signature NOT LIKE 'heartbeat:%'"
                " ORDER BY ts DESC LIMIT ?",
                (incident.subsystem, incident.ts,
                 incident.ts - window_s, limit),
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def failure_spike(self, subsystem: str, window_s: float = 600.0,
                      baseline_s: float = 3600.0,
                      z_threshold: float = 3.0) -> dict[str, Any] | None:
        """Z-score spike detection: is this subsystem failing abnormally fast?

        Compares the recent window's failure count against a baseline drawn
        from the period *before* the recent window (no overlap — otherwise
        the spike dilutes its own baseline). Returns spike details or None.
        """
        now = time.time()
        with self._lock:
            recent = self._db.execute(
                "SELECT COUNT(*) AS n FROM incidents"
                " WHERE subsystem = ? AND ts >= ?"
                " AND signature NOT LIKE 'heartbeat:ok'",
                (subsystem, now - window_s),
            ).fetchone()["n"]
            # Baseline buckets cover [now - window_s - baseline_s, now - window_s).
            buckets = max(6, int(baseline_s / window_s))
            bucket_s = baseline_s / buckets
            base_end = now - window_s
            counts: list[int] = []
            for i in range(buckets):
                hi = base_end - i * bucket_s
                lo = hi - bucket_s
                c = self._db.execute(
                    "SELECT COUNT(*) AS n FROM incidents"
                    " WHERE subsystem = ? AND ts >= ? AND ts < ?"
                    " AND signature NOT LIKE 'heartbeat:ok'",
                    (subsystem, lo, hi),
                ).fetchone()["n"]
                counts.append(c)
        n = len(counts)
        mean = sum(counts) / n if n else 0.0
        var = sum((c - mean) ** 2 for c in counts) / n if n else 0.0
        std = var ** 0.5
        if std == 0:
            # Flat baseline: any failures at all are anomalous if baseline
            # was zero.
            if mean == 0 and recent > 0:
                return {"spike": True, "z": float("inf"),
                        "recent": recent, "baseline_mean": 0.0,
                        "note": "failures from a zero baseline"}
            return None
        z = (recent - mean) / std
        if z >= z_threshold:
            return {"spike": True, "z": round(z, 2), "recent": recent,
                    "baseline_mean": round(mean, 2),
                    "baseline_std": round(std, 2)}
        return None

    def known_fix(self, exc: BaseException,
                  subsystem: str = "") -> dict[str, Any] | None:
        """Best verified fix for this failure shape, or None."""
        sig = signature_of(exc, subsystem)
        with self._lock:
            row = self._db.execute(
                "SELECT strategy, detail, successes, failures"
                " FROM verified_fixes WHERE signature = ?"
                " ORDER BY successes DESC LIMIT 1",
                (sig,),
            ).fetchone()
        return dict(row) if row else None

    def is_chronic(self, exc: BaseException, subsystem: str = "") -> bool:
        """True when this failure shape keeps recurring."""
        sig = signature_of(exc, subsystem)
        now = time.time()
        with self._lock:
            c1h = self._db.execute(
                "SELECT COUNT(*) AS n FROM incidents"
                " WHERE signature = ? AND ts >= ?",
                (sig, now - 3600),
            ).fetchone()["n"]
            if c1h >= self.CHRONIC_1H:
                return True
            c24h = self._db.execute(
                "SELECT COUNT(*) AS n FROM incidents"
                " WHERE signature = ? AND ts >= ?",
                (sig, now - 86400),
            ).fetchone()["n"]
            return c24h >= self.CHRONIC_24H

    # ── aggregates for budgets ─────────────────────────────────────────

    def record_heartbeat(self, subsystem: str, *, ok: bool,
                         ts: float | None = None) -> None:
        """Record a successful (or failed-but-unraised) operation.

        Feeds true success/failure ratios for error budgets. Cheap: one
        indexed insert.
        """
        now = ts if ts is not None else time.time()
        with self._lock:
            # Heartbeats live in incidents with a synthetic signature so
            # budgets can compute ratios without a second table.
            self._db.execute(
                "INSERT INTO incidents (signature, exc_type, subsystem,"
                " location, message, category, severity, context_json, ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"heartbeat:{'ok' if ok else 'fail'}", "Heartbeat",
                 subsystem, "-", "-", "heartbeat",
                 "low", "{}", now),
            )
            self._db.commit()

    def subsystem_ratio(self, subsystem: str, window_s: float,
                        now: float | None = None) -> tuple[int, int, float]:
        """(failures, total, failure_ratio) over the trailing window."""
        now = now if now is not None else time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT signature, COUNT(*) AS n FROM incidents"
                " WHERE subsystem = ? AND ts >= ? GROUP BY signature",
                (subsystem, now - window_s),
            ).fetchall()
        total = sum(r["n"] for r in rows)
        fails = sum(r["n"] for r in rows
                    if not r["signature"].startswith("heartbeat:ok"))
        ratio = (fails / total) if total else 0.0
        return fails, total, ratio

    def top_failing(self, window_s: float = 86400,
                    limit: int = 10) -> list[dict[str, Any]]:
        """Most frequent failure signatures in the window — the prevention list."""
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT signature, exc_type, subsystem, location,"
                " COUNT(*) AS n, MAX(ts) AS last_ts"
                " FROM incidents"
                " WHERE ts >= ? AND signature NOT LIKE 'heartbeat:%'"
                " GROUP BY signature ORDER BY n DESC LIMIT ?",
                (now - window_s, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_incidents(self, subsystem: str | None = None,
                         limit: int = 20) -> list[dict[str, Any]]:
        """Newest incidents first, optionally filtered by subsystem."""
        limit = max(1, min(100, limit))
        with self._lock:
            if subsystem:
                rows = self._db.execute(
                    "SELECT id, signature, exc_type, subsystem, location,"
                    " message, severity, ts FROM incidents"
                    " WHERE subsystem = ? AND signature NOT LIKE 'heartbeat:%'"
                    " ORDER BY ts DESC LIMIT ?",
                    (subsystem, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT id, signature, exc_type, subsystem, location,"
                    " message, severity, ts FROM incidents"
                    " WHERE signature NOT LIKE 'heartbeat:%'"
                    " ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["first_seen"] = d.pop("ts")
            d["last_seen"] = d["first_seen"]
            d["count"] = 1
            out.append(d)
        return out

    def close(self) -> None:
        with self._lock:
            self._db.close()
