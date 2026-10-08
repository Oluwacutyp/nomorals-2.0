"""Curated daily batches — scarcity-as-ritual (#83).

A small daily set of high-quality picks instead of infinite scroll:
daily gig picks, daily community highlights, daily learning content.

Rules:
- The day's batch is STABLE: same picks all day, regenerated at
  midnight. Ritual needs a repeatable rhythm.
- No repeats across days unless the pool is exhausted (then the
  least-recently-shown come back first).
- Pure scoring: tag overlap with the owner's preferences, weighted
  by each tag's importance, plus a base quality score.
- Empty pool → honest empty batch, never fabricated picks.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Candidate",
    "DailyBatchStore",
    "curate_daily",
]


@dataclass
class Candidate:
    """One matchable thing: a gig, a member highlight, a lesson."""

    candidate_id: str = ""
    title: str = ""
    summary: str = ""
    tags: tuple[str, ...] = ()
    quality: float = 0.5  # 0..1 — editor/curator score, or derived
    attributes: dict = field(default_factory=dict)  # for deal-breakers

    def __post_init__(self) -> None:
        self.tags = tuple(t.strip().lower() for t in (self.tags or ()) if t)
        try:
            self.quality = max(0.0, min(1.0, float(self.quality)))
        except (TypeError, ValueError):
            self.quality = 0.5


def curate_daily(
    candidates: list[Candidate],
    preferences: dict[str, float] | None = None,
    n: int = 5,
    *,
    exclude_ids: set[str] | None = None,
) -> list[Candidate]:
    """Score candidates and return the top ``n``.

    Score = 0.6 * preference overlap + 0.4 * quality.
    Preference overlap = sum(importance for matching tags) /
    sum(all importances) — 0.0 when the owner stated no preferences.

    ``exclude_ids`` are skipped (used for no-repeat-across-days).

    Pure function; never raises; empty input → empty batch.
    """
    try:
        prefs = {
            str(k).strip().lower(): float(v)
            for k, v in (preferences or {}).items()
            if v
        }
    except Exception:  # noqa: BLE001 — bad prefs degrade to unweighted
        prefs = {}
    total_weight = sum(prefs.values()) or 1.0
    excluded = exclude_ids or set()

    scored: list[tuple[float, Candidate]] = []
    for c in candidates or []:
        if not isinstance(c, Candidate) or c.candidate_id in excluded:
            continue
        overlap = sum(prefs.get(t, 0.0) for t in c.tags) / total_weight
        score = 0.6 * overlap + 0.4 * c.quality
        scored.append((score, c))
    scored.sort(key=lambda pair: (-pair[0], pair[1].candidate_id))
    return [c for _, c in scored[: max(0, int(n or 0))]]


_DEFAULT_DB = os.path.expanduser("~/.nomorals/matching/batches.db")
_DAY = 86400.0


class DailyBatchStore:
    """Persistent daily batches: stable all day, fresh tomorrow.

    ``today(surface, owner)`` returns the stored batch when one exists
    for the current UTC day, otherwise curates and stores a new one.
    SQLite, never raises.
    """

    def __init__(self, db_path: str = "") -> None:
        self._db = None
        try:
            path = db_path or _DEFAULT_DB
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS candidates (
                       candidate_id TEXT PRIMARY KEY, surface TEXT,
                       title TEXT, summary TEXT, tags TEXT,
                       quality REAL, attributes TEXT)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS batches (
                       id TEXT PRIMARY KEY, surface TEXT, owner TEXT,
                       day TEXT, candidate_ids TEXT, created_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS shown (
                       surface TEXT, owner TEXT, candidate_id TEXT,
                       shown_day TEXT,
                       PRIMARY KEY (surface, owner, candidate_id))"""
            )
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty store
            _log.warning("matching.batches: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    # ── candidates ──────────────────────────────────────────────

    def add_candidate(self, candidate: Candidate, surface: str = "gig") -> bool:
        """Register a candidate for future batches. Never raises."""
        try:
            if self._db is None or not candidate.candidate_id:
                return False
            import json

            self._db.execute(
                """INSERT OR REPLACE INTO candidates
                   (candidate_id, surface, title, summary, tags, quality,
                    attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (candidate.candidate_id, surface or "gig", candidate.title,
                 candidate.summary, ",".join(candidate.tags),
                 candidate.quality, json.dumps(candidate.attributes or {})),
            )
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: add_candidate failed",
                         exc_info=True)
            return False

    def candidates(self, surface: str = "gig") -> list[Candidate]:
        """All registered candidates for a surface. Never raises."""
        try:
            if self._db is None:
                return []
            import json

            rows = self._db.execute(
                "SELECT * FROM candidates WHERE surface = ?",
                (surface or "gig",)).fetchall()
            out = []
            for r in rows:
                try:
                    attrs = json.loads(r["attributes"] or "{}")
                except Exception:  # noqa: BLE001
                    attrs = {}
                out.append(Candidate(
                    candidate_id=r["candidate_id"], title=r["title"] or "",
                    summary=r["summary"] or "",
                    tags=tuple((r["tags"] or "").split(",")) if r["tags"] else (),
                    quality=r["quality"] if r["quality"] is not None else 0.5,
                    attributes=attrs if isinstance(attrs, dict) else {},
                ))
            return out
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: candidates failed", exc_info=True)
            return []

    # ── batches ─────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def today(
        self,
        surface: str = "gig",
        owner: str = "owner",
        preferences: dict[str, float] | None = None,
        n: int = 5,
        *,
        now: float | None = None,
        pool: list[Candidate] | None = None,
    ) -> list[Candidate]:
        """Today's batch: stored if fresh, curated if new. Never raises.

        ``pool`` overrides the registered candidates (e.g. after
        deal-breaker filtering) — the stored batch always reflects the
        pool it was curated from.
        """
        try:
            if self._db is None:
                return []
            surface = surface or "gig"
            owner = owner or "owner"
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            row = self._db.execute(
                """SELECT candidate_ids FROM batches
                   WHERE surface = ? AND owner = ? AND day = ?""",
                (surface, owner, day)).fetchone()
            if row and row["candidate_ids"]:
                ids = [i for i in (row["candidate_ids"] or "").split(",") if i]
                by_id = {c.candidate_id: c for c in self.candidates(surface)}
                return [by_id[i] for i in ids if i in by_id]
            # Fresh batch: avoid candidates shown in the last 7 days.
            week_ago = time.strftime(
                "%Y-%m-%d", time.gmtime((now or time.time()) - 7 * _DAY))
            recent = {
                r["candidate_id"]
                for r in self._db.execute(
                    """SELECT candidate_id FROM shown
                       WHERE surface = ? AND owner = ? AND shown_day >= ?""",
                    (surface, owner, week_ago)).fetchall()
            }
            pool = list(pool) if pool is not None else self.candidates(surface)
            batch = curate_daily(pool, preferences, n, exclude_ids=recent)
            if not batch and recent and pool:
                # Pool exhausted — recycle least-recently-shown first.
                batch = curate_daily(pool, preferences, n)
            if batch:
                self._db.execute(
                    """INSERT OR REPLACE INTO batches
                       (id, surface, owner, day, candidate_ids, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    ("b_" + uuid.uuid4().hex[:12], surface, owner, day,
                     ",".join(c.candidate_id for c in batch),
                     now or time.time()))
                for c in batch:
                    self._db.execute(
                        """INSERT OR REPLACE INTO shown
                           (surface, owner, candidate_id, shown_day)
                           VALUES (?, ?, ?, ?)""",
                        (surface, owner, c.candidate_id, day))
                self._db.commit()
            return batch
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: today failed", exc_info=True)
            return []
