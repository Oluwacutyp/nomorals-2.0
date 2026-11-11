"""Curated daily batches — scarcity-as-ritual (#83).

A small daily set of high-quality picks instead of infinite scroll:
daily gig picks, daily community highlights, daily learning content.
The reference product is Coffee Meets Bagel: a small daily batch, a
Discover-style exploration path, like/pass feedback that trains the
ranker — quality over quantity, on purpose.

Rules:
- The day's batch is STABLE: same picks all day, regenerated at
  midnight. Ritual needs a repeatable rhythm.
- No repeats across days unless the pool is exhausted (then the
  least-recently-shown come back first).
- Relevance + diversity: MMR (Carbonell & Goldstein 1998) re-ranks the
  top picks so five near-duplicates don't crowd out everything else.
- Explore/exploit: a slice of each batch is reserved for Thompson
  sampling over like/pass feedback — under-sampled and brand-new
  candidates get explored, winners get exploited.
- Empty pool → honest empty batch, never fabricated picks.
"""

from __future__ import annotations

import os
import random as _random
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Candidate",
    "DailyBatchStore",
    "curate_daily",
    "mmr_rerank",
    "explain_pick",
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


def _tag_similarity(a: Candidate, b: Candidate) -> float:
    """Jaccard similarity over tags — the MMR redundancy signal."""
    sa, sb = set(a.tags), set(b.tags)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mmr_rerank(
    candidates: list[Candidate],
    scores: dict[str, float],
    n: int,
    lambda_: float = 0.7,
) -> list[Candidate]:
    """Maximal Marginal Relevance re-rank (Carbonell & Goldstein 1998).

    Greedily builds the list picking each next candidate by
    ``λ·relevance − (1−λ)·max_similarity_to_already_picked``.
    λ=1 → pure relevance; λ=0 → pure diversity; 0.7 is the usual
    balance. Pure; never raises.
    """
    try:
        lam = max(0.0, min(1.0, float(lambda_)))
        pool = [c for c in (candidates or [])
                if isinstance(c, Candidate) and c.candidate_id]
        # Deterministic tie-breaks: score desc, id asc.
        pool.sort(key=lambda c: (-scores.get(c.candidate_id, 0.0),
                                 c.candidate_id))
        selected: list[Candidate] = []
        while pool and len(selected) < max(0, int(n or 0)):
            best, best_val = None, None
            for c in pool:
                rel = scores.get(c.candidate_id, 0.0)
                red = max((_tag_similarity(c, s) for s in selected),
                          default=0.0)
                val = lam * rel - (1.0 - lam) * red
                if best_val is None or val > best_val:
                    best, best_val = c, val
            selected.append(best)  # type: ignore[arg-type]
            pool.remove(best)
        return selected
    except Exception:  # noqa: BLE001
        _log.warning("matching.batches: mmr_rerank failed", exc_info=True)
        return [c for c in (candidates or [])
                if isinstance(c, Candidate)][: max(0, int(n or 0))]


def _base_scores(
    candidates: list[Candidate],
    prefs: dict[str, float],
    total_weight: float,
) -> dict[str, float]:
    scores = {}
    for c in candidates:
        overlap = sum(prefs.get(t, 0.0) for t in c.tags) / total_weight
        scores[c.candidate_id] = 0.6 * overlap + 0.4 * c.quality
    return scores


def curate_daily(
    candidates: list[Candidate],
    preferences: dict[str, float] | None = None,
    n: int = 5,
    *,
    exclude_ids: set[str] | None = None,
    diversify: bool = True,
    mmr_lambda: float = 0.7,
    explore: float = 0.0,
    stats: dict[str, tuple[int, int]] | None = None,
    seed: int | None = None,
) -> list[Candidate]:
    """Score candidates and return the top ``n``.

    Base score = 0.6 * preference overlap + 0.4 * quality.
    Preference overlap = sum(importance for matching tags) /
    sum(all importances) — 0.0 when the owner stated no preferences.

    ``exclude_ids`` are skipped (used for no-repeat-across-days).
    ``diversify`` applies MMR over a wide over-fetched pool (3n) so the
    batch isn't five near-duplicates — MMR needs room to choose from.
    ``explore`` (0..1) reserves that fraction of slots for Thompson
    sampling over ``stats`` = ``{id: (likes, impressions)}`` —
    under-sampled and brand-new candidates get explored (Beta(1,1)
    cold start), proven winners get exploited. ``seed`` makes the
    sampling deterministic (tests).

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
    n = max(0, int(n or 0))

    pool = [c for c in (candidates or [])
            if isinstance(c, Candidate) and c.candidate_id
            and c.candidate_id not in excluded]
    if not pool or n == 0:
        return []
    scores = _base_scores(pool, prefs, total_weight)

    # ── diversify first (MMR needs a wider pool than n — the
    # standard over-fetch pattern: select n from a diverse wide set,
    # not from the top-n by raw score) ──
    ordered = sorted(pool,
                     key=lambda c: (-scores[c.candidate_id], c.candidate_id))
    if diversify:
        wide_n = max(n * 3, n + 5)
        shortlist = mmr_rerank(ordered, scores, wide_n, mmr_lambda)
    else:
        shortlist = ordered

    # ── explore/exploit split (Thompson sampling) ──
    explore = max(0.0, min(1.0, float(explore or 0.0)))
    k_exp = min(len(shortlist), int(round(n * explore)))
    k_base = min(len(shortlist) - k_exp, n - k_exp)
    exploit = shortlist[:k_base]
    rest = [c for c in shortlist if c not in exploit]
    explore_picks: list[Candidate] = []
    if k_exp and rest:
        rng = _random.Random(seed)
        fb = stats or {}
        thetas = []
        for c in rest:
            likes, impressions = fb.get(c.candidate_id, (0, 0))
            likes = max(0, int(likes))
            impressions = max(likes, int(impressions))
            # Beta(likes+1, passes+1): cold start = Beta(1,1) uniform.
            theta = rng.betavariate(likes + 1, impressions - likes + 1)
            thetas.append((theta, scores[c.candidate_id], c.candidate_id, c))
        thetas.sort(key=lambda t: (-t[0], -t[1], t[2]))
        explore_picks = [t[3] for t in thetas[:k_exp]]

    return (exploit + explore_picks)[:n]


def explain_pick(candidate: Candidate,
                 preferences: dict[str, float] | None = None) -> str:
    """Why this candidate was picked — the explainability half of ranking.

    Mirrors ``curate_daily``'s formula and names the contributing tags.
    Pure; never raises.
    """
    try:
        prefs = {
            str(k).strip().lower(): float(v)
            for k, v in (preferences or {}).items()
            if v
        }
    except Exception:  # noqa: BLE001
        prefs = {}
    total = sum(prefs.values()) or 1.0
    matched = [(t, prefs[t]) for t in candidate.tags if t in prefs]
    matched.sort(key=lambda p: -p[1])
    overlap = sum(w for _, w in matched) / total
    score = 0.6 * overlap + 0.4 * candidate.quality
    lines = ["🎯 %s — score %.2f" % (candidate.title or candidate.candidate_id,
                                    score)]
    if matched:
        lines.append("   prefs %.2f from: %s"
                     % (overlap, ", ".join("%s ×%g" % (t, w)
                                           for t, w in matched[:6])))
    else:
        lines.append("   prefs 0.00 — no stated preferences matched")
    lines.append("   quality %.2f → +%.2f" % (candidate.quality,
                                             0.4 * candidate.quality))
    if candidate.summary:
        lines.append("   %s" % candidate.summary[:140])
    return "\n".join(lines)


_DEFAULT_DB = os.path.expanduser("~/.nomorals/matching/batches.db")
_DAY = 86400.0

#: The store's own DDL — the native path executes it and the seam
#: mapping is derived from it, so the two can never drift.
_CANDIDATES_DDL = """CREATE TABLE IF NOT EXISTS candidates (
                       candidate_id TEXT PRIMARY KEY, surface TEXT,
                       title TEXT, summary TEXT, tags TEXT,
                       quality REAL, attributes TEXT)"""
_BATCHES_DDL = """CREATE TABLE IF NOT EXISTS batches (
                       id TEXT PRIMARY KEY, surface TEXT, owner TEXT,
                       day TEXT, candidate_ids TEXT, created_at REAL)"""
_SHOWN_DDL = """CREATE TABLE IF NOT EXISTS shown (
                       surface TEXT, owner TEXT, candidate_id TEXT,
                       shown_day TEXT,
                       PRIMARY KEY (surface, owner, candidate_id))"""
_FEEDBACK_DDL = """CREATE TABLE IF NOT EXISTS feedback (
                       surface TEXT, owner TEXT, candidate_id TEXT,
                       impressions INTEGER DEFAULT 0,
                       likes INTEGER DEFAULT 0,
                       PRIMARY KEY (surface, owner, candidate_id))"""
_DDL = ";\n".join((_CANDIDATES_DDL, _BATCHES_DDL, _SHOWN_DDL,
                   _FEEDBACK_DDL))


class DailyBatchStore:
    """Persistent daily batches: stable all day, fresh tomorrow.

    ``today(surface, owner)`` returns the stored batch when one exists
    for the current UTC day, otherwise curates and stores a new one.
    Like/pass feedback trains future batches (Thompson exploration);
    every shown batch counts impressions. SQLite, never raises.

    When a document store is resolved (Mongo program P5 — an explicit
    ``store`` or the boot ambient), the same API runs over the seam
    instead (registered in storage.sidecars as matching-batches):
    all four tables are documents in ``matching-batches__candidates``
    / ``__batches`` / ``__shown`` / ``__feedback``; shown and feedback
    are keyed by their composite (surface, owner, candidate_id)
    primary key, whose parts live in the joined document id. The
    file's existing rows are imported once, through the same mapping,
    at first construction.
    """

    def __init__(self, db_path: str = "", *, store: Any = None) -> None:
        self._db = None
        self._port: Any = None
        try:
            path = db_path or _DEFAULT_DB
            os.makedirs(os.path.dirname(path), exist_ok=True)
            from ..storage.sidecars import sidecar_port

            port = sidecar_port(
                "matching-batches", _DDL,
                ("batches", "candidates", "shown", "feedback"),
                path, store=store)
            if port is not None:
                def _writer(table):
                    def write(rows):
                        for row in rows:
                            table.put(row)
                        return len(rows)
                    return write

                port.ensure_imported({
                    name: _writer(port.tables[name])
                    for name in ("batches", "candidates", "shown",
                                 "feedback")})
                self._port = port
                return
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(_CANDIDATES_DDL)
            self._db.execute(_BATCHES_DDL)
            self._db.execute(_SHOWN_DDL)
            self._db.execute(_FEEDBACK_DDL)
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty store
            _log.warning("matching.batches: db unavailable, running empty",
                         exc_info=True)
            self._db = None
            self._port = None

    # ── candidates ──────────────────────────────────────────────

    def add_candidate(self, candidate: Candidate, surface: str = "gig") -> bool:
        """Register a candidate for future batches. Never raises."""
        try:
            if (self._db is None and self._port is None) \
                    or not candidate.candidate_id:
                return False
            import json

            if self._port is not None:
                self._port.tables["candidates"].put({
                    "candidate_id": candidate.candidate_id,
                    "surface": surface or "gig",
                    "title": candidate.title,
                    "summary": candidate.summary,
                    "tags": ",".join(candidate.tags),
                    "quality": candidate.quality,
                    "attributes": json.dumps(candidate.attributes or {}),
                })
                return True
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

    def remove_candidate(self, candidate_id: str,
                         surface: str = "gig") -> bool:
        """Remove a candidate (and its feedback) from a surface."""
        try:
            if (self._db is None and self._port is None) \
                    or not candidate_id:
                return False
            surface = surface or "gig"
            if self._port is not None:
                gone = self._port.tables["candidates"].delete_where(
                    {"candidate_id": candidate_id, "surface": surface})
                # feedback's key is the composite (surface, owner,
                # candidate_id): its parts live only in the joined
                # document id, so the sweep scans restored rows and
                # deletes by id — never a find on PK fields
                feedback = self._port.tables["feedback"]
                for r in feedback.all():
                    if r["candidate_id"] == candidate_id \
                            and r["surface"] == surface:
                        feedback.delete("\x1f".join(
                            (r["surface"], r["owner"],
                             r["candidate_id"])))
                return gone > 0
            cur = self._db.execute(
                "DELETE FROM candidates WHERE candidate_id = ? AND surface = ?",
                (candidate_id, surface))
            self._db.execute(
                "DELETE FROM feedback WHERE candidate_id = ? AND surface = ?",
                (candidate_id, surface))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: remove_candidate failed",
                         exc_info=True)
            return False

    def candidates(self, surface: str = "gig") -> list[Candidate]:
        """All registered candidates for a surface. Never raises."""
        try:
            if self._db is None and self._port is None:
                return []
            import json

            if self._port is not None:
                rows = self._port.tables["candidates"].find(
                    {"surface": surface or "gig"})
            else:
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

    # ── feedback (the learning loop) ────────────────────────────

    def record_feedback(self, surface: str, owner: str, candidate_id: str,
                        liked: bool) -> bool:
        """Like/pass on a shown candidate — trains future batches."""
        try:
            if (self._db is None and self._port is None) \
                    or not candidate_id:
                return False
            if self._port is not None:
                # composite PK: the (surface, owner, candidate_id)
                # triple lives in the joined document id — the
                # upsert is a read-modify-write on that id
                table = self._port.tables["feedback"]
                joined = "\x1f".join(
                    (surface or "gig", owner or "owner", candidate_id))
                row = table.get(joined)
                table.put({
                    "surface": surface or "gig",
                    "owner": owner or "owner",
                    "candidate_id": candidate_id,
                    "impressions": (int(row["impressions"] or 0)
                                    if row else 0) + 1,
                    "likes": (int(row["likes"] or 0) if row else 0)
                             + (1 if liked else 0),
                })
                return True
            self._db.execute(
                """INSERT INTO feedback
                   (surface, owner, candidate_id, impressions, likes)
                   VALUES (?, ?, ?, 1, ?)
                   ON CONFLICT (surface, owner, candidate_id)
                   DO UPDATE SET impressions = impressions + 1,
                                 likes = likes + ?""",
                (surface or "gig", owner or "owner", candidate_id,
                 1 if liked else 0, 1 if liked else 0))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: record_feedback failed",
                         exc_info=True)
            return False

    def feedback_stats(self, surface: str,
                       owner: str) -> dict[str, tuple[int, int]]:
        """``{candidate_id: (likes, impressions)}``. Never raises."""
        try:
            if self._db is None and self._port is None:
                return {}
            if self._port is not None:
                # surface/owner are composite-PK parts — they live
                # only in the joined document id, so filter the
                # restored rows here, never via find on PK fields
                rows = [r for r in self._port.tables["feedback"].all()
                        if r["surface"] == (surface or "gig")
                        and r["owner"] == (owner or "owner")]
                return {r["candidate_id"]: (int(r["likes"] or 0),
                                            int(r["impressions"] or 0))
                        for r in rows}
            rows = self._db.execute(
                """SELECT candidate_id, likes, impressions FROM feedback
                   WHERE surface = ? AND owner = ?""",
                (surface or "gig", owner or "owner")).fetchall()
            return {r["candidate_id"]: (int(r["likes"] or 0),
                                       int(r["impressions"] or 0))
                    for r in rows}
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: feedback_stats failed",
                         exc_info=True)
            return {}

    def _bump_impressions(self, surface: str, owner: str,
                          batch: list[Candidate]) -> None:
        """A freshly shown batch counts impressions (that's what shown means)."""
        if self._port is not None:
            table = self._port.tables["feedback"]
            for c in batch:
                joined = "\x1f".join((surface, owner, c.candidate_id))
                row = table.get(joined)
                table.put({
                    "surface": surface, "owner": owner,
                    "candidate_id": c.candidate_id,
                    "impressions": (int(row["impressions"] or 0)
                                    if row else 0) + 1,
                    "likes": int(row["likes"] or 0) if row else 0,
                })
            return
        for c in batch:
            self._db.execute(
                """INSERT INTO feedback
                   (surface, owner, candidate_id, impressions, likes)
                   VALUES (?, ?, ?, 1, 0)
                   ON CONFLICT (surface, owner, candidate_id)
                   DO UPDATE SET impressions = impressions + 1""",
                (surface, owner, c.candidate_id))

    # ── batches ─────────────────────────────────────────────────

    def today(
        self,
        surface: str = "gig",
        owner: str = "owner",
        preferences: dict[str, float] | None = None,
        n: int = 5,
        *,
        now: float | None = None,
        pool: list[Candidate] | None = None,
        diversify: bool = True,
        mmr_lambda: float = 0.7,
        explore: float = 0.15,
        seed: int | None = None,
    ) -> list[Candidate]:
        """Today's batch: stored if fresh, curated if new. Never raises.

        ``pool`` overrides the registered candidates (e.g. after
        deal-breaker filtering) — the stored batch always reflects the
        pool it was curated from. ``explore`` reserves that fraction of
        slots for Thompson exploration over like/pass feedback.
        """
        try:
            if self._db is None and self._port is None:
                return []
            surface = surface or "gig"
            owner = owner or "owner"
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if self._port is not None:
                found = self._port.tables["batches"].find(
                    {"surface": surface, "owner": owner, "day": day})
                row = found[0] if found else None
            else:
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
            if self._port is not None:
                # shown's key is composite too — scan restored rows
                recent = {
                    r["candidate_id"]
                    for r in self._port.tables["shown"].all()
                    if r["surface"] == surface and r["owner"] == owner
                    and (r["shown_day"] or "") >= week_ago
                }
            else:
                recent = {
                    r["candidate_id"]
                    for r in self._db.execute(
                        """SELECT candidate_id FROM shown
                           WHERE surface = ? AND owner = ? AND shown_day >= ?""",
                        (surface, owner, week_ago)).fetchall()
                }
            pool = list(pool) if pool is not None else self.candidates(surface)
            stats = self.feedback_stats(surface, owner)
            batch = curate_daily(pool, preferences, n, exclude_ids=recent,
                                 diversify=diversify, mmr_lambda=mmr_lambda,
                                 explore=explore, stats=stats, seed=seed)
            if not batch and recent and pool:
                # Pool exhausted — recycle least-recently-shown first.
                batch = curate_daily(pool, preferences, n,
                                     diversify=diversify,
                                     mmr_lambda=mmr_lambda,
                                     explore=explore, stats=stats, seed=seed)
            if batch:
                if self._port is not None:
                    self._port.tables["batches"].put({
                        "id": "b_" + uuid.uuid4().hex[:12],
                        "surface": surface, "owner": owner, "day": day,
                        "candidate_ids": ",".join(
                            c.candidate_id for c in batch),
                        "created_at": now or time.time()})
                    shown = self._port.tables["shown"]
                    for c in batch:
                        shown.put({
                            "surface": surface, "owner": owner,
                            "candidate_id": c.candidate_id,
                            "shown_day": day})
                else:
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
                self._bump_impressions(surface, owner, batch)
                if self._db is not None:
                    self._db.commit()
            return batch
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: today failed", exc_info=True)
            return []

    def batch_history(self, surface: str = "gig", owner: str = "owner",
                      limit: int = 7) -> list[tuple[str, list[str]]]:
        """Recent batches: ``[(day, [candidate_ids])]`` newest first."""
        try:
            if self._db is None and self._port is None:
                return []
            if self._port is not None:
                rows = self._port.tables["batches"].find(
                    {"surface": surface or "gig",
                     "owner": owner or "owner"})
                rows = sorted(
                    rows, key=lambda r: r["day"] or "",
                    reverse=True)[: max(1, int(limit or 7))]
            else:
                rows = self._db.execute(
                    """SELECT day, candidate_ids FROM batches
                       WHERE surface = ? AND owner = ?
                       ORDER BY day DESC LIMIT ?""",
                    (surface or "gig", owner or "owner",
                     max(1, int(limit or 7)))).fetchall()
            return [(r["day"],
                     [i for i in (r["candidate_ids"] or "").split(",") if i])
                    for r in rows]
        except Exception:  # noqa: BLE001
            _log.warning("matching.batches: batch_history failed",
                         exc_info=True)
            return []
