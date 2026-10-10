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

import json
import logging
import math
import random
import re
import threading
import time
from pathlib import Path

from ..core.ids import new_short_id
from ..storage.db import Database

_log = logging.getLogger(__name__)

__all__ = [
    "TrajectoryStore",
    "normalize_error",
    "cluster_key_for",
    "exception_class_of",
    "wilson_lower",
    "wilson_interval",
]

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

#: Matcher vocabulary for operator fingerprint rules (Sentry-style).
FINGERPRINT_MATCHERS = ("substring", "regex", "class")

_FINGERPRINT_RULES_DDL = """
CREATE TABLE IF NOT EXISTS cog_fingerprint_rules (
    id          TEXT PRIMARY KEY,
    matcher     TEXT NOT NULL DEFAULT 'substring',
    pattern     TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    priority    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cog_fp_rules_prio
    ON cog_fingerprint_rules(priority DESC, created_at);
"""

_RE_EXCEPTION_CLASS = re.compile(r"^([A-Za-z_][\w.]*)\s*:")


def exception_class_of(error: str) -> str:
    """Extract the exception class from an error string.

    Sentry groups issues by exception *type* as well as message; the
    class is the first axis of a fingerprint.  ``"TimeoutError: boom"``
    → ``"TimeoutError"``; a bare message with no ``Class:`` prefix →
    ``""``.  Only class-like prefixes (contain an uppercase letter) are
    accepted, so ``"note: blah"`` is not misread as a class.
    """
    if not error:
        return ""
    first_line = ""
    for line in str(error).splitlines():
        line = line.strip()
        if line:
            first_line = line
            break
    match = _RE_EXCEPTION_CLASS.match(first_line)
    if not match:
        return ""
    name = match.group(1)
    return name if any(ch.isupper() for ch in name) else ""


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (Evan Miller).

    ``successes`` may be a decay-weighted (fractional) success count and
    ``n`` the effective sample size — the interval stays honest for
    sparse data where the raw mean lies.  ``z=1.96`` is the 95% interval.
    Returns ``(lower, upper)``; the 0.5 prior when ``n <= 0``.
    """
    if n <= 0:
        return (_EMPTY_PRIOR, _EMPTY_PRIOR)
    n = float(n)
    p = min(1.0, max(0.0, float(successes) / n))
    z = float(z)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_lower(successes: float, n: float, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval — the honest rank score.

    1 success in 1 trial (mean 1.0, bound ~0.21) ranks *below* 80/100
    (mean 0.8, bound ~0.71): confidence beats luck.  The standard fix
    for sorting by average rating.
    """
    return wilson_interval(successes, n, z)[0]


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
        self.db.executescript(_FINGERPRINT_RULES_DDL)
        self._migrate_failure_class()
        self._lock = threading.RLock()
        # -inf so the first record() always runs the retention prune,
        # regardless of process uptime (monotonic clocks start near 0).
        self._last_prune = float("-inf")
        self._rules_cache: list[dict] | None = None

    def _migrate_failure_class(self) -> None:
        """Add the failure_class column to pre-existing databases.

        The typed failure class (``nomorals.llm.failures.FailureClass``)
        lets ``failure_clusters`` group by *kind* of breakage
        (rate_limited vs auth vs context_overflow) instead of only by raw
        message text.  Idempotent; never raises.
        """
        try:
            cols = {r["name"] for r in self.db.query(
                "PRAGMA table_info(cog_trajectories)")}
            if "failure_class" not in cols:
                self.db.execute(
                    "ALTER TABLE cog_trajectories "
                    "ADD COLUMN failure_class TEXT NOT NULL DEFAULT ''")
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cog_traj_fclass "
                    "ON cog_trajectories(failure_class, created_at)")
        except Exception:  # noqa: BLE001 — the column is a bonus
            _log.debug("failure_class migration skipped", exc_info=True)

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
        failure_class: str = "",
    ) -> None:
        """Record one execution outcome.

        ``failure_class`` is the typed failure kind
        (``nomorals.llm.failures.FailureClass`` value); when omitted it is
        derived from the error text so clusters still group by kind.
        """
        if not failure_class and error:
            try:
                from ..llm.failures import classify_failure

                failure_class = classify_failure(error).failure_class.value
            except Exception:  # noqa: BLE001 — the class is a bonus
                failure_class = ""
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
            failure_class=failure_class,
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
        failure_class: str = "",
        created_at: float | None = None,
    ) -> str:
        """Internal insert honoring an explicit timestamp (used by tests)."""
        row_id = new_short_id("traj_")
        with self._lock:
            self.db.execute(
                "INSERT INTO cog_trajectories (id, task_kind, capability,"
                " model_id, skill_id, tool, success, latency_s, cost,"
                " error, failure_class, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    failure_class or "",
                    float(created_at if created_at is not None else time.time()),
                ),
            )
        return row_id

    # ── fingerprint rules (Sentry-style operator overrides) ────────────
    def add_fingerprint_rule(
        self,
        matcher: str,
        pattern: str,
        fingerprint: str,
        priority: int = 0,
    ) -> str:
        """Add an operator fingerprint rule; first match wins.

        ``matcher`` is one of ``FINGERPRINT_MATCHERS``: ``"substring"``
        (pattern appears anywhere in the error), ``"regex"`` (pattern is
        a regular expression, compiled now so bad patterns fail fast),
        ``"class"`` (pattern equals the exception class from
        :func:`exception_class_of`).  A hit replaces the normalized
        signature with ``fingerprint`` for grouping, so operators can
        deliberately merge (``"rate_limited"``) or split clusters.
        Higher ``priority`` rules are evaluated first.
        """
        if matcher not in FINGERPRINT_MATCHERS:
            raise ValueError(
                f"unknown matcher {matcher!r}; use {FINGERPRINT_MATCHERS}")
        if matcher == "regex":
            try:
                re.compile(pattern or "")  # fail fast on bad patterns
            except re.error as exc:
                raise ValueError(f"bad regex pattern: {exc}") from exc
        rule_id = new_short_id("fpr_")
        with self._lock:
            self.db.execute(
                "INSERT INTO cog_fingerprint_rules (id, matcher, pattern,"
                " fingerprint, priority, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (rule_id, matcher, pattern or "", fingerprint or "",
                 int(priority), time.time()),
            )
            self._rules_cache = None
        return rule_id

    def remove_fingerprint_rule(self, rule_id: str) -> bool:
        """Remove a fingerprint rule.  Returns False when unknown."""
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM cog_fingerprint_rules WHERE id = ?", (rule_id,))
            if (cur.rowcount or 0) > 0:
                self._rules_cache = None
                return True
        return False

    def fingerprint_rules(self) -> list[dict]:
        """All rules, highest priority first."""
        return self.db.query(
            "SELECT id, matcher, pattern, fingerprint, priority, created_at"
            " FROM cog_fingerprint_rules"
            " ORDER BY priority DESC, created_at ASC")

    def _rules(self) -> list[dict]:
        if self._rules_cache is None:
            rules = self.fingerprint_rules()
            compiled = []
            for rule in rules:
                if rule.get("matcher") == "regex":
                    try:
                        rule = dict(rule)
                        rule["_rx"] = re.compile(rule.get("pattern") or "")
                    except re.error:
                        continue  # bad rule stored before compile check
                compiled.append(rule)
            self._rules_cache = compiled
        return self._rules_cache

    @staticmethod
    def _apply_rules(error: str, rules: list[dict]) -> str | None:
        """First matching rule's fingerprint, or None."""
        if not rules or not error:
            return None
        exc_class = exception_class_of(error)
        text = str(error)
        for rule in rules:
            matcher = rule.get("matcher")
            pattern = rule.get("pattern") or ""
            if matcher == "substring":
                if pattern and pattern in text:
                    return rule.get("fingerprint") or ""
            elif matcher == "class":
                if pattern and pattern == exc_class:
                    return rule.get("fingerprint") or ""
            elif matcher == "regex":
                rx = rule.get("_rx")
                if rx is not None and rx.search(text):
                    return rule.get("fingerprint") or ""
        return None

    def fingerprint_for(self, task_kind: str, error: str) -> str:
        """Grouping signature for an error: rule override wins, else the
        normalized signature.  This is what failure clusters and
        :meth:`FailureKB.note_for_error` must use so operator rules
        actually change grouping."""
        hit = self._apply_rules(error, self._rules())
        if hit is not None:
            return hit
        return normalize_error(error)

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
            "SELECT success, cost, latency_s, created_at FROM cog_trajectories"
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
    def _weighted_stats(rows: list[dict], half_life_days: float,
                        now: float) -> dict:
        """Decay-weighted aggregates: rate, effective sample size (Kish),
        and decay-weighted mean cost / latency for efficiency scoring."""
        half_life_s = float(half_life_days) * _SECONDS_PER_DAY
        num = 0.0
        den = 0.0
        sum_w2 = 0.0
        cost_num = 0.0
        lat_num = 0.0
        n = 0
        successes = 0
        for row in rows:
            w = _decay_weight(now - float(row["created_at"]), half_life_s)
            if w <= 0.0:
                continue
            s = float(row.get("success", 0))
            num += w * s
            den += w
            sum_w2 += w * w
            cost_num += w * float(row.get("cost", 0.0) or 0.0)
            lat_num += w * float(row.get("latency_s", 0.0) or 0.0)
            n += 1
            successes += int(s > 0.5)
        effective_n = (den * den / sum_w2) if sum_w2 > 0 else 0.0
        rate = (num / den) if den > 0.0 else _EMPTY_PRIOR
        return {
            "n": n,
            "successes": successes,
            "rate": rate,
            "weighted_successes": num,
            "effective_n": effective_n,
            "avg_cost": (cost_num / den) if den > 0.0 else 0.0,
            "avg_latency_s": (lat_num / den) if den > 0.0 else 0.0,
        }

    @staticmethod
    def _weighted_rate(rows: list[dict], half_life_days: float,
                       now: float) -> float:
        return TrajectoryStore._weighted_stats(
            rows, half_life_days, now)["rate"]

    def slice_stats(
        self,
        task_kind: str,
        capability: str,
        model_id: str = "",
        skill_id: str = "",
        half_life_days: float = 14.0,
    ) -> dict:
        """Full statistics for a slice: counts, decay-weighted rate, the
        effective (decay-adjusted) sample size, Wilson 95% bounds, and
        decay-weighted mean cost / latency."""
        stats = self._weighted_stats(
            self._slice(task_kind, capability, model_id, skill_id),
            half_life_days,
            time.time(),
        )
        lo, hi = wilson_interval(stats["weighted_successes"],
                                 stats["effective_n"])
        stats["wilson_lower"] = lo
        stats["wilson_upper"] = hi
        return stats

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
        objective: str = "success",
    ) -> list[str]:
        """Order candidates best-first.

        ``kind`` is ``"model"`` (candidates are model ids) or
        ``"skill"`` (candidates are skill ids).  ``objective``:

        - ``"success"`` — recency-weighted success rate (legacy behavior).
        - ``"efficiency"`` — success rate per unit cost: ``rate / (1 +
          avg_cost)``.  Uses the cost the store has been recording all
          along; free candidates score their raw rate.
        - ``"speed"`` — rate penalized by mean latency: ``rate / (1 +
          avg_latency_s)``.

        Unknown candidates get the 0.5 prior and sort after any measured
        candidate above 0.5.  Ties break deterministically by name so
        results are stable.
        """
        if objective not in ("success", "efficiency", "speed"):
            raise ValueError(
                f"unknown objective {objective!r}; use"
                " 'success', 'efficiency' or 'speed'")
        scored: list[tuple[float, str]] = []
        for cand in candidates:
            stats = self.slice_stats(task_kind, capability,
                                     **{f"{kind}_id": cand})
            rate = stats["rate"]
            if objective == "efficiency":
                score = rate / (1.0 + max(0.0, stats["avg_cost"]))
            elif objective == "speed":
                score = rate / (1.0 + max(0.0, stats["avg_latency_s"]))
            else:
                score = rate
            scored.append((score, cand))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [cand for _, cand in scored]

    def recommend(
        self,
        task_kind: str,
        capability: str,
        candidates: list[str],
        kind: str = "model",
        strategy: str = "mean",
        seed: int | None = None,
    ) -> tuple[str, dict]:
        """Pick one candidate using a selection strategy.

        Strategies (mined from production LLM-router research):

        - ``"mean"`` — highest recency-weighted success rate (ties by
          name).  Deterministic; same ordering as :meth:`rank`.
        - ``"wilson"`` — highest Wilson lower bound: confidence beats
          luck, so 1/1 never outranks 80/100.
        - ``"thompson"`` — Thompson Sampling over per-model Beta
          posteriors (the L7 router pattern): each candidate's
          ``Beta(1 + weighted_successes, 1 + weighted_failures)`` is
          sampled and the max wins.  Balances exploration and
          exploitation; a new candidate can still be picked.  ``seed``
          makes the draw reproducible.
        - ``"ucb1"`` — Auer's UCB1 index: ``mean + sqrt(alpha * ln(T) /
          (2 * n))``.  Untried candidates (``n == 0``) score +inf and
          are always explored first.
        - ``"cost"`` — cheapest reliable pick: highest ``rate / (1 +
          avg_cost)``.

        Returns ``(best_candidate, details)`` where details carries the
        strategy and per-candidate stats/scores for logging.  Raises
        ``ValueError`` on an empty candidate list.
        """
        if not candidates:
            raise ValueError("recommend() needs at least one candidate")
        if strategy not in ("mean", "wilson", "thompson", "ucb1", "cost"):
            raise ValueError(f"unknown strategy {strategy!r}")
        stats_by: dict[str, dict] = {}
        for cand in candidates:
            stats_by[cand] = self.slice_stats(
                task_kind, capability, **{f"{kind}_id": cand})

        rng = random.Random(seed)
        scores: dict[str, float] = {}
        if strategy == "thompson":
            for cand, st in stats_by.items():
                a = 1.0 + st["weighted_successes"]
                b = 1.0 + (st["effective_n"] - st["weighted_successes"])
                scores[cand] = rng.betavariate(max(a, 1e-9), max(b, 1e-9))
        elif strategy == "ucb1":
            total = sum(st["effective_n"] for st in stats_by.values())
            for cand, st in stats_by.items():
                n_eff = st["effective_n"]
                if n_eff <= 0 or total <= 0:
                    scores[cand] = float("inf")  # unexplored: try it
                else:
                    scores[cand] = st["rate"] + math.sqrt(
                        4.0 * math.log(total) / (2.0 * n_eff))
        elif strategy == "wilson":
            for cand, st in stats_by.items():
                scores[cand] = st["wilson_lower"]
        elif strategy == "cost":
            for cand, st in stats_by.items():
                scores[cand] = st["rate"] / (
                    1.0 + max(0.0, st["avg_cost"]))
        else:  # mean
            for cand, st in stats_by.items():
                scores[cand] = st["rate"]

        # Deterministic tiebreak by name (except thompson draws, which
        # are already randomized; name order keeps the pick stable when
        # samples tie exactly).
        best = max(sorted(scores), key=lambda c: scores[c])
        details = {
            "strategy": strategy,
            "kind": kind,
            "candidates": {
                cand: {
                    "rate": round(stats_by[cand]["rate"], 4),
                    "n": stats_by[cand]["n"],
                    "effective_n": round(stats_by[cand]["effective_n"], 2),
                    "wilson_lower": round(stats_by[cand]["wilson_lower"], 4),
                    "avg_cost": round(stats_by[cand]["avg_cost"], 4),
                    "avg_latency_s": round(
                        stats_by[cand]["avg_latency_s"], 2),
                    "score": scores[cand],
                }
                for cand in candidates
            },
        }
        return best, details

    def trend(
        self,
        task_kind: str,
        capability: str,
        model_id: str = "",
        skill_id: str = "",
        window_days: float = 14.0,
    ) -> dict:
        """Is this slice getting better or worse?  Compares the recent
        half of ``window_days`` against the prior half (recency-weighted
        within each half).  ``direction`` is ``"improving"``,
        ``"declining"`` or ``"stable"`` (±0.05 hysteresis); ``delta`` is
        recent − prior."""
        now = time.time()
        half = float(window_days) / 2.0 * _SECONDS_PER_DAY
        rows = self.db.query(
            "SELECT success, created_at FROM cog_trajectories"
            " WHERE task_kind = ? AND capability = ?"
            " AND (model_id = ? OR ? = '')"
            " AND (skill_id = ? OR ? = '')"
            " AND created_at >= ?",
            (task_kind or "", capability or "",
             model_id or "", model_id or "",
             skill_id or "", skill_id or "",
             now - 2.0 * half),
        )
        recent = [r for r in rows if float(r["created_at"]) >= now - half]
        prior = [r for r in rows if float(r["created_at"]) < now - half]
        recent_rate = self._weighted_rate(recent, half / _SECONDS_PER_DAY,
                                          now) if recent else _EMPTY_PRIOR
        prior_rate = self._weighted_rate(prior, half / _SECONDS_PER_DAY,
                                         now) if prior else _EMPTY_PRIOR
        delta = recent_rate - prior_rate
        direction = ("improving" if delta > 0.05
                     else "declining" if delta < -0.05 else "stable")
        return {
            "direction": direction,
            "delta": round(delta, 4),
            "recent_rate": round(recent_rate, 4),
            "prior_rate": round(prior_rate, 4),
            "recent_n": len(recent),
            "prior_n": len(prior),
        }

    # ── failure clustering ───────────────────────────────────────────────
    def failure_clusters(
        self, task_kind: str | None = None, limit: int = 10
    ) -> list[dict]:
        """Group failures by failure class + fingerprint (Sentry-style).

        Grouping is by (task_kind, failure_class, fingerprint): a
        rate-limit storm and an auth outage no longer merge into one
        cluster just because their message texts look alike, and
        operator fingerprint rules can deliberately merge or split
        groups.  The fingerprint also carries the exception class
        (``exception_class_of``) as a separate field.

        Each dict: ``task_kind``, ``failure_class``,
        ``error_signature`` (the fingerprint — rule-mapped when a rule
        hit), ``raw_signature`` (normalized message before rules),
        ``exception_class``, ``count``, ``first_seen`` and ``last_seen``
        (unix time), ``models_affected`` (sorted distinct model ids),
        ``example_ids`` (up to 5 trajectory ids).
        ``task_kind=None`` aggregates across all task kinds.
        """
        params: list = []
        where = "success = 0"
        if task_kind is not None:
            where += " AND task_kind = ?"
            params.append(task_kind)
        rows = self.db.query(
            "SELECT id, task_kind, model_id, skill_id, error, failure_class,"
            " created_at FROM cog_trajectories"
            f" WHERE {where}",
            tuple(params),
        )
        rules = self._rules()
        clusters: dict[tuple[str, str, str], dict] = {}
        for row in rows:
            raw_error = row.get("error") or ""
            hit = self._apply_rules(raw_error, rules)
            raw_signature = normalize_error(raw_error)
            fingerprint = hit if hit is not None else raw_signature
            fclass = row.get("failure_class") or ""
            key = (row["task_kind"] or "", fclass, fingerprint)
            cluster = clusters.get(key)
            if cluster is None:
                cluster = {
                    "task_kind": row["task_kind"] or "",
                    "failure_class": fclass,
                    "error_signature": fingerprint,
                    "raw_signature": raw_signature,
                    "exception_class": exception_class_of(raw_error),
                    "count": 0,
                    "first_seen": float(row["created_at"]),
                    "last_seen": 0.0,
                    "models_affected": set(),
                    "example_ids": [],
                }
                clusters[key] = cluster
            cluster["count"] += 1
            created = float(row["created_at"])
            if created > cluster["last_seen"]:
                cluster["last_seen"] = created
            if created < cluster["first_seen"]:
                cluster["first_seen"] = created
            if row.get("model_id"):
                cluster["models_affected"].add(row["model_id"])
            if len(cluster["example_ids"]) < 5:
                cluster["example_ids"].append(row["id"])
        ordered = sorted(
            clusters.values(),
            key=lambda c: (-c["count"], -c["last_seen"]),
        )
        for cluster in ordered:
            cluster["models_affected"] = sorted(cluster["models_affected"])
        return ordered[: max(0, int(limit))]

    @staticmethod
    def cluster_priority(cluster: dict,
                         half_life_days: float = 14.0) -> float:
        """Triage priority for one failure cluster: count × recency decay
        (Sentry's "escalating" intuition) plus a bonus when the cluster
        fired in the last 24h.  Higher = fix first."""
        now = time.time()
        age_days = max(0.0, (now - float(cluster.get("last_seen", 0)))
                       / _SECONDS_PER_DAY)
        score = float(cluster.get("count", 0)) * (
            0.5 ** (age_days / float(half_life_days)))
        if age_days < 1.0:
            score += 2.0
        return round(score, 3)

    # ── portable artifact (Letta-style export/import) ────────────────────
    def export_json(self, path: str | Path) -> dict:
        """Export trajectories + fingerprint rules to JSON.  Returns
        ``{"trajectories": n, "rules": m, "path": str}``."""
        rows = self.db.query(
            "SELECT id, task_kind, capability, model_id, skill_id, tool,"
            " success, latency_s, cost, error, failure_class, created_at"
            " FROM cog_trajectories")
        payload = {
            "kind": "cognition-trajectories",
            "version": 1,
            "exported_at": time.time(),
            "trajectories": rows,
            "fingerprint_rules": self.fingerprint_rules(),
        }
        out = Path(path)
        out.write_text(json.dumps(payload), encoding="utf-8")
        return {"trajectories": len(rows),
                "rules": len(payload["fingerprint_rules"]),
                "path": str(out)}

    def import_json(self, path: str | Path) -> dict:
        """Import a payload written by :meth:`export_json` (or hand-made
        with the same schema).  Rows upsert by id; rules upsert by id.
        Returns ``{"trajectories": n, "rules": m}``."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "cognition-trajectories":
            raise ValueError("not a cognition-trajectories export")
        n_traj = 0
        with self._lock:
            for row in payload.get("trajectories", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO cog_trajectories (id, task_kind,"
                    " capability, model_id, skill_id, tool, success,"
                    " latency_s, cost, error, failure_class, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row.get("id") or new_short_id("traj_"),
                     row.get("task_kind") or "",
                     row.get("capability") or "",
                     row.get("model_id") or "",
                     row.get("skill_id") or "",
                     row.get("tool") or "",
                     int(row.get("success", 0)),
                     float(row.get("latency_s", 0.0)),
                     float(row.get("cost", 0.0)),
                     row.get("error") or "",
                     row.get("failure_class") or "",
                     float(row.get("created_at", time.time()))),
                )
                n_traj += 1
            n_rules = 0
            for rule in payload.get("fingerprint_rules", []):
                try:
                    self.add_fingerprint_rule(
                        rule.get("matcher", "substring"),
                        rule.get("pattern", ""),
                        rule.get("fingerprint", ""),
                        int(rule.get("priority", 0)),
                    )
                    n_rules += 1
                except ValueError:
                    continue
        return {"trajectories": n_traj, "rules": n_rules}
