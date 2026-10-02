"""Verifiable scoring for arena runs.

The arena used to be vibes — a run happened and nobody measured it.
This module gives every run a grade on five verifiable axes:

* ``research_usefulness`` (0-1): was the research actually useful?
* ``build_compiled`` (bool): did the built artifact compile/run?
* ``tests_passed`` (bool): did the verification suite pass?
* ``edit_precision`` (0-1): how much of the diff was correct?
* ``latency_s`` (seconds): wall time; mapped to a 0-1 score that
  decays to 0 at 600s.

``record_score`` stores one row per run. ``overall_score`` folds the
present axes into a single weighted number. ``category_scores`` and
``difficulty_target`` close the loop: categories the arena is bad at
get easier challenges (difficulty 1), categories it aces get harder
ones (difficulty 3). ``coverage_weights`` steers the sampler toward
categories with few scored runs.

Everything degrades gracefully: ``db=None`` or missing tables never
raise — callers get safe defaults.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "coverage_weights",
    "category_scores",
    "difficulty_target",
    "overall_score",
    "record_score",
]

#: Weighted-mean weights per axis. Latency is measured in seconds and
#: mapped to 0-1 before weighting.
_AXIS_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("research_usefulness", 0.25),
    ("build_compiled", 0.20),
    ("tests_passed", 0.25),
    ("edit_precision", 0.20),
    ("latency_s", 0.10),
)

#: Latency (seconds) at which the latency axis scores exactly 0.
_LATENCY_FLOOR_S = 600.0

_SCORE_COLUMNS = (
    "id", "ts", "topic", "category", "kind",
    "research_usefulness", "build_compiled", "tests_passed",
    "edit_precision", "latency_s", "notes",
)


def _ensure_table(db: Any) -> bool:
    """Create ``arena_scores`` exactly like migration 61. Never raises."""
    if db is None:
        return False
    try:
        with db.transaction():
            db.execute(
                "CREATE TABLE IF NOT EXISTS arena_scores "
                "(id TEXT PRIMARY KEY, ts REAL NOT NULL, "
                "topic TEXT NOT NULL, category TEXT NOT NULL, "
                "kind TEXT NOT NULL DEFAULT 'code', "
                "research_usefulness REAL, build_compiled INTEGER, "
                "tests_passed INTEGER, edit_precision REAL, "
                "latency_s REAL, notes TEXT DEFAULT '')")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_arena_scores_category "
                "ON arena_scores (category)")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_arena_scores_ts "
                "ON arena_scores (ts)")
        return True
    except Exception:  # noqa: BLE001 - fresh/unmigrated DBs
        return False


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _latency_score(latency_s: float) -> float:
    return _clamp01(max(0.0, 1.0 - float(latency_s) / _LATENCY_FLOOR_S))


def record_score(db: Any, *, topic: str, category: str, kind: str = "code",
                 research_usefulness: float | None = None,
                 build_compiled: bool | None = None,
                 tests_passed: bool | None = None,
                 edit_precision: float | None = None,
                 latency_s: float | None = None,
                 notes: str = "") -> str:
    """Store one scored arena run. Returns the 12-hex score id.

    All axes are optional except ``topic``/``category``. Returns ``""``
    (never raises) when there is no database or the table can't be
    created.
    """
    if not _ensure_table(db):
        return ""
    kind = str(kind or "code").strip().lower()
    if kind not in ("code", "research", "build"):
        kind = "code"
    score_id = secrets.token_hex(6)
    row = (
        score_id,
        time.time(),
        str(topic or "")[:500],
        str(category or "").strip().lower(),
        kind,
        None if research_usefulness is None else _clamp01(research_usefulness),
        None if build_compiled is None else int(bool(build_compiled)),
        None if tests_passed is None else int(bool(tests_passed)),
        None if edit_precision is None else _clamp01(edit_precision),
        None if latency_s is None else max(0.0, float(latency_s)),
        str(notes or "")[:2000],
    )
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO arena_scores "
                "(id, ts, topic, category, kind, research_usefulness, "
                "build_compiled, tests_passed, edit_precision, latency_s, "
                "notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        return score_id
    except Exception:  # noqa: BLE001
        return ""


def _axis_value(key: str, value: Any) -> float:
    if key in ("build_compiled", "tests_passed"):
        return 1.0 if bool(value) else 0.0
    if key == "latency_s":
        return _latency_score(value)
    return _clamp01(value)


def overall_score(axes: dict[str, Any] | None) -> float | None:
    """Weighted mean of the present axes (renormalized), or None.

    Axis keys are the ``record_score`` names; ``"latency"`` is also
    accepted as an alias for ``"latency_s"``. Boolean axes are coerced
    via ``bool()``; everything is clamped to [0, 1]. Latency scores
    ``max(0, 1 - latency_s/600)``.
    """
    if not axes:
        return None
    if "latency_s" not in axes and "latency" in axes:
        axes = dict(axes, latency_s=axes["latency"])
    total_w = 0.0
    total_v = 0.0
    for key, weight in _AXIS_WEIGHTS:
        if key not in axes or axes[key] is None:
            continue
        try:
            total_v += _axis_value(key, axes[key]) * weight
        except (TypeError, ValueError):
            continue
        total_w += weight
    if total_w <= 0.0:
        return None
    return _clamp01(total_v / total_w)


def _row_axes(row: dict[str, Any]) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for key in ("research_usefulness", "build_compiled", "tests_passed",
                "edit_precision", "latency_s"):
        if row.get(key) is not None:
            axes[key] = row[key]
    return axes


def category_scores(db: Any) -> dict[str, dict[str, Any]]:
    """Per-category aggregates.

    ``{category: {"runs": int, "avg": float|None, "avg_latency": float|None}}``
    — ``avg`` is the mean ``overall_score`` over rows that have one,
    ``avg_latency`` the mean of recorded latencies. ``{}`` on db=None
    or missing tables.
    """
    if db is None:
        return {}
    try:
        rows = db.query(
            "SELECT category, research_usefulness, build_compiled, "
            "tests_passed, edit_precision, latency_s FROM arena_scores")
    except Exception:  # noqa: BLE001
        return {}
    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        cat = str(row.get("category", "") or "").strip().lower()
        if not cat:
            continue
        slot = acc.setdefault(cat, {"runs": 0, "sum": 0.0, "n": 0,
                                    "lat_sum": 0.0, "lat_n": 0})
        slot["runs"] += 1
        overall = overall_score(_row_axes(row))
        if overall is not None:
            slot["sum"] += overall
            slot["n"] += 1
        if row.get("latency_s") is not None:
            try:
                slot["lat_sum"] += max(0.0, float(row["latency_s"]))
                slot["lat_n"] += 1
            except (TypeError, ValueError) as e:
                _log.debug("bad latency_s in arena_scores row: %s", e)
    out: dict[str, dict[str, Any]] = {}
    for cat, slot in acc.items():
        out[cat] = {
            "runs": slot["runs"],
            "avg": (slot["sum"] / slot["n"]) if slot["n"] else None,
            "avg_latency": (slot["lat_sum"] / slot["lat_n"]
                            if slot["lat_n"] else None),
        }
    return out


def difficulty_target(db: Any, category: str) -> int:
    """Which difficulty grade the sampler should aim at.

    3 when the category averages >= 0.75 over >= 3 scored runs (the
    arena aces it — push harder), 1 when it averages <= 0.35 over
    >= 3 runs (it's struggling — ease off), else 2. Unknown
    categories (or db=None) get 2.
    """
    try:
        info = category_scores(db).get(str(category or "").strip().lower(), {})
        runs = int(info.get("runs", 0) or 0)
        avg = info.get("avg")
    except Exception:  # noqa: BLE001
        return 2
    if avg is None:
        return 2
    if runs >= 3 and avg >= 0.75:
        return 3
    if runs >= 3 and avg <= 0.35:
        return 1
    return 2


def coverage_weights(db: Any) -> dict[str, float]:
    """Sampling boost per category from scored-run coverage.

    ``1.0 + 2.0 * (1 - runs / max_runs)`` — starved categories get up
    to 3.0, the best-covered get 1.0. Every category 1.0 when there
    are no scores at all (or db=None).
    """
    from .topics import all_categories

    cats = list(all_categories())
    if db is None:
        return {c: 1.0 for c in cats}
    try:
        rows = db.query(
            "SELECT category, COUNT(*) AS n FROM arena_scores GROUP BY category")
        counts = {str(r.get("category", "") or "").strip().lower():
                  int(r.get("n", 0) or 0) for r in rows}
    except Exception:  # noqa: BLE001
        return {c: 1.0 for c in cats}
    max_runs = max(counts.values(), default=0)
    if max_runs <= 0:
        return {c: 1.0 for c in cats}
    out: dict[str, float] = {}
    for cat in cats:
        n = counts.get(cat, 0)
        out[cat] = 3.0 if n == 0 else 1.0 + 2.0 * (1.0 - n / max_runs)
    return out
