"""Close the calibration -> abstention -> learning-targets loop.

``cognition/calibration.py`` (the store), ``llm/abstention.py`` (the
moving threshold) and ``autonomy/model_weakness.py`` (the ranked
learning targets) were built as separate organs and never assembled:
no production code constructed a ``CalibrationStore``, and
``threshold_from_calibration`` / ``learning_targets`` had zero callers.
This module is the assembly — one pass over the trajectory store,
driven hourly by the durable ``calibration loop tick`` job on the
LIVE scheduler (``tools/autonomy.py`` ``__calibration_loop__`` tool):

1. **Replay.**  A fresh :class:`CalibrationStore` is rebuilt from the
   trajectory store's recent rows (newest 5,000 within retention).
   The store is in-memory by design — the trajectory store stays the
   durable record — so a restart rebuilds the identical state and can
   never double-count: idempotence is structural, no watermark.
2. **Predicted confidence (the honest mapping).**  Trajectory rows
   carry no stated confidence, and inventing one would be fake data.
   The prediction for a row is the slice's success rate over strictly
   earlier rows in the same (task, model) slice — the track record
   predicting the next outcome — or the 0.5 prior with no earlier
   rows.  No look-ahead: a row's prediction never uses its own
   outcome or any later row.  Rows that cannot be read (bad success
   value, empty slice, bad timestamp) are skipped and counted, never
   fatal to the pass.
3. **Targets.**  The pass returns the rebuilt store in its result
   dict; the caller (the ``__calibration_loop__`` tool, L5) ranks it
   with :func:`learning_targets` and files the worst as model
   weaknesses in the shared ledger — the same ledger the coordinator
   and the autonomy tools already read.  The ranking call lives with
   the caller because this module is L3 and must not import the L5
   autonomy organ (the layering test walks the whole AST, lazy
   imports included).
4. **Thresholds.**  Per-slice abstention thresholds from
   :func:`threshold_from_calibration` persist in
   ``cog_calibration_thresholds`` — but only for slices with enough
   data (``min_samples``); a slice below the floor keeps the base
   threshold, honestly unmoved.  :func:`threshold_for` reads one back
   (base threshold when absent); the abstention gate is the next
   consumer — no live answer path consults ``CalibratedAbstention``
   yet, so the gate wire-up stays a named remainder, not a silent
   assumption.

Nothing here auto-applies anything: targets are filed as weaknesses
for the autonomy loop to work, thresholds are advisory values the
abstention gate reads.  A bad row never kills the tick.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from ..llm.abstention import DEFAULT_THRESHOLD, threshold_from_calibration
from .calibration import PRIOR_CONFIDENCE, CalibrationStore
from .trajectories import DEFAULT_RETENTION_DAYS, TrajectoryStore

__all__ = [
    "CALIBRATION_LOOP_JOB",
    "CALIBRATION_LOOP_TOOL",
    "CALIBRATION_LOOP_SPEC",
    "REPLAY_LIMIT",
    "build_calibration_store",
    "run_calibration_loop_once",
    "ensure_calibration_loop_job",
    "threshold_for",
    "calibrated_thresholds",
]

#: Durable calibration-loop job: name, tool payload, default cadence.
CALIBRATION_LOOP_JOB = "calibration loop tick"
CALIBRATION_LOOP_TOOL = "__calibration_loop__"
CALIBRATION_LOOP_SPEC = "every 1h"

#: At most this many trajectory rows replayed per pass (newest first).
REPLAY_LIMIT = 5000

#: Slices below this many replayed rows get no persisted threshold —
#: a single failure is a data point, not a verdict.
DEFAULT_MIN_SAMPLES = 3


_THRESHOLDS_DDL = """
CREATE TABLE IF NOT EXISTS cog_calibration_thresholds (
    task            TEXT NOT NULL,
    model           TEXT NOT NULL,
    threshold       REAL NOT NULL,
    base_threshold  REAL NOT NULL,
    n               INTEGER NOT NULL DEFAULT 0,
    calibration_gap REAL,
    updated_at      REAL NOT NULL,
    PRIMARY KEY (task, model)
);
"""

_INTERVAL_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def _interval_seconds(spec: Any) -> float | None:
    """``"every 1h"`` / ``"every 60s"`` / ``"60"`` -> seconds, else None."""
    import re

    s = str(spec or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"every\s+(\d+(?:\.\d+)?)\s*([a-z]+)", s, re.IGNORECASE)
    if not m:
        return None
    unit = m.group(2).lower()
    if unit not in _INTERVAL_UNITS:
        return None
    return float(m.group(1)) * _INTERVAL_UNITS[unit]


def _spec_matches(stored: Any, spec: str) -> bool:
    """True when a stored job spec is the requested one (numeric
    compare for ``every``-kind specs)."""
    if str(stored or "") == spec:
        return True
    want = _interval_seconds(spec)
    have = _interval_seconds(stored)
    return want is not None and have is not None and want == have


def _job_field(job: Any, name: str) -> Any:
    if isinstance(job, dict):
        return job.get(name, "")
    return getattr(job, name, "")


def _ensure_thresholds_schema(db: Any) -> None:
    db.execute(_THRESHOLDS_DDL)


def build_calibration_store(
    db: Any,
    *,
    limit: int = REPLAY_LIMIT,
    retention_days: float = DEFAULT_RETENTION_DAYS,
) -> tuple[CalibrationStore, dict[str, Any]]:
    """Replay trajectory rows into a fresh calibration store.

    Returns ``(store, stats)`` where stats counts ``replayed``,
    ``skipped`` and ``slices``.  The store is rebuilt from scratch on
    every call: replaying the same rows twice yields the same state,
    so a restart or a second tick can never double-count.
    """
    # Ensures the trajectory schema exists (a fresh DB has no table).
    TrajectoryStore(db)
    store = CalibrationStore()
    cutoff = time.time() - float(retention_days) * 86400.0
    rows = db.query(
        "SELECT task_kind, capability, model_id, success, created_at"
        " FROM (SELECT task_kind, capability, model_id, success,"
        " created_at FROM cog_trajectories WHERE created_at >= ?"
        " ORDER BY created_at DESC, rowid DESC LIMIT ?)"
        " ORDER BY created_at ASC, rowid ASC",
        (cutoff, int(limit)),
    )
    # (task, model) -> [successes, n] over strictly earlier rows:
    # each row's prediction is the slice's track record before it.
    prior: dict[tuple[str, str], list[int]] = {}
    replayed = 0
    skipped = 0
    for row in rows:
        try:
            success_raw = row["success"]
            success = int(success_raw)
            if success not in (0, 1):
                raise ValueError(f"success {success_raw!r} not 0/1")
            created_at = float(row["created_at"])
            task = str(row["task_kind"] or row["capability"] or "")
            model = str(row["model_id"] or "")
            if not task and not model:
                raise ValueError("empty (task, model) slice")
        except Exception:
            skipped += 1
            continue
        key = (task, model)
        seen = prior.setdefault(key, [0, 0])
        predicted = (seen[0] / seen[1]) if seen[1] else PRIOR_CONFIDENCE
        try:
            store.record(task, model, predicted, bool(success),
                         created_at=created_at)
        except Exception:
            skipped += 1
            continue
        seen[0] += success
        seen[1] += 1
        replayed += 1
    return store, {
        "replayed": replayed,
        "skipped": skipped,
        "slices": len(store.pairs()),
    }


def run_calibration_loop_once(
    db: Any,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    replay_limit: int = REPLAY_LIMIT,
) -> dict[str, Any]:
    """One production pass of the loop over the workspace DB.

    Replays trajectories into a fresh store and persists per-slice
    abstention thresholds for slices with enough data.  The rebuilt
    store rides in the result dict under ``"store"`` so the caller
    (the ``__calibration_loop__`` tool) can rank learning targets
    from it and file them — the ranking call lives with the caller,
    not here, for layering (L3 must not import the L5 autonomy
    organ).  A bad trajectory row never kills the pass: it is
    skipped and counted.
    """
    store, stats = build_calibration_store(db, limit=replay_limit)
    _ensure_thresholds_schema(db)
    thresholds: dict[str, float] = {}
    now = time.time()
    for task, model in store.pairs():
        rows = store.records(task, model)
        if len(rows) < min_samples:
            continue
        threshold = threshold_from_calibration(store, task, model)
        gap = store.calibration_gap(task, model)
        db.execute(
            "INSERT OR REPLACE INTO cog_calibration_thresholds"
            " (task, model, threshold, base_threshold, n,"
            " calibration_gap, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task, model, float(threshold), float(DEFAULT_THRESHOLD),
             len(rows), gap, now),
        )
        thresholds[f"{task}::{model}"] = float(threshold)
    return {
        "ok": True,
        "replayed": stats["replayed"],
        "skipped": stats["skipped"],
        "slices": stats["slices"],
        "thresholds": thresholds,
        "store": store,
    }


def threshold_for(
    db: Any,
    task: str,
    model: str,
    *,
    base_threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Persisted abstention threshold for a slice, or the base
    threshold when the loop has recorded none (honest default)."""
    try:
        row = db.query_one(
            "SELECT threshold FROM cog_calibration_thresholds"
            " WHERE task = ? AND model = ?",
            (task or "", model or ""),
        )
    except Exception:
        return float(base_threshold)
    if not row:
        return float(base_threshold)
    try:
        return float(row["threshold"])
    except (TypeError, ValueError, KeyError):
        return float(base_threshold)


def calibrated_thresholds(db: Any) -> dict[str, dict[str, Any]]:
    """All persisted thresholds, keyed ``task::model``."""
    try:
        rows = db.query(
            "SELECT task, model, threshold, base_threshold, n,"
            " calibration_gap, updated_at"
            " FROM cog_calibration_thresholds ORDER BY task, model")
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            out[f"{row['task']}::{row['model']}"] = {
                "threshold": float(row["threshold"]),
                "base_threshold": float(row["base_threshold"]),
                "n": int(row["n"]),
                "calibration_gap": row["calibration_gap"],
                "updated_at": row["updated_at"],
            }
        except (TypeError, ValueError, KeyError):
            continue
    return out


def ensure_calibration_loop_job(
    scheduler: Any,
    *,
    spec: str = CALIBRATION_LOOP_SPEC,
) -> Any:
    """Register the durable calibration-loop job. Idempotent.

    A job with the loop's name already at the requested cadence is
    returned as-is; a stale one is replaced. Best-effort: never
    raises (a missing scheduler just means no scheduled loop runs).
    The job is ``routine=True`` — the scheduler records each run but
    does not page for it; calibration housekeeping surfaces in the
    weakness ledger and the threshold table, not as alerts.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            if _job_field(job, "name") == CALIBRATION_LOOP_JOB:
                if _spec_matches(_job_field(job, "spec"), spec):
                    return job
                with contextlib.suppress(Exception):
                    scheduler.remove(_job_field(job, "id"))
                break
        return scheduler.add(CALIBRATION_LOOP_JOB, spec, "tool",
                             {"tool": CALIBRATION_LOOP_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
