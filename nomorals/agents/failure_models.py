"""Error-catcher classifier — learn which failures are real vs noise.

The failures ledger (``agents/failures.py``) records every failure;
today it surfaces everything, and the operator reads the whole list.
This module learns the triage: a classifier over failure rows,
trained on OPERATOR-GRADED rows (``grade_failure``) behind a volume
gate. Below the gate it is honestly unavailable and the ledger
surfaces everything — the current behaviour, unchanged.

The code rung is "surface everything" (every failure is real): its
held-out accuracy is the base rate of real failures. The learned
classifier promotes only when it beats that on the same held-out
slice — i.e. when it can actually separate signal from noise.

Grades arrive from the operator (or an immune/ops hook) via
:func:`grade_failure`; the grade journal is a JSONL under the home
dir (``NM_FAILURE_GRADES`` overrides). Nothing here invents labels:
an ungraded failure is waiting data, never training data.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "FAILURE_FEATURES", "ERROR_MIN_ROWS", "DEFAULT_GRADES_PATH",
    "failure_features", "grade_failure", "graded_rows",
    "fit_error_classifier", "classify_failure", "load_error_classifier",
]

_log = get_logger(__name__)

FAILURE_FEATURES = ("site_depth", "exc_known", "message_len", "burst_1h",
                    "hour_sin", "hour_cos")
#: Floor for the error classifier (operator-graded rows).
ERROR_MIN_ROWS = 30

DEFAULT_GRADES_PATH = str(Path(os.path.expanduser("~")) / ".nomorals"
                          / "failure_grades.jsonl")

#: Exception types the desk has seen often enough to name. A feature,
#: not a decision — the model learns the weights, not the list.
_KNOWN_EXC = frozenset({
    "TimeoutError", "ConnectionError", "ValueError", "KeyError",
    "AttributeError", "RuntimeError", "OSError", "IOError",
    "TypeError", "IndexError", "sqlite3.OperationalError",
    "sqlite3.DatabaseError", "json.JSONDecodeError",
})


def _failure_get(failure: Any, key: str, default: Any = None) -> Any:
    if isinstance(failure, dict):
        return failure.get(key, default)
    return getattr(failure, key, default)


def failure_features(failure: Any, *,
                     recent: list[dict[str, Any]] | None = None
                     ) -> dict[str, float]:
    """Feature vector for one failure row, all roughly 0..1 (message_len
    and burst are log-scaled). ``recent`` is the ledger's recent rows,
    used for the burst count when the row does not carry ``burst_1h``."""
    site = str(_failure_get(failure, "site", "") or "")
    exc = str(_failure_get(failure, "exc_type", "") or "")
    message = str(_failure_get(failure, "message", "") or "")
    ts = float(_failure_get(failure, "ts", 0.0) or 0.0)
    site_depth = min(4, site.count("/") + site.count(".") + 1) / 4.0
    burst = _failure_get(failure, "burst_1h")
    if burst is None:
        burst = 0
        if recent:
            for r in recent:
                r_ts = float(r.get("ts", 0.0) or 0.0)
                if (r.get("site") == site and ts - 3600.0 <= r_ts <= ts):
                    burst += 1
    hour = 0.0
    if ts:
        _now = time.gmtime(ts)
        hour = _now.tm_hour + _now.tm_min / 60.0
    return {
        "site_depth": round(site_depth, 4),
        "exc_known": 1.0 if exc in _KNOWN_EXC else 0.0,
        "message_len": round(min(3.0, math.log10(len(message) + 1.0)) / 3.0, 4),
        "burst_1h": round(min(5.0, math.log10(float(burst or 0) + 1.0))
                          / math.log10(6.0), 4),
        "hour_sin": round(math.sin(2.0 * math.pi * hour / 24.0), 4),
        "hour_cos": round(math.cos(2.0 * math.pi * hour / 24.0), 4),
    }


def _grades_path() -> Path:
    override = (os.environ.get("NM_FAILURE_GRADES") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(DEFAULT_GRADES_PATH)


def grade_failure(*, site: str, exc_type: str, message: str, real: bool,
                  grader: str = "operator") -> dict[str, Any]:
    """Grade one failure: real (needs attention) or noise. Appends to
    the grade journal. Returns the graded row. Never raises."""
    import hashlib as _hashlib

    row: dict[str, Any] = {
        "ts": time.time(), "site": site, "exc_type": exc_type,
        "message": (message or "")[:500],
        "message_hash": _hashlib.sha256(
            f"{site}|{exc_type}|{message or ''}".encode(
                "utf-8", "replace")).hexdigest()[:12],
        "real": bool(real), "grader": grader,
    }
    try:
        target = _grades_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        row["features"] = failure_features(row)
        with open(target, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:  # noqa: BLE001 - grading never kills the ledger
        _log.debug("failure grade failed", exc_info=True)
    return row


def graded_rows(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Graded rows as (features, label) — the batch-fit input.
    Label 1 = real, 0 = noise. Missing journal = []."""
    target = Path(path) if path is not None else _grades_path()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        feats = parsed.get("features") or failure_features(parsed)
        rows.append({"features": dict(feats),
                     "label": 1 if parsed.get("real") else 0})
    return rows


def fit_error_classifier(rows: list[dict[str, Any]] | None = None, *,
                         store: Any = None,
                         backend: str = "gradboost") -> Any:
    """Batch-fit the error classifier on graded failure rows (or the
    grade journal when ``rows`` is None). Returns the toolkit's
    :class:`llm.specialists.FitOutcome` — "trained" (saved under
    ``error-triage``), "held", "insufficient" (below
    ``ERROR_MIN_ROWS``), or "unavailable". The code rung is "surface
    everything" — its held-out accuracy is the base rate of real
    failures on the same slice."""
    from ..llm.specialists import LearnerSpec, ModelStore, fit_if_ready

    data = list(rows) if rows is not None else graded_rows()

    def _code_score(feats: list[dict[str, Any]],
                    labels: list[float]) -> float:
        # "surface everything": every failure is real.
        if not labels:
            return 0.0
        return sum(1 for y in labels if int(y) == 1) / len(labels)

    spec = LearnerSpec(job="error-triage", features=FAILURE_FEATURES,
                       kind="classifier", backend=backend,
                       min_examples=ERROR_MIN_ROWS, label_name="real")
    return fit_if_ready(data, spec, code_score_fn=_code_score,
                        store=(store or ModelStore()))


def classify_failure(failure: Any, model: Any, *,
                     recent: list[dict[str, Any]] | None = None) -> float:
    """P(real) for one failure under a fitted classifier. 0.5 when the
    model is absent — the ledger surfaces it, as today."""
    if model is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(model.predict_one(
            failure_features(failure, recent=recent)))))
    except Exception:  # noqa: BLE001 - a bad model never kills the ledger
        return 0.5


def load_error_classifier(store: Any = None) -> Any | None:
    """The fitted error classifier, if one was ever promoted."""
    try:
        from ..llm.specialists import ModelStore

        return (store or ModelStore()).load("error-triage")
    except Exception:  # noqa: BLE001 - a bad store never kills the ledger
        return None
