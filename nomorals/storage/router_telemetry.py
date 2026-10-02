"""Core Mind router telemetry — persisted across processes.

The Core Mind router (``nomorals/agents/coremind.py``) used to keep its
call counts in memory only, so a fresh ``nm mind`` process reported them
as "unavailable". This module persists the small telemetry record the
router writes on every decision:

* ``route:<kind>`` — decision counts per route (the organ the router
  picked, e.g. ``research_swarm``, ``coding``, ``brain``);
* ``model_consults`` / ``model_timeouts`` — the model-check call and its
  deadline hits;
* ``last_plan_error`` — JSON ``{error, route, at}`` of the most recent
  plan failure, so ``nm mind`` can show it with an age.

Backed by the ``coremind_telemetry`` table (migration 65). Everything
here is best-effort: a failed write is a debug log, never an exception,
because the router's hot path (one write per message) must not break on
storage trouble. Reads degrade to an empty snapshot.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "KEY_LAST_PLAN_ERROR",
    "KEY_MODEL_CONSULTS",
    "KEY_MODEL_TIMEOUTS",
    "record_model_check",
    "record_plan_error",
    "record_route",
    "snapshot",
]

_log = get_logger(__name__)

#: Telemetry keys that are not ``route:<kind>`` counters.
KEY_MODEL_CONSULTS = "model_consults"
KEY_MODEL_TIMEOUTS = "model_timeouts"
KEY_LAST_PLAN_ERROR = "last_plan_error"

#: Hard cap on the error text we persist (full tracebacks stay in logs).
_MAX_ERROR_LEN = 500


def _now() -> float:
    return time.time()


def _bump(db: Any, key: str) -> None:
    """Increment an integer counter key, creating it at 1."""
    now = _now()
    db.execute(
        "INSERT INTO coremind_telemetry (key, value, updated_at) "
        "VALUES (?, '1', ?) "
        "ON CONFLICT(key) DO UPDATE SET "
        "value = CAST(coremind_telemetry.value AS INTEGER) + 1, "
        "updated_at = excluded.updated_at",
        (key, now),
    )


def _set(db: Any, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO coremind_telemetry (key, value, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET "
        "value = excluded.value, updated_at = excluded.updated_at",
        (key, value, _now()),
    )


def record_route(db: Any, route: str) -> None:
    """Count one router decision for ``route``. Never raises."""
    try:
        name = (route or "brain").strip() or "brain"
        _bump(db, f"route:{name}")
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("router telemetry route write failed: %s", exc)


def record_model_check(db: Any, *, timed_out: bool = False) -> None:
    """Count one model-check call (and its deadline hit). Never raises."""
    try:
        _bump(db, KEY_MODEL_CONSULTS)
        if timed_out:
            _bump(db, KEY_MODEL_TIMEOUTS)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("router telemetry model-check write failed: %s", exc)


def record_plan_error(db: Any, error: str, *, route: str = "") -> None:
    """Persist the latest plan failure with a timestamp. Never raises."""
    try:
        text = (error or "").strip()[:_MAX_ERROR_LEN]
        if not text:
            return
        payload = json.dumps(
            {"error": text, "route": (route or "").strip(), "at": _now()},
            ensure_ascii=False,
        )
        _set(db, KEY_LAST_PLAN_ERROR, payload)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("router telemetry plan-error write failed: %s", exc)


def snapshot(db: Any) -> dict[str, Any]:
    """Read the persisted telemetry. Never raises; empty on any failure."""
    out: dict[str, Any] = {
        "routes": {},
        "model_consults": 0,
        "model_timeouts": 0,
        "last_plan_error": None,
    }
    if db is None:
        return out
    try:
        rows = db.query("SELECT key, value, updated_at FROM coremind_telemetry")
    except Exception as exc:  # noqa: BLE001 - e.g. pre-migration DBs
        _log.debug("router telemetry read failed: %s", exc)
        return out
    for row in rows:
        key = str(row.get("key") or "")
        val = str(row.get("value") or "")
        if key == KEY_LAST_PLAN_ERROR:
            try:
                out["last_plan_error"] = json.loads(val)
            except Exception:  # noqa: BLE001
                out["last_plan_error"] = {"error": val, "route": "", "at": 0}
        elif key == KEY_MODEL_CONSULTS:
            out["model_consults"] = _to_int(val)
        elif key == KEY_MODEL_TIMEOUTS:
            out["model_timeouts"] = _to_int(val)
        elif key.startswith("route:"):
            out["routes"][key[len("route:"):]] = _to_int(val)
    return out


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
