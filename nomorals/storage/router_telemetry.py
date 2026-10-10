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
* ``last_reevaluation`` — JSON ``{action, reason, goal, at}`` of the most
  recent mid-flight plan re-evaluation (wave F2), so ``nm mind`` shows
  when the orchestrator revised, trimmed, or aborted a plan.

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
from ..core.style import active_theme, header, kv_lines, styled_table

__all__ = [
    "KEY_LAST_PLAN_ERROR",
    "KEY_LAST_REEVALUATION",
    "KEY_MODEL_CONSULTS",
    "KEY_MODEL_TIMEOUTS",
    "format_snapshot",
    "record_counter",
    "record_gauge",
    "record_model_check",
    "record_plan_error",
    "record_reevaluation",
    "record_route",
    "record_timing",
    "reset",
    "snapshot",
    "top_routes",
]

_log = get_logger(__name__)

#: Telemetry keys that are not ``route:<kind>`` counters.
KEY_MODEL_CONSULTS = "model_consults"
KEY_MODEL_TIMEOUTS = "model_timeouts"
KEY_LAST_PLAN_ERROR = "last_plan_error"
KEY_LAST_REEVALUATION = "last_reevaluation"

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


def record_reevaluation(db: Any, action: str, reason: str, *,
                        goal: str = "") -> None:
    """Persist the latest mid-flight plan re-evaluation. Never raises."""
    try:
        payload = json.dumps(
            {"action": (action or "continue").strip(),
             "reason": (reason or "").strip()[:_MAX_ERROR_LEN],
             "goal": (goal or "").strip()[:200],
             "at": _now()},
            ensure_ascii=False,
        )
        _set(db, KEY_LAST_REEVALUATION, payload)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("router telemetry re-evaluation write failed: %s", exc)


# ── generic instruments ──────────────────────────────────────────────────
#
# OpenTelemetry/Prometheus-style primitives on the same table: counters,
# gauges, and timing sketches. Timings keep {count, sum, min, max} as JSON so
# snapshot() reports avg/min/max without storing every sample.

def record_counter(db: Any, key: str, delta: int = 1) -> None:
    """Add ``delta`` to a counter key. Never raises."""
    try:
        now = _now()
        db.execute(
            "INSERT INTO coremind_telemetry (key, value, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = CAST(coremind_telemetry.value AS INTEGER) + ?, "
            "updated_at = excluded.updated_at",
            (key, str(delta), now, delta),
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("telemetry counter write failed: %s", exc)


def record_gauge(db: Any, key: str, value: float) -> None:
    """Set a gauge key to ``value`` (last-write-wins). Never raises."""
    try:
        _set(db, key, repr(float(value)))
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("telemetry gauge write failed: %s", exc)


def record_timing(db: Any, key: str, seconds: float) -> None:
    """Fold one timing sample into a ``{count, sum, min, max}`` sketch."""
    try:
        row = db.query_one(
            "SELECT value FROM coremind_telemetry WHERE key = ?", (key,)
        )
        sketch = {"count": 0, "sum": 0.0, "min": None, "max": None}
        if row is not None:
            try:
                loaded = json.loads(str(row["value"]))
                if isinstance(loaded, dict):
                    sketch.update(loaded)
            except (ValueError, TypeError):
                pass
        seconds = float(seconds)
        sketch["count"] = int(sketch.get("count") or 0) + 1
        sketch["sum"] = float(sketch.get("sum") or 0.0) + seconds
        cur_min = sketch.get("min")
        cur_max = sketch.get("max")
        sketch["min"] = seconds if cur_min is None else min(cur_min, seconds)
        sketch["max"] = seconds if cur_max is None else max(cur_max, seconds)
        _set(db, key, json.dumps(sketch))
    except Exception as exc:  # noqa: BLE001 - telemetry must not break routing
        _log.debug("telemetry timing write failed: %s", exc)


def reset(db: Any, key: str) -> bool:
    """Delete one instrument. Returns True when something was cleared."""
    try:
        cursor = db.execute(
            "DELETE FROM coremind_telemetry WHERE key = ?", (key,)
        )
        return bool(cursor.rowcount)
    except Exception as exc:  # noqa: BLE001
        _log.debug("telemetry reset failed: %s", exc)
        return False


def top_routes(db: Any, limit: int = 10) -> list[tuple[str, int]]:
    """Routes sorted by decision count, most first."""
    snap = snapshot(db)
    routes = snap.get("routes", {})
    ranked = sorted(routes.items(), key=lambda kv: -kv[1])
    return ranked[: max(1, limit)]


def snapshot(db: Any) -> dict[str, Any]:
    """Read the persisted telemetry. Never raises; empty on any failure.

    Besides the router-specific keys, generic instruments surface too:
    ``timings`` holds ``{key: {count, sum, min, max, avg}}`` sketches and
    ``gauges`` holds ``{key: value}`` for plain numeric keys.
    """
    out: dict[str, Any] = {
        "routes": {},
        "model_consults": 0,
        "model_timeouts": 0,
        "last_plan_error": None,
        "last_reevaluation": None,
        "timings": {},
        "gauges": {},
    }
    if db is None:
        return out
    try:
        rows = db.query("SELECT key, value, updated_at FROM coremind_telemetry")
    except Exception as exc:  # noqa: BLE001 - e.g. pre-migration DBs
        _log.debug("router telemetry read failed: %s", exc)
        return out
    known = {KEY_LAST_PLAN_ERROR, KEY_LAST_REEVALUATION, KEY_MODEL_CONSULTS,
             KEY_MODEL_TIMEOUTS}
    for row in rows:
        key = str(row.get("key") or "")
        val = str(row.get("value") or "")
        if key == KEY_LAST_PLAN_ERROR:
            try:
                out["last_plan_error"] = json.loads(val)
            except Exception:  # noqa: BLE001
                out["last_plan_error"] = {"error": val, "route": "", "at": 0}
        elif key == KEY_LAST_REEVALUATION:
            try:
                out["last_reevaluation"] = json.loads(val)
            except Exception:  # noqa: BLE001
                out["last_reevaluation"] = {
                    "action": val, "reason": "", "goal": "", "at": 0}
        elif key == KEY_MODEL_CONSULTS:
            out["model_consults"] = _to_int(val)
        elif key == KEY_MODEL_TIMEOUTS:
            out["model_timeouts"] = _to_int(val)
        elif key.startswith("route:"):
            out["routes"][key[len("route:"):]] = _to_int(val)
        elif key not in known:
            # Generic instrument: timing sketch, gauge, or opaque counter.
            try:
                parsed = json.loads(val)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and "count" in parsed and "sum" in parsed:
                count = int(parsed.get("count") or 0)
                total = float(parsed.get("sum") or 0.0)
                out["timings"][key] = {
                    "count": count,
                    "sum": total,
                    "min": parsed.get("min"),
                    "max": parsed.get("max"),
                    "avg": (total / count) if count else 0.0,
                }
            else:
                try:
                    out["gauges"][key] = float(val)
                except (TypeError, ValueError):
                    pass
    return out


def format_snapshot(db: Any, theme: Any = None) -> str:
    """The ``nm mind`` surface: routes, checks, timings, last error with age."""
    theme = theme or active_theme()
    snap = snapshot(db)
    now = time.time()
    route_rows = [
        (route, str(count)) for route, count in top_routes(db, limit=12)
    ]
    lines = [
        header("router telemetry", theme=theme),
        *kv_lines(
            {
                "model consults": snap["model_consults"],
                "model timeouts": snap["model_timeouts"],
            },
            theme=theme,
        ),
    ]
    if route_rows:
        lines.append(styled_table(["route", "decisions"], route_rows, theme=theme))
    if snap["timings"]:
        timing_rows = [
            (key, str(t["count"]), f"{t['avg'] * 1000:.1f}ms",
             f"{(t['max'] or 0) * 1000:.1f}ms")
            for key, t in sorted(snap["timings"].items())
        ]
        lines.append(
            styled_table(["timing", "n", "avg", "max"], timing_rows, theme=theme))
    if snap["gauges"]:
        lines.extend(kv_lines(
            {k: v for k, v in sorted(snap["gauges"].items())}, theme=theme))
    last_error = snap.get("last_plan_error") or {}
    if last_error.get("error"):
        age = now - float(last_error.get("at") or now)
        lines.extend(
            kv_lines(
                {
                    "last plan error": f"{last_error.get('error')}"
                    f" (route={last_error.get('route') or '–'}, {age:.0f}s ago)",
                },
                theme=theme,
            )
        )
    last_reeval = snap.get("last_reevaluation") or {}
    if last_reeval.get("action"):
        age = now - float(last_reeval.get("at") or now)
        lines.extend(
            kv_lines(
                {
                    "last re-evaluation": f"{last_reeval.get('action')}: "
                    f"{last_reeval.get('reason') or '–'} ({age:.0f}s ago)",
                },
                theme=theme,
            )
        )
    return "\n".join(lines)


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
