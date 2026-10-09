"""Consolidation on a real cadence — scheduled, observable, additive.

The old story: ``MemoryManager.consolidate()`` ran only on memory pressure
(5000 records) or when a human remembered to invoke it.  The config key
``consolidation_interval_seconds`` existed but nothing read it — a
decorative setting.

The new story: one idempotent scheduler job (``memory-consolidation``)
ticks the *additive* consolidation on the configured interval, and every
run is observable — last report, next due, undistilled-episode backlog —
via :func:`status`, ``/memory schedule``, and ``health()``.

Additive is the rule, not the mood: the cadence runs
:meth:`MemoryManager.consolidate_additive`, which never deletes anything.
The destructive ``consolidate()`` stays available for explicit,
human-invoked use only — the schedule never touches it, so the owner's
"forget only on explicit command" rule holds while the system still
distils durable facts on its own.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.logging_setup import get_logger
from .base import MemoryKind

_log = get_logger(__name__)

__all__ = [
    "CONSOLIDATION_JOB_NAME",
    "consolidate_now",
    "ensure_consolidation_job",
    "interval_seconds",
    "maybe_run",
    "meta_get",
    "meta_set",
    "status",
]

CONSOLIDATION_JOB_NAME = "memory-consolidation"

_META_TABLE = "memory_meta"
_KEY_LAST_RUN = "consolidation.last_run_at"
_KEY_LAST_REPORT = "consolidation.last_report"
_KEY_RUNS = "consolidation.runs"


# ── durable cadence state (a tiny KV inside the memories DB) ──────────────

def _ensure_meta_table(db: Any) -> None:
    try:
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {_META_TABLE} "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')")
    except Exception:  # noqa: BLE001
        pass


def meta_get(db: Any, key: str, default: str = "") -> str:
    """Read a cadence KV. Never raises."""
    try:
        _ensure_meta_table(db)
        return str(db.scalar(
            f"SELECT value FROM {_META_TABLE} WHERE key = ?",
            (key,), default=default) or default)
    except Exception:  # noqa: BLE001
        return default


def meta_set(db: Any, key: str, value: str) -> None:
    """Write a cadence KV. Never raises."""
    try:
        _ensure_meta_table(db)
        db.execute(
            f"INSERT INTO {_META_TABLE} (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))
    except Exception:  # noqa: BLE001
        pass


def interval_seconds(manager: Any) -> float:
    """The configured cadence. ``<= 0`` disables the schedule."""
    try:
        settings = getattr(getattr(manager, "context", None),
                           "settings", None)
        memory_settings = getattr(settings, "memory", None)
        if memory_settings is not None:
            return float(getattr(memory_settings,
                                 "consolidation_interval_seconds", 3600.0))
    except Exception:  # noqa: BLE001
        pass
    return 3600.0


def undistilled_episodes(manager: Any) -> int:
    """Episodes not yet distilled by additive consolidation — the backlog
    the cadence exists to drain.  Never raises."""
    try:
        rows = manager.db.query(
            "SELECT metadata FROM memories WHERE kind = ?",
            (MemoryKind.EPISODE,))
        pending = 0
        for row in rows:
            md = row.get("metadata")
            if isinstance(md, str):
                try:
                    md = json.loads(md or "{}")
                except Exception:  # noqa: BLE001
                    md = {}
            if not (md or {}).get("distilled_into"):
                pending += 1
        return pending
    except Exception as exc:  # noqa: BLE001
        _log.debug("undistilled count failed: %s", exc)
        return -1


# ── the tick ─────────────────────────────────────────────────────────────

def consolidate_now(manager: Any, *,
                    min_episodes: int = 5) -> dict[str, Any]:
    """Run one additive consolidation tick and record it. Never raises.

    This is what the schedule runs and what ``/memory consolidate`` and
    the ``memory`` tool's ``consolidate`` action invoke.  Additive only:
    episodes are distilled into facts and marked, never deleted.
    """
    report: dict[str, Any] = {"ok": False}
    try:
        report = manager.consolidate_additive(min_episodes=min_episodes)
        report["ok"] = True
        now = time.time()
        meta_set(manager.db, _KEY_LAST_RUN, str(now))
        try:
            meta_set(manager.db, _KEY_LAST_REPORT,
                     json.dumps(report, default=str))
        except Exception:  # noqa: BLE001
            pass
        try:
            runs = int(meta_get(manager.db, _KEY_RUNS, "0") or "0")
        except Exception:  # noqa: BLE001
            runs = 0
        meta_set(manager.db, _KEY_RUNS, str(runs + 1))
        report["run_number"] = runs + 1
    except Exception as exc:  # noqa: BLE001 — the tick never breaks the host
        _log.warning("consolidation tick failed: %s", exc)
        report = {"ok": False, "error": str(exc)[:300]}
    return report


def maybe_run(manager: Any) -> dict[str, Any]:
    """Run the tick iff the interval has elapsed since the last run.

    For hosts without the scheduler (CLI, scripts, tests): call this on
    boot or on a heartbeat and consolidation happens on a cadence without
    any daemon.  Returns ``{"ran": False, ...}`` when not due.  Never raises.
    """
    try:
        interval = interval_seconds(manager)
        if interval <= 0:
            return {"ran": False, "reason": "disabled (interval <= 0)"}
        try:
            last = float(meta_get(manager.db, _KEY_LAST_RUN, "0") or "0")
        except Exception:  # noqa: BLE001
            last = 0.0
        now = time.time()
        if now - last < interval:
            return {"ran": False, "reason": "not due",
                    "due_in_s": round(interval - (now - last), 1)}
        report = consolidate_now(manager)
        report["ran"] = True
        return report
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"error: {exc}"}


def status(manager: Any) -> dict[str, Any]:
    """The observable cadence: last run, next due, backlog. Never raises."""
    out: dict[str, Any] = {"job": CONSOLIDATION_JOB_NAME}
    try:
        interval = interval_seconds(manager)
        out["interval_s"] = interval
        out["enabled"] = interval > 0
        try:
            last = float(meta_get(manager.db, _KEY_LAST_RUN, "0") or "0")
        except Exception:  # noqa: BLE001
            last = 0.0
        out["last_run_at"] = last or None
        out["last_run_ago_s"] = round(time.time() - last, 1) if last else None
        out["next_due_in_s"] = (
            max(0.0, round(interval - (time.time() - last), 1))
            if last and interval > 0 else 0.0)
        try:
            out["runs"] = int(meta_get(manager.db, _KEY_RUNS, "0") or "0")
        except Exception:  # noqa: BLE001
            out["runs"] = 0
        raw_report = meta_get(manager.db, _KEY_LAST_REPORT, "")
        try:
            out["last_report"] = json.loads(raw_report) if raw_report else {}
        except Exception:  # noqa: BLE001
            out["last_report"] = {}
        out["undistilled_episodes"] = undistilled_episodes(manager)
        out["mode"] = "additive"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:200]
    return out


# ── scheduler registration ───────────────────────────────────────────────

def ensure_consolidation_job(context: Any) -> dict[str, Any]:
    """Register the single durable ``memory-consolidation`` job (idempotent).

    Spec ``every <Ns>`` from ``consolidation_interval_seconds`` (default
    3600).  Payload is the ``memory`` tool's ``consolidate`` action — the
    same tick ``/memory consolidate`` runs by hand.  Safe to call on every
    boot; a changed interval replaces the job.  Never raises.
    """
    try:
        from ..agents.scheduler import Scheduler

        manager = getattr(context, "memory", None)
        interval = interval_seconds(manager) if manager is not None else 3600.0
        if interval <= 0:
            return {"name": CONSOLIDATION_JOB_NAME,
                    "scheduled": False, "reason": "disabled"}
        spec = f"every {int(interval)}s"
        sched = Scheduler(context)
        try:
            have = [j for j in sched.list_jobs()
                    if j.get("name") == CONSOLIDATION_JOB_NAME]
        except Exception:  # noqa: BLE001 — scheduler table may not exist yet
            have = []
        if have and have[0].get("spec") == spec:
            return {"name": CONSOLIDATION_JOB_NAME,
                    "already_scheduled": True,
                    "job_id": have[0].get("id")}
        for job in have:  # stale interval → replace
            try:
                sched.remove(job["id"])
            except Exception:  # noqa: BLE001
                pass
        job = sched.add(CONSOLIDATION_JOB_NAME, spec, "tool",
                        {"tool": "memory", "args": {"action": "consolidate"}})
        _log.info("scheduled memory-consolidation job: %s", spec)
        return {"name": CONSOLIDATION_JOB_NAME, "scheduled": True,
                "job_id": job.get("id"), "spec": spec}
    except Exception as exc:  # noqa: BLE001 — optional wiring never breaks boot
        _log.warning("consolidation job not registered: %s", exc)
        return {"name": CONSOLIDATION_JOB_NAME, "scheduled": False,
                "reason": str(exc)[:200]}
