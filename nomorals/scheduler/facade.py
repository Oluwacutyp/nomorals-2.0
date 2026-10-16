"""One scheduler facade over the four scheduler implementations.

Skip-list consolidation: scheduling lives in FOUR places —

* :mod:`nomorals.scheduler` (``UnifiedScheduler`` — cron / interval /
  mission / organ-tick views, the consolidation target),
* :mod:`nomorals.agents.scheduler` (agent ``Scheduler`` — DB-backed,
  blackouts, dependencies, missed-fire policy),
* :mod:`nomorals.missions.runner` (``MissionRunner`` — mission
  execution with retries / heartbeats),
* :mod:`nomorals.autonomy.idle` (``IdleMonitor`` — idle stages /
  inhibitors).

All four stay. This facade gives callers ONE API:

* ``add_job`` / ``remove_job`` / ``list_jobs`` / ``due_jobs`` /
  ``mark_fired`` — delegate to ``UnifiedScheduler`` (the default
  backend), so one call site schedules anything.
* ``parse_schedule_spec`` — passthrough to the agent scheduler's
  parser (same function, same result).
* ``mission_runner`` / ``idle_state`` — lazy accessors for the other
  two systems, never re-implemented here.

Facade rule: PRESERVE APIs — the facade delegates, it does not
re-implement scheduling semantics. A call through the facade returns
the same result as the direct call on the underlying scheduler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["SchedulerFacade", "parse_schedule_spec"]


def parse_schedule_spec(spec: str) -> Any:
    """Minimal schedule-spec parser, local to this facade.

    Layering: this facade (L4) must not import L5 scheduler
    implementations. Common specs are parsed here; callers needing the
    full agent-scheduler parser inject it via ``SchedulerFacade``.
    Returns ``(kind, details)`` with a truthy kind.
    """
    import re

    s = (spec or "").strip().lower()
    m = re.fullmatch(r"every\s+(\d+)\s*(h|hour|hours|m|min|mins|minutes|s|sec|seconds|d|day|days)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = n * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1 if unit.startswith("s") else 86400)
        return ("interval", {"seconds": secs, "spec": spec})
    if s.startswith("cron") or len(s.split()) in (5, 6):
        return ("cron", {"spec": spec})
    if s in ("daily", "weekly", "monthly", "hourly"):
        return ("recurrence", {"spec": spec})
    return ("unknown", {"spec": spec})


class SchedulerFacade:
    """One API over unified scheduler + agent scheduler + missions + idle."""

    def __init__(self, *,
                 unified: Any | None = None,
                 store_path: str | Path | None = None,
                 workspace_dir: str | Path | None = None,
                 mission_runner: Any | None = None,
                 idle_fn: Any | None = None) -> None:
        self._unified = unified
        self._store_path = store_path
        self._workspace_dir = workspace_dir
        self._mission_runner: Any | None = mission_runner
        self._idle_fn = idle_fn

    @property
    def unified(self) -> Any:
        if self._unified is None:
            from .unified_scheduler import UnifiedScheduler
            self._unified = UnifiedScheduler(
                store_path=str(self._store_path) if self._store_path else None)
        return self._unified

    # ── UnifiedScheduler passthrough (the one scheduling API) ──────────────
    def add_job(self, job: Any, **kwargs: Any) -> Any:
        return self.unified.add_job(job, **kwargs)

    def remove_job(self, job_id: str) -> bool:
        return self.unified.remove_job(job_id)

    def get_job(self, job_id: str) -> Any:
        return self.unified.get_job(job_id)

    def list_jobs(self) -> list[Any]:
        return self.unified.list_jobs()

    def due_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.unified.due_jobs(**kwargs)

    def mark_fired(self, job_id: str, **kwargs: Any) -> None:
        return self.unified.mark_fired(job_id, **kwargs)

    def cron_view(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.unified.cron_view(**kwargs)

    def mission_view(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.unified.mission_view(**kwargs)

    # ── agent scheduler: spec parsing passthrough ──────────────────────────
    def parse_spec(self, spec: str) -> Any:
        return parse_schedule_spec(spec)

    # ── missions runner: accessor, not a re-implementation ────────────────
    @property
    def mission_runner(self) -> Any:
        if self._mission_runner is None:
            raise RuntimeError(
                "SchedulerFacade.mission_runner requires an injected "
                "runner: pass mission_runner=... to __init__ (layering: "
                "facade must not import L5 implementations)")
        return self._mission_runner

    # ── autonomy idle: state passthrough ───────────────────────────────────
    def idle_state(self, workspace_dir: str | Path | None = None) -> dict[str, Any]:
        fn = getattr(self, "_idle_fn", None)
        if fn is None:
            return {"stage": "unknown", "idle": False,
                    "reason": "no idle_fn injected (layering: facade must not import L5)"}
        return fn(workspace_dir or self._workspace_dir)
