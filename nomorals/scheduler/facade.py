"""One scheduler facade over the scheduler implementations.

Scheduling engines (all stay; the facade fronts them in place):

* :mod:`nomorals.scheduler.unified_scheduler` (``UnifiedScheduler`` —
  cron / interval / mission / organ-tick views over one JSON store,
  the consolidation target),
* :mod:`nomorals.scheduler.scheduler` (user-facing ``Scheduler`` —
  DB-backed cron / reminders / event hooks),
* :mod:`nomorals.agents.scheduler` (agent ``Scheduler`` — DB-backed
  ``schedule_jobs``, blackouts, dependencies, missed-fire and overlap
  policies),
* :mod:`nomorals.missions.runner` (``MissionRunner`` — mission
  execution with retries / heartbeats),
* :mod:`nomorals.autonomy.idle` (``IdleMonitor`` — idle stages /
  inhibitors).

This facade gives callers ONE API:

* ``add_job`` / ``remove_job`` / ``list_jobs`` / ``due_jobs`` /
  ``mark_fired`` — delegate to ``UnifiedScheduler`` (the default
  backend), so one call site schedules anything.
* ``parse_spec`` — delegates to the agent scheduler's full parser
  when the composition root injects it (``spec_parser=``; layering
  forbids importing it here).  Uninjected, it falls back to the
  local minimal :func:`parse_schedule_spec`, whose results are a
  *subset* shape — the parity suite pins exactly where the two
  agree and where the full parser is required.
* ``add_agent_job`` / ``list_agent_jobs`` / ``remove_agent_job`` /
  ``set_agent_enabled`` — delegate to the injected agent
  ``Scheduler`` (``agent_scheduler=``); the missed-fire / overlap
  policy surface lives there and is delegated, never re-implemented.
* ``validate_cron`` / ``next_cron`` — delegate to the shared
  recurrence engine, so schedule-time refusal (garbage / impossible
  / unknown zone, by name) is reachable through the same one API.
* ``mission_runner`` / ``idle_state`` — lazy accessors for the other
  two systems, never re-implemented here.

Facade rule: PRESERVE APIs — the facade delegates, it does not
re-implement scheduling semantics. A call through the facade returns
the same result as the direct call on the underlying scheduler
(proven by ``tests/test_scheduler_facade_parity.py``).
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
                 idle_fn: Any | None = None,
                 agent_scheduler: Any | None = None,
                 spec_parser: Any | None = None) -> None:
        self._unified = unified
        self._store_path = store_path
        self._workspace_dir = workspace_dir
        self._mission_runner: Any | None = mission_runner
        self._idle_fn = idle_fn
        self._agent_scheduler: Any | None = agent_scheduler
        self._spec_parser = spec_parser

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
        """Parse a schedule spec.

        With the composition root's injected full parser (the agent
        scheduler's ``parse_schedule_spec``) the result is exactly the
        direct call's.  Uninjected, the local minimal parser answers —
        same ``(kind, details)`` shape for the specs it knows, a
        subset of the full grammar.
        """
        if self._spec_parser is not None:
            return self._spec_parser(spec)
        return parse_schedule_spec(spec)

    # ── agent scheduler: policy-bearing surface (delegated) ────────────────
    @property
    def agent_scheduler(self) -> Any:
        if self._agent_scheduler is None:
            raise RuntimeError(
                "SchedulerFacade.agent_scheduler requires an injected "
                "scheduler: pass agent_scheduler=... to __init__ "
                "(layering: facade must not import L5 implementations)")
        return self._agent_scheduler

    def add_agent_job(self, name: str, spec: str, payload_kind: str,
                      payload: dict[str, Any], **kwargs: Any) -> Any:
        """Delegate to the agent Scheduler's ``add`` — same arguments,
        same result, same missed-fire / overlap policy validation."""
        return self.agent_scheduler.add(
            name, spec, payload_kind, payload, **kwargs)

    def list_agent_jobs(self, **kwargs: Any) -> Any:
        return self.agent_scheduler.list_jobs(**kwargs)

    def remove_agent_job(self, ref: str) -> Any:
        return self.agent_scheduler.remove(ref)

    def set_agent_enabled(self, ref: str, enabled: bool) -> Any:
        return self.agent_scheduler.set_enabled(ref, enabled)

    # ── recurrence engine passthrough (schedule-time refusal) ──────────────
    def validate_cron(self, expr: str, timezone: str = "") -> Any:
        """Parse + prove a cron expression schedulable (by-name refusal)."""
        from .recurrence import validate_cron

        return validate_cron(expr, timezone)

    def next_cron(self, expr: str, after: Any, timezone: str = "") -> float:
        """Next firing strictly after ``after`` (unix timestamp)."""
        from .recurrence import next_cron

        return next_cron(expr, after, timezone)

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
