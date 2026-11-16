"""Durable scheduler registrations for the encrypted offsite backup family.

Extracted from the partner runtime's boot block (gapfix 2) so the
registrations follow the same ``ensure_*`` pattern as every other job
family (finance, rooms, watchers, cadence, ...) and can be enumerated —
and binding-tested — without booting the whole runtime.

Jobs (single source of truth: :func:`nomorals.backup.coordinator.schedule_specs`):

* ``GitHub backup (daily)`` — 02:00 daily, one full generation
  (snapshot -> encrypt -> ship-or-queue -> prune) via ``__backup_tick__``.
* ``Restore drill (monthly)`` — 03:00 on the 1st, the restore drill over
  the latest generation via ``__backup_drill__``.

Why the live agent Scheduler, not ``register_jobs``: the coordinator's
:func:`register_jobs` installs ``UnifiedJob`` rows on a
``SchedulerFacade`` / ``UnifiedScheduler`` store — and nothing in
production ever fires that store (no construction of
``scheduler.unified_scheduler.UnifiedScheduler`` anywhere, no
``due_jobs`` caller, ``UnifiedJob`` referenced nowhere else). Installing
there would record jobs that never fire: a fake fix for a backup gap,
the exact silent failure the scheduler exists to prevent. The jobs
below land on the durable scheduler whose tick actually fires (the
agent ``Scheduler``, DB-backed ``schedule_jobs``), as ``tool`` payloads
— the same shape as ``ensure_finance_jobs``. ``register_jobs`` stays
as the unified-view API for the future consolidation; the cron specs
and job ids here are derived from ``schedule_specs()`` so the two
cannot drift.

Idempotent by name; no-ops when the jobs already exist.
"""

from __future__ import annotations

from typing import Any

from .coordinator import schedule_specs

__all__ = ["ensure_backup_jobs", "BACKUP_DAILY_JOB", "BACKUP_DRILL_JOB"]

BACKUP_DAILY_JOB = "GitHub backup (daily)"
BACKUP_DRILL_JOB = "Restore drill (monthly)"

#: The live scheduler fires ``tool`` payloads, not ``module:function``
#: handler strings — translate the coordinator's handler into the tool
#: that runs it. Unknown handlers are reported, never silently skipped.
_HANDLER_TO_TOOL = {
    "nomorals.backup.coordinator:run_scheduled_backup": "__backup_tick__",
    "nomorals.backup.coordinator:run_scheduled_drill": "__backup_drill__",
}


def ensure_backup_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the daily backup + monthly restore-drill jobs.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for spec in schedule_specs():
        name = spec["name"]
        handler = str((spec.get("payload") or {}).get("handler") or "")
        tool = _HANDLER_TO_TOOL.get(handler)
        if tool is None:
            out.append({"name": name, "scheduled": False,
                        "error": f"unknown handler {handler!r}"})
            continue
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        cron_spec = str((spec.get("recurrence") or {}).get("spec") or "")
        # Heavy: a generation snapshots + encrypts + uploads; defer under
        # pressure, never drop. Skip-on-overlap: two generations must never
        # run at once (the outbox holds anything unshipped). Routine: the
        # run reports itself through the notifier, so the job stays silent
        # at the scheduler. 30-minute wall: a big upload outruns the
        # default 300s tick cap.
        job = scheduler.add(
            name, f"cron {cron_spec}", "tool",
            {"tool": tool, "args": {}},
            heavy=True, run_timeout_s=1800.0, overlap_policy="skip",
            routine=True,
        )
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
