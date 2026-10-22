"""Durable scheduler registration for the pump desk.

Same ``ensure_*`` pattern as the betting desk's jobs (extracted so
registration is enumerable and binding-testable without booting
the runtime):

* ``pumpdesk watch`` — every 5m: intake slice + frontend candidate
  poll, dossier refresh, gate evaluation, call publishing.
  Routine: calls ride their own surface (the Notifier); a failed
  cycle still records a failed run (pass-#71 semantics).
* ``pumpdesk study`` — every 30m: grade due calls, recompute dev
  profiles, flag serial ruggers into the watch list.  Also routine.

Idempotent by name; no-ops when the jobs already exist.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_pumpdesk_jobs", "PUMPDESK_WATCH_JOB",
           "PUMPDESK_STUDY_JOB"]

PUMPDESK_WATCH_JOB = "pumpdesk watch"
PUMPDESK_STUDY_JOB = "pumpdesk study"

_JOBS = (
    (PUMPDESK_WATCH_JOB, "every 5m",
     {"tool": "pumpdesk", "args": {"action": "watch", "deliver": True}}),
    (PUMPDESK_STUDY_JOB, "every 30m",
     {"tool": "pumpdesk", "args": {"action": "study"}}),
)


def ensure_pumpdesk_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the desk's watch + study jobs.  Idempotent by name.
    Returns one status dict per job."""
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=True)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
