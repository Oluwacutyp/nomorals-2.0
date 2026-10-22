"""Durable scheduler registration for the weekly paid-task hunt.

Same ``ensure_*`` pattern as every other job family (finance/jobs.py,
betdesk/jobs.py): extracted so the registration can be enumerated —
and binding-tested — without booting the whole runtime.

One job:

* ``opportunity hunt`` — Mondays 05:30 local, ahead of the morning
  briefing (07:00): the :class:`WeeklyOpportunityHunt` scans global
  paid tasks, verifies Nigerian signup eligibility + a payout method
  that works from Nigeria against each source page, and stores the
  week's short ranked shortlist.  Marked ``routine``: the result
  rides the morning briefing (the ``PaidTasksProvider`` section), not
  job alerts — a failed run still records a failed run, and the
  stored run record makes the briefing report the failure honestly.

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_opportunity_jobs", "OPPORTUNITY_HUNT_JOB"]

OPPORTUNITY_HUNT_JOB = "opportunity hunt"

_JOBS = (
    (OPPORTUNITY_HUNT_JOB, "cron 30 5 * * MON",
     {"tool": "opportunity_hunt", "args": {"action": "run"}}),
)


def ensure_opportunity_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the weekly paid-task hunt job.

    Idempotent by name. Returns one status dict per job.
    """
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
