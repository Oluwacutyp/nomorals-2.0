"""Durable scheduler registration for the betting desk.

Same ``ensure_*`` pattern as every other job family (finance/jobs.py,
news_jobs.py): extracted so the registration can be enumerated — and
binding-tested by the pass-#75 contract test — without booting the
whole runtime.

Jobs:

* ``betdesk study`` — every 30m, one full desk cycle (collect,
  import, lab, library refresh).  Marked ``routine``: the cycle's
  owner-facing output rides its own surfaces (signal pushes,
  digests), not job alerts — a FAILED cycle still records a failed
  run (pass #71 semantics).
* ``betdesk signals`` — every 10m, evaluate Library verdicts for
  callable time windows and publish the calls (>= 2 minutes of lead
  is enforced inside the engine, not here).  Also routine: calls
  self-publish through the Notifier.

Idempotent by name; no-ops when the jobs already exist.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_betdesk_jobs", "BETDESK_STUDY_JOB",
           "BETDESK_SIGNALS_JOB"]

BETDESK_STUDY_JOB = "betdesk study"
BETDESK_SIGNALS_JOB = "betdesk signals"

_JOBS = (
    (BETDESK_STUDY_JOB, "every 30m",
     {"tool": "betdesk", "args": {"action": "study"}}),
    (BETDESK_SIGNALS_JOB, "every 10m",
     {"tool": "betdesk", "args": {"action": "signals", "deliver": True}}),
)


def ensure_betdesk_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the desk's study + signal jobs.

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
