"""TrendOps autonomy — the unprompted proposal hook (owner vision
2026-10-21: "the class PROPOSES applications unprompted").

The class has :meth:`TrendOps.propose` — it ranks the live formats by
expiry urgency, least-used first, with ready apply commands — but
nothing called it.  This module is the scheduler job that calls it,
so the proposals actually reach the owner.  Same ``ensure_*`` pattern
as every other job family (``agents/news_jobs.py``,
``media/explore/jobs.py``): extracted so the registration can be
enumerated — and binding-tested — without booting the whole runtime.

One job:

* ``trendops proposals`` — daily 09:00: the ``trendops`` tool's
  ``propose`` action returns the ranked proposals.  The run publishes
  through the scheduler's own alert path (a tool run's report Doc via
  the notifier) — NOT routine, because the proposal IS the delivery:
  a routine job's successes stay silent and the owner would never
  see it.  When no formats are live the tool returns
  ``{"proposals": []}`` and the run is silent by the scheduled-
  silence rule (``job_reports.noop_reason``: "nothing to report") —
  never an empty proposal.

Cadence: daily, not a poll.  Formats expire over days, so an
hourly/30m cadence would re-send the same proposals all day; weekly
would miss a format peaking with two days left.  09:00 sits after
the morning briefing (07:00) and the exploration sweep (08:30), and
well outside the partner quiet window (22:00–08:00) so the publish
is not held.

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_trendops_jobs", "TRENDOPS_PROPOSALS_JOB"]

TRENDOPS_PROPOSALS_JOB = "trendops proposals"

TRENDOPS_PROPOSALS_SPEC = "daily 09:00"

_JOBS = (
    (TRENDOPS_PROPOSALS_JOB, TRENDOPS_PROPOSALS_SPEC,
     {"tool": "trendops", "args": {"action": "propose"}}),
)


def ensure_trendops_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the daily TrendOps proposal job.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
