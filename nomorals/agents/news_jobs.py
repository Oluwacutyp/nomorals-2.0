"""Durable scheduler registration for the news family (pass #85).

Follows the same ``ensure_*`` pattern as every other job family
(finance/jobs.py, immune, rooms, watchers, ...): extracted so the
registration can be enumerated — and binding-tested — without booting
the whole runtime.

Job:

* ``news poll`` — every 30m, the continuous poll: fetch all feeds,
  store every new story, brain-score interest, and push the
  high-interest tier anytime (quiet hours + per-day cap apply inside
  the poll).  Marked ``routine``: a poll's results reach the owner
  through pushes and digests, not through job alerts — but a FAILED
  poll still records a failed run (pass #71 semantics).

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_news_jobs", "NEWS_POLL_JOB"]

NEWS_POLL_JOB = "news poll"

_JOBS = (
    (NEWS_POLL_JOB, "every 30m",
     {"tool": "news", "args": {"action": "poll"}}),
)


def ensure_news_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the continuous news poll job.

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
