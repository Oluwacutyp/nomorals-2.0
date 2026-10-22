"""Durable scheduler registrations for the finance family.

Extracted from the partner runtime's boot block (Pass #75) so the
registrations follow the same ``ensure_*`` pattern as every other job
family (immune, rooms, watchers, cadence, ...) and can be enumerated
—and binding-tested—without booting the whole runtime.

Jobs:

* ``finance alerts`` — every 15m, the price/money watchtower check;
  fired alerts are delivered through the Notifier by the tool itself
  (``deliver: True``).
* ``finance digest`` — Sundays 09:00 local, the weekly money digest;
  the tool publishes through the Notifier itself.

Idempotent by name; no-ops when the jobs already exist.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_finance_jobs", "FINANCE_ALERTS_JOB", "FINANCE_DIGEST_JOB"]

FINANCE_ALERTS_JOB = "finance alerts"
FINANCE_DIGEST_JOB = "finance digest"

_JOBS = (
    (FINANCE_ALERTS_JOB, "every 15m",
     {"tool": "finance_alert", "args": {"action": "check", "deliver": True}}),
    (FINANCE_DIGEST_JOB, "cron 0 9 * * SUN",
     {"tool": "finance_digest", "args": {}}),
)


def ensure_finance_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the finance watchtower + weekly digest jobs.

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
