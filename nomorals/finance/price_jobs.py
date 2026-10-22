"""Durable scheduler registration for the price-source health loop
(pass #91).

Sibling of ``media/source_jobs.py`` (the download/music/book/
web-search classes live in the ONE media registry, so ONE job
covers them) — the price class gained mirrored machinery in its
own home (``integrations/market_data.py``; see that module for the
architecture decision), so it gets its own ``ensure_*`` here,
following the same pattern as every other job family.

Job:

* ``price source health`` — every 6h, re-verify every keyless
  price provider through the price layer's own health loop
  (``market_data.verify_price_sources``): one bounded real quote
  fetch per provider for a liquid symbol; keyed providers are
  never probed (quota) and are judged by real-fetch outcomes.
  Marked ``routine`` deliberately: the loop is self-reporting
  bookkeeping — its results land in the price status surface
  (``price_sources`` tool / ``market_data.source_status()``), not
  in owner alerts — but a FAILED run still records as a failed
  run (pass #71 semantics).

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_price_source_health_jobs", "PRICE_SOURCE_HEALTH_JOB"]

PRICE_SOURCE_HEALTH_JOB = "price source health"

_JOBS = (
    (PRICE_SOURCE_HEALTH_JOB, "every 6h",
     {"tool": "price_sources", "args": {"action": "health"}}),
)


def ensure_price_source_health_jobs(
        scheduler: Any) -> list[dict[str, Any]]:
    """Register the price-source health loop job.

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
