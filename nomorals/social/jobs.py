"""Durable scheduler registrations for the social family.

Follows the same ``ensure_*`` pattern as every other job family
(finance, price sources, news, ...): registrations live beside the
family they serve and are binding-tested without booting the whole
runtime.

Jobs:

* ``weekly content`` — Mondays 08:00 local, the weekly content
  pipeline (``social/content_pipeline.weekly_content`` via the
  ``weekly_content`` tool): research -> voice-shaped drafts -> queued
  for REVIEW at the audience's best windows. The pipeline never
  auto-posts; the job is NOT routine — its batch report is the point
  (the owner reviews and says the word).

Why not ``content_pipeline.install_weekly_content_job``? That
installer targets the OTHER scheduler (``nomorals/scheduler`` —
``register_action``/``schedule_cron``), which the partner runtime
does not hold. The runtime's durable scheduler
(``agents/scheduler``) runs tool payloads, so the pipeline is
installed here as a tool job, exactly like the finance family. The
cron-scheduler installer stays for that scheduler's consumers.

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_weekly_content_jobs", "WEEKLY_CONTENT_JOB"]

WEEKLY_CONTENT_JOB = "weekly content"

_JOBS = (
    (WEEKLY_CONTENT_JOB, "cron 0 8 * * MON",
     {"tool": "weekly_content", "args": {"niche": ""}}),
)


def ensure_weekly_content_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the weekly content pipeline job.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=False)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
