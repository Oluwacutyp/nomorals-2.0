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
* ``social metrics refresh`` — every 6h, the measurement spine
  (``social/metrics.refresh_post_metrics`` via the
  ``social_metrics_refresh`` tool): posted rows get their REAL
  engagement re-read from their platform adapters and written back,
  and first outcomes feed the briefs calibration loop. Marked
  ``routine``: refreshed numbers reach the owner through the windows
  and windows-driven surfaces that read them, not through job
  alerts — but a failed run still records as failed.
* ``social engagement poll`` — every 6h, the engagement loop
  (``social/engagement.poll_engagement`` via the
  ``social_engagement_poll`` tool): replies/comments on posted
  rows are read back from capable adapters into the engagement
  store for the /engage board, and per-post engagement volume
  feeds the same briefs outcome loop. Marked ``routine`` for the
  same reason — the board and the briefing carry it.

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

__all__ = [
    "ensure_weekly_content_jobs", "WEEKLY_CONTENT_JOB",
    "ensure_social_metrics_jobs", "METRICS_REFRESH_JOB",
    "ensure_social_engagement_jobs", "ENGAGEMENT_POLL_JOB",
]

WEEKLY_CONTENT_JOB = "weekly content"
METRICS_REFRESH_JOB = "social metrics refresh"
ENGAGEMENT_POLL_JOB = "social engagement poll"

_JOBS = (
    (WEEKLY_CONTENT_JOB, "cron 0 8 * * MON",
     {"tool": "weekly_content", "args": {"niche": ""}}),
)

_METRICS_JOBS = (
    (METRICS_REFRESH_JOB, "every 6h",
     {"tool": "social_metrics_refresh", "args": {}}),
)

_ENGAGEMENT_JOBS = (
    (ENGAGEMENT_POLL_JOB, "every 6h",
     {"tool": "social_engagement_poll", "args": {}}),
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


def ensure_social_metrics_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the measurement-spine refresh job.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _METRICS_JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=True)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out


def ensure_social_engagement_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the engagement-poll job (WS3 P8).

    Same 6h cadence as the metrics refresh, deliberately: the two
    spines read the same posted rows on the same rhythm, and
    engagement arrives over hours, not minutes — a tighter loop
    would spend platform quota to learn nothing sooner. Marked
    ``routine``: what needs the owner reaches him through the
    /engage board and the morning briefing, not job alerts — but
    a failed run still records as failed.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _ENGAGEMENT_JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=True)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
