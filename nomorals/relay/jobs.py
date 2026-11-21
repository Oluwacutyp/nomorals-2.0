"""The ticket relay's scheduled sync job.

Mirrors :mod:`nomorals.commerce.dunning`'s ensure shape: register
the durable hourly tick job by name, idempotent — an existing job
at the requested cadence is returned as-is, a stale one is
replaced. The job runs the ``__ticket_relay_sync__`` tool, which
polls every open ticket's issue and posts new comments and state
changes back to Discord. Best-effort: never raises.
"""
from __future__ import annotations

import contextlib
from typing import Any

__all__ = [
    "TICKET_RELAY_JOB",
    "TICKET_RELAY_SPEC",
    "TICKET_RELAY_TOOL",
    "ensure_ticket_relay_job",
]

TICKET_RELAY_JOB = "ticket relay sync"
TICKET_RELAY_SPEC = "every 1h"
TICKET_RELAY_TOOL = "__ticket_relay_sync__"


def ensure_ticket_relay_job(scheduler: Any, *,
                            spec: str = TICKET_RELAY_SPEC) -> Any:
    """Register the durable ticket-relay sync tick job."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == TICKET_RELAY_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    _reconcile_overlap_skip(scheduler, job)
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        job = scheduler.add(TICKET_RELAY_JOB, spec, "tool",
                            {"tool": TICKET_RELAY_TOOL, "args": {}},
                            overlap_policy="skip")
        # add() is idempotent by exact replay: an existing row at the
        # same spec comes back deduped carrying its OLD policy.
        _reconcile_overlap_skip(scheduler, job)
        return job
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def _reconcile_overlap_skip(scheduler: Any, job: Any) -> None:
    """Force a persisted job row onto ``overlap_policy=skip``.

    Best-effort, never raises.  A sync that wedges past the 300s
    wall must not stack a second worker behind the abandoned first
    one (2026-11-19 stall: the news poll kept "concurrent" and went
    1 → 2 abandoned workers).
    """
    try:
        row = job if isinstance(job, dict) else None
        if row is None:
            return
        if (row.get("overlap_policy") or "concurrent") != "skip":
            scheduler.set_overlap_policy(row["id"], "skip")
    except Exception:  # noqa: BLE001 - reconciliation is best-effort
        pass
