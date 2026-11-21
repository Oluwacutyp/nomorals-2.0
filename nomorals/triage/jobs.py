"""The triage scheduled sync job.

Mirrors :mod:`nomorals.relay.jobs`' ensure shape: register the
durable hourly tick job by name, idempotent — an existing job at the
requested cadence is returned as-is, a stale one is replaced. The job
runs the ``__triage_sync__`` tool, which syncs every tracked repo's
mirror and runs the code-rung triage pass (the model rung only when
the brain is up). Best-effort: never raises.
"""
from __future__ import annotations

import contextlib
from typing import Any

__all__ = [
    "TRIAGE_JOB",
    "TRIAGE_SPEC",
    "TRIAGE_SYNC_TOOL",
    "ensure_triage_job",
]

TRIAGE_JOB = "triage sync"
TRIAGE_SPEC = "every 1h"
TRIAGE_SYNC_TOOL = "__triage_sync__"


def ensure_triage_job(scheduler: Any, *,
                      spec: str = TRIAGE_SPEC) -> Any:
    """Register the durable triage sync tick job."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == TRIAGE_JOB:
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
        job = scheduler.add(TRIAGE_JOB, spec, "tool",
                            {"tool": TRIAGE_SYNC_TOOL, "args": {}},
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
