"""Startup wiring for the agency layer (audit §9 step 1).

``boot_agency(db, ...)`` is the ONE entrypoint the partner runtime
calls on boot (best-effort, never breaks boot). It runs the
supervision reconcile and returns the store + supervisor so the
runtime can keep them. Idempotent: safe to call on every boot.

The worker SHOWS UP: when a scheduler is given, boot registers the
durable ``agency-shift`` job (``__agency_shift__`` tool, every 30
min) so shifts run on schedule, not only when poked.
"""
from __future__ import annotations

from typing import Any

from .adapters import AgencyAuthority, build_adapters
from .initiatives import InitiativeStore
from .supervision import BootSupervisor
from .systems import ActiveSystems

__all__ = [
    "boot_agency", "ensure_agency_shift_job",
    "AGENCY_SHIFT_JOB", "AGENCY_SHIFT_TOOL", "AGENCY_SHIFT_SPEC",
]

#: Durable shift job: name, tool payload, default cadence.
AGENCY_SHIFT_JOB = "agency-shift"
AGENCY_SHIFT_TOOL = "__agency_shift__"
AGENCY_SHIFT_SPEC = "*/30 * * * *"


def ensure_agency_shift_job(scheduler: Any, *,
                            spec: str = AGENCY_SHIFT_SPEC) -> Any:
    """Register the agency-shift job. Idempotent by job name.

    A stale job with a different interval is replaced. Best-effort:
    never raises (a missing scheduler just means no scheduled
    shifts — the runtime can still call ``work_shift`` directly).
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = job.get("name", "") if isinstance(job, dict) \
                else getattr(job, "name", "")
            if name == AGENCY_SHIFT_JOB:
                job_spec = job.get("spec", "") if isinstance(job, dict) \
                    else getattr(job, "spec", "")
                if job_spec == spec:
                    return job
                job_id = job.get("id", "") if isinstance(job, dict) \
                    else getattr(job, "id", "")
                try:
                    scheduler.remove(job_id)
                except Exception:  # noqa: BLE001 - stale job removal
                    pass
                break
        return scheduler.add(AGENCY_SHIFT_JOB, spec, "tool",
                             {"tool": AGENCY_SHIFT_TOOL})
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def boot_agency(db: Any, *, now: float | None = None,
                scheduler_facade: Any | None = None,
                watches: list[dict[str, Any]] | None = None,
                research_organ: Any | None = None,
                wisdom_organ: Any | None = None,
                impossible: Any | None = None,
                adapters: dict[str, Any] | None = None,
                authority: Any | None = None,
                auto_wire: bool = True,
                context: Any | None = None,
                account_manager: Any | None = None,
                places: list[str] | None = None,
                inbox: Any | None = None,
                scheduler: Any | None = None,
                shift_spec: str = AGENCY_SHIFT_SPEC,
                ) -> dict[str, Any]:
    store = InitiativeStore(db)
    supervisor = BootSupervisor(store, scheduler_facade=scheduler_facade)
    report = supervisor.reconcile(now=now, watches=watches)
    # ALL AI systems actively working: seed one standing initiative per
    # system (idempotent via dedupe_key) so every system enters the
    # rotation with a live lifecycle on every boot.
    if adapters is None and auto_wire:
        # Assembly only: no organs are constructed here (layering —
        # organs are injected by the caller). Real organs arrive via
        # ``adapters=`` (see the shift tool in tools/autonomy.py).
        adapters = build_adapters(places=places, inbox=inbox)
    if authority is None:
        authority = AgencyAuthority()
    systems = ActiveSystems(db=db, store=store,
                            research_organ=research_organ,
                            wisdom_organ=wisdom_organ,
                            impossible=impossible,
                            adapters=adapters,
                            authority=authority)
    seeded = systems.seed_initiatives(now=now)
    # The worker SHOWS UP: register the durable shift job so shifts run
    # on schedule, not only when poked. Best-effort, never breaks boot.
    shift_job = None
    if scheduler is not None:
        shift_job = ensure_agency_shift_job(scheduler, spec=shift_spec)
    return {"store": store, "supervisor": supervisor, "boot": report,
            "systems": systems, "seeded_initiatives": len(seeded),
            "shift_job": shift_job}
