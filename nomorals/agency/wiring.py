"""Startup wiring for the agency layer (audit §9 step 1).

``boot_agency(db, ...)`` is the ONE entrypoint the partner runtime
calls on boot (best-effort, never breaks boot). It runs the
supervision reconcile and returns the store + supervisor so the
runtime can keep them. Idempotent: safe to call on every boot.
"""
from __future__ import annotations

from typing import Any

from .initiatives import InitiativeStore
from .supervision import BootSupervisor
from .systems import ActiveSystems

__all__ = ["boot_agency"]


def boot_agency(db: Any, *, now: float | None = None,
                scheduler_facade: Any | None = None,
                watches: list[dict[str, Any]] | None = None,
                research_organ: Any | None = None,
                wisdom_organ: Any | None = None,
                impossible: Any | None = None
                ) -> dict[str, Any]:
    store = InitiativeStore(db)
    supervisor = BootSupervisor(store, scheduler_facade=scheduler_facade)
    report = supervisor.reconcile(now=now, watches=watches)
    # ALL AI systems actively working: seed one standing initiative per
    # system (idempotent via dedupe_key) so every system enters the
    # rotation with a live lifecycle on every boot.
    systems = ActiveSystems(db=db, store=store,
                            research_organ=research_organ,
                            wisdom_organ=wisdom_organ,
                            impossible=impossible)
    seeded = systems.seed_initiatives(now=now)
    return {"store": store, "supervisor": supervisor, "boot": report,
            "systems": systems, "seeded_initiatives": len(seeded)}
