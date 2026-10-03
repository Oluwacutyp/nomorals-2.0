"""L5 — missions: long-running autonomous goals that survive process death.

A mission is the unit of work that outlives a single agent run. It persists its
plan, its progress, and its budget to SQLite, so ``kill -9`` in the middle of an
eight-hour task costs one step rather than the whole goal.

Public surface::

    from nomorals.missions import MissionRunner, MissionStore, wired_runner

    store = MissionStore(db)
    runner = wired_runner(context)  # runner + os state-machine hook attached
    result = runner.start("research X and write a report", budget_wall=3600)

    # after a crash, on startup:
    runner.resume_all()
"""

from __future__ import annotations

from .idempotency import (
    COMPLETED as IDEMPOTENCY_COMPLETED,
)
from .idempotency import (
    FAILED as IDEMPOTENCY_FAILED,
)
from .idempotency import (
    RUNNING as IDEMPOTENCY_RUNNING,
)
from .idempotency import (
    DedupeTimeout,
    DedupResult,
    IdempotencyStore,
    create_mission_once,
    dedupe,
    idempotency_key,
    mission_idempotency_key,
    step_idempotency_key,
)
from .mission import (
    ACCEPTANCE_STATE_KEY,
    Checkpoint,
    Mission,
    MissionStatus,
    MissionStore,
    mission_liveness,
    normalize_acceptance,
)
from .progress import (
    STALL_AFTER_FAILURES,
    MissionMilestones,
    MissionWatchers,
    StallCode,
    clear_stall,
    estimate_eta,
    fmt_duration,
    record_stall,
    render_status_text,
)
from .runner import MissionResult, MissionRunner, StepOutcome
from .wiring import mission_acceptance, set_acceptance, wired_runner

__all__ = [
    "ACCEPTANCE_STATE_KEY",
    "Checkpoint",
    "DedupResult",
    "DedupeTimeout",
    "IDEMPOTENCY_COMPLETED",
    "IDEMPOTENCY_FAILED",
    "IDEMPOTENCY_RUNNING",
    "IdempotencyStore",
    "Mission",
    "MissionResult",
    "MissionRunner",
    "MissionStatus",
    "MissionStore",
    "MissionMilestones",
    "MissionWatchers",
    "STALL_AFTER_FAILURES",
    "StallCode",
    "StepOutcome",
    "clear_stall",
    "create_mission_once",
    "dedupe",
    "estimate_eta",
    "fmt_duration",
    "idempotency_key",
    "mission_acceptance",
    "mission_idempotency_key",
    "mission_liveness",
    "normalize_acceptance",
    "record_stall",
    "render_status_text",
    "set_acceptance",
    "step_idempotency_key",
    "wired_runner",
]
