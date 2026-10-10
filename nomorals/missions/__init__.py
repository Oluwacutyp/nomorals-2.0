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

from .golden import (
    GOLDEN_BASELINES,
    GOLDEN_MISSIONS,
    CompensateFn,
    GoldenContext,
    GoldenMission,
    GoldenResult,
    GoldenRunner,
    GoldenStep,
    check_regression,
    list_golden_missions,
    normalize_output,
    render_golden_report,
)
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
    IdempotencyConflict,
    IdempotencyStore,
    create_mission_once,
    dedupe,
    idempotency_key,
    mission_idempotency_key,
    step_idempotency_key,
)
from .mission import (
    ACCEPTANCE_STATE_KEY,
    MISSION_TEMPLATES,
    Checkpoint,
    Mission,
    MissionStatus,
    MissionStore,
    list_mission_templates,
    mission_liveness,
    normalize_acceptance,
)
from .progress import (
    STALL_AFTER_FAILURES,
    STATUS_STYLES,
    MissionMilestones,
    MissionWatchers,
    StallCode,
    box_lines,
    clear_stall,
    estimate_eta,
    eta_breakdown,
    fmt_duration,
    record_stall,
    render_mission_card,
    render_mission_table,
    render_progress_bar,
    render_result_card,
    render_sparkline,
    render_status_card,
    render_status_text,
)
from .runner import (
    ERROR_PERMANENT,
    ERROR_RATE_LIMIT,
    ERROR_SERVER,
    ERROR_TRANSIENT,
    MissionResult,
    MissionRunner,
    StepOutcome,
    render_plan_text,
)
from .wiring import mission_acceptance, preview_mission, set_acceptance, wired_runner

__all__ = [
    "ACCEPTANCE_STATE_KEY",
    "Checkpoint",
    "CompensateFn",
    "DedupResult",
    "DedupeTimeout",
    "ERROR_PERMANENT",
    "ERROR_RATE_LIMIT",
    "ERROR_SERVER",
    "ERROR_TRANSIENT",
    "GOLDEN_BASELINES",
    "GOLDEN_MISSIONS",
    "GoldenContext",
    "GoldenMission",
    "GoldenResult",
    "GoldenRunner",
    "GoldenStep",
    "IDEMPOTENCY_COMPLETED",
    "IDEMPOTENCY_FAILED",
    "IDEMPOTENCY_RUNNING",
    "MISSION_TEMPLATES",
    "IdempotencyConflict",
    "IdempotencyStore",
    "Mission",
    "MissionResult",
    "MissionRunner",
    "MissionStatus",
    "MissionStore",
    "MissionMilestones",
    "MissionWatchers",
    "STALL_AFTER_FAILURES",
    "STATUS_STYLES",
    "StallCode",
    "StepOutcome",
    "box_lines",
    "check_regression",
    "clear_stall",
    "create_mission_once",
    "dedupe",
    "estimate_eta",
    "eta_breakdown",
    "fmt_duration",
    "idempotency_key",
    "list_golden_missions",
    "list_mission_templates",
    "mission_acceptance",
    "mission_idempotency_key",
    "mission_liveness",
    "normalize_acceptance",
    "normalize_output",
    "preview_mission",
    "record_stall",
    "render_golden_report",
    "render_mission_card",
    "render_mission_table",
    "render_plan_text",
    "render_progress_bar",
    "render_result_card",
    "render_sparkline",
    "render_status_card",
    "render_status_text",
    "set_acceptance",
    "step_idempotency_key",
    "wired_runner",
]
