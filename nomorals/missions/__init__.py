"""L6 — missions: long-running autonomous goals that survive process death.

A mission is the unit of work that outlives a single agent run. It persists its
plan, its progress, and its budget to SQLite, so ``kill -9`` in the middle of an
eight-hour task costs one step rather than the whole goal.

Public surface::

    from nomorals.missions import MissionRunner, MissionStore

    store = MissionStore(db)
    runner = MissionRunner(context)
    result = runner.start("research X and write a report", budget_wall=3600)

    # after a crash, on startup:
    runner.resume_all()
"""

from __future__ import annotations

from .mission import Checkpoint, Mission, MissionStatus, MissionStore
from .runner import MissionResult, MissionRunner, StepOutcome

__all__ = [
    "Checkpoint",
    "Mission",
    "MissionResult",
    "MissionRunner",
    "MissionStatus",
    "MissionStore",
    "StepOutcome",
]
