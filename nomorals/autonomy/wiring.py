"""Composition root for the autonomy satellite cluster (AA-35).

The six 2026-10-16 autonomy additions — the unified scheduler, the
budget governor, the owner rhythm, the why store, the preemption log,
prep-usage tracking — plus the two stateless function satellites
(:mod:`interruption` worth assessment, :mod:`model_weakness` learning
targets) were built as separate modules and then never assembled:
``UnifiedScheduler`` accepted its collaborators duck-typed and nothing
in production ever constructed the real ones, so no firing was ever
gated by the real rhythm/budget or explained into a real why store.

This module is the one place the cluster is assembled:

* :func:`build_stack` constructs the REAL satellites, persists their
  state under ``<base>/autonomy/``, and passes them to
  :class:`~nomorals.autonomy.unified.UnifiedScheduler` — replacing the
  duck-typed stand-ins its constructor used to wait for.
* The stateless satellites need no construction; they are re-exported
  here so the composition root names the whole cluster in one place
  (``focus`` injects ``interruption.assess_worth`` the same way).

Consumers: ``nm why`` (cmdline) reads the why store and runs due work
through the stack's scheduler.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import BudgetGovernor
from .interruption import assess_worth
from .model_weakness import learning_targets, record_model_weakness
from .preemption import PreemptionLog
from .rhythm import OwnerRhythm
from .unified import UnifiedScheduler
from .usage import UsageTracker
from .why import WhyStore

__all__ = [
    "AutonomyStack",
    "build_stack",
    "build_unified_scheduler",
    "assess_worth",
    "learning_targets",
    "record_model_weakness",
]


@dataclass
class AutonomyStack:
    """The assembled cluster: one scheduler + its real satellites."""

    base_dir: Path
    scheduler: UnifiedScheduler
    why_store: WhyStore
    preemption_log: PreemptionLog
    rhythm: OwnerRhythm
    budget: BudgetGovernor
    usage: UsageTracker

    def schedule_mission(self, mission: Any, *, kind: str = "") -> str:
        """Schedule a mission on the unified scheduler AND record the
        prep with the usage tracker, so "did the prepared work get
        used" is measurable from the same composition root."""
        mission_id = self.scheduler.schedule_mission(mission)
        record = self.scheduler.get_mission(mission_id) or {}
        self.usage.record_prep(
            {"title": record.get("name") or record.get("goal") or mission_id,
             "kind": kind or "mission",
             "detail": str(record.get("goal") or "")})
        return mission_id


def stack_paths(base_dir: str | Path) -> dict[str, Path]:
    """Where the cluster's durable state lives under ``base_dir``."""
    root = Path(base_dir).expanduser() / "autonomy"
    return {
        "root": root,
        "queue": root / "unified_queue.json",
        "why": root / "why.jsonl",
        "preemptions": root / "preemptions.jsonl",
        "usage": root / "usage.json",
    }


def build_stack(base_dir: str | Path, *,
                clock: Callable[[], float] | None = None) -> AutonomyStack:
    """Assemble the cluster under ``base_dir`` (the nomorals home).

    The satellites share the scheduler's clock so a test (or a
    replayed boot) sees one consistent "now" across rhythm gates,
    budget days, and why records.
    """
    paths = stack_paths(base_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    tick = clock or time.time
    why_store = WhyStore(paths["why"], clock=tick)
    preemption_log = PreemptionLog(paths["preemptions"], clock=tick)
    rhythm = OwnerRhythm(clock=tick)
    budget = BudgetGovernor(clock=tick)
    usage = UsageTracker(paths["usage"], clock=tick)
    scheduler = UnifiedScheduler(
        paths["queue"],
        why_store=why_store,
        rhythm=rhythm,
        budget=budget,
        preemption=preemption_log,
        clock=tick,
    )
    return AutonomyStack(
        base_dir=Path(base_dir).expanduser(),
        scheduler=scheduler,
        why_store=why_store,
        preemption_log=preemption_log,
        rhythm=rhythm,
        budget=budget,
        usage=usage,
    )


def build_unified_scheduler(base_dir: str | Path, *,
                            clock: Callable[[], float] | None = None,
                            ) -> UnifiedScheduler:
    """Just the scheduler, with the real satellites attached."""
    return build_stack(base_dir, clock=clock).scheduler
