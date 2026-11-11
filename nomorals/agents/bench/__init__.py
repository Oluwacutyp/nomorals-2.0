"""The multi-cursor bench: parallel coding cursors that can
never see each other's unfinished work.

- :mod:`.model` — the contracts (task specs with declared
  write-sets, state machines, verdict vocabulary), the runtime-
  created bench tables, and the monitor's append-only journal.
- :mod:`.monitor` — the Task Monitor (Lane 1): one-transaction
  task + file claiming over ``core/leases.py`` generations,
  promotion-gated readiness, heartbeat, and the watchdog that
  expires dead claims and fences their late submits.
- :mod:`.staging` — Lane 2: per-cursor staging areas (git
  worktree, plain-clone fallback) pinned to the promoted base,
  candidate capture as immutable commits, and the three read
  views (:class:`BenchViews`) that make a peer's unfinished
  work unreachable.
- :mod:`.integration` — Lane 2: the serial integration queue.
  Promotions land one at a time in submit order, exactly once
  (missions idempotency + tree/journal reconciliation), and a
  candidate that does not apply becomes a named ``conflicted``
  event — never a silent merge.

The verifier cursor is Lane 3; orchestrator/surface wiring is
Lane 4.
"""

from __future__ import annotations

from .integration import BenchIntegrator, promotion_key
from .model import (
    BenchError,
    BenchStore,
    TaskSpec,
    contract_kwargs,
    write_sets_overlap,
)
from .monitor import BenchMonitor
from .staging import BenchViews, StagingArea, StagingManager

__all__ = [
    "BenchError",
    "BenchIntegrator",
    "BenchMonitor",
    "BenchStore",
    "BenchViews",
    "StagingArea",
    "StagingManager",
    "TaskSpec",
    "contract_kwargs",
    "promotion_key",
    "write_sets_overlap",
]
