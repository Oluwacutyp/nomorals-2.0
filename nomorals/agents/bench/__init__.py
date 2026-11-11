"""The multi-cursor bench: parallel coding cursors that can
never see each other's unfinished work.

- :mod:`.model` — the contracts (task specs with declared
  write-sets, state machines, verdict vocabulary), the runtime-
  created bench tables, and the monitor's append-only journal.
- :mod:`.monitor` — the Task Monitor (Lane 1): one-transaction
  task + file claiming over ``core/leases.py`` generations,
  promotion-gated readiness, heartbeat, and the watchdog that
  expires dead claims and fences their late submits.

Staging worktrees + the integration queue are Lane 2; the
verifier cursor is Lane 3. The seams they consume are the
``submitted`` state and ``BenchMonitor.promotion_candidates``.
"""

from __future__ import annotations

from .model import (
    BenchError,
    BenchStore,
    TaskSpec,
    contract_kwargs,
    write_sets_overlap,
)
from .monitor import BenchMonitor

__all__ = [
    "BenchError",
    "BenchMonitor",
    "BenchStore",
    "TaskSpec",
    "contract_kwargs",
    "write_sets_overlap",
]
