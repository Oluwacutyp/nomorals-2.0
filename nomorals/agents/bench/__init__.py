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
- :mod:`.verifier` — Lane 3: the verifier cursor. It takes
  exactly the task spec, the base revision, and the immutable
  candidate ref (the request type has no field for a builder's
  self-report), re-runs the acceptance and adjoining suites
  itself on a clean checkout, applies the per-class check
  table, and writes the verdict the integration gate consumes:
  nothing promotes without a ``supported`` verdict for the
  current candidate ref, and a verifier can never verify its
  own work (the delegation store's independence law).

Orchestrator/surface wiring is Lane 4.
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
from .verifier import BenchVerifier, VerifyRequest

__all__ = [
    "BenchError",
    "BenchIntegrator",
    "BenchMonitor",
    "BenchStore",
    "BenchVerifier",
    "BenchViews",
    "StagingArea",
    "StagingManager",
    "TaskSpec",
    "VerifyRequest",
    "contract_kwargs",
    "promotion_key",
    "write_sets_overlap",
]
