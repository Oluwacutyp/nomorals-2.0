"""The autonomous nervous system — Devon's organs working without commands.

The user's mandate: "The system as a whole should be able to work
together without me tripping commands every now and then."

This package is how organs trigger each other through events:

* **idle** — system-wide idle detection. When nothing's happening, organs
  get their maintenance window (research, wisdom ingestion, memory
  consolidation).
* **weakness** — weakness detection → research → proposal → sandbox
  build/test. The system notices what's broken or missing, researches it,
  proposes a fix, and tests it before bothering the owner.
* **patterns** — time/pattern/interest modeling. She learns routines,
  predicts interests, acts proactively.
* **presence** — the "alive" layer. Time-aware, notices things, prepares
  ahead, surfaces serendipity without being asked. Serendipity is scored
  (novelty × relevance × unexpectedness), gated by a learned
  interruptibility score, and delivered as one batched digest.

Sweep upgrades (2026-10-10): idle runs in graduated stages
(shallow/deep/night) with inhibitors and an adaptive threshold;
weakness detection uses fuzzy error signatures, burn-rate severity,
new agent-failure kinds, and a fixed-weakness memory; patterns use
YAKE-style topic scoring, routine decay, and smoothed transitions;
the coordinator budgets per-organ slices with failure backoff and
renders human-readable cycle reports.

All coordination flows through the existing event bus
(``nomorals.core.events.global_bus``) and the organ event store
(``nomorals.organs``). No new parallel infrastructure — this is the
nervous system connecting organs that already exist.
"""

from __future__ import annotations

__all__ = [
    "idle",
    "coordinator",
    "weakness",
    "patterns",
    "presence",
]
