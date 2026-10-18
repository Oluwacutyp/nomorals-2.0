"""Autonomy-that-acts layer (audit: AUTONOMY_THAT_ACTS).

Thesis: autonomy = maintaining objectives, noticing changes, choosing
the next useful action, executing, verifying, owning follow-through.
Silence from the owner must NOT empty the work queue.

This package is the AUTONOMY layer on top of existing machinery — it
does NOT re-implement scheduling (``scheduler/facade.py`` owns that),
idempotency/delivery seams (infrastructure build), or the organs
themselves. It adds:

* :mod:`nomorals.agency.initiatives` — one durable initiative backlog
  on SQLite (NOT a new task DB: initiatives link to objectives and to
  existing goals/missions via ``objective`` / ``parent_id`` / lineage).
* :mod:`nomorals.agency.deliberation` — deliberation loop, separate
  from execution: cheap deterministic ranking, aging, exploration cap,
  starvation guard.
* :mod:`nomorals.agency.adequacy` — evidence-supported gap-resolution
  gate. A nonempty synthesis is NOT proof.
* :mod:`nomorals.agency.digestion` — source → claims → three deltas
  (knowledge / decision / action). Action delta NONE is valid.
* :mod:`nomorals.agency.followthrough` — waiting states with
  correlation ID + trigger + review + expiry; stop reasons.
* :mod:`nomorals.agency.domains` — closed-loop lifecycle validators
  and scoring for deals / weather / news / social / leads / gigs.
* :mod:`nomorals.agency.views` — Now / Since last check / Why views
  + reconciled end-of-day agenda.
* :mod:`nomorals.agency.supervision` — boot reconcile: lease recovery,
  missed-check coalescing, idempotent watch re-registration.

Invariant (enforced in :class:`InitiativeStore`): every nonterminal
initiative has an executable next action, a timed wake-up, or a
specific blocker. "Monitoring" with no next check is NOT active.
"""
from __future__ import annotations

from .initiatives import Initiative, InitiativeStore
from .deliberation import DeliberationLoop
from .adequacy import assess_gap_adequacy
from .digestion import Claim, ThreeDeltas, digest
from .followthrough import WaitRecord, PlanTracker, STOP_REASONS
from .domains import (
    LEAD_TRANSITIONS, GIG_POST_SUBMISSION, advance_lead, score_gig,
    evaluate_deal, classify_weather_item, cluster_news,
)
from .views import now_view, since_view, why_view, reconciled_agenda
from .supervision import BootSupervisor
from .knowledge import KnowledgeStore
from .rotation import RotationScheduler, SystemSlot, WorkResult
from .systems import ActiveSystems, SYSTEM_NAMES, SYSTEM_WEIGHTS

__all__ = [
    "Initiative", "InitiativeStore", "DeliberationLoop",
    "assess_gap_adequacy", "Claim", "ThreeDeltas", "digest",
    "WaitRecord", "PlanTracker", "STOP_REASONS",
    "LEAD_TRANSITIONS", "GIG_POST_SUBMISSION", "advance_lead",
    "score_gig", "evaluate_deal", "classify_weather_item",
    "cluster_news", "now_view", "since_view", "why_view",
    "reconciled_agenda", "BootSupervisor",
    "KnowledgeStore", "RotationScheduler", "SystemSlot", "WorkResult",
    "ActiveSystems", "SYSTEM_NAMES", "SYSTEM_WEIGHTS",
]
