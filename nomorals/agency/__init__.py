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
* :mod:`nomorals.agency.adapters` — real organ adapters for the domain
  loops (deals / weather / news / social / leads / gigs / planner),
  organs injected by the caller, never imported (agency is L0).
* :mod:`nomorals.agency.proactive` — proactive triggers: systems
  notice events / changes / silence and initiate work through the
  initiative lifecycle + rotation, with durable dedupe state.
* :mod:`nomorals.agency.expansion` — second-wave autonomy: the REST
  of the repo (media, code, books, games, commerce, health, travel,
  property, legal, finance, learn, marketing, community, characters,
  documents, datasci, sourcehealth) on the SAME scheduler and stores
  as the core systems — one capacity pool, no parallel system.
* :mod:`nomorals.agency.sourcehealth` — per-source live verification
  + circuit breaker. A source that doesn't work is worse than no
  source: only verified-live sources belong in an active chain.
* :mod:`nomorals.agency.sources_registry` — the single-sourced
  registry of every verifiable source, collected from the real
  modules that own the lists (no duplicated URL lists).

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
from .adapters import (
    AgencyAuthority, DealsAdapter, WeatherAdapter, NewsAdapter,
    SocialAdapter, LeadsAdapter, GigsAdapter, PlannerAdapter,
    build_adapters,
)
from .proactive import Trigger, fire_triggers, scan_proactive
from .expansion import ExpansionSystems, EXPANSION_SYSTEMS, EXPANSION_WEIGHTS
from .sourcehealth import SourceHealth, SourceRecord, SourceSpec, check_one
from .sources_registry import collect as collect_sources, spec_by_key

__all__ = [
    "Initiative", "InitiativeStore", "DeliberationLoop",
    "assess_gap_adequacy", "Claim", "ThreeDeltas", "digest",
    "WaitRecord", "PlanTracker", "STOP_REASONS",
    "LEAD_TRANSITIONS", "GIG_POST_SUBMISSION", "advance_lead", "score_gig",
    "evaluate_deal", "classify_weather_item", "cluster_news",
    "now_view", "since_view", "why_view", "reconciled_agenda",
    "BootSupervisor",
    "KnowledgeStore", "RotationScheduler", "SystemSlot", "WorkResult",
    "ActiveSystems", "SYSTEM_NAMES", "SYSTEM_WEIGHTS",
    "AgencyAuthority", "DealsAdapter", "WeatherAdapter", "NewsAdapter",
    "SocialAdapter", "LeadsAdapter", "GigsAdapter", "PlannerAdapter",
    "build_adapters", "Trigger", "fire_triggers", "scan_proactive",
    "ExpansionSystems", "EXPANSION_SYSTEMS", "EXPANSION_WEIGHTS",
    "SourceHealth", "SourceRecord", "SourceSpec", "check_one",
    "collect_sources", "spec_by_key",
]
