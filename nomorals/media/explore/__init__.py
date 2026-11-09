"""Exploration — Devon watches the field she fetches from.

WS4 Pass 1 (the substrate): an :class:`ExploreStore` that remembers
sweeps as append-only observations with provenance, a sweep engine
over the existing search organs (no new fetchers), and candidate
intake for the channels/topics discovery surfaces — the source
registry's proven loop aimed at content instead of hosts.

Trend identification is P2 (``trends.py``): cross-sweep
differencing, velocity against each beat's own baseline, the
promotion rule (observation → spike → corroborated trend claim),
freshness decay, and candidate verdicts — claims are written only
through the evaluator, in the digestion Claim shape the store
persists. Autonomy is P3 (``jobs.py``): two durable sweeps a day
on the owner's clock end in :func:`run_sweep_cycle` (gates →
sweep → evaluation); the ``/explore`` board (``board.py``) and
the morning briefing render what the loop records.

    from nomorals.media.explore import run_exploration, explore_status
    report = run_exploration(context, trigger="manual")
"""

from __future__ import annotations

from .beats import (
                    EXPLORE_SEED_BEATS,
                    beats_from_profile,
                    collect_watch_topics,
                    load_beats,
                    load_news_profile,
)
from .engine import (
                    SweepEngine,
                    default_explore_store,
                    explore_for_context,
                    explore_status,
                    note_owner_activity,
                    run_exploration,
)
from .board import (
                    PROXY_DEPENDENCY_LINE,
                    render_explore_board,
                    render_explore_candidates,
                    render_explore_trends,
                    render_sweep_cycle,
)
from .instruments import (
                    InstrumentError,
                    VideoFinderInstrument,
                    WebSearchInstrument,
                    YouTubeSearchInstrument,
                    default_instruments,
)
from .jobs import (
                    EXPLORE_SWEEP_JOB_EVENING,
                    EXPLORE_SWEEP_JOB_MORNING,
                    EXPLORE_SWEEP_JOBS,
                    MAX_SWEEPS_PER_DAY,
                    ensure_explore_jobs,
                    run_sweep_cycle,
)
from .store import ExploreStore
from .trends import (
                    CLAIM_KINDS,
                    CLAIM_SPIKE,
                    CLAIM_TREND,
                    CLAIM_UNVERIFIED,
                    claim_standing,
                    current_claims,
                    evaluate_trends,
)

__all__ = [
    "CLAIM_KINDS",
    "CLAIM_SPIKE",
    "CLAIM_TREND",
    "CLAIM_UNVERIFIED",
    "EXPLORE_SEED_BEATS",
    "EXPLORE_SWEEP_JOB_EVENING",
    "EXPLORE_SWEEP_JOB_MORNING",
    "EXPLORE_SWEEP_JOBS",
    "ExploreStore",
    "InstrumentError",
    "MAX_SWEEPS_PER_DAY",
    "PROXY_DEPENDENCY_LINE",
    "SweepEngine",
    "VideoFinderInstrument",
    "WebSearchInstrument",
    "YouTubeSearchInstrument",
    "beats_from_profile",
    "claim_standing",
    "collect_watch_topics",
    "current_claims",
    "default_explore_store",
    "default_instruments",
    "ensure_explore_jobs",
    "evaluate_trends",
    "explore_for_context",
    "explore_status",
    "load_beats",
    "load_news_profile",
    "note_owner_activity",
    "render_explore_board",
    "render_explore_candidates",
    "render_explore_trends",
    "render_sweep_cycle",
    "run_exploration",
    "run_sweep_cycle",
]
