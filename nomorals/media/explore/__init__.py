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
sweep → evaluation → niche discovery); the ``/explore`` board
(``board.py``) and the morning briefing render what the loop
records. Niche discovery is P4 (``niches.py``): clusters of the
field become niche PROPOSALS with evidence packs in the shared
overlay beside the code registry; the effective niche set is
floor ∪ adopted (:func:`effective_niches`), and adoption is the
owner's act — it scaffold-generates a draft plugin that goes
live only through human review, never by discovery. The WS3
contracts are P5 (``feed.py``): current claims project into the
content pipeline's findings shape, content consumers read the
shared niche registry through the adapter, exemplar media is
cited by WS1 artifact reference, and measured post outcomes are
recorded back against the claims they descend from.

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
                    render_explore_niches,
                    render_explore_trends,
                    render_niche_act,
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
from .niches import (
                    adopt_niche,
                    discover_niches,
                    dismiss_niche,
                    drafts_dir,
                    effective_niches,
                    floor_niches,
                    overlay_plugin,
                    proposal_board_order,
)
from .feed import (
                    claim_exemplars,
                    claim_to_finding,
                    claims_as_findings,
                    drafting_niches,
                    drafting_plugin_resolver,
                    feed_claim_outcome,
                    make_claims_feed,
                    niche_drafting_context,
)
from .store import (
                    NICHE_ADOPTED,
                    NICHE_DISMISSED,
                    NICHE_PROPOSED,
                    ExploreStore,
)
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
    "NICHE_ADOPTED",
    "NICHE_DISMISSED",
    "NICHE_PROPOSED",
    "PROXY_DEPENDENCY_LINE",
    "SweepEngine",
    "VideoFinderInstrument",
    "WebSearchInstrument",
    "YouTubeSearchInstrument",
    "adopt_niche",
    "beats_from_profile",
    "claim_exemplars",
    "claim_standing",
    "claim_to_finding",
    "claims_as_findings",
    "collect_watch_topics",
    "current_claims",
    "default_explore_store",
    "default_instruments",
    "discover_niches",
    "dismiss_niche",
    "drafting_niches",
    "drafting_plugin_resolver",
    "drafts_dir",
    "effective_niches",
    "ensure_explore_jobs",
    "evaluate_trends",
    "explore_for_context",
    "explore_status",
    "feed_claim_outcome",
    "floor_niches",
    "load_beats",
    "load_news_profile",
    "make_claims_feed",
    "niche_drafting_context",
    "note_owner_activity",
    "overlay_plugin",
    "proposal_board_order",
    "render_explore_board",
    "render_explore_candidates",
    "render_explore_niches",
    "render_explore_trends",
    "render_niche_act",
    "render_sweep_cycle",
    "run_exploration",
    "run_sweep_cycle",
]
