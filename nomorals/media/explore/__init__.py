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
persists. Autonomy (the durable scheduler) and the ``/explore``
board are P3; both call the entry points defined here unchanged.

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
from .instruments import (
                    InstrumentError,
                    VideoFinderInstrument,
                    WebSearchInstrument,
                    YouTubeSearchInstrument,
                    default_instruments,
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
    "ExploreStore",
    "InstrumentError",
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
    "evaluate_trends",
    "explore_for_context",
    "explore_status",
    "load_beats",
    "load_news_profile",
    "note_owner_activity",
    "run_exploration",
]
