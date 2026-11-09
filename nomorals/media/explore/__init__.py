"""Exploration — Devon watches the field she fetches from.

WS4 Pass 1 (the substrate): an :class:`ExploreStore` that remembers
sweeps as append-only observations with provenance, a sweep engine
over the existing search organs (no new fetchers), and candidate
intake for the channels/topics discovery surfaces — the source
registry's proven loop aimed at content instead of hosts.

Trend identification (velocity, promotion rules, claims) is P2 and
deliberately absent here: this package records facts and renders no
verdicts. Autonomy (the durable scheduler) and the ``/explore``
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

__all__ = [
    "EXPLORE_SEED_BEATS",
    "ExploreStore",
    "InstrumentError",
    "SweepEngine",
    "VideoFinderInstrument",
    "WebSearchInstrument",
    "YouTubeSearchInstrument",
    "beats_from_profile",
    "collect_watch_topics",
    "default_explore_store",
    "default_instruments",
    "explore_for_context",
    "explore_status",
    "load_beats",
    "load_news_profile",
    "note_owner_activity",
    "run_exploration",
]
