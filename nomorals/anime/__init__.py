"""Anime & manga organ — lookup, schedules, pictures, favorites.

Devon's own anime suite (FEI gap wave B, owner order 2026-11-05):
title/character/staff/studio records, episode lists, the weekly
airing grid and next-episode countdowns, rankings, recommendations,
random picks, quotes, image pools and reverse image identification —
blended into her research/media organs rather than bolted on.

Data comes from public sources treated as SEEDS, never a closed
list: AniList (GraphQL) is the primary record source, Jikan
(MyAnimeList) the second seed and cross-check, waifu.pics /
nekos.best feed the image pools, trace.moe identifies a frame,
animechan serves quotes.  Every client validates the response
shape it reads — a changed API shape is a named failure, never a
crash and never a fabricated record.
"""

from .service import AnimeService, Outcome
from .sources import (
    AniListClient,
    ImagePoolClient,
    JikanClient,
    QuoteClient,
    TraceMoeClient,
)

__all__ = [
    "AniListClient",
    "AnimeService",
    "ImagePoolClient",
    "JikanClient",
    "Outcome",
    "QuoteClient",
    "TraceMoeClient",
]
