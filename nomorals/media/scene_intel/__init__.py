"""Movie scene intelligence: segment → track → score → character reels.

High-level API: analyze_movie() runs the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .characters import CharacterProfile, build_timelines, character_reel
from .score import ScoredSegment, score_segments
from .segment import Segmentation, segment_movie
from .track import CharacterTrack, ModelUnavailable, track_characters

__all__ = [
    "analyze_movie", "MovieAnalysis",
    "Segmentation", "CharacterTrack", "ScoredSegment", "CharacterProfile",
    "ModelUnavailable", "character_reel",
]


@dataclass
class MovieAnalysis:
    source: str
    segmentation: Segmentation
    scored: list[ScoredSegment] = field(default_factory=list)
    characters: list[CharacterTrack] = field(default_factory=list)
    profiles: dict[str, CharacterProfile] = field(default_factory=dict)
    tracking_available: bool = False


def analyze_movie(src: str, *, with_tracking: bool = True,
                  top_n: int = 20) -> MovieAnalysis:
    """Full pipeline: segment the film, score every segment for coolness,
    track characters across it (when models are installed).

    Tracking degrades honestly: ModelUnavailable propagates with a clear
    message naming the missing package — never fake tracks.
    """
    seg = segment_movie(src)
    bounds = [(s.start, s.end) for s in seg.segments if s.kind == "scene"]
    scored = score_segments(src, bounds)[:top_n * 3]  # keep headroom

    analysis = MovieAnalysis(source=str(src), segmentation=seg, scored=scored)

    if with_tracking:
        characters = track_characters(src)  # raises ModelUnavailable honestly
        analysis.characters = characters
        analysis.profiles = build_timelines(characters, scored)
        analysis.tracking_available = True
    return analysis
