"""Character timelines: merge person tracks with scene scores.

Answers: who is in which scene, for how long, and which of THEIR scenes
are the cool ones. Powers "give me all of char_2's best moments."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .score import ScoredSegment
from .track import CharacterTrack


@dataclass
class CharacterScene:
    char_id: str
    seg_index: int
    start: float
    end: float
    screen_s: float          # seconds this character is visible in the scene
    score: float             # the scene's highlight score
    label: str = ""


@dataclass
class CharacterProfile:
    char_id: str
    total_screen_s: float
    scene_count: int
    best_scenes: list[CharacterScene] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0


def build_timelines(characters: list[CharacterTrack],
                    scored: list[ScoredSegment]) -> dict[str, CharacterProfile]:
    """Map each character's detections onto scored segments."""
    seg_by_index = {s.index: s for s in scored}
    # For each character, bin their detection times into segments
    profiles: dict[str, CharacterProfile] = {}
    for char in characters:
        # count detections per segment
        per_seg: dict[int, int] = {}
        for d in char.detections:
            for s in scored:
                if s.start <= d.frame_t < s.end:
                    per_seg[s.index] = per_seg.get(s.index, 0) + 1
                    break
        scenes: list[CharacterScene] = []
        for si, count in per_seg.items():
            seg = seg_by_index[si]
            # detections were sampled at ~2fps; each ≈ 0.5s of screen time
            screen_s = round(count * 0.5, 1)
            if screen_s < 1.0:
                continue  # passing glimpse, not a scene
            scenes.append(CharacterScene(
                char_id=char.char_id, seg_index=si,
                start=seg.start, end=seg.end,
                screen_s=screen_s, score=seg.score, label=seg.label))
        scenes.sort(key=lambda c: (c.score, c.screen_s), reverse=True)
        profiles[char.char_id] = CharacterProfile(
            char_id=char.char_id,
            total_screen_s=char.total_screen_s,
            scene_count=len(scenes),
            best_scenes=scenes,
            first_seen=char.first_seen,
            last_seen=char.last_seen,
        )
    return profiles


def character_reel(profiles: dict[str, CharacterProfile], char_id: str,
                   *, top_n: int = 10,
                   min_screen_s: float = 3.0) -> list[CharacterScene]:
    """Top-N scenes for one character — their highlight reel inputs."""
    prof = profiles.get(char_id)
    if not prof:
        return []
    return [s for s in prof.best_scenes if s.screen_s >= min_screen_s][:top_n]
