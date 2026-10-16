"""Async ghosts + brackets vs recorded runs — NOT fake Elo.

The old temptation was a fake Elo: invent a rating, match players
against it, pretend the population exists. There is no population —
so there is no Elo. What exists is **recorded runs**: real past
attempts (score, duration, build, moves). Ghosts are replays of those
runs; brackets are groups of those runs.

* :class:`GhostRun` — one recorded run. Immutable, factual.
* :func:`rank_vs_ghosts` — the player's percentile *among recorded
  runs*. Empty history → None (never an invented rating).
* :class:`GhostBracket` — a bracket seeded from recorded runs
  (``seed`` groups runs into terciles by score). A player competes
  against the bracket their score lands in — real opponents' real
  runs, asynchronously, with no live opponent needed.
* :func:`best_ghost` — the run to beat.

Async means: the player races a ghost's recorded splits at their own
time; the ghost never adapts, never rubber-bands, never pretends to
be a live person.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GhostRun", "GhostBracket", "rank_vs_ghosts", "best_ghost", "seed_brackets"]


@dataclass(frozen=True)
class GhostRun:
    """One recorded run — factual, never synthesized."""
    player: str
    build: str
    score: int
    duration_s: float
    moves: tuple[str, ...] = ()
    recorded_at: float = field(default_factory=time.time)


def rank_vs_ghosts(score: int, runs: list[GhostRun]) -> float | None:
    """Percentile of ``score`` among recorded runs (0–100).

    None when there are no recorded runs — the system refuses to
    invent a rating out of an empty population.
    """
    if not runs:
        return None
    below = sum(1 for r in runs if r.score < score)
    return round(100.0 * below / len(runs), 1)


def best_ghost(runs: list[GhostRun]) -> GhostRun | None:
    """The run to beat: highest score, ties → fastest duration."""
    if not runs:
        return None
    return min(runs, key=lambda r: (-r.score, r.duration_s))


@dataclass
class GhostBracket:
    """A bracket of recorded runs: low/mid/high by score tercile."""
    name: str  # "bronze" | "silver" | "gold"
    runs: list[GhostRun] = field(default_factory=list)

    @property
    def cutoff(self) -> tuple[int, int]:
        """(min_score, max_score) of the runs in this bracket."""
        if not self.runs:
            return (0, 0)
        scores = [r.score for r in self.runs]
        return (min(scores), max(scores))

    def rank_in_bracket(self, score: int) -> float | None:
        return rank_vs_ghosts(score, self.runs)


def seed_brackets(runs: list[GhostRun]) -> list[GhostBracket]:
    """Group recorded runs into bronze/silver/gold brackets by score.

    Terciles of the *recorded* population — no thresholds invented
    from nothing. Fewer than 3 runs → one bronze bracket with all of
    them (a bracket of one is still real runs, not a fake ladder).
    """
    if not runs:
        return []
    ordered = sorted(runs, key=lambda r: r.score)
    if len(ordered) < 3:
        return [GhostBracket(name="bronze", runs=list(ordered))]
    n = len(ordered)
    thirds = [ordered[: n // 3], ordered[n // 3: 2 * n // 3], ordered[2 * n // 3:]]
    names = ("bronze", "silver", "gold")
    return [GhostBracket(name=name, runs=group)
            for name, group in zip(names, thirds) if group]
