"""Counters + codex: every move has an answer, and answers are earned.

Design rule: **no uncounterable move**. Every enemy move kind in
:data:`ENEMY_MOVES` has at least one :class:`Counter` that beats it —
:func:`validate_codex` proves that, so a new move cannot ship without
its counter.

Discovery rule: counters are **discoverable, not given**. A player
starts knowing nothing; :func:`discover_counter` records that a player
has *seen* a counter work (used it, or watched it land) and only then
does :func:`known_counters` reveal it. :func:`counter_for` (the raw
lookup) is the design-time view — the game never hands it to a player
who hasn't discovered the counter.

A counter has a timing window (in turns): using it outside the window
does not count. The window is data on the counter, not a branch in
the combat code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Counter",
    "ENEMY_MOVES",
    "CODEX",
    "counter_for",
    "discover_counter",
    "known_counters",
    "validate_codex",
]

#: Enemy move kinds the codex must cover — one counter each, minimum.
ENEMY_MOVES: tuple[str, ...] = (
    "heavy_strike", "charge_attack", "aimed_shot", "lunge",
    "riposte_stance", "frenzy", "regenerate", "harden", "venom", "dread",
)


@dataclass(frozen=True)
class Counter:
    """One answer to one enemy move."""
    slug: str
    beats: str  # enemy move kind this counters
    name: str
    desc: str
    window_turns: int = 1  # timing window: counter must land within this
    requires: str = ""  # skill slug required, "" = always available


#: The codex — every move has ≥1 counter. Data, not branches.
CODEX: tuple[Counter, ...] = (
    Counter("sidestep", "heavy_strike", "Sidestep",
            "Step aside as the heavy blow falls — it whiffs.", 1),
    Counter("interrupt", "charge_attack", "Interrupt",
            "Strike mid-charge to break it.", 1, requires="quick_jab"),
    Counter("duck", "aimed_shot", "Duck",
            "Drop low; the aimed shot passes overhead.", 1),
    Counter("parry", "lunge", "Parry",
            "Deflect the lunge and open a riposte.", 1),
    Counter("feint", "riposte_stance", "Feint",
            "Fake the attack; the riposte fires at air.", 1),
    Counter("kite", "frenzy", "Kite",
            "Keep distance until the frenzy burns out.", 2),
    Counter("burst", "regenerate", "Burst",
            "Out-damage the regeneration in one window.", 1),
    Counter("shatter", "harden", "Shatter",
            "Break the hardened guard with a heavy hit.", 2, requires="crusher"),
    Counter("antidote", "venom", "Antidote",
            "Cleanse the venom before it ticks.", 2),
    Counter("taunt_break", "dread", "Taunt-break",
            "Shrug off the dread with a war cry.", 1),
)


def counter_for(move_kind: str) -> list[Counter]:
    """Design-time lookup: every counter that beats this move."""
    return [c for c in CODEX if c.beats == (move_kind or "").strip().lower()]


def discover_counter(knowledge: set[str], counter_slug: str) -> Counter | None:
    """Record that a player discovered a counter. Returns it, or None.

    ``knowledge`` is the player's mutable set of discovered counter
    slugs (persisted by the game layer). Discovering an unknown slug
    returns None — typos never invent counters.
    """
    slug = (counter_slug or "").strip().lower()
    for counter in CODEX:
        if counter.slug == slug:
            knowledge.add(slug)
            return counter
    return None


def known_counters(knowledge: set[str], move_kind: str) -> list[Counter]:
    """Counters the player has discovered for this move — nothing more."""
    known = set(knowledge or ())
    return [c for c in counter_for(move_kind) if c.slug in known]


def validate_codex() -> list[str]:
    """Prove the design rule: every move has ≥1 counter; slugs unique."""
    problems: list[str] = []
    seen: set[str] = set()
    for counter in CODEX:
        if counter.slug in seen:
            problems.append(f"duplicate counter slug {counter.slug}")
        seen.add(counter.slug)
        if counter.beats not in ENEMY_MOVES:
            problems.append(f"counter {counter.slug} beats unknown move {counter.beats}")
        if counter.window_turns < 1:
            problems.append(f"counter {counter.slug} has a non-positive window")
    for move in ENEMY_MOVES:
        if not counter_for(move):
            problems.append(f"move {move} has no counter — uncounterable moves are forbidden")
    return problems
