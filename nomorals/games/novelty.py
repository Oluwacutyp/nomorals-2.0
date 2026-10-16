"""Economy rewards for novel play.

Grinding the same build forever is the fastest way to make a game
boring — and the economy used to pay it the same as exploration. Now
novelty pays:

* **First win with a new build** — the first time a player *wins* with
  a build they have never won with before: ``FIRST_BUILD_WIN_BONUS``
  coins. Losing with a new build teaches nothing to the ledger; the
  bonus waits for the win.
* **First use of a skill** — the first time a player *uses* a skill
  (win or lose): ``FIRST_SKILL_USE_BONUS`` coins per skill. Trying
  something new is rewarded even when the fight goes badly.

:class:`NoveltyLedger` is the per-player memory: builds won with +
skills used. The game layer persists it (the economy already owns coin
balances); this module computes what is *owed* and marks it seen, so a
bonus can never be paid twice for the same first.

Invariant: the same build's second win pays no build bonus; the same
skill's second use pays no skill bonus — novelty is a one-time payment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NoveltyLedger",
    "FIRST_BUILD_WIN_BONUS",
    "FIRST_SKILL_USE_BONUS",
    "compute_novelty_rewards",
]

#: First win with a never-won build.
FIRST_BUILD_WIN_BONUS = 50
#: First use of a never-used skill.
FIRST_SKILL_USE_BONUS = 10


@dataclass
class NoveltyLedger:
    """Per-player novelty memory: builds won with + skills used."""
    builds_won: set[str] = field(default_factory=set)
    skills_used: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {"builds_won": sorted(self.builds_won),
                "skills_used": sorted(self.skills_used)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NoveltyLedger":
        data = data or {}
        return cls(builds_won=set(data.get("builds_won") or []),
                   skills_used=set(data.get("skills_used") or []))


def compute_novelty_rewards(ledger: NoveltyLedger, *, build: str,
                            skills_used: list[str] | tuple[str, ...],
                            won: bool) -> dict[str, Any]:
    """Coins owed for this fight's novelty + what was novel.

    Marks the novelties seen (the ledger is the payment record — a
    second call with the same inputs pays zero). Returns
    ``{"coins", "reasons"}``; ``reasons`` names each paid novelty so
    the player sees *why* they got coins.
    """
    coins = 0
    reasons: list[str] = []
    build_key = (build or "").strip().lower()
    if won and build_key and build_key not in ledger.builds_won:
        ledger.builds_won.add(build_key)
        coins += FIRST_BUILD_WIN_BONUS
        reasons.append(f"first win with build '{build}' (+{FIRST_BUILD_WIN_BONUS})")
    for skill in skills_used or []:
        key = str(skill or "").strip().lower()
        if key and key not in ledger.skills_used:
            ledger.skills_used.add(key)
            coins += FIRST_SKILL_USE_BONUS
            reasons.append(f"first use of skill '{skill}' (+{FIRST_SKILL_USE_BONUS})")
    return {"coins": coins, "reasons": reasons}
