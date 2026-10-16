"""Fight telemetry → difficulty tuning.

Difficulty used to be a vibes knob. Now every fight records a small
telemetry row (win/loss, duration, damage dealt/taken, skills used) and
:func:`difficulty_recommendation` derives the next tuning step from
the aggregates:

* win rate > 75% **and** average duration under 60s → ``raise``
  (players are stomping; the game is too easy).
* win rate < 35% → ``lower`` (players are stuck; the game is too hard).
* otherwise → ``hold``.
* no fights recorded → ``hold`` with reason "no data" — the tuner
  never invents a direction from zero evidence.

Per-skill usage counts expose dead skills (used in <5% of fights) so
the tuner can see *why* a difficulty feels off, not just that it does.

The store is in-memory by design: the game layer persists what it
wants (the DB-backed economy already owns coins); telemetry here is
the tuning signal, and tests prove the derivation without a database.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FightTelemetry",
    "TelemetryStore",
    "difficulty_recommendation",
    "aggregate",
]

#: Raise threshold: players win this often AND finish this fast.
RAISE_WIN_RATE = 0.75
RAISE_MAX_AVG_DURATION_S = 60.0
#: Lower threshold: players win this rarely.
LOWER_WIN_RATE = 0.35
#: Minimum fights before a raise/lower is trusted.
MIN_FIGHTS_FOR_TUNING = 10


@dataclass(frozen=True)
class FightTelemetry:
    fight_id: str
    build: str
    won: bool
    duration_s: float
    damage_dealt: int
    damage_taken: int
    skills_used: tuple[str, ...] = ()
    enemy: str = ""


@dataclass
class TelemetryStore:
    fights: list[FightTelemetry] = field(default_factory=list)

    def record(self, fight: FightTelemetry) -> None:
        self.fights.append(fight)

    def stats(self) -> dict[str, Any]:
        return aggregate(self.fights)


def aggregate(fights: list[FightTelemetry]) -> dict[str, Any]:
    """Aggregates: win rate, duration, damage, skill usage. Pure."""
    n = len(fights)
    if n == 0:
        return {"fights": 0, "win_rate": 0.0, "avg_duration_s": 0.0,
                "avg_damage_dealt": 0.0, "avg_damage_taken": 0.0,
                "skill_usage": {}, "skill_use_rate": {}}
    wins = sum(1 for f in fights if f.won)
    usage: Counter[str] = Counter()
    for f in fights:
        usage.update(set(f.skills_used))
    return {
        "fights": n,
        "win_rate": wins / n,
        "avg_duration_s": sum(f.duration_s for f in fights) / n,
        "avg_damage_dealt": sum(f.damage_dealt for f in fights) / n,
        "avg_damage_taken": sum(f.damage_taken for f in fights) / n,
        "skill_usage": dict(usage),
        "skill_use_rate": {k: v / n for k, v in usage.items()},
    }


def difficulty_recommendation(stats: dict[str, Any]) -> dict[str, Any]:
    """``{"action": raise|hold|lower, "reason": ...}`` from aggregates."""
    n = int(stats.get("fights", 0) or 0)
    if n == 0:
        return {"action": "hold", "reason": "no data — no fights recorded yet"}
    if n < MIN_FIGHTS_FOR_TUNING:
        return {"action": "hold",
                "reason": f"only {n} fight(s) — need {MIN_FIGHTS_FOR_TUNING} before tuning"}
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    avg_dur = float(stats.get("avg_duration_s", 0.0) or 0.0)
    if win_rate > RAISE_WIN_RATE and avg_dur < RAISE_MAX_AVG_DURATION_S:
        return {"action": "raise",
                "reason": f"win rate {win_rate:.0%} with avg duration {avg_dur:.0f}s — players are stomping"}
    if win_rate < LOWER_WIN_RATE:
        return {"action": "lower",
                "reason": f"win rate {win_rate:.0%} — players are stuck"}
    dead = [s for s, r in (stats.get("skill_use_rate") or {}).items() if r < 0.05]
    note = f" dead skills (<5% use): {', '.join(sorted(dead))}" if dead else ""
    return {"action": "hold",
            "reason": f"win rate {win_rate:.0%} in the healthy band.{note}"}
