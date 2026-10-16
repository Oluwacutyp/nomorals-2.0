"""Declarative game data: affixes, skills, enemies as data, not branches.

The old pattern was code-per-thing: each affix/skill/enemy behaviour
lived in its own branch, so adding content meant editing logic. Here
content is *data* — :class:`AffixDesc` / :class:`SkillDesc` /
:class:`EnemyDesc` carry tags, weights, constraints, and
incompatibilities — and one constraint solver
(:func:`solve`) picks legal combinations.

* **Affixes 5 → 12 as data.** The declarative affix catalog holds 12
  archetypal affixes (the gear module's 23 rolled affixes still exist;
  these 12 are the *declarative* set the solver composes — flat, pct,
  stat, speed, sustain, and mythic families — so new affixes are a
  one-line data entry, never a code change).
* **Role pool expansion as data.** :data:`ROLE_POOL` is the enemy role
  roster (8 roles with weights/tags); the enemy generator draws roles
  from it instead of hardcoding hunter-only foes.
* **Anti-repeat as constraint solving.** :func:`solve` treats "not seen
  in the last K picks" as a hard constraint (with graceful softening:
  when the pool is exhausted it relaxes recency before it ever returns
  an illegal combo or duplicates an incompatible pair).

Invariant: :func:`solve` never returns a set containing an
incompatible pair, a violated constraint, or (while any legal fresh
pick exists) a repeat from the recent history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AffixDesc",
    "SkillDesc",
    "EnemyDesc",
    "AFFIX_CATALOG",
    "ROLE_POOL",
    "skill_descs_from_catalog",
    "solve",
    "pick_without_repeat",
    "validate_catalog",
]


@dataclass(frozen=True)
class AffixDesc:
    """One declarative affix: what it does is data, not a branch."""
    slug: str
    name: str
    family: str  # flat | pct | stat | speed | sustain | mythic
    tags: frozenset[str] = frozenset()
    weight: float = 1.0
    min_level: int = 1
    slots: frozenset[str] = frozenset({"weapon", "armor"})
    mods: dict[str, int] = field(default_factory=dict)
    incompatible_with: frozenset[str] = frozenset()

    def allows(self, *, level: int, slot: str) -> bool:
        return level >= self.min_level and (not self.slots or slot in self.slots)


@dataclass(frozen=True)
class SkillDesc:
    """One declarative skill: tags/weights/constraints, no behaviour."""
    slug: str
    name: str
    school: str
    kind: str  # active | passive
    tags: frozenset[str] = frozenset()
    weight: float = 1.0
    level_req: int = 1
    mana_cost: int = 5
    incompatible_with: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EnemyDesc:
    """One declarative enemy archetype."""
    slug: str
    name: str
    role: str  # tank | dps | support | assassin | bruiser | controller | summoner | duelist
    tags: frozenset[str] = frozenset()
    weight: float = 1.0
    min_level: int = 1
    stat_mult: dict[str, float] = field(default_factory=dict)
    incompatible_with: frozenset[str] = frozenset()


#: 12 declarative affixes as data (flat/pct/stat/speed/sustain/mythic).
AFFIX_CATALOG: tuple[AffixDesc, ...] = (
    AffixDesc("bronze", "Bronze", "flat", frozenset({"atk"}), 10.0, 1, mods={"atk": 2}),
    AffixDesc("iron", "Iron", "flat", frozenset({"atk"}), 8.0, 12, mods={"atk": 5},
              incompatible_with=frozenset({"bronze", "steel", "mithril"})),
    AffixDesc("steel", "Steel", "flat", frozenset({"atk"}), 6.0, 28, mods={"atk": 9},
              incompatible_with=frozenset({"bronze", "iron", "mithril"})),
    AffixDesc("mithril", "Mithril", "flat", frozenset({"atk"}), 4.0, 48, mods={"atk": 14},
              incompatible_with=frozenset({"bronze", "iron", "steel"})),
    AffixDesc("cruel", "Cruel", "pct", frozenset({"atk", "crit"}), 5.0, 20, mods={"atk": 6},
              incompatible_with=frozenset({"merciless"})),
    AffixDesc("merciless", "Merciless", "pct", frozenset({"atk", "crit"}), 3.0, 45, mods={"atk": 11},
              incompatible_with=frozenset({"cruel"})),
    AffixDesc("sturdy", "Sturdy", "flat", frozenset({"def"}), 10.0, 1, mods={"def": 4}),
    AffixDesc("stone", "Stone", "flat", frozenset({"def"}), 6.0, 25, mods={"def": 8},
              incompatible_with=frozenset({"sturdy", "diamond"})),
    AffixDesc("diamond", "Diamond", "flat", frozenset({"def"}), 3.0, 50, mods={"def": 12},
              incompatible_with=frozenset({"sturdy", "stone"})),
    AffixDesc("warlord", "Warlord's", "stat", frozenset({"strength"}), 5.0, 15, mods={"strength": 3}),
    AffixDesc("swift", "Swift", "speed", frozenset({"atk", "def", "speed"}), 4.0, 30,
              mods={"atk": 2, "def": 2}),
    AffixDesc("gods", "of the Gods", "mythic", frozenset({"strength", "stamina", "intelligence"}),
              1.0, 70, mods={"strength": 6, "stamina": 6, "intelligence": 6}),
)

#: Role pool expansion as data: 8 roles the enemy generator draws from.
ROLE_POOL: tuple[EnemyDesc, ...] = (
    EnemyDesc("tank", "Bulwark", "tank", frozenset({"defensive", "taunt"}), 8.0, 1,
              {"hp": 1.6, "atk": 0.8, "def": 1.4}),
    EnemyDesc("dps", "Reaper", "dps", frozenset({"aggressive"}), 10.0, 1,
              {"hp": 0.9, "atk": 1.4, "def": 0.8}),
    EnemyDesc("support", "Mender", "support", frozenset({"heal", "buff"}), 5.0, 5,
              {"hp": 1.0, "atk": 0.8, "def": 1.0}),
    EnemyDesc("assassin", "Shade", "assassin", frozenset({"burst", "evade"}), 6.0, 8,
              {"hp": 0.8, "atk": 1.6, "def": 0.7}),
    EnemyDesc("bruiser", "Juggernaut", "bruiser", frozenset({"brawl"}), 7.0, 3,
              {"hp": 1.3, "atk": 1.2, "def": 1.1}),
    EnemyDesc("controller", "Hexer", "controller", frozenset({"debuff", "control"}), 4.0, 10,
              {"hp": 0.9, "atk": 1.0, "def": 0.9}),
    EnemyDesc("summoner", "Caller", "summoner", frozenset({"minions"}), 3.0, 15,
              {"hp": 1.0, "atk": 1.0, "def": 1.0}),
    EnemyDesc("duelist", "Blademaster", "duelist", frozenset({"counter", "precise"}), 5.0, 12,
              {"hp": 1.0, "atk": 1.3, "def": 1.0}),
)


def skill_descs_from_catalog() -> list[SkillDesc]:
    """Convert the live skill catalog to declarative SkillDescs.

    The behaviour still lives in :mod:`nomorals.games.skills`; this is
    the declarative *view* the solver and the codex read — one source
    of truth, no second hand-maintained skill list.
    """
    from .skills import SKILL_CATALOG

    out: list[SkillDesc] = []
    for slug, definition in SKILL_CATALOG.items():
        tags = {str(definition.school), str(definition.kind)}
        if definition.dodge:
            tags.add("dodge")
        if definition.heal_pct:
            tags.add("heal")
        if definition.counter_mult:
            tags.add("counter")
        out.append(SkillDesc(
            slug=str(slug), name=str(definition.name),
            school=str(definition.school), kind=str(definition.kind),
            tags=frozenset(tags), weight=1.0,
            level_req=int(definition.level_req),
            mana_cost=int(definition.mana_cost)))
    return out


def _legal(candidate: Any, chosen: list[Any]) -> bool:
    for other in chosen:
        if candidate.slug in other.incompatible_with or other.slug in candidate.incompatible_with:
            return False
    return True


def solve(pool: list[Any], count: int, *, rng: random.Random | None = None,
          history: list[str] | None = None, recent_k: int = 3,
          level: int = 1, slot: str = "weapon") -> list[Any]:
    """Pick ``count`` descs: weighted, no incompatibilities, anti-repeat.

    Anti-repeat is a hard constraint while any legal fresh pick exists:
    slugs in the last ``recent_k`` history entries are excluded first;
    only when that leaves too few candidates does the solver relax
    recency (never incompatibility). Deterministic given ``rng``.
    """
    rng = rng or random.Random()
    recent = set((history or [])[-recent_k:]) if recent_k > 0 else set()
    chosen: list[Any] = []

    def _eligible(d: Any, *, ignore_recency: bool = False) -> bool:
        if not ignore_recency and d.slug in recent:
            return False
        if hasattr(d, "min_level") and level < int(getattr(d, "min_level") or 0):
            return False
        allows = getattr(d, "allows", None)
        if allows is not None:
            try:
                if not allows(level=level, slot=slot):
                    return False
            except TypeError:
                pass
        return True

    fresh = [d for d in pool if _eligible(d)]
    relaxed = [d for d in pool if d not in fresh and _eligible(d, ignore_recency=True)]
    candidates = list(fresh)
    while len(chosen) < count:
        if not candidates:
            if not relaxed:
                break
            candidates = [d for d in relaxed if _legal(d, chosen)]
            relaxed = []
            if not candidates:
                break
        weights = [max(0.01, float(getattr(d, "weight", 1.0))) for d in candidates]
        pick = rng.choices(candidates, weights=weights, k=1)[0]
        if not _legal(pick, chosen):
            candidates.remove(pick)
            continue
        chosen.append(pick)
        candidates.remove(pick)
        relaxed = [d for d in relaxed if d is not pick]
    return chosen


def pick_without_repeat(pool: list[Any], count: int, *,
                        history: list[str], rng: random.Random | None = None,
                        **kwargs: Any) -> list[Any]:
    """Anti-repeat pick: :func:`solve` with the caller's history."""
    return solve(pool, count, rng=rng, history=history, **kwargs)


def validate_catalog() -> list[str]:
    """Catalog sanity: weights positive, no self-incompatibility, known slots."""
    problems: list[str] = []
    for desc in AFFIX_CATALOG:
        if desc.weight <= 0:
            problems.append(f"affix {desc.slug} has non-positive weight")
        if desc.slug in desc.incompatible_with:
            problems.append(f"affix {desc.slug} is incompatible with itself")
        if not desc.slots:
            problems.append(f"affix {desc.slug} has no slots")
    slugs = {d.slug for d in AFFIX_CATALOG}
    for desc in AFFIX_CATALOG:
        for other in desc.incompatible_with:
            if other not in slugs:
                problems.append(f"affix {desc.slug} incompatible with unknown {other}")
    if len({d.slug for d in ROLE_POOL}) != len(ROLE_POOL):
        problems.append("ROLE_POOL has duplicate slugs")
    return problems
