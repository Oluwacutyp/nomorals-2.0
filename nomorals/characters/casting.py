"""Casting director: which character fits this moment.

Not a menu — judgment. Given a situation (podcast topic, room vibe,
game night, deep talk), picks the right characters and explains why.
Chemistry between cast members matters as much as individual fit.
"""
from __future__ import annotations

from typing import Any

from .character import Character
from .relationships import RelationshipGraph


def _trait_fit(char: Character, want: dict[str, float]) -> float:
    if not want:
        return 0.5
    total, weight = 0.0, 0.0
    for trait, w in want.items():
        total += char.persona.get(trait, 0.3) * w
        weight += w
    return total / weight if weight else 0.5


def _skill_fit(char: Character, skills: list[str]) -> float:
    if not skills:
        return 0.5
    return sum(char.skills.get(s, 0.2) for s in skills) / len(skills)


def _role_fit(char: Character, role: str) -> float:
    if not role:
        return 0.5
    return 1.0 if role in (char.roles or []) else 0.3


def chemistry(a: Character, b: Character,
              graph: RelationshipGraph | None = None) -> float:
    """How well do these two play together? 0.0–1.0.

    Combines relationship history (if any) with complementary traits:
    opposite energy can spark, shared warmth sustains.
    """
    score = 0.5
    if graph is not None:
        e = graph.edge(a.id, b.id)
        d = e.dims
        # warmth + respect help, friction hurts (unless it's fun friction)
        score = (d["warmth"] * 0.35 + d["respect"] * 0.25
                 + d["familiarity"] * 0.2 + (1.0 - d["friction"]) * 0.2)
    # complementary traits: one high-energy + one grounded = good radio
    ea = a.persona.get("energetic", a.persona.get("extroverted", 0.5))
    eb = b.persona.get("energetic", b.persona.get("extroverted", 0.5))
    diff = abs(ea - eb)
    if 0.2 < diff < 0.6:
        score = min(1.0, score + 0.1)   # complementary energy sparks
    # shared humor bonds
    ha = a.persona.get("witty", a.persona.get("humorous", 0.3))
    hb = b.persona.get("witty", b.persona.get("humorous", 0.3))
    if ha > 0.6 and hb > 0.6:
        score = min(1.0, score + 0.08)  # banter duo
    return max(0.0, min(1.0, score))


def cast_for(chars: list[Character], *, role: str = "",
             traits: dict[str, float] | None = None,
             skills: list[str] | None = None,
             topic: str = "", n: int = 1,
             exclude: set[str] | None = None,
             graph: RelationshipGraph | None = None,
             recent: set[str] | None = None,
             rotation_penalty: float = 0.25) -> list[tuple[Character, float, str]]:
    """Pick the best-fit character(s) for a moment.

    Returns [(character, score, reason)]. For ensembles (n>1),
    optimizes for group chemistry, not just individual fit.

    ``recent``: ids/names cast lately — they get a ``rotation_penalty``
    so recurring shows rotate the spotlight instead of running the
    same faces every time.
    """
    exclude = exclude or set()
    recent = recent or set()
    cands = [c for c in chars if c.id not in exclude and c.name not in exclude]
    if not cands:
        return []

    def fit(c: Character) -> tuple[float, str]:
        reasons: list[str] = []
        s = 0.0
        s += _role_fit(c, role) * 0.4
        if role and role in (c.roles or []):
            reasons.append(f"natural {role}")
        tf = _trait_fit(c, traits or {})
        s += tf * 0.3
        if traits:
            top = max(traits, key=lambda k: traits[k])
            if c.persona.get(top, 0) > 0.6:
                reasons.append(f"strong {top}")
        sf = _skill_fit(c, skills or [])
        s += sf * 0.3
        if skills:
            best = max(skills, key=lambda sk: c.skills.get(sk, 0))
            if c.skills.get(best, 0) > 0.6:
                reasons.append(f"knows {best}")
        # topic resonance from knowledge
        if topic:
            tl = topic.lower()
            hits = [k for k in c.knowledge if any(
                w in k.lower() for w in tl.split() if len(w) > 3)]
            if hits:
                s += 0.1
                reasons.append("knows the topic")
        # rotation: the recently-cast pay a penalty
        if c.id in recent or c.name in recent:
            s = max(0.0, s - rotation_penalty)
            reasons.append("rotation pick — fresh face")
        return min(1.0, s), "; ".join(reasons) or "solid all-rounder"

    scored = [(c, *fit(c)) for c in cands]
    scored.sort(key=lambda t: -t[1])

    if n <= 1:
        return scored[:1]

    # ensemble: greedy pick maximizing fit + chemistry with already-cast
    picked: list[tuple[Character, float, str]] = [scored[0]]
    remaining = scored[1:]
    while len(picked) < n and remaining:
        def ensemble_score(t: tuple) -> float:
            c, s, _ = t
            chem = sum(chemistry(c, p[0], graph) for p in picked) / len(picked)
            return s * 0.6 + chem * 0.4
        remaining.sort(key=lambda t: -ensemble_score(t))
        nxt = remaining.pop(0)
        chem = sum(chemistry(nxt[0], p[0], graph) for p in picked) / len(picked)
        picked.append((nxt[0], nxt[1],
                       f"{nxt[2]} + chemistry {chem:.1f} with cast"))
    return picked


# ── role presets: what each moment asks for ──────────────────────────
ROLE_PRESETS: dict[str, dict[str, Any]] = {
    "podcast_host": {
        "role": "podcast_host",
        "traits": {"witty": 0.8, "curious": 0.9, "expressive": 0.7},
        "skills": ["conversation", "interviewing"],
    },
    "podcast_guest": {
        "traits": {"expressive": 0.7, "knowledgeable": 0.8},
        "skills": ["storytelling"],
    },
    "dj": {"role": "dj", "traits": {"energetic": 0.9, "expressive": 0.8},
            "skills": ["music"]},
    "gamer": {"role": "gamer", "traits": {"competitive": 0.8, "playful": 0.7},
              "skills": ["strategy"]},
    "sage": {"role": "sage", "traits": {"wise": 0.9, "calm": 0.8},
             "skills": ["philosophy", "advice"]},
    "hype": {"traits": {"energetic": 0.9, "playful": 0.9},
             "skills": ["banter"]},
    "deep_talk": {"traits": {"empathetic": 0.9, "wise": 0.7, "honest": 0.8},
                  "skills": ["listening", "advice"]},
    "roast": {"traits": {"witty": 0.9, "bold": 0.8},
              "skills": ["banter", "roasting"]},
}


def cast_preset(chars: list[Character], preset: str, n: int = 1,
                topic: str = "",
                graph: RelationshipGraph | None = None,
                exclude: set[str] | None = None,
                recent: set[str] | None = None,
                ) -> list[tuple[Character, float, str]]:
    """Shorthand: cast_for with a named preset."""
    p = ROLE_PRESETS.get(preset, {})
    return cast_for(chars, role=p.get("role", ""), traits=p.get("traits"),
                    skills=p.get("skills"), topic=topic, n=n,
                    exclude=exclude, graph=graph, recent=recent)


def cast_against(chars: list[Character], *,
                 avoid_traits: dict[str, float] | None = None,
                 avoid_skills: list[str] | None = None,
                 topic: str = "",
                 n: int = 1,
                 exclude: set[str] | None = None) -> list[tuple[Character, float, str]]:
    """Negative casting: who must NOT be in this room.

    Scores how badly each character fits the AVOIDED profile — the
    highest scorers are the worst picks. Use it to keep the wrong
    energy out, or invert it to find deliberate chaos picks.
    Returns [(character, badness_score, reason)] — higher = worse fit.
    """
    exclude = exclude or set()
    cands = [c for c in chars if c.id not in exclude and c.name not in exclude]
    if not cands:
        return []

    def badness(c: Character) -> tuple[float, str]:
        reasons: list[str] = []
        s = 0.0
        if avoid_traits:
            for trait, w in avoid_traits.items():
                v = c.persona.get(trait, 0.3)
                s += v * w
                if v > 0.7:
                    reasons.append(f"too {trait}")
        if avoid_skills:
            for sk in avoid_skills:
                v = c.skills.get(sk, 0.0)
                s += v * 0.5
                if v > 0.7:
                    reasons.append(f"brings {sk} energy")
        if topic:
            tl = topic.lower()
            hits = [k for k in c.knowledge if any(
                w in k.lower() for w in tl.split() if len(w) > 3)]
            if hits:
                s += 0.3
                reasons.append("will hijack the topic")
        return min(1.0, s), "; ".join(reasons) or "wrong room, wrong time"

    scored = [(c, *badness(c)) for c in cands]
    scored.sort(key=lambda t: -t[1])
    return scored[:n]


def cast_for_audience(chars: list[Character],
                      audience_traits: dict[str, float],
                      n: int = 2,
                      graph: RelationshipGraph | None = None,
                      exclude: set[str] | None = None) -> list[tuple[Character, float, str]]:
    """Audience-aware casting: pick the characters THIS audience will
    love. Matches character traits to what the audience vibes with —
    a hype crowd wants energy, a late-night crowd wants depth."""
    exclude = exclude or set()
    cands = [c for c in chars if c.id not in exclude and c.name not in exclude]
    if not cands:
        return []

    def appeal(c: Character) -> tuple[float, str]:
        s = _trait_fit(c, audience_traits)
        top = max(audience_traits, key=lambda k: audience_traits[k]) \
            if audience_traits else ""
        reason = (f"the crowd wants {top} — {c.name} brings it"
                  if top and c.persona.get(top, 0) > 0.6
                  else "crowd-pleaser")
        # extraverts play better to a crowd
        try:
            s = min(1.0, s + float(c.ocean.get("extraversion", 0.5)) * 0.1)
        except Exception:
            pass
        return s, reason

    scored = [(c, *appeal(c)) for c in cands]
    scored.sort(key=lambda t: -t[1])
    if n <= 1 or len(scored) <= 1:
        return scored[:n]
    # keep the chemistry pass — a crowd loves a duo that sparks
    picked: list[tuple[Character, float, str]] = [scored[0]]
    remaining = scored[1:]
    while len(picked) < n and remaining:
        def ensemble_score(t: tuple) -> float:
            c, s, _ = t
            chem = sum(chemistry(c, p[0], graph) for p in picked) / len(picked)
            return s * 0.6 + chem * 0.4
        remaining.sort(key=lambda t: -ensemble_score(t))
        nxt = remaining.pop(0)
        picked.append((nxt[0], nxt[1], nxt[2]))
    return picked
