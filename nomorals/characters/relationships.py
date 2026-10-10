"""Character relationships: directional, multi-dimensional, slow-moving.

Every edge A→B carries five dimensions (livingfeed gold):
  trust, warmth, friction, respect, familiarity — each 0.0–1.0.

Values shift SLOWLY from what actually happens between them (Musubi
gold) — one conversation never flips a relationship. Directional:
A trusting B says nothing about B trusting A.

Entities: character ids, "brain" (Devon), "owner".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

DIMS = ("trust", "warmth", "friction", "respect", "familiarity")

# how much a single interaction can move a dimension (slow by design)
LEARNING_RATE = 0.06
# mood/experience decay toward baseline per day of inactivity
DECAY_PER_DAY = 0.02

BRAIN_ID = "brain"
OWNER_ID = "owner"


def _clamp01(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


@dataclass
class Edge:
    """One directional relationship A→B."""
    a: str
    b: str
    dims: dict[str, float] = field(default_factory=lambda: {
        "trust": 0.5, "warmth": 0.5, "friction": 0.1,
        "respect": 0.5, "familiarity": 0.2,
    })
    kind: str = "acquaintance"   # acquaintance|friend|close|rival|mentor|...
    interactions: int = 0
    updated_at: float = field(default_factory=time.time)
    # transition history: [{"from": kind, "to": kind, "ts": ...}] — newest last
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        base = {"trust": 0.5, "warmth": 0.5, "friction": 0.1,
                "respect": 0.5, "familiarity": 0.2}
        base.update({k: _clamp01(v) for k, v in (self.dims or {}).items()
                     if k in DIMS})
        self.dims = base
        self.transitions = [t for t in (self.transitions or [])
                            if isinstance(t, dict)][-10:]

    def nudge(self, deltas: dict[str, float]) -> None:
        """Shift dimensions from an interaction. Slow by design."""
        for k, d in deltas.items():
            if k in DIMS:
                self.dims[k] = _clamp01(self.dims[k] + d * LEARNING_RATE)
        self.interactions += 1
        self.updated_at = time.time()
        self._recompute_kind()

    def _recompute_kind(self) -> None:
        t, w, f = self.dims["trust"], self.dims["warmth"], self.dims["friction"]
        fam = self.dims["familiarity"]
        if f > 0.6 and t < 0.4:
            new = "rival"
        elif t > 0.75 and w > 0.7 and fam > 0.6:
            new = "close"
        elif t > 0.6 and w > 0.55:
            new = "friend"
        elif fam < 0.3:
            new = "acquaintance"
        else:
            new = "familiar"
        # Inworld gold: ally↔enemy transitions are EVENTS, not silent flips.
        if new != self.kind:
            self.transitions.append(
                {"from": self.kind, "to": new, "ts": time.time()})
            self.transitions = self.transitions[-10:]
            self.kind = new

    def last_transition_line(self, name_a: str = "They",
                             name_b: str = "them") -> str:
        """Narrative line for the most recent kind flip, or ''."""
        if not self.transitions:
            return ""
        tr = self.transitions[-1]
        frm, to = tr.get("from", ""), tr.get("to", "")
        verbs = {
            ("acquaintance", "friend"): f"{name_a} and {name_b} became friends",
            ("friend", "close"): f"{name_a} and {name_b} got close",
            ("familiar", "close"): f"{name_a} and {name_b} got close",
            ("familiar", "friend"): f"{name_a} and {name_b} became friends",
            ("friend", "rival"): f"{name_a} and {name_b} are rivals now",
            ("close", "rival"): f"{name_a} and {name_b} fell out — rivals now",
            ("familiar", "rival"): f"{name_a} turned on {name_b}",
            ("rival", "friend"): f"{name_a} and {name_b} buried the hatchet",
            ("rival", "close"): f"{name_a} and {name_b} went from rivals to close",
            ("close", "friend"): f"{name_a} and {name_b} drifted a little",
            ("friend", "familiar"): f"{name_a} and {name_b} drifted apart",
        }
        return verbs.get((frm, to),
                         f"{name_a} and {name_b} went from {frm} to {to}")

    def decay(self, now: float | None = None) -> None:
        """Friction cools, warmth settles — relationships drift without contact."""
        now = now or time.time()
        days = (now - self.updated_at) / 86400.0
        if days < 1:
            return
        pull = min(0.5, days * DECAY_PER_DAY)
        # friction decays toward 0.1, warmth/trust toward 0.5
        self.dims["friction"] = _clamp01(
            self.dims["friction"] - pull * (self.dims["friction"] - 0.1))
        for k in ("warmth", "trust"):
            self.dims[k] = _clamp01(
                self.dims[k] + pull * (0.5 - self.dims[k]) * 0.5)
        self.updated_at = now

    def summary(self) -> str:
        d = self.dims
        return (f"{self.kind} (trust {d['trust']:.1f}, warmth {d['warmth']:.1f}, "
                f"friction {d['friction']:.1f}, respect {d['respect']:.1f})")

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "dims": self.dims,
                "kind": self.kind, "interactions": self.interactions,
                "updated_at": self.updated_at,
                "transitions": self.transitions}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Edge":
        e = cls(a=str(d.get("a", "")), b=str(d.get("b", "")),
                dims=dict(d.get("dims") or {}),
                kind=str(d.get("kind") or "acquaintance"),
                interactions=int(d.get("interactions") or 0),
                updated_at=float(d.get("updated_at") or time.time()),
                transitions=list(d.get("transitions") or []))
        return e


# ── interaction → delta mapping (no scripts, just direction) ──────────
# Each event nudges dimensions. Positive events build, negative erode.
EVENT_DELTAS: dict[str, dict[str, float]] = {
    "good_conversation": {"trust": 0.5, "warmth": 0.6, "familiarity": 0.8,
                          "respect": 0.3},
    "deep_conversation": {"trust": 1.0, "warmth": 0.8, "familiarity": 1.0,
                          "respect": 0.6},
    "laughed_together": {"warmth": 1.0, "familiarity": 0.5, "friction": -0.4},
    "helped": {"trust": 0.8, "respect": 0.7, "warmth": 0.4},
    "was_helped_by": {"trust": 0.7, "warmth": 0.5, "respect": 0.5},
    "disagreement": {"friction": 0.6, "respect": 0.2, "familiarity": 0.3},
    "argument": {"friction": 1.2, "trust": -0.5, "warmth": -0.6},
    "made_up": {"friction": -1.0, "trust": 0.5, "warmth": 0.6},
    "betrayed": {"trust": -1.5, "warmth": -1.0, "friction": 1.0},
    "won_game": {"respect": 0.4, "familiarity": 0.3},
    "lost_game": {"respect": -0.2, "familiarity": 0.3},
    "ignored": {"warmth": -0.4, "familiarity": -0.2},
    "praised": {"warmth": 0.6, "respect": 0.4},
    "insulted": {"friction": 0.8, "warmth": -0.7, "trust": -0.4},
    # ── sweep additions ──
    "secret_shared": {"trust": 1.2, "warmth": 0.7, "familiarity": 0.9},
    "opened_up": {"trust": 0.9, "warmth": 0.6, "familiarity": 0.7},
    "defended": {"trust": 1.0, "respect": 0.8, "warmth": 0.5},
    "celebrated": {"warmth": 0.9, "familiarity": 0.6, "friction": -0.3},
    "forgave": {"friction": -1.2, "trust": 0.6, "warmth": 0.5},
    "abandoned": {"trust": -1.2, "warmth": -0.9, "friction": 0.8},
    "stood_up_to": {"friction": 0.5, "respect": 0.4},
    "missed_them": {"warmth": 0.3, "familiarity": 0.2},
}


class RelationshipGraph:
    """All edges. Persisted inside CharacterStore as a sidecar file."""

    def __init__(self) -> None:
        self.edges: dict[tuple[str, str], Edge] = {}

    def edge(self, a: str, b: str) -> Edge:
        key = (a, b)
        e = self.edges.get(key)
        if e is None:
            e = Edge(a=a, b=b)
            self.edges[key] = e
        return e

    def interact(self, a: str, b: str, event: str,
                 mirror: str | None = None) -> None:
        """Record that ``event`` happened between a and b (a's perspective
        shifts by ``event``; b's perspective shifts by ``mirror`` or the
        same event)."""
        deltas = EVENT_DELTAS.get(event)
        if deltas:
            self.edge(a, b).nudge(deltas)
        if mirror:
            md = EVENT_DELTAS.get(mirror)
            if md:
                self.edge(b, a).nudge(md)

    def describe(self, a: str, b: str) -> str:
        return self.edge(a, b).summary()

    def story_of(self, a: str, b: str,
                 name_a: str | None = None,
                 name_b: str | None = None) -> str:
        """The relationship as a short narrative — history, current state,
        and where it's heading. Façade gold: relationships are stories,
        not stat blocks."""
        e = self.edge(a, b)
        na, nb = name_a or a, name_b or b
        d = e.dims
        lines = [f"{na} → {nb}: {e.kind}."]
        # texture from dimensions
        tex = []
        if d["trust"] > 0.75:
            tex.append(f"{na} trusts {nb} deeply")
        elif d["trust"] < 0.35:
            tex.append(f"{na} doesn't really trust {nb}")
        if d["warmth"] > 0.7:
            tex.append("genuine warmth")
        elif d["warmth"] < 0.35:
            tex.append("it's cold between them")
        if d["friction"] > 0.55:
            tex.append("real tension")
        if d["respect"] > 0.8:
            tex.append(f"{na} looks up to {nb}")
        if d["familiarity"] > 0.8:
            tex.append("they know each other inside out")
        if tex:
            lines.append("Right now: " + ", ".join(tex) + ".")
        lines.append(f"{e.interactions} shared moments on record.")
        tl = e.last_transition_line(na, nb)
        if tl:
            lines.append("Latest chapter: " + tl + ".")
        # trajectory: where is it heading?
        if len(e.transitions) >= 2:
            lines.append("Trajectory: this one keeps evolving — "
                         "watch it closely.")
        elif e.kind in ("rival",) and d["friction"] > 0.7:
            lines.append("Trajectory: heading for a blowup or a breakthrough.")
        elif e.kind in ("close", "friend") and d["warmth"] > 0.7:
            lines.append("Trajectory: solid and warming.")
        return "\n".join(lines)

    def triad(self, a: str, b: str, c: str) -> dict[str, Any]:
        """Three-way dynamics: does C's presence strain or steady A↔B?

        Returns {"tension": 0-1, "note": str}. High tension = the third
        wheel changes how the pair behaves.
        """
        eab, eac, ebc = self.edge(a, b), self.edge(a, c), self.edge(b, c)
        tension = 0.0
        notes = []
        # A's two bonds pull in opposite directions
        wa, wb = eac.dims["warmth"], ebc.dims["warmth"]
        if abs(wa - wb) > 0.4:
            tension += 0.3
            notes.append("split loyalties")
        # friction with C spills onto the pair
        fc = max(eac.dims["friction"], ebc.dims["friction"])
        if fc > 0.6:
            tension += 0.3
            notes.append("C brings friction into the room")
        # rivalry triangle
        if eac.kind == "rival" or ebc.kind == "rival":
            tension += 0.25
            notes.append("an active rivalry in the triangle")
        # mutual closeness steadies
        if eab.kind in ("close", "friend") and wa > 0.6 and wb > 0.6:
            tension = max(0.0, tension - 0.2)
            notes.append("but the pair is steady")
        return {"tension": max(0.0, min(1.0, tension)),
                "note": "; ".join(notes) or "no particular triangle drama"}

    def circle(self, who: str) -> list[tuple[str, Edge]]:
        """Everyone ``who`` has an edge with, closest first."""
        out = [(b, e) for (x, b), e in self.edges.items() if x == who]
        out.sort(key=lambda t: -(t[1].dims["familiarity"]
                                 + t[1].dims["trust"]))
        return out

    def decay_all(self) -> None:
        for e in self.edges.values():
            e.decay()

    def to_dict(self) -> dict[str, Any]:
        return {"edges": [e.to_dict() for e in self.edges.values()]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RelationshipGraph":
        g = cls()
        for ed in d.get("edges") or []:
            try:
                e = Edge.from_dict(ed)
                g.edges[(e.a, e.b)] = e
            except Exception:
                continue
        return g
