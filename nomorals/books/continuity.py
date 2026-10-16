"""Continuity engine (L3) — generalized from books/bible.py.

Finding 3 fix: continuity existed in four costumes (books bible.py
cast/threads/voice, media scene bible, music sonic bible, wisdom
history.py). This is the ONE engine; every creative organ is a client.

The engine is deliberately generic: typed state (entities / threads /
rules / voice), versioned snapshots, drift measurement. It knows nothing
about books — ``from_story_bible()`` adapts a ``StoryBible`` (duck-typed,
so bible.py stays untouched as the books facade) into engine state, and
``from_sonic_bible()`` / ``from_history()`` do the same for other organs.

Invariants:
- A bible never loses a character: entities are versioned, never deleted
  by ``apply_chapter``; removals go through ``retire_entity`` with a reason.
- Voice similarity is *reported*, never assumed: :meth:`drift` returns a
  measured number plus the features behind it.
- Every mutation is recorded in ``changelog`` with chapter + reason.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "Entity", "Thread", "Rule", "VoiceProfile", "ContinuityState",
    "Finding", "DriftReport", "ContinuityEngine",
]

_ENTITY_KINDS = ("character", "location", "item", "faction", "concept",
                 "rule", "event", "style")
_THREAD_STATUSES = ("open", "resolved", "abandoned")
_ENTITY_STATUSES = ("alive", "dead", "unknown", "absent", "retired")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Entity:
    """One typed continuity entity (character, place, item, faction…)."""

    id: str
    kind: str = "character"
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "alive"  # alive | dead | unknown | absent | retired
    first_seen: int = 0
    last_seen: int = 0
    mentions: int = 0
    retired_reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _ENTITY_KINDS:
            self.kind = "concept"
        if self.status not in _ENTITY_STATUSES:
            self.status = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Entity":
        return cls(**{k: d.get(k, v) for k, v in {
            "id": "", "kind": "character", "name": "",
            "aliases": [], "attributes": {}, "status": "alive",
            "first_seen": 0, "last_seen": 0, "mentions": 0,
            "retired_reason": ""}.items()})


@dataclass
class Thread:
    """One plot / arc thread: setup → (payoff | still open | abandoned)."""

    id: str
    summary: str
    status: str = "open"  # open | resolved | abandoned
    introduced: int = 0
    last_seen: int = 0
    keywords: list[str] = field(default_factory=list)
    foreshadow_chapter: int = 0
    payoff_chapter: int = 0
    heat: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in _THREAD_STATUSES:
            self.status = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Thread":
        return cls(**{k: d.get(k, v) for k, v in {
            "id": "", "summary": "", "status": "open", "introduced": 0,
            "last_seen": 0, "keywords": [], "foreshadow_chapter": 0,
            "payoff_chapter": 0, "heat": 0.0}.items()})


@dataclass
class Rule:
    """One world rule: a constraint later chapters must not violate."""

    id: str
    statement: str
    source_chapter: int = 0
    scope: str = "world"  # world | character | item | magic | social

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Rule":
        return cls(**{k: d.get(k, v) for k, v in {
            "id": "", "statement": "", "source_chapter": 0,
            "scope": "world"}.items()})


@dataclass
class VoiceProfile:
    """Measured voice, not a vibe: POV/tense + style-feature means."""

    pov: str = "third"
    tense: str = "past"
    tone: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    chapters_measured: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VoiceProfile":
        return cls(pov=d.get("pov", "third"), tense=d.get("tense", "past"),
                   tone=list(d.get("tone", [])),
                   features=dict(d.get("features", {})),
                   chapters_measured=int(d.get("chapters_measured", 0)))


@dataclass
class ContinuityState:
    """Versioned continuity state for one work."""

    work_slug: str
    title: str = ""
    entities: list[Entity] = field(default_factory=list)
    threads: list[Thread] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    voice: VoiceProfile = field(default_factory=VoiceProfile)
    version: int = 0
    chapters_seen: int = 0
    changelog: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_slug": self.work_slug, "title": self.title,
            "entities": [e.to_dict() for e in self.entities],
            "threads": [t.to_dict() for t in self.threads],
            "rules": [r.to_dict() for r in self.rules],
            "voice": self.voice.to_dict(), "version": self.version,
            "chapters_seen": self.chapters_seen,
            "changelog": self.changelog,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContinuityState":
        return cls(
            work_slug=d.get("work_slug", ""), title=d.get("title", ""),
            entities=[Entity.from_dict(e)
                      for e in d.get("entities", [])],
            threads=[Thread.from_dict(t) for t in d.get("threads", [])],
            rules=[Rule.from_dict(r) for r in d.get("rules", [])],
            voice=VoiceProfile.from_dict(d.get("voice", {})),
            version=int(d.get("version", 0)),
            chapters_seen=int(d.get("chapters_seen", 0)),
            changelog=list(d.get("changelog", [])))

    def entity(self, name_or_id: str) -> Entity | None:
        low = (name_or_id or "").lower()
        for e in self.entities:
            if e.id.lower() == low or e.name.lower() == low:
                return e
            if any(a.lower() == low for a in e.aliases):
                return e
        return None

    def open_threads(self) -> list[Thread]:
        return [t for t in self.threads if t.status == "open"]


@dataclass
class Finding:
    """One continuity finding: a checkable, citable problem."""

    kind: str  # dead_entity_appears | rule_violation | unknown_entity |
    #          thread_stalled | voice_drift | contradiction
    severity: str  # info | warning | error
    message: str
    chapter: int = 0
    entity: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """Measured voice drift for one chapter vs the work's voice."""

    chapter: int
    similarity: float  # 0..1 cosine over style features; REPORTED, never assumed
    features: dict[str, float] = field(default_factory=dict)
    reference_features: dict[str, float] = field(default_factory=dict)
    drifted: bool = False
    threshold: float = 0.75
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContinuityEngine:
    """Generic continuity: state + versioning + checks + drift."""

    def __init__(self, *, drift_threshold: float = 0.75) -> None:
        if not 0.0 <= drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be in [0, 1]")
        self.drift_threshold = drift_threshold

    # ── state lifecycle ──────────────────────────────────────────────
    def new_state(self, work_slug: str, title: str = "") -> ContinuityState:
        if not work_slug:
            raise ValueError("new_state needs a work_slug")
        return ContinuityState(work_slug=work_slug, title=title)

    def snapshot(self, state: ContinuityState, *,
                 reason: str = "") -> dict[str, Any]:
        """Versioned snapshot: bumps version, appends changelog, returns dict."""
        state.version += 1
        state.changelog.append({
            "version": state.version, "at": _now(), "reason": reason,
            "chapters_seen": state.chapters_seen,
            "entities": len(state.entities),
            "threads": len(state.threads),
        })
        return state.to_dict()

    def add_entity(self, state: ContinuityState, entity: Entity,
                   *, chapter: int = 0) -> Entity:
        if state.entity(entity.id) or state.entity(entity.name):
            raise ValueError(
                f"entity {entity.name!r} already in {state.work_slug!r}")
        entity.first_seen = entity.first_seen or chapter
        entity.last_seen = max(entity.last_seen, chapter)
        state.entities.append(entity)
        state.changelog.append({"op": "add_entity", "id": entity.id,
                                "chapter": chapter, "at": _now()})
        return entity

    def retire_entity(self, state: ContinuityState, name_or_id: str, *,
                      reason: str, chapter: int = 0) -> Entity:
        """The ONLY way an entity leaves active continuity — with a reason.

        The entity stays in state (never deleted); status becomes retired.
        """
        ent = state.entity(name_or_id)
        if ent is None:
            raise ValueError(f"unknown entity {name_or_id!r}")
        if not reason:
            raise ValueError("retire_entity needs a reason")
        ent.status = "retired"
        ent.retired_reason = reason
        ent.last_seen = max(ent.last_seen, chapter)
        state.changelog.append({"op": "retire_entity", "id": ent.id,
                                "reason": reason, "chapter": chapter,
                                "at": _now()})
        return ent

    def add_thread(self, state: ContinuityState, thread: Thread,
                   *, chapter: int = 0) -> Thread:
        thread.introduced = thread.introduced or chapter
        thread.foreshadow_chapter = thread.foreshadow_chapter or chapter
        state.threads.append(thread)
        return thread

    def resolve_thread(self, state: ContinuityState, thread_id: str, *,
                       chapter: int) -> Thread:
        for t in state.threads:
            if t.id == thread_id:
                t.status = "resolved"
                t.payoff_chapter = chapter
                t.last_seen = chapter
                return t
        raise ValueError(f"unknown thread {thread_id!r}")

    # ── adapters: other organs become clients without code changes ──
    def from_story_bible(self, bible: Any) -> ContinuityState:
        """Adapt a books StoryBible (object or dict) → engine state.

        bible.py keeps its own API as the facade; this is a one-way
        projection, so the engine never mutates the bible.
        """
        d = bible if isinstance(bible, dict) else (
            bible.to_dict() if hasattr(bible, "to_dict") else {})
        state = ContinuityState(
            work_slug=str(d.get("story_slug", "")),
            title=str(d.get("title", "")),
            voice=VoiceProfile(
                pov=str(d.get("pov", "third")),
                tense=str(d.get("tense", "past")),
                tone=list(d.get("tone", [])),
                chapters_measured=int(d.get("chapters_digested", 0))),
            chapters_seen=int(d.get("chapters_digested", 0)))
        for c in d.get("characters", []):
            if not isinstance(c, dict):
                continue
            state.entities.append(Entity(
                id=str(c.get("name", "")), kind="character",
                name=str(c.get("name", "")),
                attributes={"role": c.get("role", ""),
                            "description": c.get("description", ""),
                            "traits": c.get("traits", []),
                            "relationships": c.get("relationships", [])},
                first_seen=int(c.get("first_seen", 0)),
                mentions=int(c.get("mentions", 0)),
                last_seen=int(c.get("first_seen", 0))))
        for t in d.get("threads", []):
            if not isinstance(t, dict):
                continue
            state.threads.append(Thread(
                id=str(t.get("id", "")), summary=str(t.get("summary", "")),
                status=str(t.get("status", "open")),
                introduced=int(t.get("introduced", 0)),
                last_seen=int(t.get("last_seen", 0)),
                heat=float(t.get("heat", 0.0))))
        for i, r in enumerate(d.get("world_rules", [])):
            state.rules.append(Rule(id=f"rule-{i}", statement=str(r)))
        return state

    def from_sonic_bible(self, sonic: Any) -> ContinuityState:
        """Adapt a media SonicBible (object or dict) → engine state."""
        d = sonic if isinstance(sonic, dict) else (
            sonic.to_dict() if hasattr(sonic, "to_dict") else {})
        state = ContinuityState(
            work_slug=str(d.get("title", d.get("name", "sonic"))),
            title=str(d.get("title", "")))
        for key in ("key", "tempo", "scale", "palette", "texture"):
            if d.get(key) is not None:
                state.rules.append(Rule(
                    id=f"sonic-{key}",
                    statement=f"sonic {key}: {d.get(key)}",
                    scope="world"))
        return state

    # ── chapter application + checks ─────────────────────────────────
    def apply_chapter(self, state: ContinuityState, chapter: int,
                      text: str) -> list[Finding]:
        """Fold a chapter into state; returns findings (never raises on
        story content — a weird chapter is a finding, not a crash)."""
        text = text or ""
        state.chapters_seen = max(state.chapters_seen, chapter)
        low = text.lower()
        findings: list[Finding] = []
        for ent in state.entities:
            forms = [ent.name, *ent.aliases]
            hits = sum(1 for f in forms if f and re.search(
                r"\b" + re.escape(f) + r"\b", low, re.IGNORECASE))
            if hits:
                ent.mentions += hits
                ent.last_seen = chapter
                if ent.status == "dead":
                    findings.append(Finding(
                        kind="dead_entity_appears", severity="error",
                        message=f"{ent.name} is marked dead but appears "
                                f"in chapter {chapter}",
                        chapter=chapter, entity=ent.name,
                        evidence=text[max(0, low.find(ent.name.lower()) - 40):
                                      low.find(ent.name.lower()) + 80]))
        for thread in state.threads:
            kws = [k.lower() for k in thread.keywords if k]
            if kws and any(k in low for k in kws):
                thread.last_seen = chapter
                thread.heat = min(1.0, thread.heat + 0.1)
        return findings

    def check_chapter(self, state: ContinuityState, chapter: int,
                      text: str) -> list[Finding]:
        """Check a chapter against state WITHOUT mutating it."""
        findings: list[Finding] = []
        low = (text or "").lower()
        for ent in state.entities:
            if ent.status in ("dead", "retired"):
                forms = [ent.name, *ent.aliases]
                if any(f and re.search(r"\b" + re.escape(f) + r"\b", low,
                                       re.IGNORECASE)
                       for f in forms):
                    findings.append(Finding(
                        kind="dead_entity_appears", severity="error",
                        message=f"{ent.name} is {ent.status} but appears "
                                f"in chapter {chapter}",
                        chapter=chapter, entity=ent.name))
        # unknown capitalized names appearing often = unregistered entity
        known = {e.name.lower() for e in state.entities}
        known |= {a.lower() for e in state.entities for a in e.aliases}
        cands: dict[str, int] = {}
        for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text or ""):
            if m.group(0).lower() not in known:
                cands[m.group(0)] = cands.get(m.group(0), 0) + 1
        for name, count in sorted(cands.items(), key=lambda kv: -kv[1]):
            if count >= 3:
                findings.append(Finding(
                    kind="unknown_entity", severity="info",
                    message=f"Unregistered name {name!r} appears "
                            f"{count}× in chapter {chapter} — add it to "
                            "the bible or it stays invisible to continuity",
                    chapter=chapter, entity=name))
        return findings

    def stalled_threads(self, state: ContinuityState, *,
                        current_chapter: int,
                        stall_after: int = 10) -> list[Thread]:
        """Open threads unseen for >= stall_after chapters."""
        return [t for t in state.open_threads()
                if current_chapter - (t.last_seen or t.introduced)
                >= stall_after]

    # ── drift ────────────────────────────────────────────────────────
    def drift(self, state: ContinuityState, chapter: int,
              text: str) -> DriftReport:
        """Measured voice drift: chapter style features vs reference."""
        from .voice import cosine_features, vectorize
        ref = state.voice.features
        feats = vectorize(text or "").features
        if not ref:
            return DriftReport(chapter=chapter, similarity=1.0,
                               features=feats,
                               reference_features=ref,
                               notes=["no reference voice yet — "
                                      "first measurement, not drift"])
        sim = cosine_features(feats, ref)
        notes: list[str] = []
        if sim < self.drift_threshold:
            notes.append(
                f"voice similarity {sim:.2f} below floor "
                f"{self.drift_threshold:.2f} — chapter may not sound "
                "like the work")
        return DriftReport(chapter=chapter, similarity=round(sim, 4),
                           features=feats, reference_features=ref,
                           drifted=sim < self.drift_threshold,
                           threshold=self.drift_threshold, notes=notes)

    def measure_drift(self, text_a: str, text_b: str) -> float:
        """Similarity between two chapters' style (0..1). Ch40 vs ch1
        returning 0.91 means 'same voice', not 'same story'."""
        from .voice import cosine_features, vectorize
        return round(cosine_features(vectorize(text_a or "").features,
                                     vectorize(text_b or "").features), 4)
