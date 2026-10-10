"""Prompt 11 — Persona Depth: memory that deepens into a person-shaped model.

Turns Devon's memory from a searchable log into a living model of the owner:
who they are, how they like things done, who matters to them, what they're
working toward — recalled proactively, curated automatically, and fully
visible/editable by the owner.

The user model is a *materialized view* over ``MemoryManager`` records, NOT a
separate database.  It is rebuilt on a schedule (daily) and on demand.

Design notes (survey-first, per the standing rule):
- ``nomorals/memory/base.py`` (MemoryKind, MemoryRecord, score_memory) and
  ``manager.py`` (MemoryManager) are consumed, never replaced.
- ``nomorals/agents/proactive.py`` (ProactiveEngine) already does in-memory
  action *suggestions*; this module does evidence-backed *routine detection*
  from stored EPISODE records — a different job, no duplication.
- The ``private`` metadata flag is honored by recall/build_context/for_training.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .base import MemoryKind, MemoryRecord

_log = get_logger(__name__)

__all__ = [
    "UserModel",
    "ModelAttribute",
    "Routine",
    "Interest",
    "PeopleGraph",
    "PersonEntry",
    "PersonaGuide",
    "ProactiveRecall",
    "MemoryCurator",
    "register",
]

#: confidence floor — attributes below this are never acted on silently
CONFIDENCE_FLOOR = 0.45
#: routines graduate from proposed → established at this many supporting episodes
ROUTINE_ESTABLISHED_AT = 5
#: PersonaGuide hard cap (spec: 5–10 lines)
GUIDE_MAX_LINES = 10


def _is_private(record: MemoryRecord) -> bool:
    try:
        return bool((record.metadata or {}).get("private"))
    except Exception:  # noqa: BLE001
        return False


def _evidence_ids(record: MemoryRecord) -> list[str]:
    return [record.id]


# ── model attributes ───────────────────────────────────────────────────────

@dataclass
class ModelAttribute:
    """One synthesized attribute of the user model.

    Every attribute carries confidence (0..1) + evidence (record ids).
    Low-confidence attributes are offered, never acted on silently.
    """

    name: str
    value: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    kind: str = ""  # memory kind it was synthesized from

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "kind": self.kind,
        }

    @property
    def actionable(self) -> bool:
        """May Devon act on this without asking first?"""
        return self.confidence >= CONFIDENCE_FLOOR


@dataclass
class Interest:
    topic: str
    heat: float  # 0..1 normalized
    mentions: int
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "heat": round(self.heat, 3),
            "mentions": self.mentions,
            "evidence": list(self.evidence),
        }


@dataclass
class Routine:
    """A detected behavioral pattern.  Proposed, never asserted."""

    description: str
    supporting_episodes: int
    status: str = "proposed"  # proposed | established
    evidence: list[str] = field(default_factory=list)
    last_seen: float = 0.0
    time_hint: str = ""  # e.g. "weekday mornings"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "supporting_episodes": self.supporting_episodes,
            "status": self.status,
            "evidence": list(self.evidence),
            "last_seen": self.last_seen,
            "time_hint": self.time_hint,
        }


# ── user model ─────────────────────────────────────────────────────────────

_IDENTITY_KEYS = ("name", "call_me", "timezone", "locale", "location")


@dataclass
class UserModel:
    """Structured, queryable synthesis of the owner — a materialized view
    over MemoryManager records.  Rebuilt daily and on demand."""

    identity: dict[str, ModelAttribute] = field(default_factory=dict)
    preferences: list[ModelAttribute] = field(default_factory=list)
    interests: list[Interest] = field(default_factory=list)
    routines: list[Routine] = field(default_factory=list)
    goals_in_flight: list[ModelAttribute] = field(default_factory=list)
    built_at: float = 0.0
    record_count: int = 0

    # ── rebuild ──────────────────────────────────────────────────────
    @classmethod
    def rebuild(cls, manager: Any) -> "UserModel":
        """Synthesize the model from the manager's records."""
        model = cls(built_at=time.time())
        try:
            facts = manager.recall("", limit=500, kind=MemoryKind.FACT)
            prefs = manager.recall("", limit=500, kind=MemoryKind.PREFERENCE)
            episodes = manager.recall("", limit=500, kind=MemoryKind.EPISODE)
            decisions = manager.recall("", limit=200, kind=MemoryKind.DECISION)
        except Exception as exc:  # noqa: BLE001 — empty model beats crash
            _log.warning("persona rebuild failed: %s", exc)
            return model

        fact_records = [r for r in facts.records if not _is_private(r)]
        pref_records = [r for r in prefs.records if not _is_private(r)]
        ep_records = [r for r in episodes.records if not _is_private(r)]
        dec_records = [r for r in decisions.records if not _is_private(r)]
        model.record_count = (len(fact_records) + len(pref_records)
                              + len(ep_records) + len(dec_records))

        model.identity = cls._build_identity(fact_records)
        model.preferences = cls._build_preferences(pref_records)
        model.interests = cls._build_interests(ep_records + fact_records)
        model.routines = cls._detect_routines(ep_records)
        model.goals_in_flight = cls._build_goals(dec_records)
        return model

    # ── builders ─────────────────────────────────────────────────────
    @staticmethod
    def _build_identity(
            facts: list[MemoryRecord]) -> dict[str, ModelAttribute]:
        identity: dict[str, ModelAttribute] = {}
        for record in facts:
            text = record.content.strip()
            lowered = text.lower()
            for key in _IDENTITY_KEYS:
                if key in identity:
                    continue
                # explicit "key: value" or "my key is value" statements only
                m = re.search(
                    rf"(?:my\s+)?{re.escape(key.replace('_', ' '))}\s*(?:is|:)\s*(.+)",
                    lowered)
                if m:
                    value = m.group(1).strip().strip(".")[:80]
                    if value:
                        identity[key] = ModelAttribute(
                            name=key, value=value, confidence=0.9,
                            evidence=_evidence_ids(record), kind="fact")
        return identity

    @staticmethod
    def _build_preferences(
            prefs: list[MemoryRecord]) -> list[ModelAttribute]:
        out: list[ModelAttribute] = []
        for record in prefs:
            # confidence from importance + reinforcement.  A barely-stated
            # preference (importance ~0) lands below the floor: offered,
            # never acted on silently.
            conf = min(0.95, 0.30 + record.importance * 0.55
                       + record.reinforcement())
            out.append(ModelAttribute(
                name=f"preference:{record.id[:8]}",
                value=record.content.strip()[:200],
                confidence=round(conf, 3),
                evidence=_evidence_ids(record), kind="preference"))
        # most-reinforced first
        out.sort(key=lambda a: a.confidence, reverse=True)
        return out

    @staticmethod
    def _build_interests(
            records: list[MemoryRecord]) -> list[Interest]:
        counts: Counter[str] = Counter()
        evidence: dict[str, list[str]] = defaultdict(list)
        stop = {"the", "a", "an", "and", "for", "with", "this", "that",
                "from", "about", "devon", "please"}
        for record in records:
            words = re.findall(r"[a-z][a-z0-9\-]{3,}", record.content.lower())
            seen: set[str] = set()
            for w in words:
                if w in stop or w in seen:
                    continue
                seen.add(w)
                counts[w] += 1
                if len(evidence[w]) < 5:
                    evidence[w].append(record.id)
        if not counts:
            return []
        top = counts.most_common(12)
        peak = top[0][1]
        return [Interest(topic=t, heat=round(c / peak, 3), mentions=c,
                         evidence=evidence[t]) for t, c in top if c >= 2]

    @staticmethod
    def _detect_routines(
            episodes: list[MemoryRecord]) -> list[Routine]:
        """Cluster episodes by normalized action signature.

        A routine proposes itself at ≥2 occurrences and graduates to
        ``established`` at ≥5 — never asserted, always evidenced.
        """
        clusters: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in episodes:
            sig = _action_signature(record.content)
            if sig:
                clusters[sig].append(record)
        routines: list[Routine] = []
        for sig, recs in clusters.items():
            if len(recs) < 2:
                continue
            recs.sort(key=lambda r: r.created_at)
            routines.append(Routine(
                description=sig,
                supporting_episodes=len(recs),
                status=("established" if len(recs) >= ROUTINE_ESTABLISHED_AT
                        else "proposed"),
                evidence=[r.id for r in recs[-8:]],
                last_seen=recs[-1].created_at,
                time_hint=_time_hint(recs),
            ))
        routines.sort(key=lambda r: r.supporting_episodes, reverse=True)
        return routines[:10]

    @staticmethod
    def _build_goals(
            decisions: list[MemoryRecord]) -> list[ModelAttribute]:
        out: list[ModelAttribute] = []
        for record in decisions:
            if (record.metadata or {}).get("superseded_by"):
                continue  # superseded decisions aren't in flight
            out.append(ModelAttribute(
                name=f"goal:{record.id[:8]}",
                value=record.content.strip()[:200],
                confidence=round(min(0.9, 0.5 + record.importance * 0.4), 3),
                evidence=_evidence_ids(record), kind="decision"))
        return out

    # ── serialization ────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": {k: v.to_dict() for k, v in self.identity.items()},
            "preferences": [p.to_dict() for p in self.preferences],
            "interests": [i.to_dict() for i in self.interests],
            "routines": [r.to_dict() for r in self.routines],
            "goals_in_flight": [g.to_dict() for g in self.goals_in_flight],
            "built_at": self.built_at,
            "record_count": self.record_count,
        }


def _action_signature(text: str) -> str:
    """Normalize an episode into a coarse action signature for clustering."""
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # strip numbers/dates/tickers — we want the *shape* of the request
    t = re.sub(r"\b\d[\d.,]*\b", "#", t)
    words = t.split()
    # drop low-signal words, keep the verb-ish head
    drop = {"the", "a", "an", "and", "for", "with", "please", "me", "my",
            "to", "of", "on", "in", "is", "it", "devon", "hey", "hi"}
    kept = [w for w in words if w not in drop][:8]
    if len(kept) < 2:
        return ""
    return " ".join(kept)


def _time_hint(records: list[MemoryRecord]) -> str:
    """Crude time-of-day/day-of-week hint from episode timestamps."""
    if not records:
        return ""
    hours = Counter()
    weekdays = Counter()
    for r in records:
        lt = time.localtime(r.created_at)
        hours["morning" if 5 <= lt.tm_hour < 12
              else "afternoon" if 12 <= lt.tm_hour < 18
              else "evening" if 18 <= lt.tm_hour < 23 else "night"] += 1
        weekdays["weekday" if lt.tm_wday < 5 else "weekend"] += 1
    top_hour = hours.most_common(1)[0]
    top_day = weekdays.most_common(1)[0]
    hint = top_hour[0]
    if top_day[1] >= len(records) * 0.7:
        hint = f"{top_day[0]} {hint}s"
    elif top_hour[1] >= len(records) * 0.6:
        hint = f"{hint}s"
    else:
        hint = ""
    return hint


# ── people graph ───────────────────────────────────────────────────────────

@dataclass
class PersonEntry:
    name: str
    role: str = ""
    context: str = ""
    last_mentioned: float = 0.0
    mention_count: int = 0
    notes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "context": self.context,
            "last_mentioned": self.last_mentioned,
            "mention_count": self.mention_count,
            "notes": list(self.notes),
            "evidence": list(self.evidence),
        }


class Community:
    """A cluster of strongly-connected people — the community subgraph.

    Graphiti's Gc tier: label-propagation clusters over the people graph,
    each with a synthesized summary. Communities answer "who moves together"
    — the girlfriend + her family, the work crew, the trading circle — which
    no per-person lookup can give.
    """

    def __init__(self, members: list[PersonEntry], edges: int = 0) -> None:
        self.members = list(members)
        self.edges = edges
        self.built_at = time.time()

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def summary(self, max_notes: int = 3) -> str:
        """One synthesized paragraph: who they are together."""
        roles = sorted({m.role for m in self.members if m.role})
        head = ", ".join(self.names[:6])
        if len(self.names) > 6:
            head += f" (+{len(self.names) - 6} more)"
        bits = [f"Circle of {self.size}: {head}."]
        if roles:
            bits.append("Roles: " + "; ".join(roles[:4]) + ".")
        notes: list[str] = []
        for m in self.members:
            for n in m.notes:
                if n not in notes:
                    notes.append(n)
                if len(notes) >= max_notes:
                    break
            if len(notes) >= max_notes:
                break
        if notes:
            bits.append("Known: " + " ".join(notes))
        return " ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {"members": [m.to_dict() for m in self.members],
                "edges": self.edges, "size": self.size,
                "summary": self.summary(), "built_at": self.built_at}


class PeopleGraph:
    """Graph view over RELATIONSHIP records: who matters to the owner.

    Stores only what the owner stated, verbatim-ish.  No inference beyond
    what was said. Co-occurrence edges (two people named in the same
    record) power the community subgraph — "who moves together".
    """

    def __init__(self) -> None:
        self.people: dict[str, PersonEntry] = {}
        #: (key_a, key_b) → weight: people named in the same record.
        self.edges: dict[tuple[str, str], int] = {}

    @classmethod
    def build(cls, manager: Any) -> "PeopleGraph":
        graph = cls()
        try:
            result = manager.recall("", limit=500,
                                    kind=MemoryKind.RELATIONSHIP)
        except Exception as exc:  # noqa: BLE001
            _log.warning("people graph build failed: %s", exc)
            return graph
        for record in result.records:
            if _is_private(record):
                continue
            name = (record.metadata or {}).get("person") or _guess_name(
                record.content)
            if not name:
                continue
            key = name.lower()
            entry = graph.people.get(key)
            if entry is None:
                entry = PersonEntry(name=name)
                graph.people[key] = entry
            entry.mention_count += 1
            entry.last_mentioned = max(entry.last_mentioned,
                                       record.created_at)
            role = (record.metadata or {}).get("role", "")
            if role and not entry.role:
                entry.role = role
            note = record.content.strip()[:160]
            if note not in entry.notes and len(entry.notes) < 8:
                entry.notes.append(note)
            if record.id not in entry.evidence and len(entry.evidence) < 8:
                entry.evidence.append(record.id)
            # Co-occurrence edges: other people named in the same record
            # move together. ``mentioned`` metadata or extra names found
            # in the content both count; self-pairs excluded.
            others = set()
            mentioned = (record.metadata or {}).get("mentioned") or []
            if isinstance(mentioned, str):
                mentioned = [mentioned]
            for other in mentioned:
                if other and str(other).lower() != key:
                    others.add(str(other).lower())
            for other in others:
                pair = tuple(sorted((key, other)))
                graph.edges[pair] = graph.edges.get(pair, 0) + 1
        return graph

    def lookup(self, name: str) -> PersonEntry | None:
        return self.people.get((name or "").lower())

    def neighbors(self, name: str) -> list[tuple[PersonEntry, int]]:
        """People connected to ``name`` with edge weights, strongest first."""
        key = (name or "").lower()
        out: list[tuple[PersonEntry, int]] = []
        for (a, b), weight in self.edges.items():
            other = b if a == key else a if b == key else None
            if other is not None and other in self.people:
                out.append((self.people[other], weight))
        out.sort(key=lambda t: -t[1])
        return out

    def communities(self, min_size: int = 2,
                    max_rounds: int = 20) -> list[Community]:
        """Label-propagation clustering over the people graph.

        The community subgraph (Graphiti's Gc): clusters of strongly
        connected people with synthesized summaries. Deterministic —
        ties break on label id, so the same graph always yields the same
        communities. Singletons (no edges) are dropped unless
        ``min_size=1``.
        """
        nodes = sorted(self.people)
        if not nodes:
            return []
        adjacency: dict[str, dict[str, int]] = {n: {} for n in nodes}
        for (a, b), weight in self.edges.items():
            if a in adjacency and b in adjacency:
                adjacency[a][b] = adjacency[a].get(b, 0) + weight
                adjacency[b][a] = adjacency[b].get(a, 0) + weight
        labels = {n: n for n in nodes}
        for _ in range(max_rounds):
            changed = False
            for node in nodes:
                counts: dict[str, int] = {}
                for nbr, weight in adjacency[node].items():
                    lab = labels[nbr]
                    counts[lab] = counts.get(lab, 0) + weight
                if not counts:
                    continue
                best = max(sorted(counts), key=lambda lab: counts[lab])
                if best != labels[node]:
                    labels[node] = best
                    changed = True
            if not changed:
                break
        groups: dict[str, list[str]] = {}
        for node, lab in labels.items():
            groups.setdefault(lab, []).append(node)
        out: list[Community] = []
        for members in groups.values():
            if len(members) < max(1, min_size):
                continue
            edge_count = sum(
                w for (a, b), w in self.edges.items()
                if a in members and b in members)
            out.append(Community(
                [self.people[m] for m in sorted(members)],
                edges=edge_count))
        out.sort(key=lambda c: (-c.size, c.names[0] if c.names else ""))
        return out

    def subgraph(self, name: str) -> dict[str, Any]:
        """The person's entry for recall boosting — synthesized, not raw."""
        entry = self.lookup(name)
        if entry is None:
            return {}
        return entry.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.people.items()}


def _guess_name(text: str) -> str:
    """Best-effort name extraction from a relationship record.

    Only used when the record lacks explicit ``person`` metadata — and the
    result is still just a lookup key, never an inferred attribute.
    """
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text or "")
    return m.group(1) if m else ""


# ── persona guide ──────────────────────────────────────────────────────────

class PersonaGuide:
    """5–10 line system-context injection synthesized from the user model.

    Hard rules: ≤10 lines, no raw episode text, no private-marked records.
    Rebuilt when the model changes.
    """

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)[:GUIDE_MAX_LINES]
        self.built_at = time.time()

    @classmethod
    def build(cls, model: UserModel) -> "PersonaGuide":
        lines: list[str] = []
        ident = model.identity
        if "call_me" in ident:
            lines.append(f"Call the owner {ident['call_me'].value}.")
        elif "name" in ident:
            lines.append(f"Owner's name: {ident['name'].value}.")
        if "timezone" in ident:
            lines.append(f"Owner timezone: {ident['timezone'].value}.")
        # preferences — actionable ones only, synthesized phrasing
        for pref in model.preferences[:4]:
            if pref.actionable:
                lines.append(f"Preference: {pref.value}")
        # interests shape ordering
        if model.interests:
            top = ", ".join(i.topic for i in model.interests[:4])
            lines.append(f"Top interests: {top}.")
        # established routines shape anticipation
        for routine in model.routines[:2]:
            if routine.status == "established":
                lines.append(f"Routine: {routine.description}"
                             + (f" ({routine.time_hint})"
                                if routine.time_hint else "") + ".")
        # goals give direction
        if model.goals_in_flight:
            lines.append(
                f"Working toward: {model.goals_in_flight[0].value[:80]}.")
        if not lines:
            lines.append("No user model yet — defaults apply; learn as you go.")
        return cls(lines)

    def text(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {"lines": list(self.lines), "built_at": self.built_at}


# ── proactive recall ───────────────────────────────────────────────────────

class ProactiveRecall:
    """Anticipation pass over the user model: given the current request +
    time, pull relevant routines/preferences WITHOUT being asked.

    Anti-creep rules (hard):
    - NEVER surfaces raw EPISODE text unprompted — only synthesized attributes.
    - NEVER surfaces private-marked records.
    - NEVER surfaces attributes below the confidence floor.
    - Read-only over memory: changes what the agent SEES, never what it DOES.
    """

    def __init__(self, model: UserModel) -> None:
        self.model = model

    def anticipate(self, request_text: str,
                   now: float | None = None) -> list[str]:
        """Return short synthesized hints for the agent's context."""
        now = now or time.time()
        hints: list[str] = []
        req = (request_text or "").lower()

        # routines whose signature overlaps the request
        for routine in self.model.routines:
            if routine.status != "established":
                continue
            sig_words = set(routine.description.split())
            req_words = set(re.findall(r"[a-z]{3,}", req))
            if sig_words & req_words:
                hints.append(
                    f"Routine ({routine.supporting_episodes}×"
                    + (f", {routine.time_hint}" if routine.time_hint else "")
                    + f"): {routine.description}.")

        # preferences whose text overlaps the request
        for pref in self.model.preferences:
            if not pref.actionable:
                continue
            pref_words = set(re.findall(r"[a-z]{3,}", pref.value.lower()))
            if pref_words & set(re.findall(r"[a-z]{3,}", req)):
                hints.append(f"Owner preference: {pref.value[:120]}")

        return hints[:4]

    def for_context_block(self, request_text: str) -> str:
        hints = self.anticipate(request_text)
        if not hints:
            return ""
        return ("What I know about the owner (proactive recall):\n"
                + "\n".join(f"- {h}" for h in hints))


# ── memory curation ────────────────────────────────────────────────────────

class MemoryCurator:
    """Automatic hygiene over MemoryManager records.

    - Deduplication: merges near-duplicate records (same kind, high
      similarity), keeps the highest-confidence one, links merged ids.
    - Contradictions: a new FACT/PREFERENCE contradicting an old one marks
      the old ``superseded_by`` — both kept, one gentle owner note.
    - Scheduled ``forget_below()`` with a conservative default; deletions go
      to the archive table (30-day undo), never hard-deleted first pass.
    """

    # conservative default: only clear obvious cruft
    FORGET_THRESHOLD = 0.08
    DEDUP_SIMILARITY = 0.82

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.db = getattr(manager, "db", None)

    # ── dedup ────────────────────────────────────────────────────────
    def deduplicate(self, kind: str = "") -> dict[str, Any]:
        """Merge near-duplicate records.  Returns a merge report."""
        from .extract import is_duplicate

        try:
            result = self.manager.recall("", limit=2000,
                                         kind=kind or "")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        records = [r for r in result.records
                   if not _is_private(r)
                   and not (r.metadata or {}).get("superseded_by")]
        by_kind: dict[str, list[MemoryRecord]] = defaultdict(list)
        for r in records:
            by_kind[r.kind].append(r)

        merged = 0
        kept: list[str] = []
        for kind_records in by_kind.values():
            # highest confidence first → winner survives
            kind_records.sort(
                key=lambda r: (r.importance + r.reinforcement()),
                reverse=True)
            survivors: list[MemoryRecord] = []
            for record in kind_records:
                dup_of = None
                for s in survivors:
                    try:
                        if is_duplicate(record.content, s.content,
                                        self.DEDUP_SIMILARITY):
                            dup_of = s
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if dup_of is None:
                    survivors.append(record)
                    continue
                # merge into the survivor: link + reinforce
                meta = dict(dup_of.metadata or {})
                merged_ids = list(meta.get("merged_ids", []))
                if record.id not in merged_ids:
                    merged_ids.append(record.id)
                meta["merged_ids"] = merged_ids
                try:
                    self.manager.update(dup_of.id, metadata=meta)
                    self._archive(record.id, reason="dedup",
                                  superseded_by=dup_of.id)
                    self.manager.forget(record.id)
                    merged += 1
                except Exception as exc:  # noqa: BLE001
                    _log.warning("dedup merge failed: %s", exc)
            kept.extend(s.id for s in survivors)
        return {"ok": True, "merged": merged, "kept": len(kept)}

    # ── contradictions ───────────────────────────────────────────────
    # crude opposition markers — a contradiction is "X" vs "not X" /
    # "prefer X" vs "prefer Y" on the same topic, not a subtle judgment
    _NEGATIONS = ("not ", "n't ", "never ", "no longer ", "don't ",
                  "doesn't ", "dont ", "dont like")

    def check_contradiction(self, new_id: str) -> dict[str, Any]:
        """After a FACT/PREFERENCE is stored, look for an older record it
        contradicts.  Marks the old ``superseded_by``; returns a gentle
        one-time owner note (or {} when nothing contradicts)."""
        new = self.manager.get(new_id)
        if new is None or new.kind not in (MemoryKind.FACT,
                                           MemoryKind.PREFERENCE):
            return {}
        if _is_private(new):
            return {}
        try:
            result = self.manager.recall(new.content, limit=12,
                                         kind=new.kind)
        except Exception:  # noqa: BLE001
            return {}
        for old in result.records:
            if old.id == new.id or _is_private(old):
                continue
            if (old.metadata or {}).get("superseded_by"):
                continue
            if self._contradicts(old.content, new.content):
                meta = dict(old.metadata or {})
                meta["superseded_by"] = new.id
                try:
                    self.manager.update(old.id, metadata=meta)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("supersede update failed: %s", exc)
                    continue
                note = (f"noted: you used to prefer "
                        f"“{old.content.strip()[:80]}” — "
                        f"updated to “{new.content.strip()[:80]}”. "
                        f"still true?")
                return {"superseded": old.id, "by": new.id, "note": note}
        return {}

    @classmethod
    def _contradicts(cls, old: str, new: str) -> bool:
        """Cheap contradiction test: shared topic words + negation flip."""
        o, n = old.lower(), new.lower()
        o_words = set(re.findall(r"[a-z]{4,}", o))
        n_words = set(re.findall(r"[a-z]{4,}", n))
        shared = o_words & n_words
        if len(shared) < 2:
            return False
        o_neg = any(neg in o for neg in cls._NEGATIONS)
        n_neg = any(neg in n for neg in cls._NEGATIONS)
        return o_neg != n_neg

    # ── scheduled hygiene ────────────────────────────────────────────
    def scheduled_pass(self) -> dict[str, Any]:
        """One curation tick: dedup + conservative forget_below."""
        report: dict[str, Any] = {"dedup": {}, "forgotten": 0}
        report["dedup"] = self.deduplicate()
        try:
            # archive before forgetting so nothing vanishes surprisingly
            report["forgotten"] = self._forget_with_archive(
                self.FORGET_THRESHOLD)
        except Exception as exc:  # noqa: BLE001
            report["error"] = str(exc)
        return report

    def _forget_with_archive(self, threshold: float) -> int:
        """forget_below() but every deletion lands in the archive first."""
        from .base import MemoryKind as MK
        now = time.time()
        try:
            rows = self.db.query(
                "SELECT id, importance, decay, created_at, kind FROM memories")
        except Exception:  # noqa: BLE001
            return 0
        doomed: list[str] = []
        for row in rows:
            record = MemoryRecord.from_row({**row, "content": ""})
            if record.kind in {MK.FACT, MK.PREFERENCE}:
                continue  # never forget stated facts
            if (record.metadata or {}).get("private"):
                continue  # private records are never auto-forgotten
            if record.recency(self.manager.half_life, now) * \
                    record.importance < threshold:
                doomed.append(row["id"])
        for record_id in doomed:
            self._archive(record_id, reason="forget_below")
            try:
                self.manager.forget(record_id)
            except Exception as exc:  # noqa: BLE001
                _log.warning("archived forget failed: %s", exc)
        return len(doomed)

    def _archive(self, record_id: str, reason: str,
                 superseded_by: str = "") -> bool:
        """Copy a record into ``memory_archive`` (30-day undo)."""
        if self.db is None:
            return False
        try:
            record = self.manager.get(record_id)
            if record is None:
                return False
            self.db.execute(
                "INSERT OR REPLACE INTO memory_archive "
                "(id, kind, content, importance, metadata, archived_at, "
                " reason, superseded_by) VALUES (?,?,?,?,?,?,?,?)",
                (record.id, record.kind, record.content, record.importance,
                 json.dumps(record.metadata or {}), time.time(), reason,
                 superseded_by or ""))
            return True
        except Exception as exc:  # noqa: BLE001
            _log.warning("memory archive failed: %s", exc)
            return False

    def restore(self, record_id: str) -> bool:
        """Undo an archived deletion within the 30-day window."""
        if self.db is None:
            return False
        try:
            rows = self.db.query(
                "SELECT * FROM memory_archive WHERE id = ?", (record_id,))
        except Exception:  # noqa: BLE001
            return False
        if not rows:
            return False
        row = rows[0]
        if time.time() - float(row["archived_at"] or 0) > 30 * 86400:
            return False  # undo window expired
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:  # noqa: BLE001
            metadata = {}
        new_id_ = self.manager.remember(
            row["content"], kind=row["kind"],
            importance=float(row["importance"] or 0.5),
            metadata=metadata, source="restore")
        return bool(new_id_)


# ── agent tool registration ────────────────────────────────────────────────

def _manager(context: Any) -> Any:
    manager = getattr(context, "memory", None)
    if manager is None:
        raise RuntimeError("memory manager not available on this context")
    return manager


def register(registry: Any) -> None:
    """Register the ``memory`` agent tool.

    This is the correction path the spec demands: "that's wrong" /
    "forget that" in chat → immediate update()/forget() + confirmation.
    Read-only actions are the default; writes are explicit.
    """
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "memory",
        description=(
            "Inspect and manage the owner's memory/persona model. "
            "action=recall (query) | show (user model) | list (kind/query) | "
            "remember (text, kind) | update (record_id, text) | forget "
            "(record_id) | private (record_id) | public (record_id) | "
            "export | rebuild | curate | anticipate (query) | consolidate | "
            "health | repair | contradictions (record_id) | timeline (query) "
            "| deep (query) | related (record_id) | scopes | schedule | "
            "backup (dest) | import (path). "
            "Writes are explicit; recall never returns private records. "
            "Consolidation is additive — nothing is ever deleted by it."
        ),
        capability=Capability.MEM_WRITE,
        parameters={
            "action": "str — recall|show|list|remember|update|forget|private|"
                      "public|export|rebuild|curate|anticipate|consolidate|"
                      "health|repair|contradictions|timeline|deep|related|"
                      "scopes|schedule|backup|import",
            "query": "str (optional) — search text",
            "record_id": "str (optional) — target record",
            "kind": "str (optional) — memory kind filter",
            "text": "str (optional) — content for remember/update",
            "limit": "int (optional) — result limit",
            "scope": "str (optional) — memory space, e.g. project:devon",
            "dest": "str (optional) — backup destination directory",
            "path": "str (optional) — export bundle path for import",
            "dry_run": "bool (optional) — verify without writing",
        },
    )
    def memory(*, action: str = "recall", query: str = "",
               record_id: str = "", kind: str = "", text: str = "",
               limit: int = 8, scope: str = "", dest: str = "",
               path: str = "", dry_run: bool = False) -> dict[str, Any]:
        manager = _manager(context)
        if action == "recall":
            result = manager.recall(query, limit=limit, kind=kind or "",
                                    scope=scope)
            return {"records": [r.to_dict() for r in result.records]}
        if action == "show":
            model = UserModel.rebuild(manager)
            guide = PersonaGuide.build(model)
            return {"model": model.to_dict(), "guide": guide.to_dict()}
        if action == "list":
            result = manager.recall(query or "", limit=limit,
                                    kind=kind or "", scope=scope)
            return {"records": [
                {"id": r.id, "kind": r.kind,
                 "content": r.content[:200],
                 "private": _is_private(r),
                 "created_at": r.created_at}
                for r in result.records]}
        if action == "remember":
            body = text or query
            rid = manager.remember(body, kind=kind or MemoryKind.EPISODE,
                                   source="agent", scope=scope)
            note: dict[str, Any] = {}
            if kind in (MemoryKind.FACT, MemoryKind.PREFERENCE) and rid:
                note = MemoryCurator(manager).check_contradiction(rid)
            return {"ok": bool(rid), "id": rid, **note}
        if action == "update":
            if not record_id:
                return {"ok": False, "error": "record_id required"}
            n = manager.update(record_id, content=text)
            return {"ok": n > 0, "updated": n}
        if action == "forget":
            if not record_id:
                return {"ok": False, "error": "record_id required"}
            curator = MemoryCurator(manager)
            curator._archive(record_id, reason="agent-forget")
            n = manager.forget(record_id)
            return {"ok": n > 0, "forgotten": n,
                    "confirmation": "forgotten" if n else "not found"}
        if action == "private":
            n = manager.mark_private(record_id) if record_id else 0
            return {"ok": n > 0}
        if action == "public":
            n = manager.mark_public(record_id) if record_id else 0
            return {"ok": n > 0}
        if action == "export":
            return {"export": export_model(manager)}
        if action == "rebuild":
            model = UserModel.rebuild(manager)
            return {"model": model.to_dict()}
        if action == "curate":
            return MemoryCurator(manager).scheduled_pass()
        if action == "anticipate":
            model = UserModel.rebuild(manager)
            hints = ProactiveRecall(model).anticipate(query)
            return {"hints": hints}
        if action == "consolidate":
            # the scheduled tick, runnable by hand: additive only
            from .cadence import consolidate_now
            return consolidate_now(manager)
        if action == "health":
            return manager.health()
        if action == "repair":
            return manager.repair_embeddings(dry_run=bool(dry_run))
        if action == "contradictions":
            from .contradictions import detect_for
            if not record_id:
                return {"ok": False, "error": "record_id required"}
            found = detect_for(manager, record_id)
            return {"record_id": record_id,
                    "contradictions": [c.to_dict() for c in found]}
        if action == "timeline":
            from .deep_recall import timeline
            items = timeline(manager, query, limit=limit)
            return {"query": query,
                    "timeline": [r.to_dict() for r in items]}
        if action == "deep":
            from .deep_recall import recall_deep
            result = recall_deep(manager, query, limit=limit, scope=scope)
            return result.to_dict()
        if action == "related":
            from .deep_recall import related
            if not record_id:
                return {"ok": False, "error": "record_id required"}
            items = related(manager, record_id, limit=limit)
            return {"record_id": record_id,
                    "related": [r.to_dict() for r in items]}
        if action == "scopes":
            from .scopes import scopes_summary
            return {"scopes": scopes_summary(manager)}
        if action == "schedule":
            from .cadence import status
            return status(manager)
        if action == "backup":
            from .backup import backup_to
            if not dest:
                return {"ok": False, "error": "dest required"}
            return backup_to(manager, dest)
        if action == "import":
            from .backup import import_records
            import json as _json
            if not path:
                return {"ok": False, "error": "path required"}
            try:
                with open(path, encoding="utf-8") as fh:
                    bundle = _json.load(fh)
            except Exception as exc:
                return {"ok": False, "error": f"cannot read bundle: {exc}"}
            records = bundle.get("records", bundle) if isinstance(
                bundle, dict) else bundle
            return import_records(manager, records, dry_run=bool(dry_run))
        return {"ok": False, "error": f"unknown action: {action}"}



def export_model(manager: Any) -> dict[str, Any]:
    """Complete portable JSON dump: user model + people + raw records."""
    model = UserModel.rebuild(manager)
    graph = PeopleGraph.build(manager)
    records: list[dict[str, Any]] = []
    try:
        # the owner's own portable dump — includes private records (it's
        # their data), each flagged so re-import can restore the flag
        result = manager.recall("", limit=5000, include_private=True)
        for r in result.records:
            d = r.to_dict()
            d["private"] = _is_private(r)
            records.append(d)
    except Exception as exc:  # noqa: BLE001
        _log.warning("export records failed: %s", exc)
    return {
        "exported_at": time.time(),
        "user_model": model.to_dict(),
        "people": graph.to_dict(),
        "records": records,
    }
