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
    "SENSITIVE_PATTERNS",
    "is_sensitive_text",
    "ensure_persona_jobs",
    "register",
    "PERSONA_REBUILD_JOB",
    "PERSONA_CURATE_JOB",
]

#: confidence floor — attributes below this are never acted on silently
CONFIDENCE_FLOOR = 0.45
#: routines graduate from proposed → established at this many supporting episodes
ROUTINE_ESTABLISHED_AT = 5
#: PersonaGuide hard cap (spec: 5–10 lines)
GUIDE_MAX_LINES = 10


# ── sensitive-attribute guard ──────────────────────────────────────────────
# The owner owns their model.  We store only what was explicitly stated, and
# NEVER infer or persist sensitive attributes — even when baited.

SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("health", re.compile(
        r"\b(diagnos(?:ed|is)|disease|illness|disorder|syndrome|hiv|aids|"
        r"cancer|diabetes|depress(?:ion|ed)|anxi(?:ety|ous)|bipolar|"
        r"schizo|autis|adhd|medication|prescription|therap(?:y|ist)|"
        r"surgery|hospitali[sz]ed|disab(?:led|ility)|chronic pain)\b",
        re.IGNORECASE)),
    ("politics", re.compile(
        r"\b(vot(?:e|ed|ing) for|support(?:s|ed)? (?:the )?(?:party|candidate)|"
        r"democrat|republican|labour|conservative|apc\b|pdp\b|lp\b|"
        r"political (?:party|affiliation|leaning)|left-wing|right-wing|"
        r"socialist|fascist|communist)\b",
        re.IGNORECASE)),
    ("religion", re.compile(
        r"\b(christian|muslim|jew(?:ish)?|hindu|buddhist|atheist|agnostic|"
        r"pastor|imam|rabbi|church|mosque|synagogue|religio(?:n|us)|"
        r"faith\b|denomination)\b",
        re.IGNORECASE)),
    ("race", re.compile(
        r"\b(race\b|ethnic(?:ity)?|tribal\b|yoruba|igbo|hausa|fulani|"
        r"african-american|caucasian|asian\b|white\b|black\b) (?:man|woman|"
        r"person|people|guy|girl)\b|\bmy race is\b",
        re.IGNORECASE)),
    ("sexuality", re.compile(
        r"\b(gay|lesbian|bisexual|transgender|trans\b|queer|homosexual|"
        r"sexual orientation|coming out)\b|"
        r"\b(dating|attracted to|into|prefer dating)\s+"
        r"(men|women|guys|girls)\b",
        re.IGNORECASE)),
)


def is_sensitive_text(text: str) -> str:
    """Return the sensitive category a text touches, or \"\" if none.

    Used as a storage guard: FACT/PREFERENCE/RELATIONSHIP records matching
    these patterns are refused — the owner never gets a shadow profile.
    """
    for category, pattern in SENSITIVE_PATTERNS:
        if pattern.search(text or ""):
            return category
    return ""


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
                    if value and not is_sensitive_text(text):
                        identity[key] = ModelAttribute(
                            name=key, value=value, confidence=0.9,
                            evidence=_evidence_ids(record), kind="fact")
        return identity

    @staticmethod
    def _build_preferences(
            prefs: list[MemoryRecord]) -> list[ModelAttribute]:
        out: list[ModelAttribute] = []
        for record in prefs:
            if is_sensitive_text(record.content):
                continue
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


class PeopleGraph:
    """Graph view over RELATIONSHIP records: who matters to the owner.

    Stores only what the owner stated, verbatim-ish.  No inference of
    sensitive attributes — the storage guard refuses those records.
    """

    def __init__(self) -> None:
        self.people: dict[str, PersonEntry] = {}

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
            if not name or is_sensitive_text(record.content):
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
        return graph

    def lookup(self, name: str) -> PersonEntry | None:
        return self.people.get((name or "").lower())

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

    Hard rules: ≤10 lines, no raw episode text, no private-marked records,
    no sensitive attributes.  Rebuilt when the model changes.
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
        if _is_private(new) or is_sensitive_text(new.content):
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


# ── scheduler jobs ─────────────────────────────────────────────────────────

PERSONA_REBUILD_JOB = "persona rebuild"
PERSONA_CURATE_JOB = "persona curate"


def ensure_persona_jobs(context: Any) -> dict[str, Any]:
    """Register the daily persona rebuild + curation jobs (idempotent)."""
    from ..agents.scheduler import Scheduler

    out: dict[str, Any] = {}
    for name, spec, action in (
            (PERSONA_REBUILD_JOB, "daily 03:30", "rebuild"),
            (PERSONA_CURATE_JOB, "daily 04:00", "curate")):
        try:
            sched = Scheduler(context)
            have = [j for j in sched.list_jobs() if j.get("name") == name]
        except Exception:  # noqa: BLE001 — scheduler table may not exist yet
            have = []
        if have:
            out[name] = {"already_scheduled": True}
            continue
        try:
            job = sched.add(name, spec, "tool",
                            {"tool": "memory",
                             "args": {"action": action}})
            out[name] = {"scheduled": True, "job_id": job.get("id")}
            _log.info("scheduled persona job: %s", name)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
    return out


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
            "export | rebuild | curate | anticipate (query). "
            "Writes are explicit; recall never returns private records."
        ),
        capability=Capability.MEM_WRITE,
        parameters={
            "action": "str — recall|show|list|remember|update|forget|private|"
                      "public|export|rebuild|curate|anticipate",
            "query": "str (optional) — search text",
            "record_id": "str (optional) — target record",
            "kind": "str (optional) — memory kind filter",
            "text": "str (optional) — content for remember/update",
            "limit": "int (optional) — result limit",
        },
    )
    def memory(*, action: str = "recall", query: str = "",
               record_id: str = "", kind: str = "", text: str = "",
               limit: int = 8) -> dict[str, Any]:
        manager = _manager(context)
        if action == "recall":
            result = manager.recall(query, limit=limit, kind=kind or "")
            return {"records": [r.to_dict() for r in result.records]}
        if action == "show":
            model = UserModel.rebuild(manager)
            guide = PersonaGuide.build(model)
            return {"model": model.to_dict(), "guide": guide.to_dict()}
        if action == "list":
            result = manager.recall(query or "", limit=limit,
                                    kind=kind or "")
            return {"records": [
                {"id": r.id, "kind": r.kind,
                 "content": r.content[:200],
                 "private": _is_private(r),
                 "created_at": r.created_at}
                for r in result.records]}
        if action == "remember":
            body = text or query
            if is_sensitive_text(body):
                return {"ok": False, "error": "refused: sensitive attribute"}
            rid = manager.remember(body, kind=kind or MemoryKind.EPISODE,
                                   source="agent")
            note: dict[str, Any] = {}
            if kind in (MemoryKind.FACT, MemoryKind.PREFERENCE) and rid:
                note = MemoryCurator(manager).check_contradiction(rid)
            return {"ok": bool(rid), "id": rid, **note}
        if action == "update":
            if not record_id:
                return {"ok": False, "error": "record_id required"}
            if is_sensitive_text(text):
                return {"ok": False, "error": "refused: sensitive attribute"}
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
