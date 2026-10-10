"""Relationship-aware messaging: she knows who matters and how.

The problem: the system tracked ONE relationship — the owner's romantic
one. Everyone else was a stranger every time. A God-tier social OS knows
its people: who's close, who's new, who you haven't talked to in months,
who annoys you, who you light up for.

The fix: per-person relationship records, mined from actual interaction
— never hardcoded, never guessed:

* **Closeness** — derived from message frequency, recency, warmth signals,
  and the owner's own behavior (who THEY message first, who they reply
  fastest to). The owner's actions are the ground truth, not our labels.
* **Cadence** — how often you talk: daily, weekly, sporadic, dormant.
  A dormant-close friend coming back gets warmth, not "who is this".
* **Last interaction** — what you last talked about (from chat profiles),
  so "how did the interview go?" lands instead of "hey".
* **Notes** — things she learns: birthdays, kids' names, ongoing
  situations. Surfaced to the brain as context, never volunteered to
  outsiders.

Design rules:
- Scores are derived, not assigned. No manual "closeness = 8".
- The owner is never scored — they're the center, not a contact.
- Everything degrades gracefully: unknown person = polite stranger.
- Private to the owner: relationship data never leaves the system.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass
class ContactRelationship:
    """One person's relationship record."""

    person_id: str
    display_name: str = ""
    # Derived scores (0..1). Recomputed from interaction history.
    closeness: float = 0.0
    warmth: float = 0.5  # tone of recent exchanges
    # Counters.
    inbound_count: int = 0
    outbound_count: int = 0
    owner_initiated_count: int = 0  # owner messaged them first
    # Timestamps.
    first_seen: float = field(default_factory=time.time)
    last_inbound: float = 0.0
    last_outbound: float = 0.0
    # Mined facts.
    notes: list[str] = field(default_factory=list)
    last_topic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "closeness": round(self.closeness, 3),
            "warmth": round(self.warmth, 3),
            "inbound_count": self.inbound_count,
            "outbound_count": self.outbound_count,
            "owner_initiated_count": self.owner_initiated_count,
            "first_seen": self.first_seen,
            "last_inbound": self.last_inbound,
            "last_outbound": self.last_outbound,
            "notes": self.notes[-20:],
            "last_topic": self.last_topic,
        }

    def cadence(self) -> str:
        """How often you talk, in plain words."""
        total = self.inbound_count + self.outbound_count
        if total < 3:
            return "new"
        days = max(1.0, (time.time() - self.first_seen) / 86400)
        per_day = total / days
        if per_day >= 1.0:
            return "daily"
        if per_day >= 0.25:
            return "weekly"
        if time.time() - max(self.last_inbound, self.last_outbound) > 90 * 86400:
            return "dormant"
        return "sporadic"

    def prompt_lines(self) -> list[str]:
        """Context lines for the brain. Plain words, no furniture."""
        lines = [
            f"You know {self.display_name}: {self.cadence()} contact, "
            f"closeness {self.closeness:.1f}/1."
        ]
        if self.last_topic:
            lines.append(f"Last you talked about: {self.last_topic}.")
        if self.cadence() == "dormant" and self.closeness > 0.4:
            lines.append(
                "You haven't talked in a while — warmth, not awkwardness."
            )
        for note in self.notes[-3:]:
            lines.append(f"Remember: {note}")
        return lines


class RelationshipTracker:
    """Per-contact relationships, mined from interaction. JSON-backed."""

    def __init__(self, path: str = "data/social_relationships.json") -> None:
        self.path = path
        self._rels: dict[str, ContactRelationship] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for pid, prow in (raw.get("relationships") or {}).items():
            try:
                rel = ContactRelationship(person_id=pid)
                for key in (
                    "display_name", "closeness", "warmth", "inbound_count",
                    "outbound_count", "owner_initiated_count", "first_seen",
                    "last_inbound", "last_outbound", "last_topic",
                ):
                    if key in prow:
                        setattr(rel, key, prow[key])
                rel.notes = list(prow.get("notes") or [])
                self._rels[pid] = rel
            except Exception:  # noqa: BLE001
                _log.warning("relationships: skipping bad row %s", pid)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"relationships": {p: r.to_dict() for p, r in self._rels.items()}},
                    fh,
                    ensure_ascii=False,
                )
            os.replace(tmp, self.path)
        except OSError as exc:
            _log.warning("relationships: save failed: %s", exc)

    def get(self, person_id: str) -> ContactRelationship | None:
        return self._rels.get(person_id)

    def ensure(self, person_id: str, display_name: str = "") -> ContactRelationship:
        rel = self._rels.get(person_id)
        if rel is None:
            rel = ContactRelationship(person_id=person_id, display_name=display_name)
            self._rels[person_id] = rel
        elif display_name and not rel.display_name:
            rel.display_name = display_name
        return rel

    def note_inbound(
        self,
        person_id: str,
        *,
        display_name: str = "",
        warm: bool | None = None,
        topic: str = "",
    ) -> None:
        """They messaged. Bump closeness by recency-weighted frequency."""
        rel = self.ensure(person_id, display_name)
        rel.inbound_count += 1
        rel.last_inbound = time.time()
        if topic:
            rel.last_topic = topic
        if warm is not None:
            # Exponential moving average on tone.
            rel.warmth = 0.8 * rel.warmth + 0.2 * (1.0 if warm else 0.0)
        self._recompute_closeness(rel)
        self._save()

    def note_outbound(
        self, person_id: str, *, owner_initiated: bool = False, display_name: str = ""
    ) -> None:
        """We messaged them. Owner-initiated weighs heaviest — the owner's
        own behavior is the ground truth of who matters."""
        rel = self.ensure(person_id, display_name)
        rel.outbound_count += 1
        rel.last_outbound = time.time()
        if owner_initiated:
            rel.owner_initiated_count += 1
        self._recompute_closeness(rel)
        self._save()

    def add_note(self, person_id: str, note: str) -> None:
        rel = self.ensure(person_id)
        note = (note or "").strip()
        if note and note not in rel.notes:
            rel.notes.append(note)
            rel.notes = rel.notes[-20:]
            self._save()

    def _recompute_closeness(self, rel: ContactRelationship) -> None:
        """Derived, not assigned. Frequency + recency + owner initiation."""
        total = rel.inbound_count + rel.outbound_count
        # Frequency component: log-scaled, caps at ~100 messages.
        freq = min(1.0, math.log10(1 + total) / 2.0)
        # Recency component: decays over 90 days of silence.
        last = max(rel.last_inbound, rel.last_outbound, rel.first_seen)
        days_quiet = (time.time() - last) / 86400
        recency = max(0.0, 1.0 - days_quiet / 90.0)
        # Owner initiation: the owner choosing to message first is the
        # strongest signal of who matters — it dominates the score.
        initiated = min(1.0, rel.owner_initiated_count / 5.0)
        rel.closeness = round(
            min(1.0, 0.30 * freq + 0.20 * recency + 0.50 * initiated), 3
        )

    def top_contacts(self, limit: int = 10) -> list[ContactRelationship]:
        return sorted(
            self._rels.values(), key=lambda r: r.closeness, reverse=True
        )[:limit]

    def target_cadence_days(self, rel: ContactRelationship) -> int:
        """How often this contact deserves a touch, from closeness.

        Personal-CRM practice: close contacts ~monthly, mentors/strong ties
        every couple of weeks, weak ties quarterly. The score decides, not
        a manual label.
        """
        if rel.closeness >= 0.7:
            return 14
        if rel.closeness >= 0.4:
            return 30
        if rel.closeness >= 0.15:
            return 90
        return 180

    def needs_reconnect(self, days: int = 30) -> list[ContactRelationship]:
        """Close contacts gone quiet — candidates for proactive outreach."""
        cutoff = time.time() - days * 86400
        return [
            r
            for r in self._rels.values()
            if r.closeness >= 0.4
            and max(r.last_inbound, r.last_outbound) < cutoff
        ]

    def due_for_reconnect(self) -> list[tuple[ContactRelationship, int]]:
        """Contacts past their OWN target cadence: (relationship, days_quiet).

        Stronger than the flat ``needs_reconnect``: a close friend is due
        after 14 quiet days, a weak tie after 180. Sorted most-overdue
        first — this is the morning "who should I message" list.
        """
        now = time.time()
        due: list[tuple[ContactRelationship, int, float]] = []
        for rel in self._rels.values():
            last = max(rel.last_inbound, rel.last_outbound, rel.first_seen)
            days_quiet = (now - last) / 86400
            target = self.target_cadence_days(rel)
            if days_quiet >= target and rel.closeness >= 0.15:
                overdue_ratio = days_quiet / target
                due.append((rel, int(days_quiet), overdue_ratio))
        due.sort(key=lambda t: t[2], reverse=True)
        return [(rel, days) for rel, days, _ in due]

    def reconnect_prompt(self, rel: ContactRelationship, days_quiet: int) -> str:
        """A brain-ready outreach line: who, how long, last topic, warmth.

        Never a template message TO the person — context FOR the brain so
        "how did the interview go?" lands instead of "hey stranger".
        """
        name = rel.display_name or "them"
        bits = [f"{name} — {days_quiet}d quiet (cadence: {rel.cadence()})"]
        if rel.last_topic:
            bits.append(f"last talked about: {rel.last_topic}")
        if rel.warmth >= 0.7:
            bits.append("warm history — lead with affection")
        elif rel.warmth <= 0.3:
            bits.append("cooler history — keep it light, no pressure")
        for note in rel.notes[-2:]:
            bits.append(f"remember: {note}")
        if days_quiet > 120 and rel.closeness > 0.4:
            bits.append("dormant-close: warmth, not awkwardness")
        return " · ".join(bits)
