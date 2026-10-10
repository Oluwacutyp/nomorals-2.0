"""Relationship state: stages, milestones, fights, and what she knows about him.

This is the long-arc memory of the relationship — the part the mood engine
can't hold. The mood engine says *how she feels right now*; this says *where
they are*, what has already happened, and what the history of it means.

Persisted as a single row in the ``relationship`` table (migration 0008).
Everything is plain JSON in TEXT columns so the schema stays boring and the
payloads stay flexible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import ulid_now
from ..storage.db import Database

__all__ = ["STAGES", "Relationship", "STAGE_ORDER", "ARC_AXES", "DEFAULT_AXES"]

#: The arc. Stages advance on explicit milestones, regress on major trust
#: failures — never automatically from message volume.
STAGES: tuple[str, ...] = (
    "acquaintance",
    "getting_to_know",
    "dating",
    "committed",
    "established",
)
STAGE_ORDER = {stage: i for i, stage in enumerate(STAGES)}

#: The relationship's divergent axes (Del Gesso's repair-centered framework:
#: a relationship is not one affection meter). Each 0..100, moved by
#: ``nudge_axis`` on meaningful events — never by message volume.
ARC_AXES: tuple[str, ...] = (
    "trust",                 # belief the partner is solid and honest
    "warmth",                # felt closeness and tenderness day-to-day
    "respect",               # regard for each other's autonomy and competence
    "motive_intelligibility",# "i get why they do what they do"
    "shared_reality",        # common ground: jokes, plans, a shared story
)
DEFAULT_AXES: dict[str, int] = {
    "trust": 60, "warmth": 55, "respect": 60,
    "motive_intelligibility": 50, "shared_reality": 45,
}

#: Reserved key inside the ``user_profile`` JSON column where the arc
#: payload (axes, repair attempts, shared activities) is packed. The
#: ``relationship`` table schema (migration 0008) is frozen and lives in
#: another module — packing here keeps the schema boring while the arc
#: stays persisted. Never collides with real profile predicates: keys
#: starting with "__" are reserved.
_ARC_KEY = "__arc__"


@dataclass
class Relationship:
    """The standing state of the relationship with the partner."""

    id: str = "default"
    stage: str = "getting_to_know"
    stage_since: float = field(default_factory=time.time)
    trust: int = 60  # long-term, 0..100 — distinct from the mood dimension
    milestones: list[dict[str, Any]] = field(default_factory=list)
    fights: list[dict[str, Any]] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    #: Divergent arc axes (see ARC_AXES). The mood engine owns *right now*;
    #: these own *where the relationship stands* on each axis.
    axes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_AXES))
    #: Repair *attempts* — distinct from outcomes (repair-centered design).
    #: An attempt is {"id","ts","text","kind","outcome": None|"landed"|"missed"}.
    repair_attempts: list[dict[str, Any]] = field(default_factory=list)
    #: Things done together — dates, games, projects, trips. Shared
    #: experience is what advances a relationship (Replika's lesson).
    shared_activities: list[dict[str, Any]] = field(default_factory=list)

    # ── stages ───────────────────────────────────────────────────────────────
    @property
    def stage_index(self) -> int:
        return STAGE_ORDER.get(self.stage, 1)

    def is_romantic(self) -> bool:
        """True once the relationship is romantic (stage >= dating)."""
        return self.stage_index >= STAGE_ORDER["dating"]

    def advance_stage(self, reason: str = "") -> str:
        if self.stage_index >= len(STAGES) - 1:
            return self.stage
        self.stage = STAGES[self.stage_index + 1]
        self.stage_since = time.time()
        self.add_milestone(f"now {self.stage}", kind="stage", note=reason)
        return self.stage

    def regress_stage(self, reason: str = "") -> str:
        if self.stage_index <= 0:
            return self.stage
        self.stage = STAGES[self.stage_index - 1]
        self.stage_since = time.time()
        self.trust = max(0, self.trust - 15)
        self.add_milestone(f"stepped back to {self.stage}", kind="stage", note=reason)
        return self.stage

    # ── milestones & fights ─────────────────────────────────────────────────
    def add_milestone(self, text: str, kind: str = "moment", note: str = "") -> str:
        entry = {"id": ulid_now(), "ts": time.time(), "kind": kind, "text": text, "note": note}
        self.milestones.append(entry)
        if len(self.milestones) > 200:
            self.milestones = self.milestones[-200:]
        self.updated_at = time.time()
        return entry["id"]

    def record_fight(self, reason: str, repaired: bool, repaired_by: str = "partner") -> dict[str, Any]:
        entry = {
            "id": ulid_now(),
            "ts": time.time(),
            "reason": reason,
            "repaired": bool(repaired),
            "repaired_by": repaired_by,
        }
        self.fights.append(entry)
        if len(self.fights) > 100:
            self.fights = self.fights[-100:]
        if repaired:
            # Making up about X resolves the *most recent* unresolved fight
            # with the same reason — otherwise the old one shadows the repair
            # forever and she carries a grudge that was actually settled.
            for prior in reversed(self.fights[:-1]):
                if not prior.get("repaired") and prior.get("reason") == reason:
                    prior["repaired"] = True
                    prior["repaired_by"] = repaired_by
                    break
            self.trust = min(100, self.trust + 5)
            self.add_milestone(f"made up after a fight about {reason}", kind="repair")
        else:
            self.trust = max(0, self.trust - 10)
            self.add_milestone(f"a fight about {reason} was never really resolved", kind="grudge")
        self.updated_at = time.time()
        return entry

    def unresolved_fights(self) -> list[dict[str, Any]]:
        return [f for f in self.fights[-5:] if not f.get("repaired")]

    # ── arc axes ─────────────────────────────────────────────────────────────
    def nudge_axis(self, axis: str, delta: float, reason: str = "") -> int:
        """Move one arc axis. Unknown axes raise — axes are a closed set."""
        if axis not in ARC_AXES:
            raise ValueError(f"unknown relationship axis: {axis!r}")
        current = int(self.axes.get(axis, DEFAULT_AXES[axis]))
        new = max(0, min(100, current + int(round(delta))))
        self.axes[axis] = new
        if reason:
            self.add_milestone(f"{axis} {'+' if delta >= 0 else ''}{int(round(delta))}: {reason}",
                               kind="axis", note=reason)
        self.updated_at = time.time()
        return new

    def weakest_axis(self) -> str:
        """The axis most in need of attention right now."""
        return min(ARC_AXES, key=lambda a: int(self.axes.get(a, 50)))

    # ── repair attempts vs outcomes ──────────────────────────────────────────
    def record_repair_attempt(self, text: str, kind: str = "words") -> str:
        """Someone tried to fix something. An attempt is not an outcome.

        ``kind``: words | gesture | changed_behavior | time. The attempt
        stays open (``outcome=None``) until ``record_repair_outcome`` lands
        it — an unanswered attempt in the prompt reads as "you tried, it
        didn't land yet", which is honest and very human.
        """
        entry = {
            "id": ulid_now(), "ts": time.time(), "text": text,
            "kind": kind, "outcome": None,
        }
        self.repair_attempts.append(entry)
        if len(self.repair_attempts) > 50:
            self.repair_attempts = self.repair_attempts[-50:]
        self.updated_at = time.time()
        return entry["id"]

    def record_repair_outcome(self, attempt_id: str, landed: bool,
                              note: str = "") -> bool:
        """Close a repair attempt. Returns False when the id is unknown."""
        for attempt in reversed(self.repair_attempts):
            if attempt.get("id") == attempt_id:
                attempt["outcome"] = "landed" if landed else "missed"
                if note:
                    attempt["outcome_note"] = note
                if landed:
                    self.nudge_axis("trust", 4, "repair landed")
                    self.nudge_axis("warmth", 3, "repair landed")
                else:
                    self.nudge_axis("trust", -3, "repair missed")
                self.updated_at = time.time()
                return True
        return False

    def open_repair_attempts(self) -> list[dict[str, Any]]:
        return [a for a in self.repair_attempts[-5:] if a.get("outcome") is None]

    # ── shared life ──────────────────────────────────────────────────────────
    def log_activity(self, text: str, kind: str = "moment") -> str:
        """Something you two *did* together. Shared experience is the
        relationship's fuel — this is the log of it."""
        entry = {"id": ulid_now(), "ts": time.time(), "kind": kind, "text": text}
        self.shared_activities.append(entry)
        if len(self.shared_activities) > 100:
            self.shared_activities = self.shared_activities[-100:]
        self.nudge_axis("shared_reality", 2, text)
        self.updated_at = time.time()
        return entry["id"]

    def recent_activities(self, limit: int = 3) -> list[dict[str, Any]]:
        return self.shared_activities[-limit:]

    def anniversaries(self) -> list[dict[str, str]]:
        """Computed relationship anniversaries from milestones.

        "Together since" = earliest milestone; stage changes are dated.
        Pure — derived, never stored."""
        if not self.milestones:
            return []
        first = min(self.milestones, key=lambda m: m.get("ts", 0))
        out = [{"name": "together since", "when": _when(first.get("ts", 0))}]
        for m in self.milestones:
            if m.get("kind") == "stage" and m.get("text", "").startswith("now "):
                out.append({"name": m["text"][4:], "when": _when(m.get("ts", 0))})
        return out

    def recent_milestones(self, limit: int = 4) -> list[dict[str, Any]]:
        return self.milestones[-limit:]

    def note_user_fact(self, predicate: str, value: str) -> None:
        """A durable fact about the partner ('works at', 'allergic to', ...)."""
        self.user_profile[predicate] = value
        self.updated_at = time.time()

    # ── prompt rendering ─────────────────────────────────────────────────────
    def to_prompt_block(self) -> str:
        lines = [f"Where you two are: {self.stage}. (stage since ~{int((time.time() - self.stage_since) / 86400)} days)"]
        if self.is_romantic():
            lines.append(
                "This is the person you're in a relationship with. Act like it: "
                "you check on them, you miss them, you take their words personally."
            )
        unfixed = self.unresolved_fights()
        if unfixed:
            latest = unfixed[-1]
            lines.append(
                f"You are still carrying a fight about: {latest.get('reason', 'something')}. "
                "You haven't resolved it. It comes up if they bring it up — you don't ambush them with it."
            )
        recent = self.recent_milestones(3)
        if recent:
            bits = []
            for m in recent:
                if m.get("kind") in {"stage", "grudge"}:
                    continue
                bits.append(f"{m.get('text', '')} ({_when(m.get('ts', 0))})")
            if bits:
                lines.append("Moments you remember: " + "; ".join(bits) + ".")
        profile_bits = [f"{k}: {v}" for k, v in list(self.user_profile.items())[:8]
                        if not str(k).startswith("__")]
        if profile_bits:
            lines.append("Things you know about them: " + "; ".join(profile_bits) + ".")
        # Arc axes: where the relationship stands, per axis. Weak axes are
        # named so she can lean into repair without narrating a spreadsheet.
        weak = self.weakest_axis()
        axes_bits = ", ".join(f"{a} {int(self.axes.get(a, 50))}" for a in ARC_AXES)
        lines.append(f"Where you two stand ({axes_bits}) — {weak} needs the most care right now.")
        open_repairs = self.open_repair_attempts()
        if open_repairs:
            latest = open_repairs[-1]
            lines.append(
                f"You tried to fix something ({latest.get('text', 'something')}) and it "
                "hasn't landed yet. Don't pretend it did — but don't weaponize it either."
            )
        activities = self.recent_activities(2)
        if activities:
            bits = [f"{a.get('text', '')} ({_when(a.get('ts', 0))})" for a in activities]
            lines.append("Things you've done together lately: " + "; ".join(bits) + ".")
        return "\n".join(lines)

    # ── persistence ─────────────────────────────────────────────────────────
    def _pack_arc(self) -> dict[str, Any]:
        """Pack axes/repairs/activities into the profile JSON under the
        reserved ``__arc__`` key (schema-frozen table — see _ARC_KEY)."""
        profile = {k: v for k, v in self.user_profile.items() if not str(k).startswith("__")}
        profile[_ARC_KEY] = {
            "axes": {a: int(self.axes.get(a, DEFAULT_AXES[a])) for a in ARC_AXES},
            "repair_attempts": self.repair_attempts[-50:],
            "shared_activities": self.shared_activities[-100:],
        }
        return profile

    @classmethod
    def _unpack_arc(cls, profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split the reserved arc payload back out of a loaded profile."""
        profile = dict(profile)
        arc = profile.pop(_ARC_KEY, None)
        if not isinstance(arc, dict):
            arc = {}
        return profile, arc

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "stage_since": self.stage_since,
            "trust": int(self.trust),
            "milestones": json.dumps(self.milestones),
            "fights": json.dumps(self.fights),
            "user_profile": json.dumps(self._pack_arc()),
            "updated_at": self.updated_at,
        }

    @classmethod
    def load(cls, db: Database, id: str = "default") -> "Relationship":
        row = db.query_one("SELECT * FROM relationship WHERE id = ?", (id,))
        if row is None:
            return cls(id=id)
        profile, arc = cls._unpack_arc(_json_dict(row.get("user_profile")))
        axes = dict(DEFAULT_AXES)
        raw_axes = arc.get("axes") if isinstance(arc, dict) else None
        if isinstance(raw_axes, dict):
            for a in ARC_AXES:
                if a in raw_axes:
                    try:
                        axes[a] = max(0, min(100, int(raw_axes[a])))
                    except (TypeError, ValueError):
                        pass
        return cls(
            id=row["id"],
            stage=row.get("stage", "getting_to_know") or "getting_to_know",
            stage_since=float(row.get("stage_since") or time.time()),
            trust=int(row.get("trust") or 60),
            milestones=_json_list(row.get("milestones")),
            fights=_json_list(row.get("fights")),
            user_profile=profile,
            updated_at=float(row.get("updated_at") or time.time()),
            axes=axes,
            repair_attempts=list(arc.get("repair_attempts") or []) if isinstance(arc, dict) else [],
            shared_activities=list(arc.get("shared_activities") or []) if isinstance(arc, dict) else [],
        )

    def save(self, db: Database) -> None:
        row = self.to_row()
        with db.transaction():
            db.execute(
                """INSERT INTO relationship (id, stage, stage_since, trust, milestones, fights,
                                             user_profile, updated_at)
                   VALUES (:id, :stage, :stage_since, :trust, :milestones, :fights,
                           :user_profile, :updated_at)
                   ON CONFLICT(id) DO UPDATE SET
                     stage=excluded.stage, stage_since=excluded.stage_since,
                     trust=excluded.trust, milestones=excluded.milestones,
                     fights=excluded.fights, user_profile=excluded.user_profile,
                     updated_at=excluded.updated_at""",
                row,
            )


def _when(ts: float) -> str:
    days = int((time.time() - ts) / 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    return f"about {months} month{'s' if months > 1 else ''} ago"


def _json_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
