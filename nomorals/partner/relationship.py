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

__all__ = ["STAGES", "Relationship", "STAGE_ORDER"]

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
        profile_bits = [f"{k}: {v}" for k, v in list(self.user_profile.items())[:8]]
        if profile_bits:
            lines.append("Things you know about them: " + "; ".join(profile_bits) + ".")
        return "\n".join(lines)

    # ── persistence ─────────────────────────────────────────────────────────
    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "stage_since": self.stage_since,
            "trust": int(self.trust),
            "milestones": json.dumps(self.milestones),
            "fights": json.dumps(self.fights),
            "user_profile": json.dumps(self.user_profile),
            "updated_at": self.updated_at,
        }

    @classmethod
    def load(cls, db: Database, id: str = "default") -> "Relationship":
        row = db.query_one("SELECT * FROM relationship WHERE id = ?", (id,))
        if row is None:
            return cls(id=id)
        return cls(
            id=row["id"],
            stage=row.get("stage", "getting_to_know") or "getting_to_know",
            stage_since=float(row.get("stage_since") or time.time()),
            trust=int(row.get("trust") or 60),
            milestones=_json_list(row.get("milestones")),
            fights=_json_list(row.get("fights")),
            user_profile=_json_dict(row.get("user_profile")),
            updated_at=float(row.get("updated_at") or time.time()),
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
