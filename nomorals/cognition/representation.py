"""Representation-quality ledger — "How well did I represent you?"

When Devon acts on the owner's behalf (negotiates, buys, applies for
gigs, posts, sends messages), each autonomous action is logged here
with its outcome and the owner's feedback.  The ledger answers:

- :meth:`RepresentationLedger.quality_score` — recency-weighted quality
  per action type (0..1).  Owner disapproval (-1) hits harder than
  approval (+1) helps; unknown outcomes don't inflate the score.
- :meth:`RepresentationLedger.low_quality_patterns` — action types
  scoring below threshold, with example records and counterfactual
  notes — the feed for skill distillation (#2).
- :meth:`RepresentationLedger.summary` — the "/how did I do" review.

Storage follows the :class:`TrajectoryStore` convention
(``Database | str | Path | None``); scoring uses the same exponential
recency decay.  Logging never raises and never breaks the action it
observes.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..storage.db import Database

_log = logging.getLogger(__name__)

__all__ = [
    "RepresentationLedger",
    "ActionRecord",
    "ACTION_TYPES",
    "OUTCOMES",
]

#: Autonomous action types Devon can take on the owner's behalf.
ACTION_TYPES = ("negotiate", "purchase", "apply", "message", "post")
#: Outcome vocabulary.  "unknown" is honest — it never inflates a score.
OUTCOMES = ("success", "partial", "failed", "unknown")

_HALF_LIFE_DAYS = 14.0
_SECONDS_PER_DAY = 86_400.0
_EMPTY_PRIOR = 0.5
#: Owner approval weighs a bit more than a plain success.
_FEEDBACK_UP_WEIGHT = 1.5
#: Owner disapproval drags a score toward 0 by this flat amount — a
#: "success" the owner hated can never score above 0.5.
_FEEDBACK_DOWN_PENALTY = 0.5

_REPRESENTATION_DDL = """
CREATE TABLE IF NOT EXISTS cog_representation (
    id              TEXT PRIMARY KEY,
    action_type     TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    outcome         TEXT NOT NULL DEFAULT 'unknown',
    owner_feedback  INTEGER,                -- NULL none, +1 approve, -1 disapprove
    feedback_text   TEXT NOT NULL DEFAULT '',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cog_rep_type_time
    ON cog_representation(action_type, created_at);
CREATE TABLE IF NOT EXISTS cog_representation_counterfactuals (
    id          TEXT PRIMARY KEY,
    action_id   TEXT NOT NULL,
    better_action TEXT NOT NULL DEFAULT '',
    why         TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    FOREIGN KEY (action_id) REFERENCES cog_representation(id)
);
CREATE INDEX IF NOT EXISTS idx_cog_rep_cf_action
    ON cog_representation_counterfactuals(action_id);
"""


def _decay_weight(age_s: float, half_life_s: float) -> float:
    if age_s < 0:
        age_s = 0.0
    return 0.5 ** (age_s / half_life_s) if half_life_s > 0 else 1.0


@dataclass
class ActionRecord:
    """One autonomous action Devon took on the owner's behalf."""

    id: str
    action_type: str
    description: str
    outcome: str = "unknown"
    owner_feedback: int | None = None  # +1 approve / -1 disapprove / None
    feedback_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)


class RepresentationLedger:
    """Log and score how well Devon represented the owner."""

    def __init__(self, db: Database | str | Path | None = None) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_REPRESENTATION_DDL)
        self._lock = threading.RLock()

    # ── recording ────────────────────────────────────────────────────
    def log_action(
        self,
        action_type: str,
        description: str,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> ActionRecord:
        """Log an autonomous action.  Never raises — a logging failure is
        logged, never propagated, so the observed action is never broken."""
        try:
            return self._log_action(
                action_type, description, outcome=outcome,
                metadata=metadata, created_at=created_at,
            )
        except Exception as exc:  # noqa: BLE001 — logging must not break actions
            _log.warning("representation log failed: %s", exc)
            return ActionRecord(
                id="", action_type=action_type or "",
                description=description or "",
            )

    def _log_action(
        self,
        action_type: str,
        description: str,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> ActionRecord:
        if action_type not in ACTION_TYPES:
            raise ValueError(
                f"unknown action_type {action_type!r}; use {ACTION_TYPES}")
        outcome = outcome or "unknown"
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; use {OUTCOMES}")
        now = time.time()
        rec = ActionRecord(
            id=new_short_id("rep_"),
            action_type=action_type,
            description=description or "",
            outcome=outcome,
            metadata=dict(metadata or {}),
            created_at=float(created_at if created_at is not None else now),
            updated_at=now,
        )
        with self._lock:
            self.db.execute(
                "INSERT INTO cog_representation (id, action_type, description,"
                " outcome, owner_feedback, feedback_text, metadata_json,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, NULL, '', ?, ?, ?)",
                (rec.id, rec.action_type, rec.description, rec.outcome,
                 json.dumps(rec.metadata), rec.created_at, rec.updated_at),
            )
        return rec

    def record_outcome(
        self,
        action_id: str,
        outcome: str,
        *,
        owner_feedback: int | str | None = None,
        feedback_text: str = "",
    ) -> bool:
        """Set the outcome (+ optional owner feedback) for an action.

        ``owner_feedback``: +1 / -1 / "up" / "down" / None.  Returns False
        when the action id is unknown.  Never raises.
        """
        try:
            if outcome not in OUTCOMES:
                raise ValueError(f"unknown outcome {outcome!r}")
            fb: int | None = None
            if owner_feedback in ("up", "+1", 1, True):
                fb = 1
            elif owner_feedback in ("down", "-1", -1, False):
                fb = -1
            elif owner_feedback is not None:
                raise ValueError(f"bad owner_feedback {owner_feedback!r}")
            with self._lock:
                cur = self.db.execute(
                    "UPDATE cog_representation SET outcome = ?,"
                    " owner_feedback = ?, feedback_text = ?,"
                    " updated_at = ? WHERE id = ?",
                    (outcome, fb, feedback_text or "",
                     time.time(), action_id),
                )
                return (cur.rowcount or 0) > 0
        except Exception as exc:  # noqa: BLE001
            _log.warning("representation record_outcome failed: %s", exc)
            return False

    def record_counterfactual(
        self,
        action_id: str,
        better_action: str,
        why: str,
    ) -> str:
        """"Could a better action have gotten a better result?"  Stored for
        learning / skill distillation.  Returns the note id.  Never raises."""
        note_id = ""
        try:
            if not action_id or not better_action:
                raise ValueError("action_id and better_action are required")
            note_id = new_short_id("cf_")
            with self._lock:
                self.db.execute(
                    "INSERT INTO cog_representation_counterfactuals"
                    " (id, action_id, better_action, why, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (note_id, action_id, better_action, why or "",
                     time.time()),
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("representation counterfactual failed: %s", exc)
        return note_id

    # ── reading ──────────────────────────────────────────────────────
    def _row_to_record(self, row: dict[str, Any]) -> ActionRecord:
        try:
            meta = json.loads(row.get("metadata_json") or "{}")
        except (ValueError, TypeError):
            meta = {}
        return ActionRecord(
            id=row["id"],
            action_type=row.get("action_type", ""),
            description=row.get("description", ""),
            outcome=row.get("outcome", "unknown"),
            owner_feedback=row.get("owner_feedback"),
            feedback_text=row.get("feedback_text", ""),
            metadata=meta if isinstance(meta, dict) else {},
            created_at=float(row.get("created_at", 0.0)),
            updated_at=float(row.get("updated_at", 0.0)),
        )

    def get(self, action_id: str) -> ActionRecord | None:
        rows = self.db.query(
            "SELECT * FROM cog_representation WHERE id = ?", (action_id,))
        if not rows:
            return None
        rec = self._row_to_record(rows[0])
        rec.counterfactuals = self._counterfactuals_for(action_id)
        return rec

    def _counterfactuals_for(self, action_id: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT id, better_action, why, created_at"
            " FROM cog_representation_counterfactuals"
            " WHERE action_id = ? ORDER BY created_at DESC",
            (action_id,),
        )

    def recent(
        self,
        *,
        action_type: str | None = None,
        limit: int = 20,
        since_days: float | None = None,
    ) -> list[ActionRecord]:
        sql = "SELECT * FROM cog_representation"
        params: list[Any] = []
        clauses: list[str] = []
        if action_type:
            clauses.append("action_type = ?")
            params.append(action_type)
        if since_days is not None:
            clauses.append("created_at >= ?")
            params.append(time.time() - since_days * _SECONDS_PER_DAY)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [self._row_to_record(r) for r in self.db.query(sql, params)]

    # ── scoring ──────────────────────────────────────────────────────
    @staticmethod
    def _score_value(outcome: str, owner_feedback: int | None) -> float:
        """Map an outcome to a 0..1 quality value.

        Unknown outcomes contribute 0 weight (see quality_score) — they
        never inflate.  Owner approval lifts a bit; owner disapproval
        drags toward 0 by a flat 0.5 — harder than approval helps, so a
        "successful" action the owner hated can never score above 0.5.
        """
        base = {"success": 1.0, "partial": 0.5,
                "failed": 0.0, "unknown": 0.5}[outcome]
        if owner_feedback == 1:
            return min(1.0, base * _FEEDBACK_UP_WEIGHT)
        if owner_feedback == -1:
            return max(0.0, base - _FEEDBACK_DOWN_PENALTY)
        return base

    def quality_score(
        self,
        action_type: str | None = None,
        *,
        since_days: float = 30.0,
        half_life_days: float = _HALF_LIFE_DAYS,
    ) -> float:
        """Recency-weighted quality score 0..1.

        Exponential decay (14-day half-life): yesterday counts far more
        than last month.  Rows with outcome "unknown" carry zero weight —
        honesty over inflation.  Returns the 0.5 prior with no data.
        """
        now = time.time()
        cutoff = now - since_days * _SECONDS_PER_DAY
        sql = ("SELECT outcome, owner_feedback, created_at"
               " FROM cog_representation WHERE created_at >= ?")
        params: list[Any] = [cutoff]
        if action_type:
            sql += " AND action_type = ?"
            params.append(action_type)
        half_life_s = half_life_days * _SECONDS_PER_DAY
        num = 0.0
        den = 0.0
        for row in self.db.query(sql, params):
            if row.get("outcome") == "unknown" \
                    and row.get("owner_feedback") is None:
                continue  # honest: unknown never inflates
            w = _decay_weight(now - float(row["created_at"]), half_life_s)
            if w <= 0.0:
                continue
            num += w * self._score_value(row.get("outcome", "unknown"),
                                         row.get("owner_feedback"))
            den += w
        return (num / den) if den > 0.0 else _EMPTY_PRIOR

    def scores_by_type(
        self,
        *,
        since_days: float = 30.0,
    ) -> dict[str, float]:
        """Quality score per action type (only types with data)."""
        types = self.db.query(
            "SELECT DISTINCT action_type FROM cog_representation")
        out: dict[str, float] = {}
        for row in types:
            t = row["action_type"]
            if t:
                out[t] = self.quality_score(t, since_days=since_days)
        return out

    def low_quality_patterns(
        self,
        threshold: float = 0.5,
        *,
        since_days: float = 30.0,
        examples: int = 3,
    ) -> list[dict[str, Any]]:
        """Action types scoring below ``threshold`` — the skill-distillation
        feed.  Each entry carries the score, example records, and any
        counterfactual notes attached to those records."""
        patterns: list[dict[str, Any]] = []
        for action_type, score in self.scores_by_type(
                since_days=since_days).items():
            if score >= threshold:
                continue
            recs = self.recent(action_type=action_type, limit=examples,
                               since_days=since_days)
            cfs: list[dict[str, Any]] = []
            for rec in recs:
                for cf in self._counterfactuals_for(rec.id):
                    cfs.append({**cf, "action_id": rec.id})
            patterns.append({
                "action_type": action_type,
                "score": round(score, 3),
                "examples": [
                    {"id": r.id, "description": r.description,
                     "outcome": r.outcome,
                     "owner_feedback": r.owner_feedback}
                    for r in recs
                ],
                "counterfactuals": cfs,
            })
        patterns.sort(key=lambda p: p["score"])
        return patterns

    # ── learning out (Reflexion verbal-RL / Voyager skill library) ─────
    def distill(
        self,
        action_type: str | None = None,
        *,
        threshold: float = 0.7,
        since_days: float = 30.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Turn counterfactuals + owner feedback into lesson cards.

        This is the skill-distillation feed the module docstring
        promised: per action type, Reflexion-style verbal lessons —
        "when you do X, do Y instead, because Z" — assembled from the
        weak patterns, the owner's own words, and recorded
        counterfactuals.  Cards are prompt-ready; the brain injects
        ``lesson_text`` into context before the next action of that
        type (Voyager stores verified programs; we store verified
        guidance).  Types scoring at or above ``threshold`` are
        skipped unless ``action_type`` names them explicitly.
        """
        cards: list[dict[str, Any]] = []
        scores = self.scores_by_type(since_days=since_days)
        types = [action_type] if action_type else sorted(scores)
        for atype in types:
            score = scores.get(atype, _EMPTY_PRIOR)
            if action_type is None and score >= threshold:
                continue
            recs = self.recent(action_type=atype, limit=50,
                               since_days=since_days)
            if not recs:
                continue
            failed = [r for r in recs if r.outcome in ("failed", "partial")]
            pushback = [(r.feedback_text or "").strip()
                        for r in recs
                        if r.owner_feedback == -1 and r.feedback_text]
            cfs: list[str] = []
            for rec in recs:
                for cf in self._counterfactuals_for(rec.id):
                    why = f" — {cf['why']}" if cf.get("why") else ""
                    cfs.append(f"{cf['better_action']}{why}")
            # Dedupe while preserving order.
            cfs = list(dict.fromkeys(cfs))
            pushback = list(dict.fromkeys(pushback))
            lines = [
                f"LESSON — {atype} (representation quality"
                f" {score:.0%} over {len(recs)} actions)",
            ]
            if failed:
                weak = failed[0]
                desc = weak.description[:90]
                lines.append(f"Weak pattern: {len(failed)} weak of"
                             f" {len(recs)} — e.g. {desc!r} ({weak.outcome})")
            if pushback:
                lines.append("Owner pushed back: "
                             + "; ".join(f"{p!r}" for p in pushback[:3]))
            if cfs:
                lines.append("Do instead:")
                lines.extend(f"  • {c}" for c in cfs[:4])
            cards.append({
                "action_type": atype,
                "score": round(score, 3),
                "weak_count": len(failed),
                "total": len(recs),
                "counterfactuals": cfs[:4],
                "owner_pushback": pushback[:3],
                "lesson_text": "\n".join(lines),
            })
            if len(cards) >= max(0, int(limit)):
                break
        cards.sort(key=lambda c: c["score"])
        return cards

    def trend(
        self,
        action_type: str | None = None,
        *,
        window_days: float = 14.0,
    ) -> dict[str, Any]:
        """Improvement trajectory: recent half of ``window_days`` vs the
        prior half.  ``direction`` is ``"improving"`` / ``"declining"`` /
        ``"stable"`` (±0.05 hysteresis); ``delta`` is recent − prior."""
        half = float(window_days) / 2.0
        recent = self.quality_score(action_type, since_days=half)
        prior_all = self.quality_score(action_type, since_days=window_days)
        # prior half ≈ blend of the two windows, solved for the half
        # before `recent`: prior = 2*all - recent (equal weighting).
        prior = min(1.0, max(0.0, 2.0 * prior_all - recent))
        delta = recent - prior
        direction = ("improving" if delta > 0.05
                     else "declining" if delta < -0.05 else "stable")
        return {
            "direction": direction,
            "delta": round(delta, 3),
            "recent_score": round(recent, 3),
            "prior_score": round(prior, 3),
        }

    def feedback_rate(
        self,
        action_type: str | None = None,
        *,
        since_days: float = 30.0,
    ) -> float:
        """Fraction of actions in the window that got owner feedback
        (+1 or -1).  Low rate + low score = flying blind."""
        sql = ("SELECT COUNT(*) AS n, SUM(CASE WHEN owner_feedback IS NOT"
               " NULL THEN 1 ELSE 0 END) AS f FROM cog_representation"
               " WHERE created_at >= ?")
        params: list[Any] = [time.time() - since_days * _SECONDS_PER_DAY]
        if action_type:
            sql += " AND action_type = ?"
            params.append(action_type)
        row = self.db.query(sql, params)[0]
        n = row["n"] or 0
        return float(row["f"] or 0) / n if n else 0.0

    def inconsistencies(
        self,
        *,
        since_days: float = 30.0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Evaluator mismatches (Reflexion's evaluator axis): outcomes
        the owner contradicted — "success" with a 👎, "failed" with a
        👍.  These are the rows most worth re-examining: either the
        outcome detector or the owner's expectation is off."""
        rows = self.db.query(
            "SELECT id, action_type, description, outcome, owner_feedback,"
            " feedback_text, created_at FROM cog_representation"
            " WHERE created_at >= ?"
            " AND ((outcome = 'success' AND owner_feedback = -1)"
            " OR (outcome = 'failed' AND owner_feedback = 1))"
            " ORDER BY created_at DESC LIMIT ?",
            (time.time() - since_days * _SECONDS_PER_DAY,
             max(0, int(limit))),
        )
        return [
            {
                "id": row["id"],
                "action_type": row["action_type"],
                "description": row["description"],
                "outcome": row["outcome"],
                "owner_feedback": row["owner_feedback"],
                "feedback_text": row.get("feedback_text") or "",
                "note": ("owner disapproved a recorded success"
                         if row["owner_feedback"] == -1
                         else "owner approved a recorded failure"),
            }
            for row in rows
        ]

    # ── portable artifact (Letta-style export/import) ──────────────────
    def export_json(self, path: str | Path) -> dict:
        """Export actions + counterfactuals to JSON.  Returns counts."""
        actions = self.db.query("SELECT * FROM cog_representation")
        cfs = self.db.query(
            "SELECT * FROM cog_representation_counterfactuals")
        payload = {
            "kind": "cognition-representation",
            "version": 1,
            "exported_at": time.time(),
            "actions": actions,
            "counterfactuals": cfs,
        }
        out = Path(path)
        out.write_text(json.dumps(payload), encoding="utf-8")
        return {"actions": len(actions), "counterfactuals": len(cfs),
                "path": str(out)}

    def import_json(self, path: str | Path) -> dict:
        """Import a payload written by :meth:`export_json`.  Upserts by
        id.  Returns ``{"actions": n, "counterfactuals": m}``."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "cognition-representation":
            raise ValueError("not a cognition-representation export")
        n_a = n_c = 0
        with self._lock:
            for a in payload.get("actions", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO cog_representation (id,"
                    " action_type, description, outcome, owner_feedback,"
                    " feedback_text, metadata_json, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (a.get("id") or new_short_id("rep_"),
                     a.get("action_type") or "",
                     a.get("description") or "",
                     a.get("outcome") or "unknown",
                     a.get("owner_feedback"),
                     a.get("feedback_text") or "",
                     a.get("metadata_json") or "{}",
                     float(a.get("created_at", time.time())),
                     float(a.get("updated_at", time.time()))),
                )
                n_a += 1
            for c in payload.get("counterfactuals", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO"
                    " cog_representation_counterfactuals (id, action_id,"
                    " better_action, why, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (c.get("id") or new_short_id("cf_"),
                     c.get("action_id") or "",
                     c.get("better_action") or "",
                     c.get("why") or "",
                     float(c.get("created_at", time.time()))),
                )
                n_c += 1
        return {"actions": n_a, "counterfactuals": n_c}

    # ── review surface ───────────────────────────────────────────────
    def summary(self, *, limit: int = 10,
                since_days: float = 30.0) -> str:
        """The "/how did I do" review text."""
        recs = self.recent(limit=limit, since_days=since_days)
        lines = ["📊 How I represented you"
                 + (f" (last {int(since_days)}d)" if since_days else "")]
        if not recs:
            lines.append("No autonomous actions on record yet — "
                         "I haven't acted on your behalf in this window.")
            return "\n".join(lines)
        scores = self.scores_by_type(since_days=since_days)
        if scores:
            bits = []
            for t, s in sorted(scores.items()):
                tr = self.trend(t, window_days=min(14.0, since_days))
                arrow = {"improving": "↗", "declining": "↘"}.get(
                    tr["direction"], "→")
                bits.append(f"{t} {s:.0%}{arrow}")
            lines.append("Quality by type: " + ", ".join(bits))
        fb_rate = self.feedback_rate(since_days=since_days)
        lines.append(f"Owner feedback on {fb_rate:.0%} of actions"
                     + (" — flying partly blind, ask more often"
                        if fb_rate < 0.3 and len(recs) >= 5 else ""))
        lines.append("")
        lines.append(f"Recent actions ({len(recs)}):")
        for r in recs:
            fb = {1: " 👍", -1: " 👎"}.get(r.owner_feedback, "")
            when = time.strftime("%m-%d %H:%M",
                                 time.localtime(r.created_at))
            desc = (r.description[:80] + "…") \
                if len(r.description) > 80 else r.description
            lines.append(f"• [{r.action_type}] {desc} —"
                         f" {r.outcome}{fb} ({when})")
        patterns = self.low_quality_patterns(since_days=since_days)
        if patterns:
            lines.append("")
            lines.append("⚠️ Needs work:")
            for p in patterns:
                lines.append(f"• {p['action_type']} at {p['score']:.0%} —"
                             f" {len(p['counterfactuals'])} lesson(s) noted")
                for cf in p["counterfactuals"][:2]:
                    lines.append(f"  ↳ better: {cf['better_action']}"
                                 + (f" — {cf['why']}" if cf.get("why") else ""))
        else:
            lines.append("")
            lines.append("✅ No weak spots in this window.")
        mismatched = self.inconsistencies(since_days=since_days, limit=5)
        if mismatched:
            lines.append("")
            lines.append(f"🤔 {len(mismatched)} outcome(s) the owner"
                         " contradicted — worth a second look:")
            for m in mismatched[:3]:
                lines.append(f"• [{m['action_type']}]"
                             f" {m['description'][:70]} — {m['note']}")
        return "\n".join(lines)


def _ledger_for(settings: Any = None) -> RepresentationLedger:
    """Ledger bound to the agent data dir (same home pattern as finance)."""
    home = getattr(settings, "home_path", None) if settings else None
    base = Path(home) if home else Path.home()
    path = base / ".nomorals" / "representation.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return RepresentationLedger(path)


def log_representation_action(
    action_type: str,
    description: str,
    *,
    settings: Any = None,
    outcome: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """One-line hook for autonomous-action sites.  Never raises — returns
    the action id ("" when logging failed)."""
    try:
        rec = _ledger_for(settings).log_action(
            action_type, description, outcome=outcome, metadata=metadata)
        return rec.id
    except Exception as exc:  # noqa: BLE001
        _log.warning("representation hook failed: %s", exc)
        return ""
