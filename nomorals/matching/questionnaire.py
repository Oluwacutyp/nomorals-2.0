"""Importance-weighted preference capture (#83) — the OkCupid pattern.

"How important is remote work? (1–5)" — answer-more → better-results.
Deal-breakers (budget, availability, intent — the Feeld pattern) are
captured BEFORE matching: the app absorbs the awkward conversation,
so matching only ever sees candidates that clear the bar.

``Questionnaire`` is per-owner, SQLite-persisted, never raises.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Question",
    "DealBreaker",
    "Questionnaire",
]

# Curated starter prompts per surface — the "answer more" loop begins here.
STARTER_QUESTIONS: dict[str, list[tuple[str, str]]] = {
    "gig": [
        ("remote", "How important is remote work to you? (1–5)"),
        ("budget_fit", "How important is staying inside the stated budget? (1–5)"),
        ("speed", "How important is fast turnaround? (1–5)"),
        ("experience", "How important is proven experience over price? (1–5)"),
    ],
    "community": [
        ("shared_interest", "How important is a shared interest vs just vibes? (1–5)"),
        ("availability", "How important is matching availability? (1–5)"),
        ("mentor", "How important is learning something new? (1–5)"),
    ],
    "learning": [
        ("depth", "How important is depth over breadth? (1–5)"),
        ("practical", "How important is hands-on practice? (1–5)"),
        ("time", "How important is fitting into little time? (1–5)"),
    ],
}


@dataclass
class Question:
    question_id: str = ""
    text: str = ""
    attribute: str = ""        # candidate attribute this scores (e.g. "remote")
    importance: int = 0        # 1–5; 0 = unanswered


@dataclass
class DealBreaker:
    """Hard filter applied BEFORE any scoring. Never negotiable."""

    name: str = ""             # e.g. "budget", "availability", "intent"
    attribute: str = ""        # candidate attribute key
    predicate: str = ""        # "eq" | "ne" | "le" | "ge" | "in" | "contains"
    value: object = None

    def passes(self, candidate_attrs: dict) -> bool:
        try:
            actual = (candidate_attrs or {}).get(self.attribute)
            want = self.value
            if self.predicate == "eq":
                return actual == want
            if self.predicate == "ne":
                return actual != want
            if self.predicate == "le":
                return actual is not None and float(actual) <= float(want)
            if self.predicate == "ge":
                return actual is not None and float(actual) >= float(want)
            if self.predicate == "in":
                return actual in (want or [])
            if self.predicate == "contains":
                return want in (actual or [])
            return True
        except Exception:  # noqa: BLE001 — a bad predicate never blocks
            return True


_DEFAULT_DB = os.path.expanduser("~/.nomorals/matching/questionnaire.db")


class Questionnaire:
    """Per-owner preference capture + weighted candidate scoring."""

    def __init__(self, owner: str = "owner", db_path: str = "") -> None:
        self.owner = owner or "owner"
        self._db = None
        try:
            path = db_path or _DEFAULT_DB
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS answers (
                       owner TEXT, question_id TEXT, text TEXT,
                       attribute TEXT, importance INTEGER, answer TEXT,
                       updated_at REAL,
                       PRIMARY KEY (owner, question_id))"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS dealbreakers (
                       owner TEXT, name TEXT, attribute TEXT,
                       predicate TEXT, value TEXT,
                       PRIMARY KEY (owner, name))"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS ranks (
                       owner TEXT, who TEXT, side TEXT, ranking TEXT,
                       updated_at REAL,
                       PRIMARY KEY (owner, who, side))"""
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("matching.questionnaire: db unavailable",
                         exc_info=True)
            self._db = None

    # ── questions ───────────────────────────────────────────────

    def ensure_starters(self, surface: str = "gig") -> list[Question]:
        """Seed the curated starter set (idempotent). Never raises."""
        try:
            out = []
            for attr, text in STARTER_QUESTIONS.get(surface or "gig",
                                                   STARTER_QUESTIONS["gig"]):
                qid = "%s:%s" % (surface or "gig", attr)
                out.append(self.add_question(qid, text, attr))
            return [q for q in out if q.question_id]
        except Exception:  # noqa: BLE001
            return []

    def add_question(self, question_id: str, text: str,
                     attribute: str = "") -> Question:
        try:
            if self._db is None or not question_id:
                return Question()
            self._db.execute(
                """INSERT OR IGNORE INTO answers
                   (owner, question_id, text, attribute, importance, answer,
                    updated_at)
                   VALUES (?, ?, ?, ?, 0, '', ?)""",
                (self.owner, question_id, text, attribute or question_id,
                 time.time()))
            self._db.commit()
            return self.get(question_id) or Question(question_id=question_id,
                                                     text=text,
                                                     attribute=attribute)
        except Exception:  # noqa: BLE001
            return Question()

    def get(self, question_id: str) -> Question | None:
        try:
            if self._db is None:
                return None
            r = self._db.execute(
                "SELECT * FROM answers WHERE owner = ? AND question_id = ?",
                (self.owner, question_id)).fetchone()
            if not r:
                return None
            return Question(question_id=r["question_id"], text=r["text"] or "",
                            attribute=r["attribute"] or "",
                            importance=int(r["importance"] or 0))
        except Exception:  # noqa: BLE001
            return None

    def answer(self, question_id: str, importance: int,
               value: str = "") -> bool:
        """Answer 1–5 (importance). More answers → better results."""
        try:
            if self._db is None or not question_id:
                return False
            importance = max(1, min(5, int(importance)))
            self._db.execute(
                """UPDATE answers SET importance = ?, answer = ?,
                                  updated_at = ?
                   WHERE owner = ? AND question_id = ?""",
                (importance, value or "", time.time(), self.owner,
                 question_id))
            self._db.commit()
            return self._db.total_changes > 0
        except Exception:  # noqa: BLE001
            return False

    def unanswered(self) -> list[Question]:
        """The nudge list: answer more → better matches."""
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                """SELECT * FROM answers WHERE owner = ? AND importance = 0
                   ORDER BY question_id""", (self.owner,)).fetchall()
            return [Question(question_id=r["question_id"], text=r["text"] or "",
                             attribute=r["attribute"] or "")
                    for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def weights(self) -> dict[str, float]:
        """{attribute: importance} for answered questions."""
        try:
            if self._db is None:
                return {}
            rows = self._db.execute(
                """SELECT attribute, importance FROM answers
                   WHERE owner = ? AND importance > 0""",
                (self.owner,)).fetchall()
            out: dict[str, float] = {}
            for r in rows:
                attr = (r["attribute"] or "").strip().lower()
                if attr:
                    out[attr] = out.get(attr, 0.0) + float(r["importance"] or 0)
            return out
        except Exception:  # noqa: BLE001
            return {}

    # ── deal-breakers ───────────────────────────────────────────

    def set_dealbreaker(self, name: str, attribute: str, predicate: str,
                        value: object) -> bool:
        """Front-load the awkward conversation: budget, availability, intent."""
        try:
            if self._db is None or not name:
                return False
            self._db.execute(
                """INSERT OR REPLACE INTO dealbreakers
                   (owner, name, attribute, predicate, value)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.owner, name, attribute, predicate,
                 json.dumps(value, default=str)))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def dealbreakers(self) -> list[DealBreaker]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM dealbreakers WHERE owner = ?",
                (self.owner,)).fetchall()
            out = []
            for r in rows:
                try:
                    value = json.loads(r["value"] or "null")
                except Exception:  # noqa: BLE001
                    value = None
                out.append(DealBreaker(name=r["name"] or "",
                                       attribute=r["attribute"] or "",
                                       predicate=r["predicate"] or "eq",
                                       value=value))
            return out
        except Exception:  # noqa: BLE001
            return []

    # ── two-sided rankings (for stable matching) ──────────────────

    def set_ranking(self, who: str, side: str,
                    ranking: list[str]) -> bool:
        """Register ``who``'s ranked list (most-preferred first).

        ``side`` is ``"proposer"`` or ``"reviewer"``.
        """
        try:
            if self._db is None or not who or side not in ("proposer",
                                                          "reviewer"):
                return False
            clean = [r.strip() for r in (ranking or []) if r and r.strip()]
            self._db.execute(
                """INSERT OR REPLACE INTO ranks
                   (owner, who, side, ranking, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.owner, who, side, json.dumps(clean), time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def rankings(self, side: str) -> dict[str, list[str]]:
        """{who: ranked list} for one side."""
        try:
            if self._db is None or side not in ("proposer", "reviewer"):
                return {}
            rows = self._db.execute(
                "SELECT who, ranking FROM ranks WHERE owner = ? AND side = ?",
                (self.owner, side)).fetchall()
            out: dict[str, list[str]] = {}
            for r in rows:
                try:
                    lst = json.loads(r["ranking"] or "[]")
                except Exception:  # noqa: BLE001
                    lst = []
                out[r["who"]] = [x for x in lst if isinstance(x, str)]
            return out
        except Exception:  # noqa: BLE001
            return {}

    # ── scoring ─────────────────────────────────────────────────

    def filter_dealbreakers(self, candidates: list[dict]) -> list[dict]:
        """Hard filters first — matching only sees what clears the bar."""
        try:
            breakers = self.dealbreakers()
            if not breakers:
                return list(candidates or [])
            return [c for c in (candidates or [])
                    if all(b.passes((c or {}).get("attributes") or {})
                           for b in breakers)]
        except Exception:  # noqa: BLE001
            return list(candidates or [])

    def score(self, candidate: dict) -> float:
        """0..1 weighted preference score for one candidate dict.

        Each answered question contributes importance × attribute match,
        where the match is 1.0 when the candidate's attribute value is
        truthy / matches the stored answer, scaled by importance.
        """
        try:
            w = self.weights()
            if not w:
                return 0.5  # no signal yet — neutral, not zero
            attrs = (candidate or {}).get("attributes") or {}
            total = sum(w.values()) or 1.0
            got = 0.0
            for attr, importance in w.items():
                val = attrs.get(attr)
                match = 1.0 if val else 0.0
                # Numeric attributes scale by magnitude (capped at 1).
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    match = max(0.0, min(1.0, float(val)))
                got += importance * match
            return max(0.0, min(1.0, got / total))
        except Exception:  # noqa: BLE001
            return 0.5
