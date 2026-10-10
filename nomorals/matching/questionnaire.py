"""Importance-weighted preference capture (#83) — the OkCupid pattern.

OkCupid's real algorithm (Rudder's TED-Ed lesson) asks THREE things per
question: (1) my answer, (2) the answers I'd *accept* from a match,
(3) importance — irrelevant / a little / somewhat / very / mandatory,
weighted 0 / 1 / 10 / 50 / 250. Match% is the geometric mean of both
sides' satisfaction over jointly-answered questions:

    match% = sqrt((my_pts/my_max) × (their_pts/their_max)) × 100

Deal-breakers (budget, availability, intent — the Feeld pattern) are
captured BEFORE matching: the app absorbs the awkward conversation,
so matching only ever sees candidates that clear the bar. Feeld's
"Reflections" groups prompts into Desires / Boundaries / Relationships —
we keep that section split so boundary questions feed deal-breakers
naturally.

``Questionnaire`` is per-owner, SQLite-persisted, never raises.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid  # noqa: F401  (kept for forward-compat of stored ids)
from dataclasses import dataclass, field
from math import sqrt

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Question",
    "DealBreaker",
    "Questionnaire",
    "IMPORTANCE_WEIGHTS",
    "SECTIONS",
]

# OkCupid's importance ladder (Rudder, TED-Ed): exponential weights are
# what make "very important" actually dominate. Our 1–5 scale starts at
# "a little" (unanswered = 0 = irrelevant).
IMPORTANCE_WEIGHTS: dict[int, int] = {1: 1, 2: 10, 3: 50, 4: 250, 5: 250}

# Feeld "Reflections" sections: desires (what I want), boundaries
# (red flags → deal-breakers), practical (logistics).
SECTIONS = ("desires", "boundaries", "practical")

# Curated starter prompts per surface — the "answer more" loop begins here.
# (attribute, prompt text, section)
STARTER_QUESTIONS: dict[str, list[tuple[str, str, str]]] = {
    "gig": [
        ("remote", "How important is remote work to you? (1–5)", "practical"),
        ("budget_fit", "How important is staying inside the stated budget? (1–5)",
         "boundaries"),
        ("speed", "How important is fast turnaround? (1–5)", "desires"),
        ("experience", "How important is proven experience over price? (1–5)",
         "desires"),
    ],
    "community": [
        ("shared_interest", "How important is a shared interest vs just vibes? (1–5)",
         "desires"),
        ("availability", "How important is matching availability? (1–5)",
         "practical"),
        ("mentor", "How important is learning something new? (1–5)", "desires"),
        ("respect", "How important is a respectful tone, no drama? (1–5)",
         "boundaries"),
    ],
    "learning": [
        ("depth", "How important is depth over breadth? (1–5)", "desires"),
        ("practical", "How important is hands-on practice? (1–5)", "desires"),
        ("time", "How important is fitting into little time? (1–5)",
         "practical"),
    ],
}


@dataclass
class Question:
    question_id: str = ""
    text: str = ""
    attribute: str = ""        # candidate attribute this scores (e.g. "remote")
    importance: int = 0        # 1–5; 0 = unanswered
    acceptable: list = field(default_factory=list)  # answers I'd accept
    answer: str = ""           # my own answer
    section: str = ""          # desires | boundaries | practical


@dataclass
class DealBreaker:
    """Hard filter applied BEFORE any scoring. Never negotiable."""

    name: str = ""             # e.g. "budget", "availability", "intent"
    attribute: str = ""        # candidate attribute key
    predicate: str = ""        # eq|ne|le|ge|in|contains|between
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
            if self.predicate == "between":
                lo, hi = (want or [None, None])[:2]
                return (actual is not None and lo is not None
                        and hi is not None
                        and float(lo) <= float(actual) <= float(hi))
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
            self._migrate_answers()
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
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS capacities (
                       owner TEXT, reviewer TEXT, capacity INTEGER,
                       PRIMARY KEY (owner, reviewer))"""
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("matching.questionnaire: db unavailable",
                         exc_info=True)
            self._db = None

    def _migrate_answers(self) -> None:
        """Add `acceptable` + `section` columns to older databases."""
        cols = {r[1] for r in
                self._db.execute("PRAGMA table_info(answers)").fetchall()}
        if "acceptable" not in cols:
            self._db.execute(
                "ALTER TABLE answers ADD COLUMN acceptable TEXT DEFAULT '[]'")
        if "section" not in cols:
            self._db.execute(
                "ALTER TABLE answers ADD COLUMN section TEXT DEFAULT ''")

    # ── questions ───────────────────────────────────────────────

    def ensure_starters(self, surface: str = "gig") -> list[Question]:
        """Seed the curated starter set (idempotent). Never raises."""
        try:
            out = []
            for attr, text, section in STARTER_QUESTIONS.get(
                    surface or "gig", STARTER_QUESTIONS["gig"]):
                qid = "%s:%s" % (surface or "gig", attr)
                out.append(self.add_question(qid, text, attr,
                                             section=section))
            return [q for q in out if q.question_id]
        except Exception:  # noqa: BLE001
            return []

    def add_question(self, question_id: str, text: str,
                     attribute: str = "", section: str = "") -> Question:
        try:
            if self._db is None or not question_id:
                return Question()
            section = section if section in SECTIONS else ""
            self._db.execute(
                """INSERT OR IGNORE INTO answers
                   (owner, question_id, text, attribute, importance, answer,
                    acceptable, section, updated_at)
                   VALUES (?, ?, ?, ?, 0, '', '[]', ?, ?)""",
                (self.owner, question_id, text, attribute or question_id,
                 section, time.time()))
            self._db.commit()
            return self.get(question_id) or Question(question_id=question_id,
                                                     text=text,
                                                     attribute=attribute)
        except Exception:  # noqa: BLE001
            return Question()

    def _row_to_question(self, r) -> Question:
        try:
            acceptable = json.loads(r["acceptable"] or "[]")
        except Exception:  # noqa: BLE001
            acceptable = []
        try:
            section = r["section"] or ""
        except (IndexError, KeyError):
            section = ""
        return Question(
            question_id=r["question_id"], text=r["text"] or "",
            attribute=r["attribute"] or "",
            importance=int(r["importance"] or 0),
            acceptable=[a for a in acceptable if isinstance(a, str)],
            answer=r["answer"] or "", section=section)

    def get(self, question_id: str) -> Question | None:
        try:
            if self._db is None:
                return None
            r = self._db.execute(
                "SELECT * FROM answers WHERE owner = ? AND question_id = ?",
                (self.owner, question_id)).fetchone()
            return self._row_to_question(r) if r else None
        except Exception:  # noqa: BLE001
            return None

    def answer(self, question_id: str, importance: int,
               value: str = "") -> bool:
        """Answer 1–5 (importance) + my own answer value.

        More answers → better results. ``value`` is what *I* answer —
        pair with ``set_acceptable`` for the full OkCupid pattern.
        """
        try:
            if self._db is None or not question_id:
                return False
            importance = max(1, min(5, int(importance)))
            cur = self._db.execute(
                """UPDATE answers SET importance = ?, answer = ?,
                                  updated_at = ?
                   WHERE owner = ? AND question_id = ?""",
                (importance, value or "", time.time(), self.owner,
                 question_id))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def set_acceptable(self, question_id: str,
                       values: list[str]) -> bool:
        """The answers I'd ACCEPT from a match (OkCupid's 2nd input).

        Without this, satisfaction falls back to "candidate's value ==
        my value" (homophily default). Never raises.
        """
        try:
            if self._db is None or not question_id:
                return False
            clean = [str(v).strip() for v in (values or [])
                     if str(v).strip()]
            cur = self._db.execute(
                """UPDATE answers SET acceptable = ?, updated_at = ?
                   WHERE owner = ? AND question_id = ?""",
                (json.dumps(clean), time.time(), self.owner, question_id))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def unanswered(self, section: str = "") -> list[Question]:
        """The nudge list: answer more → better matches."""
        try:
            if self._db is None:
                return []
            if section in SECTIONS:
                rows = self._db.execute(
                    """SELECT * FROM answers
                       WHERE owner = ? AND importance = 0 AND section = ?
                       ORDER BY question_id""",
                    (self.owner, section)).fetchall()
            else:
                rows = self._db.execute(
                    """SELECT * FROM answers WHERE owner = ? AND importance = 0
                       ORDER BY question_id""", (self.owner,)).fetchall()
            return [self._row_to_question(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def answered(self) -> list[Question]:
        """All answered questions, most-recently-updated first."""
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                """SELECT * FROM answers WHERE owner = ? AND importance > 0
                   ORDER BY updated_at DESC""", (self.owner,)).fetchall()
            return [self._row_to_question(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def weights(self) -> dict[str, float]:
        """{attribute: importance} for answered questions (tag weights)."""
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

    def okcupid_weights(self) -> dict[str, int]:
        """{attribute: ladder weight} — 1/10/50/250, not 1–5."""
        try:
            if self._db is None:
                return {}
            rows = self._db.execute(
                """SELECT attribute, importance FROM answers
                   WHERE owner = ? AND importance > 0""",
                (self.owner,)).fetchall()
            out: dict[str, int] = {}
            for r in rows:
                attr = (r["attribute"] or "").strip().lower()
                if attr:
                    w = IMPORTANCE_WEIGHTS.get(int(r["importance"] or 1), 1)
                    out[attr] = max(out.get(attr, 0), w)
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
            if predicate not in ("eq", "ne", "le", "ge", "in", "contains",
                                 "between"):
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

    def remove_dealbreaker(self, name: str) -> bool:
        try:
            if self._db is None or not name:
                return False
            cur = self._db.execute(
                "DELETE FROM dealbreakers WHERE owner = ? AND name = ?",
                (self.owner, name))
            self._db.commit()
            return cur.rowcount > 0
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

    def boundary_questions(self) -> list[Question]:
        """'Boundaries'-section questions — Feeld's red-flags list."""
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                """SELECT * FROM answers
                   WHERE owner = ? AND section = 'boundaries'
                   ORDER BY question_id""", (self.owner,)).fetchall()
            return [self._row_to_question(r) for r in rows]
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

    # ── capacities (for Hospital–Resident matching) ───────────────

    def set_capacity(self, reviewer: str, capacity: int) -> bool:
        """A reviewer takes up to ``capacity`` proposers (mentor's seats)."""
        try:
            if self._db is None or not reviewer:
                return False
            capacity = max(1, int(capacity))
            self._db.execute(
                """INSERT OR REPLACE INTO capacities
                   (owner, reviewer, capacity) VALUES (?, ?, ?)""",
                (self.owner, reviewer, capacity))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def capacities(self) -> dict[str, int]:
        try:
            if self._db is None:
                return {}
            rows = self._db.execute(
                "SELECT reviewer, capacity FROM capacities WHERE owner = ?",
                (self.owner,)).fetchall()
            return {r["reviewer"]: int(r["capacity"] or 1) for r in rows}
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

    def match_breakdown(self, candidate_attrs: dict) -> list[dict]:
        """Per-question contribution: the explainability half of scoring.

        Returns ``[{question_id, text, weight, matched, detail}]``.
        """
        out: list[dict] = []
        try:
            for q in self.answered():
                attr = (q.attribute or "").strip().lower()
                if not attr:
                    continue
                attrs = candidate_attrs or {}
                # Attribute keys are matched case-insensitively.
                val = attrs.get(attr)
                if val is None:
                    for k, v in attrs.items():
                        if str(k).strip().lower() == attr:
                            val = v
                            break
                if val is None:
                    continue  # candidate says nothing here — not counted
                weight = IMPORTANCE_WEIGHTS.get(q.importance, 1)
                cand = str(val).strip()
                if q.acceptable:
                    matched = cand in q.acceptable
                    detail = ("accept %s" % cand if matched
                              else "want %s, got %s"
                              % ("|".join(q.acceptable), cand))
                elif q.answer:
                    matched = cand == q.answer.strip()
                    detail = ("= %s" % cand if matched
                              else "mine %s, theirs %s" % (q.answer, cand))
                else:
                    matched = bool(val) and val != "false" and val != "0"
                    detail = "present" if matched else "missing"
                out.append({"question_id": q.question_id, "text": q.text,
                            "weight": weight, "matched": matched,
                            "detail": detail})
            return out
        except Exception:  # noqa: BLE001
            return out

    def satisfaction(self, candidate_attrs: dict) -> float:
        """0..1 — how well a candidate satisfies MY answered questions.

        One-sided OkCupid: Σ weightᵢ·[their value acceptable to me] /
        Σ weightᵢ over questions the candidate answers. 0.5 when there's
        no signal yet (neutral, not zero).
        """
        try:
            rows = self.match_breakdown(candidate_attrs)
            total = sum(r["weight"] for r in rows)
            if not total:
                return 0.5
            got = sum(r["weight"] for r in rows if r["matched"])
            return max(0.0, min(1.0, got / total))
        except Exception:  # noqa: BLE001
            return 0.5

    def score(self, candidate: dict) -> float:
        """0..1 weighted preference score for one candidate dict.

        Acceptable-set aware (OkCupid semantics); falls back to the
        legacy truthy/numeric heuristic when no acceptable sets exist.
        """
        try:
            attrs = (candidate or {}).get("attributes") or {}
            rows = self.match_breakdown(attrs)
            if rows:
                return self.satisfaction(attrs)
            # Legacy path: no acceptable data — truthy heuristic.
            w = self.weights()
            if not w:
                return 0.5
            total = sum(w.values()) or 1.0
            got = 0.0
            for attr, importance in w.items():
                val = attrs.get(attr)
                match = 1.0 if val else 0.0
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    match = max(0.0, min(1.0, float(val)))
                got += importance * match
            return max(0.0, min(1.0, got / total))
        except Exception:  # noqa: BLE001
            return 0.5

    def match_percent(self, other: "Questionnaire") -> float:
        """0–100 two-sided compatibility — the real OkCupid formula.

        sqrt((my_pts/my_max) × (their_pts/their_max)) × 100 over
        jointly-answered questions. Symmetric, grounded, explainable.
        """
        try:
            mine = {q.question_id: q for q in self.answered()}
            theirs = {q.question_id: q for q in other.answered()}
            common = [qid for qid in mine
                      if qid in theirs and mine[qid].importance > 0
                      and theirs[qid].importance > 0]
            if not common:
                return 0.0

            def acceptable(q: Question) -> set[str]:
                if q.acceptable:
                    return set(q.acceptable)
                return {q.answer.strip()} if q.answer.strip() else set()

            my_pts = my_max = their_pts = their_max = 0
            for qid in common:
                mq, tq = mine[qid], theirs[qid]
                mw = IMPORTANCE_WEIGHTS.get(mq.importance, 1)
                tw = IMPORTANCE_WEIGHTS.get(tq.importance, 1)
                my_max += mw
                their_max += tw
                if tq.answer.strip() and tq.answer.strip() in acceptable(mq):
                    my_pts += mw
                if mq.answer.strip() and mq.answer.strip() in acceptable(tq):
                    their_pts += tw
            if not my_max or not their_max:
                return 0.0
            return 100.0 * sqrt((my_pts / my_max) * (their_pts / their_max))
        except Exception:  # noqa: BLE001
            _log.warning("matching.questionnaire: match_percent failed",
                         exc_info=True)
            return 0.0
