"""Mistake notebook — wrong answers become spaced-repetition flashcards.

Build-map #46. Every tutoring mistake is recorded with its correction and
queued for review on the forgetting curve instead of being repeated.

Scheduling is native FSRS-lite: the DSR (Difficulty–Stability–Retrievability)
core of FSRS-4.5 (open-spaced-repetition, MIT), with the published default
weights and the documented update rules. The short-term (same-day) learning
steps and interval fuzzing of full FSRS are deliberately omitted — the class
docstring says so. An external ``RepetitionScheduler`` may still be attached
for back-compat; it is fed in parallel and never allowed to break tutoring.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MistakeNotebook",
    "MistakeCard",
    "FSRS",
    "FSRSCardState",
    "FSRS_DEFAULT_W",
    "card_from_mistake",
    "grade_for_verdict",
    "notebooks_dir",
]

# ── FSRS-lite ────────────────────────────────────────────────────────────────
# DSR core of FSRS-4.5 (open-spaced-repetition, MIT). Default 17-weight vector
# as published for FSRS-4.5 (squeakyrobot/fsrs docs, "Default v4.5 weight
# vector"); kept as plain floats so this module stays zero-dependency.
#
# Deliberately NOT included (honest subset):
#   - short-term / same-day learning steps (w17-w20 machinery)
#   - interval fuzzing
#   - per-user weight optimization (needs cohort review logs)

FSRS_DEFAULT_W: tuple[float, ...] = (
    0.4,    # w0: initial stability, Again
    0.6,    # w1: initial stability, Hard
    2.4,    # w2: initial stability, Good
    5.8,    # w3: initial stability, Easy
    4.93,   # w4: initial difficulty base
    0.94,   # w5: initial difficulty decay
    0.86,   # w6: difficulty update factor
    0.01,   # w7: difficulty mean reversion
    1.49,   # w8: stability update base
    0.14,   # w9: stability update exp (S)
    0.94,   # w10: stability update exp (R)
    2.18,   # w11: stability fail base
    0.05,   # w12: stability fail exp (D)
    0.34,   # w13: stability fail exp (S)
    1.26,   # w14: stability fail exp (R)
    0.29,   # w15: hard penalty
    2.61,   # w16: easy bonus
)

_DECAY = -0.5                     # FSRS-4.5 forgetting-curve exponent
_FACTOR = 0.9 ** (1.0 / _DECAY) - 1.0   # = 19/81; R == 0.9 exactly when t == S
_MAX_INTERVAL_DAYS = 36500.0
_MIN_INTERVAL_DAYS = 1.0


def grade_for_verdict(verdict: str, *, easy: bool = False) -> int:
    """Map a tutor verdict to an FSRS grade 1..4 (Again/Hard/Good/Easy)."""
    v = (verdict or "").strip().lower()
    if v == "wrong":
        return 1
    if v == "partial":
        return 2
    if v == "correct":
        return 4 if easy else 3
    return 1


@dataclass
class FSRSCardState:
    """DSR state for one card. ``due_at`` is a unix timestamp."""

    difficulty: float = 5.0   # 1..10
    stability: float = 0.0    # days; 0 == never reviewed
    reps: int = 0
    lapses: int = 0
    last_review: float = 0.0
    due_at: float = 0.0

    @property
    def state(self) -> str:
        """New / Young (interval < 21d) / Mature — the FSRS card states
        relevant without the short-term learning machine."""
        if self.reps == 0:
            return "new"
        interval = (self.due_at - self.last_review) / 86400.0
        return "mature" if interval >= 21.0 else "young"

    def to_dict(self) -> dict[str, Any]:
        return {"difficulty": self.difficulty, "stability": self.stability,
                "reps": self.reps, "lapses": self.lapses,
                "last_review": self.last_review, "due_at": self.due_at}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FSRSCardState":
        return cls(difficulty=float(d.get("difficulty", 5.0)),
                   stability=float(d.get("stability", 0.0)),
                   reps=int(d.get("reps", 0)),
                   lapses=int(d.get("lapses", 0)),
                   last_review=float(d.get("last_review", 0.0)),
                   due_at=float(d.get("due_at", 0.0)))


class FSRS:
    """FSRS-lite scheduler: DSR core of FSRS-4.5 with published defaults.

    Pure math, no storage — the notebook owns persistence. ``weights`` may be
    replaced with optimizer-fitted values when review history exists.
    """

    def __init__(self, weights: tuple[float, ...] | None = None,
                 request_retention: float = 0.9) -> None:
        w = tuple(weights) if weights else FSRS_DEFAULT_W
        if len(w) < 17:
            raise ValueError("FSRS needs at least the 17 FSRS-4.5 weights")
        self.w = w
        self.request_retention = min(0.99, max(0.5, request_retention))

    # -- core --------------------------------------------------------------
    def retrievability(self, stability: float, elapsed_days: float) -> float:
        """R(t, S) = (1 + FACTOR·t/S)^DECAY — the FSRS forgetting curve."""
        if stability <= 0:
            return 0.0
        t = max(0.0, elapsed_days)
        return (1.0 + _FACTOR * t / stability) ** _DECAY

    def next_interval(self, stability: float) -> float:
        """Inverse of the forgetting curve at the requested retention,
        clamped to [1, 36500] days."""
        r = self.request_retention
        interval = (stability / _FACTOR) * (r ** (1.0 / _DECAY) - 1.0)
        return min(_MAX_INTERVAL_DAYS, max(_MIN_INTERVAL_DAYS, interval))

    def initial_stability(self, grade: int) -> float:
        return self.w[max(1, min(4, grade)) - 1]

    def initial_difficulty(self, grade: int) -> float:
        g = max(1, min(4, grade))
        d = self.w[4] - math.exp(self.w[5] * (g - 1)) + 1.0
        return min(10.0, max(1.0, d))

    def review(self, state: FSRSCardState, grade: int,
               now: float | None = None) -> FSRSCardState:
        """Apply one review. Mutates and returns ``state``."""
        now = time.time() if now is None else now
        g = max(1, min(4, int(grade)))
        w = self.w
        if state.reps == 0:
            state.difficulty = self.initial_difficulty(g)
            state.stability = self.initial_stability(g)
        else:
            elapsed = max(0.0, (now - state.last_review) / 86400.0)
            r = self.retrievability(state.stability, elapsed)
            # Difficulty update with linear damping + mean reversion to D0(Easy).
            delta_d = -w[6] * (g - 3)
            d_prime = state.difficulty + delta_d * (10.0 - state.difficulty) / 9.0
            d0_easy = self.initial_difficulty(4)
            state.difficulty = min(10.0, max(
                1.0, w[7] * d0_easy + (1.0 - w[7]) * d_prime))
            if g == 1:
                # Lapse.
                state.lapses += 1
                state.stability = (
                    w[11] * state.difficulty ** (-w[12])
                    * ((state.stability + 1.0) ** w[13] - 1.0)
                    * math.exp(w[14] * (1.0 - r)))
            else:
                # Successful recall.
                bonus = w[15] if g == 2 else (w[16] if g == 4 else 1.0)
                state.stability = state.stability * (
                    1.0 + math.exp(w[8]) * (11.0 - state.difficulty)
                    * state.stability ** (-w[9])
                    * (math.exp(w[10] * (1.0 - r)) - 1.0) * bonus)
            state.stability = max(0.01, state.stability)
        state.reps += 1
        state.last_review = now
        state.due_at = now + self.next_interval(state.stability) * 86400.0
        return state

    def due(self, state: FSRSCardState, now: float | None = None) -> bool:
        """True when retrievability has fallen to/below the request retention."""
        now = time.time() if now is None else now
        if state.reps == 0:
            return True
        elapsed = max(0.0, (now - state.last_review) / 86400.0)
        return self.retrievability(state.stability, elapsed) <= self.request_retention


# ── mistake card ─────────────────────────────────────────────────────────────

@dataclass
class MistakeCard:
    """One recorded mistake + its flashcard."""

    id: str
    question: str
    student_answer: str
    correct: str
    why: str = ""
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    reviews: int = 0
    last_grade: int | None = None
    fsrs: FSRSCardState = field(default_factory=FSRSCardState)

    @property
    def front(self) -> str:
        return f"{self.question}\n(You said: {self.student_answer})"

    @property
    def back(self) -> str:
        out = f"Correct: {self.correct}"
        if self.why:
            out += f"\nWhy: {self.why}"
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "question": self.question,
            "student_answer": self.student_answer, "correct": self.correct,
            "why": self.why, "topic": self.topic, "tags": list(self.tags),
            "created_at": self.created_at, "reviews": self.reviews,
            "last_grade": self.last_grade, "fsrs": self.fsrs.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MistakeCard":
        fsrs_d = d.get("fsrs") or {}
        return cls(
            id=str(d.get("id", "m_" + uuid.uuid4().hex[:10])),
            question=str(d.get("question", "")),
            student_answer=str(d.get("student_answer", "")),
            correct=str(d.get("correct", "")),
            why=str(d.get("why", "")),
            topic=str(d.get("topic", "")),
            tags=list(d.get("tags") or []),
            created_at=float(d.get("created_at", 0.0)),
            reviews=int(d.get("reviews", 0)),
            last_grade=d.get("last_grade"),
            fsrs=FSRSCardState.from_dict(fsrs_d),
        )


def card_from_mistake(question: str, student_answer: str, correct: str,
                      *, why: str = "", topic: str = "",
                      tags: list[str] | None = None) -> MistakeCard:
    """Build the flashcard for one wrong answer.

    Front: the question + "you said X" (so the error is confronted).
    Back: the correction + why.
    """
    return MistakeCard(
        id="m_" + uuid.uuid4().hex[:10],
        question=(question or "").strip(),
        student_answer=(student_answer or "").strip(),
        correct=(correct or "").strip(),
        why=(why or "").strip(),
        topic=(topic or "").strip(),
        tags=list(tags or []),
        created_at=time.time(),
    )


def notebooks_dir() -> Path:
    return Path.home() / ".nomorals" / "learn" / "notebooks"


class MistakeNotebook:
    """Records mistakes, schedules FSRS reviews, reports review status.

    ``scheduler`` is an optional external :class:`RepetitionScheduler`
    (back-compat — fed in parallel). The native FSRS-lite state on each card
    is the source of truth for due/review/stats. Never raises: a broken
    scheduler must not break tutoring.
    """

    def __init__(self, scheduler: Any | None = None,
                 fsrs: FSRS | None = None) -> None:
        self.scheduler = scheduler
        self.fsrs = fsrs or FSRS()
        self._cards: dict[str, MistakeCard] = {}

    # -- recording ---------------------------------------------------------
    def record(self, question: str, student_answer: str, correct: str,
               *, why: str = "", topic: str = "",
               tags: list[str] | None = None) -> MistakeCard:
        """Record a mistake and queue its flashcard. Returns the card."""
        card = card_from_mistake(question, student_answer, correct,
                                 why=why, topic=topic, tags=tags)
        self._cards[card.id] = card
        if self.scheduler is not None:
            try:
                self.scheduler.schedule(card.id, f"{card.front}\n---\n{card.back}")
            except Exception:  # noqa: BLE001
                _log.debug("flashcard scheduling failed", exc_info=True)
        return card

    def review(self, card_id: str, grade: int,
               now: float | None = None) -> bool:
        """Grade a flashcard review (1-4, FSRS). Returns False if unknown."""
        card = self._cards.get(card_id)
        if card is None:
            return False
        card.reviews += 1
        card.last_grade = max(1, min(4, int(grade)))
        try:
            self.fsrs.review(card.fsrs, card.last_grade,
                             now=time.time() if now is None else now)
        except Exception:  # noqa: BLE001
            _log.debug("FSRS review failed", exc_info=True)
        if self.scheduler is not None:
            try:
                self.scheduler.review(card_id, card.last_grade)
            except Exception:  # noqa: BLE001
                _log.debug("flashcard review failed", exc_info=True)
        return True

    # -- retrieval practice --------------------------------------------------
    def due_cards(self, now: float | None = None,
                  limit: int = 20) -> list[MistakeCard]:
        """Cards due for review, most-overdue (lowest retrievability) first.

        This is what the tutor interleaves into sessions (retrieval practice).
        """
        now = time.time() if now is None else now
        scored: list[tuple[float, MistakeCard]] = []
        for card in self._cards.values():
            try:
                if card.fsrs.reps == 0 or self.fsrs.due(card.fsrs, now):
                    elapsed = max(0.0, (now - card.fsrs.last_review) / 86400.0) \
                        if card.fsrs.reps else 0.0
                    r = self.fsrs.retrievability(card.fsrs.stability, elapsed) \
                        if card.fsrs.reps else 0.0
                    scored.append((r, card))
            except Exception:  # noqa: BLE001
                continue
        scored.sort(key=lambda pair: pair[0])
        return [c for _, c in scored[: max(1, limit)]]

    def notebook(self, now: float | None = None) -> list[dict[str, Any]]:
        """All mistakes with their cards and review status, newest first."""
        now = time.time() if now is None else now
        cards = sorted(self._cards.values(), key=lambda c: -c.created_at)
        out = []
        for card in cards:
            entry = card.to_dict()
            entry["due"] = self._due(card, now)
            entry["fsrs_state"] = card.fsrs.state
            try:
                elapsed = max(0.0, (now - card.fsrs.last_review) / 86400.0)
                entry["retrievability"] = round(
                    self.fsrs.retrievability(card.fsrs.stability, elapsed)
                    if card.fsrs.reps else 0.0, 3)
            except Exception:  # noqa: BLE001
                entry["retrievability"] = 0.0
            out.append(entry)
        return out

    def _due(self, card: MistakeCard, now: float) -> bool:
        try:
            return card.fsrs.reps == 0 or self.fsrs.due(card.fsrs, now)
        except Exception:  # noqa: BLE001
            return False

    def count(self) -> int:
        return len(self._cards)

    def get(self, card_id: str) -> MistakeCard | None:
        return self._cards.get(card_id)

    # -- stats -----------------------------------------------------------------
    def stats(self, now: float | None = None) -> dict[str, Any]:
        """FSRS state breakdown + retention estimate (autoanki-style states)."""
        now = time.time() if now is None else now
        states = {"new": 0, "young": 0, "mature": 0}
        due = 0
        retrievabilities: list[float] = []
        for card in self._cards.values():
            states[card.fsrs.state] = states.get(card.fsrs.state, 0) + 1
            if self._due(card, now):
                due += 1
            if card.fsrs.reps:
                elapsed = max(0.0, (now - card.fsrs.last_review) / 86400.0)
                try:
                    retrievabilities.append(
                        self.fsrs.retrievability(card.fsrs.stability, elapsed))
                except Exception:  # noqa: BLE001
                    pass
        avg_r = (sum(retrievabilities) / len(retrievabilities)
                 if retrievabilities else 0.0)
        return {"total": len(self._cards), "states": states, "due": due,
                "avg_retrievability": round(avg_r, 3),
                "request_retention": self.fsrs.request_retention}

    # -- Anki export -------------------------------------------------------------
    def to_anki_tsv(self, deck: str = "Devon::Mistakes",
                    tag: str = "devon-mistake") -> str:
        """The notebook as Anki's plain-text import format (UTF-8, tab
        separated, headers per the Anki manual: #separator:tab, #html:true,
        #notetype:Basic, #deck, #tags, #columns). Newlines inside fields use
        <br>; File -> Import and it lands in the deck."""
        lines = [
            "#separator:tab",
            "#html:true",
            "#notetype:Basic",
            f"#deck:{deck}",
            f"#tags:{tag}",
            "#columns:Front\tBack",
        ]
        for card in sorted(self._cards.values(),
                           key=lambda c: c.created_at):
            front = card.front.replace("\n", "<br>").replace("\t", " ")
            back = card.back.replace("\n", "<br>").replace("\t", " ")
            lines.append(f"{front}\t{back}")
        return "\n".join(lines) + "\n"

    def export_anki(self, path: str | Path, deck: str = "Devon::Mistakes",
                    tag: str = "devon-mistake") -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_anki_tsv(deck=deck, tag=tag), encoding="utf-8")
        return str(p)

    # -- persistence ---------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> str:
        p = Path(path) if path else notebooks_dir() / "default.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"cards": [c.to_dict() for c in self._cards.values()]},
            indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | Path | None = None,
             scheduler: Any | None = None) -> "MistakeNotebook":
        nb = cls(scheduler=scheduler)
        p = Path(path) if path else notebooks_dir() / "default.json"
        if not p.is_file():
            return nb
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("cards") or []:
                card = MistakeCard.from_dict(d)
                nb._cards[card.id] = card
        except Exception:  # noqa: BLE001
            _log.debug("notebook load failed", exc_info=True)
        return nb
