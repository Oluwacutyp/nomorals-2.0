"""Mistake notebook — wrong answers become spaced-repetition flashcards.

Build-map #46. Every tutoring mistake is recorded with its correction and
queued into the :class:`RepetitionScheduler` (build-map #40) so it resurfaces
on the forgetting curve instead of being repeated.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["MistakeNotebook", "MistakeCard", "card_from_mistake"]


@dataclass
class MistakeCard:
    """One recorded mistake + its flashcard."""

    id: str
    question: str
    student_answer: str
    correct: str
    why: str = ""
    topic: str = ""
    created_at: float = 0.0
    reviews: int = 0
    last_grade: int | None = None

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
            "why": self.why, "topic": self.topic,
            "created_at": self.created_at, "reviews": self.reviews,
            "last_grade": self.last_grade,
        }


def card_from_mistake(question: str, student_answer: str, correct: str,
                      *, why: str = "", topic: str = "") -> MistakeCard:
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
        created_at=time.time(),
    )


class MistakeNotebook:
    """Records mistakes, queues flashcards, reports review status.

    ``scheduler`` is a :class:`RepetitionScheduler` (or None — the notebook
    still records, just without spaced repetition). Never raises: a broken
    scheduler must not break tutoring.
    """

    def __init__(self, scheduler: Any | None = None) -> None:
        self.scheduler = scheduler
        self._cards: dict[str, MistakeCard] = {}

    def record(self, question: str, student_answer: str, correct: str,
               *, why: str = "", topic: str = "") -> MistakeCard:
        """Record a mistake and queue its flashcard. Returns the card."""
        card = card_from_mistake(question, student_answer, correct,
                                 why=why, topic=topic)
        self._cards[card.id] = card
        if self.scheduler is not None:
            try:
                self.scheduler.schedule(card.id, f"{card.front}\n---\n{card.back}")
            except Exception:  # noqa: BLE001
                _log.debug("flashcard scheduling failed", exc_info=True)
        return card

    def review(self, card_id: str, grade: int) -> bool:
        """Grade a flashcard review (1-4, SM-2). Returns False if unknown."""
        card = self._cards.get(card_id)
        if card is None:
            return False
        card.reviews += 1
        card.last_grade = grade
        if self.scheduler is not None:
            try:
                self.scheduler.review(card_id, grade)
            except Exception:  # noqa: BLE001
                _log.debug("flashcard review failed", exc_info=True)
        return True

    def notebook(self) -> list[dict[str, Any]]:
        """All mistakes with their cards and review status, newest first."""
        cards = sorted(self._cards.values(), key=lambda c: -c.created_at)
        out = []
        for card in cards:
            entry = card.to_dict()
            entry["due"] = self._due(card.id)
            out.append(entry)
        return out

    def _due(self, card_id: str) -> bool:
        if self.scheduler is None:
            return False
        try:
            got = self.scheduler.get(card_id)
            return bool(got) and got.due_at <= time.time()
        except Exception:  # noqa: BLE001
            return False

    def count(self) -> int:
        return len(self._cards)
