"""Spaced-repetition resurfacing for memories (build-map #40).

Readwise-style: memories resurface on a forgetting curve, not just recency.
SM-2-simple (not full FSRS — documented, deliberate): ease starts at 2.5,
intervals grow on success, shrink on lapse.

This is the WHEN for long-horizon resurfacing. ``delivery.py``'s
DeliveryScorer handles conversational timing; this module handles the
forgetting curve. The morning briefing's "from your past" section reads
:meth:`RepetitionScheduler.due`.

Every public method never raises — a broken scheduler must never break
chat or the briefing.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "ReviewCard",
    "RepetitionScheduler",
    "repetition_db_path",
    "GRADE_AGAIN",
    "GRADE_HARD",
    "GRADE_GOOD",
    "GRADE_EASY",
]

_log = get_logger(__name__)

DAY = 86400.0

#: Review grades (Anki-style).
GRADE_AGAIN = 1
GRADE_HARD = 2
GRADE_GOOD = 3
GRADE_EASY = 4

#: SM-2 constants.
_START_EASE = 2.5
_MIN_EASE = 1.3
_FIRST_GOOD_INTERVAL = 1.0   # days
_SECOND_GOOD_INTERVAL = 6.0  # days


def repetition_db_path(settings: Any = None) -> Path:
    """Home-dir path for the repetition store, honoring runtime settings."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home:
            return Path(home) / "memory" / "repetition.db"
    return Path.home() / ".nomorals" / "memory" / "repetition.db"


@dataclass
class ReviewCard:
    """One memory on the forgetting curve."""

    memory_id: str
    text: str
    ease: float = _START_EASE
    interval_days: float = 0.0
    due_at: float = field(default_factory=time.time)
    reps: int = 0
    lapses: int = 0
    buried_until: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_reviewed: float = 0.0


class RepetitionScheduler:
    """SM-2 scheduling for memory resurfacing. SQLite-backed, never raises.

    Usage:
        sched = RepetitionScheduler()
        sched.schedule("fact-123", "my girlfriend's name is Ada")
        for card in sched.due(limit=3):   # briefing reads these
            ...
        sched.review("fact-123", GRADE_GOOD)  # after the user engages
        sched.bury("fact-123")                # "I know this" → long interval
    """

    def __init__(self, db: Any = None) -> None:
        self._db: sqlite3.Connection | None = None
        try:
            if db is None:
                path = repetition_db_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                db = str(path)
            if isinstance(db, (str, Path)):
                self._db = sqlite3.connect(str(db))
                self._db.row_factory = sqlite3.Row
            elif isinstance(db, sqlite3.Connection):
                self._db = db
            if self._db is not None:
                self._init_schema()
        except Exception:  # noqa: BLE001 — broken store ≠ broken briefing
            _log.debug("repetition scheduler init failed", exc_info=True)
            self._db = None

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        assert self._db is not None
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS review_cards (
                   memory_id    TEXT PRIMARY KEY,
                   text         TEXT NOT NULL,
                   ease         REAL NOT NULL DEFAULT 2.5,
                   interval_days REAL NOT NULL DEFAULT 0,
                   due_at       REAL NOT NULL,
                   reps         INTEGER NOT NULL DEFAULT 0,
                   lapses       INTEGER NOT NULL DEFAULT 0,
                   buried_until REAL NOT NULL DEFAULT 0,
                   created_at   REAL NOT NULL,
                   last_reviewed REAL NOT NULL DEFAULT 0
               )""")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_cards_due "
            "ON review_cards(due_at, buried_until)")
        self._db.commit()

    def _broken(self) -> bool:
        return self._db is None

    # -- write path -------------------------------------------------------

    def schedule(self, memory_id: str, text: str) -> bool:
        """Put a memory on the curve (due immediately). Idempotent.

        Returns True on success, False if the store is broken.
        """
        if self._broken() or not memory_id or not (text or "").strip():
            return False
        try:
            now = time.time()
            self._db.execute(  # type: ignore[union-attr]
                """INSERT INTO review_cards
                   (memory_id, text, ease, interval_days, due_at, reps,
                    lapses, buried_until, created_at, last_reviewed)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(memory_id) DO UPDATE SET text=excluded.text""",
                (memory_id, text.strip(), _START_EASE, 0.0, now,
                 0, 0, 0.0, now, 0.0))
            self._db.commit()  # type: ignore[union-attr]
            return True
        except Exception:  # noqa: BLE001
            _log.debug("repetition schedule failed", exc_info=True)
            return False

    def review(self, memory_id: str, grade: int) -> ReviewCard | None:
        """Record a review outcome; update ease/interval/due (SM-2).

        Grades: 1=again, 2=hard, 3=good, 4=easy. Returns the updated card,
        or None if unknown/broken.
        """
        if self._broken() or grade not in (1, 2, 3, 4):
            return None
        try:
            card = self.get(memory_id)
            if card is None:
                return None
            now = time.time()

            # Ease moves with performance (SM-2 formula).
            card.ease += 0.1 - (4 - grade) * (0.08 + (4 - grade) * 0.02)
            card.ease = max(_MIN_EASE, card.ease)

            if grade == GRADE_AGAIN:
                card.lapses += 1
                card.reps = 0
                card.interval_days = _FIRST_GOOD_INTERVAL  # relearn tomorrow
            elif grade == GRADE_HARD:
                card.interval_days = max(1.0, card.interval_days * 1.2)
                card.reps += 1
            elif grade == GRADE_GOOD:
                if card.reps == 0:
                    card.interval_days = _FIRST_GOOD_INTERVAL
                elif card.reps == 1:
                    card.interval_days = _SECOND_GOOD_INTERVAL
                else:
                    card.interval_days *= card.ease
                card.reps += 1
            else:  # GRADE_EASY
                if card.reps <= 1:
                    card.interval_days = max(4.0, card.interval_days)
                else:
                    card.interval_days *= card.ease * 1.3
                card.reps += 1

            card.due_at = now + card.interval_days * DAY
            card.last_reviewed = now
            self._write(card)
            return card
        except Exception:  # noqa: BLE001
            _log.debug("repetition review failed", exc_info=True)
            return None

    def bury(self, memory_id: str, days: float = 30.0) -> bool:
        """The user said "I know this" — push far down the curve."""
        if self._broken():
            return False
        try:
            card = self.get(memory_id)
            if card is None:
                return False
            card.buried_until = time.time() + max(days, 1.0) * DAY
            self._write(card)
            return True
        except Exception:  # noqa: BLE001
            _log.debug("repetition bury failed", exc_info=True)
            return False

    def remove(self, memory_id: str) -> bool:
        """Drop a card entirely (memory deleted upstream)."""
        if self._broken():
            return False
        try:
            self._db.execute(  # type: ignore[union-attr]
                "DELETE FROM review_cards WHERE memory_id = ?", (memory_id,))
            self._db.commit()  # type: ignore[union-attr]
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- read path --------------------------------------------------------

    def get(self, memory_id: str) -> ReviewCard | None:
        """Fetch one card, or None."""
        if self._broken():
            return None
        try:
            row = self._db.execute(  # type: ignore[union-attr]
                "SELECT * FROM review_cards WHERE memory_id = ?",
                (memory_id,)).fetchone()
            return self._row_to_card(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def due(self, limit: int = 5) -> list[ReviewCard]:
        """Cards due for resurfacing now, oldest-due first. Never raises."""
        if self._broken() or limit <= 0:
            return []
        try:
            now = time.time()
            rows = self._db.execute(  # type: ignore[union-attr]
                """SELECT * FROM review_cards
                   WHERE due_at <= ? AND buried_until <= ?
                   ORDER BY due_at ASC LIMIT ?""",
                (now, now, limit)).fetchall()
            return [self._row_to_card(r) for r in rows]
        except Exception:  # noqa: BLE001
            _log.debug("repetition due failed", exc_info=True)
            return []

    def count(self) -> int:
        """Total cards on the curve."""
        if self._broken():
            return 0
        try:
            row = self._db.execute(  # type: ignore[union-attr]
                "SELECT COUNT(*) AS n FROM review_cards").fetchone()
            return int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # -- internals --------------------------------------------------------

    def _write(self, card: ReviewCard) -> None:
        assert self._db is not None
        self._db.execute(
            """UPDATE review_cards SET text=?, ease=?, interval_days=?,
               due_at=?, reps=?, lapses=?, buried_until=?, last_reviewed=?
               WHERE memory_id=?""",
            (card.text, card.ease, card.interval_days, card.due_at,
             card.reps, card.lapses, card.buried_until, card.last_reviewed,
             card.memory_id))
        self._db.commit()

    @staticmethod
    def _row_to_card(row: sqlite3.Row) -> ReviewCard:
        return ReviewCard(
            memory_id=str(row["memory_id"]),
            text=str(row["text"]),
            ease=float(row["ease"]),
            interval_days=float(row["interval_days"]),
            due_at=float(row["due_at"]),
            reps=int(row["reps"]),
            lapses=int(row["lapses"]),
            buried_until=float(row["buried_until"]),
            created_at=float(row["created_at"]),
            last_reviewed=float(row["last_reviewed"]),
        )
