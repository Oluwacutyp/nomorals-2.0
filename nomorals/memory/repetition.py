"""Spaced-repetition resurfacing for memories (build-map #40).

Readwise-style: memories resurface on a forgetting curve, not just recency.

Scheduling is FSRS-4.5 (Free Spaced Repetition Scheduler) — the DSR model
(Difficulty, Stability, Retrievability) that Anki has used by default since
2023 and that beat SM-2's heuristic ease factors on every published
benchmark. The old SM-2 path is kept only as a legacy migration: cards that
pre-date FSRS get D/S state synthesized from their SM-2 fields on first
review. New cards are always FSRS.

This is the WHEN for long-horizon resurfacing. ``delivery.py``'s
DeliveryScorer handles conversational timing; this module handles the
forgetting curve. The morning briefing's "from your past" section reads
:meth:`RepetitionScheduler.due`.

Every public method never raises — a broken scheduler must never break
chat or the briefing.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "ReviewCard",
    "FSCard",
    "FSRS",
    "RepetitionScheduler",
    "repetition_db_path",
    "retrievability",
    "next_interval_days",
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

#: SM-2 legacy constants (migration only).
_START_EASE = 2.5
_MIN_EASE = 1.3
_FIRST_GOOD_INTERVAL = 1.0   # days
_SECOND_GOOD_INTERVAL = 6.0  # days


# ── FSRS-4.5 ──────────────────────────────────────────────────────────────
# The DSR (Difficulty / Stability / Retrievability) model from the FSRS
# project (open-spaced-repetition/fsrs-rs, MIT). Default weights are the
# FSRS-4.5 defaults; they are tunable but the defaults are fit on ~700M
# real reviews, so "tune later" is the honest default.

#: Default FSRS-4.5 weights (w0..w16).
FSRS_WEIGHTS: tuple[float, ...] = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
    1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729,
)

#: Forgetting-curve constants: R(t, S) = (1 + FACTOR * t / S) ^ DECAY
_FSRS_DECAY = -0.5
_FSRS_FACTOR = 19.0 / 81.0

#: Difficulty bounds.
_FSRS_D_MIN, _FSRS_D_MAX = 1.0, 10.0


def retrievability(elapsed_days: float, stability: float) -> float:
    """Probability of recall after ``elapsed_days`` given ``stability``.

    The FSRS forgetting curve. SM-2 cannot compute this at all — here it is
    the primitive that lets ``due()`` sort by "closest to being forgotten".
    """
    if stability <= 0.0:
        return 1.0 if elapsed_days <= 0.0 else 0.0
    if elapsed_days <= 0.0:
        return 1.0
    return (1.0 + _FSRS_FACTOR * elapsed_days / stability) ** _FSRS_DECAY


def next_interval_days(stability: float, desired_retention: float) -> float:
    """Days until retrievability decays to ``desired_retention``.

    Retention is a *setting*, not an outcome — the scheduler solves for the
    interval. SM-2 cannot even express this question.
    """
    r = max(0.5, min(0.99, desired_retention))
    if stability <= 0.0:
        return 1.0
    return (stability / _FSRS_FACTOR) * (r ** (1.0 / _FSRS_DECAY) - 1.0)


class FSRS:
    """The FSRS-4.5 scheduler: state transitions for one review.

    Pure math — no storage, no I/O, fully testable. The
    :class:`RepetitionScheduler` drives it against SQLite.
    """

    def __init__(self, weights: tuple[float, ...] | None = None,
                 desired_retention: float = 0.90) -> None:
        self.w = tuple(weights) if weights else FSRS_WEIGHTS
        if len(self.w) < 17:
            raise ValueError("FSRS needs 17 weights (w0..w16)")
        self.desired_retention = max(0.5, min(0.99, desired_retention))

    # -- state init --------------------------------------------------------

    def init_stability(self, grade: int) -> float:
        """S₀(G): stability of a never-seen card after its first review."""
        return max(0.01, self.w[grade - 1])

    def init_difficulty(self, grade: int) -> float:
        """D₀(G): difficulty of a never-seen card after its first review."""
        return self._clamp_d(self.w[4] - (grade - 3) * self.w[5])

    @staticmethod
    def _clamp_d(d: float) -> float:
        return max(_FSRS_D_MIN, min(_FSRS_D_MAX, d))

    # -- transitions -------------------------------------------------------

    def next_difficulty(self, difficulty: float, grade: int) -> float:
        """Linear-damped difficulty update."""
        delta = -self.w[6] * (grade - 3)
        nxt = difficulty + delta * (10.0 - difficulty) / 9.0
        reverted = self.w[7] * self.init_difficulty(4) + (1.0 - self.w[7]) * nxt
        return self._clamp_d(reverted)

    def stability_after_recall(self, stability: float, difficulty: float,
                               retr: float, grade: int) -> float:
        """S′ after a successful review (grade ≥ 2)."""
        w = self.w
        hard = w[15] if grade == GRADE_HARD else 1.0
        easy = w[16] if grade == GRADE_EASY else 1.0
        inc = (math.exp(w[8]) * (11.0 - difficulty)
               * (stability ** -w[9])
               * (math.exp((1.0 - retr) * w[10]) - 1.0)
               * hard * easy)
        return max(0.01, stability * (1.0 + inc))

    def stability_after_failure(self, stability: float, difficulty: float,
                                retr: float) -> float:
        """S′ after a lapse (grade = 1)."""
        w = self.w
        return max(0.01,
                   w[11] * (difficulty ** -w[12])
                   * ((stability + 1.0) ** w[13] - 1.0)
                   * math.exp((1.0 - retr) * w[14]))

    def review(self, difficulty: float, stability: float,
               elapsed_days: float, grade: int) -> tuple[float, float, float]:
        """One review → (new_difficulty, new_stability, next_interval_days).

        ``elapsed_days`` is time since the last review; ``grade`` is
        1=again 2=hard 3=good 4=easy. Never raises on sane input.
        """
        grade = max(1, min(4, int(grade)))
        retr = retrievability(elapsed_days, stability)
        new_d = self.next_difficulty(difficulty, grade)
        if grade == GRADE_AGAIN:
            new_s = self.stability_after_failure(stability, new_d, retr)
        else:
            new_s = self.stability_after_recall(stability, new_d, retr, grade)
        return new_d, new_s, max(0.25, next_interval_days(
            new_s, self.desired_retention))


@dataclass
class FSCard:
    """FSRS view of a review card: Difficulty / Stability / Retrievability."""

    memory_id: str
    text: str
    difficulty: float = 5.0
    stability: float = 0.0
    interval_days: float = 0.0
    due_at: float = field(default_factory=time.time)
    reps: int = 0
    lapses: int = 0
    retrievability_now: float = 1.0
    last_reviewed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "difficulty": round(self.difficulty, 3),
            "stability": round(self.stability, 3),
            "interval_days": round(self.interval_days, 2),
            "due_at": self.due_at,
            "reps": self.reps,
            "lapses": self.lapses,
            "retrievability": round(self.retrievability_now, 4),
        }


def repetition_db_path(settings: Any = None) -> Path:
    """Home-dir path for the repetition store, honoring runtime settings."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home:
            return Path(home) / "memory" / "repetition.db"
    return Path.home() / ".nomorals" / "memory" / "repetition.db"


@dataclass
class ReviewCard:
    """One memory on the forgetting curve.

    ``ease`` is the legacy SM-2 field (kept for migration reads).
    ``difficulty``/``stability`` are the FSRS state; a card whose stability
    is 0 has not been FSRS-reviewed yet and will be migrated on first
    review.
    """

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
    difficulty: float = 0.0
    stability: float = 0.0
    algorithm: str = "sm2"

    @property
    def is_fsrs(self) -> bool:
        """True once this card carries real FSRS state."""
        return self.algorithm == "fsrs" and self.stability > 0.0

    def to_fs(self, *, fsrs: "FSRS", now: float | None = None) -> FSCard:
        """FSRS view: difficulty/stability/retrievability right now."""
        ts = now or time.time()
        if self.is_fsrs and self.last_reviewed > 0:
            retr = retrievability((ts - self.last_reviewed) / DAY,
                                  self.stability)
        else:
            retr = 1.0
        return FSCard(
            memory_id=self.memory_id, text=self.text,
            difficulty=self.difficulty or 5.0,
            stability=self.stability, interval_days=self.interval_days,
            due_at=self.due_at, reps=self.reps, lapses=self.lapses,
            retrievability_now=retr, last_reviewed=self.last_reviewed)


class RepetitionScheduler:
    """FSRS scheduling for memory resurfacing. SQLite-backed, never raises.

    Usage:
        sched = RepetitionScheduler(desired_retention=0.90)
        sched.schedule("fact-123", "my girlfriend's name is Ada")
        for card in sched.due(limit=3):   # briefing reads these
            ...
        sched.review("fact-123", GRADE_GOOD)  # after the user engages
        sched.bury("fact-123")                # "I know this" → long interval
        sched.retrievability_of("fact-123")   # recall probability right now
    """

    def __init__(self, db: Any = None, *,
                 desired_retention: float = 0.90,
                 weights: tuple[float, ...] | None = None) -> None:
        self.fsrs = FSRS(weights=weights, desired_retention=desired_retention)
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
                # _row_to_card needs named access; the caller may not have
                # set a row factory.
                db.row_factory = sqlite3.Row
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
        # FSRS columns — migrate in place; existing SM-2 cards keep working.
        for col, ddl in (("difficulty", "REAL NOT NULL DEFAULT 0"),
                         ("stability", "REAL NOT NULL DEFAULT 0"),
                         ("algorithm", "TEXT NOT NULL DEFAULT 'sm2'")):
            try:
                self._db.execute(
                    f"ALTER TABLE review_cards ADD COLUMN {col} {ddl}")
            except Exception:  # noqa: BLE001 — column already exists
                pass
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
        """Record a review outcome; update FSRS state, interval, due.

        Grades: 1=again, 2=hard, 3=good, 4=easy. Returns the updated card,
        or None if unknown/broken. Legacy SM-2 cards are migrated to FSRS
        on first review (difficulty/stability synthesized from the SM-2
        ease/interval, so history is not thrown away).
        """
        if self._broken() or grade not in (1, 2, 3, 4):
            return None
        try:
            card = self.get(memory_id)
            if card is None:
                return None
            now = time.time()

            if card.is_fsrs and card.last_reviewed > 0:
                elapsed = (now - card.last_reviewed) / DAY
                new_d, new_s, interval = self.fsrs.review(
                    card.difficulty or 5.0, card.stability, elapsed, grade)
            elif card.reps > 0 or card.interval_days > 0:
                # Legacy SM-2 card: migrate — map the SM-2 ease/interval
                # onto FSRS state, then apply the review to it.
                difficulty, stability = self._migrate_sm2(card)
                new_d, new_s, interval = self.fsrs.review(
                    difficulty, stability,
                    max(0.0, (now - card.last_reviewed) / DAY), grade)
            else:
                # Brand-new card: FSRS init from the first grade.
                new_d = self.fsrs.init_difficulty(grade)
                new_s = self.fsrs.init_stability(grade)
                interval = max(0.25, next_interval_days(
                    new_s, self.fsrs.desired_retention))

            if grade == GRADE_AGAIN:
                card.lapses += 1
            else:
                card.reps += 1

            card.difficulty = new_d
            card.stability = new_s
            card.algorithm = "fsrs"
            card.interval_days = interval
            card.due_at = now + interval * DAY
            card.last_reviewed = now
            # Keep ease loosely in sync for legacy readers of the column.
            card.ease = max(_MIN_EASE, min(3.0, 2.5 + (5.0 - new_d) * 0.15))
            self._write(card)
            return card
        except Exception:  # noqa: BLE001
            _log.debug("repetition review failed", exc_info=True)
            return None

    @staticmethod
    def _migrate_sm2(card: ReviewCard) -> tuple[float, float]:
        """Synthesize FSRS (difficulty, stability) from SM-2 state.

        Ease 2.5 → difficulty 5.0; lower ease (struggled) → higher
        difficulty. Interval maps straight onto stability — an SM-2
        interval is already a decent stability estimate.
        """
        difficulty = max(_FSRS_D_MIN, min(_FSRS_D_MAX,
                                         5.0 + (2.5 - card.ease) * 2.0))
        stability = max(1.0, card.interval_days or 1.0)
        return difficulty, stability

    # -- FSRS read path -----------------------------------------------------

    def retrievability_of(self, memory_id: str,
                          now: float | None = None) -> float | None:
        """Recall probability of a card right now (FSRS R(t, S)).

        None when the card is unknown or the store is broken. SM-2 could
        never answer this — it's the new primitive that powers forgetting
        -aware due ordering.
        """
        if self._broken():
            return None
        try:
            card = self.get(memory_id)
            if card is None:
                return None
            return card.to_fs(fsrs=self.fsrs, now=now).retrievability_now
        except Exception:  # noqa: BLE001
            return None

    def fs_card(self, memory_id: str) -> FSCard | None:
        """The FSRS view of a card (D/S/R), or None."""
        if self._broken():
            return None
        try:
            card = self.get(memory_id)
            return card.to_fs(fsrs=self.fsrs) if card else None
        except Exception:  # noqa: BLE001
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

    def due_by_forgetting(self, limit: int = 5,
                          now: float | None = None) -> list[FSCard]:
        """Cards closest to being forgotten first, regardless of due date.

        The FSRS way: being late is information, and recall probability is
        a computable quantity. A card at R=0.55 that is "not due yet" is a
        better resurfacing candidate than one at R=0.95. Used by the
        morning briefing's "from your past" section for the highest-value
        reminders.
        """
        if self._broken() or limit <= 0:
            return []
        try:
            ts = now or time.time()
            rows = self._db.execute(  # type: ignore[union-attr]
                """SELECT * FROM review_cards
                   WHERE buried_until <= ? AND last_reviewed > 0
                   ORDER BY due_at ASC LIMIT ?""",
                (ts, max(limit * 4, 20))).fetchall()
            cards = [self._row_to_card(r).to_fs(fsrs=self.fsrs, now=ts)
                     for r in rows]
            cards.sort(key=lambda c: c.retrievability_now)
            return cards[:limit]
        except Exception:  # noqa: BLE001
            _log.debug("repetition due_by_forgetting failed", exc_info=True)
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
               due_at=?, reps=?, lapses=?, buried_until=?, last_reviewed=?,
               difficulty=?, stability=?, algorithm=?
               WHERE memory_id=?""",
            (card.text, card.ease, card.interval_days, card.due_at,
             card.reps, card.lapses, card.buried_until, card.last_reviewed,
             card.difficulty, card.stability, card.algorithm,
             card.memory_id))
        self._db.commit()

    @staticmethod
    def _row_to_card(row: sqlite3.Row) -> ReviewCard:
        keys = set(row.keys())
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
            difficulty=float(row["difficulty"]) if "difficulty" in keys else 0.0,
            stability=float(row["stability"]) if "stability" in keys else 0.0,
            algorithm=str(row["algorithm"]) if "algorithm" in keys else "sm2",
        )
