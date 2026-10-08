"""Memory delivery scoring: WHEN to surface a memory, not just WHAT to recall.

The memory manager's recall answers "what is relevant?" This module answers
"is this the right moment?" — the timing and room-reading layer.

A memory can be highly relevant but wrong to mention right now:
- The conversation moved on (topic discontinuity)
- It was just mentioned (repetition)
- The emotional tone doesn't fit (bringing up a sad memory in a happy chat)
- It's a non-sequitur (no natural bridge from current topic)
- The user already knows it (no novelty — see KnowledgeState)

The DeliveryScorer assigns each candidate memory a delivery score (0.0-1.0)
based on conversational appropriateness. The caller decides the threshold.

This is deliberately heuristic and lightweight — no LLM call needed. It uses:
- Topic overlap between memory content and recent conversation
- Time since the memory was last surfaced (anti-repetition)
- Emotional tone compatibility (simple lexicon-based)
- Memory importance (important memories earn more leeway)
- Novelty vs the user's knowledge state (build-map #40)
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "DeliveryScorer", "DeliveryScore", "KnowledgeState", "score_delivery",
    "knowledge_state_db_path",
]

_log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

#: Jaccard similarity at or above this → the user already knows this text.
_KNOWS_FUZZY_THRESHOLD = 0.8

#: How many recent briefings count as "don't resurface".
_RECENT_BRIEFINGS_N = 3


# ── tone lexicons (tiny, deliberate — not a sentiment model) ──────────────────

_POSITIVE = frozenset({
    "happy", "great", "awesome", "love", "excited", "fun", "good", "nice",
    "wonderful", "amazing", "best", "win", "won", "celebrat", "congrat",
    "laugh", "smile", "joy", "joke", "funny",
})

_NEGATIVE = frozenset({
    "sad", "bad", "terrible", "awful", "hate", "angry", "upset", "worried",
    "anxious", "scared", "afraid", "cry", "crying", "death", "died", "loss",
    "lost", "fail", "failed", "problem", "trouble", "pain", "hurt",
})

_QUESTION_WORDS = frozenset({
    "what", "when", "where", "who", "whom", "whose", "which", "how", "why",
})


@dataclass
class DeliveryScore:
    """How appropriate it is to surface a memory right now."""

    memory_id: str
    score: float  # 0.0-1.0, higher = better moment
    reasons: list[str] = field(default_factory=list)
    # Component scores for debugging/tuning
    topic_fit: float = 0.0
    freshness: float = 0.0
    tone_fit: float = 0.0
    importance_boost: float = 0.0
    novelty: float = 1.0  # 0.0 = user already knows it; 1.0 = genuinely new


def knowledge_state_db_path(settings: Any = None) -> Path:
    """Home-dir path for the knowledge-state store."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home:
            return Path(home) / "memory" / "knowledge_state.db"
    return Path.home() / ".nomorals" / "memory" / "knowledge_state.db"


class KnowledgeState:
    """What the user already knows — the anti-redundancy layer (Petrarca).

    Devon tracks which memories the user has seen/heard so she only
    surfaces what's NEW relative to that. Kills "you already told me this".

    Sources of knowledge:
    - ``mark_known`` — the user demonstrably knows this (they said it,
      confirmed it, or acted on it).
    - ``mark_surfaced`` — Devon already told them (surfaced in chat or a
      briefing).
    - ``mark_briefing`` — a whole briefing's worth of surfacings, so the
      last N briefings don't repeat each other.

    ``knows(text)`` is fuzzy (Jaccard ≥ 0.8 over tokens) because the user
    may know the fact in different words.

    Persistent (SQLite) so briefings days apart still dedupe. Never raises.
    """

    def __init__(self, db: Any = None) -> None:
        self._db: sqlite3.Connection | None = None
        # memory_id -> token set, for fuzzy matching without hitting disk
        self._known_tokens: dict[str, set[str]] = {}
        self._known_ids: set[str] = set()
        try:
            if db is None:
                path = knowledge_state_db_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                db = str(path)
            if isinstance(db, (str, Path)):
                self._db = sqlite3.connect(str(db))
                self._db.row_factory = sqlite3.Row
            elif isinstance(db, sqlite3.Connection):
                self._db = db
            if self._db is not None:
                self._init_schema()
                self._load_known()
        except Exception:  # noqa: BLE001
            _log.debug("knowledge state init failed", exc_info=True)
            self._db = None

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        assert self._db is not None
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS known_memories (
                   memory_id TEXT PRIMARY KEY,
                   text      TEXT NOT NULL,
                   added_at  REAL NOT NULL
               )""")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS surfacings (
                   id         INTEGER PRIMARY KEY AUTOINCREMENT,
                   memory_id  TEXT NOT NULL,
                   text       TEXT NOT NULL DEFAULT '',
                   briefing_id TEXT NOT NULL DEFAULT '',
                   surfaced_at REAL NOT NULL
               )""")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_surfacings_mid "
            "ON surfacings(memory_id, surfaced_at)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_surfacings_brief "
            "ON surfacings(briefing_id, surfaced_at)")
        self._db.commit()

    def _load_known(self) -> None:
        assert self._db is not None
        for row in self._db.execute(
                "SELECT memory_id, text FROM known_memories").fetchall():
            mid = str(row["memory_id"])
            self._known_ids.add(mid)
            self._known_tokens[mid] = _tokens(str(row["text"]))

    def _broken(self) -> bool:
        return self._db is None

    # -- write path -------------------------------------------------------

    def mark_known(self, memory_id: str, text: str) -> bool:
        """The user demonstrably knows this memory."""
        if self._broken() or not memory_id:
            return False
        try:
            self._db.execute(  # type: ignore[union-attr]
                """INSERT INTO known_memories (memory_id, text, added_at)
                   VALUES (?,?,?)
                   ON CONFLICT(memory_id) DO UPDATE SET text=excluded.text""",
                (memory_id, text or "", time.time()))
            self._db.commit()  # type: ignore[union-attr]
            self._known_ids.add(memory_id)
            self._known_tokens[memory_id] = _tokens(text or "")
            return True
        except Exception:  # noqa: BLE001
            return False

    def mark_surfaced(self, memory_id: str, text: str = "",
                       briefing_id: str = "") -> bool:
        """Devon already told the user (chat reply or briefing)."""
        if self._broken() or not memory_id:
            return False
        try:
            self._db.execute(  # type: ignore[union-attr]
                """INSERT INTO surfacings
                   (memory_id, text, briefing_id, surfaced_at)
                   VALUES (?,?,?,?)""",
                (memory_id, text or "", briefing_id or "", time.time()))
            self._db.commit()  # type: ignore[union-attr]
            return True
        except Exception:  # noqa: BLE001
            return False

    def mark_briefing(self, briefing_id: str, memory_ids: list[str]) -> bool:
        """Record everything one briefing surfaced (for the last-N rule)."""
        if self._broken() or not briefing_id:
            return False
        ok = True
        for mid in memory_ids:
            ok = self.mark_surfaced(mid, briefing_id=briefing_id) and ok
        return ok

    # -- read path --------------------------------------------------------

    def knows_id(self, memory_id: str) -> bool:
        """This exact memory is known."""
        return memory_id in self._known_ids

    def knows(self, text: str) -> bool:
        """Fuzzy: the user already knows this content (Jaccard ≥ 0.8)."""
        toks = _tokens(text or "")
        if not toks:
            return False
        for known_toks in self._known_tokens.values():
            if _jaccard(toks, known_toks) >= _KNOWS_FUZZY_THRESHOLD:
                return True
        return False

    def surfaced_count(self, memory_id: str) -> int:
        """How many times Devon has surfaced this memory."""
        if self._broken():
            return 0
        try:
            row = self._db.execute(  # type: ignore[union-attr]
                "SELECT COUNT(*) AS n FROM surfacings WHERE memory_id = ?",
                (memory_id,)).fetchone()
            return int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    def was_in_recent_briefings(self, memory_id: str,
                                n: int = _RECENT_BRIEFINGS_N) -> bool:
        """Was this memory surfaced in any of the last ``n`` briefings?"""
        if self._broken() or n <= 0:
            return False
        try:
            rows = self._db.execute(  # type: ignore[union-attr]
                """SELECT DISTINCT briefing_id FROM surfacings
                   WHERE briefing_id != ''
                   ORDER BY surfaced_at DESC""").fetchall()
            recent = [str(r["briefing_id"]) for r in rows][:n]
            if not recent:
                return False
            placeholders = ",".join("?" for _ in recent)
            row = self._db.execute(  # type: ignore[union-attr]
                f"""SELECT COUNT(*) AS n FROM surfacings
                    WHERE memory_id = ? AND briefing_id IN ({placeholders})""",
                (memory_id, *recent)).fetchone()
            return bool(row and row["n"])
        except Exception:  # noqa: BLE001
            return False

    def known_count(self) -> int:
        return len(self._known_ids)


class DeliveryScorer:
    """Scores recalled memories on conversational appropriateness.

    Usage:
        scorer = DeliveryScorer()
        scored = scorer.score(candidates, context={
            "recent_texts": [...],  # last few conversation turns
            "current_text": "...",   # the message being replied to
        })
        # Filter or rank by scored[i].score
    """

    def __init__(
        self,
        *,
        repeat_cooldown_seconds: float = 3600.0,  # don't re-surface within 1h
        topic_weight: float = 0.4,
        freshness_weight: float = 0.25,
        tone_weight: float = 0.2,
        importance_weight: float = 0.15,
        novelty_weight: float = 0.25,  # only used when a knowledge_state is passed
    ) -> None:
        self.repeat_cooldown = repeat_cooldown_seconds
        self.weights = {
            "topic": topic_weight,
            "freshness": freshness_weight,
            "tone": tone_weight,
            "importance": importance_weight,
        }
        self.novelty_weight = novelty_weight
        # memory_id -> last surfaced timestamp
        self._last_surfaced: dict[str, float] = {}

    def novelty_score(self, memory: Any,
                      knowledge_state: "KnowledgeState | None") -> float:
        """0.0 = the user already knows this (suppress), 1.0 = genuinely new.

        ``None`` knowledge state → 1.0 (fully novel — backward compatible).
        """
        if knowledge_state is None:
            return 1.0
        mem_id = getattr(memory, "id", str(id(memory)))
        content = getattr(memory, "content", "") or ""
        try:
            if knowledge_state.knows_id(mem_id):
                return 0.0
            if knowledge_state.was_in_recent_briefings(mem_id):
                return 0.0
            if knowledge_state.knows(content):
                return 0.0
        except Exception:  # noqa: BLE001 — knowledge check never breaks scoring
            _log.debug("novelty check failed", exc_info=True)
        return 1.0

    def score(
        self,
        candidates: list[Any],
        *,
        recent_texts: list[str] | None = None,
        current_text: str = "",
        knowledge_state: "KnowledgeState | None" = None,
    ) -> list[DeliveryScore]:
        """Score each candidate memory for delivery appropriateness.

        Pass ``knowledge_state`` to downweight memories the user already
        knows (build-map #40). Without it, behavior is exactly the old one.
        """
        recent_texts = recent_texts or []
        now = time.time()
        conversation_tokens = _tokens(" ".join(recent_texts + [current_text]))
        conversation_tone = _tone(" ".join(recent_texts + [current_text]))

        # When a knowledge state is present, carve room for novelty out of
        # the existing weights (they keep their relative proportions).
        novelty_on = knowledge_state is not None
        if novelty_on:
            scale = 1.0 - self.novelty_weight
            weights = {k: v * scale for k, v in self.weights.items()}
        else:
            weights = self.weights

        results: list[DeliveryScore] = []
        for memory in candidates:
            mem_id = getattr(memory, "id", str(id(memory)))
            content = getattr(memory, "content", "") or ""
            importance = float(getattr(memory, "importance", 0.5) or 0.5)

            reasons: list[str] = []

            # 1. Topic fit: lexical overlap with recent conversation.
            mem_tokens = _tokens(content)
            topic_fit = _jaccard(mem_tokens, conversation_tokens)
            # Boost if the memory directly answers the current message.
            if current_text.strip().endswith("?") and topic_fit > 0.1:
                topic_fit = min(1.0, topic_fit + 0.2)
                reasons.append("answers-question")

            # 2. Freshness: penalize recently-surfaced memories.
            last = self._last_surfaced.get(mem_id, 0.0)
            age = now - last
            if age < self.repeat_cooldown:
                freshness = age / self.repeat_cooldown  # 0.0 = just surfaced
                reasons.append("recently-surfaced")
            else:
                freshness = 1.0

            # 3. Tone fit: don't drop sad memories into happy chats.
            mem_tone = _tone(content)
            tone_fit = _tone_compatibility(mem_tone, conversation_tone)
            if tone_fit < 0.5:
                reasons.append("tone-mismatch")

            # 4. Importance: important memories earn leeway.
            importance_boost = importance  # 0.0-1.0 directly

            if novelty_on:
                # 5. Novelty: the user already knows this → suppress.
                novelty = self.novelty_score(memory, knowledge_state)
                if novelty == 0.0:
                    reasons.append("already-known")
                score = (
                    weights["topic"] * topic_fit
                    + weights["freshness"] * freshness
                    + weights["tone"] * tone_fit
                    + weights["importance"] * importance_boost
                    + self.novelty_weight * novelty
                )
            else:
                novelty = 1.0
                score = (
                    weights["topic"] * topic_fit
                    + weights["freshness"] * freshness
                    + weights["tone"] * tone_fit
                    + weights["importance"] * importance_boost
                )

            results.append(DeliveryScore(
                memory_id=mem_id,
                score=min(1.0, max(0.0, score)),
                reasons=reasons,
                topic_fit=topic_fit,
                freshness=freshness,
                tone_fit=tone_fit,
                importance_boost=importance_boost,
                novelty=novelty,
            ))

        results.sort(key=lambda s: -s.score)
        return results

    def mark_surfaced(self, memory_id: str) -> None:
        """Record that a memory was surfaced (for anti-repetition)."""
        self._last_surfaced[memory_id] = time.time()

    def mark_surfaced_many(self, memory_ids: list[str]) -> None:
        now = time.time()
        for mid in memory_ids:
            self._last_surfaced[mid] = now


def score_delivery(
    candidates: list[Any],
    *,
    recent_texts: list[str] | None = None,
    current_text: str = "",
    knowledge_state: "KnowledgeState | None" = None,
) -> list[DeliveryScore]:
    """One-shot convenience: score without managing a scorer instance."""
    return DeliveryScorer().score(
        candidates, recent_texts=recent_texts, current_text=current_text,
        knowledge_state=knowledge_state,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _tone(text: str) -> str:
    """Crude tone classification: positive, negative, or neutral."""
    tokens = _tokens(text)
    pos = len(tokens & _POSITIVE)
    neg = len(tokens & _NEGATIVE)
    if pos > neg and pos > 0:
        return "positive"
    if neg > pos and neg > 0:
        return "negative"
    return "neutral"


def _tone_compatibility(mem_tone: str, conv_tone: str) -> float:
    """How well does the memory's tone fit the conversation's tone?"""
    if mem_tone == "neutral" or conv_tone == "neutral":
        return 1.0  # neutral fits anywhere
    if mem_tone == conv_tone:
        return 1.0  # matching tones fit
    # Opposite tones clash — but not a hard zero, sometimes contrast works.
    return 0.3
