"""Memory delivery scoring: WHEN to surface a memory, not just WHAT to recall.

The memory manager's recall answers "what is relevant?" This module answers
"is this the right moment?" — the timing and room-reading layer.

A memory can be highly relevant but wrong to mention right now:
- The conversation moved on (topic discontinuity)
- It was just mentioned (repetition)
- The emotional tone doesn't fit (bringing up a sad memory in a happy chat)
- It's a non-sequitur (no natural bridge from current topic)

The DeliveryScorer assigns each candidate memory a delivery score (0.0-1.0)
based on conversational appropriateness. The caller decides the threshold.

This is deliberately heuristic and lightweight — no LLM call needed. It uses:
- Topic overlap between memory content and recent conversation
- Time since the memory was last surfaced (anti-repetition)
- Emotional tone compatibility (simple lexicon-based)
- Memory importance (important memories earn more leeway)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

__all__ = ["DeliveryScorer", "DeliveryScore", "score_delivery"]

_log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


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
    ) -> None:
        self.repeat_cooldown = repeat_cooldown_seconds
        self.weights = {
            "topic": topic_weight,
            "freshness": freshness_weight,
            "tone": tone_weight,
            "importance": importance_weight,
        }
        # memory_id -> last surfaced timestamp
        self._last_surfaced: dict[str, float] = {}

    def score(
        self,
        candidates: list[Any],
        *,
        recent_texts: list[str] | None = None,
        current_text: str = "",
    ) -> list[DeliveryScore]:
        """Score each candidate memory for delivery appropriateness."""
        recent_texts = recent_texts or []
        now = time.time()
        conversation_tokens = _tokens(" ".join(recent_texts + [current_text]))
        conversation_tone = _tone(" ".join(recent_texts + [current_text]))

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

            score = (
                self.weights["topic"] * topic_fit
                + self.weights["freshness"] * freshness
                + self.weights["tone"] * tone_fit
                + self.weights["importance"] * importance_boost
            )

            results.append(DeliveryScore(
                memory_id=mem_id,
                score=min(1.0, max(0.0, score)),
                reasons=reasons,
                topic_fit=topic_fit,
                freshness=freshness,
                tone_fit=tone_fit,
                importance_boost=importance_boost,
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
) -> list[DeliveryScore]:
    """One-shot convenience: score without managing a scorer instance."""
    return DeliveryScorer().score(
        candidates, recent_texts=recent_texts, current_text=current_text
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
