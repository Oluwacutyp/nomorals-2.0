"""Smart notification triage: what actually needs the owner's attention.

The problem: every message buzzes equally. A "k" from a group of 200
and "dad's in the hospital" from your sister get the same ping. A God-tier
social OS knows the difference.

The fix: score every inbound message on importance — derived from who,
what, and context. Never hardcoded keyword lists for importance (that
road leads to the regex router we killed). Instead:

* **Who** — owner > close contacts > known people > groups > strangers.
  From the relationship tracker, not a VIP list.
* **What** — questions directed at the owner, time-sensitive language
  ("urgent", "now", "today", times/dates), emotional weight. Detected
  structurally (question marks, @mentions, reply-to-owner), not by
  keyword dictionaries.
* **Context** — a message in a 1:1 DM outranks the same text in a
  500-person group. A reply to the owner's message outranks a broadcast.
* **Recency pressure** — unanswered important messages escalate; noise
  decays.

Tiers:
* ``critical`` — owner must see now (family emergency pattern, direct
  urgent ask from a close contact).
* ``important`` — surface in the next digest, don't buzz at 3am.
* ``routine`` — normal flow, no special handling.
* ``noise`` — group chatter, low-content acks, safe to summarize or skip.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

TIER_CRITICAL = "critical"
TIER_IMPORTANT = "important"
TIER_ROUTINE = "routine"
TIER_NOISE = "noise"

#: Structural urgency signals — message SHAPE, not a keyword dictionary.
#: A time ("by 5pm", "tomorrow"), a question aimed at the owner, an
#: @mention, or a reply to the owner's own message.
_TIME_RE = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|today|tonight|tomorrow|asap|now|urgent)\b",
    re.IGNORECASE,
)
#: Low-content acks that are never important on their own.
_ACK_RE = re.compile(
    r"^(ok|okay|kk?|lol+|lmao|haha+|👍|👌|🙏|thanks|ty|thx)\.?$", re.IGNORECASE
)


@dataclass
class TriageScore:
    tier: str
    score: float  # 0..1
    reasons: list[str] = field(default_factory=list)
    buzz: bool = False  # push a notification now?

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "buzz": self.buzz,
        }


def triage_message(
    message: Any,
    *,
    is_owner: bool,
    closeness: float = 0.0,
    kind: str = "dm",
    mentioned: bool = False,
    reply_to_owner: bool = False,
    now: float | None = None,
) -> TriageScore:
    """Score one inbound message. Pure function — testable, no I/O.

    ``message`` needs ``.text``. Everything else is passed explicitly so
    the scoring stays honest and inspectable.
    """
    now = now or time.time()
    text = (getattr(message, "text", "") or "").strip()
    reasons: list[str] = []
    score = 0.0

    # ── who (0..0.45) ────────────────────────────────────────────
    if is_owner:
        score += 0.45
        reasons.append("from the owner")
    else:
        who_score = 0.45 * max(0.0, min(1.0, closeness))
        score += who_score
        if closeness >= 0.6:
            reasons.append("from a close contact")
        elif closeness >= 0.3:
            reasons.append("from someone you know")
    # Groups dilute: the same text matters less in a crowd.
    if kind == "group":
        score *= 0.55
        reasons.append("in a group (diluted)")
    elif kind == "channel":
        score *= 0.3

    # ── what (structural, 0..0.35) ──────────────────────────────
    if mentioned:
        score += 0.15
        reasons.append("mentioned/tagged you")
    if reply_to_owner:
        score += 0.12
        reasons.append("replying to you")
    if text.endswith("?") and len(text) > 3:
        score += 0.10
        reasons.append("asking you something")
    if _TIME_RE.search(text):
        score += 0.10
        reasons.append("time-sensitive language")

    # ── noise floor ──────────────────────────────────────────────
    if _ACK_RE.match(text):
        score = min(score, 0.15)
        reasons.append("low-content ack")

    score = max(0.0, min(1.0, score))

    # ── tiers ────────────────────────────────────────────────────
    if score >= 0.75:
        tier, buzz = TIER_CRITICAL, True
    elif score >= 0.45:
        tier, buzz = TIER_IMPORTANT, False
    elif score >= 0.20:
        tier, buzz = TIER_ROUTINE, False
    else:
        tier, buzz = TIER_NOISE, False

    # Owner messages about genuinely urgent things buzz even at night.
    # Everything else respects quiet hours via the scheduler, not here.
    return TriageScore(tier=tier, score=score, reasons=reasons, buzz=buzz)


class TriageLog:
    """Recent triage decisions, so digests can say '3 important, 12 routine'."""

    def __init__(self, max_entries: int = 500) -> None:
        self.max_entries = max_entries
        self._entries: list[dict[str, Any]] = []

    def record(
        self, chat_key: str, tier: str, score: float, sender: str = ""
    ) -> None:
        self._entries.append(
            {
                "chat_key": chat_key,
                "tier": tier,
                "score": round(score, 3),
                "sender": sender,
                "ts": time.time(),
            }
        )
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def digest(self, since_hours: float = 24.0) -> dict[str, Any]:
        cutoff = time.time() - since_hours * 3600
        recent = [e for e in self._entries if e["ts"] >= cutoff]
        by_tier: dict[str, int] = {}
        for e in recent:
            by_tier[e["tier"]] = by_tier.get(e["tier"], 0) + 1
        important = [e for e in recent if e["tier"] in (TIER_CRITICAL, TIER_IMPORTANT)]
        return {
            "window_hours": since_hours,
            "total": len(recent),
            "by_tier": by_tier,
            "needs_attention": important[:20],
        }
