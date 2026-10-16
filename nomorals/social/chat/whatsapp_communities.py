"""WhatsApp Communities: cross-post + announcements with preview + limits.

A community announcement is a broadcast: once posted it reaches every
linked group, and it cannot be unsent from everyone's memory. This
module makes cross-posts deliberate:

* **Preview first.** :func:`plan_crosspost` renders exactly what will
  go to each group (announcement text + per-group prefix) so the owner
  approves the real text, not a summary of it.
* **Rate limits.** A per-minute cap on community posts (default 1/min)
  plus a per-post group cap — a mis-tap cannot blast 40 groups in a
  second. Exceeding either refuses with the numbers, it does not queue
  silently.
* **No fake sends.** The module plans and previews; the caller supplies
  the ``sender`` callable (the Baileys bridge). Tests prove the plan,
  the preview, and the refusal — the send itself belongs to the
  adapter, which already owns ``community_broadcast``.

Invariant: ``plan_crosspost`` returns ``ok=False`` (and sends nothing)
when the rate limit would be exceeded or the target list is empty —
never a partial blast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "CommunityPost",
    "CrosspostPlan",
    "CommunityRateLimiter",
    "plan_crosspost",
    "preview_crosspost",
]

#: Max community posts per minute (a broadcast is expensive attention).
DEFAULT_POSTS_PER_MINUTE = 1
#: Max groups in one cross-post (a community with more needs batching).
DEFAULT_MAX_GROUPS_PER_POST = 20


@dataclass
class CommunityPost:
    """One announcement to cross-post."""
    community: str
    text: str
    groups: list[str] = field(default_factory=list)
    announcement: bool = True  # announcement group vs every subgroup


@dataclass
class CrosspostPlan:
    ok: bool
    reason: str = ""
    deliveries: list[dict[str, str]] = field(default_factory=list)
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason,
                "deliveries": self.deliveries, "preview": self.preview}


class CommunityRateLimiter:
    """Rolling per-minute cap on community posts."""

    def __init__(self, posts_per_minute: int = DEFAULT_POSTS_PER_MINUTE,
                 clock: Callable[[], float] | None = None) -> None:
        self.limit = max(1, int(posts_per_minute))
        self._clock = clock or time.time
        self._events: list[float] = []

    def allow(self) -> bool:
        """True iff a post now stays within the cap (and records it)."""
        now = self._clock()
        cutoff = now - 60.0
        self._events = [t for t in self._events if t >= cutoff]
        if len(self._events) >= self.limit:
            return False
        self._events.append(now)
        return True

    def remaining(self) -> int:
        now = self._clock()
        cutoff = now - 60.0
        self._events = [t for t in self._events if t >= cutoff]
        return max(0, self.limit - len(self._events))


def preview_crosspost(post: CommunityPost) -> str:
    """Exactly what each group will receive — the approval surface."""
    lines = [f"📣 Community: {post.community}",
             f"   mode: {'announcement' if post.announcement else 'every subgroup'}",
             f"   groups ({len(post.groups)}): {', '.join(post.groups) or '(none)'}",
             "   — text —"]
    lines.append(post.text or "(empty)")
    lines.append("   — end —")
    return "\n".join(lines)


def plan_crosspost(post: CommunityPost, *,
                   limiter: CommunityRateLimiter | None = None,
                   max_groups: int = DEFAULT_MAX_GROUPS_PER_POST) -> CrosspostPlan:
    """Plan a cross-post: validate, rate-check, preview. Never sends."""
    if not (post.text or "").strip():
        return CrosspostPlan(ok=False, reason="empty announcement text — nothing to post")
    if not post.groups:
        return CrosspostPlan(ok=False, reason="no target groups — refusing an empty blast")
    if len(post.groups) > max_groups:
        return CrosspostPlan(
            ok=False,
            reason=f"{len(post.groups)} groups exceeds the per-post cap ({max_groups}) — split into batches")
    if limiter is not None and not limiter.allow():
        return CrosspostPlan(
            ok=False,
            reason=f"community rate limit: {limiter.limit} post(s)/min already used — wait a minute")
    deliveries = [{"group": g,
                   "text": (f"📣 [{post.community}] {post.text}" if post.announcement
                            else post.text)} for g in post.groups]
    preview = preview_crosspost(post)
    return CrosspostPlan(ok=True, deliveries=deliveries, preview=preview)
