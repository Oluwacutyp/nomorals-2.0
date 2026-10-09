"""Short-form video publishers — YouTube, TikTok, Instagram/Facebook, X.

One adapted payload per platform, real API uploads where they work
headless, and honest capability gates where they don't. Nothing here
fakes a post: when credentials or a manual step are missing, the
publisher raises :class:`CapabilityUnavailable` with the EXACT manual
step required — never a silent no-op, never fake success.

Capability map (researched 2026-10-09, see module docstrings):

* ``youtube`` — full headless upload with OAuth keys: resumable upload
  (any size), Shorts, thumbnails, scheduled publishing, playlists.
  Costs 1600 quota units per upload out of the default 10,000/day.
* ``tiktok`` — full Direct Post flow headless with keys, BUT TikTok
  forces unaudited apps' posts to private-only (``SELF_ONLY``) and
  caps them at 5 posting users / 24h until the app passes TikTok's
  Content Posting API audit (a manual review step).
* ``meta`` (Instagram Reels + Facebook Page video) — full headless
  flow with keys, BUT Meta fetches the media itself: Instagram Reels
  REQUIRE the video at a public HTTPS URL, and the account must be a
  Business/Creator account. Facebook Page video needs a Page token.
* ``x`` — full chunked-upload flow headless with a user-context token,
  BUT X write APIs require a paid API tier and OAuth 2.0 user context
  (``tweet.write`` + ``media.write``); app-only bearer tokens can't post.

Usage::

    from nomorals.media.contentops.publish import (
        YouTubePublisher, PublishLedger, adapt_title,
    )
    pub = YouTubePublisher(vault, http=http)
    result = pub.publish("clip.mp4", title=adapt_title("youtube", title),
                         description=..., confirmed=True, ledger=ledger)
"""

from __future__ import annotations

from typing import Any

from ....core.errors import NoMoralsError

__all__ = [
    "CapabilityUnavailable",
    "PLATFORM_LIMITS",
    "adapt_title",
    "adapt_description",
    "format_hashtags",
    "YouTubePublisher",
    "TikTokPublisher",
    "MetaPublisher",
    "XPublisher",
    "PublishLedger",
]


class CapabilityUnavailable(NoMoralsError):
    """A platform can't be posted to headless right now.

    Raised instead of faking success. ``manual_step`` is the EXACT
    human/owner step that unblocks this — do it, then retry.
    """

    code = "capability_unavailable"

    def __init__(
        self,
        platform: str,
        message: str,
        *,
        manual_step: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.platform = platform
        self.manual_step = manual_step
        self.details = details or {}


#: Per-platform text limits, researched from the platforms' own docs.
#: YouTube: title 100 chars, description 5000 chars.
#: TikTok: caption (post_info.title) max 2200 chars.
#: Instagram: caption max 2200 chars.
#: X: tweet text max 280 chars.
PLATFORM_LIMITS: dict[str, dict[str, int]] = {
    "youtube": {"title": 100, "description": 5000, "tags": 500},
    "tiktok": {"title": 150, "caption": 2200},
    "instagram": {"caption": 2200},
    "facebook": {"description": 63206},
    "x": {"text": 280},
}


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def format_hashtags(tags: list[str] | tuple[str, ...] | None) -> str:
    """Normalize tags to ``#CamelCase``-style hashtags, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags or []:
        clean = "".join(
            ch for ch in str(tag).strip().lstrip("#") if ch.isalnum()
        )
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append("#" + clean)
    return " ".join(out)


def adapt_title(platform: str, title: str) -> str:
    """Fit a title to a platform's limit (YouTube 100 chars)."""
    limit = PLATFORM_LIMITS.get(platform, {}).get("title")
    if limit is None:
        return (title or "").strip()
    return _truncate(title, limit)


def adapt_description(
    platform: str,
    description: str,
    tags: list[str] | None = None,
    *,
    cta: str = "",
) -> str:
    """Build a per-platform caption/description with hashtag conventions.

    * youtube — description body, hashtags appended at the end.
    * tiktok — caption with hashtags inline at the end (TikTok convention).
    * instagram — caption, breathing room, then the hashtag block
      (the API has no first-comment trick; this is the standard layout).
    * facebook — long-form description, hashtags appended.
    * x — text + hashtags, hard-truncated to 280 chars.
    """
    body = (description or "").strip()
    hashes = format_hashtags(tags)
    if cta:
        body = f"{body}\n\n{cta.strip()}" if body else cta.strip()

    if platform == "instagram":
        text = f"{body}\n.\n.\n.\n{hashes}" if hashes else body
        return _truncate(text, PLATFORM_LIMITS["instagram"]["caption"])
    if platform == "tiktok":
        text = f"{body} {hashes}".strip() if hashes else body
        return _truncate(text, PLATFORM_LIMITS["tiktok"]["caption"])
    if platform == "x":
        limit = PLATFORM_LIMITS["x"]["text"]
        if hashes:
            # Hashtags carry discovery — the body yields to them.
            room = max(0, limit - len(hashes) - 1)
            body = _truncate(body, room)
            return f"{body} {hashes}".strip() if body else hashes
        return _truncate(body, limit)
    if platform == "youtube":
        text = f"{body}\n\n{hashes}".strip() if hashes else body
        return _truncate(text, PLATFORM_LIMITS["youtube"]["description"])
    if platform == "facebook":
        text = f"{body}\n\n{hashes}".strip() if hashes else body
        return _truncate(text, PLATFORM_LIMITS["facebook"]["description"])
    return body


def __getattr__(name: str) -> Any:
    """Lazy imports — keep ``import publish`` cheap and cycle-free."""
    if name == "YouTubePublisher":
        from .youtube import YouTubePublisher

        return YouTubePublisher
    if name == "TikTokPublisher":
        from .tiktok import TikTokPublisher

        return TikTokPublisher
    if name == "MetaPublisher":
        from .meta import MetaPublisher

        return MetaPublisher
    if name == "XPublisher":
        from .x import XPublisher

        return XPublisher
    if name == "PublishLedger":
        from .ledger import PublishLedger

        return PublishLedger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
