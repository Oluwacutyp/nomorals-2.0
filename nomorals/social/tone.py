"""Per-platform tone adaptation for cross-network publishing.

One draft rarely fits everywhere: LinkedIn wants formal paragraphs, X wants
punch, Threads wants casual. ``adapt_tone`` rewrites a draft per platform —
with an LLM when one is available, with honest rule-based shaping otherwise.

The rule-based fallback is exactly that: length trims, hashtag handling, and
light formatting. It never invents content or claims an LLM wrote it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger
from .adapters.postiz import Adapter as PostizAdapter
from .base import Account, PostResult

__all__ = ["PLATFORM_PROFILES", "PlatformProfile", "adapt_tone", "Publisher"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class PlatformProfile:
    """Voice + hard limits for one network."""
    name: str
    max_chars: int
    hashtags: str  # "keep" | "trim" | "strip"
    style: str     # short human description used by the LLM path


#: Conservative real-world limits. Postiz enforces the true ones; these keep
#: the rule-based fallback honest.
PLATFORM_PROFILES: dict[str, PlatformProfile] = {
    "x": PlatformProfile("x", 280, "keep",
                         "punchy, direct, one sharp idea, no corporate fluff"),
    "twitter": PlatformProfile("x", 280, "keep",
                               "punchy, direct, one sharp idea, no corporate fluff"),
    "threads": PlatformProfile("threads", 500, "keep",
                               "casual, conversational, lowercase-friendly"),
    "linkedin": PlatformProfile("linkedin", 3000, "strip",
                                "formal, professional, clear paragraphs, no slang"),
    "instagram": PlatformProfile("instagram", 2200, "keep",
                                 "visual-first caption, line breaks, emojis welcome"),
    "facebook": PlatformProfile("facebook", 2000, "trim",
                                "warm, conversational, community tone"),
    "bluesky": PlatformProfile("bluesky", 300, "keep",
                               "witty, internet-native, concise"),
    "mastodon": PlatformProfile("mastodon", 500, "keep",
                                "thoughtful, unhurried, no growth-hacking voice"),
    "tiktok": PlatformProfile("tiktok", 150, "keep",
                              "hook-first caption, casual, trend-aware"),
    "youtube": PlatformProfile("youtube", 5000, "trim",
                               "clear description, timestamps welcome, SEO-aware"),
    "discord": PlatformProfile("discord", 2000, "strip",
                               "plain, direct, announcement-style"),
    "telegram": PlatformProfile("telegram", 4096, "strip",
                                "plain, direct, announcement-style"),
    "pinterest": PlatformProfile("pinterest", 500, "keep",
                                 "descriptive, keyword-rich, helpful"),
}


def _profile(platform: str) -> PlatformProfile:
    key = (platform or "").strip().lower()
    return PLATFORM_PROFILES.get(key, PlatformProfile(key or "generic", 2000, "trim",
                                                      "clear and natural"))


def adapt_tone(
    text: str,
    platform: str,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """Rewrite ``text`` for ``platform``'s voice.

    With ``llm_fn``: a real style rewrite via the caller's model.
    Without: rule-based shaping (length trim at word boundaries, hashtag
    handling, light formatting). The fallback is documented, not disguised.
    """
    profile = _profile(platform)
    text = (text or "").strip()
    if not text:
        return text
    if llm_fn is not None:
        prompt = (
            f"Rewrite the following post for {profile.name}. "
            f"Voice: {profile.style}. "
            f"Keep every fact identical — change style only, never add claims. "
            f"Max {profile.max_chars} characters.\n\n{text}"
        )
        try:
            rewritten = (llm_fn(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001 - LLM failure → fallback
            _log.warning("tone llm failed for %s, using rules: %s", platform, exc)
            rewritten = ""
        if rewritten:
            return _hard_trim(rewritten, profile.max_chars)
        # fall through to rules on empty/exception
    return _rule_shape(text, profile)


def _hard_trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 1)].rsplit(" ", 1)[0] or text[: max(0, limit - 1)]
    return cut.rstrip() + "…"


_HASHTAG = re.compile(r"#\w+")


def _rule_shape(text: str, profile: PlatformProfile) -> str:
    if profile.hashtags == "strip":
        text = _HASHTAG.sub("", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
    elif profile.hashtags == "trim":
        tags = _HASHTAG.findall(text)
        text = _HASHTAG.sub("", text).strip()
        kept = " ".join(tags[:3])
        if kept:
            text = f"{text}\n\n{kept}".strip()
    if profile.name == "linkedin":
        # Light professional formatting: collapse runs, keep paragraphs.
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        text = "\n\n".join(paras) if paras else text
    return _hard_trim(text, profile.max_chars)


@dataclass
class ChannelSpec:
    """One Postiz channel to publish to."""
    integration_id: str
    platform: str                      # e.g. "linkedin", "x" — for tone
    settings: dict[str, Any] | None = None  # platform-specific (see postiz.py)


class Publisher:
    """One draft → per-platform adapted posts → Postiz.

    Usage::

        pub = Publisher(adapter, account)
        results = pub.publish_adapted(
            "We just shipped v2 …",
            [ChannelSpec("abc123", "linkedin"), ChannelSpec("def456", "x")],
        )
    """

    def __init__(self, adapter: PostizAdapter, account: Account) -> None:
        self.adapter = adapter
        self.account = account

    def adapt_all(
        self,
        draft: str,
        channels: Sequence[ChannelSpec],
        *,
        llm_fn: Callable[[str], str] | None = None,
    ) -> dict[str, str]:
        """→ {integration_id: adapted text}. Pure: no network."""
        return {
            ch.integration_id: adapt_tone(draft, ch.platform, llm_fn=llm_fn)
            for ch in channels
        }

    def publish_adapted(
        self,
        draft: str,
        channels: Sequence[ChannelSpec],
        *,
        media_paths: Sequence[str] = (),
        schedule_at: Any = None,
        llm_fn: Callable[[str], str] | None = None,
        short_link: bool = False,
    ) -> dict[str, PostResult]:
        """Adapt per platform and publish in ONE Postiz call.

        Returns {integration_id: PostResult}. Per-channel failures are in
        each result's ``error`` and in ``metrics["channels"]`` — never
        silent, never a fake success.
        """
        channels = list(channels)
        if not channels:
            raise ValueError("publish_adapted needs at least one channel")
        variants = self.adapt_all(draft, channels, llm_fn=llm_fn)
        settings = {ch.integration_id: dict(ch.settings or {}) for ch in channels}
        combined = self.adapter.post(
            self.account,
            draft,  # fallback content; per-channel variants override it
            integration_ids=[ch.integration_id for ch in channels],
            variants=variants,
            settings=settings,
            media_paths=list(media_paths),
            schedule_at=schedule_at,
            short_link=short_link,
        )
        per_channel = {
            c.get("integration_id", ""): c
            for c in combined.metrics.get("channels", [])
            if isinstance(c, dict)
        }
        out: dict[str, PostResult] = {}
        for ch in channels:
            info = per_channel.get(ch.integration_id, {})
            ok = bool(info.get("ok", combined.ok))
            out[ch.integration_id] = PostResult(
                platform=f"postiz:{ch.platform}",
                ok=ok,
                external_id=str(info.get("external_id", "") or combined.external_id),
                url=combined.url,
                error=str(info.get("error", "")),
                status_code=combined.status_code,
                seconds=combined.seconds,
                metrics={"adapted_text": variants[ch.integration_id]},
            )
        return out
