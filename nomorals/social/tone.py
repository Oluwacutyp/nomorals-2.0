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

__all__ = [
    "PLATFORM_PROFILES", "PlatformProfile", "adapt_tone", "Publisher",
    "hook_check", "split_thread",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class PlatformProfile:
    """Voice + hard limits for one network."""
    name: str
    max_chars: int
    hashtags: str  # "keep" | "trim" | "strip"
    style: str     # short human description used by the LLM path
    hook_len: int = 140  # chars visible before the fold ("see more")
    cta_style: str = "question"  # "question" | "link" | "none"


#: Conservative real-world limits. Postiz enforces the true ones; these keep
#: the rule-based fallback honest. ``hook_len`` is what the reader sees
#: before the fold — the hook must land inside it (LinkedIn truncates at
#: ~210 chars on mobile before "see more").
PLATFORM_PROFILES: dict[str, PlatformProfile] = {
    "x": PlatformProfile("x", 280, "keep",
                         "punchy, direct, one sharp idea, no corporate fluff",
                         hook_len=120, cta_style="question"),
    "twitter": PlatformProfile("x", 280, "keep",
                               "punchy, direct, one sharp idea, no corporate fluff",
                               hook_len=120, cta_style="question"),
    "threads": PlatformProfile("threads", 500, "keep",
                               "casual, conversational, lowercase-friendly",
                               hook_len=140, cta_style="question"),
    "linkedin": PlatformProfile("linkedin", 3000, "strip",
                                "formal, professional, clear paragraphs, no slang",
                                hook_len=210, cta_style="question"),
    "instagram": PlatformProfile("instagram", 2200, "keep",
                                 "visual-first caption, line breaks, emojis welcome",
                                 hook_len=125, cta_style="question"),
    "facebook": PlatformProfile("facebook", 2000, "trim",
                                "warm, conversational, community tone",
                                hook_len=150, cta_style="question"),
    "bluesky": PlatformProfile("bluesky", 300, "keep",
                               "witty, internet-native, concise",
                               hook_len=130, cta_style="question"),
    "mastodon": PlatformProfile("mastodon", 500, "keep",
                                "thoughtful, unhurried, no growth-hacking voice",
                                hook_len=160, cta_style="none"),
    "tiktok": PlatformProfile("tiktok", 150, "keep",
                              "hook-first caption, casual, trend-aware",
                              hook_len=80, cta_style="none"),
    "youtube": PlatformProfile("youtube", 5000, "trim",
                               "clear description, timestamps welcome, SEO-aware",
                               hook_len=150, cta_style="link"),
    "discord": PlatformProfile("discord", 2000, "strip",
                               "plain, direct, announcement-style",
                               hook_len=200, cta_style="none"),
    "telegram": PlatformProfile("telegram", 4096, "strip",
                                "plain, direct, announcement-style",
                                hook_len=200, cta_style="none"),
    "pinterest": PlatformProfile("pinterest", 500, "keep",
                                 "descriptive, keyword-rich, helpful",
                                 hook_len=100, cta_style="link"),
}


def _profile(platform: str) -> PlatformProfile:
    key = (platform or "").strip().lower()
    return PLATFORM_PROFILES.get(key, PlatformProfile(key or "generic", 2000, "trim",
                                                      "clear and natural"))


def hook_check(text: str, platform: str) -> str | None:
    """Warn when the hook dies below the fold. Returns None when fine.

    The fold: LinkedIn truncates ~210 chars on mobile before "see more".
    If the first line doesn't create a reason to expand, the rest of the
    post is invisible — this flags drafts where the payoff starts too late.
    """
    profile = _profile(platform)
    text = (text or "").strip()
    if not text:
        return None
    hook_zone = text[: profile.hook_len]
    first_line = text.split("\n", 1)[0].strip()
    if len(first_line) < 20:
        return (f"weak hook for {profile.name}: the first line is only "
                f"{len(first_line)} chars — nothing to stop the scroll")
    # The reveal/claim lands after the fold while the hook zone is filler.
    if len(first_line) > profile.hook_len and not any(
        tok in first_line.lower() for tok in ("?", ":", "!")
    ):
        return (f"hook for {profile.name} runs {len(first_line)} chars before "
                f"any punctuation — the fold ({profile.hook_len} chars) "
                f"hides the payoff")
    _filler = re.compile(
        r"^(excited to|thrilled to|happy to|proud to|just wanted to|"
        r"i've been thinking|in today's|in this post)",
        re.IGNORECASE,
    )
    if _filler.match(hook_zone):
        return (f"throat-clearing in the hook zone for {profile.name} — "
                f"the first {profile.hook_len} chars decide whether anyone "
                f"expands")
    return None


def _grapheme_len(text: str) -> int:
    """Best-effort grapheme count (emoji clusters count as ~1)."""
    import unicodedata

    count = 0
    for ch in text:
        # Zero-width joiners and combining marks continue the cluster.
        if ch in ("\u200d", "\ufe0f") or unicodedata.combining(ch):
            continue
        count += 1
    return count


def split_thread(text: str, platform: str, *, numbered: bool = True) -> list[str]:
    """Split long text into thread-safe posts for ``platform``.

    Like atproto.dart's token-aware ``split()``: never cuts a handle,
    link, or #tag in half, and every chunk respects BOTH the platform's
    grapheme limit and its byte budget. Chunks are numbered ``(1/n)`` so
    a reader can follow the thread even when numbering survives posting.
    """
    profile = _profile(platform)
    limit = profile.max_chars
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= limit and _grapheme_len(text) <= limit:
        return [text]

    # Never cut inside a token (URL, @mention, #tag).
    words: list[str] = []
    for word in text.split(" "):
        # A pathological single word longer than the whole limit gets
        # hard-split — nothing else can carry it.
        while len(word) > limit:
            words.append(word[:limit])
            word = word[limit:]
        words.append(word)
    tokens: list[str] = []
    buf: list[str] = []
    for word in words:
        buf.append(word)
        if len(" ".join(buf)) > limit:
            tokens.append(" ".join(buf[:-1]))
            buf = [buf[-1]]
    if buf:
        tokens.append(" ".join(buf))
    # Merge tiny tokens so one long word doesn't produce stub posts.
    chunks: list[str] = []
    cur = ""
    for tok in tokens:
        piece = f"{cur} {tok}".strip() if cur else tok
        if _grapheme_len(piece) <= limit and len(piece.encode("utf-8")) <= 3000:
            cur = piece
        else:
            if cur:
                chunks.append(cur)
            cur = tok
    if cur:
        chunks.append(cur)
    chunks = [c for c in chunks if c] or [text[:limit]]
    if numbered and len(chunks) > 1:
        n = len(chunks)
        chunks = [f"({i + 1}/{n}) {c}" for i, c in enumerate(chunks)]
    return chunks


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
