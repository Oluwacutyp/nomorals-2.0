"""Per-platform transmission limits — the descriptor table.

First adopter of the descriptor rule (``docs/DESCRIPTOR_RULE.md``):
a new capability of kind X is a new descriptor, not a new code path.
Adding a platform is adding one :class:`PlatformLimits` row; the gateway,
the chunker, and the adapters all read from this table instead of
hardcoding constants.

Field semantics:

* ``max_text`` — hard per-message character cap. Text longer than this is
  chunked (never truncated) by :mod:`nomorals.social.chat.chunking`.
* ``max_caption`` — hard media-caption cap. A longer caption is chunked,
  first chunk as the caption, the rest as follow-up text.
* ``max_file_mb`` — inline-media cap in megabytes. ``0`` means no cap
  (in-process surfaces). Media over the cap is sent as a document.
* ``supports_buttons`` — the platform renders keyboards (inline buttons
  *and* reply keyboards). The gateway strips keyboards for platforms
  that cannot render them instead of sending bytes the platform ignores.
* ``supports_html`` — the platform accepts an HTML parse mode.
* ``supports_media_edit`` — in-place media editing is implemented.
* ``rate_per_hour`` — outbound hourly budget. ``0`` means unlimited
  (the gateway's own windows govern); a positive value is a hard cap.
* ``cost_model`` — ``"free"`` or ``"per_segment"`` (SMS bills per segment).
* ``gateway_chunks`` — False when the adapter owns segmentation
  (SMS: encoding-aware splitting lives in the adapter, and gateway
  chunking would double-label segments).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

__all__ = ["PlatformLimits", "for_platform", "register_limits", "all_limits"]


@dataclass(frozen=True)
class PlatformLimits:
    platform: str
    max_text: int
    max_caption: int
    max_file_mb: float
    supports_buttons: bool
    supports_html: bool
    supports_media_edit: bool
    rate_per_hour: int
    cost_model: str
    gateway_chunks: bool = True


#: One row per platform. This is the whole table — see the descriptor rule.
_LIMITS: dict[str, PlatformLimits] = {
    # Telegram userbot (MTProto). No edit_media on the userbot adapter.
    "telegram": PlatformLimits(
        platform="telegram",
        max_text=4096, max_caption=1024, max_file_mb=50,
        supports_buttons=True, supports_html=True, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
    ),
    # Telegram Bot API. editMessageMedia is implemented.
    "telegram-bot": PlatformLimits(
        platform="telegram-bot",
        max_text=4096, max_caption=1024, max_file_mb=50,
        supports_buttons=True, supports_html=True, supports_media_edit=True,
        rate_per_hour=0, cost_model="free",
    ),
    # Second BotFather bot (personal, owner-only). Same Bot API limits
    # as the gated bot — a separate row so chunking/gateway routing
    # keyed on "telegram-personal" gets Bot API caps, not the 2000-char
    # unknown-platform default.
    "telegram-personal": PlatformLimits(
        platform="telegram-personal",
        max_text=4096, max_caption=1024, max_file_mb=50,
        supports_buttons=True, supports_html=True, supports_media_edit=True,
        rate_per_hour=0, cost_model="free",
    ),
    # WhatsApp via the local Baileys bridge. max_text is the bridge-effective
    # cap the adapter already split at (4000), not the protocol maximum.
    "whatsapp": PlatformLimits(
        platform="whatsapp",
        max_text=4000, max_caption=1024, max_file_mb=16,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
    ),
    # Discord: 2000-char hard cap, 25 MB free-tier upload cap.
    "discord": PlatformLimits(
        platform="discord",
        max_text=2000, max_caption=2000, max_file_mb=25,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
    ),
    # SMS: single-segment GSM-7 budget; the adapter owns multi-part
    # segmentation (encoding-aware), so the gateway does not chunk.
    # No media, no caption, billed per segment.
    "sms": PlatformLimits(
        platform="sms",
        max_text=160, max_caption=0, max_file_mb=0,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="per_segment",
        gateway_chunks=False,
    ),
    # In-process surfaces: no platform cap, so a generous bound that keeps
    # one message from being an unbounded blob.
    "web": PlatformLimits(
        platform="web",
        max_text=8000, max_caption=8000, max_file_mb=0,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
    ),
    "local": PlatformLimits(
        platform="local",
        max_text=8000, max_caption=8000, max_file_mb=0,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
    ),
    "webhook": PlatformLimits(
        platform="webhook",
        max_text=8000, max_caption=8000, max_file_mb=0,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
    ),
}

#: Short / legacy names → canonical platform.
_ALIASES: dict[str, str] = {
    "tgbot": "telegram-bot",
    "tgpersonal": "telegram-personal",
    "tg": "telegram",
    "wa": "whatsapp",
    "twilio": "sms",
    "webapp": "web",
    "console": "local",
    "cli": "local",
}

#: Unknown platforms get this — conservative, never a crash, never a
#: guessed branch. The platform name is preserved so diagnostics stay honest.
_DEFAULT = PlatformLimits(
    platform="",
    max_text=2000, max_caption=1024, max_file_mb=0,
    supports_buttons=False, supports_html=False, supports_media_edit=False,
    rate_per_hour=0, cost_model="free",
    gateway_chunks=True,
)


def _normalize(platform: object) -> str:
    key = str(platform or "").strip().lower()
    return _ALIASES.get(key, key)


def for_platform(platform: object) -> PlatformLimits:
    """The limits row for ``platform`` (aliases resolved). Never raises."""
    try:
        key = _normalize(platform)
        row = _LIMITS.get(key)
        if row is None:
            return dataclasses.replace(_DEFAULT, platform=key or "unknown")
        return row
    except Exception:  # noqa: BLE001 - limits lookup never breaks a send
        return dataclasses.replace(_DEFAULT, platform="unknown")


def register_limits(limits: PlatformLimits) -> None:
    """Add (or replace) one platform row. This is how a new platform lands:
    one row, no code path."""
    _LIMITS[_normalize(limits.platform)] = limits


def all_limits() -> dict[str, PlatformLimits]:
    """A copy of the whole table, for dashboards and tests."""
    return dict(_LIMITS)
