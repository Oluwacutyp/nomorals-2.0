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
* ``supports_text_edit`` — in-place TEXT editing of the account's
  own already-sent messages is implemented (the progressive-display
  seam: adapters expose ``edit_progress``). ``text_edit_scope``
  names, in plain words, exactly what that edit path covers and
  what remains absent — an absence is recorded BY NAME here, never
  faked downstream and never silently whole-message when an edit
  path exists.
* ``rate_per_hour`` — outbound hourly budget. ``0`` means unlimited
  (the gateway's own windows govern); a positive value is a hard cap.
* ``cost_model`` — ``"free"`` or ``"per_segment"`` (SMS bills per segment).
* ``gateway_chunks`` — False when the adapter owns segmentation
  (SMS: encoding-aware splitting lives in the adapter, and gateway
  chunking would double-label segments).
* ``utf16_lengths`` — True when the platform's caps are counted in
  UTF-16 code units rather than Unicode code points (Telegram's Bot
  API / MTProto limits, and the JS stacks behind Discord and WhatsApp,
  all measure string length in UTF-16 units: an astral-plane emoji
  counts 2).  Chunking and caption splitting for such platforms
  measure with :func:`nomorals.social.chat.chunking.utf16_len` — a
  4096-code-point chunk of emoji is 8192 UTF-16 units and Telegram
  rejects it as too long.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from .failures import record_failure

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
    utf16_lengths: bool = False
    supports_text_edit: bool = False
    text_edit_scope: str = ""


#: One row per platform. This is the whole table — see the descriptor rule.
_LIMITS: dict[str, PlatformLimits] = {
    # Telegram userbot (MTProto). Buttons are real end-to-end since
    # AA-77: native inline keyboards (Telethon Button.inline) on
    # send/edit plus a CallbackQuery handler routing taps as
    # commands.  Text edit is real via the edit_message verb; what
    # the surface lacks is the throttled edit_progress seam
    # (progressive display binds the Bot API adapters) and
    # edit_media — those absences are named, not hidden.
    "telegram": PlatformLimits(
        platform="telegram",
        max_text=4096, max_caption=1024, max_file_mb=50,
        supports_buttons=True, supports_html=True, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
        utf16_lengths=True,
        supports_text_edit=True,
        text_edit_scope=("own messages via MTProto edit (edit_message "
                         "verb, unthrottled); no edit_progress seam "
                         "(the Bot API adapters have one)"),
    ),
    # Telegram Bot API. editMessageMedia is implemented.
    "telegram-bot": PlatformLimits(
        platform="telegram-bot",
        max_text=4096, max_caption=1024, max_file_mb=50,
        supports_buttons=True, supports_html=True, supports_media_edit=True,
        rate_per_hour=0, cost_model="free",
        utf16_lengths=True,
        supports_text_edit=True,
        text_edit_scope=("own messages via editMessageText; progress "
                         "edits throttled (~1/3s, on change only)"),
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
        utf16_lengths=True,
        supports_text_edit=True,
        text_edit_scope=("own messages via editMessageText; progress "
                         "edits throttled (~1/3s, on change only)"),
    ),
    # WhatsApp via the local bridges. max_text is the protocol cap:
    # WhatsApp text messages allow 65,536 characters, and neither
    # bridge imposes a lower limit (the Go/whatsmeow send handler and
    # the Node/Baileys sendMessage path both pass text through
    # untouched — verified 2026-11-04). The old 4000 here was a
    # Devon-invented "bridge-effective" cap that shredded ordinary
    # long replies (the full WhatsApp menu went out as ~13 chunks).
    "whatsapp": PlatformLimits(
        platform="whatsapp",
        max_text=65536, max_caption=1024, max_file_mb=16,
        # Buttons are REAL here (register #27, 2026-11-05): both
        # bridges implement the button contract — the Go bridge
        # (bridge/gobridge/messages.go cmdSend) and the Node bridge
        # (bridge/whatsapp-bridge.mjs cmdSend) send the first 3
        # buttons as an InteractiveMessage (NativeFlow quick_reply
        # set) with overflow as text action lines, and taps
        # re-enter as commands. (The legacy ButtonsMessage both
        # bridges originally sent no longer renders on current
        # WhatsApp clients for unofficial senders — replaced
        # 2026-11-07; a refused interactive send degrades to text
        # lines and the bridge reports the form, which the adapter
        # journals.) The old False made the gateway strip every
        # authored button bundle in silence while reply copy still
        # said "tap a button below". Buttons completion
        # (2026-11-08): media sends carry the same keyboard (the
        # media heads an InteractiveMessage), edits carry it too
        # (see text_edit_scope), and an account whose quick_reply
        # form degrades falls back to the ListMessage form on the
        # bridge-reported evidence (the /pair buttons probe).
        supports_buttons=True, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
        utf16_lengths=True,
        # Progressive display (register #31): real, via BOTH
        # bridges' edit commands behind the adapter's
        # edit_progress / edit_message. The scope names the edges.
        supports_text_edit=True,
        text_edit_scope=(
            "own messages only, inside WhatsApp's ~15-minute edit "
            "window, via both bridges' edit commands (Go: whatsmeow "
            "BuildEdit; Baileys: its MESSAGE_EDIT wrap) — and since "
            "the buttons completion (2026-11-08) the replacement "
            "carries the declared keyboard too: interactive/list "
            "rebuilds, an empty keyboard clears, and a refused "
            "clearing edit falls back at the adapter to revoke + a "
            "fresh settled send. Absent by name: editing other "
            "people's messages (the protocol cannot) and "
            "media/caption edits"),
    ),
    # Discord: 2000-char hard cap, 25 MB free-tier upload cap.
    "discord": PlatformLimits(
        platform="discord",
        max_text=2000, max_caption=2000, max_file_mb=25,
        supports_buttons=False, supports_html=False, supports_media_edit=False,
        rate_per_hour=0, cost_model="free",
        utf16_lengths=True,
        supports_text_edit=False,
        text_edit_scope=("absent: Discord's API can edit own messages, "
                         "but this adapter has no edit seam yet"),
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
    except Exception as exc:  # noqa: BLE001 - limits lookup never breaks a send
        record_failure('platform_limits.for_platform', exc)
        return dataclasses.replace(_DEFAULT, platform="unknown")


def register_limits(limits: PlatformLimits) -> None:
    """Add (or replace) one platform row. This is how a new platform lands:
    one row, no code path."""
    _LIMITS[_normalize(limits.platform)] = limits


def all_limits() -> dict[str, PlatformLimits]:
    """A copy of the whole table, for dashboards and tests."""
    return dict(_LIMITS)
