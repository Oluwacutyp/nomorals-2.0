"""Per-surface voice-note transports — one interface, honest results.

The TTS pipeline produces an audio file; *sending* it as a voice note
is a different operation on every surface:

- **Bot API** (``TelegramBotAdapter`` — the gated bot AND the personal
  bot): ``sendVoice`` with an OGG/Opus upload, through the adapter's
  core-HttpClient session contract.
- **MTProto userbot** (``TelegramAdapter`` — the owner's own account):
  Telethon ``send_file(..., voice_note=True)``. Generated artifacts on
  this personal surface default to the owner's **Saved Messages**.
- **WhatsApp** (``WhatsAppAdapter``): the bridge's ``ptt`` voice send.

Before this module, the TTS paths hand-rolled "convert + send_media"
per caller and the Bot adapter's media send assumed a requests-style
session the production session (core ``HttpClient``) doesn't have —
voice notes died with ``'HttpClient' object has no attribute 'post'``
and callers papered over it with plain file sends that were easy to
mistake for real voice notes.

The contract here:

- :func:`send_voice_note` returns a :class:`VoiceDelivery`. It claims
  ``voice_note=True`` only when a real voice bubble landed.
- When the surface *rejects* a voice send, the transport itself
  delivers the artifact as a plain file, marked ``degraded`` **with
  the reason**. When conversion to OGG/Opus is impossible, no voice
  bubble can exist on any surface: the transport sends nothing and
  returns the reason, and the caller runs its labelled file fallback
  ("sent as file instead — <reason>"). Either way a file is never
  passed off as a voice note.
- Transports only ever call their OWN surface's adapter API. The
  MTProto transport cannot touch a Bot API client shape and vice
  versa — the boundary is the adapter's own ``send_media`` /
  ``send_voice``, typed per surface below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatRef, MediaRef, SendResult

_log = get_logger(__name__)

__all__ = [
    "VoiceDelivery",
    "VoiceTransport",
    "BotApiVoiceTransport",
    "MTProtoVoiceTransport",
    "WhatsAppVoiceTransport",
    "AdapterVoiceTransport",
    "voice_transport_for",
    "send_voice_note",
    "saved_messages_chat",
    "deliver_voice_artifact",
]


@dataclass
class VoiceDelivery:
    """The honest outcome of one voice-note send attempt."""

    ok: bool
    #: True only when the message landed as a playable voice bubble.
    voice_note: bool = False
    #: True when the artifact was delivered as a plain file instead —
    #: ``reason`` says why, and the user-facing reply must carry it.
    degraded: bool = False
    reason: str = ""
    message_id: str = ""
    platform: str = ""
    sent_path: str = ""


class VoiceTransport:
    """Base transport: convert → surface voice send → labelled file fallback."""

    surface = "generic"

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    @property
    def platform(self) -> str:
        return str(getattr(self.adapter, "name", "") or "")

    # ── the one entry point ──────────────────────────────────────────
    def send_voice(self, chat: ChatRef, audio_path: str, *,
                   caption: str = "") -> VoiceDelivery:
        from ...tools.audio import wav_to_voice_ogg

        try:
            send_path, mime = wav_to_voice_ogg(audio_path)
        except Exception as exc:  # noqa: BLE001 - conversion is best-effort
            return VoiceDelivery(
                ok=False, platform=self.platform,
                reason=f"voice-note conversion error: {exc}")
        if mime != "audio/ogg":
            # No OGG/Opus → no voice bubble is possible. Nothing is
            # sent at this layer: the caller runs its labelled file
            # fallback (the pipeline's gateway file route) with this
            # reason in the message.
            return VoiceDelivery(
                ok=False, platform=self.platform,
                reason=("ffmpeg not available for WAV → OGG/Opus "
                        f"conversion (source stayed "
                        f"{mime or 'an unknown format'})"))
        result = self._call(self._send_voice_media, chat, send_path,
                            caption)
        if result is not None and result.ok:
            return VoiceDelivery(
                ok=True, voice_note=True,
                    message_id=getattr(result, "message_id", "") or "",
                platform=self.platform, sent_path=send_path)
        voice_error = ("voice send failed: "
                       + ((result.error if result is not None else "")
                          or "adapter unavailable"))
        # Degraded path: the voice send was rejected — the artifact
        # still goes out, as a plain file, and the delivery SAYS so.
        file_result = self._call(self._send_file_media, chat, audio_path,
                                 caption)
        if file_result is not None and file_result.ok:
            _log.info("voice transport (%s): degraded file send for %s: %s",
                      self.surface, chat.key, voice_error)
            return VoiceDelivery(
                ok=True, voice_note=False, degraded=True,
                reason=voice_error,
                message_id=getattr(file_result, "message_id", "") or "",
                platform=self.platform, sent_path=audio_path)
        file_error = ((file_result.error if file_result is not None else "")
                      or "adapter unavailable")
        return VoiceDelivery(
            ok=False, platform=self.platform,
            reason=f"{voice_error}; file fallback also failed: {file_error}")

    # ── surface hooks ────────────────────────────────────────────────
    def _invoke_send_media(self, chat: ChatRef, media: MediaRef,
                           caption: str) -> SendResult:
        # ``caption`` is the base contract's optional arg — omitted
        # when empty so minimal adapters see the plain two-arg call.
        if caption:
            return self.adapter.send_media(chat, media, caption=caption)
        return self.adapter.send_media(chat, media)

    def _send_voice_media(self, chat: ChatRef, path: str,
                          caption: str) -> SendResult:
        """Send a converted OGG as a voice bubble on this surface."""
        media = MediaRef(path=path, kind="voice", name="voice.ogg",
                         mime="audio/ogg", voice_note=True)
        return self._invoke_send_media(chat, media, caption)

    def _send_file_media(self, chat: ChatRef, path: str,
                         caption: str) -> SendResult:
        """Send the artifact as a plain file (the degraded route)."""
        media = MediaRef(path=path, kind="document",
                         name=os.path.basename(path))
        return self._invoke_send_media(chat, media, caption)

    def _call(self, fn: Any, *args: Any) -> SendResult | None:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - a raising adapter is a result
            return SendResult(ok=False, platform=self.platform,
                              error=str(exc))

    def saved_messages_ref(self) -> ChatRef | None:
        """Saved Messages target on this surface, when it has one."""
        return None


class BotApiVoiceTransport(VoiceTransport):
    """Telegram Bot API surfaces (gated bot + personal bot): sendVoice."""

    surface = "telegram-bot-api"


class MTProtoVoiceTransport(VoiceTransport):
    """The owner's own account (Telethon): send_file voice_note=True.

    This transport only ever calls the userbot adapter's own
    ``send_media`` — no Bot API client shape exists on this surface.
    """

    surface = "telegram-mtproto"

    def saved_messages_ref(self) -> ChatRef | None:
        ref = getattr(self.adapter, "saved_messages_ref", None)
        if not callable(ref):
            return None
        try:
            return ref()
        except Exception:  # noqa: BLE001 - unknown id → no target, honestly
            return None


class WhatsAppVoiceTransport(VoiceTransport):
    """WhatsApp: the bridge's native ptt voice send."""

    surface = "whatsapp"

    def _send_voice_media(self, chat: ChatRef, path: str,
                          caption: str) -> SendResult:
        return self.adapter.send_voice(chat, path, caption=caption)


class AdapterVoiceTransport(VoiceTransport):
    """Any other chat adapter: its own send_media with the voice flag."""

    surface = "adapter"


def voice_transport_for(adapter: Any) -> VoiceTransport:
    """The voice transport owning this adapter's surface."""
    from .telegram import TelegramAdapter, TelegramBotAdapter
    from .whatsapp import WhatsAppAdapter

    if isinstance(adapter, TelegramBotAdapter):
        return BotApiVoiceTransport(adapter)
    if isinstance(adapter, TelegramAdapter):
        return MTProtoVoiceTransport(adapter)
    if isinstance(adapter, WhatsAppAdapter):
        return WhatsAppVoiceTransport(adapter)
    return AdapterVoiceTransport(adapter)


def send_voice_note(adapter: Any, chat: ChatRef, audio_path: str, *,
                    caption: str = "") -> VoiceDelivery:
    """Send ``audio_path`` as a voice note on ``adapter``'s surface.

    THE interface the TTS pipeline calls. Never raises: the result
    carries ok / voice_note / degraded + reason.
    """
    if adapter is None:
        return VoiceDelivery(ok=False, reason="no adapter for this surface")
    try:
        return voice_transport_for(adapter).send_voice(
            chat, audio_path, caption=caption)
    except Exception as exc:  # noqa: BLE001 - delivery must report, not crash
        return VoiceDelivery(ok=False, platform=str(
            getattr(adapter, "name", "") or ""), reason=str(exc))


def saved_messages_chat(adapter: Any) -> ChatRef | None:
    """The Saved Messages ChatRef for this adapter's surface, if any.

    Only the MTProto userbot can address the owner's Saved Messages;
    every other surface returns None.
    """
    if adapter is None:
        return None
    try:
        return voice_transport_for(adapter).saved_messages_ref()
    except Exception:  # noqa: BLE001
        return None


def deliver_voice_artifact(adapter: Any, audio_path: str, *,
                           chat: ChatRef | None = None,
                           caption: str = "") -> VoiceDelivery:
    """Deliver a GENERATED voice artifact (no conversational home).

    With an explicit ``chat`` the artifact goes there. Without one, the
    personal (MTProto) surface defaults to the owner's Saved Messages;
    surfaces with no Saved Messages fail honestly rather than guessing
    a destination.
    """
    target = chat or saved_messages_chat(adapter)
    if target is None:
        return VoiceDelivery(
            ok=False, platform=str(getattr(adapter, "name", "") or ""),
            reason=("no destination chat given and this surface has no "
                    "Saved Messages target"))
    return send_voice_note(adapter, target, audio_path, caption=caption)
