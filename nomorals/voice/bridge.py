"""Voice bridge - ties TTS → WhatsApp/Telegram voice notes together.

Closes the loop: text → TTS → audio file → send as voice note.

The ``voice`` engine is duck-typed, newest first:

1. :class:`nomorals.voice.tts.UniversalTTS` — the owner's own engine
   (local neural backends, voice cloning, director). Preferred.
   ``speak(text, voice_name=...)`` → ``{"path": ...}``.
2. :class:`nomorals.voice.stt.UniversalSTT` — for the transcribe side.
   ``transcribe(path, language=...)`` → ``{"text": ...}``.
3. Legacy :class:`nomorals.integrations.voice_integration.VoiceIntegration`
   (cloud edge-tts/gtts/elevenlabs) — still works, kept for backward
   compatibility. Prefer the owner's engine for new code.

Usage:
    from nomorals.voice.tts import UniversalTTS

    bridge = VoiceBridge(UniversalTTS(backend="auto"),
                         whatsapp_adapter, telegram_adapter)

    # Send voice note to WhatsApp
    await bridge.send_voice_whatsapp(chat_ref, "Hello! This is a voice message.")

    # Send voice note to Telegram
    await bridge.send_voice_telegram(chat_id, "Hello! This is a voice message.")

    # Transcribe incoming voice note
    text = await bridge.transcribe_voice(audio_path)
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger
from ..integrations.voice_integration import VoiceIntegration
from ..social.chat.base import ChatRef
from ..social.chat.whatsapp import WhatsAppAdapter

__all__ = ["VoiceBridge"]


_log = get_logger(__name__)


class VoiceBridge:
    """Bridges TTS → WhatsApp/Telegram voice notes.

    ``voice`` is duck-typed: a ``UniversalTTS`` (preferred, the owner's
    own engine), a ``UniversalSTT`` (transcribe side), or the legacy
    ``VoiceIntegration``. ``None`` is allowed at construction; the first
    synthesis/transcription then raises a clear error instead of failing
    deep inside a send.
    """

    def __init__(
        self,
        voice: Any = None,
        whatsapp: Optional[WhatsAppAdapter] = None,
        telegram: Any = None,  # TelegramAdapter
    ) -> None:
        self.voice = voice
        self.whatsapp = whatsapp
        self.telegram = telegram
        _log.info("Voice bridge initialized (engine=%s)",
                  type(voice).__name__ if voice is not None else "none")

    # ── engine adaptation (duck-typed, newest first) ──────────────────

    def _require_voice(self) -> Any:
        if self.voice is None:
            raise RuntimeError(
                "VoiceBridge has no voice engine — pass UniversalTTS (or "
                "the legacy VoiceIntegration) as the first constructor arg")
        return self.voice

    async def _synthesize(self, text: str, *, voice: str = "default",
                          rate: str = "+0%") -> str:
        """Text → audio file path, via whichever engine is attached."""
        engine = self._require_voice()
        speak = getattr(engine, "speak", None)
        if callable(speak):
            # UniversalTTS: sync, returns {"path": ...}. The owner's own
            # engine — no cloud, voice profiles live in its VoiceLibrary.
            out = speak(text,
                        voice_name=voice if voice != "default" else None)
            path = out.get("path") if isinstance(out, dict) else None
            if not path:
                raise RuntimeError("TTS engine returned no audio path")
            return path
        tts = getattr(engine, "tts", None)
        if tts is not None and callable(getattr(tts, "synthesize", None)):
            # Legacy VoiceIntegration (cloud edge-tts / gtts / elevenlabs).
            return await tts.synthesize(text, voice=voice, rate=rate)
        raise RuntimeError(
            f"voice engine {type(engine).__name__!r} has neither "
            ".speak() (UniversalTTS) nor .tts.synthesize() (legacy)")

    async def _transcribe(self, audio_path: str,
                          *, language: str = "en") -> str:
        """Audio file path → transcript text."""
        engine = self._require_voice()
        transcribe = getattr(engine, "transcribe", None)
        if callable(transcribe):
            res = transcribe(audio_path, language=language)
            if inspect.isawaitable(res):
                res = await res
            # UniversalSTT returns {"text": ...}; legacy returns str.
            if isinstance(res, dict):
                return str(res.get("text", "") or "")
            return str(res or "")
        stt = getattr(engine, "stt", None)
        if stt is not None and callable(getattr(stt, "transcribe", None)):
            res = stt.transcribe(audio_path, language=language)
            if inspect.isawaitable(res):
                res = await res
            if isinstance(res, dict):
                return str(res.get("text", "") or "")
            return str(res or "")
        raise RuntimeError(
            f"voice engine {type(engine).__name__!r} has neither "
            ".transcribe() (UniversalSTT) nor .stt.transcribe() (legacy)")

    async def send_voice_whatsapp(
        self,
        chat: ChatRef,
        text: str,
        *,
        voice: str = "default",
        rate: str = "+0%",
        caption: str = "",
    ) -> bool:
        """Send a voice note to WhatsApp.

        Args:
            chat: WhatsApp chat reference
            text: Text to speak
            voice: TTS voice name
            rate: Speech rate adjustment
            caption: Optional caption

        Returns:
            True if sent successfully
        """
        if not self.whatsapp:
            _log.error("WhatsApp adapter not available")
            return False

        try:
            # Generate audio via TTS
            audio_path = await self._synthesize(text, voice=voice, rate=rate)

            if not audio_path or not Path(audio_path).exists():
                _log.error("TTS failed to generate audio")
                return False

            # Send as voice note
            result = self.whatsapp.send_voice(chat, audio_path, caption=caption)

            # Cleanup temp file
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:  # noqa: E103 - temp file cleanup is best-effort
                pass

            if result.ok:
                _log.info(f"Sent voice note to WhatsApp: {chat.chat_id}")
                return True
            else:
                _log.error(f"Failed to send voice note: {result.error}")
                return False

        except Exception as e:
            _log.error(f"Voice bridge error: {e}")
            return False

    async def send_voice_telegram(
        self,
        chat_id: int | str,
        text: str,
        *,
        voice: str = "default",
        rate: str = "+0%",
        caption: str = "",
    ) -> bool:
        """Send a voice note to Telegram.

        Args:
            chat_id: Telegram chat ID
            text: Text to speak
            voice: TTS voice name
            rate: Speech rate adjustment
            caption: Optional caption

        Returns:
            True if sent successfully
        """
        if not self.telegram:
            _log.error("Telegram adapter not available")
            return False

        try:
            # Generate audio via TTS
            audio_path = await self._synthesize(text, voice=voice, rate=rate)

            if not audio_path or not Path(audio_path).exists():
                _log.error("TTS failed to generate audio")
                return False

            # Send as voice note via Telegram
            # Telegram uses voice_note=True flag
            await self.telegram.client.send_file(
                chat_id,
                audio_path,
                voice_note=True,
                caption=caption,
            )

            # Cleanup temp file
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:  # noqa: E103 - temp file cleanup is best-effort
                pass

            _log.info(f"Sent voice note to Telegram: {chat_id}")
            return True

        except Exception as e:
            _log.error(f"Voice bridge error: {e}")
            return False

    async def transcribe_voice(
        self,
        audio_path: str,
        language: str = "en",
    ) -> str:
        """Transcribe a voice message to text.

        Args:
            audio_path: Path to audio file
            language: Expected language

        Returns:
            Transcribed text
        """
        try:
            return await self._transcribe(audio_path, language=language)
        except Exception as e:
            _log.error(f"Transcription failed: {e}")
            return ""

    async def transcribe_whatsapp_voice(
        self,
        message: Any,
        *,
        language: str = "en",
    ) -> str:
        """Transcribe a WhatsApp voice message.

        Args:
            message: WhatsApp message with voice media
            language: Expected language

        Returns:
            Transcribed text
        """
        # Extract audio path from message media
        if hasattr(message, "media") and message.media:
            for media in message.media:
                if media.get("kind") in ("audio", "voice") and media.get("path"):
                    return await self.transcribe_voice(media["path"],
                                                       language=language)

        return ""

    async def send_voice_auto(
        self,
        platform: str,
        chat_id: Any,
        text: str,
        **kwargs: Any,
    ) -> bool:
        """Send voice note to the appropriate platform.

        Args:
            platform: "whatsapp" or "telegram"
            chat_id: Chat identifier
            text: Text to speak
            **kwargs: Platform-specific options

        Returns:
            True if sent successfully
        """
        if platform == "whatsapp":
            if isinstance(chat_id, ChatRef):
                return await self.send_voice_whatsapp(chat_id, text, **kwargs)
            else:
                _log.error("WhatsApp requires ChatRef, not raw ID")
                return False
        elif platform == "telegram":
            return await self.send_voice_telegram(chat_id, text, **kwargs)
        else:
            _log.error(f"Unsupported platform: {platform}")
            return False
