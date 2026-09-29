"""Voice bridge - ties TTS → WhatsApp/Telegram voice notes together.

Closes the loop: text → TTS → audio file → send as voice note.

Usage:
    bridge = VoiceBridge(tts_engine, whatsapp_adapter, telegram_adapter)
    
    # Send voice note to WhatsApp
    await bridge.send_voice_whatsapp(chat_ref, "Hello! This is a voice message.")
    
    # Send voice note to Telegram
    await bridge.send_voice_telegram(chat_id, "Hello! This is a voice message.")
    
    # Transcribe incoming voice note
    text = await bridge.transcribe_voice(audio_path)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger
from ..integrations.voice_integration import VoiceIntegration
from ..social.chat.base import ChatRef
from ..social.chat.whatsapp import WhatsAppAdapter

__all__ = ["VoiceBridge"]

_log = get_logger(__name__)


class VoiceBridge:
    """Bridges TTS → WhatsApp/Telegram voice notes."""
    
    def __init__(
        self,
        voice: VoiceIntegration,
        whatsapp: Optional[WhatsAppAdapter] = None,
        telegram: Any = None,  # TelegramAdapter
    ) -> None:
        self.voice = voice
        self.whatsapp = whatsapp
        self.telegram = telegram
        _log.info("Voice bridge initialized")
    
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
            audio_path = await self.voice.tts.synthesize(text, voice=voice, rate=rate)
            
            if not audio_path or not Path(audio_path).exists():
                _log.error("TTS failed to generate audio")
                return False
            
            # Send as voice note
            result = self.whatsapp.send_voice(chat, audio_path, caption=caption)
            
            # Cleanup temp file
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:
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
            audio_path = await self.voice.tts.synthesize(text, voice=voice, rate=rate)
            
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
            except Exception:
                pass
            
            _log.info(f"Sent voice note to Telegram: {chat_id}")
            return True
                
        except Exception as e:
            _log.error(f"Voice bridge error: {e}")
            return False
    
    async def transcribe_voice(
        self,
        audio_path: str,
        *,
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
            text = await self.voice.transcribe(audio_path, language=language)
            return text
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
                    return await self.transcribe_voice(media["path"], language=language)
        
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
