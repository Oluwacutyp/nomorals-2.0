"""Voice message integration for Telegram and other platforms.

Supports:
1. Speech-to-text (STT) via Whisper (local or API)
2. Text-to-speech (TTS) via multiple engines
3. Voice message send/receive on Telegram
4. Voice command recognition

Usage:
    voice = VoiceIntegration(telegram_client)
    
    # Send voice message
    await voice.send_voice_message(
        chat_id=12345,
        text="Hello! This is a voice message.",
        voice="default"
    )
    
    # Transcribe received voice message
    text = await voice.transcribe(audio_file_path)
    
    # Recognize voice command
    command = await voice.recognize_command(audio_file_path)
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger

__all__ = [
    "VoiceIntegration",
    "VoiceMessage",
    "VoiceCommand",
    "TTSEngine",
    "STTEngine",
]

_log = get_logger(__name__)


@dataclass
class VoiceMessage:
    """Represents a voice message."""
    
    text: str
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0
    language: str = "en"
    voice: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceCommand:
    """Recognized voice command."""
    
    command: str
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


class TTSEngine:
    """Text-to-speech engine abstraction.
    
    Supports multiple backends:
    - edge-tts (Microsoft Edge TTS, free, high quality)
    - gtts (Google TTS, free)
    - pyttsx3 (offline, system voices)
    - ElevenLabs API (premium quality)
    """
    
    def __init__(self, engine: str = "edge-tts") -> None:
        self.engine = engine
        self._voices: dict[str, str] = {}

    # Curated voice catalog: style → voice id per engine family.
    VOICE_CATALOG: dict[str, dict[str, str]] = {
        "default": {"label": "Aria — warm, general", "edge": "en-US-AriaNeural"},
        "male": {"label": "Guy — masculine, steady", "edge": "en-US-GuyNeural"},
        "female": {"label": "Aria — female, warm", "edge": "en-US-AriaNeural"},
        "british": {"label": "Sonia — British English", "edge": "en-GB-SoniaNeural"},
        "nigerian": {"label": "Ezinne — Nigerian English", "edge": "en-NG-EzinneNeural"},
        "deep": {"label": "Davis — deep narrator", "edge": "en-US-DavisNeural"},
        "cheerful": {"label": "Jenny — bright, upbeat", "edge": "en-US-JennyNeural"},
        "calm": {"label": "Ana — soft, calm", "edge": "en-US-AnaNeural"},
        "news": {"label": "Christopher — newsreader", "edge": "en-US-ChristopherNeural"},
        "assistant": {"label": "Ava — crisp assistant", "edge": "en-US-AvaNeural"},
    }

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        output_path: str | None = None,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Convert text to speech.
        
        Args:
            text: Text to speak
            voice: Voice name/ID
            output_path: Output file path (auto-generated if None)
            rate: Speech rate adjustment (e.g., "+20%", "-10%")
            volume: Volume adjustment
            pitch: Pitch adjustment (e.g., "+10Hz", "-5Hz")
            
        Returns:
            Path to generated audio file
        """
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                output_path = f.name
        
        if self.engine == "edge-tts":
            return await self._tts_edge(text, voice, output_path, rate,
                                        volume, pitch)
        elif self.engine == "gtts":
            return await self._tts_gtts(text, voice, output_path)
        elif self.engine == "elevenlabs":
            return await self._tts_elevenlabs(text, voice, output_path,
                                              rate=rate)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")

    async def synthesize_stream(self, text: str, *,
                                voice: str = "default",
                                rate: str = "+0%", volume: str = "+0%",
                                pitch: str = "+0Hz"):
        """Async generator of MP3 audio chunks (edge-tts streaming).

        Yields ``bytes`` as they arrive — play the first chunk while
        the rest synthesizes. Long texts are chunked sentence-aware.
        """
        if self.engine != "edge-tts":
            # non-streaming engines: synthesize whole, yield once
            path = await self.synthesize(text, voice=voice, rate=rate,
                                         volume=volume, pitch=pitch)
            with open(path, "rb") as f:
                yield f.read()
            return
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts not installed") from exc
        selected = self._resolve_voice(voice)
        for chunk in self._chunk_text(text):
            communicate = edge_tts.Communicate(
                chunk, selected, rate=rate, volume=volume, pitch=pitch)
            async for part in communicate.stream():
                if part["type"] == "audio" and part.get("data"):
                    yield part["data"]

    async def synthesize_long(self, text: str, *,
                              voice: str = "default",
                              output_path: str | None = None,
                              rate: str = "+0%", volume: str = "+0%",
                              pitch: str = "+0Hz",
                              on_progress: Any = None) -> str:
        """Long-form synthesis: sentence-aware chunking under edge-tts's
        ~2000-char limit, chunks concatenated to one MP3."""
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                output_path = f.name
        chunks = self._chunk_text(text)
        with open(output_path, "wb") as out:
            for i, part in enumerate(chunks):
                data = b"".join(
                    [c async for c in self.synthesize_stream(
                        part, voice=voice, rate=rate, volume=volume,
                        pitch=pitch)])
                out.write(data)
                if on_progress is not None:
                    try:
                        on_progress(i + 1, len(chunks))
                    except Exception:  # noqa: BLE001
                        pass
        return output_path

    @staticmethod
    def _chunk_text(text: str, limit: int = 1800) -> list[str]:
        """Split on sentence boundaries under ``limit`` chars."""
        import re as _re
        sentences = _re.split(r"(?<=[.!?])\s+", (text or "").strip())
        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) + 1 <= limit:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                # a single monster sentence: hard-split on commas
                while len(s) > limit:
                    cut = s.rfind(",", 0, limit)
                    cut = cut if cut > 0 else limit
                    chunks.append(s[:cut].strip())
                    s = s[cut:].strip()
                cur = s
        if cur:
            chunks.append(cur)
        return chunks or [text]

    def _resolve_voice(self, voice: str) -> str:
        entry = self.VOICE_CATALOG.get(voice or "default", {})
        return entry.get("edge", voice or "en-US-AriaNeural")

    def pick_voice(self, *, style: str = "", language: str = "",
                   gender: str = "") -> str:
        """Pick a voice id by style/language/gender hints.

        ``style`` matches catalog keys or labels; ``language`` matches
        e.g. "en-NG"; ``gender`` matches "male"/"female" in labels.
        Falls back to "default" — never raises.
        """
        want = (style or "").lower()
        if want in self.VOICE_CATALOG:
            return want
        for key, entry in self.VOICE_CATALOG.items():
            label = entry.get("label", "").lower()
            vid = entry.get("edge", "")
            if want and want in label:
                return key
            if language and language.lower() in vid.lower():
                return key
            if gender and gender.lower() in label:
                return key
        return "default"

    def format_voices(self) -> str:
        """God-tier voice browser for chat."""
        lines = ["🎙️ **voices** (edge-tts, free)"]
        for key, entry in self.VOICE_CATALOG.items():
            lines.append(f"• `{key}` — {entry['label']} "
                         f"({entry['edge']})")
        lines.append("\n_use `pick_voice(style=...)` or pass a voice key_")
        return "\n".join(lines)
    
    async def _tts_edge(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str,
        volume: str,
        pitch: str = "+0Hz",
    ) -> str:
        """Use Microsoft Edge TTS (free, high quality)."""
        try:
            import edge_tts

            selected_voice = self._resolve_voice(voice)

            communicate = edge_tts.Communicate(
                text,
                selected_voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
            )

            await communicate.save(output_path)
            _log.info(f"Generated voice message: {output_path}")
            return output_path

        except ImportError:
            _log.warning("edge-tts not installed, falling back to gtts")
            return await self._tts_gtts(text, voice, output_path)
    
    async def _tts_gtts(
        self,
        text: str,
        voice: str,
        output_path: str,
    ) -> str:
        """Use Google TTS (free, basic quality)."""
        try:
            from gtts import gTTS
            
            tts = gTTS(text=text, lang="en")
            tts.save(output_path)
            return output_path
            
        except ImportError:
            _log.error("No TTS engine available. Install edge-tts or gtts.")
            raise RuntimeError("No TTS engine available")
    
    async def _tts_elevenlabs(
        self,
        text: str,
        voice: str,
        output_path: str,
        *,
        rate: str = "+0%",
        api_key: str = "",
    ) -> str:
        """Use ElevenLabs API (premium quality, requires API key).

        Real implementation: POST /v1/text-to-speech/{voice_id} with
        the ElevenLabs voice id. ``voice`` may be a voice id or a
        friendly alias ("rachel", "adam"…).
        """
        key = api_key or getattr(self, "_elevenlabs_key", "") or ""
        if not key:
            _log.warning("ElevenLabs needs an API key — falling back to "
                         "edge-tts (set TTSEngine._elevenlabs_key)")
            return await self._tts_edge(text, voice, output_path, rate,
                                        "+0%", "+0Hz")
        voice_ids = {
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "adam": "pNInz6obpgDQGcFmaJgB",
            "sam": "yoZ06aMxZJJ28mfd3POQ",
            "bella": "EXAVITQu4vr4xnSDxMaL",
        }
        voice_id = voice_ids.get((voice or "").lower(), voice)
        if not voice_id or voice_id == "default":
            voice_id = voice_ids["rachel"]
        payload = json.dumps({
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }).encode()
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            data=payload,
            headers={"xi-api-key": key, "Content-Type": "application/json",
                     "Accept": "audio/mpeg"})
        try:
            import asyncio
            loop = asyncio.get_running_loop()

            def _do() -> None:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(output_path, "wb") as f:
                        f.write(resp.read())

            await loop.run_in_executor(None, _do)
            _log.info("ElevenLabs TTS saved: %s", output_path)
            return output_path
        except Exception as e:
            _log.error(f"ElevenLabs TTS failed: {e}")
            raise

    def list_voices(self) -> list[str]:
        """List available voice keys."""
        if self.engine == "edge-tts":
            return list(self.VOICE_CATALOG)
        return ["default"]


class STTEngine:
    """Speech-to-text engine abstraction.
    
    Supports multiple backends:
    - whisper (local OpenAI Whisper model)
    - whisper-api (OpenAI Whisper API)
    - google-speech (Google Speech Recognition)
    """
    
    def __init__(self, engine: str = "whisper") -> None:
        self.engine = engine
    
    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str = "en",
        model: str = "base",
    ) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Expected language code
            model: Whisper model size (tiny, base, small, medium, large)
            
        Returns:
            Transcribed text
        """
        if self.engine == "whisper":
            return await self._stt_whisper_local(audio_path, language, model)
        elif self.engine == "whisper-api":
            return await self._stt_whisper_api(audio_path, language)
        elif self.engine == "google-speech":
            return await self._stt_google(audio_path, language)
        else:
            raise ValueError(f"Unknown STT engine: {self.engine}")
    
    async def _stt_whisper_local(
        self,
        audio_path: str,
        language: str,
        model: str,
    ) -> str:
        """Use local Whisper model."""
        try:
            import whisper
            
            whisp_model = whisper.load_model(model)
            result = whisp_model.transcribe(audio_path, language=language)
            return result["text"].strip()
            
        except ImportError:
            _log.warning("Whisper not installed, trying whisper CLI")
            return await self._stt_whisper_cli(audio_path, language, model)
    
    async def _stt_whisper_cli(
        self,
        audio_path: str,
        language: str,
        model: str,
    ) -> str:
        """Use Whisper CLI."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    [
                        "whisper", audio_path,
                        "--model", model,
                        "--language", language,
                        "--output_dir", tmpdir,
                        "--output_format", "txt",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                
                if result.returncode == 0:
                    # Read output file
                    txt_file = Path(tmpdir) / f"{Path(audio_path).stem}.txt"
                    if txt_file.exists():
                        return txt_file.read_text().strip()
                
                _log.error(f"Whisper CLI failed: {result.stderr}")
                return ""
        except FileNotFoundError:
            _log.error("Whisper CLI not found")
            return ""
        except subprocess.TimeoutExpired:
            _log.error("Whisper transcription timed out")
            return ""
    
    async def _stt_whisper_api(
        self,
        audio_path: str,
        language: str,
    ) -> str:
        """Use OpenAI Whisper API."""
        # Would need API key from credential vault
        _log.warning("Whisper API not implemented yet")
        return await self._stt_whisper_local(audio_path, language, "base")
    
    async def _stt_google(
        self,
        audio_path: str,
        language: str,
    ) -> str:
        """Use Google Speech Recognition."""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # Convert to WAV if needed
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            
            text = recognizer.recognize_google(audio_data, language=language)
            return text
            
        except ImportError:
            _log.error("SpeechRecognition not installed")
            return ""
        except Exception as e:
            _log.error(f"Google STT failed: {e}")
            return ""


class VoiceIntegration:
    """Voice message integration with STT and TTS."""
    
    def __init__(
        self,
        telegram_client: Any = None,
        *,
        tts_engine: str = "edge-tts",
        stt_engine: str = "whisper",
    ) -> None:
        self.telegram = telegram_client
        self.tts = TTSEngine(tts_engine)
        self.stt = STTEngine(stt_engine)
        _log.info(f"Voice integration initialized (TTS: {tts_engine}, STT: {stt_engine})")
    
    async def send_voice_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        voice: str = "default",
        rate: str = "+0%",
    ) -> Optional[str]:
        """Send a voice message to a chat.
        
        Args:
            chat_id: Telegram chat ID
            text: Text to speak
            voice: Voice to use
            rate: Speech rate adjustment
            
        Returns:
            Path to generated audio file
        """
        # Generate audio
        audio_path = await self.tts.synthesize(text, voice=voice, rate=rate)
        
        # Send via Telegram if available
        if self.telegram:
            try:
                await self.telegram.send_file(
                    chat_id,
                    audio_path,
                    voice_note=True,
                )
                _log.info(f"Sent voice message to {chat_id}")
            except Exception as e:
                _log.error(f"Failed to send voice message: {e}")
        
        return audio_path
    
    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str = "en",
    ) -> str:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Expected language
            
        Returns:
            Transcribed text
        """
        return await self.stt.transcribe(audio_path, language=language)
    
    async def transcribe_telegram_voice(
        self,
        message: Any,
        *,
        language: str = "en",
    ) -> str:
        """Transcribe a Telegram voice message.
        
        Args:
            message: Telegram message with voice
            language: Expected language
            
        Returns:
            Transcribed text
        """
        if not self.telegram:
            raise RuntimeError("Telegram client not available")
        
        # Download the voice message
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                audio_path = f.name
            
            await self.telegram.download_media(message, audio_path)
            
            # Transcribe
            text = await self.transcribe(audio_path, language=language)
            
            # Cleanup
            Path(audio_path).unlink(missing_ok=True)
            
            return text
        except Exception as e:
            _log.error(f"Failed to transcribe Telegram voice: {e}")
            return ""
    
    async def recognize_command(
        self,
        audio_path: str,
        *,
        language: str = "en",
    ) -> Optional[VoiceCommand]:
        """Recognize a voice command.
        
        Args:
            audio_path: Path to audio file
            language: Expected language
            
        Returns:
            VoiceCommand object or None
        """
        text = await self.transcribe(audio_path, language=language)
        
        if not text:
            return None
        
        # Parse command from text
        return self._parse_command(text)
    
    def _parse_command(self, text: str) -> VoiceCommand:
        """Parse a command from transcribed text."""
        text_lower = text.lower().strip()
        
        # Command patterns
        command_patterns = {
            "turn_on_light": [
                r"turn on (?:the )?(.+?)(?: light)?$",
                r"switch on (?:the )?(.+?)(?: light)?$",
            ],
            "turn_off_light": [
                r"turn off (?:the )?(.+?)(?: light)?$",
                r"switch off (?:the )?(.+?)(?: light)?$",
            ],
            "set_temperature": [
                r"set (?:the )?temperature to (\d+)",
                r"set (?:the )?thermostat to (\d+)",
            ],
            "send_email": [
                r"send (?:an )?email to (.+)",
                r"email (.+)",
            ],
            "search": [
                r"search (?:for )?(.+)",
                r"look up (.+)",
                r"find (.+)",
            ],
            "play_music": [
                r"play (.+)",
            ],
            "set_reminder": [
                r"remind me to (.+?)(?: at (\d+))?$",
            ],
        }
        
        import re
        
        for command, patterns in command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    params = {}
                    groups = match.groups()
                    
                    if command in ("turn_on_light", "turn_off_light"):
                        params["device"] = groups[0].strip()
                    elif command == "set_temperature":
                        params["temperature"] = int(groups[0])
                    elif command == "send_email":
                        params["recipient"] = groups[0].strip()
                    elif command == "search":
                        params["query"] = groups[0].strip()
                    elif command == "play_music":
                        params["song"] = groups[0].strip()
                    elif command == "set_reminder":
                        params["task"] = groups[0].strip()
                        if len(groups) > 1 and groups[1]:
                            params["time"] = int(groups[1])
                    
                    return VoiceCommand(
                        command=command,
                        confidence=0.8,
                        parameters=params,
                        raw_text=text,
                    )
        
        # Unknown command
        return VoiceCommand(
            command="unknown",
            confidence=0.3,
            parameters={"text": text},
            raw_text=text,
        )
    
    def list_voices(self) -> list[str]:
        """List available TTS voices."""
        return self.tts.list_voices()
