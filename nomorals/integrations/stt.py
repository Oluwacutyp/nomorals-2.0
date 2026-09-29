"""Speech-to-text (STT) engine - transcribe voice messages.

Supports multiple backends:
- Whisper (OpenAI's open-source model, runs locally)
- Whisper API (cloud-based, requires API key)
- Google Speech Recognition (free, requires internet)

Usage:
    stt = SpeechToText(engine="whisper")
    
    # Transcribe audio file
    text = await stt.transcribe("/path/to/audio.ogg", language="en")
    
    # Transcribe with timestamps
    result = await stt.transcribe_with_timestamps("/path/to/audio.ogg")
    print(result.text)
    for segment in result.segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger

__all__ = [
    "SpeechToText",
    "TranscriptionResult",
    "TranscriptionSegment",
    "STTEngine",
]

_log = get_logger(__name__)


class STTEngine:
    """Supported STT engines."""
    
    WHISPER = "whisper"           # Local Whisper model
    WHISPER_API = "whisper_api"   # OpenAI Whisper API
    GOOGLE = "google"             # Google Speech Recognition
    ASSEMBLYAI = "assemblyai"     # AssemblyAI API


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timing."""
    
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    
    text: str
    language: str = "en"
    duration: float = 0.0
    segments: list[TranscriptionSegment] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [s.to_dict() for s in self.segments],
            "confidence": self.confidence,
            "engine": self.engine,
        }


class SpeechToText:
    """Speech-to-text transcription engine."""
    
    def __init__(
        self,
        engine: str = STTEngine.WHISPER,
        *,
        whisper_model: str = "base",
        api_key: str = "",
    ) -> None:
        self.engine = engine
        self.whisper_model = whisper_model
        self.api_key = api_key
        _log.info(f"STT initialized with engine: {engine}")
    
    async def transcribe(
        self,
        audio_path: str,
        *,
        language: str = "en",
        prompt: str = "",
    ) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (ogg, mp3, wav, etc.)
            language: Expected language code
            prompt: Optional prompt to guide transcription
            
        Returns:
            Transcribed text
        """
        result = await self.transcribe_full(
            audio_path,
            language=language,
            prompt=prompt,
        )
        return result.text
    
    async def transcribe_full(
        self,
        audio_path: str,
        *,
        language: str = "en",
        prompt: str = "",
    ) -> TranscriptionResult:
        """Transcribe audio with full metadata.
        
        Args:
            audio_path: Path to audio file
            language: Expected language code
            prompt: Optional prompt to guide transcription
            
        Returns:
            TranscriptionResult with segments and timing
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.engine == STTEngine.WHISPER:
            return await self._transcribe_whisper(audio_path, language, prompt)
        elif self.engine == STTEngine.WHISPER_API:
            return await self._transcribe_whisper_api(audio_path, language, prompt)
        elif self.engine == STTEngine.GOOGLE:
            return await self._transcribe_google(audio_path, language)
        elif self.engine == STTEngine.ASSEMBLYAI:
            return await self._transcribe_assemblyai(audio_path, language)
        else:
            raise ValueError(f"Unknown STT engine: {self.engine}")
    
    async def _transcribe_whisper(
        self,
        audio_path: str,
        language: str,
        prompt: str,
    ) -> TranscriptionResult:
        """Transcribe using local Whisper model."""
        try:
            import whisper
            
            _log.info(f"Loading Whisper model: {self.whisper_model}")
            model = whisper.load_model(self.whisper_model)
            
            _log.info(f"Transcribing: {audio_path}")
            result = model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                initial_prompt=prompt if prompt else None,
            )
            
            # Extract segments with timing
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptionSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                ))
            
            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language", language),
                duration=result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
                segments=segments,
                confidence=1.0,  # Whisper doesn't provide confidence
                engine=STTEngine.WHISPER,
            )
            
        except ImportError:
            _log.error("Whisper not installed. Run: pip install openai-whisper")
            raise
        except Exception as e:
            _log.error(f"Whisper transcription failed: {e}")
            raise
    
    async def _transcribe_whisper_api(
        self,
        audio_path: str,
        language: str,
        prompt: str,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API."""
        if not self.api_key:
            raise ValueError("OpenAI API key required for Whisper API")
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    prompt=prompt if prompt else None,
                    response_format="verbose_json",
                )
            
            # Parse response
            segments = []
            for seg in response.segments or []:
                segments.append(TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                ))
            
            return TranscriptionResult(
                text=response.text.strip(),
                language=response.language,
                duration=response.duration,
                segments=segments,
                confidence=1.0,
                engine=STTEngine.WHISPER_API,
            )
            
        except ImportError:
            _log.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            _log.error(f"Whisper API transcription failed: {e}")
            raise
    
    async def _transcribe_google(
        self,
        audio_path: str,
        language: str,
    ) -> TranscriptionResult:
        """Transcribe using Google Speech Recognition."""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # Convert to WAV if needed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Use ffmpeg to convert if needed
            import subprocess
            subprocess.run(
                ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_path],
                check=True,
                capture_output=True,
            )
            
            # Transcribe
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=language)
            
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
            
            return TranscriptionResult(
                text=text,
                language=language,
                duration=0,  # Google doesn't provide duration
                segments=[],
                confidence=1.0,
                engine=STTEngine.GOOGLE,
            )
            
        except ImportError:
            _log.error("SpeechRecognition not installed. Run: pip install SpeechRecognition")
            raise
        except Exception as e:
            _log.error(f"Google STT failed: {e}")
            raise
    
    async def _transcribe_assemblyai(
        self,
        audio_path: str,
        language: str,
    ) -> TranscriptionResult:
        """Transcribe using AssemblyAI API."""
        if not self.api_key:
            raise ValueError("AssemblyAI API key required")
        
        try:
            import assemblyai as aai
            
            aai.settings.api_key = self.api_key
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI error: {transcript.error}")
            
            # Extract segments
            segments = []
            for utterance in transcript.utterances or []:
                segments.append(TranscriptionSegment(
                    start=utterance.start / 1000,  # Convert ms to seconds
                    end=utterance.end / 1000,
                    text=utterance.text.strip(),
                ))
            
            return TranscriptionResult(
                text=transcript.text,
                language=language,
                duration=transcript.audio_duration,
                segments=segments,
                confidence=transcript.confidence,
                engine=STTEngine.ASSEMBLYAI,
            )
            
        except ImportError:
            _log.error("AssemblyAI not installed. Run: pip install assemblyai")
            raise
        except Exception as e:
            _log.error(f"AssemblyAI transcription failed: {e}")
            raise
