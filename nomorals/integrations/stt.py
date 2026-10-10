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
    FASTER_WHISPER = "faster_whisper"  # faster-whisper (CTranslate2, 4x)
    WHISPER_API = "whisper_api"   # OpenAI Whisper API
    GOOGLE = "google"             # Google Speech Recognition
    ASSEMBLYAI = "assemblyai"     # AssemblyAI API


@dataclass
class TranscriptionWord:
    """One word with timing (word_timestamps=True)."""

    word: str
    start: float
    end: float
    probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "start": self.start, "end": self.end,
                "probability": self.probability}


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timing."""

    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str
    words: list[TranscriptionWord] = field(default_factory=list)
    speaker: str = ""  # diarization label, e.g. "SPEAKER_00"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "speaker": self.speaker,
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

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def to_srt(self) -> str:
        """SubRip subtitles from segments."""
        out = []
        for i, seg in enumerate(self.segments, 1):
            out.append(str(i))
            out.append(f"{self._fmt_ts(seg.start)} --> "
                       f"{self._fmt_ts(seg.end)}")
            label = f"[{seg.speaker}] " if seg.speaker else ""
            out.append(f"{label}{seg.text.strip()}")
            out.append("")
        return "\n".join(out)

    def to_vtt(self) -> str:
        """WebVTT subtitles from segments."""
        out = ["WEBVTT", ""]
        for seg in self.segments:
            out.append(f"{self._fmt_ts(seg.start).replace(',', '.')} --> "
                       f"{self._fmt_ts(seg.end).replace(',', '.')}")
            label = f"<v {seg.speaker}>" if seg.speaker else ""
            out.append(f"{label}{seg.text.strip()}")
            out.append("")
        return "\n".join(out)


class SpeechToText:
    """Speech-to-text transcription engine."""
    
    def __init__(
        self,
        engine: str = STTEngine.WHISPER,
        *,
        whisper_model: str = "base",
        api_key: str = "",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.engine = engine
        self.whisper_model = whisper_model
        self.api_key = api_key
        # faster-whisper hardware ladder: cuda→float16, cpu→int8.
        self.device = device
        self.compute_type = compute_type
        self._fw_model: Any = None
        _log.info(f"STT initialized with engine: {engine}")

    def _fw_device(self) -> tuple[str, str]:
        if self.device != "auto":
            dev = self.device
        else:
            try:
                import torch  # type: ignore[import]
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        if self.compute_type != "auto":
            ct = self.compute_type
        else:
            ct = "float16" if dev == "cuda" else "int8"
        return dev, ct
    
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
        word_timestamps: bool = False,
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> TranscriptionResult:
        """Transcribe audio with full metadata.
        
        Args:
            audio_path: Path to audio file
            language: Expected language code (pinned — avoids
                mis-detection on short clips)
            prompt: Optional prompt to guide transcription
            word_timestamps: Per-word timing (faster-whisper)
            vad_filter: Silero VAD — strips silence, kills the
                hallucinated-text-in-silence failure mode
            beam_size: Decoding beam (5 = standard accuracy/speed trade)
            
        Returns:
            TranscriptionResult with segments and timing
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.engine == STTEngine.WHISPER:
            return await self._transcribe_whisper(audio_path, language, prompt)
        elif self.engine == STTEngine.FASTER_WHISPER:
            return await self._transcribe_faster_whisper(
                audio_path, language, prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter, beam_size=beam_size)
        elif self.engine == STTEngine.WHISPER_API:
            return await self._transcribe_whisper_api(audio_path, language, prompt)
        elif self.engine == STTEngine.GOOGLE:
            return await self._transcribe_google(audio_path, language)
        elif self.engine == STTEngine.ASSEMBLYAI:
            return await self._transcribe_assemblyai(audio_path, language)
        else:
            raise ValueError(f"Unknown STT engine: {self.engine}")

    async def transcribe_diarized(
        self,
        audio_path: str,
        *,
        language: str = "en",
        min_speakers: int = 1,
        max_speakers: int = 4,
        hf_token: str = "",
    ) -> TranscriptionResult:
        """Transcribe + speaker diarization (WhisperX-style).

        Needs ``whisperx`` + a HuggingFace token with access to the
        pyannote diarization model. Raises a clear error when the
        stack isn't installed — never a fake transcript.
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            import whisperx  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "diarization needs the whisperx stack: "
                "pip install whisperx  (plus a HuggingFace token with "
                "pyannote/speaker-diarization access)") from exc
        device, compute_type = self._fw_device()
        model = whisperx.load_model(self.whisper_model, device,
                                    compute_type=compute_type,
                                    language=language)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token or None, device=device)
        diarize_segments = diarize_model(audio, min_speakers=min_speakers,
                                         max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        segments = []
        for seg in result.get("segments", []):
            words = [TranscriptionWord(
                word=w.get("word", ""), start=w.get("start", 0.0),
                end=w.get("end", 0.0))
                for w in seg.get("words", []) or []]
            segments.append(TranscriptionSegment(
                start=seg.get("start", 0.0), end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(), words=words,
                speaker=seg.get("speaker", "")))
        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=result.get("language", language),
            duration=segments[-1].end if segments else 0.0,
            segments=segments,
            engine="whisperx",
        )
    
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
    
    async def _transcribe_faster_whisper(
        self,
        audio_path: str,
        language: str,
        prompt: str,
        *,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> TranscriptionResult:
        """Transcribe with faster-whisper (CTranslate2, ~4x faster).

        Hallucination guards on by default: ``vad_filter=True`` strips
        silence (Whisper invents text in long quiet stretches),
        ``condition_on_previous_text=False`` stops drift on long-form
        audio. Language is pinned, not detected.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper not installed: pip install faster-whisper"
            ) from exc
        device, compute_type = self._fw_device()
        if self._fw_model is None:
            _log.info("Loading faster-whisper %s (%s/%s)",
                      self.whisper_model, device, compute_type)
            self._fw_model = WhisperModel(self.whisper_model,
                                          device=device,
                                          compute_type=compute_type)
        import asyncio
        loop = asyncio.get_running_loop()
        segments_iter, info = await loop.run_in_executor(
            None, lambda: self._fw_model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
            ))
        segments = []
        for seg in segments_iter:  # generator — must be consumed
            words = []
            if word_timestamps:
                for w in seg.words or []:
                    words.append(TranscriptionWord(
                        word=w.word, start=w.start, end=w.end,
                        probability=w.probability))
            segments.append(TranscriptionSegment(
                start=seg.start, end=seg.end,
                text=seg.text.strip(), words=words))
        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=getattr(info, "language", language),
            duration=segments[-1].end if segments else 0.0,
            segments=segments,
            engine=STTEngine.FASTER_WHISPER,
        )

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
