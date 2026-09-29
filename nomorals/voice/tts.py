"""Universal TTS: swappable free/open-source neural backends, one interface.

Port of the owner's universal_tts.py (from the main branch upload),
integrated with the framework:

- Backends: **Bark** (best tag/non-speech support), **XTTS v2** (best
  cloning quality, NON-COMMERCIAL CPML license), **Kokoro** (lightest,
  CPU-friendly, Apache-2.0 — commercial OK).  Each is lazy-imported;
  nothing breaks when a backend is not installed.
- A tag system for emotion / pauses / non-speech sounds that the AI can
  use directly in its own text: [happy] [whisper] [laughs] [pause:300] …
- Voice profiles persisted to disk, with a hard consent gate on cloning
  backends (reference audio is only used when consent_confirmed=True).
- A mood bridge: the partner's mood system maps straight into tags.
- Pure-stdlib WAV writing (no scipy); numpy is only touched by the
  backends themselves, never at import time.

Install one backend on the phone:
    pip install git+https://github.com/suno-ai/bark.git   # Bark
    pip install TTS                                       # XTTS v2
    pip install kokoro                                    # Kokoro
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, List, Optional

__all__ = [
    "VoiceProfile",
    "VoiceLibrary",
    "Segment",
    "TagProcessor",
    "mood_to_tagged_text",
    "available_backends",
    "UniversalTTS",
    "write_wav",
]

# ----------------------------------------------------
# VOICE PROFILE
# ----------------------------------------------------


@dataclass
class VoiceProfile:
    """A voice you can generate speech in.

    ``reference_audio_path`` is only needed for cloning backends (XTTS);
    Bark/Kokoro use built-in preset voices instead.
    ``consent_confirmed`` exists on purpose: cloning backends refuse to
    run without it.  Set it True only for your own voice, a
    public-domain recording, or someone who explicitly agreed to be
    cloned.
    """

    name: str
    reference_audio_path: Optional[str] = None
    preset_id: Optional[str] = None   # for Bark/Kokoro built-in voices
    language: str = "en"
    consent_confirmed: bool = False
    description: str = ""

    def validate_for_cloning(self) -> None:
        if self.reference_audio_path and not self.consent_confirmed:
            raise PermissionError(
                f"VoiceProfile '{self.name}' has reference audio but "
                "consent_confirmed=False. Set it True only if this is "
                "your own voice or you have explicit permission to use it.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reference_audio_path": self.reference_audio_path,
            "preset_id": self.preset_id,
            "language": self.language,
            "consent_confirmed": self.consent_confirmed,
            "description": self.description,
        }


class VoiceLibrary:
    """Upload/register/manage voice profiles, persisted to disk."""

    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = str(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.profiles: dict[str, VoiceProfile] = {}
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self.storage_dir, "index.json")

    def _load_index(self) -> None:
        path = self._index_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                self.profiles = {
                    k: VoiceProfile(**{
                        key: value for key, value in v.items()
                        if key in VoiceProfile.__dataclass_fields__
                    })
                    for k, v in (raw or {}).items()
                }
            except (OSError, ValueError, TypeError):
                self.profiles = {}

    def _save_index(self) -> None:
        try:
            with open(self._index_path(), "w", encoding="utf-8") as fh:
                json.dump(
                    {k: v.to_dict() for k, v in self.profiles.items()},
                    fh, indent=2)
        except OSError:
            pass

    def upload_voice(self, name: str, audio_file_path: str,
                     language: str = "en", consent_confirmed: bool = False,
                     preset_id: Optional[str] = None,
                     description: str = "") -> VoiceProfile:
        """Register a reference clip.  Copies it into managed storage."""
        dest = os.path.join(self.storage_dir, f"{name}.wav")
        try:
            shutil.copy(audio_file_path, dest)
        except OSError as exc:
            raise ValueError(f"cannot copy voice sample: {exc}") from exc
        profile = VoiceProfile(
            name=name,
            reference_audio_path=dest,
            preset_id=preset_id,
            language=language,
            consent_confirmed=consent_confirmed,
            description=description,
        )
        self.profiles[name] = profile
        self._save_index()
        return profile

    def register_preset(self, name: str, preset_id: str,
                        language: str = "en",
                        description: str = "") -> VoiceProfile:
        """Register a built-in preset voice (no reference audio)."""
        profile = VoiceProfile(
            name=name, preset_id=preset_id, language=language,
            description=description)
        self.profiles[name] = profile
        self._save_index()
        return profile

    def remove(self, name: str) -> bool:
        profile = self.profiles.pop(name, None)
        if profile is None:
            return False
        try:
            if profile.reference_audio_path and \
                    os.path.exists(profile.reference_audio_path):
                os.remove(profile.reference_audio_path)
        except OSError:
            pass
        self._save_index()
        return True

    def get(self, name: str) -> Optional[VoiceProfile]:
        return self.profiles.get(name)

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in self.profiles.values():
            out.append({
                "name": p.name,
                "preset_id": p.preset_id,
                "cloning": bool(p.reference_audio_path),
                "consent_confirmed": p.consent_confirmed,
                "language": p.language,
                "description": p.description,
            })
        return out


# ----------------------------------------------------
# TAG SYSTEM: emotion, pauses, non-speech sounds
# ----------------------------------------------------


@dataclass
class Segment:
    text: str
    tags: List[str] = field(default_factory=list)
    pause_after_ms: int = 0


# Tags usable directly in generated text:
#   [happy] [sad] [angry] [jealous] [tired] [excited] [whisper]
#   [annoyed] [serious] [horny]
#   [laughs] [sighs] [gasps] [clears_throat] [chuckles]
#   [pause:300]  -> 300ms silence after the preceding chunk
TAG_PATTERN = re.compile(r"\[([a-z_]+)(?::(\d+))?\]")


class TagProcessor:
    """Parses inline tags out of generated text into structured segments."""

    EMOTION_TAGS = {"happy", "sad", "angry", "jealous", "tired",
                    "excited", "whisper", "serious", "horny", "annoyed"}
    SOUND_TAGS = {"laughs", "sighs", "gasps", "clears_throat", "chuckles"}

    def parse(self, raw_text: str) -> List[Segment]:
        segments: list[Segment] = []
        active_tags: list[str] = []
        pos = 0
        for match in TAG_PATTERN.finditer(raw_text or ""):
            chunk = raw_text[pos:match.start()].strip()
            if chunk:
                segments.append(Segment(text=chunk, tags=list(active_tags)))
            tag_name = match.group(1)
            pause_ms = match.group(2)
            if tag_name == "pause" and pause_ms:
                if segments:
                    segments[-1].pause_after_ms = int(pause_ms)
            elif tag_name in self.SOUND_TAGS:
                segments.append(Segment(text=f"[{tag_name}]", tags=["_sound_"]))
            elif tag_name in self.EMOTION_TAGS:
                active_tags = [tag_name]
            pos = match.end()
        tail = raw_text[pos:].strip()
        if tail:
            segments.append(Segment(text=tail, tags=list(active_tags)))
        return segments

    def to_bark_format(self, segments: List[Segment]) -> str:
        """Bark understands bracket tags natively — reassemble directly."""
        out: list[str] = []
        for seg in segments:
            if seg.tags == ["_sound_"]:
                out.append(seg.text)
                continue
            piece = (f"[whispers] {seg.text}" if "whisper" in seg.tags
                     else seg.text)
            out.append(piece)
            if seg.pause_after_ms:
                out.append("..." if seg.pause_after_ms < 500 else "... ...")
        return " ".join(out)

    def to_plain_with_pauses(self, segments: List[Segment]) -> tuple:
        """For backends with no native tag support (XTTS, Kokoro): strip
        tags to clean text, return pause points so the engine can splice
        real silence in during post-processing."""
        clean_parts: list[str] = []
        pause_points: list[tuple[int, int]] = []
        running_text = ""
        for seg in segments:
            if seg.tags == ["_sound_"]:
                continue
            clean_parts.append(seg.text)
            running_text += seg.text + " "
            if seg.pause_after_ms:
                pause_points.append((len(running_text), seg.pause_after_ms))
        return running_text.strip(), pause_points


def mood_to_tagged_text(text: str, mood: str, mood_level: int) -> str:
    """Bridges the existing mood system straight into TTS tags — call this
    on the bot's reply text before passing it to the engine."""
    mood_map = {
        "happy": "happy", "excited": "excited", "annoyed": "annoyed",
        "angry": "angry", "sad": "sad", "jealous": "jealous",
        "tired": "tired", "stressed": "annoyed", "horny": "horny",
        "neutral": None,
    }
    tag = mood_map.get((mood or "").lower())
    if not tag:
        return text
    prefix = f"[{tag}] " if mood_level >= 4 else ""
    return prefix + text


# ----------------------------------------------------
# BACKENDS
# ----------------------------------------------------


def _spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> list[str]:
    """Which neural backends are actually installed (in preference order)."""
    out = []
    if _spec("bark"):
        out.append("bark")
    if _spec("TTS"):
        out.append("xtts")
    if _spec("kokoro"):
        out.append("kokoro")
    return out


class BarkBackend:
    """Bark: best tag/non-speech support (laughs, sighs, pauses)."""

    name = "bark"
    supports_native_tags = True
    supports_cloning = False   # preset-based, not true cloning
    sample_rate = 24000

    def __init__(self) -> None:
        from bark import generate_audio, preload_models

        preload_models()
        self._generate_audio = generate_audio

    def synthesize(self, text: str,
                   voice: Optional[VoiceProfile]) -> Any:
        history_prompt = voice.preset_id if voice else None
        return self._generate_audio(text, history_prompt=history_prompt)


class XTTSBackend:
    """XTTS v2: best raw cloning quality from ~6s reference audio.

    NON-COMMERCIAL license (CPML) — personal use only.
    """

    name = "xtts"
    supports_native_tags = False
    supports_cloning = True
    sample_rate = 24000

    def __init__(self) -> None:
        from TTS.api import TTS

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def synthesize(self, text: str, voice: Optional[VoiceProfile]) -> Any:
        if voice:
            voice.validate_for_cloning()
        # TTS.api returns (audio_chunks, sampled_rate, length)
        result = self.model.tts(
            text=text,
            speaker_wav=voice.reference_audio_path if voice else None,
            language=voice.language if voice else "en",
        )
        if isinstance(result, tuple) and len(result) >= 1:
            chunks = result[0]
            if isinstance(chunks, (list, tuple)):
                import numpy as np

                return np.concatenate(list(chunks))
            return chunks
        return result


class KokoroBackend:
    """Kokoro: lightest/fastest, runs on CPU, Apache-2.0 (commercial OK).

    No cloning — good fallback on current hardware, or once the bot is
    being sold as a product.
    """

    name = "kokoro"
    supports_native_tags = False
    supports_cloning = False
    sample_rate = 24000

    def __init__(self) -> None:
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="a")

    def synthesize(self, text: str, voice: Optional[VoiceProfile]) -> Any:
        voice_id = (voice.preset_id if voice and voice.preset_id
                    else "af_heart")
        generator = self.pipeline(text, voice=voice_id)
        chunks = [audio for _idx, _len, audio in generator]
        if not chunks:
            raise ValueError("kokoro produced no audio")
        try:
            import numpy as np

            return np.concatenate(chunks)
        except ImportError:
            # flatten the float chunks by hand
            flat: list[float] = []
            for chunk in chunks:
                flat.extend(chunk.tolist() if hasattr(chunk, "tolist")
                            else list(chunk))
            return flat


_BACKENDS = {"bark": BarkBackend, "xtts": XTTSBackend, "kokoro": KokoroBackend}


# ----------------------------------------------------
# WAV WRITING (stdlib only — no scipy)
# ----------------------------------------------------


def _to_int16(samples: Any, sample_rate: int) -> bytes:
    """Normalize samples (float list or numpy array, range ~[-1, 1]) to
    16-bit little-endian PCM bytes."""
    has_astype = hasattr(samples, "astype")
    if has_astype:
        arr = samples.astype("float64") if samples.dtype.kind == "f" \
            else (samples / 32768.0).astype("float64")
        clipped = [max(-1.0, min(1.0, float(v))) for v in arr.tolist()]
    else:
        clipped = [max(-1.0, min(1.0, float(v))) for v in samples]
    return b"".join(struct.pack("<h", int(v * 32767)) for v in clipped)


def write_wav(path: str, samples: Any, sample_rate: int = 24000) -> int:
    """Write 16-bit mono PCM WAV.  Returns bytes written."""
    pcm = _to_int16(samples, sample_rate)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return len(pcm) + 44


# ----------------------------------------------------
# UNIFIED ENGINE
# ----------------------------------------------------


class UniversalTTS:
    """The unified voice engine.

    engine = UniversalTTS(backend="auto", voices_dir=".../voices")
    engine.voices.upload_voice("me", "my_sample.wav", consent_confirmed=True)
    out = engine.speak("[happy] omg baby I missed you [laughs]")
    # out -> {"path": ".../say.wav", "bytes": 240000,
    #         "sample_rate": 24000, "backend": "bark"}
    """

    def __init__(self, backend: str = "auto", voices_dir: str = "voices",
                 default_sample_rate: int = 24000) -> None:
        self.tag_processor = TagProcessor()
        self.voices = VoiceLibrary(voices_dir)
        self.default_sample_rate = default_sample_rate
        self._backend_name = (backend or "auto").lower()
        self._loaded = False
        self._impl: Any = None

    # -- backend lifecycle ---------------------------------------------------
    def _load_backend(self) -> Any:
        if self._loaded:
            return self._impl
        wanted = self._backend_name
        if wanted == "auto":
            available = available_backends()
            if not available:
                raise RuntimeError(
                    "no neural TTS backend installed — pip install one of: "
                    "kokoro (lightest, Apache-2.0) | TTS (XTTS v2, "
                    "non-commercial) | git+https://github.com/suno-ai/bark.git")
            wanted = available[0]
        if wanted not in _BACKENDS:
            raise RuntimeError(
                f"unknown TTS backend {wanted!r}; use one of "
                f"{', '.join(_BACKENDS)} or 'auto'")
        if not _spec({"bark": "bark", "xtts": "TTS",
                      "kokoro": "kokoro"}[wanted]):
            raise RuntimeError(
                f"TTS backend {wanted!r} is not installed on this machine")
        self._impl = _BACKENDS[wanted]()
        self._loaded = True
        return self._impl

    @property
    def backend_name(self) -> str:
        if self._impl is not None:
            return self._impl.name
        if self._backend_name != "auto":
            return self._backend_name
        return available_backends()[0] if available_backends() else ""

    # -- synthesis ------------------------------------------------------------
    def speak(self, tagged_text: str, voice_name: Optional[str] = None,
              out_path: str = "", mood: str = "", mood_level: int = 5) -> dict:
        """Synthesize `tagged_text` (emotion/pause tags allowed) to a WAV."""
        text = mood_to_tagged_text(tagged_text, mood, mood_level) if mood \
            else tagged_text
        voice = self.voices.get(voice_name) if voice_name else None
        backend = self._load_backend()
        segments = self.tag_processor.parse(text)
        sample_rate = getattr(backend, "sample_rate", self.default_sample_rate)

        if backend.supports_native_tags:
            final_text = self.tag_processor.to_bark_format(segments)
            audio = backend.synthesize(final_text, voice)
            audio = self._insert_pauses(audio, [], text, sample_rate)
        else:
            clean_text, pause_points = \
                self.tag_processor.to_plain_with_pauses(segments)
            audio = backend.synthesize(clean_text, voice)
            audio = self._insert_pauses(audio, pause_points, clean_text,
                                        sample_rate)

        path = out_path or os.path.join(
            self.voices.storage_dir, "..",
            f"tts-{int(__import__('time').time() * 1000)}.wav")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        written = write_wav(path, audio, sample_rate)
        return {
            "path": path,
            "bytes": written,
            "sample_rate": sample_rate,
            "backend": backend.name,
            "segments": len(segments),
        }

    def _insert_pauses(self, audio: Any, pause_points: List[tuple],
                       text: str, sample_rate: int) -> Any:
        if not pause_points:
            return audio
        try:
            import numpy as np
        except ImportError:
            return audio  # pausing needs numpy; the backend needs it anyway
        if len(text) == 0 or len(audio) == 0:
            return audio
        chars_per_sample = len(text) / float(len(audio))
        result: list = []
        last_cut = 0
        for char_idx, ms in pause_points:
            sample_idx = min(int(char_idx / chars_per_sample)
                             if chars_per_sample else 0, len(audio))
            result.append(audio[last_cut:sample_idx])
            result.append(np.zeros(int(sample_rate * ms / 1000)))
            last_cut = sample_idx
        result.append(audio[last_cut:])
        return np.concatenate(result)
