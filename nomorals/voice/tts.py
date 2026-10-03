"""Universal TTS: swappable free/open-source neural backends, one interface.

Port of the owner's universal_tts.py (from the main branch upload),
integrated with the framework:

- Backends: **Chatterbox** (best free cloning — MIT, blind-test
  winner vs ElevenLabs, 23 languages, native ``[laugh]``/``[chuckle]``/
  ``[cough]`` tags on Turbo/Nano, Nano runs 3× realtime on CPU),
  **F5-TTS** (highest-fidelity single-shot cloning — MIT code but
  CC-BY-NC pretrained checkpoints), **OmniVoice** (600+ languages,
  Apache-2.0, RTF 0.025), **Qwen3-TTS** (expressive Apache-2.0,
  native ``[laugh]``/``[sigh]`` tags, 0.6B CPU-friendly), **Bark**
  (best tag/non-speech support), **XTTS v2** (best cloning quality,
  NON-COMMERCIAL CPML license), **Kokoro** (lightest, CPU-friendly,
  Apache-2.0 — commercial OK), **Piper** (best on-device phone TTS,
  ONNX/CPU, MIT), **CosyVoice** (multilingual + instruction-driven
  paralinguistics: laughter, breaths, emphasis — MIT, fetched from
  HuggingFace), **Orpheus** (LLM-class expressivity, native
  ``<laugh>``/``<sigh>``/``<cough>`` tags, zero-shot cloning —
  Apache-2.0), **Dia** (nari-labs dialogue model, native ``(laughs)``
  / ``(coughs)`` / ``(sneezes)`` non-verbals — Apache-2.0, GPU-only),
  **hf-endpoint** (Fish Audio S2 via HF serverless, no local weights),
  **system** (the OS's own speech service — ``say``/``espeak-ng``/
  PowerShell System.Speech — pure stdlib, zero pip packages, dead-last
  fallback so the module works with nothing installed).
  Each is lazy-imported; nothing breaks when a backend is not installed.
- A tag system for emotion / pauses / non-speech sounds that the AI can
  use directly in its own text: [happy] [whisper] [laughs] [pause:300] …
- The **director** (``nomorals/voice/director.py``) turns plain text
  into a performance script — 30 vocal bursts (laughs, giggles, coughs,
  sneezes, sighs, gasps, yawns, whistles, …), 53 emotion tags, fillers,
  stutters, pauses, emphasis, pacing — and each backend renders the
  canonical markup in its own native vocabulary. ``UniversalTTS.perform()``
  is the one-call path: text in, human-sounding wav out.
- Voice profiles persisted to disk, with a hard consent gate on cloning
  backends (reference audio is only used when consent_confirmed=True).
  Multiple reference samples per voice are blended where the backend
  supports it (XTTS averages the embeddings).
- A mood bridge: the partner's mood system maps straight into tags.
- Pure-stdlib WAV writing (no scipy); numpy is only touched by the
  backends themselves, never at import time.

Install one backend on the phone:
    pip install piper-tts                          # Piper (CPU, MIT)
    python -m piper.download_voices en_US-lessac-medium
    pip install kokoro                             # Kokoro (CPU)
    pip install chatterbox-tts                     # Chatterbox (MIT)
    pip install qwen-tts                           # Qwen3-TTS (0.6B)
    pip install f5-tts                             # F5-TTS (needs ref clip)
    pip install TTS                                # XTTS v2
    pip install cosyvoice                          # CosyVoice
    pip install orpheus-speech                     # Orpheus (GPU)
    pip install omnivoice                          # OmniVoice (GPU)
    pip install git+https://github.com/suno-ai/bark.git   # Bark
    nm voice fetch --backend cosyvoice              # pull the weights
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import struct
import sys
import wave
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .director import CANONICAL_BURSTS

_log = logging.getLogger(__name__)

__all__ = [
    "VoiceProfile",
    "VoiceLibrary",
    "Segment",
    "TagProcessor",
    "mood_to_tagged_text",
    "available_backends",
    "probe_reference_audio",
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
    #: Transcript of the reference clip. Zero-shot backends (CosyVoice)
    #: need it to clone; XTTS does not.
    prompt_text: str = ""
    #: Extra reference clips for the same voice. Cloning backends that
    #: accept several samples (XTTS averages the speaker embeddings)
    #: blend them; single-sample backends use the first clip.
    extra_samples: list = field(default_factory=list)

    def validate_for_cloning(self) -> None:
        if self.reference_audio_path and not self.consent_confirmed:
            raise PermissionError(
                f"VoiceProfile '{self.name}' has reference audio but "
                "consent_confirmed=False. Set it True only if this is "
                "your own voice or you have explicit permission to use it.")

    @property
    def reference_audios(self) -> list[str]:
        """All reference clips: the primary plus any extra samples."""
        auds = ([self.reference_audio_path]
                if self.reference_audio_path else [])
        auds.extend(a for a in self.extra_samples if a)
        return auds

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reference_audio_path": self.reference_audio_path,
            "preset_id": self.preset_id,
            "language": self.language,
            "consent_confirmed": self.consent_confirmed,
            "description": self.description,
            "prompt_text": self.prompt_text,
            "extra_samples": list(self.extra_samples),
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
        except OSError as e:
            _log.warning("could not persist voice profile index: %s", e)

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

    def add_sample(self, name: str, audio_file_path: str) -> VoiceProfile:
        """Attach an extra reference clip to a voice.

        Cloning backends that accept several samples (XTTS) blend them
        for a more stable clone; single-sample backends ignore extras.
        The consent gate still applies to every clip.
        """
        profile = self.profiles.get(name)
        if profile is None:
            raise KeyError(f"unknown voice profile {name!r}")
        n = len(profile.extra_samples) + 1
        dest = os.path.join(self.storage_dir, f"{name}_sample{n}.wav")
        try:
            shutil.copy(audio_file_path, dest)
        except OSError as exc:
            raise ValueError(f"cannot copy voice sample: {exc}") from exc
        profile.extra_samples.append(dest)
        self._save_index()
        return profile

    def set_transcript(self, name: str, transcript: str) -> VoiceProfile:
        """Set/replace the prompt transcript of a voice's reference clip.

        Zero-shot backends (CosyVoice, Dia voice-clone) need the exact
        words spoken in the reference audio to clone well.
        """
        profile = self.profiles.get(name)
        if profile is None:
            raise KeyError(f"unknown voice profile {name!r}")
        profile.prompt_text = (transcript or "").strip()
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
        except OSError:  # noqa: E103 - reference file already gone; index save below is the source of truth
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
#   [annoyed] [serious] [horny] [terrified] [ecstatic] …
#   [laughs] [sighs] [gasps] [clears_throat] [chuckles] + every
#   canonical director burst ([sneeze] [bellylaugh] [whistle] …)
#   [pause:300]  -> 300ms silence after the preceding chunk
TAG_PATTERN = re.compile(r"\[([a-z_]+)(?::(\d+))?\]")

#: Legacy Bark-style sound tags, kept for backward compatibility, plus
#: every canonical director burst name (mapped to Bark-native in
#: ``to_bark_format``).
_LEGACY_SOUND_TAGS = {"laughs", "sighs", "gasps", "clears_throat",
                      "chuckles"}


def probe_reference_audio(path: str) -> dict[str, Any]:
    """Inspect a reference clip before cloning (stdlib WAV probe).

    Returns ``{"ok", "seconds", "sample_rate", "channels", "warnings"}``.
    Warnings flag the usual clone-killers: clips under 3s, over 30s,
    telephone-grade sample rates, stereo files, or unreadable data.
    Never raises on bad input — ``ok`` is False and ``warnings``
    explains why.
    """
    info: dict[str, Any] = {
        "path": path, "ok": False, "seconds": 0.0,
        "sample_rate": 0, "channels": 0, "warnings": [],
    }
    try:
        with wave.open(path, "rb") as wav:
            n_frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
    except (wave.Error, OSError, EOFError) as exc:
        info["warnings"].append(f"unreadable as WAV: {exc}")
        return info
    seconds = (n_frames / rate) if rate else 0.0
    info.update(ok=True, seconds=round(seconds, 2),
                sample_rate=rate, channels=channels)
    warns = info["warnings"]
    if seconds < 3:
        warns.append("very short (<3s) — cloning will be unstable; "
                     "6–30s of clean speech is ideal")
    elif seconds > 30:
        warns.append("longer than 30s — most zero-shot backends only "
                     "need ~10s; trim it to save memory")
    if rate and rate < 16000:
        warns.append(f"low sample rate ({rate}Hz) — telephone-grade "
                     "audio clones poorly")
    if channels > 1:
        warns.append("stereo — will be mixed down to mono")
    return info


class TagProcessor:
    """Parses inline tags out of generated text into structured segments."""

    EMOTION_TAGS = {"happy", "sad", "angry", "jealous", "tired",
                    "excited", "whisper", "serious", "horny", "annoyed",
                    "nervous", "scared", "proud", "sarcastic", "curious",
                    "surprised", "thoughtful", "confident", "empathetic",
                    "reassuring", "tender", "playful", "nostalgic",
                    "terrified", "ecstatic", "deadpan", "smug", "wistful",
                    "hopeful", "triumphant", "desperate", "panicked",
                    "disgusted", "amused", "relieved", "eager", "hesitant",
                    "skeptical", "whispering", "shouting", "singing",
                    "muttering", "soft", "loud", "crying", "screaming",
                    "panting", "chanting", "aside"}
    #: Legacy Bark-style tags plus every canonical director burst name.
    SOUND_TAGS = _LEGACY_SOUND_TAGS | set(CANONICAL_BURSTS)

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
        """Bark understands bracket tags natively — reassemble directly.

        Canonical director bursts (``[sneeze]``) are mapped to their
        Bark-native form (``[sneezes]``); legacy tags (``[laughs]``)
        pass through untouched.
        """
        from .director import _BARK_BURSTS

        out: list[str] = []
        for seg in segments:
            if seg.tags == ["_sound_"]:
                tag = seg.text.strip("[]").lower()
                out.append(_BARK_BURSTS.get(tag, seg.text))
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
    # an already-imported module is definitionally installed
    if name in sys.modules:
        return True
    if not name:  # e.g. "system": no pip package, probed separately
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> list[str]:
    """Which backends are actually usable (in preference order).

    Quality-first among the fully-free licenses: Chatterbox (MIT,
    blind-test winner vs ElevenLabs) → F5-TTS → OmniVoice → Qwen3-TTS →
    Orpheus → Dia → XTTS → CosyVoice → Kokoro → Piper → Bark →
    hf-endpoint (cloud, needs no local weights) → system (the OS's own
    speech service: pure stdlib, zero pip packages, dead last).
    CPU-only boxes land on Kokoro or Piper; GPU boxes land on
    Chatterbox; a bare box with espeak-ng/say still talks via system.
    """
    order = [("chatterbox", "chatterbox"), ("f5tts", "f5_tts"),
             ("omnivoice", "omnivoice"), ("qwen3tts", "qwen_tts"),
             ("orpheus", "orpheus_tts"), ("dia", "dia"), ("xtts", "TTS"),
             ("cosyvoice", "cosyvoice"), ("kokoro", "kokoro"),
             ("piper", "piper"), ("bark", "bark"),
             ("hf-endpoint", "huggingface_hub")]
    found = [name for name, spec in order if _spec(spec)]
    if SystemTTSBackend.available():  # stdlib-only fallback, dead last
        found.append("system")
    return found


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
                   voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
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

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if voice:
            voice.validate_for_cloning()
        # XTTS averages the speaker embedding over several reference
        # clips when given a list — multi-sample voices blend here.
        wavs = voice.reference_audios if voice else []
        result = self.model.tts(
            text=text,
            speaker_wav=wavs[0] if len(wavs) == 1 else (wavs or None),
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

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
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


class CosyVoiceBackend:
    """CosyVoice: multilingual + instruction-driven paralinguistics.

    Weights fetched from HuggingFace (``nm voice fetch --backend
    cosyvoice``), MIT license. Two modes:

    - **instruct** (default): ``inference_instruct`` with a preset speaker
      id and a natural-language instruction — this is where the
      director's cues land: ``[laughter]`` / ``[breath]`` bursts,
      ``<laughter>`` / ``<strong>`` spans, emotion and rate.
    - **zero-shot**: ``inference_zero_shot`` clones a voice profile's
      reference clip (3–10s). Needs ``voice.prompt_text`` — the
      transcript of the reference clip — plus the consent gate.

    API follows the documented ``cosyvoice.cli.cosyvoice.CosyVoice``
    interface (300M-Instruct / CosyVoice-3). Not live-tested here —
    needs a GPU box to verify; the director renderers and the fetch
    plumbing are covered by tests with a stubbed ``cosyvoice`` module.
    """

    name = "cosyvoice"
    supports_native_tags = True    # instruct tokens, see director.py
    supports_cloning = True        # zero-shot from reference clip
    sample_rate = 22050

    #: Where ``nm voice fetch`` puts the weights; override with
    #: ``COSYVOICE_MODEL_DIR``.
    @staticmethod
    def _default_model_dir() -> str:
        from .fetch import default_cache_dir

        return default_cache_dir("cosyvoice")

    def __init__(self, model_dir: str = "") -> None:
        from cosyvoice.cli.cosyvoice import CosyVoice

        self.model_dir = (
            model_dir
            or os.environ.get("COSYVOICE_MODEL_DIR", "")
            or self._default_model_dir()
        )
        if not os.path.isdir(self.model_dir):
            raise RuntimeError(
                f"CosyVoice weights not found at {self.model_dir} — "
                "run: nm voice fetch --backend cosyvoice")
        self.cosyvoice = CosyVoice(self.model_dir)

    def _collect(self, gen: Any) -> Any:
        chunks = []
        for out in gen:
            chunks.append(out["tts_speech"])
        if not chunks:
            raise ValueError("cosyvoice produced no audio")
        try:
            import torch

            return torch.cat([c.reshape(-1) for c in chunks],
                             dim=0).cpu().numpy()
        except ImportError:
            import numpy as np

            return np.concatenate(
                [np.asarray(c).reshape(-1) for c in chunks])

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        cv = self.cosyvoice
        ref = voice.reference_audio_path if voice else None
        if ref:
            if voice is not None:
                voice.validate_for_cloning()
            prompt_text = (voice.prompt_text if voice else "").strip()
            if not prompt_text:
                raise ValueError(
                    f"voice '{voice.name}' needs prompt_text (the transcript "
                    "of its reference clip) for CosyVoice zero-shot cloning")
            gen = cv.inference_zero_shot(text, prompt_text, ref, stream=False)
        else:
            spk_id = (voice.preset_id if voice and voice.preset_id
                      else "英文女")
            gen = cv.inference_instruct(
                text, spk_id, instruct or "Speak naturally.", stream=False)
        return self._collect(gen)


class DiaBackend:
    """Dia (nari-labs): dialogue-grade paralinguistics, Apache-2.0.

    Native parenthesized non-verbals — ``(laughs)`` ``(coughs)``
    ``(sighs)`` ``(sneezes)`` ``(whistles)`` ``(groans)`` … — inside a
    ``[S1]`` / ``[S2]`` dialogue script (the renderer adds the speaker
    prefix). Voice cloning via an audio prompt, per the official
    ``example/voice_clone.py``.

    GPU-ONLY: ~10GB VRAM in fp16. Not for CPU boxes or phones — but
    when a GPU is around it is the most human dialogue renderer here.
    """

    name = "dia"
    supports_native_tags = True    # paren tags, see director.render_dia
    supports_cloning = True        # via audio prompt
    sample_rate = 44100

    def __init__(self, model_id: str = "") -> None:
        try:
            from dia.model import Dia
        except ImportError as exc:
            raise RuntimeError(
                "dia backend needs the dia package (GPU-only, ~10GB "
                "VRAM): pip install git+https://github.com/nari-labs/"
                "dia.git") from exc
        self.model_id = (model_id or os.environ.get("DIA_MODEL_ID", "")
                         or "nari-labs/Dia-1.6B-0626")
        self.model = Dia.from_pretrained(self.model_id)

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        ref = voice.reference_audio_path if voice else None
        if ref:
            if voice is not None:
                voice.validate_for_cloning()
            try:
                out = self.model.generate(text, audio_prompt=ref)
            except TypeError:
                # older dia releases without audio-prompt cloning
                out = self.model.generate(text)
        else:
            out = self.model.generate(text)
        try:
            import numpy as np

            return np.asarray(out, dtype=np.float32).reshape(-1)
        except ImportError:
            return [float(v) for v in out]


class OrpheusBackend:
    """Orpheus (Canopy Labs): LLM-class expressive TTS, Apache-2.0.

    A Llama-3B fine-tune that emits SNAC audio codes, with native
    angle-bracket emotion tags — ``<laugh>`` ``<chuckle>`` ``<sigh>``
    ``<cough>`` ``<sniffle>`` ``<groan>`` ``<yawn>`` ``<gasp>`` —
    zero-shot voice cloning, 8 preset voices (``tara``, ``josh``,
    ``emma``, …), and ~200ms streaming latency.

    Install: ``pip install orpheus-speech`` (vLLM under the hood —
    needs a GPU). CPU path: run a GGUF quant (2–3.5 GB) under
    llama.cpp / Orpheus-FastAPI and point this backend at it later.
    Not live-tested here — needs a GPU box; the renderer and the
    plumbing are covered by tests with a stubbed ``orpheus_tts``.
    """

    name = "orpheus"
    supports_native_tags = True    # angle tags, see director.render_orpheus
    supports_cloning = True        # zero-shot
    sample_rate = 24000

    def __init__(self, model_id: str = "") -> None:
        try:
            from orpheus_tts import OrpheusModel
        except ImportError as exc:
            raise RuntimeError(
                "orpheus backend needs: pip install orpheus-speech "
                "(GPU via vLLM) — or serve a GGUF quant over HTTP for "
                "the CPU path") from exc
        self.model_id = (model_id or os.environ.get("ORPHEUS_MODEL_ID", "")
                         or "canopylabs/orpheus-tts-0.1-finetune-prod")
        self.model = OrpheusModel(model_name=self.model_id)

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if voice is not None:
            voice.validate_for_cloning()
        voice_id = (voice.preset_id if voice and voice.preset_id
                    else "tara")
        # generate_speech yields 16-bit mono PCM chunks at 24kHz
        pcm = b"".join(
            self.model.generate_speech(prompt=text, voice=voice_id))
        if not pcm:
            raise ValueError("orpheus produced no audio")
        vals = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
        return [v / 32768.0 for v in vals]


class HFEndpointBackend:
    """Hugging Face Inference Endpoints / serverless Inference API.

    Reach a TTS model *without* local weights — the no-GPU path:

    1. **Serverless** (default): ``huggingface_hub.InferenceClient`` with a
       model id, e.g. ``fishaudio/s2-pro`` — HF routes to shared GPUs.
       Needs ``pip install huggingface_hub``; ``HF_TOKEN`` env only for
       gated/private models.
    2. **Dedicated endpoint**: set ``HF_TTS_ENDPOINT_URL`` to an
       Inference Endpoint you deployed (CosyVoice / Fish / whatever
       container you run). The backend POSTs ``{"inputs": text,
       "parameters": {...}}`` and expects audio bytes back.

    The director's script is rendered fish-native (``render_fish``) when
    the model id mentions fish — S2 speaks canonical tags natively —
    otherwise plain speakable text (``render_plain``), since endpoint
    APIs speak plain text. Auth is env-only; never paste tokens in chat.
    """

    name = "hf-endpoint"
    sample_rate = 24000

    @property
    def supports_native_tags(self) -> bool:
        # Fish models speak the director's free-form [tags] natively
        # (render_fish keeps them inline); anything else gets plain text.
        return self._is_fish()

    supports_cloning = True        # via endpoint voice/reference params

    def __init__(self, model: str = "", endpoint_url: str = "") -> None:
        try:
            from huggingface_hub import InferenceClient  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "hf-endpoint backend needs huggingface_hub: "
                "pip install huggingface_hub") from exc
        self.model = (model or os.environ.get("HF_TTS_MODEL", "")
                      or "fishaudio/s2-pro")
        self.endpoint_url = (endpoint_url
                             or os.environ.get("HF_TTS_ENDPOINT_URL", ""))

    def _is_fish(self) -> bool:
        return "fish" in self.model.lower()

    def _client(self) -> Any:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN", "") or None
        if self.endpoint_url:
            return InferenceClient(base_url=self.endpoint_url, token=token)
        return InferenceClient(model=self.model, token=token)

    @staticmethod
    def _decode_wav(blob: bytes) -> tuple[list, int]:
        """WAV bytes → (float samples, sample_rate). Stdlib only.

        Handles 8/16/32-bit PCM, mono or multi-channel (mixed to mono).
        """
        import io
        import struct as _struct

        with wave.open(io.BytesIO(blob), "rb") as wav:
            n = wav.getnframes()
            raw = wav.readframes(n)
            width = wav.getsampwidth()
            rate = wav.getframerate()
            channels = wav.getnchannels()
        if width == 1:
            # 8-bit WAV PCM is unsigned — offset to signed
            vals = [b - 128 for b in raw]
            scale = 128.0
        elif width == 2:
            vals = list(_struct.unpack("<%dh" % (len(raw) // 2), raw))
            scale = 32768.0
        elif width == 4:
            vals = list(_struct.unpack("<%di" % (len(raw) // 4), raw))
            scale = float(1 << 31)
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")
        mono = [sum(vals[i:i + channels]) / (channels * scale)
                for i in range(0, len(vals), channels)]
        return mono, rate

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        client = self._client()
        params: dict = {}
        if instruct:
            params["instruct"] = instruct
        if voice is not None:
            voice.validate_for_cloning()
            if voice.preset_id:
                params["speaker"] = voice.preset_id
            if voice.language:
                params["language"] = voice.language
        if self.endpoint_url:
            blob = client.post(json={"inputs": text, "parameters": params})
        else:
            blob = client.text_to_speech(text)
        if not isinstance(blob, (bytes, bytearray)) or not blob:
            raise ValueError("hf-endpoint returned no audio bytes")
        try:
            samples, rate = self._decode_wav(bytes(blob))
        except Exception as exc:
            head = bytes(blob)[:100]
            raise ValueError(
                "hf-endpoint did not return WAV audio "
                f"(first bytes: {head!r})") from exc
        self.sample_rate = rate
        return samples


class ChatterboxBackend:
    """Chatterbox (Resemble AI): the best FREE cloning TTS today, MIT.

    Zero-shot cloning from ~5–10s of reference audio, 23 languages,
    emotion ``exaggeration`` control — preferred over ElevenLabs in
    blind Podonos evaluations (Turbo vs ElevenLabs Turbo v2.5).

    Three variants, via the ``variant`` arg or ``CHATTERBOX_VARIANT``:

    - ``multilingual`` (default): Chatterbox Multilingual V3 (500M),
      per-voice ``language_id`` (en/fr/de/es/zh/…).
    - ``turbo``: English only, distilled one-step decoder (~sub-200ms
      latency), native paralinguistic tags ``[laugh]`` ``[chuckle]``
      ``[cough]`` — the director's ``render_chatterbox`` speaks them.
    - ``nano``: same as turbo at 110M — **3× realtime on 8 CPU cores**,
      the on-device path when no GPU exists.

    ``pip install chatterbox-tts`` (Python 3.11+). Weights auto-download
    from HuggingFace on first use. ``CHATTERBOX_DEVICE`` overrides the
    cuda/cpu auto-detect.
    """

    name = "chatterbox"
    supports_cloning = True
    sample_rate = 24000

    _VARIANTS = ("multilingual", "turbo", "nano")

    @property
    def supports_native_tags(self) -> bool:
        # turbo/nano speak [laugh]/[chuckle]/[cough] natively; the
        # multilingual V3 model does not — the engine renders plain
        # text for it instead
        return self.variant in ("turbo", "nano")

    def __init__(self, variant: str = "") -> None:
        self.variant = (variant
                        or os.environ.get("CHATTERBOX_VARIANT", "")
                        or "multilingual").lower()
        if self.variant not in self._VARIANTS:
            raise ValueError(
                f"unknown chatterbox variant {self.variant!r}; "
                f"use one of {', '.join(self._VARIANTS)}")
        self.device = os.environ.get("CHATTERBOX_DEVICE", "")
        if not self.device:
            try:
                import torch

                self.device = ("cuda" if torch.cuda.is_available()
                               else "cpu")
            except ImportError:
                self.device = "cpu"
        try:
            if self.variant == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                self.model = ChatterboxMultilingualTTS.from_pretrained(
                    device=self.device, t3_model="v3")
            else:
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                self.model = ChatterboxTurboTTS.from_pretrained(
                    device=self.device, nano=(self.variant == "nano"))
        except ImportError as exc:
            raise RuntimeError(
                "chatterbox backend needs: pip install chatterbox-tts "
                "(Python 3.11+)") from exc
        self.sample_rate = int(getattr(self.model, "sr", 24000) or 24000)

    @staticmethod
    def _language_id(voice: Optional[VoiceProfile]) -> str:
        code = (voice.language if voice else "") or "en"
        return code.replace("_", "-").split("-")[0].lower() or "en"

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if voice is not None:
            voice.validate_for_cloning()
        ref = voice.reference_audio_path if voice else None
        kwargs: dict[str, Any] = {"exaggeration": 0.5, "cfg_weight": 0.5}
        if self.variant == "multilingual":
            out = self.model.generate(
                text, language_id=self._language_id(voice),
                audio_prompt_path=ref, **kwargs)
        else:
            out = self.model.generate(text, audio_prompt_path=ref, **kwargs)
        try:
            import torch

            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().float().numpy()
        except ImportError:
            pass
        try:
            import numpy as np

            return np.asarray(out, dtype=np.float32).reshape(-1)
        except ImportError:
            return [float(v) for v in out]


class PiperBackend:
    """Piper: the best FREE on-device TTS (phone/embedded CPU), MIT.

    ONNX-runtime VITS voices — ~60MB per voice, RTF ≈ 0.28 on plain CPU
    (3.6× realtime, no GPU, no network at inference), 22.05kHz output.
    The no-GPU fallback that still sounds human on a phone.

    Voice resolution: ``PIPER_VOICE`` (absolute ``.onnx`` path) wins;
    else ``PIPER_VOICES_DIR`` plus ``VoiceProfile.preset_id`` as the
    voice filename stem (``en_US-lessac-medium`` → the ``.onnx`` beside
    it); else a ``en_US-lessac-medium.onnx`` / ``*.onnx`` found in
    ``PIPER_VOICES_DIR`` or the fetch cache. Download voices with::

        pip install piper-tts
        python -m piper.download_voices en_US-lessac-medium

    (espeak-ng phonemizer ships embedded with piper-tts.)
    """

    name = "piper"
    supports_native_tags = False
    supports_cloning = False

    def __init__(self, voice_path: str = "") -> None:
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError(
                "piper backend needs: pip install piper-tts") from exc
        self._PiperVoice = PiperVoice
        self._voices: dict[str, Any] = {}
        self.sample_rate = 22050
        self.voice_path = voice_path or self._find_default_voice()
        if self.voice_path:
            self._load_named("default", self.voice_path)
        # without a voice file the backend loads lazily per profile —
        # synthesis raises a helpful error only if no voice resolves

    @staticmethod
    def _search_dirs() -> list[str]:
        from .fetch import default_cache_dir

        dirs = [os.environ.get("PIPER_VOICES_DIR", ""),
                default_cache_dir("piper")]
        return [d for d in dirs if d]

    @classmethod
    def _find_default_voice(cls) -> str:
        explicit = os.environ.get("PIPER_VOICE", "").strip()
        if explicit and os.path.isfile(explicit):
            return explicit
        for directory in cls._search_dirs():
            candidate = os.path.join(directory, "en_US-lessac-medium.onnx")
            if os.path.isfile(candidate):
                return candidate
            try:
                for entry in sorted(os.listdir(directory)):
                    if entry.endswith(".onnx") and not entry.endswith(
                            ".onnx.json"):
                        return os.path.join(directory, entry)
            except OSError:
                continue
        return ""

    def _load_named(self, key: str, path: str) -> Any:
        voice = self._PiperVoice.load(path)
        self._voices[key] = voice
        rate = getattr(getattr(voice, "config", None), "sample_rate", 0)
        if rate:
            self.sample_rate = int(rate)
        return voice

    def _resolve(self, voice: Optional[VoiceProfile]) -> Any:
        if voice and voice.preset_id:
            for directory in self._search_dirs():
                candidate = os.path.join(
                    directory, f"{voice.preset_id}.onnx")
                if os.path.isfile(candidate):
                    return self._voices.get(voice.preset_id) or \
                        self._load_named(voice.preset_id, candidate)
            # a named voice that does not resolve is a config error —
            # never silently speak in a different voice
            raise RuntimeError(
                f"piper: no voice file for preset "
                f"{voice.preset_id!r} — download one with "
                f"'python -m piper.download_voices {voice.preset_id}' "
                f"into PIPER_VOICES_DIR")
        if "default" in self._voices:
            return self._voices["default"]
        raise RuntimeError(
            "piper: no voice file found — download one with "
            "'python -m piper.download_voices en_US-lessac-medium' "
            "and set PIPER_VOICE or PIPER_VOICES_DIR")

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        pv = self._resolve(voice)
        # preferred: raw int16 PCM chunks, no temp files
        try:
            raw = b"".join(pv.synthesize_stream_raw(text))
        except AttributeError:
            import io

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                pv.synthesize(text, wav_file)
            buf.seek(0)
            with wave.open(buf, "rb") as wav_file:
                raw = wav_file.readframes(wav_file.getnframes())
        if not raw:
            raise ValueError("piper produced no audio")
        vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
        return [v / 32768.0 for v in vals]


class F5TTSBackend:
    """F5-TTS: highest-fidelity single-shot cloning, flow matching.

    MIT code — but the **pretrained checkpoints are CC-BY-NC**
    (Emilia-trained): research/personal use, NOT commercial products.
    Needs a 5–15s reference clip (``VoiceProfile.reference_audio_path``)
    — F5-TTS has no preset voices, so a voice without reference audio
    raises a helpful error. ``voice.prompt_text`` is used as the
    reference transcript when set; otherwise F5's own ASR transcribes
    the clip (extra GPU memory/time).

    ``pip install f5-tts``. ~3GB VRAM for the base model.
    """

    name = "f5tts"
    supports_native_tags = False
    supports_cloning = True
    sample_rate = 24000

    def __init__(self, model: str = "") -> None:
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise RuntimeError(
                "f5tts backend needs: pip install f5-tts") from exc
        self.model_id = (model or os.environ.get("F5TTS_MODEL", "")
                         or "F5TTS_v1_Base")
        self.model = F5TTS(model=self.model_id)

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if voice is not None:
            voice.validate_for_cloning()
        ref = voice.reference_audio_path if voice else None
        if not ref:
            raise RuntimeError(
                "f5tts needs a voice with reference audio — register one "
                "with upload_voice()/clone() first; F5-TTS has no preset "
                "voices")
        ref_text = (voice.prompt_text or "").strip() if voice else ""
        if not ref_text:
            _log.info("f5tts: no prompt_text — F5 will ASR the "
                      "reference clip itself (needs extra GPU memory)")
        wav, sr, _spec = self.model.infer(
            ref_file=ref, ref_text=ref_text, gen_text=text)
        self.sample_rate = int(sr or 24000)
        try:
            import numpy as np

            return np.asarray(wav, dtype=np.float32).reshape(-1)
        except ImportError:
            return [float(v) for v in wav]


class OmniVoiceBackend:
    """OmniVoice (k2-fsa): 600+ languages, Apache-2.0, RTF ~0.025.

    Diffusion-LM zero-shot TTS: cloning from 3–15s reference audio,
    natural-language voice design ("female, low pitch, british accent"
    via the ``instruct`` arg or the voice profile's ``description``),
    native non-verbal symbols (``[laughter]`` ``[sigh]`` ``[sniff]`` —
    see the director's ``render_omnivoice``), and multi-speaker
    ``[Speaker_N]:`` scripts.

    ``pip install omnivoice`` (+ torch). GPU recommended; CPU offload
    is automatic. Model: ``k2-fsa/OmniVoice`` on HuggingFace
    (override with ``OMNIVOICE_MODEL_ID``).
    """

    name = "omnivoice"
    supports_native_tags = True
    supports_cloning = True
    sample_rate = 24000

    def __init__(self, model_id: str = "") -> None:
        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise RuntimeError(
                "omnivoice backend needs: pip install omnivoice") from exc
        self.model_id = (model_id or os.environ.get("OMNIVOICE_MODEL_ID",
                                                    "")
                         or "k2-fsa/OmniVoice")
        device_map = os.environ.get("OMNIVOICE_DEVICE_MAP", "") or "auto"
        self.model = OmniVoice.from_pretrained(
            self.model_id, device_map=device_map, dtype="auto")

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if voice is not None:
            voice.validate_for_cloning()
        ref = voice.reference_audio_path if voice else None
        kwargs: dict[str, Any] = {}
        if ref:
            kwargs["ref_audio"] = ref
            ref_text = (voice.prompt_text or "").strip() if voice else ""
            if ref_text:
                kwargs["ref_text"] = ref_text
        design = instruct or (voice.description if voice else "")
        if design:
            kwargs["instruct"] = design
        out = self.model.generate(text=text, **kwargs)
        try:
            import numpy as np

            return np.asarray(out, dtype=np.float32).reshape(-1)
        except ImportError:
            return [float(v) for v in out]


class Qwen3TTSBackend:
    """Qwen3-TTS (Alibaba): expressive Apache-2.0 TTS, 0.6B / 1.7B.

    Native paralinguistic tags (``[laugh]`` ``[sigh]`` ``[yawn]``
    ``[wow]`` ``[giggle]`` ``[scoff]``) and per-line ``[emotion]``
    switching — canonical markup is already its vocabulary — plus
    instruction-driven style ("speak with great enthusiasm") through
    the ``instruct`` arg, 10 languages, streaming-capable.

    Two model flavors, picked automatically:

    - voice **with** reference audio → the ``Base`` model
      (default ``Qwen/Qwen3-TTS-12Hz-0.6B-Base``), 3s rapid cloning;
    - voice **without** reference → the ``CustomVoice`` model, premium
      preset speakers (``VoiceProfile.preset_id``, default "Vivian").

    ``pip install qwen-tts`` (+ torch; flash-attn recommended on GPU).
    The 0.6B is the most CPU-friendly expressive open model here.
    """

    name = "qwen3tts"
    supports_native_tags = True    # inline [laugh]/[sigh]/… + [emotion]
    supports_cloning = True        # via the Base model
    sample_rate = 24000

    _LANG_NAMES = {
        "en": "English", "zh": "Chinese", "ja": "Japanese",
        "ko": "Korean", "es": "Spanish", "fr": "French",
        "de": "German", "it": "Italian", "pt": "Portuguese",
        "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    }

    def __init__(self, model: str = "", base_model: str = "",
                 custom_model: str = "") -> None:
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "qwen3tts backend needs: pip install qwen-tts") from exc
        self._Qwen3TTSModel = Qwen3TTSModel
        self.base_model_id = (base_model
                              or os.environ.get("QWEN3_TTS_BASE_MODEL", "")
                              or "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self.custom_model_id = (custom_model or model
                                or os.environ.get("QWEN3_TTS_MODEL", "")
                                or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self.device_map = os.environ.get("QWEN3_TTS_DEVICE_MAP", "") or "auto"
        self._models: dict[str, Any] = {}

    def _load(self, kind: str) -> Any:
        if kind not in self._models:
            model_id = (self.base_model_id if kind == "base"
                        else self.custom_model_id)
            self._models[kind] = self._Qwen3TTSModel.from_pretrained(
                model_id, device_map=self.device_map, dtype="auto")
        return self._models[kind]

    @classmethod
    def _language(cls, voice: Optional[VoiceProfile]) -> str:
        code = (voice.language if voice else "") or "en"
        code = code.replace("_", "-").split("-")[0].lower()
        return cls._LANG_NAMES.get(code, "Auto")

    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if voice is not None:
            voice.validate_for_cloning()
        language = self._language(voice)
        ref = voice.reference_audio_path if voice else None
        if ref:
            model = self._load("base")
            ref_text = (voice.prompt_text or "").strip() if voice else ""
            wavs, sr = model.generate_voice_clone(
                text=text, language=language, ref_audio=ref,
                ref_text=ref_text or " ")
        else:
            model = self._load("custom")
            speaker = (voice.preset_id if voice and voice.preset_id
                       else "Vivian")
            kwargs: dict[str, Any] = {"instruct": instruct} if instruct \
                else {}
            wavs, sr = model.generate_custom_voice(
                text=text, language=language, speaker=speaker, **kwargs)
        self.sample_rate = int(sr or 24000)
        first = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        try:
            import numpy as np

            return np.asarray(first, dtype=np.float32).reshape(-1)
        except ImportError:
            return [float(v) for v in first]


def _resample_linear(samples: list, src_rate: int, dst_rate: int) -> list:
    """Pure-Python linear resampler (stdlib only).

    Normalizes whatever the OS speech service emits to the backend's
    canonical rate.
    """
    if src_rate == dst_rate or not samples:
        return list(samples)
    ratio = src_rate / dst_rate
    out_len = int(len(samples) / ratio)
    out: list = []
    for i in range(out_len):
        pos = i * ratio
        j = int(pos)
        frac = pos - j
        a = samples[j]
        b = samples[j + 1] if j + 1 < len(samples) else a
        out.append(a + (b - a) * frac)
    return out


def _extended80_to_float(data: bytes) -> float:
    """IEEE 754 80-bit extended → float (AIFF sample rates)."""
    import struct as _struct

    if len(data) != 10:
        raise ValueError("bad 80-bit extended float")
    expon = _struct.unpack(">H", data[0:2])[0]
    mant = int.from_bytes(data[2:10], "big")
    if expon & 0x7FFF == 0 and mant == 0:
        return 0.0
    sign = -1.0 if expon & 0x8000 else 1.0
    return sign * (mant / float(1 << 63)) * (2.0 ** ((expon & 0x7FFF)
                                                   - 16383))


def _aiff_to_wav_bytes(blob: bytes) -> bytes:
    """AIFF/AIFF-C bytes → WAV bytes. Pure stdlib, no aifc.

    macOS `say` writes AIFF-C with 'sowt' (little-endian) or plain AIFF
    (big-endian); both are handled. Only uncompressed PCM is supported —
    anything else fails fast with a clear error instead of garbage
    audio.
    """
    import io
    import struct as _struct

    if len(blob) < 12 or blob[0:4] != b"FORM":
        raise ValueError("not an AIFF file")
    form_type = blob[8:12]
    if form_type not in (b"AIFF", b"AIFC"):
        raise ValueError(f"unsupported AIFF form type: {form_type!r}")

    channels = width = rate = None
    frames = b""
    little = False
    pos = 12
    while pos + 8 <= len(blob):
        ck_id = blob[pos:pos + 4]
        ck_size = _struct.unpack(">I", blob[pos + 4:pos + 8])[0]
        data = blob[pos + 8:pos + 8 + ck_size]
        if ck_id == b"COMM":
            channels = _struct.unpack(">h", data[0:2])[0]
            width = _struct.unpack(">h", data[6:8])[0] // 8
            rate = int(_extended80_to_float(data[8:18]))
            if form_type == b"AIFC":
                comp = data[18:22]
                if comp == b"sowt":
                    little = True
                elif comp != b"NONE":
                    raise ValueError(
                        f"unsupported AIFF-C compression: {comp!r}")
        elif ck_id == b"SSND":
            offset = _struct.unpack(">I", data[0:4])[0]
            frames = data[8 + offset:]
        pos += 8 + ck_size + (ck_size & 1)  # chunks are even-padded
    if channels is None or width is None or rate is None:
        raise ValueError("AIFF missing COMM chunk")
    if not frames:
        raise ValueError("AIFF missing SSND chunk")
    if width != 2:
        raise ValueError(f"unsupported AIFF sample width: {width * 8}-bit")
    if not little:
        # big-endian → little-endian 16-bit swap
        frames = b"".join(frames[i:i + 2][::-1]
                          for i in range(0, len(frames) - 1, 2))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return buf.getvalue()


class SystemTTSBackend:
    """The OS's own speech service — pure stdlib, zero pip packages.

    macOS: `say` · Linux: `espeak-ng`/`espeak --stdout` · Windows:
    PowerShell System.Speech. Dead last in auto mode: worst quality of
    the lot, but it makes the module *work with nothing installed* —
    the standing fallback behind every neural backend. Not cloning (no
    consent gate); voice profiles' preset_id maps to the OS voice name
    on macOS (`say -v`) and Windows (SelectVoice).
    """

    name = "system"
    supports_native_tags = False
    supports_cloning = False
    sample_rate = 22050

    _MISSING = ("system TTS: no OS speech service found — macOS ships "
                "`say`; on Linux install espeak-ng (`apt install "
                "espeak-ng`, Termux: `pkg install espeak-ng`); Windows "
                "needs PowerShell")

    def __init__(self, *, lang: Optional[str] = None) -> None:
        self.lang = lang or os.environ.get("SYSTEM_TTS_LANG", "")
        self._kind, self._exe = self.detect()

    @staticmethod
    def detect() -> tuple[Optional[str], Optional[str]]:
        """(kind, executable) of the OS speech service, or (None, None)."""
        plat = sys.platform
        if plat == "darwin":
            exe = shutil.which("say")
            return ("say", exe) if exe else (None, None)
        if plat.startswith("linux"):
            for candidate in ("espeak-ng", "espeak"):
                exe = shutil.which(candidate)
                if exe:
                    return ("espeak", exe)
            return (None, None)
        if plat == "win32":
            exe = shutil.which("powershell") or shutil.which("pwsh")
            return ("powershell", exe) if exe else (None, None)
        return (None, None)

    @classmethod
    def available(cls) -> bool:
        kind, _exe = cls.detect()
        return kind is not None

    # -- synthesis -----------------------------------------------------
    def synthesize(self, text: str, voice: Optional[VoiceProfile],
                   *, instruct: str = "") -> Any:
        if not self._exe:
            raise RuntimeError(self._MISSING)
        text = " ".join(str(text or "").split())
        if not text:
            return []
        import tempfile

        with tempfile.TemporaryDirectory(prefix="devon-system-tts-") as tmp:
            if self._kind == "say":
                blob = self._via_say(text, voice, tmp)
            elif self._kind == "espeak":
                blob = self._via_espeak(text)
            elif self._kind == "powershell":
                blob = self._via_powershell(text, voice, tmp)
            else:  # pragma: no cover — detect() already failed above
                raise RuntimeError(self._MISSING)
        samples, rate = HFEndpointBackend._decode_wav(blob)
        if rate != self.sample_rate:
            samples = _resample_linear(samples, rate, self.sample_rate)
        return samples

    def _run(self, cmd: list) -> Any:
        import subprocess

        try:
            return subprocess.run(cmd, check=True, capture_output=True,
                                  timeout=180)
        except FileNotFoundError:
            raise RuntimeError(self._MISSING)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"").decode("utf-8", "replace")[:300]
            raise RuntimeError(f"system TTS ({self._kind}) failed: "
                               f"{err or exc}")

    def _via_say(self, text: str, voice: Optional[VoiceProfile],
                 tmp: str) -> bytes:
        out = os.path.join(tmp, "out.aiff")
        cmd = [self._exe, "-o", out]
        vname = voice.preset_id if voice and voice.preset_id else ""
        if vname:
            cmd += ["-v", vname]
        cmd.append(text)
        self._run(cmd)
        with open(out, "rb") as fh:
            return _aiff_to_wav_bytes(fh.read())

    def _via_espeak(self, text: str) -> bytes:
        cmd = [self._exe, "--stdout", "-s", "175"]
        if self.lang:
            cmd += ["-v", self.lang]
        cmd.append(text)
        return self._run(cmd).stdout

    def _via_powershell(self, text: str, voice: Optional[VoiceProfile],
                        tmp: str) -> bytes:
        out = os.path.join(tmp, "out.wav")
        vname = (voice.preset_id if voice and voice.preset_id
                 else "").replace("'", "''")
        select = f"$s.SelectVoice('{vname}');" if vname else ""
        script = ("Add-Type -AssemblyName System.Speech;"
                  "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                  f"{select}$s.SetOutputToWaveFile('{out}');"
                  "$s.Speak(@\"\n" + text.replace('"', '""') + "\n\"@);"
                  "$s.Dispose()")
        self._run([self._exe, "-NoProfile", "-NonInteractive",
                   "-Command", script])
        with open(out, "rb") as fh:
            return fh.read()


_BACKENDS = {"bark": BarkBackend, "xtts": XTTSBackend,
             "kokoro": KokoroBackend, "cosyvoice": CosyVoiceBackend,
             "dia": DiaBackend, "orpheus": OrpheusBackend,
             "hf-endpoint": HFEndpointBackend,
             "chatterbox": ChatterboxBackend, "piper": PiperBackend,
             "f5tts": F5TTSBackend, "omnivoice": OmniVoiceBackend,
             "qwen3tts": Qwen3TTSBackend, "system": SystemTTSBackend}

#: backend name → importable module proving it is installed. (This also
#: fixes a latent KeyError: "hf-endpoint" was in _BACKENDS but missing
#: from the old inline spec map.)
#: "system" maps to None — it needs no pip package, only an OS speech
#: service (see SystemTTSBackend.available()).
_BACKEND_SPECS = {"bark": "bark", "xtts": "TTS", "kokoro": "kokoro",
                  "cosyvoice": "cosyvoice", "dia": "dia",
                  "orpheus": "orpheus_tts",
                  "hf-endpoint": "huggingface_hub",
                  "chatterbox": "chatterbox", "piper": "piper",
                  "f5tts": "f5_tts", "omnivoice": "omnivoice",
                  "qwen3tts": "qwen_tts", "system": None}


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
                    "no TTS backend usable — pip install one of: "
                    "chatterbox-tts (best free cloning, MIT) | piper-tts "
                    "(phone/CPU, MIT) | kokoro (lightest, Apache-2.0) | "
                    "qwen-tts (0.6B expressive, Apache-2.0) | f5-tts | "
                    "omnivoice (600+ langs, Apache-2.0) | TTS (XTTS v2, "
                    "non-commercial) | cosyvoice (multilingual+paralinguistics, "
                    "MIT) | orpheus-speech (Orpheus, Apache-2.0, GPU) | "
                    "git+https://github.com/suno-ai/bark.git | "
                    "git+https://github.com/nari-labs/dia.git (GPU-only) — "
                    "or install an OS speech service (espeak-ng) for the "
                    "zero-dependency 'system' backend")
            wanted = available[0]
        if wanted not in _BACKENDS:
            raise RuntimeError(
                f"unknown TTS backend {wanted!r}; use one of "
                f"{', '.join(_BACKENDS)} or 'auto'")
        if wanted == "system":
            if not SystemTTSBackend.available():
                raise RuntimeError(SystemTTSBackend._MISSING)
        elif not _spec(_BACKEND_SPECS[wanted]):
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

    #: legacy sound tags (TagProcessor.SOUND_TAGS) → canonical burst names,
    #: so the director renderers can consume parsed segments losslessly.
    _SOUND_TO_CANONICAL = {
        "laughs": "laugh", "sighs": "sigh", "gasps": "gasp",
        "chuckles": "chuckle", "giggles": "giggle",
        "clears_throat": "clearthroat",
    }

    def _segments_to_canonical(self, segments: List[Segment]) -> str:
        """Parsed segments → canonical director markup.

        Re-inserts the tags TagProcessor parsed out (emotion opens, sound
        tags, pauses) so the per-backend director renderers can translate
        them into each backend's native vocabulary. Legacy sound tags
        (``[laughs]``) are normalized to canonical burst names
        (``[laugh]``) first.
        """
        parts: list[str] = []
        for seg in segments:
            if seg.tags == ["_sound_"]:
                burst = seg.text.strip("[]").lower()
                parts.append(f"[{self._SOUND_TO_CANONICAL.get(burst, burst)}]")
            else:
                opens = " ".join(f"[{t}]" for t in seg.tags)
                parts.append(f"{opens} {seg.text}".strip())
            if seg.pause_after_ms:
                parts.append(f"[pause:{seg.pause_after_ms}]")
        return " ".join(parts)

    def _render_native(self, backend: Any,
                       segments: List[Segment]) -> tuple[str, str]:
        """Render parsed segments in the backend's native vocabulary.

        Returns ``(text, instruct)`` — ``instruct`` is only non-empty for
        CosyVoice's instruct mode. Backends with no dedicated renderer
        keep the legacy Bark-format path.
        """
        from .director import (render_bark, render_chatterbox, render_cosyvoice,
                               render_dia, render_fish, render_for,
                               render_omnivoice, render_orpheus,
                               render_plain)

        name = getattr(backend, "name", "")
        if name == "bark":
            return self.tag_processor.to_bark_format(segments), ""
        canonical = self._segments_to_canonical(segments)
        if name == "cosyvoice":
            return render_cosyvoice(canonical)
        if name == "dia":
            return render_dia(canonical), ""
        if name == "orpheus":
            return render_orpheus(canonical), ""
        if name == "chatterbox":
            if getattr(backend, "supports_native_tags", False):
                return render_chatterbox(canonical), ""
            # multilingual V3 has no native paralinguistics: speakable
            # words + onomatopoeia instead of Bark-format tags
            text, _pauses = render_plain(canonical, speak_bursts=True)
            return text, ""
        if name == "omnivoice":
            return render_omnivoice(canonical), ""
        if name == "qwen3tts":
            # native [laugh]/[sigh]/[yawn]/[wow]/[giggle]/[scoff] +
            # [emotion] tags — canonical markup is already its
            # vocabulary (render_for handles pause normalization)
            text, _extra = render_for("qwen3tts", canonical)
            return text, ""
        if name == "hf-endpoint" and getattr(
                backend, "_is_fish", lambda: False)():
            return render_fish(canonical), ""
        # a native-tag backend with no dedicated renderer: Bark format
        return self.tag_processor.to_bark_format(segments), ""

    def speak(self, tagged_text: str, voice_name: Optional[str] = None,
              out_path: str = "", mood: str = "", mood_level: int = 5) -> dict:
        """Synthesize `tagged_text` (emotion/pause tags allowed) to a WAV."""
        text = mood_to_tagged_text(tagged_text, mood, mood_level) if mood \
            else tagged_text
        voice = self.voices.get(voice_name) if voice_name else None
        backend = self._load_backend()
        segments = self.tag_processor.parse(text)
        sample_rate = getattr(backend, "sample_rate", self.default_sample_rate)

        instruct = ""
        if backend.supports_native_tags:
            final_text, instruct = self._render_native(backend, segments)
            audio = backend.synthesize(final_text, voice, instruct=instruct)
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

    def perform(self, text: str, voice_name: Optional[str] = None,
                out_path: str = "", *, mood: str = "neutral",
                intensity: int = 3, seed: Optional[int] = None,
                effect: Optional[str] = None) -> dict:
        """Text in, human-sounding wav out.

        Runs the director (``nomorals/voice/director.py``) over ``text``
        — the PerformanceTuner normalizes it, detects each sentence's
        intent, and routes a natural effect chain (laughs, sighs,
        coughs, breaths, stutters, fillers, pauses, emphasis) — then
        renders the performance script in whatever the active backend
        natively understands:

        - fish: near pass-through (S2 speaks free-form [tags] natively)
        - dia: parenthesized non-verbals (laughs) (coughs) (sneezes)
        - orpheus: angle-bracket emotion tags <laugh> <sigh> <cough>
        - cosyvoice: instruct tokens + emotion/rate instruction
        - bark: native paralinguistic tags
        - chatterbox: native paralinguistic tags (Turbo) or speakable
          words + onomatopoeia (multilingual V3)
        - omnivoice: native [laughter]/[sigh]/[sniff] markup
        - qwen3tts: native [laugh]/[sigh]/[emotion] markup
        - xtts/kokoro/piper/hf-endpoint/system: speakable words +
          onomatopoeia bursts + spliced silence

        ``effect`` picks a named director house style (see
        ``director.EFFECT_PRESETS``). Returns the usual speak() dict plus
        ``script`` (the marked-up performance text) and ``cues``.
        """
        from .director import (direct, render_bark, render_chatterbox,
                               render_cosyvoice, render_dia, render_fish,
                               render_for, render_omnivoice, render_orpheus,
                               render_plain)

        script = direct(text, mood=mood, intensity=intensity, seed=seed,
                        effect=effect)
        voice = self.voices.get(voice_name) if voice_name else None
        backend = self._load_backend()
        sample_rate = getattr(backend, "sample_rate", self.default_sample_rate)

        instruct = ""
        pause_points: list = []
        if backend.name == "cosyvoice":
            final_text, instruct = render_cosyvoice(script)
        elif backend.name == "bark":
            final_text = render_bark(script)
        elif backend.name == "dia":
            final_text = render_dia(script)
        elif backend.name == "orpheus":
            final_text = render_orpheus(script)
        elif backend.name == "chatterbox":
            if getattr(backend, "supports_native_tags", False):
                final_text = render_chatterbox(script)
            else:
                # multilingual V3: speakable words + onomatopoeia
                final_text, pause_points = render_plain(script,
                                                        speak_bursts=True)
        elif backend.name == "omnivoice":
            final_text = render_omnivoice(script)
        elif backend.name == "qwen3tts":
            final_text = render_for("qwen3tts", script)[0]
        elif backend.name == "fish" or (
                backend.name == "hf-endpoint"
                and getattr(backend, "_is_fish", lambda: False)()):
            final_text = render_fish(script)
        else:
            # plain backends: bursts become speakable onomatopoeia
            # (achoo, ha-ha, ahem) instead of vanishing
            final_text, pause_points = render_plain(script,
                                                    speak_bursts=True)

        audio = backend.synthesize(final_text, voice, instruct=instruct)
        audio = self._insert_pauses(audio, pause_points, final_text,
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
            "script": script.text,
            "cues": script.cues,
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
