"""Universal TTS: swappable free/open-source neural backends, one interface.

Port of the owner's universal_tts.py (from the main branch upload),
integrated with the framework:

- Backends: **Bark** (best tag/non-speech support), **XTTS v2** (best
  cloning quality, NON-COMMERCIAL CPML license), **Kokoro** (lightest,
  CPU-friendly, Apache-2.0 — commercial OK), **CosyVoice** (multilingual
  + instruction-driven paralinguistics: laughter, breaths, emphasis —
  MIT, fetched from HuggingFace), **Orpheus** (LLM-class expressivity,
  native ``<laugh>``/``<sigh>``/``<cough>`` tags, zero-shot cloning —
  Apache-2.0), **Dia** (nari-labs dialogue model, native ``(laughs)``
  / ``(coughs)`` / ``(sneezes)`` non-verbals — Apache-2.0, GPU-only).
  Each is lazy-imported; nothing breaks when a backend is not installed.
- A tag system for emotion / pauses / non-speech sounds that the AI can
  use directly in its own text: [happy] [whisper] [laughs] [pause:300] …
- The **director** (``nomorals/voice/director.py``) turns plain text
  into a performance script — 30 vocal bursts (laughs, giggles, coughs,
  sneezes, sighs, gasps, yawns, whistles, …), 53 emotion tags, fillers,
  stutters, pauses, emphasis, pacing — and each backend renders the
  canonical markup in its own native vocabulary. ``UniversalTTS.perform()``
  is the one-call path: text in, human-sounding wav out.
- Voice profiles persisted to disk. Clone operations are audit-logged.
  Multiple reference samples per voice are blended where the backend
  supports it (XTTS averages the embeddings).
- A mood bridge: the partner's mood system maps straight into tags.
- Pure-stdlib WAV writing (no scipy); numpy is only touched by the
  backends themselves, never at import time.

Install one backend on the phone:
    pip install git+https://github.com/suno-ai/bark.git   # Bark
    pip install TTS                                       # XTTS v2
    pip install kokoro                                    # Kokoro
    pip install cosyvoice                                 # CosyVoice
    pip install orpheus-speech                            # Orpheus (GPU)
    nm voice fetch --backend cosyvoice                    # pull the weights
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import struct
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
    Clone operations are audit-logged (voice name, timestamp, backend).
    """

    name: str
    reference_audio_path: Optional[str] = None
    preset_id: Optional[str] = None   # for Bark/Kokoro built-in voices
    language: str = "en"
    description: str = ""
    #: Transcript of the reference clip. Zero-shot backends (CosyVoice)
    #: need it to clone; XTTS does not.
    prompt_text: str = ""
    #: Extra reference clips for the same voice. Cloning backends that
    #: accept several samples (XTTS averages the speaker embeddings)
    #: blend them; single-sample backends use the first clip.
    extra_samples: list = field(default_factory=list)

    def audit_clone(self, backend: str) -> None:
        """Log a voice-clone operation for the owner's audit trail."""
        from ..core.logging_setup import get_logger
        _log = get_logger(__name__)
        from datetime import datetime, timezone
        _log.info(
            "voice clone initiated: name=%s backend=%s time=%s",
            self.name, backend,
            datetime.now(timezone.utc).isoformat(),
        )

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
                     language: str = "en",
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
    if _spec("cosyvoice"):
        out.append("cosyvoice")
    if _spec("orpheus_tts"):
        out.append("orpheus")
    if _spec("dia"):
        out.append("dia")
    if _spec("huggingface_hub"):
        out.append("hf-endpoint")
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
            voice.audit_clone(self.name)
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
                voice.audit_clone(self.name)
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
                voice.audit_clone(self.name)
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
            voice.audit_clone(self.name)
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
            voice.audit_clone(self.name)
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


_BACKENDS = {"bark": BarkBackend, "xtts": XTTSBackend,
             "kokoro": KokoroBackend, "cosyvoice": CosyVoiceBackend,
             "dia": DiaBackend, "orpheus": OrpheusBackend,
             "hf-endpoint": HFEndpointBackend}

#: backend name → importable module proving it is installed. (This also
#: fixes a latent KeyError: "hf-endpoint" was in _BACKENDS but missing
#: from the old inline spec map.)
_BACKEND_SPECS = {"bark": "bark", "xtts": "TTS", "kokoro": "kokoro",
                  "cosyvoice": "cosyvoice", "dia": "dia",
                  "orpheus": "orpheus_tts",
                  "hf-endpoint": "huggingface_hub"}


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
    engine.voices.upload_voice("me", "my_sample.wav")
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
                    "non-commercial) | cosyvoice (multilingual+paralinguistics, "
                    "MIT) | orpheus-speech (Orpheus, Apache-2.0, GPU) | "
                    "git+https://github.com/suno-ai/bark.git | "
                    "git+https://github.com/nari-labs/dia.git (GPU-only)")
            wanted = available[0]
        if wanted not in _BACKENDS:
            raise RuntimeError(
                f"unknown TTS backend {wanted!r}; use one of "
                f"{', '.join(_BACKENDS)} or 'auto'")
        if not _spec(_BACKEND_SPECS[wanted]):
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
        from .director import (render_bark, render_cosyvoice, render_dia,
                               render_fish, render_orpheus)

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
        - xtts/kokoro/hf-endpoint: speakable words + onomatopoeia bursts
          + spliced silence

        ``effect`` picks a named director house style (see
        ``director.EFFECT_PRESETS``). Returns the usual speak() dict plus
        ``script`` (the marked-up performance text) and ``cues``.
        """
        from .director import (direct, render_bark, render_cosyvoice,
                               render_dia, render_fish, render_orpheus,
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
