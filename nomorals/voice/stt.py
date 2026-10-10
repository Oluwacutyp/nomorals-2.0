"""Universal STT: swappable free/open-source speech-to-text backends.

The ``nomorals/voice/`` side of transcription — the mirror of
``tts.py``'s ``UniversalTTS``. Where ``integrations/voice_integration``
carries an older engine (openai-whisper, paid whisper-api, google),
this module wires the best free options behind one interface:

- **faster-whisper** (default/primary): CTranslate2-backed Whisper —
  up to ~12× realtime, 99 languages, MIT, ``large-v3-turbo`` is the
  sweet spot (7.7% WER, tiny decoder stack). ``pip install
  faster-whisper``.
- **parakeet**: NVIDIA Parakeet TDT 0.6B v3 via the lightweight
  ``onnx-asr`` package — 6.3% WER English, built-in punctuation/casing,
  ~27ms/10s on GPU and ~0.3–0.5s on CPU, int8 quant. CC-BY-4.0
  (commercial OK). English + 25 European languages — fastest free
  dictation when Whisper's 99 languages aren't needed.
  ``pip install onnx-asr`` (optionally ``onnx-asr[cpu,hub]``).
- **whisper.cpp**: the llama.cpp author's C++ port through the
  ``pywhispercpp`` bindings — no torch at all, GGUF/quantized models,
  good on weak hardware. ``pip install pywhispercpp``.
- **whisper** (openai-whisper): the classic reference — slowest, kept
  as a fallback when nothing else is installed.
- **hf-stt**: Hugging Face serverless ASR (``InferenceClient``) — no
  local weights, for the no-GPU path. ``pip install huggingface_hub``;
  ``HF_STT_MODEL`` picks the model (default
  ``openai/whisper-large-v3-turbo``).

Usage::

    stt = UniversalSTT(backend="auto")
    result = stt.transcribe("voice_note.wav", language="en")
    # {"text": "...", "language": "en", "backend": "faster-whisper",
    #  "segments": [{"start": 0.0, "end": 1.2, "text": "..."}], ...}

    # adapt to VoiceSession's stt callable (wav path -> str)
    session_stt = make_session_stt(stt)

All backends are lazy-imported; nothing breaks when one is missing.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

__all__ = [
    "available_stt_backends",
    "UniversalSTT",
    "make_session_stt",
    "StreamingSTT",
    "diarize",
    "to_srt",
    "to_vtt",
    "format_transcript",
]


def _spec(name: str) -> bool:
    # an already-imported module is definitionally installed
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ----------------------------------------------------
# BACKENDS
# ----------------------------------------------------


class FasterWhisperBackend:
    """faster-whisper: the best free general STT. MIT.

    CTranslate2-backed Whisper — the same 99-language accuracy as
    openai-whisper at ~4–12× the speed, CPU int8 viable. ``large-v3-turbo``
    (default) is the 2025–26 sweet spot: ~7.7% WER, 4 decoder layers,
    ~0.3–0.6s per 10s clip on GPU. ``pip install faster-whisper``.
    """

    name = "faster-whisper"

    def __init__(self, model: str = "") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper backend needs: pip install faster-whisper"
            ) from exc
        self.model_id = (model or os.environ.get("FASTER_WHISPER_MODEL", "")
                         or "large-v3-turbo")
        device = os.environ.get("FASTER_WHISPER_DEVICE", "")
        if not device:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute_type = os.environ.get(
            "FASTER_WHISPER_COMPUTE",
            "float16" if device == "cuda" else "int8")
        self.model = WhisperModel(self.model_id, device=device,
                                  compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str = "en",
                   **kwargs: Any) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            audio_path, language=language or None,
            vad_filter=kwargs.pop("vad_filter", True),
            vad_parameters=kwargs.pop(
                "vad_parameters",
                {"min_silence_duration_ms": 500}),
            condition_on_previous_text=False,
            **kwargs)
        want_words = bool(kwargs.get("word_timestamps", False))
        segs = []
        for s in segments:
            seg: dict[str, Any] = {
                "start": float(s.start), "end": float(s.end),
                "text": s.text.strip()}
            if want_words:
                seg["words"] = [
                    {"start": float(w.start), "end": float(w.end),
                     "word": w.word, "prob": float(getattr(w, "probability",
                                                           0.0) or 0.0)}
                    for w in (s.words or [])]
            segs.append(seg)
        text = " ".join(s["text"] for s in segs).strip()
        return {"text": text, "language": info.language or language,
                "backend": self.name, "segments": segs}


class ParakeetBackend:
    """NVIDIA Parakeet TDT 0.6B v3: fastest free English dictation.

    6.3% English WER (Open ASR LB), built-in punctuation and casing,
    ~27ms per 10s of audio on GPU, ~0.3–0.5s on desktop CPU (int8 ONNX
    via ``onnx-asr`` — no NeMo, no torch needed). CC-BY-4.0
    (commercial OK). Covers English + 25 European languages; pick
    faster-whisper for the other ~74. ``pip install onnx-asr[cpu,hub]``.
    """

    name = "parakeet"

    def __init__(self, model: str = "") -> None:
        try:
            import onnx_asr
        except ImportError as exc:
            raise RuntimeError(
                "parakeet backend needs: pip install onnx-asr "
                "(or onnx-asr[cpu,hub])") from exc
        self.model_id = (model or os.environ.get("PARAKEET_MODEL", "")
                         or "nemo-parakeet-tdt-0.6b-v3")
        self.quantization = os.environ.get("PARAKEET_QUANTIZATION",
                                           "") or None
        self.model = onnx_asr.load_model(self.model_id,
                                         quantization=self.quantization)

    def transcribe(self, audio_path: str, language: str = "en",
                   **kwargs: Any) -> dict[str, Any]:
        result = self.model.recognize(audio_path)
        if isinstance(result, (list, tuple)):
            text = " ".join(
                r.text if hasattr(r, "text") else str(r) for r in result
            ).strip()
        else:
            text = (result.text if hasattr(result, "text")
                    else str(result)).strip()
        return {"text": text, "language": language, "backend": self.name,
                "segments": []}


class WhisperCppBackend:
    """whisper.cpp via pywhispercpp: Whisper without torch. MIT.

    The llama.cpp author's C++ port — quantized GGUF models, tiny
    footprint, good on weak hardware. Slower than faster-whisper's
    CTranslate2 path but needs no Python ML stack at all.
    ``pip install pywhispercpp``.
    """

    name = "whisper.cpp"

    def __init__(self, model: str = "") -> None:
        try:
            from pywhispercpp.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "whisper.cpp backend needs: pip install pywhispercpp"
            ) from exc
        self.model_id = (model or os.environ.get("WHISPERCPP_MODEL", "")
                         or "base")
        self._Model = Model
        self.model = Model(self.model_id)

    def transcribe(self, audio_path: str, language: str = "en",
                   **kwargs: Any) -> dict[str, Any]:
        segments = self.model.transcribe(audio_path,
                                         language=language or "auto")
        segs = [{"start": float(s.t0 / 100.0), "end": float(s.t1 / 100.0),
                 "text": s.text.strip()} for s in segments]
        text = " ".join(s["text"] for s in segs).strip()
        return {"text": text, "language": language, "backend": self.name,
                "segments": segs}


class WhisperBackend:
    """openai-whisper: the classic reference. MIT.

    Slowest option (plain PyTorch, no CTranslate2), kept as a fallback
    for when nothing else is installed. ``pip install openai-whisper``.
    """

    name = "whisper"

    def __init__(self, model: str = "") -> None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "whisper backend needs: pip install openai-whisper "
                "(or use the faster-whisper backend)") from exc
        self.model_id = (model or os.environ.get("WHISPER_MODEL", "")
                         or "base")
        self.model = whisper.load_model(self.model_id)

    def transcribe(self, audio_path: str, language: str = "en",
                   **kwargs: Any) -> dict[str, Any]:
        result = self.model.transcribe(audio_path,
                                       language=language or None)
        text = str(result.get("text", "")).strip()
        segs = [{"start": float(s.get("start", 0.0)),
                 "end": float(s.get("end", 0.0)),
                 "text": str(s.get("text", "")).strip()}
                for s in result.get("segments", []) or []]
        return {"text": text, "language": result.get("language", language),
                "backend": self.name, "segments": segs}


class HFEndpointSTTBackend:
    """Hugging Face serverless ASR — no local weights. The no-GPU path.

    Uses ``huggingface_hub.InferenceClient`` against an ASR model
    (``HF_STT_MODEL``, default ``openai/whisper-large-v3-turbo``).
    ``HF_TOKEN`` only needed for gated/private models. Needs
    ``pip install huggingface_hub``.
    """

    name = "hf-stt"

    def __init__(self, model: str = "") -> None:
        try:
            from huggingface_hub import InferenceClient  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "hf-stt backend needs: pip install huggingface_hub"
            ) from exc
        self.model = (model or os.environ.get("HF_STT_MODEL", "")
                      or "openai/whisper-large-v3-turbo")

    def transcribe(self, audio_path: str, language: str = "en",
                   **kwargs: Any) -> dict[str, Any]:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN", "") or None
        client = InferenceClient(model=self.model, token=token)
        with open(audio_path, "rb") as fh:
            blob = fh.read()
        out = client.automatic_speech_recognition(blob)
        if isinstance(out, dict):
            text = str(out.get("text", "")).strip()
        else:
            text = str(out or "").strip()
        return {"text": text, "language": language, "backend": self.name,
                "segments": []}


_STT_BACKENDS = {"faster-whisper": FasterWhisperBackend,
                 "parakeet": ParakeetBackend,
                 "whisper.cpp": WhisperCppBackend,
                 "whisper": WhisperBackend,
                 "hf-stt": HFEndpointSTTBackend}

_STT_SPECS = {"faster-whisper": "faster_whisper", "parakeet": "onnx_asr",
              "whisper.cpp": "pywhispercpp", "whisper": "whisper",
              "hf-stt": "huggingface_hub"}


def available_stt_backends() -> list[str]:
    """Which STT backends are actually installed (in preference order).

    faster-whisper first (broad 99-language coverage), then Parakeet
    (fastest free English dictation), whisper.cpp (no torch), classic
    whisper, and the cloud fallback.
    """
    order = [("faster-whisper", "faster_whisper"), ("parakeet", "onnx_asr"),
             ("whisper.cpp", "pywhispercpp"), ("whisper", "whisper"),
             ("hf-stt", "huggingface_hub")]
    return [name for name, spec in order if _spec(spec)]


# ----------------------------------------------------
# UNIFIED ENGINE
# ----------------------------------------------------


class UniversalSTT:
    """The unified speech-to-text engine.

    stt = UniversalSTT(backend="auto")
    result = stt.transcribe("voice_note.wav", language="en")
    # result -> {"text": "...", "language": "en",
    #            "backend": "faster-whisper", "segments": [...]}
    """

    def __init__(self, backend: str = "auto") -> None:
        self._backend_name = (backend or "auto").lower()
        self._loaded = False
        self._impl: Any = None

    def _load_backend(self) -> Any:
        if self._loaded:
            return self._impl
        wanted = self._backend_name
        if wanted == "auto":
            available = available_stt_backends()
            if not available:
                raise RuntimeError(
                    "no STT backend installed — pip install one of: "
                    "faster-whisper (best free, MIT) | onnx-asr (Parakeet, "
                    "fastest English) | pywhispercpp (no torch) | "
                    "openai-whisper | huggingface_hub (cloud)")
            wanted = available[0]
        if wanted not in _STT_BACKENDS:
            raise RuntimeError(
                f"unknown STT backend {wanted!r}; use one of "
                f"{', '.join(_STT_BACKENDS)} or 'auto'")
        if not _spec(_STT_SPECS[wanted]):
            raise RuntimeError(
                f"STT backend {wanted!r} is not installed on this machine")
        self._impl = _STT_BACKENDS[wanted]()
        self._loaded = True
        return self._impl

    @property
    def backend_name(self) -> str:
        if self._impl is not None:
            return self._impl.name
        if self._backend_name != "auto":
            return self._backend_name
        return available_stt_backends()[0] if available_stt_backends() \
            else ""

    def transcribe(self, audio_path: str, language: str = "en",
                   **kwargs: Any) -> dict:
        """Transcribe an audio file. Returns ``{"text", "language",
        "backend", "segments"}`` — never raises on empty audio, only on
        genuinely broken input.

        Extra kwargs pass through to the backend: faster-whisper honors
        ``word_timestamps=True`` (per-word ``start``/``end``/``word``),
        ``vad_filter`` and ``vad_parameters`` (Silero VAD tuning —
        ``min_silence_duration_ms=500`` is the natural-cadence default).
        """
        backend = self._load_backend()
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"audio file not found: {audio_path!r}")
        return backend.transcribe(audio_path, language=language, **kwargs)


def make_session_stt(stt: Any, *,
                     language: str = "en") -> Callable[[str], str]:
    """Adapt a ``UniversalSTT`` (or anything with ``transcribe(path,
    language=)``) to the sync ``stt: Callable[[str], str]`` signature
    ``VoiceSession`` (``nomorals/voice/session.py``) expects.

    Errors degrade to ``""`` (session keeps listening) rather than
    killing the voice loop.
    """

    def _stt(wav_path: str) -> str:
        try:
            result = stt.transcribe(wav_path, language=language)
        except Exception as exc:  # noqa: BLE001 - a dead mic/empty clip must not kill the voice loop
            _log.warning("STT failed on %s: %s", wav_path, exc)
            return ""
        if isinstance(result, dict):
            return str(result.get("text", "") or "").strip()
        return str(result or "").strip()

    _stt.supports_partial = False  # type: ignore[attr-defined]
    return _stt


# ---------------------------------------------------------------------------
# Transcript presentation + streaming + diarization (sweep additions)
# ---------------------------------------------------------------------------


def _srt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    return _srt_ts(seconds).replace(",", ".")


def to_srt(segments: list[dict[str, Any]]) -> str:
    """Segments → SubRip subtitles (the tts-audio-suite SRT pattern)."""
    lines: list[str] = []
    for i, seg in enumerate(segments or [], 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg.get('start', 0.0))} --> "
                     f"{_srt_ts(seg.get('end', 0.0))}")
        speaker = seg.get("speaker")
        text = str(seg.get("text", "") or "").strip()
        lines.append(f"{speaker}: {text}" if speaker else text)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def to_vtt(segments: list[dict[str, Any]]) -> str:
    """Segments → WebVTT subtitles."""
    lines = ["WEBVTT", ""]
    for seg in segments or []:
        lines.append(f"{_vtt_ts(seg.get('start', 0.0))} --> "
                     f"{_vtt_ts(seg.get('end', 0.0))}")
        speaker = seg.get("speaker")
        text = str(seg.get("text", "") or "").strip()
        lines.append(f"<v {speaker}>{text}" if speaker else text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_transcript(result: dict[str, Any],
                      style: str = "text") -> str:
    """Render a transcribe() result as text, SRT, VTT, or words.

    - ``text``: the plain transcript.
    - ``srt``/``vtt``: subtitle files from segments.
    - ``words``: one word per line with timestamps (needs
      ``word_timestamps=True`` at transcribe time).
    - ``pretty``: timestamped lines, ``[00:01.2 → 00:03.4] hello``.
    """
    style = (style or "text").lower()
    segments = result.get("segments", []) or []
    if style == "srt":
        return to_srt(segments)
    if style == "vtt":
        return to_vtt(segments)
    if style == "words":
        lines = []
        for seg in segments:
            for w in seg.get("words", []) or []:
                lines.append(f"[{_vtt_ts(w.get('start', 0.0))}] "
                             f"{w.get('word', '')}")
        return "\n".join(lines)
    if style == "pretty":
        lines = []
        for seg in segments:
            spk = f"{seg['speaker']}: " if seg.get("speaker") else ""
            lines.append(f"[{_vtt_ts(seg.get('start', 0.0))} → "
                         f"{_vtt_ts(seg.get('end', 0.0))}] {spk}"
                         f"{seg.get('text', '')}")
        return "\n".join(lines)
    return str(result.get("text", "") or "")


class StreamingSTT:
    """Incremental transcription for the live loop (RealtimeSTT pattern).

    Feeds PCM chunks in, emits partial transcripts out. Backed by
    faster-whisper: chunks accumulate until ``min_chunk_s`` of audio is
    buffered, then one transcribe pass runs on the window. No extra
    dependency — the stdlib + faster-whisper path the module already has.

    Usage::

        sstt = StreamingSTT(language="en")
        for pcm in mic_chunks():          # bytes, int16 mono 16kHz
            for partial in sstt.feed(pcm):
                print("partial:", partial["text"])
        final = sstt.flush()             # last window, finalized
    """

    def __init__(self, backend: str = "auto", language: str = "en",
                 sample_rate: int = 16000,
                 min_chunk_s: float = 1.0) -> None:
        self.stt = UniversalSTT(backend=backend)
        self.language = language
        self.sample_rate = sample_rate
        self.min_chunk_s = min_chunk_s
        self._buf = bytearray()
        self._finalized = ""
        self._lock = None

    def _need(self) -> int:
        return int(self.sample_rate * 2 * self.min_chunk_s)

    def feed(self, pcm: bytes) -> list[dict[str, Any]]:
        """Feed int16 mono PCM bytes. Returns newly completed partials."""
        if pcm:
            self._buf.extend(pcm)
        out: list[dict[str, Any]] = []
        while len(self._buf) >= self._need():
            window = bytes(self._buf[:self._need()])
            del self._buf[:self._need()]
            text = self._transcribe_window(window)
            if text:
                out.append({"text": text, "partial": True})
        return out

    def _transcribe_window(self, pcm: bytes) -> str:
        import tempfile
        import wave
        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav",
                                             delete=False) as fh:
                path = fh.name
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(pcm)
            res = self.stt.transcribe(path, language=self.language)
            return str(res.get("text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001 - a bad window is skipped
            _log.debug("streaming STT window failed: %s", exc)
            return ""
        finally:
            try:
                if path:
                    os.remove(path)
            except OSError:
                pass

    def flush(self) -> dict[str, Any]:
        """Transcribe the remainder. Returns {"text", "partial": False}."""
        text = self._transcribe_window(bytes(self._buf)) if self._buf \
            else ""
        self._buf.clear()
        if text:
            self._finalized = (self._finalized + " " + text).strip()
        return {"text": self._finalized, "partial": False}


def diarize(audio_path: str, *,
            num_speakers: int = 0,
            speaker_names: dict[str, str] | None = None,
            hysteresis_ms: int = 500) -> dict[str, Any]:
    """Who spoke when — optional pyannote pipeline, honest when absent.

    Returns ``{"ok", "segments": [{"start", "end", "speaker"}]}``.
    Speaker IDs are mapped to names server-side via ``speaker_names``
    (``{"SPEAKER_00": "Ada"}``) — raw indices never leak to callers.
    A ``hysteresis_ms`` debounce merges speaker flips shorter than the
    window (the forasoft 2026 diarization guidance: one cough must not
    re-attribute two words).

    Needs ``pip install pyannote.audio`` + a HF token for the gated
    diarization models; raises RuntimeError with the recipe otherwise.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "diarize needs: pip install pyannote.audio — plus a "
            "HuggingFace token (HF_TOKEN) accepting the pyannote "
            "speaker-diarization model terms") from exc
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio file not found: {audio_path!r}")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1")
    kwargs: dict[str, Any] = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kwargs)
    names = speaker_names or {}
    raw: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        label = names.get(speaker, speaker)
        raw.append({"start": float(turn.start), "end": float(turn.end),
                    "speaker": label})
    # hysteresis: merge same-speaker-adjacent flips shorter than window
    merged: list[dict[str, Any]] = []
    window = hysteresis_ms / 1000.0
    for seg in raw:
        if (merged and merged[-1]["speaker"] == seg["speaker"]
                and seg["start"] - merged[-1]["end"] <= window):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    speakers = sorted({s["speaker"] for s in merged})
    return {"ok": True, "segments": merged, "speakers": speakers,
            "num_speakers": len(speakers)}
