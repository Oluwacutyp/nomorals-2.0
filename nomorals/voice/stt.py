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
            vad_filter=True,
            condition_on_previous_text=False,
            **kwargs)
        segs = [{"start": float(s.start), "end": float(s.end),
                 "text": s.text.strip()} for s in segments]
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

    def transcribe(self, audio_path: str, language: str = "en") -> dict:
        """Transcribe an audio file. Returns ``{"text", "language",
        "backend", "segments"}`` — never raises on empty audio, only on
        genuinely broken input."""
        backend = self._load_backend()
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"audio file not found: {audio_path!r}")
        return backend.transcribe(audio_path, language=language)


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
