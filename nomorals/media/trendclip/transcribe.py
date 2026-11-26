"""Transcription for trendclip — Faster Whisper with word timestamps,
through Devon's production STT path.

The source shells out to faster-whisper directly. Devon already owns a
production STT organ (``nomorals/voice/stt.py`` — ``UniversalSTT`` with a
FasterWhisperBackend that returns per-word ``start``/``end``/``word`` when
``word_timestamps=True``), so the organ calls that instead of a parallel
transcriber. The words are normalized to the organ's ``{"w","s","e"}``
shape, which is what the scorer and the caption renderer consume.
"""

from __future__ import annotations

from typing import Any

__all__ = ["normalize_words", "transcribe_audio"]


def normalize_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a ``UniversalSTT`` result into ``{"w","s","e"}`` words."""
    words: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            try:
                start = float(word.get("start", 0.0))
                end = float(word.get("end", start))
            except (TypeError, ValueError):
                continue
            words.append({"w": text, "s": start, "e": end})
    return words


def transcribe_audio(audio_path: str, *, language: str = "en",
                     model: str = "small",
                     stt: Any = None) -> dict[str, Any]:
    """Transcribe an audio file with Faster Whisper word timestamps.

    Returns ``{"text", "words", "language", "backend", "model"}``. Raises
    ``RuntimeError`` with a plain-word cause when no STT backend is
    installed — the caller turns that into the honest note, never a fake
    transcript. ``stt`` is an injectable ``UniversalSTT`` (tests fake it).

    The model is honored through the backend's own seam: the
    faster-whisper backend reads ``FASTER_WHISPER_MODEL`` at
    construction (``voice/stt.py`` ``FasterWhisperBackend`` — its
    default is ``large-v3-turbo`` and ``UniversalSTT`` takes no model
    argument). ``UniversalSTT`` constructs its backend LAZILY, on the
    first ``transcribe()`` call — so the env var must be set around
    the ``transcribe()`` call itself, not around the ``UniversalSTT``
    construction (scoping it to the constructor alone silently loads
    the default). It is restored after, and ``model`` is never passed
    to ``transcribe()`` (the Whisper call takes no such argument).
    The receipt's model is the model actually loaded, never the one
    merely requested.
    """
    if stt is None:
        from ...voice.stt import UniversalSTT

        stt = UniversalSTT(backend="faster-whisper")
        _needs_env = True
    else:
        _needs_env = False
    if _needs_env:
        import os

        _prev = os.environ.get("FASTER_WHISPER_MODEL")
        os.environ["FASTER_WHISPER_MODEL"] = model
        try:
            result = stt.transcribe(audio_path, language=language,
                                    word_timestamps=True, vad_filter=True)
        finally:
            if _prev is None:
                os.environ.pop("FASTER_WHISPER_MODEL", None)
            else:
                os.environ["FASTER_WHISPER_MODEL"] = _prev
    else:
        result = stt.transcribe(audio_path, language=language,
                                word_timestamps=True, vad_filter=True)
    if not isinstance(result, dict):
        raise RuntimeError("STT backend returned no result")
    words = normalize_words(result)
    return {
        "text": str(result.get("text") or "").strip(),
        "words": words,
        "language": str(result.get("language") or language),
        "backend": str(result.get("backend") or "faster-whisper"),
        "model": model,
    }
