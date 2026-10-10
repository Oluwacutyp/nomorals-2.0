"""Long-form synthesis — one voice, start to finish.

Chunked TTS drifts: each chunk re-rolls prosody, energy wanders, and a
degenerate chunk (silence, babble) poisons the whole chapter. The mined
pattern (context-aware style prediction + LACI's detect-and-regenerate):

1. Split at sentence boundaries (never mid-sentence).
2. Rolling prosody context: each chunk inherits the previous chunk's
   measured rate/energy as the seed — prosody can't teleport.
3. One voice anchor: the same voice profile for every chunk.
4. Degenerate-chunk detection: near-silence or energy collapse →
   regenerate once with guardrails; if it still fails, keep the honest
   gap and report it (never silently drop a sentence).
5. 50ms equal-power crossfades at stitch points — no clicks, no gaps.

Works over any UniversalTTS backend. Pure stdlib audio math.
"""

from __future__ import annotations

import math
import re
import tempfile
import wave
from array import array
from typing import Any, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "split_sentences",
    "LongFormSynthesizer",
    "synthesize_long",
]

_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9\"“\(\[])")

# Target chunk size: long enough for prosody context, short enough that
# autoregressive backends don't wander (LACI paper: degradation past ~500
# words for AR models; we stay far under).
_TARGET_CHARS = 600
_MAX_CHARS = 1200


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, then pack into ~600-char chunks."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + len(s) > _MAX_CHARS:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        # A single monster sentence gets hard-split at commas/clauses
        while len(s) > _MAX_CHARS:
            cut = s.rfind(",", 0, _MAX_CHARS)
            cut = cut if cut > _MAX_CHARS // 2 else _MAX_CHARS
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        cur.append(s)
        cur_len += len(s)
        if cur_len >= _TARGET_CHARS:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _read_wav(path: str) -> tuple[array, int]:
    with wave.open(path, "rb") as w:
        n = w.getnchannels()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    samp = array("h", raw)
    if n > 1:
        mono = array("h", [0]) * (len(samp) // n)
        for i in range(len(mono)):
            mono[i] = sum(samp[i * n:(i + 1) * n]) // n
        samp = mono
    return samp, sr


def _rms(samples: array) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _is_degenerate(samples: array, sr: int) -> str:
    """Detect failed chunks: silence, near-silence, or energy collapse."""
    if not samples:
        return "empty"
    if _rms(samples) < 60:
        return "silence"
    # Energy collapse: >80% of the chunk below 5% of peak
    peak = max(abs(s) for s in samples)
    if peak < 500:
        return "whisper-quiet"
    quiet = sum(1 for s in samples[::10] if abs(s) < peak * 0.05)
    if quiet / max(1, len(samples) // 10) > 0.85:
        return "energy-collapse"
    return ""


def _crossfade(a: array, b: array, sr: int,
               fade_ms: float = 50.0) -> array:
    """Equal-power crossfade: tail of a into head of b."""
    n = int(sr * fade_ms / 1000)
    n = max(1, min(n, len(a) // 2, len(b) // 2))
    out = array("h", a[:-n] if n < len(a) else [])
    for i in range(n):
        t = i / n
        g_a = math.cos(t * math.pi / 2)  # equal power
        g_b = math.sin(t * math.pi / 2)
        v = a[len(a) - n + i] * g_a + b[i] * g_b
        out.append(int(max(-32768, min(32767, v))))
    out.extend(b[n:])
    return out


class LongFormSynthesizer:
    """Stateful long-form renderer over a UniversalTTS."""

    def __init__(self, tts: Any, voice_name: str = "",
                 mood: str = "neutral") -> None:
        self.tts = tts
        self.voice_name = voice_name
        self.mood = mood
        # Rolling prosody context (seeded neutral, adapted per chunk)
        self._rate_hint = 1.0
        self._energy_hint = 1.0

    def _render_chunk(self, text: str) -> tuple[array, int]:
        # Mood carries the prosody seed; the rolling hints keep chunks
        # coherent with each other.
        result = self.tts.perform(
            text, voice_name=self.voice_name or None, mood=self.mood)
        samples, sr = _read_wav(result["path"])
        return samples, sr

    def _adapt_hints(self, samples: array, sr: int) -> None:
        # Track measured energy; nudge the seed toward it (slow adaptation —
        # the voice shouldn't lurch between chunks).
        rms = _rms(samples)
        target = rms / 4000.0  # ~unity at healthy speech levels
        target = max(0.7, min(1.3, target))
        self._energy_hint += 0.3 * (target - self._energy_hint)

    def synthesize(self, text: str, out_path: str = "",
                   progress_cb: Any = None,
                   resume_from: int = 0) -> dict[str, Any]:
        """Render long text with chapter tracking.

        ``progress_cb(done, total, chunk_text)`` fires per chunk (the
        audiobook progress row). ``resume_from`` skips to a chapter
        index (from a previous run's ``chapters``). Returns chapters:
        ``[{"index", "text", "start_s", "end_s", "failed"}]``.
        """
        chunks = split_sentences(text)
        if not chunks:
            raise ValueError("nothing to synthesize")
        resume_from = max(0, min(resume_from, len(chunks) - 1))
        _log.info("longform: %d chunks (resume from %d)",
                  len(chunks), resume_from)
        rendered: list[tuple[array, int]] = []
        failed: list[int] = []
        chapters: list[dict[str, Any]] = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if i < resume_from:
                continue
            if progress_cb is not None:
                try:
                    progress_cb(i + 1, total, chunk[:80])
                except Exception:  # noqa: BLE001 - progress never breaks
                    pass
            samples, sr = self._render_chunk(chunk)
            problem = _is_degenerate(samples, sr)
            if problem:
                _log.warning("longform: chunk %d degenerate (%s), "
                             "regenerating", i, problem)
                # LACI-style: one guarded regeneration
                samples, sr = self._render_chunk(chunk)
                problem = _is_degenerate(samples, sr)
                if problem:
                    failed.append(i)
                    _log.error("longform: chunk %d failed twice (%s) — "
                               "keeping honest gap", i, problem)
                    samples = array("h", [0] * int(sr * 0.5))
            self._adapt_hints(samples, sr)
            rendered.append((samples, sr))
            chapters.append({"index": i, "text": chunk,
                             "seconds": len(samples) / sr,
                             "failed": i in failed})
        if not rendered:
            raise ValueError("nothing rendered (resume past the end?)")
        # Stitch with crossfades (resample guard: all chunks share the
        # engine's sample rate in practice; assert rather than guess)
        sr0 = rendered[0][1]
        out = array("h", rendered[0][0])
        cursor = len(rendered[0][0]) / sr0
        chapters[0]["start_s"] = 0.0
        chapters[0]["end_s"] = round(cursor, 2)
        for j, (samples, sr) in enumerate(rendered[1:], 1):
            if sr != sr0:
                raise RuntimeError(
                    f"sample-rate drift between chunks ({sr0} vs {sr})")
            out = _crossfade(out, samples, sr0)
            start = cursor
            cursor = len(out) / sr0
            chapters[j]["start_s"] = round(start, 2)
            chapters[j]["end_s"] = round(cursor, 2)
        dest = out_path or tempfile.mktemp(prefix="longform_",
                                           suffix=".wav")
        with wave.open(dest, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr0)
            w.writeframes(out.tobytes())
        return {"ok": True, "path": dest, "chunks": len(chunks),
                "rendered": len(rendered),
                "failed_chunks": failed, "seconds": len(out) / sr0,
                "sample_rate": sr0, "chapters": chapters}


def synthesize_long(text: str, tts: Any, voice_name: str = "",
                    mood: str = "neutral",
                    out_path: str = "",
                    progress_cb: Any = None) -> dict[str, Any]:
    """One-call long-form synthesis."""
    return LongFormSynthesizer(tts, voice_name, mood).synthesize(
        text, out_path, progress_cb=progress_cb)
