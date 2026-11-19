"""SoundCraft offline render — bounce a session to WAV (lane 107).

Ported from storytold/soundcraft's ``crates/mix`` offline render:
the same MixEngine that drives the realtime callback renders the
whole session, sample-exact and deterministic — two bounces of the
same session are byte-identical. The output is 16-bit PCM stereo
WAV via the stdlib, and the result reports the peak, RMS, and
whether the master clipped, honestly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .errors import SoundError, SoundErrorKind
from .mix import MixEngine
from .model import SessionDocument

__all__ = ["render_session"]


def render_session(doc: SessionDocument, output: str | Path | None = None,
                   length: int | None = None) -> dict[str, Any]:
    """Render the whole session (or ``length`` samples) to stereo WAV."""
    total = int(length) if length is not None else doc.length_samples()
    if total <= 0:
        raise SoundError("nothing to render — the session is empty",
                         kind=SoundErrorKind.INVALID_ARGUMENT)
    engine = MixEngine(doc)
    left, right, report = engine.render(0, total)
    stereo = np.empty(2 * total, dtype=np.float64)
    stereo[0::2] = left
    stereo[1::2] = right
    # Interleave manually into a stereo WAV with the stdlib wave module.
    import io
    import wave
    clipped = np.clip(stereo, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(doc.sample_rate)
        wf.writeframes(pcm)
    raw = buf.getvalue()
    result: dict[str, Any] = {
        "sample_rate": doc.sample_rate,
        "frames": total,
        "peak": report["master_peak"],
        "rms": float(np.sqrt(np.mean(stereo ** 2))) if total else 0.0,
        "clipped": report["clipped"],
        "clip_first_sample": report["clip_first_sample"],
        "strip_peaks": report["strip_peaks"],
        "bus_peaks": report["bus_peaks"],
    }
    if output is not None:
        Path(output).write_bytes(raw)
        result["path"] = str(output)
    else:
        result["wav_bytes"] = raw
    return result
