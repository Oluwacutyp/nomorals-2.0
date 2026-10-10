"""Neural emotion — true emotion transfer, not DSP hacks.

The problem with :mod:`nomorals.voice.emotion_dsp` is honest: pitch/rate/
energy shaping is *suggestive*, not neural. The mined solution (the RVC
hybrid pattern from voice-studio/dhwani): RVC preserves the source audio's
emotion, prosody and timing while swapping identity. So:

    expressive render (any voice) → RVC → target identity

The emotion is genuinely in the waveform the neural converter preserves —
a real performance, not a pitch knob. When RVC (or a model for the voice)
is unavailable, this module degrades through an honest chain:

1. RVC emotion transfer (neural) — best
2. Expressive-backend native tags (Chatterbox/Dia/Bark/CosyVoice) — neural,
   no identity change
3. DSP shaping (emotion_dsp) — honest fallback, documented as such

Never claims neural when it used DSP: the result dict always names the
tier that actually ran.
"""

from __future__ import annotations

import os
import tempfile
import wave
from array import array
from typing import Any, Callable, Optional

from ..core.logging_setup import get_logger
from . import rvc_bridge
from .emotion_dsp import shape_for_direction
from .nl_director import Direction, parse_direction

_log = get_logger(__name__)

__all__ = [
    "EMOTION_TIERS",
    "render_emotional",
]

#: The honest capability ladder, best first.
EMOTION_TIERS = ("rvc", "native", "dsp")


def _write_wav(path: str, samples: array, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def render_emotional(text: str, tts: Any, voice_name: str = "",
                     direction: str | Direction = "",
                     rvc_model: str = "",
                     tiers: tuple = EMOTION_TIERS) -> dict[str, Any]:
    """Render text with real emotion. Returns {"path", "tier", ...}.

    ``tts``: a UniversalTTS instance. ``direction``: "[said angrily]" or a
    parsed Direction. ``rvc_model``: RVC model name for the target voice
    (defaults to a model matching voice_name when present).
    """
    if isinstance(direction, str):
        direction = parse_direction(direction)
    tag = direction.raw or ""
    last_error: Optional[Exception] = None

    # Tier 1 — RVC neural transfer
    if "rvc" in tiers:
        try:
            probe = rvc_bridge.detect_rvc()
            model_name = rvc_model or voice_name
            models = rvc_bridge.list_models()
            model = next((m for m in models if m.name == model_name), None)
            if not probe["ok"]:
                raise rvc_bridge.RVCUnavailable(probe["detail"])
            if model is None:
                raise rvc_bridge.RVCUnavailable(
                    f"no RVC model for voice '{model_name}'")
            # Expressive source render: the director's perform() on the
            # *default* voice — identity comes from RVC, so the source
            # voice doesn't matter, the performance does.
            src = tts.perform(text, mood=direction.emotion or "neutral",
                              intensity=7)
            src_path = src["path"]
            conv = rvc_bridge.convert(src_path, model)
            return {"ok": True, "path": conv["path"], "tier": "rvc",
                    "model": model.name, "emotion": direction.emotion,
                    "delivery": direction.delivery}
        except Exception as exc:  # noqa: BLE001 — fall through honestly
            last_error = exc
            _log.info("neural emotion: rvc tier unavailable: %s", exc)

    # Tier 2 — expressive backend's native tags (perform = director)
    if "native" in tiers:
        try:
            native = tts.perform(text, voice_name=voice_name or None,
                                 mood=direction.emotion or "neutral",
                                 intensity=7)
            return {"ok": True, "path": native["path"], "tier": "native",
                    "backend": native.get("backend")}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _log.info("neural emotion: native tier unavailable: %s", exc)

    # Tier 3 — honest DSP
    if "dsp" in tiers:
        base = tts.speak(text, voice_name=voice_name or None)
        with wave.open(base["path"], "rb") as w:
            sr = w.getframerate()
            samples = array("h", w.readframes(w.getnframes()))
        shaped = shape_for_direction(samples, sr, direction)
        out = tempfile.mktemp(prefix="emo_", suffix=".wav")
        _write_wav(out, shaped, sr)
        return {"ok": True, "path": out, "tier": "dsp",
                "note": "DSP shaping — suggestive, not neural"}

    raise RuntimeError(
        f"no emotion tier available (last error: {last_error})")
