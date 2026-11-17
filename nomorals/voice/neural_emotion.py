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


def _segment_direction(cue: Any, fallback: Direction) -> Direction:
    """A track cue (+ whole-line fallback) → nl_director.Direction.

    Fields the cue doesn't name fall back to the whole-line direction
    per field, so an untagged stretch of a directed line still renders
    with the line's direction instead of dropping to neutral.
    """
    return Direction(
        emotion=(cue.emotion if cue else "") or fallback.emotion,
        delivery=(cue.delivery if cue else "") or fallback.delivery,
        accent=(cue.accent if cue else "") or fallback.accent,
        pace=(cue.pace if cue else "") or fallback.pace,
        intensity=fallback.intensity,
        raw=fallback.raw)


def _stitch_wavs(parts: list[tuple[str, float]]) -> str:
    """Concatenate per-segment wavs (+ pause silence) into one wav.

    All parts must share one sample rate and be 16-bit mono — what
    tts.write_wav produces. A mismatch refuses loudly instead of
    silently resampling; segments are concatenated as-is (no
    crossfade — longform owns the crossfade stitcher if a later lane
    wants it).
    """
    out_samples: array = array("h")
    sr = 0
    for path, pause_after_s in parts:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise RuntimeError(
                    f"cannot stitch {path}: expected 16-bit mono WAV")
            samples = array("h", w.readframes(w.getnframes()))
        if sr == 0:
            sr = rate
        elif rate != sr:
            raise RuntimeError(
                f"cannot stitch wavs at mixed sample rates ({sr} vs "
                f"{rate})")
        out_samples.extend(samples)
        if pause_after_s > 0:
            out_samples.extend(
                array("h", [0]) * int(pause_after_s * sr))
    if sr == 0:
        raise RuntimeError("cannot stitch: no segments rendered")
    out = tempfile.mktemp(prefix="emo_seg_", suffix=".wav")
    _write_wav(out, out_samples, sr)
    return out


def _render_directed_per_segment(text: str, track: Any, tts: Any,
                                 voice_name: str, direction: Direction,
                                 tiers: tuple, intensity: int
                                 ) -> dict[str, Any]:
    """Render a directed line segment-by-segment through its track.

    Each spoken segment re-enters :func:`render_emotional` with its
    own direction — its text is tag-free, so the recursion takes the
    flat path exactly once, through the same tier ladder (RVC →
    native → DSP). Sound / ambient segments render with an empty
    direction — the same tiers, nothing to direct. The stitched wav
    keeps segment order (plus each cue's pause-after silence); the
    result keeps the whole-line mood in ``emotion`` so the flat mood
    stays derivable, and carries the track so callers can see what
    ran.
    """
    from .tts import TagProcessor

    tp_segments = TagProcessor().parse(text)
    parts: list[tuple[str, float]] = []
    seg_tiers: list[str] = []
    word_index = 0
    for seg in tp_segments:
        if list(seg.tags or []) in (["_sound_"], ["_ambient_"]):
            seg_direction = Direction()
            pause_after = (seg.pause_after_ms or 0) / 1000.0
        else:
            cue = track.cue_for_word(word_index)
            word_index += len(seg.text.split())
            seg_direction = _segment_direction(cue, direction)
            pause_after = ((cue.pause_after_s if cue else 0.0)
                           or (seg.pause_after_ms or 0) / 1000.0)
        res = render_emotional(seg.text, tts, voice_name=voice_name,
                               direction=seg_direction, tiers=tiers,
                               intensity=intensity)
        parts.append((res["path"], pause_after))
        seg_tiers.append(str(res.get("tier", "")))
    tier = (seg_tiers[0] if seg_tiers and all(
        t == seg_tiers[0] for t in seg_tiers) else "mixed")
    return {"ok": True, "path": _stitch_wavs(parts), "tier": tier,
            "segment_tiers": seg_tiers, "emotion": direction.emotion,
            "delivery": direction.delivery, "segments": len(parts),
            "directed": "per-segment", "track": track.to_dict()}


def render_emotional(text: str, tts: Any, voice_name: str = "",
                     direction: str | Direction = "",
                     rvc_model: str = "",
                     tiers: tuple = EMOTION_TIERS,
                     intensity: int = 7) -> dict[str, Any]:
    """Render text with real emotion. Returns {"path", "tier", ...}.

    ``tts``: a UniversalTTS instance. ``direction``: "[said angrily]" or a
    parsed Direction. ``rvc_model``: RVC model name for the target voice
    (defaults to a model matching voice_name when present).
    ``intensity`` 1–10: director intensity for the expressive render;
    on the DSP tier it scales the shaping (parsed adverbs like "very"
    feed the same scale).
    """
    if isinstance(direction, str):
        direction = parse_direction(direction)
    tag = direction.raw or ""
    last_error: Optional[Exception] = None
    intensity = max(1, min(10, intensity))
    # the DSP tier reads Direction.intensity (1–5). A parsed adverb
    # ("very angry" → 4) wins; otherwise the caller's intensity maps
    # onto the same scale so every tier pushes equally hard.
    if isinstance(direction, Direction) and direction.intensity == 3:
        direction.intensity = max(1, min(5, round(intensity / 2)))

    # Per-segment directed render: inline direction tags anchor
    # direction to word ranges. When the text carries its own tags,
    # build the track and render each segment through the same tier
    # ladder with its own direction — instead of one flat mood for
    # the whole line. An untagged line has no track and takes the
    # flat path below, byte-identical to before; a builder failure
    # fails open to the flat path.
    _track = None
    try:
        from .direction import track_from_text
        _, _track = track_from_text(text, default=direction)
    except Exception:  # noqa: BLE001 — fail open to the flat mood
        _log.debug("neural emotion: track build failed", exc_info=True)
        _track = None
    if _track is not None:
        return _render_directed_per_segment(
            text, _track, tts, voice_name, direction, tiers, intensity)

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
                              intensity=intensity)
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
                                 intensity=intensity)
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
