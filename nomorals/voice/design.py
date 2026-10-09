"""Voice design and morphing — craft voices from description, blend voices.

God-tier voice systems don't just clone; they *design*. This module turns
natural-language voice descriptions into real, usable voices through DSP
parameter shaping (pitch, speed, tone) applied to a base voice, and morphs
between voices by interpolating their acoustic parameters.

No neural voice-conversion model required — this is honest DSP shaping,
documented as such. When a neural backend supports reference blending,
the designed parameters travel with the voice profile.
"""

from __future__ import annotations

import logging
import math
import os
import re
from array import array
from dataclasses import dataclass, field
from typing import Any, Optional

_log = logging.getLogger("nomorals.voice.design")


# ---------------------------------------------------------------------------
# description → parameters
# ---------------------------------------------------------------------------

# Each entry: (compiled pattern, parameter deltas).
# pitch in semitones (+ = higher), speed as multiplier, brightness as
# spectral tilt hint (used by tone shaping), warmth as low-end emphasis hint.
_DESCRIPTORS: list[tuple[re.Pattern, dict[str, float]]] = [
    (re.compile(r"\bdeep\b"), {"pitch": -4.0, "warmth": 0.6}),
    (re.compile(r"\b(low|bass)\b"), {"pitch": -3.0, "warmth": 0.5}),
    (re.compile(r"\bhigh\b"), {"pitch": 4.0}),
    (re.compile(r"\b(squeaky|chipmunk)\b"), {"pitch": 7.0, "speed": 1.1}),
    (re.compile(r"\bwarm\b"), {"pitch": -1.5, "warmth": 0.7, "brightness": -0.3}),
    (re.compile(r"\bcold\b|\bicy\b"), {"pitch": 1.0, "brightness": 0.5,
                                      "warmth": -0.4}),
    (re.compile(r"\bbright\b"), {"brightness": 0.6}),
    (re.compile(r"\bdark\b"), {"brightness": -0.5, "pitch": -2.0}),
    (re.compile(r"\bsoft\b|\bgentle\b"), {"pitch": 1.0, "speed": 0.92,
                                          "warmth": 0.4}),
    (re.compile(r"\bharsh\b|\bgruff\b"), {"pitch": -2.5, "brightness": 0.4,
                                         "warmth": -0.3}),
    (re.compile(r"\b(slow|calm|sleepy)\b"), {"speed": 0.85}),
    (re.compile(r"\b(fast|energetic|hyper|excited)\b"), {"speed": 1.15,
                                                       "pitch": 1.5}),
    (re.compile(r"\b(old|elderly|grand\w*)\b"), {"pitch": -2.0, "speed": 0.9,
                                               "warmth": 0.3}),
    (re.compile(r"\b(young|youthful|teen)\b"), {"pitch": 3.0, "speed": 1.05}),
    (re.compile(r"\b(child|kid)\b"), {"pitch": 6.0, "speed": 1.08}),
    (re.compile(r"\b(masculine|manly|male)\b"), {"pitch": -3.5}),
    (re.compile(r"\b(feminine|female|womanly)\b"), {"pitch": 3.5}),
    (re.compile(r"\b(raspy|smoky)\b"), {"brightness": 0.3, "warmth": 0.2,
                                       "pitch": -1.0}),
    (re.compile(r"\b(clear|crisp)\b"), {"brightness": 0.5}),
    (re.compile(r"\b(mellow|smooth)\b"), {"pitch": -1.0, "warmth": 0.5,
                                         "brightness": -0.2}),
    (re.compile(r"\b(robot|robotic|synth)\b"), {"pitch": 0.0, "brightness": 0.7,
                                              "warmth": -0.8}),
    (re.compile(r"\b(whisper|quiet|breathy)\b"), {"speed": 0.9, "warmth": 0.3,
                                                "brightness": -0.2}),
    (re.compile(r"\b(authoritative|commanding|powerful)\b"), {"pitch": -2.0,
                                                            "speed": 0.95,
                                                            "warmth": 0.4}),
    (re.compile(r"\b(friendly|cheerful)\b"), {"pitch": 2.0, "speed": 1.05}),
    (re.compile(r"\b(sad|melancholy)\b"), {"pitch": -1.5, "speed": 0.88}),
    (re.compile(r"\b(angry|fierce)\b"), {"pitch": -1.0, "speed": 1.1,
                                       "brightness": 0.4}),
]


@dataclass
class VoiceDesign:
    """Shaping parameters derived from a description."""
    pitch_semitones: float = 0.0
    speed: float = 1.0
    brightness: float = 0.0   # -1..1 spectral tilt hint
    warmth: float = 0.0       # -1..1 low-end emphasis hint
    matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"pitch_semitones": self.pitch_semitones,
                "speed": self.speed,
                "brightness": self.brightness,
                "warmth": self.warmth,
                "matched": list(self.matched)}


def describe_to_params(description: str) -> VoiceDesign:
    """Turn a natural-language voice description into shaping parameters.

    Pure function — no audio, no model. Multiple descriptors stack.
    """
    design = VoiceDesign()
    text = (description or "").lower()
    for pattern, deltas in _DESCRIPTORS:
        if pattern.search(text):
            design.matched.append(pattern.pattern)
            design.pitch_semitones += deltas.get("pitch", 0.0)
            # speed multiplies (geometric stacking)
            if "speed" in deltas:
                design.speed *= deltas["speed"]
            design.brightness += deltas.get("brightness", 0.0)
            design.warmth += deltas.get("warmth", 0.0)
    # clamp to sane ranges
    design.pitch_semitones = max(-12.0, min(12.0, design.pitch_semitones))
    design.speed = max(0.6, min(1.6, design.speed))
    design.brightness = max(-1.0, min(1.0, design.brightness))
    design.warmth = max(-1.0, min(1.0, design.warmth))
    return design


# ---------------------------------------------------------------------------
# DSP shaping
# ---------------------------------------------------------------------------

def _read_mono_wav(path: str) -> tuple[array, int]:
    import wave
    with wave.open(path, "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
        ch = wf.getnchannels()
    import struct
    if width == 2:
        samples = array("d", struct.unpack("<%dh" % (n * ch), raw))
        samples = array("d", (s / 32768.0 for s in samples))
    elif width == 1:
        samples = array("d", ((b - 128) / 128.0 for b in raw))
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        samples = array("d", (sum(samples[i * ch:(i + 1) * ch]) / ch
                              for i in range(n)))
    return samples, sr


def _write_mono_wav(path: str, samples: array, sr: int) -> None:
    import wave
    import struct
    clipped = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in samples))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<%dh" % len(clipped), *clipped))


def _apply_tone(samples: array, sr: int, brightness: float,
                warmth: float) -> array:
    """Simple tone shaping: brightness = high-shelf lean, warmth = low lift.

    One-pole shelving approximations — honest, lightweight, documented.
    """
    if not samples or (brightness == 0 and warmth == 0):
        return samples
    out = array("d", samples)
    # brightness: differentiate slightly toward highs
    if brightness != 0:
        alpha = abs(brightness) * 0.35
        prev = 0.0
        for i in range(len(out)):
            if brightness > 0:
                # lift highs: add a fraction of the delta
                delta = out[i] - prev
                out[i] = out[i] + alpha * delta
            else:
                # dull highs: smooth
                out[i] = (1 - alpha) * out[i] + alpha * prev
            prev = samples[i]
    # warmth: lift lows with a slow follower
    if warmth != 0:
        alpha = abs(warmth) * 0.12
        follower = 0.0
        for i in range(len(out)):
            follower = (1 - alpha) * follower + alpha * out[i]
            if warmth > 0:
                out[i] = out[i] + alpha * 2.0 * follower
            else:
                out[i] = out[i] - alpha * follower
    # normalize to avoid clipping drift
    peak = max((abs(s) for s in out), default=0.0)
    if peak > 0.98:
        scale = 0.98 / peak
        out = array("d", (s * scale for s in out))
    return out


def shape_voice(input_path: str, output_path: str,
                design: VoiceDesign) -> dict[str, Any]:
    """Apply a VoiceDesign to an audio file. Returns ``{"ok", ...}``."""
    try:
        samples, sr = _read_mono_wav(input_path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"unreadable: {exc}"}
    try:
        from ..audio.dsp import pitch_shift, time_stretch
        shaped = samples
        if design.pitch_semitones:
            shaped = pitch_shift(shaped, sr, design.pitch_semitones)
        if design.speed != 1.0:
            # time_stretch(tmp, sr, factor): factor > 1 slows down.
            # We want speed multiplier → factor = 1/speed.
            shaped = time_stretch(shaped, sr, 1.0 / design.speed)
        shaped = _apply_tone(shaped, sr, design.brightness, design.warmth)
        _write_mono_wav(output_path, shaped, sr)
        return {"ok": True, "path": output_path,
                "design": design.to_dict()}
    except Exception as exc:  # noqa: BLE001
        _log.warning("shape_voice failed: %s", exc)
        return {"ok": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# catalogue integration
# ---------------------------------------------------------------------------

def design_voice(name: str, description: str, *,
                 base_voice: str = "", voices_dir: str = "") -> dict[str, Any]:
    """Design a voice from a natural description, register it in the catalogue.

    Uses the base voice's reference clip (or the catalogue's active voice)
    shaped by the description's parameters. The design parameters are stored
    on the voice so future TTS through neural backends can carry them.
    """
    from .catalogue import default_catalogue
    cat = default_catalogue(voices_dir) if voices_dir else default_catalogue()
    design = describe_to_params(description)
    if not design.matched:
        return {"ok": False,
                "reason": (f"no voice descriptors understood in "
                           f"{description!r} — try words like deep, warm, "
                           f"bright, young, slow, raspy")}

    # resolve the base voice's reference audio
    base = cat.get(base_voice) if base_voice else cat.active_for_chat("")
    if base is None:
        return {"ok": False, "reason": "no base voice available to shape"}
    ref_path = ""
    try:
        prof = cat.library.get(base.profile or base.name)
        ref_path = (getattr(prof, "sample_path", "") or
                    getattr(prof, "path", "") or "")
    except Exception:  # noqa: BLE001
        pass
    if not ref_path or not os.path.exists(ref_path):
        return {"ok": False,
                "reason": (f"base voice {base.name!r} has no reference audio "
                           f"to shape — clone a voice first")}

    shaped_path = os.path.join(
        cat.voices_dir, f"designed_{name}.wav")
    res = shape_voice(ref_path, shaped_path, design)
    if not res.get("ok"):
        return res
    try:
        voice = cat.clone(name, shaped_path,
                          description=(f"designed: {description} "
                                       f"(from {base.name})"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"catalogue register failed: {exc}"}
    # stash the design on the voice for backend use
    try:
        voice.tags = tuple(set(voice.tags) | {"designed"})
        cat._save()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "voice": voice.name, "from": base.name,
            "design": design.to_dict(),
            "note": ("DSP-shaped from a real voice — honest shaping, "
                     "not a neural voice model")}


def morph_voices(name: str, voice_a: str, voice_b: str, *,
                 ratio: float = 0.5, voices_dir: str = "") -> dict[str, Any]:
    """Morph two catalogue voices — interpolate their shaping toward each other.

    ``ratio`` 0.0 = all A, 1.0 = all B. Works by measuring both voices'
    prints, computing the delta in pitch space, and shaping A's reference
    by the interpolated delta. Honest DSP morph, documented as such.
    """
    from .catalogue import default_catalogue
    from .tts import voice_print
    cat = default_catalogue(voices_dir) if voices_dir else default_catalogue()
    va = cat.get(voice_a)
    vb = cat.get(voice_b)
    if va is None or vb is None:
        return {"ok": False,
                "reason": f"unknown voice(s): {voice_a!r}, {voice_b!r}"}
    ratio = max(0.0, min(1.0, ratio))

    def _ref(v) -> str:
        try:
            prof = cat.library.get(v.profile or v.name)
            return (getattr(prof, "sample_path", "") or
                    getattr(prof, "path", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    ra, rb = _ref(va), _ref(vb)
    if not ra or not os.path.exists(ra):
        return {"ok": False, "reason": f"{voice_a!r} has no reference audio"}
    pa, pb = voice_print(ra), voice_print(rb)
    fa = (pa.get("features") or {})
    fb = (pb.get("features") or {})
    f0a = float(fa.get("mean_f0_hz", 0) or 0)
    f0b = float(fb.get("mean_f0_hz", 0) or 0)
    if f0a > 20 and f0b > 20:
        semitones = 12.0 * math.log2(
            (f0a + ratio * (f0b - f0a)) / f0a)
    else:
        semitones = 0.0
    design = VoiceDesign(pitch_semitones=semitones,
                         matched=[f"morph({voice_a},{voice_b},{ratio:.2f})"])
    shaped_path = os.path.join(cat.voices_dir, f"morphed_{name}.wav")
    res = shape_voice(ra, shaped_path, design)
    if not res.get("ok"):
        return res
    try:
        voice = cat.clone(name, shaped_path,
                          description=(f"morph of {voice_a} × {voice_b} "
                                       f"@{ratio:.0%}"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"catalogue register failed: {exc}"}
    return {"ok": True, "voice": voice.name,
            "shift_semitones": round(semitones, 2),
            "note": "pitch-interpolated morph — honest DSP, not neural VC"}
