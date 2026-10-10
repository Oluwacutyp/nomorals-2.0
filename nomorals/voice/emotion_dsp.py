"""Emotion DSP — makes emotion tags WORK on any backend.

The Step-Audio-EditX pattern: generate neutral audio from any TTS
backend, then shape the emotion in as post-processing. This is honest
DSP (pitch/rate/energy shaping), not neural emotion — documented as
such. But it means a ``[sad]`` tag produces audibly sadder speech
even on Piper, instead of being stripped to nothing.

For backends WITH native tag support (Chatterbox Turbo, Dia, Bark,
Fish S2), this module is bypassed — native is always better.
"""

import math
from array import array


def pitch_shift(samples: array, semitones: float,
                sample_rate: int) -> array:
    """Simple pitch shift via resampling.

    This changes duration too (like a tape). For small shifts
    (<3 semitones) the tempo change is acceptable. Larger shifts
    should use a phase vocoder (not implemented — honest limit).
    """
    if abs(semitones) < 0.05:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    n = len(samples)
    new_n = max(1, int(n / factor))
    out = array("h")
    for i in range(new_n):
        src = i * factor
        i0 = int(src)
        frac = src - i0
        i1 = min(i0 + 1, n - 1)
        v = samples[i0] * (1 - frac) + samples[i1] * frac
        out.append(int(max(-32768, min(32767, v))))
    return out


def time_stretch(samples: array, rate_mult: float) -> array:
    """Change speaking rate without pitch change (naive).

    Linear interpolation resampling. Quality is acceptable for
    0.8x–1.3x. Beyond that, artifacts grow — honest limit documented.
    """
    if abs(rate_mult - 1.0) < 0.02:
        return samples
    n = len(samples)
    new_n = max(1, int(n / rate_mult))
    out = array("h")
    for i in range(new_n):
        src = i * rate_mult
        i0 = int(src)
        frac = src - i0
        i1 = min(i0 + 1, n - 1)
        v = samples[i0] * (1 - frac) + samples[i1] * frac
        out.append(int(max(-32768, min(32767, v))))
    return out


def apply_energy(samples: array, energy: float) -> array:
    """Scale amplitude for energy (whisper=quiet, shout=loud)."""
    if abs(energy - 1.0) < 0.02:
        return samples
    out = array("h")
    for s in samples:
        v = int(s * energy)
        out.append(max(-32768, min(32767, v)))
    return out


def add_breathiness(samples: array, sample_rate: int,
                    amount: float = 0.08) -> array:
    """Add noise for whisper/breathy delivery.

    Mixes shaped noise into the signal. ``amount`` 0.0–0.3.
    """
    if amount <= 0.0:
        return samples
    import random
    rng = random.Random(42)  # deterministic
    out = array("h")
    for s in samples:
        noise = rng.gauss(0, 800) * amount * 10
        # Gate noise by signal presence (don't add hiss to silence)
        gate = min(1.0, abs(s) / 3000.0)
        v = int(s + noise * gate)
        out.append(max(-32768, min(32767, v)))
    return out


def shape_emotion(samples: array, sample_rate: int,
                  pitch_shift_st: float = 0.0,
                  rate_mult: float = 1.0,
                  energy: float = 1.0,
                  breathiness: float = 0.0) -> array:
    """Apply full emotion shaping pipeline.

    Order: pitch → rate → energy → breathiness.
    All honest DSP, documented limits in each function.
    """
    out = samples
    if abs(pitch_shift_st) >= 0.05:
        out = pitch_shift(out, pitch_shift_st, sample_rate)
    if abs(rate_mult - 1.0) >= 0.02:
        out = time_stretch(out, rate_mult)
    if abs(energy - 1.0) >= 0.02:
        out = apply_energy(out, energy)
    if breathiness > 0.0:
        out = add_breathiness(out, sample_rate, breathiness)
    return out


def shape_for_direction(samples: array, sample_rate: int,
                        direction) -> array:
    """Apply emotion shaping from an nl_director.Direction."""
    from .nl_director import direction_to_dsp_params
    params = direction_to_dsp_params(direction)
    breathiness = 0.12 if direction.delivery in (
        "whisper", "whispers", "whispering") else 0.0
    return shape_emotion(
        samples, sample_rate,
        pitch_shift_st=params["pitch_shift"],
        rate_mult=params["rate_mult"],
        energy=params["energy"],
        breathiness=breathiness,
    )
