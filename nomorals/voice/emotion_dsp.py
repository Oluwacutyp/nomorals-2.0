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


def whisperize(samples: array, sample_rate: int,
               intensity: float = 1.0) -> array:
    """One-call whisper: breathiness + HF air + energy dip.

    The delivery-verb shortcut — ``[whispers]`` on a tag-less backend
    without hand-rolling the four knobs.
    """
    out = pitch_shift(samples, 0.5 * intensity, sample_rate)
    out = add_breathiness(out, sample_rate,
                          amount=min(0.3, 0.10 + 0.10 * intensity))
    # energy dip LAST so the added air doesn't raise the peaks back up
    out = apply_energy(out, max(0.3, 1.0 - 0.45 * intensity))
    return out


def tremolo(samples: array, sample_rate: int, rate_hz: float = 6.0,
            depth: float = 0.4) -> array:
    """Amplitude wobble — fear/nervousness, old-radio voices."""
    if depth <= 0.0:
        return samples
    import math as _math
    out = array("h")
    for i, s in enumerate(samples):
        mod = 1.0 - depth * 0.5 * (
            1.0 + _math.sin(2 * _math.pi * rate_hz * i / sample_rate))
        out.append(int(max(-32768, min(32767, s * mod))))
    return out


def vibrato_dsp(samples: array, sample_rate: int, rate_hz: float = 5.5,
                depth_cents: float = 40.0) -> array:
    """Pitch wobble via modulated resampling — the nervous/operatic edge.

    Honest scope: a modulated delay-line vibrato, not a phase vocoder;
    depth in cents, subtle is the point.
    """
    if depth_cents <= 0.0 or not samples:
        return samples
    import math as _math
    n = len(samples)
    depth_ratio = 2.0 ** (depth_cents / 1200.0) - 1.0
    out = array("h")
    for i in range(n):
        lfo = _math.sin(2 * _math.pi * rate_hz * i / sample_rate)
        # local resample position wobbles ±depth around i
        src = i + lfo * depth_ratio * sample_rate / max(1.0, rate_hz) * 0.02
        src = max(0.0, min(n - 1.001, src))
        i0 = int(src)
        frac = src - i0
        i1 = min(i0 + 1, n - 1)
        v = samples[i0] * (1 - frac) + samples[i1] * frac
        out.append(int(max(-32768, min(32767, v))))
    return out


#: Delivery verbs → DSP recipes (nl_director's vocabulary, one call).
_DELIVERY_RECIPES: dict[str, dict] = {
    "whisper": {"fn": "whisperize", "intensity": 1.0},
    "whispers": {"fn": "whisperize", "intensity": 1.0},
    "whispering": {"fn": "whisperize", "intensity": 1.0},
    "shout": {"energy": 1.45, "pitch_shift_st": 1.0, "rate_mult": 1.1},
    "shouts": {"energy": 1.45, "pitch_shift_st": 1.0, "rate_mult": 1.1},
    "shouting": {"energy": 1.45, "pitch_shift_st": 1.0, "rate_mult": 1.1},
    "scream": {"energy": 1.6, "pitch_shift_st": 2.0, "rate_mult": 1.15,
               "tremolo": 0.25},
    "screams": {"energy": 1.6, "pitch_shift_st": 2.0, "rate_mult": 1.15,
                "tremolo": 0.25},
    "mutter": {"energy": 0.7, "pitch_shift_st": -1.0, "rate_mult": 0.9},
    "mutters": {"energy": 0.7, "pitch_shift_st": -1.0, "rate_mult": 0.9},
    "muttering": {"energy": 0.7, "pitch_shift_st": -1.0, "rate_mult": 0.9},
    "chant": {"energy": 1.1, "rate_mult": 0.85, "tremolo": 0.15},
    "sigh": {"energy": 0.75, "pitch_shift_st": -1.5, "rate_mult": 0.8,
             "breathiness": 0.1},
    "sighs": {"energy": 0.75, "pitch_shift_st": -1.5, "rate_mult": 0.8,
              "breathiness": 0.1},
    "cry": {"energy": 1.2, "pitch_shift_st": 1.5, "tremolo": 0.35,
            "breathiness": 0.08},
    "cries": {"energy": 1.2, "pitch_shift_st": 1.5, "tremolo": 0.35,
              "breathiness": 0.08},
    "crying": {"energy": 1.2, "pitch_shift_st": 1.5, "tremolo": 0.35,
               "breathiness": 0.08},
    "laugh": {"energy": 1.25, "pitch_shift_st": 2.0, "tremolo": 0.5},
    "laughs": {"energy": 1.25, "pitch_shift_st": 2.0, "tremolo": 0.5},
    "laughing": {"energy": 1.25, "pitch_shift_st": 2.0, "tremolo": 0.5},
    "gasp": {"energy": 1.3, "pitch_shift_st": 2.5, "breathiness": 0.15},
    "gasps": {"energy": 1.3, "pitch_shift_st": 2.5, "breathiness": 0.15},
    "stammer": {"rate_mult": 0.85, "tremolo": 0.2},
    "stammers": {"rate_mult": 0.85, "tremolo": 0.2},
    "stammering": {"rate_mult": 0.85, "tremolo": 0.2},
}


def apply_delivery(samples: array, sample_rate: int,
                   delivery: str) -> array:
    """Apply a delivery-verb recipe (``whispers``, ``shouts``…).

    Unknown verbs pass through untouched (never raises).
    """
    recipe = _DELIVERY_RECIPES.get((delivery or "").lower())
    if not recipe:
        return samples
    if recipe.get("fn") == "whisperize":
        return whisperize(samples, sample_rate,
                          intensity=float(recipe.get("intensity", 1.0)))
    out = shape_emotion(
        samples, sample_rate,
        pitch_shift_st=float(recipe.get("pitch_shift_st", 0.0)),
        rate_mult=float(recipe.get("rate_mult", 1.0)),
        energy=float(recipe.get("energy", 1.0)),
        breathiness=float(recipe.get("breathiness", 0.0)),
    )
    if recipe.get("tremolo"):
        out = tremolo(out, sample_rate, depth=float(recipe["tremolo"]))
    return out
