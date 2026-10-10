"""Mastering — the post-synthesis polish chain.

Mined from dhwani's pipeline: dynamic EQ, de-esser, multiband compressor,
LUFS normalization, true-peak limiter. This is the honest pure-Python
subset that actually matters for speech:

1. DC removal (cheap highpass)
2. De-ess (sibilance tamer — the "sss" harshness)
3. Normalize to a broadcast-ish target level
4. Soft limiter (true-peak-ish ceiling, no clipping)

Every stage is optional and the chain reports what ran. Nothing here
pretends to be a studio — it just makes every backend's output sit at a
consistent, non-harsh level.
"""

from __future__ import annotations

import math
from array import array

__all__ = [
    "remove_dc",
    "deess",
    "normalize",
    "soft_limit",
    "trim_silence",
    "fade_in_out",
    "loudness_match",
    "compress",
    "master",
]


def remove_dc(samples: array) -> array:
    """Cheap DC blocker: one-pole highpass at ~20Hz."""
    out = array("h")
    y = 0.0
    prev = 0.0
    alpha = 0.995
    for s in samples:
        x = float(s)
        y = alpha * (y + x - prev)
        prev = x
        out.append(int(max(-32768, min(32767, y))))
    return out


def deess(samples: array, sample_rate: int,
          amount: float = 0.4) -> array:
    """Tame sibilance: detect high-frequency bursts, dip them.

    Sibilance lives ~5–9kHz. We approximate with a highpassed energy
    detector (difference of the signal and its lowpassed self) and apply
    a fast gain dip when it spikes. ``amount`` 0.0–1.0.
    """
    if amount <= 0.0:
        return samples
    # Lowpass for the "body" reference
    alpha = 0.12
    body = 0.0
    out = array("h")
    for s in samples:
        body += alpha * (s - body)
        sibilant = abs(s - body)
        # Fast attack/release gain dip
        if not hasattr(deess, "_g"):
            deess._g = 1.0  # noqa: SLF001 — simple stateful DSP
        target = 1.0
        if sibilant > 9000:
            target = max(1.0 - amount, 1.0 - amount * (sibilant / 20000))
        # attack fast, release slow
        rate = 0.3 if target < deess._g else 0.02
        deess._g += rate * (target - deess._g)
        out.append(int(max(-32768, min(32767, s * deess._g))))
    try:
        del deess._g  # noqa: SLF001 — reset for next call
    except AttributeError:
        pass
    return out


def normalize(samples: array, target_peak: float = 0.89) -> array:
    """Peak-normalize to target_peak (0.0–1.0). No-op on silence."""
    if not samples:
        return samples
    peak = max(abs(s) for s in samples)
    if peak < 10:
        return samples
    gain = (32767 * target_peak) / peak
    # Don't amplify noise floors aggressively
    gain = min(gain, 8.0)
    out = array("h")
    for s in samples:
        out.append(int(max(-32768, min(32767, s * gain))))
    return out


def soft_limit(samples: array, ceiling: float = 0.98) -> array:
    """Soft-clip limiter: tanh-style curve above the ceiling region."""
    out = array("h")
    c = 32767 * ceiling
    for s in samples:
        x = float(s)
        ax = abs(x)
        if ax <= c:
            out.append(int(x))
        else:
            # Soft knee: compress the overshoot logarithmically
            over = ax - c
            y = c + (32767 - c) * (1 - math.exp(-over / (32767 - c) * 3))
            out.append(int(math.copysign(y, x)))
    return out


def master(samples: array, sample_rate: int = 24000,
           target_peak: float = 0.89,
           deess_amount: float = 0.4,
           trim: bool = False,
           fade_ms: float = 0.0,
           loudness_db: float | None = None,
           compress_amount: float = 0.0) -> tuple[array, list[str]]:
    """Full chain. Returns (samples, stages_that_ran).

    - ``trim``: strip leading/trailing digital silence first.
    - ``fade_ms``: symmetric fade in/out (kills edge clicks).
    - ``loudness_db``: RMS-match to this dBFS before peak normalize
      (the LUFS-inspired level ride — honest RMS, not gated LUFS).
    - ``compress_amount`` 0–1: gentle feedforward compression for
      shouted/whispered takes with wild dynamics.
    """
    stages: list[str] = []
    out = samples
    if trim:
        out = trim_silence(out)
        stages.append("trim")
    out = remove_dc(out)
    stages.append("dc")
    if compress_amount > 0:
        out = compress(out, amount=compress_amount)
        stages.append("compress")
    out = deess(out, sample_rate, deess_amount)
    stages.append("deess")
    if loudness_db is not None:
        out = loudness_match(out, loudness_db)
        stages.append("loudness")
    out = normalize(out, target_peak)
    stages.append("normalize")
    if fade_ms > 0:
        out = fade_in_out(out, sample_rate, fade_ms)
        stages.append("fade")
    out = soft_limit(out)
    stages.append("limit")
    return out, stages


def trim_silence(samples: array, threshold: int = 200) -> array:
    """Strip leading/trailing near-silence. No-op on all-silence."""
    n = len(samples)
    start = 0
    while start < n and abs(samples[start]) < threshold:
        start += 1
    end = n
    while end > start and abs(samples[end - 1]) < threshold:
        end -= 1
    if start == 0 and end == n:
        return samples
    return array("h", samples[start:end])


def fade_in_out(samples: array, sample_rate: int,
                fade_ms: float = 25.0) -> array:
    """Symmetric raised-cosine fades — kills edge clicks on splices."""
    n = len(samples)
    fade = min(n // 2, int(sample_rate * fade_ms / 1000))
    if fade <= 1:
        return samples
    out = array("h", samples)
    for i in range(fade):
        g = 0.5 - 0.5 * math.cos(math.pi * i / fade)
        out[i] = int(out[i] * g)
        out[n - 1 - i] = int(out[n - 1 - i] * g)
    return out


def _rms_db(samples: array) -> float:
    if not samples:
        return -96.0
    mean_sq = sum(float(s) * s for s in samples) / len(samples)
    if mean_sq <= 0:
        return -96.0
    return 10.0 * math.log10(mean_sq / (32768.0 ** 2))


def loudness_match(samples: array, target_db: float = -20.0) -> array:
    """RMS level ride toward target_db dBFS (LUFS-inspired, honest RMS).

    Capped at ±12dB so a whisper doesn't become a noise floor showcase.
    """
    if not samples:
        return samples
    current = _rms_db(samples)
    if current <= -90:
        return samples
    gain_db = max(-12.0, min(12.0, target_db - current))
    gain = 10.0 ** (gain_db / 20.0)
    out = array("h")
    for s in samples:
        out.append(int(max(-32768, min(32767, s * gain))))
    return out


def compress(samples: array, amount: float = 0.5, threshold_db: float = -18.0,
             ratio: float = 3.0, attack_ms: float = 5.0,
             release_ms: float = 80.0,
             sample_rate: int = 24000) -> array:
    """Feedforward compressor — tames wild shout/whisper dynamics.

    ``amount`` 0–1 blends dry/wet (parallel compression at low values).
    Simple peak detector, honest scope: not multiband, not a studio.
    """
    if amount <= 0.0 or not samples:
        return samples
    thresh = 32768.0 * (10.0 ** (threshold_db / 20.0))
    attack = math.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    release = math.exp(-1.0 / (sample_rate * release_ms / 1000.0))
    env = 0.0
    wet = array("h")
    for s in samples:
        peak = abs(float(s))
        coeff = attack if peak > env else release
        env = coeff * env + (1.0 - coeff) * peak
        gain = 1.0
        if env > thresh:
            over_db = 20.0 * math.log10(env / thresh)
            gain = 10.0 ** (-(over_db - over_db / ratio) / 20.0)
        v = s * gain
        wet.append(int(max(-32768, min(32767, v))))
    out = array("h")
    for d, w in zip(samples, wet):
        out.append(int(d * (1.0 - amount) + w * amount))
    return out
