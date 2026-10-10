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
           deess_amount: float = 0.4) -> tuple[array, list[str]]:
    """Full chain. Returns (samples, stages_that_ran)."""
    stages: list[str] = []
    out = remove_dc(samples)
    stages.append("dc")
    out = deess(out, sample_rate, deess_amount)
    stages.append("deess")
    out = normalize(out, target_peak)
    stages.append("normalize")
    out = soft_limit(out)
    stages.append("limit")
    return out, stages
