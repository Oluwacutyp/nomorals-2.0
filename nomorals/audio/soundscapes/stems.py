"""Procedural scene stems — every sound a scene is built from.

Each stem is synthesised in code from first principles (filtered
noise, swept sines, sparse impulses) and labelled with provenance
"synthesised" — no field recording is ever faked, and no stem
pretends to be a recording it is not. A stem renders ``n`` mono
samples at ``sr``; every stem is deterministic per seed and peaks
near 0.7 so a scene mix stays predictable.

A recorded stem can join later: a WAV file dropped under
``<home>/soundscapes/stems/<name>.wav`` is mixed in looped, and the
stem's provenance becomes "recorded". The shipped catalogue carries
no recordings — the engine is the portable part.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

__all__ = ["STEMS", "stem_names"]

# A stem is a function (n, sr, seed) -> mono floats in [-1, 1].


def _noise(n: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _normalize(samples: list[float], target: float = 0.7) -> list[float]:
    peak = max((abs(s) for s in samples), default=0.0)
    if peak <= 0.0:
        return samples
    scale = target / peak
    return [s * scale for s in samples]


def _lowpass(samples: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    y = 0.0
    for x in samples:
        y += alpha * (x - y)
        out.append(y)
    return out


def _highpass(samples: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    y = 0.0
    prev = 0.0
    for x in samples:
        y = alpha * (y + x - prev)
        prev = x
        out.append(y)
    return out


def _smooth_random_walk(n: int, sr: int, rng: random.Random,
                        rate_hz: float, lo: float, hi: float) -> list[float]:
    """A slowly wandering control signal between lo and hi."""
    out: list[float] = []
    value = (lo + hi) / 2.0
    target = rng.uniform(lo, hi)
    step = max(1, int(sr / (rate_hz * 20.0)))
    for i in range(n):
        if i % step == 0:
            target = rng.uniform(lo, hi)
        value += (target - value) * 0.02
        out.append(value)
    return out


def _burst(out: list[float], at: int, freq: float, sr: int,
           dur_s: float, amp: float, sweep_to: float | None = None) -> None:
    """Add one damped sine burst at sample ``at`` (in place)."""
    n = int(dur_s * sr)
    end = min(len(out), at + n)
    for i in range(at, end):
        t = (i - at) / sr
        f = freq if sweep_to is None else freq + (sweep_to - freq) * (i - at) / n
        env = math.exp(-6.0 * (i - at) / n)
        out[i] += amp * env * math.sin(2.0 * math.pi * f * t)


def stem_rain(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Steady rain: broadband patter with droplet pings."""
    rng = random.Random(seed)
    bed = _lowpass(_noise(n, rng), 0.25)
    bed = _highpass(bed, 0.02)
    for _ in range(max(1, n // (sr // 2))):
        at = rng.randrange(0, n)
        _burst(bed, at, rng.uniform(2000.0, 6000.0), sr,
               rng.uniform(0.004, 0.010), rng.uniform(0.15, 0.4))
    return _normalize(bed)


def stem_drizzle(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Light drizzle: softer, higher, sparser than rain."""
    rng = random.Random(seed)
    bed = _lowpass(_noise(n, rng), 0.12)
    bed = _highpass(bed, 0.05)
    for _ in range(max(1, n // (sr * 2))):
        at = rng.randrange(0, n)
        _burst(bed, at, rng.uniform(3500.0, 7000.0), sr,
               rng.uniform(0.003, 0.007), rng.uniform(0.1, 0.25))
    return _normalize(bed, 0.5)


def stem_thunder(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Distant thunder: low rumble swells, minutes apart in feel."""
    rng = random.Random(seed)
    bed = _lowpass(_noise(n, rng), 0.008)
    swells = _smooth_random_walk(n, sr, rng, 0.05, 0.15, 1.0)
    out = [s * w for s, w in zip(bed, swells, strict=True)]
    return _normalize(out)


def stem_wind(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Wind: pink-ish bed breathing on a slow swell, cutoff wandering."""
    rng = random.Random(seed)
    bed = _noise(n, rng)
    # Pink-ish: three cascaded one-pole stages.
    for alpha in (0.08, 0.05, 0.03):
        bed = _lowpass(bed, alpha)
    swell = _smooth_random_walk(n, sr, rng, 0.08, 0.25, 1.0)
    out = [s * w for s, w in zip(bed, swell, strict=True)]
    return _normalize(out)


def stem_leaves(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Rustling leaves: high shimmer in gusts."""
    rng = random.Random(seed)
    bed = _highpass(_noise(n, rng), 0.35)
    gusts = _smooth_random_walk(n, sr, rng, 0.15, 0.1, 1.0)
    out = [s * w for s, w in zip(bed, gusts, strict=True)]
    return _normalize(out, 0.5)


def stem_stream(n: int, sr: int, seed: int | None = None) -> list[float]:
    """A stream: bright burble over a steady bed."""
    rng = random.Random(seed)
    bed = _highpass(_noise(n, rng), 0.08)
    burble = _smooth_random_walk(n, sr, rng, 0.4, 0.5, 1.0)
    out = [s * w for s, w in zip(bed, burble, strict=True)]
    return _normalize(out)


def stem_ocean(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Ocean: slow swell under surf that breaks on each swell peak."""
    rng = random.Random(seed)
    bed = _lowpass(_noise(n, rng), 0.02)
    swell = [0.5 + 0.5 * math.sin(2.0 * math.pi * 0.09 * i / sr)
             for i in range(n)]
    # Surf: highpassed noise gated by the swell's rising edge.
    surf = _highpass(_noise(n, rng), 0.10)
    out = []
    prev = 0.0
    for i in range(n):
        gate = max(0.0, swell[i] - prev)
        prev = swell[i]
        out.append(bed[i] * (0.4 + 0.6 * swell[i]) + surf[i] * gate * 2.0)
    return _normalize(out)


def stem_fire(n: int, sr: int, seed: int | None = None) -> list[float]:
    """A fire bed: warm low roar with crackle impulses."""
    rng = random.Random(seed)
    bed = _lowpass(_noise(n, rng), 0.05)
    for _ in range(n // (sr // 100)):
        at = rng.randrange(0, n)
        amp = rng.uniform(0.3, 1.0)
        decay = rng.uniform(0.002, 0.012)
        span = min(n - at, int(decay * sr * 4))
        for i in range(span):
            bed[at + i] += amp * math.exp(-(i / sr) / decay) * (
                1.0 if rng.random() > 0.5 else -1.0) * 0.3
    return _normalize(bed)


def stem_birds(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Birdsong: sparse chirps — 2–4 swept blips each — never constant."""
    rng = random.Random(seed)
    out = [0.0] * n
    chirps = max(1, n // (sr * 2))
    for _ in range(chirps):
        at = rng.randrange(0, max(1, n - sr // 2))
        blips = rng.randint(2, 4)
        for b in range(blips):
            start = at + b * int(0.12 * sr)
            f0 = rng.uniform(2400.0, 3400.0)
            _burst(out, start, f0, sr, 0.09,
                   rng.uniform(0.3, 0.6), sweep_to=f0 * 1.4)
    return _normalize(out)


def stem_crickets(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Crickets: a 4.2 kHz carrier pulsed ~18 Hz, in slow phrases."""
    rng = random.Random(seed)
    out: list[float] = []
    phrase = _smooth_random_walk(n, sr, rng, 0.1, 0.2, 1.0)
    for i in range(n):
        t = i / sr
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 18.0 * t)
        carrier = math.sin(2.0 * math.pi * 4200.0 * t)
        out.append(0.5 * pulse * carrier * phrase[i])
    return _normalize(out, 0.5)


def stem_murmur(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Cafe murmur: band-limited voices as syllabic swells.

    Synthesised — formant-shaped noise with speech-like amplitude
    segments. It reads as a room full of quiet talk, never as words.
    """
    rng = random.Random(seed)
    bed = _lowpass(_noise(n, rng), 0.10)
    bed = _highpass(bed, 0.015)
    # Syllabic envelope: random segment gains, smoothed.
    seg_len = max(1, sr // 8)
    gains = [rng.uniform(0.2, 1.0) for _ in range(n // seg_len + 2)]
    out: list[float] = []
    prev = gains[0]
    for i, s in enumerate(bed):
        g = gains[i // seg_len]
        prev += (g - prev) * 0.05
        out.append(s * prev)
    return _normalize(out, 0.6)


def stem_clinks(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Cup clinks: sparse damped high pings."""
    rng = random.Random(seed)
    out = [0.0] * n
    for _ in range(max(1, n // (sr * 3))):
        at = rng.randrange(0, n)
        _burst(out, at, rng.uniform(1800.0, 5200.0), sr,
               rng.uniform(0.06, 0.15), rng.uniform(0.2, 0.5))
    return _normalize(out, 0.5)


def stem_gulls(n: int, sr: int, seed: int | None = None) -> list[float]:
    """Distant gulls: slow descending cries, far apart."""
    rng = random.Random(seed)
    out = [0.0] * n
    for _ in range(max(1, n // (sr * 4))):
        at = rng.randrange(0, max(1, n - sr))
        _burst(out, at, rng.uniform(1400.0, 1700.0), sr, 0.4,
               rng.uniform(0.15, 0.3), sweep_to=900.0)
    return _normalize(out, 0.5)


def stem_owl(n: int, sr: int, seed: int | None = None) -> list[float]:
    """An owl: low hoot pairs, far apart, slightly trembling."""
    rng = random.Random(seed)
    out = [0.0] * n
    for _ in range(max(1, n // (sr * 6))):
        at = rng.randrange(0, max(1, n - sr))
        for k in range(2):
            start = at + k * int(0.35 * sr)
            dur = int(0.25 * sr)
            for i in range(start, min(n, start + dur)):
                t = (i - start) / sr
                env = math.sin(math.pi * (i - start) / dur) ** 2
                vib = 1.0 + 0.03 * math.sin(2.0 * math.pi * 6.0 * t)
                out[i] += 0.4 * env * math.sin(
                    2.0 * math.pi * 380.0 * vib * t)
    return _normalize(out, 0.5)


STEMS: dict[str, Callable[[int, int, int | None], list[float]]] = {
    "rain": stem_rain,
    "drizzle": stem_drizzle,
    "thunder": stem_thunder,
    "wind": stem_wind,
    "leaves": stem_leaves,
    "stream": stem_stream,
    "ocean": stem_ocean,
    "fire": stem_fire,
    "birds": stem_birds,
    "crickets": stem_crickets,
    "murmur": stem_murmur,
    "clinks": stem_clinks,
    "gulls": stem_gulls,
    "owl": stem_owl,
}


def stem_names() -> list[str]:
    """The stem catalogue, sorted."""
    return sorted(STEMS)
