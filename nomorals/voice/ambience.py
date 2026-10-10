"""Procedural ambience — environments that actually mix.

``nl_director`` has parsed ``[light rain]`` / ``[phone buzzing]`` for a
while; nothing ever *did* anything with it. This module generates the
environments procedurally (no samples, no downloads, deterministic) and
mixes them under speech with proper ducking, plus a parametric room
reverb (Schroeder-style comb/allpass network — no impulse-response file
needed).

Everything is 16-bit mono sample arrays; stdlib only.
"""

from __future__ import annotations

import math
import random
from array import array
from typing import Any, Optional

__all__ = [
    "AMBIENCES",
    "generate_ambience",
    "mix_under",
    "apply_room",
    "Room",
]

#: Ambience kinds the generator knows. nl_director's ambient patterns map here.
AMBIENCES = (
    "rain", "heavy_rain", "wind", "thunder", "crowd", "applause",
    "room_tone", "phone_line", "traffic", "birds", "ocean", "fire",
)


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _noise(rng: random.Random, n: int, scale: float = 1.0) -> array:
    out = array("h")
    for _ in range(n):
        out.append(int(max(-32768, min(32767, rng.gauss(0, 9000) * scale))))
    return out


def _lowpass_1pole(samples: array, alpha: float) -> array:
    """One-pole lowpass — the workhorse for shaping noise into weather."""
    out = array("h")
    y = 0.0
    for s in samples:
        y += alpha * (s - y)
        out.append(int(max(-32768, min(32767, y))))
    return out


def _rain(sr: int, n: int, rng: random.Random, heavy: bool) -> array:
    # Diffuse body: bandlimited noise (the "shhh")
    body = _lowpass_1pole(_noise(rng, n, 0.5 if not heavy else 0.9), 0.25)
    # Droplets: sparse decaying chirps (the "pitter")
    out = array("h", body)
    drop_n = int(n / sr * (40 if heavy else 12))
    for _ in range(drop_n):
        pos = rng.randrange(n)
        length = rng.randrange(int(sr * 0.01), int(sr * 0.05))
        freq = rng.uniform(1800, 6000)
        amp = rng.uniform(1500, 6000) * (1.6 if heavy else 1.0)
        for i in range(length):
            if pos + i >= n:
                break
            env = math.exp(-i / (length * 0.25))
            f = freq * math.exp(-i / (length * 0.5))  # downward chirp
            out[pos + i] = int(max(-32768, min(
                32767, out[pos + i] + amp * env * math.sin(2 * math.pi * f * i / sr))))
    return out


def _wind(sr: int, n: int, rng: random.Random) -> array:
    base = _lowpass_1pole(_noise(rng, n, 0.6), 0.08)
    # Slow gusts: amplitude modulation by summed sines
    out = array("h")
    for i, s in enumerate(base):
        t = i / sr
        gust = (0.55 + 0.30 * math.sin(2 * math.pi * 0.07 * t)
                + 0.15 * math.sin(2 * math.pi * 0.23 * t + 1.7))
        out.append(int(max(-32768, min(32767, s * gust))))
    return out


def _thunder(sr: int, n: int, rng: random.Random) -> array:
    # One distant rumble: lowpassed noise burst with slow decay,
    # placed ~1/3 in so speech can lead.
    out = array("h", [0] * n)
    start = n // 3
    length = min(n - start, int(sr * 4))
    burst = _lowpass_1pole(_noise(rng, length, 1.4), 0.03)
    for i in range(length):
        env = math.exp(-i / (length * 0.4))
        rumble = 0.7 + 0.3 * math.sin(2 * math.pi * 3 * i / sr)
        out[start + i] = int(max(-32768, min(
            32767, burst[i] * env * rumble)))
    return out


def _crowd(sr: int, n: int, rng: random.Random) -> array:
    # Murmur: many lowpassed noise streams at different rates, none
    # intelligible — the cocktail-party wash.
    out = array("h", [0] * n)
    for voice in range(14):
        stream = _lowpass_1pole(_noise(rng, n, 0.16), 0.35)
        mod_rate = rng.uniform(2.0, 7.0)
        for i in range(n):
            t = i / sr
            # Syllabic babble: gated amplitude
            gate = 0.5 + 0.5 * math.sin(2 * math.pi * mod_rate * t
                                        + rng.uniform(0, 6.28))
            gate = gate * gate
            out[i] = int(max(-32768, min(32767, out[i] + stream[i] * gate)))
    return out


def _applause(sr: int, n: int, rng: random.Random) -> array:
    out = array("h", [0] * n)
    claps = int(n / sr * 25)
    for _ in range(claps):
        pos = rng.randrange(n)
        length = rng.randrange(int(sr * 0.015), int(sr * 0.05))
        amp = rng.uniform(3000, 9000)
        for i in range(length):
            if pos + i >= n:
                break
            env = math.exp(-i / (length * 0.2))
            out[pos + i] = int(max(-32768, min(
                32767, out[pos + i] + amp * env * rng.uniform(-1, 1))))
    return _lowpass_1pole(out, 0.6)


def _simple(kind: str, sr: int, n: int, rng: random.Random) -> array:
    if kind == "room_tone":
        return _lowpass_1pole(_noise(rng, n, 0.06), 0.02)
    if kind == "phone_line":
        # 300–3400Hz-ish band + faint hum
        x = _lowpass_1pole(_noise(rng, n, 0.10), 0.35)
        out = array("h")
        for i, s in enumerate(x):
            hum = 300 * math.sin(2 * math.pi * 50 * i / sr)
            out.append(int(max(-32768, min(32767, s + hum))))
        return out
    if kind == "traffic":
        base = _lowpass_1pole(_noise(rng, n, 0.35), 0.10)
        # Occasional pass-bys: slow swells
        out = array("h")
        for i, s in enumerate(base):
            t = i / sr
            swell = 0.6 + 0.4 * math.sin(2 * math.pi * 0.05 * t) ** 2
            out.append(int(max(-32768, min(32767, s * swell))))
        return out
    if kind == "birds":
        out = array("h", [0] * n)
        chirps = int(n / sr * 3)
        for _ in range(chirps):
            pos = rng.randrange(n)
            length = rng.randrange(int(sr * 0.05), int(sr * 0.2))
            f0 = rng.uniform(2500, 4500)
            for i in range(length):
                if pos + i >= n:
                    break
                env = math.sin(math.pi * i / length)  # smooth chirp envelope
                f = f0 * (1 + 0.3 * math.sin(2 * math.pi * 12 * i / sr))
                out[pos + i] = int(max(-32768, min(
                    32767, out[pos + i]
                    + 2500 * env * math.sin(2 * math.pi * f * i / sr))))
        return out
    if kind == "ocean":
        base = _lowpass_1pole(_noise(rng, n, 0.5), 0.12)
        out = array("h")
        for i, s in enumerate(base):
            t = i / sr
            wave_m = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.12 * t)) ** 2
            out.append(int(max(-32768, min(32767, s * wave_m))))
        return out
    if kind == "fire":
        # Crackle: sparse sharp pops over low rumble
        out = _lowpass_1pole(_noise(rng, n, 0.25), 0.05)
        pops = int(n / sr * 30)
        for _ in range(pops):
            pos = rng.randrange(n)
            length = rng.randrange(4, int(sr * 0.02))
            amp = rng.uniform(2000, 8000)
            for i in range(length):
                if pos + i >= n:
                    break
                out[pos + i] = int(max(-32768, min(
                    32767, out[pos + i]
                    + amp * math.exp(-i / 8) * rng.uniform(-1, 1))))
        return out
    return _lowpass_1pole(_noise(rng, n, 0.2), 0.1)


def generate_ambience(kind: str, duration_s: float,
                      sample_rate: int = 24000, seed: int = 7) -> tuple[array, int]:
    """Deterministic procedural ambience. (samples, sample_rate).

    Unknown kinds fall back to room_tone — never raise.
    """
    kind = (kind or "").lower().strip().replace(" ", "_")
    n = max(1, int(sample_rate * max(0.1, duration_s)))
    rng = _rng(seed + hash(kind) % 100000)
    if kind in ("rain", "light_rain"):
        gen = _rain(sr=sample_rate, n=n, rng=rng, heavy=False)
    elif kind in ("heavy_rain", "storm"):
        gen = _rain(sr=sample_rate, n=n, rng=rng, heavy=True)
    elif kind == "wind":
        gen = _wind(sample_rate, n, rng)
    elif kind == "thunder":
        gen = _thunder(sample_rate, n, rng)
    elif kind in ("crowd", "murmur", "cafe"):
        gen = _crowd(sample_rate, n, rng)
    elif kind == "applause":
        gen = _applause(sample_rate, n, rng)
    else:
        gen = _simple(kind, sample_rate, n, rng)
    return gen, sample_rate


def mix_under(voice: array, ambience: array, ambience_level: float = 0.18,
              duck: bool = True) -> array:
    """Mix ambience under voice with sidechain-style ducking.

    When the voice is loud, the ambience drops; in pauses it breathes back.
    """
    n = len(voice)
    m = len(ambience)
    if m == 0:
        return array("h", voice)
    out = array("h")
    for i in range(n):
        a = ambience[i % m]
        v = voice[i]
        if duck:
            # Duck amount from voice envelope (simple, causal)
            env = min(1.0, abs(v) / 12000.0)
            duck_gain = 1.0 - 0.6 * env
        else:
            duck_gain = 1.0
        mixed = v + a * ambience_level * duck_gain
        out.append(int(max(-32768, min(32767, mixed))))
    return out


class Room:
    """Parametric room: small/large/hall/cave/phone."""
    PRESETS: dict[str, dict[str, float]] = {
        "small":  {"decay": 0.25, "wet": 0.12, "damping": 0.5},
        "room":   {"decay": 0.45, "wet": 0.18, "damping": 0.4},
        "hall":   {"decay": 1.4,  "wet": 0.28, "damping": 0.25},
        "cave":   {"decay": 2.8,  "wet": 0.35, "damping": 0.15},
        "phone":  {"decay": 0.08, "wet": 0.06, "damping": 0.7},
    }

    def __init__(self, preset: str = "room", **over):
        p = dict(self.PRESETS.get(preset, self.PRESETS["room"]))
        p.update(over)
        self.decay = p["decay"]
        self.wet = p["wet"]
        self.damping = p["damping"]


def apply_room(samples: array, sample_rate: int,
               room: Room | str = "room") -> array:
    """Schroeder reverb: parallel combs → series allpasses.

    Pure Python, no IR files. Honest scope: a believable space, not a
    convolution of a real hall.
    """
    if isinstance(room, str):
        room = Room(room)
    # Comb delays (ms) tuned for 24k; scale with sample rate
    comb_ms = (29.7, 37.1, 41.1, 43.7)
    ap_ms = (5.0, 1.7)
    combs = [int(sample_rate * ms / 1000) for ms in comb_ms]
    aps = [int(sample_rate * ms / 1000) for ms in ap_ms]
    decay_gain = 10 ** (-3 * max(combs) / (room.decay * sample_rate))
    # Parallel combs with damping
    comb_bufs = [[0.0] * d for d in combs]
    comb_idx = [0] * len(combs)
    comb_lp = [0.0] * len(combs)
    wet = array("h")
    for s in samples:
        acc = 0.0
        for ci, d in enumerate(combs):
            buf = comb_bufs[ci]
            i = comb_idx[ci]
            delayed = buf[i]
            # Damping: lowpass the feedback
            comb_lp[ci] += (1 - room.damping) * (delayed - comb_lp[ci])
            buf[i] = s + comb_lp[ci] * decay_gain
            comb_idx[ci] = (i + 1) % d
            acc += delayed
        wet.append(int(max(-32768, min(32767, acc / len(combs)))))
    # Series allpasses
    for d in aps:
        buf = [0.0] * d
        i = 0
        g = 0.7
        out = array("h")
        for s in wet:
            delayed = buf[i]
            y = -g * s + delayed
            buf[i] = s + g * delayed
            i = (i + 1) % d
            out.append(int(max(-32768, min(32767, y))))
        wet = out
    # Dry/wet mix
    out = array("h")
    for dry, w in zip(samples, wet):
        out.append(int(max(-32768, min(
            32767, dry * (1 - room.wet) + w * room.wet))))
    return out


def with_ambience(voice: array, sample_rate: int, ambient_desc: str,
                  room: str = "") -> array:
    """One call: parse an ambient description, generate, mix, room.

    ``ambient_desc`` is the free text from nl_director (e.g. "light rain").
    """
    desc = (ambient_desc or "").lower()
    kind = "room_tone"
    for k in AMBIENCES:
        if k.replace("_", " ") in desc or k in desc:
            kind = k
            break
    # "light"/"heavy" modifiers
    if "heavy" in desc or "storm" in desc:
        kind = "heavy_rain" if "rain" in kind else kind
    dur = len(voice) / max(1, sample_rate)
    amb, _ = generate_ambience(kind, dur, sample_rate)
    mixed = mix_under(voice, amb)
    if room:
        mixed = apply_room(mixed, sample_rate, room)
    return mixed
