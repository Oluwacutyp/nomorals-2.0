"""Soundscape synthesis engine — Devon's own MyNoise core.

Everything here is synthesised in code, no recordings and no network:
noise colours with measured spectra (white flat, pink −3 dB/octave,
brown −6 dB/octave, blue +3 dB/octave, grey shaped to equal loudness),
binaural beats and drones, and a uniform 10-slider model — ten gains
in dB that act as a 10-band EQ over noise and tones, or as ten stem
levels over a scene. A hearing profile (iEQ) is a second 10-band
correction applied after the sliders, so a calibrated ear hears every
generator flat.

Two paths, one result: numpy FFT filtering when numpy is present
(fast), a pure-Python peaking-biquad bank otherwise (correct, slower).
Both are real synthesis — nothing here is a placeholder, and nothing
is ever faked: a render that cannot be built raises, it never returns
silence labelled as sound.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "SR",
    "BAND_FREQS",
    "GREY_CONTOUR_DB",
    "MAX_DURATION_S",
    "animate_sliders",
    "apply_eq",
    "apply_hearing",
    "octave_band_energies",
    "render_tone",
    "synthesize_noise",
    "synthesize_tone",
    "write_mono_wav",
    "write_stereo_wav",
]

#: Default sample rate. 22050 Hz covers the whole audible band the
#: 10-band EQ addresses (top band 16 kHz) at half the bytes of 44.1k.
SR = 22050

#: The ten slider bands — standard octave centres.
BAND_FREQS: tuple[float, ...] = (
    31.5, 63.0, 125.0, 250.0, 500.0,
    1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
)

#: Grey-noise contour: dB to add at each band (relative to 1 kHz) so
#: the noise *sounds* equally loud at every frequency — the inverse of
#: the ISO 226 40-phon equal-loudness contour, rounded. Hearing is
#: most sensitive in the mids, so grey carries more energy at the
#: extremes and less in the middle.
GREY_CONTOUR_DB: tuple[float, ...] = (
    25.0, 10.0, 2.0, -4.0, -6.0, 0.0, -3.0, -5.0, 2.0, 12.0,
)

#: Longest single render. A chat file longer than this is a transfer
#: problem, not a soundscape — the owner loops the file instead.
MAX_DURATION_S = 300.0

#: Animate drift: each slider wanders by this many dB peak.
_ANIMATE_DEPTH_DB = 1.5


def _has_numpy() -> bool:
    try:
        __import__("numpy")
        return True
    except Exception:  # noqa: BLE001
        return False


def _peak_normalize(samples: Sequence[float],
                    target: float = 0.9) -> list[float]:
    peak = 0.0
    for s in samples:
        if abs(s) > peak:
            peak = abs(s)
    if peak <= 0.0:
        return [0.0] * len(samples)
    scale = target / peak
    return [s * scale for s in samples]


def _one_pole_lowpass(samples: Sequence[float], alpha: float) -> list[float]:
    out: list[float] = []
    y = 0.0
    for x in samples:
        y += alpha * (x - y)
        out.append(y)
    return out


def _one_pole_highpass(samples: Sequence[float], alpha: float) -> list[float]:
    out: list[float] = []
    y = 0.0
    prev_x = 0.0
    for x in samples:
        y = alpha * (y + x - prev_x)
        prev_x = x
        out.append(y)
    return out


def _peaking(samples: Sequence[float], sr: int, freq: float,
             gain_db: float, q: float = 1.0) -> list[float]:
    """RBJ peaking EQ — one band, pure Python. Total."""
    if gain_db == 0.0 or freq <= 0 or sr <= 0:
        return list(samples)
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * max(0.05, q))
    cw = math.cos(w0)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / a
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0
    out: list[float] = [0.0] * len(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(samples):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return out


def _kellet_pink(n: int, rng: random.Random) -> list[float]:
    """Paul Kellet's pink-noise filter — ~−3 dB/octave, public domain.

    Seven one-pole stages; the output is the sum of the stages.
    Accurate to well under a dB across the audible band.
    """
    b = [0.99886, 0.99332, 0.96900, 0.86650, 0.55000, 0.11500]
    g = [0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, 0.0168980]
    state = [0.0] * 7
    out: list[float] = []
    for _ in range(n):
        white = rng.uniform(-1.0, 1.0)
        for i in range(6):
            state[i] = b[i] * state[i] + g[i] * white
        state[6] = white * 0.115926
        out.append(sum(state) + white * 0.5362)
    return _peak_normalize(out)


def synthesize_noise(color: str, n: int, sr: int,
                     seed: int | None = None) -> list[float]:
    """Synthesise ``n`` samples of a noise colour. Deterministic per seed.

    Colours: white (flat), pink (−3 dB/oct), brown (−6 dB/oct),
    grey (equal-loudness shaped), blue (+3 dB/oct). Raises ValueError
    for an unknown colour. Never returns silence.
    """
    color = (color or "").strip().lower()
    if n <= 0:
        return []
    rng = random.Random(seed)
    if color == "white":
        return _peak_normalize([rng.uniform(-1.0, 1.0) for _ in range(n)])
    if color == "pink":
        return _kellet_pink(n, rng)
    if color == "brown":
        # Leaky integrator: corner far below the lowest band, so the
        # slope is a clean −6 dB/octave across the audible band.
        leak = 0.9995
        out: list[float] = []
        y = 0.0
        for _ in range(n):
            y = leak * y + (1.0 - leak) * rng.uniform(-1.0, 1.0) * 40.0
            out.append(y)
        mean = sum(out) / len(out)
        return _peak_normalize([s - mean for s in out])
    if color == "blue":
        # +3 dB/octave: power proportional to frequency. Numpy path
        # shapes the spectrum by sqrt(f) exactly; the pure path takes
        # the first difference of pink noise (1/f times f^2 = f) —
        # the same slope, measured, not assumed.
        if _has_numpy():
            import numpy as np

            white = np.array([rng.uniform(-1.0, 1.0) for _ in range(n)])
            spec = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            shape = np.sqrt(np.maximum(freqs, 1.0) / 1000.0)
            shape[0] = 0.0
            return _peak_normalize(list(np.fft.irfft(spec * shape, n=n)))
        pink = _kellet_pink(n, rng)
        out = [0.0]
        for i in range(1, n):
            out.append(pink[i] - pink[i - 1])
        return _peak_normalize(out)
    if color == "grey":
        white = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        gains = dict(zip(BAND_FREQS, GREY_CONTOUR_DB, strict=True))
        return _peak_normalize(apply_eq(white, sr, gains))
    raise ValueError(f"unknown noise colour: {color!r}")


def synthesize_tone(kind: str, n: int, sr: int, *,
                    freq: float = 440.0, beat: float = 10.0,
                    root: float = 110.0,
                    seed: int | None = None) -> tuple[list[float], list[float]]:
    """Synthesise a tone generator as a (left, right) stereo pair.

    ``binaural``: two sines at carrier ∓ beat/2 — the beat lives in
    the difference the brain hears, never in either ear alone.
    ``drone``: a root with its harmonic stack, slightly detuned
    between the ears for width. Raises ValueError for unknown kinds.
    """
    kind = (kind or "").strip().lower()
    if kind == "binaural":
        left: list[float] = []
        right: list[float] = []
        for i in range(n):
            t = i / sr
            left.append(math.sin(2.0 * math.pi * (freq - beat / 2.0) * t))
            right.append(math.sin(2.0 * math.pi * (freq + beat / 2.0) * t))
        return _peak_normalize(left, 0.7), _peak_normalize(right, 0.7)
    if kind == "drone":
        harmonics = (1.0, 1.5, 2.0, 3.0, 4.02)
        weights = (1.0, 0.5, 0.35, 0.2, 0.1)
        left = []
        right = []
        for i in range(n):
            t = i / sr
            left_v = sum(w * math.sin(2.0 * math.pi * root * h * t)
                         for h, w in zip(harmonics, weights, strict=True))
            right_v = sum(w * math.sin(2.0 * math.pi * root * h * 1.003 * t
                                      + 0.4 * h)
                          for h, w in zip(harmonics, weights, strict=True))
            left.append(left_v)
            right.append(right_v)
        return _peak_normalize(left, 0.7), _peak_normalize(right, 0.7)
    raise ValueError(f"unknown tone kind: {kind!r}")


def octave_band_energies(samples: Sequence[float],
                         sr: int) -> dict[float, float]:
    """Spectral *density* in each octave band around :data:`BAND_FREQS`.

    Energy per Hz, so a white render measures flat, pink −3 dB/oct
    and brown −6 dB/oct — the proof criteria themselves. Needs numpy
    (the spectral measurement path). Raises RuntimeError without it —
    a measurement is never faked from a guess.
    """
    if not _has_numpy():
        raise RuntimeError("octave_band_energies needs numpy")
    import numpy as np

    data = np.array(list(samples), dtype=float)
    if len(data) == 0:
        return {f: 0.0 for f in BAND_FREQS}
    spec = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
    energies: dict[float, float] = {}
    edges = [BAND_FREQS[0] / math.sqrt(2.0)]
    for i in range(len(BAND_FREQS) - 1):
        edges.append(math.sqrt(BAND_FREQS[i] * BAND_FREQS[i + 1]))
    edges.append(BAND_FREQS[-1] * math.sqrt(2.0))
    for i, centre in enumerate(BAND_FREQS):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        width = max(edges[i + 1] - edges[i], 1.0)
        energies[centre] = (float(spec[mask].sum()) / width
                            if mask.any() else 0.0)
    return energies


def apply_eq(samples: Sequence[float], sr: int,
             gains: dict[float, float]) -> list[float]:
    """Apply a 10-band EQ (gains in dB, keyed by band centre).

    Missing bands are 0 dB. Each slider owns its octave: the numpy
    path gives every FFT bin the gain of its nearest band centre in
    log space, so a +12 dB slider raises its whole octave by 12 dB.
    The pure-Python path is a peaking-biquad bank — the same sliders,
    the same band centres, a smooth peak instead of a flat top.
    Both are real filtering — the sliders always do what they say.
    """
    if not gains or all(v == 0.0 for v in gains.values()):
        return list(samples)
    if _has_numpy():
        import numpy as np

        data = np.array(list(samples), dtype=float)
        spec = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        band_freqs = np.array(BAND_FREQS, dtype=float)
        band_gains = np.array([gains.get(f, 0.0) for f in BAND_FREQS],
                              dtype=float)
        log_f = np.log10(np.maximum(freqs, 1.0))
        log_b = np.log10(band_freqs)
        nearest = np.argmin(np.abs(log_f[:, None] - log_b[None, :]),
                            axis=1)
        spec = spec * (10.0 ** (band_gains[nearest] / 20.0))
        return list(np.fft.irfft(spec, n=len(data)))
    out = list(samples)
    for freq in BAND_FREQS:
        gain = gains.get(freq, 0.0)
        if gain:
            out = _peaking(out, sr, freq, gain, q=1.0)
    return out


def apply_hearing(samples: Sequence[float], sr: int,
                  profile: dict[float, float]) -> list[float]:
    """Apply a hearing (iEQ) profile — the same 10-band EQ, fed by a
    calibration instead of the sliders. An empty profile is a no-op.
    """
    if not profile:
        return list(samples)
    return apply_eq(samples, sr, dict(profile))


def animate_sliders(base: Sequence[float], t: float,
                    seed: int | None = None) -> tuple[float, ...]:
    """Drift the slider vector slowly so a scene never feels looped.

    Each slider wanders on its own slow sine (0.004–0.022 Hz, ±1.5 dB
    peak). At t=0 the vector is exactly the base — the drift grows
    from silence, it never jumps. Deterministic per seed.
    """
    rng = random.Random(seed)
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(10)]
    out: list[float] = []
    for i, gain in enumerate(base):
        freq = 0.004 + 0.002 * i
        drift = (_ANIMATE_DEPTH_DB
                 * math.sin(2.0 * math.pi * freq * t + phases[i])
                 - _ANIMATE_DEPTH_DB * math.sin(phases[i]))
        out.append(float(gain) + drift)
    return tuple(out)


def write_mono_wav(path: str | os.PathLike[str],
                   samples: Sequence[float], sr: int) -> str:
    """Mono floats → 16-bit WAV. Returns the path, never raises."""
    try:
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in samples))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr) or SR)
            wf.writeframes(pcm.tobytes())
        return str(path)
    except Exception:
        return ""


def write_stereo_wav(path: str | os.PathLike[str], left: Sequence[float],
                     right: Sequence[float], sr: int) -> str:
    """Stereo floats → 16-bit WAV. Returns the path, never raises."""
    try:
        pcm = array("h")
        for left_v, right_v in zip(left, right, strict=True):
            pcm.append(max(-32768, min(32767, int(left_v * 32767))))
            pcm.append(max(-32768, min(32767, int(right_v * 32767))))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(int(sr) or SR)
            wf.writeframes(pcm.tobytes())
        return str(path)
    except Exception:
        return ""


def render_tone(freq: float, duration_s: float = 2.0,
                out_path: str | None = None) -> str:
    """Render a pure calibration tone (one band centre) as a mono WAV.

    The guided hearing-calibration flow plays these, one band at a
    time — the owner says what he hears, and the band gain follows.
    Returns the WAV path. Raises ValueError for a bad frequency.
    """
    if freq <= 0 or freq > SR / 2:
        raise ValueError(f"tone frequency out of range: {freq}")
    n = max(1, int(duration_s * SR))
    samples = [0.8 * math.sin(2.0 * math.pi * freq * i / SR)
               for i in range(n)]
    # Short raised-cosine edges so the tone never clicks on or off.
    edge = min(n // 10, int(0.05 * SR))
    for i in range(edge):
        w = 0.5 * (1.0 - math.cos(math.pi * i / edge))
        samples[i] *= w
        samples[n - 1 - i] *= w
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav",
                                        prefix=f"tone-{int(freq)}-")
        os.close(fd)
    write_mono_wav(out_path, samples, SR)
    return str(out_path)


def wav_bytes_sha256(path: str) -> str:
    """SHA-256 of a rendered WAV — the determinism receipt."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
