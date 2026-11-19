"""SoundCraft effects — the insert chain (lane 107).

Real DSP, measured against known signals:

* :class:`ParametricEQ` — RBJ Audio EQ Cookbook peaking / shelving
  biquads, the same cookbook upstream's ``crates/dsp`` eq.rs uses.
  Devon's :mod:`nomorals.audio.dsp` biquad has no peaking or shelf
  kinds, so the coefficients live here; the filter state is kept
  across blocks, exactly like a real insert.
* :class:`Compressor` — wraps Devon's own
  :func:`nomorals.audio.dsp.compressor` (soft-knee feed-forward)
  and adds measured gain reduction.
* :class:`Limiter` — wraps Devon's
  :func:`nomorals.audio.dsp.soft_limiter` (tanh ceiling).
* :class:`GainPlugin` / :class:`TonePlugin` — the test plugins the
  quarantined hosts load; a real plugin is any object with
  ``process(samples, sample_rate)`` and ``info()``.
"""

from __future__ import annotations

import math

import numpy as np

from ...audio import dsp as _dsp

__all__ = [
    "Compressor",
    "GainPlugin",
    "Limiter",
    "ParametricEQ",
    "TonePlugin",
]


class ParametricEQ:
    """RBJ peaking / low-shelf / high-shelf biquad insert."""

    def __init__(self, sample_rate: int, freq: float, gain_db: float,
                 q: float = 1.0, kind: str = "peak"):
        if kind not in ("peak", "lowshelf", "highshelf"):
            raise ValueError(f"unknown EQ kind {kind!r}")
        if freq <= 0 or sample_rate <= 0:
            raise ValueError("freq and sample_rate must be positive")
        self.sample_rate = int(sample_rate)
        self.freq = float(freq)
        self.gain_db = float(gain_db)
        self.q = float(max(0.05, q))
        self.kind = kind
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0
        self._coefficients()

    def _coefficients(self) -> None:
        a = 10.0 ** (self.gain_db / 40.0)
        w0 = 2.0 * math.pi * self.freq / self.sample_rate
        alpha = math.sin(w0) / (2.0 * self.q)
        cw = math.cos(w0)
        if self.kind == "peak":
            b0 = 1.0 + alpha * a
            b1 = -2.0 * cw
            b2 = 1.0 - alpha * a
            a0 = 1.0 + alpha / a
            a1 = -2.0 * cw
            a2 = 1.0 - alpha / a
        elif self.kind == "lowshelf":
            sq = 2.0 * math.sqrt(a) * alpha
            b0 = a * ((a + 1.0) - (a - 1.0) * cw + sq)
            b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw)
            b2 = a * ((a + 1.0) - (a - 1.0) * cw - sq)
            a0 = (a + 1.0) + (a - 1.0) * cw + sq
            a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw)
            a2 = (a + 1.0) + (a - 1.0) * cw - sq
        else:  # highshelf
            sq = 2.0 * math.sqrt(a) * alpha
            b0 = a * ((a + 1.0) + (a - 1.0) * cw + sq)
            b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw)
            b2 = a * ((a + 1.0) + (a - 1.0) * cw - sq)
            a0 = (a + 1.0) - (a - 1.0) * cw + sq
            a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw)
            a2 = (a + 1.0) - (a - 1.0) * cw - sq
        self._b = (b0 / a0, b1 / a0, b2 / a0)
        self._a = (a1 / a0, a2 / a0)

    def info(self) -> dict[str, object]:
        return {"id": "parametric-eq", "name": "Parametric EQ",
                "params": {"freq": self.freq, "gain_db": self.gain_db,
                           "q": self.q, "kind": self.kind}}

    def process(self, samples: np.ndarray,
                sample_rate: int | None = None) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float64)
        out = np.empty_like(x)
        b0, b1, b2 = self._b
        a1, a2 = self._a
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        for i, x0 in enumerate(x):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
        self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2
        return out


class Compressor:
    """Soft-knee feed-forward compressor over Devon's DSP toolbox."""

    def __init__(self, sample_rate: int, threshold_db: float = -18.0,
                 ratio: float = 4.0, attack_s: float = 0.005,
                 release_s: float = 0.1):
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.ratio = float(max(1.0, ratio))
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)

    def info(self) -> dict[str, object]:
        return {"id": "compressor", "name": "Compressor/Limiter",
                "params": {"threshold_db": self.threshold_db,
                           "ratio": self.ratio,
                           "attack_s": self.attack_s,
                           "release_s": self.release_s}}

    def process(self, samples: np.ndarray,
                sample_rate: int | None = None) -> np.ndarray:
        out = _dsp.compressor(
            np.asarray(samples, dtype=np.float64),
            int(sample_rate or self.sample_rate),
            threshold_db=self.threshold_db, ratio=self.ratio,
            attack_s=self.attack_s, release_s=self.release_s)
        return np.asarray(out, dtype=np.float64)

    def gain_reduction_db(self, samples: np.ndarray) -> float:
        """Measured gain reduction: peak in vs peak out, in dB.

        Measured on the settled second half — a feed-forward
        compressor lets the initial attack transient through, and
        that transient is not gain reduction.
        """
        x = np.asarray(samples, dtype=np.float64)
        if len(x) < 2:
            return 0.0
        half = x[len(x) // 2:]
        peak_in = float(np.max(np.abs(half)))
        if peak_in <= 0.0:
            return 0.0
        out = self.process(x, self.sample_rate)
        out_half = out[len(out) // 2:]
        peak_out = float(np.max(np.abs(out_half))) if len(out_half) else 0.0
        if peak_out <= 0.0 or peak_out >= peak_in:
            return 0.0
        return 20.0 * math.log10(peak_in / peak_out)


class Limiter:
    """Tanh soft-clip ceiling over Devon's DSP toolbox."""

    def __init__(self, ceiling: float = 0.95):
        self.ceiling = float(ceiling)

    def info(self) -> dict[str, object]:
        return {"id": "limiter", "name": "Limiter",
                "params": {"ceiling": self.ceiling}}

    def process(self, samples: np.ndarray,
                sample_rate: int | None = None) -> np.ndarray:
        out = _dsp.soft_limiter(np.asarray(samples, dtype=np.float64),
                                ceiling=self.ceiling)
        return np.asarray(out, dtype=np.float64)


class GainPlugin:
    """Test plugin: a plain gain stage, the hosts' smoke test."""

    def __init__(self, gain_db: float = 0.0, **kwargs: object):
        if "gain_db" in kwargs:
            gain_db = float(kwargs["gain_db"])  # type: ignore[arg-type]
        self.gain_db = float(gain_db)

    def info(self) -> dict[str, object]:
        return {"id": "gain", "name": "Gain",
                "params": {"gain_db": self.gain_db}}

    def process(self, samples: np.ndarray,
                sample_rate: int | None = None) -> np.ndarray:
        return np.asarray(samples, dtype=np.float64) * (
            10.0 ** (self.gain_db / 20.0))


class TonePlugin:
    """Test plugin: a sine oscillator that replaces its input."""

    def __init__(self, freq: float = 440.0, amp: float = 0.5, **kwargs: object):
        if "freq" in kwargs:
            freq = float(kwargs["freq"])  # type: ignore[arg-type]
        if "amp" in kwargs:
            amp = float(kwargs["amp"])  # type: ignore[arg-type]
        self.freq = float(freq)
        self.amp = float(amp)
        self._phase = 0.0

    def info(self) -> dict[str, object]:
        return {"id": "tone", "name": "Tone Oscillator",
                "params": {"freq": self.freq, "amp": self.amp}}

    def process(self, samples: np.ndarray,
                sample_rate: int | None = None) -> np.ndarray:
        n = len(samples)
        sr = int(sample_rate or 48000)
        phase = self._phase + 2.0 * math.pi * self.freq * np.arange(n) / sr
        self._phase = float((self._phase + 2.0 * math.pi * self.freq * n
                             / sr) % (2.0 * math.pi))
        return self.amp * np.sin(phase)
