"""Devon's own DSP toolbox — audio effects and enhancement, natively.

Every effect here is implemented by Devon, in this file: no ffmpeg, no
plugins, no network. The toolbox powers:

* :func:`enhance` — the native enhancement chain behind
  ``nomorals.audio.edit.enhance_audio``'s ``devon`` engine (denoise →
  de-hum → trim → normalize → limit), for voice notes and speech.
* :class:`EffectChain` — Devon-owned effect chains (reverb, echo, EQ,
  compressor, pitch shift) wired into ``/audio fx``.

Math: biquad IIR (notch), spectral gating (FFT), generated impulse
responses (reverb convolution), delay lines (echo), soft-knee dynamics.
FFT comes from :mod:`nomorals.audio.fingerprint` (numpy when present,
pure-Python Cooley-Tukey otherwise).

All functions are total: they take and return ``array('d')`` mono
float buffers and never raise — a failed effect returns the input
unchanged (logged), never fake audio.
"""

from __future__ import annotations

import logging
import math
import os
import wave
from array import array
from pathlib import Path
from typing import Any, Sequence

from ..core.logging_setup import get_logger
from .fingerprint import has_numpy

_log = logging.getLogger(__name__)

__all__ = [
    "read_mono_wav",
    "write_mono_wav",
    "normalize_peak",
    "remove_dc",
    "trim_silence",
    "fade_edges",
    "notch",
    "dehum",
    "spectral_gate",
    "compressor",
    "soft_limiter",
    "reverb",
    "echo",
    "eq_3band",
    "pitch_shift",
    "pitch_shift_to",
    "time_stretch",
    "resample",
    "mix_under",
    "EffectChain",
    "EFFECTS",
    "enhance",
    "ENHANCE_PROFILES",
]

_SAMPLE_EPS = 1e-9


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_mono_wav(path: str | os.PathLike[str]) -> tuple[array, int]:
    """WAV → (mono float samples, sample rate). Total."""
    from .fingerprint import _read_wav

    try:
        return _read_wav(str(path))
    except Exception as exc:  # noqa: BLE001
        _log.debug("read_mono_wav failed for %s: %s", path, exc)
        return array("d"), 0


def write_mono_wav(path: str | os.PathLike[str], samples: Sequence[float],
                   sr: int) -> str:
    """Mono floats → 16-bit WAV. Returns path or "" (never raises)."""
    try:
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in samples))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr) or 22050)
            wf.writeframes(pcm.tobytes())
        return str(path)
    except Exception as exc:  # noqa: BLE001
        _log.debug("write_mono_wav failed: %s", exc)
        return ""


def resample(samples: Sequence[float], src_sr: int,
             dst_sr: int) -> array:
    """Linear resample. Total."""
    from .fingerprint import _resample_linear

    return _resample_linear(array("d", samples), src_sr, dst_sr)


# ---------------------------------------------------------------------------
# cleanup primitives
# ---------------------------------------------------------------------------

def normalize_peak(samples: Sequence[float], target: float = 0.89) -> array:
    """Peak-normalize to ``target``. Total."""
    try:
        peak = max((abs(s) for s in samples), default=0.0)
        if peak <= _SAMPLE_EPS:
            return array("d", samples)
        g = target / peak
        return array("d", (s * g for s in samples))
    except Exception:  # noqa: BLE001
        return array("d", samples)


def remove_dc(samples: Sequence[float]) -> array:
    """Subtract the DC offset. Total."""
    try:
        n = len(samples)
        if not n:
            return array("d")
        mean = sum(samples) / n
        return array("d", (s - mean for s in samples))
    except Exception:  # noqa: BLE001
        return array("d", samples)


def trim_silence(samples: Sequence[float], sr: int,
                 threshold_db: float = -45.0,
                 pad_s: float = 0.15) -> array:
    """Cut leading/trailing near-silence. Total."""
    try:
        thr = 10.0 ** (threshold_db / 20.0)
        n = len(samples)
        start, end = 0, n
        while start < n and abs(samples[start]) < thr:
            start += 1
        while end > start and abs(samples[end - 1]) < thr:
            end -= 1
        pad = int(pad_s * sr)
        start = max(0, start - pad)
        end = min(n, end + pad)
        return array("d", samples[start:end]) if end > start else \
            array("d", samples)
    except Exception:  # noqa: BLE001
        return array("d", samples)


def fade_edges(samples: Sequence[float], sr: int,
               fade_s: float = 0.015) -> array:
    """Short raised-cosine fades at both ends (de-click). Total."""
    try:
        n = len(samples)
        m = max(1, min(n // 2, int(fade_s * sr)))
        out = array("d", samples)
        for i in range(m):
            f = 0.5 - 0.5 * math.cos(math.pi * i / m)
            out[i] *= f
            out[n - 1 - i] *= f
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


# ---------------------------------------------------------------------------
# biquad notch — mains hum removal (50/60 Hz + harmonics)
# ---------------------------------------------------------------------------

def notch(samples: Sequence[float], sr: int, freq: float,
          q: float = 30.0) -> array:
    """Biquad notch at ``freq`` Hz. Total."""
    try:
        n = len(samples)
        if n == 0 or freq <= 0 or sr <= 0:
            return array("d", samples)
        w0 = 2.0 * math.pi * freq / sr
        alpha = math.sin(w0) / (2.0 * q)
        b0, b1, b2 = 1.0, -2.0 * math.cos(w0), 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * math.cos(w0), 1.0 - alpha
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        out = array("d", [0.0]) * n
        x1 = x2 = y1 = y2 = 0.0
        for i, x0 in enumerate(samples):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def dehum(samples: Sequence[float], sr: int,
          freqs: Sequence[float] = (50.0, 60.0)) -> array:
    """Notch the mains hum + its first two harmonics. Total."""
    try:
        out = array("d", samples)
        for f in freqs:
            for mult in (1, 2, 3):
                if f * mult < sr / 2 - 50:
                    out = notch(out, sr, f * mult)
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


# ---------------------------------------------------------------------------
# spectral gating — the native denoiser
# ---------------------------------------------------------------------------

def _noise_profile(frames: list[list[float]]) -> list[float]:
    """Median magnitude per bin over the quietest 10% of frames.

    Minimum-statistics: the noise floor is what the quietest moments
    look like, not the first moments (speech often starts immediately).
    """
    n = len(frames)
    if not n:
        return []
    energies = sorted((sum(m * m for m in fr), i)
                      for i, fr in enumerate(frames))
    k = max(1, n // 10)
    idx = [i for _, i in energies[:k]]
    nbins = len(frames[0])
    prof = [0.0] * nbins
    for b in range(nbins):
        col = sorted(frames[i][b] for i in idx)
        prof[b] = col[len(col) // 2] if col else 0.0
    return prof


def spectral_gate(samples: Sequence[float], sr: int, *,
                  reduction_db: float = 12.0,
                  win: int = 2048, hop: int = 1024) -> array:
    """Spectral-subtraction denoiser: bins near the noise floor get
    attenuated by ``reduction_db``. Total (numpy or pure-Python FFT via
    the fingerprint module)."""
    try:
        from .fingerprint import _fft_pure

        n = len(samples)
        if n < win:
            return array("d", samples)
        red = 10.0 ** (-reduction_db / 20.0)

        def _analyze(frame: Sequence[float]) -> list[float]:
            hann = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (win - 1))
                    for i in range(win)]
            w = [frame[i] * hann[i] for i in range(win)]
            if has_numpy():
                import numpy as np

                return [complex(v) for v in
                        np.fft.rfft(np.asarray(w, dtype=np.float64))]
            full = _fft_pure([complex(v, 0.0) for v in w])
            return full[:win // 2 + 1]

        def _synth(spec: list[complex]) -> list[float]:
            if has_numpy():
                import numpy as np

                return [float(v) for v in
                        np.fft.irfft(np.asarray(spec, dtype=np.complex128))]
            # pure-python inverse via conjugate trick
            conj = [c.conjugate() for c in spec]
            # rebuild full spectrum (mirror, dropping DC/Nyquist dupes)
            full = conj + [c.conjugate() for c in
                           reversed(conj[1:-1])] if len(conj) > 2 else conj
            inv = _fft_pure(full)
            return [v.real / len(full) for v in inv[:win]]

        frames_c: list[list[complex]] = []
        starts: list[int] = []
        for start in range(0, n - win + 1, hop):
            frames_c.append(_analyze(samples[start:start + win]))
            starts.append(start)
        mags = [[abs(c) for c in fr] for fr in frames_c]
        prof = _noise_profile(mags)
        out = array("d", [0.0]) * n
        norm = array("d", [0.0]) * n
        hann = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (win - 1))
                for i in range(win)]
        for fr, mag, start in zip(frames_c, mags, starts):
            gated = [c * (red if m < prof[b] * 1.8 else 1.0)
                     for b, (c, m) in enumerate(zip(fr, mag))]
            y = _synth(gated)
            for i in range(win):
                out[start + i] += y[i] * hann[i]
                norm[start + i] += hann[i] * hann[i]
        # normalize only where the window overlap is meaningful —
        # the tail edge has ~zero overlap and must not be divided
        for i in range(n):
            if norm[i] > 1e-3:
                out[i] /= norm[i]
            else:
                out[i] = 0.0
        return out
    except Exception:  # noqa: BLE001
        _log.debug("spectral_gate failed", exc_info=True)
        return array("d", samples)


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def compressor(samples: Sequence[float], sr: int, *,
               threshold_db: float = -18.0,
               ratio: float = 3.0,
               attack_s: float = 0.005,
               release_s: float = 0.12) -> array:
    """Soft-knee feed-forward compressor. Total."""
    try:
        n = len(samples)
        if not n:
            return array("d")
        thr = 10.0 ** (threshold_db / 20.0)
        a_att = math.exp(-1.0 / max(1, attack_s * sr))
        a_rel = math.exp(-1.0 / max(1, release_s * sr))
        out = array("d", [0.0]) * n
        env = 0.0
        for i, s in enumerate(samples):
            a = abs(s)
            env = a_att * env + (1.0 - a_att) * a if a > env \
                else a_rel * env + (1.0 - a_rel) * a
            if env > thr:
                over = env / thr
                gain = (thr * over ** (1.0 / ratio)) / env
            else:
                gain = 1.0
            out[i] = s * gain
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def soft_limiter(samples: Sequence[float], ceiling: float = 0.95) -> array:
    """tanh soft-clip ceiling. Total."""
    try:
        out = array("d", [0.0]) * len(samples)
        for i, s in enumerate(samples):
            v = s / max(_SAMPLE_EPS, ceiling)
            out[i] = math.tanh(v) * ceiling
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


# ---------------------------------------------------------------------------
# space + tone — reverb, echo, EQ
# ---------------------------------------------------------------------------

def _convolve_fft(x: Sequence[float], h: Sequence[float]) -> array:
    """FFT convolution (numpy) with an O(n*m) pure-Python fallback for
    short kernels. Total."""
    try:
        import numpy as np

        n = len(x) + len(h) - 1
        size = 1
        while size < n:
            size <<= 1
        X = np.fft.rfft(np.asarray(list(x) + [0.0] * (size - len(x)),
                                   dtype=np.float64))
        H = np.fft.rfft(np.asarray(list(h) + [0.0] * (size - len(h)),
                                   dtype=np.float64))
        y = np.fft.irfft(X * H, n=size)
        return array("d", (float(v) for v in y[:n]))
    except Exception:  # noqa: BLE001
        # short-kernel direct convolution (used by the Schroeder path
        # only for tiny kernels — never the full reverb tail)
        out = array("d", [0.0]) * (len(x) + len(h) - 1)
        for i, xv in enumerate(x):
            if xv == 0.0:
                continue
            for j, hv in enumerate(h):
                out[i + j] += xv * hv
        return out


def _schroeder(x: Sequence[float], sr: int, decay_s: float) -> array:
    """Schroeder reverb: 4 parallel combs → 2 series allpasses. O(n),
    the pure-Python fallback when numpy is unavailable. Total."""
    n = len(x)
    out = array("d", [0.0]) * n
    # comb delays (samples at 44.1k, scaled to sr)
    scale = sr / 44100.0
    comb_ds = [int(1557 * scale), int(1617 * scale),
               int(1491 * scale), int(1422 * scale)]
    decay = 10.0 ** (-3.0 * max(comb_ds) / (decay_s * sr + 1e-9))
    for d in comb_ds:
        g = decay ** (d / max(comb_ds))
        buf = array("d", [0.0]) * d
        bi = 0
        for i in range(n):
            v = x[i] + buf[bi] * g
            buf[bi] = v
            out[i] += v
            bi = (bi + 1) % d
    for i in range(n):
        out[i] /= 4.0
    # two allpasses
    for d_ms, g in ((5.0, 0.7), (1.7, 0.7)):
        d = max(2, int(d_ms / 1000.0 * sr))
        buf = array("d", [0.0]) * d
        bi = 0
        for i in range(n):
            buf_out = buf[bi]
            v = -g * out[i] + buf_out
            buf[bi] = out[i] + g * buf_out
            out[i] = v
            bi = (bi + 1) % d
    return out


def reverb(samples: Sequence[float], sr: int, *,
           decay_s: float = 1.2, wet: float = 0.25,
           pre_delay_ms: float = 12.0) -> array:
    """Generated-impulse convolution reverb (exponential noise tail).

    Deterministic (seeded) — same input, same output. FFT convolution
    when numpy is present; a Schroeder comb/allpass network otherwise
    (O(n), honest about the different algorithm in the docstring).
    Total.
    """
    try:
        import random

        n = len(samples)
        if not n:
            return array("d")
        if has_numpy():
            ir_n = max(16, int(decay_s * sr))
            pre = int(pre_delay_ms / 1000.0 * sr)
            rng = random.Random(0xD3707)
            ir = [0.0] * (ir_n + pre)
            for i in range(ir_n):
                t = i / ir_n
                ir[pre + i] = rng.gauss(0.0, 1.0) * math.exp(-4.0 * t)
            wet_buf = _convolve_fft(samples, ir)
            scale = 1.0 / max(_SAMPLE_EPS,
                              math.sqrt(sum(c * c for c in ir)) * 4.0)
            out = array("d", [0.0]) * n
            for i in range(n):
                out[i] = (samples[i] * (1.0 - wet)
                          + wet_buf[i] * scale * wet)
            return out
        wet_buf = _schroeder(samples, sr, decay_s)
        out = array("d", [0.0]) * n
        for i in range(n):
            out[i] = samples[i] * (1.0 - wet) + wet_buf[i] * wet * 0.5
        return out
    except Exception:  # noqa: BLE001
        _log.debug("reverb failed", exc_info=True)
        return array("d", samples)


def echo(samples: Sequence[float], sr: int, *,
         delay_ms: float = 320.0, decay: float = 0.35,
         repeats: int = 4) -> array:
    """Feedback-style multi-tap echo. Total."""
    try:
        n = len(samples)
        if not n:
            return array("d")
        d = max(1, int(delay_ms / 1000.0 * sr))
        out = array("d", samples)
        taps = min(repeats, 8)
        for r in range(1, taps + 1):
            g = decay ** r
            off = d * r
            for i in range(n - off):
                out[i + off] += samples[i] * g
        return normalize_peak(out, 0.95)
    except Exception:  # noqa: BLE001
        return array("d", samples)


def eq_3band(samples: Sequence[float], sr: int, *,
             low_db: float = 0.0, mid_db: float = 0.0,
             high_db: float = 0.0,
             low_xo: float = 250.0, high_xo: float = 4000.0) -> array:
    """3-band EQ via FFT spectral scaling (whole-buffer, zero-phase).

    Honest and simple: per-bin gain by band. Not a surgical EQ — the
    docstring says so. Total.
    """
    try:
        if not samples or (low_db == 0 and mid_db == 0 and high_db == 0):
            return array("d", samples)
        from .fingerprint import _fft_pure

        n = len(samples)
        size = 1
        while size < n:
            size <<= 1
        x = [complex(s, 0.0) for s in samples] + [0j] * (size - n)
        if has_numpy():
            import numpy as np

            spec = np.fft.rfft(np.asarray(
                [c.real for c in x], dtype=np.float64))
            spec = [complex(v) for v in spec]
            nb = len(spec)
            freqs = [i * sr / size for i in range(nb)]
        else:
            spec = _fft_pure(x)[:size // 2 + 1]
            nb = len(spec)
            freqs = [i * sr / size for i in range(nb)]
        gl = 10.0 ** (low_db / 20.0)
        gm = 10.0 ** (mid_db / 20.0)
        gh = 10.0 ** (high_db / 20.0)
        shaped = [c * (gl if f < low_xo else gh if f > high_xo else gm)
                  for c, f in zip(spec, freqs)]
        if has_numpy():
            import numpy as np

            y = np.fft.irfft(np.asarray(shaped, dtype=np.complex128),
                             n=size)
            return array("d", (float(v) for v in y[:n]))
        full = shaped + [c.conjugate() for c in reversed(shaped[1:-1])]
        inv = _fft_pure(full)
        return array("d", (v.real / len(full) for v in inv[:n]))
    except Exception:  # noqa: BLE001
        _log.debug("eq_3band failed", exc_info=True)
        return array("d", samples)


# ---------------------------------------------------------------------------
# pitch
# ---------------------------------------------------------------------------

def time_stretch(samples: Sequence[float], sr: int,
                 factor: float) -> array:
    """WSOLA granular time-stretch: ``factor`` > 1 lengthens, pitch kept.

    Waveform-Similarity Overlap-Add: each new grain is chosen by
    searching ±8 ms around its nominal position for the segment that
    best continues the previous grain's waveform (max normalized
    cross-correlation) — grains stay phase-coherent, so pitch is
    preserved while time stretches. The speech/vocal standard;
    crossfaded Hann grains. Total.
    """
    try:
        n = len(samples)
        if n < 64 or factor <= 0:
            return array("d", samples)
        if abs(factor - 1.0) < 1e-6:
            return array("d", samples)
        grain = max(64, int(0.04 * sr))
        ha = grain // 2
        hs = max(1, int(round(ha * factor)))
        ov = grain - hs
        if ov < 8:  # grains barely overlap — widen the match region
            ov = grain // 2
        tol = max(8, int(0.008 * sr))
        win = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (grain - 1))
               for i in range(grain)]
        x = [float(v) for v in samples]
        n_out = int(n * factor) + grain
        out = [0.0] * n_out
        norm = [0.0] * n_out

        def _place(a_pos: int, s_pos: int) -> None:
            for i in range(grain):
                if a_pos + i >= n or s_pos + i >= n_out:
                    break
                out[s_pos + i] += x[a_pos + i] * win[i]
                norm[s_pos + i] += win[i]

        def _best_match(ref: list, a_nom: int) -> int:
            lo = max(0, a_nom - tol)
            hi = min(n - grain, a_nom + tol)
            if hi <= lo:
                return max(0, min(n - grain, a_nom))
            m = len(ref)
            if has_numpy():
                import numpy as np

                seg = np.asarray(x[lo:hi + m], dtype=np.float64)
                r = np.asarray(ref, dtype=np.float64)
                corr = np.correlate(seg, r, mode="valid")
                r_e = float(np.dot(r, r)) + 1e-12
                sq = seg * seg
                cs = np.concatenate(([0.0], np.cumsum(sq)))
                e = cs[m:] - cs[:len(corr)]
                scores = corr / np.sqrt(e * r_e + 1e-12)
                return lo + int(np.argmax(scores))
            best_a, best_s = lo, -2.0
            r_e = sum(v * v for v in ref) + 1e-12
            for a in range(lo, hi + 1):
                num = sum(x[a + i] * ref[i] for i in range(m))
                den = math.sqrt(
                    sum(x[a + i] * x[a + i] for i in range(m)) * r_e)
                s_ = num / (den + 1e-12)
                if s_ > best_s:
                    best_s, best_a = s_, a
            return best_a

        # first grain anchors at 0
        _place(0, 0)
        a_prev, s_prev = 0, 0
        while True:
            s_new = s_prev + hs
            if s_new + grain > n_out:
                break
            a_nom = a_prev + ha
            if a_nom + grain > n:
                break
            ref = [x[a_prev + grain - ov + i] for i in range(ov)]
            a_new = _best_match(ref, a_nom)
            _place(a_new, s_new)
            a_prev, s_prev = a_new, s_new
        for i in range(n_out):
            if norm[i] > 1e-3:
                out[i] /= norm[i]
        target = int(round(n * factor))
        res = array("d", out[:target])
        if len(res) < target:
            res.extend([0.0] * (target - len(res)))
        return res
    except Exception:  # noqa: BLE001
        _log.debug("time_stretch failed", exc_info=True)
        return array("d", samples)


def _varispeed(samples: Sequence[float], factor: float) -> array:
    """Resample by ``factor`` — pitch AND speed move together (the
    classic tape varispeed). Total."""
    try:
        n = len(samples)
        if n < 2 or factor <= 0:
            return array("d", samples)
        m = max(2, int(n / factor))
        out = array("d", [0.0]) * m
        for i in range(m):
            pos = i * factor
            lo = int(pos)
            hi = min(n - 1, lo + 1)
            frac = pos - lo
            out[i] = samples[lo] * (1.0 - frac) + samples[hi] * frac
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def pitch_shift(samples: Sequence[float], sr: int,
                semitones: float) -> array:
    """Pitch-shift by ``semitones`` with duration preserved.

    Varispeed resample (the honest pitch change) followed by WSOLA
    granular time-stretch back to the original length. Formants move
    with the pitch — the classic shifter sound, documented as such.
    Total.
    """
    try:
        if not samples or semitones == 0:
            return array("d", samples)
        n = len(samples)
        factor = 2.0 ** (semitones / 12.0)
        tmp = _varispeed(samples, factor)
        out = time_stretch(tmp, sr, factor)
        # exact length: trim or zero-pad
        if len(out) > n:
            return out[:n]
        if len(out) < n:
            out.extend([0.0] * (n - len(out)))
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def pitch_shift_to(samples: Sequence[float], sr: int,
                   from_midi: float, to_midi: float) -> array:
    """Shift so a note at ``from_midi`` lands on ``to_midi``. Total."""
    return pitch_shift(samples, sr, to_midi - from_midi)


# ---------------------------------------------------------------------------
# mixing helper
# ---------------------------------------------------------------------------

def mix_under(bed: Sequence[float], top: Sequence[float],
              gain: float = 0.85) -> array:
    """Sum ``top`` (×gain) under ``bed``. Total."""
    try:
        n = max(len(bed), len(top))
        out = array("d", [0.0]) * n
        for i, s in enumerate(bed):
            out[i] += s
        for i, s in enumerate(top):
            out[i] += s * gain
        return out
    except Exception:  # noqa: BLE001
        return array("d", bed)


# ---------------------------------------------------------------------------
# effect chain — Devon-owned chains, named and inspectable
# ---------------------------------------------------------------------------

#: Every chainable effect: name → (function, description).
EFFECTS: dict[str, tuple[Any, str]] = {
    "denoise": (lambda s, sr, **k: spectral_gate(
        s, sr, reduction_db=float(k.get("db", 12.0))),
        "spectral-gate denoise (db=)"),
    "dehum": (lambda s, sr, **k: dehum(s, sr), "mains-hum notch"),
    "normalize": (lambda s, sr, **k: normalize_peak(
        s, float(k.get("target", 0.89))), "peak normalize (target=)"),
    "compress": (lambda s, sr, **k: compressor(s, sr), "soft-knee compressor"),
    "limit": (lambda s, sr, **k: soft_limiter(s), "tanh soft limiter"),
    "trim": (lambda s, sr, **k: trim_silence(s, sr), "trim edge silence"),
    "fade": (lambda s, sr, **k: fade_edges(s, sr), "de-click edge fades"),
    "reverb": (lambda s, sr, **k: reverb(
        s, sr, decay_s=float(k.get("decay", 1.2)),
        wet=float(k.get("wet", 0.25))), "generated-IR reverb (decay=, wet=)"),
    "echo": (lambda s, sr, **k: echo(
        s, sr, delay_ms=float(k.get("delay", 320.0)),
        decay=float(k.get("decay", 0.35))), "multi-tap echo (delay=, decay=)"),
    "eq": (lambda s, sr, **k: eq_3band(
        s, sr, low_db=float(k.get("low", 0.0)),
        mid_db=float(k.get("mid", 0.0)),
        high_db=float(k.get("high", 0.0))), "3-band EQ (low=, mid=, high= dB)"),
    "pitch": (lambda s, sr, **k: pitch_shift(
        s, sr, float(k.get("semitones", 0.0))),
        "resample pitch shift (semitones=)"),
}


class EffectChain:
    """An ordered, named, inspectable chain of Devon-owned effects.

    ``EffectChain([("denoise", {}), ("reverb", {"wet": 0.3})])``.
    Unknown effect names raise ValueError at construction (fail fast);
    ``run()`` never raises — a failing effect is skipped with its name
    recorded in ``self.skipped``.
    """

    def __init__(self, steps: Sequence[tuple[str, dict[str, Any]]]) -> None:
        self.steps: list[tuple[str, dict[str, Any]]] = []
        for name, params in steps:
            if name not in EFFECTS:
                raise ValueError(
                    f"unknown effect {name!r} — known: "
                    f"{', '.join(sorted(EFFECTS))}")
            self.steps.append((name, dict(params or {})))
        self.skipped: list[str] = []

    def describe(self) -> str:
        parts = []
        for name, params in self.steps:
            desc = EFFECTS[name][1]
            if params:
                desc += " " + " ".join(f"{k}={v}"
                                       for k, v in params.items())
            parts.append(f"{name}({desc})")
        return " → ".join(parts) if parts else "(empty chain)"

    def run(self, samples: Sequence[float], sr: int) -> array:
        """Apply the chain. Never raises."""
        out = array("d", samples)
        self.skipped = []
        for name, params in self.steps:
            try:
                fn = EFFECTS[name][0]
                out = fn(out, sr, **params)
            except Exception as exc:  # noqa: BLE001
                _log.debug("effect %s failed: %s", name, exc)
                self.skipped.append(name)
        return out

    @classmethod
    def parse(cls, text: str) -> "EffectChain":
        """Parse ``"reverb wet=0.3, eq low=3 high=-2, normalize"``.

        Never raises — unparsable steps are dropped with a note in
        ``skipped`` after construction... (construction raises on unknown
        names; use :meth:`parse_lenient` for chat input.)
        """
        steps: list[tuple[str, dict[str, Any]]] = []
        for chunk in (text or "").replace(";", ",").split(","):
            toks = chunk.strip().split()
            if not toks:
                continue
            name = toks[0].lower()
            params: dict[str, Any] = {}
            for tok in toks[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        params[k.strip()] = float(v)
                    except ValueError:
                        params[k.strip()] = v
            steps.append((name, params))
        return cls(steps)

    @classmethod
    def parse_lenient(cls, text: str) -> tuple["EffectChain", list[str]]:
        """Parse chat input; unknown effects are reported, not fatal."""
        steps: list[tuple[str, dict[str, Any]]] = []
        unknown: list[str] = []
        for chunk in (text or "").replace(";", ",").split(","):
            toks = chunk.strip().split()
            if not toks:
                continue
            name = toks[0].lower()
            if name not in EFFECTS:
                unknown.append(name)
                continue
            params: dict[str, Any] = {}
            for tok in toks[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        params[k.strip()] = float(v)
                    except ValueError:
                        params[k.strip()] = v
            steps.append((name, params))
        return cls(steps), unknown


# ---------------------------------------------------------------------------
# enhancement — the native "Adobe Enhance" Devon owns
# ---------------------------------------------------------------------------

#: Enhancement profiles: ordered effect steps per material.
ENHANCE_PROFILES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "voice": [("dehum", {}), ("denoise", {"db": 12.0}),
              ("trim", {}), ("normalize", {"target": 0.89}),
              ("compress", {}), ("limit", {}), ("fade", {})],
    "music": [("dehum", {}), ("denoise", {"db": 6.0}),
              ("normalize", {"target": 0.89}), ("limit", {}),
              ("fade", {})],
    "light": [("trim", {}), ("normalize", {"target": 0.89}),
              ("fade", {})],
}


def enhance(samples: Sequence[float], sr: int,
            profile: str = "voice") -> dict[str, Any]:
    """Run the native enhancement chain over mono samples.

    Returns ``{"ok", "samples", "chain", "skipped"}``. Never raises.
    """
    try:
        steps = ENHANCE_PROFILES.get((profile or "voice").lower(),
                                     ENHANCE_PROFILES["voice"])
        chain = EffectChain(steps)
        out = chain.run(samples, sr)
        return {"ok": True, "samples": out,
                "chain": chain.describe(), "skipped": chain.skipped}
    except Exception as exc:  # noqa: BLE001
        _log.debug("enhance failed", exc_info=True)
        return {"ok": False, "reason": f"enhance failed: {exc}",
                "samples": array("d", samples)}
