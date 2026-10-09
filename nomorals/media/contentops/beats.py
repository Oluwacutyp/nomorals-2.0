"""Beat detection for the content-edit engine.

Two paths, honest about which one ran:

1. **librosa** (preferred) — ``librosa.beat.beat_track`` + ``frames_to_time``.
   Used only when the ``librosa`` package actually imports; every call is
   guarded so an API drift or a runtime failure falls through to (2)
   instead of crashing the render.
2. **numpy fallback** — log-spectral-flux onset envelope → autocorrelation
   tempo estimate (60–200 BPM, octave disambiguation) → phase-aligned beat
   grid snapped to local onset peaks. Deterministic, no heavy deps.

``detect_beats()`` is the public entry point. It never raises for a missing
librosa; it raises :class:`MediaEditError` only for a missing/undecodable
file (same honesty contract as ``nomorals.media_edit.videos``).
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...media_edit.videos import MediaEditError, ffmpeg_path, run_ffmpeg
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["detect_beats", "detect_beats_full", "beats_available", "BeatInfo"]

_SR = 22050          # analysis sample rate (librosa's default)
_N_FFT = 1024
_HOP = 256
_MIN_BPM = 60.0
_MAX_BPM = 200.0


# ---------------------------------------------------------------------------
# audio decoding (no soundfile/scipy needed — ffmpeg + stdlib wave)
# ---------------------------------------------------------------------------

def _decode_mono(path: str | os.PathLike[str], sr: int = _SR) -> np.ndarray:
    """Any audio/video file → mono float32/float64 waveform at ``sr`` Hz."""
    p = Path(path)
    if not p.exists():
        raise MediaEditError(f"no such audio file: {path}")
    fd, tmp = tempfile.mkstemp(prefix="beats-", suffix=".wav")
    os.close(fd)
    try:
        run_ffmpeg(["-i", str(p), "-vn", "-ar", str(sr), "-ac", "1",
                    "-c:a", "pcm_s16le", tmp],
                   timeout=120.0)
        with wave.open(tmp, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise MediaEditError(f"unexpected decoded format for {path}")
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    except MediaEditError:
        raise
    except (OSError, wave.Error, ValueError) as exc:
        raise MediaEditError(f"could not decode audio from {path}: {exc}") from exc
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def beats_available() -> dict[str, Any]:
    """Which beat-detection backend is usable right now."""
    return {
        "librosa": importlib.util.find_spec("librosa") is not None,
        "numpy_fallback": True,
    }


# ---------------------------------------------------------------------------
# librosa path (verified against librosa 0.10 docs; fully guarded)
# ---------------------------------------------------------------------------

def _beats_librosa(y: np.ndarray, sr: int) -> list[float] | None:
    """librosa.beat.beat_track(y, sr) → tempo, beat_frames (units='frames'),
    then librosa.frames_to_time(beat_frames, sr=sr).

    Returns None on ANY failure (missing package, API drift, runtime
    error) so the caller can fall through to the numpy engine.
    """
    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        beat_track = getattr(getattr(librosa, "beat", None), "beat_track", None)
        frames_to_time = getattr(librosa, "frames_to_time", None)
        if beat_track is None or frames_to_time is None:
            _log.warning("librosa present but beat_track/frames_to_time "
                         "missing — using numpy fallback")
            return None
        tempo, beat_frames = beat_track(y=y, sr=sr)
        times = frames_to_time(np.asarray(beat_frames).ravel(), sr=sr)
        beats = sorted(float(t) for t in np.ravel(times) if t >= 0)
        _log.info("librosa beat_track: tempo=%.1f bpm, %d beats",
                  float(np.ravel(tempo)[0]) if np.size(tempo) else 0.0,
                  len(beats))
        return beats
    except Exception as exc:  # noqa: BLE001 — any librosa failure → fallback
        _log.warning("librosa beat_track failed (%s) — numpy fallback", exc)
        return None


# ---------------------------------------------------------------------------
# numpy fallback: onset envelope → tempo → phase-aligned grid
# ---------------------------------------------------------------------------

def _onset_envelope(y: np.ndarray, sr: int,
                    n_fft: int = _N_FFT, hop: int = _HOP) -> tuple[np.ndarray, np.ndarray]:
    """Log-spectral-flux onset envelope (centred frames, like librosa).

    Returns (times, envelope) with the envelope normalised to a max of 1.
    An energy gate kills noise-floor flux spikes in near-silent regions.
    """
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad))
    win = np.hanning(n_fft)
    n_frames = 1 + (len(yp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    X = np.abs(np.fft.rfft(yp[idx] * win, axis=1))
    logX = np.log1p(1000.0 * X)
    flux = np.diff(logX, axis=0)
    flux[flux < 0] = 0.0
    # Energy gate: a flux row is only meaningful if at least one of the
    # two frames it spans carries real energy (>=1% of the peak frame).
    frame_peak = X.max(axis=1)
    gate = np.maximum(frame_peak[:-1], frame_peak[1:]) > 0.01 * frame_peak.max()
    flux = flux * gate[:, None]
    env = np.concatenate([[flux[0].sum()], flux.sum(axis=1)])
    mx = env.max()
    if mx > 0:
        env = env / mx
    times = np.arange(len(env)) * hop / sr
    return times, env


def _estimate_period(env: np.ndarray, sr: int, hop: int,
                     min_bpm: float, max_bpm: float) -> float:
    """Autocorrelation tempo estimate → period in seconds (0.0 = none)."""
    e = env - env.mean()
    if not np.any(e):
        return 0.0
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(round(60.0 * sr / (max_bpm * hop)))
    hi = int(round(60.0 * sr / (min_bpm * hop)))
    if hi >= len(ac):
        hi = len(ac) - 1
    seg = ac[lo:hi + 1]
    if len(seg) < 3:
        return 0.0
    k = int(np.argmax(seg))
    if seg[k] < 0.25:  # no meaningful periodicity — don't hallucinate
        return 0.0

    def _is_peak(j: int) -> bool:
        return 0 < j < len(seg) - 1 and seg[j] >= seg[j - 1] and seg[j] >= seg[j + 1]

    # Octave disambiguation: prefer the faster tempo when a sub-harmonic
    # peak (period/2 or /3) is strong — autocorr loves the half-tempo lag.
    lag_k = lo + k
    best_lag = lag_k
    for lag_j in range(lo, lag_k):
        j = lag_j - lo
        if not _is_peak(j):
            continue
        ratio = lag_k / max(lag_j, 1)
        if abs(ratio - round(ratio)) < 0.08 and seg[j] > 0.75 * seg[k] \
                and lag_j < best_lag:
            best_lag = lag_j
    k = best_lag - lo
    if 0 < k < len(seg) - 1:  # parabolic sub-bin refinement
        a, b, c = seg[k - 1], seg[k], seg[k + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            k = k + float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
    return (lo + k) * hop / sr


def _onset_peaks(env: np.ndarray) -> np.ndarray:
    """Local-maximum onset peak frame indices above median+std."""
    thr = float(np.median(env) + env.std())
    peaks = [i for i in range(1, len(env) - 1)
             if env[i] > thr and env[i] >= env[i - 1] and env[i] > env[i + 1]]
    return np.asarray(peaks, dtype=np.int64)


def _score_grid(times: np.ndarray, env: np.ndarray, sr: int, hop: int,
                period: float, phase: float, dur: float,
                support_floor: float) -> tuple[tuple[int, float], np.ndarray, np.ndarray]:
    """(score, grid, supported-mask) for one (period, phase) hypothesis.

    Score = (number of grid points landing on real onset energy, then total
    onset energy) — the count dominates so grids don't stretch into silence.
    """
    grid = np.arange(phase, dur, period)
    t = phase
    while True:  # extend the grid backwards too
        t -= period
        if t < 0:
            break
        grid = np.concatenate([[t], grid])
    idx = np.clip((grid * sr / hop).astype(int), 0, len(env) - 1)
    supported = env[idx] > support_floor
    key = (int(supported.sum()), float(env[idx].sum()))
    return key, grid, supported


def _beats_numpy(y: np.ndarray, sr: int,
                 min_bpm: float = _MIN_BPM,
                 max_bpm: float = _MAX_BPM) -> tuple[float, list[float]]:
    """Full numpy beat tracker → (bpm, beats)."""
    hop = _HOP
    times, env = _onset_envelope(y, sr)
    dur = len(y) / sr
    if dur <= 0 or env.max() <= 0:
        return 0.0, []
    p0 = _estimate_period(env, sr, hop, min_bpm, max_bpm)
    if p0 <= 0:
        return 0.0, []
    peaks = _onset_peaks(env)
    thr = float(np.median(env) + env.std())
    support_floor = max(thr * 0.75, 0.15)

    # Refine the period: the autocorr estimate can be ~1% off, which drifts
    # audibly over a track — grid-search ±3% and keep the best-scoring grid.
    best: tuple[tuple[int, float], float, float] | None = None
    for period in np.linspace(0.97 * p0, 1.03 * p0, 21):
        for phase in np.linspace(0.0, period, 60, endpoint=False):
            key, _grid, _sup = _score_grid(times, env, sr, hop, float(period),
                                           float(phase), dur, support_floor)
            if best is None or key > best[0]:
                best = (key, float(period), float(phase))
    assert best is not None
    _, period, phase = best
    _, grid, supported = _score_grid(times, env, sr, hop, period, phase,
                                     dur, support_floor)
    grid = grid[supported]
    bpm = 60.0 / period

    # Snap each grid point to the nearest local onset peak (±40% of period),
    # then drop leading/trailing points with no onset support at all
    # (grid hallucinations past the first/last real hit).
    snapped: list[float] = []
    for b in grid:
        if len(peaks):
            d = np.abs(peaks * hop / sr - b)
            cand = peaks[d <= 0.4 * period]
            if len(cand):
                snapped.append(float(cand[np.argmax(env[cand])] * hop / sr))
                continue
        snapped.append(float(b))
    kept = [s for s in sorted(snapped)
            if (not len(peaks)
                or np.any(np.abs(peaks * hop / sr - s) <= 0.4 * period))]
    beats: list[float] = []
    for s in kept:  # dedup
        if not beats or s - beats[-1] > 0.12:
            beats.append(s)
    return bpm, beats


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

@dataclass
class BeatInfo:
    """Beat detection result: tempo, grid, and which engine produced it."""
    bpm: float
    beats: list[float] = field(default_factory=list)
    backend: str = "numpy"  # "librosa" | "numpy"


def detect_beats(audio_path: str | os.PathLike[str], *,
                 sr: int = _SR,
                 min_bpm: float = _MIN_BPM,
                 max_bpm: float = _MAX_BPM,
                 backend: str = "auto") -> list[float]:
    """Detect beat times (seconds) in ``audio_path``.

    ``backend="auto"`` tries librosa first and falls back to the numpy
    engine; ``"librosa"``/``"numpy"`` pin one (librosa still degrades to
    numpy on failure rather than raising). Returns [] when no periodic
    beat structure is found — never hallucinates a grid on silence.

    The ``min_bpm``/``max_bpm`` range only affects the numpy engine
    (librosa uses its own defaults); values are clamped to 30–240.
    """
    info = detect_beats_full(audio_path, sr=sr, min_bpm=min_bpm,
                             max_bpm=max_bpm, backend=backend)
    return info.beats


def detect_beats_full(audio_path: str | os.PathLike[str], *,
                      sr: int = _SR,
                      min_bpm: float = _MIN_BPM,
                      max_bpm: float = _MAX_BPM,
                      backend: str = "auto") -> BeatInfo:
    """Like :func:`detect_beats` but also returns tempo + backend used."""
    min_bpm = float(min(240.0, max(30.0, min_bpm)))
    max_bpm = float(min(240.0, max(30.0, max_bpm)))
    if min_bpm > max_bpm:
        min_bpm, max_bpm = max_bpm, min_bpm
    y = _decode_mono(audio_path, sr=sr)
    if y.size == 0:
        return BeatInfo(bpm=0.0, beats=[], backend="numpy")
    beats: list[float] | None = None
    if backend in ("auto", "librosa"):
        beats = _beats_librosa(y, sr)
    if beats is not None:
        used = "librosa"
        bpm = 0.0
        if len(beats) >= 2:
            gaps = np.diff(sorted(beats))
            gaps = gaps[gaps > 0]
            if len(gaps):
                bpm = 60.0 / float(np.median(gaps))
    else:
        _log.info("beat detection: numpy fallback engine")
        bpm, beats = _beats_numpy(y, sr, min_bpm=min_bpm, max_bpm=max_bpm)
        used = "numpy"
        _log.info("numpy beat engine: %.1f bpm, %d beats", bpm, len(beats))
    # sanitize: finite, non-negative, sorted, deduplicated
    clean = sorted({round(float(b), 4) for b in beats
                    if np.isfinite(b) and b >= 0})
    return BeatInfo(bpm=round(float(bpm), 2), beats=clean, backend=used)
