"""Devon's own audio fingerprinting + acoustic analysis.

Native-first music recognition: Devon listens HERSELF.

* :func:`fingerprint` — Shazam-style spectral-constellation hashes
  extracted locally from any audio file. No API, no key, no network.
  ``peak_mode="localmax"`` uses the true Wang-2003 local-maximum filter;
  fan-out 15 per anchor.
* :func:`identify` / :func:`match_batch` — one-call and batch recognition.
* :class:`FingerprintDB` — a local sqlite database of fingerprints:
  Devon's own productions, the user's music library, anything she has
  heard before. :meth:`FingerprintDB.match` identifies those with
  time-coherent offset scoring (the real Shazam matching idea).
  :meth:`FingerprintDB.index_directory` fingerprints a whole folder.
* :func:`analyze` / :func:`describe_audio` — native acoustic analysis:
  tempo (onset autocorrelation), key (chroma + Krumhansl profiles),
  chord, tuning (cents off A440), loudness, brightness, speech-vs-music,
  clipping, silence, plus Essentia-style quality flags (SNR estimate,
  mains-hum presence, saturation). What Devon can honestly say about
  audio she has never heard before.

What this is NOT: a global commercial catalogue. Identifying arbitrary
songs from the world still needs an external service (AudD stays as the
fallback for that). But anything Devon made, or anything in the local
library, she recognizes on her own — and she never pays an API to hear
what she already knows.

Dependencies: stdlib + optional numpy (fast FFT). Without numpy a
pure-Python Cooley-Tukey FFT is used — correct, slower; the analysis
window shrinks so it stays usable on small profiles (honest
profile-gating, not a silent downgrade).
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import struct
import tempfile
import threading
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..core.logging_setup import get_logger

_log = logging.getLogger(__name__)

__all__ = [
    "AudioReadError",
    "AudioAnalysis",
    "read_mono",
    "has_numpy",
    "fingerprint",
    "export_constellation",
    "FingerprintDB",
    "default_fingerprint_db",
    "analyze",
    "describe_audio",
    "index_own",
    "identify",
    "match_local",
    "match_local_db",
    "match_batch",
    "ANALYSIS_WINDOW_SECONDS",
    "FINGERPRINT_WINDOW_SECONDS",
]

#: How much audio the analysis stage listens to (seconds). Keeps the
#: pure-Python FFT path usable and bounds memory on long files.
ANALYSIS_WINDOW_SECONDS = 60.0
#: How much audio gets fingerprinted per file (seconds). Fingerprints
#: are a summary — 2 minutes covers verses + choruses for matching.
FINGERPRINT_WINDOW_SECONDS = 120.0
#: Fingerprint analysis sample rate (mono). High enough for the
#: musically interesting band, low enough to stay cheap.
_FP_SR = 11025
#: STFT window / hop (samples at _FP_SR). 4096 ≈ 370ms.
_WIN = 4096
_HOP = 2048
#: Constellation: frequency bands and peaks per frame.
_BANDS = 32
_FMIN, _FMAX = 80.0, 8000.0
#: Hash fan-out: pairs per anchor peak. 15 is the measured sweet spot
#: (pair-hashing separates the true match ~18x further from the
#: runner-up than small fan-outs — the constellation stays robust).
_FANOUT = 15
#: Max time delta between paired peaks (seconds).
_MAX_DT = 3.0
#: Cap on stored hashes per file.
_MAX_HASHES = 4000
#: Local-match confidence threshold (coherent-offset fraction).
MATCH_THRESHOLD = 0.12

_LOCK = threading.Lock()


class AudioReadError(RuntimeError):
    """The audio could not be decoded — engine named, not blamed."""


# ---------------------------------------------------------------------------
# loading — stdlib WAV first, ffmpeg decode as the strategy fallback
# ---------------------------------------------------------------------------

def _ffmpeg_path() -> str | None:
    try:
        from ..media_edit.videos import ffmpeg_path
        return ffmpeg_path()
    except Exception:  # noqa: BLE001
        return None


def _read_wav(path: str) -> tuple[array, int]:
    """WAV → (mono float samples as array('d'), sample rate). Never fakes."""
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    n = len(raw) // max(1, width)
    if width == 1:
        vals = [(b - 128) / 128.0 for b in raw[:n]]
    elif width == 2:
        vals = [v / 32768.0 for v in struct.unpack("<%dh" % n, raw[: n * 2])]
    elif width == 4:
        vals = [v / 2147483648.0
                for v in struct.unpack("<%di" % n, raw[: n * 4])]
    else:
        raise AudioReadError(f"unsupported WAV sample width: {width}")
    if nch > 1 and n:
        frames = n // nch
        vals = [sum(vals[i * nch:(i + 1) * nch]) / nch
                for i in range(frames)]
    return array("d", vals), int(sr or 8000)


def _resample_linear(samples: array, src_sr: int, dst_sr: int) -> array:
    """Linear-interpolation resample. Total (never raises)."""
    try:
        if src_sr == dst_sr or len(samples) < 2:
            return array("d", samples)
        ratio = dst_sr / src_sr
        n = max(1, int(round(len(samples) * ratio)))
        out = array("d", [0.0]) * n
        last = len(samples) - 1
        for i in range(n):
            pos = i / ratio
            lo = int(pos)
            hi = min(last, lo + 1)
            frac = pos - lo
            out[i] = samples[lo] * (1.0 - frac) + samples[hi] * frac
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def read_mono(path: str | os.PathLike[str], *,
              target_sr: int = _FP_SR,
              max_seconds: float = FINGERPRINT_WINDOW_SECONDS) -> tuple[array, int]:
    """Decode ``path`` → (mono float samples, sample rate).

    Strategy chain: stdlib ``wave`` for WAV, ffmpeg decode for
    everything else (ogg/opus/m4a/mp3). Raises :class:`AudioReadError`
    with the honest reason when neither can read the file.
    """
    p = str(path)
    if not os.path.isfile(p):
        raise AudioReadError(f"no such file: {path}")
    samples: array | None = None
    sr = 0
    if p.lower().endswith(".wav"):
        try:
            samples, sr = _read_wav(p)
        except Exception as exc:  # noqa: BLE001 - fall to ffmpeg
            _log.debug("stdlib wav read failed for %s: %s", p, exc)
            samples = None
    if samples is None:
        ff = _ffmpeg_path()
        if ff is None:
            raise AudioReadError(
                f"cannot decode {os.path.basename(p)} — not a readable WAV "
                "and ffmpeg is not installed (needed for ogg/mp3/m4a)")
        tmp = tempfile.mkdtemp(prefix="devon-fp-")
        out = os.path.join(tmp, "decoded.wav")
        try:
            from ..media_edit.videos import run_ffmpeg
            run_ffmpeg(["-i", p, "-ac", "1", "-ar", str(target_sr), out],
                       timeout=300.0)
            samples, sr = _read_wav(out)
        except Exception as exc:
            raise AudioReadError(
                f"ffmpeg could not decode {os.path.basename(p)}: {exc}"
            ) from exc
        finally:
            try:
                os.remove(out)
                os.rmdir(tmp)
            except OSError:
                pass
    if max_seconds and max_seconds > 0:
        cap = int(max_seconds * sr)
        if len(samples) > cap:
            samples = samples[:cap]
    samples = _resample_linear(samples, sr, target_sr)
    return samples, target_sr


# ---------------------------------------------------------------------------
# FFT — numpy when present, honest pure-Python Cooley-Tukey otherwise
# ---------------------------------------------------------------------------

def has_numpy() -> bool:
    try:
        __import__("numpy")
        return True
    except Exception:  # noqa: BLE001
        return False


def _fft_pure(x: list[complex]) -> list[complex]:
    """Iterative radix-2 Cooley-Tukey. len(x) must be a power of 2."""
    n = len(x)
    a = list(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wlen = complex(math.cos(ang), math.sin(ang))
        for i in range(0, n, length):
            w = 1 + 0j
            for k in range(length // 2):
                u = a[i + k]
                v = a[i + k + length // 2] * w
                a[i + k] = u + v
                a[i + k + length // 2] = u - v
                w *= wlen
        length <<= 1
    return a


def _spectrum_magnitudes(frame: Sequence[float]) -> list[float]:
    """|FFT| of one windowed frame, first half only. Total."""
    n = len(frame)
    hann = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / max(1, n - 1))
            for i in range(n)]
    win = [frame[i] * hann[i] for i in range(n)]
    if has_numpy():
        import numpy as np

        spec = np.abs(np.fft.rfft(np.asarray(win, dtype=np.float64)))
        return [float(v) for v in spec]
    comp = _fft_pure([complex(v, 0.0) for v in win])
    half = n // 2 + 1
    return [abs(comp[i]) for i in range(half)]


def _frames(samples: Sequence[float], sr: int,
            win: int = _WIN, hop: int = _HOP) -> list[list[float]]:
    """Windowed magnitude spectra over the samples. Total."""
    n = len(samples)
    if n < win:
        return []
    out: list[list[float]] = []
    for start in range(0, n - win + 1, hop):
        out.append(_spectrum_magnitudes(samples[start:start + win]))
    return out


def _bin_freqs(n_bins: int, sr: int, win: int) -> list[float]:
    return [i * sr / win for i in range(n_bins)]


# ---------------------------------------------------------------------------
# spectral constellation + Shazam-style hashing
# ---------------------------------------------------------------------------

def _constellation_peaks(frames: list[list[float]], sr: int,
                         win: int = _WIN,
                         with_magnitude: bool = False
                         ) -> list[tuple[float, float]] | list[tuple[float, float, float]]:
    """Peaks = (freq_hz, time_s): strongest bin in each log band per frame.

    A peak must be a local maximum clearing twice the frame median —
    noise raises the median, so weak maxima are not landmarks. With
    ``with_magnitude=True`` returns (freq, time, magnitude) triples so
    callers can keep only the strongest landmarks. Total.
    """
    peaks: list = []
    try:
        if not frames:
            return peaks
        freqs = _bin_freqs(len(frames[0]), sr, win)
        # log-spaced band edges
        edges = [_FMIN * ((_FMAX / _FMIN) ** (b / _BANDS))
                 for b in range(_BANDS + 1)]
        for fi, mags in enumerate(frames):
            # peak must be a real standout: clear twice the frame median
            # (noise raises the median — weak maxima are not landmarks)
            med = sorted(mags)[len(mags) // 2] if mags else 0.0
            floor = med * 2.0
            t = fi * _HOP / sr
            bi = 0
            for b in range(_BANDS):
                lo, hi = edges[b], edges[b + 1]
                best_i, best_v = -1, floor
                while bi < len(freqs) and freqs[bi] < lo:
                    bi += 1
                j = bi
                while j < len(freqs) and freqs[j] < hi:
                    # local maximum (not a noise shoulder)
                    if (mags[j] > best_v and
                            (j == 0 or mags[j] >= mags[j - 1]) and
                            (j + 1 >= len(mags) or
                             mags[j] >= mags[j + 1])):
                        best_v, best_i = mags[j], j
                    j += 1
                if best_i >= 0:
                    if with_magnitude:
                        peaks.append((freqs[best_i], t, best_v))
                    else:
                        peaks.append((freqs[best_i], t))
        return peaks
    except Exception:  # noqa: BLE001
        _log.debug("constellation failed", exc_info=True)
        return []


#: Fraction of the global peak magnitude a landmark must reach to be
#: hashed. Keeps the constellation on the dominant tones — the part of
#: the spectrum that survives noise, encoding, and room coloration.
_PEAK_STRENGTH_FRACTION = 0.12


def _localmax_peaks(frames: list[list[float]], sr: int,
                     win: int = _WIN,
                     with_magnitude: bool = False
                     ) -> list:
    """Constellation via a true local-maximum filter (the Wang 2003
    pattern): a bin is a peak only if it is the maximum of its
    freq×time neighborhood AND clears twice the frame median.

    More faithful to the original Shazam algorithm than band slicing —
    a sustained note no longer plants a peak in every frame of an empty
    band. Requires numpy; falls back to an empty list otherwise (the
    caller then uses the band-based peaks). Total.
    """
    peaks: list = []
    try:
        if not frames or not has_numpy():
            return peaks
        import numpy as np

        spec = np.asarray(frames, dtype=np.float64)  # (time, freq)
        if spec.shape[0] < 3 or spec.shape[1] < 3:
            return peaks
        med = np.median(spec, axis=1, keepdims=True)
        floor = med * 2.0
        # local max over a 3x3 freq×time neighborhood (symmetric padding
        # keeps the output shape identical to the input)
        padded = np.pad(spec, 1, mode="constant")
        nb = np.stack([padded[i:i + spec.shape[0], j:j + spec.shape[1]]
                       for i in range(3) for j in range(3)], axis=-1)
        nbmax = nb.max(axis=-1)
        is_peak = (spec >= floor) & (spec >= nbmax)
        freqs = _bin_freqs(spec.shape[1], sr, win)
        ts, fs = np.nonzero(is_peak)
        for t_i, f_i in zip(ts.tolist(), fs.tolist()):
            f = freqs[f_i] if f_i < len(freqs) else 0.0
            tm = t_i * _HOP / sr
            if with_magnitude:
                peaks.append((f, tm, float(spec[t_i, f_i])))
            else:
                peaks.append((f, tm))
        return peaks
    except Exception:  # noqa: BLE001
        _log.debug("localmax constellation failed", exc_info=True)
        return []


def _hash_pair(f1: float, f2: float, dt: float) -> int:
    """Quantized constellation hash: f1 | f2 | Δt → one int.

    8 Hz frequency steps — coarse enough that noise-driven ±1-bin
    wobble lands in the same bucket, fine enough to stay distinctive.
    """
    q1 = int(f1 // 8) & 0x3FF          # 8 Hz steps, 10 bits
    q2 = int(f2 // 8) & 0x3FF
    qd = int(dt / 0.02) & 0x3FF       # 20 ms steps, 10 bits
    return (q1 << 20) | (q2 << 10) | qd


def fingerprint(path: str | os.PathLike[str], *,
                max_hashes: int = _MAX_HASHES,
                max_seconds: float = FINGERPRINT_WINDOW_SECONDS,
                peak_mode: str = "bands",
                ) -> list[tuple[int, float]]:
    """Extract Shazam-style constellation hashes from ``path``.

    ``peak_mode="bands"`` (default) takes the strongest local maximum in
    each log-spaced frequency band per frame; ``peak_mode="localmax"``
    uses a true freq×time local-maximum filter (the Wang 2003 pattern —
    more faithful, needs numpy, falls back to bands without it).

    Returns ``[(hash, offset_seconds), ...]`` — Devon's own fingerprint,
    extracted locally. Raises :class:`AudioReadError` when the file
    cannot be decoded.
    """
    samples, sr = read_mono(path, target_sr=_FP_SR,
                            max_seconds=max_seconds)
    frames = _frames(samples, sr)
    if (peak_mode or "bands").lower() == "localmax":
        peaks = _localmax_peaks(frames, sr, with_magnitude=True)
        if not peaks:
            peaks = _constellation_peaks(frames, sr, with_magnitude=True)
    else:
        peaks = _constellation_peaks(frames, sr, with_magnitude=True)
    # keep the strong landmarks only — dominant tones survive noise,
    # encoding, and room coloration; weak ones don't
    if peaks:
        gmax = max(m for _, _, m in peaks)
        peaks = [(f, t) for f, t, m in peaks
                 if m >= gmax * _PEAK_STRENGTH_FRACTION]
    else:
        peaks = []
    hashes: list[tuple[int, float]] = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        for j in range(i + 1, min(n, i + 1 + _FANOUT)):
            f2, t2 = peaks[j]
            dt = t2 - t1
            if dt <= 0 or dt > _MAX_DT:
                continue
            hashes.append((_hash_pair(f1, f2, dt), t1))
            if len(hashes) >= max_hashes:
                return hashes
    return hashes


# ---------------------------------------------------------------------------
# local fingerprint database — Devon recognizes what she has heard
# ---------------------------------------------------------------------------

_DEFAULT_DB = ""


def _media_duration(path: str | os.PathLike[str]) -> float:
    """Honest duration in seconds. WAV via stdlib, else ffprobe, else 0."""
    try:
        p = str(path)
        if p.lower().endswith(".wav"):
            with wave.open(p, "rb") as wf:
                return wf.getnframes() / max(1, wf.getframerate())
    except Exception:  # noqa: BLE001
        pass
    try:
        import shutil
        import subprocess

        if shutil.which("ffprobe") is None:
            return 0.0
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return float((out.stdout or "").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def export_constellation(path: str | os.PathLike[str], *,
                         peak_mode: str = "bands",
                         max_peaks: int = 500) -> dict[str, Any]:
    """Export the spectral constellation as JSON-able data: the peaks
    Devon hears when she fingerprints a track. Powers "show the work"
    displays (spectrogram + constellation + offset histogram).

    Never raises; returns ``{"ok", "peaks": [(freq_hz, time_s), ...]}``.
    """
    try:
        samples, sr = read_mono(path, target_sr=_FP_SR,
                                max_seconds=FINGERPRINT_WINDOW_SECONDS)
        frames = _frames(samples, sr)
        if (peak_mode or "bands").lower() == "localmax":
            peaks = _localmax_peaks(frames, sr)
            if not peaks:
                peaks = _constellation_peaks(frames, sr)
        else:
            peaks = _constellation_peaks(frames, sr)
        peaks = [(round(float(f), 1), round(float(t), 3))
                 for f, t in peaks[:max(1, max_peaks)]]
        return {"ok": True, "peaks": peaks, "count": len(peaks),
                "peak_mode": peak_mode,
                "window_s": FINGERPRINT_WINDOW_SECONDS}
    except AudioReadError as exc:
        return {"ok": False, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"constellation export failed: {exc}"}


def _default_db_path() -> str:
    d = Path.home() / ".nomorals" / "audio"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:  # noqa: BLE001
        pass
    return str(d / "fingerprints.db")


class FingerprintDB:
    """Devon's local recognition memory: fingerprints of her own
    productions, the user's music library, anything indexed via
    :func:`index_own` or ``/audio fingerprint``.

    Matching is time-coherent: a track scores by the largest set of
    hashes that agree on ONE time offset (the Shazam idea) — random
    collisions don't form a coherent line. ``match()`` never raises.
    """

    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS tracks ("
                "id INTEGER PRIMARY KEY, title TEXT, artist TEXT, "
                "source TEXT, duration REAL, created REAL)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS hashes ("
                "track_id INTEGER, hash INTEGER, offset_ms INTEGER)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hashes_hash "
                "ON hashes(hash)")
            self._conn.commit()
        return self._conn

    def add_track(self, path: str | os.PathLike[str], *,
                  title: str = "", artist: str = "",
                  source: str = "devon") -> dict[str, Any]:
        """Fingerprint ``path`` into the local database. Never raises."""
        try:
            hashes = fingerprint(path)
            if not hashes:
                return {"ok": False,
                        "reason": "no fingerprints extracted — "
                                  "audio too quiet or too short"}
            import time as _t
            duration = _media_duration(path)
            with self._lock:
                conn = self._connect()
                cur = conn.execute(
                    "INSERT INTO tracks (title, artist, source, duration, "
                    "created) VALUES (?,?,?,?,?)",
                    (title or Path(str(path)).stem,
                     artist or "unknown", source, duration, _t.time()))
                tid = cur.lastrowid
                conn.executemany(
                    "INSERT INTO hashes (track_id, hash, offset_ms) "
                    "VALUES (?,?,?)",
                    [(tid, h, int(t * 1000)) for h, t in hashes])
                conn.commit()
            return {"ok": True, "track_id": tid, "hashes": len(hashes),
                    "title": title or Path(str(path)).stem,
                    "duration_s": round(duration, 1)}
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            _log.debug("add_track failed", exc_info=True)
            return {"ok": False, "reason": f"indexing failed: {exc}"}

    def index_directory(self, directory: str | os.PathLike[str], *,
                        extensions: tuple[str, ...] = (
                            ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"),
                        source: str = "library",
                        artist: str = "") -> dict[str, Any]:
        """Fingerprint every audio file in ``directory`` (the
        ``build_database.py`` pattern). Never raises."""
        try:
            d = Path(directory)
            if not d.is_dir():
                return {"ok": False,
                        "reason": f"no such directory: {directory}"}
            files = sorted(
                p for p in d.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions)
            indexed, failed = 0, 0
            for f in files:
                res = self.add_track(f, artist=artist, source=source)
                if res.get("ok"):
                    indexed += 1
                else:
                    failed += 1
            return {"ok": True, "indexed": indexed, "failed": failed,
                    "scanned": len(files)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"directory index failed: {exc}"}

    def stats(self) -> dict[str, Any]:
        """Database size at a glance. Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                tracks = conn.execute(
                    "SELECT COUNT(*) FROM tracks").fetchone()[0]
                hashes = conn.execute(
                    "SELECT COUNT(*) FROM hashes").fetchone()[0]
            return {"ok": True, "tracks": tracks, "hashes": hashes}
        except Exception:  # noqa: BLE001
            return {"ok": False, "tracks": 0, "hashes": 0}

    def match(self, path: str | os.PathLike[str], *,
              threshold: float = MATCH_THRESHOLD
              ) -> dict[str, Any]:
        """Identify ``path`` against the local database. Never raises.

        Returns ``{"ok": True, "title", "artist", "score", "track_id"}``
        on a confident match, else ``{"ok": False, "reason"}``.
        """
        try:
            hashes = fingerprint(path)
            if not hashes:
                return {"ok": False,
                        "reason": "no fingerprints extracted — "
                                  "audio too quiet or too short"}
            with self._lock:
                conn = self._connect()
                # pull candidate (track_id, offset_ms) rows per hash
                cand: dict[int, list[int]] = {}
                for h, t in hashes:
                    rows = conn.execute(
                        "SELECT track_id, offset_ms FROM hashes "
                        "WHERE hash = ?", (h,)).fetchall()
                    q_ms = int(t * 1000)
                    for tid, off in rows:
                        cand.setdefault(tid, []).append(off - q_ms)
                best: dict[str, Any] = {"ok": False,
                                       "reason": "no match in local library"}
                best_score = 0.0
                for tid, deltas in cand.items():
                    # coherent-offset histogram (50 ms bins)
                    bins: dict[int, int] = {}
                    for d in deltas:
                        b = int(d // 50)
                        bins[b] = bins.get(b, 0) + 1
                    if not bins:
                        continue
                    score = max(bins.values()) / max(1, len(hashes))
                    if score > best_score:
                        best_score = score
                        row = conn.execute(
                            "SELECT title, artist, source FROM tracks "
                            "WHERE id = ?", (tid,)).fetchone()
                        if row:
                            best = {"ok": True, "track_id": tid,
                                    "title": row[0], "artist": row[1],
                                    "source": row[2], "score": round(score, 3)}
                if best.get("ok") and best_score >= threshold:
                    return best
                return {"ok": False,
                        "reason": "no confident match in local library "
                                  f"(best score {best_score:.2f} < "
                                  f"{threshold:.2f})"}
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            _log.debug("match failed", exc_info=True)
            return {"ok": False, "reason": f"local match failed: {exc}"}

    def list_tracks(self) -> list[dict[str, Any]]:
        """Everything Devon can recognize locally. Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                rows = conn.execute(
                    "SELECT id, title, artist, source, duration, created "
                    "FROM tracks ORDER BY created DESC").fetchall()
            return [{"track_id": r[0], "title": r[1], "artist": r[2],
                     "source": r[3], "duration_s": round(r[4] or 0.0, 1)}
                    for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def remove_track(self, track_id: int) -> bool:
        """Drop a track and its hashes. Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                conn.execute("DELETE FROM hashes WHERE track_id = ?",
                             (track_id,))
                cur = conn.execute("DELETE FROM tracks WHERE id = ?",
                                   (track_id,))
                conn.commit()
                return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._conn = None


def default_fingerprint_db(db_path: str = "") -> FingerprintDB:
    """The shared local recognition memory."""
    return FingerprintDB(db_path or _default_db_path())


def index_own(path: str | os.PathLike[str], title: str = "",
              artist: str = "Devon") -> dict[str, Any]:
    """Index one of Devon's own productions for local recognition.

    Called after /produce renders a song — from then on /sham
    recognizes it without any API. Never raises.
    """
    try:
        return default_fingerprint_db().add_track(
            path, title=title or Path(str(path)).stem, artist=artist,
            source="devon")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"index_own failed: {exc}"}


def match_local_db(path: str | os.PathLike[str], *,
                   db: FingerprintDB | None = None) -> dict[str, Any]:
    """Match against Devon's local recognition memory. Never raises."""
    try:
        return (db or default_fingerprint_db()).match(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"local match failed: {exc}"}


#: Backwards-friendly alias.
match_local = match_local_db


def identify(path: str | os.PathLike[str], *,
             db: FingerprintDB | None = None,
             threshold: float = MATCH_THRESHOLD) -> dict[str, Any]:
    """One call: "what is this audio?" → match result dict. Never raises."""
    try:
        return (db or default_fingerprint_db()).match(
            path, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"identify failed: {exc}"}


def match_batch(paths: Sequence[str | os.PathLike[str]], *,
                db: FingerprintDB | None = None,
                threshold: float = MATCH_THRESHOLD) -> list[dict[str, Any]]:
    """Identify many clips at once (batch Shazam). Never raises per item —
    each entry carries its own ``{"ok", ...}`` or ``{"ok": False}``.
    """
    out: list[dict[str, Any]] = []
    try:
        db = db or default_fingerprint_db()
        for p in (paths or []):
            res = db.match(p, threshold=threshold)
            res["path"] = str(p)
            out.append(res)
        return out
    except Exception as exc:  # noqa: BLE001
        return [{"ok": False, "reason": f"batch match failed: {exc}",
                 "path": str(p)} for p in (paths or [])]


# ---------------------------------------------------------------------------
# acoustic analysis — what Devon can honestly say about unknown audio
# ---------------------------------------------------------------------------

#: Krumhansl-Schmuckler key profiles (research standard).
_KRUMHANSL_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KRUMHANSL_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                    2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B")


@dataclass
class AudioAnalysis:
    """Everything Devon hears in a clip, natively."""

    path: str = ""
    ok: bool = False
    reason: str = ""
    seconds: float = 0.0
    sample_rate: int = 0
    #: dBFS
    rms_db: float = -96.0
    peak_db: float = -96.0
    spectral_centroid_hz: float = 0.0
    zero_crossing_rate: float = 0.0
    tempo_bpm: float = 0.0
    tempo_confidence: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    #: dominant chord (Essentia ChordsDetection pattern)
    chord: str = ""
    chord_confidence: float = 0.0
    #: tuning deviation in cents from A440 (+ = sharp)
    tuning_cents: float = 0.0
    #: Essentia-style audio-problem flags
    quality: dict[str, Any] = field(default_factory=dict)
    #: 0 = pure speech, 1 = pure music (heuristic — labeled as such)
    music_score: float = 0.5
    silence_ratio: float = 0.0
    clipped_ratio: float = 0.0
    engine: str = "devon-native"

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "path", "ok", "reason", "seconds", "sample_rate", "rms_db",
            "peak_db", "spectral_centroid_hz", "zero_crossing_rate",
            "tempo_bpm", "tempo_confidence", "key", "key_confidence",
            "chord", "chord_confidence", "tuning_cents", "quality",
            "music_score", "silence_ratio", "clipped_ratio", "engine")}


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(max(1e-9, abs(x)))


def _estimate_tempo(flux: list[float], sr: int, hop: int) -> tuple[float, float]:
    """Onset-envelope autocorrelation → BPM in [60, 200]."""
    try:
        n = len(flux)
        if n < 32:
            return 0.0, 0.0
        mean = sum(flux) / n
        dev = [f - mean for f in flux]
        var = sum(d * d for d in dev) / n
        if var <= 1e-12:
            return 0.0, 0.0
        frame_s = hop / sr
        best_bpm, best_corr = 0.0, 0.0
        for bpm in range(60, 201):
            lag = int(round(60.0 / bpm / frame_s))
            if lag < 2 or lag >= n:
                continue
            num = sum(dev[i] * dev[i + lag] for i in range(n - lag))
            corr = num / (n - lag) / var
            if corr > best_corr:
                best_corr, best_bpm = corr, float(bpm)
        return best_bpm, max(0.0, min(1.0, best_corr))
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _chroma_vector(frames: list[list[float]], sr: int,
                   win: int = _WIN) -> list[float]:
    """12-bin normalized chroma vector (shared by key + chord + tuning).

    Total.
    """
    try:
        if not frames:
            return [0.0] * 12
        freqs = _bin_freqs(len(frames[0]), sr, win)
        chroma = [0.0] * 12
        for mags in frames:
            for i, m in enumerate(mags):
                f = freqs[i]
                if f < 55.0 or f > 4000.0 or m <= 0:
                    continue
                midi = 69 + 12.0 * math.log2(f / 440.0)
                pc = int(round(midi)) % 12
                chroma[pc] += m
        total = sum(chroma)
        if total <= 0:
            return [0.0] * 12
        return [c / total for c in chroma]
    except Exception:  # noqa: BLE001
        return [0.0] * 12


def _chroma_key(frames: list[list[float]], sr: int,
                win: int = _WIN) -> tuple[str, float]:
    """12-bin chroma → best Krumhansl major/minor rotation."""
    try:
        chroma = _chroma_vector(frames, sr, win)
        if sum(chroma) <= 0:
            return "", 0.0
        best_name, best_corr = "", -2.0
        for mode, prof in (("major", _KRUMHANSL_MAJOR),
                           ("minor", _KRUMHANSL_MINOR)):
            pm = sum(prof) / 12.0
            for root in range(12):
                # profile anchored at pitch class `root`: pitch class i
                # sits (i - root) semitones above the tonic
                rot = [prof[(i - root) % 12] for i in range(12)]
                num = sum((chroma[i] - sum(chroma) / 12.0) * (rot[i] - pm)
                          for i in range(12))
                den = math.sqrt(
                    sum((chroma[i] - sum(chroma) / 12.0) ** 2
                        for i in range(12))
                    * sum((r - pm) ** 2 for r in rot))
                corr = num / den if den > 0 else 0.0
                if corr > best_corr:
                    best_corr = corr
                    suffix = "" if mode == "major" else "m"
                    best_name = f"{_NOTE_NAMES[root]}{suffix}"
        conf = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))
        return best_name, conf
    except Exception:  # noqa: BLE001
        return "", 0.0


_CHORD_TEMPLATES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
}


def _detect_chord(frames: list[list[float]], sr: int,
                  win: int = _WIN) -> tuple[str, float]:
    """Dominant chord: chroma-template match over 12 roots × qualities
    (the Essentia ChordsDetection pattern, simplified). Returns
    (\"Am\", confidence). Total.
    """
    try:
        chroma = _chroma_vector(frames, sr, win)
        if sum(chroma) <= 0:
            return "", 0.0
        best, best_s = "", -1.0
        for root in range(12):
            for qual, ivs in _CHORD_TEMPLATES.items():
                tmpl = [0.0] * 12
                for iv in ivs:
                    tmpl[(root + iv) % 12] = 1.0
                # cosine similarity
                num = sum(c * t for c, t in zip(chroma, tmpl))
                den = math.sqrt(sum(c * c for c in chroma)
                                * sum(t * t for t in tmpl))
                s = num / den if den > 0 else 0.0
                if s > best_s:
                    best_s = s
                    best = f"{_NOTE_NAMES[root]}{qual}"
        return best, round(max(0.0, min(1.0, best_s)), 3)
    except Exception:  # noqa: BLE001
        return "", 0.0


def _estimate_tuning(frames: list[list[float]], sr: int,
                     win: int = _WIN) -> float:
    """Tuning deviation in cents from A440 (Essentia TuningFrequency
    pattern): strongest spectral peak in 80–1000 Hz → cents off the
    nearest equal-tempered note. +50 = a quarter-tone sharp. Total.
    """
    try:
        if not frames:
            return 0.0
        freqs = _bin_freqs(len(frames[0]), sr, win)
        acc = [0.0] * len(freqs)
        for mags in frames:
            for i, m in enumerate(mags):
                acc[i] += m
        best_i, best_v = -1, 0.0
        for i, f in enumerate(freqs):
            if 80.0 <= f <= 1000.0 and acc[i] > best_v:
                best_v, best_i = acc[i], i
        if best_i < 0:
            return 0.0
        f = freqs[best_i]
        midi = 69 + 12.0 * math.log2(f / 440.0)
        return round((midi - round(midi)) * 100.0, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def _quality_flags(samples: Sequence[float], sr: int,
                    frames: list[list[float]]) -> dict[str, Any]:
    """Essentia-style audio-problem flags: SNR estimate, mains-hum
    presence, saturation. Total — never raises.
    """
    out: dict[str, Any] = {"snr_db": 0.0, "hum": False,
                           "saturated": False, "dc_offset": 0.0}
    try:
        n = len(samples)
        if n == 0 or not frames:
            return out
        # SNR: median frame energy of the loudest half vs the noise
        # floor (minimum-statistics: the noise floor is what the
        # quietest moments look like — same idea as the DSP denoiser)
        mags = [[abs(c) for c in fr] for fr in frames]
        nfr = len(mags)
        energies = sorted((sum(m * m for m in fr), i)
                          for i, fr in enumerate(mags))
        k = max(1, nfr // 10)
        quiet = [i for _, i in energies[:k]]
        nbins = len(mags[0])
        noise_e = 0.0
        for b in range(nbins):
            col = sorted(mags[i][b] for i in quiet)
            p = col[len(col) // 2] if col else 0.0
            noise_e += p * p
        noise_e /= max(1, nbins)
        sig_e = sum(m * m for fr in mags for m in fr) / max(1, nfr * nbins)
        out["snr_db"] = round(
            10.0 * math.log10(sig_e / max(1e-12, noise_e)), 1)
        # hum: is there a standout at 50/60 Hz (+harmonics)?
        freqs = _bin_freqs(len(frames[0]), sr, _WIN)
        acc = [0.0] * len(freqs)
        for mg in mags:
            for i, m in enumerate(mg):
                acc[i] += m
        med = sorted(acc)[len(acc) // 2] if acc else 0.0
        hum_hit = False
        for base in (50.0, 60.0):
            for mult in (1, 2, 3):
                f = base * mult
                if f >= sr / 2 - 50:
                    continue
                i = min(len(acc) - 1, int(f / (sr / _WIN)))
                if acc[i] > med * 6.0:
                    hum_hit = True
        out["hum"] = hum_hit
        out["saturated"] = any(abs(s) >= 0.99 for s in samples[:sr * 5])
        out["dc_offset"] = round(
            sum(samples[:sr * 5]) / max(1, min(n, sr * 5)), 4)
        return out
    except Exception:  # noqa: BLE001
        return out


def analyze(path: str | os.PathLike[str], *,
            max_seconds: float = ANALYSIS_WINDOW_SECONDS) -> AudioAnalysis:
    """Listen to ``path`` natively and describe what is in it.

    Never raises — failures land in ``AudioAnalysis.ok/reason``.
    """
    res = AudioAnalysis(path=str(path))
    try:
        samples, sr = read_mono(path, target_sr=_FP_SR,
                                max_seconds=max_seconds)
    except AudioReadError as exc:
        res.reason = str(exc)
        return res
    try:
        n = len(samples)
        if n < _WIN:
            res.reason = "audio too short to analyze"
            return res
        res.seconds = round(n / sr, 2)
        res.sample_rate = sr
        peak = 0.0
        ssum = 0.0
        clip = 0
        quiet = 0
        zc = 0
        prev = samples[0]
        for s in samples:
            a = abs(s)
            if a > peak:
                peak = a
            ssum += s * s
            if a >= 0.99:
                clip += 1
            if a < 0.01:
                quiet += 1
            if (s >= 0) != (prev >= 0):
                zc += 1
            prev = s
        res.rms_db = round(_dbfs(math.sqrt(ssum / n)), 1)
        res.peak_db = round(_dbfs(peak), 1)
        res.zero_crossing_rate = round(zc / max(1, n - 1), 4)
        res.clipped_ratio = round(clip / n, 4)
        res.silence_ratio = round(quiet / n, 3)

        frames = _frames(samples, sr)
        # spectral centroid + flux (onset envelope)
        flux: list[float] = []
        prev_mags: list[float] | None = None
        num = 0.0
        den = 0.0
        for mags in frames:
            freqs = _bin_freqs(len(mags), sr, _WIN)
            for i, m in enumerate(mags):
                num += freqs[i] * m
                den += m
            if prev_mags is not None:
                flux.append(sum(max(0.0, mags[i] - prev_mags[i])
                                for i in range(len(mags))))
            prev_mags = mags
        res.spectral_centroid_hz = round(num / den, 1) if den > 0 else 0.0

        bpm, conf = _estimate_tempo(flux, sr, _HOP)
        res.tempo_bpm = round(bpm, 1)
        res.tempo_confidence = round(conf, 3)
        key, kconf = _chroma_key(frames, sr)
        res.key = key
        res.key_confidence = round(kconf, 3)
        chord, cconf = _detect_chord(frames, sr)
        res.chord = chord
        res.chord_confidence = cconf
        res.tuning_cents = _estimate_tuning(frames, sr)
        res.quality = _quality_flags(samples, sr, frames)

        # music vs speech: music sustains harmonic energy with lower
        # frame-to-frame variance relative to its level; speech is
        # burstier. Heuristic — labeled as such in the docs.
        if flux:
            fmean = sum(flux) / len(flux)
            fvar = sum((f - fmean) ** 2 for f in flux) / len(flux)
            cv = math.sqrt(fvar) / (fmean + 1e-9)
            # low centroid + low cv → music; high zcr + high cv → speech
            score = 0.6 - 0.35 * min(1.0, res.zero_crossing_rate * 8.0) \
                - 0.35 * min(1.0, cv / 3.0) + 0.2
            res.music_score = round(max(0.0, min(1.0, score)), 2)
        res.ok = True
        return res
    except Exception as exc:  # noqa: BLE001
        _log.debug("analyze failed", exc_info=True)
        res.reason = f"analysis failed: {exc}"
        return res


def describe_audio(path: str | os.PathLike[str]) -> str:
    """One honest human line about what is in the audio. Never raises."""
    a = analyze(path)
    if not a.ok:
        return f"couldn't listen to that ({a.reason})"
    kind = ("music" if a.music_score >= 0.6 else
            "speech" if a.music_score <= 0.4 else "mixed audio")
    parts = [f"{a.seconds:.0f}s of {kind}"]
    if a.tempo_bpm and a.tempo_confidence >= 0.15:
        parts.append(f"~{a.tempo_bpm:.0f} BPM")
    if a.key and a.key_confidence >= 0.55:
        parts.append(f"key of {a.key}")
    if a.chord and a.chord_confidence >= 0.5:
        parts.append(f"leaning on {a.chord}")
    if abs(a.tuning_cents) >= 8:
        drift = "sharp" if a.tuning_cents > 0 else "flat"
        parts.append(f"tuned {abs(a.tuning_cents):.0f}¢ {drift}")
    parts.append(f"loudness {a.rms_db:.0f} dBFS")
    q = a.quality or {}
    flags: list[str] = []
    if a.clipped_ratio > 0.001 or q.get("saturated"):
        flags.append("clipped")
    if q.get("hum"):
        flags.append("mains hum")
    if q.get("snr_db") and q["snr_db"] < 10:
        flags.append(f"noisy (SNR {q['snr_db']:.0f} dB)")
    if a.silence_ratio > 0.5:
        flags.append("mostly silence")
    if flags:
        parts.append("⚠ " + ", ".join(flags))
    return " · ".join(parts) + f"  (heard natively, {a.engine})"
