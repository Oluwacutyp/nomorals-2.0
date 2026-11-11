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

import hashlib
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

from .failures import record_failure

_log = logging.getLogger(__name__)

__all__ = [
    "AudioReadError",
    "AudioAnalysis",
    "RecognitionReceipt",
    "external_recognition_receipt",
    "no_match_receipt",
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
#: Local-match acceptance threshold (coherent-offset fraction).
MATCH_THRESHOLD = 0.12

#: Distinct-landmark support floors for accepting a local match
#: (doc 24 SR2 §9 benchmark, 2026-11-10). A winner must clear the
#: coherent-offset threshold AND at least one breadth test:
#: ``support_fraction`` (distinct landmark types in the winning bin ÷
#: all query hashes) ≥ 0.05, or ``type_recall`` (distinct types in
#: the bin ÷ distinct types in the query) ≥ 0.25. Repetition is not
#: breadth — on the benchmark (12-track synthesized catalog, 322
#: accepted candidates) every false accept failed BOTH tests
#: (support ≤ 0.049, recall ≤ 0.104) while every correct accept
#: passed at least one (support ≥ 0.057, or recall ≈ 1.0 for
#: simple few-tone recordings whose whole vocabulary is small).
MIN_SUPPORT_FRACTION = 0.05
MIN_TYPE_RECALL = 0.25

#: Playback-rate corrections retried when direct voting accepts
#: nothing. A coupled speed/pitch edit (sped-up social clip, DJ
#: pitch-up) scales landmark frequencies by the rate and pair deltas
#: by its inverse, so unscaled voting finds ~nothing (benchmark:
#: 0.000 accepted-correct at every rate ±3–10% before this retry).
_RATE_RETRY_GRID = (0.90, 0.92, 0.94, 0.96, 0.98,
                    1.02, 1.04, 1.06, 1.08, 1.10)

#: Measured calibration of the local matcher (doc 24 SR2, §9
#: benchmark run 2026-11-10, schema songrecon.bench/v1, harness
#: ``nomorals.audio.songrecon_bench``). The claim carries its
#: evidence: on that run every accepted positive named the correct
#: track (359 of 359; the one unaccepted query ranked correctly but
#: scored under the threshold) and no non-catalog negative was
#: accepted (0 of 48). The per-score calibration the same run
#: measured for the unscaled top-1 ranking is weaker and stays
#: named in the benchmark report: fraction bin 0.12–0.20 was
#: correct only 0.1765 of the time (n=17), 0.20–0.35 was 0.9649
#: (n=57), and every bin at/over 0.35 was 1.0 (n=241). The scope
#: is the synthesized fixture catalog — never read this as
#: real-library accuracy.
MEASURED_CALIBRATION: dict[str, Any] = {
    "benchmark": "songrecon.bench/v1",
    "measured_on": "2026-11-10",
    "catalog_tracks": 12,
    "positives": 360,
    "accepted": 359,
    "accepted_correct": 359,
    "negatives": 48,
    "negatives_accepted": 0,
    "speed_conditions": (
        "playback-rate corrections 0.94-1.10: 60 of 60 positive "
        "queries accepted with the correct track"
    ),
    "fixture_scope": (
        "synthesized fixture catalog (12 generated tracks, "
        "doc-24 section 9 degradation conditions) - measured on "
        "generated fixtures, not a real music library"
    ),
}

#: SongRecon receipt schema emitted by local and external recognition.
RECEIPT_SCHEMA = "songrecon.receipt/v1"
#: The shipped landmark engine solves exact / near-exact recordings.
MATCH_CLASS_EXACT_RECORDING = "class_1_exact_or_near_exact_recording"
#: What the local ``score`` actually measures. It is the fraction of
#: query landmark hashes whose reference offsets agree on one 50 ms
#: bin. It is not a calibrated probability.
SCORE_NAME = "coherent_offset_fraction"
SCORE_UNIT = "coherent landmark-hash votes per query landmark hash"
SCORE_BASIS = (
    "largest set of query landmark hashes agreeing on one 50 ms "
    "reference-offset bin, divided by all query landmark hashes"
)
LOCAL_RIGHTS_NOTE = (
    "Local catalog scope: Devon's own productions and tracks indexed "
    "from the owner's library."
)
AUDD_RIGHTS_NOTE = (
    "External catalog scope: AudD's provider-managed catalogue, used "
    "only as a fallback; its catalog version is not exposed to Devon."
)


@dataclass(frozen=True)
class RecognitionReceipt:
    """Typed evidence record for one SongRecon answer.

    A local score is carried only as ``match_strength`` with its
    measured name, numerator, denominator, unit, and basis. No field
    upgrades that offset-alignment fraction into a probability.
    """

    status: str
    source: str
    source_label: str
    match_class: str
    schema: str = RECEIPT_SCHEMA
    artist: str = ""
    title: str = ""
    album: str = ""
    release_date: str = ""
    song_link: str = ""
    song_id: str = ""
    recording_id: str = ""
    work_id: str = ""
    catalog_version: str = ""
    catalog_track_count: int | None = None
    catalog_hash_count: int | None = None
    query_hash: str = ""
    query_fingerprint_count: int = 0
    match_strength: dict[str, Any] = field(default_factory=dict)
    evidence_windows: tuple[dict[str, Any], ...] = ()
    alignment: dict[str, Any] = field(default_factory=dict)
    offset_map: dict[str, Any] = field(default_factory=dict)
    feature_sources: tuple[str, ...] = ()
    alternatives: tuple[dict[str, Any], ...] = ()
    attempted_sources: tuple[str, ...] = ()
    local_search: dict[str, Any] = field(default_factory=dict)
    best_candidate: dict[str, Any] = field(default_factory=dict)
    rights_note: str = ""
    no_match_reason: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of the receipt."""
        return {
            "schema": self.schema,
            "status": self.status,
            "source": self.source,
            "source_label": self.source_label,
            "match_class": self.match_class,
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "release_date": self.release_date,
            "song_link": self.song_link,
            "song_id": self.song_id,
            "recording_id": self.recording_id,
            "work_id": self.work_id,
            "catalog_version": self.catalog_version,
            "catalog_track_count": self.catalog_track_count,
            "catalog_hash_count": self.catalog_hash_count,
            "query_hash": self.query_hash,
            "query_fingerprint_count": self.query_fingerprint_count,
            "match_strength": dict(self.match_strength),
            "evidence_windows": [dict(w) for w in self.evidence_windows],
            "alignment": dict(self.alignment),
            "offset_map": dict(self.offset_map),
            "feature_sources": list(self.feature_sources),
            "alternatives": [dict(a) for a in self.alternatives],
            "attempted_sources": list(self.attempted_sources),
            "local_search": dict(self.local_search),
            "best_candidate": dict(self.best_candidate),
            "rights_note": self.rights_note,
            "no_match_reason": self.no_match_reason,
            "reason": self.reason,
        }


def _query_fingerprint_hash(
        hashes: Sequence[tuple[int, float]]) -> str:
    """Stable hash of the query fingerprint, not of the raw audio."""
    if not hashes:
        return ""
    canonical = "\n".join(
        f"{int(h)}:{int(round(float(t) * 1000))}"
        for h, t in sorted(hashes, key=lambda item: (int(item[0]),
                                                     float(item[1]))))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _match_strength(value: float, votes: int,
                    query_count: int) -> dict[str, Any]:
    """Name the local score by the quantity the matcher computed."""
    return {
        "name": SCORE_NAME,
        "value": round(float(value), 3),
        "numerator": int(votes),
        "denominator": int(query_count),
        "unit": SCORE_UNIT,
        "basis": SCORE_BASIS,
        "calibration": "uncalibrated — not a probability",
        "measured_calibration": MEASURED_CALIBRATION,
    }


def external_recognition_receipt(
        result: dict[str, Any], *,
        local_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the SongRecon receipt for an AudD fallback match.

    AudD does not expose a Devon-computed offset score through this
    connector, so the receipt deliberately leaves ``match_strength``
    empty instead of borrowing the local score or inventing one.
    """
    local = dict(local_receipt or {})
    timecode = str(result.get("timecode") or "").strip()
    windows: tuple[dict[str, Any], ...] = ()
    if timecode:
        windows = ({"provider_timecode": timecode},)
    local_search = {
        "status": str(local.get("status") or "not_run"),
        "reason": str(local.get("no_match_reason")
                      or local.get("reason") or ""),
        "catalog_version": str(local.get("catalog_version") or ""),
        "catalog_track_count": local.get("catalog_track_count"),
        "catalog_hash_count": local.get("catalog_hash_count"),
    }
    return RecognitionReceipt(
        status="matched",
        source="audd",
        source_label="AudD external catalogue (fallback)",
        match_class=MATCH_CLASS_EXACT_RECORDING,
        artist=str(result.get("artist") or "Unknown artist"),
        title=str(result.get("title") or "Unknown title"),
        album=str(result.get("album") or ""),
        release_date=str(result.get("release_date") or ""),
        song_link=str(result.get("song_link") or ""),
        catalog_version="audd:provider-managed (version not exposed)",
        query_hash=str(local.get("query_hash") or ""),
        query_fingerprint_count=int(
            local.get("query_fingerprint_count") or 0),
        evidence_windows=windows,
        alignment={"method": "AudD provider recognition",
                   "provider_timecode": timecode,
                   "provider_label": str(result.get("label") or "")},
        offset_map={"provider_timecode": timecode} if timecode else {},
        feature_sources=("audd_recognition_api",),
        attempted_sources=("devon_local_fingerprint", "audd"),
        local_search=local_search,
        rights_note=AUDD_RIGHTS_NOTE,
    ).to_dict()


def no_match_receipt(local_receipt: dict[str, Any] | None, *,
                     reason: str,
                     attempted_sources: Sequence[str]) -> dict[str, Any]:
    """Return the final no-match receipt after every attempted source."""
    receipt = dict(local_receipt or {})
    receipt.update({
        "schema": str(receipt.get("schema") or RECEIPT_SCHEMA),
        "status": "no_match",
        "source": "none",
        "source_label": "No recognition source named the track",
        "match_class": "no_reliable_match",
        "attempted_sources": list(attempted_sources),
        "no_match_reason": reason,
        "reason": reason,
    })
    return receipt

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
    except Exception as exc:  # noqa: BLE001
        record_failure("fingerprint.ffmpeg_path", exc)
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
    except Exception as exc:  # noqa: BLE001
        record_failure("fingerprint.resample_linear", exc)
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
            except OSError as exc:
                record_failure("fingerprint.decode_tmp_cleanup", exc)
            try:
                os.rmdir(tmp)
            except OSError as exc:
                record_failure("fingerprint.decode_tmp_cleanup", exc)
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


def _hashes_from_samples(samples: Sequence[float], sr: int, *,
                         max_hashes: int = _MAX_HASHES,
                         peak_mode: str = "bands"
                         ) -> list[tuple[int, float]]:
    """Constellation hashes from decoded mono samples.

    The sample-domain half of :func:`fingerprint`, split out so the
    matcher's rate retry can re-extract landmarks from a
    rate-corrected waveform (re-picking peaks on the corrected
    spectrum — rescaling packed hash ints instead loses most
    landmarks to quantization error, measured in SR2).
    """
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
    return _hashes_from_samples(samples, sr, max_hashes=max_hashes,
                                peak_mode=peak_mode)


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
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
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


#: The file's DDL, hoisted to a constant for the sidecar port
#: (Mongo residue Batch 3, C7). Byte-identical to the statements
#: :meth:`FingerprintDB._connect` runs — the port's mapping is
#: derived from this, never a second hand-written shape.
_DDL = (
    "CREATE TABLE IF NOT EXISTS tracks ("
    "id INTEGER PRIMARY KEY, title TEXT, artist TEXT, "
    "source TEXT, duration REAL, created REAL);"
    "CREATE TABLE IF NOT EXISTS hashes ("
    "track_id INTEGER, hash INTEGER, offset_ms INTEGER);"
    "CREATE INDEX IF NOT EXISTS idx_hashes_hash ON hashes(hash)"
)

#: Seam collection names for the inverted-index documents (C7).
#: ``hashes`` documents are keyed by hash value, body
#: ``{"occurrences": [[track_id, offset_ms], ...]}`` in native
#: rowid order, duplicates preserved verbatim — the document twin
#: of the native ``idx_hashes_hash`` inverted index. ``track_hashes``
#: documents are keyed by track id, body ``{"pairs":
#: [[hash, offset_ms], ...]}`` in offset order: one get serves
#: :meth:`FingerprintDB.track_hashes` and hands
#: :meth:`FingerprintDB.remove_track` its exact scrub list.
_HASHES_COLLECTION = "hashes"
_TRACK_HASHES_COLLECTION = "track_hashes"


class FingerprintDB:
    """Devon's local recognition memory: fingerprints of her own
    productions, the user's music library, anything indexed via
    :func:`index_own` or ``/audio fingerprint``.

    Matching is time-coherent: a track scores by the largest set of
    hashes that agree on ONE time offset (the Shazam idea) — random
    collisions don't form a coherent line. ``match()`` never raises.

    When a document store is resolved (Mongo program P5 — an explicit
    ``store`` or the boot ambient), the same API runs over the seam
    instead (registered in storage.sidecars as audio-fingerprints):
    ``tracks`` rides its RowTable, ``hashes`` rides the inverted-index
    documents above, and a meta counter keeps ``stats()`` exact
    without scanning. The native sqlite3 path serves the SQLite
    default byte-for-byte. Class separation is structural: these
    collections are ``audio-fingerprints__*`` and never merge with
    ``audio-melodic__works`` (SongRecon class-1 vs class-3).
    """

    def __init__(self, db_path: str = "", *, store: Any = None) -> None:
        self.db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._port: Any = None
        # The port attempt is best-effort by this organ's
        # never-raises posture: a store that cannot be wired
        # degrades to the native file path, loudly journaled.
        # (sidecar_port returns None on the SQLite default, so
        # the native path below is the common case.)
        try:
            from ..storage.sidecars import sidecar_port

            port = sidecar_port(
                "audio-fingerprints", _DDL, ("tracks",),
                None if self.db_path == ":memory:" else self.db_path,
                store=store)
            if port is not None:
                self._import_into(port)
                self._port = port
                return
        except Exception:  # noqa: BLE001 — degraded, not fatal
            _log.warning("fingerprint: seam unavailable, "
                         "running native", exc_info=True)
            self._port = None

    def _import_into(self, port: Any) -> None:
        """One-time lazy import of fingerprints.db into the seam.

        ``tracks`` ports as-is through its RowTable; ``hashes``
        streams in rowid order and groups into the inverted-index
        documents (rowid order preserved inside each occurrence
        list) plus the per-track ``track_hashes`` documents (offset
        order, stable). Counters are seeded from the imported
        maximum track id and total hash rows, so seam-allocated ids
        continue the file's sequence. Runs through ensure_imported:
        one-time, idempotent — a live store is adopted, never
        overwritten.
        """
        max_track = [0]
        total_hashes = [0]

        def write_tracks(rows: list[dict[str, Any]]) -> int:
            table = port.tables["tracks"]
            for row in rows:
                max_track[0] = max(max_track[0], int(row["id"]))
                table.put(row)
            return len(rows)

        def write_hashes(rows: list[dict[str, Any]]) -> int:
            occurrences: dict[int, list[list[int]]] = {}
            per_track: dict[int, list[list[int]]] = {}
            for row in rows:
                tid = int(row["track_id"])
                h = int(row["hash"])
                off = int(row["offset_ms"])
                occurrences.setdefault(h, []).append([tid, off])
                per_track.setdefault(tid, []).append([h, off])
            for h, occ in occurrences.items():
                port.put_raw(_HASHES_COLLECTION, str(h),
                             {"occurrences": occ})
            for tid, pairs in per_track.items():
                pairs.sort(key=lambda pair: pair[1])
                port.put_raw(_TRACK_HASHES_COLLECTION, str(tid),
                             {"pairs": pairs})
            total_hashes[0] = len(rows)
            return len(rows)

        def seed_counters(_counts: dict[str, int]) -> None:
            doc = port.store.get(port.meta_collection, "counters") or {}
            seeded = dict(doc)
            if int(doc.get("tracks") or 0) < max_track[0]:
                seeded["tracks"] = max_track[0]
            if int(doc.get("hashes") or 0) < total_hashes[0]:
                seeded["hashes"] = total_hashes[0]
            if seeded != doc:
                port.store.put(port.meta_collection, "counters", seeded)

        port.ensure_imported(
            {"tracks": write_tracks, "hashes": write_hashes},
            after_import=seed_counters)

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

    # -- seam accessors (C7) ------------------------------------------------
    # Every read the voting code needs goes through one of these, so
    # the native SQL and the inverted-index documents answer the
    # same questions in the same order. All callers hold ``_lock``.

    def _hash_rows(self, h: int) -> list[tuple[int, int]]:
        """Occurrences of one hash, ``[(track_id, offset_ms), ...]``.

        Native: the indexed lookup, rowid order. Seam: one get of
        the inverted-index document, whose list preserves that
        order (and duplicates) verbatim.
        """
        if self._port is not None:
            doc = self._port.store.get(
                "audio-fingerprints__" + _HASHES_COLLECTION, str(h))
            if not doc:
                return []
            return [(int(tid), int(off))
                    for tid, off in doc.get("occurrences") or []]
        rows = self._connect().execute(
            "SELECT track_id, offset_ms FROM hashes WHERE hash = ?",
            (h,)).fetchall()
        return [(int(tid), int(off)) for tid, off in rows]

    def _track_meta(self, tid: int) -> tuple[str, str, str] | None:
        """``(title, artist, source)`` for one track, or None."""
        if self._port is not None:
            row = self._port.tables["tracks"].get(tid)
            if row is None:
                return None
            return (str(row["title"]), str(row["artist"]),
                    str(row["source"]))
        row = self._connect().execute(
            "SELECT title, artist, source FROM tracks WHERE id = ?",
            (tid,)).fetchone()
        if not row:
            return None
        return (row[0], row[1], row[2])

    def _all_track_rows(self) -> list[dict[str, Any]]:
        """Every track row, id order — the catalog snapshot's input."""
        if self._port is not None:
            rows = self._port.tables["tracks"].all()
            rows.sort(key=lambda r: int(r["id"]))
            return rows
        rows = self._connect().execute(
            "SELECT id, title, artist, source, duration, created "
            "FROM tracks ORDER BY id").fetchall()
        return [{"id": r[0], "title": r[1], "artist": r[2],
                 "source": r[3], "duration": r[4], "created": r[5]}
                for r in rows]

    def _hash_total(self) -> int:
        """Total hash rows. Native: COUNT(*). Seam: the meta counter
        (maintained by every seam write, seeded at import) — the
        vote never scans for it."""
        if self._port is not None:
            doc = self._port.store.get(
                self._port.meta_collection, "counters") or {}
            return int(doc.get("hashes") or 0)
        return int(self._connect().execute(
            "SELECT COUNT(*) FROM hashes").fetchone()[0])

    def _seam_write_hashes(self, tid: int,
                           pairs: list[tuple[int, int]]) -> None:
        """Write one track's hash rows into the inverted index.

        Per distinct hash: get the occurrence document, append this
        track's pairs, put. The track_hashes document carries the
        full list in offset order. The meta counter advances by the
        row count. Composition order is chosen so a crash leaves
        extra occurrences for a track rather than a track missing
        hashes — and the voting code already skips occurrences
        whose track row is gone, so crash residue is inert.
        """
        port = self._port
        seen: dict[int, list[list[int]]] = {}
        for h, off in pairs:
            seen.setdefault(h, []).append([tid, off])
        for h, mine in seen.items():
            doc = port.store.get(
                "audio-fingerprints__" + _HASHES_COLLECTION, str(h))
            occ = [list(p) for p in (doc.get("occurrences") or [])] \
                if doc else []
            occ.extend(mine)
            port.put_raw(_HASHES_COLLECTION, str(h),
                         {"occurrences": occ})
        ordered = sorted(pairs, key=lambda pair: pair[1])
        port.put_raw(_TRACK_HASHES_COLLECTION, str(tid),
                     {"pairs": [[h, off] for h, off in ordered]})
        if pairs:
            port.store.inc(port.meta_collection, "counters",
                           "hashes", float(len(pairs)))

    def _seam_drop_hashes(self, tid: int,
                          pairs: list[tuple[int, int]]) -> None:
        """Scrub one track's pairs out of the inverted index (the
        remove_track half). A document emptied of occurrences is
        deleted outright. The meta counter retreats by the row count.
        """
        port = self._port
        for h in {h for h, _off in pairs}:
            doc = port.store.get(
                "audio-fingerprints__" + _HASHES_COLLECTION, str(h))
            if not doc:
                continue
            kept = [p for p in (doc.get("occurrences") or [])
                    if int(p[0]) != tid]
            if kept:
                port.put_raw(_HASHES_COLLECTION, str(h),
                             {"occurrences": kept})
            else:
                port.store.delete(
                    "audio-fingerprints__" + _HASHES_COLLECTION, str(h))
        if pairs:
            port.store.inc(port.meta_collection, "counters",
                           "hashes", float(-len(pairs)))

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
            created = _t.time()
            pairs = [(h, int(t * 1000)) for h, t in hashes]
            with self._lock:
                if self._port is not None:
                    tid = self._port.next_id("tracks")
                    self._port.tables["tracks"].put({
                        "id": tid,
                        "title": title or Path(str(path)).stem,
                        "artist": artist or "unknown",
                        "source": source, "duration": duration,
                        "created": created})
                    self._seam_write_hashes(tid, pairs)
                else:
                    conn = self._connect()
                    cur = conn.execute(
                        "INSERT INTO tracks (title, artist, source, "
                        "duration, created) VALUES (?,?,?,?,?)",
                        (title or Path(str(path)).stem,
                         artist or "unknown", source, duration, created))
                    tid = cur.lastrowid
                    conn.executemany(
                        "INSERT INTO hashes (track_id, hash, offset_ms) "
                        "VALUES (?,?,?)",
                        [(tid, h, off) for h, off in pairs])
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
                if self._port is not None:
                    tracks = self._port.tables["tracks"].count()
                    hashes = self._hash_total()
                else:
                    conn = self._connect()
                    tracks = conn.execute(
                        "SELECT COUNT(*) FROM tracks").fetchone()[0]
                    hashes = conn.execute(
                        "SELECT COUNT(*) FROM hashes").fetchone()[0]
            return {"ok": True, "tracks": tracks, "hashes": hashes}
        except Exception as exc:  # noqa: BLE001
            record_failure("fingerprint.stats", exc)
            return {"ok": False, "tracks": 0, "hashes": 0}

    def _catalog_snapshot_locked(self) -> dict[str, Any]:
        """Catalog stamp for a receipt; caller must hold ``_lock``."""
        rows = self._all_track_rows()
        hash_count = self._hash_total()
        canonical = "\n".join(
            "|".join("" if value is None else str(value)
                     for value in (row["id"], row["title"],
                                   row["artist"], row["source"],
                                   row["duration"], row["created"]))
            for row in rows)
        canonical += f"\nhashes:{int(hash_count)}"
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {
            "version": f"local-fingerprint-v1:sha256:{digest}",
            "tracks": len(rows),
            "hashes": int(hash_count),
        }

    def catalog_snapshot(self) -> dict[str, Any]:
        """Current local catalog size and version stamp. Never raises."""
        try:
            with self._lock:
                return self._catalog_snapshot_locked()
        except Exception as exc:  # noqa: BLE001
            record_failure("fingerprint.catalog_snapshot", exc)
            return {"version": "", "tracks": 0, "hashes": 0}

    def _receipt_locked(
            self, *,
            status: str,
            query_hash: str,
            query_count: int,
            reason: str = "",
            winner: dict[str, Any] | None = None,
            alternatives: Sequence[dict[str, Any]] = (),
            best_candidate: dict[str, Any] | None = None,
            catalog: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
        """Assemble a local receipt; caller must hold ``_lock``."""
        catalog = catalog or self._catalog_snapshot_locked()
        if winner is not None:
            return RecognitionReceipt(
                status="matched",
                source="devon_local_fingerprint",
                source_label="Devon's local fingerprint catalog",
                match_class=MATCH_CLASS_EXACT_RECORDING,
                artist=str(winner.get("artist") or ""),
                title=str(winner.get("title") or ""),
                recording_id=str(winner.get("recording_id") or ""),
                catalog_version=str(catalog["version"]),
                catalog_track_count=int(catalog["tracks"]),
                catalog_hash_count=int(catalog["hashes"]),
                query_hash=query_hash,
                query_fingerprint_count=int(query_count),
                match_strength=dict(winner.get("match_strength") or {}),
                evidence_windows=tuple(
                    winner.get("evidence_windows") or ()),
                alignment=dict(winner.get("alignment") or {}),
                offset_map=dict(winner.get("offset_map") or {}),
                feature_sources=("wang_2003_constellation_landmarks",),
                alternatives=tuple(alternatives),
                attempted_sources=("devon_local_fingerprint",),
                rights_note=LOCAL_RIGHTS_NOTE,
            ).to_dict()
        return RecognitionReceipt(
            status=status,
            source="devon_local_fingerprint",
            source_label="Devon's local fingerprint catalog",
            match_class=("no_reliable_match" if status == "no_match"
                         else "unavailable"),
            catalog_version=str(catalog["version"]),
            catalog_track_count=int(catalog["tracks"]),
            catalog_hash_count=int(catalog["hashes"]),
            query_hash=query_hash,
            query_fingerprint_count=int(query_count),
            feature_sources=("wang_2003_constellation_landmarks",),
            attempted_sources=("devon_local_fingerprint",),
            best_candidate=dict(best_candidate or {}),
            rights_note=LOCAL_RIGHTS_NOTE,
            no_match_reason=reason if status == "no_match" else "",
            reason=reason,
        ).to_dict()

    def _unavailable_receipt(self, *, status: str, reason: str,
                             query_hash: str = "",
                             query_count: int = 0) -> dict[str, Any]:
        """Receipt for a local failure before candidate matching."""
        catalog = self.catalog_snapshot()
        return RecognitionReceipt(
            status=status,
            source="devon_local_fingerprint",
            source_label="Devon's local fingerprint catalog",
            match_class=("no_reliable_match" if status == "no_match"
                         else "unavailable"),
            catalog_version=str(catalog.get("version") or ""),
            catalog_track_count=int(catalog.get("tracks") or 0),
            catalog_hash_count=int(catalog.get("hashes") or 0),
            query_hash=query_hash,
            query_fingerprint_count=int(query_count),
            feature_sources=("wang_2003_constellation_landmarks",),
            attempted_sources=("devon_local_fingerprint",),
            rights_note=LOCAL_RIGHTS_NOTE,
            no_match_reason=reason if status == "no_match" else "",
            reason=reason,
        ).to_dict()

    def _rank_locked(self, hashes: Sequence[tuple[int, float]]
                     ) -> list[dict[str, Any]]:
        """Rank catalog candidates for one query fingerprint.

        Caller must hold ``_lock``. Coherent-offset voting: each
        candidate's score is its largest set of landmark hashes
        agreeing on one 50 ms reference-offset bin, divided by the
        number of query hashes. Returns every candidate that shared
        at least one hash, best first. Total.

        Occurrence lookups go through :meth:`_hash_rows`, so the
        native index and the seam's inverted documents feed the
        identical voting code in the identical order — including
        duplicate query hashes, which re-pull the same occurrences
        (the cache below is per call, keyed by hash value).
        """
        query_types = len({h for h, _t in hashes})
        cand: dict[int, list[tuple[int, int, int, int]]] = {}
        occurrence_cache: dict[int, list[tuple[int, int]]] = {}
        for h, t in hashes:
            if h not in occurrence_cache:
                occurrence_cache[h] = self._hash_rows(h)
            q_ms = int(t * 1000)
            for tid, off in occurrence_cache[h]:
                cand.setdefault(tid, []).append(
                    (int(off) - q_ms, q_ms, int(off), int(h)))
        ranked: list[dict[str, Any]] = []

        def distinct_votes(
                values: list[tuple[int, int, int, int]]) -> int:
            # A query landmark votes ONCE per bin, however many
            # reference copies of its hash agree there: the score
            # is a fraction of query landmark hashes, so its
            # numerator counts distinct (hash, query-time) pairs.
            # Raw entry counting let repetitive material print
            # fractions over 1.0 (measured 1.024, SR2 benchmark).
            return len({(v[3], v[1]) for v in values})

        for tid, evidence in cand.items():
            # Coherent-offset histogram (50 ms bins).
            bins: dict[int, list[tuple[int, int, int, int]]] = {}
            for item in evidence:
                bins.setdefault(int(item[0] // 50),
                                []).append(item)
            if not bins:
                continue
            coherent = max(
                bins.values(),
                key=lambda values: (distinct_votes(values),
                                    -abs(values[0][0])))
            votes = distinct_votes(coherent)
            distinct_types = len({item[3] for item in coherent})
            score = votes / max(1, len(hashes))
            row = self._track_meta(tid)
            if not row:
                continue
            deltas = sorted(item[0] for item in coherent)
            offset_ms = deltas[len(deltas) // 2]
            query_times = [item[1] for item in coherent]
            reference_times = [item[2] for item in coherent]
            window = {
                "query_start_s": round(min(query_times) / 1000, 3),
                "query_end_s": round(max(query_times) / 1000, 3),
                "reference_start_s": round(
                    min(reference_times) / 1000, 3),
                "reference_end_s": round(
                    max(reference_times) / 1000, 3),
                "coherent_votes": votes,
            }
            offset_map = {
                "reference_minus_query_ms": int(offset_ms),
                "query_start_s": window["query_start_s"],
                "reference_start_s": window["reference_start_s"],
            }
            ranked.append({
                "track_id": tid,
                "recording_id": f"local:{tid}",
                "title": row[0],
                "artist": row[1],
                "source": row[2],
                "score": round(score, 3),
                "match_strength": _match_strength(
                    score, votes, len(hashes)),
                "coherent_votes": votes,
                "distinct_types": distinct_types,
                "support_fraction": round(
                    distinct_types / max(1, len(hashes)), 4),
                "type_recall": round(
                    distinct_types / max(1, query_types), 4),
                "offset_ms": int(offset_ms),
                "evidence_windows": [window],
                "alignment": {
                    "method": "Wang-2003 constellation "
                              "landmark offset voting",
                    "bin_width_ms": 50,
                    "coherent_votes": votes,
                    "offset_ms": int(offset_ms),
                },
                "offset_map": offset_map,
            })
        ranked.sort(
            key=lambda item: (
                -float(item["score"]),
                -int(item["coherent_votes"]),
                str(item.get("title") or "").casefold()))
        return ranked

    @staticmethod
    def _clears_acceptance(ranked: list[dict[str, Any]],
                           threshold: float) -> bool:
        """The acceptance tests: coherent-offset fraction at or over
        ``threshold`` AND at least one distinct-landmark breadth test
        (:data:`MIN_SUPPORT_FRACTION` support, or
        :data:`MIN_TYPE_RECALL` of the query's own landmark
        vocabulary explained by the winner)."""
        if not ranked:
            return False
        top = ranked[0]
        if float(top["score"]) < threshold:
            return False
        return (float(top.get("support_fraction") or 0.0)
                >= MIN_SUPPORT_FRACTION
                or float(top.get("type_recall") or 0.0)
                >= MIN_TYPE_RECALL)

    def _rate_retry_locked(self, path: str | os.PathLike[str],
                           threshold: float
                           ) -> tuple[list[dict[str, Any]], float] | None:
        """Retry voting under playback-rate corrections.

        Caller must hold ``_lock``. For each candidate rate the query
        waveform is resampled back toward reference speed and its
        landmarks re-extracted from the corrected spectrum (rescaling
        packed hash ints instead loses most landmarks to quantization
        error — measured in SR2: 0.035 fraction at the exact rate).
        Tries the coarse ±10% grid, then refines ±0.01 around every
        coarse rate — together the two passes probe every hundredth
        from 0.89 to 1.11.
        Returns ``(ranked, rate)`` for the best
        correction that clears strict acceptance, else None. Costs extra
        extraction+voting passes, so it runs only when direct voting
        accepted nothing.
        """
        try:
            samples, sr = read_mono(path)
        except AudioReadError:
            return None
        if len(samples) < sr:
            return None
        # Retry acceptance is stricter than direct: the hypothesis
        # space was searched across rates, so a corrected winner must
        # clear the threshold AND both breadth legs. (Measured: a
        # whole-tone-transposed three-tone chord otherwise "matches"
        # at fraction 1.000 under a rate hypothesis while resting on
        # 42 distinct landmarks — support 0.035. Genuine corrected
        # winners on the benchmark sit at support ≥ 0.12, recall
        # ≥ 0.28.)
        def clears_strict(ranked: list[dict[str, Any]]) -> bool:
            if not ranked:
                return False
            top = ranked[0]
            return (float(top["score"]) >= threshold
                    and float(top.get("support_fraction") or 0.0)
                    >= MIN_SUPPORT_FRACTION
                    and float(top.get("type_recall") or 0.0)
                    >= MIN_TYPE_RECALL)

        best: tuple[float, float, list[dict[str, Any]]] | None = None

        def consider(rate: float) -> None:
            nonlocal best
            corrected = _resample_linear(
                samples, sr, max(1, int(round(sr * rate))))
            hashes = _hashes_from_samples(corrected, _FP_SR)
            if not hashes:
                return
            ranked = self._rank_locked(hashes)
            if not ranked:
                return
            score = float(ranked[0]["score"])
            if clears_strict(ranked) and (
                    best is None or score > best[0]):
                best = (score, rate, ranked)

        for grid_rate in _RATE_RETRY_GRID:
            consider(grid_rate)
        # Refine ±0.01 around EVERY coarse rate. An earlier design
        # refined only around the ≤4 loudest coarse rates at/over
        # the threshold; the SR2 benchmark's final 3 speed misses
        # indicted exactly that gate+cap — each query cleared strict
        # acceptance at its applied rate (scores 0.371–0.493,
        # support 0.126–0.244, recall 0.293–0.657) but the applied
        # rate is an odd hundredth whose even neighbours either
        # heard nothing at/over the threshold or were crowded out
        # of the loudest-4 by chance maxima on wrong tracks. The
        # correct peak is sharp (±0.01 away loses nearly all
        # landmarks), so every hundredth in range gets probed; the
        # cost is bounded extra passes on a path that runs only
        # when direct voting already accepted nothing, and strict
        # acceptance — not the search width — is what guards
        # against false accepts (benchmark negatives verify).
        for centre in _RATE_RETRY_GRID:
            for refine in (centre - 0.01, centre + 0.01):
                rate = round(refine, 2)
                if 0.88 <= rate <= 1.12 and rate not in _RATE_RETRY_GRID:
                    consider(rate)
        if best is not None:
            return best[2], best[1]
        return None

    @staticmethod
    def _apply_rate_to_winner(winner: dict[str, Any],
                              rate: float) -> dict[str, Any]:
        """Annotate a rate-corrected winner: query-side times back in
        query-file seconds, correction named on the alignment (and so
        on the receipt that carries it). Mutates and returns it."""
        for window in winner.get("evidence_windows") or ():
            window["query_start_s"] = round(
                float(window["query_start_s"]) / rate, 3)
            window["query_end_s"] = round(
                float(window["query_end_s"]) / rate, 3)
        offset_map = winner.get("offset_map") or {}
        if offset_map:
            offset_map["query_start_s"] = round(
                float(offset_map["query_start_s"]) / rate, 3)
            offset_map["reference_minus_query_ms"] = int(
                int(offset_map["reference_minus_query_ms"]) / rate)
            offset_map["rate_correction"] = rate
        alignment = winner.get("alignment") or {}
        if alignment:
            alignment["offset_ms"] = int(
                int(alignment["offset_ms"]) / rate)
            alignment["rate_correction"] = rate
        winner["offset_ms"] = int(int(winner["offset_ms"]) / rate)
        winner["rate_correction"] = rate
        return winner

    def rank_candidates(self, path: str | os.PathLike[str], *,
                        limit: int = 5) -> list[dict[str, Any]]:
        """Ranked catalog candidates for ``path``, best first.

        The measurement view over the same voting :meth:`match`
        performs — no threshold applied, so callers (the §9
        benchmark, diagnostics) can see top-5 behaviour and near
        misses. Never raises; returns [] when nothing can be said.
        """
        try:
            hashes = fingerprint(path)
            if not hashes:
                return []
            with self._lock:
                return self._rank_locked(hashes)[:limit]
        except Exception:  # noqa: BLE001
            _log.debug("rank_candidates failed", exc_info=True)
            return []

    def hash_occurrences(
            self, hashes: Sequence[tuple[int, float]]
            ) -> dict[int, list[tuple[int, int]]]:
        """Raw catalog occurrences for a query fingerprint.

        Returns ``{hash: [(track_id, reference_offset_ms), ...]}`` —
        every enrolled occurrence of every distinct query hash. This
        is the evidence substrate for windowed/partial matching
        (doc 24 SR4 lineage): :meth:`match` collapses it into one
        whole-track vote, while a sample is a *locally* coherent
        slice of it. Never raises; returns {} when nothing can be
        said.
        """
        try:
            wanted = {int(h) for h, _t in hashes}
            if not wanted:
                return {}
            out: dict[int, list[tuple[int, int]]] = {}
            with self._lock:
                for h in sorted(wanted):
                    rows = self._hash_rows(h)
                    if rows:
                        out[h] = rows
            return out
        except Exception:  # noqa: BLE001
            _log.debug("hash_occurrences failed", exc_info=True)
            return {}

    def track_hashes(self, track_id: int
                     ) -> list[tuple[int, float]]:
        """The stored fingerprint of one enrolled track:
        ``[(hash, offset_seconds), ...]``, anchor-time order.

        Lets a lineage scan treat an enrolled work as the query —
        lineage records *between* works in the catalog, not only
        between an upload and the catalog. Never raises.
        """
        try:
            with self._lock:
                if self._port is not None:
                    doc = self._port.store.get(
                        "audio-fingerprints__" + _TRACK_HASHES_COLLECTION,
                        str(int(track_id)))
                    if not doc:
                        return []
                    return [(int(h), int(off) / 1000.0)
                            for h, off in doc.get("pairs") or []]
                conn = self._connect()
                rows = conn.execute(
                    "SELECT hash, offset_ms FROM hashes "
                    "WHERE track_id = ? ORDER BY offset_ms",
                    (int(track_id),)).fetchall()
            return [(int(h), int(off) / 1000.0) for h, off in rows]
        except Exception:  # noqa: BLE001
            _log.debug("track_hashes failed", exc_info=True)
            return []

    def match(self, path: str | os.PathLike[str], *,
              threshold: float = MATCH_THRESHOLD
              ) -> dict[str, Any]:
        """Identify ``path`` against the local database. Never raises.

        On an accepted match, returns the backwards-compatible
        ``{"ok", "title", "artist", "score", "track_id"}`` fields plus
        a typed ``receipt``. ``score`` remains as the machine-readable
        coherent-offset fraction; the receipt gives it its full name,
        numerator, denominator, unit, and uncalibrated basis. A
        no-match or failure result also carries a receipt stating what
        was searched and why no recording was accepted.

        Acceptance (SR2): the fraction must clear ``threshold`` AND
        the winner must pass a distinct-landmark breadth test —
        support ≥ :data:`MIN_SUPPORT_FRACTION` of the query's hashes,
        or type recall ≥ :data:`MIN_TYPE_RECALL` of the query's own
        landmark vocabulary. When direct voting accepts nothing, voting is
        retried under playback-rate corrections (coupled speed/pitch
        edits); a winner found that way carries ``rate_correction``
        on the result and on the receipt's alignment.
        """
        query_hash = ""
        query_count = 0
        try:
            hashes = fingerprint(path)
            query_count = len(hashes)
            query_hash = _query_fingerprint_hash(hashes)
            if not hashes:
                reason = ("no fingerprints extracted — audio too quiet "
                          "or too short")
                return {"ok": False, "reason": reason,
                        "receipt": self._unavailable_receipt(
                            status="no_match", reason=reason,
                            query_hash=query_hash,
                            query_count=query_count)}
            with self._lock:
                # Pull candidate evidence per hash. Keeping the query
                # and reference times lets the receipt name the actual
                # windows that voted for the winning offset.
                ranked = self._rank_locked(hashes)
                rate_used: float | None = None
                if not self._clears_acceptance(ranked, threshold):
                    # Coupled speed/pitch edits defeat unscaled
                    # voting by construction; retry rate-corrected
                    # before answering no-match (SR2 hardening).
                    retry = self._rate_retry_locked(path, threshold)
                    if retry is not None:
                        ranked, rate_used = retry
                catalog = self._catalog_snapshot_locked()
                if self._clears_acceptance(ranked, threshold):
                    winner = ranked[0]
                    if rate_used is not None:
                        winner = self._apply_rate_to_winner(
                            winner, rate_used)
                    receipt = self._receipt_locked(
                        status="matched", query_hash=query_hash,
                        query_count=query_count, winner=winner,
                        alternatives=ranked[1:4], catalog=catalog)
                    result = {
                        "ok": True,
                        "track_id": winner["track_id"],
                        "recording_id": winner["recording_id"],
                        "title": winner["title"],
                        "artist": winner["artist"],
                        "source": winner["source"],
                        "score": winner["score"],
                        "match_strength": winner["match_strength"],
                        "evidence_windows": winner["evidence_windows"],
                        "offset_map": winner["offset_map"],
                        "alternatives": ranked[1:4],
                        "catalog": catalog,
                        "catalog_version": catalog["version"],
                        "query_hash": query_hash,
                        "query_fingerprint_count": query_count,
                        "receipt": receipt,
                    }
                    if rate_used is not None:
                        result["rate_correction"] = rate_used
                    return result
                if ranked:
                    best_score = float(ranked[0]["score"])
                    support = float(
                        ranked[0].get("support_fraction") or 0.0)
                    recall = float(
                        ranked[0].get("type_recall") or 0.0)
                    if best_score >= threshold:
                        reason = (
                            "no reliable match in local library "
                            f"(searched {int(catalog['tracks'])} indexed "
                            "track(s); strongest coherent-offset fraction "
                            f"{best_score:.3f} rests on too few distinct "
                            "landmarks — support "
                            f"{support:.3f} < {MIN_SUPPORT_FRACTION:.3f} "
                            f"and type recall {recall:.3f} < "
                            f"{MIN_TYPE_RECALL:.3f}; repeated landmarks "
                            "are not independent evidence)")
                    else:
                        reason = (
                            "no reliable match in local library "
                            f"(searched {int(catalog['tracks'])} indexed "
                            "track(s); strongest coherent-offset fraction "
                            f"{best_score:.3f} < {threshold:.3f})")
                    best_candidate = ranked[0]
                else:
                    best_score = 0.0
                    reason = (
                        "no reliable match in local library "
                        f"(searched {int(catalog['tracks'])} indexed "
                        "track(s); strongest coherent-offset fraction "
                        "0.000 — no candidate shared a coherent "
                        "landmark-offset pattern)")
                    best_candidate = None
                receipt = self._receipt_locked(
                    status="no_match", query_hash=query_hash,
                    query_count=query_count, reason=reason,
                    best_candidate=best_candidate, catalog=catalog)
                return {
                    "ok": False,
                    "reason": reason,
                    "best_score": round(best_score, 3),
                    "best_candidate": best_candidate,
                    "catalog": catalog,
                    "catalog_version": catalog["version"],
                    "query_hash": query_hash,
                    "query_fingerprint_count": query_count,
                    "receipt": receipt,
                }
        except AudioReadError as exc:
            reason = str(exc)
            return {"ok": False, "reason": reason,
                    "receipt": self._unavailable_receipt(
                        status="unavailable", reason=reason,
                        query_hash=query_hash, query_count=query_count)}
        except Exception as exc:  # noqa: BLE001
            _log.debug("match failed", exc_info=True)
            reason = f"local match failed: {exc}"
            return {"ok": False, "reason": reason,
                    "receipt": self._unavailable_receipt(
                        status="unavailable", reason=reason,
                        query_hash=query_hash, query_count=query_count)}

    def list_tracks(self) -> list[dict[str, Any]]:
        """Everything Devon can recognize locally. Never raises."""
        try:
            with self._lock:
                if self._port is not None:
                    rows = self._port.tables["tracks"].all()
                    rows.sort(key=lambda r: float(r["created"] or 0.0),
                              reverse=True)
                    return [{"track_id": int(r["id"]), "title": r["title"],
                             "artist": r["artist"], "source": r["source"],
                             "duration_s": round(float(r["duration"] or 0.0), 1)}
                            for r in rows]
                conn = self._connect()
                rows = conn.execute(
                    "SELECT id, title, artist, source, duration, created "
                    "FROM tracks ORDER BY created DESC").fetchall()
            return [{"track_id": r[0], "title": r[1], "artist": r[2],
                     "source": r[3], "duration_s": round(r[4] or 0.0, 1)}
                    for r in rows]
        except Exception as exc:  # noqa: BLE001
            record_failure("fingerprint.list_tracks", exc)
            return []

    def remove_track(self, track_id: int) -> bool:
        """Drop a track and its hashes. Never raises.

        Seam: the track_hashes document hands back the exact scrub
        list, so no occurrence of a deleted track survives in the
        inverted index; the meta counter retreats by the row count.
        Composition order (occurrences first, track row last) leaves
        a crash with extra occurrences for a track — inert, because
        the voting code already skips occurrences whose track row
        is gone — never a track missing hashes.
        """
        try:
            with self._lock:
                if self._port is not None:
                    doc = self._port.store.get(
                        "audio-fingerprints__" + _TRACK_HASHES_COLLECTION,
                        str(int(track_id)))
                    pairs = [(int(h), int(off))
                             for h, off in (doc.get("pairs") or [])] \
                        if doc else []
                    self._seam_drop_hashes(int(track_id), pairs)
                    self._port.store.delete(
                        "audio-fingerprints__" + _TRACK_HASHES_COLLECTION,
                        str(int(track_id)))
                    return self._port.tables["tracks"].delete(
                        int(track_id))
                conn = self._connect()
                conn.execute("DELETE FROM hashes WHERE track_id = ?",
                             (track_id,))
                cur = conn.execute("DELETE FROM tracks WHERE id = ?",
                                   (track_id,))
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:  # noqa: BLE001
            record_failure("fingerprint.remove_track", exc)
            return False

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception as exc:  # noqa: BLE001
            record_failure("fingerprint.close", exc)
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


def _failed_local_result(reason: str) -> dict[str, Any]:
    """A receipt-carrying local failure result. Never raises."""
    receipt = RecognitionReceipt(
        status="unavailable",
        source="devon_local_fingerprint",
        source_label="Devon's local fingerprint catalog",
        match_class="unavailable",
        feature_sources=("wang_2003_constellation_landmarks",),
        attempted_sources=("devon_local_fingerprint",),
        rights_note=LOCAL_RIGHTS_NOTE,
        reason=reason,
    ).to_dict()
    return {"ok": False, "reason": reason, "receipt": receipt}


def match_local_db(path: str | os.PathLike[str], *,
                   db: FingerprintDB | None = None) -> dict[str, Any]:
    """Match against Devon's local recognition memory. Never raises."""
    try:
        return (db or default_fingerprint_db()).match(path)
    except Exception as exc:  # noqa: BLE001
        return _failed_local_result(f"local match failed: {exc}")


#: Backwards-friendly alias.
match_local = match_local_db


def identify(path: str | os.PathLike[str], *,
             db: FingerprintDB | None = None,
             threshold: float = MATCH_THRESHOLD) -> dict[str, Any]:
    """One call: "what is this audio?" → receipt-carrying result.

    Never raises.
    """
    try:
        return (db or default_fingerprint_db()).match(
            path, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        return _failed_local_result(f"identify failed: {exc}")


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
        return [{**_failed_local_result(
            f"batch match failed: {exc}"), "path": str(p)}
            for p in (paths or [])]


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
