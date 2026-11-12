"""The DJ DESK — the part of the DJ you play.

The picker (dj_engine LearnedTaste) decides WHAT to play. The desk is
HOW it is played: two decks with play/pause, jog, scratch, pitch bend,
tempo fader + key lock and sync; a mixer with 3-band EQ, channel
faders and an equal-power crossfader; effects (filter, echo, reverb,
flanger); a beat grid with downbeat, named hot cues, a sample-pad
bank, beat-quantized loops; a library of crates with search; per-track
waveforms; recording of a live set; and a live-set runner that plays a
show THROUGH the desk on real rendered audio. Real songs come through
the music chain (fetch -> analyze -> decks); composed tracks are the
fallback input.

A grid that is wrong says so (confidence / unknown), never fakes
beats. A cue jump lands exactly on a grid beat. Pads quantize to the
grid. Loops are exact beats. Everything here is DSP (code rung) except
the two learned rungs registered on the ladder: the madmom RNN beat
tracker (``dj_beat_grid`` model rung) and the cue suggester
(``dj_cue_suggest`` model rung).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import struct
import subprocess
import sys
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dj_engine import TrackAnalysis, _frame_rms, _read_mono, analyze_track
from .dj_mixdrop import fetch_track_audio
from .failures import record_failure
from .vocal_lite import _resample_linear

_log = logging.getLogger(__name__)

SR = 22050
_FRAME, _HOP = 1024, 512
_MIN_RATE, _MAX_RATE = 0.92, 1.08  # tempo fader: +/-8%
_LOOP_LENGTHS = (1, 2, 4, 8)


# ── beat grid ───────────────────────────────────────────────────────────────

@dataclass
class BeatGrid:
    """Beat times for one track, anchored to its detected BPM.

    ``known`` is False (empty beats, confidence 0) when the grid could
    not be found — a wrong grid says so, never fakes beats.
    """

    bpm: float | None = None
    beat_times: list[float] = field(default_factory=list)
    downbeat_index: int | None = None
    confidence: float = 0.0
    source: str = "unknown"  # "onset-grid" | "madmom" | "unknown"

    @property
    def known(self) -> bool:
        return bool(self.beat_times) and self.confidence > 0.0

    @property
    def beat_s(self) -> float | None:
        return 60.0 / self.bpm if self.bpm else None

    def nearest_beat(self, t: float) -> float | None:
        if not self.beat_times:
            return None
        return min(self.beat_times, key=lambda b: abs(b - t))

    def next_beat(self, t: float) -> float | None:
        """First beat strictly after ``t`` (a trigger on a beat waits
        for the next one — the beat that just sounded is gone). The
        5ms epsilon treats a trigger landing within a hair of a beat
        as ON that beat: the grid itself carries ~0.1ms of anchor
        error, and firing on the beat that just sounded is the wrong
        musical answer."""
        for b in self.beat_times:
            if b > t + 0.005:
                return b
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"bpm": self.bpm,
                "beat_times": [round(b, 4) for b in self.beat_times],
                "downbeat_index": self.downbeat_index,
                "confidence": round(self.confidence, 3),
                "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeatGrid:
        try:
            return cls(bpm=data.get("bpm"),
                       beat_times=[float(b)
                                   for b in data.get("beat_times", [])],
                       downbeat_index=data.get("downbeat_index"),
                       confidence=float(data.get("confidence", 0.0)),
                       source=data.get("source", "unknown"))
        except Exception:  # noqa: BLE001 - a bad stored grid is unknown
            return cls()


def _onset_flux(samples: array, sr: int) -> list[float]:
    """Positive energy flux between consecutive frames (as in
    dj_engine.detect_bpm)."""
    rms = _frame_rms(samples, _FRAME, _HOP)
    if len(rms) < 2:
        return []
    return [max(0.0, rms[i] - rms[i - 1]) for i in range(1, len(rms))]


def detect_beat_grid(samples: array, sr: int,
                     bpm: float | None) -> BeatGrid:
    """Beat times from the onset grid anchored to ``bpm``.

    Phase-search over the flux envelope: the offset whose grid frames
    carry the most onset energy wins; the offset is then refined to
    sample level by the median peak offset (a click's loudest sample
    IS its beat). Confidence is the share of beats with real onset
    energy. Unknown BPM or no onsets -> unknown grid. Never raises.
    """
    grid = BeatGrid(bpm=bpm)
    try:
        if not bpm or bpm <= 0 or len(samples) < sr * 2:
            return grid
        flux = _onset_flux(samples, sr)
        if len(flux) < 32:
            return grid
        # flux[j] compares frame j+1 with frame j, so its natural time
        # is the START of frame j+1 — prepend a zero so index j maps
        # to time j/fps. Without this the grid sits ~one frame early
        # for every beat.
        flux = [0.0] + flux
        fps = sr / _HOP
        period = fps * 60.0 / bpm  # frames per beat
        if period < 2:
            return grid
        duration = len(samples) / sr
        n_beats = int(duration / (60.0 / bpm))
        if n_beats < 4:
            return grid
        # phase search: score each frame offset by its grid flux
        best_off, best_score = 0.0, -1.0
        step = 1.0
        off = 0.0
        while off < period:
            score = 0.0
            k = 0
            while True:
                idx = int(round(off + k * period))
                if idx >= len(flux):
                    break
                score += flux[idx]
                k += 1
            if k and score / k > best_score:
                best_score, best_off = score / k, off
            off += step
        if best_score <= 1e-9:
            return grid
        # parabolic refine of the frame offset
        off0 = best_off
        if 0 < best_off < period - 1:
            def _score(o: float) -> float:
                s, k = 0.0, 0
                while True:
                    idx = int(round(o + k * period))
                    if idx >= len(flux):
                        break
                    s += flux[idx]
                    k += 1
                return s / max(k, 1)
            y0, y1, y2 = _score(best_off - 1), best_score, _score(best_off + 1)
            denom = (y0 - 2 * y1 + y2)
            if denom:
                shift = 0.5 * (y0 - y2) / denom
                if abs(shift) <= 1.0:
                    off0 = best_off + shift
        off_s = off0 / fps
        beat_s = 60.0 / bpm
        beats = [off_s + k * beat_s for k in range(n_beats + 1)]
        beats = [b for b in beats if b < duration]
        if len(beats) < 4:
            return grid
        # sample-level refine: the ONSET of each beat, not its peak.
        # Walk back from the loudest sample in +-half a beat to where
        # the attack crosses 10% of the peak — a click's loudest sample
        # sits a quarter-period after its onset, so the peak itself is
        # a biased anchor. The median across beats kills stray peaks.
        # Capped: a grid that needs a bigger shift is wrong, and stays
        # frame-anchored.
        half = beat_s / 2.0
        deltas: list[float] = []
        for b in beats:
            lo = max(0, int((b - half) * sr))
            hi = min(len(samples), int((b + half) * sr) + 1)
            if hi <= lo:
                continue
            peak = max(range(lo, hi), key=lambda i: abs(samples[i]))
            peak_val = abs(samples[peak])
            if peak_val <= 1e-9:
                continue
            floor = 0.1 * peak_val
            back_stop = max(lo, peak - int(0.05 * sr))
            onset = back_stop
            for i in range(peak, back_stop - 1, -1):
                if abs(samples[i]) < floor:
                    onset = i + 1
                    break
            deltas.append(onset / sr - b)
        if deltas:
            deltas.sort()
            shift = deltas[len(deltas) // 2]
            if abs(shift) <= 0.15:
                beats = [round(b + shift, 4) for b in beats]
        beats = [b for b in beats if 0.0 <= b < duration]
        if len(beats) < 4:
            return grid
        # the grid is a periodic lattice: extend it to cover [0,
        # duration) at the beat spacing, even when the first onset is
        # invisible to the flux (a click at t=0 has no rising edge).
        while beats and beats[0] - beat_s >= -0.001:
            beats.insert(0, round(beats[0] - beat_s, 4))
        while beats and beats[-1] + beat_s < duration:
            beats.append(round(beats[-1] + beat_s, 4))
        # confidence: share of beats with a real onset NEAR them — the
        # max flux within -2..+1 frames, since the first frame
        # containing an onset starts up to a frame before it, so the
        # flux peak sits 1-2 frames EARLY in index terms.
        grid_flux = []
        for b in beats:
            idx = int(round(b * fps))
            lo = max(0, idx - 2)
            hi = min(len(flux), idx + 2)
            grid_flux.append(max(flux[lo:hi]) if hi > lo else 0.0)
        peak_flux = max(grid_flux) if grid_flux else 0.0
        if peak_flux <= 1e-9:
            return grid
        matched = sum(1 for g in grid_flux if g >= 0.25 * peak_flux)
        grid.beat_times = beats
        grid.confidence = round(matched / max(len(beats), 1), 3)
        grid.source = "onset-grid"
        return grid
    except Exception as exc:  # noqa: BLE001 - a failed grid is unknown
        record_failure("dj_desk.detect_beat_grid", exc)
        return BeatGrid(bpm=bpm)


def detect_downbeat(samples: array, sr: int,
                    grid: BeatGrid) -> tuple[int | None, float]:
    """Which beat of the bar is the downbeat, from the energy pattern.

    Scores each candidate offset 0..3 by the mean energy of the bars
    starting on it. The winner needs a real margin, or the downbeat is
    unknown (a uniform click track has no downbeat to find — and the
    desk says so instead of faking one). Returns (index, confidence).
    Never raises.
    """
    try:
        if not grid.known or grid.beat_s is None or len(grid.beat_times) < 8:
            return None, 0.0
        beat_s = grid.beat_s
        # sample-level RMS per bar window: frame windows LEAK the next
        # beat's onset (a frame starting just before it contains it),
        # which makes every candidate look accented. Sample-exact
        # windows keep each bar's energy its own.
        scores: list[float] = []
        for d in range(4):
            bar_energies: list[float] = []
            for b in range(d, len(grid.beat_times) - 3, 4):
                t0 = grid.beat_times[b]
                s0 = int(t0 * sr)
                s1 = min(len(samples), int((t0 + beat_s) * sr))
                if s1 > s0:
                    ss = 0.0
                    for i in range(s0, s1):
                        v = samples[i]
                        ss += v * v
                    bar_energies.append(math.sqrt(ss / (s1 - s0)))
            scores.append(sum(bar_energies) / max(len(bar_energies), 1)
                          if bar_energies else 0.0)
        best = max(range(4), key=lambda i: scores[i])
        ranked = sorted(scores, reverse=True)
        if ranked[0] <= 1e-9:
            return None, 0.0
        margin = (ranked[0] - ranked[1]) / ranked[0]
        if margin < 0.15:
            return None, round(margin, 3)
        return best, round(margin, 3)
    except Exception as exc:  # noqa: BLE001 - an unknown downbeat is fine
        record_failure("dj_desk.detect_downbeat", exc)
        return None, 0.0


def analyze_grid(path: str, analysis: TrackAnalysis) -> BeatGrid:
    """Grid + downbeat for one analyzed track. Never raises."""
    try:
        samples, sr = _read_mono(path)
        bpm = analysis.bpm
        if bpm is None:
            from .dj_engine import detect_bpm
            bpm = detect_bpm(samples, sr)
        grid = detect_beat_grid(samples, sr, bpm)
        if grid.known:
            idx, _conf = detect_downbeat(samples, sr, grid)
            grid.downbeat_index = idx
        return grid
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.analyze_grid", exc)
        return BeatGrid(bpm=analysis.bpm)


# ── hot cues ────────────────────────────────────────────────────────────────

@dataclass
class CueBank:
    """Named cue points for one track: beat index -> exact grid time.

    A cue IS a grid beat — set_cue refuses an out-of-range beat or an
    unknown grid, so a jump can never land between beats.
    """

    cues: dict[str, tuple[int, float]] = field(default_factory=dict)

    def set_cue(self, name: str, beat_index: int, grid: BeatGrid) -> float:
        if not grid.known:
            raise ValueError("cannot set a cue without a beat grid")
        if beat_index < 0 or beat_index >= len(grid.beat_times):
            raise ValueError(
                f"cue beat {beat_index} out of range "
                f"(grid has {len(grid.beat_times)} beats)")
        t = grid.beat_times[beat_index]
        self.cues[str(name)] = (int(beat_index), float(t))
        return t

    def jump(self, name: str) -> float:
        return self.cues[str(name)][1]

    def delete(self, name: str) -> None:
        self.cues.pop(str(name), None)

    def list_cues(self) -> list[tuple[str, int, float]]:
        return sorted((n, b, t) for n, (b, t) in self.cues.items())

    def to_dict(self) -> dict[str, Any]:
        return {n: {"beat": b, "time": t}
                for n, (b, t) in self.cues.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CueBank:
        bank = cls()
        try:
            for name, val in (data or {}).items():
                bank.cues[str(name)] = (int(val["beat"]),
                                        float(val["time"]))
        except Exception:  # noqa: BLE001 - a bad stored bank is empty
            return cls()
        return bank


# ── persistence (runtime JSON store — never migrations.py) ───────────────────

def _desk_store_path() -> Path:
    override = (os.environ.get("NM_DJ_DESK_PATH") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return (Path(os.path.expanduser("~")) / ".cache" / "nomorals" / "dj"
            / "desk.json")


class DeskStore:
    """Per-track grids + cue banks + the record library, in one JSON
    file. Keyed by track id (path or fingerprint — the caller picks).
    Atomic tmp+replace writes; a corrupt file loads empty, never raises.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _desk_store_path()
        self.tracks: dict[str, dict[str, Any]] = {}

    def put_track(self, track_id: str, grid: BeatGrid,
                  bank: CueBank,
                  waveform: list[float] | None = None) -> None:
        entry: dict[str, Any] = {"grid": grid.to_dict(),
                                 "cues": bank.to_dict()}
        if waveform is not None:
            entry["waveform"] = [round(float(v), 3) for v in waveform]
        self.tracks[str(track_id)] = entry

    def get_track(self, track_id: str
                  ) -> tuple[BeatGrid | None, CueBank | None]:
        entry = self.tracks.get(str(track_id))
        if not isinstance(entry, dict):
            return None, None
        return (BeatGrid.from_dict(entry.get("grid", {})),
                CueBank.from_dict(entry.get("cues", {})))

    def get_waveform(self, track_id: str) -> list[float]:
        entry = self.tracks.get(str(track_id))
        if not isinstance(entry, dict):
            return []
        wf = entry.get("waveform", [])
        return [float(v) for v in wf] if isinstance(wf, list) else []

    # ── library (crates + track records, same file, same atomic write) ──
    @property
    def library(self) -> dict[str, Any]:
        lib = self.__dict__.setdefault(
            "_library", {"tracks": {}, "crates": {}})
        if not isinstance(lib, dict):
            lib = {"tracks": {}, "crates": {}}
            self.__dict__["_library"] = lib
        lib.setdefault("tracks", {})
        lib.setdefault("crates", {})
        return lib

    def load(self) -> None:
        try:
            if self.path.is_file():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.tracks = data.get("tracks", {}) or {}
                    lib = data.get("library")
                    if isinstance(lib, dict):
                        self.__dict__["_library"] = {
                            "tracks": lib.get("tracks", {}) or {},
                            "crates": lib.get("crates", {}) or {}}
        except Exception as exc:  # noqa: BLE001 - corrupt store is empty
            record_failure("dj_desk.DeskStore.load", exc)
            self.tracks = {}
            self.__dict__["_library"] = {"tracks": {}, "crates": {}}

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"tracks": self.tracks, "library": self.library},
                indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("dj_desk.DeskStore.save", exc)
            return False


# ── pitch-preserving stretch (key lock) ─────────────────────────────────────

def _wsola_stretch(seg: array, rate: float) -> array:
    """Pitch-preserving time stretch (WSOLA): play ``rate``x faster
    without the pitch riding the fader.

    Overlap-add of Hann-windowed frames, each frame's analysis offset
    chosen by cross-correlation against the overlap already written,
    so transients stay aligned. Pure Python — slow on long audio
    (roughly a second of compute per second of audio), so key lock is
    an explicit opt-in; short segments fall back to the plain
    resample. Never raises.
    """
    n = len(seg)
    try:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if abs(rate - 1.0) < 1e-6 or n < 1024:
            return _resample_linear(seg, SR, int(SR / rate))
        W, Ha = 512, 128
        Hs = max(1, int(round(Ha / rate)))
        win = [0.5 - 0.5 * math.cos(2 * math.pi * i / (W - 1))
               for i in range(W)]
        out_len = max(1, int(n / rate))
        out = array("d", [0.0]) * out_len
        norm = array("d", [0.0]) * out_len
        pos = 0.0
        out_pos = 0
        tol = 64
        while out_pos < out_len:
            base = int(round(pos))
            best_off = 0
            o1 = min(out_len, out_pos + Ha)
            if out_pos > 0 and o1 > out_pos:
                span = o1 - out_pos
                target = out[out_pos:o1]
                t_norm = math.sqrt(sum(v * v for v in target)) or 1.0

                def _score(off: int, base: int = base, span: int = span,
                           target: array = target,
                           t_norm: float = t_norm) -> float | None:
                    idx = base + off
                    if idx < 0 or idx + span > n:
                        return None
                    num = 0.0
                    den = 0.0
                    for k in range(0, span, 4):
                        c = seg[idx + k]
                        num += c * target[k]
                        den += c * c
                    return num / (math.sqrt(den) * t_norm or 1.0)

                best_score: float | None = None
                for off in range(-tol, tol + 1, 4):
                    s = _score(off)
                    if s is not None and (best_score is None
                                          or s > best_score):
                        best_score, best_off = s, off
                for off in range(best_off - 3, best_off + 4):
                    s = _score(off)
                    if s is not None and (best_score is None
                                          or s > best_score):
                        best_score, best_off = s, off
            idx = base + best_off
            flen = min(W, n - idx, out_len - out_pos)
            for i in range(flen):
                w = win[i]
                out[out_pos + i] += seg[idx + i] * w
                norm[out_pos + i] += w
            pos += Ha
            out_pos += Hs
        for i in range(out_len):
            if norm[i] > 1e-9:
                out[i] /= norm[i]
        return out
    except Exception as exc:  # noqa: BLE001 - a failed stretch is a resample
        record_failure("dj_desk._wsola_stretch", exc)
        return _resample_linear(seg, SR, int(SR / max(rate, 1e-6)))


def _biquad_coeffs(kind: str, cutoff: float, q: float,
                   sr: int) -> tuple[float, float, float, float, float]:
    """RBJ biquad coefficients for a lowpass/highpass/bandpass filter."""
    w0 = 2 * math.pi * cutoff / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    if kind == "lowpass":
        b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
    elif kind == "highpass":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
    elif kind == "bandpass":
        # constant 0 dB peak gain
        b0, b1, b2 = alpha, 0.0, -alpha
    else:
        raise ValueError(f"unknown biquad kind {kind!r}")
    a0 = 1 + alpha
    a1, a2 = -2 * cw, 1 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_run(seg: array, coeffs: tuple[float, ...],
                state: list[float]) -> tuple[array, list[float]]:
    """One biquad pass; `state` is [x1, x2, y1, y2] and comes back
    updated, so the filter rings correctly across render chunks."""
    b0, b1, b2, a1, a2 = coeffs
    x1, x2, y1, y2 = state
    out = array("d", [0.0]) * len(seg)
    for i, x in enumerate(seg):
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, x, y1, y
        out[i] = y
    return out, [x1, x2, y1, y2]


# ── effects ─────────────────────────────────────────────────────────────────

@dataclass
class FXChain:
    """One deck's effect chain: filter, echo, reverb, flanger.

    All four are real DSP with state held on the deck (see _FXState);
    everything defaults OFF — a flat deck sounds exactly like no FX.
    """

    filter_mode: str = "off"  # off | lowpass | highpass
    filter_cutoff: float = 8000.0
    filter_q: float = 0.707
    echo_on: bool = False
    echo_beats: float = 1.0
    echo_feedback: float = 0.4
    echo_mix: float = 0.5
    reverb_on: bool = False
    reverb_mix: float = 0.5
    flanger_on: bool = False
    flanger_mix: float = 0.5
    flanger_rate_hz: float = 0.25

    def set_filter(self, mode: str, cutoff: float = 8000.0,
                   q: float = 0.707) -> str:
        mode = str(mode).lower()
        if mode not in ("off", "lowpass", "highpass"):
            raise ValueError(
                f"filter mode must be off/lowpass/highpass, got {mode!r}")
        cutoff = float(cutoff)
        if not 20.0 <= cutoff <= 18000.0:
            raise ValueError(
                f"filter cutoff must be 20..18000 Hz, got {cutoff}")
        q = float(q)
        if not 0.1 <= q <= 10.0:
            raise ValueError(f"filter Q must be 0.1..10, got {q}")
        self.filter_mode = mode
        self.filter_cutoff = cutoff
        self.filter_q = q
        return (f"filter {mode}"
                + (f" {cutoff:g} Hz" if mode != "off" else ""))

    def set_echo(self, on: bool = True, beats: float = 1.0,
                 feedback: float = 0.4, mix: float = 0.5) -> str:
        self.echo_on = bool(on)
        self.echo_beats = float(beats)
        if not 0.25 <= self.echo_beats <= 8.0:
            raise ValueError(
                f"echo beats must be 0.25..8, got {self.echo_beats}")
        self.echo_feedback = float(feedback)
        if not 0.0 <= self.echo_feedback <= 0.95:
            raise ValueError(
                f"echo feedback must be 0..0.95, got {self.echo_feedback}")
        self.echo_mix = float(mix)
        if not 0.0 <= self.echo_mix <= 1.0:
            raise ValueError(f"echo mix must be 0..1, got {self.echo_mix}")
        return ("echo on"
                if self.echo_on else "echo off")

    def set_reverb(self, on: bool = True, mix: float = 0.5) -> str:
        self.reverb_on = bool(on)
        self.reverb_mix = float(mix)
        if not 0.0 <= self.reverb_mix <= 1.0:
            raise ValueError(
                f"reverb mix must be 0..1, got {self.reverb_mix}")
        return "reverb on" if self.reverb_on else "reverb off"

    def set_flanger(self, on: bool = True, mix: float = 0.5,
                    rate_hz: float = 0.25) -> str:
        self.flanger_on = bool(on)
        self.flanger_mix = float(mix)
        if not 0.0 <= self.flanger_mix <= 1.0:
            raise ValueError(
                f"flanger mix must be 0..1, got {self.flanger_mix}")
        self.flanger_rate_hz = float(rate_hz)
        if not 0.05 <= self.flanger_rate_hz <= 5.0:
            raise ValueError(
                f"flanger rate must be 0.05..5 Hz, got "
                f"{self.flanger_rate_hz}")
        return "flanger on" if self.flanger_on else "flanger off"

    def off(self, name: str | None = None) -> str:
        if name is None:
            self.filter_mode = "off"
            self.echo_on = self.reverb_on = self.flanger_on = False
            return "all FX off"
        name = str(name).lower()
        if name == "filter":
            self.filter_mode = "off"
        elif name == "echo":
            self.echo_on = False
        elif name == "reverb":
            self.reverb_on = False
        elif name == "flanger":
            self.flanger_on = False
        else:
            raise ValueError(f"unknown FX {name!r}")
        return f"{name} off"

    def summary(self) -> str:
        parts: list[str] = []
        if self.filter_mode != "off":
            parts.append(f"filter {self.filter_mode} "
                         f"{self.filter_cutoff:g}Hz")
        if self.echo_on:
            parts.append(f"echo {self.echo_beats:g}b")
        if self.reverb_on:
            parts.append("reverb")
        if self.flanger_on:
            parts.append("flanger")
        return " · ".join(parts) if parts else "fx off"


#: Freeverb-style tunings, halved for the desk's 22050 Hz rate.
_COMB_DELAYS = (558, 594, 638, 678)
_AP_DELAYS = (278, 220)


@dataclass
class _FXState:
    """Per-deck effect state: biquad memory, delay lines, LFO phase.

    Reset on every load() so a fresh track never inherits the last
    one's ringing tail.
    """

    bq: list[float] = field(default_factory=lambda: [0.0] * 4)
    bq_key: tuple = ()
    bq_coeffs: tuple[float, ...] = ()
    echo_buf: array = field(default_factory=lambda: array("d"))
    echo_pos: int = 0
    echo_delay: int = 0
    combs: list = field(default_factory=list)  # [[buf, pos], ...]
    aps: list = field(default_factory=list)
    comb_sr: int = 0
    fl_buf: array = field(default_factory=lambda: array("d"))
    fl_pos: int = 0
    fl_phase: float = 0.0

    def reset(self) -> None:
        self.bq = [0.0] * 4
        self.bq_key = ()
        self.bq_coeffs = ()
        self.echo_buf = array("d")
        self.echo_pos = 0
        self.echo_delay = 0
        self.combs = []
        self.aps = []
        self.comb_sr = 0
        self.fl_buf = array("d")
        self.fl_pos = 0
        self.fl_phase = 0.0


# ── waveforms ───────────────────────────────────────────────────────────────

def compute_waveform(samples: array, sr: int,
                     columns: int = 200) -> list[float]:
    """Overview waveform: RMS per column, normalized to the peak
    (0..1). Silence is all zeros — never fakes peaks. Never raises."""
    try:
        if not samples or columns <= 0:
            return [0.0] * max(columns, 0)
        n = len(samples)
        per = n / columns
        cols: list[float] = []
        peak = 0.0
        for c in range(columns):
            s0 = int(c * per)
            s1 = max(s0 + 1, int((c + 1) * per))
            seg = samples[s0:s1]
            rms = math.sqrt(sum(v * v for v in seg) / (s1 - s0))
            cols.append(rms)
            peak = max(peak, rms)
        if peak <= 1e-12:
            return [0.0] * columns
        return [round(v / peak, 3) for v in cols]
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.compute_waveform", exc)
        return [0.0] * max(columns, 0)


# ── library: crates + search ────────────────────────────────────────────────

class Library:
    """The desk's record library: tracks fetched or composed, in
    named crates, searchable by title/artist/BPM/key.

    Persisted in the DeskStore JSON (same atomic write as grids and
    cues — never migrations.py). A track id is the fetch path for a
    real track, or the caller-chosen id for a composed one.
    """

    def __init__(self, store: DeskStore) -> None:
        self.store = store

    def add_track(self, track_id: str, *, title: str, artist: str = "",
                  bpm: float | None = None, key: str = "",
                  mode: str = "", source: str = "", path: str = "",
                  duration_s: float = 0.0) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "track_id": str(track_id),
            "title": str(title or ""),
            "artist": str(artist or ""),
            "bpm": float(bpm) if bpm else None,
            "key": str(key or ""),
            "mode": str(mode or ""),
            "source": str(source or ""),
            "path": str(path or track_id),
            "duration_s": round(float(duration_s or 0.0), 2),
            "added_ts": time.time()}
        self.store.library["tracks"][str(track_id)] = rec
        return rec

    def get_track(self, track_id: str) -> dict[str, Any] | None:
        rec = self.store.library["tracks"].get(str(track_id))
        return dict(rec) if isinstance(rec, dict) else None

    def search(self, query: str = "", *, bpm_min: float | None = None,
               bpm_max: float | None = None,
               key: str = "") -> list[dict[str, Any]]:
        q = (query or "").strip().lower()
        want_key = (key or "").strip().lower()
        out: list[dict[str, Any]] = []
        for rec in self.store.library["tracks"].values():
            if not isinstance(rec, dict):
                continue
            if q and q not in (rec.get("title", "") + " "
                               + rec.get("artist", "")).lower():
                continue
            bpm = rec.get("bpm")
            if bpm_min is not None and (bpm is None or bpm < bpm_min):
                continue
            if bpm_max is not None and (bpm is None or bpm > bpm_max):
                continue
            if want_key and (rec.get("key", "") or "").lower() != want_key:
                continue
            out.append(dict(rec))
        out.sort(key=lambda r: (r.get("title", ""), r.get("artist", "")))
        return out

    def create_crate(self, name: str) -> bool:
        name = str(name or "").strip()
        if not name:
            raise ValueError("crate name must not be empty")
        crates = self.store.library["crates"]
        if name in crates:
            return False
        crates[name] = []
        return True

    def crates(self) -> list[str]:
        return sorted(self.store.library["crates"].keys())

    def add_to_crate(self, crate: str, track_id: str) -> bool:
        crates = self.store.library["crates"]
        if crate not in crates:
            return False
        if str(track_id) not in self.store.library["tracks"]:
            return False
        ids = crates[crate]
        if str(track_id) not in ids:
            ids.append(str(track_id))
        return True

    def remove_from_crate(self, crate: str, track_id: str) -> bool:
        crates = self.store.library["crates"]
        if crate not in crates:
            return False
        ids = crates[crate]
        if str(track_id) not in ids:
            return False
        ids.remove(str(track_id))
        return True

    def crate_tracks(self, crate: str) -> list[str]:
        ids = self.store.library["crates"].get(str(crate))
        return list(ids) if isinstance(ids, list) else []

    def delete_crate(self, name: str) -> bool:
        return self.store.library["crates"].pop(str(name), None) is not None

    def load_deck(self, desk: DJDesk, which: str,
                  track_id: str) -> Deck:
        """Load a library track onto a deck: the record's BPM/key are
        the analysis (they were measured when the track was fetched),
        the audio is read from the record's path. Raises ValueError
        when the track is unknown or unreadable."""
        rec = self.get_track(track_id)
        if rec is None:
            raise ValueError(f"no library track {track_id!r}")
        path = rec.get("path") or str(track_id)
        try:
            samples, sr = _read_mono(path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot read {path!r}: {exc}") from exc
        if not samples:
            raise ValueError(f"cannot read {path!r}: empty audio")
        if sr != SR:
            samples = _resample_linear(samples, sr, SR)
            sr = SR
        analysis = TrackAnalysis(
            path=path, title=rec.get("title", ""),
            bpm=rec.get("bpm"), key=rec.get("key", ""),
            mode=rec.get("mode", ""), duration_s=rec.get("duration_s", 0.0)
            or len(samples) / SR,
            bpm_source="detected" if rec.get("bpm") else "unknown")
        return desk.load_deck(which, analysis, samples, sr)


# ── sample pads ─────────────────────────────────────────────────────────────

def _noise(n: int, seed: int) -> list[float]:
    """Deterministic white noise (LCG) — pads sound the same every run."""
    out: list[float] = []
    x = seed
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(x / 0x7FFFFFFF * 2.0 - 1.0)
    return out


@dataclass
class PadEvent:
    pad: str
    scheduled_s: float  # next grid beat >= trigger time (or raw time)
    samples: array


class PadBank:
    """One-shot sample pads. Five synthesized built-ins + three custom
    slots loaded from a WAV file. A trigger quantizes to the grid.
    """

    slots = ["kick", "snare", "clap", "hat", "airhorn",
             "custom1", "custom2", "custom3"]
    builtins = ("kick", "snare", "clap", "hat", "airhorn")

    def __init__(self) -> None:
        self.custom: dict[str, array] = {}

    def load_custom(self, slot: str, path: str) -> bool:
        """Load a WAV into a custom slot. Returns False when unreadable."""
        if slot not in ("custom1", "custom2", "custom3"):
            raise ValueError(f"{slot!r} is not a custom slot")
        samples, _sr = _read_mono(path)
        if not samples:
            return False
        self.custom[slot] = samples
        return True

    def sample(self, pad: str, sr: int = SR) -> array:
        if pad in self.custom:
            return array("d", self.custom[pad])
        if pad not in self.builtins:
            raise ValueError(f"unknown pad {pad!r}")
        return _synthesize_pad(pad, sr)

    def trigger(self, pad: str, at_s: float,
                grid: BeatGrid | None) -> PadEvent:
        """Quantize the trigger to the next grid beat (or fire raw when
        there is no grid — the event says which)."""
        if pad not in self.slots and pad not in self.custom:
            raise ValueError(f"unknown pad {pad!r}")
        when = at_s
        if grid is not None and grid.known:
            nxt = grid.next_beat(at_s)
            if nxt is not None:
                when = nxt
        return PadEvent(pad=pad, scheduled_s=when,
                        samples=self.sample(pad, SR))


def _synthesize_pad(name: str, sr: int) -> array:
    """One-shot drum/horn synthesis, pure Python. Each pad has a
    distinct spectral signature (the tests read it back with Goertzel).
    """
    if name == "kick":
        n = int(sr * 0.25)
        out = array("d", [0.0]) * n
        phase = 0.0
        for i in range(n):
            t = i / sr
            f = 160.0 * math.exp(-t * 18.0) + 45.0
            phase += 2 * math.pi * f / sr
            out[i] = math.sin(phase) * math.exp(-t * 22.0)
        return out
    if name == "snare":
        n = int(sr * 0.18)
        nz = _noise(n, 7)
        out = array("d", [0.0]) * n
        for i in range(n):
            t = i / sr
            tone = math.sin(2 * math.pi * 190.0 * t)
            out[i] = (0.7 * nz[i] + 0.5 * tone) * math.exp(-t * 30.0)
        return out
    if name == "clap":
        n = int(sr * 0.25)
        nz = _noise(n, 21)
        out = array("d", [0.0]) * n
        for burst in (0.0, 0.02, 0.04):
            start = int(burst * sr)
            for i in range(start, min(n, start + int(0.03 * sr))):
                out[i] += nz[i] * math.exp(-(i - start) / sr * 90.0)
        for i in range(n):
            out[i] *= math.exp(-i / sr * 14.0)
        return out
    if name == "hat":
        n = int(sr * 0.06)
        nz = _noise(n, 99)
        out = array("d", [0.0]) * n
        prev = 0.0
        for i in range(n):
            # difference filter: highpassed noise
            cur = nz[i] - prev
            prev = nz[i]
            out[i] = cur * math.exp(-i / sr * 120.0)
        return out
    if name == "airhorn":
        n = int(sr * 0.7)
        out = array("d", [0.0]) * n
        ph1 = ph2 = 0.0
        for i in range(n):
            t = i / sr
            ph1 += 2 * math.pi * 440.0 / sr
            ph2 += 2 * math.pi * 466.16 / sr
            saw = ((ph1 % (2 * math.pi)) / math.pi - 1.0)
            saw2 = ((ph2 % (2 * math.pi)) / math.pi - 1.0)
            env = min(1.0, t / 0.02) * (0.6 + 0.4 * math.exp(-t * 2.0))
            out[i] = 0.5 * (saw + saw2) * env
        return out
    raise ValueError(f"unknown built-in pad {name!r}")


# ── decks ───────────────────────────────────────────────────────────────────

@dataclass
class Deck:
    """One deck: a loaded track, a playhead, tempo, cue bank, pad bank,
    a beat-quantized loop, a 3-band EQ, an FX chain, key lock, jog /
    scratch / pitch bend — and a waveform. render() advances the
    playhead and returns real audio for the window — pads, loops, EQ
    and FX are IN the render, not a side narration.
    """

    name: str
    analysis: TrackAnalysis | None = None
    samples: array = field(default_factory=lambda: array("d"))
    sr: int = SR
    position_s: float = 0.0
    playing: bool = False
    rate: float = 1.0  # tempo fader, 0.92..1.08
    gain: float = 1.0  # trim
    fader: float = 1.0  # channel fader, 0..1 (applied by the desk mix)
    key_lock: bool = False  # tempo without pitch (WSOLA stretch)
    eq_low: float = 1.0
    eq_mid: float = 1.0
    eq_high: float = 1.0
    quantize: bool = True  # pads snap to the grid; off fires raw
    fx: FXChain = field(default_factory=FXChain)
    waveform: list[float] = field(default_factory=list)
    grid: BeatGrid = field(default_factory=BeatGrid)
    cue_bank: CueBank = field(default_factory=CueBank)
    pad_bank: PadBank = field(default_factory=PadBank)
    # loop state
    loop_active: bool = False
    loop_in_s: float = 0.0
    loop_len_s: float = 0.0
    loop_roll: bool = False
    ghost_s: float = 0.0  # where the playhead WOULD be without the loop
    _pad_events: list[PadEvent] = field(default_factory=list)
    # 3-band EQ state: each band is TWO cascaded biquads (4th order)
    # so a cut band is really gone. (The first version derived mid by
    # subtraction, x - low - high — the lowpass's phase shift leaked
    # ~40% of a 100 Hz tone into "mid", so cutting low left the bass
    # audible. Three real filters, no subtraction.)
    _eq_lp1: list[float] = field(default_factory=lambda: [0.0] * 4,
                                 repr=False)
    _eq_lp2: list[float] = field(default_factory=lambda: [0.0] * 4,
                                 repr=False)
    _eq_bp1: list[float] = field(default_factory=lambda: [0.0] * 4,
                                 repr=False)
    _eq_bp2: list[float] = field(default_factory=lambda: [0.0] * 4,
                                 repr=False)
    _eq_hp1: list[float] = field(default_factory=lambda: [0.0] * 4,
                                 repr=False)
    _eq_hp2: list[float] = field(default_factory=lambda: [0.0] * 4,
                                 repr=False)
    _eq_lp_c: tuple[float, ...] = field(default=(), repr=False)
    _eq_bp_c: tuple[float, ...] = field(default=(), repr=False)
    _eq_hp_c: tuple[float, ...] = field(default=(), repr=False)
    _bend_factor: float = field(default=1.0, repr=False)
    _bend_beats_left: float = field(default=0.0, repr=False)
    _fx_state: _FXState = field(default_factory=_FXState, repr=False)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / max(1, self.sr)

    @property
    def bpm(self) -> float | None:
        return self.analysis.bpm if self.analysis else None

    @property
    def effective_bpm(self) -> float | None:
        if self.bpm is None:
            return None
        return round(self.bpm * self.rate, 2)

    def position_beats(self) -> float:
        if self.bpm is None:
            return 0.0
        return self.position_s * self.bpm / 60.0

    def load(self, analysis: TrackAnalysis, samples: array,
             sr: int = SR) -> None:
        self.analysis = analysis
        self.samples = array("d", samples)
        self.sr = sr
        self.position_s = 0.0
        self.ghost_s = 0.0
        self.playing = False
        self.rate = 1.0
        self.loop_active = False
        self._pad_events = []
        self._eq_lp1 = [0.0] * 4
        self._eq_lp2 = [0.0] * 4
        self._eq_bp1 = [0.0] * 4
        self._eq_bp2 = [0.0] * 4
        self._eq_hp1 = [0.0] * 4
        self._eq_hp2 = [0.0] * 4
        self._eq_lp_c = _biquad_coeffs("lowpass", 220.0, 0.707, sr)
        self._eq_bp_c = _biquad_coeffs("bandpass", 938.0, 0.5, sr)
        self._eq_hp_c = _biquad_coeffs("highpass", 4000.0, 0.707, sr)
        self._bend_factor = 1.0
        self._bend_beats_left = 0.0
        self._fx_state.reset()
        self.waveform = compute_waveform(samples, sr)
        self.grid = detect_beat_grid(samples, sr, analysis.bpm)
        if self.grid.known:
            idx, _c = detect_downbeat(samples, sr, self.grid)
            self.grid.downbeat_index = idx
        self.cue_bank = CueBank()

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def set_rate(self, rate: float) -> float:
        self.rate = max(_MIN_RATE, min(_MAX_RATE, float(rate)))
        return self.rate

    def sync_to(self, other: Deck) -> bool:
        """Match the other deck's effective BPM. False when out of the
        fader's range — the rate is left untouched, not faked."""
        if self.bpm is None or other.effective_bpm is None:
            return False
        need = other.effective_bpm / self.bpm
        if _MIN_RATE <= need <= _MAX_RATE:
            self.rate = need
            return True
        return False

    def jump_to_beat(self, beat_index: int) -> float:
        if not self.grid.known:
            raise ValueError("no beat grid to jump on")
        if beat_index < 0 or beat_index >= len(self.grid.beat_times):
            raise ValueError(f"beat {beat_index} out of range")
        self.position_s = self.grid.beat_times[beat_index]
        self.ghost_s = self.position_s
        return self.position_s

    def jump_to_cue(self, name: str) -> float:
        t = self.cue_bank.jump(name)
        self.position_s = t
        self.ghost_s = t
        return t

    def set_loop(self, in_beat: int, length_beats: int,
                 roll: bool = False) -> None:
        if length_beats not in _LOOP_LENGTHS:
            raise ValueError(
                f"loop length must be one of {_LOOP_LENGTHS}, "
                f"got {length_beats}")
        if not self.grid.known:
            raise ValueError("no beat grid to loop on")
        if in_beat < 0 or in_beat >= len(self.grid.beat_times):
            raise ValueError(f"loop in-beat {in_beat} out of range")
        beat_s = 60.0 / self.bpm
        self.loop_in_s = self.grid.beat_times[in_beat]
        self.loop_len_s = length_beats * beat_s
        self.loop_roll = roll
        self.loop_active = True
        self.ghost_s = self.position_s

    def release_loop(self) -> None:
        if not self.loop_active:
            return
        if self.loop_roll:
            # the ghost kept moving: resume where the track would be
            self.position_s = self.ghost_s
        else:
            # plain loop: continue from the end of the loop
            self.position_s = self.loop_in_s + self.loop_len_s
        self.loop_active = False

    def trigger_pad(self, pad: str) -> PadEvent:
        """Trigger now, quantized to the next grid beat after the
        current playhead (raw when quantize is off or no grid)."""
        grid = self.grid if self.quantize else None
        ev = self.pad_bank.trigger(pad, self.position_s, grid)
        self._pad_events.append(ev)
        return ev

    def queued_pads(self) -> list[tuple[str, float, array]]:
        """Pending pad/scratch hits: (pad, scheduled_s, samples)."""
        return [(ev.pad, ev.scheduled_s, ev.samples)
                for ev in self._pad_events]

    # ── jog / beat jump / pitch bend / scratch ────────────────────────

    def jog(self, beats: float) -> float:
        """Nudge the playhead by `beats` (the pitch-bend wheel's
        cousin, for beat-matching by ear). Clamped to the track."""
        if not self.samples:
            raise ValueError("deck has no track loaded to jog")
        if self.bpm is None:
            raise ValueError("no BPM to jog in beats")
        self.position_s = max(
            0.0, min(self.duration_s,
                     self.position_s + beats * 60.0 / self.bpm))
        self.ghost_s = self.position_s
        return self.position_s

    def beat_jump(self, n_beats: int) -> float:
        """Jump `n_beats` from the nearest grid beat (a pro deck's
        beat jump: forward/back by whole beats, always on the grid)."""
        if not self.grid.known:
            raise ValueError("no beat grid to jump on")
        nearest = self.grid.nearest_beat(self.position_s)
        if nearest is None:
            raise ValueError("no beat grid to jump on")
        idx = self.grid.beat_times.index(nearest) + int(n_beats)
        idx = max(0, min(len(self.grid.beat_times) - 1, idx))
        self.position_s = self.grid.beat_times[idx]
        self.ghost_s = self.position_s
        return self.position_s

    def bend(self, factor: float, beats: float) -> float:
        """Temporary pitch bend: play `factor`x fast for the next
        `beats` beats, then the rate returns on its own. (The nudge
        a DJ holds while two tracks drift apart.)"""
        factor = float(factor)
        if not 0.5 <= factor <= 2.0:
            raise ValueError(f"bend factor must be 0.5..2.0, got {factor}")
        beats = float(beats)
        if beats <= 0:
            raise ValueError("bend beats must be > 0")
        if self.bpm is None:
            raise ValueError("no BPM to bend in beats")
        self._bend_factor = factor
        self._bend_beats_left = beats
        return factor

    def scratch(self, delta_s: float) -> float:
        """Move the playhead by `delta_s` and queue the scratch burst:
        the 0.12s of audio at the new position — forward when the
        platter moves forward, REVERSED when it moves back, so the
        scratch is audible in the render. Returns the new position."""
        if not self.samples:
            raise ValueError("deck has no track loaded to scratch")
        new_pos = max(0.0, min(self.duration_s,
                               self.position_s + float(delta_s)))
        burst_n = max(1, int(0.12 * self.sr))
        end_i = int(new_pos * self.sr)
        if delta_s < 0:
            start_i = max(0, end_i - burst_n)
            burst = array("d", reversed(self.samples[start_i:end_i]))
        else:
            burst = array("d", self.samples[end_i:end_i + burst_n])
        self.position_s = new_pos
        self.ghost_s = new_pos
        self._pad_events.append(
            PadEvent(pad="__scratch__", scheduled_s=new_pos,
                     samples=burst))
        return new_pos

    # ── mixer: EQ + channel fader ─────────────────────────────────────

    def set_eq(self, band: str, value: float) -> float:
        """3-band EQ, linear gain 0..2 per band (0 = cut, 1 = flat,
        2 = +6 dB). Raises on a bad band or range — never clamps a
        wrong value into silence."""
        band = str(band).lower()
        if band not in ("low", "mid", "high"):
            raise ValueError(
                f"EQ band must be low/mid/high, got {band!r}")
        value = float(value)
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"EQ {band} must be 0..2, got {value}")
        setattr(self, f"eq_{band}", value)
        return value

    def set_fader(self, value: float) -> float:
        """Channel fader 0..1 (clamped — a fader is a continuous
        control, unlike the EQ which refuses a wrong value)."""
        self.fader = max(0.0, min(1.0, float(value)))
        return self.fader

    # ── effects ───────────────────────────────────────────────────────

    def set_fx(self, kind: str, **kw: Any) -> str:
        """Route one FX change to the chain. kind is filter / echo /
        reverb / flanger / off; the rest are the chain's own params
        (mode, cutoff, q, on, beats, feedback, mix, rate_hz)."""
        kind = str(kind).lower()
        if kind == "filter":
            return self.fx.set_filter(kw.get("mode", "lowpass"),
                                      kw.get("cutoff", 8000.0),
                                      kw.get("q", 0.707))
        if kind == "echo":
            return self.fx.set_echo(kw.get("on", True),
                                    kw.get("beats", 1.0),
                                    kw.get("feedback", 0.4),
                                    kw.get("mix", 0.5))
        if kind == "reverb":
            return self.fx.set_reverb(kw.get("on", True),
                                      kw.get("mix", 0.5))
        if kind == "flanger":
            return self.fx.set_flanger(kw.get("on", True),
                                       kw.get("mix", 0.5),
                                       kw.get("rate_hz", 0.25))
        if kind == "off":
            return self.fx.off(kw.get("name"))
        raise ValueError(f"unknown FX {kind!r}")

    def _apply_eq(self, seg: array) -> array:
        # Three real filters (low/band/high), each cascaded twice —
        # never a subtraction, so a cut band is really gone and flat
        # sums to ~1 across the spectrum.
        if (self.eq_low == 1.0 and self.eq_mid == 1.0
                and self.eq_high == 1.0):
            return seg
        if not self._eq_lp_c:
            self._eq_lp_c = _biquad_coeffs("lowpass", 220.0, 0.707,
                                           self.sr)
            self._eq_bp_c = _biquad_coeffs("bandpass", 938.0, 0.5,
                                           self.sr)
            self._eq_hp_c = _biquad_coeffs("highpass", 4000.0, 0.707,
                                           self.sr)
        low, self._eq_lp1 = _biquad_run(seg, self._eq_lp_c,
                                        self._eq_lp1)
        low, self._eq_lp2 = _biquad_run(low, self._eq_lp_c,
                                        self._eq_lp2)
        mid, self._eq_bp1 = _biquad_run(seg, self._eq_bp_c,
                                        self._eq_bp1)
        mid, self._eq_bp2 = _biquad_run(mid, self._eq_bp_c,
                                        self._eq_bp2)
        high, self._eq_hp1 = _biquad_run(seg, self._eq_hp_c,
                                         self._eq_hp1)
        high, self._eq_hp2 = _biquad_run(high, self._eq_hp_c,
                                         self._eq_hp2)
        out = array("d", [0.0]) * len(seg)
        for i in range(len(seg)):
            out[i] = (low[i] * self.eq_low + mid[i] * self.eq_mid
                      + high[i] * self.eq_high)
        return out

    def _biquad(self, seg: array, kind: str, cutoff: float,
                q: float) -> array:
        st = self._fx_state
        key = (kind, cutoff, q, self.sr)
        if st.bq_key != key:
            st.bq_coeffs = _biquad_coeffs(kind, cutoff, q, self.sr)
            st.bq_key = key
            st.bq = [0.0] * 4
        out, st.bq = _biquad_run(seg, st.bq_coeffs, st.bq)
        return out

    def _echo(self, seg: array, delay_s: float, feedback: float,
              mix: float) -> array:
        st = self._fx_state
        d = max(1, int(delay_s * self.sr))
        if st.echo_delay != d or len(st.echo_buf) != d:
            st.echo_buf = array("d", [0.0]) * d
            st.echo_pos = 0
            st.echo_delay = d
        buf = st.echo_buf
        pos = st.echo_pos
        out = array("d", [0.0]) * len(seg)
        for i, x in enumerate(seg):
            y = buf[pos]
            buf[pos] = x + feedback * y
            out[i] = x * (1.0 - mix) + y * mix
            pos += 1
            if pos >= d:
                pos = 0
        st.echo_pos = pos
        return out

    def _reverb(self, seg: array, mix: float) -> array:
        st = self._fx_state
        if not st.combs or st.comb_sr != self.sr:
            scale = self.sr / SR
            st.combs = [[array("d", [0.0])
                         * max(1, int(d * scale)), 0]
                        for d in _COMB_DELAYS]
            st.aps = [[array("d", [0.0]) * max(1, int(d * scale)), 0]
                      for d in _AP_DELAYS]
            st.comb_sr = self.sr
        out = array("d", [0.0]) * len(seg)
        for i, x in enumerate(seg):
            verb = 0.0
            for pair in st.combs:
                buf, pos = pair
                y = buf[pos]
                buf[pos] = x + 0.84 * y
                pair[1] = pos + 1 if pos + 1 < len(buf) else 0
                verb += y
            verb /= 4.0
            for pair in st.aps:
                buf, pos = pair
                y = buf[pos]
                buf[pos] = verb + 0.5 * y
                pair[1] = pos + 1 if pos + 1 < len(buf) else 0
                verb = -0.5 * verb + y
            out[i] = x * (1.0 - mix) + verb * mix
        return out

    def _flanger(self, seg: array, mix: float, rate_hz: float) -> array:
        st = self._fx_state
        max_d = int(0.006 * self.sr) + 2
        if len(st.fl_buf) != max_d:
            st.fl_buf = array("d", [0.0]) * max_d
            st.fl_pos = 0
        buf = st.fl_buf
        pos = st.fl_pos
        phase = st.fl_phase
        out = array("d", [0.0]) * len(seg)
        for i, x in enumerate(seg):
            lfo = math.sin(2 * math.pi * phase)
            ds = (0.002 + 0.002 * lfo) * self.sr
            read = pos - ds
            fl = math.floor(read)
            r0 = int(fl) % max_d
            r1 = (r0 + 1) % max_d
            frac = read - fl
            delayed = buf[r0] * (1.0 - frac) + buf[r1] * frac
            buf[pos] = x + 0.3 * delayed
            out[i] = x * (1.0 - mix) + delayed * mix
            pos = (pos + 1) % max_d
            phase += rate_hz / self.sr
            if phase >= 1.0:
                phase -= 1.0
        st.fl_pos, st.fl_phase = pos, phase
        return out

    def _apply_fx(self, seg: array) -> array:
        fx = self.fx
        if (fx.filter_mode == "off" and not fx.echo_on
                and not fx.reverb_on and not fx.flanger_on):
            return seg
        out = seg
        if fx.filter_mode in ("lowpass", "highpass"):
            out = self._biquad(out, fx.filter_mode, fx.filter_cutoff,
                               fx.filter_q)
        if fx.echo_on:
            beat_s = 60.0 / self.bpm if self.bpm else 0.5
            out = self._echo(out, fx.echo_beats * beat_s,
                             fx.echo_feedback, fx.echo_mix)
        if fx.reverb_on:
            out = self._reverb(out, fx.reverb_mix)
        if fx.flanger_on:
            out = self._flanger(out, fx.flanger_mix,
                                fx.flanger_rate_hz)
        return out

    def render(self, duration_s: float, gain: float = 1.0) -> array:
        """Real audio for the next `duration_s` of set time.

        Consumes duration_s * rate seconds of source (tempo fader,
        times any active pitch bend), wraps the playhead inside an
        active loop, advances the ghost, and mixes in any pad hits
        whose quantized time falls in the window. Key lock stretches
        pitch-preserving (WSOLA) instead of resampling. EQ and FX are
        applied to the track audio (pads stay dry). Silence when
        paused or at the end of the track.
        """
        n = max(1, int(duration_s * self.sr))
        out = array("d", [0.0]) * n
        if not self.playing or not self.samples:
            return out
        loop_end = self.loop_in_s + self.loop_len_s
        cursor = 0
        remaining = duration_s
        beat_s = 60.0 / self.bpm if self.bpm else 0.0
        while remaining > 1e-6 and cursor < n:
            eff_rate = self.rate
            if self._bend_beats_left > 0:
                eff_rate = self.rate * self._bend_factor
            pos = self.position_s
            if self.loop_active and pos >= loop_end - 1e-9:
                pos = self.loop_in_s
                self.position_s = pos
            chunk_src = remaining * eff_rate
            if self.loop_active:
                chunk_src = min(chunk_src, loop_end - pos)
            chunk_src = min(chunk_src, self.duration_s - pos)
            if chunk_src <= 1e-6:
                break
            chunk_out_s = chunk_src / eff_rate
            start_i = int(pos * self.sr)
            end_i = min(len(self.samples), int((pos + chunk_src) * self.sr))
            seg = array("d", self.samples[start_i:end_i])
            if eff_rate != 1.0:
                if self.key_lock:
                    seg = _wsola_stretch(seg, eff_rate)
                else:
                    seg = _resample_linear(seg, self.sr,
                                           int(self.sr / eff_rate))
            seg = self._apply_eq(seg)
            seg = self._apply_fx(seg)
            # pads firing inside this chunk
            chunk_end = pos + chunk_src
            for ev in list(self._pad_events):
                if pos - 1e-9 <= ev.scheduled_s < chunk_end:
                    off = cursor + int((ev.scheduled_s - pos)
                                       / eff_rate * self.sr)
                    for i, v in enumerate(ev.samples):
                        j = off + i
                        if 0 <= j < n:
                            out[j] += v * gain
                    self._pad_events.remove(ev)
            for i, v in enumerate(seg):
                j = cursor + i
                if j >= n:
                    break
                out[j] += v * self.gain * gain
            self.position_s = pos + chunk_src
            if self.loop_active:
                self.ghost_s += chunk_src
            else:
                self.ghost_s = self.position_s
            if self._bend_beats_left > 0 and beat_s > 0:
                self._bend_beats_left -= chunk_src / beat_s
                if self._bend_beats_left <= 0:
                    self._bend_factor = 1.0
                    self._bend_beats_left = 0.0
            cursor += len(seg)
            remaining -= chunk_out_s
        # drop pad events the playhead jumped past
        self._pad_events = [e for e in self._pad_events
                            if e.scheduled_s >= self.position_s - 1e-9]
        return out


# ── the desk ────────────────────────────────────────────────────────────────

class DJDesk:
    """Two decks, a mixer (3-band EQ per deck, channel faders,
    equal-power crossfader), and the live-set runner's render core.
    record_start/record_stop capture a live session to WAV."""

    def __init__(self) -> None:
        self.deck_a = Deck("A")
        self.deck_b = Deck("B")
        self._xf = 0.0  # 0 = A only, 1 = B only
        self.master_gain = 1.0
        self._recording_on = False
        self._recording: list[array] = []

    def deck(self, which: str) -> Deck:
        return self.deck_a if str(which).upper() == "A" else self.deck_b

    def load_deck(self, which: str, analysis: TrackAnalysis,
                  samples: array, sr: int = SR) -> Deck:
        d = self.deck(which)
        d.load(analysis, samples, sr)
        return d

    def crossfader(self, x: float) -> float:
        self._xf = max(0.0, min(1.0, float(x)))
        return self._xf

    def record_start(self) -> bool:
        """Start capturing the desk mix. False when already recording."""
        if self._recording_on:
            return False
        self._recording_on = True
        self._recording = []
        return True

    @property
    def recording(self) -> bool:
        return self._recording_on

    def record_stop(self, path: str) -> dict[str, Any]:
        """Stop capturing and write the mix to a WAV. The recording is
        the live counterpart of run_live_set's scripted render."""
        if not self._recording_on:
            return {"ok": False,
                    "reason": "not recording — "
                              "/dj desk record start first"}
        self._recording_on = False
        mix = array("d")
        for chunk in self._recording:
            mix.extend(chunk)
        self._recording = []
        if not mix:
            return {"ok": False, "reason": "recording is empty"}
        out_path = _write_mono_wav(path, mix, SR)
        if not out_path:
            return {"ok": False, "reason": "could not write recording"}
        return {"ok": True, "path": out_path,
                "duration_s": round(len(mix) / SR, 2)}

    def render(self, duration_s: float) -> array:
        """Equal-power crossfade mix of both decks for the window,
        with each deck's channel fader applied. Captured when
        recording."""
        a = self.deck_a.render(duration_s)
        b = self.deck_b.render(duration_s)
        ga = math.cos(self._xf * math.pi / 2.0) * self.deck_a.fader
        gb = math.sin(self._xf * math.pi / 2.0) * self.deck_b.fader
        n = max(len(a), len(b))
        out = array("d", [0.0]) * n
        for i in range(n):
            va = a[i] if i < len(a) else 0.0
            vb = b[i] if i < len(b) else 0.0
            v = (va * ga + vb * gb) * self.master_gain
            out[i] = max(-1.0, min(1.0, v))
        if self._recording_on:
            self._recording.append(array("d", out))
        return out

    def status(self) -> dict[str, Any]:
        def _deck(d: Deck) -> dict[str, Any]:
            return {"name": d.name, "bpm": d.bpm,
                    "effective_bpm": d.effective_bpm,
                    "position_s": round(d.position_s, 2),
                    "playing": d.playing,
                    "grid_beats": len(d.grid.beat_times),
                    "grid_confidence": d.grid.confidence,
                    "downbeat": d.grid.downbeat_index,
                    "cues": [c[0] for c in d.cue_bank.list_cues()],
                    "loop": (d.loop_active, d.loop_len_s),
                    "pads": d.pad_bank.slots,
                    "eq": (d.eq_low, d.eq_mid, d.eq_high),
                    "fader": d.fader,
                    "key_lock": d.key_lock,
                    "fx": d.fx.summary(),
                    "waveform": len(d.waveform)}
        return {"crossfader": round(self._xf, 2),
                "recording": self._recording_on,
                "deck_a": _deck(self.deck_a),
                "deck_b": _deck(self.deck_b)}


def _write_mono_wav(path: str, samples: array, sr: int = SR) -> str:
    try:
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(struct.pack(
                f"<{len(samples)}h",
                *[max(-32768, min(32767, int(s * 32767)))
                  for s in samples]))
        return path
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk._write_mono_wav", exc)
        return ""


def run_live_set(tracks: list[tuple[str, TrackAnalysis, array]],
                 script: list[dict[str, Any]], out_path: str,
                 duration_s: float | None = None) -> dict[str, Any]:
    """Play a scripted set THROUGH the desk and render it to a WAV.

    ``tracks`` is [(track_id, analysis, samples), ...] — track 0 on
    deck A, track 1 (if any) on deck B. ``script`` is a list of actions
    {"at_s", "deck", "kind", ...}:
    cue (set + jump to a beat), pad (quantized trigger), loop
    (beat-quantized in/length, roll optional), loop_off, xfade (ramp
    to a value over N beats), play, pause, jog (nudge by beats), bend
    (temporary pitch bend), eq (band + value), fx (filter/echo/reverb/
    flanger), fader, keylock. ``duration_s`` bounds the set (a looped
    deck never ends on its own). Returns the render report; the WAV
    is the proof, not a narration.
    """
    try:
        if not tracks:
            return {"ok": False, "reason": "no tracks for the set"}
        desk = DJDesk()
        desk.load_deck("A", tracks[0][1], tracks[0][2])
        desk.deck_a.play()
        if len(tracks) > 1:
            desk.load_deck("B", tracks[1][1], tracks[1][2])
        total = duration_s
        if total is None:
            total = max(t[1].duration_s for t in tracks)
        total = max(1.0, float(total))
        actions = sorted(script or [], key=lambda a: float(a.get("at_s", 0)))
        applied = 0
        out = array("d")
        t = 0.0
        step = 0.25
        xf_ramp: tuple[float, float, float, float] | None = None
        while t < total - 1e-9:
            chunk = min(step, total - t)
            # actions due at this set time fire before the chunk renders
            while actions and float(actions[0].get("at_s", 0)) <= t + 1e-9:
                act = actions.pop(0)
                deck = desk.deck(act.get("deck", "A"))
                kind = str(act.get("kind", ""))
                try:
                    if kind == "cue":
                        beat = int(act.get("beat", 0))
                        name = str(act.get("name", "cue"))
                        deck.cue_bank.set_cue(name, beat, deck.grid)
                        deck.jump_to_cue(name)
                    elif kind == "pad":
                        deck.trigger_pad(str(act.get("pad", "kick")))
                    elif kind == "loop":
                        deck.set_loop(int(act.get("in_beat", 0)),
                                      int(act.get("length_beats", 4)),
                                      roll=bool(act.get("roll", False)))
                    elif kind == "loop_off":
                        deck.release_loop()
                    elif kind == "xfade":
                        to = float(act.get("to", 1.0))
                        beats = float(act.get("beats", 4) or 4)
                        b_s = 60.0 / (deck.bpm or 120.0)
                        xf_ramp = (desk._xf, to, t, beats * b_s)
                        # start the other deck when the fade starts
                        other = (desk.deck_b if deck is desk.deck_a
                                 else desk.deck_a)
                        other.play()
                    elif kind == "play":
                        deck.play()
                    elif kind == "pause":
                        deck.pause()
                    elif kind == "jog":
                        deck.jog(float(act.get("beats", 0)))
                    elif kind == "bend":
                        deck.bend(float(act.get("factor", 1.0)),
                                  float(act.get("beats", 4)))
                    elif kind == "eq":
                        deck.set_eq(str(act.get("band", "low")),
                                    float(act.get("value", 1.0)))
                    elif kind == "fx":
                        deck.set_fx(
                            str(act.get("fx", "off")),
                            mode=act.get("mode", "lowpass"),
                            cutoff=float(act.get("cutoff", 8000.0)),
                            on=bool(act.get("on", True)),
                            beats=float(act.get("beats", 1.0)),
                            feedback=float(act.get("feedback", 0.4)),
                            mix=float(act.get("mix", 0.5)),
                            rate_hz=float(act.get("rate_hz", 0.25)))
                    elif kind == "fader":
                        deck.set_fader(float(act.get("value", 1.0)))
                    elif kind == "keylock":
                        deck.key_lock = bool(act.get("on", True))
                    applied += 1
                except Exception as exc:  # noqa: BLE001 - one bad
                    # action never kills the set; it is reported
                    record_failure("dj_desk.run_live_set.action", exc)
            if xf_ramp is not None:
                frm, to, t0, dur = xf_ramp
                frac = (t - t0) / dur if dur > 0 else 1.0
                if frac >= 1.0:
                    desk.crossfader(to)
                    xf_ramp = None
                else:
                    desk.crossfader(frm + (to - frm) * frac)
            out.extend(desk.render(chunk))
            t += chunk
        path = _write_mono_wav(out_path, out, SR)
        if not path:
            return {"ok": False, "reason": "could not write set WAV"}
        return {"ok": True, "path": path,
                "duration_s": round(len(out) / SR, 2),
                "actions_applied": applied,
                "tracks": [t[0] for t in tracks]}
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.run_live_set", exc)
        return {"ok": False, "reason": str(exc)}


# ── cue suggester (learned rung) ────────────────────────────────────────────

#: Bar features the cue model reads. One row per bar.
_CUE_FEATURES = ("energy", "energy_delta", "onset_density", "position_frac")


def _bar_features(samples: array, sr: int, bpm: float) -> list[dict[str, float]]:
    """One feature row per 4-beat bar: energy, jump vs the previous
    bar, onset density, and position in the track."""
    try:
        if not bpm or bpm <= 0 or len(samples) < sr:
            return []
        beat_s = 60.0 / bpm
        bar_s = beat_s * 4
        n_bars = max(1, int(len(samples) / sr / bar_s))
        rms = _frame_rms(samples, _FRAME, _HOP)
        flux = _onset_flux(samples, sr)
        fps = sr / _HOP
        raw: list[dict[str, float]] = []
        for b in range(n_bars):
            t0 = b * bar_s
            f0 = int(t0 * fps)
            f1 = min(len(rms), int((t0 + bar_s) * fps))
            e = (sum(rms[f0:f1]) / (f1 - f0)) if f1 > f0 else 0.0
            dens = (sum(flux[f0:f1]) / bar_s) if f1 > f0 else 0.0
            raw.append({"energy": e, "onset_density": dens,
                        "position_frac": round(b / max(n_bars, 1), 4)})
        # normalize per track (0..1): a drop is a drop at any volume,
        # and the code rule's 0.4 jump threshold is on this scale.
        max_e = max((r["energy"] for r in raw), default=0.0)
        max_d = max((r["onset_density"] for r in raw), default=0.0)
        rows: list[dict[str, float]] = []
        prev = 0.0
        for r in raw:
            e = r["energy"] / max_e if max_e > 0 else 0.0
            rows.append({"energy": round(e, 4),
                         "energy_delta": round(e - prev, 4),
                         "onset_density": round(
                             r["onset_density"] / max_d, 4)
                         if max_d > 0 else 0.0,
                         "position_frac": r["position_frac"]})
            prev = e
        return rows
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk._bar_features", exc)
        return []


def _code_cue_rule(feats: dict[str, float]) -> int:
    """The code rung: a cue on the set start and on any big energy
    jump. It cannot see a build bar coming — that is the model's edge.
    """
    if feats.get("position_frac", 1.0) == 0.0:
        return 1
    return 1 if feats.get("energy_delta", 0.0) >= 0.4 else 0


def _code_cue_score(feats: list[dict[str, float]],
                    labels: list[float]) -> float:
    if not labels:
        return 0.0
    hits = sum(1 for f, y in zip(feats, labels, strict=True)
               if _code_cue_rule(f) == int(y))
    return hits / len(labels)


def build_cue_training_rows(n_tracks: int, seed: int = 0) -> list[dict[str, Any]]:
    """Synthetic labeled structure rows: structured tracks whose bar
    energies follow intro -> build -> drop -> breakdown -> drop, with
    the ground-truth cue bars known (set start, drops, AND the build
    bar before each drop — the bar the code rule cannot see coming).

    Deterministic per seed; the model must beat the code rule on the
    held-out slice to be promoted, per the specialists gate.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for ti in range(max(0, int(n_tracks))):
        n_bars = rng.choice((8, 8, 12))
        energies: list[float] = []
        e = rng.uniform(0.25, 0.35)
        for _ in range(n_bars):
            energies.append(round(e, 3))
            e = max(0.2, min(0.95, e + rng.uniform(-0.1, 0.18)))
        # plant the structure: a drop (jump >= 0.4) preceded by a build
        cues: set[int] = {0}
        for drop_at in (3, 6):
            if drop_at < n_bars:
                energies[drop_at - 1] = round(
                    min(0.6, energies[drop_at - 2] + 0.2), 3)
                energies[drop_at] = round(
                    min(0.95, energies[drop_at - 1] + 0.45), 3)
                cues.add(drop_at)
                cues.add(drop_at - 1)  # the build bar: a cue, too
        # rebuild samples so the features are REAL measurements
        samples = _structured_samples(energies, 120.0)
        feats = _bar_features(samples, SR, 120.0)
        for b, f in enumerate(feats):
            rows.append({"features": f, "label": 1 if b in cues else 0,
                         "track": ti, "bar": b})
    return rows


def _structured_samples(bar_energy: list[float], bpm: float) -> array:
    """Clicks + a sine bed at each bar's energy (the training-track
    synth — the same shape the desk tests use as ground truth)."""
    beat_s = 60.0 / bpm
    bar_s = beat_s * 4
    n = int(SR * len(bar_energy) * bar_s)
    out = array("d", [0.0]) * n
    for b, e in enumerate(bar_energy):
        start = int(b * bar_s * SR)
        for beat in range(4):
            cs = start + int(beat * beat_s * SR)
            for i in range(min(2205, n - cs)):
                t = i / SR
                out[cs + i] += e * math.exp(-t * 60.0) * math.sin(
                    2 * math.pi * 180.0 * t)
        for i in range(start, min(n, start + int(bar_s * SR))):
            t = i / SR
            out[i] += 0.25 * e * math.sin(2 * math.pi * 110.0 * t)
    return out


def fit_cue_suggester(rows: list[dict[str, Any]], store: Any = None,
                      seed: int = 0) -> Any:
    """Train the cue suggester through the specialists gate: floor,
    held-out score, and it must BEAT the energy-jump code rule on the
    same slice, or the code rung keeps carrying. Never raises."""
    try:
        from ..llm.specialists import LearnerSpec, fit_if_ready
        spec = LearnerSpec(job="dj_cue_suggest",
                           features=_CUE_FEATURES, kind="classifier",
                           backend="logistic", min_examples=30)
        return fit_if_ready(rows, spec, code_score_fn=_code_cue_score,
                            store=store, seed=seed)
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.fit_cue_suggester", exc)
        from ..llm.specialists import FitOutcome
        return FitOutcome(status="unavailable", n_rows=len(rows or []),
                          reason=f"cue suggester fit failed: {exc}")


_CUE_MODEL: Any = None
_CUE_MODEL_LOADED = False


def _cue_model() -> Any:
    """The trained cue model from the versioned store, if one was ever
    promoted; None until then (the code rule carries). Cached."""
    global _CUE_MODEL, _CUE_MODEL_LOADED
    if _CUE_MODEL_LOADED:
        return _CUE_MODEL
    _CUE_MODEL_LOADED = True
    try:
        from ..llm.specialists import ModelStore
        _CUE_MODEL = ModelStore().load("dj_cue_suggest")
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk._cue_model", exc)
        _CUE_MODEL = None
    return _CUE_MODEL


def suggest_cues(samples: array, sr: int, bpm: float,
                 model: Any = None) -> list[tuple[int, float]]:
    """Cue bars for a track, best first: (bar_index, score).

    The learned model when one is trained (score = P(cue)), else the
    energy-jump code rule (score 1.0 for rule hits). Never raises.
    """
    try:
        feats = _bar_features(samples, sr, bpm)
        if not feats:
            return []
        if model is None:
            model = _cue_model()
        out: list[tuple[int, float]] = []
        for b, f in enumerate(feats):
            s = (float(model.predict_one(f)) if model is not None
                 else float(_code_cue_rule(f)))
            if s >= 0.5:
                out.append((b, round(s, 3)))
        out.sort(key=lambda kv: -kv[1])
        return out
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.suggest_cues", exc)
        return []


# ── beat-tracking model rung (madmom) ───────────────────────────────────────

#: Default home for the provisioned madmom venv (Devon installs her
#: own runtimes; the venv is NOT part of the repo).
_MADMOM_VENV_HOME = (Path(os.path.expanduser("~")) / ".nomorals" / "dj"
                     / "madmom-venv")

_MADMOM_DRIVER = '''
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
for _n, _v in (("int", int), ("float", float), ("bool", bool),
               ("object", object)):
    if not hasattr(np, _n):
        setattr(np, _n, _v)
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.processors import SequentialProcessor
proc = SequentialProcessor([RNNBeatProcessor(),
                            DBNBeatTrackingProcessor(fps=100)])
beats = proc(sys.argv[1])
print(json.dumps([round(float(b), 4) for b in beats]))
'''


def _numpy_compat_shim() -> None:
    """madmom 0.16.1 predates the numpy 1.24 alias removals; its
    compiled HMM still reads np.int. The shim restores the aliases
    before any madmom import (in-process or in the venv driver)."""
    try:
        import warnings

        import numpy as np
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _n, _v in (("int", int), ("float", float), ("bool", bool),
                           ("object", object)):
                if not hasattr(np, _n):
                    setattr(np, _n, _v)
    except Exception:  # noqa: BLE001 - no numpy, no madmom either
        pass


def _madmom_venv_python(venv_python: str | None = None) -> str:
    if venv_python:
        return venv_python
    env = (os.environ.get("NM_MADMOM_VENV") or "").strip()
    if env:
        return env
    return str(_MADMOM_VENV_HOME / "bin" / "python")


def madmom_available(venv_python: str | None = None) -> tuple[bool, str]:
    """Is the madmom RNN beat tracker usable here? (ok, reason).

    Tries the in-process import first, then the provisioned venv
    python (NM_MADMOM_VENV or ~/.nomorals/dj/madmom-venv). Never
    raises; False names the wall.
    """
    _numpy_compat_shim()
    try:
        import madmom  # noqa: F401
        return True, f"madmom {madmom.__version__} (in-process)"
    except Exception:  # noqa: BLE001
        pass
    py = _madmom_venv_python(venv_python)
    try:
        proc = subprocess.run(
            [py, "-c", "import madmom; print(madmom.__version__)"],
            capture_output=True, text=True, timeout=60)
        if proc.returncode == 0 and proc.stdout.strip():
            return True, (f"madmom {proc.stdout.strip()} "
                         f"(venv {py})")
        return False, (f"madmom not installed at {py} — run "
                       f"/dj desk provision, or provision_madmom()")
    except Exception as exc:  # noqa: BLE001
        return False, f"madmom check failed at {py}: {exc}"


def madmom_beat_track(path: str,
                      venv_python: str | None = None) -> list[float]:
    """Beat times from the madmom RNN tracker. Raises RuntimeError
    with the named wall when the model rung is unavailable."""
    _numpy_compat_shim()
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # noqa: F401
        from madmom.processors import SequentialProcessor
        proc = SequentialProcessor(
            [RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100)])
        return [round(float(b), 4) for b in proc(path)]
    except ImportError:
        pass
    py = _madmom_venv_python(venv_python)
    try:
        proc = subprocess.run([py, "-", path], input=_MADMOM_DRIVER,
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"madmom venv run failed ({py}): "
                f"{(proc.stderr or '')[-300:]}")
        return [float(b) for b in json.loads(proc.stdout.strip())]
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"madmom beat tracking failed: {exc}") from exc


def provision_madmom(dest: str | Path | None = None) -> dict[str, Any]:
    """Self-provision the madmom beat tracker: venv + Cython + numpy
    + scipy + setuptools, madmom built from source, the two Python
    3.12 compatibility patches, then a smoke beat-run on a click
    track. Explicit only — never runs on import. Returns a report
    dict; slow (minutes) by nature. Never raises.
    """
    dest = Path(dest) if dest else _MADMOM_VENV_HOME
    steps: list[str] = []
    try:
        py = dest / "bin" / "python"
        pip = dest / "bin" / "pip"
        if not py.is_file():
            proc = subprocess.run(
                [sys.executable, "-m", "venv", str(dest)],
                capture_output=True, text=True, timeout=300)
            steps.append(f"venv: rc={proc.returncode}")
            if proc.returncode != 0:
                return {"ok": False, "path": str(dest), "steps": steps,
                        "error": f"venv failed: "
                                 f"{(proc.stderr or '')[-300:]}"}
        for args, label in (
                (["install", "-q", "--upgrade", "pip"], "pip upgrade"),
                (["install", "-q", "Cython>=0.29", "numpy<2", "scipy",
                  "setuptools", "wheel"], "build deps"),
                (["install", "--no-build-isolation", "madmom"],
                 "madmom build"),
                (["install", "-q", "setuptools<81"], "setuptools<81")):
            proc = subprocess.run([str(pip), *args], capture_output=True,
                                  text=True, timeout=900)
            steps.append(f"{label}: rc={proc.returncode}")
            if proc.returncode != 0:
                return {"ok": False, "path": str(dest), "steps": steps,
                        "error": f"{label} failed: "
                                 f"{(proc.stderr or proc.stdout or '')[-300:]}"}
        # Python 3.12 patches, venv-local (madmom 0.16.1 predates 3.10+):
        # collections.abc + the removed numpy aliases in pure-Python files.
        site = next((dest / "lib").glob("python*/site-packages"))
        mm = site / "madmom"
        proc = subprocess.run(
            ["sed", "-i",
             "s/^from collections import MutableSequence$/"
             "from collections.abc import MutableSequence/",
             str(mm / "processors.py")],
            capture_output=True, text=True, timeout=60)
        steps.append(f"patch collections.abc: rc={proc.returncode}")
        import re as _re
        patched = 0
        for f in mm.rglob("*.py"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            new = _re.sub(r"np\.(float|int|bool|object)\b",
                          lambda m: {"float": "float", "int": "int",
                                     "bool": "bool",
                                     "object": "object"}[m.group(1)],
                          text)
            if new != text:
                f.write_text(new, encoding="utf-8")
                patched += 1
        steps.append(f"patch numpy aliases: {patched} files")
        ok, reason = madmom_available(str(py))
        steps.append(f"verify: {reason}")
        if not ok:
            return {"ok": False, "path": str(dest), "steps": steps,
                    "error": reason}
        return {"ok": True, "path": str(dest), "steps": steps,
                "error": ""}
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.provision_madmom", exc)
        return {"ok": False, "path": str(dest), "steps": steps,
                "error": str(exc)}


# ── real songs through the music chain ──────────────────────────────────────

def load_real_track(query: str, workdir: str,
                    context: Any = None) -> dict[str, Any]:
    """Fetch one REAL track through the music chain and put it on the
    desk: search (all 20 MUSIC_SEARCH_SOURCES strategies) + download
    via dj_mixdrop.fetch_track_audio, then analyze_track with BPM/key
    DETECTED from the audio — never ground-truth. Returns the loaded
    track dict, or {"ok": False, "error"} — a failed fetch never
    silently becomes a composed track; the caller names the fallback.
    Never raises.
    """
    try:
        fetch = fetch_track_audio(query, workdir, context)
        if not fetch.ok:
            return {"ok": False, "query": query,
                    "error": fetch.error or "fetch failed"}
        analysis = analyze_track(fetch.path, title=fetch.title or query)
        samples, sr = _read_mono(fetch.path)
        if sr != SR and samples:
            samples = _resample_linear(samples, sr, SR)
            sr = SR
        grid = detect_beat_grid(samples, sr, analysis.bpm)
        if grid.known:
            idx, _c = detect_downbeat(samples, sr, grid)
            grid.downbeat_index = idx
        cues = suggest_cues(samples, sr, analysis.bpm or 0.0)
        # the library records every real track fetched — the desk's
        # crates are built from what was actually played, not typed
        # in by hand. Best-effort: the record never blocks the fetch.
        try:
            _store = DeskStore()
            _store.load()
            Library(_store).add_track(
                fetch.path, title=fetch.title or query,
                artist=fetch.artist or "", bpm=analysis.bpm,
                key=analysis.key, mode=analysis.mode,
                source=fetch.source or "music-chain",
                path=fetch.path, duration_s=analysis.duration_s)
            _store.save()
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "query": query, "track_id": fetch.path,
                "path": fetch.path, "title": fetch.title,
                "artist": fetch.artist, "source": fetch.source,
                "analysis": analysis, "samples": samples, "grid": grid,
                "cues": cues}
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.load_real_track", exc)
        return {"ok": False, "query": query, "error": str(exc)}


def dig_real_crate(queries: list[str], workdir: str,
                   context: Any = None
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch a crate of real tracks for a set. Returns (loaded,
    failures) — partial success is normal and reported per track."""
    loaded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for q in queries or []:
        got = load_real_track(q, workdir, context)
        if got.get("ok"):
            loaded.append(got)
        else:
            failures.append({"query": q, "error": got.get("error", "?")})
    return loaded, failures


def resolve_show_track(pick: dict[str, Any] | None, workdir: str,
                       context: Any = None, fetch_fn: Any = None,
                       compose_fn: Any = None) -> dict[str, Any]:
    """One show slot: real song first, composed fallback, named.

    ``fetch_fn`` is the music-chain fetch (defaults to
    fetch_track_audio); ``compose_fn`` composes a track when the real
    fetch fails. The returned dict always names its source — "real",
    "composed" (with the fetch failure as the reason), or "none".
    """
    fetch = fetch_fn or fetch_track_audio
    title = (pick or {}).get("title", "") or ""
    artist = (pick or {}).get("artist", "") or ""
    query = f"{artist} {title}".strip() or title or "Devon FM original"
    try:
        got = fetch(query, workdir, context)
    except Exception as exc:  # noqa: BLE001
        got = None
        fetch_error = str(exc)
    else:
        fetch_error = getattr(got, "error", "") or ""
    if got is not None and getattr(got, "ok", False) and got.path:
        return {"source": "real", "path": got.path, "title": got.title,
                "artist": got.artist, "fetch_source": got.source,
                "reason": ""}
    if compose_fn is not None:
        try:
            path = compose_fn(pick, workdir) or ""
        except Exception as exc:  # noqa: BLE001
            path = ""
            compose_error = str(exc)
        else:
            compose_error = ""
        if path and os.path.isfile(path):
            return {"source": "composed", "path": path,
                    "title": title, "artist": artist or "Devon FM",
                    "reason": "real fetch failed: "
                              f"{fetch_error or 'no result'} — "
                              "composed fallback"}
        return {"source": "none", "path": "",
                "reason": "real fetch failed: "
                          f"{fetch_error or 'no result'}; compose failed: "
                          f"{compose_error or 'no audio'}"}
    return {"source": "none", "path": "",
            "reason": f"real fetch failed: {fetch_error or 'no result'}"}


# ── show-builder wiring ─────────────────────────────────────────────────────

def attach_desk_plan(analyses: list[TrackAnalysis],
                     samples: dict[str, array] | None = None
                     ) -> list[dict[str, Any]]:
    """Grid + suggested cues for each analyzed track, for the show
    builder to place on the decks. Never raises."""
    plans: list[dict[str, Any]] = []
    for a in analyses or []:
        try:
            grid = BeatGrid(bpm=a.bpm)
            cues: list[tuple[int, float]] = []
            waveform: list[float] = []
            s = (samples or {}).get(a.path) if samples else None
            if s is not None and a.bpm:
                grid = detect_beat_grid(s, SR, a.bpm)
                if grid.known:
                    idx, _c = detect_downbeat(s, SR, grid)
                    grid.downbeat_index = idx
                cues = suggest_cues(s, SR, a.bpm)
                waveform = compute_waveform(s, SR)
            plans.append({"track": a, "grid": grid, "cues": cues,
                          "waveform": waveform})
        except Exception as exc:  # noqa: BLE001
            record_failure("dj_desk.attach_desk_plan", exc)
            plans.append({"track": a, "grid": BeatGrid(bpm=a.bpm),
                          "cues": [], "waveform": []})
    return plans


# ── /dj desk surface ────────────────────────────────────────────────────────

_LIVE_DESK = DJDesk()


def _desk_rungs_text() -> str:
    ok, reason = madmom_available()
    beat_rung = ("madmom RNN beat tracker: " + reason) if ok else (
        "beat-grid model rung: unavailable — " + reason)
    cue = _cue_model()
    cue_rung = ("cue suggester: learned model trained"
                if cue is not None else
                "cue suggester: code rule (energy jumps) — "
                "no trained model yet")
    return f"{beat_rung}\n{cue_rung}"


def control_dj_desk(tail: str, *, chat_key: str = "",
                    deliver_audio: Any = None, deliver_text: Any = None,
                    context: Any = None, workdir: str | None = None
                    ) -> str:
    """`/dj desk` — the live desk surface.

    * ``/dj desk`` — status: decks, grids, cues, EQ, FX, model rungs
    * ``/dj desk fetch <song>`` — fetch a REAL track, analyze it,
      load it on deck A
    * ``/dj desk live [genre|song, song, ...]`` — a live set of REAL
      tracks played through the desk, dropped in chat
    * ``/dj desk eq <A|B> <low|mid|high> <0..2>`` — 3-band EQ
    * ``/dj desk fx <A|B> <filter|echo|reverb|flanger> …`` — effects
    * ``/dj desk fader <A|B> <0..1>`` — channel fader
    * ``/dj desk keylock <A|B> <on|off>`` — tempo without pitch
    * ``/dj desk jog <A|B> <beats>`` — nudge the playhead
    * ``/dj desk crate [list|search <q>|new <name>|put <crate> <title>|<name>]``
    * ``/dj desk wave [A|B]`` — the track's waveform
    * ``/dj desk record <start|stop>`` — capture the live mix to WAV
    * ``/dj desk provision`` — self-provision the madmom beat tracker
    Never raises.
    """
    try:
        tail = (tail or "").strip()
        low = tail.lower()
        if low.startswith("provision"):
            res = provision_madmom()
            lines = ["madmom provision: "
                     + ("OK — " + res["path"] if res["ok"]
                        else "FAILED — " + res.get("error", "?"))]
            lines.extend("  " + s for s in res.get("steps", []))
            return "\n".join(lines)
        if low.startswith("fetch "):
            query = tail[6:].strip()
            if not query:
                return "usage: /dj desk fetch <song title / artist>"
            wd = workdir or str(Path("dj_desk") / "fetch")
            got = load_real_track(query, wd, context)
            if not got.get("ok"):
                return (f"couldn't fetch '{query}': {got.get('error', '?')} "
                        "— try another title, or /dj desk live for "
                        "trending picks.")
            _LIVE_DESK.load_deck("A", got["analysis"], got["samples"])
            a = got["analysis"]
            grid = got["grid"]
            cues = ", ".join(f"bar {b} ({s})" for b, s in got["cues"][:4])
            return (f"on deck A (REAL — {got['source']}): "
                    f"{got['title']} — {got['artist']}\n"
                    f"BPM {a.bpm} ({a.bpm_source}) · key {a.key} {a.mode} "
                    f"· grid {len(grid.beat_times)} beats "
                    f"(conf {grid.confidence}) · downbeat "
                    f"{grid.downbeat_index}\n"
                    f"suggested cues: {cues or 'none yet'}")
        if low.startswith("live"):
            rest = tail[4:].strip()
            queries: list[str] = []
            if rest:
                if "," in rest:
                    queries = [q.strip() for q in rest.split(",") if q.strip()]
                else:
                    from .dj_mixdrop import _trending_queries
                    queries = _trending_queries(rest, 4, context)
                    if not queries:
                        queries = [rest]
            else:
                from .dj_mixdrop import _trending_queries
                queries = _trending_queries("", 4, context)
            if not queries:
                return ("no tracks to play live — charts are empty and "
                        "no songs were named")
            wd = workdir or str(Path("dj_desk") / f"live_{int(time.time())}")
            tracks, failures = dig_real_crate(queries[:4], wd, context)
            if not tracks:
                why = "; ".join(f["error"] for f in failures[:2])
                return (f"no real tracks fetched for the live set: {why} "
                        "— the desk plays real songs; composed tracks are "
                        "the show fallback, not the live desk.")
            script: list[dict[str, Any]] = []
            first_cues = tracks[0].get("cues") or []
            cue_beat = (first_cues[0][0] * 4) if first_cues else 4
            script.append({"at_s": 0.0, "deck": "A", "kind": "cue",
                           "name": "drop", "beat": cue_beat})
            script.append({"at_s": 3.5, "deck": "A", "kind": "pad",
                           "pad": "airhorn"})
            if len(tracks) > 1:
                script.append({"at_s": 4.0, "deck": "A", "kind": "xfade",
                               "to": 1.0, "beats": 4})
                script.append({"at_s": 4.0, "deck": "B", "kind": "loop",
                               "in_beat": 0, "length_beats": 4})
            dur = min(60.0, max(t["analysis"].duration_s
                                for t in tracks))
            out_path = str(Path(wd) / "desk-live-set.wav")
            res = run_live_set(
                [(t["track_id"], t["analysis"], t["samples"])
                 for t in tracks], script, out_path, duration_s=dur)
            if not res.get("ok"):
                return f"live set failed: {res.get('reason', '?')}"
            names = " + ".join(
                f"{t['title']} ({t['source']})" for t in tracks)
            failed = (f"\n{len(failures)} fetch(es) failed: " + "; ".join(
                f"{f['query']}: {f['error']}" for f in failures[:2])
                if failures else "")
            caption = (f"🎧 live from the desk — REAL tracks: {names} · "
                       f"{res['duration_s']}s through cues, pads, loops "
                       f"and the crossfader")
            if deliver_audio is not None:
                deliver_audio(res["path"], caption)
                return caption + failed
            return caption + f"\nfile: {res['path']}" + failed
        parts = tail.split()
        verb = parts[0].lower() if parts else ""
        if verb == "eq":
            if len(parts) < 4:
                return ("usage: /dj desk eq <A|B> <low|mid|high> <0..2> "
                        "(0 = cut, 1 = flat, 2 = +6 dB)")
            deck = _LIVE_DESK.deck(parts[1])
            value = deck.set_eq(parts[2], float(parts[3]))
            return f"deck {deck.name} EQ {parts[2].lower()} = {value}"
        if verb == "fx":
            if len(parts) < 3:
                return ("usage: /dj desk fx <A|B> filter "
                        "<lowpass|highpass|off> [cutoff] · "
                        "fx <A|B> echo <on|off> [beats feedback mix] · "
                        "fx <A|B> reverb <on|off> [mix] · "
                        "fx <A|B> flanger <on|off> [mix rate] · "
                        "fx <A|B> off")
            deck = _LIVE_DESK.deck(parts[1])
            kind = parts[2].lower()
            if kind == "filter":
                mode = parts[3].lower() if len(parts) > 3 else "lowpass"
                cutoff = float(parts[4]) if len(parts) > 4 else 8000.0
                res = deck.set_fx("filter", mode=mode, cutoff=cutoff)
            elif kind == "echo":
                on = (parts[3].lower() != "off") if len(parts) > 3 else True
                res = deck.set_fx(
                    "echo", on=on,
                    beats=float(parts[4]) if len(parts) > 4 else 1.0,
                    feedback=float(parts[5]) if len(parts) > 5 else 0.4,
                    mix=float(parts[6]) if len(parts) > 6 else 0.5)
            elif kind == "reverb":
                on = (parts[3].lower() != "off") if len(parts) > 3 else True
                res = deck.set_fx(
                    "reverb", on=on,
                    mix=float(parts[4]) if len(parts) > 4 else 0.5)
            elif kind == "flanger":
                on = (parts[3].lower() != "off") if len(parts) > 3 else True
                res = deck.set_fx(
                    "flanger", on=on,
                    mix=float(parts[4]) if len(parts) > 4 else 0.5,
                    rate_hz=float(parts[5]) if len(parts) > 5 else 0.25)
            elif kind == "off":
                res = deck.set_fx("off")
            else:
                return (f"unknown FX {kind!r} — filter, echo, reverb, "
                        "flanger, off")
            return f"deck {deck.name} FX: {res}"
        if verb == "fader":
            if len(parts) < 3:
                return "usage: /dj desk fader <A|B> <0..1>"
            deck = _LIVE_DESK.deck(parts[1])
            value = deck.set_fader(float(parts[2]))
            return f"deck {deck.name} fader = {value}"
        if verb == "keylock":
            if len(parts) < 3:
                return "usage: /dj desk keylock <A|B> <on|off>"
            deck = _LIVE_DESK.deck(parts[1])
            deck.key_lock = parts[2].lower() in ("on", "1", "true")
            return (f"deck {deck.name} key lock "
                    f"{'on' if deck.key_lock else 'off'}")
        if verb == "jog":
            if len(parts) < 3:
                return "usage: /dj desk jog <A|B> <beats>"
            deck = _LIVE_DESK.deck(parts[1])
            if not deck.samples:
                return (f"deck {deck.name} has no track to jog — "
                        "/dj desk fetch <song> first")
            pos = deck.jog(float(parts[2]))
            return (f"deck {deck.name} jogged to {pos:.2f}s "
                    f"(beat {deck.position_beats():.1f})")
        if verb == "crate":
            store = DeskStore()
            store.load()
            lib = Library(store)
            if len(parts) < 2 or parts[1].lower() == "list":
                crates = lib.crates()
                if not crates:
                    return ("library is empty — no crates yet "
                            "(fetch a track first: "
                            "/dj desk fetch <song>)")
                lines = ["crates:"]
                for c in crates:
                    lines.append(
                        f"  {c} ({len(lib.crate_tracks(c))} tracks)")
                return "\n".join(lines)
            sub = parts[1].lower()
            if sub == "search":
                q = tail[len("crate search"):].strip()
                hits = lib.search(q)
                if not hits:
                    if not lib.search():
                        return (f"no tracks match {q!r} — "
                                "library is empty")
                    return f"no tracks match {q!r}"
                return "\n".join(
                    f"{h['title']} — {h['artist']} · {h['bpm']} BPM · "
                    f"{h['key']} {h['mode']} ({h['source']})"
                    for h in hits[:10])
            if sub == "new":
                if len(parts) < 3:
                    return "usage: /dj desk crate new <name>"
                name = " ".join(parts[2:])
                if lib.create_crate(name):
                    store.save()
                    return f"crate '{name}' created"
                return f"crate '{name}' already exists"
            if sub == "put":
                if len(parts) < 4:
                    return "usage: /dj desk crate put <crate> <title…>"
                crate = parts[2]
                q = " ".join(parts[3:])
                hits = lib.search(q)
                if not hits:
                    return f"no track matches {q!r}"
                if lib.add_to_crate(crate, hits[0]["track_id"]):
                    store.save()
                    return (f"added {hits[0]['title']} "
                            f"to crate '{crate}'")
                return f"no crate '{crate}' — /dj desk crate list"
            tracks = lib.crate_tracks(parts[1])
            if not tracks and parts[1] not in lib.crates():
                return f"no crate '{parts[1]}' — /dj desk crate list"
            if not tracks:
                return f"crate '{parts[1]}' is empty"
            lines = [f"crate '{parts[1]}':"]
            for tid in tracks:
                rec = lib.get_track(tid) or {}
                lines.append(f"  {rec.get('title', tid)} — "
                             f"{rec.get('artist', '')} · "
                             f"{rec.get('bpm')} BPM")
            return "\n".join(lines)
        if verb == "record":
            sub = parts[1].lower() if len(parts) > 1 else ""
            if sub == "start":
                if _LIVE_DESK.record_start():
                    return ("recording the desk — play, mix, then "
                            "/dj desk record stop")
                return "already recording"
            if sub == "stop":
                wd = workdir or str(Path("dj_desk"))
                path = str(Path(wd)
                           / f"desk-record-{int(time.time())}.wav")
                rep = _LIVE_DESK.record_stop(path)
                if not rep.get("ok"):
                    return f"record stop: {rep.get('reason', '?')}"
                caption = (f"🎧 desk recording — {rep['duration_s']}s of "
                           "the live mix")
                if deliver_audio is not None:
                    deliver_audio(rep["path"], caption)
                return caption + f"\nfile: {rep['path']}"
            return "usage: /dj desk record <start|stop>"
        if verb == "wave":
            which = parts[1] if len(parts) > 1 else "A"
            deck = _LIVE_DESK.deck(which)
            if not deck.samples or not deck.waveform:
                return (f"deck {deck.name} has no track — "
                        "waveform is empty")
            bars = " ▁▂▃▄▅▆▇█"
            cols = deck.waveform
            step = max(1, len(cols) // 40)
            spark = "".join(bars[min(8, int(cols[i] * 8))]
                            for i in range(0, len(cols), step))
            return (f"deck {deck.name} waveform "
                    f"({len(cols)} columns):\n{spark}")
        st = _LIVE_DESK.status()
        lines = ["🎧 DJ DESK — the decks you play",
                 f"crossfader: {st['crossfader']} "
                 f"(0 = A only, 1 = B only)"
                 + (" · 🔴 recording" if st["recording"] else "")]
        for key in ("deck_a", "deck_b"):
            d = st[key]
            lines.append(
                f"deck {d['name']}: "
                f"{'playing' if d['playing'] else 'stopped'} · "
                f"BPM {d['bpm']} (eff {d['effective_bpm']}) · "
                f"grid {d['grid_beats']} beats "
                f"(conf {d['grid_confidence']}) · "
                f"downbeat {d['downbeat']} · "
                f"cues {d['cues'] or 'none'} · "
                f"pads {len(d['pads'])} · "
                f"EQ {d['eq'][0]:g}/{d['eq'][1]:g}/{d['eq'][2]:g} · "
                f"fader {d['fader']} · "
                f"key lock {'on' if d['key_lock'] else 'off'} · "
                f"{d['fx']} · waveform {d['waveform']} cols")
        lines.append(_desk_rungs_text())
        lines.append("try: /dj desk fetch <song> · /dj desk live [genre] · "
                     "/dj desk eq A low 0 · /dj desk fx A filter lowpass 800 "
                     "· /dj desk crate list · /dj desk record start · "
                     "/dj desk provision")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        record_failure("dj_desk.control_dj_desk", exc)
        return f"desk error: {exc}"
