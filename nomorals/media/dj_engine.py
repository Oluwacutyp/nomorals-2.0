"""DJ engine — real mixing: tempo/key detection, beatmatching, harmonic mixing.

This is what makes the DJ real instead of a playlist assembler:

* **Track analysis** — BPM via onset-envelope autocorrelation, musical key
  via chromagram + Krumhansl profiles, energy via RMS/brightness. Tracks
  composed by our own ``MusicCreator`` carry ground-truth tempo/key in the
  ``Song`` object — analysis uses that (ground truth, not a guess) and only
  runs DSP detection on external audio.
* **Beatmatching** — incoming track tempo-synced to the outgoing track
  (resample ratio, ±8% like real decks) when within range.
* **Harmonic mixing** — Camelot wheel compatibility scoring (deterministic
  music theory, not vibes).
* **Phrase-aligned transitions** — blends start on bar boundaries, sized in
  beats (16 = one 4-bar phrase), not arbitrary seconds.
* **Energy arc** — track ordering follows warm-up → peak → cool-down,
  not chart order.
* **Taste interface** — ``TasteModel`` protocol: the compatibility and
  sequencing scores live behind a clean interface so a small model trained
  on the owner's skip/like feedback can slot in later. ``HeuristicTaste``
  is the deterministic implementation used today.

Pure stdlib DSP, Termux-safe. Never raises out of the public functions —
analysis degrades honestly (``bpm=None`` means "couldn't detect", and the
planner falls back to an echo-drop instead of claiming a sync it can't do).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "TrackAnalysis",
    "TransitionPlan",
    "analyze_track",
    "detect_bpm",
    "detect_key",
    "track_energy",
    "camelot_code",
    "harmonic_score",
    "harmonic_path",
    "path_harmonic_score",
    "tempo_compatible",
    "sync_ratio",
    "sync_mult",
    "plan_transition",
    "plan_energy_arc",
    "energy_curve",
    "ENERGY_CURVES",
    "arc_report",
    "render_beatmatched_transition",
    "TasteModel",
    "HeuristicTaste",
    "log_feedback",
    "TASTE_FEATURE_SCHEMA",
]

_SR = 22050  # analysis sample rate (tracks are resampled to this)


# ───────────────────────── track analysis ────────────────────────────────────

@dataclass
class TrackAnalysis:
    path: str = ""
    title: str = ""
    bpm: float | None = None       # None = could not detect
    key: str = ""                  # e.g. "C", "F#", "" = unknown
    mode: str = ""                 # "major" | "minor" | ""
    camelot: str = ""              # e.g. "8A" — "" when key unknown
    energy: float = 0.5            # 0..1
    duration_s: float = 0.0
    bpm_source: str = ""           # "ground-truth" | "detected" | "unknown"
    key_source: str = ""           # "ground-truth" | "detected" | "unknown"

    @property
    def beat_s(self) -> float | None:
        return 60.0 / self.bpm if self.bpm else None

    @property
    def bar_s(self) -> float | None:
        b = self.beat_s
        return b * 4.0 if b else None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "title": self.title, "bpm": self.bpm,
                "key": self.key, "mode": self.mode, "camelot": self.camelot,
                "energy": round(self.energy, 3),
                "duration_s": round(self.duration_s, 1),
                "bpm_source": self.bpm_source, "key_source": self.key_source}


def _read_mono(path: str, max_s: float = 90.0) -> tuple[array, int]:
    """Read mono float samples, capped at `max_s` seconds. Never raises."""
    try:
        with wave.open(path, "rb") as f:
            n = f.getnframes()
            sr = f.getframerate()
            ch = f.getnchannels()
            sw = f.getsampwidth()
            cap = int(sr * max_s)
            n = min(n, cap)
            raw = f.readframes(n)
        if sw == 2:
            import struct
            vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
            scale = 32768.0
        elif sw == 1:
            vals = bytes(raw)
            return array("d", [(v - 128) / 128.0 for v in vals][::ch]), sr
        else:
            return array("d"), sr
        if ch > 1:
            vals = vals[::ch]
        return array("d", [v / scale for v in vals]), sr
    except Exception:  # noqa: BLE001
        return array("d"), _SR


def _frame_rms(samples: array, frame: int, hop: int) -> list[float]:
    """RMS energy per frame. Pure Python, fast enough at these sizes."""
    out: list[float] = []
    n = len(samples)
    for start in range(0, max(0, n - frame), hop):
        s = 0.0
        for i in range(start, start + frame):
            v = samples[i]
            s += v * v
        out.append(math.sqrt(s / frame))
    return out


def detect_bpm(samples: array, sr: int = _SR) -> float | None:
    """Tempo via onset-envelope autocorrelation.

    Onset envelope = positive energy flux between consecutive frames;
    autocorrelate it; the strongest lag in the 60–200 BPM window wins,
    with octave correction toward the 90–170 BPM sweet spot.
    Returns None when nothing rhythmic is found. Never raises.
    """
    try:
        if len(samples) < sr * 4:
            return None
        frame, hop = 1024, 512
        rms = _frame_rms(samples, frame, hop)
        if len(rms) < 64:
            return None
        # onset envelope: positive flux
        flux = [max(0.0, rms[i] - rms[i - 1]) for i in range(1, len(rms))]
        mean = sum(flux) / len(flux)
        if mean <= 1e-9:
            return None
        flux = [f - mean for f in flux]
        fps = sr / hop  # frames per second
        # lag window for 60..200 BPM
        lag_min = max(2, int(fps * 60.0 / 200.0))
        lag_max = int(fps * 60.0 / 60.0)
        n = len(flux)
        # subsample for speed: autocorrelation is O(n * lags)
        step = max(1, n // 1500)
        xs = flux[::step]
        nn = len(xs)
        smin = max(2, lag_min // step)
        smax = min(nn // 2, lag_max // step)
        if smax <= smin:
            return None
        best_lag, best_val = smin, -1.0
        for lag in range(smin, smax + 1):
            acc = 0.0
            for i in range(nn - lag):
                acc += xs[i] * xs[i + lag]
            if acc > best_val:
                best_val, best_lag = acc, lag
        if best_val <= 0:
            return None
        lag = best_lag * step
        bpm = 60.0 * fps / lag
        # octave correction: fold into 90..180
        while bpm < 90.0:
            bpm *= 2.0
        while bpm > 180.0:
            bpm /= 2.0
        return round(bpm, 1)
    except Exception:  # noqa: BLE001
        _log.debug("bpm detection failed", exc_info=True)
        return None


# Krumhansl-Schmuckler key profiles
_KRUMHANSL_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KRUMHANSL_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                    2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B")


def _goertzel_mag(samples: array, start: int, n: int,
                  freq: float, sr: int) -> float:
    """Single-frequency magnitude via Goertzel. Never raises."""
    try:
        k = 0.5 + n * freq / sr
        w = 2.0 * math.pi * k / n
        cw = math.cos(w)
        coeff = 2.0 * cw
        s0 = s1 = s2 = 0.0
        end = min(start + n, len(samples))
        for i in range(start, end):
            s0 = samples[i] + coeff * s1 - s2
            s2, s1 = s1, s0
        return math.sqrt(max(0.0, s1 * s1 + s2 * s2 - coeff * s1 * s2))
    except Exception:  # noqa: BLE001
        return 0.0


def detect_key(samples: array, sr: int = _SR) -> tuple[str, str]:
    """Musical key via chromagram + Krumhansl correlation.

    Chroma from Goertzel magnitudes at 12 pitch classes × 2 octaves
    (A2=110 Hz base), accumulated over the middle 30 s, then correlated
    against rotated major/minor profiles. Returns (key, mode);
    ("", "") when nothing tonal is found. Never raises.
    """
    try:
        if len(samples) < sr * 8:
            return "", ""
        # middle 30 s — skips intros/outros that may be atonal
        total = len(samples)
        seg_n = min(total, sr * 30)
        seg_start = max(0, (total - seg_n) // 2)
        seg = samples[seg_start:seg_start + seg_n]
        win, hop = 4096, 8192
        chroma = [0.0] * 12
        base = 110.0  # A2
        for start in range(0, max(1, len(seg) - win), hop):
            for pc in range(12):
                mag = 0.0
                for octv in range(2):
                    f = base * (2.0 ** (pc / 12.0)) * (2.0 ** octv)
                    if f > sr / 2.5:
                        continue
                    mag += _goertzel_mag(seg, start, win, f, sr)
                chroma[pc] += mag
        total_c = sum(chroma)
        if total_c <= 1e-9:
            return "", ""
        # remap: Goertzel bins are semitones above A (bin 0 = A), but the
        # key profiles are C-based (index 0 = C). C_index = (bin + 9) % 12,
        # so C-based chroma[i] = raw bin (i + 3) % 12.
        chroma = [chroma[(i + 3) % 12] / total_c for i in range(12)]
        # correlate against rotated profiles
        best = ("", "", -2.0)
        for root in range(12):
            for mode, prof in (("major", _KRUMHANSL_MAJOR),
                               ("minor", _KRUMHANSL_MINOR)):
                # rotate profile so it starts at `root`
                rot = [prof[(i - root) % 12] for i in range(12)]
                num = sum((chroma[i] - sum(chroma) / 12)
                            * (rot[i] - sum(rot) / 12) for i in range(12))
                den_c = math.sqrt(sum((c - sum(chroma) / 12) ** 2
                                      for c in chroma))
                den_r = math.sqrt(sum((r - sum(rot) / 12) ** 2 for r in rot))
                corr = num / (den_c * den_r) if den_c and den_r else -1.0
                if corr > best[2]:
                    best = (_PITCH_NAMES[root], mode, corr)
        if best[2] < 0.35:  # too atonal to call
            return "", ""
        return best[0], best[1]
    except Exception:  # noqa: BLE001
        _log.debug("key detection failed", exc_info=True)
        return "", ""


def track_energy(samples: array, sr: int = _SR) -> float:
    """Perceived energy 0..1 from RMS loudness + brightness. Never raises."""
    try:
        if not samples:
            return 0.5
        n = len(samples)
        rms = math.sqrt(sum(v * v for v in samples) / n)
        # brightness proxy: zero-crossing rate
        zc = sum(1 for i in range(1, min(n, sr * 20), 7)
                 if samples[i] * samples[i - 1] < 0)
        zc_rate = zc / max(1, min(n, sr * 20) / 7)
        loud = min(1.0, rms / 0.35)
        bright = min(1.0, zc_rate / 0.45)
        return round(max(0.0, min(1.0, 0.65 * loud + 0.35 * bright)), 3)
    except Exception:  # noqa: BLE001
        return 0.5


def analyze_track(path: str, *, title: str = "",
                  known_bpm: float | None = None,
                  known_key: str = "", known_mode: str = "") -> TrackAnalysis:
    """Full analysis of one track file.

    `known_bpm`/`known_key`/`known_mode` are ground truth (e.g. from the
    composer) — used directly, never "detected". DSP detection runs only
    on external audio without metadata. Never raises.
    """
    a = TrackAnalysis(path=path, title=title or os.path.basename(path))
    samples, sr = _read_mono(path)
    a.duration_s = round(len(samples) / max(1, sr), 1)
    if not samples:
        return a
    # resample analysis copy to 22050 for consistent DSP
    if sr != _SR:
        try:
            from .vocal_lite import _resample_linear as _rs
            samples = _rs(samples, sr, _SR)
            sr = _SR
        except Exception:  # noqa: BLE001
            pass
    if (isinstance(known_bpm, (int, float)) and not isinstance(known_bpm, bool)
            and 40.0 <= known_bpm <= 220.0):
        a.bpm, a.bpm_source = round(float(known_bpm), 1), "ground-truth"
    else:
        a.bpm = detect_bpm(samples, sr)
        a.bpm_source = "detected" if a.bpm else "unknown"
    if isinstance(known_key, str) and known_key.strip():
        a.key, a.mode = known_key.strip(), (known_mode or "major")
        a.key_source = "ground-truth"
    else:
        a.key, a.mode = detect_key(samples, sr)
        a.key_source = "detected" if a.key else "unknown"
    a.camelot = camelot_code(a.key, a.mode)
    a.energy = track_energy(samples, sr)
    return a


# ───────────────────────── harmonic mixing ───────────────────────────────────

_MAJOR_CAMELOT = {"B": "1B", "F#": "2B", "Gb": "2B", "Db": "3B", "C#": "1B",
                  "Ab": "4B", "Eb": "5B", "Bb": "6B", "F": "7B", "C": "8B",
                  "G": "9B", "D": "10B", "A": "11B", "E": "12B", "D#": "3B",
                  "G#": "4B", "A#": "6B"}
_MINOR_CAMELOT = {"G#": "1A", "Ab": "1A", "Eb": "2A", "D#": "2A",
                  "Bb": "3A", "A#": "3A", "F": "4A", "C": "5A",
                  "G": "6A", "D": "7A", "A": "8A", "E": "9A",
                  "B": "10A", "F#": "11A", "Gb": "11A", "C#": "12A",
                  "Db": "12A"}


def camelot_code(key: str, mode: str) -> str:
    """Key name → Camelot code ("8A"). "" when unknown. Never raises."""
    try:
        table = _MINOR_CAMELOT if (mode or "").lower().startswith("min") \
            else _MAJOR_CAMELOT
        return table.get((key or "").strip(), "")
    except Exception:  # noqa: BLE001
        return ""


def _camelot_parts(code: str) -> tuple[int, str] | None:
    try:
        return int(code[:-1]), code[-1].upper()
    except Exception:  # noqa: BLE001
        return None


def _camelot_neighbors(code: str) -> list[str]:
    """The T-shape: same key, ±1 step same letter, same number letter-flip.

    (Mixed In Key's three rules, as graph edges.)
    """
    parts = _camelot_parts(code)
    if not parts:
        return []
    n, letter = parts
    up = n % 12 + 1
    down = (n - 2) % 12 + 1
    flip = "B" if letter == "A" else "A"
    return [code, f"{up}{letter}", f"{down}{letter}", f"{n}{flip}"]


def harmonic_path(from_code: str, to_code: str) -> list[str]:
    """Shortest harmonic journey between two Camelot codes (BFS).

    Lets the planner route a set between distant keys — e.g. 8A→3A —
    through intermediate compatible keys instead of jumping and
    clashing. Returns [] when either code is unknown.
    Deterministic. Mined from dj-harmonic-analyzer's key-to-key paths.
    """
    if not _camelot_parts(from_code) or not _camelot_parts(to_code):
        return []
    if from_code == to_code:
        return [from_code]
    from collections import deque
    seen = {from_code}
    queue: deque[tuple[str, list[str]]] = deque([(from_code, [from_code])])
    while queue:
        cur, path = queue.popleft()
        for nxt in _camelot_neighbors(cur):
            if nxt in seen:
                continue
            seen.add(nxt)
            if nxt == to_code:
                return path + [nxt]
            queue.append((nxt, path + [nxt]))
    return [from_code, to_code]  # unreachable in practice (wheel is connected)


def path_harmonic_score(path: list[str]) -> float:
    """Mean step score along a harmonic path (1.0 = perfectly smooth)."""
    if len(path) < 2:
        return 1.0
    scores = []
    for a, b in zip(path, path[1:]):
        pa, pb = _camelot_parts(a), _camelot_parts(b)
        if not pa or not pb:
            scores.append(0.5)
            continue
        (na, la), (nb, lb) = pa, pb
        if (na, la) == (nb, lb):
            scores.append(1.0)
        elif na == nb:
            scores.append(0.7)
        elif lb == la:
            scores.append(0.8)
        else:
            scores.append(0.3)
    return round(sum(scores) / len(scores), 3)


def harmonic_score(a: TrackAnalysis, b: TrackAnalysis) -> float:
    """0..1 harmonic compatibility between two tracks.

    1.0 same Camelot code · 0.8 ±1 step (energy lift/drop) ·
    0.7 relative major/minor (same number, A↔B) · 0.3 diagonal
    (±1 step AND letter flip) · 0.0 clash. Unknown keys score 0.5
    (neutral — don't punish what we can't hear). Deterministic.
    """
    pa, pb = _camelot_parts(a.camelot), _camelot_parts(b.camelot)
    if not pa or not pb:
        return 0.5
    na, la = pa
    nb, lb = pb
    if (na, la) == (nb, lb):
        return 1.0
    if na == nb and la != lb:
        return 0.7  # relative major/minor
    up = na % 12 + 1
    down = (na - 2) % 12 + 1
    if nb in (up, down):
        return 0.8 if lb == la else 0.3
    return 0.0


def tempo_compatible(a: TrackAnalysis, b: TrackAnalysis,
                     max_ratio: float = 0.08) -> bool:
    """Can `b` be pitch-shifted onto `a`'s tempo within ±8%? Deterministic.

    Also tries half/double-time equivalence (87 ≈ 174 BPM) — returns
    True when any of 1×, 2×, 0.5× lands within range. Use
    :func:`sync_ratio` for the exact ratio to apply.
    """
    if not a.bpm or not b.bpm:
        return False
    for mult in (1.0, 2.0, 0.5):
        if abs(b.bpm * mult - a.bpm) / a.bpm <= max_ratio:
            return True
    return False


def sync_ratio(out_trk: TrackAnalysis, in_trk: TrackAnalysis,
               max_ratio: float = 0.08) -> float | None:
    """Exact resample ratio to sync `in_trk` onto `out_trk`'s tempo.

    Returns None when no 1×/2×/0.5× ratio lands within ±8% — the
    caller must NOT claim a sync then (honest degradation).
    """
    mult = sync_mult(out_trk, in_trk, max_ratio)
    if mult is None or not out_trk.bpm or not in_trk.bpm:
        return None
    return out_trk.bpm / (in_trk.bpm * mult)


def sync_mult(out_trk: TrackAnalysis, in_trk: TrackAnalysis,
              max_ratio: float = 0.08) -> float | None:
    """Which tempo multiple (1.0, 2.0, 0.5) syncs `in_trk` to `out_trk`.

    None = no multiple lands within range. Deterministic.
    """
    if not out_trk.bpm or not in_trk.bpm:
        return None
    for mult in (1.0, 2.0, 0.5):
        if abs(in_trk.bpm * mult - out_trk.bpm) / out_trk.bpm <= max_ratio:
            return mult
    return None


# ───────────────────────── transition planning ───────────────────────────────

@dataclass
class TransitionPlan:
    kind: str              # "blend" | "echo-drop" | "break"
    blend_beats: int = 16  # blend length in beats (phrase units)
    sync_ratio: float = 1.0  # resample ratio applied to incoming track
    start_beat: int = 0    # beat index in outgoing track where blend starts
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "blend_beats": self.blend_beats,
                "sync_ratio": round(self.sync_ratio, 4),
                "start_beat": self.start_beat, "reason": self.reason}


def plan_transition(out_trk: TrackAnalysis, in_trk: TrackAnalysis) -> TransitionPlan:
    """Choose the transition between two analyzed tracks. Deterministic.

    * both BPM known + syncable (1×/2×/0.5× within ±8%) → beatmatched
      phrase blend (16 beats)
    * harmonic score ≥ 0.7 → echo-out + drop on the one (key-safe)
    * otherwise → break (voice break between, no fake blend)
    """
    harm = harmonic_score(out_trk, in_trk)
    ratio = sync_ratio(out_trk, in_trk)
    if ratio is not None:
        # phrase-align: blend starts on a bar boundary near the track end
        blend_beats = 16
        total_beats = int((out_trk.duration_s * out_trk.bpm) / 60.0)
        start_beat = max(0, (total_beats - blend_beats) // 4 * 4)
        mult = sync_mult(out_trk, in_trk)
        half = " (half/double-time)" if mult not in (None, 1.0) else ""
        return TransitionPlan(
            kind="blend", blend_beats=blend_beats, sync_ratio=ratio,
            start_beat=start_beat,
            reason=(f"beatmatched {in_trk.bpm}→{out_trk.bpm} BPM "
                    f"(×{ratio:.3f}){half}, {in_trk.camelot or '?'}→"
                    f"{out_trk.camelot or '?'} harmonic {harm:.1f}"))
    if harm >= 0.7:
        return TransitionPlan(
            kind="echo-drop",
            reason=(f"echo-out + drop (harmonic {harm:.1f}, "
                    f"tempo gap too wide for sync)"))
    return TransitionPlan(
        kind="break",
        reason=(f"clean break (harmonic {harm:.1f} — no fake blend)"))


# ── energy curves (spotify-mixmaster pattern) ─────────────────────────────────

#: Named energy-curve presets: target energy at each set position (0..1).
#: ``late_peak`` is the classic club arc (warm-up → peak at ~70% → cool-down).
ENERGY_CURVES: dict[str, tuple[float, ...]] = {
    "late_peak": (0.25, 0.35, 0.45, 0.55, 0.68, 0.8, 0.92, 1.0, 0.85, 0.6),
    "linear": (0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.78, 0.86, 0.94, 1.0),
    "wave": (0.4, 0.6, 0.45, 0.7, 0.5, 0.85, 0.6, 1.0, 0.7, 0.45),
    "flat": (0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
}


def energy_curve(name: str, n: int) -> list[float]:
    """Resample a named energy curve to `n` positions. Never raises."""
    curve = ENERGY_CURVES.get(name, ENERGY_CURVES["late_peak"])
    if n <= 0:
        return []
    if n == 1:
        return [curve[len(curve) // 2]]
    out = []
    for i in range(n):
        pos = i * (len(curve) - 1) / (n - 1)
        lo, hi = int(pos), min(len(curve) - 1, int(pos) + 1)
        frac = pos - lo
        out.append(round(curve[lo] * (1 - frac) + curve[hi] * frac, 3))
    return out


def plan_energy_arc(tracks: list[TrackAnalysis],
                    seed: int | None = None, *,
                    curve: str = "late_peak",
                    quality_floor: float = 0.0,
                    taste: "TasteModel | None" = None) -> list[TrackAnalysis]:
    """Order tracks along an energy curve with harmonic sanity.

    * Positions follow the named ``curve`` (``late_peak``/``linear``/
      ``wave``/``flat``): each slot gets the remaining track whose
      energy best matches the slot's target.
    * Consecutive tracks are checked with the taste model (harmonic +
      tempo compatibility); a track that would force a transition below
      ``quality_floor`` is skipped for a later slot.
    * **Quality floor honesty** (spotify-mixmaster): when no remaining
      track clears the floor for a slot, the set ENDS SHORT with the
      reason recorded on the dropped tracks' plan — never padded with
      a bad transition.
    * ``seed`` makes tie-breaking deterministic and reproducible.

    Returns the ordered tracks. Dropped tracks are attached as
    ``plan.dropped`` on the function's ``last_dropped`` attribute.
    """
    plan_energy_arc.last_dropped = []
    if len(tracks) <= 2:
        return list(tracks)
    rng = random.Random(seed)
    targets = energy_curve(curve, len(tracks))
    remaining = list(tracks)
    ordered: list[TrackAnalysis] = []
    dropped: list[TrackAnalysis] = []
    scorer = taste or HeuristicTaste()

    for slot, target in enumerate(targets):
        if not remaining:
            break
        # rank by |energy - target|, tie-broken by seed
        cands = sorted(remaining,
                       key=lambda t: (abs(t.energy - target),
                                      rng.random()))
        pick = None
        for cand in cands:
            if not ordered:
                pick = cand
                break
            prev = ordered[-1]
            plan = plan_transition(prev, cand)
            t_score = scorer.score_transition(prev, cand, plan)
            h_score = harmonic_score(prev, cand)
            combined = 0.6 * t_score + 0.4 * h_score
            if combined >= quality_floor:
                pick = cand
                break
        if pick is None:
            # quality floor: end the set short, honestly
            for cand in remaining:
                cand_dict = cand.to_dict()
                cand_dict["dropped_reason"] = (
                    f"no transition from "
                    f"{ordered[-1].title if ordered else '?'} cleared the "
                    f"quality floor {quality_floor:.2f}")
                dropped.append(cand)
            remaining.clear()
            break
        ordered.append(pick)
        remaining.remove(pick)

    plan_energy_arc.last_dropped = dropped
    return ordered


plan_energy_arc.last_dropped = []  # type: ignore[attr-defined]


def arc_report(ordered: list[TrackAnalysis],
               curve: str = "late_peak") -> str:
    """God-tier set report: journey map, energy sparkbar, transitions."""
    from .style import theme as _theme
    th = _theme()
    lines = [th.banner("DJ Set Plan", f"curve: {curve}")]
    stops = [t.camelot or "?" for t in ordered]
    lines.append(th.section("Harmonic journey"))
    lines.append(th.journey_map(stops))
    lines.append("")
    lines.append(th.section("Energy arc"))
    lines.append("  " + th.sparkbar([t.energy for t in ordered]))
    lines.append(th.kv([
        ("tracks", len(ordered)),
        ("bpm range",
         f"{min((t.bpm or 0) for t in ordered):.0f}–"
         f"{max((t.bpm or 0) for t in ordered):.0f}"
         if any(t.bpm for t in ordered) else "unknown"),
    ]))
    lines.append(th.section("Transitions"))
    for a, b in zip(ordered, ordered[1:]):
        plan = plan_transition(a, b)
        glyph = th.ok if plan.kind == "blend" else th.warn
        lines.append(f"  [{glyph}] {a.title or '?'} → {b.title or '?'}: "
                     f"{plan.kind} — {plan.reason}")
    dropped = getattr(plan_energy_arc, "last_dropped", [])
    if dropped:
        lines.append(th.section("Dropped (quality floor)"))
        for t in dropped:
            d = t.to_dict() if hasattr(t, "to_dict") else {}
            lines.append(f"  [{th.fail}] {t.title or '?'} — "
                         f"{d.get('dropped_reason', '')}")
    return "\n".join(lines)


# ───────────────────────── rendering ─────────────────────────────────────────

def render_beatmatched_transition(mix: array, out_samples: array,
                                  in_samples: array, plan: TransitionPlan,
                                  out_trk: TrackAnalysis,
                                  in_trk: TrackAnalysis,
                                  sr: int = _SR) -> array:
    """Apply a TransitionPlan: tempo-sync + phrase-aligned blend.

    `mix` already ends with the outgoing track. The incoming track is
    resampled to the outgoing tempo (when plan says so), then blended
    over `blend_beats` starting at `start_beat` of the outgoing track.
    Returns the extended mix. Never raises.
    """
    try:
        from .vocal_lite import _resample_linear as _rs
        incoming = array("d", in_samples)
        if plan.kind == "blend" and abs(plan.sync_ratio - 1.0) > 1e-4:
            # tempo-sync: resample incoming to outgoing tempo
            target_len = int(len(incoming) * plan.sync_ratio)
            if target_len > 0:
                tmp = _rs(incoming, sr, int(sr / plan.sync_ratio))
                incoming = tmp
        if plan.kind == "blend" and out_trk.beat_s and in_trk.beat_s:
            out_beat_n = int(out_trk.beat_s * sr)
            in_beat_n = int(in_trk.beat_s * sr / max(1e-6, plan.sync_ratio))
            blend_n = plan.blend_beats * out_beat_n
            start_n = plan.start_beat * out_beat_n
            # mix tail currently = full outgoing track; overlay region:
            ov_start = max(0, len(mix) - len(out_samples) + start_n)
            # align incoming to its own bar start (assume phrase starts at 0)
            need = min(blend_n, len(incoming))
            out = array("d", mix)
            for i in range(need):
                t = i / max(1, need - 1)
                g_out = math.cos(t * math.pi / 2.0)
                g_in = math.sin(t * math.pi / 2.0)
                idx = ov_start + i
                if idx < len(out):
                    out[idx] = out[idx] * g_out + incoming[i] * g_in
            # append the rest of the incoming track
            out.extend(incoming[need:])
            return out
        # echo-drop / break: caller handles spacing; just append
        out = array("d", mix)
        out.extend(incoming)
        return out
    except Exception:  # noqa: BLE001
        _log.debug("beatmatched transition failed", exc_info=True)
        out = array("d", mix)
        out.extend(in_samples)
        return out


# ───────────────────────── taste interface (model-ready) ─────────────────────

#: Feature schema a future taste model trains on. Every transition the DJ
#: renders logs one row (see log_feedback); a small model can learn
#: P(keep | features) from the owner's skip/like/rating events.
TASTE_FEATURE_SCHEMA = {
    "bpm_delta_pct": "abs(b.bpm - a.bpm) / a.bpm",
    "harmonic_score": "0..1 Camelot compatibility",
    "energy_delta": "b.energy - a.energy",
    "energy_abs_b": "incoming track energy 0..1",
    "transition_kind": "blend | echo-drop | break (one-hot)",
    "same_style": "1 if same style tag else 0",
    "label": "1 = kept/played through, 0 = skipped",
}


class TasteModel(Protocol):
    """Interface a learned taste model implements. The DJ calls it;
    HeuristicTaste is the deterministic default. A future small model
    (trained on log_feedback rows) slots in without touching the engine.
    """

    def score_transition(self, out_trk: TrackAnalysis,
                         in_trk: TrackAnalysis,
                         plan: TransitionPlan) -> float:
        """0..1 — how good is this transition?"""
        ...

    def score_sequence(self, tracks: list[TrackAnalysis]) -> float:
        """0..1 — how good is this ordering?"""
        ...


class HeuristicTaste:
    """Deterministic taste: harmonic + tempo + energy-flow heuristics."""

    def score_transition(self, out_trk: TrackAnalysis,
                         in_trk: TrackAnalysis,
                         plan: TransitionPlan) -> float:
        harm = harmonic_score(out_trk, in_trk)
        score = 0.5 * harm
        if plan.kind == "blend":
            score += 0.3
            # small tempo nudges score higher than big ones
            drift = abs(plan.sync_ratio - 1.0)
            score += 0.1 * max(0.0, 1.0 - drift / 0.08)
        elif plan.kind == "echo-drop":
            score += 0.15
        # energy should not collapse mid-show
        e_delta = in_trk.energy - out_trk.energy
        score += 0.1 * max(-1.0, min(1.0, e_delta * 2.0))
        return round(max(0.0, min(1.0, score)), 3)

    def score_sequence(self, tracks: list[TrackAnalysis]) -> float:
        if len(tracks) < 2:
            return 1.0
        scores = [self.score_transition(tracks[i], tracks[i + 1],
                                        plan_transition(tracks[i],
                                                        tracks[i + 1]))
                  for i in range(len(tracks) - 1)]
        return round(sum(scores) / len(scores), 3)


_FEEDBACK_DIR = Path(os.path.expanduser("~")) / ".cache" / "nomorals" / "dj"


def log_feedback(event: str, *, track_id: str = "",
                 out_id: str = "", in_id: str = "",
                 features: dict[str, Any] | None = None) -> None:
    """Log owner feedback for future taste-model training.

    event: "kept" | "skipped" | "liked" | "rated" (+ optional rating).
    Appends one JSONL row per event. Never raises.
    """
    try:
        _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.time(), "event": event, "track_id": track_id,
               "out_id": out_id, "in_id": in_id,
               "features": features or {}}
        with open(_FEEDBACK_DIR / "taste-feedback.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:  # noqa: BLE001
        _log.debug("taste feedback log failed", exc_info=True)
