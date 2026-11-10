"""Melodic matching — SongRecon class 3: covers, re-performances, hums.

Doc 24 SR3. The class-1 engine (``audio/fingerprint.py``, Wang-2003
landmarks) answers "is this that *recording*". This module answers a
different question — "is this that *song*" — over a melody catalog
derived from enrolled recordings:

* **Melody extraction.** Spectral-flux onsets over the shipped
  STFT substrate; each segment's pitch is estimated from its attack
  frames (harmonic-sum F0 candidates with a subharmonic octave
  guard and parabolic interpolation), where the attacking melody
  dominates the mix. Where Demucs is installed, enrollment/query
  extraction runs on the vocals stem; everywhere else on the full
  mix — the receipt names which.
* **Representation.** A work is its successive-note *interval*
  sequence (semitone differences) plus coarse duration buckets
  (short/medium/long relative to the work's own median note). The
  interval sequence is transposition-invariant by construction; the
  buckets are tempo-invariant by construction (ratios, not seconds).
* **Alignment.** Smith–Waterman local alignment of the query's
  interval sequence against each enrolled work's, with a substitution
  kernel that pays full price only for exact interval matches and a
  duration-bucket agreement term. The reported strength is the
  fraction of the query's intervals the best alignment accounts
  for — an alignment statistic in its own units, never a probability.

Honesty ceiling (the doc's, enforced here): melodic answers are
**ranked hypotheses only**. A hypothesis never renders as an
identification, never upgrades or blends into a class-1 result, and
the receipt's match class names it (``class_3_cover_or_
reperformance_hypothesis``). Cover/hum accuracy is *unproven* on real
recordings: the only measurements that exist are over synthesized
fixtures (see the SR3 suite / benchmark idiom), and every melodic
surface says so. The blind ground-truth run over the owner's real
library is what could change that sentence; this module does not.

Enrollment mirrors the shipped fingerprint idiom: works enter from
audio files (``MelodyCatalog.add_work``), or are derived from tracks
already enrolled in the fingerprint DB
(``MelodyCatalog.enroll_fingerprint_track`` — the work inherits the
recording's id, so a class-1 row and its melody stay linked). A query
is never enrolled as a work.

Dependencies: stdlib + the fingerprint module's substrate (numpy
when present, honest pure-Python FFT otherwise). No network, no
external service.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .failures import record_failure
from .fingerprint import (
    LOCAL_RIGHTS_NOTE,
    AudioReadError,
    FingerprintDB,
    RecognitionReceipt,
    _spectrum_magnitudes,
    read_mono,
)

_log = logging.getLogger(__name__)

__all__ = [
    "ALIGNMENT_BASIS",
    "ALIGNMENT_CALIBRATION",
    "ALIGNMENT_NAME",
    "ALIGNMENT_UNIT",
    "MATCH_CLASS_MELODIC_HYPOTHESIS",
    "MAX_TRANSPOSITION_MAD",
    "MIN_ALIGNED_COVERAGE",
    "MIN_ALIGNED_INTERVALS",
    "MIN_ALIGNMENT_FRACTION",
    "MIN_CHROMA_COSINE",
    "Melody",
    "MelodyCatalog",
    "MelodyNote",
    "default_melody_catalog",
    "enroll_work",
    "extract_melody",
    "match_melody_db",
    "melody_from_path",
    "render_hypothesis_lines",
]

#: Match class for every melodic answer. Hypotheses, never an
#: identification — the class name itself carries the ceiling.
MATCH_CLASS_MELODIC_HYPOTHESIS = "class_3_cover_or_reperformance_hypothesis"

#: What the melodic score actually measures: the share of the query
#: melody's successive-note intervals accounted for by the best local
#: alignment against the enrolled work's interval sequence.
ALIGNMENT_NAME = "interval_alignment_fraction"
ALIGNMENT_UNIT = "aligned query melody intervals per query melody interval"
ALIGNMENT_BASIS = (
    "share of the query melody's successive-note intervals aligned, "
    "in order, to intervals of the enrolled work's melody in the "
    "best local (Smith-Waterman) alignment; intervals are "
    "transposition-invariant, duration buckets tempo-invariant"
)
ALIGNMENT_CALIBRATION = (
    "uncalibrated — not a probability; cover/hum accuracy is "
    "unproven on real recordings (only synthesized-fixture "
    "measurements exist)"
)

#: Hypothesis floor (doc 24 SR3, measured on the SR3 fixture
#: battery — bench sibling pool + hand-built covers/hums, see
#: tests/test_d24sr3_melodic_matcher.py). Four evidence gates, each
#: an independent representation agreeing:
#: * alignment fraction >= 0.34 — polyphonic extraction is lossy, so
#:   true re-performances measured 0.38-0.98 while fragment-fitted
#:   wrong works reached 0.70 on fraction ALONE;
#: * >= 5 aligned intervals;
#: * chroma cosine >= 0.84 at the chroma's own best key rotation —
#:   an independent witness: every true pair measured >= 0.866;
#: * transposition MAD <= 2 semitones — one consistent key
#:   relationship along the whole path;
#: * aligned coverage >= 0.50 — at least half the query's intervals
#:   actually align (the gate that keeps unrelated melodies out:
#:   the long unrelated probe melody reached fraction 0.43 through
#:   scattered exact hits but covered only 6 of its 13 intervals,
#:   while every measured true pair covered >= 0.67).
#: Wrong works from the same synthesized family can still clear
#: the gates as lower-ranked competing hypotheses — the receipt
#: presents them as exactly that. The gates are evidence
#: requirements, not a confidence.
MIN_ALIGNMENT_FRACTION = 0.34
MIN_ALIGNED_INTERVALS = 5
MIN_CHROMA_COSINE = 0.84
MAX_TRANSPOSITION_MAD = 2
MIN_ALIGNED_COVERAGE = 0.50
#: A query with fewer intervals than this cannot support a hypothesis
#: at all — too little melody to distinguish one work from another.
#: (Short fragments genuinely collide on contour alone: an unrelated
#: 8-interval melody locally aligned to a wrong work at 0.536 on the
#: probe battery. What keeps such fragments out is the compound
#: floor below — that pair's chroma agreement measured 0.739 against
#: the 0.84 gate — not fragment length by itself.)
MIN_QUERY_INTERVALS = 8

_MELODY_SR = 11025
_MWIN = 4096
_MHOP = 1024
_FMIN, _FMAX = 65.0, 1400.0
#: A frame's candidates are kept when their harmonic-sum score
#: clears 5x the frame's band median (noise raises the median; hums
#: and sung lines clear it easily). After tracking, frames quieter
#: than 5% of the clip's loudest peak are silence, not melody.
_VOICE_MEDIAN_RATIO = 5.0
_VOICE_PEAK_RATIO = 0.05
#: Notes shorter than two frames (~0.19 s) are extraction artifacts
#: (octave flicker, onset spill), not melody.
_MIN_NOTE_FRAMES = 2
#: Onset-anchored segmentation: a melody note is estimated from
#: the frames right after its onset, where the attacking voice
#: dominates the mix. Per-frame tracking over a whole note instead
#: zigzags between voices (a decayed melody note's tail loses to
#: the bass; a sustained pad wins a gap) and — measured on the SR3
#: probe battery — makes two performances of one composition
#: extract to different "melodies". A Viterbi continuity tracker
#: over-commits the other way, locking onto the smoothest sustained
#: line (the pads). Onset anchoring is the middle path that the
#: physics supports: attacks are where the melody is loudest.
_CANDIDATES = 3
#: Minimum frames between onsets (~0.19 s) — below this, flux peaks
#: are one note's shimmer, not two notes.
_MIN_ONSET_GAP_FRAMES = 2
#: A segment longer than this is re-sampled in sub-windows — a
#: legato phrase with weak internal onsets still changes pitch.
_LONG_SEGMENT_S = 1.5
_SUBWINDOW_S = 0.75

#: Alignment kernel. Exact interval match pays 2.0; off-by-one
#: semitone (a singer's slip, an extraction wobble) pays 0.6; any
#: other substitution costs 1.6. Duration buckets agreeing add 0.4,
#: adjacent buckets are neutral, short-vs-long costs 0.4. A gap step
#: (an ornament note, a dropped note) costs 1.1. The maximum cell
#: value is therefore 2.4 — the fraction's denominator basis.
_SIM_EXACT = 2.0
_SIM_OFF_BY_ONE = 0.6
_SIM_OTHER = -1.6
_DUR_SAME = 0.4
_DUR_ADJACENT = 0.0
_DUR_FAR = -0.4
_GAP = -1.6
_CELL_MAX = _SIM_EXACT + _DUR_SAME


# ---------------------------------------------------------------------------
# melody representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MelodyNote:
    """One segmented note: pitch (MIDI), start and duration (seconds).

    ``alt_midi`` carries the segment's second-plausible pitch when
    the attack frames supported two readings (a pad tone shadowing
    the melody, a bass root under it). Alignment may use either
    reading — extraction uncertainty is represented in the data
    instead of being silently decided at the microphone.
    """

    midi: int
    start_s: float
    dur_s: float
    alt_midi: int | None = None


@dataclass(frozen=True)
class Melody:
    """An extracted melody: notes + the clip's mean chroma vector."""

    notes: tuple[MelodyNote, ...] = ()
    chroma: tuple[float, ...] = ()
    duration_s: float = 0.0


def melody_intervals(melody: Melody) -> tuple[int, ...]:
    """Successive-note pitch differences, in semitones.

    Transposition-invariant by construction: add any constant to
    every note and this tuple is unchanged.
    """
    notes = melody.notes
    return tuple(notes[i + 1].midi - notes[i].midi
                 for i in range(len(notes) - 1))


def duration_buckets(melody: Melody) -> tuple[str, ...]:
    """Per-interval duration bucket of the interval's *first* note,
    relative to the melody's median note duration ("s"/"m"/"l").

    Ratios, not seconds — a uniformly faster or slower performance
    keeps the same bucket sequence.
    """
    notes = melody.notes
    if len(notes) < 2:
        return ()
    durs = sorted(n.dur_s for n in notes)
    median = durs[len(durs) // 2] or 1.0
    out: list[str] = []
    for note in notes[:-1]:
        ratio = note.dur_s / median
        out.append("s" if ratio < 0.7 else "l" if ratio > 1.45 else "m")
    return tuple(out)


# ---------------------------------------------------------------------------
# F0 contour extraction over the shipped STFT substrate
# ---------------------------------------------------------------------------

def _parabolic_midi(mags: Sequence[float], bin_idx: int, sr: int,
                    win: int) -> float:
    """Sub-bin pitch (MIDI float) via parabolic interpolation on
    log magnitudes around the peak bin."""
    if 0 < bin_idx < len(mags) - 1:
        a = math.log(max(mags[bin_idx - 1], 1e-12))
        b = math.log(max(mags[bin_idx], 1e-12))
        c = math.log(max(mags[bin_idx + 1], 1e-12))
        denom = a - 2.0 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        shift = max(-0.5, min(0.5, shift))
    else:
        shift = 0.0
    freq = (bin_idx + shift) * sr / win
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def _frame_candidates(mags: Sequence[float], sr: int, win: int
                      ) -> tuple[float, list[tuple[float, float]]]:
    """One frame's pitch candidates: (frame peak, [(midi, emission)]).

    Candidates are local maxima in [_FMIN, _FMAX] of a harmonic-sum
    score (f + 0.6*2f + 0.3*3f) — a note's harmonics support its
    fundamental instead of competing with it. Each candidate passes
    an octave guard (walk down while the subharmonic bin carries at
    least 55% of the chosen bin's magnitude — the classic trap of a
    strong second harmonic outscoring a weak fundamental) and is
    refined to sub-bin pitch by parabolic interpolation. Emission is
    log harmonic-sum over the frame's band median, so it compares
    across frames and against the unvoiced state's constant.
    """
    n_bins = len(mags)
    lo = max(1, int(math.ceil(_FMIN * win / sr)))
    hi = min(n_bins - 1, int(_FMAX * win / sr))
    if hi <= lo:
        return 0.0, []
    band = sorted(mags[lo:hi + 1])
    median = band[len(band) // 2] or 1e-12
    hsum: dict[int, float] = {}
    for k in range(lo, hi + 1):
        score = mags[k]
        if 2 * k < n_bins:
            score += 0.6 * mags[2 * k]
        if 3 * k < n_bins:
            score += 0.3 * mags[3 * k]
        hsum[k] = score
    maxima = [k for k in range(lo, hi + 1)
              if hsum[k] >= hsum.get(k - 1, 0.0)
              and hsum[k] >= hsum.get(k + 1, 0.0)
              and hsum[k] >= _VOICE_MEDIAN_RATIO * median]
    maxima.sort(key=lambda k: -hsum[k])
    out: list[tuple[float, float]] = []
    frame_peak = 0.0
    for k in maxima:
        frame_peak = max(frame_peak, mags[k])
        chosen = k
        while chosen // 2 >= lo and chosen % 2 == 0:
            half = chosen // 2
            if mags[half] >= 0.55 * mags[chosen]:
                chosen = half
            else:
                break
        midi = _parabolic_midi(mags, chosen, sr, win)
        if any(abs(midi - kept) < 0.7 for kept, _e in out):
            continue
        out.append((midi, math.log1p(hsum[k] / median)))
        if len(out) >= _CANDIDATES:
            break
    return frame_peak, out


def _detect_onsets(flux: list[float]) -> list[int]:
    """Frame indices where a new note attacks: flux local maxima
    above an adaptive threshold (median + one standard deviation),
    at least _MIN_ONSET_GAP_FRAMES apart. Frame 0 always onsets."""
    if not flux:
        return []
    n = len(flux)
    ordered = sorted(flux)
    median = ordered[n // 2]
    mean = sum(flux) / n
    std = math.sqrt(sum((f - mean) ** 2 for f in flux) / n)
    threshold = median + std
    onsets = [0]
    for i in range(1, n):
        if flux[i] < threshold:
            continue
        left = flux[i - 1] if i > 0 else 0.0
        right = flux[i + 1] if i + 1 < n else 0.0
        if flux[i] < left or flux[i] < right:
            continue
        if i - onsets[-1] >= _MIN_ONSET_GAP_FRAMES:
            onsets.append(i)
    return onsets


def _segment_candidates(
        cand_frames: list[list[tuple[float, float]]],
        peaks: list[float]) -> tuple[int | None, int | None]:
    """A segment's (primary, alternate) pitches from its attack
    frames: every candidate in every frame votes with weight
    peak x emission, aggregated per semitone. The alternate is kept
    when it draws at least 35% of the primary's weight — two voices
    genuinely contest the segment, and the aligner, not the
    extractor, should settle which line the work is on."""
    weights: dict[int, float] = {}
    for cands, peak in zip(cand_frames, peaks, strict=True):
        for midi, emission in cands:
            key = int(round(midi))
            weights[key] = weights.get(key, 0.0) + peak * emission
    if not weights:
        return None, None
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    primary, top_weight = ranked[0]
    alt: int | None = None
    for midi, weight in ranked[1:]:
        if abs(midi - primary) >= 1 and weight >= 0.35 * top_weight:
            alt = midi
            break
    return primary, alt


def _accumulate_chroma(chroma: list[float], mags: Sequence[float],
                       sr: int, win: int) -> None:
    """Fold one frame's spectrum into a 12-bin chroma accumulator.

    Same weighting as ``fingerprint._chroma_vector`` (magnitude per
    pitch class over 55–4000 Hz), accumulated incrementally so long
    clips never hold every frame in memory.
    """
    step = sr / win
    for i, m in enumerate(mags):
        if m <= 0:
            continue
        f = i * step
        if f < 55.0 or f > 4000.0:
            continue
        midi = 69 + 12.0 * math.log2(f / 440.0)
        chroma[int(round(midi)) % 12] += m


def extract_melody(samples: Sequence[float], sr: int = _MELODY_SR
                   ) -> Melody:
    """Samples → segmented melody (notes + chroma). Total.

    Frames of 4096 samples at a 1024 hop (~93 ms). Note onsets come
    from band spectral flux; each segment's pitch is the
    magnitude-weighted median of its attack frames' strongest F0
    candidates (the first 60% of the segment), where the attacking
    melody dominates the mix. Consecutive segments at the same pitch
    merge — a re-articulated note is one note.
    """
    n = len(samples)
    if n < _MWIN:
        return Melody()
    lo = max(1, int(math.ceil(_FMIN * _MWIN / sr)))
    votes: list[float | None] = []
    cand_frames: list[list[tuple[float, float]]] = []
    peaks: list[float] = []
    flux: list[float] = []
    prev_band: list[float] | None = None
    chroma = [0.0] * 12
    for start in range(0, n - _MWIN + 1, _MHOP):
        mags = _spectrum_magnitudes(samples[start:start + _MWIN])
        _accumulate_chroma(chroma, mags, sr, _MWIN)
        peak, cands = _frame_candidates(mags, sr, _MWIN)
        peaks.append(peak)
        cand_frames.append(cands)
        votes.append(cands[0][0] if cands else None)
        hi = min(len(mags) - 1, int(_FMAX * _MWIN / sr))
        band = mags[lo:hi + 1]
        if prev_band is None:
            flux.append(0.0)
        else:
            num = sum(max(0.0, b - a)
                      for a, b in zip(prev_band, band, strict=True))
            den = sum(prev_band) or 1.0
            flux.append(num / den)
        prev_band = band
    if not votes:
        return Melody()
    loudest = max(peaks) or 1.0
    frame_s = _MHOP / sr
    onsets = _detect_onsets(flux)
    bounds = onsets + [len(votes)]
    long_frames = int(_LONG_SEGMENT_S / frame_s)
    sub_frames = max(2, int(_SUBWINDOW_S / frame_s))
    raw: list[MelodyNote] = []
    for k in range(len(bounds) - 1):
        seg_start, seg_end = bounds[k], bounds[k + 1]
        # Split over-long segments (legato phrases) into sub-windows.
        cuts = [seg_start]
        while seg_end - cuts[-1] > long_frames:
            cuts.append(cuts[-1] + sub_frames)
        cuts.append(seg_end)
        for c in range(len(cuts) - 1):
            w_start, w_end = cuts[c], cuts[c + 1]
            span = w_end - w_start
            if span < _MIN_NOTE_FRAMES:
                continue
            head = max(1, int(math.ceil(0.6 * span)))
            if not any(v is not None
                       for v in votes[w_start:w_start + head]):
                continue
            strength = sum(peaks[w_start:w_start + head]) / head
            if strength < _VOICE_PEAK_RATIO * loudest:
                continue
            primary, alt = _segment_candidates(
                cand_frames[w_start:w_start + head],
                peaks[w_start:w_start + head])
            if primary is None:
                continue
            raw.append(MelodyNote(
                midi=primary, alt_midi=alt,
                start_s=round(w_start * frame_s, 3),
                dur_s=round(span * frame_s, 3)))
    # Merge consecutive same-pitch segments into single notes.
    notes: list[MelodyNote] = []
    for note in raw:
        if notes and notes[-1].midi == note.midi:
            prev = notes[-1]
            notes[-1] = MelodyNote(
                midi=prev.midi, start_s=prev.start_s,
                dur_s=round(prev.dur_s + note.dur_s, 3),
                alt_midi=prev.alt_midi)
        else:
            notes.append(note)
    total = sum(chroma)
    chroma_t = tuple(c / total for c in chroma) if total > 0 \
        else tuple(0.0 for _ in range(12))
    return Melody(notes=tuple(notes), chroma=chroma_t,
                  duration_s=round(n / sr, 3))


def melody_from_path(path: str | os.PathLike[str], *,
                     prefer_stem: bool = True
                     ) -> tuple[Melody, str]:
    """Decode a file → (melody, feature source name).

    Stems-assisted where Demucs is available (the eval's SR3 shape):
    when the separator probes available, extraction runs on the
    vocals stem; any stem failure falls back to the full mix and the
    returned source name says which path produced the melody. Raises
    :class:`AudioReadError` only when the file itself is unreadable.
    """
    samples, sr = read_mono(path)
    source = "f0_contour_full_mix"
    if prefer_stem:
        stem = _vocals_stem_samples(path)
        if stem is not None:
            samples, sr = stem
            source = "demucs_vocals_stem_f0_contour"
    return extract_melody(samples, sr), source


def _vocals_stem_samples(path: str | os.PathLike[str]
                         ) -> tuple[Any, int] | None:
    """Vocals-stem samples when Demucs is installed; else None.

    Fail-closed in the stems module's own idiom: probe first, and
    any failure returns None so the caller uses the full mix — a
    missing separator is a fact about the host, not an error.
    """
    try:
        from ..media.stems import probe_demucs, separate

        if not probe_demucs().available:
            return None
        tmp = tempfile.mkdtemp(prefix="devon-melodic-")
        stems = separate(str(path), out_dir=tmp)
        if not stems.ok or not stems.vocals_path:
            return None
        return read_mono(stems.vocals_path)
    except Exception as exc:  # noqa: BLE001 - fallback is the design
        _log.debug("vocals stem unavailable for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# alignment — Smith-Waterman over interval sequences
# ---------------------------------------------------------------------------

def _interval_sim(a: int, b: int) -> float:
    d = abs(a - b)
    if d == 0:
        return _SIM_EXACT
    if d == 1:
        return _SIM_OFF_BY_ONE
    return _SIM_OTHER


_BUCKET_ORDER = {"s": 0, "m": 1, "l": 2}


def _duration_adj(a: str, b: str) -> float:
    d = abs(_BUCKET_ORDER.get(a, 1) - _BUCKET_ORDER.get(b, 1))
    if d == 0:
        return _DUR_SAME
    if d == 1:
        return _DUR_ADJACENT
    return _DUR_FAR


#: Score for the first aligned pair after a skipped note (an
#: ornament in the cover, a dropped note in the hum): the interval
#: across the skip carries no evidence, so the resumed pair earns a
#: partial, fixed credit — and the skip itself pays _GAP.
_RESUME = 0.5
#: Cost of reading a note on its alternate candidate, per side.
#: The alternate is the segment's less-supported pitch; using it is
#: legitimate (it rescues register flips) but never free — without
#: the toll, two candidates per note let the aligner spell almost
#: any interval sequence out of a polyphonic catalog's chord tones
#: (measured: wrong-work fractions to 0.90 on the probe battery).
_ALT_COST = 0.45


def _note_candidates(note: MelodyNote) -> tuple[int, ...]:
    if note.alt_midi is not None and note.alt_midi != note.midi:
        return (note.midi, note.alt_midi)
    return (note.midi,)


def align_melodies(query: Melody, work: Melody) -> dict[str, Any] | None:
    """Best local alignment of the query's melody against a work's.

    Smith–Waterman over NOTE pairs where each note carries up to
    two candidate pitches (see :class:`MelodyNote`): a state is an
    aligned pair under one candidate choice per side, and a diagonal
    step scores the interval implied by the *chosen* candidates on
    both sides of the step. Extraction flips (a segment read on the
    pad tone instead of the melody) therefore cost nothing when the
    true pitch was the segment's other candidate. Steps may skip
    one note on either side (gap + resume credit, the skipped
    interval unscored). Any state may also start a fresh local
    alignment at zero — the best-scoring contiguous correspondence
    is what is reported.

    Returns the alignment evidence dict, or None when either side
    has fewer than two notes. ``fraction`` is the best path score
    over (max cell x query interval count) — the share of the
    query's melody the alignment accounts for, in [0, 1].
    """
    qn, rn = query.notes, work.notes
    if len(qn) < 2 or len(rn) < 2:
        return None
    qc = [_note_candidates(note) for note in qn]
    rc = [_note_candidates(note) for note in rn]
    qb = duration_buckets(query)
    rb = duration_buckets(work)
    n, m = len(qn), len(rn)
    # h[i][j][a][b]: best score of a local alignment whose last
    # aligned pair is query note i (candidate a) with work note j
    # (candidate b). ptr mirrors it with the winning predecessor:
    # (kind, pa, pb) with kind in {"diag", "skipq", "skipr"}, or
    # None when the state starts an alignment.
    h: list[list[list[list[float]]]] = [
        [[[0.0, 0.0], [0.0, 0.0]] for _ in range(m)] for _ in range(n)]
    ptr: list[list[list[list[Any]]]] = [
        [[[None, None], [None, None]] for _ in range(m)]
        for _ in range(n)]
    best, bi, bj, ba, bb = 0.0, 0, 0, 0, 0
    for i in range(n):
        for j in range(m):
            for a in range(len(qc[i])):
                for b in range(len(rc[j])):
                    value, back = 0.0, None
                    if i >= 1 and j >= 1:
                        for pa in range(len(qc[i - 1])):
                            for pb in range(len(rc[j - 1])):
                                prev = h[i - 1][j - 1][pa][pb]
                                cell = _interval_sim(
                                    qc[i][a] - qc[i - 1][pa],
                                    rc[j][b] - rc[j - 1][pb]) \
                                    + _duration_adj(qb[i - 1], rb[j - 1])
                                cand = prev + cell
                                if cand > value:
                                    value, back = cand, ("diag", pa, pb)
                    if i >= 2 and j >= 1:
                        for pa in range(len(qc[i - 2])):
                            for pb in range(len(rc[j - 1])):
                                cand = h[i - 2][j - 1][pa][pb] \
                                    + _GAP + _RESUME
                                if cand > value:
                                    value, back = cand, ("skipq", pa, pb)
                    if i >= 1 and j >= 2:
                        for pa in range(len(qc[i - 1])):
                            for pb in range(len(rc[j - 2])):
                                cand = h[i - 1][j - 2][pa][pb] \
                                    + _GAP + _RESUME
                                if cand > value:
                                    value, back = cand, ("skipr", pa, pb)
                    # Reading either side on its alternate candidate
                    # pays the toll — including for a start state.
                    value -= (a > 0) * _ALT_COST + (b > 0) * _ALT_COST
                    h[i][j][a][b] = value
                    ptr[i][j][a][b] = back
                    if value > best:
                        best, bi, bj, ba, bb = value, i, j, a, b
    if best <= 0.0:
        return None
    # Traceback: aligned note pairs (with chosen candidates) and
    # the interval steps that were scored diagonally.
    states: list[tuple[int, int, int, int]] = []
    steps: list[float] = []
    dur_steps: list[float] = []
    i, j, a, b = bi, bj, ba, bb
    while True:
        states.append((i, j, a, b))
        back = ptr[i][j][a][b]
        if back is None:
            break
        kind, pa, pb = back
        if kind == "diag":
            steps.append(_interval_sim(qc[i][a] - qc[i - 1][pa],
                                       rc[j][b] - rc[j - 1][pb]))
            dur_steps.append(_duration_adj(qb[i - 1], rb[j - 1]))
            i, j, a, b = i - 1, j - 1, pa, pb
        elif kind == "skipq":
            i, j, a, b = i - 2, j - 1, pa, pb
        else:
            i, j, a, b = i - 1, j - 2, pa, pb
    states.reverse()
    steps.reverse()
    aligned = sum(1 for s in steps if s >= _SIM_OFF_BY_ONE)
    n_intervals = n - 1
    fraction = max(0.0, min(1.0, best / (_CELL_MAX * n_intervals)))
    transpositions = [qc[i][a] - rc[j][b] for i, j, a, b in states]
    med_transposition = sorted(transpositions)[len(transpositions) // 2] \
        if transpositions else 0
    transposition_mad = sorted(
        abs(t - med_transposition) for t in transpositions)[
            len(transpositions) // 2] if transpositions else 0
    dur_agreement = (sum(1 for d in dur_steps if d >= _DUR_SAME)
                     / len(dur_steps)) if dur_steps else 0.0
    tempo_ratios = [qn[i].dur_s / rn[j].dur_s
                    for i, j, _a, _b in states if rn[j].dur_s > 0]
    transposition = int(round(sorted(transpositions)[
        len(transpositions) // 2])) if transpositions else 0
    tempo_ratio = round(sorted(tempo_ratios)[len(tempo_ratios) // 2], 2) \
        if tempo_ratios else 0.0
    q_first, w_first = states[0][0], states[0][1]
    q_last, w_last = states[-1][0], states[-1][1]
    # Chroma agreement at the excerpt's own key relationship: the
    # rotation of the work's chroma that best matches the query's.
    # Deliberately NOT forced to the melodic path's transposition —
    # the path can lock onto a shadow line of the same composition
    # (measured: path +3, chroma 0.67, while at the true relationship
    # 0 the same pair's chroma is 0.96); each representation is an
    # independent witness and the receipt reports both.
    chroma_cos = 0.0
    chroma_t = 0
    if query.chroma and work.chroma:
        for t in range(12):
            rotated = tuple(work.chroma[(k - t) % 12]
                            for k in range(12))
            cos = _cosine(query.chroma, rotated)
            if cos > chroma_cos:
                chroma_cos, chroma_t = cos, t
    if chroma_t > 6:
        chroma_t -= 12
    return {
        "fraction": round(fraction, 3),
        "aligned_intervals": aligned,
        "query_intervals": n_intervals,
        "work_intervals": m - 1,
        "transposition_semitones": transposition,
        "transposition_mad": transposition_mad,
        "chroma_transposition_semitones": chroma_t,
        "duration_agreement": round(dur_agreement, 3),
        "query_span": states[-1][0] - states[0][0] + 1 if states else 0,
        "work_span": states[-1][1] - states[0][1] + 1 if states else 0,
        "tempo_ratio": tempo_ratio,
        "chroma_cosine": round(chroma_cos, 3),
        "query_window": {
            "query_start_s": qn[q_first].start_s,
            "query_end_s": round(qn[q_last].start_s
                                 + qn[q_last].dur_s, 3),
            "reference_start_s": rn[w_first].start_s,
            "reference_end_s": round(rn[w_last].start_s
                                     + rn[w_last].dur_s, 3),
        },
    }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b, strict=False))
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# the melody catalog — works derived from enrolled recordings
# ---------------------------------------------------------------------------

def _default_catalog_path() -> str:
    d = Path.home() / ".nomorals" / "audio"
    with contextlib.suppress(OSError):
        d.mkdir(parents=True, exist_ok=True)
    return str(d / "melodies.db")


def _query_melody_hash(melody: Melody) -> str:
    """Stable hash of a melody's interval shape, not of raw audio."""
    canonical = ";".join(
        f"{iv}/{b}" for iv, b in zip(melody_intervals(melody),
                                     duration_buckets(melody),
                                     strict=False))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MelodyCatalog:
    """Devon's melody memory: the works she can hypothesize about.

    One row per enrolled work: its segmented melody (notes, chroma),
    derived from a recording at enrollment time. Deliberately a
    sibling of :class:`FingerprintDB`, not an extension of it — the
    two catalogs answer different questions (recording vs work) and
    a class-3 row must never be able to masquerade as class-1
    evidence. ``match`` never raises.
    """

    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path or _default_catalog_path()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path,
                                         check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS works ("
                "work_id TEXT PRIMARY KEY, title TEXT, artist TEXT, "
                "source TEXT, track_id INTEGER, notes TEXT, "
                "chroma TEXT, feature_source TEXT, created REAL)")
            self._conn.commit()
        return self._conn

    # -- enrollment ----------------------------------------------------

    def add_work(self, path: str | os.PathLike[str], *,
                 title: str = "", artist: str = "",
                 source: str = "library",
                 track_id: int | None = None,
                 prefer_stem: bool = True) -> dict[str, Any]:
        """Extract ``path``'s melody and enroll it as a work.

        Never raises. A hum or query file is enrolled only when a
        caller explicitly enrolls it as a work — matching never
        enrolls.
        """
        try:
            melody, feature_source = melody_from_path(
                path, prefer_stem=prefer_stem)
            if len(melody.notes) < 3:
                return {"ok": False,
                        "reason": "too little melody extracted — "
                                  f"{len(melody.notes)} note(s) heard; "
                                  "a work needs a melody line"}
            title = title or Path(str(path)).stem
            artist = artist or "unknown"
            if track_id is not None:
                work_id = f"fp-track-{int(track_id)}"
            else:
                # Keyed by (title, artist): re-enrolling a work
                # supersedes its earlier extraction, never duplicates
                # it in the hypothesis list.
                digest = hashlib.sha256(
                    f"{title}|{artist}".encode()
                ).hexdigest()[:16]
                work_id = f"mel-{digest}"
            notes_json = json.dumps(
                [[n.midi, n.start_s, n.dur_s, n.alt_midi]
                 for n in melody.notes])
            with self._lock:
                conn = self._connect()
                conn.execute(
                    "INSERT OR REPLACE INTO works (work_id, title, "
                    "artist, source, track_id, notes, chroma, "
                    "feature_source, created) VALUES (?,?,?,?,?,?,?,?,?)",
                    (work_id, title, artist, source, track_id,
                     notes_json, json.dumps(list(melody.chroma)),
                     feature_source, time.time()))
                conn.commit()
            return {"ok": True, "work_id": work_id,
                    "track_id": track_id, "title": title,
                    "notes": len(melody.notes),
                    "intervals": len(melody.notes) - 1,
                    "feature_source": feature_source}
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            _log.debug("add_work failed", exc_info=True)
            return {"ok": False, "reason": f"enrollment failed: {exc}"}

    def enroll_fingerprint_track(
            self, fdb: FingerprintDB, track_id: int,
            path: str | os.PathLike[str], *,
            prefer_stem: bool = True) -> dict[str, Any]:
        """Derive a work from a recording already enrolled in the
        fingerprint DB. The work inherits the recording's track id,
        title and artist, so the class-1 row and its class-3 melody
        stay linked without ever sharing an index. Never raises."""
        try:
            meta = next((t for t in fdb.list_tracks()
                         if int(t.get("track_id") or 0) == int(track_id)),
                        None)
            if meta is None:
                return {"ok": False,
                        "reason": f"no fingerprint track {track_id} "
                                  "to derive a work from"}
            return self.add_work(
                path, title=str(meta.get("title") or ""),
                artist=str(meta.get("artist") or ""),
                source=str(meta.get("source") or "library"),
                track_id=int(track_id), prefer_stem=prefer_stem)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False,
                    "reason": f"fingerprint-track enrollment failed: {exc}"}

    # -- catalog facts ---------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Catalog size at a glance. Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                works = conn.execute(
                    "SELECT COUNT(*) FROM works").fetchone()[0]
                rows = conn.execute("SELECT notes FROM works").fetchall()
            intervals = sum(max(0, len(json.loads(r[0])) - 1)
                            for r in rows)
            return {"ok": True, "works": works, "intervals": intervals}
        except Exception as exc:  # noqa: BLE001
            record_failure("melodic.stats", exc)
            return {"ok": False, "works": 0, "intervals": 0}

    def list_works(self) -> list[dict[str, Any]]:
        """Every enrolled work. Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                rows = conn.execute(
                    "SELECT work_id, title, artist, source, track_id, "
                    "notes, feature_source FROM works ORDER BY created"
                ).fetchall()
            return [{"work_id": r[0], "title": r[1], "artist": r[2],
                     "source": r[3], "track_id": r[4],
                     "notes": len(json.loads(r[5])),
                     "feature_source": r[6]} for r in rows]
        except Exception as exc:  # noqa: BLE001
            record_failure("melodic.list_works", exc)
            return []

    def catalog_snapshot(self) -> dict[str, Any]:
        """Catalog stamp for a receipt (same digest idiom as the
        fingerprint catalog). Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                rows = conn.execute(
                    "SELECT work_id, title, artist, notes FROM works "
                    "ORDER BY work_id").fetchall()
            canonical = "\n".join(
                "|".join(str(v) for v in row) for row in rows)
            digest = hashlib.sha256(
                canonical.encode("utf-8")).hexdigest()
            intervals = sum(max(0, len(json.loads(r[3])) - 1)
                            for r in rows)
            return {"version": f"local-melody-v1:sha256:{digest}",
                    "works": len(rows), "intervals": intervals}
        except Exception as exc:  # noqa: BLE001
            record_failure("melodic.catalog_snapshot", exc)
            return {"version": "", "works": 0, "intervals": 0}

    def remove_work(self, work_id: str) -> bool:
        """Drop one enrolled work. Never raises."""
        try:
            with self._lock:
                conn = self._connect()
                cur = conn.execute(
                    "DELETE FROM works WHERE work_id = ?", (work_id,))
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:  # noqa: BLE001
            record_failure("melodic.remove_work", exc)
            return False

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception as exc:  # noqa: BLE001
            record_failure("melodic.close", exc)
        finally:
            self._conn = None

    # -- matching --------------------------------------------------------

    def _works_locked(self, conn: sqlite3.Connection
                      ) -> list[tuple[dict[str, Any], Melody]]:
        rows = conn.execute(
            "SELECT work_id, title, artist, track_id, notes, chroma, "
            "feature_source FROM works").fetchall()
        out: list[tuple[dict[str, Any], Melody]] = []
        for r in rows:
            notes = tuple(MelodyNote(
                midi=int(n[0]), start_s=float(n[1]),
                dur_s=float(n[2]),
                alt_midi=(int(n[3]) if len(n) > 3 and n[3] is not None
                          else None))
                for n in json.loads(r[4]))
            chroma = tuple(float(c) for c in json.loads(r[5]))
            meta = {"work_id": r[0], "title": r[1], "artist": r[2],
                    "track_id": r[3], "feature_source": r[6]}
            out.append((meta, Melody(notes=notes, chroma=chroma)))
        return out

    def rank_hypotheses(self, query: Melody, *,
                        limit: int = 5) -> list[dict[str, Any]]:
        """Ranked melodic hypotheses for an extracted query melody.

        Only works clearing the hypothesis floor are returned — an
        unrelated melody gets an empty list, never a forced match.
        Never raises.
        """
        try:
            if len(melody_intervals(query)) < MIN_QUERY_INTERVALS:
                return []
            with self._lock:
                works = self._works_locked(self._connect())
            ranked: list[dict[str, Any]] = []
            for meta, work in works:
                ev = align_melodies(query, work)
                if ev is None:
                    continue
                if ev["fraction"] < MIN_ALIGNMENT_FRACTION \
                        or ev["aligned_intervals"] < MIN_ALIGNED_INTERVALS \
                        or ev["chroma_cosine"] < MIN_CHROMA_COSINE \
                        or ev["transposition_mad"] > MAX_TRANSPOSITION_MAD \
                        or ev["aligned_intervals"] \
                        < MIN_ALIGNED_COVERAGE * ev["query_intervals"]:
                    continue
                ranked.append({**meta, **ev})
            ranked.sort(key=lambda h: (-h["fraction"],
                                       -h["aligned_intervals"],
                                       -h["chroma_cosine"]))
            out: list[dict[str, Any]] = []
            for rank, h in enumerate(ranked[:limit], start=1):
                out.append({
                    "rank": rank,
                    "artist": h["artist"],
                    "title": h["title"],
                    "work_id": h["work_id"],
                    "recording_id": (str(h["track_id"])
                                     if h["track_id"] is not None else ""),
                    "match_strength": {
                        "name": ALIGNMENT_NAME,
                        "value": h["fraction"],
                        "numerator": h["aligned_intervals"],
                        "denominator": h["query_intervals"],
                        "unit": ALIGNMENT_UNIT,
                        "basis": ALIGNMENT_BASIS,
                        "calibration": ALIGNMENT_CALIBRATION,
                    },
                    "aligned_intervals": h["aligned_intervals"],
                    "query_intervals": h["query_intervals"],
                    "transposition_semitones":
                        h["transposition_semitones"],
                    "transposition_mad": h["transposition_mad"],
                    "chroma_transposition_semitones":
                        h["chroma_transposition_semitones"],
                    "duration_agreement": h["duration_agreement"],
                    "tempo_ratio": h["tempo_ratio"],
                    "chroma_cosine": h["chroma_cosine"],
                    "work_feature_source": h["feature_source"],
                    "evidence_windows": [h["query_window"]],
                })
            return out
        except Exception as exc:  # noqa: BLE001
            record_failure("melodic.rank_hypotheses", exc)
            return []

    def match(self, path: str | os.PathLike[str], *,
              limit: int = 5, prefer_stem: bool = True
              ) -> dict[str, Any]:
        """Hypothesize about one audio file against the catalog.

        Returns ``{"ok": bool, "hypotheses": [...], "receipt": {...}}``
        — ``ok`` means *hypotheses exist*, never that a work was
        identified. Never raises.
        """
        try:
            melody, feature_source = melody_from_path(
                path, prefer_stem=prefer_stem)
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc),
                    "receipt": self._receipt(
                        status="unavailable", reason=str(exc),
                        query=None, hypotheses=(),
                        feature_source="")}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"melody match failed: {exc}",
                    "receipt": self._receipt(
                        status="unavailable",
                        reason=f"melody match failed: {exc}",
                        query=None, hypotheses=(), feature_source="")}
        hypotheses = self.rank_hypotheses(melody, limit=limit)
        if hypotheses:
            return {"ok": True, "hypotheses": hypotheses,
                    "receipt": self._receipt(
                        status="hypotheses", reason="",
                        query=melody, hypotheses=tuple(hypotheses),
                        feature_source=feature_source)}
        n_intervals = len(melody_intervals(melody))
        if n_intervals < MIN_QUERY_INTERVALS:
            reason = (f"too little melody in the query to hypothesize "
                      f"from ({n_intervals} interval(s) extracted; "
                      f"{MIN_QUERY_INTERVALS} needed)")
        else:
            snap = self.catalog_snapshot()
            reason = (f"no enrolled work cleared the hypothesis "
                      f"floor (interval alignment fraction >= "
                      f"{MIN_ALIGNMENT_FRACTION}, >= "
                      f"{MIN_ALIGNED_INTERVALS} aligned intervals, "
                      f"chroma cosine >= {MIN_CHROMA_COSINE} at its "
                      f"best key rotation, transposition drift "
                      f"<= {MAX_TRANSPOSITION_MAD} semitones, aligned "
                      f"coverage >= {MIN_ALIGNED_COVERAGE}) across "
                      f"{snap['works']} enrolled work(s)")
        return {"ok": False, "hypotheses": [], "reason": reason,
                "receipt": self._receipt(
                    status="no_match", reason=reason, query=melody,
                    hypotheses=(), feature_source=feature_source)}

    def _receipt(self, *, status: str, reason: str,
                 query: Melody | None,
                 hypotheses: tuple[dict[str, Any], ...],
                 feature_source: str) -> dict[str, Any]:
        """Assemble the class-3 receipt. Hypotheses ride in
        ``alternatives`` — there is no winner slot to fill, and the
        status is never "matched"."""
        snap = self.catalog_snapshot()
        sources = ("f0_melody_contour", "chroma")
        if feature_source.startswith("demucs"):
            sources = ("demucs_vocals_stem",) + sources
        top = hypotheses[0] if hypotheses else {}
        return RecognitionReceipt(
            status=status,
            source="devon_local_melody",
            source_label="Devon's local melody catalog",
            match_class=(MATCH_CLASS_MELODIC_HYPOTHESIS
                         if status == "hypotheses"
                         else "no_reliable_match"
                         if status == "no_match" else "unavailable"),
            catalog_version=str(snap["version"]),
            catalog_track_count=int(snap["works"]),
            catalog_hash_count=int(snap["intervals"]),
            query_hash=_query_melody_hash(query) if query else "",
            query_fingerprint_count=(len(query.notes) if query else 0),
            match_strength=dict(top.get("match_strength") or {}),
            evidence_windows=tuple(top.get("evidence_windows") or ()),
            alignment={"method": "smith_waterman_candidate_"
                                 "interval_alignment",
                       "transposition_invariant": True,
                       "hypothesis_floor": {
                           "min_fraction": MIN_ALIGNMENT_FRACTION,
                           "min_aligned_intervals": MIN_ALIGNED_INTERVALS,
                           "min_chroma_cosine": MIN_CHROMA_COSINE,
                           "max_transposition_mad":
                               MAX_TRANSPOSITION_MAD,
                           "min_aligned_coverage": MIN_ALIGNED_COVERAGE,
                       }},
            feature_sources=sources,
            alternatives=tuple(hypotheses),
            attempted_sources=("devon_local_melody",),
            rights_note=LOCAL_RIGHTS_NOTE,
            no_match_reason=reason if status == "no_match" else "",
            reason=reason,
        ).to_dict()


# ---------------------------------------------------------------------------
# module-level API — mirroring the fingerprint module's idiom
# ---------------------------------------------------------------------------

def default_melody_catalog(db_path: str = "") -> MelodyCatalog:
    """The shared local melody memory."""
    return MelodyCatalog(db_path or _default_catalog_path())


def enroll_work(path: str | os.PathLike[str], *, title: str = "",
                artist: str = "", source: str = "library",
                catalog: MelodyCatalog | None = None) -> dict[str, Any]:
    """Enroll one recording as a melodic work. Never raises."""
    try:
        return (catalog or default_melody_catalog()).add_work(
            path, title=title, artist=artist, source=source)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"enrollment failed: {exc}"}


def match_melody_db(path: str | os.PathLike[str], *,
                    catalog: MelodyCatalog | None = None,
                    limit: int = 5) -> dict[str, Any]:
    """Ranked melodic hypotheses for ``path``. Never raises.

    ``ok`` in the result means hypotheses exist — it is not, and
    never renders as, an identification.
    """
    try:
        return (catalog or default_melody_catalog()).match(
            path, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "hypotheses": [],
                "reason": f"melody match failed: {exc}"}


def render_hypothesis_lines(hypotheses: list[dict[str, Any]], *,
                            limit: int = 3) -> list[str]:
    """Render ranked melodic hypotheses for chat surfaces.

    The wording is the doctrine: hypotheses with their evidence in
    its own units — never an identification, never a probability,
    never a /play line. Shared by /sham and /audio so every surface
    says it the same way.
    """
    lines: list[str] = []
    for idx, h in enumerate(hypotheses[:limit], 1):
        artist = h.get("artist") or "unknown artist"
        title = h.get("title") or "unknown title"
        transposition = h.get("transposition_semitones", 0)
        sign = "+" if isinstance(transposition, (int, float)) \
            and transposition > 0 else ""
        strength = h.get("match_strength") or {}
        fraction = strength.get("value", h.get("fraction"))
        lines.append(
            f"{idx}. {artist} — {title}: interval alignment "
            f"fraction {fraction} — "
            f"{h.get('aligned_intervals')} of "
            f"{h.get('query_intervals')} query melody intervals "
            f"aligned; key relationship {sign}{transposition} "
            f"semitones; query notes ≈{h.get('tempo_ratio')}× the "
            f"work's durations; chroma cosine "
            f"{h.get('chroma_cosine')}")
    lines.append(
        "These are melodic hypotheses — the tune resembles the "
        "enrolled work — not an identification of the recording. "
        "Cover/hum accuracy on real recordings is unproven; treat "
        "them as candidates to check.")
    return lines
