"""SongRecon §9 benchmark — measured robustness for the local matcher.

Doc 24 SR2. The shipped class-1 engine (``audio/fingerprint.py``,
Wang-2003 constellation landmarks over Devon's own catalog) carried an
uncalibrated coherent-offset fraction and *no* measured tolerance:
nobody knew which degradations it survives. This harness turns that
assertion into numbers, offline and repeatably:

* a ground-truth catalog of deterministically synthesized tracks
  (seeded compositions — no downloads, no external corpora);
* a §9 degradation battery over excerpts of those tracks: clip length,
  background noise, room reverb, codec/bitrate round-trips, EQ /
  microphone response, coupled speed+pitch edits, pitch-preserving
  tempo edits, gain, clipping, and background-music overlap;
* per-condition top-1 / top-5 accuracy, false accepts, median query
  time, and the empirical calibration of the coherent-offset fraction
  (P(top-1 correct) per score bin, n stated per bin);
* a false-candidate rate over non-catalog negative queries.

The harness drives the REAL engine end to end (``FingerprintDB.add_track``
/ ``match`` / ``rank_candidates``) — nothing is mocked, so the numbers
indict or vindicate the shipped matcher itself. The report is a
date-stamped JSON + Markdown artifact written to the caller's output
directory; it is evidence, not a surface claim.

Run it::

    python -m nomorals.audio.songrecon_bench --out ~/bench-out
    python -m nomorals.audio.songrecon_bench --out ~/bench-out --quick

Fixture-design note (measured, 2026-11-10): a first version of this
harness synthesized tracks on small discrete lattices (13 registers,
3 tempo grids, exact-ratio sustained pads). Distinct "songs" then
shared their dominant landmark stream and the false-candidate rate
read 0.90 — a fixture artifact, not an engine measurement (distinct
query occurrences genuinely aligned against wrong tracks). The
shipped generator therefore uses continuous registers/timing,
humanized onsets, detune and vibrato: distinct compositions share
~no exact landmarks, as distinct real recordings don't. The
re-performance *sibling* pool keeps the honest hard case in view,
reported separately.

Determinism: every fixture derives from integer seeds via SHA-256
sub-seeding, so the same spec regenerates byte-identical fixtures and
identical results on the same host/toolchain.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import wave
from array import array
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .fingerprint import (
    MATCH_THRESHOLD,
    FingerprintDB,
    _ffmpeg_path,
    fingerprint,
    read_mono,
)

BENCH_SCHEMA = "songrecon.bench/v1"
SAMPLE_RATE = 11025

_CATALOG_SEED0 = 20000
_NEGATIVE_SEED0 = 40000
_BACKGROUND_SEED = 60000


def _subseed(*parts: object) -> int:
    """Stable integer seed from parts (never Python's salted hash)."""
    text = ":".join(str(p) for p in parts)
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------
# deterministic ground-truth compositions
# ---------------------------------------------------------------------------

def _composition(seed: int, seconds: float) -> dict[str, Any]:
    """The seeded *notes* of one pseudo-song, performance-free.

    Splitting composition from performance is what makes the
    benchmark's hardest negatives possible: a sibling query renders
    the SAME composition with a different performance seed (humanized
    timing, detune, vibrato, timbre drift) — a re-performance, not
    the same recording.
    """
    import random

    rng = random.Random(seed)
    base_root = 75.0 * (2.0 ** rng.uniform(0.0, 1.2))
    slot = rng.uniform(0.38, 0.65)
    n_chords = rng.choice([3, 4, 4, 5])
    qualities = [(1.0, 1.25, 1.5), (1.0, 1.20, 1.5), (1.0, 1.335, 1.5)]
    prog = []
    for _ in range(n_chords):
        semi = rng.choice([0, 2, 4, 5, 7, 9])
        prog.append((base_root * (2.0 ** (semi / 12.0)),
                     rng.choice(qualities)))
    seg = seconds / n_chords
    events: list[tuple[float, float, float, str]] = []
    motif: list[tuple[float, float]] | None = None
    for ci, (root, quality) in enumerate(prog):
        start = ci * seg
        chord = [root * 2.0 * q for q in quality] + [root * 4.0]
        scale = chord + [root * 2.0 * 1.125, root * 2.0 * 1.75]
        steps = max(2, int(seg / slot))
        walk = rng.randrange(len(scale))
        notes: list[tuple[float, float]] = []
        for k in range(steps):
            if ci == 2 and motif is not None and k < len(motif):
                f, dur = motif[k]
            else:
                walk = max(0, min(len(scale) - 1,
                                  walk + rng.choice([-2, -1, 0, 1, 2])))
                f = scale[walk] * rng.choice([1.0, 1.0, 1.0, 2.0])
                dur = slot * rng.uniform(0.8, 1.5)
            notes.append((f, dur))
        if ci == 0:
            motif = notes
        for k, (f, dur) in enumerate(notes):
            events.append((start + k * slot, dur, f, "melody"))
            events.append((start + k * slot, slot * 0.9, root, "bass"))
        events.append((start, seg, chord[0], "pad"))
        events.append((start, seg, chord[2], "pad"))
    return {"events": events, "seconds": seconds}


def _render(comp: dict[str, Any], perf_seed: int) -> array:
    """Render a composition with one performance's micro-structure:
    humanized onsets (±30 ms), per-track detune (±15 cents) and
    per-note detune, melody vibrato, per-note timbre/velocity drift,
    per-performance pad tremolo rate. Continuous parameters (never
    small lattices) are deliberate: two different compositions must
    share ~no exact landmark stream, as two real songs don't."""
    import random

    rng = random.Random(perf_seed)
    seconds = float(comp["seconds"])
    n = int(SAMPLE_RATE * seconds)
    out = [0.0] * n
    detune = 2.0 ** (rng.uniform(-15.0, 15.0) / 1200.0)
    w2 = rng.uniform(0.30, 0.60)
    w3 = rng.uniform(0.10, 0.35)
    trem_hz = rng.uniform(0.2, 0.5)
    for start_s, dur_s, freq, kind in comp["events"]:
        jitter = rng.uniform(-0.030, 0.030) if kind != "pad" else 0.0
        f = freq * detune * (2.0 ** (rng.uniform(-6.0, 6.0) / 1200.0))
        vel = rng.uniform(0.8, 1.15)
        ns = max(0, int((start_s + jitter) * SAMPLE_RATE))
        dn = int(dur_s * SAMPLE_RATE)
        attack = max(1, int(0.015 * SAMPLE_RATE))
        vib_hz = rng.uniform(4.0, 6.0)
        vib_depth = rng.uniform(0.001, 0.003) if kind == "melody" else 0.0
        nw2 = w2 * rng.uniform(0.85, 1.15)
        nw3 = w3 * rng.uniform(0.85, 1.15)
        for i in range(dn):
            idx = ns + i
            if idx >= n:
                break
            t = i / SAMPLE_RATE
            if kind == "melody":
                env = math.exp(-2.5 * i / dn) * min(1.0, i / attack)
                vib = 1.0 + vib_depth * math.sin(2.0 * math.pi * vib_hz * t)
                ph = 2.0 * math.pi * f * vib * i / SAMPLE_RATE
                out[idx] += env * 0.30 * vel * (
                    math.sin(ph) + nw2 * math.sin(2.0 * ph)
                    + nw3 * math.sin(3.0 * ph))
            elif kind == "bass":
                env = math.exp(-1.5 * i / dn)
                out[idx] += env * 0.16 * vel * math.sin(
                    2.0 * math.pi * f * i / SAMPLE_RATE)
            else:  # pad
                trem = 0.75 + 0.25 * math.sin(2.0 * math.pi * trem_hz * t)
                out[idx] += 0.045 * trem * math.sin(
                    2.0 * math.pi * f * i / SAMPLE_RATE)
    peak = max((abs(v) for v in out), default=0.0) or 1.0
    noise_rng = random.Random(_subseed(perf_seed, "floor"))
    return array("d", [v * 0.72 / peak + noise_rng.gauss(0, 0.0012)
                       for v in out])


def synth_track(seed: int, seconds: float = 20.0) -> array:
    """One deterministic pseudo-recording: composition ``seed``
    performed by performance ``seed``. Different seeds are different
    songs in different registers with continuous timing/tuning, so
    distinct tracks share no long identical landmark stretch."""
    return _render(_composition(seed, seconds), seed)


def synth_sibling(seed: int, seconds: float = 20.0) -> array:
    """The same composition as ``synth_track(seed)``, re-performed
    under a different performance seed — the near-duplicate stress
    negative (same notes, NOT the same recording)."""
    return _render(_composition(seed, seconds), seed + 777000)


def write_wav(path: str | os.PathLike[str], samples: Sequence[float],
              sr: int = SAMPLE_RATE) -> str:
    """Write mono 16-bit WAV. Returns the path as a string."""
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in samples))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return str(path)


# ---------------------------------------------------------------------------
# degradations (doc §9 axes) — pure, seeded, sample-domain unless noted
# ---------------------------------------------------------------------------

def excerpt(samples: Sequence[float], at_s: float, len_s: float) -> array:
    a = int(at_s * SAMPLE_RATE)
    return array("d", samples[a:a + int(len_s * SAMPLE_RATE)])


def add_noise(samples: Sequence[float], snr_db: float, seed: int) -> array:
    import random

    rng = random.Random(seed)
    if not samples:
        return array("d", samples)
    sig = math.sqrt(sum(v * v for v in samples) / len(samples))
    nsig = sig / (10.0 ** (snr_db / 20.0))
    return array("d", [v + rng.gauss(0, nsig) for v in samples])


def mix_background(samples: Sequence[float], background: Sequence[float],
                   snr_db: float) -> array:
    """Overlap: another (non-catalog) composition under the query."""
    if not samples:
        return array("d", samples)
    sig = math.sqrt(sum(v * v for v in samples) / len(samples))
    bsig = math.sqrt(sum(v * v for v in background) / max(1, len(background)))
    if bsig <= 0:
        return array("d", samples)
    gain = (sig / bsig) / (10.0 ** (snr_db / 20.0))
    return array("d", [
        v + gain * background[i % len(background)]
        for i, v in enumerate(samples)])


def apply_reverb(samples: Sequence[float], *, strong: bool,
                 seed: int) -> array:
    """Room reverb. With numpy: convolution against a synthetic
    exponentially decaying noise impulse response. Without: a
    deterministic multi-tap echo mix standing in for the room.
    Either way the same seed reproduces the same output on a host."""
    import random

    rng = random.Random(seed)
    decay_s = 0.9 if strong else 0.35
    wet = 0.45 if strong else 0.25
    try:
        import numpy as np

        ir_len = int(decay_s * SAMPLE_RATE)
        ir = np.array([rng.gauss(0, 1.0) * math.exp(-3.0 * i / ir_len)
                       for i in range(ir_len)], dtype=np.float64)
        dry = np.array(samples, dtype=np.float64)
        wet_sig = np.convolve(dry, ir)[:len(dry)]
        peak = float(np.max(np.abs(wet_sig))) or 1.0
        out = dry + (wet / peak) * wet_sig * (
            float(np.max(np.abs(dry))) or 1.0)
        return array("d", [float(v) for v in out])
    except ImportError:
        out = [float(v) for v in samples]
        taps = [(0.09, 0.50), (0.17, 0.32), (0.29, 0.20), (0.43, 0.12)]
        for delay_s, amp in taps:
            d = int(delay_s * SAMPLE_RATE)
            a = amp * wet / 0.45
            for i in range(d, len(out)):
                out[i] += a * samples[i - d]
        return array("d", out)


def _one_pole(samples: Sequence[float], coeff: float,
              *, highpass: bool = False) -> array:
    out = array("d", [0.0]) * len(samples)
    prev_x = 0.0
    prev_y = 0.0
    for i, x in enumerate(samples):
        y = (coeff * (prev_y + x - prev_x) if highpass
             else prev_y + coeff * (x - prev_y))
        out[i] = y
        prev_x, prev_y = x, y
    return out


def _pole_coeff(cutoff_hz: float) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / SAMPLE_RATE)


def eq_phone(samples: Sequence[float]) -> array:
    """Telephone / small-microphone response: ~300 Hz–3.4 kHz band."""
    hp = _one_pole(samples, _pole_coeff(300.0), highpass=True)
    return _one_pole(hp, _pole_coeff(3400.0))


def eq_bass_boost(samples: Sequence[float]) -> array:
    low = _one_pole(samples, _pole_coeff(250.0))
    return array("d", [v + 1.2 * lv
                       for v, lv in zip(samples, low, strict=True)])


def eq_treble_cut(samples: Sequence[float]) -> array:
    """Dull radio: everything above ~2 kHz gone."""
    return _one_pole(samples, _pole_coeff(2000.0))


def apply_speed(samples: Sequence[float], factor: float) -> array:
    """Coupled speed+pitch edit (the sped-up social clip / DJ pitch-up):
    playing the samples at ``factor``× rate scales pitch by ``factor``
    and durations by 1/``factor``. Linear-interp resample."""
    if factor <= 0 or not samples:
        return array("d", samples)
    n_out = max(1, int(len(samples) / factor))
    last = len(samples) - 1
    out = array("d", [0.0]) * n_out
    for i in range(n_out):
        pos = i * factor
        lo = int(pos)
        hi = min(last, lo + 1)
        frac = pos - lo
        out[i] = samples[lo] * (1.0 - frac) + samples[hi] * frac
    return out


def apply_gain(samples: Sequence[float], factor: float) -> array:
    return array("d", [v * factor for v in samples])


def hard_clip(samples: Sequence[float], ceiling: float) -> array:
    return array("d", [max(-ceiling, min(ceiling, v)) for v in samples])


def soft_clip(samples: Sequence[float], drive: float) -> array:
    return array("d", [math.tanh(drive * v) / math.tanh(drive)
                       for v in samples])


class BenchUnavailable(RuntimeError):
    """A degradation's external tool is missing on this host."""


def _ffmpeg() -> str:
    ff = _ffmpeg_path()
    if not ff:
        raise BenchUnavailable("ffmpeg not installed on this host")
    return ff


def codec_roundtrip(samples: Sequence[float], codec: str, bitrate_k: int,
                    workdir: str) -> array:
    """Encode → decode through a real codec at a real bitrate."""
    ff = _ffmpeg()
    src = write_wav(os.path.join(workdir, "codec-src.wav"), samples)
    if codec == "mp3":
        enc = os.path.join(workdir, "codec.mp3")
        cmd = [ff, "-y", "-i", src, "-codec:a", "libmp3lame",
               "-b:a", f"{bitrate_k}k", "-ac", "1", enc]
    elif codec == "aac":
        enc = os.path.join(workdir, "codec.m4a")
        cmd = [ff, "-y", "-i", src, "-codec:a", "aac",
               "-b:a", f"{bitrate_k}k", "-ac", "1", enc]
    else:
        raise BenchUnavailable(f"unknown codec: {codec}")
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0 or not os.path.isfile(enc):
        raise BenchUnavailable(
            f"ffmpeg {codec} encode failed: "
            + proc.stderr.decode("utf-8", "replace")[-200:])
    decoded, _sr = read_mono(enc)
    return decoded


def tempo_stretch(samples: Sequence[float], factor: float,
                  workdir: str) -> array:
    """Pitch-preserving tempo change via ffmpeg's atempo filter."""
    ff = _ffmpeg()
    src = write_wav(os.path.join(workdir, "tempo-src.wav"), samples)
    out = os.path.join(workdir, "tempo-out.wav")
    proc = subprocess.run(
        [ff, "-y", "-i", src, "-filter:a", f"atempo={factor}",
         "-ac", "1", "-ar", str(SAMPLE_RATE), out],
        capture_output=True, timeout=120)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise BenchUnavailable(
            "ffmpeg atempo failed: "
            + proc.stderr.decode("utf-8", "replace")[-200:])
    decoded, _sr = read_mono(out)
    return decoded


# ---------------------------------------------------------------------------
# condition table
# ---------------------------------------------------------------------------

class _Ctx:
    """Per-run helpers handed to condition builders."""

    def __init__(self, workdir: str, background: array) -> None:
        self.workdir = workdir
        self.background = background


Builder = Callable[[Sequence[float], int, _Ctx], array]


def _at(seconds: float) -> Callable[[Sequence[float]], array]:
    return lambda s: excerpt(s, 8.0, seconds)


def _chain(*fns: Callable[[Sequence[float]], array]
           ) -> Callable[[Sequence[float]], array]:
    def run(samples: Sequence[float]) -> array:
        out = array("d", samples)
        for fn in fns:
            out = fn(out)
        return out
    return run


def _b(fn: Callable[[Sequence[float]], array]) -> Builder:
    return lambda samples, _seed, _ctx: fn(samples)


def _seeded(fn: Callable[[Sequence[float], int], array]) -> Builder:
    return lambda samples, seed, _ctx: fn(samples, seed)


CONDITIONS: list[tuple[str, Builder]] = [
    ("clean_full", _b(lambda s: array("d", s))),
    ("excerpt_10s", _b(_at(10.0))),
    ("excerpt_5s", _b(_at(5.0))),
    ("excerpt_3s", _b(_at(3.0))),
    ("excerpt_2s", _b(_at(2.0))),
    ("noise_snr20", _seeded(
        lambda s, sd: _chain(_at(10.0),
                             lambda x: add_noise(x, 20.0, sd))(s))),
    ("noise_snr10", _seeded(
        lambda s, sd: _chain(_at(10.0),
                             lambda x: add_noise(x, 10.0, sd))(s))),
    ("noise_snr5", _seeded(
        lambda s, sd: _chain(_at(10.0),
                             lambda x: add_noise(x, 5.0, sd))(s))),
    ("noise_snr0", _seeded(
        lambda s, sd: _chain(_at(10.0),
                             lambda x: add_noise(x, 0.0, sd))(s))),
    ("reverb_mild", _seeded(
        lambda s, sd: _chain(
            _at(10.0),
            lambda x: apply_reverb(x, strong=False, seed=sd))(s))),
    ("reverb_strong", _seeded(
        lambda s, sd: _chain(
            _at(10.0),
            lambda x: apply_reverb(x, strong=True, seed=sd))(s))),
    ("codec_mp3_128", lambda s, _sd, ctx: codec_roundtrip(
        _at(10.0)(s), "mp3", 128, ctx.workdir)),
    ("codec_mp3_64", lambda s, _sd, ctx: codec_roundtrip(
        _at(10.0)(s), "mp3", 64, ctx.workdir)),
    ("codec_mp3_32", lambda s, _sd, ctx: codec_roundtrip(
        _at(10.0)(s), "mp3", 32, ctx.workdir)),
    ("codec_aac_64", lambda s, _sd, ctx: codec_roundtrip(
        _at(10.0)(s), "aac", 64, ctx.workdir)),
    ("eq_phone", _b(_chain(_at(10.0), eq_phone))),
    ("eq_bass_boost", _b(_chain(_at(10.0), eq_bass_boost))),
    ("eq_treble_cut", _b(_chain(_at(10.0), eq_treble_cut))),
    ("speed_0.94", _b(_chain(_at(10.0), lambda x: apply_speed(x, 0.94)))),
    ("speed_0.97", _b(_chain(_at(10.0), lambda x: apply_speed(x, 0.97)))),
    ("speed_1.03", _b(_chain(_at(10.0), lambda x: apply_speed(x, 1.03)))),
    ("speed_1.06", _b(_chain(_at(10.0), lambda x: apply_speed(x, 1.06)))),
    ("speed_1.10", _b(_chain(_at(10.0), lambda x: apply_speed(x, 1.10)))),
    ("tempo_0.94", lambda s, _sd, ctx: tempo_stretch(
        _at(10.0)(s), 0.94, ctx.workdir)),
    ("tempo_1.06", lambda s, _sd, ctx: tempo_stretch(
        _at(10.0)(s), 1.06, ctx.workdir)),
    ("gain_quiet", _b(_chain(_at(10.0), lambda x: apply_gain(x, 0.25)))),
    ("clip_hard", _b(_chain(_at(10.0), lambda x: hard_clip(x, 0.4)))),
    ("soft_clip", _b(_chain(_at(10.0), lambda x: soft_clip(x, 3.0)))),
    ("overlap_snr12", lambda s, _sd, ctx: mix_background(
        _at(10.0)(s), excerpt(ctx.background, 3.0, 10.0), 12.0)),
    ("overlap_snr6", lambda s, _sd, ctx: mix_background(
        _at(10.0)(s), excerpt(ctx.background, 3.0, 10.0), 6.0)),
]

QUICK_CONDITIONS = [
    "clean_full", "excerpt_5s", "excerpt_3s", "excerpt_2s",
    "noise_snr10", "noise_snr5", "codec_mp3_64", "eq_phone",
    "speed_0.97", "speed_1.06", "clip_hard", "reverb_mild",
]

NEGATIVE_CONDITIONS = ["excerpt_10s", "noise_snr10", "speed_1.06",
                       "codec_mp3_64"]

_SCORE_BINS = [(0.12, 0.20), (0.20, 0.35), (0.35, 0.50),
               (0.50, 0.70), (0.70, 1.01)]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _condition_map(quick: bool) -> list[tuple[str, Builder]]:
    if not quick:
        return list(CONDITIONS)
    wanted = set(QUICK_CONDITIONS)
    return [(n, b) for n, b in CONDITIONS if n in wanted]


def run_benchmark(out_dir: str | os.PathLike[str], *,
                  n_tracks: int = 12, track_seconds: float = 20.0,
                  quick: bool = False,
                  threshold: float = MATCH_THRESHOLD) -> dict[str, Any]:
    """Run the full §9 battery; write report.json + report.md.

    Returns the report dict. Honest by construction: a condition whose
    tool is unavailable is recorded as unavailable with its reason and
    excluded from every rate (never silently passed, never zeroed).
    """
    out = Path(out_dir)
    (out / "queries").mkdir(parents=True, exist_ok=True)
    started = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))

    catalog_dir = out / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    db = FingerprintDB(str(out / "bench.db"))
    titles: dict[int, str] = {}
    samples_by_seed: dict[int, array] = {}
    try:
        for i in range(n_tracks):
            seed = _CATALOG_SEED0 + i
            samples = synth_track(seed, track_seconds)
            samples_by_seed[seed] = samples
            title = f"Bench Track {i:02d}"
            path = write_wav(catalog_dir / f"track{i:02d}.wav", samples)
            res = db.add_track(path, title=title, artist="SongRecon Bench")
            if not res.get("ok"):
                raise RuntimeError(f"catalog build failed: {res}")
            titles[seed] = title
        stats = db.stats()
        background = synth_track(_BACKGROUND_SEED, track_seconds)
        ctx = _Ctx(str(out / "queries"), background)

        rows: list[dict[str, Any]] = []
        unavailable: dict[str, str] = {}
        for name, builder in _condition_map(quick):
            for seed, title in titles.items():
                sub = _subseed(seed, name)
                try:
                    query = builder(samples_by_seed[seed], sub, ctx)
                except BenchUnavailable as exc:
                    unavailable[name] = str(exc)
                    continue
                qpath = write_wav(
                    out / "queries" / f"q-{name}-{seed}.wav", query)
                t0 = time.perf_counter()
                ranked = db.rank_candidates(qpath, limit=5)
                match = db.match(qpath, threshold=threshold)
                latency = time.perf_counter() - t0
                top1 = ranked[0] if ranked else {}
                rows.append({
                    "condition": name,
                    "expected": title,
                    "accepted": bool(match.get("ok")),
                    "matched_title": match.get("title") or "",
                    "top1_title": top1.get("title") or "",
                    "top1_score": float(top1.get("score") or 0.0),
                    "top1_votes": int(top1.get("coherent_votes") or 0),
                    "top1_correct": (top1.get("title") or "") == title,
                    "accepted_correct": bool(match.get("ok"))
                    and (match.get("title") or "") == title,
                    "top5_hit": any(
                        (r.get("title") or "") == title for r in ranked),
                    "correct_score": next(
                        (float(r["score"]) for r in ranked
                         if (r.get("title") or "") == title), None),
                    "rate_correction": match.get("rate_correction"),
                    "latency_s": round(latency, 4),
                })

        neg_rows: list[dict[str, Any]] = []
        neg_builders = dict(_condition_map(False))
        neg_samples = {_NEGATIVE_SEED0 + i: synth_track(
            _NEGATIVE_SEED0 + i, track_seconds) for i in range(n_tracks)}
        for name in NEGATIVE_CONDITIONS:
            builder = neg_builders[name]
            for seed, samples in neg_samples.items():
                sub = _subseed(seed, name)
                try:
                    query = builder(samples, sub, ctx)
                except BenchUnavailable as exc:
                    unavailable[name] = str(exc)
                    continue
                qpath = write_wav(
                    out / "queries" / f"neg-{name}-{seed}.wav", query)
                match = db.match(qpath, threshold=threshold)
                neg_rows.append({
                    "condition": name,
                    "false_accept": bool(match.get("ok")),
                    "matched_title": match.get("title") or "",
                    "best_score": float(match.get("best_score") or
                                        match.get("score") or 0.0),
                })

        # Sibling stress pool: re-performances of catalog compositions
        # (same notes, different performance) — NOT the same recording,
        # so a class-1 accept is a false candidate; reported separately
        # from the different-composition headline rate.
        sib_rows: list[dict[str, Any]] = []
        for name in ("excerpt_10s", "noise_snr10"):
            builder = neg_builders[name]
            for seed, title in titles.items():
                samples = synth_sibling(seed, track_seconds)
                sub = _subseed(seed, "sibling", name)
                try:
                    query = builder(samples, sub, ctx)
                except BenchUnavailable as exc:
                    unavailable[f"sibling:{name}"] = str(exc)
                    continue
                qpath = write_wav(
                    out / "queries" / f"sib-{name}-{seed}.wav", query)
                match = db.match(qpath, threshold=threshold)
                sib_rows.append({
                    "condition": name,
                    "composition": title,
                    "accepted": bool(match.get("ok")),
                    "matched_title": match.get("title") or "",
                    "named_own_composition": bool(match.get("ok"))
                    and (match.get("title") or "") == title,
                    "best_score": float(match.get("best_score") or
                                        match.get("score") or 0.0),
                })
    finally:
        db.close()

    conditions: list[dict[str, Any]] = []
    for name, _builder in _condition_map(quick):
        cell = [r for r in rows if r["condition"] == name]
        if name in unavailable and not cell:
            conditions.append({"name": name, "unavailable": True,
                               "reason": unavailable[name], "n": 0})
            continue
        n = len(cell)
        lat = [r["latency_s"] for r in cell]
        conditions.append({
            "name": name,
            "n": n,
            "top1_accuracy": round(
                sum(1 for r in cell if r["top1_correct"]) / n, 4) if n else 0.0,
            "accept_accuracy": round(
                sum(1 for r in cell if r["accepted_correct"]) / n, 4)
            if n else 0.0,
            "top5_accuracy": round(
                sum(1 for r in cell if r["top5_hit"]) / n, 4) if n else 0.0,
            "false_accepts": sum(
                1 for r in cell
                if r["accepted"] and not r["accepted_correct"]),
            "rate_corrected": sum(
                1 for r in cell if r.get("rate_correction")),
            "median_query_s": round(statistics.median(lat), 4) if lat else 0.0,
        })

    neg_n = len(neg_rows)
    neg_false = sum(1 for r in neg_rows if r["false_accept"])
    calibration = []
    for lo, hi in _SCORE_BINS:
        cell = [r for r in rows if lo <= r["top1_score"] < hi]
        calibration.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": len(cell),
            "top1_correct_fraction": round(
                sum(1 for r in cell if r["top1_correct"]) / len(cell), 4)
            if cell else None,
        })
    accepted_rows = [r for r in rows if r["accepted"]]
    report: dict[str, Any] = {
        "schema": BENCH_SCHEMA,
        "generated_at": stamp,
        "engine": {
            "threshold": threshold,
            "catalog_tracks": int(stats.get("tracks") or 0),
            "catalog_hashes": int(stats.get("hashes") or 0),
            "track_seconds": track_seconds,
            "quick": quick,
        },
        "conditions": conditions,
        "negatives": {
            "n": neg_n,
            "false_accepts": neg_false,
            "false_candidate_rate": round(neg_false / neg_n, 4)
            if neg_n else None,
            "rows": neg_rows,
        },
        "siblings": {
            "n": len(sib_rows),
            "accepts": sum(1 for r in sib_rows if r["accepted"]),
            "accept_rate": round(
                sum(1 for r in sib_rows if r["accepted"])
                / len(sib_rows), 4) if sib_rows else None,
            "named_own_composition": sum(
                1 for r in sib_rows if r["named_own_composition"]),
            "rows": sib_rows,
        },
        "calibration": calibration,
        "accepted_precision": {
            "n_accepted": len(accepted_rows),
            "n_correct": sum(1 for r in accepted_rows
                             if r["accepted_correct"]),
            "precision": round(
                sum(1 for r in accepted_rows if r["accepted_correct"])
                / len(accepted_rows), 4) if accepted_rows else None,
        },
        "rows": rows,
        "wall_s": round(time.time() - started, 1),
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report),
                                   encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable rendering of a benchmark report dict."""
    eng = report["engine"]
    lines = [
        "# SongRecon §9 benchmark",
        "",
        f"Generated: {report['generated_at']} · schema "
        f"{report['schema']} · threshold {eng['threshold']} · "
        f"catalog {eng['catalog_tracks']} tracks / "
        f"{eng['catalog_hashes']} hashes · wall {report['wall_s']}s",
        "",
        "| condition | n | top-1 | accepted-correct | top-5 | "
        "false accepts | median query |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in report["conditions"]:
        if c.get("unavailable"):
            lines.append(
                f"| {c['name']} | 0 | unavailable: {c['reason']} "
                "| | | | |")
            continue
        lines.append(
            f"| {c['name']} | {c['n']} | {c['top1_accuracy']:.3f} | "
            f"{c['accept_accuracy']:.3f} | {c['top5_accuracy']:.3f} | "
            f"{c['false_accepts']} | {c['median_query_s']:.3f}s |")
    neg = report["negatives"]
    sib = report.get("siblings") or {}
    prec = report["accepted_precision"]
    lines += [
        "",
        f"Negatives: n={neg['n']} · false accepts="
        f"{neg['false_accepts']} · false-candidate rate="
        f"{neg['false_candidate_rate']}",
        f"Siblings (re-performances, not the same recording): "
        f"n={sib.get('n')} · accepts={sib.get('accepts')} · "
        f"accept rate={sib.get('accept_rate')} · named own "
        f"composition={sib.get('named_own_composition')}",
        f"Accepted precision (positives): {prec['precision']} "
        f"({prec['n_correct']}/{prec['n_accepted']} accepted)",
        "",
        "Calibration — P(top-1 correct) by top-1 coherent-offset "
        "fraction bin (positive queries):",
        "",
        "| score bin | n | P(correct) |",
        "|---|---|---|",
    ]
    for cell in report["calibration"]:
        frac = cell["top1_correct_fraction"]
        lines.append(
            f"| {cell['bin']} | {cell['n']} | "
            f"{frac if frac is not None else '—'} |")
    lines.append("")
    return "\n".join(lines)


def fingerprint_fixture_check(seed: int = _CATALOG_SEED0,
                              seconds: float = 6.0) -> bool:
    """Determinism probe: the same seed regenerates the same
    fingerprint twice (used by the SR2 suite)."""
    import tempfile

    samples = synth_track(seed, seconds)
    again = synth_track(seed, seconds)
    if list(samples) != list(again):
        return False
    with tempfile.TemporaryDirectory() as td:
        path = write_wav(os.path.join(td, "probe.wav"), samples)
        return fingerprint(path) == fingerprint(path)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="songrecon_bench",
        description="Run the SongRecon §9 robustness benchmark "
                    "against the shipped local fingerprint engine.")
    parser.add_argument("--out", required=True,
                        help="output directory for report.json/report.md")
    parser.add_argument("--tracks", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--quick", action="store_true",
                        help="reduced condition set (suite-sized)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_benchmark(args.out, n_tracks=args.tracks,
                           track_seconds=args.seconds, quick=args.quick)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
