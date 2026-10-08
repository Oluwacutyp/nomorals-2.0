"""Pure-Python MIDI→audio renderer.

Turns arranged :class:`NoteEvent` tracks into a real playable WAV file
with zero dependencies — stdlib only (``wave``, ``math``, ``array``,
``random``).  This is what makes ``/music compose`` deliver actual audio
instead of just a MIDI file.

Each track gets its own timbre:
* melody / counter — bright lead (sine + harmonics, fast attack)
* chords — soft pad (sine + light 2nd harmonic, slow attack)
* bass — deep sine with a touch of 2nd harmonic
* drums — synthesized kick (pitch-dropping sine), snare (noise + tone),
  hats/cymbals (highpassed noise)

GM percussion note numbers (matching :mod:`nomorals.core.midi`):
35/36 kick, 38/40 snare, 42 closed hat, 46 open hat, 49/51/59 cymbals,
everything else → generic percussive tick.
"""

from __future__ import annotations

import logging
import math
import random
import struct
import wave
from array import array
from io import BytesIO
from typing import Sequence

from . import caps

_log = logging.getLogger(__name__)

__all__ = ["render_wav", "write_wav", "mix_tracks"]

#: sample rate — 22050 Hz keeps CPU and file size sane on a phone
SAMPLE_RATE = 22050

#: safety bound: one note never renders more than this many samples
#: (10 minutes). Corrupt arrangement data (a duration of thousands of
#: beats) used to turn the per-sample Python loops below into an
#: effective hang; now it is bounded work plus a debug log.
_MAX_NOTE_SAMPLES = 10 * 60 * SAMPLE_RATE


def _np():
    """numpy when importable, else None.

    The fast vectorized render path uses numpy if it is installed;
    every function below keeps a pure-stdlib twin so the synth still
    works on machines without it (the ``fast`` extra in pyproject).
    """
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return np

#: wavetable size for the sine lookup
_TABLE_SIZE = 2048
_SINE_TABLE = array("d", (math.sin(2.0 * math.pi * i / _TABLE_SIZE)
                           for i in range(_TABLE_SIZE)))


def _sine(phase: float) -> float:
    """Fast sine via wavetable; phase in radians."""
    idx = int(phase * (_TABLE_SIZE / (2.0 * math.pi))) % _TABLE_SIZE
    return _SINE_TABLE[idx]


def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _adsr(n: int, attack: float, decay: float, sustain: float,
          release: float, dur_samples: int) -> array:
    """ADSR envelope as an array of length n (n <= dur_samples)."""
    env = array("d", [0.0]) * n
    sr = SAMPLE_RATE
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    for i in range(n):
        if i < a:
            env[i] = i / a
        elif i < a + d:
            env[i] = 1.0 - (1.0 - sustain) * ((i - a) / d)
        elif i < n - r:
            env[i] = sustain
        else:
            env[i] = sustain * max(0.0, (n - i) / r)
    return env


def _render_tone(freq: float, n: int, velocity: float, harmonics: Sequence[float],
                 attack: float, decay: float, sustain: float,
                 release: float) -> array:
    """A pitched note: harmonic stack × ADSR."""
    out = array("d", [0.0]) * n
    env = _adsr(n, attack, decay, sustain, release, n)
    phases = [0.0] * len(harmonics)
    for h, amp in enumerate(harmonics, start=1):
        if amp <= 0:
            continue
        f = freq * h
        inc = 2.0 * math.pi * f / SAMPLE_RATE
        ph = 0.0
        for i in range(n):
            out[i] += amp * _sine(ph)
            ph += inc
    vel = max(0.05, min(1.0, velocity / 100.0))
    for i in range(n):
        out[i] *= env[i] * vel * 0.5
    return out


def _render_kick(n: int, velocity: float) -> array:
    """Kick: sine dropping 120 Hz → 45 Hz with a fast decay."""
    out = array("d", [0.0]) * n
    env = _adsr(n, 0.002, 0.09, 0.0, 0.02, n)
    ph = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        freq = 45.0 + 75.0 * math.exp(-t * 30.0)
        ph += 2.0 * math.pi * freq / SAMPLE_RATE
        out[i] = _sine(ph) * env[i]
    vel = max(0.05, min(1.0, velocity / 100.0))
    for i in range(n):
        out[i] *= vel
    return out


def _render_snare(n: int, velocity: float, rng: random.Random) -> array:
    """Snare: noise burst + 180 Hz body tone."""
    out = array("d", [0.0]) * n
    env = _adsr(n, 0.001, 0.08, 0.0, 0.03, n)
    ph = 0.0
    inc = 2.0 * math.pi * 180.0 / SAMPLE_RATE
    for i in range(n):
        noise = rng.uniform(-1.0, 1.0)
        out[i] = (0.6 * noise + 0.4 * _sine(ph)) * env[i]
        ph += inc
    vel = max(0.05, min(1.0, velocity / 100.0))
    for i in range(n):
        out[i] *= vel * 0.7
    return out


def _render_hat(n: int, velocity: float, rng: random.Random,
                open: bool = False) -> array:
    """Hat/cymbal: highpassed noise (differenced white noise)."""
    out = array("d", [0.0]) * n
    decay = 0.25 if open else 0.05
    env = _adsr(n, 0.001, decay, 0.0, 0.02, n)
    prev = 0.0
    for i in range(n):
        noise = rng.uniform(-1.0, 1.0)
        hp = noise - prev  # crude highpass: hats live in the highs
        prev = noise
        out[i] = hp * env[i] * 0.5
    vel = max(0.05, min(1.0, velocity / 100.0))
    for i in range(n):
        out[i] *= vel * 0.5
    return out


#: timbre per track: (harmonics, attack, decay, sustain, release, gain)
#: bass carries extra upper harmonics (2nd/3rd) so the line stays audible
#: on small phone speakers (psychoacoustic "missing fundamental"), plus a
#: sub-octave doubler (see _SUB_BASS_GAIN) for systems with real low end.
#: Clipping is impossible: both mix paths peak-normalize + soft-clip.
_TIMBRES = {
    "melody":  ((1.0, 0.45, 0.22, 0.1), 0.008, 0.06, 0.75, 0.09, 1.0),
    "counter": ((1.0, 0.35, 0.15, 0.0), 0.01, 0.08, 0.7, 0.1, 0.8),
    "chords":  ((1.0, 0.28, 0.08, 0.0), 0.05, 0.12, 0.65, 0.18, 0.55),
    "bass":    ((1.0, 0.5, 0.22, 0.08), 0.006, 0.05, 0.85, 0.06, 1.2),
}

#: sub-bass octave-doubler gain for the bass track: a pure sine one octave
#: below each bass note, mixed under. Inaudible on phone speakers (they
#: can't reproduce it) but felt on real systems; the mix normalizer keeps
#: it from ever pushing the master into clipping. 0.0 disables.
_SUB_BASS_GAIN = 0.35

_KICK_NOTES = {35, 36}
_SNARE_NOTES = {38, 40}
_HAT_NOTES = {42, 44}
_OPEN_HAT_NOTES = {46, 49, 51, 52, 55, 57, 59}


def _render_drum(note: int, dur_beats: float, tempo: float,
                 velocity: int, rng: random.Random) -> array:
    dur_sec = max(0.05, dur_beats * 60.0 / tempo + 0.15)
    n = min(_MAX_NOTE_SAMPLES, int(dur_sec * SAMPLE_RATE))
    if note in _KICK_NOTES:
        return _render_kick(n, velocity)
    if note in _SNARE_NOTES:
        return _render_snare(n, velocity, rng)
    if note in _HAT_NOTES:
        return _render_hat(n, velocity, rng, open=False)
    if note in _OPEN_HAT_NOTES:
        return _render_hat(n, velocity, rng, open=True)
    # unknown percussion → short tick
    return _render_hat(n, velocity, rng, open=False)


# ── numpy fast path ─────────────────────────────────────────────────────────
# Vectorized twins of the renderers above. Same math, same envelopes,
# ~100× faster: the stdlib versions above are per-sample Python loops
# (a 50-second song was ~40s of CPU; the numpy path is <1s). Used
# automatically when numpy is importable; the stdlib versions remain
# the fallback and the reference implementation.


def _adsr_np(np, n: int, attack: float, decay: float, sustain: float,
             release: float):
    """ADSR envelope as a float64 vector of length n."""
    sr = SAMPLE_RATE
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    i = np.arange(n, dtype=np.float64)
    env = np.empty(n, dtype=np.float64)
    m_attack = i < a
    m_decay = (i >= a) & (i < a + d)
    m_sustain = (i >= a + d) & (i < n - r)
    m_rel = ~(m_attack | m_decay | m_sustain)
    env[m_attack] = i[m_attack] / a
    env[m_decay] = 1.0 - (1.0 - sustain) * ((i[m_decay] - a) / d)
    env[m_sustain] = sustain
    env[m_rel] = sustain * np.maximum(0.0, (n - i[m_rel]) / r)
    return env


def _render_tone_np(np, freq: float, n: int, velocity: float,
                    harmonics: Sequence[float], attack: float, decay: float,
                    sustain: float, release: float):
    """A pitched note: harmonic stack × ADSR, vectorized."""
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    env = _adsr_np(np, n, attack, decay, sustain, release)
    out = np.zeros(n, dtype=np.float64)
    for h, amp in enumerate(harmonics, start=1):
        if amp <= 0:
            continue
        out += amp * np.sin(2.0 * np.pi * freq * h * t)
    vel = max(0.05, min(1.0, velocity / 100.0))
    out *= env * vel * 0.5
    return out


def _render_kick_np(np, n: int, velocity: float):
    """Kick: sine dropping 120 Hz → 45 Hz with a fast decay."""
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    freq = 45.0 + 75.0 * np.exp(-t * 30.0)
    phase = np.cumsum(2.0 * np.pi * freq / SAMPLE_RATE)
    env = _adsr_np(np, n, 0.002, 0.09, 0.0, 0.02)
    vel = max(0.05, min(1.0, velocity / 100.0))
    return np.sin(phase) * env * vel


def _render_snare_np(np, n: int, velocity: float, rng) -> object:
    """Snare: noise burst + 180 Hz body tone."""
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    noise = rng.uniform(-1.0, 1.0, n)
    tone = np.sin(2.0 * np.pi * 180.0 * t)
    env = _adsr_np(np, n, 0.001, 0.08, 0.0, 0.03)
    vel = max(0.05, min(1.0, velocity / 100.0))
    return (0.6 * noise + 0.4 * tone) * env * vel * 0.7


def _render_hat_np(np, n: int, velocity: float, rng,
                   open: bool = False) -> object:
    """Hat/cymbal: highpassed noise (differenced white noise)."""
    decay = 0.25 if open else 0.05
    env = _adsr_np(np, n, 0.001, decay, 0.0, 0.02)
    noise = rng.uniform(-1.0, 1.0, n)
    hp = np.empty(n, dtype=np.float64)
    hp[0] = noise[0]  # prev starts at 0.0, matching the stdlib version
    hp[1:] = noise[1:] - noise[:-1]
    vel = max(0.05, min(1.0, velocity / 100.0))
    return hp * env * 0.5 * vel * 0.5


def _render_drum_np(np, note: int, dur_beats: float, tempo: float,
                    velocity: int, rng) -> object:
    dur_sec = max(0.05, dur_beats * 60.0 / tempo + 0.15)
    n = min(_MAX_NOTE_SAMPLES, int(dur_sec * SAMPLE_RATE))
    if note in _KICK_NOTES:
        return _render_kick_np(np, n, velocity)
    if note in _SNARE_NOTES:
        return _render_snare_np(np, n, velocity, rng)
    if note in _OPEN_HAT_NOTES:
        return _render_hat_np(np, n, velocity, rng, open=True)
    # hats and unknown percussion → short tick
    return _render_hat_np(np, n, velocity, rng, open=False)


def _mix_tracks_np(np, parts: dict[str, list], tempo: float,
                   seed: int = 0) -> object:
    """Vectorized mix. Returns a float64 numpy buffer."""
    beat_sec = 60.0 / tempo
    total_beats = 0.0
    for events in parts.values():
        for e in events:
            total_beats = max(total_beats, e.start + e.duration)
    total_samples = int(total_beats * beat_sec * SAMPLE_RATE) + SAMPLE_RATE
    mix = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for track, events in parts.items():
        if track == "drums":
            for e in events:
                start = int(e.start * beat_sec * SAMPLE_RATE)
                if start >= total_samples:
                    continue
                sig = _render_drum_np(np, e.note, e.duration, tempo,
                                      e.velocity, rng)
                end = min(total_samples, start + len(sig))
                mix[start:end] += sig[:end - start] * 0.8
            continue
        timbre = _TIMBRES.get(track, _TIMBRES["melody"])
        harmonics, attack, decay, sustain, release, gain = timbre
        for e in events:
            start = int(e.start * beat_sec * SAMPLE_RATE)
            if start >= total_samples:
                continue
            dur_sec = e.duration * beat_sec
            n = min(_MAX_NOTE_SAMPLES, max(16, int(dur_sec * SAMPLE_RATE)))
            freq = midi_to_freq(max(0, min(127, int(e.note))))
            sig = _render_tone_np(np, freq, n, e.velocity, harmonics,
                                  attack, decay, sustain, release)
            end = min(total_samples, start + len(sig))
            mix[start:end] += sig[:end - start] * gain
            if track == "bass" and _SUB_BASS_GAIN > 0:
                # sub-octave doubler: pure sine one octave down
                sub = _render_tone_np(np, freq / 2.0, n, e.velocity,
                                      (1.0,), attack, decay, sustain,
                                      release)
                mix[start:end] += sub[:end - start] * _SUB_BASS_GAIN
    # soft clip + normalize (same curve as the stdlib path)
    if total_samples:
        peak = float(np.max(np.abs(mix)))
        if peak > 0:
            mix[:] = np.tanh(mix * (0.89 / peak) * 1.2) * 0.95
    return mix


def _mix_tracks_stdlib(parts: dict[str, list], tempo: float,
                       seed: int = 0) -> array:
    """Reference implementation: per-sample Python loops."""
    beat_sec = 60.0 / tempo
    # find the total length
    total_beats = 0.0
    for events in parts.values():
        for e in events:
            total_beats = max(total_beats, e.start + e.duration)
    total_samples = int(total_beats * beat_sec * SAMPLE_RATE) + SAMPLE_RATE
    mix = array("d", [0.0]) * total_samples
    rng = random.Random(seed)

    for track, events in parts.items():
        if track == "drums":
            for e in events:
                start = int(e.start * beat_sec * SAMPLE_RATE)
                sig = _render_drum(e.note, e.duration, tempo, e.velocity, rng)
                for i, s in enumerate(sig):
                    idx = start + i
                    if idx < total_samples:
                        mix[idx] += s * 0.8
            continue
        timbre = _TIMBRES.get(track, _TIMBRES["melody"])
        harmonics, attack, decay, sustain, release, gain = timbre
        for e in events:
            start = int(e.start * beat_sec * SAMPLE_RATE)
            dur_sec = e.duration * beat_sec
            n = min(_MAX_NOTE_SAMPLES, max(16, int(dur_sec * SAMPLE_RATE)))
            freq = midi_to_freq(max(0, min(127, int(e.note))))
            sig = _render_tone(freq, n, e.velocity, harmonics,
                               attack, decay, sustain, release)
            for i, s in enumerate(sig):
                idx = start + i
                if idx < total_samples:
                    mix[idx] += s * gain
            if track == "bass" and _SUB_BASS_GAIN > 0:
                # sub-octave doubler: pure sine one octave down
                sub = _render_tone(freq / 2.0, n, e.velocity, (1.0,),
                                   attack, decay, sustain, release)
                for i, s in enumerate(sub):
                    idx = start + i
                    if idx < total_samples:
                        mix[idx] += s * _SUB_BASS_GAIN
    # soft clip + normalize
    peak = max((abs(s) for s in mix), default=0.0)
    if peak > 0:
        norm = 0.89 / peak
        # gentle tanh-style soft clip on the way
        for i, s in enumerate(mix):
            v = s * norm
            mix[i] = math.tanh(v * 1.2) * 0.95
    return mix


def mix_tracks(parts: dict[str, list], tempo: float,
               seed: int = 0) -> array:
    """Mix all tracks into one mono float buffer.

    Uses the vectorized numpy path when numpy is importable, else the
    pure-stdlib reference implementation. Both return ``array('d')``.
    """
    np = _np()
    if np is not None:
        return array("d", _mix_tracks_np(np, parts, tempo, seed=seed))
    return _mix_tracks_stdlib(parts, tempo, seed=seed)


def render_wav(parts: dict[str, list], tempo: float, seed: int = 0) -> bytes:
    """Render arranged tracks to WAV bytes (16-bit mono)."""
    np = _np()
    if np is not None:
        mix = _mix_tracks_np(np, parts, tempo, seed=seed)
        # truncate toward zero, same as int() in the stdlib path
        pcm = np.clip(mix * 32767.0, -32768.0, 32767.0).astype(np.int16)
        frames = pcm.tobytes()
    else:
        mix = _mix_tracks_stdlib(parts, tempo, seed=seed)
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in mix))
        frames = pcm.tobytes()
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(frames)
    return buf.getvalue()


def write_wav(path: str, parts: dict[str, list], tempo: float,
              seed: int = 0) -> str | None:
    """Render and write a WAV file.

    Returns the path on success, or ``None`` when the render would exceed
    :data:`.caps.MAX_AUDIO_WRITE_BYTES` — the cap refusal is logged and
    never raises, and nothing is written (no truncation).
    """
    data = render_wav(parts, tempo, seed=seed)
    ok, reason = caps.check_write_size(len(data), caps.MAX_AUDIO_WRITE_BYTES)
    if not ok:
        caps.refuse_write(f"write_wav({path})", reason)
        return None
    with open(path, "wb") as f:
        f.write(data)
    return path
