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


def _render_slide_tone(freq_start: float, freq_end: float, n: int,
                       velocity: float, harmonics: Sequence[float],
                       attack: float, decay: float, sustain: float,
                       release: float) -> array:
    """An 808-style gliding note: pitch sweeps exponentially from
    ``freq_start`` to ``freq_end`` over the note (portamento).

    The exponential curve sounds like a real 808 slide — fast at first,
    settling into the target — instead of a linear robot sweep.
    """
    out = array("d", [0.0]) * n
    env = _adsr(n, attack, decay, sustain, release, n)
    ratio = max(1e-6, freq_end / max(1e-6, freq_start))
    for h, amp in enumerate(harmonics, start=1):
        if amp <= 0:
            continue
        f0 = freq_start * h
        ph = 0.0
        for i in range(n):
            frac = i / max(1, n - 1)
            freq = f0 * (ratio ** frac)
            ph += 2.0 * math.pi * freq / SAMPLE_RATE
            out[i] += amp * _sine(ph)
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
#: Raised for heavier 808s — drill/trap/phonk lean on the sub.
_SUB_BASS_GAIN = 0.5

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


def _render_slide_tone_np(np, freq_start: float, freq_end: float, n: int,
                           velocity: float, harmonics: Sequence[float],
                           attack: float, decay: float, sustain: float,
                           release: float):
    """Vectorized 808-style pitch glide: exponential sweep."""
    ratio = max(1e-6, freq_end / max(1e-6, freq_start))
    frac = np.arange(n, dtype=np.float64) / max(1, n - 1)
    env = _adsr_np(np, n, attack, decay, sustain, release)
    out = np.zeros(n, dtype=np.float64)
    for h, amp in enumerate(harmonics, start=1):
        if amp <= 0:
            continue
        freq = (freq_start * h) * (ratio ** frac)
        phase = np.cumsum(2.0 * np.pi * freq / SAMPLE_RATE)
        out += amp * np.sin(phase)
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
            slide = getattr(e, "slide_to", None)
            if slide is not None:
                freq_end = midi_to_freq(max(0, min(127, int(slide))))
                sig = _render_slide_tone_np(np, freq, freq_end, n,
                                            e.velocity, harmonics,
                                            attack, decay, sustain, release)
            else:
                sig = _render_tone_np(np, freq, n, e.velocity, harmonics,
                                      attack, decay, sustain, release)
            end = min(total_samples, start + len(sig))
            mix[start:end] += sig[:end - start] * gain
            if track == "bass" and _SUB_BASS_GAIN > 0:
                # sub-octave doubler: pure sine one octave down, following
                # any 808 glide so the sub sweeps with the bass
                if slide is not None:
                    sub = _render_slide_tone_np(
                        np, freq / 2.0, freq_end / 2.0, n, e.velocity,
                        (1.0,), attack, decay, sustain, release)
                else:
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
            slide = getattr(e, "slide_to", None)
            if slide is not None:
                freq_end = midi_to_freq(max(0, min(127, int(slide))))
                sig = _render_slide_tone(freq, freq_end, n, e.velocity,
                                         harmonics, attack, decay, sustain,
                                         release)
            else:
                sig = _render_tone(freq, n, e.velocity, harmonics,
                                   attack, decay, sustain, release)
            for i, s in enumerate(sig):
                idx = start + i
                if idx < total_samples:
                    mix[idx] += s * gain
            if track == "bass" and _SUB_BASS_GAIN > 0:
                # sub-octave doubler: pure sine one octave down, following
                # any 808 glide so the sub sweeps with the bass
                if slide is not None:
                    sub = _render_slide_tone(freq / 2.0, freq_end / 2.0, n,
                                             e.velocity, (1.0,),
                                             attack, decay, sustain, release)
                else:
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


# ── voice / patch / bus layer (synth8 + kengine pattern) ──────────────────────
#
# The render functions above are the offline path. This layer adds the
# instrument abstraction the gold has: a Patch is a named sound, a Voice
# is one triggered note on that patch, and the SynthBus is the
# polyphonic manager — voices → one-pole lowpass → feedback delay send
# → master. Pure stdlib (array), Termux-safe.

from dataclasses import dataclass, field as _dc_field


@dataclass
class Patch:
    """A named synth sound: harmonic recipe + envelope + filter + FX."""
    name: str = "keys"
    harmonics: tuple[float, ...] = (1.0, 0.35, 0.15, 0.05)
    attack: float = 0.01
    decay: float = 0.08
    sustain: float = 0.7
    release: float = 0.15
    cutoff: float = 8000.0     # one-pole lowpass Hz (20000 = wide open)
    resonance: float = 0.0     # 0..1 — pre-filter drive
    delay_send: float = 0.0    # 0..1 — feedback delay send amount
    delay_time: float = 0.375  # seconds (dotted-eighth at 120bpm)
    velocity: float = 96.0

    def to_dict(self) -> dict:
        return {"name": self.name, "harmonics": list(self.harmonics),
                "attack": self.attack, "decay": self.decay,
                "sustain": self.sustain, "release": self.release,
                "cutoff": self.cutoff, "resonance": self.resonance,
                "delay_send": self.delay_send,
                "delay_time": self.delay_time,
                "velocity": self.velocity}


#: Factory presets — the working palette (kengine-style patch list).
PATCHES: dict[str, Patch] = {
    "bass": Patch("bass", harmonics=(1.0, 0.5, 0.2, 0.08),
                  attack=0.005, decay=0.05, sustain=0.85, release=0.08,
                  cutoff=900.0, resonance=0.25, velocity=100.0),
    "sub": Patch("sub", harmonics=(1.0, 0.12),
                 attack=0.005, decay=0.02, sustain=1.0, release=0.05,
                 cutoff=300.0, velocity=104.0),
    "lead": Patch("lead", harmonics=(1.0, 0.6, 0.35, 0.18, 0.08),
                  attack=0.01, decay=0.12, sustain=0.75, release=0.2,
                  cutoff=6500.0, resonance=0.15,
                  delay_send=0.25, velocity=96.0),
    "pluck": Patch("pluck", harmonics=(1.0, 0.45, 0.22, 0.1),
                   attack=0.003, decay=0.18, sustain=0.25, release=0.12,
                   cutoff=4200.0, resonance=0.3,
                   delay_send=0.18, velocity=92.0),
    "pad": Patch("pad", harmonics=(1.0, 0.5, 0.3, 0.18, 0.1, 0.05),
                 attack=0.4, decay=0.6, sustain=0.85, release=0.8,
                 cutoff=2800.0, velocity=80.0),
    "stab": Patch("stab", harmonics=(1.0, 0.55, 0.3, 0.12),
                  attack=0.004, decay=0.09, sustain=0.4, release=0.06,
                  cutoff=5200.0, resonance=0.2, velocity=98.0),
    "keys": Patch("keys", harmonics=(1.0, 0.35, 0.15, 0.05),
                  attack=0.01, decay=0.08, sustain=0.7, release=0.15,
                  cutoff=8000.0, velocity=90.0),
}


class Voice:
    """One triggered note on a Patch (synth8 voice pattern).

    ``note_on`` renders the attack/decay/sustain body; ``note_off``
    appends the release tail. Voices are cheap — the bus pools them.
    """

    def __init__(self, patch: Patch, midi_note: int,
                 velocity: float | None = None,
                 dur_beats: float = 1.0, tempo: float = 120.0) -> None:
        self.patch = patch
        self.midi_note = int(midi_note)
        self.velocity = float(patch.velocity if velocity is None
                              else velocity)
        self.dur_beats = float(dur_beats)
        self.tempo = float(tempo)
        self.released = False

    @property
    def dur_s(self) -> float:
        return self.dur_beats * 60.0 / max(1.0, self.tempo)

    def render(self, tail_beats: float = 0.5) -> array:
        """Render body + release tail. Never raises."""
        try:
            p = self.patch
            freq = midi_to_freq(max(0, min(127, self.midi_note)))
            n = max(1, int(self.dur_s * SAMPLE_RATE))
            body = _render_tone(freq, n, self.velocity, p.harmonics,
                                p.attack, p.decay, p.sustain, 0.01)
            tail_n = max(1, int(tail_beats * 60.0 / max(1.0, self.tempo)
                                * SAMPLE_RATE))
            tail = _render_tone(freq, tail_n, self.velocity * 0.6,
                                p.harmonics, 0.005, p.release, 0.0, 0.01)
            out = array("d", [0.0]) * (len(body) + len(tail))
            for i, v in enumerate(body):
                out[i] += v
            # crossfade the tail in over the last 10% of the body
            xf = max(1, len(body) // 10)
            for i, v in enumerate(tail):
                j = len(body) - xf + i
                if 0 <= j < len(out):
                    w = min(1.0, i / max(1, xf))
                    out[j] = out[j] * (1 - w) + (out[j] + v) * w \
                        if j >= len(body) else out[j] + v * w
                elif j < len(out):
                    out[j] += v
            return out
        except Exception:  # noqa: BLE001 — a voice never kills the bus
            return array("d", [0.0])


def _one_pole_lowpass(buf: array, cutoff: float) -> array:
    """One-pole lowpass (kengine SVF-lite). cutoff<=0 or >=20000 = bypass."""
    if cutoff <= 0 or cutoff >= 20000 or not buf:
        return buf
    import math as _m
    rc = 1.0 / (2.0 * _m.pi * cutoff)
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (rc + dt)
    out = array("d", [0.0]) * len(buf)
    y = 0.0
    for i, x in enumerate(buf):
        y += alpha * (x - y)
        out[i] = y
    return out


class SynthBus:
    """Polyphonic voice manager: voices → filter → delay send → master.

    ``play_notes([(midi, start_beat, dur_beats, velocity), ...])``
    renders a full part on one patch. ``play_chord`` stacks a chord as
    one event. The bus is the per-part counterpart to ``mix_tracks``
    (which mixes whole parts together).
    """

    def __init__(self, patch: Patch | str = "keys",
                 tempo: float = 120.0, master: float = 0.9) -> None:
        self.patch = (PATCHES[patch] if isinstance(patch, str)
                      else patch)
        self.tempo = float(tempo)
        self.master = float(master)
        self._delay_line: array = array("d")

    def _apply_bus_fx(self, buf: array) -> array:
        p = self.patch
        # filter stage
        buf = _one_pole_lowpass(buf, p.cutoff)
        # resonance = pre-filter drive (simple tanh-ish saturation)
        if p.resonance > 0 and buf:
            import math as _m
            drive = 1.0 + p.resonance * 2.0
            buf = array("d", (_m.tanh(v * drive) / _m.tanh(drive)
                              for v in buf))
        # feedback delay send
        if p.delay_send > 0 and buf:
            d_n = max(1, int(p.delay_time * SAMPLE_RATE))
            wet = array("d", [0.0]) * len(buf)
            for i in range(len(buf)):
                echo = wet[i - d_n] * 0.35 if i >= d_n else 0.0
                wet[i] = buf[i] * p.delay_send + echo
            buf = array("d", (a + b for a, b in zip(buf, wet)))
        if self.master != 1.0:
            buf = array("d", (v * self.master for v in buf))
        return buf

    def play_notes(self, notes: list[tuple],
                   patch: Patch | str | None = None) -> array:
        """Render [(midi, start_beat, dur_beats[, velocity]), ...].

        Never raises — bad entries are skipped honestly.
        """
        p = (PATCHES[patch] if isinstance(patch, str)
             else (patch or self.patch))
        beat_s = 60.0 / max(1.0, self.tempo)
        events: list[tuple[int, array]] = []
        total = 0
        for entry in notes:
            try:
                midi = int(entry[0])
                start = float(entry[1])
                dur = float(entry[2]) if len(entry) > 2 else 1.0
                vel = float(entry[3]) if len(entry) > 3 else p.velocity
            except Exception:  # noqa: BLE001 — skip bad entries
                continue
            v = Voice(p, midi, velocity=vel, dur_beats=dur,
                      tempo=self.tempo)
            sig = v.render()
            at = int(start * beat_s * SAMPLE_RATE)
            events.append((at, sig))
            total = max(total, at + len(sig))
        out = array("d", [0.0]) * max(1, total)
        for at, sig in events:
            for i, s in enumerate(sig):
                if at + i < len(out):
                    out[at + i] += s
        # gentle peak normalize to the bus
        peak = max((abs(v) for v in out), default=0.0)
        if peak > 1.0:
            out = array("d", (v / peak for v in out))
        return self._apply_bus_fx(out)

    def play_chord(self, midi_notes: list[int], start_beat: float = 0.0,
                   dur_beats: float = 4.0,
                   velocity: float | None = None) -> array:
        """One chord event: all notes start together."""
        notes = [(m, start_beat, dur_beats,
                  self.patch.velocity if velocity is None else velocity)
                 for m in midi_notes]
        return self.play_notes(notes)


__all__ = ["render_wav", "write_wav", "mix_tracks",
           "Patch", "PATCHES", "Voice", "SynthBus", "midi_to_freq"]
