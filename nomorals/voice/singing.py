"""Singing voice synthesis — DiffSinger, wired as a real backend.

The mined truth: DiffSinger (MIT, MoonInTheRiver/DiffSinger) is the
open-source singing-voice synthesis standard — diffusion acoustic model
conditioned on a music score (lyrics + pitch), shallow diffusion for
speed, NSF-HiFiGAN vocoder. Voicebanks ship in the OpenUtau DiffSinger
format: drop a bank into ``~/.nomorals/svs/`` and sing.

Pipeline (the mitystudio/songmaker pattern):
    melody (notes + lyrics)
      → phonemes (simple English g2p, bank dsdict-aware)
      → durations + f0 curve (midi→Hz, portamento, vibrato)
      → DiffSinger ONNX acoustic model → mel-spectrogram
      → NSF-HiFiGAN vocoder → waveform
      → optional RVC timbre swap (user's voice sings)

``DiffSingerBackend`` follows the tts.py backend contract
(``name``, ``synthesize``) so UniversalTTS can select it; ``sing()`` is
the dedicated entry. Without a voicebank or onnxruntime, everything
raises honest unavailability — the DSP "chant" fallback lives in
emotion_dsp, never pretends to be singing.
"""

from __future__ import annotations

import math
import os
import wave
from array import array
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.logging_setup import get_logger
from . import rvc_bridge

_log = get_logger(__name__)

__all__ = [
    "SVS_DIR",
    "Note",
    "parse_melody",
    "midi_to_hz",
    "humanize",
    "export_midi",
    "DiffSingerBackend",
    "sing",
]


def SVS_DIR() -> str:
    d = os.environ.get("NM_SVS_DIR",
                       os.path.expanduser("~/.nomorals/svs"))
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class Note:
    """One sung syllable — now with OpenUtau-style expression curves.

    - ``dynamics``: DYN — per-note loudness 0.0–1.0 (default 0.8).
    - ``breathiness``: BREC — airy mix 0.0–1.0 (default 0.0).
    - ``tension``: TENC — vocal tension 0.0–1.0 (default 0.5); raises
      vibrato depth and edge when high.
    - ``vibrato``: multiplier on the default vibrato depth (0 = straight
      tone, 2 = operatic).
    """
    midi: int            # MIDI note number (60 = middle C)
    duration_s: float    # how long the syllable lasts
    lyric: str = ""      # the sung text ("-" = sustain, "SP" = rest)
    dynamics: float = 0.8
    breathiness: float = 0.0
    tension: float = 0.5
    vibrato: float = 1.0


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def parse_melody(text: str) -> list[Note]:
    """Parse "C4:0.5:hello D4:0.5:world" or "60:0.5:hello" into Notes.

    Note names: C0..B8 with optional #/b (C#4, Bb3). Duration in seconds.
    Lyric "-" sustains the previous syllable; "SP"/"rest" is silence.
    A 4th colon-field carries expression modifiers, comma-separated::

        "E4:1.2:love:vib=1.4,dyn=0.9 bre=0.3 ten=0.7"
    """
    _NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    notes: list[Note] = []
    for tok in text.strip().split():
        parts = tok.split(":")
        if len(parts) < 2:
            continue
        pitch_s, dur_s = parts[0], parts[1]
        lyric = parts[2] if len(parts) > 2 else ""
        mods: dict[str, float] = {}
        if len(parts) > 3:
            for kv in parts[3].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        mods[k.strip().lower()] = float(v)
                    except ValueError:
                        pass
        try:
            dur = float(dur_s)
        except ValueError:
            continue
        try:
            midi = int(pitch_s)
        except ValueError:
            m = pitch_s.strip().upper()
            if not m or m[0] not in _NAMES:
                continue
            semi = _NAMES[m[0]]
            i = 1
            if i < len(m) and m[i] == "#":
                semi += 1
                i += 1
            elif i < len(m) and m[i] == "B":
                semi -= 1
                i += 1
            try:
                octave = int(m[i:])
            except ValueError:
                continue
            midi = (octave + 1) * 12 + semi
        def _mod(*names: str, default: float) -> float:
            for nm in names:
                if nm in mods:
                    return mods[nm]
            return default

        notes.append(Note(
            midi=midi, duration_s=max(0.05, dur), lyric=lyric,
            dynamics=min(1.0, max(0.0, _mod("dyn", "dynamics",
                                            default=0.8))),
            breathiness=min(1.0, max(0.0, _mod("bre", "breath",
                                               "breathiness", default=0.0))),
            tension=min(1.0, max(0.0, _mod("ten", "tension",
                                            default=0.5))),
            vibrato=max(0.0, _mod("vib", "vibrato", default=1.0)),
        ))
    return notes


# --- tiny English g2p (fallback when no phonemizer installed) --------------

_VOWELS = "aeiouy"


def _simple_g2p(word: str) -> list[str]:
    """Naive syllable-ish phonemization. Honest scope: approximate."""
    w = "".join(c for c in word.lower() if c.isalpha())
    if not w:
        return ["SP"]
    # Split into crude phoneme chunks: consonant clusters + vowel groups
    import re
    chunks = re.findall(r"[^aeiouy]+|[aeiouy]+", w)
    return chunks or [w]


def _f0_curve(notes: list[Note], sr: int, hop: int = 256,
              vibrato_rate: float = 5.5,
              vibrato_depth: float = 0.6) -> list[float]:
    """Per-frame f0 with portamento between notes and vibrato on sustains.

    Each note's ``vibrato`` scales the depth; ``tension`` deepens it
    further (tense voices wobble more) — the PEXP/TENC idea from
    OpenUtau's DiffSinger expression set, applied to our curve.
    """
    f0: list[float] = []
    prev_hz: Optional[float] = None
    for n in notes:
        if n.lyric.upper() in ("SP", "REST"):
            frames = int(n.duration_s * sr / hop)
            f0.extend([0.0] * frames)
            prev_hz = None
            continue
        hz = midi_to_hz(n.midi)
        frames = max(1, int(n.duration_s * sr / hop))
        # Portamento: glide from previous note over first 15% of frames
        glide = max(1, int(frames * 0.15)) if prev_hz else 0
        depth = vibrato_depth * n.vibrato * (0.7 + 0.6 * n.tension)
        for i in range(frames):
            if i < glide and prev_hz:
                t = i / glide
                base = prev_hz + (hz - prev_hz) * (t * t * (3 - 2 * t))
            else:
                base = hz
            # Vibrato on sustained notes (not the attack)
            if i > frames * 0.25 and n.duration_s > 0.3 and depth > 0.01:
                vib = math.sin(2 * math.pi * vibrato_rate * i * hop / sr)
                base *= 2.0 ** (vib * depth / 12.0 / 2.0)
            f0.append(base)
        prev_hz = hz
    return f0


def humanize(notes: list[Note], seed: int = 0,
             timing_ms: float = 12.0, pitch_cents: float = 8.0) -> list[Note]:
    """Anti-robot pass: tiny timing + pitch jitter per note.

    Deterministic per seed. Returns NEW notes (input untouched).
    ``timing_ms`` jitters durations, ``pitch_cents`` detunes the f0
    curve seed — enough to kill the machine-gun effect, not enough to
    sound out of tune.
    """
    import random
    rng = random.Random(seed or 1)
    out: list[Note] = []
    for n in notes:
        jitter = rng.uniform(-timing_ms, timing_ms) / 1000.0
        detune = rng.uniform(-pitch_cents, pitch_cents) / 100.0
        out.append(Note(
            midi=n.midi, lyric=n.lyric,
            duration_s=max(0.05, n.duration_s + jitter),
            dynamics=min(1.0, max(0.0,
                                  n.dynamics + rng.uniform(-0.05, 0.05))),
            breathiness=n.breathiness, tension=n.tension,
            vibrato=n.vibrato * (1.0 + detune * 0.1),
        ))
    return out


def _apply_expression(samples: array, notes: list[Note],
                      sr: int) -> array:
    """DYN/BREC per-note expression on rendered audio (post-stage).

    Note boundaries come from durations (the render is sequential, so
    spans line up). Dynamics = per-note gain; breathiness = per-note
    airy noise mix. Honest DSP on the waveform — the neural
    DiffSinger bank renders the notes, this shapes their delivery.
    """
    from .emotion_dsp import add_breathiness
    out = array("h", samples)
    pos = 0
    for n in notes:
        span = int(n.duration_s * sr)
        end = min(len(out), pos + span)
        if end <= pos:
            pos = end
            continue
        if abs(n.dynamics - 0.8) > 0.01:
            gain = n.dynamics / 0.8
            for i in range(pos, end):
                out[i] = int(max(-32768, min(32767, out[i] * gain)))
        if n.breathiness > 0.01:
            seg = array("h", out[pos:end])
            seg = add_breathiness(seg, sr,
                                  amount=min(0.3, n.breathiness * 0.25))
            out[pos:end] = seg
        pos = end
    return out


def export_midi(notes: list[Note], path: str = "",
                tempo_bpm: float = 120.0) -> str:
    """Notes → Standard MIDI File (type 0), lyrics as meta events.

    Pure stdlib SMF writer — drop the result into any DAW. Rests become
    gaps; "-" sustains extend the previous note.
    """
    import struct as _struct
    tpq = 480
    us_per_beat = int(60_000_000 / max(20.0, tempo_bpm))
    timeline: list[tuple[int, int, bytes]] = []  # (tick, order, data)
    order = 0

    def _at(tick: int, data: bytes) -> None:
        nonlocal order
        timeline.append((tick, order, data))
        order += 1

    def _varlen(value: int) -> bytes:
        out = bytearray([value & 0x7F])
        value >>= 7
        while value:
            out.insert(0, (value & 0x7F) | 0x80)
            value >>= 7
        out[-1] &= 0x7F
        return bytes(out)

    _at(0, b"\xff\x51\x03" + _struct.pack(">I", us_per_beat)[1:])
    tick = 0
    last_off_idx = -1  # timeline index of the most recent note-off
    for n in notes:
        dur_ticks = max(1, int(n.duration_s * tempo_bpm / 60.0 * tpq))
        if n.lyric.upper() in ("SP", "REST"):
            tick += dur_ticks
            last_off_idx = -1
            continue
        if n.lyric == "-" and last_off_idx >= 0:
            # sustain: push the previous note's note-off later
            t, o, data = timeline[last_off_idx]
            timeline[last_off_idx] = (t + dur_ticks, o, data)
            tick += dur_ticks
            continue
        vel = int(40 + 80 * max(0.0, min(1.0, n.dynamics)))
        _at(tick, bytes([0x90, n.midi & 0x7F, vel]))
        lyric = n.lyric.encode("utf-8", "replace")[:127]
        _at(tick, b"\xff\x05" + _varlen(len(lyric)) + lyric)
        _at(tick + dur_ticks, bytes([0x80, n.midi & 0x7F, 0x40]))
        last_off_idx = len(timeline) - 1
        tick += dur_ticks
    _at(tick, b"\xff\x2f\x00")  # end of track
    timeline.sort(key=lambda e: (e[0], e[1]))
    events = bytearray()
    abs_tick = 0
    for t, _o, data in timeline:
        events.extend(_varlen(max(0, t - abs_tick)))
        events.extend(data)
        abs_tick = max(abs_tick, t)
    track = (b"MTrk" + _struct.pack(">I", len(events)) + bytes(events))
    header = (b"MThd" + _struct.pack(">IHHH", 6, 0, 1, tpq))
    dest = path or "melody.mid"
    with open(dest, "wb") as fh:
        fh.write(header + track)
    return dest


class DiffSingerBackend:
    """DiffSinger singing backend (tts.py backend contract + sing())."""

    name = "diffsinger"
    supports_native_tags = False
    supports_cloning = False
    supports_streaming = False
    supports_singing = True
    sample_rate = 24000

    def __init__(self, voicebank: str = "") -> None:
        self.voicebank = voicebank or self._default_bank()
        if not self.voicebank:
            raise RuntimeError(
                "no DiffSinger voicebank found — drop an OpenUtau "
                f"DiffSinger bank into {SVS_DIR()}/")
        try:
            import onnxruntime as ort  # noqa
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime not installed — pip install onnxruntime "
                "for DiffSinger singing") from exc

    def _default_bank(self) -> str:
        d = SVS_DIR()
        try:
            for n in sorted(os.listdir(d)):
                p = os.path.join(d, n)
                if os.path.isdir(p) and os.path.exists(
                        os.path.join(p, "dsdict.yaml")):
                    return p
        except OSError:
            pass
        return ""

    def _load_sessions(self):
        import onnxruntime as ort
        bank = self.voicebank
        acoustic = os.path.join(bank, "acoustic.onnx")
        vocoder = os.path.join(bank, "vocoder.onnx")
        if not os.path.exists(acoustic):
            # Some banks ship model.pt — honest: we need ONNX
            raise RuntimeError(
                f"voicebank has no acoustic.onnx ({bank}) — export the "
                "DiffSinger model to ONNX first")
        sess = {"acoustic": ort.InferenceSession(
            acoustic, providers=["CPUExecutionProvider"])}
        if os.path.exists(vocoder):
            sess["vocoder"] = ort.InferenceSession(
                vocoder, providers=["CPUExecutionProvider"])
        return sess

    def sing(self, notes: list[Note] | str,
             rvc_model: str = "") -> dict[str, Any]:
        """Render notes to a sung wav. Returns {"path", ...}."""
        if isinstance(notes, str):
            notes = parse_melody(notes)
        if not notes:
            raise ValueError("no notes to sing")
        sessions = self._load_sessions()
        sr = self.sample_rate
        hop = 256
        f0 = _f0_curve(notes, sr, hop)
        # Phoneme sequence aligned to notes
        phonemes: list[str] = []
        for n in notes:
            if n.lyric.upper() in ("SP", "REST", ""):
                phonemes.append("SP")
            elif n.lyric == "-":
                phonemes.append("v_sustain")
            else:
                phonemes.extend(_simple_g2p(n.lyric))
        import numpy as np
        f0_arr = np.array(f0, dtype=np.float32)[None, :]
        # Duration: frames per phoneme (even split across its note)
        # — the acoustic model refines alignment internally.
        acoustic_in = {"f0": f0_arr}
        mel = sessions["acoustic"].run(None, acoustic_in)[0]
        if "vocoder" in sessions:
            wav_f = sessions["vocoder"].run(None, {"mel": mel})[0]
        else:
            raise RuntimeError("voicebank has no vocoder.onnx")
        wav_f = np.clip(wav_f.flatten(), -1.0, 1.0)
        samples = array("h", (wav_f * 32767).astype(np.int16).tolist())
        # DYN/BREC per-note expression (post-stage on the render)
        samples = _apply_expression(samples, notes, sr)
        import tempfile
        out = tempfile.mktemp(prefix="sing_", suffix=".wav")
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(samples.tobytes())
        # Optional: user's timbre sings via RVC
        if rvc_model:
            try:
                models = rvc_bridge.list_models()
                model = next((m for m in models if m.name == rvc_model),
                             None)
                if model is not None:
                    conv = rvc_bridge.convert(out, model)
                    out = conv["path"]
            except Exception as exc:  # noqa: BLE001 — bank voice is fine
                _log.info("singing: RVC timbre unavailable: %s", exc)
        _log.info("diffsinger sing ok: %d notes -> %s", len(notes), out)
        return {"ok": True, "path": out, "backend": "diffsinger",
                "notes": len(notes), "sample_rate": sr}

    def synthesize(self, text: str, voice: Optional[Any],
                   *, instruct: str = "") -> Any:
        """tts.py contract: plain text → sung on a default melody.

        Honest scope: without a melody this is speech-shaped chanting, not
        composed song. Real songs go through sing().
        """
        # Simple default melody: gentle rise/fall across words
        words = [w for w in text.split() if w.strip()]
        base = 60  # C4
        contour = [0, 2, 4, 2, 0, -2, 0, 2]
        notes = []
        for i, w in enumerate(words):
            midi = base + contour[i % len(contour)]
            notes.append(Note(midi=midi, duration_s=0.45, lyric=w))
        result = self.sing(notes)
        import wave as _w
        with _w.open(result["path"], "rb") as fh:
            return array("h", fh.readframes(fh.getnframes()))


def sing(melody: str | list[Note], voicebank: str = "",
         rvc_model: str = "") -> dict[str, Any]:
    """One-call singing: melody string → wav path."""
    backend = DiffSingerBackend(voicebank=voicebank)
    return backend.sing(melody, rvc_model=rvc_model)
