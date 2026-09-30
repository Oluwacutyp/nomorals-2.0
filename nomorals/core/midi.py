"""A real MIDI file writer — no dependencies.

Produces format-1 Standard MIDI Files (SMF) that open in any DAW, phone
music app, or web player.  The MusicWriter uses it to turn a generated
song (chords + melody) into an actually-playable ``.mid`` file.

Capabilities:
* note-name ⇄ MIDI-number conversion (``C4`` ↔ 60)
* scales (major, natural/minor, harmonic minor, pentatonic, mixolydian,
  dorian, blues)
* roman-numeral chord progressions → concrete pitches
* seeded melody generation (scale-conformant, rhythm patterns, contour)
* a :class:`MidiBuilder` that assembles tracks (tempo, time signature,
  program change, note on/off, per-track name) and writes valid SMF bytes

The format implementation follows the SMF spec (chunk structure,
variable-length quantities, meta events).  A round-trip parser lives in
the test suite to prove the bytes are real.

    from nomorals.core.midi import MidiBuilder, generate_melody, scale_notes
    b = MidiBuilder(tempo=110, time_signature=(4, 4))
    b.set_program(0)                 # acoustic grand piano
    for n, start, dur in generate_melody("C", "major", bars=4, seed=7):
        b.note(n, start, dur, velocity=95)
    b.write("song.mid")
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "note_midi", "midi_note",
    "scale_notes", "chord_progression",
    "generate_melody", "generate_chord_bass",
    "MidiBuilder", "MidiTrack", "NoteEvent",
    "SCALES", "ROMAN_DEGREES",
]

NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()
SHARP_TO_FLAT = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

#: degree index (0-based) → semitones from the root
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
}

#: roman numerals → (degree index, quality) within a key
ROMAN_DEGREES: dict[str, tuple[int, bool]] = {
    "I": (0, True), "ii": (1, False), "iii": (2, False),
    "IV": (3, True), "V": (4, True), "vi": (5, False), "vii°": (6, False),
    "bVII": (6, True), "bVI": (5, True),
    # minor-key lowercase
    "i": (0, False), "iv": (3, False), "v": (4, False),
    "vii": (6, False), "bVI": (5, True), "bVII": (6, True),
}


# ───────────────────────────── note arithmetic ──────────────────────────────

def note_midi(name: str, middle_c: int = 60) -> int:
    """``"C4"`` → 60, ``"F#3"`` → 53.  Accepts sharps or flats."""
    name = name.strip()
    if not name or len(name) < 2:
        raise ValueError(f"bad note name {name!r}")
    if name[1] in "#b" and len(name) >= 3 and name[2].lstrip("#b").isdigit() or \
       (name[1] in "#b" and len(name) == 3 and name[2].isdigit()):
        letter, acc, octave = name[0].upper(), name[1], int(name[2])
    else:
        letter, acc, octave = name[0].upper(), "", int(name[1:])
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if letter not in base:
        raise ValueError(f"bad note letter {letter!r}")
    semis = base[letter]
    if acc == "#":
        semis += 1
    elif acc == "b":
        semis -= 1
    return middle_c + (octave - 4) * 12 + (semis - 0)


def midi_note(n: int) -> str:
    n = int(n) % 128
    octave = n // 12 - 1
    name = NOTE_NAMES[n % 12]
    return f"{name}{octave}"


def _parse_root(root: str) -> int:
    """Key root like "C", "F#", "Bb" → semitone offset from C."""
    root = root.strip().upper().replace("BB", "B")
    if not root:
        raise ValueError("empty key root")
    letter = root[0]
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    return base + (1 if "#" in root else (-1 if "b" in root else 0))


def scale_notes(root: str, mode: str, octaves: int = 2,
                base_octave: int = 4) -> list[int]:
    """Ascending scale as MIDI numbers.  ``mode`` is a key in SCALES."""
    mode = (mode or "major").lower()
    if mode == "minor" and "natural_minor" in SCALES and mode not in SCALES:
        mode = "natural_minor"
    if mode not in SCALES:
        raise ValueError(f"unknown scale {mode!r} (choose from {sorted(SCALES)})")
    pattern = SCALES[mode]
    root_off = _parse_root(root)
    out: list[int] = []
    for octv in range(octaves):
        for step in pattern:
            out.append(60 + (base_octave - 4) * 12 + root_off + step +
                       octv * 12)
    return out


def chord_progression(root: str, mode: str, romans: Sequence[str],
                       base_octave: int = 3) -> list[list[int]]:
    """``("C", "major", ["I","V","vi","IV"])`` → triads as MIDI number lists."""
    mode = (mode or "major").lower()
    if mode not in SCALES:
        raise ValueError(f"unknown scale {mode!r} (choose from {sorted(SCALES)})")
    root_off = _parse_root(root)
    base_c = 60 + (base_octave - 4) * 12
    chords: list[list[int]] = []
    for roman in romans:
        try:
            deg, _quality = ROMAN_DEGREES[roman]
        except KeyError as exc:
            raise ValueError(f"unknown roman numeral {roman!r}") from exc
        # triad = scale degree + its 3rd and 5th (wrapping the scale cycle)
        triad = [base_c + root_off + SCALES[mode][deg]]
        for step in (2, 4):
            idx = (deg + step) % len(SCALES[mode])
            extra_oct = 1 if idx < deg else 0
            triad.append(base_c + root_off + SCALES[mode][idx] + extra_oct * 12)
        chords.append(triad)
    return chords


# ─────────────────────────── melody generation ──────────────────────────────

#: rhythm pattern: (probability, beats) — beats in quarter notes
_RHYTHM = ((0.34, 1.0), (0.27, 0.5), (0.18, 0.25), (0.12, 1.5), (0.09, 2.0))
#: contour moves: (probability, semitone-ish step in scale-degree units)
_MOVES = ((0.22, 0), (0.18, 1), (0.18, -1), (0.14, 2), (0.12, -2),
          (0.09, 3), (0.07, -3))


def generate_melody(root: str, mode: str, *, bars: int = 4, beats: int = 4,
                    seed: int | None = None, velocity: int = 95,
                    base_octave: int = 4) -> list["NoteEvent"]:
    """Seeded, scale-conformant melody as a list of NoteEvents (in beats)."""
    rng = random.Random(seed)
    scale = scale_notes(root, mode, octaves=3, base_octave=base_octave)
    mid = len(scale) // 2
    events: list[NoteEvent] = []
    pos = 0.0
    degree = mid
    for bar in range(bars):
        bar_start = bar * beats
        bar_end = bar_start + beats
        first_of_bar = True
        while pos < bar_end - 1e-6:
            # pick rhythm
            r = rng.random()
            acc = 0.0
            dur = 0.5
            for p, d in _RHYTHM:
                acc += p
                if r <= acc:
                    dur = d
                    break
            # pick contour
            m = rng.random()
            acc = 0.0
            step = 0
            for p, s in _MOVES:
                acc += p
                if m <= acc:
                    step = s
                    break
            degree = max(0, min(len(scale) - 1, degree + step))
            # snap to the beat on downbeats for a musical feel
            if first_of_bar and bar == 0:
                degree = mid
            events.append(NoteEvent(scale[degree], pos, max(0.25, dur),
                                     velocity=velocity))
            pos += dur
            first_of_bar = False
    return events


def generate_chord_bass(root: str, mode: str, romans: Sequence[str], *,
                        bars_per_chord: int = 1, beats: int = 4,
                        seed: int | None = None) -> list[tuple[list[int], float, float]]:
    """Chord voicings over time: [(chord_notes, start_beat, dur_beats), …]."""
    rng = random.Random(seed)
    chords = chord_progression(root, mode, list(romans))
    out: list[tuple[list[int], float, float]] = []
    for i, chord in enumerate(chords):
        start = i * bars_per_chord * beats
        dur = bars_per_chord * beats
        # drop the lowest note to bass register for a fuller sound
        voicing = [n - 12 for n in chord]
        if rng.random() < 0.5:
            voicing = voicing + [voicing[0] + 12]
        out.append((voicing, float(start), float(dur)))
    return out


# ─────────────────────────── SMF file format ────────────────────────────────

def _vlq(value: int) -> bytes:
    """Variable-length quantity (SMF delta-time encoding)."""
    if value < 0:
        raise ValueError("delta must be >= 0")
    buf = [value & 0x7F]
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buf))


@dataclass
class NoteEvent:
    note: int
    start: float          # beats
    duration: float       # beats
    velocity: int = 96
    channel: int = 0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class _TickEvent:
    delta: int
    data: bytes


class MidiTrack:
    """Collects events, sorts by tick, emits a valid track chunk."""

    def __init__(self) -> None:
        self._events: list[_TickEvent] = []
        self.name: str = ""

    def _add_meta(self, tick: int, data: bytes) -> None:
        self._events.append(_TickEvent(tick, b"\xff" + data))

    # internal: raw event injection
    def _raw(self, delta: int, data: bytes) -> None:
        self._events.append(_TickEvent(delta, data))

    def _sorted(self) -> list[_TickEvent]:
        return sorted(self._events, key=lambda e: e.delta)

    def to_chunk(self) -> bytes:
        body = b""
        last = 0
        for ev in self._sorted():
            delta = ev.delta - last
            last = ev.delta
            body += _vlq(max(0, delta)) + ev.data
        return b"MTrk" + struct.pack(">I", len(body)) + body


class MidiBuilder:
    """Assemble a format-1 MIDI file.

    One tick = one 16th note (480 ticks per quarter at 4/4).
    """

    TICKS_PER_BEAT = 480

    def __init__(self, tempo: float = 120.0, time_signature: tuple[int, int] = (4, 4)) -> None:
        self.tempo = tempo
        self.time_signature = time_signature
        self.tracks: list[MidiTrack] = []
        self._programs: dict[int, int] = {}
        self._finalized = False
        track0 = MidiTrack()
        track0.name = "meta"
        self.tracks.append(track0)
        self.melody = MidiTrack()
        self.melody.name = "melody"
        self.tracks.append(self.melody)
        self.chords = MidiTrack()
        self.chords.name = "chords"
        self.tracks.append(self.chords)

    def set_program(self, program: int, channel: int = 0) -> None:
        self._programs[channel] = int(program) & 0x7F

    def _add_notes(self, track: MidiTrack, notes: Iterable[NoteEvent],
                   offset_beats: float = 0.0) -> None:
        evs: list[tuple[int, bytes]] = []
        for n in notes:
            ch = n.channel & 0x0F
            on = int(n.velocity) & 0x7F
            off = max(1, int(n.velocity * 0.8)) & 0x7F
            note_b = int(n.note) & 0x7F
            start_t = int((n.start + offset_beats) * self.TICKS_PER_BEAT)
            end_t = int(n.end * self.TICKS_PER_BEAT) + int(offset_beats * self.TICKS_PER_BEAT)
            evs.append((start_t, bytes([0x90 | ch, note_b, on])))
            evs.append((max(start_t + 1, end_t), bytes([0x80 | ch, note_b, off])))
        evs.sort()
        last = 0
        for tick, data in evs:
            track._raw(tick, data)
            last = tick

    def add_melody(self, notes: Iterable[NoteEvent],
                   offset_beats: float = 0.0) -> None:
        self._add_notes(self.melody, notes, offset_beats)

    def add_chords(self, chord_notes: Sequence[int], start_beat: float,
                   duration_beats: float, velocity: int = 80) -> None:
        for n in chord_notes:
            self.chords._raw(
                int(start_beat * self.TICKS_PER_BEAT),
                bytes([0x90, int(n) & 0x7F, velocity & 0x7F]))
            self.chords._raw(
                int((start_beat + duration_beats) * self.TICKS_PER_BEAT),
                bytes([0x80, int(n) & 0x7F, 0]))

    def _finalize_meta(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        t0 = self.tracks[0]
        # tempo meta (microseconds per quarter)
        usq = int(60_000_000 / max(20.0, self.tempo))
        t0._raw(0, b"\xff\x51\x03" + struct.pack(">I", usq)[1:])
        # time signature
        num, den = self.time_signature
        t0._raw(0, b"\xff\x58\x04" + bytes([num, den, 24, 8]))
        # program changes per channel
        for ch, prog in sorted(self._programs.items()):
            t0._raw(0, bytes([0xC0 | (ch & 0x0F), prog & 0x7F]))
        # end-of-track must be the last event of every track
        for tr in self.tracks:
            max_tick = max((e.delta for e in tr._events), default=0)
            tr._raw(max_tick + 1, b"\xff\x2f\x00")

    def build(self) -> bytes:
        self._finalize_meta()
        tracks = b"".join(tr.to_chunk() for tr in self.tracks)
        header = b"MThd" + struct.pack(">I", 6) + \
            struct.pack(">HHH", 1, len(self.tracks), 480)
        return header + tracks

    def write(self, path: str) -> str:
        with open(path, "wb") as fh:
            fh.write(self.build())
        return path
