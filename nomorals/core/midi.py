"""A real MIDI file writer — no dependencies.

Produces format-1 Standard MIDI Files (SMF) that open in any DAW, phone
music app, or web player.  The MusicWriter uses it to turn a generated
song (chords + melody) into an actually-playable ``.mid`` file.

Capabilities:
* note-name ⇄ MIDI-number conversion (``C4`` ↔ 60)
* scales (major, natural/minor, harmonic minor, pentatonic, mixolydian,
  dorian, blues)
* roman-numeral chord progressions → concrete pitches
* chord voicings (closed, drop-2, spread, stabs) plus functional chord
  extensions (7ths/9ths: V gets a dominant 7th, ii a minor 7th, …)
* seeded melody generation: motif-based (a one-bar motif with variations,
  so the tune has an identity), downbeats anchored to chord tones, final
  note cadencing on the tonic
* seeded bass lines (roots, driving eighths, walking, syncopated,
  log-drum, halftime, riff, reggae bubble, breakdown pad, metal gallop)
* a real drum kit: 22 genre patterns (four-on-the-floor, boom bap, trap,
  afrobeats, amapiano, one-drop, swing, lofi, highlife, disco, reggaeton,
  country, metal, drill, funk, …) on the GM percussion channel, with
  ghost notes
* call-and-response counter-melodies, humanize (micro-timing + velocity
  jitter), root transposition for key changes
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
    "generate_bass_line", "generate_drums", "generate_counter_melody",
    "apply_voicing", "add_chord_extensions", "humanize", "transpose_root",
    "MidiBuilder", "MidiTrack", "NoteEvent",
    "SCALES", "ROMAN_DEGREES", "DRUM_PATTERNS",
    "KICK", "SNARE", "RIM", "CLAP",
    "CLOSED_HAT", "PEDAL_HAT", "OPEN_HAT",
    "CRASH", "RIDE", "TAMBOURINE", "COWBELL",
    "HIGH_TOM", "LOW_TOM", "LOW_FLOOR_TOM",
    "CONGA_HIGH", "CONGA_LOW", "TIMBALE_HIGH", "SHAKER",
    "GM_INSTRUMENTS", "GM_FAMILIES", "DRUM_MAP",
    "program_name", "program_number", "gm_family", "quantize",
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
    "bIII": (2, True),
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
                    base_octave: int = 4,
                    chord_tones: Sequence[Sequence[int]] | None = None,
                    ) -> list[NoteEvent]:
    """Seeded, scale-conformant melody as a list of NoteEvents (in beats).

    Built like a real tune: a one-bar **motif** is written first, then
    replayed across the bars with small variations (so the line has an
    identity instead of wandering), and the final note **cadences on the
    tonic**.  When ``chord_tones`` is given (one pitch set per bar — the
    arrangement passes the actual chord under each bar), the downbeat of
    every bar snaps to the nearest chord tone, so the melody always sits
    on the harmony.
    """
    rng = random.Random(seed)
    scale = scale_notes(root, mode, octaves=3, base_octave=base_octave)
    top = len(scale) - 1
    mid = len(scale) // 2
    cycle = len(SCALES[(mode or "major").lower()])

    def pick_rhythm() -> float:
        r = rng.random()
        acc = 0.0
        for p, d in _RHYTHM:
            acc += p
            if r <= acc:
                return d
        return 0.5

    def pick_step() -> int:
        m = rng.random()
        acc = 0.0
        for p, s in _MOVES:
            acc += p
            if m <= acc:
                return s
        return 0

    def clamp(d: int) -> int:
        return max(0, min(top, d))

    # 1) write the motif: one bar of rhythm + contour
    motif: list[tuple[float, float, int]] = []   # (offset, dur, degree)
    pos = 0.0
    degree = mid
    while pos < beats - 1e-6:
        dur = min(pick_rhythm(), beats - pos)
        degree = clamp(degree + pick_step())
        motif.append((pos, max(0.25, dur), degree))
        pos += dur
    if motif:
        # the motif opens on the tonic area: identity starts here
        motif[0] = (motif[0][0], motif[0][1], mid)

    def nearest_degree(tones: Sequence[int]) -> int:
        best, best_d = mid, 10 ** 9
        for d in range(len(scale)):
            dd = min(abs(scale[d] - t) for t in tones)
            if dd < best_d or (dd == best_d and abs(d - mid) < abs(best - mid)):
                best, best_d = d, dd
        return best

    # 2) replay the motif across the bars with variation
    events: list[NoteEvent] = []
    n_tones = len(chord_tones) if chord_tones else 0
    for bar in range(bars):
        tones = chord_tones[bar % n_tones] if n_tones else ()
        for off, dur, mdeg in motif:
            var = 0 if bar == 0 else rng.choice((-1, 0, 0, 0, 1))
            degree = clamp(mdeg + var)
            if off == 0 and tones:
                degree = nearest_degree(tones)
            events.append(NoteEvent(scale[degree], bar * beats + off,
                                    max(0.25, dur), velocity=velocity))

    # 3) cadence: the last note lands on the tonic root
    if events:
        deg0 = mid - (mid % cycle)
        last = events[-1]
        events[-1] = NoteEvent(scale[deg0], last.start, last.duration,
                               last.velocity, last.channel)
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


# ─────────────────────────── rhythm section ──────────────────────────────────

#: General MIDI percussion note numbers (always channel 9).
KICK = 36
SNARE = 38
RIM = 37
CLAP = 39
CLOSED_HAT = 42
PEDAL_HAT = 44
OPEN_HAT = 46
CRASH = 49
RIDE = 51
TAMBOURINE = 54
COWBELL = 56
HIGH_TOM = 50
LOW_TOM = 45
LOW_FLOOR_TOM = 43
CONGA_HIGH = 63
CONGA_LOW = 64
TIMBALE_HIGH = 65
SHAKER = 82

#: One bar of drums as (offset_beats, note, velocity_scale).  Offsets assume
#: 4/4; the bar loops for every bar of the section.
DRUM_PATTERNS: dict[str, tuple[tuple[float, int, float], ...]] = {
    "four_floor": (
        (0.0, KICK, 1.0), (1.0, KICK, 1.0), (2.0, KICK, 1.0), (3.0, KICK, 1.0),
        (1.0, CLAP, 1.0), (3.0, CLAP, 1.0),
        (0.5, CLOSED_HAT, 0.75), (1.5, CLOSED_HAT, 0.75),
        (2.5, CLOSED_HAT, 0.75), (3.5, CLOSED_HAT, 0.75),
        (3.75, OPEN_HAT, 0.7), (0.0, CRASH, 0.8),
    ),
    "boombap": (
        (0.0, KICK, 1.0), (1.75, KICK, 0.9), (2.5, KICK, 0.8),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (0.0, CLOSED_HAT, 0.6), (0.5, CLOSED_HAT, 0.6),
        (1.0, CLOSED_HAT, 0.6), (1.5, CLOSED_HAT, 0.6),
        (2.0, CLOSED_HAT, 0.6), (2.5, CLOSED_HAT, 0.6),
        (3.0, CLOSED_HAT, 0.6), (3.5, CLOSED_HAT, 0.6),
        (3.75, OPEN_HAT, 0.6),
    ),
    "trap": (
        (0.0, KICK, 1.0), (2.75, KICK, 0.9),
        (2.0, SNARE, 1.0),
        *[(i * 0.25, CLOSED_HAT, 0.55) for i in range(16)],
        (3.5, CLOSED_HAT, 0.8), (3.625, CLOSED_HAT, 0.8),
        (3.75, CLOSED_HAT, 0.85), (3.875, CLOSED_HAT, 0.9),
        (3.9, OPEN_HAT, 0.7),
    ),
    "afrobeats": (
        (0.0, KICK, 1.0), (2.0, KICK, 0.9), (3.5, KICK, 0.7),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        *[(i * 0.25, SHAKER, 0.5) for i in range(16)],
        (1.75, CONGA_HIGH, 0.7), (2.75, CONGA_HIGH, 0.7),
        (0.5, CONGA_LOW, 0.6),
    ),
    "amapiano": (
        (0.0, KICK, 1.0), (1.0, KICK, 1.0), (2.0, KICK, 1.0), (3.0, KICK, 1.0),
        *[(i * 0.25, SHAKER, 0.55) for i in range(16)],
        (0.5, OPEN_HAT, 0.75), (1.5, OPEN_HAT, 0.75),
        (2.5, OPEN_HAT, 0.75), (3.5, OPEN_HAT, 0.75),
        (1.0, SNARE, 0.8), (3.0, SNARE, 0.8),
        (2.5, COWBELL, 0.6),
    ),
    "one_drop": (
        (2.0, KICK, 1.0), (2.0, RIM, 0.9),
        (0.0, CLOSED_HAT, 0.7), (0.5, CLOSED_HAT, 0.7),
        (1.0, CLOSED_HAT, 0.7), (1.5, CLOSED_HAT, 0.7),
        (2.0, CLOSED_HAT, 0.7), (2.5, CLOSED_HAT, 0.7),
        (3.0, CLOSED_HAT, 0.7), (3.5, CLOSED_HAT, 0.7),
        (0.5, TAMBOURINE, 0.6), (1.5, TAMBOURINE, 0.6),
        (2.5, TAMBOURINE, 0.6), (3.5, TAMBOURINE, 0.6),
        (0.0, CRASH, 0.7),
    ),
    "swing": (
        (0.0, RIDE, 0.9), (0.67, RIDE, 0.6), (1.0, RIDE, 0.9),
        (1.67, RIDE, 0.6), (2.0, RIDE, 0.9), (2.67, RIDE, 0.6),
        (3.0, RIDE, 0.9), (3.67, RIDE, 0.6),
        (0.0, KICK, 0.7), (2.0, KICK, 0.7),
        (1.5, SNARE, 0.45), (3.75, SNARE, 0.45),
        (3.5, CRASH, 0.5),
    ),
    "shuffle": (
        (0.0, KICK, 0.9), (2.0, KICK, 0.9), (2.5, KICK, 0.7),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (0.0, CLOSED_HAT, 0.6), (0.67, CLOSED_HAT, 0.5),
        (1.33, CLOSED_HAT, 0.6), (2.0, CLOSED_HAT, 0.6),
        (2.67, CLOSED_HAT, 0.5), (3.33, CLOSED_HAT, 0.6),
    ),
    "rock": (
        (0.0, KICK, 1.0), (2.0, KICK, 1.0), (2.5, KICK, 0.8),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        *[(i * 0.5, CLOSED_HAT, 0.75) for i in range(8)],
        (0.0, CRASH, 0.9),
    ),
    "lofi": (
        (0.0, KICK, 0.9), (2.5, KICK, 0.7),
        (1.0, SNARE, 0.9), (3.0, SNARE, 0.9),
        (0.0, CLOSED_HAT, 0.45), (0.67, CLOSED_HAT, 0.4),
        (1.33, CLOSED_HAT, 0.45), (2.0, CLOSED_HAT, 0.45),
        (2.67, CLOSED_HAT, 0.4), (3.33, CLOSED_HAT, 0.45),
        *[(i * 0.5, SHAKER, 0.35) for i in range(8)],
    ),
    "halftime_pop": (
        (0.0, KICK, 1.0),
        (2.0, SNARE, 1.0),
        *[(i * 0.5, CLOSED_HAT, 0.7) for i in range(8)],
        (3.5, OPEN_HAT, 0.6), (0.0, CRASH, 0.7),
    ),
    "highlife": (
        (0.0, KICK, 1.0), (2.0, KICK, 0.9),
        (1.0, SNARE, 0.9), (3.0, SNARE, 0.9),
        (0.5, OPEN_HAT, 0.7), (1.5, OPEN_HAT, 0.7),
        (2.5, OPEN_HAT, 0.7), (3.5, OPEN_HAT, 0.7),
        (0.75, CONGA_HIGH, 0.6), (1.75, CONGA_HIGH, 0.6),
        (2.75, CONGA_HIGH, 0.6), (3.75, CONGA_HIGH, 0.6),
        (0.0, CRASH, 0.7),
    ),
    "rnb_slow": (
        (0.0, KICK, 0.9), (2.75, KICK, 0.7),
        (2.0, SNARE, 0.95),
        *[(i * 0.5, CLOSED_HAT, 0.5) for i in range(8)],
        (1.0, TAMBOURINE, 0.5), (3.0, TAMBOURINE, 0.5),
    ),
    "gospel": (
        (0.0, KICK, 0.9), (1.0, KICK, 0.9), (2.0, KICK, 0.9), (3.0, KICK, 0.9),
        (1.0, CLAP, 1.0), (3.0, CLAP, 1.0),
        (1.0, SNARE, 0.9), (3.0, SNARE, 0.9),
        *[(i * 0.5, TAMBOURINE, 0.7) for i in range(8)],
        (0.0, CRASH, 0.8),
    ),
    "dancehall": (
        (0.0, KICK, 1.0), (2.75, KICK, 0.95),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (0.5, CLOSED_HAT, 0.7), (1.5, CLOSED_HAT, 0.7),
        (2.5, CLOSED_HAT, 0.7), (3.5, CLOSED_HAT, 0.7),
        (3.5, RIM, 0.6), (0.0, CRASH, 0.6),
    ),
    "buildup": (
        (0.0, KICK, 0.9), (1.0, KICK, 0.9), (2.0, KICK, 0.9), (3.0, KICK, 0.9),
        *[(i * 0.25, SNARE, 0.55 + 0.45 * (i / 15)) for i in range(16)],
        *[(i * 0.5, OPEN_HAT, 0.6) for i in range(8)],
    ),
    "breakdown": (
        *[(i * 0.5, SHAKER, 0.4) for i in range(8)],
        (1.0, TAMBOURINE, 0.35), (3.0, TAMBOURINE, 0.35),
    ),
    "disco": (
        (0.0, KICK, 1.0), (1.0, KICK, 1.0), (2.0, KICK, 1.0), (3.0, KICK, 1.0),
        (1.0, CLAP, 1.0), (3.0, CLAP, 1.0),
        *[(i * 0.25, OPEN_HAT, 0.7) for i in range(16)],
        (0.0, CRASH, 0.8), (2.0, TAMBOURINE, 0.6),
    ),
    "reggaeton": (
        # dembow: kick-snare-shell pattern, the riddim's heartbeat
        (0.0, KICK, 1.0), (0.75, KICK, 0.7), (1.75, KICK, 0.8),
        (2.5, KICK, 0.9),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        *[(i * 0.5, CLOSED_HAT, 0.6) for i in range(8)],
        (0.5, CONGA_LOW, 0.6), (2.25, CONGA_HIGH, 0.6),
        (3.5, TAMBOURINE, 0.55),
    ),
    "country": (
        # train-beat shuffle: steady chug with a backbeat snap
        (0.0, KICK, 0.9), (1.0, KICK, 0.8), (2.0, KICK, 0.9),
        (3.0, KICK, 0.8),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (0.0, CLOSED_HAT, 0.6), (0.67, CLOSED_HAT, 0.5),
        (1.33, CLOSED_HAT, 0.6), (2.0, CLOSED_HAT, 0.6),
        (2.67, CLOSED_HAT, 0.5), (3.33, CLOSED_HAT, 0.6),
        (0.5, TAMBOURINE, 0.55), (2.5, TAMBOURINE, 0.55),
        (0.0, CRASH, 0.7),
    ),
    "metal": (
        # double-bass gallop under a driving rock kit
        *[(i * 0.5, KICK, 0.95 if i % 2 == 0 else 0.75)
          for i in range(8)],
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        *[(i * 0.25, CLOSED_HAT, 0.6) for i in range(16)],
        (0.0, CRASH, 0.9), (2.0, CRASH, 0.85),
        (1.5, HIGH_TOM, 0.7), (3.5, LOW_TOM, 0.75),
    ),
    "drill": (
        # sliding 808s + swung hats, snare on the 3
        (0.0, KICK, 1.0), (0.75, KICK, 0.9), (2.5, KICK, 0.9),
        (2.0, SNARE, 1.0),
        *[(i * (1 / 3), CLOSED_HAT, 0.55 if i % 3 else 0.75)
          for i in range(12)],
        (3.875, CLOSED_HAT, 0.85), (0.0, CRASH, 0.6),
    ),
    "funk": (
        # tight pocket: ghosted snare, chicken-scratch 16ths
        (0.0, KICK, 1.0), (1.75, KICK, 0.8), (2.5, KICK, 0.85),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (2.875, SNARE, 0.4), (3.625, SNARE, 0.35),
        *[(i * 0.25, CLOSED_HAT, 0.65) for i in range(16)],
        (2.0, OPEN_HAT, 0.6), (0.5, TAMBOURINE, 0.6),
        (2.5, TAMBOURINE, 0.6), (0.0, CRASH, 0.7),
    ),
    "uk_drill": (
        # UK drill: sparse menacing kicks, snare on the 3, swung hats,
        # 32nd-note hat roll into the next bar
        (0.0, KICK, 1.0), (0.75, KICK, 0.85), (2.5, KICK, 0.95),
        (3.25, KICK, 0.7),
        (2.0, SNARE, 1.0),
        *[(i * (1 / 3), CLOSED_HAT, 0.7 if i % 3 == 0 else 0.5)
          for i in range(11)],
        (3.625, CLOSED_HAT, 0.85), (3.6875, CLOSED_HAT, 0.85),
        (3.75, CLOSED_HAT, 0.9), (3.8125, CLOSED_HAT, 0.9),
        (3.875, CLOSED_HAT, 0.95), (3.9375, CLOSED_HAT, 0.95),
        (0.0, CRASH, 0.55),
    ),
    "grime": (
        # eski grime: cold square-wave bounce, syncopated kicks,
        # snare on 2 and 4 with eski rim clicks
        (0.0, KICK, 1.0), (0.75, KICK, 0.8), (1.5, KICK, 0.9),
        (2.75, KICK, 0.85), (3.5, KICK, 0.75),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (0.5, RIM, 0.7), (2.5, RIM, 0.7),
        *[(i * 0.25, CLOSED_HAT, 0.6) for i in range(16)],
        (3.875, OPEN_HAT, 0.65), (0.0, CRASH, 0.6),
    ),
    "phonk": (
        # dark memphis phonk: cowbell lead, heavy swung kicks,
        # half-time snare, rolling hats
        (0.0, KICK, 1.0), (0.75, KICK, 0.85), (1.75, KICK, 0.9),
        (2.5, KICK, 0.95),
        (2.0, SNARE, 1.0),
        (0.0, COWBELL, 0.85), (0.75, COWBELL, 0.85),
        (1.5, COWBELL, 0.85), (2.25, COWBELL, 0.7),
        (3.0, COWBELL, 0.85),
        *[(i * 0.25, CLOSED_HAT, 0.5) for i in range(16)],
        (3.75, OPEN_HAT, 0.65), (0.0, CRASH, 0.6),
    ),
    "jersey": (
        # jersey club: triplet kick pattern, clap/snare stabs,
        # chopped feel
        (0.0, KICK, 1.0), (0.33, KICK, 0.85), (0.67, KICK, 0.9),
        (1.5, KICK, 0.9), (2.0, KICK, 1.0),
        (2.33, KICK, 0.85), (2.67, KICK, 0.9), (3.5, KICK, 0.8),
        (1.0, CLAP, 1.0), (3.0, CLAP, 1.0),
        (1.75, SNARE, 0.6), (3.75, SNARE, 0.7),
        *[(i * 0.25, CLOSED_HAT, 0.6) for i in range(16)],
        (0.0, CRASH, 0.65),
    ),
    "dnb": (
        # amen-style drum & bass 2-step: rolling break, ghost snares,
        # driving hats at ~174 bpm
        (0.0, KICK, 1.0), (1.75, KICK, 0.85), (2.5, KICK, 0.9),
        (1.0, SNARE, 1.0), (3.0, SNARE, 1.0),
        (0.875, SNARE, 0.45), (2.875, SNARE, 0.45),
        (3.5, SNARE, 0.55), (3.75, SNARE, 0.6),
        *[(i * 0.25, CLOSED_HAT, 0.62) for i in range(16)],
        (0.0, CRASH, 0.7), (2.0, RIDE, 0.6),
    ),
}


def generate_drums(pattern: str, *, bars: int = 4, beats: int = 4,
                   seed: int | None = None, velocity: int = 100,
                   channel: int = 9) -> list[NoteEvent]:
    """Seeded drum pattern as NoteEvents on the GM percussion channel.

    ``pattern`` is a key in :data:`DRUM_PATTERNS`; ``""`` (or an unknown
    name) yields no drums — used by styles with no kit (e.g. classical).
    """
    rng = random.Random(seed)
    hits = DRUM_PATTERNS.get(pattern or "")
    if not hits or bars <= 0:
        return []
    events: list[NoteEvent] = []
    for bar in range(bars):
        base = bar * beats
        for off, note, scale in hits:
            dur = 0.5 if note in (OPEN_HAT, CRASH, RIDE) else 0.25
            vel = max(1, min(127, int(velocity * scale)))
            # ghost notes: a few extra feathered snare/hat taps keep the
            # groove human instead of grid-locked
            events.append(NoteEvent(note, base + off, dur,
                                   velocity=vel, channel=channel))
            if rng.random() < 0.06 and note == CLOSED_HAT:
                events.append(NoteEvent(
                    note, base + off + 0.125, 0.125,
                    velocity=max(1, vel // 3), channel=channel))
    return events


#: bass-line styles: how the bass moves under each chord
BASS_STYLES = ("roots", "driving_eighths", "walking", "syncopated",
               "logdrum", "halftime", "riff", "reggae_bubble", "breakdown_pad",
               "gallop", "808_slide", "sub_pulse", "reese")


def generate_bass_line(root: str, mode: str, romans: Sequence[str], *,
                       style: str = "roots", bars_per_chord: int = 1,
                       beats: int = 4, seed: int | None = None,
                       velocity: int = 90, channel: int = 3) -> list[NoteEvent]:
    """Seeded bass line under a chord progression.

    Styles: ``roots`` (half-note roots), ``driving_eighths`` (eighth-note
    roots), ``walking`` (jazz quarter-note walk approaching each next root),
    ``syncopated`` (afrobeats offbeat groove), ``logdrum`` (amapiano-style
    syncopated mid-register stabs), ``halftime`` (dotted-half roots),
    ``riff`` (root/fifth/octave rock riff), ``reggae_bubble`` (offbeat
    stabs), ``breakdown_pad`` (one long root per chord),
    ``808_slide`` (long 808 notes with portamento glides via
    ``NoteEvent.slide_to``), ``sub_pulse`` (trap/phonk eighth-note sub
    pulses), ``reese`` (dnb sustained root + octave stack).
    """
    rng = random.Random(seed)
    chords = chord_progression(root, mode, list(romans))
    if not chords:
        return []
    style = style or "roots"
    if style not in BASS_STYLES:
        style = "roots"
    scale = scale_notes(root, mode, octaves=3, base_octave=2)
    out: list[NoteEvent] = []

    def ev(note: int, start: float, dur: float, vel: int = velocity) -> None:
        out.append(NoteEvent(max(1, min(127, note)), start,
                             max(0.1, dur), velocity=max(1, min(127, vel)),
                             channel=channel))

    for i, chord in enumerate(chords):
        start = i * bars_per_chord * beats
        bar_len = bars_per_chord * beats
        root_n = chord[0] - 12                      # bass register
        fifth_n = root_n + 7
        oct_n = root_n + 12
        nxt = chords[(i + 1) % len(chords)][0] - 12
        accent = velocity
        soft = max(1, velocity - 18)
        if style == "roots":
            ev(root_n, start, bar_len / 2, accent)
            ev(root_n, start + bar_len / 2, bar_len / 2, soft)
        elif style == "driving_eighths":
            n = int(bar_len / 0.5)
            for j in range(n):
                ev(root_n if j % 4 else root_n, start + j * 0.5, 0.45,
                   accent if j % 2 == 0 else soft)
        elif style == "walking":
            # quarter-note walk: root, scale step, approach note, next root
            ev(root_n, start, 0.9, accent)
            idx = min(range(len(scale)),
                      key=lambda k: abs(scale[k] - root_n))
            step = scale[min(len(scale) - 1, idx + 2)]
            ev(step, start + 1, 0.9, soft)
            approach = nxt - 1 if rng.random() < 0.5 else nxt - 2
            ev(approach, start + 2, 0.9, soft)
            ev(nxt, start + 3, 1.6, accent)
        elif style == "syncopated":
            for off, nn, vv in ((0.0, root_n, accent), (0.75, fifth_n, soft),
                                (1.5, root_n, accent), (2.5, oct_n, soft),
                                (3.25, fifth_n, soft)):
                ev(nn, start + off, 0.45, vv)
        elif style == "logdrum":
            # amapiano log drum: syncopated stabs an octave up, melodic
            hi = root_n + 12
            seq = ((0.5, hi, 0.28), (1.0, hi + 7, 0.28),
                   (1.75, hi, 0.28), (2.5, hi + 5, 0.28),
                   (3.0, hi + 7, 0.4))
            for off, nn, dd in seq:
                ev(nn, start + off, dd, accent)
        elif style == "halftime":
            ev(root_n, start, bar_len * 0.72, accent)
            ev(fifth_n, start + bar_len * 0.75, bar_len * 0.22, soft)
        elif style == "riff":
            for off, nn, vv in ((0.0, root_n, accent), (0.5, root_n, soft),
                                (0.75, fifth_n, soft), (1.5, root_n, accent),
                                (2.0, oct_n, accent), (3.0, fifth_n, soft),
                                (3.5, root_n, soft)):
                ev(nn, start + off, 0.42, vv)
        elif style == "reggae_bubble":
            for off, nn in ((0.5, root_n), (1.5, fifth_n),
                            (2.5, root_n), (3.5, fifth_n)):
                ev(nn, start + off, 0.35, soft)
        elif style == "gallop":
            # metal gallop: 8th-16th-16th ride on the root, octave punch
            for beat in range(int(bar_len)):
                ev(root_n, start + beat, 0.32, accent)
                ev(root_n, start + beat + 0.5, 0.18, soft)
                ev(root_n, start + beat + 0.75, 0.2, soft)
                if beat % 2 == 1:
                    ev(oct_n, start + beat + 0.5, 0.2, accent)
        elif style == "808_slide":
            # drill/trap 808s: long sustained sub notes with portamento
            # slides into the fifth/octave — the signature glide. Each
            # long note carries slide_to so the synth sweeps the pitch.
            nxt_root = chords[(i + 1) % len(chords)][0] - 12
            long1 = bar_len * 0.55
            long2 = bar_len * 0.30
            out.append(NoteEvent(max(1, min(127, root_n)), start,
                                 max(0.1, long1), velocity=accent,
                                 channel=channel, slide_to=fifth_n))
            out.append(NoteEvent(max(1, min(127, fifth_n)),
                                 start + long1, max(0.1, long2),
                                 velocity=soft, channel=channel,
                                 slide_to=nxt_root))
            out.append(NoteEvent(max(1, min(127, nxt_root)),
                                 start + long1 + long2,
                                 max(0.1, bar_len - long1 - long2),
                                 velocity=accent, channel=channel))
        elif style == "sub_pulse":
            # trap/phonk sub: deep eighth-note pulses hammering the root
            n = int(bar_len / 0.5)
            for j in range(n):
                nn = root_n if j % 8 < 6 else (fifth_n if j % 2 else oct_n)
                ev(nn, start + j * 0.5, 0.46,
                   accent if j % 2 == 0 else soft)
        elif style == "reese":
            # dnb reese: sustained root doubled an octave up — thick,
            # menacing, constantly moving under the break
            ev(root_n, start, bar_len * 0.9, accent)
            ev(oct_n, start, bar_len * 0.9, soft)
            ev(fifth_n, start + bar_len * 0.5, bar_len * 0.4, soft)
        else:  # breakdown_pad
            ev(root_n, start, bar_len * 0.95, soft)
    return out


def generate_counter_melody(root: str, mode: str, *, bars: int = 4,
                            beats: int = 4, seed: int | None = None,
                            base_octave: int = 5, velocity: int = 88,
                            channel: int = 2) -> list[NoteEvent]:
    """Call-and-response counter line: fills that answer the lead melody.

    Plays only on odd bars (the "gaps"), beats 2–4, in a higher register —
    the classic backing-vocal / lead-guitar answer phrase.
    """
    rng = random.Random(seed)
    scale = scale_notes(root, mode, octaves=3, base_octave=base_octave)
    mid = len(scale) // 2
    out: list[NoteEvent] = []
    for bar in range(bars):
        if bar % 2 == 0:
            continue
        pos = bar * beats + 2.0
        end = bar * beats + beats
        degree = mid + rng.choice((2, 3, 4))
        while pos < end - 1e-6:
            dur = rng.choice((0.5, 0.5, 0.75, 1.0))
            step = rng.choice((-2, -1, -1, 0, 1, 1, 2))
            degree = max(0, min(len(scale) - 1, degree + step))
            out.append(NoteEvent(scale[degree], pos, min(dur, end - pos),
                                 velocity=velocity, channel=channel))
            pos += dur
    return out


def apply_voicing(notes: Sequence[int], kind: str = "closed") -> list[int]:
    """Revoice a chord: ``closed`` (identity), ``drop2`` (second-highest
    voice down an octave — the jazz/R&B spread), ``spread`` (widen the
    outer voices), ``stabs`` (identity; the caller shortens durations)."""
    n = sorted(int(x) for x in notes)
    if kind == "drop2" and len(n) >= 3:
        idx = len(n) - 2
        n[idx] -= 12
    elif kind == "spread" and len(n) >= 3:
        n[0] -= 12
        n[-1] += 12
    return sorted(set(max(1, min(127, x)) for x in n))


def add_chord_extensions(notes: Sequence[int], roman: str, mode: str,
                         kind: str = "sevenths") -> list[int]:
    """Color a root-position triad with its 7th (``"sevenths"``) or 7th+9th
    (``"ninths"``) — the jazz/R&B/lofi color.

    The 7th is the scale degree three diatonic steps above the chord's own
    root (so V gets a dominant 7th, ii a minor 7th, I a major 7th — real
    functional color, not a pasted-on interval).  Unknown ``roman`` or
    ``kind`` returns the chord unchanged so arrangement code never dies
    mid-song.
    """
    n = sorted(int(x) for x in notes)
    if len(n) < 3 or kind not in ("sevenths", "ninths"):
        return n
    mode = (mode or "major").lower()
    cycle = SCALES.get(mode)
    if cycle is None:
        return n
    try:
        deg, _quality = ROMAN_DEGREES[roman]
    except KeyError:
        return n
    root_n = n[0]
    L = len(cycle)
    root_semis = cycle[deg]

    def stack(step_deg: int, octave_up: int) -> int:
        semis = cycle[step_deg % L]
        diff = (semis - root_semis) % 12
        return root_n + 12 * octave_up + diff

    out = list(n)
    seventh = stack(deg + 6, 1)          # a seventh above the chord root
    if 1 <= seventh <= 127:
        out.append(seventh)
    if kind == "ninths":
        ninth = stack(deg + 1, 2)        # a ninth above the chord root
        if 1 <= ninth <= 127:
            out.append(ninth)
    return sorted(set(out))


def humanize(events: list[NoteEvent], *, timing: float = 0.02,
             velocity: int = 6, seed: int | None = None) -> list[NoteEvent]:
    """Micro-timing + velocity jitter so parts breathe like a performance.

    Deterministic under ``seed``; never moves a note before beat 0.
    """
    rng = random.Random(seed)
    out: list[NoteEvent] = []
    for e in events:
        out.append(NoteEvent(
            e.note,
            max(0.0, e.start + rng.uniform(-timing, timing)),
            e.duration,
            velocity=max(1, min(127, e.velocity + rng.randint(-velocity,
                                                             velocity))),
            channel=e.channel))
    return out


def transpose_root(root: str, semitones: int) -> str:
    """Transpose a key root name by semitones: ``("B", 1)`` → ``"C"``."""
    off = _parse_root(root)
    name = NOTE_NAMES[(off + semitones) % 12]
    # keep flats flat-ish for readability
    flat = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}
    if "b" in root and name in flat:
        return flat[name]
    return name


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
    slide_to: int | None = None  # portamento target (MIDI note): when set,
                                 # audio renderers glide the pitch from
                                 # ``note`` to ``slide_to`` over the
                                 # duration (808 slides). MIDI writers
                                 # ignore it (plain note-on).

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

    def new_track(self, name: str) -> MidiTrack:
        """Add a named track (bass, drums, counter-melody, …) to the file."""
        track = MidiTrack()
        track.name = name
        self.tracks.append(track)
        return track

    def add_notes(self, track: MidiTrack, notes: Iterable[NoteEvent],
                  offset_beats: float = 0.0) -> None:
        """Add note events to any track (public wrapper)."""
        self._add_notes(track, notes, offset_beats)

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
                   duration_beats: float, velocity: int = 80,
                   channel: int = 0) -> None:
        ch = channel & 0x0F
        for n in chord_notes:
            self.chords._raw(
                int(start_beat * self.TICKS_PER_BEAT),
                bytes([0x90 | ch, int(n) & 0x7F, velocity & 0x7F]))
            self.chords._raw(
                int((start_beat + duration_beats) * self.TICKS_PER_BEAT),
                bytes([0x80 | ch, int(n) & 0x7F, 0]))

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


# ─────────────────────── General MIDI tables ─────────────────────────────────

#: The 128 General MIDI program names, in order (8 families × 16).
#: Every MIDI tool ships this table; composition code can now say
#: ``program_number("Nylon Guitar")`` instead of memorizing 25.
GM_INSTRUMENTS: tuple[str, ...] = (
    # Piano
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2",
    "Harpsichord", "Clavinet",
    # Chromatic Percussion
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer",
    # Organ
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    # Guitar
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar",
    "Guitar harmonics",
    # Bass
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
    "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1",
    "Synth Bass 2",
    # Strings
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
    "Pizzicato Strings", "Orchestral Harp", "Timpani",
    # Ensemble
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1",
    "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
    "Orchestra Hit",
    # Brass
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn",
    "Brass Section", "Synth Brass 1", "Synth Brass 2",
    # Reed
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet",
    # Pipe
    "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle",
    "Shakuhachi", "Whistle", "Ocarina",
    # Synth Lead
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
    "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)",
    "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    # Synth Pad
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)",
    "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)",
    "Pad 8 (sweep)",
    # Synth Effects
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    # Ethnic
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle",
    "Shanai",
    # Percussive
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
    "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    # Sound Effects
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
)

#: GM family names for program ranges (program // 8).
GM_FAMILIES: tuple[str, ...] = (
    "Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings",
    "Ensemble", "Brass", "Reed", "Pipe", "Synth Lead", "Synth Pad",
    "Synth Effects", "Ethnic", "Percussive", "Sound Effects",
)

#: Full GM percussion map: note number → name (channel 9/10).
DRUM_MAP: dict[int, str] = {
    35: "Acoustic Bass Drum", 36: "Bass Drum 1", 37: "Side Stick",
    38: "Acoustic Snare", 39: "Hand Clap", 40: "Electric Snare",
    41: "Low Floor Tom", 42: "Closed Hi Hat", 43: "High Floor Tom",
    44: "Pedal Hi-Hat", 45: "Low Tom", 46: "Open Hi-Hat",
    47: "Low-Mid Tom", 48: "Hi-Mid Tom", 49: "Crash Cymbal 1",
    50: "High Tom", 51: "Ride Cymbal 1", 52: "Chinese Cymbal",
    53: "Ride Bell", 54: "Tambourine", 55: "Splash Cymbal",
    56: "Cowbell", 57: "Crash Cymbal 2", 58: "Vibraslap",
    59: "Ride Cymbal 2", 60: "Hi Bongo", 61: "Low Bongo",
    62: "Mute Hi Conga", 63: "Open Hi Conga", 64: "Low Conga",
    65: "High Timbale", 66: "Low Timbale", 67: "High Agogo",
    68: "Low Agogo", 69: "Cabasa", 70: "Maracas",
    71: "Short Whistle", 72: "Long Whistle", 73: "Short Guiro",
    74: "Long Guiro", 75: "Claves", 76: "Hi Wood Block",
    77: "Low Wood Block", 78: "Mute Cuica", 79: "Open Cuica",
    80: "Mute Triangle", 81: "Open Triangle",
}


def program_name(program: int) -> str:
    """GM program number (0–127) → instrument name."""
    if not 0 <= program <= 127:
        raise ValueError(f"GM program must be 0–127, got {program}")
    return GM_INSTRUMENTS[program]


def program_number(name: str) -> int:
    """Instrument name → GM program number (case-insensitive).

    Matches exact name first, then unique prefix, then unique substring —
    ``"nylon"`` finds ``Acoustic Guitar (nylon)``.
    """
    want = name.strip().lower()
    exact = [i for i, n in enumerate(GM_INSTRUMENTS) if n.lower() == want]
    if exact:
        return exact[0]
    prefix = [i for i, n in enumerate(GM_INSTRUMENTS)
              if n.lower().startswith(want)]
    if len(prefix) == 1:
        return prefix[0]
    sub = [i for i, n in enumerate(GM_INSTRUMENTS) if want in n.lower()]
    if len(sub) == 1:
        return sub[0]
    cands = prefix or sub
    raise ValueError(
        f"ambiguous or unknown GM instrument {name!r}"
        + (f" (matches: {', '.join(GM_INSTRUMENTS[i] for i in cands[:5])})"
           if cands else ""))


def gm_family(program: int) -> str:
    """The GM family name for a program number (e.g. 25 → "Guitar")."""
    if not 0 <= program <= 127:
        raise ValueError(f"GM program must be 0–127, got {program}")
    return GM_FAMILIES[program // 8]


def quantize(events: list["NoteEvent"], grid: float = 0.25,
             *, strength: float = 1.0) -> list["NoteEvent"]:
    """Snap event starts (and durations) to a beat grid.

    ``grid=0.25`` = 16th notes. ``strength`` 0.0–1.0 blends between the
    original timing and full quantization — 1.0 is robotic, 0.5 keeps the
    human feel while tightening the groove (the standard DAW control).
    """
    import dataclasses as _dc

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be 0.0–1.0")
    out: list[NoteEvent] = []
    for ev in events:
        q_start = round(ev.start / grid) * grid
        q_dur = max(grid, round(ev.duration / grid) * grid)
        out.append(_dc.replace(
            ev,
            start=ev.start + (q_start - ev.start) * strength,
            duration=ev.duration + (q_dur - ev.duration) * strength,
        ))
    return out
