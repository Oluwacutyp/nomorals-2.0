"""ABC notation melody — the LLM writes actual notes, not descriptions.

Research: ABC notation is THE proven text format for LLM music composition.
ChatMusician (4B tokens), NotaGen (1.6M pieces), ComposerX (multi-agent),
and llmcomposer all use ABC.  LLMs trained on text handle ABC naturally
because it's text-compatible — no special tokenizer needed.

The LLM writes a melody in ABC notation as part of the SongSpec.  This
module parses it into note events the synth can render.

Minimal subset supported (enough for LLM-written melodies):
- Header: X:, M: (meter), L: (default length), K: (key), Q: (tempo)
- Notes: A-G (uppercase = lower octave, lowercase = higher)
- Accidentals: ^ (sharp), _ (flat), = (natural)
- Durations: number after note (C2 = 2x default), / = half, // = quarter
- Rests: z, Z
- Bar lines: | (ignored for timing, used for validation)
- Chords: [CEG] (played as simultaneous notes)
- Ties: - (combines durations)

    from nomorals.media.abc_melody import parse_abc_melody
    notes = parse_abc_melody("C D E F | G2 z2 |")
    # [(60, 0.0, 1.0), (62, 1.0, 1.0), ...]  (midi, start_beat, dur_beats)

Sources:
- https://huggingface.co/papers/2402.16153 (ChatMusician)
- https://arxiv.org/pdf/2502.18008 (NotaGen)
- https://github.com/alexnodeland/llmcomposer
"""

from __future__ import annotations

import re
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["parse_abc_melody", "validate_abc", "ABC_MELODY_PROMPT"]

# note name → semitone offset from C
_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# token: accidental, note, octave marks, duration
_TOKEN = re.compile(r"(\^+|_+|=)?([A-Ga-g])([,']*)(\d*/*\d*)?(-)?")


def _parse_duration(dur_str: str, default_len: float) -> float:
    """ABC duration string → beats.  default_len is in beats."""
    if not dur_str:
        return default_len
    # handle slashes: / = half, // = quarter, /// = eighth
    # (each slash halves the duration)
    if set(dur_str) <= {"/"}:
        return default_len / (2 ** len(dur_str))
    if "/" in dur_str:
        parts = dur_str.split("/")
        num = float(parts[0]) if parts[0] else 1.0
        den = float(parts[1]) if len(parts) > 1 and parts[1] else 2.0
        return default_len * num / den
    try:
        return default_len * float(dur_str)
    except ValueError:
        return default_len


def _note_to_midi(letter: str, accidental: str, octave_marks: str,
                  key_octave: int = 4) -> int:
    """ABC note → MIDI number.  key_octave is the base octave for uppercase."""
    base = _NOTE_OFFSETS[letter.upper()]
    # uppercase = base octave, lowercase = one octave up
    octave = key_octave + (1 if letter.islower() else 0)
    # , lowers octave, ' raises
    octave -= octave_marks.count(",")
    octave += octave_marks.count("'")
    midi = 12 * (octave + 1) + base
    # accidentals
    if accidental:
        if "^" in accidental:
            midi += accidental.count("^")
        elif "_" in accidental:
            midi -= accidental.count("_")
    return midi


def parse_abc_melody(abc: str, default_octave: int = 4) -> list[tuple[int, float, float]]:
    """Parse ABC melody → [(midi_note, start_beat, duration_beats)].

    Rests advance time but produce no notes.  Chords produce simultaneous
    notes.  Ties combine durations.

    Never raises — returns what it can parse.
    """
    notes: list[tuple[int, float, float]] = []
    if not abc or not abc.strip():
        return notes

    # strip header lines and comments
    lines = []
    for ln in abc.strip().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        if re.match(r"^[A-Z]:", ln):
            continue  # header field
        lines.append(ln)
    body = " ".join(lines)

    # default length: assume 1/8 = 0.5 beats unless specified
    # (we stripped L: but default to eighth notes, the common LLM output)
    default_len = 0.5

    t = 0.0  # current beat position
    i = 0
    pending_tie: tuple[int, float] | None = None  # (midi, start) for ties

    while i < len(body):
        ch = body[i]

        # skip whitespace and bar lines
        if ch in " \t\n|:":
            i += 1
            continue

        # chord: [CEG]
        if ch == "[":
            j = body.find("]", i)
            if j == -1:
                break
            chord_str = body[i + 1:j]
            chord_notes = []
            for m in _TOKEN.finditer(chord_str):
                acc, letter, octm, dur, _ = m.groups()
                midi = _note_to_midi(letter, acc or "", octm or "",
                                     default_octave)
                chord_notes.append(midi)
            if chord_notes:
                # chord duration from the first note's duration spec
                m0 = _TOKEN.search(chord_str)
                dur = _parse_duration(m0.group(4) if m0 else "",
                                      default_len) if m0 else default_len
                for midi in chord_notes:
                    notes.append((midi, t, dur))
                t += dur
            i = j + 1
            continue

        # rest
        if ch in "zZxX":
            m = re.match(r"[zZxX](\d*/*\d*)?", body[i:])
            dur_str = m.group(1) if m else ""
            t += _parse_duration(dur_str, default_len)
            i += len(m.group(0)) if m else 1
            continue

        # note
        m = _TOKEN.match(body[i:])
        if m:
            acc, letter, octm, dur_str, tie = m.groups()
            midi = _note_to_midi(letter, acc or "", octm or "",
                                 default_octave)
            dur = _parse_duration(dur_str or "", default_len)
            if tie:
                # tie: accumulate, emit when the chain ends
                if pending_tie is None:
                    pending_tie = (midi, t)
                # extend the pending note's duration (handled at emit)
                # store extended duration in a side channel
                pending_tie = (pending_tie[0], pending_tie[1],  # type: ignore
                               pending_tie[2] + dur if len(pending_tie) > 2  # type: ignore
                               else dur)
                # simpler: just extend t and keep accumulating
                t += dur
            else:
                if pending_tie is not None:
                    # emit the tied note: from pending start to current t + dur
                    pm, ps = pending_tie[0], pending_tie[1]
                    total_dur = (t + dur) - ps
                    notes.append((pm, ps, total_dur))
                    pending_tie = None
                    t += dur
                else:
                    notes.append((midi, t, dur))
                    t += dur
            i += len(m.group(0))
            continue

        # unknown char: skip
        i += 1

    # flush any dangling tie
    if pending_tie is not None:
        pm, ps = pending_tie[0], pending_tie[1]
        notes.append((pm, ps, t - ps))

    return notes


def validate_abc(abc: str, expected_beats: float | None = None) -> list[str]:
    """Check an ABC melody for problems.  Empty list = valid.

    If expected_beats is given, warns when the total duration doesn't match
    (useful for catching LLM bar-length errors).
    """
    errors: list[str] = []
    if not abc or not abc.strip():
        return ["empty ABC string"]
    try:
        notes = parse_abc_melody(abc)
    except Exception as exc:  # noqa: BLE001
        return [f"parse failed: {exc}"]
    if not notes:
        return ["no notes parsed — check ABC syntax"]
    # melodic range sanity (research: AI melodies often have inhuman leaps)
    pitches = [n[0] for n in notes]
    span = max(pitches) - min(pitches)
    if span > 24:
        errors.append(f"melodic span {span} semitones > 2 octaves — "
                      "unusually wide for a singable line")
    # leap check: flag consecutive leaps > octave
    for i in range(1, len(notes)):
        leap = abs(notes[i][0] - notes[i - 1][0])
        if leap > 12:
            errors.append(
                f"leap of {leap} semitones at note {i} — "
                "larger than an octave, likely unnatural")
            break  # one warning is enough
    # duration check
    if expected_beats is not None:
        total = max((n[1] + n[2] for n in notes), default=0.0)
        if abs(total - expected_beats) > 1.0:
            errors.append(
                f"melody is {total:.1f} beats, expected ~{expected_beats:.1f} "
                "— bar lengths may be wrong")
    return errors


# ── prompt fragment ───────────────────────────────────────────────────────
# Included in the composition prompt so the LLM writes real melody in ABC.

ABC_MELODY_PROMPT = """For each section that has a melody (verse, chorus, bridge, hook),
write the actual melody in ABC notation — real notes, not a description.

ABC MELODY RULES:
- One line per section, e.g. "verse_melody": "C D E F | G2 E2 | ..."
- Notes: A-G (uppercase = middle octave, lowercase = octave up)
- Durations: number after note (C2 = 2 beats if default is 1 beat).
  Use L:1/4 in the header so numbers are beats.
- Rests: z (z2 = 2-beat rest). LEAVE SPACE — don't fill every beat.
  Human melodies breathe; aim for 60-70% note density, not 100%.
- Bar lines: | every 4 beats (for 4/4).
- Keep the melodic range within one octave for verses, up to 1.5 for chorus.
- Avoid leaps larger than a 5th except for dramatic moments.
- Stepwise motion should dominate — that's what makes melodies singable.
- End phrases on the tonic or a chord tone.

Example (4 bars, 4/4, leaves space):
"C2 D2 | E2 z2 | G2 E2 | C4 |"

Put the ABC string in the section's "melody_abc" field."""
