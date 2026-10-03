"""End-to-end tests for the Devon music composer (Wave: music-1.0).

Covers :mod:`nomorals.core.midi` (SMF writer, scales, chord extensions,
motif melodies, drum kit, bass lines, voicings, humanize) and
:mod:`nomorals.media.music` (24 genres, lyric rhyme engine, arrangement,
dynamics, key changes).  Every genre must produce a real, playable MIDI
file — no placeholder styles.
"""

from __future__ import annotations

import random
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core import midi as M
from nomorals.core.midi import (
    BASS_STYLES,
    DRUM_PATTERNS,
    ROMAN_DEGREES,
    SCALES,
    MidiBuilder,
    NoteEvent,
    add_chord_extensions,
    apply_voicing,
    chord_progression,
    generate_bass_line,
    generate_chord_bass,
    generate_counter_melody,
    generate_drums,
    generate_melody,
    humanize,
    midi_note,
    note_midi,
    scale_notes,
    transpose_root,
)
from nomorals.media.music import (
    STYLES,
    MusicCreator,
    resolve_style,
)


def _context() -> SimpleNamespace:
    root = tempfile.mkdtemp(prefix="music_composer_")
    return SimpleNamespace(settings=SimpleNamespace(workspace_dir=root),
                           _root=root)


def _parse_smf(data: bytes) -> dict:
    """Minimal SMF parser: header + per-track event bytes.  Proves the
    bytes the composer writes are a real format-1 MIDI file."""
    assert data[:4] == b"MThd", "missing MThd magic"
    hlen, fmt, ntracks, division = struct.unpack(">IHHH", data[4:14])
    assert hlen == 6 and fmt == 1, f"expected format-1, got {fmt}"
    tracks = []
    pos = 14
    for _ in range(ntracks):
        assert data[pos:pos + 4] == b"MTrk", "missing MTrk magic"
        (tlen,) = struct.unpack(">I", data[pos + 4:pos + 8])
        tracks.append(data[pos + 8:pos + 8 + tlen])
        pos += 8 + tlen
    assert pos == len(data), "trailing bytes after last track"
    return {"ntracks": ntracks, "division": division, "tracks": tracks}


def _has_status(track: bytes, status_hi: int) -> bool:
    """Scan a track's raw bytes for a MIDI status nibble (0x90 note-on,
    0x80 note-off, 0xC0 program change).  VLQ deltas are skipped by the
    standard variable-length walk."""
    i = 0
    while i < len(track):
        # skip VLQ delta
        while i < len(track) and track[i] & 0x80:
            i += 1
        i += 1  # last delta byte
        if i >= len(track):
            break
        b = track[i]
        if b == 0xFF:  # meta event: FF type len bytes...
            i += 1
            if i >= len(track):
                break
            i += 1  # type
            ln = 0
            while i < len(track) and track[i] & 0x80:
                ln = (ln << 7) | (track[i] & 0x7F)
                i += 1
            if i < len(track):
                ln = (ln << 7) | track[i]
                i += 1
            i += ln
            continue
        if (b & 0xF0) == status_hi:
            return True
        # channel voice message length
        hi = b & 0xF0
        i += 1 + (1 if hi in (0xC0, 0xD0) else 2)
    return False


# ─────────────────────────── midi primitives ───────────────────────────────

class NoteArithmeticTests(unittest.TestCase):
    def test_note_midi_roundtrip(self) -> None:
        self.assertEqual(note_midi("C4"), 60)
        self.assertEqual(note_midi("A4"), 69)
        self.assertEqual(note_midi("F#3"), 54)
        self.assertEqual(note_midi("Bb2"), 46)
        self.assertEqual(midi_note(60), "C4")
        self.assertEqual(midi_note(69), "A4")

    def test_bad_note_names_raise(self) -> None:
        for bad in ("", "H4", "C", "4"):
            with self.assertRaises(ValueError, msg=bad):
                note_midi(bad)

    def test_scale_notes(self) -> None:
        self.assertEqual(
            [n % 12 for n in scale_notes("C", "major", octaves=1)],
            [0, 2, 4, 5, 7, 9, 11])
        self.assertEqual(len(scale_notes("F#", "dorian", octaves=2)), 14)
        with self.assertRaises(ValueError):
            scale_notes("C", "nope")

    def test_chord_progression_triads(self) -> None:
        chords = chord_progression("C", "major", ["I", "V", "vi", "IV"])
        self.assertEqual([n % 12 for n in chords[0]], [0, 4, 7])   # C E G
        self.assertEqual([n % 12 for n in chords[1]], [7, 11, 2])  # G B D
        with self.assertRaises(ValueError):
            chord_progression("C", "major", ["X9"])

    def test_transpose_root(self) -> None:
        self.assertEqual(transpose_root("B", 1), "C")
        self.assertEqual(transpose_root("C", -1), "B")
        self.assertEqual(transpose_root("F#", 6), "C")


# ─────────────────────────── chord extensions ──────────────────────────────

class ChordExtensionTests(unittest.TestCase):
    def test_chord_bass_voicings_over_time(self) -> None:
        rows = generate_chord_bass("C", "major", ["I", "V", "vi", "IV"],
                                   bars_per_chord=1, beats=4, seed=1)
        self.assertEqual(len(rows), 4)
        for i, (chord, start, dur) in enumerate(rows):
            self.assertEqual(start, i * 4.0)
            self.assertEqual(dur, 4.0)
            self.assertGreaterEqual(len(chord), 3)  # triad, maybe doubled
    def test_cmaj7(self) -> None:
        tri = chord_progression("C", "major", ["I"])[0]
        self.assertEqual(add_chord_extensions(tri, "I", "major", "sevenths"),
                         [48, 52, 55, 71])  # + B (major 7th)

    def test_dominant_seventh_on_V(self) -> None:
        tri = chord_progression("C", "major", ["V"])[0]
        ext = add_chord_extensions(tri, "V", "major", "sevenths")
        self.assertEqual(ext[-1] % 12, 5)  # F natural: G7, not Gmaj7

    def test_minor_seventh_on_ii(self) -> None:
        tri = chord_progression("C", "major", ["ii"])[0]
        ext = add_chord_extensions(tri, "ii", "major", "sevenths")
        self.assertEqual(ext, [50, 53, 57, 72])  # Dm7

    def test_ninths_adds_both(self) -> None:
        tri = chord_progression("C", "major", ["I"])[0]
        ext = add_chord_extensions(tri, "I", "major", "ninths")
        self.assertEqual(ext, [48, 52, 55, 71, 74])  # Cmaj9

    def test_unknowns_are_noops(self) -> None:
        tri = chord_progression("C", "major", ["I"])[0]
        self.assertEqual(add_chord_extensions(tri, "X9", "major", "sevenths"),
                         sorted(tri))
        self.assertEqual(add_chord_extensions(tri, "I", "major", "elevenths"),
                         sorted(tri))
        self.assertEqual(add_chord_extensions([60, 64], "I", "major",
                                              "sevenths"), [60, 64])


# ─────────────────────────── motif melodies ────────────────────────────────

class MelodyTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        a = generate_melody("C", "major", bars=4, seed=7)
        b = generate_melody("C", "major", bars=4, seed=7)
        self.assertEqual([(e.note, e.start, e.duration) for e in a],
                         [(e.note, e.start, e.duration) for e in b])

    def test_scale_conformant(self) -> None:
        pcs = {n % 12 for n in scale_notes("D", "dorian", octaves=3)}
        for e in generate_melody("D", "dorian", bars=8, seed=3):
            self.assertIn(e.note % 12, pcs)

    def test_motif_identity_across_bars(self) -> None:
        # the motif gives the tune an identity: bar 0 and bar 2 share
        # their rhythmic skeleton (same count of notes per bar)
        mel = generate_melody("C", "major", bars=4, seed=11)
        per_bar = [[e for e in mel if int(e.start // 4) == b]
                   for b in range(4)]
        self.assertEqual(len(per_bar[0]), len(per_bar[2]))
        self.assertGreater(len(per_bar[0]), 0)

    def test_cadence_lands_on_tonic(self) -> None:
        for root, mode in (("C", "major"), ("A", "minor"), ("G", "mixolydian"),
                           ("E", "harmonic_minor")):
            mel = generate_melody(root, mode, bars=4, seed=5)
            tonic_pc = {"C": 0, "A": 9, "G": 7, "E": 4}[root]
            self.assertEqual(mel[-1].note % 12, tonic_pc,
                              f"{root} {mode} did not cadence on tonic")

    def test_chord_tone_downbeats(self) -> None:
        prog = ["I", "V", "vi", "IV"]
        chords = chord_progression("C", "major", prog)
        mel = generate_melody("C", "major", bars=4, seed=9,
                              chord_tones=chords)
        for bar in range(4):
            first = min((e for e in mel if int(e.start // 4) == bar),
                        key=lambda e: e.start)
            pcs = {n % 12 for n in chords[bar]}
            self.assertIn(first.note % 12, pcs,
                          f"bar {bar} downbeat not on a chord tone")

    def test_covers_full_length(self) -> None:
        mel = generate_melody("C", "major", bars=6, seed=1)
        self.assertLessEqual(max(e.end for e in mel), 6 * 4 + 1e-6)
        self.assertGreaterEqual(min(e.start for e in mel), 0.0)


# ─────────────────────────── drums & bass ──────────────────────────────────

class DrumTests(unittest.TestCase):
    def test_all_patterns_wellformed(self) -> None:
        for name, hits in DRUM_PATTERNS.items():
            self.assertTrue(hits, f"{name} is empty")
            for off, note, scale in hits:
                self.assertGreaterEqual(off, 0.0, name)
                self.assertLess(off, 4.0, name)
                self.assertGreaterEqual(note, 35, name)
                self.assertLessEqual(note, 82, name)
                self.assertGreater(scale, 0.0, name)
                self.assertLessEqual(scale, 1.0, name)

    def test_new_patterns_have_backbeat(self) -> None:
        for name in ("disco", "reggaeton", "country", "metal", "drill",
                     "funk"):
            hits = DRUM_PATTERNS[name]
            notes = {n for _, n, _ in hits}
            self.assertIn(M.KICK, notes, name)
            self.assertTrue(M.SNARE in notes or M.CLAP in notes, name)

    def test_disco_is_four_on_floor_with_claps(self) -> None:
        kicks = [o for o, n, _ in DRUM_PATTERNS["disco"] if n == M.KICK]
        self.assertEqual(kicks, [0.0, 1.0, 2.0, 3.0])
        claps = [o for o, n, _ in DRUM_PATTERNS["disco"] if n == M.CLAP]
        self.assertEqual(claps, [1.0, 3.0])

    def test_metal_has_double_kick(self) -> None:
        kicks = [o for o, n, _ in DRUM_PATTERNS["metal"] if n == M.KICK]
        self.assertGreaterEqual(len(kicks), 8)

    def test_generate_drums_empty_and_unknown(self) -> None:
        self.assertEqual(generate_drums("", bars=4), [])
        self.assertEqual(generate_drums("nope", bars=4), [])
        self.assertEqual(generate_drums("rock", bars=0), [])

    def test_generate_drums_on_channel_9(self) -> None:
        for e in generate_drums("trap", bars=2, seed=4):
            self.assertEqual(e.channel, 9)
            self.assertGreaterEqual(e.start, 0.0)
            self.assertLess(e.start, 8.0)
            self.assertTrue(1 <= e.velocity <= 127)

    def test_generate_drums_deterministic(self) -> None:
        a = generate_drums("afrobeats", bars=4, seed=8)
        b = generate_drums("afrobeats", bars=4, seed=8)
        self.assertEqual([(e.note, e.start) for e in a],
                         [(e.note, e.start) for e in b])


class BassTests(unittest.TestCase):
    def test_all_styles_wellformed(self) -> None:
        prog = ["i", "bVI", "bVII", "i"]
        for style in BASS_STYLES:
            evs = generate_bass_line("A", "minor", prog, style=style, seed=2)
            self.assertTrue(evs, style)
            for e in evs:
                self.assertTrue(1 <= e.note <= 127, style)
                self.assertGreater(e.duration, 0, style)
                self.assertGreaterEqual(e.start, 0.0, style)
                self.assertLess(e.start, 16.0, style)
                self.assertEqual(e.channel, 3, style)

    def test_unknown_style_falls_back_to_roots(self) -> None:
        a = generate_bass_line("C", "major", ["I", "IV"], style="nope",
                               seed=1)
        b = generate_bass_line("C", "major", ["I", "IV"], style="roots",
                               seed=1)
        self.assertEqual([(e.note, e.start) for e in a],
                         [(e.note, e.start) for e in b])

    def test_gallop_rhythm(self) -> None:
        evs = generate_bass_line("E", "harmonic_minor", ["i"], style="gallop",
                                 seed=1)
        offs = sorted({round(e.start % 1, 2) for e in evs})
        self.assertIn(0.0, offs)
        self.assertIn(0.5, offs)
        self.assertIn(0.75, offs)

    def test_walking_approaches_next_root(self) -> None:
        evs = generate_bass_line("C", "major", ["I", "IV"], style="walking",
                                 seed=1)
        # the beat-2 note approaches the next root from a semitone or two
        # below (beat 3 lands on the next root itself)
        bar0 = [e for e in evs if e.start < 3.0]
        approach = max(bar0, key=lambda e: e.start)
        self.assertAlmostEqual(approach.start, 2.0)
        f_root = chord_progression("C", "major", ["IV"])[0][0] - 12
        self.assertIn(f_root - approach.note, (1, 2))


# ─────────────────────────── voicings & humanize ────────────────────────────

class VoicingTests(unittest.TestCase):
    def test_drop2(self) -> None:
        self.assertEqual(apply_voicing([60, 64, 67, 71], "drop2"),
                         [55, 60, 64, 71])

    def test_spread_widens(self) -> None:
        v = apply_voicing([60, 64, 67], "spread")
        self.assertLess(v[0], 60)
        self.assertGreater(v[-1], 67)

    def test_closed_is_identity(self) -> None:
        self.assertEqual(apply_voicing([67, 60, 64], "closed"), [60, 64, 67])

    def test_unknown_voicing_is_identity(self) -> None:
        self.assertEqual(apply_voicing([60, 64, 67], "nope"), [60, 64, 67])

    def test_humanize_deterministic_and_bounded(self) -> None:
        evs = [NoteEvent(60, 1.0, 1.0, velocity=100)]
        a = humanize(evs, seed=3)
        b = humanize(evs, seed=3)
        self.assertEqual(a[0].start, b[0].start)
        self.assertEqual(a[0].velocity, b[0].velocity)
        self.assertGreaterEqual(a[0].start, 0.0)
        self.assertTrue(1 <= a[0].velocity <= 127)
        # never pushes a beat-0 note before zero
        z = humanize([NoteEvent(60, 0.0, 1.0)], seed=3)
        self.assertGreaterEqual(z[0].start, 0.0)


class CounterMelodyTests(unittest.TestCase):
    def test_answers_on_odd_bars(self) -> None:
        ctr = generate_counter_melody("C", "major", bars=4, seed=6)
        self.assertTrue(ctr)
        for e in ctr:
            bar = int(e.start // 4)
            self.assertEqual(bar % 2, 1)
            self.assertGreaterEqual(e.start % 4, 2.0)

    def test_empty_for_single_bar(self) -> None:
        self.assertEqual(generate_counter_melody("C", "major", bars=1,
                                                 seed=6), [])


# ─────────────────────────── SMF writer ────────────────────────────────────

class SmfWriterTests(unittest.TestCase):
    def test_format1_file_parses(self) -> None:
        b = MidiBuilder(tempo=128, time_signature=(4, 4))
        b.set_program(81, channel=0)
        b.add_melody(generate_melody("C", "minor", bars=4, seed=2))
        b.add_notes(b.new_track("bass"),
                    generate_bass_line("C", "minor", ["i", "i", "i", "i"],
                                       seed=2))
        b.add_notes(b.new_track("drums"), generate_drums("four_floor",
                                                          bars=4, seed=2))
        parsed = _parse_smf(b.build())
        # meta + melody + chords + bass + drums
        self.assertEqual(parsed["ntracks"], 5)
        self.assertEqual(parsed["division"], 480)

    def test_note_on_off_and_program_change_present(self) -> None:
        b = MidiBuilder(tempo=120)
        b.set_program(40, channel=0)
        b.add_melody([NoteEvent(60, 0.0, 1.0, velocity=100)])
        parsed = _parse_smf(b.build())
        blob = b"".join(parsed["tracks"])
        self.assertTrue(_has_status(blob, 0x90))   # note on
        self.assertTrue(_has_status(blob, 0x80))   # note off
        self.assertTrue(_has_status(blob, 0xC0))   # program change

    def test_end_of_track_on_every_track(self) -> None:
        b = MidiBuilder(tempo=100)
        b.add_melody(generate_melody("G", "major", bars=2, seed=1))
        for tr in _parse_smf(b.build())["tracks"]:
            self.assertTrue(tr.endswith(b"\xff\x2f\x00"))

    def test_write_returns_path_and_reads_back(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="smf_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        target = tmp / "x.mid"
        b = MidiBuilder(tempo=90)
        b.add_melody(generate_melody("F", "major", bars=2, seed=1))
        self.assertEqual(b.write(str(target)), str(target))
        _parse_smf(target.read_bytes())


# ─────────────────────────── style specs ───────────────────────────────────

class StyleSpecTests(unittest.TestCase):
    VALID_VOICINGS = {"closed", "drop2", "spread", "stabs"}
    VALID_EXT = {"", "sevenths", "ninths"}

    def test_all_styles_have_full_spec(self) -> None:
        self.assertGreaterEqual(len(STYLES), 24)
        for key, spec in STYLES.items():
            with self.subTest(style=key):
                self.assertTrue(spec.label)
                lo, hi = spec.tempo
                self.assertLess(lo, hi)
                self.assertIn(spec.mode, SCALES)
                self.assertGreaterEqual(len(spec.progressions), 2)
                for prog in spec.progressions:
                    self.assertEqual(len(prog), 4)
                    for roman in prog:
                        self.assertIn(roman, ROMAN_DEGREES, roman)
                self.assertTrue(spec.sections)
                self.assertGreater(sum(b for _, b in spec.sections), 8)
                self.assertTrue(spec.instrumentation)
                self.assertTrue(spec.palette)
                if spec.drum_pattern:
                    self.assertIn(spec.drum_pattern, DRUM_PATTERNS)
                self.assertIn(spec.bass_pattern, BASS_STYLES)
                self.assertIn(spec.voicing, self.VALID_VOICINGS)
                self.assertIn(spec.extensions, self.VALID_EXT)
                for p in spec.programs:
                    self.assertTrue(0 <= p <= 127)

    def test_aliases(self) -> None:
        cases = {"lo-fi": "lofi", "boom bap": "hiphop", "r&b": "rnb",
                 "dembow": "reggaeton", "nashville": "country",
                 "thrash": "metal", "boogie": "disco", "hymn": "choir",
                 "p-funk": "funk", "uk drill": "drill", "drone": "ambient",
                 "alté": "afrobeats"}
        for alias, want in cases.items():
            self.assertEqual(resolve_style(alias).name, want)

    def test_unknown_style_raises(self) -> None:
        from nomorals.core.errors import ToolError
        with self.assertRaises(ToolError):
            resolve_style("not-a-genre-xyz")

    def test_every_section_name_known_to_arranger(self) -> None:
        from nomorals.media import music as mu
        known = set(mu._SECTION_DYN) | {"verse", "chorus"}
        for key, spec in STYLES.items():
            for name, _ in spec.sections:
                self.assertIn(name, known,
                              f"{key} uses unknown section {name!r}")


# ─────────────────────────── compose end-to-end ────────────────────────────

class ComposeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _context()
        self.addCleanup(shutil.rmtree, self.ctx._root, True)
        self.creator = MusicCreator(self.ctx)

    def test_every_style_produces_real_midi(self) -> None:
        for key in sorted(STYLES):
            with self.subTest(style=key):
                song = self.creator.compose("midnight city", style=key,
                                            seed=123, with_midi=True)
                self.assertTrue(song.midi_path, key)
                data = Path(song.midi_path).read_bytes()
                self.assertGreater(len(data), 1024, key)
                parsed = _parse_smf(data)
                # meta + melody + chords + bass + drums + counter
                self.assertEqual(parsed["ntracks"], 6, key)
                blob = b"".join(parsed["tracks"])
                self.assertTrue(_has_status(blob, 0x90), key)
                for s in song.sections:
                    self.assertTrue(s.chords, f"{key}/{s.name}")

    def test_compose_is_seed_deterministic(self) -> None:
        a = self.creator.compose("ocean drive", style="pop", seed=99,
                                 with_midi=True)
        b = self.creator.compose("ocean drive", style="pop", seed=99,
                                 with_midi=True)
        self.assertEqual(
            [ln for s in a.sections for ln in s.lyrics],
            [ln for s in b.sections for ln in s.lyrics])
        self.assertEqual(Path(a.midi_path).read_bytes(),
                         Path(b.midi_path).read_bytes())

    def test_verse_follows_rhyme_scheme(self) -> None:
        from nomorals.media import music as mu
        groups = {w: g for g in mu._RHYME_GROUPS for w in g}
        song = self.creator.compose("city lights", style="pop", seed=7,
                                    with_midi=False)
        verse = next(s for s in song.sections if s.name == "verse")
        self.assertEqual(len(verse.lyrics), 8)
        tails = [ln.rstrip(".,!?").split()[-1].lower()
                 for ln in verse.lyrics]
        # every line ending belongs to a real rhyme group…
        for t in tails:
            self.assertIn(t, groups, f"unrhymed tail {t!r}")
        # …and the 8 lines follow one of the declared schemes
        # (the 4-letter scheme repeats across the 8 lines)
        schemes = {"ABAB", "AABB", "ABBA"}
        letters = [chr(65 + list(mu._RHYME_GROUPS).index(groups[t]))
                   for t in tails]
        # normalize to first-seen-letter form
        seen: dict[str, str] = {}
        norm = "".join(seen.setdefault(ch, chr(65 + len(seen)))
                       for ch in letters)
        self.assertIn(norm[:4], schemes, f"scheme {norm} tails {tails}")
        self.assertEqual(norm[:4], norm[4:],
                         f"scheme does not repeat: {norm}")

    def test_lyric_tails_are_grammatical(self) -> None:
        # regression: the old tail-swap produced "across the paper folder".
        # tails must come from rhyme groups, never mid-phrase fragments.
        from nomorals.media import music as mu
        vocab = {w for g in mu._RHYME_GROUPS for w in g}
        for key in ("pop", "hiphop", "jazz", "rock", "country", "choir",
                    "afrobeats", "dancehall", "rnb"):
            song = self.creator.compose("midnight city", style=key, seed=21,
                                        with_midi=False)
            for s in song.sections:
                if not s.lyrics or s.name == "verse":
                    continue  # verse covered above
                for ln in s.lyrics:
                    tail = ln.rstrip(".,!?").split()[-1].lower()
                    if any(p in ln.lower()
                           for p in ("dey ", "omo", "na so e be", "jaiye")):
                        continue  # pidgin ad-libs don't rhyme
                    self.assertIn(tail, vocab,
                                  f"{key}/{s.name}: bad tail in {ln!r}")

    def test_chorus_repeats_the_hook(self) -> None:
        song = self.creator.compose("fire", style="pop", seed=5,
                                    with_midi=False)
        chorus = next(s for s in song.sections if s.name == "chorus")
        self.assertEqual(chorus.lyrics[2], chorus.lyrics[0])

    def test_tag_sections(self) -> None:
        for key in ("pop", "gospel", "rnb", "country", "choir"):
            song = self.creator.compose("home", style=key, seed=3,
                                        with_midi=False)
            tag = next(s for s in song.sections if s.name == "tag")
            self.assertEqual(len(tag.lyrics), 2)
            self.assertTrue(tag.chords)

    def test_meter_compliance(self) -> None:
        from nomorals.media import music as mu
        song = self.creator.compose("neon rain", style="hiphop", seed=13,
                                    with_midi=False)
        lo, hi = mu._METER["hiphop"]
        for s in song.sections:
            for ln in s.lyrics:
                n = mu._syllables(ln)
                self.assertLessEqual(n, hi + 3, ln)
                self.assertGreaterEqual(n, lo - 3, ln)

    def test_to_dict_and_markdown_roundtrip(self) -> None:
        song = self.creator.compose("dawn", style="lofi", seed=4,
                                    with_midi=False)
        d = song.to_dict()
        self.assertEqual(d["style"], "lofi")
        self.assertTrue(d["sections"])
        md = song.to_markdown()
        self.assertIn(song.title, md)
        self.assertIn("## CHORUS", md)


# ─────────────────────────── arrangement ───────────────────────────────────

class ArrangementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _context()
        self.addCleanup(shutil.rmtree, self.ctx._root, True)
        self.creator = MusicCreator(self.ctx)

    def _arrange(self, style: str, seed: int = 11):
        from nomorals.media import music as mu
        song = self.creator.compose("test", style=style, seed=seed,
                                    with_midi=False)
        spec = resolve_style(style)
        return song, mu.MusicCreator._arrange(
            self.creator, song, spec, random.Random(seed))

    def test_outro_quieter_than_chorus(self) -> None:
        # outros fade: the last chords of the song average quieter than
        # the song's overall chord average (dynamics are baked in)
        _song, parts = self._arrange("pop")
        chords = parts["chords"]
        self.assertTrue(chords)
        avg_all = sum(e.velocity for e in chords) / len(chords)
        tail = chords[-8:]
        avg_tail = sum(e.velocity for e in tail) / len(tail)
        self.assertLess(avg_tail, avg_all)

    def test_key_change_on_modulating_styles(self) -> None:
        song, _parts = self._arrange("pop")
        notes = [s.note for s in song.sections]
        self.assertTrue(any("key change" in n for n in notes),
                        f"no key change marked: {notes}")

    def test_no_key_change_without_modulate(self) -> None:
        song, _parts = self._arrange("rock")
        self.assertFalse(any("key change" in s.note for s in song.sections))

    def test_jazz_chords_have_extensions(self) -> None:
        song, parts = self._arrange("jazz")
        # ninths → ≥4-note chords on the chords track
        self.assertGreater(max(len({e.note for e in parts["chords"]
                                    if abs(e.start - s) < 0.01})
                               for s in range(0, 8)), 3)

    def test_pop_chords_are_triads(self) -> None:
        # no extensions on pop: triads, occasionally with the root doubled
        # an octave up for fullness
        _song, parts = self._arrange("pop")
        first_bar = {e.note for e in parts["chords"] if e.start < 4.0}
        self.assertIn(len(first_bar), (3, 4))

    def test_chorus_melody_lifted_over_verse(self) -> None:
        song, parts = self._arrange("pop")
        bars = []
        bar = 0
        for s in song.sections:
            bars.append((s.name, bar * 4, (bar + s.bars) * 4))
            bar += s.bars
        def avg_pitch(name: str) -> float:
            ns = [e.note for e in parts["melody"]
                  for n, lo, hi in bars
                  if n == name and lo <= e.start < hi]
            return sum(ns) / len(ns)
        self.assertGreater(avg_pitch("chorus"), avg_pitch("verse"))

    def test_counter_melody_only_on_big_sections(self) -> None:
        song, parts = self._arrange("pop")
        bars = []
        bar = 0
        for s in song.sections:
            bars.append((s.name, bar * 4, (bar + s.bars) * 4))
            bar += s.bars
        verse_hits = [e for e in parts["counter"]
                      for n, lo, hi in bars
                      if n == "verse" and lo <= e.start < hi]
        chorus_hits = [e for e in parts["counter"]
                       for n, lo, hi in bars
                       if n == "chorus" and lo <= e.start < hi]
        self.assertEqual(verse_hits, [])
        self.assertTrue(chorus_hits)

    def test_metal_uses_gallop_and_double_kick(self) -> None:
        _song, parts = self._arrange("metal")
        kicks = [e for e in parts["drums"] if e.note == M.KICK]
        self.assertGreaterEqual(len(kicks), 16)
        self.assertTrue(parts["bass"])

    def test_ambient_has_no_drums(self) -> None:
        _song, parts = self._arrange("ambient")
        self.assertEqual(parts["drums"], [])

    def test_downbeat_accents(self) -> None:
        _song, parts = self._arrange("rock")
        downs = [e for e in parts["melody"] if e.start % 4 < 0.05]
        offs = [e for e in parts["melody"]
                if 0.4 < e.start % 4 < 3.6]
        self.assertTrue(downs and offs)
        avg_down = sum(e.velocity for e in downs) / len(downs)
        avg_off = sum(e.velocity for e in offs) / len(offs)
        self.assertGreater(avg_down, avg_off)


# ─────────────────────────── tool wiring ───────────────────────────────────

class ToolWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _context()
        self.addCleanup(shutil.rmtree, self.ctx._root, True)

    def _fn(self):
        import nomorals.media.music as mu
        fns: dict = {}

        def register(name: str, **kw):
            def deco(fn):
                fns[name] = fn
                return fn
            return deco

        mu.register(SimpleNamespace(context=self.ctx, register=register,
                                    fns=fns))
        return fns["music_writer"]

    def test_styles_listing_covers_new_genres(self) -> None:
        fn = self._fn()
        styles = fn(action="styles")["styles"]
        for key in ("funk", "disco", "reggaeton", "country", "metal",
                    "ambient", "drill", "choir"):
            self.assertIn(key, styles)
        self.assertEqual(len(styles), len(STYLES))

    def test_compose_action_end_to_end(self) -> None:
        fn = self._fn()
        song = fn(action="compose", topic="victory lap", style="disco",
                  seed=2026)
        self.assertEqual(song["style"], "disco")
        self.assertTrue(Path(song["midi_path"]).exists())
        _parse_smf(Path(song["midi_path"]).read_bytes())

    def test_compose_needs_topic(self) -> None:
        from nomorals.core.errors import ToolError
        fn = self._fn()
        with self.assertRaises(ToolError):
            fn(action="compose", topic="  ")

    def test_song_lookup_lists_saved(self) -> None:
        fn = self._fn()
        fn(action="compose", topic="saved tune", style="lofi", seed=1,
           with_midi=True)
        out = fn(action="song", topic="")
        self.assertGreaterEqual(out["count"], 1)
        found = fn(action="song", topic="saved tune")
        self.assertTrue(found["found"])

    def test_unknown_action_raises(self) -> None:
        from nomorals.core.errors import ToolError
        fn = self._fn()
        with self.assertRaises(ToolError):
            fn(action="remix")


if __name__ == "__main__":
    unittest.main()
