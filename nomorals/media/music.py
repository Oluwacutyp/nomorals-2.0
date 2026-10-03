"""MusicCreator — turn a topic into a real, performable song.

Two lyric engines, chosen by availability:
* **model** — when a real model is active, it writes the lyrics to your
  exact structure (verse/chorus/bridge) in the requested style.
* **template** — a genuine offline engine: a rhyme-group bank,
  topic-seeded vocabulary, syllable-aware line assembly, real rhyme
  schemes (ABAB/AABB/ABBA), per-style syllable meter, section-specific
  templates (hooks, pre-chorus lifts, bridges), and Naija pidgin flavor
  for afrobeats/highlife/amapiano/dancehall. No lorem-ipsum — every line
  rhymes, scans, and carries the topic.

On top of the words it produces a full musical plan: key, mode, tempo,
a chord progression per section, a melody description, and — via
:mod:`nomorals.core.midi` — an actual playable ``.mid`` file you can open
in any DAW or phone music app.  The MIDI is a real arrangement, not a
demo: motif-based lead melody with chord-tone downbeats and a tonic
cadence, plus call-and-response counter-melodies, style-voiced chords
(closed/drop-2/spread/stabs) with functional 7th/9th extensions on the
jazzy styles, a proper bass line (walking, syncopated, log-drum,
gallop, …), a 22-pattern drum kit on the GM percussion channel,
per-section dynamics (intros build, outros fade, buildups rise,
drops slam), tag refrains after final choruses, humanized timing,
and last-chorus key changes.

    from nomorals.media.music import MusicCreator
    creator = MusicCreator(context)
    song = creator.compose("late night drive through the city",
                           style="lofi", with_midi=True)
    song.midi_path          # …/workspace/music/<slug>.mid  (real SMF)
    print(song.to_markdown())

Registered as the ``music_writer`` tool.
"""

from __future__ import annotations

import difflib
import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.midi import (
    MidiBuilder,
    NoteEvent,
    add_chord_extensions,
    apply_voicing,
    generate_bass_line,
    generate_chord_bass,
    generate_counter_melody,
    generate_drums,
    generate_melody,
    humanize,
    transpose_root,
)
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["MusicCreator", "Song", "StyleSpec", "STYLES", "register"]


# ─────────────────────────────── styles ──────────────────────────────────────

@dataclass(frozen=True)
class StyleSpec:
    name: str
    label: str
    tempo: tuple[int, int]          # (min, max) bpm
    mode: str                        # midi scale name
    progressions: tuple[tuple[str, ...], ...]
    sections: tuple[tuple[str, int], ...]   # (section, bars)
    energy: str
    palette: tuple[str, ...]        # mood words used in descriptions
    instrumentation: tuple[str, ...]
    # arrangement — how the song is actually played, not just described
    drum_pattern: str = ""          # key in midi.DRUM_PATTERNS; "" = no kit
    bass_pattern: str = "roots"     # key in midi.BASS_STYLES
    voicing: str = "closed"         # closed | drop2 | spread | stabs
    extensions: str = ""            # "" | sevenths | ninths — 7th/9th color
    modulate: bool = False          # lift the final chorus a semitone
    programs: tuple[int, int, int] = (0, 4, 32)  # GM: melody, chords, bass


def _style(name, label, tempo, mode, progressions, sections, energy,
           palette, instrumentation, drum_pattern="", bass_pattern="roots",
           voicing="closed", modulate=False, extensions="",
           programs=(0, 4, 32)) -> StyleSpec:
    return StyleSpec(name, label, tempo, mode, tuple(progressions),
                     tuple(sections), energy, tuple(palette),
                     tuple(instrumentation), drum_pattern, bass_pattern,
                     voicing, extensions, modulate, programs)


STYLES: dict[str, StyleSpec] = {
    "lofi": _style("lofi", "Lo-fi / chillhop", (70, 90), "dorian",
                   (("ii", "V", "I", "vi"), ("iii", "IV", "ii", "V"),
                    ("vi", "IV", "I", "V")),
                   (("intro", 2), ("verse", 8), ("chorus", 8),
                    ("bridge", 4), ("outro", 2)),
                   "mellow, hazy, introspective",
                   ("dusty vinyl", "tape warmth", "dull keys", "rain on glass",
                    "amber light", "half-remembered"),
                   ("lofi keys", "dusty drums", "soft bass", "vinyl crackle",
                    "muted guitar"),
                   drum_pattern="lofi", bass_pattern="roots", voicing="drop2",
                   extensions="sevenths", programs=(4, 0, 32)),
    "hiphop": _style("hiphop", "Hip-hop / rap", (85, 100), "minor",
                     (("i", "iv", "v", "i"), ("i", "vii", "iv", "v"),
                      ("i", "bVI", "bVII", "i")),
                     (("intro", 2), ("verse", 16), ("chorus", 8),
                      ("verse", 16), ("chorus", 8), ("outro", 2)),
                     "hard, confident, rhythmic",
                     ("street lights", "concrete", "hustle", "crown",
                      "midnight", "blue flame"),
                     ("808s", "hard kick", "snappy snare", "sub bass",
                      "hi-hat rolls"),
                     drum_pattern="trap", bass_pattern="riff",
                     voicing="closed", programs=(0, 4, 39)),
    "afrobeats": _style("afrobeats", "Afrobeats", (100, 110), "major",
                        (("I", "V", "vi", "IV"), ("I", "IV", "vi", "V"),
                         ("IV", "I", "V", "vi")),
                        (("intro", 2), ("verse", 8), ("chorus", 8),
                         ("bridge", 4), ("chorus", 8), ("outro", 2)),
                        "groovy, warm, danceable",
                        ("sun", "garden", "rhythm", "home", "golden",
                         "laughter"),
                        ("log drums", "shakers", "bright guitar", "warm bass",
                         "congas"),
                        drum_pattern="afrobeats", bass_pattern="syncopated",
                        voicing="closed", programs=(26, 4, 33)),
    "amapiano": _style("amapiano", "Amapiano", (110, 115), "minor",
                       (("i", "bVI", "bVII", "i"), ("i", "iv", "bVI", "v"),
                        ("i", "bVII", "iv", "i")),
                       (("intro", 4), ("verse", 8), ("chorus", 8),
                        ("break", 4), ("chorus", 8), ("outro", 2)),
                       "bass-heavy, jazzy, hypnotic",
                       ("night drive", "deep space", "smoke", "slow motion",
                        "neon", "gravity"),
                       ("log drum bass", "piano stabs", "shakers", "soft keys",
                        "deep sub"),
                       drum_pattern="amapiano", bass_pattern="logdrum",
                       voicing="stabs", programs=(4, 89, 38)),
    "pop": _style("pop", "Pop", (100, 120), "major",
                  (("I", "V", "vi", "IV"), ("vi", "IV", "I", "V"),
                   ("I", "IV", "V", "I")),
                  (("intro", 2), ("verse", 8), ("pre-chorus", 4),
                   ("chorus", 8), ("bridge", 4), ("chorus", 8),
                   ("tag", 2), ("outro", 2)),
                  "bright, hooky, uplifting, singalong",
                  ("spark", "hearts", "sky", "lightning", "gold", "forever"),
                  ("synth stabs", "four-on-the-floor", "bright bass",
                   "claps", "supersaw pads"),
                  drum_pattern="halftime_pop",
                  bass_pattern="driving_eighths", voicing="closed",
                  modulate=True, programs=(0, 4, 33)),
    "rock": _style("rock", "Rock", (120, 140), "mixolydian",
                   (("I", "bVII", "IV", "I"), ("I", "IV", "bVII", "IV"),
                    ("I", "V", "IV", "I")),
                   (("intro", 2), ("verse", 8), ("chorus", 8),
                    ("solo", 8), ("chorus", 8), ("outro", 2)),
                   "driving, raw, anthemic, tense",
                   ("thunder", "engine", "fists", "wire", "static", "fire"),
                   ("overdriven guitars", "tight drums", "walking bass",
                    "power chords", "double bass drum"),
                   drum_pattern="rock", bass_pattern="driving_eighths",
                   voicing="closed", programs=(29, 30, 33)),
    "rnb": _style("rnb", "R&B / soul", (70, 95), "natural_minor",
                  (("i", "bVI", "bVII", "i"), ("i", "iv", "bVI", "v"),
                   ("i", "v", "iv", "i")),
                  (("intro", 2), ("verse", 8), ("pre-chorus", 4),
                   ("chorus", 8), ("bridge", 4), ("chorus", 8),
                   ("tag", 2), ("outro", 2)),
                  "sensual, smooth, emotional, late-night",
                  ("velvet", "moon", "silk", "embers", "slow breath",
                   "afterglow"),
                  ("smooth keys", "round drums", "gliding bass", "strings",
                   "whisper vox"),
                  drum_pattern="rnb_slow", bass_pattern="halftime",
                  voicing="drop2", extensions="sevenths", programs=(52, 48, 33)),
    "gospel": _style("gospel", "Gospel", (60, 90), "major",
                     (("I", "IV", "V", "I"), ("I", "vi", "IV", "V"),
                      ("I", "V", "IV", "I")),
                     (("intro", 2), ("verse", 8), ("chorus", 8),
                      ("bridge", 8), ("chorus", 8),
                      ("tag", 2), ("outro", 4)),
                     "lifting, joyful, testimony, spirited",
                     ("light", "morning", "grace", "hands raised", "dawn",
                      "praise"),
                     ("piano", "handclaps", "brass stabs", "bass",
                      "choir pads"),
                     drum_pattern="gospel", bass_pattern="roots",
                     voicing="spread", extensions="sevenths", modulate=True, programs=(52, 0, 33)),
    "edm": _style("edm", "EDM / big room", (126, 132), "minor",
                  (("i", "bVI", "bVII", "i"), ("i", "iv", "v", "i"),
                   ("i", "bVII", "bVI", "bVII")),
                  (("intro", 4), ("verse", 8), ("buildup", 4), ("drop", 8),
                   ("break", 4), ("drop", 8), ("outro", 2)),
                  "euphoric, massive, hands-up",
                  ("stadium", "lasers", "sweat", "anthem", "midnight",
                   "electric sky"),
                  ("supersaw lead", "pounding kick", "risers", "white noise",
                   "sub drop", "crowd"),
                  drum_pattern="four_floor", bass_pattern="driving_eighths",
                  voicing="stabs", modulate=True, programs=(81, 62, 38)),
    "house": _style("house", "House", (120, 124), "minor",
                    (("i", "bVII", "iv", "i"), ("i", "iv", "v", "i"),
                     ("i", "bVI", "bVII", "i")),
                    (("intro", 4), ("verse", 8), ("chorus", 8),
                     ("break", 4), ("chorus", 8), ("outro", 4)),
                    "hypnotic, rolling, late-night",
                    ("basement", "strobe", "sweat", "loop", "velvet rope",
                     "4am"),
                    ("piano stabs", "four-on-the-floor", "rolling bass",
                     "open hats", "diva vox chops"),
                    drum_pattern="four_floor", bass_pattern="riff",
                    voicing="stabs", programs=(81, 89, 38)),
    "jazz": _style("jazz", "Jazz", (100, 130), "dorian",
                   (("ii", "V", "I", "vi"), ("iii", "vi", "ii", "V"),
                    ("I", "vi", "ii", "V")),
                   (("intro", 2), ("verse", 8), ("chorus", 8),
                    ("solo", 8), ("chorus", 8), ("outro", 2)),
                   "swinging, sophisticated, alive",
                   ("smoke", "blue hour", "brass", "velvet booths",
                    "improvisation", "after hours"),
                   ("ride cymbal", "walking bass", "piano comping",
                    "brushed snare", "tenor sax"),
                   drum_pattern="swing", bass_pattern="walking",
                   voicing="drop2", extensions="ninths", programs=(66, 4, 32)),
    "classical": _style("classical", "Classical / orchestral", (70, 110),
                        "major",
                        (("I", "IV", "V", "I"), ("I", "vi", "ii", "V"),
                         ("IV", "I", "V", "I")),
                        (("intro", 4), ("verse", 8), ("development", 8),
                         ("chorus", 8), ("outro", 4)),
                        "grand, sweeping, emotional",
                        ("cathedral", "candlelight", "marble", "storm",
                         "silk", "eternity"),
                        ("strings", "timpani", "woodwinds", "brass",
                         "harp", "choir"),
                        drum_pattern="", bass_pattern="roots",
                        voicing="spread", programs=(40, 48, 43)),
    "reggae": _style("reggae", "Reggae", (76, 96), "major",
                     (("I", "V", "vi", "IV"), ("I", "IV", "V", "I"),
                      ("vi", "IV", "I", "V")),
                     (("intro", 2), ("verse", 8), ("chorus", 8),
                      ("verse", 8), ("chorus", 8), ("outro", 2)),
                     "laid-back, righteous, sun-baked",
                     ("island", "ganja smoke", "redemption", "zion",
                      "ocean", "lionheart"),
                     ("skank guitar", "one-drop drums", "bubble organ",
                      "deep bass", "dub echo"),
                     drum_pattern="one_drop", bass_pattern="reggae_bubble",
                     voicing="stabs", programs=(26, 4, 33)),
    "highlife": _style("highlife", "Highlife", (100, 112), "major",
                       (("I", "IV", "V", "I"), ("I", "vi", "IV", "V"),
                        ("IV", "V", "I", "I")),
                       (("intro", 4), ("verse", 8), ("chorus", 8),
                        ("solo", 4), ("chorus", 8), ("outro", 4)),
                       "joyful, celebratory, golden",
                       ("owambe", "party", "palm wine", "dance floor",
                        "sunshine", "family"),
                       ("juju guitar", "talking drum", "horns", "congas",
                        "groovy bass", "shekere"),
                       drum_pattern="highlife", bass_pattern="syncopated",
                       voicing="closed", programs=(26, 48, 33)),
    "blues": _style("blues", "Blues", (70, 100), "minor",
                    (("i", "iv", "i", "v"), ("i", "i", "iv", "i"),
                     ("v", "iv", "i", "i")),
                    (("intro", 2), ("verse", 12), ("chorus", 8),
                     ("solo", 8), ("verse", 12), ("outro", 2)),
                    "raw, aching, honest",
                    ("whiskey", "crossroads", "freight train", "lonesome",
                     "juke joint", "midnight"),
                    ("wailing guitar", "shuffle drums", "walking bass",
                     "harmonica", " Hammond organ"),
                    drum_pattern="shuffle", bass_pattern="walking",
                    voicing="closed", programs=(29, 4, 33)),
    "dancehall": _style("dancehall", "Dancehall", (95, 105), "minor",
                        (("i", "bVII", "iv", "i"), ("i", "iv", "v", "i"),
                         ("i", "bVI", "bVII", "i")),
                        (("intro", 2), ("verse", 8), ("chorus", 8),
                         ("break", 4), ("chorus", 8), ("outro", 2)),
                        "bad, bouncy, yard-hot",
                        ("dancefloor", "bassline", "rum", "whine",
                         "street party", "spotlight"),
                        ("riddim guitar", "808 slides", "dembow drums",
                         "airhorn", "sub bass"),
                        drum_pattern="dancehall", bass_pattern="syncopated",
                        voicing="stabs", programs=(26, 4, 39)),
    "funk": _style("funk", "Funk", (95, 110), "minor",
                   (("i", "bVII", "iv", "i"), ("i", "iv", "bVII", "v"),
                    ("i", "bVI", "bVII", "i")),
                   (("intro", 2), ("verse", 8), ("chorus", 8),
                    ("break", 2), ("chorus", 8), ("outro", 2)),
                   "tight, greasy, in-the-pocket",
                   ("sweat", "velvet rope", "brass section", "midnight oil",
                    "groove", "hot wax"),
                   ("scratch guitar", "slap bass", "horn stabs", "clavinet",
                    "tight kit"),
                   drum_pattern="funk", bass_pattern="syncopated",
                   voicing="stabs", programs=(27, 7, 34)),
    "disco": _style("disco", "Disco", (115, 120), "minor",
                    (("i", "bVI", "bVII", "i"), ("i", "iv", "v", "i"),
                     ("i", "bVII", "iv", "v")),
                    (("intro", 4), ("verse", 8), ("chorus", 8),
                     ("break", 4), ("chorus", 8), ("outro", 2)),
                    "glittering, euphoric, mirrorball",
                    ("mirrorball", "velvet", "spotlight", "midnight",
                     "sequins", "city lights"),
                    ("string section", "octave bass", "four-on-the-floor",
                     "wah guitar", "orchestral hits"),
                    drum_pattern="disco", bass_pattern="driving_eighths",
                    voicing="stabs", extensions="sevenths",
                    modulate=True, programs=(81, 48, 33)),
    "reggaeton": _style("reggaeton", "Reggaeton", (92, 100), "minor",
                        (("i", "bVII", "iv", "i"), ("i", "iv", "v", "i"),
                         ("i", "bVI", "bVII", "v")),
                        (("intro", 2), ("verse", 8), ("chorus", 8),
                         ("break", 2), ("chorus", 8), ("outro", 2)),
                        "perreable, hypnotic, dembow-driven",
                        ("neón", "calle", "baila", "verano", "rumba",
                         "medianoche"),
                        ("dembow drums", "synth brass", "sub bass",
                         "shakers", "reggaeton vox"),
                        drum_pattern="reggaeton", bass_pattern="syncopated",
                        voicing="stabs", programs=(62, 62, 39)),
    "country": _style("country", "Country", (90, 115), "major",
                      (("I", "V", "vi", "IV"), ("I", "IV", "V", "I"),
                       ("vi", "IV", "I", "V")),
                      (("intro", 2), ("verse", 8), ("chorus", 8),
                       ("solo", 4), ("chorus", 8),
                       ("tag", 2), ("outro", 2)),
                      "heartfelt, wide-open, honest",
                      ("dirt road", "porch light", "whiskey", "tailgate",
                       "red sun", "hometown"),
                      ("telecaster twang", "pedal steel", "train-beat drums",
                       "acoustic guitar", "fiddle"),
                      drum_pattern="country", bass_pattern="roots",
                      voicing="closed", programs=(26, 24, 33)),
    "metal": _style("metal", "Metal", (140, 160), "harmonic_minor",
                    (("i", "bVI", "bVII", "i"), ("i", "iv", "v", "i"),
                     ("i", "bVII", "bVI", "bVII")),
                    (("intro", 2), ("verse", 8), ("chorus", 8),
                     ("solo", 8), ("breakdown", 4), ("chorus", 8),
                     ("outro", 2)),
                    "crushing, relentless, epic",
                    ("iron", "storm", "thunder", "obsidian", "war drums",
                     "fire"),
                    ("distorted guitars", "double-bass drums", "gallop bass",
                     "shredding lead", "choir of doom"),
                    drum_pattern="metal", bass_pattern="gallop",
                    voicing="closed", programs=(30, 30, 34)),
    "ambient": _style("ambient", "Ambient / drone", (60, 80), "major",
                      (("I", "vi", "IV", "V"), ("I", "IV", "vi", "V"),
                       ("vi", "IV", "I", "V")),
                      (("intro", 4), ("verse", 8), ("break", 4),
                       ("chorus", 8), ("outro", 4)),
                      "weightless, drifting, cathedral-quiet",
                      ("fog", "cathedral", "tide", "slow light", "vastness",
                       "stillness"),
                      ("shimmer pads", "felt piano", "soft bass swells",
                       "field recordings", "glass harmonica"),
                      drum_pattern="", bass_pattern="breakdown_pad",
                      voicing="spread", programs=(89, 89, 48)),
    "drill": _style("drill", "Drill", (138, 142), "minor",
                    (("i", "bVI", "bVII", "i"), ("i", "iv", "v", "i"),
                     ("i", "bVII", "iv", "i")),
                    (("intro", 2), ("verse", 16), ("chorus", 8),
                     ("verse", 8), ("chorus", 8), ("outro", 2)),
                    "dark, sliding, menacing",
                    ("block", "night bus", "fog", "sirens", "concrete",
                     "shadows"),
                    ("sliding 808s", "swung hats", "dark piano",
                     "choir stabs", "deep sub"),
                    drum_pattern="drill", bass_pattern="riff",
                    voicing="stabs", programs=(0, 4, 39)),
    "choir": _style("choir", "Choir / hymnal", (70, 90), "major",
                    (("I", "IV", "V", "I"), ("I", "vi", "ii", "V"),
                     ("IV", "I", "V", "I")),
                    (("intro", 2), ("verse", 8), ("chorus", 8),
                     ("bridge", 4), ("chorus", 8),
                     ("tag", 2), ("outro", 4)),
                    "soaring, reverent, united",
                    ("cathedral", "candlelight", "voices", "dawn", "grace",
                     "eternity"),
                    ("SATB choir", "organ", "strings", "handbells",
                     "acoustic bass"),
                    drum_pattern="", bass_pattern="roots",
                    voicing="spread", extensions="sevenths",
                    programs=(52, 19, 43)),
}
#: aliases so natural phrasing maps onto a style
STYLE_ALIASES = {
    "lofi": "lofi", "lo-fi": "lofi", "chill": "lofi", "chillhop": "lofi",
    "chillbeat": "lofi",
    "hiphop": "hiphop", "hip-hop": "hiphop", "rap": "hiphop", "trap": "hiphop",
    "boom bap": "hiphop", "boom-bap": "hiphop",
    "afro": "afrobeats", "afrobeats": "afrobeats", "afrobeat": "afrobeats",
    "amapiano": "amapiano", "mapiano": "amapiano",
    "pop": "pop",
    "rock": "rock", "punk": "rock", "guitar": "rock",
    "rnb": "rnb", "r&b": "rnb", "soul": "rnb",
    "gospel": "gospel",
    "edm": "edm", "electronic": "edm", "dance": "edm", "big room": "edm",
    "bigroom": "edm", "festival": "edm", "dubstep": "edm",
    "house": "house", "deep house": "house", "deep-house": "house",
    "techno": "house",
    "jazz": "jazz", "bebop": "jazz", "swing": "jazz", "bop": "jazz",
    "classical": "classical", "orchestral": "classical", "symphony": "classical",
    "reggae": "reggae", "dub": "reggae", "ska": "reggae", "roots": "reggae",
    "highlife": "highlife", "juju": "highlife", "owambe": "highlife",
    "blues": "blues", "shuffle": "blues",
    "dancehall": "dancehall",
    "alte": "afrobeats", "alté": "afrobeats", "afro-fusion": "afrobeats",
    "afroswing": "afrobeats",
    "funk": "funk", "p-funk": "funk", "george clinton": "funk",
    "disco": "disco", "boogie": "disco", "nu-disco": "disco",
    "reggaeton": "reggaeton", "dembow": "reggaeton", "perreo": "reggaeton",
    "latin": "reggaeton",
    "country": "country", "nashville": "country", "americana": "country",
    "metal": "metal", "heavy metal": "metal", "thrash": "metal",
    "ambient": "ambient", "drone": "ambient", "chillout": "ambient",
    "sleep": "ambient",
    "drill": "drill", "uk drill": "drill",
    "choir": "choir", "hymn": "choir", "choral": "choir",
    "acappella": "choir", "a cappella": "choir",
}


def resolve_style(name: str) -> StyleSpec:
    key = (name or "").strip().lower()
    if not key:
        return STYLES["pop"]
    if key in STYLES:
        return STYLES[key]
    if key in STYLE_ALIASES:
        return STYLES[STYLE_ALIASES[key]]
    for k, v in STYLE_ALIASES.items():
        if k in key or key in k:
            return STYLES[v]
    # Fuzzy match: "dream" -> "ambient", "hip hop" -> "hiphop", etc.
    # Also try matching against style names with spaces removed.
    candidates = list(STYLES.keys()) + list(STYLE_ALIASES.keys())
    matches = difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
    if matches:
        matched = matches[0]
        if matched in STYLES:
            return STYLES[matched]
        return STYLES[STYLE_ALIASES[matched]]
    # Vibe-word fallback: map common mood words to the closest style.
    _VIBE_MAP = {
        "dream": "ambient", "dreamy": "ambient", "sleep": "ambient",
        "chill": "lofi", "relax": "lofi", "calm": "ambient",
        "party": "dancehall", "club": "edm", "dance": "edm",
        "sad": "blues", "happy": "pop", "love": "rnb",
        "worship": "gospel", "church": "gospel",
        "street": "drill", "hard": "drill",
    }
    for vibe, style_key in _VIBE_MAP.items():
        if vibe in key or key in vibe:
            return STYLES[style_key]
    raise ToolError(f"unknown style {name!r} — choose from "
                    f"{sorted(STYLES)}")


# ─────────────────────────── offline lyric engine ────────────────────────────

#: rhyme groups: every word must work as a sung line-ending — punchy,
#: singable nouns/adjectives, no Latinate abstractions, no verb forms.
_RHYME_GROUPS: tuple[tuple[str, ...], ...] = (
    ("night", "light", "flight", "tonight", "moonlight", "spotlight",
     "satellite"),
    ("fire", "desire", "wire"),
    ("heart", "start", "art"),
    ("dream", "stream", "gleam", "beam"),
    ("time", "shine", "rhyme"),
    ("run", "sun", "song", "dawn"),
    ("home", "stone"),
    ("love", "dove", "groove"),
    ("fall", "call", "hall"),
    ("rise", "eyes", "skies", "surprise"),
    ("away", "day", "stay", "sway"),
    ("deep", "sleep"),
    ("sound", "ground", "found", "round"),
    ("blue", "view"),
    ("sea", "key", "memory"),
    ("glow", "flow", "show"),
    ("wild", "child"),
    ("ocean", "motion", "emotion", "devotion"),
    ("rhythm", "prism", "schism"),
    ("midnight", "first light", "dynamite"),
    ("gravity", "sanity", "clarity", "vanity"),
    ("fever", "believer", "dreamer", "deceiver"),
    ("soldier", "shoulder"),
    ("afterglow", "overflow", "vertigo", "indigo"),
    ("avenue", "rendezvous"),
    ("dance", "trance", "romance", "chance"),
    ("whine", "shine", "design"),
    ("money", "honey"),
    ("danger", "stranger", "anger"),
    ("victory", "glory", "story"),
    ("power", "tower", "hour", "flower"),
    ("wonder", "thunder"),
    ("baby", "lady"),
    ("moon", "tune"),
    ("star", "guitar", "boulevard"),
    ("king", "ring", "bling", "everything"),
    ("road", "load"),
    ("rain", "pain", "champagne", "refrain"),
    ("hero", "zero"),
)

#: templates that can carry a forced rhyme: they end in an {imagery} or
#: {em} slot, so the tail word can be swapped in grammatically (the
#: article stays: "across the paper moons" -> "across the light").
#: A few are excluded even so: their tail slot is governed by "of",
#: "never", or a verb, where a swapped-in noun would read wrong
#: ("nights of tonight", "never dreamer", "we will dream home").
_RHYME_BLOCKLIST = (
    "{ing} through {em} nights of {imagery}",
    "We {verb}, we {verb}, never {em}",
    "Say the word and we will {verb} {em}",
    "We were born to {verb} {em}",
)


def _rhymable(pool: tuple[str, ...]) -> tuple[str, ...]:
    out = tuple(t for t in pool
                if t.rstrip().endswith(("{imagery}", "{em}"))
                and t not in _RHYME_BLOCKLIST)
    return out or pool

_VERB_BANK = ("chase", "hold", "carve", "burn", "wake", "fall", "rise",
              "carry", "breathe", "turn", "run", "wait", "build", "break",
              "learn", "dream", "search", "remember", "follow", "find",
              "gather", "spill", "mend", "sway", "drift", "climb", "fold",
              "ignite", "whisper", "roar", "stumble", "soar", "weave",
              "unravel", "glisten", "tremble", "bloom", "fade", "linger",
              "charge", "surrender", "reclaim", "wander", "shine", "ache")


def _ing(verb: str) -> str:
    v = verb
    if v.endswith(("e",)) and not v.endswith(("ee",)):
        v = v[:-1]
    return v + "ing"
_IMAGERY = ("neon rain", "paper stars", "midnight smoke", "gold horizon",
            "silver wires", "distant thunder", "soft static", "open sky",
            "broken mirrors", "fading film", "slow lightning", "warm static",
            "city embers", "quiet flames", "heavy air", "bright concrete",
            "velvet dusk", "chrome rivers", "hollow cathedrals", "wild mercury",
            "amber traffic", "frozen fireworks", "tangled satellites",
            "moonlit gravel", "electric moss", "rusted halos", "paper moons",
            "glass deserts", "midnight orchards", "copper skies",
            "sleeping sirens", "velvet static", "iron blossoms",
            "quicksilver tides", "burning atlases", "silent carnivals",
            "phosphor dreams", "withered neon", "salt cathedrals",
            "obsidian waves", "lantern smoke", "fractured auroras",
            "midnight terminals", "gilded wreckage", "pale machinery",
            "velvet thunder", "hollow suns", "drifting embers")
_EMOTION = ("quiet", "restless", "golden", "distant", "endless", "hollow",
            "fierce", "tender", "electric", "weightless",
            "haunted", "feverish", "patient", "wild",
            "sleepless", "gentle", "reckless", "luminous", "aching",
            "defiant", "honeyed", "fractured", "brave",
            "lonely", "radiant", "unbroken", "velvet", "smoldering")

_LINE_TEMPLATES = (
    "{ing} the {topic_short} under the {em} {imagery}",
    "{imagery} in motion, we are still {ing}",
    "I found a {noun} in the {em} {imagery}",
    "The {topic_short} never sleeps, {ing} like {imagery}",
    "We {verb} through the {em} {imagery} alone",
    "Every {noun} {ing} a little {em}",
    "The {imagery} hold what the {topic_short} is worth",
    "Say the word and we will {verb} {em}",
    "{ing} past the {imagery}, never looking {em}",
    "A {noun} for the {em}, a spark for the {imagery}",
    "We {verb} like the {imagery} owes us {em}",
    "Hold a {noun} up to the {em} {imagery}",
    "{ing} where the {imagery} forgets to be {em}",
    "The {em} {imagery} taught the {noun} to {verb}",
    "Nobody {ing} here without a little {imagery}",
    "Trade your {noun} for a pocket of {em} {imagery}",
    "We {verb} the {topic_short} till the {imagery} bends",
    "{ing} on {em} feet across the {imagery}",
    "A {noun} in the {imagery} keeps {ing} {em}",
    "The {topic_short} {ing}, and the {imagery} answer {em}",
    "We {verb} our names into the {em} {imagery}",
    "{imagery} on repeat while the {noun} keeps {ing}",
    "{ing} through {em} nights of {imagery}",
    "Give the {imagery} a {noun} and watch it {verb}",
)
#: hook lines for choruses — short, repeatable, singable
_HOOK_TEMPLATES = (
    "We {verb} the {em} {imagery}",
    "This is the {topic_short}, baby, {verb}",
    "Hold the {em} {imagery} tight",
    "We were made to {verb} in the {imagery}",
    "Sing it {em}, sing it like {imagery}",
    "Oh, we {verb} tonight",
    "Forever {ing} through the {imagery}",
    "{em} hearts in the {imagery}",
    "We {verb}, we {verb}, never {em}",
    "Light up the {imagery}, {verb} with me",
    "This {topic_short} is ours to {verb}",
    "Higher than {imagery}, we {verb}",
    "Stay {em} inside the {imagery}",
    "We were born to {verb} {em}",
)
#: pre-chorus lines: tension, rising, leading into the hook
_PRECHORUS_TEMPLATES = (
    "Can you feel it {ing} under the {imagery}",
    "The {em} night is {ing}, don't you {verb}",
    "Hold your breath now, the {topic_short} is {ing}",
    "Every {noun} {ing} louder than the {imagery}",
    "We are {ing} closer to the {em} {imagery}",
    "Don't you {verb} now, don't you dare go {em}",
)
#: bridge lines: reflective, a turn in the story
_BRIDGE_TEMPLATES = (
    "Maybe the {topic_short} was {ing} all along",
    "I gave my {noun} to the {em} {imagery}",
    "And if we {verb} tomorrow, we still {verb} today",
    "The {imagery} remembers every {em} {noun}",
    "Strip it down to {imagery} and a {em} {noun}",
    "This is where the {topic_short} learns to {verb}",
)

#: rhymable subsets (end in {imagery} or {em}): used when a line must
#: carry a forced rhyme from a rhyme group
_LINE_TEMPLATES_R = _rhymable(_LINE_TEMPLATES)
_HOOK_TEMPLATES_R = _rhymable(_HOOK_TEMPLATES)
_PRECHORUS_TEMPLATES_R = _rhymable(_PRECHORUS_TEMPLATES)
_BRIDGE_TEMPLATES_R = _rhymable(_BRIDGE_TEMPLATES)
#: Naija flavor for afrobeats / highlife / amapiano / dancehall —
#: pidgin-inflected lines, used sparingly so they land like ad-libs
_PIDGIN_TEMPLATES = (
    "Na {em} {imagery}, we dey {verb} o",
    "Omo, the {topic_short} no dey {verb} small",
    "We go {verb} till the {imagery} bend",
    "No long talk, just {ing} for the {imagery}",
    "{em} {imagery}, we {verb} am well-well",
    "Shey you see the {imagery}? We dey {ing}",
)
_PIDGIN_HOOKS = (
    "We dey {verb} o, we dey {verb}",
    "Na so e be, {em} {imagery}",
    "{topic_short} to the world, we {verb}",
    "Omo jaiye, {verb} with the {imagery}",
)

#: target syllables per line by style: (min, max). Keeps verses singable
#: and rap verses dense without turning into tongue-twisters.
_METER: dict[str, tuple[int, int]] = {
    "hiphop": (9, 14),
    "lofi": (6, 10), "rnb": (6, 10),
    "pop": (7, 11), "rock": (7, 11),
    "edm": (5, 9), "house": (5, 9), "dancehall": (7, 11),
    "afrobeats": (7, 11), "amapiano": (6, 10), "highlife": (7, 11),
    "reggae": (7, 11), "jazz": (6, 10), "blues": (6, 10),
    "gospel": (7, 11), "classical": (6, 10),
    "funk": (6, 10), "disco": (5, 9), "reggaeton": (7, 11),
    "country": (7, 11), "metal": (7, 11), "ambient": (5, 9),
    "drill": (9, 14), "choir": (6, 10),
}

_NAIJA_STYLES = {"afrobeats", "highlife", "amapiano", "dancehall"}

#: verse rhyme schemes: which lines share a rhyme group
_VERSE_SCHEMES = ("ABAB", "AABB", "ABBA")


def _syllables(text: str) -> int:
    """Rough syllable count: vowel groups per word, silent-e adjusted."""
    total = 0
    for word in re.findall(r"[a-zA-Z']+", text.lower()):
        groups = re.findall(r"[aeiouy]+", word)
        n = len(groups)
        if word.endswith("e") and n > 1 and not word.endswith(("ee", "ye")):
            n -= 1
        total += max(1, n)
    return total


#: section dynamics: (base velocity scale, ramp across the section).
#: Intros build, pre-choruses and buildups rise, outros fade, drops slam.
_SECTION_DYN: dict[str, tuple[float, float]] = {
    "intro": (0.55, 0.35),
    "verse": (0.82, 0.0),
    "pre-chorus": (0.85, 0.12),
    "chorus": (1.00, 0.0),
    "bridge": (0.70, 0.0),
    "break": (0.55, 0.0),
    "breakdown": (0.55, 0.0),
    "buildup": (0.85, 0.20),
    "drop": (1.05, 0.0),
    "solo": (0.95, 0.0),
    "development": (0.80, 0.10),
    "tag": (1.05, 0.0),
    "outro": (0.85, -0.40),
}

#: sections that get a call-and-response counter-melody
_COUNTER_SECTIONS = {"chorus", "bridge", "drop", "solo", "development",
                     "pre-chorus", "tag"}

#: per-section lead-melody register shift, semitones (choruses soar)
_MELODY_LIFT = {"chorus": 5, "pre-chorus": 3, "drop": 7, "solo": 7,
                "bridge": -2, "verse": 0, "intro": 0, "outro": 0,
                "break": 0, "breakdown": 0, "buildup": 3,
                "development": 2, "tag": 5}

#: sections with no lyrics — pure arrangement
_INSTRUMENTAL_SECTIONS = {"intro", "outro", "break", "solo", "buildup",
                          "drop", "breakdown", "development"}


def _dyn_scale(name: str, bar_i: int, bars: int) -> float:
    base, ramp = _SECTION_DYN.get(name, (0.85, 0.0))
    if bars <= 1 or ramp == 0.0:
        return base
    return base + ramp * (bar_i / (bars - 1))


def _slugify(text: str, limit: int = 48) -> str:
    from ..core.text import slugify

    # canonical: nomorals.core.text.slugify
    return slugify(text, limit=limit, fallback="untitled")


@dataclass
class Section:
    name: str
    bars: int
    lyrics: list[str] = field(default_factory=list)
    chords: tuple[str, ...] = ()
    note: str = ""


@dataclass
class Song:
    title: str
    style: str
    topic: str
    key: str
    mode: str
    tempo: int
    sections: list[Section] = field(default_factory=list)
    melody_description: str = ""
    instrumentation: tuple[str, ...] = ()
    mood: str = ""
    midi_path: str = ""
    audio_path: str = ""
    score_pdf_path: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "style": self.style, "topic": self.topic,
            "key": self.key, "mode": self.mode, "tempo": self.tempo,
            "mood": self.mood, "seed": self.seed,
            "melody_description": self.melody_description,
            "instrumentation": list(self.instrumentation),
            "sections": [
                {"name": s.name, "bars": s.bars, "lyrics": s.lyrics,
                 "chords": list(s.chords), "note": s.note}
                for s in self.sections],
            "midi_path": self.midi_path,
            "audio_path": self.audio_path,
            "score_pdf_path": self.score_pdf_path,
        }

    def to_score_markdown(self) -> str:
        """A proper lead sheet: title block + sections with chord symbols,
        lyrics, and arrangement notes.  This is what the score PDF renders."""
        lines = [
            f"# {self.title}",
            f"*{self.style.title()} · Key of {self.key} {self.mode} · "
            f"{self.tempo} BPM · Mood: {self.mood}*",
            "",
        ]
        if self.melody_description:
            lines += [f"**Melody:** {self.melody_description}", ""]
        for s in self.sections:
            tag = s.name.upper()
            lines.append(f"## {tag} — {s.bars} bars")
            if s.chords:
                # chord symbols as a lead-sheet row
                lines.append("**Chords:** " + " | ".join(s.chords))
            if s.note:
                lines.append(f"*{s.note}*")
            if s.lyrics:
                lines.append("")
                lines += [f"> {ln}" for ln in s.lyrics]
            lines.append("")
        if self.instrumentation:
            lines += ["## Instrumentation",
                      ", ".join(self.instrumentation), ""]
        return "\n".join(lines).rstrip() + "\n"

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "",
                 f"**Style:** {self.style}  ·  **Key:** {self.key} "
                 f"{self.mode}  ·  **Tempo:** {self.tempo} BPM  ·  "
                 f"**Mood:** {self.mood}", ""]
        if self.melody_description:
            lines += [f"**Melody:** {self.melody_description}", ""]
        for s in self.sections:
            tag = s.name.upper()
            head = f"## {tag}"
            if s.chords:
                head += f"  ({' — '.join(s.chords)})"
            lines.append(head)
            if s.lyrics:
                lines += [f"> {ln}" for ln in s.lyrics]
            lines.append("")
        if self.instrumentation:
            lines += ["**Instrumentation:**",
                      ", ".join(self.instrumentation), ""]
        if self.midi_path:
            lines.append(f"**MIDI file:** `{self.midi_path}`")
        if self.audio_path:
            lines.append(f"**Audio:** `{self.audio_path}`")
        if self.score_pdf_path:
            lines.append(f"**Score PDF:** `{self.score_pdf_path}`")
        return "\n".join(lines).rstrip() + "\n"


def _topic_words(topic: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", topic or "")
    stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "of", "for", "with", "through", "by", "from", "is", "are", "was",
            "were", "my", "your", "our", "their", "this", "that", "it",
            "about", "into", "over", "under", "around", "between", "during"}
    keep = [w for w in words if w.lower() not in stop and len(w) > 2]
    return keep or ["everything"]


def _model_available(context: Any) -> bool:
    router = getattr(context, "router", None)
    snap_fn = getattr(router, "stats_snapshot", None)
    if snap_fn is None:
        return False
    try:
        snap = snap_fn()
    except Exception:  # noqa: BLE001
        return False
    active = str(snap.get("active") or "")
    return bool(active) and active not in {"mock", "offline", "test"}


class MusicCreator:
    """Compose a full song (lyrics + music plan + MIDI) from a topic."""

    role = "music"

    def __init__(self, context: Any) -> None:
        self.context = context

    def compose(self, topic: str, *, style: str = "pop", title: str = "",
                key: str = "", seed: int | None = None,
                with_midi: bool = True, with_audio: bool = True,
                with_score: bool = True, workdir: str = "music") -> Song:
        spec = resolve_style(style)
        if seed is None:
            seed = int(hashlib.sha256(
                f"{topic}|{spec.name}|{time.time():.0f}".encode()
            ).hexdigest()[:8], 16)
        rng = random.Random(seed)

        title = (title or "").strip() or self._make_title(topic, spec, rng)
        key = key or rng.choice(["C", "A", "D", "G", "E", "F", "B", "C#"])
        tempo = rng.randint(spec.tempo[0], spec.tempo[1])
        progression = rng.choice(spec.progressions)

        topic_words = _topic_words(topic)
        primary = topic_words[0].capitalize()

        sections: list[Section] = []
        for name, bars in spec.sections:
            chords = self._section_chords(name, progression, rng)
            if name in _INSTRUMENTAL_SECTIONS:
                note = self._instrument_note(name, spec, rng)
                sections.append(Section(name, bars, chords=chords, note=note))
            else:
                lyrics = self._lyrics_for(name, topic, primary, topic_words,
                                          spec, rng)
                sections.append(Section(name, bars, lyrics=lyrics,
                                        chords=chords))

        melody_desc = self._melody_description(spec, rng, key, tempo)
        song = Song(title=title, style=spec.name, topic=topic, key=key,
                    mode=spec.mode, tempo=tempo, sections=sections,
                    melody_description=melody_desc,
                    instrumentation=spec.instrumentation,
                    mood=", ".join(rng.sample(spec.palette, 3)), seed=seed)

        if with_midi:
            try:
                song.midi_path = self._write_midi(song, workdir)
            except Exception as exc:  # noqa: BLE001
                _log.warning("midi generation failed: %s", exc)
        if with_audio:
            try:
                song.audio_path = self._render_audio(song, workdir)
            except Exception as exc:  # noqa: BLE001
                _log.warning("audio render failed: %s", exc)
        if with_score:
            try:
                song.score_pdf_path = self._write_score_pdf(song, workdir)
            except Exception as exc:  # noqa: BLE001
                _log.warning("score pdf failed: %s", exc)
        return song

    # ── lyrics ──────────────────────────────────────────────────────────
    def _make_title(self, topic: str, spec: StyleSpec, rng: random.Random) -> str:
        words = _topic_words(topic)
        if not words:
            return f"{spec.label} (untitled)"
        core = " ".join(w.capitalize() for w in words[:3])
        ornaments = ("After the", "Beneath", "Inside the", "Beyond",
                     "Under the", "Through the", "Chasing", "Holding")
        if rng.random() < 0.5:
            return f"{rng.choice(ornaments)} {core}"
        return core

    def _section_chords(self, name: str, progression: tuple[str, ...],
                        rng: random.Random) -> tuple[str, ...]:
        n = len(progression)
        if n == 0:
            return ()
        start = rng.randrange(n)
        if name in ("bridge", "break", "breakdown", "buildup", "development"):
            # a lift: rotate the progression for contrast
            start = (start + 2) % n
        return tuple(progression[(start + i) % n] for i in range(n))

    def _instrument_note(self, name: str, spec: StyleSpec,
                         rng: random.Random) -> str:
        inst = rng.sample(list(spec.instrumentation),
                          k=min(2, len(spec.instrumentation)))
        if name == "intro":
            return f"soft {inst[0]} entrance, building"
        if name == "outro":
            return f"fading {inst[0]}, {spec.palette[0]} tail"
        if name == "break":
            return f"space out — {inst[0]} drops, tension"
        if name == "solo":
            return f"instrumental feature on {inst[0]}"
        if name == "buildup":
            return f"riser — {inst[0]} climbs, snare roll into the drop"
        if name == "drop":
            return f"full drop — {inst[0]} hits hardest"
        if name == "breakdown":
            return f"stripped back — {inst[0]} and space"
        if name == "development":
            return f"{inst[0]} develops the theme"
        return f"driven by {inst[0]}"

    def _lyrics_for(self, name: str, topic: str, primary: str,
                    topic_words: list[str], spec: StyleSpec,
                    rng: random.Random) -> list[str]:
        if _model_available(self.context):
            try:
                return self._model_lyrics(name, topic, spec)
            except Exception as exc:  # noqa: BLE001
                _log.debug("model lyrics failed, using template: %s", exc)
        n_lines = 8 if name == "verse" else (2 if name == "tag" else 4)
        topic_short = " ".join(topic_words[:3]).lower()
        lines: list[str] = []
        used_imagery: set[str] = set()
        # rhyme scheme: every line index maps to a rhyme group up front, so
        # verses follow a real scheme (ABAB / AABB / ABBA) instead of just
        # pairwise rhymes, and choruses hammer one hook group.
        groups: list[tuple[str, ...] | None] = [None] * n_lines
        scheme_letters: dict[str, tuple[str, ...]] = {}
        if name == "verse":
            scheme = rng.choice(_VERSE_SCHEMES)
            for i in range(n_lines):
                letter = scheme[i % len(scheme)]
                if letter not in scheme_letters:
                    scheme_letters[letter] = rng.choice(_RHYME_GROUPS)
                groups[i] = scheme_letters[letter]
        elif name in ("chorus", "tag"):
            hook_group = rng.choice(_RHYME_GROUPS)
            groups = [hook_group] * n_lines
        elif name in ("bridge", "pre-chorus"):
            for i in range(0, n_lines, 2):
                grp = rng.choice(_RHYME_GROUPS)
                groups[i] = grp
                if i + 1 < n_lines:
                    groups[i + 1] = grp
        for i in range(n_lines):
            if name == "chorus" and i == 2 and lines:
                # real choruses repeat the hook verbatim
                lines.append(lines[0])
                continue
            line, imagery, _em = self._build_line(
                name, topic, primary, topic_words, topic_short, spec,
                rng, end_group=groups[i], used_imagery=used_imagery)
            used_imagery.add(imagery)
            lines.append(line)
        return lines

    def _rhyme_for(self, anchor: str, rng: random.Random,
                   used: set[str] | None) -> tuple[str, ...] | None:
        anchor_last = anchor.strip().rstrip(".,!?").split()[-1].lower() \
            if anchor.strip() else ""
        for group in _RHYME_GROUPS:
            if anchor_last in (g.lower() for g in group):
                if used is not None:
                    used.discard(group[0])
                return group
        return None

    def _end_word(self, group: tuple[str, ...] | None,
                  rng: random.Random) -> str:
        if group:
            return rng.choice(group)
        return rng.choice(rng.choice(_RHYME_GROUPS))

    def _rhyme_tail(self, line: str, imagery: str, em: str,
                    group: tuple[str, ...], rng: random.Random) -> str:
        """Swap a trailing {imagery}/{em} slot for a rhyme-group word.

        The rest of the line is untouched, so articles and prepositions
        stay grammatical.  Returns the line unchanged when it doesn't
        end in one of those slots.
        """
        tail = self._end_word(group, rng)
        if imagery and line.endswith(imagery):
            return (line[: -len(imagery)].rstrip() + " " + tail).strip()
        if em and line.endswith(" " + em):
            return (line[: -len(em)].rstrip() + " " + tail).strip()
        return line

    def _build_line(self, section: str, topic: str, primary: str,
                    topic_words: list[str], topic_short: str,
                    spec: StyleSpec, rng: random.Random,
                    end_group: tuple[str, ...] | None = None,
                    used_imagery: set[str] | None = None
                    ) -> tuple[str, str, str]:
        used_imagery = used_imagery or set()
        naija = spec.name in _NAIJA_STYLES and rng.random() < 0.30
        meter = _METER.get(spec.name, (6, 11))
        # draw a few candidates and keep the first that fits the meter —
        # a line that can't be sung in the groove is a bad line
        best: tuple[str, str, str] | None = None  # (line, imagery, em)
        best_miss = 10 ** 9
        for _ in range(6):
            pidgin_line = False
            if section == "chorus":
                if naija and rng.random() < 0.5:
                    pool = _PIDGIN_HOOKS
                    pidgin_line = True
                else:
                    pool = (_HOOK_TEMPLATES_R if end_group is not None
                            else _HOOK_TEMPLATES)
            elif section == "pre-chorus":
                pool = (_PRECHORUS_TEMPLATES_R if end_group is not None
                        else _PRECHORUS_TEMPLATES)
            elif section == "bridge":
                pool = (_BRIDGE_TEMPLATES_R if end_group is not None
                        else _BRIDGE_TEMPLATES)
            elif section == "tag":
                pool = (_HOOK_TEMPLATES_R if end_group is not None
                        else _HOOK_TEMPLATES)
            elif naija:
                pool = _PIDGIN_TEMPLATES
                pidgin_line = True
            else:
                pool = (_LINE_TEMPLATES_R if end_group is not None
                        else _LINE_TEMPLATES)
            template = rng.choice(pool)
            # pick imagery we haven't leaned on yet in this section
            avail = [im for im in _IMAGERY if im not in used_imagery]
            imagery = rng.choice(avail or list(_IMAGERY))
            # {em} is always a single-word emotion: an adjective-class word
            # that sits grammatically before {imagery} ("the golden light")
            # and can be swapped for a rhyme word when the line must rhyme
            for _ in range(4):
                em = rng.choice(_EMOTION)
                if em.lower() not in imagery.lower():
                    break
            verb = rng.choice(_VERB_BANK)
            line = template.format(
                verb=verb,
                ing=_ing(verb),
                topic_short=topic_short,
                em=em,
                imagery=imagery,
                noun=rng.choice(topic_words),
            )
            # guarantee a real rhyme on the tail when we were given a group:
            # swap the whole trailing {imagery}/{em} slot for the rhyme
            # word, so the article survives ("across the paper moons" ->
            # "across the light").  Pidgin ad-libs are left unrhy med.
            if end_group is not None and not pidgin_line:
                line = self._rhyme_tail(line, imagery, em, end_group, rng)
            line = re.sub(r"\s+", " ", line).strip()
            line = line[0].upper() + line[1:]
            n = _syllables(line)
            miss = meter[0] - n if n < meter[0] else (
                n - meter[1] if n > meter[1] else 0)
            if miss == 0:
                return line, imagery, em
            if miss < best_miss:
                best_miss = miss
                best = (line, imagery, em)
        assert best is not None  # 6 draws always produce one
        return best

    def _model_lyrics(self, section: str, topic: str,
                      spec: StyleSpec) -> list[str]:
        from ..llm.base import Message, SamplingParams

        router = self.context.router
        n = 8 if section == "verse" else 4
        prompt = (
            f"Write exactly {n} lines of {section} lyrics for a "
            f"{spec.label} song about: {topic!r}. "
            f"Style/mood: {', '.join(spec.palette)}. "
            "No titles, no labels, no markdown — just the lines, one per "
            "line. Make them rhyme and singable."
        )
        resp = router.chat(
            [Message.user(prompt)],
            SamplingParams(temperature=0.9))
        text = (getattr(resp, "text", "") or "").strip()
        lines = [ln.strip(" \t-•“”\"'") for ln in text.splitlines()
                 if ln.strip()]
        lines = [ln for ln in lines if not ln.lower().startswith(section)]
        if not lines:
            raise ValueError("model returned no usable lyric lines")
        return lines[:n]

    def _melody_description(self, spec: StyleSpec, rng: random.Random,
                            key: str, tempo: int) -> str:
        move = rng.choice(("stepwise, conversational", "leaping, expressive",
                           "rising and falling around the root",
                           "pentatonic, call-and-response",
                           "syncopated, rhythm-first",
                           "arch-shaped, peaking on the chorus",
                           "descending, confessional",
                           "repetitive motif with small variations",
                           "wide-interval, anthemic leaps",
                           "staccato, percussive phrasing"))
        register = rng.choice(("low and warm", "mid-range, intimate",
                               "soaring on the chorus",
                               "breathy and close-mic'd",
                               "belting the bridge",
                               "falsetto flourishes on the hook",
                               "spoken-word verses, sung chorus",
                               "layered harmonies stacking upward"))
        mod_note = (", final chorus modulates up a semitone"
                   if spec.modulate else "")
        return (f"{key} {spec.mode}, {tempo} BPM — a {move} line that sits "
                f"{register}; chorus lifts a third over the pre-chorus, "
                f"bridges drop to a half-phrase and resolve on the downbeat. "
                f"Arrangement: {spec.drum_pattern or 'no drums'} kit, "
                f"{spec.bass_pattern} bass, {spec.voicing} voicings, "
                f"call-and-response counter-lines{mod_note}; "
                f"intros build, outros fade, drops hit hardest.")

    # ── MIDI ─────────────────────────────────────────────────────────────
    def _arrange(self, song: Song, spec: StyleSpec,
                 rng: random.Random) -> dict[str, list[NoteEvent]]:
        """Full arrangement: every part, every section, in absolute beats.

        Returns track name → NoteEvents for ``melody``, ``counter``,
        ``chords``, ``bass`` and ``drums``.  Builds and fades, breakdowns,
        drops, style voicings, per-section dynamics and key changes are all
        baked into the events here; :meth:`_write_midi` only renders them.
        """
        parts: dict[str, list[NoteEvent]] = {
            "melody": [], "counter": [], "chords": [], "bass": [],
            "drums": []}
        last_chorus = max(
            (i for i, s in enumerate(song.sections) if s.name == "chorus"),
            default=-1)
        bar = 0
        for idx, s in enumerate(song.sections):
            # key change: modulating styles lift the final chorus (and
            # everything after it) a semitone — the classic last-chorus lift
            lift = 1 if (spec.modulate and last_chorus > 0
                         and idx >= last_chorus) else 0
            if lift and "key change" not in s.note:
                s.note = (s.note + " · " if s.note else "") + "key change +1 st"
            root = transpose_root(song.key, lift)
            # fit the progression to the section: loop short progressions
            # so an 8-bar chorus gets 8 bars of chords, and truncate so a
            # 2-bar outro doesn't ring past its own end
            raw = list(s.chords) or ["I", "V", "vi", "IV"]
            prog = (raw * ((s.bars // len(raw)) + 1))[:s.bars]
            seed = song.seed
            sec: dict[str, list[NoteEvent]] = {k: [] for k in parts}

            # chords, colored + revoiced for the style
            stabs = spec.voicing == "stabs"
            chord_rows = generate_chord_bass(
                root, song.mode, prog, bars_per_chord=1, beats=4,
                seed=seed + idx * 7 + 1)
            bar_chords: list[list[int]] = []   # per-bar harmony, for melody
            for (chord, st, dur), roman in zip(chord_rows, prog,
                                                  strict=True):
                rich = list(chord)
                if spec.extensions:
                    rich = add_chord_extensions(rich, roman, song.mode,
                                                spec.extensions)
                bar_chords.append(rich)
                voiced = apply_voicing(rich, spec.voicing)
                vel = 80
                if stabs:
                    dur = min(dur, 0.5)
                    vel = 86
                sec["chords"].extend(
                    NoteEvent(n, st, dur, velocity=vel, channel=1)
                    for n in voiced)

            # bass line in the style's groove
            sec["bass"].extend(generate_bass_line(
                root, song.mode, prog, style=spec.bass_pattern,
                bars_per_chord=1, beats=4, seed=seed + idx * 13 + 2,
                velocity=92, channel=3))

            # drums: pattern per section role
            pattern = spec.drum_pattern
            if pattern and s.name in ("break", "breakdown"):
                pattern = "breakdown"
            elif pattern and s.name == "buildup":
                pattern = "buildup"
            drums = generate_drums(pattern, bars=s.bars,
                                   seed=seed + idx * 17 + 3, velocity=100)
            if s.name == "intro" and s.bars > 1:
                # the kit kicks in on the last intro bar
                drums = [d for d in drums if d.start >= (s.bars - 1) * 4]
            if s.name == "outro" and s.bars > 1:
                # the kit drops out for the final bar
                drums = [d for d in drums if d.start < (s.bars - 1) * 4]
            sec["drums"].extend(drums)

            # lead melody: motif-based, downbeats anchored to the chord tones
            mel = generate_melody(root, song.mode, bars=s.bars, beats=4,
                                  seed=seed + idx * 31 + 4, velocity=96,
                                  chord_tones=bar_chords)
            up = _MELODY_LIFT.get(s.name, 0)
            if up:
                mel = [NoteEvent(min(127, e.note + up), e.start, e.duration,
                                 e.velocity, e.channel) for e in mel]
            if s.name == "intro":
                mel = mel[::2]  # sparse, spacious entrance
            sec["melody"].extend(humanize(mel, seed=seed + idx * 41 + 5))

            # counter-melody: call and response on the big sections
            if s.name in _COUNTER_SECTIONS and s.bars >= 2:
                ctr = generate_counter_melody(
                    root, song.mode, bars=s.bars, seed=seed + idx * 19 + 6)
                sec["counter"].extend(
                    humanize(ctr, seed=seed + idx * 43 + 7))

            # dynamics: section profile + downbeat accents, then to absolute
            base = float(bar * 4)
            for track, events in sec.items():
                for e in events:
                    scale = _dyn_scale(s.name, int(e.start // 4), s.bars)
                    vel = int(e.velocity * scale)
                    if track in ("melody", "counter", "chords", "bass") \
                            and e.start % 4 < 0.01:
                        vel += 8  # downbeat accent
                    e.velocity = max(1, min(127, vel))
                    e.start += base
                    parts[track].append(e)
            bar += s.bars
        return parts

    def _write_midi(self, song: Song, workdir: str) -> str:
        from ..tools.filesystem import safe_path

        base = safe_path(self.context, (workdir or "music").strip("/"))
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{_slugify(song.title)}.mid"

        spec = resolve_style(song.style)
        rng = random.Random(song.seed ^ 0x5EED)
        parts = self._arrange(song, spec, rng)

        b = MidiBuilder(tempo=song.tempo, time_signature=(4, 4))
        b.set_program(spec.programs[0], channel=0)  # lead
        b.set_program(spec.programs[1], channel=1)  # chords/keys
        b.set_program(spec.programs[0], channel=2)  # counter (same family)
        b.set_program(spec.programs[2], channel=3)  # bass
        # channel 9 is the GM drum kit: no program change needed
        bass_tr = b.new_track("bass")
        drum_tr = b.new_track("drums")
        counter_tr = b.new_track("counter")

        b.add_melody(parts["melody"])
        b.add_notes(counter_tr, parts["counter"])
        b.add_notes(b.chords, parts["chords"])
        b.add_notes(bass_tr, parts["bass"])
        b.add_notes(drum_tr, parts["drums"])
        return b.write(str(target))

    def _render_audio(self, song: Song, workdir: str) -> str:
        """Render the arrangement to a playable WAV (pure-Python synth)."""
        from ..tools.filesystem import safe_path
        from .synth import write_wav

        base = safe_path(self.context, (workdir or "music").strip("/"))
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{_slugify(song.title)}.wav"

        spec = resolve_style(song.style)
        rng = random.Random(song.seed ^ 0x5EED)
        parts = self._arrange(song, spec, rng)
        return write_wav(str(target), parts, float(song.tempo),
                         seed=song.seed ^ 0xA071)

    def _write_score_pdf(self, song: Song, workdir: str) -> str:
        """Render the lead sheet (chords + lyrics + arrangement) to PDF."""
        from ..tools.filesystem import safe_path
        from ..core.pdf import render_pdf

        base = safe_path(self.context, (workdir or "music").strip("/"))
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{_slugify(song.title)}-score.pdf"

        data = render_pdf(
            song.to_score_markdown(),
            title=f"{song.title} — Score",
            headings=True,
            toc=False,
        )
        target.write_bytes(data)
        return str(target)


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "music_writer",
        description=(
            "Compose a real song from a topic: style-aware lyrics (model or "
            "offline rhyme engine), section structure, chord progression, "
            "melody description, a playable .mid file, a rendered WAV audio "
            "file, and a score PDF (lead sheet with chords + lyrics). "
            "action=compose (topic, style, title, key, seed, with_midi, "
            "with_audio, with_score) | styles | "
            "song (slug|title) to re-fetch a saved one."
        ),
        capability=Capability.FS_WRITE,
    )
    def music_writer(
        action: str = "compose", topic: str = "", style: str = "pop",
        title: str = "", key: str = "", seed: int = 0,
        with_midi: bool = True, with_audio: bool = True,
        with_score: bool = True,
    ) -> dict[str, Any]:
        if action == "styles":
            return {"styles": {
                k: {"label": v.label, "tempo": list(v.tempo), "mode": v.mode,
                    "energy": v.energy, "drums": v.drum_pattern or "none",
                    "bass": v.bass_pattern, "voicing": v.voicing,
                    "sections": [n for n, _ in v.sections]}
                for k, v in STYLES.items()}}
        if action == "song":
            return _saved_songs(context, topic)
        creator = MusicCreator(context)
        if action == "compose":
            if not topic.strip():
                raise ToolError("music_writer compose needs a topic")
            song = creator.compose(
                topic, style=style, title=title, key=key,
                seed=seed or None, with_midi=with_midi,
                with_audio=with_audio, with_score=with_score)
            return song.to_dict()
        raise ToolError(f"unknown music_writer action {action!r}")


def _saved_songs(context: Any, lookup: str) -> dict[str, Any]:
    """List previously composed songs, or re-fetch one by slug/title."""
    from ..tools.filesystem import safe_path

    music_dir = safe_path(context, "music", must_exist=False)
    if not music_dir.exists():
        return {"songs": [], "count": 0}
    mids = sorted(music_dir.glob("*.mid"))
    songs = [{"title": p.stem, "midi": str(p),
              "bytes": p.stat().st_size} for p in mids]
    if lookup.strip():
        want = _slugify(lookup)
        for s in songs:
            if s["title"] == want or want in s["title"].lower() \
                    or lookup.lower() in s["title"].lower():
                return {"found": True, "song": s}
        return {"found": False, "songs": songs[:20],
                "note": f"no saved song matching {lookup!r}"}
    return {"songs": songs, "count": len(songs)}
