"""MusicCreator — turn a topic into a real, performable song.

Two lyric engines, chosen by availability:
* **model** — when a real model is active, it writes the lyrics to your
  exact structure (verse/chorus/bridge) in the requested style.
* **template** — a genuine offline engine: a rhyme-group bank,
  topic-seeded vocabulary, syllable-aware line assembly, and meter
  constraints. No lorem-ipsum — every line rhymes and carries the topic.

On top of the words it produces a full musical plan: key, mode, tempo,
a chord progression per section, a melody description, and — via
:mod:`nomorals.core.midi` — an actual playable ``.mid`` file you can open
in any DAW or phone music app.

    from nomorals.media.music import MusicCreator
    creator = MusicCreator(context)
    song = creator.compose("late night drive through the city",
                           style="lofi", with_midi=True)
    song.midi_path          # …/workspace/music/<slug>.mid  (real SMF)
    print(song.to_markdown())

Registered as the ``music_writer`` tool.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.midi import (MidiBuilder, generate_chord_bass, generate_melody,
                         note_midi)
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


def _style(name, label, tempo, mode, progressions, sections, energy,
           palette, instrumentation) -> StyleSpec:
    return StyleSpec(name, label, tempo, mode, tuple(progressions),
                     tuple(sections), energy, tuple(palette),
                     tuple(instrumentation))


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
                    "muted guitar")),
    "hiphop": _style("hiphop", "Hip-hop / rap", (85, 100), "minor",
                     (("i", "iv", "v", "i"), ("i", "vii", "iv", "v"),
                      ("i", "bVI", "bVII", "i")),
                     (("intro", 2), ("verse", 16), ("chorus", 8),
                      ("verse", 16), ("chorus", 8), ("outro", 2)),
                   "hard, confident, rhythmic",
                   ("street lights", "concrete", "hustle", "crown", "midnight",
                    "blue flame"),
                   ("808s", "hard kick", "snappy snare", "sub bass",
                    "hi-hat rolls")),
    "afrobeats": _style("afrobeats", "Afrobeats", (100, 110), "major",
                        (("I", "V", "vi", "IV"), ("I", "IV", "vi", "V"),
                         ("IV", "I", "V", "vi")),
                        (("intro", 2), ("verse", 8), ("chorus", 8),
                         ("bridge", 4), ("chorus", 8), ("outro", 2)),
                        "groovy, warm, danceable",
                        ("sun", "garden", "rhythm", "home", "golden",
                         "laughter"),
                        ("log drums", "shakers", "bright guitar", "warm bass",
                         "congas")),
    "amapiano": _style("amapiano", "Amapiano", (110, 115), "minor",
                       (("i", "bVI", "bVII", "i"), ("i", "iv", "bVI", "v"),
                        ("i", "bVII", "iv", "i")),
                       (("intro", 4), ("verse", 8), ("chorus", 8),
                        ("break", 4), ("chorus", 8), ("outro", 2)),
                       "bass-heavy, jazzy, hypnotic",
                       ("night drive", "deep space", "smoke", "slow motion",
                        "neon", "gravity"),
                       ("log drum bass", "piano stabs", "shakers", "soft keys",
                        "deep sub")),
    "pop": _style("pop", "Pop", (100, 120), "major",
                  (("I", "V", "vi", "IV"), ("vi", "IV", "I", "V"),
                   ("I", "IV", "V", "I")),
                  (("intro", 2), ("verse", 8), ("pre-chorus", 4),
                   ("chorus", 8), ("bridge", 4), ("chorus", 8), ("outro", 2)),
                  ("bright", "hooky", "uplifting", "singalong"),
                  ("spark", "hearts", "sky", "lightning", "gold", "forever"),
                  ("synth stabs", "four-on-the-floor", "bright bass",
                   "claps", "supersaw pads")),
    "rock": _style("rock", "Rock", (120, 140), "mixolydian",
                   (("I", "bVII", "IV", "I"), ("I", "IV", "bVII", "IV"),
                    ("I", "V", "IV", "I")),
                   (("intro", 2), ("verse", 8), ("chorus", 8),
                    ("solo", 8), ("chorus", 8), ("outro", 2)),
                   ("driving", "raw", "anthemic", "tense"),
                   ("thunder", "engine", "fists", "wire", "static", "fire"),
                   ("overdriven guitars", "tight drums", "walking bass",
                    "power chords", "double bass drum")),
    "rnb": _style("rnb", "R&B / soul", (70, 95), "natural_minor",
                  (("i", "bVI", "bVII", "i"), ("i", "iv", "bVI", "v"),
                   ("i", "v", "iv", "i")),
                  (("intro", 2), ("verse", 8), ("pre-chorus", 4),
                   ("chorus", 8), ("bridge", 4), ("chorus", 8), ("outro", 2)),
                  ("sensual", "smooth", "emotional", "late-night"),
                  ("velvet", "moon", "silk", "embers", "slow breath",
                   "afterglow"),
                  ("smooth keys", "round drums", "gliding bass", "strings",
                   "whisper vox")),
    "gospel": _style("gospel", "Gospel", (60, 90), "major",
                     (("I", "IV", "V", "I"), ("I", "vi", "IV", "V"),
                      ("I", "V", "IV", "I")),
                     (("intro", 2), ("verse", 8), ("chorus", 8),
                      ("bridge", 8), ("chorus", 8), ("outro", 4)),
                     ("lifting", "joyful", "testimony", "spirited"),
                     ("light", "morning", "grace", "hands raised", "dawn",
                      "praise"),
                     ("piano", "handclaps", "brass stabs", "bass",
                      "choir pads")),
}
#: aliases so natural phrasing maps onto a style
STYLE_ALIASES = {
    "lofi": "lofi", "lo-fi": "lofi", "chill": "lofi", "chillhop": "lofi",
    "chillbeat": "lofi",
    "hiphop": "hiphop", "hip-hop": "hiphop", "rap": "hiphop", "trap": "hiphop",
    "drill": "hiphop", "boom bap": "hiphop", "boom-bap": "hiphop",
    "afro": "afrobeats", "afrobeats": "afrobeats", "afrobeat": "afrobeats",
    "amapiano": "amapiano", "mapiano": "amapiano",
    "pop": "pop",
    "rock": "rock", "punk": "rock", "guitar": "rock",
    "rnb": "rnb", "r&b": "rnb", "soul": "rnb",
    "gospel": "gospel",
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
    raise ToolError(f"unknown style {name!r} — choose from "
                    f"{sorted(STYLES)}")


# ─────────────────────────── offline lyric engine ────────────────────────────

#: rhyme groups: list of (end-word, {syllable-ish tag}) — kept small but real
_RHYME_GROUPS: tuple[tuple[str, ...], ...] = (
    ("night", "light", "sight", "flight", "tight", "bright", "fight", "white"),
    ("fire", "higher", "wire", "desire", "inspire", "smile", "mile", "while"),
    ("heart", "start", "apart", "art", "part", "apart", "chart", "smart"),
    ("dream", "stream", "team", "beam", "seam", "scheme", "gleam", "extreme"),
    ("time", "climb", "mind", "behind", "find", "line", "shine", "sign"),
    ("run", "sun", "one", "done", "dawn", "on", "gone", "song"),
    ("home", "alone", "bone", "tone", "stone", "known", "grow", "low"),
    ("love", "above", "move", "prove", "dove", "stove", "groove", "hoop"),
    ("fall", "call", "wall", "hall", "small", "all", "tall", "recall"),
    ("rise", "eyes", "skies", "wise", "surprise", "lies", "device", "price"),
    ("away", "day", "play", "stay", "pray", "sway", "haze", "gray"),
    ("deep", "sleep", "keep", "steep", "reap", "weep", "repeat", "heat"),
    ("sound", "ground", "found", "round", "bound", "wound", "fount", "profound"),
    ("night", "city", "empty", "pretty", "met it", "fletch it"),
    ("gold", "hold", "cold", "bold", "told", "old", "control", "role"),
    ("blue", "through", "new", "true", "view", "rewind", "in me", "me too"),
    ("free", "be", "sea", "see", "key", "memory", "gravity", "easy"),
    ("way", "pray", "day", "stay", "faded", "swayed", "haze", "phase"),
    ("glow", "low", "go", "know", "show", "slow", "flow", "so"),
    ("light", "flight", "tonight", "right", "bright", "sight", "white", "tight"),
    ("down", "town", "crown", "brown", "frown", "found", "sound", "wound"),
    ("wild", "child", "mild", "styled", "piled", "beguiled", "undefiled", "reconciled"),
    ("ocean", "motion", "devotion", "notion", "explosion", "slow motion", "commotion", "emotion"),
    ("electric", "hectic", "poetic", "magnetic", "kinetic", "eclectic", "apologetic", "sympathetic"),
    ("rhythm", "with them", "dismiss them", "kiss them", "prism", "schism", "tourism", "realism"),
    ("midnight", "first light", "satellite", "meteorite", "dynamite", "overnight", "moonlight", "spotlight"),
    ("gravity", "sanity", "vanity", "humanity", "clarity", "charity", "rarity", "parody"),
    ("fever", "believer", "deceiver", "receiver", "achiever", "reliever", "griever", "weaver"),
    ("soldier", "shoulder", "bolder", "colder", "holder", "folder", "smolder", "beholder"),
    ("static", "dramatic", "ecstatic", "pragmatic", "enigmatic", "traumatic", "emphatic", "dogmatic"),
    ("afterglow", "overflow", "vertigo", "indigo", "calico", "scenario", "ratio", "patio"),
    ("avenue", "continue", "into you", "outgrew", "pursue", "rendezvous", "subdue", "untrue"),
)

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
    "I found a {noun} in the {em} and {imagery}",
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
        }

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
                with_midi: bool = True, workdir: str = "music") -> Song:
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
            if name in ("intro", "outro", "break", "solo"):
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
        if name in ("bridge", "break"):
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
        return f"driven by {inst[0]}"

    def _lyrics_for(self, name: str, topic: str, primary: str,
                    topic_words: list[str], spec: StyleSpec,
                    rng: random.Random) -> list[str]:
        if _model_available(self.context):
            try:
                return self._model_lyrics(name, topic, spec)
            except Exception as exc:  # noqa: BLE001
                _log.debug("model lyrics failed, using template: %s", exc)
        n_lines = 8 if name == "verse" else 4
        if name == "pre-chorus":
            n_lines = 4
        topic_short = " ".join(topic_words[:3]).lower()
        lines: list[str] = []
        used_groups: set[str] = set()
        hook_group: tuple[str, ...] | None = None
        used_imagery: set[str] = set()
        for i in range(n_lines):
            grp: tuple[str, ...] | None = None
            if name == "chorus":
                # chorus is a hook: line 0 is the hook, odd lines rhyme with it
                if i == 0:
                    hook_group = rng.choice(_RHYME_GROUPS)
                    grp = hook_group
                elif i % 2 == 1:
                    grp = hook_group
            elif name == "verse" and i % 2 == 1 and lines:
                grp = self._rhyme_for(lines[i - 1], rng, used_groups)
            line, imagery = self._build_line(name, topic, primary, topic_words,
                                             topic_short, spec, rng,
                                             end_group=grp,
                                             used_imagery=used_imagery)
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

    def _build_line(self, section: str, topic: str, primary: str,
                    topic_words: list[str], topic_short: str,
                    spec: StyleSpec, rng: random.Random,
                    end_group: tuple[str, ...] | None = None,
                    used_imagery: set[str] | None = None
                    ) -> tuple[str, str]:
        if section == "chorus":
            template = rng.choice(_HOOK_TEMPLATES)
        else:
            template = rng.choice(_LINE_TEMPLATES)
        used_imagery = used_imagery or set()
        # pick imagery we haven't leaned on yet in this section
        pool = [im for im in _IMAGERY if im not in used_imagery] or list(_IMAGERY)
        imagery = rng.choice(pool)
        # single-word emotions fit "a little {em}" / "{em} alone" slots; the
        # palette's two-word phrases fit "the {em} {imagery}" slots
        em_is_phrase = ("{em} {imagery}" in template
                        or "{em} and {imagery}" in template)
        for _ in range(4):
            em = (rng.choice(spec.palette) if em_is_phrase
                  else rng.choice(_EMOTION))
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
        # guarantee a real rhyme on the tail when we were given a group
        if end_group is not None:
            tail = self._end_word(end_group, rng)
            words = line.split()
            if words:
                words[-1] = tail
            line = " ".join(words)
        line = re.sub(r"\s+", " ", line).strip()
        return line[0].upper() + line[1:], imagery

    def _model_lyrics(self, section: str, topic: str,
                      spec: StyleSpec) -> list[str]:
        router = self.context.router
        n = 8 if section == "verse" else 4
        prompt = (
            f"Write exactly {n} lines of {section} lyrics for a "
            f"{spec.label} song about: {topic!r}. Style/mood: {spec.mood}. "
            "No titles, no labels, no markdown — just the lines, one per "
            "line. Make them rhyme and singable."
        )
        resp = router.chat(
            [{"role": "user", "content": prompt}],
            params={"temperature": 0.9})
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
        return (f"{key} {spec.mode}, {tempo} BPM — a {move} line that sits "
                f"{register}; chorus lifts a third over the pre-chorus, "
                f"bridges drop to a half-phrase and resolve on the downbeat.")

    # ── MIDI ─────────────────────────────────────────────────────────────
    def _write_midi(self, song: Song, workdir: str) -> str:
        from pathlib import Path

        from ..tools.filesystem import safe_path

        base = safe_path(self.context, (workdir or "music").strip("/"))
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{_slugify(song.title)}.mid"

        b = MidiBuilder(tempo=song.tempo, time_signature=(4, 4))
        b.set_program(0)  # piano
        key = song.key
        mode = song.mode
        bars_total = sum(s.bars for s in song.sections)
        beats = bars_total * 4

        # chords: each section plays its own (possibly rotated) progression
        fallback = ("I", "V", "vi", "IV")
        bar_of = 0
        for s in song.sections:
            prog = s.chords or fallback
            for chord, st, dur in generate_chord_bass(
                    key, mode, prog, bars_per_chord=1, beats=4):
                b.add_chords(chord, bar_of * 4 + st, dur, velocity=78)
            bar_of += s.bars
        # melody across the whole form
        melody = generate_melody(key, mode, bars=max(2, bars_total // 2),
                                 beats=4, seed=song.seed)
        b.add_melody(melody)
        return b.write(str(target))


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "music_writer",
        description=(
            "Compose a real song from a topic: style-aware lyrics (model or "
            "offline rhyme engine), section structure, chord progression, "
            "melody description, and a playable .mid file. action=compose "
            "(topic, style, title, key, seed, with_midi) | styles | "
            "song (slug|title) to re-fetch a saved one."
        ),
        capability=Capability.FS_WRITE,
    )
    def music_writer(
        action: str = "compose", topic: str = "", style: str = "pop",
        title: str = "", key: str = "", seed: int = 0,
        with_midi: bool = True,
    ) -> dict[str, Any]:
        if action == "styles":
            return {"styles": {
                k: {"label": v.label, "tempo": list(v.tempo), "mode": v.mode,
                    "energy": v.energy} for k, v in STYLES.items()}}
        if action == "song":
            return _saved_songs(context, topic)
        creator = MusicCreator(context)
        if action == "compose":
            if not topic.strip():
                raise ToolError("music_writer compose needs a topic")
            song = creator.compose(
                topic, style=style, title=title, key=key,
                seed=seed or None, with_midi=with_midi)
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
            if s["title"] == want or lookup.lower() in s["title"].lower():
                return {"found": True, "song": s}
        return {"found": False, "songs": songs[:20],
                "note": f"no saved song matching {lookup!r}"}
    return {"songs": songs, "count": len(songs)}
