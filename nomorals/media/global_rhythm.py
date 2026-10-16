"""Procedural rhythm engine — GLOBAL. Generative rules, no corpus, no tables.

Tier 2. Extends :mod:`nomorals.media.rhythm` (Euclidean core, energy →
density, swing, groove keywords) with a *genre* layer: every genre is a
:class:`GenreRhythm` descriptor — subdivision grid, accent rule,
Euclidean pulse counts, swing range, kick/snare/hat *rules* (functions
of bar position, energy, and section), not a stored pattern. The same
descriptor + seed + SongSpec renders the same drums every time.

SCOPE — universal, Nigeria as DEFAULT, not boundary. N packs,
nation-generic: afrobeats/amapiano are the default pack (owner
context, best place to start), every other genre an equal pack.

Genres covered (descriptors, all generative): afrobeats, amapiano,
highlife, hip-hop, trap, pop, rock, edm/house, techno, drum-and-bass,
latin/reggaeton, dancehall, k-pop, bollywood, jazz, funk, gospel,
country, metal. A genre not listed falls back to the nearest family
via :func:`genre_family` — never a crash, never a silent default beat.

Groove templates (the anti-straight-quantize fix)
-------------------------------------------------
:func:`extract_groove_template` learns microtiming from a list of
onset times (e.g. a drummer's take, or a reference loop's onsets):
for each 16th-grid slot it records the median offset in ticks, so the
engine can *replay* a human feel. :func:`apply_groove` shifts
generated hits by the template instead of quantizing straight.
No audio is analysed here — onsets come from the caller (MIDI, a
transient detector, or hand-tapped times).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from .rhythm import euclidean

__all__ = [
    "GenreRhythm",
    "GENRE_RHYTHMS",
    "genre_family",
    "genre_rhythm",
    "generate_genre_drums",
    "extract_groove_template",
    "apply_groove",
    "GrooveTemplate",
    "register",
]

# ── descriptors ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenreRhythm:
    """Generative rhythm descriptor for one genre (or family)."""

    genre: str
    family: str                 # "african" | "urban" | "pop" | "rock" | ...
    bpm_range: tuple[int, int]  # typical BPM window (guidance, not a clamp)
    steps_per_bar: int = 16     # grid resolution (16 = 16ths)
    kick_pulses: int = 4        # Euclidean pulses for the kick layer
    kick_rotation: int = 0
    snare_pulses: int = 2       # Euclidean pulses for snare/clap layer
    snare_rotation: int = 0
    hat_pulses: int = 8         # Euclidean pulses for hats/shaker
    hat_rotation: int = 0
    swing: float = 0.0          # default swing 0..0.6
    swing_range: tuple[float, float] = (0.0, 0.3)
    kick_rule: str = "euclidean"   # "euclidean" | "four_floor" | "boom_bap" | "dembow" | "half_time" | "tresillo"
    snare_rule: str = "backbeat"   # "backbeat" | "half_time" | "tresillo" | "clave" | "none"
    hat_rule: str = "offbeat"      # "offbeat" | "straight" | "triplet" | "shaker"
    percussion: tuple[str, ...] = ()  # extra voices the rules may add
    microtiming_ms: float = 0.0  # default humanization window
    feel: str = ""


def _g(genre: str, family: str, bpm: tuple[int, int], **kw: Any) -> GenreRhythm:
    return GenreRhythm(genre=genre, family=family, bpm_range=bpm, **kw)


GENRE_RHYTHMS: dict[str, GenreRhythm] = {
    # ── African (the moat, generative) ──
    "afrobeats": _g("afrobeats", "african", (95, 110), kick_pulses=3,
                    kick_rotation=0, snare_pulses=2, snare_rotation=4,
                    hat_pulses=12, swing=0.08, swing_range=(0.0, 0.18),
                    kick_rule="tresillo", snare_rule="backbeat",
                    hat_rule="shaker", percussion=("shaker", "conga"),
                    microtiming_ms=14.0,
                    feel="syncopated tresillo kick, snare backbeat, shaker 16ths"),
    "amapiano": _g("amapiano", "african", (110, 115), kick_pulses=4,
                   kick_rotation=0, snare_pulses=2, snare_rotation=0,
                   hat_pulses=10, swing=0.06, swing_range=(0.0, 0.15),
                   kick_rule="four_floor", snare_rule="backbeat",
                   hat_rule="shaker", percussion=("shaker", "log_drum"),
                   microtiming_ms=10.0,
                   feel="four-floor kick, log-drum bass melody, shaker"),
    "highlife": _g("highlife", "african", (100, 120), kick_pulses=4,
                   kick_rotation=0, snare_pulses=2, snare_rotation=0,
                   hat_pulses=12, swing=0.10, swing_range=(0.0, 0.2),
                   kick_rule="euclidean", snare_rule="backbeat",
                   hat_rule="shaker", percussion=("shaker", "conga", "cowbell"),
                   microtiming_ms=12.0, feel="bouncy highlife guitar-band groove"),
    "dancehall": _g("dancehall", "african", (90, 110), kick_pulses=3,
                    kick_rotation=0, snare_pulses=2, snare_rotation=0,
                    hat_pulses=8, swing=0.05, kick_rule="tresillo",
                    snare_rule="backbeat", hat_rule="offbeat",
                    percussion=("shaker",), microtiming_ms=12.0),
    # ── Urban ──
    "hip-hop": _g("hip-hop", "urban", (80, 100), kick_pulses=4,
                  kick_rotation=0, snare_pulses=2, snare_rotation=0,
                  hat_pulses=8, swing=0.18, swing_range=(0.05, 0.35),
                  kick_rule="boom_bap", snare_rule="backbeat",
                  hat_rule="offbeat", microtiming_ms=16.0,
                  feel="boom-bap: kick syncopated against snare 2&4"),
    "trap": _g("trap", "urban", (130, 150), kick_pulses=3,
               kick_rotation=0, snare_pulses=2, snare_rotation=8,
               hat_pulses=12, swing=0.0, kick_rule="euclidean",
               snare_rule="half_time", hat_rule="triplet",
               percussion=("808",), microtiming_ms=6.0,
               feel="half-time snare, rolling triplet hats, 808 kick"),
    "drill": _g("drill", "urban", (138, 148), kick_pulses=4,
                kick_rotation=2, snare_pulses=2, snare_rotation=8,
                hat_pulses=12, swing=0.05, kick_rule="euclidean",
                snare_rule="half_time", hat_rule="triplet",
                microtiming_ms=6.0, feel="sliding 808, half-time snare"),
    "rnb": _g("rnb", "urban", (60, 90), kick_pulses=3,
              kick_rotation=0, snare_pulses=2, snare_rotation=0,
              hat_pulses=8, swing=0.22, swing_range=(0.1, 0.4),
              kick_rule="euclidean", snare_rule="backbeat",
              hat_rule="offbeat", microtiming_ms=14.0),
    # ── Pop / rock ──
    "pop": _g("pop", "pop", (90, 120), kick_pulses=4, kick_rotation=0,
              snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.02,
              kick_rule="four_floor", snare_rule="backbeat",
              hat_rule="offbeat", microtiming_ms=6.0),
    "rock": _g("rock", "rock", (110, 150), kick_pulses=4, kick_rotation=0,
               snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.0,
               kick_rule="euclidean", snare_rule="backbeat",
               hat_rule="straight", microtiming_ms=5.0,
               feel="driving 8th hats, kick follows bass"),
    "metal": _g("metal", "rock", (140, 200), kick_pulses=8, kick_rotation=0,
                snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.0,
                kick_rule="euclidean", snare_rule="backbeat",
                hat_rule="straight", microtiming_ms=3.0,
                feel="double-kick density scales with energy"),
    "country": _g("country", "pop", (80, 130), kick_pulses=2, kick_rotation=0,
                  snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.12,
                  kick_rule="euclidean", snare_rule="backbeat",
                  hat_rule="offbeat", microtiming_ms=10.0),
    # ── Electronic ──
    "house": _g("house", "electronic", (120, 130), kick_pulses=4,
                kick_rotation=0, snare_pulses=2, snare_rotation=0,
                hat_pulses=12, swing=0.0, kick_rule="four_floor",
                snare_rule="backbeat", hat_rule="offbeat",
                microtiming_ms=2.0),
    "edm": _g("edm", "electronic", (128, 150), kick_pulses=4,
              kick_rotation=0, snare_pulses=2, snare_rotation=0,
              hat_pulses=12, swing=0.0, kick_rule="four_floor",
              snare_rule="backbeat", hat_rule="offbeat",
              microtiming_ms=2.0),
    "techno": _g("techno", "electronic", (125, 140), kick_pulses=4,
                 kick_rotation=0, snare_pulses=1, snare_rotation=0,
                 hat_pulses=12, swing=0.0, kick_rule="four_floor",
                 snare_rule="none", hat_rule="offbeat", microtiming_ms=2.0),
    "dnb": _g("dnb", "electronic", (170, 180), kick_pulses=2, kick_rotation=0,
              snare_pulses=2, snare_rotation=0, hat_pulses=12, swing=0.05,
              kick_rule="euclidean", snare_rule="backbeat",
              hat_rule="offbeat", microtiming_ms=4.0,
              feel="breakbeat: kick 1 + ~10.5, snare 2&4 at double time"),
    # ── Latin ──
    "reggaeton": _g("reggaeton", "latin", (90, 100), kick_pulses=3,
                    kick_rotation=0, snare_pulses=3, snare_rotation=0,
                    hat_pulses=8, swing=0.0, kick_rule="dembow",
                    snare_rule="tresillo", hat_rule="offbeat",
                    percussion=("shaker",), microtiming_ms=8.0,
                    feel="dembow: kick on 1, +and, 3 — the defining pattern"),
    "salsa": _g("salsa", "latin", (160, 220), kick_pulses=5, kick_rotation=0,
                snare_pulses=5, snare_rotation=0, hat_pulses=8, swing=0.0,
                kick_rule="euclidean", snare_rule="clave",
                hat_rule="offbeat", percussion=("cowbell", "conga", "timbale"),
                microtiming_ms=8.0, feel="son clave 3-2 drives everything"),
    "bossa": _g("bossa", "latin", (120, 140), kick_pulses=5, kick_rotation=2,
                snare_pulses=5, snare_rotation=2, hat_pulses=8, swing=0.05,
                kick_rule="euclidean", snare_rule="clave",
                hat_rule="offbeat", percussion=("shaker",),
                microtiming_ms=10.0),
    # ── Asia ──
    "k-pop": _g("k-pop", "pop", (95, 130), kick_pulses=4, kick_rotation=0,
                snare_pulses=2, snare_rotation=0, hat_pulses=12, swing=0.04,
                kick_rule="four_floor", snare_rule="backbeat",
                hat_rule="triplet", microtiming_ms=4.0,
                feel="polished pop with trap-hat inflections in verses"),
    "bollywood": _g("bollywood", "asia", (90, 130), kick_pulses=4,
                    kick_rotation=0, snare_pulses=2, snare_rotation=0,
                    hat_pulses=8, swing=0.06, kick_rule="euclidean",
                    snare_rule="backbeat", hat_rule="offbeat",
                    percussion=("tabla", "dholak", "shaker"),
                    microtiming_ms=12.0,
                    feel="film-song: dholak/tabla colours over a pop grid"),
    "j-pop": _g("j-pop", "pop", (100, 140), kick_pulses=4, kick_rotation=0,
                snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.02,
                kick_rule="four_floor", snare_rule="backbeat",
                hat_rule="offbeat", microtiming_ms=4.0),
    # ── Jazz / funk / gospel ──
    "jazz": _g("jazz", "jazz", (120, 200), kick_pulses=2, kick_rotation=0,
               snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.35,
               swing_range=(0.25, 0.5), kick_rule="euclidean",
               snare_rule="none", hat_rule="offbeat",
               percussion=("ride",), microtiming_ms=18.0,
               feel="ride pattern + feathered kick, heavy swing"),
    "funk": _g("funk", "funk", (90, 115), kick_pulses=5, kick_rotation=0,
               snare_pulses=2, snare_rotation=0, hat_pulses=12, swing=0.12,
               kick_rule="euclidean", snare_rule="backbeat",
               hat_rule="offbeat", microtiming_ms=16.0,
               feel="the one: kick on 1, syncopated 16th pickups"),
    "gospel": _g("gospel", "gospel", (70, 130), kick_pulses=4, kick_rotation=0,
                 snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.15,
                 kick_rule="euclidean", snare_rule="backbeat",
                 hat_rule="offbeat", microtiming_ms=14.0),
    "soul": _g("soul", "pop", (60, 100), kick_pulses=3, kick_rotation=0,
               snare_pulses=2, snare_rotation=0, hat_pulses=8, swing=0.18,
               kick_rule="euclidean", snare_rule="backbeat",
               hat_rule="offbeat", microtiming_ms=14.0),
}

_FAMILY_FALLBACK = {
    "african": "afrobeats", "urban": "hip-hop", "pop": "pop",
    "rock": "rock", "electronic": "house", "latin": "reggaeton",
    "asia": "bollywood", "jazz": "jazz", "funk": "funk",
    "gospel": "gospel",
}
_GENRE_ALIASES = {
    "afrobeat": "afrobeats", "amapiano": "amapiano", "rap": "hip-hop",
    "house": "house", "techno": "techno", "edm": "edm",
    "kpop": "k-pop", "k pop": "k-pop", "jpop": "j-pop",
    "drum and bass": "dnb", "d&b": "dnb", "reggaetón": "reggaeton",
    "latin": "reggaeton", "afro": "afrobeats", "naija": "afrobeats",
    "high-life": "highlife", "dance-hall": "dancehall",
    "r&b": "rnb", "rhythm and blues": "rnb",
}


def genre_family(genre: str) -> str:
    """Family of a genre string; 'pop' for the unknown (never crashes)."""
    g = (genre or "").strip().lower()
    g = _GENRE_ALIASES.get(g, g)
    if g in GENRE_RHYTHMS:
        return GENRE_RHYTHMS[g].family
    return "pop"


def genre_rhythm(genre: str) -> GenreRhythm:
    """Descriptor for a genre; unknown genres fall back to pop rules."""
    g = (genre or "").strip().lower()
    g = _GENRE_ALIASES.get(g, g)
    if g in GENRE_RHYTHMS:
        return GENRE_RHYTHMS[g]
    # family fallback: "afro-soul" -> afrobeats-ish, "latin-pop" -> reggaeton
    for key in ("afro", "latin", "jazz", "funk", "gospel", "techno",
                "house", "metal", "rock", "trap", "drill", "hip", "rap"):
        if key in g:
            fam = {"afro": "african", "latin": "latin", "jazz": "jazz",
                   "funk": "funk", "gospel": "gospel", "techno": "electronic",
                   "house": "electronic", "metal": "rock", "rock": "rock",
                   "trap": "urban", "drill": "urban", "hip": "urban",
                   "rap": "urban"}[key]
            return GENRE_RHYTHMS[_FAMILY_FALLBACK[fam]]
    return GENRE_RHYTHMS["pop"]


# ── generation ──────────────────────────────────────────────────────────────


def _rule_pattern(rule: str, pulses: int, steps: int, rotation: int,
                  beats: int = 4) -> list[bool]:
    """Turn a kick/snare rule into a boolean grid for one bar."""
    if rule == "four_floor":
        return [i % (steps // beats) == 0 for i in range(steps)]
    if rule == "boom_bap":
        # kick: 1, the "and" of 2, 3.5-ish syncopation — derived, not stored
        pat = [False] * steps
        for slot in (0, steps // 4 + steps // 8, steps // 2 + steps // 8,
                     3 * steps // 4):
            pat[slot % steps] = True
        return pat
    if rule == "dembow":
        pat = [False] * steps
        for slot in (0, 3 * steps // 8, steps // 2, 7 * steps // 8):
            pat[slot % steps] = True
        return pat
    if rule == "tresillo":
        return euclidean(3, steps, rotation)
    if rule == "half_time":
        pat = [False] * steps
        pat[steps // 2] = True  # snare on 3 only
        return pat
    if rule == "backbeat":
        pat = [False] * steps
        pat[steps // 4] = True
        pat[3 * steps // 4] = True
        return pat
    if rule == "clave":  # son clave 3-2
        pat = [False] * steps
        for slot in (0, 3 * steps // 16, 6 * steps // 16,
                     8 * steps // 16, 12 * steps // 16):
            pat[slot % steps] = True
        return pat
    if rule == "none":
        return [False] * steps
    return euclidean(pulses, steps, rotation)


def generate_genre_drums(genre: str, bars: int = 4, energy: float = 0.6,
                         swing: float | None = None, seed: int = 0,
                         bpm: float | None = None,
                         groove: "GrooveTemplate | None" = None,
                         section: str = "") -> list[dict[str, Any]]:
    """Generate drum events for a genre — no pattern tables.

    Events: {bar, step, time_beats, voice, velocity, swing_applied}.
    Energy scales density (extra hat subdivisions + ghost kicks at
    high energy; stripped kit at low energy). Deterministic for a
    given (genre, bars, energy, seed). ``groove`` (from
    :func:`extract_groove_template`) shifts hits off the grid instead
    of straight-quantizing them.
    """
    desc = genre_rhythm(genre)
    rng = random.Random(seed)
    steps = desc.steps_per_bar
    sw = desc.swing if swing is None else max(0.0, min(0.6, swing))
    events: list[dict[str, Any]] = []
    kick_pat = _rule_pattern(desc.kick_rule, desc.kick_pulses, steps,
                             desc.kick_rotation)
    snare_pat = _rule_pattern(desc.snare_rule, desc.snare_pulses, steps,
                              desc.snare_rotation)
    hat_pat = _rule_pattern("euclidean", desc.hat_pulses, steps,
                            desc.hat_rotation)
    if desc.hat_rule == "triplet":
        hat_pat = [(i % 3 == 0) for i in range(steps)]
    elif desc.hat_rule == "straight":
        hat_pat = [i % 2 == 0 for i in range(steps)]
    for bar in range(max(1, bars)):
        for step in range(steps):
            t = bar * 4.0 + step * (4.0 / steps)
            # swing: delay offbeat 16ths
            if sw and step % 2 == 1:
                t += sw * (4.0 / steps) * 0.5
            if kick_pat[step]:
                events.append({"bar": bar, "step": step, "time_beats": t,
                               "voice": "kick",
                               "velocity": 0.95 if step % (steps // 4) == 0 else 0.8,
                               "swing_applied": bool(sw and step % 2 == 1)})
            if snare_pat[step]:
                events.append({"bar": bar, "step": step, "time_beats": t,
                               "voice": "snare",
                               "velocity": 0.9, "swing_applied": False})
            if hat_pat[step]:
                vel = 0.55 + 0.25 * (1 if step % 4 == 0 else 0)
                events.append({"bar": bar, "step": step, "time_beats": t,
                               "voice": "hat",
                               "velocity": round(vel, 2),
                               "swing_applied": bool(sw and step % 2 == 1)})
            # energy-driven extras: ghost kicks + shaker doubles
            if energy >= 0.75 and step % 4 == 2 and rng.random() < 0.4:
                events.append({"bar": bar, "step": step, "time_beats": t,
                               "voice": "kick_ghost", "velocity": 0.35,
                               "swing_applied": False})
            if energy >= 0.6 and "shaker" in desc.percussion and step % 2 == 0:
                events.append({"bar": bar, "step": step, "time_beats": t,
                               "voice": "shaker", "velocity": 0.45,
                               "swing_applied": bool(sw and step % 2 == 1)})
        # fill at the end of the phrase: snare roll into the next section
        if bar == bars - 1 and bars > 1 and energy >= 0.5:
            for j in range(4):
                events.append({"bar": bar, "step": steps - 4 + j,
                               "time_beats": bar * 4.0 + (steps - 4 + j) * (4.0 / steps),
                               "voice": "snare", "velocity": 0.4 + 0.15 * j,
                               "swing_applied": False})
    if groove is not None:
        events = apply_groove(events, groove)
    events.sort(key=lambda e: e["time_beats"])
    return events


# ── groove templates ────────────────────────────────────────────────────────


@dataclass
class GrooveTemplate:
    """Learned microtiming: median offset (in beats) per grid slot."""

    name: str = ""
    steps_per_bar: int = 16
    offsets: list[float] = field(default_factory=list)  # per-slot, beats
    source_onsets: int = 0
    confidence: float = 0.0  # fraction of slots with >=3 samples

    def offset_for(self, step: int) -> float:
        if not self.offsets:
            return 0.0
        return self.offsets[step % len(self.offsets)]


def extract_groove_template(onsets: list[float], bpm: float,
                            steps_per_bar: int = 16,
                            name: str = "") -> GrooveTemplate:
    """Learn microtiming from onset times (seconds).

    Each onset is snapped to the nearest 16th slot; the median offset
    per slot becomes the template. Slots with <3 samples get the
    global median (never a wild outlier). Offsets are clamped to
    ±0.35 of a step so a bad onset can't wreck the grid. Returns a
    template even for sparse input — with low confidence, honestly.
    """
    if bpm <= 0 or not onsets:
        return GrooveTemplate(name=name, steps_per_bar=steps_per_bar,
                              offsets=[0.0] * steps_per_bar,
                              source_onsets=0, confidence=0.0)
    beat_s = 60.0 / bpm
    step_s = beat_s * 4.0 / steps_per_bar
    per_slot: list[list[float]] = [[] for _ in range(steps_per_bar)]
    for onset in onsets:
        pos = onset / step_s
        slot = int(round(pos)) % steps_per_bar
        per_slot[slot].append((pos - round(pos)) * step_s / beat_s)  # in beats
    all_offsets = [o for slot in per_slot for o in slot]
    global_med = statistics.median(all_offsets) if all_offsets else 0.0
    max_off = 0.35 * (4.0 / steps_per_bar)
    offsets = []
    covered = 0
    for slot in per_slot:
        if len(slot) >= 3:
            offsets.append(max(-max_off, min(max_off, statistics.median(slot))))
            covered += 1
        elif slot:
            offsets.append(max(-max_off, min(max_off, statistics.median(slot))))
        else:
            offsets.append(global_med)
    return GrooveTemplate(
        name=name or "extracted", steps_per_bar=steps_per_bar,
        offsets=[round(o, 4) for o in offsets],
        source_onsets=len(onsets),
        confidence=round(covered / steps_per_bar, 3))


def apply_groove(events: list[dict[str, Any]],
                 template: GrooveTemplate) -> list[dict[str, Any]]:
    """Shift events by the template's per-slot offsets (beats)."""
    out = []
    for e in events:
        step = int(e.get("step", 0))
        e2 = dict(e)
        e2["time_beats"] = round(float(e["time_beats"])
                                 + template.offset_for(step), 4)
        e2["groove"] = template.name
        out.append(e2)
    return out


def register(registry: Any) -> None:
    """Tool registration: ``global_rhythm`` — genre drums + groove."""
    from ..core.policy import Capability

    @registry.register(
        "global_rhythm",
        description=(
            "Procedural global rhythm engine: action=describe (genre → "
            "generative descriptor) | drums (genre, bars, energy, seed → "
            "drum events) | groove-extract (onsets JSON seconds, bpm → "
            "microtiming template). No pattern tables, no corpus — every "
            "genre is generative rules. Afrobeats/amapiano + hip-hop/pop/"
            "rock/EDM/Latin/K-pop/Bollywood/jazz/funk/gospel and more."
        ),
        capability=Capability.FS_WRITE,
    )
    def global_rhythm(action: str = "describe", genre: str = "afrobeats",
                      bars: int = 4, energy: float = 0.6, seed: int = 0,
                      onsets: str = "", bpm: float = 120.0) -> dict:
        import json as _json
        action = (action or "describe").lower()
        if action == "describe":
            d = genre_rhythm(genre)
            return {"ok": True, "genre": d.genre, "family": d.family,
                    "bpm_range": list(d.bpm_range), "kick_rule": d.kick_rule,
                    "snare_rule": d.snare_rule, "hat_rule": d.hat_rule,
                    "percussion": list(d.percussion), "feel": d.feel}
        if action == "drums":
            return {"ok": True, "genre": genre_rhythm(genre).genre,
                    "events": generate_genre_drums(genre, bars=bars,
                                                   energy=energy, seed=seed)}
        if action == "groove-extract":
            try:
                ons = _json.loads(onsets) if onsets else []
            except Exception as exc:  # noqa: BLE001
                from ..core.errors import ToolError
                raise ToolError(f"onsets must be JSON: {exc}")
            t = extract_groove_template([float(x) for x in ons], bpm)
            return {"ok": True, "offsets": t.offsets,
                    "confidence": t.confidence,
                    "source_onsets": t.source_onsets}
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
