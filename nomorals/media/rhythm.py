"""Dynamic rhythm engine — generative drums, not preset patterns.

There are NO beat pattern tables in this module.  Every drum hit is placed
by rules that respond to the song's actual parameters:

* **energy** (0-1) → rhythmic density
* **swing** (0-0.6) → timing offset on offbeat subdivisions
* **harmonic rhythm** → kick accents land where chords change
* **bass attacks** → kick locks with the bassline (they hit together)
* **groove keywords** → the model's free-text feel description steers feel
* **section position** → fills happen at structural boundaries, not randomly

Same SongSpec + same seed → same drums.  Deterministic and testable.

    from nomorals.media.rhythm import generate_drums
    events = generate_drums(spec, section, start_bar=0, rng=rng,
                            bass_attacks=[(0.0,), (1.5,)])
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["generate_drums", "generate_bass", "parse_groove_keywords",
           "euclidean", "humanize"]

# MIDI drum notes (General MIDI)
KICK = 36
SNARE = 38
HAT_CLOSED = 42
HAT_OPEN = 46
CRASH = 49
RIDE = 51
TOM_LOW = 43
TOM_MID = 47
SHAKER = 82
CONGA_LOW = 64
CONGA_HIGH = 63
CLAP = 39
RIM = 37
COWBELL = 56


# ── groove keyword parsing ────────────────────────────────────────────────
# The model's free-text groove description steers the engine.  Keywords are
# matched case-insensitively across feel + kick/snare/hat style + percussion.

def parse_groove_keywords(groove: Any) -> dict[str, bool]:
    """Extract feel flags from a GrooveSpec (or anything with those attrs)."""
    text = " ".join([
        str(getattr(groove, "feel", "") or ""),
        str(getattr(groove, "kick_style", "") or ""),
        str(getattr(groove, "snare_style", "") or ""),
        str(getattr(groove, "hat_style", "") or ""),
        str(getattr(groove, "percussion", "") or ""),
    ]).lower()
    has = lambda *words: any(w in text for w in words)
    return {
        "four_floor": has("four-on-floor", "four on the floor", "four-on-the-floor"),
        "half_time": has("half-time", "half time"),
        "double_time": has("double-time", "double time"),
        "syncopated": has("syncopat"),
        "sparse": has("sparse", "minimal", "stripped"),
        "driving": has("driv", "relentless", "pumping", "pounding"),
        "laid_back": has("laid-back", "laid back", "lazy", "behind the beat"),
        "trap": has("trap", "808"),
        "swing_feel": has("swing", "shuffle"),
        "rim": has("rim click", "rimclick"),
        "shaker": has("shaker"),
        "congas": has("conga"),
        "rolls": has("roll", "skitter", "triplet"),
        "breakbeat": has("breakbeat", "break beat", "jungle"),
        "boom_bap": has("boom-bap", "boom bap"),
    }


# ── Euclidean rhythm core ───────────────────────────────────────────────
# Toussaint's research: genre-defining grooves ARE Euclidean distributions.
# Tresillo = E(3,8), cinquillo = E(5,8), bossa nova = E(5,16) shifted.
# This is the generative foundation — not a pattern table, but the
# mathematical principle that GENERATES the patterns.
# Source: https://en.wikipedia.org/wiki/Euclidean_rhythm

def euclidean(pulses: int, steps: int, rotation: int = 0) -> list[bool]:
    """Bjorklund's algorithm: distribute `pulses` hits as evenly as possible
    across `steps` slots.  Returns a boolean list; True = hit.

    ``rotation`` spins the pattern (e.g. tresillo vs son-clave are rotations
    of E(3,8)).

    Deterministic.  Never raises.
    """
    if pulses <= 0:
        return [False] * steps
    if pulses >= steps:
        return [True] * steps
    # Bjorklund
    pattern: list[list[int]] = [[1]] * pulses + [[0]] * (steps - pulses)
    while True:
        # find the trailing zeros to redistribute
        zeros = [p for p in pattern if p == [0]]
        ones = [p for p in pattern if p != [0]]
        if len(zeros) <= 1 or not ones:
            break
        # pair each zero-run with a one-run
        new_pattern: list[list[int]] = []
        for i, o in enumerate(ones):
            if i < len(zeros):
                new_pattern.append(o + zeros[i])
            else:
                new_pattern.append(o)
        # leftover zeros stay at the end
        if len(zeros) > len(ones):
            new_pattern.extend(zeros[len(ones):])
        if len(new_pattern) == len(pattern):
            break
        pattern = new_pattern
    flat = [x for sub in pattern for x in sub]
    # rotate
    rotation = rotation % steps
    flat = flat[-rotation:] + flat[:-rotation] if rotation else flat
    return [bool(x) for x in flat]


# Named Euclidean grooves: the LLM (or the groove keywords) picks a FEEL,
# the engine generates the exact hits via Euclidean distribution.
# These are NOT preset patterns — they're (pulses, steps, rotation) triples
# that generate mathematically correct rhythms for the feel.
_GROOVE_EUCLID = {
    # feel keyword → (kick_pulses, kick_steps, kick_rot,
    #                 hat_pulses, hat_steps, hat_rot)
    "afrobeats":  ((3, 8, 0), (5, 8, 2)),    # tresillo kick, cinquillo hats
    "trap":       ((3, 8, 1), (7, 16, 0)),   # sparse syncopated kick, dense hats
    "house":      ((4, 4, 0), (4, 16, 0)),   # four-floor kick (E(4,4)), 16th hats
    "boom_bap":   ((3, 8, 2), (3, 8, 0)),    # swung kick, tresillo hats
    "bossa":      ((3, 8, 0), (5, 16, 3)),   # tresillo kick, bossa hats
    "half_time":  ((2, 8, 0), (3, 8, 1)),    # sparse kick, light hats
    "driving":    ((5, 8, 0), (8, 16, 0)),   # cinquillo kick, full 16th hats
    "sparse":     ((2, 8, 3), (2, 8, 0)),    # minimal
}


def _euclid_for_groove(kw: dict[str, bool], energy: float,
                       rng: random.Random) -> tuple[tuple, tuple]:
    """Pick the Euclidean parameters from groove keywords + energy.

    Returns ((kick_pulses, kick_steps, kick_rot),
             (hat_pulses, hat_steps, hat_rot)).
    """
    if kw.get("four_floor"):
        base = _GROOVE_EUCLID["house"]
    elif kw.get("trap"):
        base = _GROOVE_EUCLID["trap"]
    elif kw.get("boom_bap"):
        base = _GROOVE_EUCLID["boom_bap"]
    elif kw.get("half_time"):
        base = _GROOVE_EUCLID["half_time"]
    elif kw.get("driving") and energy > 0.65:
        base = _GROOVE_EUCLID["driving"]
    elif energy < 0.3 or kw.get("sparse"):
        base = _GROOVE_EUCLID["sparse"]
    else:
        base = _GROOVE_EUCLID["afrobeats"]  # default: syncopated bounce

    # energy modulates pulse COUNT, not the structure — denser when hotter
    (kp, ks, kr), (hp, hs, hr) = base
    kp = max(1, min(ks, kp + (1 if energy > 0.75 else 0)
                    - (1 if energy < 0.25 else 0)))
    hp = max(1, min(hs, hp + (2 if energy > 0.75 else 0)
                    - (2 if energy < 0.25 else 0)))
    # slight rotation jitter for variety (seeded, deterministic)
    kr = (kr + rng.randint(0, 1)) % ks
    return (kp, ks, kr), (hp, hs, hr)


# ── humanization ──────────────────────────────────────────────────────────
# The "chaotic glue" — micro-timing and velocity variation that makes drums
# feel alive.  Research: AI filters this out as noise; humans add it
# deliberately.  We add it back, subtly.
# Source: https://medium.com/@seonghoonjeon95/how-ai-music-misses-the-mark-on-natural-blending-da847e9da8a2

def humanize(events: list, rng: random.Random,
             timing_ms: float = 8.0, vel_var: int = 6) -> list:
    """Apply micro-timing shifts (±timing_ms) and velocity variation
    (±vel_var) to drum events.  In-place.  Returns the list.

    timing_ms is in milliseconds at 120 BPM — scaled by actual tempo
    by the caller if needed.  Never raises.
    """
    try:
        # ms → beats at 120 BPM: 1 beat = 500ms
        beat_jitter = (timing_ms / 500.0)
        for ev in events:
            # timing: ±jitter, but NEVER move the downbeat
            start = float(getattr(ev, "start", 0.0))
            if abs(start % 4.0) > 0.01:  # not a downbeat
                ev.start = start + rng.uniform(-beat_jitter, beat_jitter)
            # velocity: ±var, clamped
            vel = int(getattr(ev, "velocity", 96))
            ev.velocity = max(1, min(127, vel + rng.randint(-vel_var, vel_var)))
    except Exception:  # noqa: BLE001 - humanization never kills drums
        pass
    return events


# ── main entry ────────────────────────────────────────────────────────────

def generate_drums(spec: Any, section: Any, start_bar: int,
                   rng: random.Random,
                   bass_attacks: list[tuple[float, ...]] | None = None,
                   next_energy: float | None = None) -> list:
    """Generate drum events for one section.

    ``spec``: SongSpec (groove, tempo). ``section``: SectionSpec (energy,
    bars, groove_note). ``start_bar``: absolute bar offset. ``bass_attacks``:
    per-bar list of beat-offsets where the bass hits (for kick lock).
    ``next_energy``: energy of the following section (drives fill intensity).

    Returns a list of NoteEvent.  Never raises.
    """
    from ..core.midi import NoteEvent

    # explicit None guards — getattr defaults aren't enough for None inputs
    if spec is None or section is None:
        return []
    try:
        return _generate_drums(spec, section, start_bar, rng,
                               bass_attacks or [], next_energy)
    except Exception as exc:  # noqa: BLE001 - drums never kill the song
        _log.warning("rhythm engine failed: %s", exc)
        return []


def _generate_drums(spec, section, start_bar, rng, bass_attacks,
                    next_energy) -> list:
    from ..core.midi import NoteEvent

    out: list = []
    groove = getattr(spec, "groove", None)
    kw = parse_groove_keywords(groove) if groove else {}
    swing = float(getattr(groove, "swing", 0.0) or 0.0)
    energy = float(getattr(section, "energy", 0.5) or 0.5)
    bars = int(getattr(section, "bars", 8) or 8)
    name = str(getattr(section, "name", "verse") or "verse").lower()
    chords = list(getattr(section, "chords", "") or [])

    # harmonic rhythm: which beats have chord changes (kick accents land here)
    change_beats = _chord_change_beats(chords, bars)

    # Euclidean groove parameters for this section — computed once,
    # consistent across all bars of the section
    euclid_params = _euclid_for_groove(kw, energy, rng)

    for b in range(bars):
        bar_start = (start_bar + b) * 4.0
        is_first = (b == 0)
        is_last = (b == bars - 1)
        bass_bar = bass_attacks[b] if b < len(bass_attacks) else ()

        # crash on section starts (not intro/outro)
        if is_first and name not in ("intro", "outro"):
            out.append(NoteEvent(note=CRASH, start=bar_start,
                                 duration=1.0, velocity=96))

        _kick(out, bar_start, energy, swing, kw, change_beats,
              bass_bar, name, rng, NoteEvent, euclid_params)
        _snare(out, bar_start, energy, swing, kw, name, rng, NoteEvent)
        _hats(out, bar_start, energy, swing, kw, name, rng, NoteEvent,
              euclid_params)
        _percussion(out, bar_start, energy, kw, name, rng, NoteEvent)

        # fill into the next section: structural, intensity from energy delta
        if is_last and next_energy is not None:
            delta = next_energy - energy
            if abs(delta) > 0.05 or name in ("buildup", "pre-chorus"):
                _fill(out, bar_start, energy, delta, kw, rng, NoteEvent)

    # humanize: the "chaotic glue" — micro-timing + velocity variation
    # (research: this is what makes drums feel alive vs mechanical)
    humanize(out, rng)
    return out


# ── harmonic rhythm ───────────────────────────────────────────────────────

def _chord_change_beats(chords: list[str], bars: int) -> set[float]:
    """Beat offsets (within a bar) where the harmony changes.

    With one chord per bar cycling, changes happen on beat 1 of each bar
    where the chord differs from the previous bar's.
    """
    if not chords:
        return {0.0}
    changes: set[float] = set()
    prog = (chords * ((bars // max(1, len(chords))) + 1))[:bars]
    for b in range(bars):
        if b == 0 or prog[b] != prog[b - 1]:
            changes.add(0.0)  # downbeat of a new chord
    # within-bar changes aren't tracked (one chord per bar model) —
    # the downbeat accent is what matters for kick placement
    return changes


# ── kick ──────────────────────────────────────────────────────────────────
# Generative via Euclidean distributions (Toussaint): the groove keywords
# select a Euclidean feel, energy modulates density, harmonic rhythm and
# bass lock add accents.  No pattern tables — the math generates the hits.

def _kick(out, bar_start, energy, swing, kw, change_beats, bass_bar,
          name, rng, NoteEvent, euclid_params=None) -> None:
    (kp, ks, kr), _ = euclid_params or _euclid_for_groove(kw, energy, rng)

    # Euclidean pattern on an 8th-note grid (ks=8) or 16th (ks=16)
    # Normalize to beat positions
    hits = euclidean(kp, ks, kr)
    grid = ks / 4.0  # slots per beat

    for i, hit in enumerate(hits):
        if not hit:
            continue
        slot = i / grid  # beat position within the bar

        # harmonic rhythm: boost kick where chords change
        # bass lock: boost where the bass hits
        boost = 0.0
        if any(abs(slot - cb) < 0.26 for cb in change_beats):
            boost += 0.25
        if any(abs(slot - ba) < 0.13 for ba in bass_bar):
            boost += 0.3

        # energy gate: sparser when quiet (but keep the Euclidean shape)
        gate = 0.6 + energy * 0.4 + boost
        if name in ("intro", "outro", "break"):
            gate *= 0.7

        if rng.random() < min(0.98, gate):
            # velocity: downbeats hit harder
            if abs(slot) < 0.01:
                vel = 114
            elif abs(slot % 1.0) < 0.01:
                vel = 106
            else:
                vel = rng.randint(88, 100)
            out.append(NoteEvent(
                note=KICK,
                start=_swung(bar_start + slot, slot, swing),
                duration=0.25, velocity=vel))


def _swung(beat_pos: float, slot: float, swing: float) -> float:
    """Apply swing: delay offbeat 8ths/16ths proportionally."""
    if swing <= 0:
        return beat_pos
    # swing affects the "and" of each beat (x.5 positions)
    frac = slot % 1.0
    if abs(frac - 0.5) < 0.01:
        return beat_pos + swing * 0.33  # push the offbeat late
    return beat_pos


# ── snare ─────────────────────────────────────────────────────────────────

def _snare(out, bar_start, energy, swing, kw, name, rng, NoteEvent) -> None:
    if kw.get("half_time"):
        # half-time: snare on 3 only
        beats = [2.0]
    elif kw.get("double_time") or (kw.get("driving") and energy > 0.8):
        beats = [1.0, 3.0]
        # extra ghost snares when very hot
        if energy > 0.85 and rng.random() < 0.5:
            beats.append(3.5)
    else:
        # standard backbeat
        beats = [1.0, 3.0]

    # breakdowns strip the snare
    if name in ("intro", "break") and energy < 0.35:
        beats = [3.0] if rng.random() < 0.5 else []

    for beat in beats:
        # rim clicks when quiet or requested
        use_rim = kw.get("rim") or (energy < 0.3 and name in ("verse", "intro"))
        note = RIM if use_rim else SNARE
        vel = 70 if use_rim else (104 if energy > 0.6 else 92)
        # clap layer on backbeats when driving
        out.append(NoteEvent(note=note,
                             start=_swung(bar_start + beat, beat, swing),
                             duration=0.25, velocity=vel))
        if not use_rim and kw.get("driving") and energy > 0.65:
            out.append(NoteEvent(note=CLAP,
                                 start=_swung(bar_start + beat, beat, swing),
                                 duration=0.2, velocity=max(60, vel - 20)))


# ── hats ──────────────────────────────────────────────────────────────────

def _hats(out, bar_start, energy, swing, kw, name, rng, NoteEvent,
          euclid_params=None) -> None:
    # Euclidean hat distribution — the groove's mathematical fingerprint
    _, (hp, hs, hr) = euclid_params or _euclid_for_groove(kw, energy, rng)
    hits = euclidean(hp, hs, hr)
    grid = hs / 4.0  # slots per beat

    # trap triplets: subdivide some 8ths into triplets
    triplet_spots: set[float] = set()
    if kw.get("trap") and energy > 0.5:
        for beat in range(4):
            if rng.random() < 0.3:
                triplet_spots.add(float(beat))

    for i, hit in enumerate(hits):
        if not hit:
            continue
        pos = i / grid

        if pos in triplet_spots:
            # triplet burst: 3 hits in the space of 2 16ths
            for j in range(3):
                t = pos + j * (0.5 / 3)
                out.append(NoteEvent(
                    note=HAT_CLOSED,
                    start=_swung(bar_start + t, t, swing * 0.5),
                    duration=0.08,
                    velocity=rng.randint(60, 78)))
            continue

        # density gate: sparser when quiet (but keep the Euclidean shape)
        gate = 0.55 + energy * 0.45
        if name in ("intro", "break"):
            gate *= 0.7
        if rng.random() < gate:
            # open hat on offbeats when driving
            is_offbeat = abs((pos % 1.0) - 0.5) < 0.26
            use_open = (is_offbeat and kw.get("driving")
                        and energy > 0.6 and rng.random() < 0.3)
            note = HAT_OPEN if use_open else HAT_CLOSED
            # velocity accents: on-beats stronger, natural lilt
            if abs(pos % 1.0) < 0.01:
                vel = 84
            elif is_offbeat:
                vel = 72
            else:
                vel = rng.randint(58, 70)
            out.append(NoteEvent(
                note=note,
                start=_swung(bar_start + pos, pos, swing),
                duration=0.5 if use_open else 0.1,
                velocity=vel))


# ── percussion ────────────────────────────────────────────────────────────

def _percussion(out, bar_start, energy, kw, name, rng, NoteEvent) -> None:
    if name in ("intro",) and energy < 0.3:
        return  # let the intro breathe

    if kw.get("shaker"):
        # shaker: steady 16ths, velocity wave (the "bounce")
        for i in range(16):
            pos = i * 0.25
            # wave: accents on beats, dips between — this is the bounce
            wave = 0.75 + 0.25 * (1 - abs((i % 4) - 0) / 4)
            if rng.random() < 0.7 + energy * 0.3:
                out.append(NoteEvent(
                    note=SHAKER, start=bar_start + pos, duration=0.08,
                    velocity=int(68 * wave)))

    if kw.get("congas"):
        # conga pattern: syncopated tumbao-ish, generative not fixed
        base_hits = [0.0, 0.75, 1.5, 2.0, 2.75, 3.5]
        for pos in base_hits:
            if rng.random() < 0.5 + energy * 0.4:
                note = CONGA_LOW if pos % 1.0 == 0 else CONGA_HIGH
                out.append(NoteEvent(
                    note=note, start=bar_start + pos, duration=0.2,
                    velocity=rng.randint(72, 92)))

    if "cowbell" in str(kw) or kw.get("trap"):
        # sparse cowbell accents
        if rng.random() < 0.25 + energy * 0.2:
            pos = rng.choice([0.5, 1.5, 2.5, 3.5])
            out.append(NoteEvent(note=COWBELL, start=bar_start + pos,
                                 duration=0.15, velocity=76))


# ── fills ─────────────────────────────────────────────────────────────────
# Structural: last bar of a section, intensity from the energy delta into
# the next section.  Bigger lift → bigger fill.

def _fill(out, bar_start, energy, delta, kw, rng, NoteEvent) -> None:
    # fill occupies the last 2 beats of the bar
    fill_start = bar_start + 2.0
    intensity = min(1.0, 0.3 + abs(delta) * 1.5 + energy * 0.3)

    if delta > 0.15 or energy > 0.7:
        # snare build: accelerating hits into the downbeat
        n = 8 if intensity > 0.7 else 4
        for i in range(n):
            t = fill_start + (i / n) * 2.0
            vel = int(70 + (i / max(1, n - 1)) * 40)
            out.append(NoteEvent(note=SNARE, start=t, duration=0.15,
                                 velocity=min(120, vel)))
        # crash on the downbeat is added by the next section's start
    elif delta < -0.15:
        # winding down: tom descent, sparse
        for i, tom in enumerate([TOM_MID, TOM_LOW]):
            t = fill_start + i * 1.0
            out.append(NoteEvent(note=tom, start=t, duration=0.4,
                                 velocity=80))
    else:
        # neutral: small snare pickup
        if rng.random() < 0.6:
            for i in range(2):
                t = fill_start + 1.0 + i * 0.5
                out.append(NoteEvent(note=SNARE, start=t, duration=0.15,
                                     velocity=78 + i * 8))


# ── bass ──────────────────────────────────────────────────────────────────
# The bass LOCKS WITH THE KICK.  It takes the kick pattern's attack points
# and builds a bassline that dances around them — this is the groove.

def generate_bass(spec: Any, section: Any, start_bar: int,
                  chord_roots: list[int],
                  kick_attacks: list[list[float]] | None,
                  rng: random.Random) -> tuple[list, list[list[float]]]:
    """Generate a bassline that grooves with the drums.

    Returns (events, bass_attacks_per_bar) — the attacks feed back into
    the kick placement for lock-tight rhythm section interplay.

    Never raises.
    """
    from ..core.midi import NoteEvent

    if spec is None or section is None:
        return [], []
    try:
        return _generate_bass(spec, section, start_bar, chord_roots,
                              kick_attacks or [], rng, NoteEvent)
    except Exception as exc:  # noqa: BLE001
        _log.warning("bass generation failed: %s", exc)
        return [], []


def _generate_bass(spec, section, start_bar, chord_roots, kick_attacks,
                   rng, NoteEvent) -> tuple[list, list[list[float]]]:
    out: list = []
    attacks_per_bar: list[list[float]] = []

    groove = getattr(spec, "groove", None)
    kw = parse_groove_keywords(groove) if groove else {}
    bass_approach = str(getattr(spec, "bass_approach", "") or "").lower()
    energy = float(getattr(section, "energy", 0.5) or 0.5)
    bars = int(getattr(section, "bars", 8) or 8)
    chords = list(getattr(section, "chords", "") or [])

    # approach keywords from the model's bass_approach text
    melodic = any(w in bass_approach for w in
                  ("melodic", "dances", "log drum", "log-drum"))
    glides = any(w in bass_approach for w in
                 ("glide", "slide", "808"))
    octave_pops = "octave" in bass_approach
    pump = any(w in bass_approach for w in ("pump", "driving", "locked"))

    for b in range(bars):
        bar_start = (start_bar + b) * 4.0
        root = chord_roots[b] if b < len(chord_roots) else 36
        kicks = kick_attacks[b] if b < len(kick_attacks) else [0.0]
        bar_attacks: list[float] = []

        if melodic:
            # melodic bass: root + movement, syncopated around the kick
            # hit WITH the kick on downbeats, dance between
            positions = [0.0]  # always with the kick on 1
            # add syncopated positions that complement (not duplicate) kicks
            for off in (0.75, 1.5, 2.25, 2.75, 3.25):
                kick_near = any(abs(off - k) < 0.2 for k in kicks)
                # play in the gaps OR double the kick for punch
                if (not kick_near and rng.random() < 0.3 + energy * 0.3) or \
                   (kick_near and rng.random() < 0.5):
                    positions.append(off)
            for pos in sorted(positions):
                # note choice: root mostly, fifth/octave for movement
                choice = rng.random()
                if choice < 0.6:
                    note = root
                elif choice < 0.8:
                    note = root + 7  # fifth
                elif octave_pops and choice < 0.9:
                    note = root + 12  # octave pop
                else:
                    note = root + rng.choice([3, 5, 10])  # color tones
                out.append(NoteEvent(
                    note=note, start=bar_start + pos,
                    duration=0.4 if pos % 1 else 0.6,
                    velocity=rng.randint(96, 110)))
                bar_attacks.append(pos)

        elif glides or kw.get("trap"):
            # 808 style: long notes on kick hits, with slides
            for k in kicks:
                if rng.random() < 0.8:
                    ev = NoteEvent(note=root, start=bar_start + k,
                                   duration=1.2, velocity=110)
                    # slide to the next root on some hits
                    if rng.random() < 0.25 and b + 1 < len(chord_roots):
                        ev.slide_to = chord_roots[b + 1]
                    out.append(ev)
                    bar_attacks.append(k)

        elif pump or kw.get("four_floor"):
            # driving pump: 8ths locked to kick quarters
            for beat in range(4):
                for sub, dur in ((0.0, 0.4), (0.5, 0.3)):
                    pos = beat + sub
                    note = root + (12 if octave_pops and beat >= 2
                                   and rng.random() < 0.3 else 0)
                    out.append(NoteEvent(
                        note=note, start=bar_start + pos,
                        duration=dur, velocity=104 if sub == 0 else 96))
                    bar_attacks.append(pos)

        else:
            # default: roots on kick hits + sparse movement
            for k in kicks:
                out.append(NoteEvent(note=root, start=bar_start + k,
                                     duration=0.5, velocity=102))
                bar_attacks.append(k)
            # occasional fifth on the "and" for movement
            if energy > 0.4 and rng.random() < 0.4:
                pos = 2.5
                out.append(NoteEvent(note=root + 7, start=bar_start + pos,
                                     duration=0.4, velocity=94))
                bar_attacks.append(pos)

        attacks_per_bar.append(sorted(set(bar_attacks)))

    return out, attacks_per_bar
