"""Producer — a composition engine, not a template renderer.

The music system fills style presets; the Producer *writes* music:

* **Reference analysis** — a Spotify URL is analyzed for its musical
  DNA (tempo, key, mode, energy, mood) via the Spotify oEmbed endpoint
  plus songdata.io's public track pages.  Nothing is copied — the
  reference's melody is never extracted or reused; only its
  characteristics (tempo, key, energy, mood) steer the composition.
* **Motivic composition** — an original 2-bar motif is written from the
  scale (stepwise motion, occasional leaps, resolves to tonic), then
  developed with real compositional techniques: sequence, inversion,
  retrograde, diminution, augmentation, octave displacement.  Each
  section of the song develops the motif differently, so the piece is
  thematically unified instead of stitched from presets.
* **Dynamic arrangement** — section lengths and order come from the
  reference's energy curve (high-energy refs get short intros and
  early drops; melancholic refs get longer builds).  Transitions are
  *composed*: a 2-bar line that melodically connects section A to
  section B (rising run into the drop, falling line into the break).
* **Independent bass** — bass lines follow the harmony but have their
  own rhythm (syncopated pumps, approach notes), not roots-on-beats.
* **Counter-melodies** in breaks, drum parts matched to energy.

    from nomorals.media.producer import Producer, analyze_reference
    p = Producer(context)
    res = p.produce("https://open.spotify.com/track/6r7b1UHvO3fBZe7wBXWTaZ")
    res["path"]   # composed WAV
    res["notes"]  # production notes: motif, developments, arrangement

Chat command is ``/produce``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Producer", "analyze_reference", "describe_to_profile",
           "ReferenceProfile", "register"]

_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "nomorals" / "producer"
_CACHE_TTL = 7 * 24 * 3600
_UA = ("Mozilla/5.0 (Linux; Android 13; Mobile) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36")


# ──────────────────────── reference analysis ────────────────────────────────

@dataclass
class ReferenceProfile:
    title: str = ""
    artist: str = ""
    bpm: float = 128.0
    key: str = "A"
    mode: str = "minor"
    energy_words: tuple[str, ...] = ()
    genre: str = "electronic"
    mood: str = ""
    ok: bool = True
    reason: str = ""


def _http_get(url: str, timeout: float = 15.0) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - network is best-effort
        _log.debug("producer fetch failed %s: %s", url, exc)
        return None


def _oembed_title(track_id: str) -> str:
    url = ("https://open.spotify.com/oembed?url="
           + urllib.parse.quote(f"https://open.spotify.com/track/{track_id}",
                                safe=""))
    try:
        raw = _http_get(url, timeout=10.0)
        if raw:
            return str(json.loads(raw).get("title", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _songdata_profile(track_id: str) -> dict[str, Any]:
    """Scrape songdata.io's public track page for BPM/key/mood.

    Pages carry sentences like: "X is a melancholic, high-energy track
    by A, B … It runs 2:52 at 155 BPM … in F♯ Minor".
    """
    out: dict[str, Any] = {}
    html = _http_get(f"https://songdata.io/track/{track_id}")
    if not html:
        return out
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"is an? ([^.]{3,120}?) track", text, re.IGNORECASE)
    if m:
        out["energy_words"] = tuple(
            w.strip().lower() for w in m.group(1).split(",") if w.strip())
        out["mood"] = m.group(1).strip()
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*BPM", text)
    if m:
        try:
            out["bpm"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"in ([A-G][♯#b]?)\s*(Major|Minor)", text)
    if m:
        out["key"] = m.group(1).replace("♯", "#")
        out["mode"] = m.group(2).lower()
    return out


def _cache_path(track_id: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / (hashlib.sha256(track_id.encode()).hexdigest()[:16]
                         + ".json")


def analyze_reference(url: str) -> ReferenceProfile:
    """Analyze a Spotify track URL into a ReferenceProfile. Never raises."""
    try:
        m = re.search(r"open\.spotify\.com/track/([A-Za-z0-9]+)", url or "")
        if not m:
            return ReferenceProfile(ok=False,
                                    reason="not a Spotify track URL")
        track_id = m.group(1)
        cp = _cache_path(track_id)
        if cp.exists() and time.time() - cp.stat().st_mtime < _CACHE_TTL:
            try:
                data = json.loads(cp.read_text())
                return ReferenceProfile(**data)
            except Exception:  # noqa: BLE001 - corrupt cache, refetch
                pass
        title = _oembed_title(track_id)
        info = _songdata_profile(track_id)
        prof = ReferenceProfile(
            title=title,
            bpm=float(info.get("bpm", 128.0)),
            key=str(info.get("key", "A")),
            mode=str(info.get("mode", "minor")),
            energy_words=tuple(info.get("energy_words", ())),
            mood=str(info.get("mood", "")),
            genre="electronic",
        )
        try:
            cp.write_text(json.dumps({
                "title": prof.title, "artist": prof.artist, "bpm": prof.bpm,
                "key": prof.key, "mode": prof.mode,
                "energy_words": list(prof.energy_words), "genre": prof.genre,
                "mood": prof.mood, "ok": True, "reason": ""}))
        except Exception:  # noqa: BLE001
            pass
        return prof
    except Exception as exc:  # noqa: BLE001 - analysis never raises
        return ReferenceProfile(ok=False, reason=str(exc)[:120])


_GENRE_DEFAULTS = (
    (("dark edm", "melodic techno", "black out", "stay away"), 152, "F#", "minor",
     ("dark", "driving", "melancholic", "relentless")),
    (("uk drill", "uk-drill", "drill"), 141, "C", "minor",
     ("dark", "menacing", "sliding", "sparse")),
    (("trap",), 140, "F", "minor", ("hard", "rolling", "deep")),
    (("phonk", "drift phonk"), 126, "E", "minor", ("dark", "swung", "night")),
    (("house", "deep house"), 124, "A", "minor", ("groovy", "warm")),
    (("edm", "big room", "dance", "electronic"), 128, "G", "minor",
     ("euphoric", "massive", "hands-up")),
    (("afrobeats", "afro"), 104, "D", "major", ("groovy", "warm", "danceable")),
    (("rap", "boom bap", "hip hop", "hiphop"), 92, "C", "minor",
     ("hard", "dusty", "confident")),
    (("rnb", "r&b", "soul"), 96, "Bb", "major", ("smooth", "late-night")),
    (("pop",), 118, "C", "major", ("bright", "catchy")),
)


def describe_to_profile(text: str) -> ReferenceProfile:
    """Build a ReferenceProfile from a plain description (no URL needed)."""
    low = (text or "").lower()
    bpm, key, mode, energy = 128.0, "A", "minor", ("driving",)
    for words, b, k, mo, en in _GENRE_DEFAULTS:
        if any(w in low for w in words):
            bpm, key, mode, energy = float(b), k, mo, en
            break
    extra = []
    for w in ("dark", "melancholic", "relentless", "euphoric", "chill",
              "aggressive", "dreamy", "menacing"):
        if w in low and w not in energy:
            extra.append(w)
    return ReferenceProfile(
        title=(text or "").strip()[:60], bpm=bpm, key=key, mode=mode,
        energy_words=tuple(list(energy) + extra), genre="electronic",
        mood=", ".join(list(energy) + extra))


# ──────────────────────── motivic composition ───────────────────────────────

@dataclass
class MotifNote:
    degree: int      # scale-degree index (0 = tonic, 7 = octave)
    dur: float       # beats


def _scale_degrees(key: str, mode: str) -> list[int]:
    from ..core.midi import scale_notes
    return scale_notes(key, mode, octaves=3, base_octave=4)


def write_motif(key: str, mode: str, bars: int = 2,
                rng: random.Random | None = None) -> list[MotifNote]:
    """Write an original motif: stepwise motion, occasional leaps, resolves
    to tonic.  Degrees are scale indices so it always stays in key."""
    rng = rng or random.Random()
    total = bars * 4.0
    motif: list[MotifNote] = []
    degree = rng.choice([0, 4, 2])          # start: tonic, 5th, or 3rd
    t = 0.0
    while t < total - 0.01:
        remaining = total - t
        # rhythm: mostly quarters/eighths, occasional dotted
        dur = rng.choices([1.0, 0.5, 0.5, 1.5, 0.25],
                          weights=[34, 30, 12, 12, 12])[0]
        dur = min(dur, remaining)
        motif.append(MotifNote(degree=degree, dur=dur))
        t += dur
        if t >= total - 0.01:
            break
        # contour: 65% stepwise, 20% leap, 15% repeat
        r = rng.random()
        if r < 0.65:
            degree += rng.choice([-2, -1, -1, 1, 1, 2])
        elif r < 0.85:
            degree += rng.choice([-4, -3, 3, 4])
        # else repeat
        degree = max(-2, min(10, degree))    # keep singable range
    # resolve: final note lands on the tonic
    if motif:
        motif[-1] = MotifNote(degree=0, dur=motif[-1].dur)
    return motif


def develop_motif(motif: list[MotifNote], technique: str,
                  **kw: Any) -> list[MotifNote]:
    """Classical developmental techniques.  Pure function — motif untouched."""
    tech = (technique or "").lower()
    if tech == "sequence":
        steps = int(kw.get("steps", 2))
        return [MotifNote(n.degree + steps, n.dur) for n in motif]
    if tech == "inversion":
        if not motif:
            return []
        out = [MotifNote(motif[0].degree, motif[0].dur)]
        for i in range(1, len(motif)):
            interval = motif[i].degree - motif[i - 1].degree
            out.append(MotifNote(out[-1].degree - interval, motif[i].dur))
        return out
    if tech == "retrograde":
        return [MotifNote(n.degree, n.dur) for n in reversed(motif)]
    if tech == "diminution":
        return [MotifNote(n.degree, max(0.125, n.dur / 2)) for n in motif]
    if tech == "augmentation":
        return [MotifNote(n.degree, n.dur * 2) for n in motif]
    if tech == "octave":
        up = kw.get("up", True)
        shift = 7 if up else -7
        return [MotifNote(n.degree + shift, n.dur) for n in motif]
    if tech == "fragment":
        # first half only — tightens the idea for transitions
        return [MotifNote(n.degree, n.dur) for n in motif[:max(1, len(motif) // 2)]]
    return [MotifNote(n.degree, n.dur) for n in motif]


def realize_motif(motif: list[MotifNote], scale: list[int],
                  start_beat: float, velocity: int = 96) -> list:
    """Motif → NoteEvents at an absolute beat position."""
    from ..core.midi import NoteEvent
    events: list = []
    t = start_beat
    for n in motif:
        idx = max(0, min(len(scale) - 1, n.degree + 7))
        events.append(NoteEvent(note=scale[idx], start=t,
                               duration=n.dur, velocity=velocity))
        t += n.dur
    return events


# ──────────────────────── arrangement planning ──────────────────────────────

_HIGH = {"high-energy", "high energy", "relentless", "driving", "aggressive",
         "euphoric", "massive", "hands-up", "pounding"}
_MELANCHOLIC = {"melancholic", "melancholy", "dark", "haunting", "dreamy",
                "emotional", "sad", "midnight"}


def energy_score(profile: ReferenceProfile) -> float:
    words = {w.lower() for w in profile.energy_words}
    score = 0.55
    score += 0.25 * len(words & _HIGH) / max(1, len(_HIGH)) * 4
    score -= 0.10 * len(words & _MELANCHOLIC) / max(1, len(_MELANCHOLIC)) * 4
    if profile.bpm >= 145:
        score += 0.15
    elif profile.bpm <= 100:
        score -= 0.10
    return max(0.15, min(1.0, score))


def plan_arrangement(profile: ReferenceProfile,
                     rng: random.Random) -> list[tuple[str, int]]:
    """Sections chosen from the energy curve — not a fixed template."""
    e = energy_score(profile)
    melancholic = bool({w.lower() for w in profile.energy_words}
                       & _MELANCHOLIC)
    if e >= 0.75:
        # high energy: short intro, early drop
        plan = [("intro", 1), ("verse", 4), ("buildup", 2), ("drop", 8),
                ("break", 4), ("buildup", 2), ("drop", 8), ("outro", 2)]
    elif melancholic:
        # melancholic: longer builds, breathing room
        plan = [("intro", 4), ("verse", 8), ("buildup", 4), ("drop", 8),
                ("break", 8), ("verse", 4), ("buildup", 4), ("drop", 8),
                ("outro", 4)]
    else:
        plan = [("intro", 2), ("verse", 8), ("buildup", 4), ("drop", 8),
                ("break", 4), ("drop", 8), ("outro", 2)]
    # small humanizing jitter: ±0 on most, never below 1
    return [(name, max(1, bars)) for name, bars in plan]


_SECTION_TECHNIQUE = {
    "intro": ("augmentation", {}),
    "verse": ("plain", {}),
    "buildup": ("diminution", {}),
    "drop": ("sequence", {"steps": 2}),
    "chorus": ("sequence", {"steps": 2}),
    "break": ("inversion", {}),
    "outro": ("retrograde", {}),
}


def _progression_for(profile: ReferenceProfile) -> tuple[str, ...]:
    words = {w.lower() for w in profile.energy_words}
    if words & _MELANCHOLIC or profile.mode == "minor":
        return ("i", "bVI", "bVII", "v")
    return ("i", "bVII", "bVI", "bVII")


# ──────────────────────── the Producer ─────────────────────────────────────

class Producer:
    """Writes original music in a reference's lane.  Never raises."""

    role = "producer"

    def __init__(self, context: Any) -> None:
        self.context = context

    # ── public ────────────────────────────────────────────────────────────
    def produce(self, ref: str, *, seed: int | None = None,
                workdir: str = "produced") -> dict[str, Any]:
        """Compose an original piece from a Spotify URL or a description.

        Returns {"ok", "path", "title", "notes", ...}; never raises.
        """
        try:
            return self._produce(ref, seed=seed, workdir=workdir)
        except Exception as exc:  # noqa: BLE001 - produce never raises
            _log.exception("produce failed: %s", exc)
            return {"ok": False, "reason": str(exc)[:200]}

    def _produce(self, ref: str, *, seed: int | None,
                 workdir: str) -> dict[str, Any]:
        ref = (ref or "").strip()
        if not ref:
            return {"ok": False, "reason": "give me a Spotify URL or a vibe "
                                           "description: /produce <url|words>"}
        if "open.spotify.com" in ref:
            profile = analyze_reference(ref)
            if not profile.ok:
                return {"ok": False, "reason": profile.reason}
            from_url = True
        else:
            profile = describe_to_profile(ref)
            from_url = False

        if seed is None:
            seed = int(hashlib.sha256(
                f"{ref}|{time.time():.0f}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        motif = write_motif(profile.key, profile.mode, bars=2, rng=rng)
        plan = plan_arrangement(profile, rng)
        progression = _progression_for(profile)
        scale = _scale_degrees(profile.key, profile.mode)

        parts = self._compose_parts(profile, motif, plan, progression,
                                    scale, rng, seed)

        out_path = self._render(profile, parts, seed, workdir)
        notes = self._production_notes(profile, motif, plan, progression,
                                       from_url, ref)
        title = self._title(profile, rng)
        return {"ok": True, "path": out_path, "title": title,
                "notes": notes, "bpm": profile.bpm, "key": profile.key,
                "mode": profile.mode, "seed": seed,
                "sections": [n for n, _ in plan]}

    # ── composition ───────────────────────────────────────────────────────
    def _compose_parts(self, profile: ReferenceProfile,
                       motif: list[MotifNote],
                       plan: list[tuple[str, int]],
                       progression: tuple[str, ...], scale: list[int],
                       rng: random.Random, seed: int) -> dict[str, list]:
        from ..core.midi import NoteEvent, chord_progression

        parts: dict[str, list] = {"melody": [], "counter": [], "chords": [],
                                 "bass": [], "drums": []}
        e = energy_score(profile)
        bar = 0

        for idx, (name, bars) in enumerate(plan):
            # — melody: develop the motif per section role —
            tech, kw = _SECTION_TECHNIQUE.get(name, ("plain", {}))
            dev = motif if tech == "plain" else develop_motif(motif, tech, **kw)
            if name == "drop" and idx > 0:
                dev = develop_motif(dev, "octave", up=True)  # lift the drop
            sec_beats = bars * 4.0
            # lay the developed motif across the section, varying repeats.
            # The tail of the section is reserved for the composed
            # transition into the next section (nothing reserved after
            # the final one, and nothing reserved in sections too short
            # to hold both motif and transition).
            t = bar * 4.0
            end = t + sec_beats
            has_next = idx < len(plan) - 1
            trans_beats = 8.0 if (has_next and bars >= 4) else 0.0
            melody_end = end - trans_beats
            rep = 0
            vel = 100 if name in ("drop", "chorus", "buildup") else 88
            while t < melody_end - 0.01:
                use = dev
                if rep > 0 and rng.random() < 0.35:
                    # variation on repeats: fragment or sequence ±1
                    use = develop_motif(
                        dev, rng.choice(["fragment", "sequence"]),
                        **({"steps": rng.choice([-1, 1])}
                           if rng.random() < 0.5 else {}))
                evs = realize_motif(use, scale, t, velocity=vel)
                # truncate anything spilling past the melody region
                for evn in evs:
                    if evn.start < melody_end:
                        evn.duration = min(evn.duration,
                                           melody_end - evn.start)
                        parts["melody"].append(evn)
                motif_len = sum(n.dur for n in use)
                t += motif_len if motif_len > 0 else 4.0
                rep += 1

            # — transition INTO the next section: composed connector —
            if trans_beats > 0:
                nxt = plan[idx + 1][0]
                trans = self._transition(scale, name, nxt, rng)
                parts["melody"].extend(
                    realize_motif(trans, scale, melody_end, velocity=104))

            # — harmony: chords follow the progression —
            prog = (list(progression) *
                    ((bars // len(progression)) + 1))[:bars]
            chord_sets = chord_progression(profile.key, profile.mode, prog,
                                           base_octave=3)
            for b in range(bars):
                triad = chord_sets[b]
                bstart = (bar + b) * 4.0
                if name in ("drop", "chorus"):
                    # stabs on offbeats
                    for off in (0.0, 1.5, 2.5):
                        for n in triad:
                            parts["chords"].append(NoteEvent(
                                note=n, start=bstart + off, duration=0.4,
                                velocity=92))
                elif name in ("intro", "break", "outro"):
                    # pads: whole-bar swells
                    for n in triad:
                        parts["chords"].append(NoteEvent(
                            note=n + 12, start=bstart, duration=4.0,
                            velocity=64))
                else:
                    # verse/buildup: sustained + pulse
                    for n in triad:
                        parts["chords"].append(NoteEvent(
                            note=n + 12, start=bstart, duration=2.0,
                            velocity=76))

            # — bass: follows harmony, own rhythm —
            for b in range(bars):
                triad = chord_sets[b]
                root = min(triad) - 12  # sub octave
                bstart = (bar + b) * 4.0
                if name in ("drop", "chorus"):
                    # driving pump with syncopated accents
                    pat = [(0.0, 0.5, 110), (0.5, 0.5, 100), (1.0, 0.5, 110),
                           (1.5, 0.5, 100), (2.0, 0.5, 110), (2.5, 0.25, 104),
                           (2.75, 0.25, 104), (3.0, 0.5, 112), (3.5, 0.5, 100)]
                elif e > 0.7:
                    pat = [(0.0, 0.5, 104), (0.5, 0.5, 96), (1.0, 0.5, 104),
                           (2.0, 0.5, 104), (2.5, 0.5, 96), (3.0, 1.0, 108)]
                else:
                    pat = [(0.0, 1.0, 100), (2.0, 1.0, 96), (3.0, 1.0, 102)]
                for off, dur, velb in pat:
                    note = root
                    # approach notes: fifth below on the pickup
                    if off == 3.5 and rng.random() < 0.5:
                        note = root + 7
                    parts["bass"].append(NoteEvent(
                        note=note, start=bstart + off, duration=dur,
                        velocity=velb))

            # — counter-melody in breaks: inversion, lower octave —
            if name == "break":
                inv = develop_motif(motif, "inversion")
                inv = develop_motif(inv, "octave", up=False)
                parts["counter"].extend(
                    realize_motif(inv, scale, bar * 4.0, velocity=72))

            # — drums matched to energy —
            self._drums(parts["drums"], bar, bars, name, e, rng)

            bar += bars
        return parts

    def _transition(self, scale: list[int], frm: str, to: str,
                    rng: random.Random) -> list[MotifNote]:
        """A 2-bar composed connector between sections."""
        rising = to in ("drop", "chorus", "buildup")
        notes: list[MotifNote] = []
        if rising:
            # scalar ascent, accelerating
            deg = rng.choice([0, 2])
            durs = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
            for d in durs:
                notes.append(MotifNote(degree=deg, dur=d))
                deg += rng.choice([1, 2])
        else:
            # falling line, breathing out
            deg = rng.choice([7, 8])
            durs = [1.5, 1.5, 1.0, 1.0, 2.0, 1.0]
            for d in durs:
                notes.append(MotifNote(degree=deg, dur=d))
                deg -= rng.choice([1, 2])
        return notes

    def _drums(self, out: list, bar: int, bars: int, name: str,
               energy: float, rng: random.Random) -> None:
        from ..core.midi import NoteEvent
        KICK, SNARE, HAT, CRASH = 36, 38, 42, 49
        for b in range(bars):
            s0 = (bar + b) * 4.0
            if b == 0:  # crash on section starts (not intro/outro)
                if name not in ("intro", "outro"):
                    out.append(NoteEvent(note=CRASH, start=s0,
                                         duration=1.0, velocity=100))
            for beat in range(4):
                out.append(NoteEvent(note=KICK, start=s0 + beat,
                                     duration=0.25, velocity=112))
                if beat in (1, 3):
                    out.append(NoteEvent(note=SNARE, start=s0 + beat,
                                         duration=0.25, velocity=104))
                # hats: 8ths, 16ths when hot
                step = 0.5 if energy < 0.75 else 0.25
                h = 0.0
                while h < 4.0:
                    out.append(NoteEvent(note=HAT, start=s0 + h,
                                         duration=0.1,
                                         velocity=70 if h % 1 else 82))
                    h += step
            if name == "buildup" and b == bars - 1:
                # snare roll into the drop
                for i in range(8):
                    out.append(NoteEvent(note=SNARE,
                                         start=s0 + i * 0.5,
                                         duration=0.2,
                                         velocity=80 + i * 4))

    # ── render ────────────────────────────────────────────────────────────
    def _render(self, profile: ReferenceProfile, parts: dict[str, list],
                seed: int, workdir: str) -> str:
        from ..tools.filesystem import safe_path
        from .synth_backend import render_wav
        from ..core.midi import MidiBuilder

        base = safe_path(self.context, (workdir or "produced").strip("/"))
        base.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-",
                      (profile.title or "produced").lower()).strip("-") or "produced"
        target = base / f"{slug}-{seed % 100000:05d}.wav"

        # minimal MIDI so fluidsynth (if chosen) has something real
        b = MidiBuilder(tempo=profile.bpm, time_signature=(4, 4))
        for ch, pname in enumerate(["melody", "chords", "bass"]):
            b.add_notes(b.new_track(pname), parts.get(pname, []))
        midi_path = str(target.with_suffix(".mid"))
        try:
            b.write(midi_path)
        except Exception:  # noqa: BLE001 - midi is best-effort
            midi_path = ""

        choice = render_wav(parts, float(profile.bpm), seed, midi_path,
                            str(target), context=self.context)
        _log.info("produced via %s (%s)", choice.name, choice.reason)
        return str(target)

    # ── notes & title ─────────────────────────────────────────────────────
    def _title(self, profile: ReferenceProfile,
               rng: random.Random) -> str:
        words = (profile.title or "").split()
        core = " ".join(w.capitalize() for w in words[:3]) or "Untitled"
        if "open.spotify.com" in (profile.title or ""):
            core = "Untitled"
        prefix = rng.choice(["After the", "Beneath", "Inside", "Beyond",
                             "Under", "Through"])
        return f"{prefix} {core}" if rng.random() < 0.5 else core

    def _production_notes(self, profile: ReferenceProfile,
                          motif: list[MotifNote],
                          plan: list[tuple[str, int]],
                          progression: tuple[str, ...],
                          from_url: bool, ref: str) -> str:
        scale_names = ["1", "2", "3", "4", "5", "6", "7", "8"]
        motif_str = " – ".join(
            f"{scale_names[min(7, max(0, n.degree))]} ({n.dur:g}b)"
            for n in motif[:8])
        e = energy_score(profile)
        lines = [
            f"🎛 PRODUCED — original composition, {profile.bpm:g} BPM, "
            f"{profile.key} {profile.mode}",
        ]
        if from_url:
            lines.append(f"reference lane: {profile.title or ref[:60]} "
                         f"(characteristics only — melody is original)")
        else:
            lines.append(f"vibe brief: {(ref or '')[:80]}")
        if profile.energy_words:
            lines.append(f"energy: {', '.join(profile.energy_words)} "
                         f"(score {e:.2f})")
        lines += [
            f"motif (2 bars, resolves to tonic): {motif_str}",
            "developments: verse = motif plain · buildup = diminution "
            "(rhythmic squeeze) · drop = sequence up 2 + octave lift · "
            "break = inversion (counter-melody, lower octave) · "
            "outro = retrograde",
            f"harmony: {' – '.join(progression)} "
            f"(bass follows roots with syncopated pump + approach notes)",
            "arrangement: " + " → ".join(f"{n}({b})" for n, b in plan),
            "transitions: composed 2-bar connectors — rising scalar run "
            "into drops, falling line into breaks",
        ]
        return "\n".join(lines)


def register() -> dict[str, Any]:
    return {"produce": Producer}
