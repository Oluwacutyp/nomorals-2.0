"""LLM-first song composition.

When the brain is available, THE MODEL WRITES THE SONG — lyrics, structure,
chord progressions, groove feel, melodic ideas, arrangement arc — as a
structured :class:`SongSpec`.  The producer engine then RENDERS that vision.
The algorithm doesn't guess; it executes the model's composition.

When no brain is available (offline, no providers), an algorithmic composer
builds the SongSpec from the existing producer machinery — but it still
flows through the same schema, so the render pipeline is unified.

    from nomorals.media.composer_llm import compose_song_spec
    spec = compose_song_spec(context, "a sad afrobeats song about Lagos traffic")
    # spec.source == "llm" or "algorithmic"
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from ..core.logging_setup import get_logger
from .songspec import GrooveSpec, SectionSpec, SongSpec, SONGSPEC_JSON_SCHEMA

_log = get_logger(__name__)

__all__ = ["compose_song_spec", "algorithmic_spec", "brain_available"]

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def brain_available(context: Any) -> bool:
    """Is there a live LLM behind the router?"""
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


# ── the composition prompt ────────────────────────────────────────────────

_SYSTEM = (
    "You are a professional songwriter and music producer. "
    "You write ORIGINAL songs — never copy existing lyrics or melodies. "
    "Output ONLY valid JSON matching the schema. No prose, no markdown, "
    "no commentary around the JSON."
)

_PROMPT_TEMPLATE = """Write an original song for this request: {prompt!r}

Style hint: {style_hint}
Taste profile (what the listener likes): {taste}

RULES:
- Lyrics must be original, singable, and rhyme within each section.
- Chord progressions use roman numerals (i, VI, III, VII for minor; I, V, vi, IV for major).
- Sections should form a real arc: build energy toward the chorus/drop, breathe in the bridge/break.
  TENSION AND RELEASE — the #1 thing that separates real music from AI slop. Build-ups must
  actually build (rising energy, tightening rhythm). Drops must hit (maximum energy, full drums).
  Bridges/breaks must breathe (pull back, leave space). Never flatline.
- The groove description must be SPECIFIC — not "good beat" but *how* the drums move:
  where the kick sits, what the snare does per section, hat density, swing feel.
- Bass approach must describe how the bassline moves against the kick.
- 3-7 sections total. Keep it tight.

{abc_prompt}

Output ONLY this JSON schema filled in:
""" + SONGSPEC_JSON_SCHEMA


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of model output.  Never raises."""
    if not text:
        return None
    text = text.strip()
    # strip markdown fences if the model added them anyway
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _taste_summary(context: Any) -> str:
    """One-line taste profile for the prompt, best-effort."""
    try:
        from .taste import get_taste
        t = get_taste(context)
        likes = (t.get("liked_styles") or [])[:5]
        if likes:
            return "likes: " + ", ".join(str(x) for x in likes)
    except Exception:  # noqa: BLE001
        pass
    return "unknown — write to the request"


def compose_song_spec(context: Any, prompt: str, *,
                      style_hint: str = "",
                      seed: int | None = None,
                      max_retries: int = 2) -> SongSpec:
    """Compose a SongSpec via the LLM, falling back to algorithmic.

    Never raises — worst case returns an algorithmic spec.
    """
    prompt = (prompt or "").strip() or "an original song"
    if brain_available(context):
        for attempt in range(max_retries + 1):
            try:
                spec = _llm_compose(context, prompt, style_hint, seed)
                if spec is not None:
                    return spec
            except Exception as exc:  # noqa: BLE001
                _log.warning("LLM composition attempt %d failed: %s",
                             attempt, exc)
    _log.info("composing algorithmically (no brain or LLM failed)")
    return algorithmic_spec(prompt, style_hint=style_hint, seed=seed)


def _llm_compose(context: Any, prompt: str, style_hint: str,
                 seed: int | None) -> SongSpec | None:
    from ..llm.base import Message, SamplingParams
    from .abc_melody import ABC_MELODY_PROMPT

    router = context.router
    full_prompt = _PROMPT_TEMPLATE.format(
        prompt=prompt,
        style_hint=style_hint or "any — choose what fits the request",
        taste=_taste_summary(context),
        abc_prompt=ABC_MELODY_PROMPT,
    )
    params = SamplingParams(
        temperature=0.85, max_tokens=3000,
        seed=seed,
    )
    resp = router.chat(
        [Message.system(_SYSTEM), Message.user(full_prompt)],
        params,
    )
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        _log.warning("LLM composition returned empty text")
        return None
    data = _extract_json(text)
    if not data:
        _log.warning("LLM composition: no JSON found in %d chars",
                     len(text))
        return None
    spec = SongSpec.from_dict(data)
    spec.source = "llm"
    spec.topic = prompt
    errors = spec.validate()
    if errors:
        _log.warning("LLM spec invalid: %s", "; ".join(errors[:3]))
        # repair the fatal ones, keep the model's creativity
        if not spec.sections:
            return None
        spec.sections = [s for s in spec.sections if s.chords] or spec.sections
        if spec.mode not in ("major", "minor"):
            spec.mode = "minor"
    _log.info("LLM composed '%s' (%d sections, %d bars)",
              spec.title, len(spec.sections), spec.total_bars)
    return spec


# ── algorithmic fallback ──────────────────────────────────────────────────
# Builds a SongSpec from the existing producer machinery so the render
# pipeline stays unified when no brain is available.

def algorithmic_spec(prompt: str, *, style_hint: str = "",
                     seed: int | None = None) -> SongSpec:
    """Compose a SongSpec without the LLM.  Deterministic given seed."""
    from .producer import (
        ReferenceProfile, describe_to_profile, energy_score,
        plan_arrangement, _progression_for, write_motif, realize_motif,
        _scale_degrees,
    )

    if seed is None:
        seed = int.from_bytes(
            f"{prompt}|{time.time():.0f}".encode(), "little") % (2 ** 31)
    rng = random.Random(seed)

    prof: ReferenceProfile = describe_to_profile(
        f"{prompt} {style_hint}".strip())
    e = energy_score(prof)
    plan = plan_arrangement(prof, rng)
    progression = _progression_for(prof)
    genre = _detect_genre(f"{prompt} {style_hint}")

    # groove words from the profile — the algorithmic composer still
    # describes feel specifically, so the rhythm engine has something real
    groove = _groove_for_profile(prof, e, rng, genre)

    sections: list[SectionSpec] = []
    for name, bars in plan:
        # one chord per bar, cycling the progression
        chords = [(list(progression) * ((bars // len(progression)) + 1))[b]
                  for b in range(bars)]
        sec_energy = _section_energy(name, e, rng)
        sections.append(SectionSpec(
            name=name, bars=bars, chords=chords, energy=sec_energy,
            lyrics=[],  # lyrics come from the vocal/lyric path, not here
            melody_idea=_melody_idea_for(name, prof, rng),
            groove_note=_groove_note_for(name, e, rng),
        ))

    return SongSpec(
        title=_title_for(prompt, prof, rng),
        style=genre, topic=prompt,
        key=prof.key, mode=prof.mode,
        tempo=int(round(prof.bpm)),
        mood=prof.mood or ", ".join(prof.energy_words[:2]),
        groove=groove, sections=sections,
        bass_approach=_bass_approach_for(prof, e, rng, genre),
        instrumentation=_instrumentation_for(prof, rng, genre),
        arrangement_notes=(
            f"energy arc {_energy_arc_words(e)}; "
            f"{len(plan)} sections over {sum(b for _, b in plan)} bars"),
        source="algorithmic",
    )


def _section_energy(name: str, base: float, rng: random.Random) -> float:
    offsets = {
        "intro": -0.25, "verse": -0.1, "pre-chorus": 0.05,
        "chorus": 0.2, "drop": 0.3, "buildup": 0.15,
        "bridge": -0.05, "break": -0.2, "outro": -0.3,
        "interlude": -0.15, "hook": 0.15, "tag": -0.1,
        "post-chorus": 0.1,
    }
    return max(0.05, min(1.0, base + offsets.get(name, 0.0)
                         + rng.uniform(-0.05, 0.05)))


def _detect_genre(text: str) -> str:
    """Detect genre from prompt text (describe_to_profile hardcodes
    'electronic' — we do better for groove selection)."""
    low = (text or "").lower()
    if any(w in low for w in ("afrobeats", "afrobeat", "afro ")):
        return "afrobeats"
    if any(w in low for w in ("trap", "hip-hop", "hip hop", "drill")):
        return "trap"
    if any(w in low for w in ("lo-fi", "lofi", "chillhop")):
        return "lo-fi"
    if any(w in low for w in ("house", "techno", "edm", "dance")):
        return "house"
    if any(w in low for w in ("rnb", "r&b", "soul")):
        return "rnb"
    if any(w in low for w in ("jazz", "bossa")):
        return "jazz"
    return "electronic"


def _groove_for_profile(prof: Any, e: float,
                        rng: random.Random, genre: str = "") -> GrooveSpec:
    genre = (genre or prof.genre or "").lower()
    if "afro" in genre:
        return GrooveSpec(
            feel="bouncy afrobeats — kick syncopated around the log drum, "
                 "shaker driving 16ths with a lazy swing",
            swing=round(rng.uniform(0.1, 0.25), 2),
            kick_style=("syncopated, sparse in verse — sits with the bass; "
                        "denser in chorus"),
            snare_style="backbeat on 2 and 4, rim clicks in verse",
            hat_style="16th shaker with swing, open hats on offbeats",
            percussion="shaker, congas, log-drum accents",
        )
    if "trap" in genre or "hip" in genre:
        return GrooveSpec(
            feel="heavy trap — booming 808, skittering hats, half-time snare",
            swing=round(rng.uniform(0.0, 0.1), 2),
            kick_style="808-driven, sparse but huge — follows the bassline",
            snare_style="half-time on 3, layered clap",
            hat_style="16th rolls with triplet fills, velocity waves",
            percussion="808 cowbell accents, risers",
        )
    if e > 0.7:
        return GrooveSpec(
            feel="driving four-on-the-floor energy, relentless forward motion",
            swing=0.0,
            kick_style="four-on-floor, pumping with the bass",
            snare_style="big backbeat on 2 and 4, clap layer",
            hat_style="driving 8ths, 16ths in the drop",
            percussion="crash on section hits, ride in breaks",
        )
    return GrooveSpec(
        feel="laid-back and spacious — drums breathe around the melody",
        swing=round(rng.uniform(0.1, 0.3), 2),
        kick_style="soft pulse on 1, sparse syncopated accents",
        snare_style="brushy backbeat, rim clicks when quiet",
        hat_style="swung 8ths, lots of air",
        percussion="shaker, soft conga",
    )


def _melody_idea_for(name: str, prof: Any, rng: random.Random) -> str:
    ideas = {
        "intro": ["sparse motif, lots of space", "filtered swell into theme"],
        "verse": ["conversational, stepwise — leaves room for words",
                  "rhythm-first phrasing, syncopated"],
        "chorus": ["wide arch peaking on the hook", "anthemic, singable top line"],
        "drop": ["rhythmic stabs, call-and-response with bass"],
        "bridge": ["contrasting register, introspective turn"],
        "buildup": ["rising sequence, tension climbing"],
        "break": ["stripped motif, inversion for contrast"],
        "outro": ["motif dissolving, fading fragments"],
    }
    return rng.choice(ideas.get(name, ["develop the motif naturally"]))


def _groove_note_for(name: str, e: float, rng: random.Random) -> str:
    notes = {
        "intro": "drums enter gradually — shaker first, then kick",
        "verse": "pull back: kick + hats, snare light",
        "chorus": "full kit, maximum drive",
        "drop": "everything hits — biggest drums of the song",
        "buildup": "rising intensity, snare roll into the drop",
        "bridge": "half-time feel, space for the vocal",
        "break": "strip to percussion and bass",
        "outro": "drums thin out, end on kick + crash",
    }
    return notes.get(name, "")


def _bass_approach_for(prof: Any, e: float, rng: random.Random,
                       genre: str = "") -> str:
    genre = (genre or prof.genre or "").lower()
    if "afro" in genre:
        return ("melodic log-drum style bass — syncopated, dances around the "
                "kick, octave pops on chord changes")
    if "trap" in genre or "hip" in genre:
        return ("808 bass locked to the kick — long glides, follows the "
                "chord roots with slides between")
    if e > 0.7:
        return ("driving root pump locked to the kick, fifth accents, "
                "octave jumps into high-energy sections")
    return ("sparse melodic bass — roots with approach notes, "
            "breathes with the kick pattern")


def _instrumentation_for(prof: Any, rng: random.Random,
                         genre: str = "") -> list[str]:
    genre = (genre or prof.genre or "").lower()
    base = ["drums", "bass", "chords"]
    if "afro" in genre:
        return base + ["log drum", "shaker", "congas", "pluck synth", "pad"]
    if "trap" in genre:
        return base + ["808", "dark pad", "pluck lead", "riser"]
    if "lo-fi" in genre:
        return base + ["rhodes", "vinyl texture", "soft pad"]
    return base + ["piano", "pad", "lead synth"]


def _title_for(prompt: str, prof: Any, rng: random.Random) -> str:
    words = re.findall(r"[a-zA-Z']+", prompt or "")
    stop = {"the", "a", "an", "and", "make", "me", "song", "about",
            "with", "that", "this", "for"}
    keep = [w for w in words if w.lower() not in stop and len(w) > 2]
    core = " ".join(w.capitalize() for w in keep[:3]) or "Untitled"
    return core


def _energy_arc_words(e: float) -> str:
    if e > 0.75:
        return "high throughout with a breathing bridge"
    if e > 0.5:
        return "builds verse → chorus, breathes in the bridge"
    return "intimate verses opening into an emotional chorus"
