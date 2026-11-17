"""The soundscape catalogue — generators and presets as data.

One uniform interface, three kinds:

* ``noise`` — white / pink / brown / grey / blue. The ten sliders
  are a 10-band EQ over the octave centres.
* ``tones`` — binaural beats and drones. The ten sliders are the
  same 10-band EQ over the tone's spectrum.
* ``scene`` — rain, forest, ocean, fire, cafe, night. Each slider
  is one synthesised stem of the scene (thunder, birds, …).

Presets are named mixes: one word — "sleep", "focus", "rainy night"
— renders a whole soundscape. The owner grows the catalogue with
``save``; nothing here is invented, and an unknown name is refused
with the list of real ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .engine import BAND_FREQS

__all__ = [
    "GENERATORS",
    "PRESETS",
    "GeneratorSpec",
    "parse_soundscape_query",
    "resolve_generator",
]

_BAND_LABELS: tuple[str, ...] = tuple(
    f"{f:g} Hz" if f >= 100 else f"{f:g} Hz" for f in BAND_FREQS)


@dataclass(frozen=True)
class GeneratorSpec:
    """One generator: what it is, what its ten sliders mean."""

    id: str
    name: str
    kind: str  # noise | tones | scene
    sliders: tuple[str, ...]
    defaults: tuple[float, ...] = ()
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.defaults:
            object.__setattr__(self, "defaults",
                               tuple(0.0 for _ in self.sliders))
        if len(self.defaults) != len(self.sliders):
            raise ValueError(
                f"generator {self.id}: {len(self.sliders)} sliders but "
                f"{len(self.defaults)} defaults")


GENERATORS: dict[str, GeneratorSpec] = {
    "white": GeneratorSpec(
        id="white", name="White Noise", kind="noise", sliders=_BAND_LABELS,
        description="Flat spectrum — the masking workhorse."),
    "pink": GeneratorSpec(
        id="pink", name="Pink Noise", kind="noise", sliders=_BAND_LABELS,
        description="−3 dB per octave — the balanced, sleep-friendly one."),
    "brown": GeneratorSpec(
        id="brown", name="Brown Noise", kind="noise", sliders=_BAND_LABELS,
        description="−6 dB per octave — deep, low, the focus favourite."),
    "grey": GeneratorSpec(
        id="grey", name="Grey Noise", kind="noise", sliders=_BAND_LABELS,
        description="Shaped to equal loudness — sounds flat to a human ear."),
    "blue": GeneratorSpec(
        id="blue", name="Blue Noise", kind="noise", sliders=_BAND_LABELS,
        description="+3 dB per octave — bright, airy, hiss-like."),
    "binaural": GeneratorSpec(
        id="binaural", name="Binaural Beats", kind="tones",
        sliders=_BAND_LABELS,
        description="Two tones, one per ear — the beat lives in the "
                    "difference. Carrier 200 Hz, beat 10 Hz by default.",
        params={"carrier": 200.0, "beat": 10.0}),
    "drone": GeneratorSpec(
        id="drone", name="Drone", kind="tones", sliders=_BAND_LABELS,
        description="A root with its harmonic stack, slightly detuned "
                    "between the ears. Root 110 Hz by default.",
        params={"root": 110.0}),
    "rain": GeneratorSpec(
        id="rain", name="Rain", kind="scene",
        sliders=("rain", "drizzle", "thunder", "wind", "leaves",
                 "stream", "birds"),
        defaults=(0.0, -6.0, -3.0, -9.0, -12.0, -12.0, -30.0),
        description="Steady rain with distant thunder and a wet wind."),
    "forest": GeneratorSpec(
        id="forest", name="Forest", kind="scene",
        sliders=("wind", "birds", "leaves", "stream", "crickets",
                 "rain", "thunder", "owl"),
        defaults=(-6.0, 0.0, -3.0, -6.0, -12.0, -30.0, -30.0, -12.0),
        description="Morning forest — birds over wind and leaves."),
    "ocean": GeneratorSpec(
        id="ocean", name="Ocean", kind="scene",
        sliders=("ocean", "wind", "gulls", "rain", "thunder",
                 "stream", "birds", "fire"),
        defaults=(0.0, -6.0, -9.0, -30.0, -30.0, -30.0, -12.0, -30.0),
        description="Slow swell and surf, gulls far off."),
    "fire": GeneratorSpec(
        id="fire", name="Fireplace", kind="scene",
        sliders=("fire", "wind", "rain", "thunder", "crickets",
                 "owl", "stream", "birds"),
        defaults=(0.0, -12.0, -30.0, -30.0, -12.0, -18.0, -30.0, -30.0),
        description="A fire bed with crackle, night outside."),
    "cafe": GeneratorSpec(
        id="cafe", name="Cafe", kind="scene",
        sliders=("murmur", "clinks", "wind", "rain", "stream",
                 "birds", "fire", "ocean"),
        defaults=(0.0, -6.0, -18.0, -30.0, -30.0, -30.0, -30.0, -30.0),
        description="Quiet cafe murmur with cup clinks."),
    "night": GeneratorSpec(
        id="night", name="Night", kind="scene",
        sliders=("crickets", "wind", "owl", "stream", "rain",
                 "thunder", "fire", "birds"),
        defaults=(0.0, -9.0, -6.0, -12.0, -30.0, -30.0, -18.0, -30.0),
        description="Crickets and an owl over a still night."),
}

#: Named mixes — one word renders a whole soundscape.
PRESETS: dict[str, dict[str, Any]] = {
    "focus": {"generator": "brown",
              "sliders": (0.0, 0.0, 0.0, 0.0, 0.0, -3.0, -6.0, -9.0,
                          -12.0, -12.0),
              "animate": False, "duration_s": 60.0},
    "sleep": {"generator": "pink",
              "sliders": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, -6.0,
                          -9.0, -12.0),
              "animate": True, "duration_s": 120.0},
    "deep sleep": {"generator": "brown",
                   "sliders": (3.0, 3.0, 0.0, 0.0, -3.0, -6.0, -9.0,
                               -12.0, -12.0, -12.0),
                   "animate": True, "duration_s": 120.0},
    "rainy night": {"generator": "rain",
                    "sliders": (0.0, -6.0, 0.0, -9.0, -12.0, -12.0, -30.0),
                    "animate": True, "duration_s": 120.0},
    "forest morning": {"generator": "forest",
                       "sliders": (-6.0, 3.0, -3.0, -6.0, -12.0, -30.0,
                                   -30.0, -12.0),
                       "animate": True, "duration_s": 120.0},
    "cafe work": {"generator": "cafe",
                  "sliders": (0.0, -6.0, -18.0, -30.0, -30.0, -30.0,
                              -30.0, -30.0),
                  "animate": False, "duration_s": 60.0},
    "ocean night": {"generator": "ocean",
                    "sliders": (0.0, -6.0, -9.0, -30.0, -30.0, -30.0,
                                -12.0, -30.0),
                    "animate": True, "duration_s": 120.0},
    "meditation": {"generator": "drone",
                   "sliders": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, -9.0,
                               -12.0, -12.0),
                   "animate": False, "duration_s": 120.0},
}

_ALIASES = {
    "white noise": "white",
    "pink noise": "pink",
    "brown noise": "brown",
    "grey noise": "grey",
    "gray noise": "grey",
    "blue noise": "blue",
    "binaural beats": "binaural",
    "rain sounds": "rain",
    "rain sound": "rain",
    "forest sounds": "forest",
    "ocean sounds": "ocean",
    "sea sounds": "ocean",
    "fire sounds": "fire",
    "fireplace": "fire",
    "cafe sounds": "cafe",
    "coffee shop": "cafe",
    "night sounds": "night",
}


def resolve_generator(name: str) -> GeneratorSpec:
    """Resolve a generator id, alias or preset name to a spec.

    Presets resolve to their generator. Raises ValueError naming the
    real generators when nothing matches — never a guess.
    """
    key = (name or "").strip().lower()
    if key in PRESETS:
        return GENERATORS[PRESETS[key]["generator"]]
    key = _ALIASES.get(key, key)
    if key in GENERATORS:
        return GENERATORS[key]
    known = ", ".join(sorted(GENERATORS))
    raise ValueError(f"unknown soundscape {name!r} — try: {known}")


def _parse_duration(text: str) -> float | None:
    """Parse '40m', '90s', '2h' or 'fade out in 40 minutes'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)\b",
                      text.lower())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("h"):
        return value * 3600.0
    if unit.startswith("m"):
        return value * 60.0
    return value


def parse_soundscape_query(text: str) -> dict[str, Any]:
    """Parse a natural soundscape request into a render spec.

    "rain sounds, more thunder, less birds" → rain with thunder +3 dB
    and birds −3 dB. "brown noise for sleep, fade out in 40 minutes"
    → brown, 40-minute timer. A preset name ("sleep", "focus") renders
    the preset. Unknown words are ignored; an unknown generator is
    refused by resolve_generator, never guessed.
    """
    raw = (text or "").strip()
    lowered = raw.lower()
    result: dict[str, Any] = {"generator": None, "preset": None,
                              "adjust": {}, "timer_min": None,
                              "duration_s": None, "animate": None}
    for name in sorted(PRESETS, key=len, reverse=True):
        if name in lowered:
            result["preset"] = name
            result["generator"] = PRESETS[name]["generator"]
            break
    if result["generator"] is None:
        for alias in sorted(_ALIASES, key=len, reverse=True):
            if alias in lowered:
                result["generator"] = _ALIASES[alias]
                break
    if result["generator"] is None:
        for gid in sorted(GENERATORS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(gid)}\b", lowered):
                result["generator"] = gid
                break
    # "more thunder" / "less birds" — ±3 dB on the named stem or band.
    for direction, delta in (("more", 3.0), ("less", -3.0)):
        for match in re.finditer(rf"\b{direction}\s+([a-z]+)\b", lowered):
            word = match.group(1)
            spec = None
            if result["generator"]:
                try:
                    spec = resolve_generator(result["generator"])
                except ValueError:
                    spec = None
            if spec is not None:
                for slider in spec.sliders:
                    if slider.lower().startswith(word):
                        result["adjust"][slider] = (
                            result["adjust"].get(slider, 0.0) + delta)
                        break
                else:
                    # A band name ("more bass") maps to the low bands.
                    if word in ("bass", "low", "lows"):
                        for band in ("31.5 Hz", "63 Hz", "125 Hz"):
                            result["adjust"][band] = (
                                result["adjust"].get(band, 0.0) + delta)
                    elif word in ("treble", "high", "highs"):
                        for band in ("8000 Hz", "16000 Hz"):
                            result["adjust"][band] = (
                                result["adjust"].get(band, 0.0) + delta)
    if "sleep" in lowered or "fade" in lowered or "timer" in lowered:
        seconds = _parse_duration(lowered)
        if seconds:
            result["timer_min"] = seconds / 60.0
    if "animate" in lowered:
        result["animate"] = True
    if "still" in lowered or "static" in lowered:
        result["animate"] = False
    return result
