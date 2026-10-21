"""Shot design — lighting + camera as composable primitive data.

The style-agnostic law (owner, 2026-10-09) applied to directing:
this module owns *primitives* — shot sizes, angles, movements, lens
feels, motivated light sources, fill ratios, exposure intents — and a
designer that composes them from the shot's dramatic job. No look is
baked in. A *style pack* is data (a dict of preferences over the same
primitives) that biases the choices the rules leave open; packs
compose primitives, they never replace the grammar.

Two design paths, both real:

- **Rule designer** — cinematography heuristics keyed to the shot's
  structural job (establishing / dialogue / reaction / action /
  detail / transition) and the scene's time of day. Deterministic.
- **Brain refinement** — when a brain is offered, it picks per-shot
  values *from the vocabularies* with a reason; every pick is
  validated against the vocabulary and an out-of-vocab pick falls
  back to the rule value. The brain directs within the grammar, it
  does not invent a new one mid-production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "SHOT_SIZES", "ANGLES", "MOVEMENTS", "LENS_FEELS", "FOCUS_INTENTS",
    "KEY_SOURCES", "KEY_QUALITIES", "KEY_DIRECTIONS", "FILL_RATIOS",
    "EXPOSURE_INTENTS", "COLOR_TEMPS",
    "LightingSpec", "CameraSpec", "DesignSpec",
    "STYLE_PACKS", "get_style_pack",
    "design_shot", "refine_design",
]

# ---------------------------------------------------------------------------
# Primitive vocabularies (the grammar — closed sets, validated everywhere)
# ---------------------------------------------------------------------------

SHOT_SIZES = ("extreme wide", "wide", "medium wide", "medium",
              "medium close-up", "close-up", "extreme close-up")
ANGLES = ("eye-level", "low", "high", "dutch", "overhead",
          "over-the-shoulder", "point-of-view")
MOVEMENTS = ("static", "slow push-in", "pull-back", "pan", "tilt",
             "tracking", "handheld", "gimbal glide", "crane")
LENS_FEELS = ("wide", "normal", "telephoto", "anamorphic")
FOCUS_INTENTS = ("deep focus", "shallow depth of field",
                 "rack focus to subject", "soft background")

#: motivated key sources — light that exists inside the story world
KEY_SOURCES = ("daylight through window", "open daylight", "overcast sky",
               "practical lamp", "overhead fluorescent", "neon signage",
               "streetlight", "candlelight", "screen glow", "moonlight",
               "firelight", "studio softbox")
KEY_QUALITIES = ("soft", "hard", "diffused daylight")
KEY_DIRECTIONS = ("front", "side", "three-quarter", "back", "top")
FILL_RATIOS = ("1:1 flat", "1:2 gentle", "1:4 modelling", "1:8 low-key")
EXPOSURE_INTENTS = ("bright and open", "balanced natural",
                    "moody underexposed", "high-contrast chiaroscuro",
                    "silhouette against source")
COLOR_TEMPS = ("warm golden", "neutral daylight", "cool blue",
               "mixed practical warmth", "monochrome-leaning")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class LightingSpec:
    key_source: str = "daylight through window"
    key_quality: str = "soft"
    key_direction: str = "three-quarter"
    fill_ratio: str = "1:2 gentle"
    rim: str = "none"                # "none" | "subtle edge" | "strong halo"
    practicals: str = ""             # visible in-world lights, "" = none
    exposure: str = "balanced natural"
    color_temp: str = "neutral daylight"

    def fragments(self) -> list[str]:
        parts = [f"key light: {self.key_quality} {self.key_source} "
                 f"from {self.key_direction}",
                 f"fill at {self.fill_ratio}"]
        if self.rim != "none":
            parts.append(f"rim light: {self.rim}")
        if self.practicals:
            parts.append(f"practicals in frame: {self.practicals}")
        parts.append(f"exposure {self.exposure}")
        parts.append(f"color temperature {self.color_temp}")
        return parts

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LightingSpec":
        return cls(**{k: str(d.get(k, getattr(cls(), k)))
                      for k in vars(cls())})


@dataclass
class CameraSpec:
    shot_size: str = "medium"
    angle: str = "eye-level"
    movement: str = "static"
    lens: str = "normal"
    focus: str = "shallow depth of field"

    def fragments(self) -> list[str]:
        return [f"{self.shot_size} shot", f"{self.angle} angle",
                f"camera {self.movement}", f"{self.lens} lens feel",
                self.focus]

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraSpec":
        return cls(**{k: str(d.get(k, getattr(cls(), k)))
                      for k in vars(cls())})


@dataclass
class DesignSpec:
    lighting: LightingSpec = field(default_factory=LightingSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    mood: str = ""
    rationale: str = ""

    def prompt_fragments(self) -> list[str]:
        parts = self.camera.fragments() + self.lighting.fragments()
        if self.mood:
            parts.append(f"mood: {self.mood}")
        return parts

    def to_dict(self) -> dict[str, Any]:
        return {"lighting": self.lighting.to_dict(),
                "camera": self.camera.to_dict(),
                "mood": self.mood, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DesignSpec":
        return cls(lighting=LightingSpec.from_dict(d.get("lighting", {})),
                   camera=CameraSpec.from_dict(d.get("camera", {})),
                   mood=str(d.get("mood", "")),
                   rationale=str(d.get("rationale", "")))


# ---------------------------------------------------------------------------
# Style packs — data over the primitives, never a baked look
# ---------------------------------------------------------------------------

STYLE_PACKS: dict[str, dict[str, Any]] = {
    "neutral": {
        "label": "Neutral — the scene's own logic leads",
        "prefer": {},
    },
    "naturalist": {
        "label": "Naturalist — motivated light, invisible camera",
        "prefer": {
            "key_quality": "soft", "fill_ratio": "1:2 gentle",
            "movement": "static", "angle": "eye-level",
            "lens": "normal", "focus": "deep focus",
            "exposure": "balanced natural",
        },
    },
    "noir": {
        "label": "Noir — low-key, hard edges, low angles",
        "prefer": {
            "key_quality": "hard", "fill_ratio": "1:8 low-key",
            "rim": "strong halo", "angle": "low",
            "exposure": "high-contrast chiaroscuro",
            "color_temp": "monochrome-leaning",
        },
    },
    "neon": {
        "label": "Neon — saturated practicals, glossy night",
        "prefer": {
            "key_source_night": "neon signage", "rim": "strong halo",
            "color_temp": "mixed practical warmth",
            "exposure": "moody underexposed", "lens": "anamorphic",
        },
    },
    "golden": {
        "label": "Golden — warm backlight, slow moves",
        "prefer": {
            "color_temp": "warm golden", "rim": "subtle edge",
            "movement": "slow push-in", "key_direction": "back",
            "exposure": "bright and open",
        },
    },
}


def get_style_pack(name: str) -> dict[str, Any]:
    """A style pack by name; unknown names raise (never silently neutral)."""
    key = (name or "neutral").strip().lower()
    if key not in STYLE_PACKS:
        known = ", ".join(sorted(STYLE_PACKS))
        raise ValueError(f"unknown style pack {name!r} — packs: {known}")
    return {"name": key, **STYLE_PACKS[key]}


# ---------------------------------------------------------------------------
# The rule designer
# ---------------------------------------------------------------------------

def _motivated_key(time_of_day: str, interior: str) -> tuple[str, str]:
    """(key_source, color_temp) the story world itself provides."""
    tod = (time_of_day or "").upper()
    if "NIGHT" in tod:
        return ("practical lamp", "mixed practical warmth") \
            if interior in ("INT", "INT/EXT") else \
            ("streetlight", "cool blue")
    if any(w in tod for w in ("DUSK", "DAWN", "MAGIC", "SUNSET", "EVENING")):
        return "open daylight", "warm golden"
    if "DAY" in tod or "MORNING" in tod or "AFTERNOON" in tod:
        return (("daylight through window", "neutral daylight")
                if interior in ("INT", "INT/EXT")
                else ("open daylight", "neutral daylight"))
    return ("daylight through window", "neutral daylight") \
        if interior in ("INT", "INT/EXT") else \
        ("overcast sky", "neutral daylight")


def design_shot(shot_kind: str, *, time_of_day: str = "",
                interior: str = "", style_pack: str = "neutral",
                mood_hint: str = "") -> DesignSpec:
    """Compose a design spec for one shot from the primitives.

    ``shot_kind`` is the shot's structural job (establishing | action |
    dialogue | reaction | detail | transition). Choices start from
    cinematography rules for that job, lit by the scene's motivated
    source; the style pack then biases only the choices the rules
    leave open.
    """
    pack = get_style_pack(style_pack)
    pref: dict[str, Any] = dict(pack.get("prefer") or {})
    kind = (shot_kind or "action").lower()
    key_source, color_temp = _motivated_key(time_of_day, interior)
    if (time_of_day or "").upper().find("NIGHT") >= 0 \
            and "key_source_night" in pref:
        key_source = pref["key_source_night"]

    cam = CameraSpec()
    light = LightingSpec(key_source=key_source, color_temp=color_temp)
    mood = mood_hint

    if kind == "establishing":
        cam.shot_size, cam.angle = "wide", "eye-level"
        cam.movement, cam.focus = "slow push-in", "deep focus"
        light.fill_ratio = "1:2 gentle"
        rationale = "open the space: geography before people"
    elif kind == "dialogue":
        cam.shot_size = "medium close-up"
        cam.movement = "slow push-in"
        light.key_direction = "three-quarter"
        rationale = "performance carries it; gentle push as the line lands"
    elif kind == "reaction":
        cam.shot_size, cam.movement = "close-up", "static"
        light.key_quality = "soft"
        rationale = "hold on the face; the reaction IS the beat"
    elif kind == "action":
        cam.shot_size = "medium wide"
        cam.movement = "tracking"
        cam.focus = "deep focus"
        rationale = "stay wide enough to read the movement, move with it"
    elif kind == "detail":
        cam.shot_size, cam.movement = "extreme close-up", "static"
        light.key_direction = "side"
        rationale = "isolate the object the story is pointing at"
    elif kind == "transition":
        cam.shot_size, cam.movement = "wide", "static"
        light.exposure = "balanced natural"
        rationale = "a breath between scenes"
    else:
        rationale = f"general coverage for a {kind} beat"

    # style pack biases the open choices (data over primitives)
    if "key_quality" in pref:
        light.key_quality = pref["key_quality"]
    if "fill_ratio" in pref:
        light.fill_ratio = pref["fill_ratio"]
    if "rim" in pref:
        light.rim = pref["rim"]
    if "exposure" in pref:
        light.exposure = pref["exposure"]
    if "color_temp" in pref:
        light.color_temp = pref["color_temp"]
    if "key_direction" in pref:
        light.key_direction = pref["key_direction"]
    if "angle" in pref and kind in ("dialogue", "action", "establishing"):
        cam.angle = pref["angle"]
    if "movement" in pref and kind in ("dialogue", "establishing"):
        cam.movement = pref["movement"]
    if "lens" in pref:
        cam.lens = pref["lens"]
    if "focus" in pref and kind != "detail":
        cam.focus = pref["focus"]
    if not mood:
        mood = {"noir": "tension under the surface",
                "neon": "electric, restless",
                "golden": "warmth with an edge of loss",
                "naturalist": "unforced, lived-in"}.get(pack["name"], "")

    return DesignSpec(lighting=light, camera=cam, mood=mood,
                      rationale=rationale)


# ---------------------------------------------------------------------------
# Brain refinement — picks validated against the vocabularies
# ---------------------------------------------------------------------------

_CAMERA_VOCAB = {"shot_size": SHOT_SIZES, "angle": ANGLES,
                 "movement": MOVEMENTS, "lens": LENS_FEELS,
                 "focus": FOCUS_INTENTS}
_LIGHT_VOCAB = {"key_source": KEY_SOURCES, "key_quality": KEY_QUALITIES,
                "key_direction": KEY_DIRECTIONS, "fill_ratio": FILL_RATIOS,
                "exposure": EXPOSURE_INTENTS, "color_temp": COLOR_TEMPS}


def refine_design(spec: DesignSpec, shot_description: str, *,
                  brain: Any) -> DesignSpec:
    """Let the brain re-pick design values — from the vocabularies only.

    Returns a new spec. Any field the brain answers out-of-vocabulary
    (or doesn't answer) keeps the rule value; the brain's stated
    reason is kept in ``rationale`` when it changes anything. A brain
    failure returns the rule spec unchanged — refinement is an
    upgrade, never a dependency.
    """
    from ...llm.base import Message

    prompt = (
        "You are a cinematographer refining one shot's design. Choose "
        "values ONLY from the vocabularies given. Reply with JSON: "
        '{"camera": {"shot_size": ..., "angle": ..., "movement": ..., '
        '"lens": ..., "focus": ...}, "lighting": {"key_source": ..., '
        '"key_quality": ..., "key_direction": ..., "fill_ratio": ..., '
        '"exposure": ..., "color_temp": ...}, "mood": "...", '
        '"reason": "..."}.\n'
        f"Shot: {shot_description}\n"
        f"Camera vocab: { {k: list(v) for k, v in _CAMERA_VOCAB.items()} }\n"
        f"Lighting vocab: { {k: list(v) for k, v in _LIGHT_VOCAB.items()} }\n"
        f"Rule design so far: camera={spec.camera.to_dict()} "
        f"lighting={spec.lighting.to_dict()}")
    try:
        data, resp = brain.chat_json([Message.user(prompt)],
                                     task_kind="design")
    except Exception as exc:  # noqa: BLE001 - refinement is optional
        _log.debug("director: design refinement failed: %s", exc)
        return spec
    if not isinstance(data, dict):
        _log.debug("director: design refinement unusable (%s)",
                   getattr(resp, "error", ""))
        return spec
    cam = CameraSpec.from_dict(spec.camera.to_dict())
    light = LightingSpec.from_dict(spec.lighting.to_dict())
    changed = False
    for field_name, vocab in _CAMERA_VOCAB.items():
        val = (data.get("camera") or {}).get(field_name)
        if val in vocab:
            setattr(cam, field_name, val)
            changed = True
    for field_name, vocab in _LIGHT_VOCAB.items():
        val = (data.get("lighting") or {}).get(field_name)
        if val in vocab:
            setattr(light, field_name, val)
            changed = True
    if not changed:
        return spec
    reason = str(data.get("reason") or "").strip()
    return DesignSpec(
        lighting=light, camera=cam,
        mood=str(data.get("mood") or spec.mood),
        rationale=(f"{spec.rationale} · refined: {reason}"
                   if reason else spec.rationale))
