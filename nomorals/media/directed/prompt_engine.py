"""Prompt structuring engine — first-class system, not a helper.

Takes plain descriptions ("make the person raise two fingers, selfie style")
and expands them into model-optimized prompts:

- action sentence (single, leading)
- motion choreography broken into time-stamped beats
- camera specs (angle, movement, lens)
- lighting direction
- physical details (anti "plastic AI smoothness")
- style tags
- negative prompts
- temporal consistency anchors

Different structure per backend: LTX wants chronological paragraphs <200
words; Wan tolerates stylized language; CPU motion wants dense keywords;
SD1.5 image wants comma tags.

Two fill modes:
- rule-based structural assembly (offline, deterministic) — always works
- LLM expansion hook (the brain fills sections — best quality)

From EXPANSION_MINING.md §1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── curated vocabularies (mined, not guessed) ────────────────────────
NEGATIVE_VIDEO = (
    "blurry, low quality, distorted, watermark, text, logo, "
    "overexposed, underexposed, shaky, jittery, artifacts, "
    "morphing faces, unnatural motion, plastic skin, frozen frame"
)

NEGATIVE_IMAGE = (
    "blurry, low quality, distorted, watermark, text, logo, "
    "bad anatomy, extra fingers, deformed hands"
)

PHYSICAL_DETAIL_PACK = (
    "natural skin texture with visible pores, flyaway hair strands, "
    "fabric weave detail, subtle lens grain, realistic light falloff"
)

TEMPORAL_ANCHORS = (
    "high-fidelity temporal consistency, smooth motion transitions, "
    "consistent identity across frames"
)

CAMERA_MOVEMENTS = {
    "dolly": "slow dolly push-in",
    "tracking": "cinematic tracking shot",
    "handheld": "subtle handheld camera shake",
    "pan": "smooth pan",
    "tilt": "slow tilt up",
    "static": "locked-off static camera",
    "selfie": "slight handheld phone wobble, front-camera feel",
    "orbit": "slow orbital camera move",
    "zoom": "slow dolly zoom",
}

CAMERA_ANGLES = {
    "closeup": "close-up, tight framing on the face",
    "medium": "medium shot, waist up",
    "wide": "wide shot, full environment visible",
    "low": "low angle, looking up",
    "high": "high angle, looking down",
    "aerial": "aerial shot",
    "profile": "side profile shot",
    "over_shoulder": "over-the-shoulder shot",
}

LIGHTING = {
    "cinematic": "cinematic three-point lighting with soft key",
    "natural": "soft natural daylight",
    "golden": "warm golden-hour light, long shadows",
    "neon": "neon practical lights, colored rim light",
    "studio": "clean studio softbox lighting",
    "moody": "low-key moody lighting, deep shadows",
    "night": "dim blue-toned night lighting",
}

# vibe adjectives -> physical descriptors (anti vibe-coding)
VIBE_MAP = {
    "beautiful": "symmetrical features, clear skin",
    "epic": "dramatic scale, sweeping composition",
    "amazing": "",
    "cool": "",
    "awesome": "",
    "cinematic": "anamorphic lens feel, shallow depth of field, film grain",
}


@dataclass
class StructuredPrompt:
    """The structured prompt — inspectable, backend-formattable."""
    action_sentence: str = ""
    beats: list[str] = field(default_factory=list)   # time-stamped
    appearance: str = ""
    environment: str = ""
    camera_angle: str = ""
    camera_movement: str = ""
    lens: str = ""
    lighting: str = ""
    style_tags: list[str] = field(default_factory=list)
    physical_details: str = PHYSICAL_DETAIL_PACK
    negative: str = NEGATIVE_VIDEO
    consistency: str = TEMPORAL_ANCHORS
    backend: str = "ltx"

    # ── backend formatters ──────────────────────────────────────────
    def for_ltx(self) -> str:
        """Chronological paragraph, action-first, <200 words (LTX spec)."""
        parts = [self.action_sentence.rstrip(".") + "."]
        if self.beats:
            parts.append(" ".join(self.beats))
        if self.appearance:
            parts.append(self.appearance.rstrip(".") + ".")
        if self.environment:
            parts.append(self.environment.rstrip(".") + ".")
        cam = " ".join(x for x in
                       [self.camera_angle, self.camera_movement, self.lens]
                       if x)
        if cam:
            parts.append(f"The camera: {cam}.")
        if self.lighting:
            parts.append(self.lighting.rstrip(".") + ".")
        if self.style_tags:
            parts.append("Style: " + ", ".join(self.style_tags) + ".")
        parts.append(self.physical_details.rstrip(".") + ".")
        parts.append(self.consistency.rstrip(".") + ".")
        text = " ".join(parts)
        words = text.split()
        if len(words) > 200:
            text = " ".join(words[:200])
        return text

    def for_wan(self) -> str:
        """Wan: like LTX but tolerates a style-forward open."""
        base = self.for_ltx()
        if self.style_tags:
            return f"{', '.join(self.style_tags)}. {base}"
        return base

    def for_motion(self) -> str:
        """CPU motion fallback: dense keywords, no grammar to parse."""
        bits = [self.action_sentence]
        bits += self.beats
        if self.camera_movement:
            bits.append(self.camera_movement)
        bits += self.style_tags
        return ", ".join(b for b in bits if b)

    def for_image(self) -> str:
        """SD1.5-style comma tags."""
        bits = [self.action_sentence]
        if self.appearance:
            bits.append(self.appearance)
        if self.environment:
            bits.append(self.environment)
        bits += self.style_tags
        bits.append(self.physical_details)
        return ", ".join(b for b in bits if b)

    def render(self, backend: str | None = None) -> dict[str, str]:
        b = (backend or self.backend).lower()
        fmt = {"ltx": self.for_ltx, "wan": self.for_wan,
               "motion": self.for_motion, "image": self.for_image,
               "sd": self.for_image}.get(b, self.for_ltx)
        neg = self.negative if b in ("ltx", "wan") else NEGATIVE_IMAGE \
            if b in ("image", "sd") else ""
        return {"prompt": fmt(), "negative_prompt": neg, "backend": b}


# ── parsing (rule-based structural assembly) ─────────────────────────
_ACTION_VERBS = (
    "raise|raises|raising|wave|waves|waving|point|points|pointing|"
    "nod|nods|nodding|shake|shakes|shaking|dance|dances|dancing|"
    "walk|walks|walking|run|runs|running|smile|smiles|smiling|"
    "look|looks|looking|turn|turns|turning|sit|sits|sitting|"
    "jump|jumps|jumping|spin|spins|spinning"
)


def _extract_action(text: str) -> str:
    t = text.lower()
    m = re.search(rf"\b({_ACTION_VERBS})\b([^.,;]*)", t)
    if m:
        chunk = (m.group(1) + m.group(2)).strip()
        # de-vibe: drop empty adjectives
        for vibe, phys in VIBE_MAP.items():
            if vibe in chunk and not phys:
                chunk = chunk.replace(vibe, "").strip()
        return chunk[:160] or t[:160]
    return t[:160]


def _extract_camera(text: str) -> tuple[str, str, str]:
    t = text.lower()
    angle, movement, lens = "", "", ""
    for key, val in CAMERA_ANGLES.items():
        if key.replace("_", " ") in t or key in t:
            angle = val
            break
    for key, val in CAMERA_MOVEMENTS.items():
        if key in t:
            movement = val
            break
    if "selfie" in t and not movement:
        movement = CAMERA_MOVEMENTS["selfie"]
    m = re.search(r"(\d+)\s?mm", t)
    if m:
        lens = f"shot on {m.group(1)}mm lens"
    elif "wide angle" in t or "wide-angle" in t:
        lens = "ultra-wide lens"
    elif "telephoto" in t:
        lens = "telephoto lens, compressed background"
    return angle, movement, lens


def _extract_lighting(text: str) -> str:
    t = text.lower()
    for key, val in LIGHTING.items():
        if key in t:
            return val
    return ""


def _extract_style(text: str) -> list[str]:
    t = text.lower()
    tags = []
    for w in ("cinematic", "documentary", "anime", "photorealistic",
              "vintage", "cyberpunk", "noir", "selfie", "vlog",
              "music video", "commercial"):
        if w in t:
            tags.append(w)
    return tags


def _build_beats(action: str, duration_s: float = 5.0) -> list[str]:
    """Break the action into time-stamped beats (the choreography)."""
    thirds = [0.0, round(duration_s * 0.33, 1), round(duration_s * 0.66, 1)]
    return [
        f"At {thirds[0]}s: the motion begins, {action}.",
        f"At {thirds[1]}s: the action reaches its peak expression.",
        f"At {thirds[2]}s: the motion settles naturally to rest.",
    ]


def structure(plain: str, *, backend: str = "ltx",
              duration_s: float = 5.0) -> StructuredPrompt:
    """Plain description -> structured, backend-optimized prompt.

    Always works offline (rule-based). The brain can refine sections
    via refine() when an LLM is available.
    """
    text = (plain or "").strip()
    if not text:
        raise ValueError("empty prompt")
    action = _extract_action(text)
    # action sentence: capitalize, ensure it reads as a shot description
    sentence = action[0].upper() + action[1:] if action else text
    if not sentence.endswith("."):
        sentence += "."
    angle, movement, lens = _extract_camera(text)
    sp = StructuredPrompt(
        action_sentence=sentence,
        beats=_build_beats(action, duration_s),
        camera_angle=angle,
        camera_movement=movement,
        lens=lens,
        lighting=_extract_lighting(text),
        style_tags=_extract_style(text),
        backend=backend,
    )
    return sp


def refine(sp: StructuredPrompt, *, appearance: str = "",
           environment: str = "", beats: list[str] | None = None,
           lighting: str = "") -> StructuredPrompt:
    """LLM/brain refinement hook: fill or override sections."""
    if appearance:
        sp.appearance = appearance
    if environment:
        sp.environment = environment
    if beats is not None:
        sp.beats = beats
    if lighting:
        sp.lighting = lighting
    return sp
