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

#: HunyuanVideo — official 1.x default negative is the empty string
#: (guidance-distilled, true_cfg); community motion-focused fallback
#: from the prompting experiments (see module notes).
NEGATIVE_HUNYUAN = (
    "low quality, blurry, distorted, artifacts, watermark, text, logo, "
    "static camera, no motion, jerky motion, stuttering, flickering"
)

#: Mochi — T5 encoder, strong adherence; temporal artifacts are the
#: failure mode (mined from video prompt-engineering practice).
NEGATIVE_MOCHI = (
    "blurry, low quality, morphing face, extra limbs, extra fingers, "
    "flickering, warping background, distorted hands, text, watermark, "
    "sudden scene change, deformed body, disappearing objects"
)

#: CogVideoX — negative from the official diffusers CogVideoX docs.
NEGATIVE_COGVIDEOX = (
    "inconsistent motion, blurry motion, worse quality, degenerate outputs, "
    "deformed outputs, watermark, text, logo, flickering, distorted faces"
)

NEGATIVE_IMAGE = (
    "blurry, low quality, distorted, watermark, text, logo, "
    "bad anatomy, extra fingers, deformed hands"
)

#: per-backend negative defaults. SVD has no text encoder (image-to-video)
#: and the CPU motion path takes keywords only — both get "".
NEGATIVE_BACKENDS: dict[str, str] = {
    "ltx": NEGATIVE_VIDEO,
    "wan": NEGATIVE_VIDEO,
    "hunyuanvideo": NEGATIVE_HUNYUAN,
    "hunyuan": NEGATIVE_HUNYUAN,
    "mochi": NEGATIVE_MOCHI,
    "cogvideox": NEGATIVE_COGVIDEOX,
    "cogvideo": NEGATIVE_COGVIDEOX,
    "image": NEGATIVE_IMAGE,
    "sd": NEGATIVE_IMAGE,
    "svd": "",
    "motion": "",
}


def negative_prompt(backend: str, extra: str = "") -> str:
    """Per-backend negative prompt builder.

    Returns the backend's curated default negative; ``extra`` appends
    user/brain-supplied terms. SVD and the motion path take no negative
    (no text encoder / keyword-only) and return "".
    """
    base = NEGATIVE_BACKENDS.get((backend or "").lower(), NEGATIVE_VIDEO)
    extra = (extra or "").strip().rstrip(",")
    if extra:
        return f"{base}, {extra}" if base else extra
    return base

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

# ── shot-composition vocabulary ──────────────────────────────────
#: composition cues extracted from plain NL and woven into renders.
#: Style-agnostic framing primitives — no aesthetic baked in.
COMPOSITION = {
    "rule of thirds": "subject placed on a rule-of-thirds intersection",
    "leading lines": "strong leading lines draw the eye toward the subject",
    "depth layers": "layered foreground, midground and background for depth",
    "headroom": "generous headroom above the subject",
    "symmetry": "symmetrical composition with the subject centered",
    "negative space": "expansive negative space around the subject",
    "dutch angle": "dutch angle, tilted horizon for tension",
    "frame within a frame": "a natural frame-within-the-frame around the subject",
    "centered": "subject centered in frame",
}

#: vibe adjectives -> physical descriptors (anti vibe-coding)
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
    composition: list[str] = field(default_factory=list)  # COMPOSITION phrases
    physical_details: str = PHYSICAL_DETAIL_PACK
    #: explicit negative override — when empty, render() uses the
    #: per-backend default from negative_prompt(backend)
    negative: str = ""
    consistency: str = TEMPORAL_ANCHORS
    backend: str = "ltx"

    def _composition_sentence(self) -> str:
        if not self.composition:
            return ""
        return "Composition: " + "; ".join(self.composition) + "."

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

    # ── new backend renderers (Phase 8C) ───────────────────────────

    def for_hunyuanvideo(self) -> str:
        """HunyuanVideo: detailed 5-aspect description — subject, action,
        scene, camera, lighting — with explicit camera and timing
        language ("The camera pans…", "over N seconds")."""
        parts = [self.action_sentence.rstrip(".") + "."]
        if self.appearance:
            parts.append(self.appearance.rstrip(".") + ".")
        if self.environment:
            parts.append(self.environment.rstrip(".") + ".")
        cam = " ".join(x for x in
                       [self.camera_angle, self.camera_movement, self.lens]
                       if x)
        if cam:
            parts.append(f"The camera: {cam}.")
        if self.beats:
            parts.append(
                "The motion unfolds " + " ".join(self.beats).lower())
        if self.lighting:
            parts.append(self.lighting.rstrip(".") + ".")
        comp = self._composition_sentence()
        if comp:
            parts.append(comp)
        if self.style_tags:
            parts.append("Style: " + ", ".join(self.style_tags) + ".")
        parts.append(self.physical_details.rstrip(".") + ".")
        parts.append(self.consistency.rstrip(".") + ".")
        text = " ".join(parts)
        words = text.split()
        if len(words) > 300:
            text = " ".join(words[:300])
        return text

    def for_mochi(self) -> str:
        """Mochi: cinematic brief — style tags open, then a dense visual
        description with shot, lens and lighting named explicitly
        (Mochi's T5 encoder rewards named cinematography)."""
        parts = []
        if self.style_tags:
            parts.append(", ".join(self.style_tags).capitalize() + ".")
        desc = [self.action_sentence.rstrip(".") + "."]
        if self.appearance:
            desc.append(self.appearance.rstrip(".") + ".")
        if self.environment:
            desc.append(self.environment.rstrip(".") + ".")
        shot = " ".join(x for x in
                        [self.camera_angle, self.camera_movement, self.lens]
                        if x)
        if shot:
            desc.append(f"{shot.capitalize()}.")
        if self.lighting:
            desc.append(self.lighting.rstrip(".") + ".")
        comp = self._composition_sentence()
        if comp:
            desc.append(comp)
        parts.append(" ".join(desc))
        if self.beats:
            parts.append(" ".join(self.beats))
        parts.append(self.physical_details.rstrip(".") + ".")
        parts.append(self.consistency.rstrip(".") + ".")
        text = " ".join(parts)
        words = text.split()
        if len(words) > 250:
            text = " ".join(words[:250])
        return text

    def for_cogvideox(self) -> str:
        """CogVideoX: one long descriptive paragraph (the docs' own
        examples are detailed prose), action-first, camera last."""
        parts = [self.action_sentence.rstrip(".") + "."]
        if self.appearance:
            parts.append(self.appearance.rstrip(".") + ".")
        if self.environment:
            parts.append(self.environment.rstrip(".") + ".")
        if self.beats:
            parts.append(" ".join(self.beats))
        if self.lighting:
            parts.append(self.lighting.rstrip(".") + ".")
        comp = self._composition_sentence()
        if comp:
            parts.append(comp)
        cam = " ".join(x for x in
                       [self.camera_angle, self.camera_movement, self.lens]
                       if x)
        if cam:
            parts.append(f"Camera work: {cam}.")
        if self.style_tags:
            parts.append(", ".join(self.style_tags) + ".")
        parts.append(self.physical_details.rstrip(".") + ".")
        parts.append(self.consistency.rstrip(".") + ".")
        return " ".join(parts)

    def for_svd(self) -> str:
        """Stable Video Diffusion: honest I2V — SVD has no text encoder,
        so this renders a MOTION DIRECTIVE (what moves, how the camera
        moves), not a scene description. Pair with a start frame."""
        bits = [self.action_sentence.rstrip(".") + "."]
        bits += self.beats
        if self.camera_movement:
            bits.append(f"Camera: {self.camera_movement}.")
        if self.camera_angle:
            bits.append(f"Framing: {self.camera_angle}.")
        comp = self._composition_sentence()
        if comp:
            bits.append(comp)
        return " ".join(b for b in bits if b)

    #: alias → canonical backend name (render() always reports canonical)
    BACKEND_ALIASES = {"hunyuan": "hunyuanvideo",
                       "cogvideo": "cogvideox",
                       "sd": "image"}

    def render(self, backend: str | None = None) -> dict[str, str]:
        b = (backend or self.backend).lower()
        b = self.BACKEND_ALIASES.get(b, b)
        fmt = {"ltx": self.for_ltx, "wan": self.for_wan,
               "hunyuanvideo": self.for_hunyuanvideo,
               "mochi": self.for_mochi,
               "cogvideox": self.for_cogvideox,
               "svd": self.for_svd,
               "motion": self.for_motion,
               "image": self.for_image}.get(b, self.for_ltx)
        neg = self.negative or negative_prompt(b)
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
    # camera language first: it handles compounds the keyword table can't
    # ("dolly in slowly, then orbit left" -> "slow dolly in, then orbit left")
    from .camera import parse_camera_language
    prog = parse_camera_language(t)
    if not prog.empty:
        # Bare keywords keep their tuned legacy phrases ("dolly" ->
        # "slow dolly push-in"); the program wins when it adds real
        # information: compounds, explicit directions, or speeds.
        use_program = True
        if len(prog.moves) == 1:
            mv = prog.moves[0]
            if (mv.verb in CAMERA_MOVEMENTS and mv.direction in ("", "in")
                    and mv.speed == "normal"):
                use_program = False
        if use_program:
            movement = prog.describe()
    if not movement:
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


def _extract_composition(text: str) -> list[str]:
    """Shot-composition vocabulary from NL: rule of thirds, leading
    lines, depth layers, headroom, symmetry, negative space,
    dutch angle, frame-within-a-frame, centered."""
    t = text.lower()
    found = []
    for key, phrase in COMPOSITION.items():
        if key in t and phrase not in found:
            found.append(phrase)
    return found


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
        composition=_extract_composition(text),
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
