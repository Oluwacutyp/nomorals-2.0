"""Natural-language direction parser — ElevenLabs v4 style.

Takes free-form direction like ``[said angrily in British accent]`` or
``[whispers fearfully]`` and produces structured direction that any
backend can consume — not just tag lookup, but intent parsing.

This is the v4 shift: from rigid markup to prose direction. The parser
handles:
- Emotion + delivery: "[said angrily]", "[whispers fearfully]"
- Accent: "[in British accent]", "[said angrily in French accent]"
- Combined: "[shouts excitedly in Nigerian English]"
- Ambient: "[light rain]", "[phone buzzing]", "[door slams]"
"""

import re
from dataclasses import dataclass, field


@dataclass
class Direction:
    """Structured performance direction from a free-form tag."""
    emotion: str = ""           # happy, angry, fearful, ...
    delivery: str = ""          # whispers, shouts, sings, ...
    accent: str = ""            # british, nigerian, french, ...
    ambient: str = ""           # light rain, phone buzzing, ...
    pace: str = ""              # slow, fast, rushed, drawn out
    intensity: int = 3          # 1–5 from adverbs ("very angry" → 4)
    raw: str = ""               # original tag text

    def is_empty(self) -> bool:
        return not (self.emotion or self.delivery or self.accent
                    or self.ambient or self.pace)

    def to_dsp_params(self) -> dict:
        """This direction → DSP shaping params (method form)."""
        return direction_to_dsp_params(self)

    def describe(self) -> str:
        """Pretty one-liner: 'angry(4) · shouts · british · fast'."""
        parts = []
        if self.emotion:
            parts.append(f"{self.emotion}({self.intensity})")
        if self.delivery:
            parts.append(self.delivery)
        if self.accent:
            parts.append(self.accent)
        if self.pace:
            parts.append(self.pace)
        if self.ambient:
            parts.append(f"~{self.ambient}")
        return " · ".join(parts) if parts else "(no direction)"


# Delivery verbs that change HOW something is said
_DELIVERY_VERBS = {
    "whisper", "whispers", "whispering", "whispered",
    "shout", "shouts", "shouting", "shouted",
    "scream", "screams", "screaming",
    "sing", "sings", "singing", "sang",
    "mutter", "mutters", "muttering",
    "chant", "chants", "chanting",
    "cry", "cries", "crying", "weeps", "weeping",
    "laugh", "laughs", "laughing",
    "sigh", "sighs", "sighing",
    "stammer", "stammers", "stammering",
    "gasp", "gasps", "gasping",
    # sweep: performance deliveries
    "rap", "raps", "rapping",
    "croon", "croons", "crooning",
    "preach", "preaches", "preaching",
    "narrate", "narrates", "narrating",
    "announce", "announces", "announcing",
    "plead", "pleads", "pleading",
    "taunt", "taunts", "taunting",
    "tease", "teases", "teasing",
    "bark", "barks", "barking",
    "drawl", "drawls", "drawling",
    "stutter", "stutters", "stuttering",
    "yell", "yells", "yelling",
    "hiss", "hisses", "hissing",
    "purr", "purrs", "purring",
    "growl", "growls", "growling",
}

# Emotion adjectives
_EMOTION_WORDS = {
    "angry", "angrily", "furious", "happy", "happily", "joyful",
    "sad", "sadly", "sorrowful", "excited", "excitedly",
    "nervous", "nervously", "anxious", "anxiously",
    "scared", "fearful", "fearfully", "terrified",
    "proud", "proudly", "sarcastic", "sarcastically",
    "curious", "curiously", "surprised", "thoughtful",
    "confident", "confidently", "annoyed", "tender",
    "playful", "playfully", "bored", "determined",
    "guilty", "shy", "shyly", "calm", "calmly", "tired",
    "reassuring", "empathetic", "desperate", "hopeful",
    "mischievous", "mischievously", "smug", "wistful",
    "bitter", "triumphant", "panicked", "disgusted",
    "amused", "relieved", "nostalgic", "ecstatic",
    # sweep: wider affect (director's canonical set + Hume-style states)
    "jealous", "jealously", "contemptuous", "hysterical",
    "resigned", "deadpan", "flustered", "lonely",
    "awkward", "eager", "hesitant", "skeptical", "skeptically",
    "reflective", "envious", "remorseful", "suspicious",
    "appalled", "defiant", "defiantly", "melancholy", "melancholic",
    "exhausted", "restless", "serene", "serenely", "passionate",
    "passionately", "vengeful", "mournful", "mournfully",
    "jubilant", "giddy", "giddily", "sheepish", "sheepishly",
    "indignant", "indignantly", "bewildered", "drowsy", "manic",
}

# Intensity adverbs → 1–5 level multiplier on the parsed emotion
_INTENSITY_WORDS = {
    "slightly": 1, "a bit": 1, "a little": 1, "somewhat": 2,
    "quite": 3, "fairly": 3, "pretty": 3,
    "very": 4, "really": 4, "so": 4,
    "extremely": 5, "incredibly": 5, "utterly": 5,
}

# Pace descriptors
_PACE_WORDS = {
    "slow": "slow", "slowly": "slow",
    "fast": "fast", "quickly": "fast", "rapidly": "fast",
    "rushed": "rushed", "hurried": "rushed",
    "drawn out": "drawn out", "drawn-out": "drawn out",
    "hesitant": "hesitant", "hesitantly": "hesitant",
}

# Ambient/sound-effect patterns (not voice direction)
_AMBIENT_PATTERNS = [
    "rain", "thunder", "wind", "buzzing", "phone",
    "door", "slam", "gunshot", "explosion", "applause",
    "clapping", "crowd", "traffic", "birds", "ocean",
    "waves", "fire", "alarm", "siren",
]

# Accent patterns
_ACCENT_RE = re.compile(
    r"\bin\s+(?:a\s+|an\s+|the\s+)?"
    r"(british|american|french|nigerian|yoruba|igbo|hausa|"
    r"australian|indian|russian|german|spanish|italian|"
    r"scottish|irish|south african|jamaican|canadian|"
    r"pidgin|caribbean|west african|swahili|arabic|chinese|"
    r"japanese|korean|dutch|portuguese|welsh)\s+accent\b",
    re.IGNORECASE,
)


def _normalize_emotion(word: str) -> str:
    """Normalize adverb → adjective: angrily→angry, happily→happy."""
    if word.endswith("ily") and word[:-3] + "y" in _EMOTION_WORDS:
        return word[:-3] + "y"
    if word.endswith("ly") and word[:-2] in _EMOTION_WORDS:
        return word[:-2]
    return word


def parse_direction(tag_text: str) -> Direction:
    """Parse a free-form direction tag into structured Direction.

    Examples:
        "[said angrily in British accent]" ->
            Direction(emotion="angry", accent="british")
        "[whispers fearfully]" ->
            Direction(delivery="whispers", emotion="fearful")
        "[light rain]" ->
            Direction(ambient="light rain")
    """
    text = tag_text.strip().strip("[]").strip()
    d = Direction(raw=text)
    lower = text.lower()

    # Check for ambient first — it's not voice direction
    for pattern in _AMBIENT_PATTERNS:
        if pattern in lower:
            d.ambient = text
            return d

    # Extract accent
    m = _ACCENT_RE.search(text)
    if m:
        d.accent = m.group(1).lower()
        # Remove accent clause for cleaner remaining parse
        lower = _ACCENT_RE.sub("", lower).strip()
    else:
        # compound fallback: "in Nigerian pidgin accent" → normalize the
        # whole phrase ("nigerian pidgin" → "pidgin", longest match wins)
        m2 = re.search(r"\bin\s+(?:a\s+|an\s+|the\s+)?([\w ]+?)\s+accent\b",
                       text, re.IGNORECASE)
        if m2:
            from .accent import normalize_accent as _norm_accent
            normed = _norm_accent(m2.group(1))
            if normed:
                d.accent = normed
                lower = lower.replace(m2.group(0).lower(), " ").strip()

    # "said X" / "says X" pattern — X is the emotion/delivery
    said_m = re.search(r"\bsa(?:id|ys)\s+(\w+)", lower)
    if said_m:
        word = said_m.group(1)
        if word in _DELIVERY_VERBS or word.rstrip("s") in _DELIVERY_VERBS:
            d.delivery = word
        elif word in _EMOTION_WORDS:
            d.emotion = _normalize_emotion(word)

    # Standalone delivery verbs
    words = re.findall(r"\w+", lower)
    for w in words:
        if w in _DELIVERY_VERBS and not d.delivery:
            d.delivery = w
        elif w in _EMOTION_WORDS and not d.emotion:
            d.emotion = _normalize_emotion(w)

    # Pace
    for phrase, pace in _PACE_WORDS.items():
        if phrase in lower and not d.pace:
            d.pace = pace

    # Intensity adverbs ("very angry" → 4, "slightly nervous" → 1)
    for phrase, level in sorted(_INTENSITY_WORDS.items(),
                                key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(phrase) + r"\b", lower):
            d.intensity = level
            break

    return d


def direction_to_dsp_params(d: Direction) -> dict:
    """Convert a Direction to DSP shaping parameters.

    Returns pitch_shift (semitones), rate_mult, and energy hints.
    These are honest DSP approximations, not neural emotion.
    ``d.intensity`` (1–5) scales the deviation from neutral — "very
    angry" shapes harder than "slightly angry".
    """
    params = {"pitch_shift": 0.0, "rate_mult": 1.0, "energy": 1.0}

    # Emotion → pitch/rate shaping (Zonos-inspired dimensional mapping)
    emotion_map = {
        "angry": {"pitch_shift": 1.5, "rate_mult": 1.15, "energy": 1.3},
        "furious": {"pitch_shift": 2.0, "rate_mult": 1.2, "energy": 1.4},
        "excited": {"pitch_shift": 2.0, "rate_mult": 1.2, "energy": 1.25},
        "happy": {"pitch_shift": 1.0, "rate_mult": 1.05, "energy": 1.1},
        "joyful": {"pitch_shift": 1.5, "rate_mult": 1.1, "energy": 1.2},
        "jubilant": {"pitch_shift": 2.0, "rate_mult": 1.15, "energy": 1.3},
        "sad": {"pitch_shift": -2.0, "rate_mult": 0.85, "energy": 0.8},
        "sorrowful": {"pitch_shift": -2.5, "rate_mult": 0.8, "energy": 0.75},
        "melancholy": {"pitch_shift": -2.0, "rate_mult": 0.85,
                       "energy": 0.8},
        "melancholic": {"pitch_shift": -2.0, "rate_mult": 0.85,
                        "energy": 0.8},
        "mournful": {"pitch_shift": -2.5, "rate_mult": 0.78,
                     "energy": 0.7},
        "fearful": {"pitch_shift": 2.5, "rate_mult": 1.25, "energy": 1.1},
        "scared": {"pitch_shift": 2.5, "rate_mult": 1.25, "energy": 1.1},
        "terrified": {"pitch_shift": 3.0, "rate_mult": 1.3, "energy": 1.15},
        "panicked": {"pitch_shift": 2.8, "rate_mult": 1.35, "energy": 1.2},
        "calm": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 0.9},
        "serene": {"pitch_shift": -0.8, "rate_mult": 0.88, "energy": 0.85},
        "tired": {"pitch_shift": -1.5, "rate_mult": 0.8, "energy": 0.75},
        "exhausted": {"pitch_shift": -2.0, "rate_mult": 0.75,
                      "energy": 0.7},
        "drowsy": {"pitch_shift": -1.8, "rate_mult": 0.78, "energy": 0.7},
        "nervous": {"pitch_shift": 1.0, "rate_mult": 1.15, "energy": 0.95},
        "anxious": {"pitch_shift": 1.2, "rate_mult": 1.18, "energy": 0.95},
        "restless": {"pitch_shift": 1.0, "rate_mult": 1.2, "energy": 1.0},
        "confident": {"pitch_shift": 0.5, "rate_mult": 0.95, "energy": 1.15},
        "defiant": {"pitch_shift": 0.8, "rate_mult": 1.0, "energy": 1.25},
        "tender": {"pitch_shift": -1.0, "rate_mult": 0.85, "energy": 0.85},
        "sarcastic": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 1.0},
        "smug": {"pitch_shift": -0.3, "rate_mult": 0.92, "energy": 1.05},
        "contemptuous": {"pitch_shift": -0.8, "rate_mult": 0.9,
                         "energy": 1.05},
        "bored": {"pitch_shift": -1.2, "rate_mult": 0.85, "energy": 0.8},
        "deadpan": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 0.85},
        "playful": {"pitch_shift": 1.5, "rate_mult": 1.1, "energy": 1.15},
        "giddy": {"pitch_shift": 2.0, "rate_mult": 1.2, "energy": 1.2},
        "mischievous": {"pitch_shift": 1.0, "rate_mult": 1.05,
                        "energy": 1.1},
        "amused": {"pitch_shift": 1.0, "rate_mult": 1.0, "energy": 1.05},
        "curious": {"pitch_shift": 1.2, "rate_mult": 1.0, "energy": 1.0},
        "surprised": {"pitch_shift": 2.0, "rate_mult": 1.1, "energy": 1.1},
        "bewildered": {"pitch_shift": 1.5, "rate_mult": 0.95, "energy": 0.9},
        "hopeful": {"pitch_shift": 1.0, "rate_mult": 1.0, "energy": 1.05},
        "desperate": {"pitch_shift": 1.8, "rate_mult": 1.2, "energy": 1.15},
        "passionate": {"pitch_shift": 1.5, "rate_mult": 1.1, "energy": 1.25},
        "vengeful": {"pitch_shift": 0.5, "rate_mult": 0.95, "energy": 1.3},
        "indignant": {"pitch_shift": 1.0, "rate_mult": 1.1, "energy": 1.2},
        "bitter": {"pitch_shift": -1.0, "rate_mult": 0.9, "energy": 1.0},
        "resigned": {"pitch_shift": -1.5, "rate_mult": 0.85, "energy": 0.8},
        "lonely": {"pitch_shift": -1.8, "rate_mult": 0.85, "energy": 0.8},
        "nostalgic": {"pitch_shift": -0.8, "rate_mult": 0.9, "energy": 0.9},
        "wistful": {"pitch_shift": -1.0, "rate_mult": 0.88, "energy": 0.85},
        "reflective": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 0.9},
        "thoughtful": {"pitch_shift": -0.3, "rate_mult": 0.92,
                       "energy": 0.95},
        "empathetic": {"pitch_shift": 0.3, "rate_mult": 0.95,
                       "energy": 1.0},
        "reassuring": {"pitch_shift": -0.5, "rate_mult": 0.9,
                       "energy": 1.0},
        "proud": {"pitch_shift": 0.8, "rate_mult": 0.95, "energy": 1.15},
        "triumphant": {"pitch_shift": 1.5, "rate_mult": 1.05,
                       "energy": 1.25},
        "determined": {"pitch_shift": 0.3, "rate_mult": 1.0, "energy": 1.2},
        "eager": {"pitch_shift": 1.2, "rate_mult": 1.15, "energy": 1.15},
        "manic": {"pitch_shift": 2.0, "rate_mult": 1.3, "energy": 1.3},
        "hysterical": {"pitch_shift": 2.5, "rate_mult": 1.3, "energy": 1.25},
        "flustered": {"pitch_shift": 1.5, "rate_mult": 1.2, "energy": 1.0},
        "awkward": {"pitch_shift": 0.5, "rate_mult": 0.9, "energy": 0.9},
        "hesitant": {"pitch_shift": 0.3, "rate_mult": 0.85, "energy": 0.9},
        "shy": {"pitch_shift": 0.8, "rate_mult": 0.9, "energy": 0.85},
        "sheepish": {"pitch_shift": 0.8, "rate_mult": 0.88, "energy": 0.85},
        "guilty": {"pitch_shift": -0.8, "rate_mult": 0.88, "energy": 0.85},
        "remorseful": {"pitch_shift": -1.2, "rate_mult": 0.85,
                       "energy": 0.8},
        "disgusted": {"pitch_shift": -0.5, "rate_mult": 0.95,
                      "energy": 1.05},
        "annoyed": {"pitch_shift": 0.8, "rate_mult": 1.05, "energy": 1.1},
        "appalled": {"pitch_shift": 1.5, "rate_mult": 1.0, "energy": 1.1},
        "jealous": {"pitch_shift": 0.5, "rate_mult": 1.0, "energy": 1.1},
        "envious": {"pitch_shift": 0.5, "rate_mult": 1.0, "energy": 1.05},
        "suspicious": {"pitch_shift": -0.3, "rate_mult": 0.92,
                       "energy": 1.0},
        "skeptical": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 1.0},
        "relieved": {"pitch_shift": -0.8, "rate_mult": 0.92,
                     "energy": 0.95},
        "ecstatic": {"pitch_shift": 2.5, "rate_mult": 1.2, "energy": 1.35},
    }
    if d.emotion in emotion_map:
        # intensity 1–5 scales the deviation from neutral (3 = as mapped)
        scale = 0.4 + 0.3 * max(1, min(5, d.intensity or 3)) / 1.5
        for k, v in emotion_map[d.emotion].items():
            if k == "pitch_shift":
                params[k] = v * scale
            elif k == "rate_mult":
                params[k] = 1.0 + (v - 1.0) * scale
            else:
                params[k] = 1.0 + (v - 1.0) * scale

    # Delivery overrides
    if d.delivery in ("whisper", "whispers", "whispering"):
        params["energy"] = 0.5
        params["pitch_shift"] += 0.5
    elif d.delivery in ("shout", "shouts", "shouting"):
        params["energy"] = 1.4
        params["pitch_shift"] += 1.0
    elif d.delivery in ("sing", "sings", "singing"):
        params["energy"] = 1.2  # singing handled separately

    # Pace overrides rate
    pace_map = {"slow": 0.8, "fast": 1.25, "rushed": 1.35,
                "drawn out": 0.7, "hesitant": 0.85}
    if d.pace in pace_map:
        params["rate_mult"] = pace_map[d.pace]

    return params
