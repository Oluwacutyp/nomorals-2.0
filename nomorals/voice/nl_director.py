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
    raw: str = ""               # original tag text

    def is_empty(self) -> bool:
        return not (self.emotion or self.delivery or self.accent
                    or self.ambient or self.pace)


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
    r"scottish|irish|south african|jamaican|canadian)\s+accent\b",
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

    return d


def direction_to_dsp_params(d: Direction) -> dict:
    """Convert a Direction to DSP shaping parameters.

    Returns pitch_shift (semitones), rate_mult, and energy hints.
    These are honest DSP approximations, not neural emotion.
    """
    params = {"pitch_shift": 0.0, "rate_mult": 1.0, "energy": 1.0}

    # Emotion → pitch/rate shaping (Zonos-inspired dimensional mapping)
    emotion_map = {
        "angry": {"pitch_shift": 1.5, "rate_mult": 1.15, "energy": 1.3},
        "excited": {"pitch_shift": 2.0, "rate_mult": 1.2, "energy": 1.25},
        "happy": {"pitch_shift": 1.0, "rate_mult": 1.05, "energy": 1.1},
        "sad": {"pitch_shift": -2.0, "rate_mult": 0.85, "energy": 0.8},
        "fearful": {"pitch_shift": 2.5, "rate_mult": 1.25, "energy": 1.1},
        "scared": {"pitch_shift": 2.5, "rate_mult": 1.25, "energy": 1.1},
        "calm": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 0.9},
        "tired": {"pitch_shift": -1.5, "rate_mult": 0.8, "energy": 0.75},
        "nervous": {"pitch_shift": 1.0, "rate_mult": 1.15, "energy": 0.95},
        "confident": {"pitch_shift": 0.5, "rate_mult": 0.95, "energy": 1.15},
        "tender": {"pitch_shift": -1.0, "rate_mult": 0.85, "energy": 0.85},
        "sarcastic": {"pitch_shift": -0.5, "rate_mult": 0.9, "energy": 1.0},
    }
    if d.emotion in emotion_map:
        for k, v in emotion_map[d.emotion].items():
            params[k] = v

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
