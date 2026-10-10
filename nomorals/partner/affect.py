"""Offline affect sensing: a small, transparent emotion lexicon.

The responder's signal bank is regex — precise but brittle: it only fires on
phrasings someone hand-wrote. "this week has been a lot" carries sadness no
regex knows. This module is the second pass behind the regex bank:

* **NRC-style word→emotion lexicon** — hand-curated (~150 stems), GoEmotions /
  NRC-flavored categories, no model, no downloads, stdlib only.
* **VADER-style modifiers** — degree words (very/so/really ×1.5), hedges
  (kind of/a bit ×0.6), negation windows ("not happy" flips), ALL-CAPS emphasis,
  and !!! / ??? punctuation boosts. Tuned for chat text, like VADER.
* **Capped and subordinate** — affect-derived events are capped at low intensity
  and never override a regex kind. The regex bank is the precision instrument;
  this is the recall net.

Deterministic: the same text always scores the same. Pure: no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .mood import MoodEvent

__all__ = [
    "AFFECT_CATEGORIES",
    "AffectReading",
    "AffectScorer",
    "affect_to_events",
    "score_affect",
]

#: The emotion categories the lexicon covers. GoEmotions-flavored
#: (27-emotion taxonomy, Demszky et al. 2020) collapsed to the ones that
#: actually move a relationship; NRC-flavored word associations.
AFFECT_CATEGORIES: tuple[str, ...] = (
    "joy", "love", "excitement", "gratitude", "amusement", "affection",
    "pride", "hope", "relief", "curiosity", "calm",
    "sadness", "anger", "fear", "anxiety", "disgust", "surprise",
    "shame", "disappointment", "jealousy", "loneliness", "frustration",
    "nervousness", "remorse",
)

#: word -> (emotion, base weight). Stems: matching is prefix-on-word so
#: "loved"/"loving" hit "lov". Curated for chat text, not news text.
_WORD_AFFECT: dict[str, tuple[str, float]] = {
    # ── warm ──
    "lov": ("love", 1.0), "adore": ("love", 1.0), "cherish": ("love", 0.9),
    "miss": ("love", 0.7), "darling": ("love", 0.8),
    "happy": ("joy", 0.9), "glad": ("joy", 0.8), "delight": ("joy", 0.9),
    "wonderful": ("joy", 0.8), "amazing": ("joy", 0.8), "great": ("joy", 0.6),
    "awesome": ("joy", 0.7), "fantastic": ("joy", 0.8), "yay": ("joy", 0.8),
    "haha": ("amusement", 0.6), "lol": ("amusement", 0.5), "lmao": ("amusement", 0.6),
    "funny": ("amusement", 0.7), "hilarious": ("amusement", 0.8),
    "excit": ("excitement", 0.9), "thrilled": ("excitement", 1.0),
    "can't wait": ("excitement", 0.8), "stoked": ("excitement", 0.9),
    "thank": ("gratitude", 0.9), "grateful": ("gratitude", 1.0),
    "appreciate": ("gratitude", 0.9), "blessed": ("gratitude", 0.7),
    "sweet": ("affection", 0.7), "tender": ("affection", 0.8),
    "cuddle": ("affection", 0.8), "hug": ("affection", 0.7),
    "care": ("affection", 0.6),
    "proud": ("pride", 0.9), "accomplish": ("pride", 0.8),
    "hope": ("hope", 0.8), "wish": ("hope", 0.5), "fingers crossed": ("hope", 0.7),
    "reliev": ("relief", 0.8), "phew": ("relief", 0.8),
    "curious": ("curiosity", 0.7), "wonder": ("curiosity", 0.6),
    "calm": ("calm", 0.8), "peaceful": ("calm", 0.9), "chill": ("calm", 0.6),
    "relax": ("calm", 0.7),
    # ── cold ──
    "sad": ("sadness", 0.9), "cry": ("sadness", 0.8), "tears": ("sadness", 0.8),
    "depress": ("sadness", 0.9), "lonely": ("loneliness", 1.0),
    "alone": ("loneliness", 0.7), "empty": ("loneliness", 0.7),
    "angry": ("anger", 1.0), "furious": ("anger", 1.0), "rage": ("anger", 0.9),
    "hate": ("anger", 0.9), "annoy": ("anger", 0.7), "irritat": ("anger", 0.7),
    "piss": ("anger", 0.8), "mad": ("anger", 0.8),
    "scare": ("fear", 0.9), "afraid": ("fear", 0.9), "terrifi": ("fear", 0.8),
    "fear": ("fear", 0.9),
    "anxious": ("anxiety", 1.0), "worri": ("anxiety", 0.8),
    "nervous": ("nervousness", 0.9), "stress": ("anxiety", 0.8),
    "overwhelm": ("anxiety", 0.8), "panic": ("anxiety", 0.9),
    "disgust": ("disgust", 0.9), "gross": ("disgust", 0.7), "ew": ("disgust", 0.6),
    "vile": ("disgust", 0.8),
    "surpris": ("surprise", 0.7), "shock": ("surprise", 0.8),
    "unbeliev": ("surprise", 0.6), "wow": ("surprise", 0.5),
    "ashamed": ("shame", 0.9), "embarrass": ("shame", 0.8),
    "humiliat": ("shame", 0.9),
    "disappoint": ("disappointment", 0.9), "let down": ("disappointment", 0.9),
    "bummed": ("disappointment", 0.7), "meh": ("disappointment", 0.5),
    "jealous": ("jealousy", 1.0), "envious": ("jealousy", 0.8),
    "frustrat": ("frustration", 0.9), "ugh": ("frustration", 0.6),
    "sick of": ("frustration", 0.8), "tired of": ("frustration", 0.7),
    "sorry": ("remorse", 0.8), "regret": ("remorse", 0.9),
    "guilt": ("remorse", 0.9), "my fault": ("remorse", 0.9),
    "hurt": ("sadness", 0.7), "pain": ("sadness", 0.7),
    "tired": ("sadness", 0.4), "exhaust": ("sadness", 0.6),
    "awful": ("sadness", 0.6), "terrible": ("sadness", 0.6),
    "horrible": ("sadness", 0.6), "bad day": ("sadness", 0.7),
}

#: Degree modifiers, VADER-style. Multi-word entries checked first.
_INTENSIFIERS: dict[str, float] = {
    "extremely": 1.6, "incredibly": 1.6, "absolutely": 1.5, "totally": 1.5,
    "completely": 1.5, "utterly": 1.5, "so": 1.4, "very": 1.4, "really": 1.4,
    "super": 1.4, "pretty": 1.2, "quite": 1.2, "rather": 1.1,
}
_HEDGES: dict[str, float] = {
    "kind of": 0.6, "sort of": 0.6, "a bit": 0.6, "a little": 0.6,
    "slightly": 0.6, "barely": 0.5, "mildly": 0.6,
}
_NEGATIONS = frozenset({
    "not", "no", "never", "n't", "dont", "don't", "cant", "can't",
    "wont", "won't", "isnt", "isn't", "arent", "aren't", "wasnt",
    "wasn't", "werent", "weren't", "havent", "haven't", "hasnt",
    "hasn't", "hadnt", "hadn't", "couldnt", "couldn't", "shouldnt",
    "shouldn't", "wouldnt", "wouldn't", "without",
})

#: Negated warm emotion -> this cold one; negated cold emotion -> relief-ish.
#: (VADER flips polarity; we map to the closest relationship-relevant label.)
_NEGATION_FLIP: dict[str, str] = {
    "joy": "disappointment", "love": "loneliness", "excitement": "disappointment",
    "gratitude": "disappointment", "amusement": "disappointment",
    "affection": "loneliness", "pride": "shame", "hope": "disappointment",
    "relief": "anxiety", "curiosity": "disgust", "calm": "anxiety",
    "sadness": "relief", "anger": "calm", "fear": "relief",
    "anxiety": "relief", "disgust": "amusement", "surprise": "calm",
    "shame": "pride", "disappointment": "hope", "jealousy": "relief",
    "loneliness": "love", "frustration": "relief",
    "nervousness": "relief", "remorse": "pride",
}

#: Affect emotion -> EVENT_TABLE kind. The second pass only ever produces
#: these; the regex bank owns the sharper kinds (insult, fight, apology…).
_EMOTION_TO_EVENT: dict[str, str] = {
    "joy": "warm_response", "love": "affectionate", "excitement": "good_news",
    "gratitude": "warm_response", "amusement": "warm_response",
    "affection": "affectionate", "pride": "good_news", "hope": "warm_response",
    "relief": "forgiven", "curiosity": "deep_conversation", "calm": "warm_response",
    "sadness": "bad_news", "anger": "fight_start", "fear": "bad_news",
    "anxiety": "bad_news", "disgust": "cold_response", "surprise": "warm_response",
    "shame": "apology", "disappointment": "cold_response",
    "jealousy": "jealousy_trigger", "loneliness": "bad_news",
    "frustration": "fight_start", "nervousness": "bad_news",
    "remorse": "apology",
}

_WORD = re.compile(r"[a-z0-9']+")
_NEGATION_WINDOW = 3  # words back a "not" still flips


@dataclass
class AffectReading:
    """One scored message: the emotion profile, dominant label, valence."""

    scores: dict[str, float]
    dominant: str
    intensity: float  # 0..1, strength of the dominant emotion
    valence: float    # -1..1, warm-minus-cold

    def to_dict(self) -> dict[str, object]:
        return {
            "dominant": self.dominant,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "scores": {k: round(v, 3) for k, v in sorted(
                self.scores.items(), key=lambda kv: kv[1], reverse=True)[:6]},
        }


class AffectScorer:
    """Score a message's emotional content with the lexicon. Pure, deterministic."""

    #: Emotions with positive relationship valence (for the valence readout).
    POSITIVE = frozenset({
        "joy", "love", "excitement", "gratitude", "amusement", "affection",
        "pride", "hope", "relief", "curiosity", "calm",
    })

    def score(self, text: str) -> AffectReading:
        scores: dict[str, float] = {}
        if not text or not text.strip():
            return AffectReading({}, "neutral", 0.0, 0.0)
        lowered = text.lower()
        words = _WORD.findall(lowered)
        if not words:
            return AffectReading({}, "neutral", 0.0, 0.0)

        # Punctuation emphasis, VADER-style.
        bangs = min(3, lowered.count("!"))
        questions = min(2, lowered.count("?"))
        caps_ratio = sum(1 for w in text.split() if w.isalpha() and w.isupper()) / max(1, len(text.split()))

        i = 0
        while i < len(words):
            # Multi-word entries first ("can't wait", "kind of", "sick of").
            hit = None
            for n in (3, 2):
                if i + n <= len(words):
                    phrase = " ".join(words[i:i + n])
                    if phrase in _WORD_AFFECT:
                        hit = (phrase, n)
                        break
            if hit is None and words[i] in _WORD_AFFECT:
                hit = (words[i], 1)
            if hit is None:
                # Prefix/stem match for single words.
                w = words[i]
                for key, (emo, base) in _WORD_AFFECT.items():
                    if " " not in key and len(key) > 3 and w.startswith(key):
                        hit = (key, 1)
                        break
            if hit is None:
                i += 1
                continue
            key, n = hit
            emotion, base = _WORD_AFFECT[key]
            weight = base

            # Look back for modifiers / negation.
            window = words[max(0, i - _NEGATION_WINDOW):i]
            negated = any(w in _NEGATIONS or w.endswith("n't") for w in window)
            for mod, mult in _INTENSIFIERS.items():
                if mod in window:
                    weight *= mult
                    break
            else:
                for hedge, mult in _HEDGES.items():
                    if hedge in " ".join(window):
                        weight *= mult
                        break
            # ALL-CAPS emphasis on the word itself.
            raw_tokens = text.split()
            if any(tok.upper() == words[i].upper() and tok.isupper() and len(tok) > 2
                   for tok in raw_tokens):
                weight *= 1.25
            if negated:
                emotion = _NEGATION_FLIP.get(emotion, emotion)
                weight *= 0.8
            scores[emotion] = scores.get(emotion, 0.0) + weight
            i += n

        if not scores:
            return AffectReading({}, "neutral", 0.0, 0.0)

        # Punctuation boosts arousal-carrying emotions.
        if bangs:
            for emo in ("anger", "excitement", "joy", "frustration", "surprise"):
                if emo in scores:
                    scores[emo] *= 1.0 + 0.15 * bangs
        if questions:
            for emo in ("anxiety", "curiosity", "fear", "nervousness"):
                if emo in scores:
                    scores[emo] *= 1.0 + 0.15 * questions
        if caps_ratio > 0.4:
            for emo in scores:
                scores[emo] *= 1.0 + 0.2 * caps_ratio

        # Normalize: dominant -> 1.0 scale-ish. Keep raw sums under it.
        top = max(scores.values())
        norm = {k: v / top for k, v in scores.items()}
        dominant = max(scores, key=lambda k: scores[k])
        pos = sum(v for k, v in norm.items() if k in self.POSITIVE)
        neg = sum(v for k, v in norm.items() if k not in self.POSITIVE)
        valence = (pos - neg) / max(1e-9, pos + neg)
        intensity = min(1.0, top / 2.5)
        return AffectReading(
            scores={k: round(v, 3) for k, v in norm.items()},
            dominant=dominant,
            intensity=round(intensity, 3),
            valence=round(valence, 3),
        )


#: Module-level default scorer (stateless — safe to share).
_default_scorer = AffectScorer()


def score_affect(text: str) -> AffectReading:
    """Score one message with the default scorer."""
    return _default_scorer.score(text)


#: Cap on affect-derived event intensity: the second pass whispers.
AFFECT_EVENT_CAP = 0.35


def affect_to_events(
    reading: AffectReading,
    *,
    skip_kinds: frozenset[str] = frozenset(),
    cap: float = AFFECT_EVENT_CAP,
) -> list[MoodEvent]:
    """Turn an affect reading into MoodEvents.

    Only the top two emotions convert, each at ``0.15 + intensity * 0.5``
    capped at ``cap``. Kinds already produced by the regex bank
    (``skip_kinds``) are never duplicated — the regex bank wins ties.
    """
    events: list[MoodEvent] = []
    ranked = sorted(reading.scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    for emotion, strength in ranked:
        if strength < 0.35:
            continue
        kind = _EMOTION_TO_EVENT.get(emotion)
        if not kind or kind in skip_kinds:
            continue
        if any(e.kind == kind for e in events):
            continue
        intensity = min(cap, 0.15 + strength * 0.5)
        events.append(MoodEvent(
            kind=kind,
            intensity=round(intensity, 3),
            note=f"affect:{emotion}",
        ))
        skip_kinds = skip_kinds | {kind}
    return events
