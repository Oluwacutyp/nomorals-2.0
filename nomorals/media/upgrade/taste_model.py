"""Tier 2 moat — Taste model from keep vs discard.

Learns from the owner's own keep/discard judgements: each training
example is a feature dict (measured image features + style tags) with
a kept/discarded label. The model is a per-feature log-odds weight
table — transparent, inspectable, no black box. After ~100 renders it
starts to express real preferences ("warm shadows", "hates centred
compositions") as readable rules. With <10 examples it abstains
(returns neutral) instead of hallucinating a taste.

Features expected (all 0..1 unless noted): warmth, shadow_warmth,
centeredness, symmetry, saturation, contrast, negative_space,
skin_tone_present (0/1), text_present (0/1). Extra keys are ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FEATURES = ("warmth", "shadow_warmth", "centeredness", "symmetry",
            "saturation", "contrast", "negative_space",
            "skin_tone_present", "text_present")
MIN_EXAMPLES = 10
LEARNED_AT = 100


@dataclass
class TasteModel:
    weights: dict[str, float] = field(default_factory=dict)
    examples: int = 0
    keeps: int = 0

    def learn(self, features: dict[str, float], kept: bool) -> None:
        """One keep/discard judgement updates per-feature weights.

        Kept examples push weights toward their feature values;
        discarded examples push away. Learning rate decays with
        example count so early judgements do not dominate forever.
        """
        lr = 1.0 / (1.0 + self.examples / 25.0)
        y = 1.0 if kept else -1.0
        for f in FEATURES:
            v = float(features.get(f, 0.5)) - 0.5  # centre at 0
            self.weights[f] = self.weights.get(f, 0.0) + lr * y * v
        self.examples += 1
        if kept:
            self.keeps += 1

    def score(self, features: dict[str, float]) -> float:
        """Predicted keep-probability 0..1; 0.5 (neutral) when unlearned."""
        if self.examples < MIN_EXAMPLES:
            return 0.5
        logit = sum(self.weights.get(f, 0.0) * (float(features.get(f, 0.5)) - 0.5)
                    for f in FEATURES)
        return 1.0 / (1.0 + math.exp(-logit))

    def rules(self, *, threshold: float = 0.15) -> list[str]:
        """Human-readable taste rules from learned weights."""
        labels = {
            "warmth": ("warm tones", "cool tones"),
            "shadow_warmth": ("warm shadows", "cool shadows"),
            "centeredness": ("centred compositions", "off-centre compositions"),
            "symmetry": ("symmetric framing", "asymmetric framing"),
            "saturation": ("saturated colour", "muted colour"),
            "contrast": ("high contrast", "low contrast"),
            "negative_space": ("lots of negative space", "filled frames"),
            "skin_tone_present": ("skin tones present", "no skin tones"),
            "text_present": ("text in frame", "no text in frame"),
        }
        out = []
        for f, w in sorted(self.weights.items(), key=lambda kv: -abs(kv[1])):
            if abs(w) < threshold:
                continue
            like, dislike = labels[f]
            out.append(f"likes {like}" if w > 0 else f"dislikes {like} (prefers {dislike})")
        return out

    @property
    def learned(self) -> bool:
        return self.examples >= LEARNED_AT

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "examples": self.examples,
                "keeps": self.keeps}


def keep_or_discard(model: TasteModel, features: dict[str, float], *,
                    threshold: float = 0.55) -> str:
    """Taste verdict: 'keep' / 'discard' / 'undecided' (unlearned)."""
    if model.examples < MIN_EXAMPLES:
        return "undecided"
    return "keep" if model.score(features) >= threshold else "discard"
