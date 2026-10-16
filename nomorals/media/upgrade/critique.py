"""Tier 0 — Critique engine.

A brief becomes a checklist of checkable claims. A render (a dict of
measured features, a score card, or a PIL image) is scored on named
axes: prompt adherence, technical, composition, consistency,
aesthetic. The quality floor refuses to ship garbage: ``passes_floor``
is False unless every axis clears its minimum AND the weighted overall
clears the floor. Nothing here invents a model judgement — scores come
from caller-supplied measurements or from real, documented image
features (exposure, sharpness, saturation) computed in
:func:`measure_image`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

#: Weighted overall below this → do not ship. Per-axis minima in AXIS_MIN.
QUALITY_FLOOR = 0.55

AXES = ("prompt_adherence", "technical", "composition", "consistency", "aesthetic")
WEIGHTS = {
    "prompt_adherence": 0.30,
    "technical": 0.25,
    "composition": 0.15,
    "consistency": 0.15,
    "aesthetic": 0.15,
}
AXIS_MIN = {
    "prompt_adherence": 0.40,
    "technical": 0.40,
    "composition": 0.25,
    "consistency": 0.30,
    "aesthetic": 0.25,
}

# Words that turn a brief sentence into a checkable claim. A claim is
# checkable when it names something observable: a subject, a setting, a
# count, a colour, a mood word, or a framing word.
_CLAIM_WORDS = (
    r"wearing|holding|standing|sitting|smiling|looking|facing|walking|"
    r"red|blue|green|gold|black|white|yellow|orange|purple|pink|warm|cool|"
    r"two|three|four|five|one|single|group|crowd|"
    r"close-up|wide|aerial|portrait|landscape|night|day|sunset|rain|"
    r"market|street|beach|studio|stage|kitchen|office|village|city"
)


@dataclass
class Claim:
    """One checkable claim extracted from a brief."""

    text: str
    kind: str  # subject | setting | attribute | framing | count
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "kind": self.kind,
                "keywords": list(self.keywords)}


@dataclass
class Critique:
    """Named-axis scores + verdict for one render."""

    scores: dict[str, float]
    overall: float
    claims_checked: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    verdict: str = "reject"  # ship | revise | reject

    @property
    def ship(self) -> bool:
        return self.verdict == "ship"

    def to_dict(self) -> dict[str, Any]:
        return {"scores": dict(self.scores), "overall": self.overall,
                "claims_checked": self.claims_checked,
                "failures": list(self.failures), "verdict": self.verdict}


def _kind_for(sentence: str) -> str:
    low = sentence.lower()
    if re.search(r"\b(two|three|four|five|\d+)\b", low):
        return "count"
    if re.search(r"\b(close-up|wide|aerial|portrait|landscape|night|day|sunset)\b", low):
        return "framing"
    if re.search(r"\b(market|street|beach|studio|stage|kitchen|office|village|city|room)\b", low):
        return "setting"
    if re.search(r"\b(wearing|holding|standing|sitting|smiling|looking|facing|walking)\b", low):
        return "subject"
    return "attribute"


def brief_to_checklist(brief: str) -> list[Claim]:
    """Split a brief into checkable claims.

    Every sentence that names something observable becomes a claim.
    Sentences with no observable content are dropped (they cannot be
    checked, so they must not pass silently).
    """
    claims: list[Claim] = []
    for raw in re.split(r"[.!?\n]+", brief or ""):
        sentence = raw.strip()
        if not sentence:
            continue
        keywords = tuple(sorted(set(re.findall(_CLAIM_WORDS, sentence.lower()))))
        if not keywords and len(sentence.split()) < 3:
            continue
        claims.append(Claim(text=sentence, kind=_kind_for(sentence),
                            keywords=keywords))
    return claims


def measure_image(image: Any) -> dict[str, float]:
    """Real, documented image features -> technical/aesthetic proxies.

    exposure: mean luminance closeness to mid-grey (0..1).
    sharpness: mean Laplacian magnitude, normalised (0..1).
    saturation: mean HSV saturation (0..1).
    These are measurements, not model judgements; callers may override
    any axis with a real model score via ``scores=`` in critique_render.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - env always has these
        raise RuntimeError("measure_image needs Pillow + numpy") from exc
    if not isinstance(image, Image.Image):
        raise TypeError("measure_image expects a PIL Image")
    grey = image.convert("L")
    arr = np.asarray(grey, dtype=np.float64)
    mean = float(arr.mean()) / 255.0
    exposure = 1.0 - abs(mean - 0.5) * 2.0
    lap = np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()
    sharpness = min(1.0, float(lap) / 24.0)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float64)
    saturation = float(hsv[:, :, 1].mean()) / 255.0
    return {"exposure": max(0.0, exposure),
            "sharpness": max(0.0, min(1.0, sharpness)),
            "saturation": max(0.0, min(1.0, saturation))}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def critique_render(brief: str = "", *,
                    scores: dict[str, float] | None = None,
                    image: Any = None,
                    claim_results: list[bool] | None = None,
                    consistency: float | None = None) -> Critique:
    """Score one render on the five named axes.

    Sources, in priority order: explicit ``scores`` (real model /
    human measurements win), then ``image`` features for technical +
    aesthetic, then ``claim_results`` (per-claim pass/fail) for prompt
    adherence. Missing evidence scores 0 on that axis — never a
    default pass.
    """
    claims = brief_to_checklist(brief)
    out: dict[str, float] = {}
    supplied = dict(scores or {})
    if claim_results is not None and claims:
        out["prompt_adherence"] = sum(1 for r in claim_results if r) / max(1, len(claims))
    elif "prompt_adherence" in supplied:
        out["prompt_adherence"] = supplied["prompt_adherence"]
    elif not claims:
        out["prompt_adherence"] = 0.0
    else:
        out["prompt_adherence"] = 0.0
    if image is not None:
        m = measure_image(image)
        out["technical"] = _clamp01(0.5 * m["exposure"] + 0.5 * m["sharpness"])
        out["aesthetic"] = _clamp01(0.6 * m["exposure"] + 0.4 * (1.0 - abs(m["saturation"] - 0.45) * 2.0))
    for axis in AXES:
        if axis not in out and axis in supplied:
            out[axis] = supplied[axis]
    if "consistency" not in out:
        out["consistency"] = consistency if consistency is not None else supplied.get("consistency", 0.0)
    if "composition" not in out:
        out["composition"] = supplied.get("composition", 0.0)
    for axis in AXES:
        out[axis] = _clamp01(out.get(axis, 0.0))
    overall = sum(WEIGHTS[a] * out[a] for a in AXES)
    failures = [a for a in AXES if out[a] < AXIS_MIN[a]]
    if overall < QUALITY_FLOOR:
        failures.append("overall_below_floor")
    checked = [{"claim": c.text, "passed": bool(claim_results[i]) if claim_results and i < len(claim_results) else None}
               for i, c in enumerate(claims)]
    verdict = "ship" if not failures else ("revise" if overall >= QUALITY_FLOOR - 0.15 else "reject")
    return Critique(scores=out, overall=overall, claims_checked=checked,
                    failures=failures, verdict=verdict)


def passes_floor(critique: Critique) -> bool:
    """The one invariant: nothing below the quality floor ships."""
    return critique.verdict == "ship" and critique.overall >= QUALITY_FLOOR
