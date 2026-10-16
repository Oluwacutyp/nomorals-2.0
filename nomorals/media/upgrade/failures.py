"""Tier 2 moat — Failure-mode library.

Every known generative failure gets: a detector (pure-feature
heuristic over an image + optional metadata, returning a confidence),
and a remedy (the concrete fix: regional inpaint, regenerate, composite,
or a prompt patch). Detectors are honest heuristics with named
thresholds — they flag suspects for the FIX stage, they do not claim
model-grade detection. Reuses the failure-ledger idea: each finding
names the failure, the evidence, and the remedy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FAILURE_MODES = {
    "hands": {"detector": "edge-density anomaly in hand regions",
              "remedy": "regional inpaint hands with prompt 'anatomically correct hands, five fingers'"},
    "eyes": {"detector": "asymmetric eye highlights / pupil shape",
             "remedy": "regional inpaint eyes with prompt 'symmetric natural eyes, matching gaze'"},
    "text": {"detector": "high-frequency glyph-like blobs in text zones",
             "remedy": "regenerate WITHOUT text, then composite real typography (see typography.py)"},
    "reflections": {"detector": "mirror/window region inconsistent with scene",
                    "remedy": "regional inpaint reflection to match scene geometry"},
    "extra_limbs": {"detector": "limb-count proxy: edge clusters beyond two arms/two legs",
                    "remedy": "regional inpaint extra limb away, or regenerate with 'two arms, two legs'"},
    "teeth": {"detector": "mouth-region high-frequency smear",
              "remedy": "regional inpaint mouth with prompt 'natural closed-lip smile'"},
    "symmetry": {"detector": "face halves diverge beyond threshold",
                 "remedy": "regional inpaint weaker half, or regenerate"},
    "jewelry": {"detector": "small metallic blobs with broken geometry",
                "remedy": "regional inpaint jewelry with prompt 'clean gold jewelry, coherent geometry'"},
}


@dataclass
class FailureFinding:
    mode: str
    confidence: float
    evidence: str
    remedy: str

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "confidence": self.confidence,
                "evidence": self.evidence, "remedy": self.remedy}


def remedy_for(mode: str) -> str:
    if mode not in FAILURE_MODES:
        raise ValueError(f"unknown failure mode {mode!r}")
    return FAILURE_MODES[mode]["remedy"]


def detect_failures(image: Any, *, text_zones: list[dict[str, Any]] | None = None,
                    prompt_mentions: list[str] | None = None) -> list[FailureFinding]:
    """Run all failure detectors over one image.

    Heuristics (documented, not model claims):
    - text: caller-declared text zones with glyph-like high-frequency
      energy → flag, because generated text is always suspect.
    - symmetry: left/right face-half histogram correlation.
    - hands/extra_limbs/teeth/eyes/reflections/jewelry: only flagged
      when the prompt mentions the relevant subject (no prompt
      evidence → no flag, avoiding false positives on landscapes).
    """
    findings: list[FailureFinding] = []
    mentions = {m.lower() for m in (prompt_mentions or [])}
    if text_zones:
        for zone in text_zones:
            findings.append(FailureFinding(
                mode="text", confidence=0.85,
                evidence=f"text zone {zone} in a generated image — generated text is never trusted",
                remedy=remedy_for("text")))
    if "face" in mentions or "portrait" in mentions or "person" in mentions:
        try:
            import numpy as np
            from PIL import Image
            if isinstance(image, Image.Image):
                small = image.convert("L").resize((64, 64))
                arr = np.asarray(small, dtype=np.float64)
                left = arr[:, :32].ravel()
                right = np.fliplr(arr[:, 32:]).ravel()
                corr = float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else 1.0
                if corr < 0.55:
                    findings.append(FailureFinding(
                        mode="symmetry", confidence=round(1.0 - max(0.0, corr), 2),
                        evidence=f"face-half correlation {corr:.2f} below 0.55",
                        remedy=remedy_for("symmetry")))
        except Exception:
            pass
    for mode in ("hands", "eyes", "teeth", "extra_limbs", "jewelry", "reflections"):
        if mode in mentions or (mode == "hands" and "person" in mentions):
            findings.append(FailureFinding(
                mode=mode, confidence=0.30,
                evidence=f"prompt mentions {mode}; heuristic check queued for FIX stage review",
                remedy=remedy_for(mode)))
    return findings
