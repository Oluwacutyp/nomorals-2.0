"""Taste model — learn what the owner likes from accept / reject.

Tier 3. Complements :mod:`nomorals.media.taste` (which remembers
*what* was liked as a reference profile). This module learns *why*:
a small, transparent, per-feature weight model over song descriptors
(genre, bpm band, energy, key/mode, language, palette words, vocal
style). Each accept/reject nudges the weights; :func:`score` predicts
how much the owner will like a candidate spec. No neural net, no
black box — the weights are inspectable JSON, and a cold start (no
feedback) scores everything 0.5 honestly instead of hallucinating
a preference.

Update rule: logistic-style online learning. For a candidate with
feature vector x (0/1 per feature) and label y in {+1 accept, -1
reject}, w += lr * (y - tanh(w·x)) * x, then weights decay slightly
toward 0 so stale taste fades. Bounded weights keep one feedback
from hijacking the model.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["TasteModel", "candidate_features", "register"]

_DEFAULT_PATH = os.path.expanduser("~/.nomorals/media/taste_model.json")
_LR = 0.35
_DECAY = 0.995
_MAX_W = 3.0


def _bpm_band(bpm: float) -> str:
    if bpm < 80:
        return "bpm:slow"
    if bpm < 105:
        return "bpm:mid"
    if bpm < 135:
        return "bpm:fast"
    return "bpm:veryfast"


def _energy_band(energy: float) -> str:
    if energy < 0.35:
        return "energy:low"
    if energy < 0.7:
        return "energy:mid"
    return "energy:high"


def candidate_features(spec: dict[str, Any]) -> list[str]:
    """SongSpec-like dict -> binary feature names for the taste model."""
    spec = spec or {}
    feats = ["bias"]
    genre = str(spec.get("genre", "") or spec.get("style", "") or "").lower()
    if genre:
        feats.append(f"genre:{genre}")
    bpm = spec.get("bpm", spec.get("tempo", 0)) or 0
    try:
        feats.append(_bpm_band(float(bpm)))
    except (TypeError, ValueError):
        pass
    try:
        feats.append(_energy_band(float(spec.get("energy", 0.5))))
    except (TypeError, ValueError):
        pass
    key = str(spec.get("key", "") or "").upper()
    if key:
        feats.append(f"key:{key}")
    mode = str(spec.get("mode", "") or "").lower()
    if mode:
        feats.append(f"mode:{mode}")
    lang = str(spec.get("language", "") or "").lower()
    if lang:
        feats.append(f"language:{lang}")
    vocal = str(spec.get("vocal_style", "") or "").lower()
    if vocal:
        feats.append(f"vocal:{vocal}")
    for p in spec.get("palette", []) or []:
        feats.append(f"palette:{str(p).lower()}")
    return sorted(set(feats))


@dataclass
class TasteModel:
    """Online taste weights, persisted as JSON."""

    weights: dict[str, float] = field(default_factory=dict)
    accepts: int = 0
    rejects: int = 0
    path: str = _DEFAULT_PATH

    def score(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Predicted liking 0..1 for a candidate spec.

        Cold start (no feedback): 0.5 with confidence 0.0 — honest.
        Confidence grows with feedback count (capped at 0.9).
        """
        feats = candidate_features(spec)
        raw = sum(self.weights.get(f, 0.0) for f in feats)
        prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, raw))))
        total = self.accepts + self.rejects
        confidence = min(0.9, total / 20.0) if total else 0.0
        top = sorted(((self.weights.get(f, 0.0), f) for f in feats),
                     reverse=True)[:3]
        return {"score": round(prob, 3), "confidence": round(confidence, 3),
                "features": feats,
                "top_drivers": [f for _, f in top if _ != 0.0],
                "feedback": total}

    def feedback(self, spec: dict[str, Any], accept: bool) -> dict[str, Any]:
        """Record an accept (True) / reject (False) for a spec."""
        feats = candidate_features(spec)
        y = 1.0 if accept else -1.0
        raw = sum(self.weights.get(f, 0.0) for f in feats)
        pred = math.tanh(raw)
        err = y - pred
        for f in feats:
            w = self.weights.get(f, 0.0) * _DECAY + _LR * err
            self.weights[f] = max(-_MAX_W, min(_MAX_W, w))
        if accept:
            self.accepts += 1
        else:
            self.rejects += 1
        return {"ok": True, "accept": accept,
                "new_score": self.score(spec)["score"],
                "accepts": self.accepts, "rejects": self.rejects}

    def save(self, path: str = "") -> str:
        p = path or self.path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"weights": self.weights, "accepts": self.accepts,
                       "rejects": self.rejects}, fh, indent=2, sort_keys=True)
        return p

    @classmethod
    def load(cls, path: str = "") -> "TasteModel":
        p = path or _DEFAULT_PATH
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            return cls(weights={k: float(v) for k, v in
                                d.get("weights", {}).items()},
                       accepts=int(d.get("accepts", 0)),
                       rejects=int(d.get("rejects", 0)), path=p)
        except Exception:  # noqa: BLE001 — missing/corrupt starts fresh
            return cls(path=p)

    def top_weights(self, n: int = 10) -> list[dict[str, Any]]:
        return [{"feature": k, "weight": round(v, 3)}
                for k, v in sorted(self.weights.items(),
                                   key=lambda kv: -abs(kv[1]))[:n]]


def register(registry: Any) -> None:
    """Tool registration: ``taste_model`` — accept/reject learning."""
    from ..core.policy import Capability

    @registry.register(
        "taste_model",
        description=(
            "Taste model from accept/reject: action=score (spec JSON → "
            "predicted liking 0..1 + confidence) | feedback (spec JSON, "
            "accept true/false → updated weights) | top (strongest "
            "learned preferences). Cold start scores 0.5 honestly."
        ),
        capability=Capability.FS_WRITE,
    )
    def taste_model(action: str = "score", spec: str = "",
                    accept: bool = True) -> dict:
        import json as _json
        try:
            s = _json.loads(spec) if spec else {}
        except Exception as exc:  # noqa: BLE001
            from ..core.errors import ToolError
            raise ToolError(f"spec must be JSON: {exc}")
        model = TasteModel.load()
        if action == "score":
            return {"ok": True, **model.score(s)}
        if action == "feedback":
            r = model.feedback(s, accept)
            model.save()
            return {"ok": True, **r}
        if action == "top":
            return {"ok": True, "weights": model.top_weights(),
                    "accepts": model.accepts, "rejects": model.rejects}
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
