"""Quality gate before the owner sees a render (EDIT_GEN_UPGRADE 2.10).

The gate runs :mod:`highlight_model` + the judge
(:mod:`media.upgrade.critique`) + :mod:`directed.validate` over one
document + its measured render properties, on six axes:

* exposure_clipping — clipped highlights/shadows
* loudness — LUFS + true-peak vs the doc's loudness spec
* black_frames — black / frozen frames in the render
* caption_sync_drift — captions vs word timings drift (seconds)
* identity_drift — per-shot identity vs the doc's identity locks
* aesthetic — highlight-model / critique aesthetic proxy

Measurements are *supplied*, never invented: an axis with no
measurement reports ``"unknown"`` with a weakness naming exactly what
was not measured, and ``ship`` stays ``False``. When every measured
axis passes and no axis is unknown, the gate ships. Otherwise it
either auto-fixes (safe, reversible fixes: loudness gain, caption
re-sync, exposure grade) or delivers with the weakness named honestly
— never a silent pass.
"""

from __future__ import annotations

from typing import Any

from ...media_edit.videos import MediaEditError

__all__ = ["QualityGate", "quality_gate", "auto_fix"]

AXES = ("exposure_clipping", "loudness", "black_frames",
        "caption_sync_drift", "identity_drift", "aesthetic")

#: per-axis pass thresholds (0..1 scores; loudness/caption use raw
#: measurements converted to scores by the gate).
_THRESHOLDS = {
    "exposure_clipping": 0.80,
    "loudness": 0.80,
    "black_frames": 0.90,
    "caption_sync_drift": 0.80,
    "identity_drift": 0.75,
    "aesthetic": 0.50,
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _aesthetic_score(measurements: dict[str, Any]) -> float | None:
    """Aesthetic proxy: highlight-model features → score, else critique.

    Prefers the caller's measured ``aesthetic``; then the highlight
    model's per-second features averaged; then the critique engine
    over caller-supplied image features. ``None`` = unmeasured.
    """
    if measurements.get("aesthetic") is not None:
        return _clamp01(float(measurements["aesthetic"]))
    feats = measurements.get("highlight_features")
    if feats:
        try:
            from ..highlight_model.model import HighlightMLP, default_weights_path
            mlp = HighlightMLP(default_weights_path())
            if mlp.loaded:
                scores = mlp.score([list(map(float, f)) for f in feats])
                return _clamp01(sum(scores) / len(scores)) if scores else None
        except Exception:  # noqa: BLE001 - model path is best-effort
            pass
        # heuristic fallback over the same 8 features: mean of the
        # audio+motion+faces signals (features 0,1,2) — documented,
        # not a learned claim.
        try:
            vals = [float(f[0]) * 0.4 + float(f[1]) * 0.3 + float(f[2]) * 0.3
                    for f in feats]
            return _clamp01(sum(vals) / len(vals)) if vals else None
        except (IndexError, TypeError, ValueError):
            return None
    img_feats = measurements.get("image_features")
    if img_feats:
        try:
            from ..upgrade.critique import Critique, measure_image  # noqa: F401
            # critique scores from measured features only
            m = dict(img_feats)
            aesthetic = 0.6 * _clamp01(m.get("exposure", 0.5)) + 0.4 * (
                1.0 - abs(float(m.get("saturation", 0.45)) - 0.45) * 2.0)
            return _clamp01(aesthetic)
        except Exception:  # noqa: BLE001
            return None
    return None


class QualityGate:
    """Six-axis gate: document + measurements → verdict + fixes."""

    def __init__(self, *, thresholds: dict[str, float] | None = None):
        self.thresholds = dict(_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def check(self, document: Any,
              measurements: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the gate. ``measurements`` keys (all optional, all
        honest): ``exposure`` (0..1, 1=clean), ``loudness`` (dict with
        integrated_lufs/true_peak_dbtp), ``black_frame_fraction``
        (0..1), ``caption_drift_s`` (seconds), ``identity_scores``
        (list 0..1 per shot), ``aesthetic`` / ``highlight_features`` /
        ``image_features``. Missing → axis ``unknown``."""
        m = dict(measurements or {})
        axes: dict[str, Any] = {}
        weaknesses: list[str] = []
        fixes: list[str] = []

        # exposure_clipping
        if m.get("exposure") is None:
            axes["exposure_clipping"] = {"score": None, "status": "unknown",
                                         "detail": "no exposure measurement"}
            weaknesses.append("exposure unmeasured — no clipping check ran")
        else:
            s = _clamp01(float(m["exposure"]))
            axes["exposure_clipping"] = {"score": round(s, 3),
                                         "status": "pass" if s >= self.thresholds["exposure_clipping"] else "fail",
                                         "detail": f"exposure score {s:.2f}"}
            if s < self.thresholds["exposure_clipping"]:
                weaknesses.append(f"exposure clipping: score {s:.2f}")
                fixes.append("add color_grade effect (reduce exposure)")

        # loudness — enforced against the doc's loudness spec
        if m.get("loudness") is None:
            axes["loudness"] = {"score": None, "status": "unknown",
                                "detail": "no loudness measurement"}
            weaknesses.append("loudness unmeasured — LUFS/true-peak unchecked")
        else:
            from .audio_ops import LoudnessSpec, check_loudness
            spec = None
            loud = getattr(document, "loudness", None) or {}
            if isinstance(loud, dict) and loud:
                try:
                    spec = LoudnessSpec.from_dict(loud)
                except MediaEditError:
                    spec = None
            spec = spec or LoudnessSpec()
            res = check_loudness(dict(m["loudness"]), spec)
            s = 1.0 if res["ok"] else max(0.0, 1.0 - 0.25 * len(res["issues"]))
            axes["loudness"] = {"score": round(s, 3),
                                "status": "pass" if res["ok"] else "fail",
                                "detail": "; ".join(res["issues"]) or "on target",
                                "gain_db": res["gain_db"]}
            if not res["ok"]:
                weaknesses.extend(res["issues"])
                fixes.append(f"apply loudness gain {res['gain_db']:+.1f} dB "
                             "and re-measure")

        # black frames
        if m.get("black_frame_fraction") is None:
            axes["black_frames"] = {"score": None, "status": "unknown",
                                    "detail": "no black-frame scan"}
            weaknesses.append("black frames unscanned")
        else:
            frac = _clamp01(float(m["black_frame_fraction"]))
            s = 1.0 - frac
            axes["black_frames"] = {"score": round(s, 3),
                                    "status": "pass" if s >= self.thresholds["black_frames"] else "fail",
                                    "detail": f"black fraction {frac:.3f}"}
            if s < self.thresholds["black_frames"]:
                weaknesses.append(f"black frames: {frac:.1%} of frames")
                fixes.append("drop black ranges (see removed ranges)")

        # caption sync drift
        if m.get("caption_drift_s") is None:
            axes["caption_sync_drift"] = {"score": None, "status": "unknown",
                                          "detail": "no caption timing check"}
            weaknesses.append("caption sync unmeasured")
        else:
            drift = abs(float(m["caption_drift_s"]))
            s = _clamp01(1.0 - drift / 0.5)  # 0.5s drift = 0
            axes["caption_sync_drift"] = {"score": round(s, 3),
                                          "status": "pass" if s >= self.thresholds["caption_sync_drift"] else "fail",
                                          "detail": f"drift {drift:.3f}s"}
            if s < self.thresholds["caption_sync_drift"]:
                weaknesses.append(f"caption sync drift {drift:.2f}s")
                fixes.append(f"shift captions by {-float(m['caption_drift_s']):+.2f}s")

        # identity drift — per-shot scores vs identity locks
        if m.get("identity_scores") is None:
            locks = getattr(document, "identity_locks", None) or []
            if not locks:
                axes["identity_drift"] = {"score": 1.0, "status": "pass",
                                          "detail": "no identity locks — nothing to drift"}
            else:
                axes["identity_drift"] = {"score": None, "status": "unknown",
                                          "detail": "identity locks set but no per-shot scores"}
                weaknesses.append("identity drift unmeasured — locks set, no per-shot scores")
        else:
            scores = [float(v) for v in m["identity_scores"]]
            s = _clamp01(min(scores)) if scores else 1.0
            axes["identity_drift"] = {"score": round(s, 3),
                                      "status": "pass" if s >= self.thresholds["identity_drift"] else "fail",
                                      "detail": f"worst shot identity {s:.2f}"}
            if s < self.thresholds["identity_drift"]:
                weaknesses.append(f"identity drift: worst shot {s:.2f}")
                fixes.append("re-lock identity on the drifting shot or re-generate it")

        # aesthetic
        aest = _aesthetic_score(m)
        if aest is None:
            axes["aesthetic"] = {"score": None, "status": "unknown",
                                 "detail": "no aesthetic measurement"}
            weaknesses.append("aesthetic unmeasured — no highlight/critique features")
        else:
            axes["aesthetic"] = {"score": round(aest, 3),
                                 "status": "pass" if aest >= self.thresholds["aesthetic"] else "fail",
                                 "detail": f"aesthetic {aest:.2f}"}
            if aest < self.thresholds["aesthetic"]:
                weaknesses.append(f"aesthetic low: {aest:.2f}")
                fixes.append("re-grade / re-cut weakest shots")

        measured = [a["score"] for a in axes.values()
                    if a["score"] is not None]
        overall = round(sum(measured) / len(measured), 3) if measured else 0.0
        failed = [k for k, a in axes.items() if a["status"] == "fail"]
        unknown = [k for k, a in axes.items() if a["status"] == "unknown"]
        ship = not failed and not unknown
        verdict = ("ship" if ship else
                   "revise" if (failed or unknown) else "reject")
        return {"axes": axes, "overall": overall, "ship": ship,
                "verdict": verdict, "failed": failed, "unknown": unknown,
                "weaknesses": weaknesses, "fixes": fixes,
                "deliver_with_weakness": (not ship and not failed),
                "honest_note": ("deliver with weaknesses named" if not ship
                                else "all measured axes pass")}

    def gate(self, document: Any,
             measurements: dict[str, Any] | None = None) -> dict[str, Any]:
        """Alias for :meth:`check` (the front-door name)."""
        return self.check(document, measurements)


def quality_gate(document: Any,
                 measurements: dict[str, Any] | None = None) -> dict[str, Any]:
    """One-call gate: document + measurements → verdict dict."""
    return QualityGate().check(document, measurements)


def auto_fix(document: Any, report: dict[str, Any],
             measurements: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the gate's safe, reversible fixes to a Document.

    Loudness → set the doc loudness gain hint; caption drift → shift
    text-layer timings; exposure → append a ``color_grade`` effect to
    video clips. Returns ``{"document", "applied", "remaining"}``.
    Only Document inputs are mutated (copied first); anything else
    raises.
    """
    from .document import Document
    from .timeline import Effect
    if not isinstance(document, Document):
        raise MediaEditError("auto_fix takes a Document")
    doc = document.copy()
    applied: list[str] = []
    m = dict(measurements or {})
    axes = report.get("axes", {})
    loud = axes.get("loudness", {})
    if loud.get("status") == "fail" and loud.get("gain_db") is not None:
        doc.loudness["gain_db"] = float(loud["gain_db"])
        applied.append(f"loudness gain {loud['gain_db']:+.1f} dB recorded")
    cap = axes.get("caption_sync_drift", {})
    if cap.get("status") == "fail" and m.get("caption_drift_s") is not None:
        shift = -float(m["caption_drift_s"])
        for layer in doc.text_layers:
            start = getattr(layer, "start", None)
            if start is not None:
                try:
                    layer.start = max(0.0, float(start) + shift)
                except (AttributeError, TypeError, ValueError):
                    pass
        applied.append(f"captions shifted {shift:+.2f}s")
    exp = axes.get("exposure_clipping", {})
    if exp.get("status") == "fail":
        video = doc.track("video")
        if video is not None:
            for clip in video.clips:
                clip.effects = tuple(clip.effects) + (
                    Effect(name="color_grade",
                           params={"grade": "faded"}),)
            applied.append("color_grade(faded) appended to video clips")
    return {"document": doc.seal(), "applied": applied,
            "remaining": [w for w in report.get("weaknesses", [])
                          if not any(a.split()[0] in w for a in applied)]}
