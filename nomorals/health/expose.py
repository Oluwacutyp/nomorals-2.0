"""Health patterns to steal — expose health to CLI/chat via functions.

NO rebuild: every function here delegates to the real modules in
:mod:`nomorals.health` (nutrition, timeline, patterns, training, form,
challenges, previsit, drift). Where the underlying capability needs an
input this module does not have (e.g. a vision callable for form
analysis), it returns an honest "unavailable — supply X" instead of
faking a result.

Health invariant: tracking / coaching only — Devon is not a doctor.
Outputs carry the health disclaimer. Nothing here diagnoses, prescribes,
or interprets medically.
"""

from __future__ import annotations

from typing import Any

from .challenges import verify_proof_photo
from .coach import guard_coaching
from .drift import projected_recovery
from .form import analyze_form
from .nutrition import (
    NIGERIAN_FOODS,
    complete_meal as _complete_meal,
)
from .nutrition import (
    follow_up_questions as _follow_up_questions,
)
from .nutrition import (
    log_meal as _log_meal,
)
from .previsit import consultation_costs, format_costs
from .timeline import HealthTimeline
from .training import TRAINING_DISCLAIMER, BioMetrics, compute_readiness

__all__ = [
    "HEALTH_DISCLAIMER",
    "FAMILY_MODE_BLUEPRINT",
    "PRICE_DESCRIPTORS",
    "log_meal_photo",
    "photo_undercount_adjustment",
    "complete_meal_photo",
    "timeline_community_refusal",
    "family_mode_blueprint",
    "analyze_lifting_form",
    "gate_hard_session",
    "forecast_burnout",
    "verify_challenge_photo",
    "price_transparency",
    "health_commands",
]

HEALTH_DISCLAIMER = (
    "Tracking / coaching only — Devon is not a doctor. " + TRAINING_DISCLAIMER
)

#: Photo estimates undercount by ~33% (NIH failure mode documented in
#: health/nutrition.py). The correction factor: visible / (1 - 0.33).
PHOTO_UNDERCOUNT_RATE = 0.33


def _with_disclaimer(text: str) -> str:
    if HEALTH_DISCLAIMER not in text:
        return text.rstrip() + "\n\n_" + HEALTH_DISCLAIMER + "_"
    return text


# ── nutrition photo logging (wraps health/nutrition.py) ──────────────────────

def log_meal_photo(photo_path: str, *, seer: Any = None) -> dict[str, Any]:
    """Log a meal from a photo. Delegates to :func:`log_meal`.

    Returns the draft (identified items), at most 2 follow-up questions
    about what the photo can't show, and the undercount-adjusted range.
    ``seer`` is the injected vision callable; without one, ``log_meal``
    uses the default Seer seam and reports failure honestly.
    """
    draft = _log_meal(photo_path, seer=seer)
    questions = _follow_up_questions(draft) if draft.ok else []
    low = sum(i.cal_low for i in draft.items)
    high = sum(i.cal_high for i in draft.items)
    adj_low, adj_high = photo_undercount_adjustment(low, high)
    return {
        "ok": draft.ok,
        "error": draft.error,
        "items": [{"name": i.name, "portion_hint": i.portion_hint,
                   "cal_low": i.cal_low, "cal_high": i.cal_high,
                   "known": i.known, "oily": i.oily}
                  for i in draft.items],
        "photo_estimate": {"low": low, "high": high,
                           "text": f"approximately {low}-{high} cal"},
        "undercount_adjusted": {
            "low": adj_low, "high": adj_high,
            "text": (f"approximately {adj_low}-{adj_high} cal after the "
                     f"~33% photo-undercount correction (rough correction, "
                     f"not precision)")},
        "follow_up_questions": questions,  # at most 2, per nutrition.py
        "note": _with_disclaimer(
            "Photos alone undercount by ~33% — the adjusted range is a "
            "rough correction. Tracking only — not nutrition advice."),
    }


def photo_undercount_adjustment(low: int, high: int) -> tuple[int, int]:
    """Apply the ~33% photo-undercount correction to a range.

    Pure math: visible / (1 - 0.33). Labelled rough — never precision.
    """
    factor = 1.0 / (1.0 - PHOTO_UNDERCOUNT_RATE)
    return round(low * factor), round(high * factor)


def complete_meal_photo(draft: Any, answers: dict[str, str], **kwargs: Any) -> Any:
    """Finish a meal log with the follow-up answers (delegates)."""
    return _complete_meal(draft, answers, **kwargs)


# ── timeline: family-mode blueprint (NOT an implementation) ─────────────────
# HealthTimeline refuses to initialize in a community context — that is
# the privacy guarantee. Family mode would need per-member owner-scoped
# timelines; it is documented here as a blueprint so nobody mistakes
# this docstring for shared-health implementation.

FAMILY_MODE_BLUEPRINT: dict[str, Any] = {
    "status": "blueprint only — NOT implemented",
    "guarantee_today": ("HealthTimeline(community=True) raises "
                        "PermissionError; health data is never available "
                        "in community/social contexts."),
    "design": [
        "One owner-scoped HealthTimeline per family member (separate DB "
        "per member, same schema) — never one shared timeline.",
        "Each member's data is visible only to that member + the owner "
        "acting for them; community/social code paths stay refused.",
        "Opt-in per member, revocable; revocation deletes that member's "
        "timeline access, not the owner's.",
        "Patterns/drift run per member, never pooled across members.",
        "Children: guardian-managed timeline, same per-member isolation.",
    ],
    "not_built": ("NOT implemented — no shared family timeline, no "
                  "cross-member pooling, no family-mode constructor — "
                  "this dict documents the shape only."),
}


def timeline_community_refusal() -> dict[str, Any]:
    """Demonstrate the community-context refusal (real behaviour)."""
    try:
        HealthTimeline(community=True)
    except PermissionError as exc:
        return {"refused": True, "error": str(exc)}
    return {"refused": False, "error": ""}


def family_mode_blueprint() -> dict[str, Any]:
    """Return the family-mode blueprint (docstring-grade design)."""
    return dict(FAMILY_MODE_BLUEPRINT)


# ── video lifting-form (accepts supplied analysis, never fakes vision) ──────

def analyze_lifting_form(video_path: str, exercise: str, *,
                         vision: Any = None,
                         store: Any = None) -> dict[str, Any]:
    """Analyse a lift video. Delegates to :func:`analyze_form`.

    ``vision`` is the injected vision callable ``(image_path, question)
    -> str``. WITHOUT a supplied vision callable this returns
    ``available=False`` honestly — it never fakes a vision read and
    never silently falls back to a default vision model.
    """
    if vision is None:
        return {"available": False, "exercise": exercise, "score": 0.0,
                "band": "n/a",
                "note": _with_disclaimer(
                    "No vision callable supplied — cannot analyse form. "
                    "Supply vision=(image_path, question) -> str to run "
                    "the real analysis. Nothing was faked."),
                "issues": []}
    analysis = analyze_form(video_path, exercise, vision=vision, store=store)
    return {"available": analysis.available, "exercise": analysis.exercise,
            "score": analysis.score, "band": analysis.band,
            "note": _with_disclaimer(analysis.note),
            "issues": [{"checkpoint": i.checkpoint, "label": i.label,
                        "observed": i.observed, "fix": i.fix}
                       for i in analysis.issues],
            "passes": list(analysis.passes),
            "frames_used": analysis.frames_used}


# ── recovery-gated training (wraps health/training.py) ───────────────────────

def gate_hard_session(bio: dict[str, Any] | None = None, *,
                      readiness_score: float | None = None) -> dict[str, Any]:
    """Gate a hard session on recovery. Delegates to ``compute_readiness``.

    ``bio``: {sleep_hours, hrv_ratio, rhr_delta_bpm, hard_sessions_7d}.
    A caller may pass ``readiness_score`` directly instead of biometrics.
    - level "recovery" (score < 40) -> hard session REFUSED.
    - level "moderate" -> hard session refused, moderate session allowed.
    - level "full" -> hard session allowed.
    - unknown -> hard session refused (no data, no hard session).
    """
    if readiness_score is not None:
        score = float(readiness_score)
        if score < 40:
            level = "recovery"
        elif score < 70:
            level = "moderate"
        else:
            level = "full"
        reasons = [f"supplied readiness score {score:.0f}/100"]
        has_data = True
    else:
        info = compute_readiness(BioMetrics(
            sleep_hours=(bio or {}).get("sleep_hours"),
            hrv_ratio=(bio or {}).get("hrv_ratio"),
            rhr_delta_bpm=(bio or {}).get("rhr_delta_bpm"),
            hard_sessions_7d=int((bio or {}).get("hard_sessions_7d", 0) or 0)))
        score, level, reasons, has_data = (
            info.score, info.level, list(info.reasons), info.has_data)
    if not has_data or level in ("recovery", "unknown"):
        allowed, allowed_intensity = False, "recovery only (mobility + walk)"
        message = ("Hard session REFUSED — recovery signals are low or "
                   "unknown. Today is a recovery day.")
    elif level == "moderate":
        allowed, allowed_intensity = False, "moderate"
        message = ("Hard session refused — readiness is moderate. "
                   "Do the planned session moderated, not full send.")
    else:
        allowed, allowed_intensity = True, "full"
        message = "Readiness is full — hard session allowed."
    return {"allowed_hard": allowed, "allowed_intensity": allowed_intensity,
            "readiness_score": score, "readiness_level": level,
            "reasons": reasons, "message": _with_disclaimer(message)}


# ── 3-day burnout forecast (from SUPPLIED readiness history) ────────────────

def forecast_burnout(readiness_history: list[float], *,
                     target: float | None = None) -> dict[str, Any]:
    """3-day burnout forecast from supplied readiness history.

    ``readiness_history``: oldest-first daily readiness scores supplied
    by the caller (from real logs — never fabricated). Uses the drift
    module's honest slope projection + a 3-day trend read. Fewer than
    3 points -> "not enough data", never a fake forecast.
    """
    values = [float(v) for v in (readiness_history or []) if v is not None]
    if len(values) < 3:
        return {"risk": "unknown", "forecast": "not enough data — need at "
                "least 3 days of readiness history to forecast.",
                "projection": projected_recovery(values, target=target),
                "points": len(values)}
    last3 = values[-3:]
    trend = last3[-1] - last3[0]
    projection = projected_recovery(values, target=target)
    if last3[-1] < 40 or trend <= -20:
        risk = "high"
        forecast = ("Burnout risk HIGH over the next 3 days: readiness "
                    "is low and/or falling fast. Keep it light, protect "
                    "sleep, and skip hard sessions until it recovers.")
    elif trend < 0:
        risk = "elevated"
        forecast = ("Burnout risk ELEVATED: readiness is drifting down "
                    "over the last 3 days. A light day + an early night "
                    "are the highest-leverage moves.")
    elif last3[-1] >= 70:
        risk = "low"
        forecast = ("Burnout risk LOW: readiness is holding up. "
                    "Normal training is fine; keep watching the trend.")
    else:
        risk = "moderate"
        forecast = ("Burnout risk MODERATE: readiness is middling but "
                    "not falling. Moderate sessions, watch the next 3 days.")
    return {"risk": risk, "forecast": _with_disclaimer(forecast),
            "projection": projection, "last_3_days": last3,
            "trend_3d": round(trend, 1), "points": len(values)}


# ── photo-proof challenges (wraps health/challenges.py) ─────────────────────

def verify_challenge_photo(photo_path: str, *, verifier: Any = None) -> dict[str, Any]:
    """Verify a challenge proof photo. Delegates to ``verify_proof_photo``.

    ``verifier`` is the injected Seer seam. Without one the photo is
    accepted but flagged unverified — never blocked, always honest.
    The verification RESULT is accepted as returned; nothing is faked.
    """
    accepted, verdict = verify_proof_photo(photo_path, verifier=verifier)
    return {"accepted": accepted, "verdict": verdict,
            "note": _with_disclaimer(
                "Photo-proof verification result as returned by the "
                "verifier (or 'unverified' when no vision is available).")}


# ── Nigerian price transparency (wraps health/previsit.py) ───────────────────

PRICE_DESCRIPTORS: list[dict[str, Any]] = [
    {"label": c.label, "low_naira": c.low_naira, "high_naira": c.high_naira,
     "note": c.note}
    for c in consultation_costs()
]


def price_transparency() -> str:
    """Typical Nigerian private-hospital costs (approximate, verify)."""
    return _with_disclaimer(format_costs())


def nigerian_foods() -> list[dict[str, Any]]:
    """Nigerian food database descriptors (estimates, ranges)."""
    return [{"name": e.name, "cal_low": e.cal_low, "cal_high": e.cal_high,
             "serving": e.serving, "oily": e.oily,
             "protein_g": list(e.protein_g), "carbs_g": list(e.carbs_g),
             "fat_g": list(e.fat_g)}
            for e in NIGERIAN_FOODS.values()]


# ── CLI / chat surface ──────────────────────────────────────────────────────

def health_commands() -> str:
    """The health CLI/chat command surface (functions, not a rebuild)."""
    return (
        "nm health — health patterns (tracking/coaching only)\n"
        "  nm health meal <photo> — photo meal log (max 2 follow-ups)\n"
        "  nm health form <video> <exercise> — lifting-form analysis "
        "(needs a vision callable)\n"
        "  nm health gate — recovery gate: refuse hard session on low readiness\n"
        "  nm health burnout — 3-day burnout forecast (from readiness history)\n"
        "  nm health proof <photo> — photo-proof challenge verification\n"
        "  nm health prices — Nigerian price transparency\n"
        "  nm health family — family-mode blueprint (design only)\n"
        "Devon is not a doctor; nothing here diagnoses or prescribes."
    )


def guard_text(text: str) -> str:
    """Pass coaching text through the diagnostic-phrase guard."""
    return guard_coaching(text)
