"""The captcha estate aggregator: every kind, its real mechanism.

Importing this module imports every kind-solver module, so every
registration this estate owns is live:

* ``local_slider``  — slider + GeeTest slider (Pass #78)
* ``local_ocr``     — image captchas via tesseract, Seer second read
* ``local_audio``   — audio captchas via local whisper ASR
* ``local_click``   — click-selection grids via vision grounding
  (also owns the ``geetest`` registration: slider vs click dispatch)
* ``local_brain``   — question captchas via the brain
* ``local_cloudflare`` — Cloudflare interstitials via the download
  kit's state machine

:func:`estate_status` is the audit surface the owner's order
("update all captcha solver for all sorts of captcha") is checked
against: one row per kind naming the mechanism that actually
handles it — local solver / service / takeover / wait — with the
local solver's availability probed NOW, on this machine. Token
kinds get no fake local rows: a service solve followed by takeover
is their real mechanism, and reCAPTCHA v3 says plainly that it is
score-based and no local solve exists.
"""
from __future__ import annotations

from typing import Any

from ..tools.captcha import CaptchaKind
from . import (  # noqa: F401 - imports ARE the registration
    audio_solver,
    click_solver,
    cloudflare_solver,
    ocr_solver,
    question_solver,
    slider_solver,
)
from .kind_solvers import kind_solver_for, registered_kinds

__all__ = ["estate_status", "MECHANISMS"]


def _service_backend_state(settings: Any = None) -> dict[str, Any]:
    """The paid-service rung's live state (key presence, probed now)."""
    try:
        from ..tools.captcha import ServiceBackend
        svc = ServiceBackend(settings=settings)
        avail = bool(svc.available())
        return {"available": avail,
                "detail": "a solving-service key is configured "
                          "(CAPTCHA_API_KEY)" if avail else
                          "no solving-service key configured "
                          "(CAPTCHA_API_KEY unset)"}
    except Exception as exc:  # noqa: BLE001 - status never crashes
        return {"available": False,
                "detail": f"service probe failed: {exc}"}


def _row(kind: str, mechanism: str, local: str = "", *,
         available: bool, detail: str,
         fallbacks: str = "service → takeover") -> dict[str, Any]:
    return {"kind": kind, "mechanism": mechanism,
            "local_solver": local, "available": bool(available),
            "detail": detail, "fallbacks": fallbacks,
            "registered": kind_solver_for(kind) is not None}


#: Human-readable name for each mechanism code.
MECHANISMS = {
    "local": "local solver first, then service → takeover",
    "local_only": "local solver, then takeover (services sell no "
                  "solve for this kind)",
    "service": "solving service → owner takeover",
    "wait": "wait rung — the widget's own proof-of-work completes "
            "in a real browser session",
    "cloudflare": "state machine: managed wait → vault cookie → "
                  "owner ask",
    "takeover": "owner takeover (human solves in the live tab)",
}


def estate_status(settings: Any = None) -> dict[str, Any]:
    """The full kind→mechanism map with live availability probes.

    ``kinds`` maps every kind the estate knows — captcha.py's
    ``CaptchaKind.ALL`` plus the browser-layer kinds (``slider``,
    ``cloudflare``, ``cloudflare-challenge``, ``question_captcha``)
    — to ``{mechanism, local_solver, available, detail, fallbacks,
    registered}``. ``available`` answers "could the FIRST rung for
    this kind run right now on this machine".
    """
    kinds: dict[str, Any] = {}

    ocr = ocr_solver.ocr_status()
    kinds[CaptchaKind.IMAGE_CAPTCHA] = _row(
        CaptchaKind.IMAGE_CAPTCHA, "local", "local_ocr",
        available=bool(ocr["available"]), detail=str(ocr["detail"]))

    asr = audio_solver.audio_asr_status()
    kinds[CaptchaKind.AUDIO_CAPTCHA] = _row(
        CaptchaKind.AUDIO_CAPTCHA, "local", "local_audio",
        available=bool(asr["available"]), detail=str(asr["detail"]))

    ground = click_solver.grounding_status()
    kinds[CaptchaKind.RECAPTCHA_V2] = _row(
        CaptchaKind.RECAPTCHA_V2, "local", "local_click",
        available=bool(ground["available"]),
        detail="image-grid challenges: vision-grounded clicking "
               "locally; checkbox token challenges: "
               + str(ground["detail"])
               if ground["available"] else str(ground["detail"]))
    kinds[CaptchaKind.HCAPTCHA] = _row(
        CaptchaKind.HCAPTCHA, "local", "local_click",
        available=bool(ground["available"]),
        detail="click challenges: vision-grounded clicking "
               "locally; token challenges fall through. "
               + str(ground["detail"])
        if ground["available"] else str(ground["detail"]))

    slider = slider_solver.local_slider_status()
    slider_detail = str(slider.get("detail") or
                        slider.get("vision", {}).get("detail") or "")
    kinds[CaptchaKind.GEETEST] = _row(
        CaptchaKind.GEETEST, "local", "local_slider / local_click",
        available=bool(slider["available"] or ground["available"]),
        detail="dispatcher: slider markup → the vision gap solver; "
               "click markup → vision-grounded clicking")
    kinds["slider"] = _row(
        "slider", "local", "local_slider",
        available=bool(slider["available"]),
        detail=slider_detail)

    brain = question_solver.brain_status()
    kinds[question_solver.QUESTION_KIND] = _row(
        question_solver.QUESTION_KIND, "local", "local_brain",
        available=bool(brain["available"]),
        detail=str(brain["detail"]),
        fallbacks="takeover (services sell no solve for this kind)")

    kinds[cloudflare_solver.CLOUDFLARE_KIND] = _row(
        cloudflare_solver.CLOUDFLARE_KIND, "cloudflare",
        "local_cloudflare",
        available=True,
        detail="state machine: managed wait needs only a live "
               "tab; the vault-cookie rung needs a wired cookie "
               "store",
        fallbacks="owner ask → takeover")
    kinds[cloudflare_solver.CLOUDFLARE_LADDER_KIND] = _row(
        cloudflare_solver.CLOUDFLARE_LADDER_KIND, "cloudflare",
        "local_cloudflare (flow/backend seams)",
        available=True,
        detail="the browser ladder's signal name for the same wall "
               "— deliberately NOT kind-registered: the ladder "
               "drives it natively (wait rung = the managed wait, "
               "takeover manager = the owner ask, solved-session "
               "save = the clearance capture); the state-machine "
               "solver serves this wall under the 'cloudflare' "
               "kind and the local_cloudflare backend",
        fallbacks="ladder wait → takeover")

    service = _service_backend_state(settings)
    service_rows = {
        CaptchaKind.RECAPTCHA_V3: (
            "score-based — the site scores the session; no local "
            "solve exists, a service mints the token"),
        CaptchaKind.RECAPTCHA_ENTERPRISE: (
            "enterprise token challenge — a service mints the "
            "token"),
        CaptchaKind.TURNSTILE: (
            "managed token challenge — a service mints the token"),
        CaptchaKind.ARKOSE: (
            "challenge token — a service mints the token when a "
            "puzzle is served"),
        CaptchaKind.AWS_WAF: (
            "AWS WAF token challenge — no local solver; service / "
            "session persistence is the path"),
        CaptchaKind.SMARTCAPTCHA: (
            "Yandex SmartCaptcha token — a service mints the "
            "token"),
        CaptchaKind.PERIMETERX: (
            "bot-management wall — takeover-primary: a human "
            "session is the reliable solve"),
    }
    for kind, note in service_rows.items():
        kinds[kind] = _row(
            kind, "service", "",
            available=bool(service["available"]),
            detail=f"{note}. Service rung: {service['detail']}",
            fallbacks="takeover" if kind == CaptchaKind.PERIMETERX
            else "service → takeover")

    kinds[CaptchaKind.FRIENDLY] = _row(
        CaptchaKind.FRIENDLY, "wait", "",
        available=True,
        detail="proof-of-work widget — the browser ladder's wait "
               "rung lets its own crypto puzzle complete in-session "
               "(pinned by test); no token is bought for it",
        fallbacks="wait → takeover")

    kinds[CaptchaKind.UNKNOWN] = _row(
        CaptchaKind.UNKNOWN, "takeover", "",
        available=True,
        detail="unclassified walls route to owner takeover; "
               "Cloudflare interstitials arrive as unknown + "
               "provider metadata and are served by local_cloudflare "
               "through the backend seam",
        fallbacks="takeover")

    return {"kinds": kinds,
            "registered_kinds": registered_kinds(),
            "mechanisms": MECHANISMS}
