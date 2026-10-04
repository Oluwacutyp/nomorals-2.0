"""Captcha detection and solving for owner-directed browser automation.

Devon drives browser sessions (see :mod:`nomorals.tools.browser`) for the
owner — logins, forms, checkouts — and those flows regularly hit captchas.
This module gives her a way through without paging the owner every time:

* :func:`detect` — scan page HTML for reCAPTCHA v2/v2-audio/v3/enterprise,
  hCaptcha, Cloudflare Turnstile, Cloudflare challenge pages, GeeTest
  v3/v4, Arkose FunCaptcha, AWS WAF, FriendlyCaptcha, Yandex
  SmartCaptcha, PerimeterX px-captcha, and image/audio captchas (inline
  ``data:`` images are decoded on the spot so the solver can use them
  directly).
* :class:`CaptchaBackend` — pluggable solvers behind one interface:
  ``service`` (a 2captcha-shaped commercial solving API, key from the
  environment only), ``takeover`` (pause and hand the challenge to the
  owner), ``detect`` (report only, never solve).
* Every solve attempt is audit-logged (kind, page domain, sitekey,
  timestamp, backend, outcome). The API key and page content are never
  logged.

Credentials: ``CAPTCHA_API_KEY`` from the environment. Nothing here asks
for a key in chat, and the key never touches a log line.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

__all__ = [
    "CaptchaKind",
    "CaptchaChallenge",
    "CaptchaError",
    "SolveResult",
    "CaptchaBackend",
    "ServiceBackend",
    "TakeoverBackend",
    "DetectOnlyBackend",
    "SolverRateLimiter",
    "rate_limiter",
    "register_backend",
    "detect",
    "detect_in_session",
    "extract_image_captcha_urls",
    "fetch_image_bytes",
    "backend_for",
    "solve",
    "creator_solver_adapter",
    "register",
]

_log = get_logger(__name__)


# ── kinds ────────────────────────────────────────────────────────────────────


class CaptchaKind:
    """Challenge taxonomy. Plain strings so they serialize cleanly."""

    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    RECAPTCHA_ENTERPRISE = "recaptcha_enterprise"
    HCAPTCHA = "hcaptcha"
    TURNSTILE = "turnstile"
    GEETEST = "geetest"              # GeeTest v3 slider / v4 behavioral
    ARKOSE = "arkose"                # Arkose FunCaptcha
    AWS_WAF = "aws_waf"              # AWS WAF captcha widget
    FRIENDLY = "friendly"            # FriendlyCaptcha (PoW puzzle)
    SMARTCAPTCHA = "smartcaptcha"    # Yandex SmartCaptcha
    PERIMETERX = "perimeterx"        # PerimeterX/HUMAN px-captcha
    IMAGE_CAPTCHA = "image_captcha"
    AUDIO_CAPTCHA = "audio_captcha"
    UNKNOWN = "unknown"

    ALL = (
        RECAPTCHA_V2, RECAPTCHA_V3, RECAPTCHA_ENTERPRISE,
        HCAPTCHA, TURNSTILE, GEETEST, ARKOSE, AWS_WAF, FRIENDLY,
        SMARTCAPTCHA, PERIMETERX, IMAGE_CAPTCHA, AUDIO_CAPTCHA, UNKNOWN,
    )

    #: kinds that resolve to a token the browser submits with the form
    TOKEN_KINDS = frozenset({
        RECAPTCHA_V2, RECAPTCHA_V3, RECAPTCHA_ENTERPRISE,
        HCAPTCHA, TURNSTILE, ARKOSE,
    })

    #: field name the solved token goes into on submit
    TOKEN_FIELD = {
        RECAPTCHA_V2: "g-recaptcha-response",
        RECAPTCHA_V3: "g-recaptcha-response",
        RECAPTCHA_ENTERPRISE: "g-recaptcha-response",
        HCAPTCHA: "h-captcha-response",
        TURNSTILE: "cf-turnstile-response",
        ARKOSE: "fc-token",
    }

    #: kinds that resolve to readable text (typed into a captcha field)
    TEXT_KINDS = frozenset({IMAGE_CAPTCHA, AUDIO_CAPTCHA})


class CaptchaError(ToolError):
    """Raised when a solve fails hard (bad key, API error, timeout)."""


@dataclass
class CaptchaChallenge:
    """One detected challenge."""

    kind: str
    sitekey: str = ""
    page_url: str = ""
    image_url: str = ""
    image_bytes: bytes = b""
    action: str = ""          # reCAPTCHA v3 action name, when known
    min_score: float = 0.3    # reCAPTCHA v3 score floor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return (urllib.parse.urlparse(self.page_url).hostname or "").lower()

    def summary(self) -> dict[str, Any]:
        img_url = self.image_url
        if img_url.startswith("data:"):
            # inline image — never dump kilobytes of base64 into a summary
            img_url = img_url[:64] + "…<inline image>"
        return {
            "kind": self.kind,
            "sitekey": self.sitekey,
            "domain": self.domain,
            "image_url": img_url,
            "image_bytes": len(self.image_bytes),
            "action": self.action,
        }


@dataclass
class SolveResult:
    """Outcome of one solve attempt."""

    ok: bool
    kind: str
    backend: str
    token: str = ""           # the g-recaptcha-response / h-captcha-response value
    text: str = ""            # solved text for image captchas
    takeover: bool = False    # True when the owner must solve it by hand
    elapsed_ms: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ok": self.ok, "kind": self.kind, "backend": self.backend,
            "takeover": self.takeover, "elapsed_ms": self.elapsed_ms,
            "detail": self.detail,
        }
        if self.token:
            d["token"] = self.token
        if self.text:
            d["text"] = self.text
        return d


# ── detection ────────────────────────────────────────────────────────────────

_SITEKEY_RE = re.compile(r'data-sitekey\s*=\s*["\']([^"\']+)["\']', re.I)
_V3_RENDER_RE = re.compile(
    r"recaptcha/api\.js\?render=([A-Za-z0-9_\-]+)", re.I)
_V3_EXECUTE_RE = re.compile(
    r"grecaptcha(?:\.enterprise)?\.execute\(\s*['\"]([A-Za-z0-9_\-]+)['\"]", re.I)
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.I)
_IMG_SRC_RE = re.compile(r'src\s*=\s*["\']([^"\']+)["\']', re.I)
# provider-specific identifiers
_GT_RE = re.compile(r"\bgt\s*[:=]\s*[\"']([A-Za-z0-9]{20,})[\"']", re.I)
_GT_CHALLENGE_RE = re.compile(
    r"\bchallenge\s*[:=]\s*[\"']([A-Za-z0-9_\-]{20,})[\"']", re.I)
_CAPTCHAID_RE = re.compile(
    r"\bcaptch?a-?id\s*[:=]\s*[\"']([A-Za-z0-9]{8,})[\"']", re.I)
_PKEY_RE = re.compile(
    r"(?:data-pkey|pkey)\s*[:=]\s*[\"']([A-Za-z0-9\-]{8,})[\"']", re.I)


def _gt_params(html: str) -> tuple[str, str]:
    """(gt, challenge) for a GeeTest v3 inline config, '' when absent."""
    gt_m = _GT_RE.search(html)
    ch_m = _GT_CHALLENGE_RE.search(html)
    return ((gt_m.group(1) if gt_m else ""),
            (ch_m.group(1) if ch_m else ""))


def _captcha_id(html: str) -> str:
    m = _CAPTCHAID_RE.search(html)
    return m.group(1).strip() if m else ""


def _pkey(html: str) -> str:
    m = _PKEY_RE.search(html)
    return m.group(1).strip() if m else ""


def _has(html: str, *needles: str) -> bool:
    low = html.lower()
    return any(n.lower() in low for n in needles)


def _decode_data_uri(src: str) -> bytes:
    """Decode a ``data:image/...;base64,...`` URI to raw bytes.

    Returns b"" when the URI is not base64-encoded image data.
    """
    try:
        header, _, payload = src.partition(",")
        if ";base64" not in header.lower() or not payload:
            return b""
        return base64.b64decode(payload, validate=True)
    except Exception:  # noqa: BLE001 — malformed data URI, not a captcha
        return b""


def detect(html: str, url: str = "", fetch_bytes: bool = False) -> list[CaptchaChallenge]:
    """Scan page HTML, return every captcha challenge found.

    Pure function when ``fetch_bytes=False`` (default) — no network, no
    side effects. ``sitekey`` values are public site keys, safe to surface.

    Inline (``data:`` URI) captcha images are decoded into
    ``image_bytes`` on the spot, so the service backend can solve them
    without a second fetch. Pass ``fetch_bytes=True`` to also download
    the bytes of remote (http/https) captcha images — best effort, one
    short-timeout request per image; failures leave ``image_bytes``
    empty rather than failing detection.
    """
    html = html or ""
    out: list[CaptchaChallenge] = []
    seen: set[tuple[str, str]] = set()

    def add(ch: CaptchaChallenge) -> None:
        key = (ch.kind, ch.sitekey or ch.image_url)
        if key not in seen:
            seen.add(key)
            out.append(ch)

    is_enterprise = _has(html, "recaptcha/enterprise.js", "grecaptcha.enterprise")
    is_recaptcha = _has(html, "google.com/recaptcha", "gstatic.com/recaptcha",
                        "g-recaptcha")
    is_hcaptcha = _has(html, "hcaptcha.com", "h-captcha")
    is_turnstile = _has(html, "challenges.cloudflare.com/turnstile",
                        "cf-turnstile")
    is_geetest = _has(html, "geetest.com", "initgeetest", "geetestv4",
                      "geetest_v4", "gt.js")
    is_arkose = _has(html, "arkoselabs.com", "funcaptcha", "fc-token")
    is_awswaf = _has(html, "awswaf", "aws-waf")
    is_friendly = _has(html, "friendlycaptcha", "frc-captcha")
    is_smartcaptcha = _has(html, "smartcaptcha",
                           "smart-captcha.yandexcloud.net")
    is_perimeterx = _has(html, "perimeterx", "px-captcha", "_px/")

    # reCAPTCHA enterprise / v2 — data-sitekey driven. Providers that share
    # the data-sitekey markup (hCaptcha, Turnstile, FriendlyCaptcha,
    # Yandex SmartCaptcha) are disambiguated by surrounding markup, in
    # order of specificity.
    for m in _SITEKEY_RE.finditer(html):
        sitekey = m.group(1).strip()
        if not sitekey or sitekey == "explicit":
            continue
        window = html[max(0, m.start() - 600):m.start()].lower()
        if is_enterprise or "enterprise" in window:
            add(CaptchaChallenge(CaptchaKind.RECAPTCHA_ENTERPRISE,
                                 sitekey=sitekey, page_url=url))
        elif is_geetest and "geetest" in window:
            # GeeTest v3 inline config (gt/challenge live in metadata)
            gt, challenge = _gt_params(html)
            add(CaptchaChallenge(CaptchaKind.GEETEST, sitekey=sitekey,
                                 page_url=url,
                                 metadata={"gt": gt, "challenge": challenge,
                                           "variant": "v3"}))
        elif is_hcaptcha or "h-captcha" in window:
            add(CaptchaChallenge(CaptchaKind.HCAPTCHA,
                                 sitekey=sitekey, page_url=url))
        elif is_turnstile or "cf-turnstile" in window:
            add(CaptchaChallenge(CaptchaKind.TURNSTILE,
                                 sitekey=sitekey, page_url=url))
        elif is_friendly or "frc-captcha" in window:
            add(CaptchaChallenge(CaptchaKind.FRIENDLY,
                                 sitekey=sitekey, page_url=url,
                                 metadata={"note": "proof-of-work puzzle"}))
        elif is_smartcaptcha or "smartcaptcha" in window:
            add(CaptchaChallenge(CaptchaKind.SMARTCAPTCHA,
                                 sitekey=sitekey, page_url=url))
        elif is_recaptcha or "g-recaptcha" in window:
            add(CaptchaChallenge(CaptchaKind.RECAPTCHA_V2,
                                 sitekey=sitekey, page_url=url))

    # reCAPTCHA v3 — render= key or grecaptcha.execute('key')
    for m in _V3_RENDER_RE.finditer(html):
        sitekey = m.group(1)
        if sitekey != "explicit":
            kind = (CaptchaKind.RECAPTCHA_ENTERPRISE if is_enterprise
                    else CaptchaKind.RECAPTCHA_V3)
            add(CaptchaChallenge(kind, sitekey=sitekey, page_url=url))
    for m in _V3_EXECUTE_RE.finditer(html):
        sitekey = m.group(1)
        kind = (CaptchaKind.RECAPTCHA_ENTERPRISE if is_enterprise
                else CaptchaKind.RECAPTCHA_V3)
        add(CaptchaChallenge(kind, sitekey=sitekey, page_url=url))

    # GeeTest v4 — initGeetest4({captchaId: "..."}) or a captcha-id attr.
    if is_geetest:
        cid = _captcha_id(html)
        gt, challenge = _gt_params(html)
        if cid:
            add(CaptchaChallenge(
                CaptchaKind.GEETEST, sitekey=cid, page_url=url,
                metadata={"gt": "", "challenge": "", "variant": "v4"}))
        elif gt and challenge:
            add(CaptchaChallenge(
                CaptchaKind.GEETEST, sitekey=gt, page_url=url,
                metadata={"gt": gt, "challenge": challenge,
                          "variant": "v3"}))
        else:
            add(CaptchaChallenge(
                CaptchaKind.GEETEST, page_url=url,
                metadata={"note": "geetest markup present, "
                                  "no gt/captchaId extracted"}))

    # Arkose FunCaptcha — api.js with a public key (data-pkey).
    if is_arkose:
        pkey = _pkey(html)
        add(CaptchaChallenge(
            CaptchaKind.ARKOSE, sitekey=pkey, page_url=url,
            metadata={"note": "public key extracted" if pkey
                              else "no public key extracted"}))

    # AWS WAF captcha — the captcha.js widget on a WAF-fronted page.
    if is_awswaf and _has(html, "captcha"):
        add(CaptchaChallenge(
            CaptchaKind.AWS_WAF, page_url=url,
            metadata={"provider": "aws_waf",
                      "note": "WAF captcha widget — interactive, "
                              "token injected by the widget itself"}))

    # PerimeterX / HUMAN px-captcha — cookie-clearance interstitial.
    if is_perimeterx:
        add(CaptchaChallenge(
            CaptchaKind.PERIMETERX, page_url=url,
            metadata={"provider": "perimeterx",
                      "note": "interactive challenge page"}))

    # Cloudflare challenge / "verify you are human" interstitials
    if _has(html, "cf-challenge", "__cf_chl", "cf_clearance") and _has(
            html, "verifying you are human", "verify you are human",
            "just a moment"):
        add(CaptchaChallenge(
            CaptchaKind.UNKNOWN, page_url=url,
            metadata={"provider": "cloudflare",
                      "note": "interactive challenge page"}))

    # image captchas — <img> whose src/alt/class/id smells like a captcha,
    # including inline data: URIs (decoded into image_bytes so the solver
    # can use them directly).
    for img_url in extract_image_captcha_urls(html, url):
        img_bytes = b""
        if img_url.startswith("data:"):
            img_bytes = _decode_data_uri(img_url)
        elif fetch_bytes and img_url.startswith(("http://", "https://")):
            try:
                img_bytes = fetch_image_bytes(img_url, timeout=10.0)
            except Exception:  # noqa: BLE001 — best effort
                img_bytes = b""
        add(CaptchaChallenge(CaptchaKind.IMAGE_CAPTCHA, page_url=url,
                             image_url=img_url, image_bytes=img_bytes))

    # audio captchas — reCAPTCHA v2's "audio challenge" fallback (and any
    # <audio> element in a captcha context). The audio bytes are the
    # challenge payload; services solve them the same way as images.
    if _has(html, "audio challenge", "recaptcha/api2/payload/audio",
            "get audio challenge"):
        audio_url = ""
        audio_bytes = b""
        m = re.search(
            r"<audio\b[^>]*src\s*=\s*[\"']([^\"']+)[\"']", html, re.I)
        if m:
            audio_url = urllib.parse.urljoin(url, m.group(1).strip())
            if audio_url.startswith("data:"):
                audio_bytes = _decode_data_uri(audio_url)
            elif fetch_bytes and audio_url.startswith(
                    ("http://", "https://")):
                try:
                    audio_bytes = fetch_image_bytes(audio_url, timeout=10.0)
                except Exception:  # noqa: BLE001 — best effort
                    audio_bytes = b""
        add(CaptchaChallenge(CaptchaKind.AUDIO_CAPTCHA, page_url=url,
                             image_url=audio_url, image_bytes=audio_bytes,
                             metadata={"note": "audio challenge"}))

    return out


def extract_image_captcha_urls(html: str, base_url: str = "") -> list[str]:
    """Pull candidate image-captcha URLs out of page HTML.

    Inline ``data:`` URIs are kept (not resolved against ``base_url``) so
    callers can decode the image bytes directly.
    """
    urls: list[str] = []
    for tag in _IMG_TAG_RE.findall(html or ""):
        low = tag.lower()
        if "captcha" not in low:
            continue
        m = _IMG_SRC_RE.search(tag)
        if not m:
            continue
        src = m.group(1).strip()
        if src.startswith("data:"):
            urls.append(src)
            continue
        urls.append(urllib.parse.urljoin(base_url, src))
    # de-dupe, keep order
    return list(dict.fromkeys(urls))


def detect_in_session(session: Any) -> list[CaptchaChallenge]:
    """Run :func:`detect` against a live
    :class:`nomorals.tools.browser.BrowserSession`.

    Accepts anything with ``url`` and ``_raw`` attributes so it stays
    decoupled from the browser module's import graph.
    """
    url = getattr(session, "url", "") or ""
    raw = getattr(session, "_raw", "") or ""
    return detect(raw, url)


# ── backends ─────────────────────────────────────────────────────────────────


class CaptchaBackend:
    """One way of turning a challenge into a token/text."""

    name = "base"

    def available(self) -> bool:
        return True

    def solve(self, challenge: CaptchaChallenge) -> SolveResult:
        raise NotImplementedError


class DetectOnlyBackend(CaptchaBackend):
    """Never solves — just reports kind + sitekey so the agent can decide."""

    name = "detect"

    def solve(self, challenge: CaptchaChallenge) -> SolveResult:
        return SolveResult(
            ok=False, kind=challenge.kind, backend=self.name,
            detail=(f"detected {challenge.kind} "
                    f"(sitekey {challenge.sitekey or 'n/a'}, "
                    f"domain {challenge.domain or 'n/a'}) — "
                    "detect-only backend, no solve attempted"))


class TakeoverBackend(CaptchaBackend):
    """Pause the session and hand the challenge to the owner.

    Used when no solving service is configured (or the owner prefers to
    click the checkbox herself). Never auto-solves anything. The
    ``detail`` carries everything the owner needs: where the challenge
    is, what kind it is, and exactly what to do.
    """

    name = "takeover"

    _HINTS = {
        CaptchaKind.RECAPTCHA_V2: "tick the 'I'm not a robot' checkbox",
        CaptchaKind.RECAPTCHA_ENTERPRISE: "complete the reCAPTCHA widget",
        CaptchaKind.HCAPTCHA: "complete the hCaptcha checkbox",
        CaptchaKind.TURNSTILE: "tick the Cloudflare checkbox",
        CaptchaKind.GEETEST: "drag the slider (or complete the GeeTest "
                             "puzzle) in the page",
        CaptchaKind.ARKOSE: "complete the FunCaptcha puzzle",
        CaptchaKind.AWS_WAF: "complete the AWS WAF captcha widget",
        CaptchaKind.FRIENDLY: "wait for the FriendlyCaptcha proof-of-work "
                              "(solves itself, just wait)",
        CaptchaKind.SMARTCAPTCHA: "complete the SmartCaptcha",
        CaptchaKind.PERIMETERX: "complete the PerimeterX challenge",
        CaptchaKind.IMAGE_CAPTCHA: "read the image and type the characters",
        CaptchaKind.AUDIO_CAPTCHA: "listen to the audio and type the words",
        CaptchaKind.UNKNOWN: "complete the challenge in the browser",
    }

    def solve(self, challenge: CaptchaChallenge) -> SolveResult:
        hint = self._HINTS.get(challenge.kind, self._HINTS[CaptchaKind.UNKNOWN])
        where = challenge.page_url or challenge.domain or "the page"
        detail = (f"owner takeover: {challenge.kind} on {where} — "
                  f"{hint}. Reply when done and the automation will resume.")
        return SolveResult(
            ok=False, kind=challenge.kind, backend=self.name, takeover=True,
            detail=detail)


# ── rate limiting ──────────────────────────────────────────────────────────────

def _rate_limit_path(settings: Any = None) -> str:
    if settings is not None:
        try:
            return str(settings.resolve("data/captcha/rate_limit.json"))
        except Exception:  # noqa: BLE001
            pass
    root = os.environ.get("NOMORALS_CAPTCHA_DIR") or os.path.join(
        os.path.expanduser("~"), ".config", "nomorals", "captcha")
    return os.path.join(root, "rate_limit.json")


class SolverRateLimiter:
    """Keep the solver from hammering a commercial solving service.

    Budgets are configurable through the environment and persist to a
    JSON state file so they survive restarts:

    * ``NM_CAPTCHA_PER_MINUTE`` — max task submissions per rolling
      minute (default 30).
    * ``NM_CAPTCHA_PER_DAY`` — max task submissions per rolling
      24 hours (default 1000).
    * ``NM_CAPTCHA_ERROR_COOLDOWN`` — base seconds of backoff after a
      failed solve; doubles per consecutive failure, capped at 30
      minutes (default 30).

    Hard-stop failures (bad key, zero balance, banned IP) trip a
    6-hour global cooldown: no point spending against a dead account.
    """

    _HARD_COOLDOWN_S = 6 * 3600
    _BACKOFF_CAP_S = 1800

    def __init__(self, settings: Any = None, *,
                 per_minute: int | None = None,
                 per_day: int | None = None,
                 error_cooldown_s: float | None = None) -> None:
        self._path = _rate_limit_path(settings)
        self._per_minute = per_minute if per_minute is not None else int(
            os.environ.get("NM_CAPTCHA_PER_MINUTE", "30"))
        self._per_day = per_day if per_day is not None else int(
            os.environ.get("NM_CAPTCHA_PER_DAY", "1000"))
        self._error_cooldown = (error_cooldown_s if error_cooldown_s
                                is not None else float(
                                    os.environ.get(
                                        "NM_CAPTCHA_ERROR_COOLDOWN", "30")))

    # -- state ----------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                state = json.load(fh)
            if isinstance(state, dict):
                return state
        except Exception:  # noqa: BLE001 — missing/corrupt state is fine
            pass
        return {}

    def _save(self, state: dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
            os.replace(tmp, self._path)
        except Exception:  # noqa: BLE001 — best effort
            _log.warning("captcha rate-limit state write failed")

    # -- checks -----------------------------------------------------------------
    def check(self) -> None:
        """Raise :class:`CaptchaError` when a new solve must not be sent."""
        now = time.time()
        state = self._load()
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        if now < cooldown_until:
            wait = int(cooldown_until - now)
            raise CaptchaError(
                f"solver cooling down — next solve allowed in {wait}s")
        stamps = [s for s in state.get("submits", [])
                  if now - float(s) < 86400.0]
        if len(stamps) >= self._per_day:
            raise CaptchaError(
                f"daily solving budget reached ({self._per_day})")
        minute = [s for s in stamps if now - float(s) < 60.0]
        if len(minute) >= self._per_minute:
            raise CaptchaError(
                f"solve rate too high (>{self._per_minute}/min) — backing off")

    def record_submit(self) -> None:
        state = self._load()
        now = time.time()
        stamps = [s for s in state.get("submits", [])
                  if now - float(s) < 86400.0]
        stamps.append(now)
        state["submits"] = stamps
        self._save(state)

    def record_success(self) -> None:
        state = self._load()
        state["consecutive_failures"] = 0
        self._save(state)

    def record_failure(self, hard: bool = False) -> None:
        """Note a failed solve. ``hard`` = account-level failure —
        trip the long cooldown."""
        state = self._load()
        if hard:
            state["cooldown_until"] = time.time() + self._HARD_COOLDOWN_S
            state["hard_failure"] = True
            _log.warning("captcha solver hard-stop: 6h cooldown")
        else:
            failures = int(state.get("consecutive_failures", 0) or 0) + 1
            state["consecutive_failures"] = failures
            backoff = min(self._error_cooldown * (2 ** (failures - 1)),
                          self._BACKOFF_CAP_S)
            state["cooldown_until"] = time.time() + backoff
            _log.info("captcha solver backoff %.0fs after failure #%d",
                      backoff, failures)
        self._save(state)

    def status(self) -> dict[str, Any]:
        """Snapshot for ``nm captcha status``."""
        now = time.time()
        state = self._load()
        stamps = [s for s in state.get("submits", [])
                  if now - float(s) < 86400.0]
        return {
            "submits_last_24h": len(stamps),
            "per_minute": self._per_minute,
            "per_day": self._per_day,
            "cooldown_until": state.get("cooldown_until", 0.0),
            "cooling_down": now < float(state.get("cooldown_until", 0.0) or 0),
            "consecutive_failures": int(
                state.get("consecutive_failures", 0) or 0),
            "hard_failure": bool(state.get("hard_failure", False)),
        }


_LIMITERS: dict[str, SolverRateLimiter] = {}


def rate_limiter(settings: Any = None) -> SolverRateLimiter:
    """Process-wide shared limiter (one per state-file path)."""
    path = _rate_limit_path(settings)
    if path not in _LIMITERS:
        _LIMITERS[path] = SolverRateLimiter(settings)
    return _LIMITERS[path]


# 2captcha-shaped commercial solving service ────────────────────────────────

_DEFAULT_API_URL = "https://2captcha.com"
_POLL_INTERVAL = 5.0
_MAX_POLLS = 24  # ~2 minutes

_METHOD_FOR_KIND = {
    CaptchaKind.RECAPTCHA_V2: "userrecaptcha",
    CaptchaKind.RECAPTCHA_V3: "userrecaptcha",
    CaptchaKind.RECAPTCHA_ENTERPRISE: "userrecaptcha",
    CaptchaKind.HCAPTCHA: "hcaptcha",
    CaptchaKind.TURNSTILE: "turnstile",
    CaptchaKind.GEETEST: "geetest",
    CaptchaKind.ARKOSE: "funcaptcha",
    CaptchaKind.IMAGE_CAPTCHA: "base64",
    CaptchaKind.AUDIO_CAPTCHA: "audio",
}

#: API errors that mean the account itself is dead — never retry, and
#: trip the 6h limiter cooldown instead of spending more tasks.
_HARD_STOP_ERRORS = {
    "ERROR_ZERO_BALANCE",
    "ERROR_KEY_DOES_NOT_EXIST",
    "ERROR_WRONG_USER_KEY",
    "ERROR_IP_NOT_ALLOWED",
    "ERROR_IP_BLOCKED",
}


def _hard_error(error_text: str) -> bool:
    """Is this a 2captcha-shaped error code we must never retry?"""
    up = (error_text or "").upper()
    return any(code in up for code in _HARD_STOP_ERRORS)


_FRIENDLY_ERRORS = {
    "ERROR_ZERO_BALANCE": "solving account has zero balance — "
                          "add funds before retrying",
    "ERROR_KEY_DOES_NOT_EXIST": "CAPTCHA_API_KEY is invalid/unknown — "
                                "check the key",
    "ERROR_WRONG_USER_KEY": "CAPTCHA_API_KEY is wrong — check the key",
    "ERROR_IP_NOT_ALLOWED": "solving account blocks this IP",
    "ERROR_IP_BLOCKED": "solving account blocked this IP",
    "ERROR_NO_SLOT_AVAILABLE": "solver busy — retry shortly",
    "ERROR_TASK_ABSENT": "task expired at the solver — retry",
    "ERROR_WRONG_CAPTCHA_ID": "stale task id — retry",
    "ERROR_IMAGE_TYPE_NOT_SUPPORTED": "image type unsupported — "
                                      "convert to PNG/JPG first",
    "ERROR_CAPTCHA_UNSOLVABLE": "solver workers could not solve it",
    "ERROR_BAD_TOKEN_OR_PAGEURL": "sitekey/pageurl rejected by the solver",
}


def _friendly_error(error_text: str) -> str:
    """Turn a raw solver error code into a human-usable message."""
    up = (error_text or "").strip().upper()
    msg = _FRIENDLY_ERRORS.get(up)
    if msg:
        return f"solver API error: {msg}"
    short = (error_text or "").strip()[:120] or "unknown error"
    return f"solver API rejected the request ({short})"


class ServiceBackend(CaptchaBackend):
    """Commercial solving service behind the 2captcha in.php/res.php shape.

    The API key comes **only** from the ``CAPTCHA_API_KEY`` environment
    variable (``CAPTCHA_API_URL`` optionally overrides the endpoint for
    compatible services, e.g. capmonster — its 2captcha-compatible mode
    uses the same in.php/res.php calls). The key is never logged, never
    stored in the audit trail, and never echoed back in errors.

    A shared :class:`SolverRateLimiter` guards submissions: per-minute
    and per-day budgets plus exponential backoff on failures. Hard
    account-level errors (bad key, zero balance) trip a 6-hour global
    cooldown so we never burn against a dead account.
    """

    name = "service"

    def __init__(self, api_key: str = "", api_url: str = "",
                 sleeper: Callable[[float], None] | None = None,
                 limiter: "SolverRateLimiter | None" = None,
                 proxy: str = "", settings: Any = None) -> None:
        self._key = api_key or os.environ.get("CAPTCHA_API_KEY", "")
        self._api_url = (api_url or os.environ.get("CAPTCHA_API_URL", "")
                         or _DEFAULT_API_URL).rstrip("/")
        self._sleep = sleeper or time.sleep
        self._limiter = limiter if limiter is not None else rate_limiter(
            settings)
        self._proxy = proxy or os.environ.get("CAPTCHA_PROXY", "")

    # -- availability -------------------------------------------------------
    def available(self) -> bool:
        return bool(self._key)

    # -- HTTP ---------------------------------------------------------------
    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{self._api_url}{path}?{qs}",
            headers={"User-Agent": "nomorals-captcha/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            # a 429 from the solving service is a hard stop for this task
            if exc.code == 429:
                self._limiter.record_failure(hard=False)
                raise CaptchaError(
                    "solver API rate-limited us (429) — backing off") from exc
            self._limiter.record_failure()
            raise CaptchaError(f"solver API unreachable: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            self._limiter.record_failure()
            raise CaptchaError(f"solver API unreachable: {exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # plain-text shape: OK|<id>, CAPCHA_NOT_READY, or <ERROR>
            if body.startswith("OK|"):
                return {"status": 1, "request": body[3:].strip()}
            if body.strip() == "CAPCHA_NOT_READY":
                return {"status": 0, "request": "CAPCHA_NOT_READY"}
            self._limiter.record_failure(hard=_hard_error(body))
            raise CaptchaError(_friendly_error(body))
        if data.get("status") != 1 and data.get("request") != "CAPCHA_NOT_READY":
            error = str(data.get("request", ""))
            self._limiter.record_failure(hard=_hard_error(error))
            raise CaptchaError(_friendly_error(error))
        return data

    # -- solve --------------------------------------------------------------
    def _task_params(self, challenge: CaptchaChallenge,
                     method: str) -> dict[str, str]:
        params: dict[str, str] = {"key": self._key, "json": "1",
                                  "method": method}
        if method in ("base64", "audio"):
            if not challenge.image_bytes:
                raise CaptchaError(
                    f"{challenge.kind} needs audio/image bytes "
                    "(detect(..., fetch_bytes=True) or supply image_bytes)")
            params["body"] = base64.b64encode(challenge.image_bytes).decode()
        elif method == "funcaptcha":
            if not challenge.sitekey:
                raise CaptchaError("arkose/funcaptcha needs a public key")
            params["publickey"] = challenge.sitekey
            domain = challenge.domain or ""
            params["surl"] = (challenge.metadata.get("surl")
                              or f"https://{domain}" if domain else "")
            params["pageurl"] = challenge.page_url or "about:blank"
        elif method == "geetest":
            variant = challenge.metadata.get("variant", "v4")
            if variant == "v3":
                gt = challenge.metadata.get("gt") or challenge.sitekey
                ch = challenge.metadata.get("challenge", "")
                if not gt or not ch:
                    raise CaptchaError(
                        "geetest v3 needs gt + challenge parameters")
                params["gt"] = gt
                params["challenge"] = ch
                params["geetest"] = "1"
            else:
                if not challenge.sitekey:
                    raise CaptchaError("geetest v4 needs a captchaId")
                params = {"key": self._key, "json": "1",
                          "method": "geetest_v4",
                          "captcha_id": challenge.sitekey}
            params.setdefault("pageurl", challenge.page_url or "about:blank")
        else:
            if not challenge.sitekey:
                raise CaptchaError(f"{challenge.kind} needs a sitekey")
            if method == "userrecaptcha":
                params["googlekey"] = challenge.sitekey
            else:
                params["sitekey"] = challenge.sitekey
            params["pageurl"] = challenge.page_url or "about:blank"
            if challenge.kind == CaptchaKind.RECAPTCHA_V3:
                params["version"] = "v3"
                params["action"] = challenge.action or "verify"
                params["min_score"] = str(challenge.min_score)
            elif challenge.kind == CaptchaKind.RECAPTCHA_ENTERPRISE:
                params["enterprise"] = "1"
        if self._proxy and method not in ("base64", "audio"):
            ptype, _, paddr = self._proxy.partition("://")
            params["proxy"] = paddr or self._proxy
            params["proxytype"] = (ptype.upper() if ptype in
                                   ("http", "https", "socks4", "socks5")
                                   else "HTTP")
        return params

    def solve(self, challenge: CaptchaChallenge) -> SolveResult:
        started = time.time()
        if not self.available():
            raise CaptchaError(
                "no CAPTCHA_API_KEY configured — set the environment "
                "variable or use the takeover backend")
        method = _METHOD_FOR_KIND.get(challenge.kind)
        if not method:
            raise CaptchaError(f"service backend cannot solve kind "
                               f"{challenge.kind!r} "
                               f"(try takeover)")

        params = self._task_params(challenge, method)

        # rate-limit check happens right before submission, and the
        # submission is recorded, so rapid-fire callers get blocked.
        self._limiter.check()
        self._limiter.record_submit()

        created = self._get("/in.php", params)
        task_id = str(created["request"])

        token = ""
        for _ in range(_MAX_POLLS):
            self._sleep(_POLL_INTERVAL)
            got = self._get("/res.php", {"key": self._key, "json": "1",
                                         "action": "get", "id": task_id})
            if got.get("status") == 1:
                token = str(got["request"])
                break
            # status 0 + CAPCHA_NOT_READY → keep polling; anything else
            # already raised inside _get.
        if not token:
            self._limiter.record_failure()
            raise CaptchaError("solver timed out waiting for a token")

        self._limiter.record_success()
        elapsed = int((time.time() - started) * 1000)
        if challenge.kind in (CaptchaKind.IMAGE_CAPTCHA,
                              CaptchaKind.AUDIO_CAPTCHA):
            return SolveResult(ok=True, kind=challenge.kind,
                               backend=self.name, text=token,
                               elapsed_ms=elapsed)
        if challenge.kind == CaptchaKind.ARKOSE:
            # funcaptcha answers are tokens for the fc-token field
            return SolveResult(ok=True, kind=challenge.kind,
                               backend=self.name, token=token,
                               elapsed_ms=elapsed)
        return SolveResult(ok=True, kind=challenge.kind, backend=self.name,
                           token=token, elapsed_ms=elapsed)


# ── audit trail ──────────────────────────────────────────────────────────────

def _audit_path(settings: Any = None) -> str:
    if settings is not None:
        try:
            return str(settings.resolve("data/captcha/audit.jsonl"))
        except Exception:  # noqa: BLE001
            pass
    root = os.environ.get("NOMORALS_CAPTCHA_DIR") or os.path.join(
        os.path.expanduser("~"), ".config", "nomorals", "captcha")
    return os.path.join(root, "audit.jsonl")


def _audit(entry: dict[str, Any], settings: Any = None) -> None:
    """Append one JSONL audit line. Best-effort: never breaks a solve."""
    try:
        path = _audit_path(settings)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001
        _log.warning("captcha audit write failed")


# ── orchestration ────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[CaptchaBackend]] = {
    "service": ServiceBackend,
    "takeover": TakeoverBackend,
    "detect": DetectOnlyBackend,
}


def register_backend(name: str, cls: type[CaptchaBackend]) -> None:
    """Register a custom backend (e.g. a local OCR solver) under ``name``.

    ``cls`` must subclass :class:`CaptchaBackend`. Use with
    ``backend="<name>"`` in :func:`solve`.
    """
    if not (isinstance(name, str) and name.strip()):
        raise CaptchaError("backend name must be a non-empty string")
    if not (isinstance(cls, type) and issubclass(cls, CaptchaBackend)):
        raise CaptchaError("backend class must subclass CaptchaBackend")
    _BACKENDS[name.strip().lower()] = cls


def backend_for(name: str, **kwargs: Any) -> CaptchaBackend:
    """Build a backend by name; ``auto`` picks service when a key exists,
    otherwise takeover."""
    name = (name or "auto").lower()
    if name == "auto":
        name = "service" if ServiceBackend(**kwargs).available() else "takeover"
    cls = _BACKENDS.get(name)
    if cls is None:
        raise CaptchaError(f"unknown captcha backend {name!r} "
                           f"(want one of: {sorted(_BACKENDS)})")
    return cls(**kwargs)


def _notify_takeover(challenge: CaptchaChallenge, detail: str,
                     context: Any = None) -> dict[str, Any]:
    """Ping the owner when auto-solve fails or was never available.

    Best-effort: never raises. Without a live gateway the notification
    is persisted so the owner still sees it (redeliverable later).
    Identical challenges dedupe inside the notifier's 10-minute window.
    """
    title = (f"CAPTCHA needs you — {challenge.kind} "
             f"@ {challenge.domain or 'a page'}")
    body = (
        f"Auto-solve could not clear the challenge on "
        f"{challenge.page_url or challenge.domain or 'a page'}.\n\n"
        f"Kind: {challenge.kind}\n"
        f"Sitekey: {challenge.sitekey or 'n/a'}\n"
        f"{detail}\n\n"
        f"Solve it in the browser, then the flow will resume "
        f"(re-run the step — detection finds nothing once it's cleared)."
    )
    try:
        from ..agents.notifier import notify as _notify
        return _notify(context, "captcha", title, body, critical=True)
    except Exception as exc:  # noqa: BLE001 — notification never breaks flow
        _log.warning("captcha takeover notification failed: %s", exc)
        return {"delivered": False, "error": str(exc)}


def solve(challenge: CaptchaChallenge, backend: str = "auto",
          settings: Any = None, solver_enabled: bool = True,
          notify_owner: bool = True, context: Any = None,
          **backend_kwargs: Any) -> SolveResult:
    """Solve one challenge through the named backend, audit-logging it.

    The audit entry carries kind, domain, sitekey, timestamp, backend,
    outcome — never the API key, never page content.

    When backend="auto" (default) and solver_enabled=True (default):
    tries the service backend first; if it fails, falls back to takeover
    (owner solves by hand). Set solver_enabled=False to skip the service
    and go straight to takeover.

    ``notify_owner`` (default True): whenever the outcome is a
    takeover, ping the owner with the challenge details through the
    notifier. Pass ``context`` from a runtime context when one is
    available so the ping can deliver live; without one it is
    persisted for the owner to see.
    """
    backend_name = (backend or "auto").lower()
    # Resolve auto: service if enabled and key available, else takeover.
    # If service fails, fall back to takeover.
    fallback_to_takeover = False
    if backend_name == "auto":
        if solver_enabled and ServiceBackend(**backend_kwargs).available():
            backend_name = "service"
            fallback_to_takeover = True
        else:
            backend_name = "takeover"

    solver = backend_for(backend_name, **backend_kwargs)
    started = time.time()
    try:
        result = solver.solve(challenge)
        ok, detail = result.ok, result.detail
    except CaptchaError as exc:
        ok, detail = False, str(exc)
        result = SolveResult(ok=False, kind=challenge.kind,
                             backend=solver.name, detail=detail)
    except Exception as exc:  # noqa: BLE001 — audit, then re-raise
        _audit({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": challenge.kind, "domain": challenge.domain,
            "sitekey": challenge.sitekey, "backend": solver.name,
            "ok": False, "takeover": False,
            "elapsed_ms": int((time.time() - started) * 1000),
            "error": "internal solver failure",
        }, settings)
        raise

    # Fallback: service failed → takeover (owner solves by hand).
    if not ok and fallback_to_takeover:
        _audit({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": challenge.kind, "domain": challenge.domain,
            "sitekey": challenge.sitekey, "backend": solver.name,
            "ok": False, "takeover": False,
            "elapsed_ms": int((time.time() - started) * 1000),
            "error": f"service failed, falling back to takeover: {detail[:200]}",
        }, settings)
        takeover = TakeoverBackend(**backend_kwargs)
        result = takeover.solve(challenge)
        ok, detail = result.ok, result.detail

    if result.takeover and notify_owner:
        # Owner must solve by hand — make sure they hear about it.
        _notify_takeover(challenge, result.detail, context)
        _audit({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": challenge.kind, "domain": challenge.domain,
            "sitekey": challenge.sitekey, "backend": result.backend,
            "ok": False, "takeover": True,
            "elapsed_ms": result.elapsed_ms or int((time.time() - started) * 1000),
            "error": "owner notified of takeover",
        }, settings)

    _audit({
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": challenge.kind, "domain": challenge.domain,
        "sitekey": challenge.sitekey, "backend": result.backend,
        "ok": ok, "takeover": result.takeover,
        "elapsed_ms": result.elapsed_ms or int((time.time() - started) * 1000),
        "error": "" if ok else detail[:200],
    }, settings)
    return result


# ── AccountCreator adapter (dependency injection bridge) ───────────────────

def creator_solver_adapter(
    solver_enabled: bool | None = None,
    settings: Any = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a solver callable for ``nomorals.accounts.AccountCreator``.

    ``accounts`` is L2 and this module is L4, so the creator cannot
    import the solver — it takes an injected callable instead. This
    adapter is that callable: it wraps :func:`solve` with
    ``backend="auto"`` (service first, owner-takeover fallback) and
    honors the same enablement rule as the CLI — explicit
    ``solver_enabled`` wins, otherwise ``NM_CAPTCHA_SOLVER`` (default
    ON). Every attempt is audit-logged by :func:`solve`.

    The challenge dict carries ``kind`` plus ``sitekey``/``page_url``/
    ``image_url``/``image_bytes``/``action``/``min_score`` as known; the
    returned dict is ``SolveResult.to_dict()``.
    """
    def _solve(challenge: dict[str, Any]) -> dict[str, Any]:
        on = solver_enabled
        if on is None:
            on = os.environ.get("NM_CAPTCHA_SOLVER", "1") != "0"
        ch = CaptchaChallenge(
            kind=challenge.get("kind", CaptchaKind.UNKNOWN),
            sitekey=challenge.get("sitekey", "") or "",
            page_url=challenge.get("page_url", "") or "",
            image_url=challenge.get("image_url", "") or "",
            image_bytes=challenge.get("image_bytes", b"") or b"",
            action=challenge.get("action", "") or "",
            min_score=float(challenge.get("min_score", 0.3) or 0.3),
        )
        return solve(ch, backend="auto", settings=settings,
                     solver_enabled=on).to_dict()

    return _solve


def fetch_image_bytes(url: str, timeout: float = 20.0) -> bytes:
    """Download an image-captcha's bytes (stdlib, no deps)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-captcha/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ── tool registration ────────────────────────────────────────────────────────

def register(registry: Any) -> None:
    from ..core.policy import Capability

    @registry.register(
        "captcha",
        description=(
            "captcha handling for owner-directed browser automation: detect "
            "captchas in page HTML (reCAPTCHA v2/v3/enterprise, hCaptcha, "
            "Turnstile, Cloudflare challenges, GeeTest v3/v4, Arkose "
            "FunCaptcha, AWS WAF, FriendlyCaptcha, Yandex SmartCaptcha, "
            "PerimeterX, image and audio captchas) and solve "
            "them via a pluggable backend — 'service' (commercial solving "
            "API, key from CAPTCHA_API_KEY env only, rate-limited with "
            "budgets + error backoff), 'takeover' (pause, notify the owner, "
            "hand the challenge to them), or 'detect' (report only). "
            "Auto mode tries the service first, then owner takeover with "
            "a clear notification. Every attempt is audit-logged."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — detect|status|solve (default detect)",
            "html": "str — page HTML for detect (or use session)",
            "url": "str — page URL (hints detection, used as pageurl)",
            "session": "str — browser session name to detect against",
            "kind": "str — for solve: recaptcha_v2|recaptcha_v3|"
                    "recaptcha_enterprise|hcaptcha|turnstile|image_captcha|"
                    "audio_captcha",
            "sitekey": "str — for solve: the data-sitekey",
            "backend": "str — service|takeover|detect|auto (default auto)",
            "notify_owner": "bool — ping the owner when a takeover is "
                            "needed (default true)",
            "image": "str — for solve of image_captcha: path or URL",
            "v3_action": "str — reCAPTCHA v3 action name",
            "min_score": "float — reCAPTCHA v3 score floor (default 0.3)",
        },
    )
    def captcha(action: str = "detect", html: str = "", url: str = "",
                session: str = "", kind: str = "", sitekey: str = "",
                backend: str = "auto", notify_owner: bool = True,
                image: str = "",
                v3_action: str = "", min_score: float = 0.3,
                **_: Any) -> dict[str, Any]:
        action = (action or "detect").lower()
        context = registry.context
        settings = getattr(context, "settings", None) if context else None

        if action == "status":
            svc = ServiceBackend()
            return {
                "backends": sorted(_BACKENDS),
                "service_available": svc.available(),
                "api_key_configured": svc.available(),
                "solver_enabled": os.environ.get("NM_CAPTCHA_SOLVER", "1") != "0",
                "rate_limit": rate_limiter(settings).status(),
                "audit_log": _audit_path(settings),
                "kinds": list(CaptchaKind.ALL),
            }

        if action == "detect":
            page_html, page_url = html, url
            if session and not page_html:
                from .browser import get_session
                sess = get_session(session)
                page_html, page_url = (getattr(sess, "_raw", "") or "",
                                      sess.url or url)
            found = detect(page_html, page_url)
            return {"count": len(found),
                    "challenges": [c.summary() for c in found]}

        if action == "solve":
            if kind not in CaptchaKind.ALL or kind == CaptchaKind.UNKNOWN:
                raise ToolError(
                    f"captcha solve needs a known kind "
                    f"(want one of: {list(CaptchaKind.ALL)})")
            image_bytes = b""
            image_url = ""
            if image:
                if image.startswith(("http://", "https://")):
                    image_url = image
                    image_bytes = fetch_image_bytes(image)
                else:
                    with open(os.path.expanduser(image), "rb") as fh:
                        image_bytes = fh.read()
            challenge = CaptchaChallenge(
                kind=kind, sitekey=sitekey, page_url=url,
                image_url=image_url, image_bytes=image_bytes,
                action=v3_action, min_score=min_score)
            result = solve(challenge, backend=backend, settings=settings,
                           notify_owner=notify_owner, context=context)
            return result.to_dict()

        raise ToolError(f"unknown captcha action {action!r} "
                        "(want detect|status|solve)")


def _redact_key(text: str, key: str) -> str:
    """Defense-in-depth: scrub the key out of any string before logging."""
    if key and key in text:
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return text.replace(key, f"<api-key:{digest}>")
    return text
