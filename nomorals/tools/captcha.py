"""Captcha detection and solving for owner-directed browser automation.

Devon drives browser sessions (see :mod:`nomorals.tools.browser`) for the
owner — logins, forms, checkouts — and those flows regularly hit captchas.
This module gives her a way through without paging the owner every time:

* :func:`detect` — scan page HTML for reCAPTCHA v2/v3/enterprise, hCaptcha,
  Cloudflare Turnstile, Cloudflare challenge pages, and image captchas.
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
    "detect",
    "detect_in_session",
    "extract_image_captcha_urls",
    "backend_for",
    "solve",
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
    IMAGE_CAPTCHA = "image_captcha"
    UNKNOWN = "unknown"

    ALL = (
        RECAPTCHA_V2, RECAPTCHA_V3, RECAPTCHA_ENTERPRISE,
        HCAPTCHA, TURNSTILE, IMAGE_CAPTCHA, UNKNOWN,
    )


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
        return {
            "kind": self.kind,
            "sitekey": self.sitekey,
            "domain": self.domain,
            "image_url": self.image_url,
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


def _has(html: str, *needles: str) -> bool:
    low = html.lower()
    return any(n.lower() in low for n in needles)


def detect(html: str, url: str = "") -> list[CaptchaChallenge]:
    """Scan page HTML, return every captcha challenge found.

    Pure function — no network, no side effects. ``sitekey`` values are
    public site keys, safe to surface.
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

    # reCAPTCHA enterprise / v2 — data-sitekey driven
    for m in _SITEKEY_RE.finditer(html):
        sitekey = m.group(1).strip()
        if not sitekey or sitekey == "explicit":
            continue
        window = html[max(0, m.start() - 600):m.start()].lower()
        if is_enterprise or "enterprise" in window:
            add(CaptchaChallenge(CaptchaKind.RECAPTCHA_ENTERPRISE,
                                 sitekey=sitekey, page_url=url))
        elif is_recaptcha or "g-recaptcha" in window:
            add(CaptchaChallenge(CaptchaKind.RECAPTCHA_V2,
                                 sitekey=sitekey, page_url=url))
        elif is_hcaptcha or "h-captcha" in window:
            add(CaptchaChallenge(CaptchaKind.HCAPTCHA,
                                 sitekey=sitekey, page_url=url))
        elif is_turnstile or "cf-turnstile" in window:
            add(CaptchaChallenge(CaptchaKind.TURNSTILE,
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

    # Cloudflare challenge / "verify you are human" interstitials
    if _has(html, "cf-challenge", "__cf_chl", "cf_clearance") and _has(
            html, "verifying you are human", "verify you are human",
            "just a moment"):
        add(CaptchaChallenge(
            CaptchaKind.UNKNOWN, page_url=url,
            metadata={"provider": "cloudflare",
                      "note": "interactive challenge page"}))

    # image captchas — <img> whose src/alt/class/id smells like a captcha
    for img_url in extract_image_captcha_urls(html, url):
        add(CaptchaChallenge(CaptchaKind.IMAGE_CAPTCHA, page_url=url,
                             image_url=img_url))

    return out


def extract_image_captcha_urls(html: str, base_url: str = "") -> list[str]:
    """Pull candidate image-captcha URLs out of page HTML."""
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
    click the checkbox herself). Never auto-solves anything.
    """

    name = "takeover"

    def solve(self, challenge: CaptchaChallenge) -> SolveResult:
        hint = {
            CaptchaKind.RECAPTCHA_V2: "tick the 'I'm not a robot' checkbox",
            CaptchaKind.IMAGE_CAPTCHA: "read the image and type the characters",
        }.get(challenge.kind, "complete the challenge in the browser")
        return SolveResult(
            ok=False, kind=challenge.kind, backend=self.name, takeover=True,
            detail=(f"owner takeover needed on {challenge.domain or 'the page'}: "
                    f"{hint}"))


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
    CaptchaKind.IMAGE_CAPTCHA: "base64",
}


class ServiceBackend(CaptchaBackend):
    """Commercial solving service behind the 2captcha in.php/res.php shape.

    The API key comes **only** from the ``CAPTCHA_API_KEY`` environment
    variable (``CAPTCHA_API_URL`` optionally overrides the endpoint for
    compatible services). The key is never logged, never stored in the
    audit trail, and never echoed back in errors.
    """

    name = "service"

    def __init__(self, api_key: str = "", api_url: str = "",
                 sleeper: Callable[[float], None] | None = None) -> None:
        self._key = api_key or os.environ.get("CAPTCHA_API_KEY", "")
        self._api_url = (api_url or os.environ.get("CAPTCHA_API_URL", "")
                         or _DEFAULT_API_URL).rstrip("/")
        self._sleep = sleeper or time.sleep

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
        except Exception as exc:  # noqa: BLE001
            raise CaptchaError(f"solver API unreachable: {exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # plain-text shape: OK|<id>, CAPCHA_NOT_READY, or <ERROR>
            if body.startswith("OK|"):
                return {"status": 1, "request": body[3:].strip()}
            if body.strip() == "CAPCHA_NOT_READY":
                return {"status": 0, "request": "CAPCHA_NOT_READY"}
            raise CaptchaError("solver API returned an unreadable response")
        if data.get("status") != 1 and data.get("request") != "CAPCHA_NOT_READY":
            raise CaptchaError("solver API rejected the request")
        return data

    # -- solve --------------------------------------------------------------
    def solve(self, challenge: CaptchaChallenge) -> SolveResult:
        started = time.time()
        if not self.available():
            raise CaptchaError(
                "no CAPTCHA_API_KEY configured — set the environment "
                "variable or use the takeover backend")
        method = _METHOD_FOR_KIND.get(challenge.kind)
        if not method:
            raise CaptchaError(f"service backend cannot solve kind "
                               f"{challenge.kind!r}")

        params: dict[str, str] = {"key": self._key, "json": "1",
                                  "method": method}
        if method == "base64":
            if not challenge.image_bytes:
                raise CaptchaError("image captcha needs image bytes")
            params["body"] = base64.b64encode(challenge.image_bytes).decode()
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
            raise CaptchaError("solver timed out waiting for a token")

        elapsed = int((time.time() - started) * 1000)
        if challenge.kind == CaptchaKind.IMAGE_CAPTCHA:
            return SolveResult(ok=True, kind=challenge.kind,
                               backend=self.name, text=token,
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


def solve(challenge: CaptchaChallenge, backend: str = "auto",
          settings: Any = None, **backend_kwargs: Any) -> SolveResult:
    """Solve one challenge through the named backend, audit-logging it.

    The audit entry carries kind, domain, sitekey, timestamp, backend,
    outcome — never the API key, never page content.
    """
    solver = backend_for(backend, **backend_kwargs)
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
    _audit({
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": challenge.kind, "domain": challenge.domain,
        "sitekey": challenge.sitekey, "backend": solver.name,
        "ok": ok, "takeover": result.takeover,
        "elapsed_ms": result.elapsed_ms or int((time.time() - started) * 1000),
        "error": "" if ok else detail[:200],
    }, settings)
    return result


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
            "Turnstile, Cloudflare challenges, image captchas) and solve "
            "them via a pluggable backend — 'service' (commercial solving "
            "API, key from CAPTCHA_API_KEY env only), 'takeover' (pause and "
            "hand the challenge to the owner), or 'detect' (report only). "
            "Every attempt is audit-logged."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — detect|status|solve (default detect)",
            "html": "str — page HTML for detect (or use session)",
            "url": "str — page URL (hints detection, used as pageurl)",
            "session": "str — browser session name to detect against",
            "kind": "str — for solve: recaptcha_v2|recaptcha_v3|"
                    "recaptcha_enterprise|hcaptcha|turnstile|image_captcha",
            "sitekey": "str — for solve: the data-sitekey",
            "backend": "str — service|takeover|detect|auto (default auto)",
            "image": "str — for solve of image_captcha: path or URL",
            "v3_action": "str — reCAPTCHA v3 action name",
            "min_score": "float — reCAPTCHA v3 score floor (default 0.3)",
        },
    )
    def captcha(action: str = "detect", html: str = "", url: str = "",
                session: str = "", kind: str = "", sitekey: str = "",
                backend: str = "auto", image: str = "",
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
            result = solve(challenge, backend=backend, settings=settings)
            return result.to_dict()

        raise ToolError(f"unknown captcha action {action!r} "
                        "(want detect|status|solve)")


def _redact_key(text: str, key: str) -> str:
    """Defense-in-depth: scrub the key out of any string before logging."""
    if key and key in text:
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return text.replace(key, f"<api-key:{digest}>")
    return text
