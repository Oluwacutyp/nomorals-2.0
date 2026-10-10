"""Browser error taxonomy (layer 4).

Every failure the browser stack reports is exactly one of three kinds —
never misreported as another:

* :class:`BrowserNetworkError` — the request never reached a working
  server: DNS failures, timeouts, refused/reset connections, TLS errors.
* :class:`BrowserBotDetectedError` — the site answered but blocked the
  client: WAF/anti-bot challenge pages (Cloudflare, PerimeterX, DataDome,
  ...), captcha walls, HTTP 403/429 blocks.
* :class:`BrowserSiteError` — the site answered with a genuine error:
  4xx (not found, gone, bad request) or 5xx (the server is broken).

Anything that matches none of these signals stays a plain
:class:`BrowserError` with the raw message — an unclassified failure is
reported honestly, never force-fit into a category.

Each typed error carries ``kind``, machine-readable detail (``reason`` /
``detection`` / ``status``), a ``next_steps`` list of concrete follow-ups,
and optional ``evidence`` (screenshot/DOM paths captured at failure time).
All three subclass :class:`BrowserError`, so existing ``except
BrowserError`` handlers keep working unchanged.
"""

from __future__ import annotations

import re
import socket
import urllib.error
from typing import Any

__all__ = [
    "BrowserError",
    "BrowserNetworkError",
    "BrowserSiteError",
    "BrowserBotDetectedError",
    "classify_exception",
    "classify_http_status",
    "detect_challenge",
    "describe",
    "summarize",
]


class BrowserError(Exception):
    """Anything the browser service refuses to fake: navigation failures,
    unknown tabs/sessions, HTTP errors on download, missing playwright."""

    #: machine-readable bucket: browser_error | network_error |
    #: site_error | bot_detection.
    kind = "browser_error"

    def __init__(self, message: str, *,
                 next_steps: list[str] | tuple[str, ...] = (),
                 evidence: dict[str, str] | None = None,
                 url: str = "",
                 retryable: bool = False,
                 retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        #: concrete follow-ups for the caller/owner, in priority order.
        self.next_steps: list[str] = list(next_steps)
        #: failure evidence, e.g. {"screenshot": path, "dom": path}.
        self.evidence: dict[str, str] = dict(evidence or {})
        #: the URL being acted on when the failure happened ("").
        self.url: str = url or ""
        #: honest automation guidance: may this operation simply be tried
        #: again later? A rate limit is retryable (with backoff); a captcha
        #: wall is not — retrying it blindly is hammering.
        self.retryable: bool = bool(retryable)
        #: suggested wait before retrying, in seconds (None = no guidance).
        #: Honored from ``Retry-After`` / ``RateLimit-Reset`` headers when
        #: classifying HTTP 429 responses.
        self.retry_after_seconds: int | None = (
            int(retry_after_seconds)
            if retry_after_seconds is not None else None)

    def with_message(self, message: str) -> "BrowserError":
        """A same-type copy with a new message, preserving kind, detail
        attributes, next steps, evidence, url, and retry guidance."""
        cls = type(self)
        new = cls.__new__(cls)
        Exception.__init__(new, message)
        new.next_steps = list(self.next_steps)
        new.evidence = dict(self.evidence)
        new.url = self.url
        new.retryable = self.retryable
        new.retry_after_seconds = self.retry_after_seconds
        for key, value in vars(self).items():
            if key not in {"args", "next_steps", "evidence", "url",
                           "retryable", "retry_after_seconds"}:
                setattr(new, key, value)
        return new

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "error": str(self),
            "url": self.url,
            "next_steps": list(self.next_steps),
            "evidence": dict(self.evidence),
            "retryable": self.retryable,
        }
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        return payload


class BrowserNetworkError(BrowserError):
    """The request never reached a working server.

    ``reason`` is one of: dns, timeout, connection_refused,
    connection_reset, offline, tls, proxy, unknown.
    """

    kind = "network_error"

    _NEXT_STEPS = {
        "dns": [
            "check DNS / network connectivity on this machine",
            "retry — transient resolver failures happen",
            "the domain may not exist; verify the URL spelling",
        ],
        "timeout": [
            "retry — the site may just be slow",
            "raise the navigation timeout for this site",
            "the site may be unreachable from this network; try a proxy",
        ],
        "connection_refused": [
            "the server actively refused the connection — it may be down",
            "verify the host/port; retry later",
        ],
        "connection_reset": [
            "retry — resets are often transient",
            "persistent resets can mean a middlebox/WAF dropping the client; "
            "try the session proxy",
        ],
        "offline": [
            "this machine looks offline — restore connectivity and retry",
        ],
        "tls": [
            "TLS handshake failed — the site's certificate may be invalid "
            "or a proxy is intercepting",
            "do not disable certificate verification to work around this",
        ],
        "proxy": [
            "the configured proxy failed — check the proxy URL/credentials",
            "rotate or clear the session proxy and retry",
        ],
        "unknown": [
            "retry — transient network failures happen",
            "check connectivity on this machine",
        ],
    }

    def __init__(self, message: str, *, reason: str = "unknown",
                 **kwargs: Any) -> None:
        self.reason = reason if reason in self._NEXT_STEPS else "unknown"
        # Network failures are the canonical retryable failure: the
        # request never reached a working server, so nothing about the
        # attempt is held against the client.
        kwargs.setdefault("retryable", True)
        super().__init__(
            message,
            next_steps=kwargs.pop("next_steps", None)
            or self._NEXT_STEPS[self.reason],
            **kwargs)


class BrowserSiteError(BrowserError):
    """The site answered with a genuine HTTP error — not a block, not the
    network. ``status`` is the HTTP status code."""

    kind = "site_error"

    def __init__(self, message: str, *, status: int = 0,
                 **kwargs: Any) -> None:
        self.status = int(status or 0)
        steps = list(kwargs.pop("next_steps", None) or ())
        if not steps:
            if 400 <= self.status < 500:
                steps = [
                    f"HTTP {self.status} is the site refusing this request — "
                    "check the URL and parameters",
                    "the page may have moved or been removed",
                ]
            elif self.status >= 500:
                steps = [
                    f"HTTP {self.status} means the site's own server is "
                    "broken — nothing wrong on this end",
                    "wait and retry; alert the owner if it persists",
                ]
            else:
                steps = ["check the URL and retry"]
        # A broken server may heal; a refused request will not.
        if self.status >= 500:
            kwargs.setdefault("retryable", True)
        super().__init__(message, next_steps=steps, **kwargs)


class BrowserBotDetectedError(BrowserError):
    """The site blocked the client: anti-bot challenge, captcha wall, or
    an HTTP 403/429 block.

    ``detection`` is one of: cloudflare-challenge, turnstile, perimeterx,
    datadome, akamai, imperva, kasada, f5-shape, aws-waf, arkose,
    geetest, captcha-wall, rate-limit, forbidden.

    This is a *report*, not an evasion tool: the next steps point at the
    existing captcha flow (solver default-on, owner takeover when
    unsolvable) — never at defeating the control.
    """

    kind = "bot_detection"

    _NEXT_STEPS = {
        "cloudflare-challenge": [
            "run check_captcha on the tab to see the live challenge",
            "the captcha solver runs by default where one exists; "
            "if it cannot solve it, the owner takes over manually",
            "rotating the session proxy sometimes helps — it never "
            "guarantees a pass",
        ],
        "turnstile": [
            "Cloudflare Turnstile usually auto-passes a real browser — "
            "switch this tab to a rendered (playwright) tab and retry",
            "if the widget still spins, the IP/network is flagged: "
            "rotate the session proxy",
        ],
        "perimeterx": [
            "run check_captcha on the tab to see the live challenge",
            "the captcha solver runs by default where one exists; "
            "if it cannot solve it, the owner takes over manually",
        ],
        "datadome": [
            "run check_captcha on the tab to see the live challenge",
            "the captcha solver runs by default where one exists; "
            "if it cannot solve it, the owner takes over manually",
        ],
        "akamai": [
            "Akamai Bot Manager flagged this client (sensor/cookie "
            "challenge) — a rendered tab with clean storage sometimes "
            "passes on retry",
            "persistent blocks need a residential proxy; datacenter IPs "
            "rarely pass Akamai",
        ],
        "imperva": [
            "Imperva/Incapsula is challenging the client — run "
            "check_captcha on the tab to see the live interstitial",
            "the captcha solver runs by default where one exists; "
            "otherwise the owner takes over manually",
        ],
        "kasada": [
            "Kasada flagged the client — slow down hard: enable pacing, "
            "wait several minutes, then retry",
            "rotating the session proxy sometimes helps",
        ],
        "f5-shape": [
            "F5 Shape is running its interstitial — this usually needs a "
            "real browser session; retry with a rendered tab",
        ],
        "aws-waf": [
            "AWS WAF is challenging the client — a rendered tab with "
            "fresh storage usually clears the token challenge",
        ],
        "arkose": [
            "Arkose (FunCaptcha) is up — run check_captcha; the solver "
            "handles some variants, the owner takes the rest",
        ],
        "geetest": [
            "GeeTest slider/click challenge — run check_captcha; the "
            "solver handles some variants, the owner takes the rest",
        ],
        "captcha-wall": [
            "run check_captcha on the tab to identify the challenge",
            "the captcha solver runs by default where one exists; "
            "if it cannot solve it, the owner takes over manually",
        ],
        "rate-limit": [
            "slow down: enable pacing between actions, then retry",
            "wait before retrying — hammering a rate limit extends it",
            "rotating the session proxy can help",
        ],
        "forbidden": [
            "HTTP 403 with no challenge markers — a WAF rule or "
            "site-level deny for this client/IP",
            "run check_captcha to confirm whether a challenge is present",
            "rotating the session proxy sometimes helps",
        ],
    }

    #: detections that are retryable with backoff (the block is the site
    #: asking for slower/fresher traffic, not a hard wall).
    _RETRYABLE_DETECTIONS = frozenset({
        "rate-limit", "kasada", "turnstile", "aws-waf", "f5-shape"})

    def __init__(self, message: str, *, detection: str = "forbidden",
                 **kwargs: Any) -> None:
        self.detection = (detection or "forbidden")
        if self.detection not in self._NEXT_STEPS:
            self.detection = "forbidden"
        if self.detection in self._RETRYABLE_DETECTIONS:
            kwargs.setdefault("retryable", True)
        super().__init__(
            message,
            next_steps=kwargs.pop("next_steps", None)
            or self._NEXT_STEPS[self.detection],
            **kwargs)


# ── challenge-page detection ───────────────────────────────────────────────
# Markers verified against real WAF interstitials (see module docstring
# research notes): Cloudflare's "Just a moment..." title family (localized),
# _cf_chl_opt / cf-chl / challenge-platform structure, PerimeterX's
# px-captcha/_px3, DataDome's captcha-delivery.com. The content tier only
# runs on challenge-compatible signals (block statuses or WAF headers), so
# an article that merely mentions "captcha" never trips it.

#: Localized Cloudflare interstitial titles (exact, lowercased).
_CF_CHALLENGE_TITLES = frozenset({
    "just a moment...",
    "please wait...",
    "checking your browser before you access",
    "attention required!",
    "attention required! | cloudflare",
    "un momento...",
    "einen moment...",
    "veuillez patienter...",
    "подождите...",
    "请稍候…",
    "잠시만 기다려주세요...",
})

#: Body markers that only appear on real challenge interstitials.
_CF_BODY_MARKERS = (
    "_cf_chl_opt",
    "__cf_chl_rt_tk",
    "cf-chl",
    "cf-challenge-platform",
    "cf-browser-verification",
    "challenges.cloudflare.com",
    "/cdn-cgi/challenge-platform/",
    "cf_chl_opt",
)

_PX_BODY_MARKERS = ("px-captcha", "_px3", "perimeterx.net",
                     "client.perimeterx.net", "/api/v1/px/")
_DD_BODY_MARKERS = ("captcha-delivery.com", "ct.captcha-delivery.com",
                    "datadome", "js.datadome.co")

#: Extra WAF body/script markers, trusted on challenge-shaped responses.
_AKAMAI_BODY_MARKERS = ("_abck", "bm_sz", "bm_sv", "ak_bmsc",
                        "akamai/sensor", "/akam/13/")
_IMPERVA_BODY_MARKERS = ("___utmvc", "reese84", "incapsula")
_KASADA_BODY_MARKERS = ("ips.js", "kpsdk", "kasada")
_F5_BODY_MARKERS = ("istlwashere", "f5 shape", "shape security")
_AWSWAF_BODY_MARKERS = ("aws-waf-token", "awswafintegration",
                        "awswaf-captcha")
_TURNSTILE_MARKERS = ("challenges.cloudflare.com/turnstile",
                      "cf-turnstile")
_ARKOSE_MARKERS = ("arkoselabs.com", "funcaptcha", "arkose")
_GEETEST_MARKERS = ("initgeetest4", "gcaptcha4.geetest.com", "gt4.js",
                    "geetest.com")

#: Set-Cookie NAME prefixes that identify the WAF on a challenge-shaped
#: response (header value irrelevant — the name is the signal).
_COOKIE_PREFIX_MARKERS: tuple[tuple[str, str], ...] = (
    ("__cf_bm", "cloudflare-challenge"),
    ("cf_clearance", "cloudflare-challenge"),
    ("_abck", "akamai"),
    ("bm_sz", "akamai"),
    ("bm_sv", "akamai"),
    ("ak_bmsc", "akamai"),
    ("datadome", "datadome"),
    ("_px3", "perimeterx"),
    ("_pxvid", "perimeterx"),
    ("_pxhd", "perimeterx"),
    ("incap_ses_", "imperva"),
    ("visid_incap_", "imperva"),
    ("nlbi_", "imperva"),
    ("aws-waf-token", "aws-waf"),
    ("kp_uidz", "kasada"),
    ("acw_sc__v2", "imperva"),  # Alibaba WAF rides the Imperva guidance
)

#: Response headers that identify the WAF vendor (name -> detection).
_WAF_HEADER_MARKERS: tuple[tuple[str, str], ...] = (
    ("x-datadome", "datadome"),
    ("x-dd-b", "datadome"),
    ("x-datadome-cid", "datadome"),
    ("x-iinfo", "imperva"),
    ("x-cdn", "imperva"),  # value checked for "incapsula"
    ("x-kpsdk-ct", "kasada"),
    ("x-kpsdk-v", "kasada"),
    ("x-kpsdk-cd", "kasada"),
    ("x-amzn-waf-action", "aws-waf"),
    ("x-sucuri-id", "forbidden"),
    ("akamai-grn", "akamai"),
)

#: Wall-text + widget pairs that mark a generic captcha wall (both halves
#: required — a bare "captcha" mention on a login form is not a wall).
_WALL_TEXT = (
    "verify you are human",
    "verify you're human",
    "verifying you are human",
    "are you a robot",
    "prove you are human",
    "complete the security check",
    "confirm you are not a robot",
)
_WALL_WIDGETS = ("g-recaptcha", "h-captcha", "cf-turnstile",
                 "data-sitekey", "recaptcha/api.js")


def _headers_lower(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = headers.items() if hasattr(headers, "items") else headers
        for k, v in (items or []):
            out[str(k).lower()] = str(v)
    except Exception:  # noqa: BLE001 - headers are best-effort evidence
        pass
    return out


def detect_challenge(html: str = "", *, title: str = "",
                     headers: Any = None,
                     set_cookies: Any = None) -> str:
    """Identify an anti-bot challenge page.

    Returns ``cloudflare-challenge`` | ``turnstile`` | ``perimeterx`` |
    ``datadome`` | ``akamai`` | ``imperva`` | ``kasada`` | ``f5-shape`` |
    ``aws-waf`` | ``arkose`` | ``geetest`` | ``captcha-wall`` | ``""``
    (no challenge detected).

    Detection is gated: body markers are only trusted on
    challenge-compatible responses (WAF headers present) or alongside the
    interstitial title family, so ordinary pages that quote the markers
    are not misclassified. ``title`` alone (the "Just a moment..." family)
    is trusted — sites do not title real pages that. ``set_cookies``
    accepts the response's Set-Cookie values (a header string or a list
    of them): vendor-specific cookie *names* are strong signals.
    """
    low_title = (title or "").strip().lower()
    low_html = (html or "").lower()
    if not low_title and "<title" in low_html:
        # Callers that only have raw HTML: extract the <title> text.
        match = re.search(r"<title[^>]*>(.*?)</title>", low_html,
                          re.DOTALL)
        if match:
            low_title = re.sub(r"\s+", " ", match.group(1)).strip()
    hdrs = _headers_lower(headers)
    server = hdrs.get("server", "")
    waf_headers = (
        "cloudflare" in server
        or "cf-ray" in hdrs
        or hdrs.get("cf-mitigated", "") == "challenge"
    )

    if low_title in _CF_CHALLENGE_TITLES:
        return "cloudflare-challenge"
    if "cloudflare" in low_title and "attention required" in low_title:
        return "cloudflare-challenge"
    if hdrs.get("cf-mitigated", "") == "challenge":
        return "cloudflare-challenge"

    # Vendor header signals (passive, documented fingerprints).
    for hname, detection in _WAF_HEADER_MARKERS:
        value = hdrs.get(hname, "")
        if not value:
            continue
        if hname == "x-cdn" and "incapsula" not in value:
            continue
        if detection == "forbidden":
            continue  # Sucuri presence alone is not a challenge
        return detection
    if "incapsula" in hdrs.get("x-cdn", ""):
        return "imperva"

    # Set-Cookie name prefixes — the cookie name is the vendor's tell.
    cookie_blob = _set_cookies_blob(set_cookies).lower()
    if cookie_blob:
        for prefix, detection in _COOKIE_PREFIX_MARKERS:
            if prefix in cookie_blob:
                return detection

    if waf_headers:
        for marker in _CF_BODY_MARKERS:
            if marker in low_html:
                return "cloudflare-challenge"
    # Cloudflare structural markers are challenge-specific enough to
    # trust even without WAF headers (they never appear on real pages).
    for marker in ("_cf_chl_opt", "__cf_chl_rt_tk", "/cdn-cgi/challenge-platform/"):
        if marker in low_html:
            return "cloudflare-challenge"
    # Turnstile is Cloudflare's widget but behaves differently (usually
    # auto-passes a real browser), so it gets its own detection.
    for marker in _TURNSTILE_MARKERS:
        if marker in low_html:
            return "turnstile"

    for marker in _PX_BODY_MARKERS:
        if marker in low_html:
            return "perimeterx"
    for marker in _DD_BODY_MARKERS:
        if marker in low_html:
            return "datadome"
    for marker in _AKAMAI_BODY_MARKERS:
        if marker in low_html:
            return "akamai"
    for marker in _IMPERVA_BODY_MARKERS:
        if marker in low_html:
            return "imperva"
    for marker in _KASADA_BODY_MARKERS:
        if marker in low_html:
            return "kasada"
    for marker in _F5_BODY_MARKERS:
        if marker in low_html:
            return "f5-shape"
    for marker in _AWSWAF_BODY_MARKERS:
        if marker in low_html:
            return "aws-waf"
    for marker in _ARKOSE_MARKERS:
        if marker in low_html:
            return "arkose"
    for marker in _GEETEST_MARKERS:
        if marker in low_html:
            return "geetest"

    if any(t in low_html for t in _WALL_TEXT) and any(
            w in low_html for w in _WALL_WIDGETS):
        return "captcha-wall"
    return ""


def _set_cookies_blob(set_cookies: Any) -> str:
    """Flatten Set-Cookie header value(s) into one searchable string."""
    if not set_cookies:
        return ""
    if isinstance(set_cookies, str):
        return set_cookies
    try:
        return "\n".join(str(c) for c in set_cookies)
    except TypeError:
        return str(set_cookies)


def _retry_after_seconds(headers: Any) -> int | None:
    """Parse ``Retry-After`` (delta-seconds) or ``RateLimit-Reset`` into an
    int number of seconds. None when absent/unparseable."""
    hdrs = _headers_lower(headers)
    raw = (hdrs.get("retry-after") or "").strip()
    if raw.isdigit():
        return max(0, int(raw))
    # HTTP-date form: convert to a delta against now.
    if raw:
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = int((dt - datetime.now(timezone.utc)).total_seconds())
            return max(0, delta)
        except Exception:  # noqa: BLE001 - best-effort header parsing
            pass
    reset = (hdrs.get("ratelimit-reset") or "").strip()
    if reset.isdigit():
        return max(0, int(reset))
    return None


# ── classification ─────────────────────────────────────────────────────────

#: playwright / chromium network error tokens → reason.
_NET_ERROR_TOKENS: tuple[tuple[str, str], ...] = (
    ("net::err_name_not_resolved", "dns"),
    ("net::err_connection_timed_out", "timeout"),
    ("net::err_timed_out", "timeout"),
    ("net::err_connection_refused", "connection_refused"),
    ("net::err_connection_reset", "connection_reset"),
    ("net::err_connection_closed", "connection_reset"),
    ("net::err_internet_disconnected", "offline"),
    ("net::err_network_changed", "offline"),
    ("net::err_cert_", "tls"),
    ("net::err_ssl_", "tls"),
    ("net::err_proxy_", "proxy"),
    ("net::err_tunnel_connection_failed", "proxy"),
)

#: plain-text network signals (lowercased substring) → reason. Only
#: matched when they clearly describe transport failure — never bare
#: "timeout", which also fires on selector waits.
_TEXT_NET_SIGNALS: tuple[tuple[str, str], ...] = (
    ("name or service not known", "dns"),
    ("temporary failure in name resolution", "dns"),
    ("nodename nor servname provided", "dns"),
    ("getaddrinfo failed", "dns"),
    ("connection timed out", "timeout"),
    ("connection reset by peer", "connection_reset"),
    ("connection refused", "connection_refused"),
    ("network is unreachable", "offline"),
    ("network unreachable", "offline"),
    ("certificate verify failed", "tls"),
    ("ssl: certificate_verify_failed", "tls"),
    ("proxy error", "proxy"),
)


def _walk_chain(exc: BaseException) -> list[str]:
    """Every message in the exception chain, lowercased."""
    texts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        texts.append(f"{type(current).__name__}: {current}".lower())
        current = current.__cause__ or current.__context__
    return texts


def _network_reason_from_text(text: str) -> str:
    for token, reason in _NET_ERROR_TOKENS:
        if token in text:
            return reason
    for signal, reason in _TEXT_NET_SIGNALS:
        if signal in text:
            return reason
    return ""


def classify_exception(exc: BaseException, *, url: str = "",
                       evidence: dict[str, str] | None = None) -> BrowserError:
    """Classify a raised exception into the taxonomy.

    Already-typed :class:`BrowserError`s pass through untouched (same
    object — never re-wrapped into a different kind). Network signals
    (net::ERR_*, DNS/TLS/connection text, socket/urllib errors) become
    :class:`BrowserNetworkError`. Everything else becomes a plain
    :class:`BrowserError` with the raw message — an unrecognized failure
    is reported as-is, never force-fit.
    """
    if isinstance(exc, BrowserError):
        if evidence:
            merged = dict(exc.evidence)
            merged.update(evidence)
            exc.evidence = merged
        if url and not exc.url:
            exc.url = url
        return exc
    texts = _walk_chain(exc)
    for text in texts:
        reason = _network_reason_from_text(text)
        if reason:
            short = str(exc).strip().splitlines()[0][:220] or type(exc).__name__
            return BrowserNetworkError(
                f"network failure ({reason}) on {url or '(no url)'}: {short}",
                reason=reason, url=url, evidence=evidence)
    if isinstance(exc, socket.gaierror):
        return BrowserNetworkError(
            f"DNS resolution failed on {url or '(no url)'}: {exc}",
            reason="dns", url=url, evidence=evidence)
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return BrowserNetworkError(
            f"connection timed out on {url or '(no url)'}: {exc}",
            reason="timeout", url=url, evidence=evidence)
    if isinstance(exc, ConnectionError):
        return BrowserNetworkError(
            f"connection failed on {url or '(no url)'}: {exc}",
            reason="connection_reset", url=url, evidence=evidence)
    if isinstance(exc, urllib.error.URLError):
        # The reason's type is authoritative: a URLError wrapping a
        # timeout/DNS error is that failure, whatever the text says.
        inner = getattr(exc, "reason", None)
        if isinstance(inner, socket.gaierror):
            reason = "dns"
        elif isinstance(inner, (socket.timeout, TimeoutError)):
            reason = "timeout"
        elif isinstance(inner, ConnectionError):
            reason = "connection_reset"
        else:
            reason = _network_reason_from_text(
                str(inner or "").lower()) or "unknown"
        return BrowserNetworkError(
            f"request to {url or '(no url)'} failed: {exc.reason}",
            reason=reason, url=url, evidence=evidence)
    short = (str(exc).strip().splitlines() or [""])[0][:300]
    return BrowserError(
        f"{url + ': ' if url else ''}{short or type(exc).__name__}",
        url=url, evidence=evidence)


_CHALLENGE_STATUSES = {401, 403, 429, 503}


def classify_http_status(status: int, *, url: str = "", html: str = "",
                         title: str = "", headers: Any = None,
                         set_cookies: Any = None,
                         evidence: dict[str, str] | None = None) -> BrowserError:
    """Classify an HTTP error status into the taxonomy.

    403/429/503 (or any status whose body carries challenge markers)
    → :class:`BrowserBotDetectedError`; other 4xx and 5xx →
    :class:`BrowserSiteError`. A 429 without challenge markers is still a
    block (rate-limit), not a broken page — and ``Retry-After`` /
    ``RateLimit-Reset`` headers become ``retry_after_seconds``. 2xx/3xx
    here is a caller bug and raises ValueError — never silently
    "classify" success.
    """
    status = int(status or 0)
    if status < 400:
        raise ValueError(
            f"classify_http_status got non-error status {status}")
    where = f" on {url}" if url else ""
    challenge = ""
    if status in _CHALLENGE_STATUSES or headers:
        challenge = detect_challenge(html, title=title, headers=headers,
                                     set_cookies=set_cookies)
    if challenge:
        label = challenge.replace("-", " ")
        return BrowserBotDetectedError(
            f"blocked by {label}{where} (HTTP {status}) — "
            f"the page is a {label} interstitial, not the site",
            detection=challenge, url=url, evidence=evidence)
    if status == 429:
        wait = _retry_after_seconds(headers)
        return BrowserBotDetectedError(
            f"rate-limited{where} (HTTP 429) — the site is throttling "
            f"this client",
            detection="rate-limit", url=url, evidence=evidence,
            retry_after_seconds=wait)
    if status == 403:
        return BrowserBotDetectedError(
            f"access denied{where} (HTTP 403, no challenge markers) — "
            f"a WAF rule or site-level deny for this client",
            detection="forbidden", url=url, evidence=evidence)
    if status == 404:
        return BrowserSiteError(
            f"page not found{where} (HTTP 404)",
            status=status, url=url, evidence=evidence)
    if status == 410:
        return BrowserSiteError(
            f"page is gone{where} (HTTP 410)",
            status=status, url=url, evidence=evidence)
    if 400 <= status < 500:
        return BrowserSiteError(
            f"request rejected{where} (HTTP {status})",
            status=status, url=url, evidence=evidence)
    edge = ""
    if headers:
        hdrs = _headers_lower(headers)
        if "cloudflare" in hdrs.get("server", "") or "cf-ray" in hdrs:
            edge = " (served by the Cloudflare edge — the origin may be " \
                   "down, or an edge challenge without markers)"
    return BrowserSiteError(
        f"server error{where} (HTTP {status}){edge}",
        status=status, url=url, evidence=evidence)


#: human one-liner per kind, for logs and CLI output.
KIND_LABELS = {
    "browser_error": "browser error",
    "network_error": "network failure",
    "site_error": "site error",
    "bot_detection": "blocked (anti-bot)",
}

#: icon per kind for styled output.
KIND_ICONS = {
    "browser_error": "\u2753",
    "network_error": "\U0001F50C",
    "site_error": "\U0001F6A7",
    "bot_detection": "\U0001F6E1",
}


def describe(error: BrowserError) -> dict[str, Any]:
    """JSON-safe summary of a classified error for CLI/API output."""
    payload = error.to_dict()
    detail: dict[str, Any] = {}
    if isinstance(error, BrowserNetworkError):
        detail["reason"] = error.reason
    elif isinstance(error, BrowserSiteError):
        detail["status"] = error.status
    elif isinstance(error, BrowserBotDetectedError):
        detail["detection"] = error.detection
    payload["detail"] = detail
    payload["label"] = KIND_LABELS.get(error.kind, error.kind)
    payload["icon"] = KIND_ICONS.get(error.kind, "")
    return payload


def summarize(error: BrowserError, *, style: str = "rich") -> str:
    """A human-readable failure card for CLI/chat output.

    ``style="rich"`` uses icons and section markers; ``style="plain"``
    is ASCII-only. This is the anti-bot-doctor shape: symptom →
    diagnosis → prescription, with honest retry guidance instead of a
    bare traceback.
    """
    icon = KIND_ICONS.get(error.kind, "") if style == "rich" else ""
    label = KIND_LABELS.get(error.kind, error.kind)
    head = f"{icon} {label}".strip() if icon else label
    lines = [head, str(error)]
    if error.url:
        lines.append(f"url: {error.url}")
    detail: dict[str, Any] = {}
    if isinstance(error, BrowserNetworkError):
        detail["reason"] = error.reason
    elif isinstance(error, BrowserSiteError):
        detail["status"] = error.status
    elif isinstance(error, BrowserBotDetectedError):
        detail["detection"] = error.detection
    if detail:
        lines.append(
            "detail: " + ", ".join(f"{k}={v}" for k, v in detail.items()))
    if error.retryable:
        wait = (f" — wait {error.retry_after_seconds}s first"
                if error.retry_after_seconds else "")
        lines.append(f"retryable: yes{wait}")
    else:
        lines.append("retryable: no — retrying blindly will not help")
    if error.next_steps:
        lines.append("next steps:")
        for i, step in enumerate(error.next_steps, 1):
            lines.append(f"  {i}. {step}")
    if error.evidence:
        lines.append("evidence: " + ", ".join(
            f"{k}={v}" for k, v in error.evidence.items()))
    return "\n".join(lines)
