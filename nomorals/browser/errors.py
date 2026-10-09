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
                 url: str = "") -> None:
        super().__init__(message)
        #: concrete follow-ups for the caller/owner, in priority order.
        self.next_steps: list[str] = list(next_steps)
        #: failure evidence, e.g. {"screenshot": path, "dom": path}.
        self.evidence: dict[str, str] = dict(evidence or {})
        #: the URL being acted on when the failure happened ("").
        self.url: str = url or ""

    def with_message(self, message: str) -> "BrowserError":
        """A same-type copy with a new message, preserving kind, detail
        attributes, next steps, evidence, and url."""
        cls = type(self)
        new = cls.__new__(cls)
        Exception.__init__(new, message)
        new.next_steps = list(self.next_steps)
        new.evidence = dict(self.evidence)
        new.url = self.url
        for key, value in vars(self).items():
            if key not in {"args", "next_steps", "evidence", "url"}:
                setattr(new, key, value)
        return new

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "error": str(self),
            "url": self.url,
            "next_steps": list(self.next_steps),
            "evidence": dict(self.evidence),
        }


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
        super().__init__(message, next_steps=steps, **kwargs)


class BrowserBotDetectedError(BrowserError):
    """The site blocked the client: anti-bot challenge, captcha wall, or
    an HTTP 403/429 block.

    ``detection`` is one of: cloudflare-challenge, perimeterx,
    datadome, captcha-wall, rate-limit, forbidden.

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

    def __init__(self, message: str, *, detection: str = "forbidden",
                 **kwargs: Any) -> None:
        self.detection = (detection or "forbidden")
        if self.detection not in self._NEXT_STEPS:
            self.detection = "forbidden"
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

_PX_BODY_MARKERS = ("px-captcha", "_px3", "perimeterx.net")
_DD_BODY_MARKERS = ("captcha-delivery.com", "ct.captcha-delivery.com",
                    "datadome")

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
                     headers: Any = None) -> str:
    """Identify an anti-bot challenge page.

    Returns ``cloudflare-challenge`` | ``perimeterx`` | ``datadome`` |
    ``captcha-wall`` | ``""`` (no challenge detected).

    Detection is gated: body markers are only trusted on
    challenge-compatible responses (WAF headers present) or alongside the
    interstitial title family, so ordinary pages that quote the markers
    are not misclassified. ``title`` alone (the "Just a moment..." family)
    is trusted — sites do not title real pages that.
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

    if waf_headers:
        for marker in _CF_BODY_MARKERS:
            if marker in low_html:
                return "cloudflare-challenge"
    # Cloudflare structural markers are challenge-specific enough to
    # trust even without WAF headers (they never appear on real pages).
    for marker in ("_cf_chl_opt", "__cf_chl_rt_tk", "/cdn-cgi/challenge-platform/"):
        if marker in low_html:
            return "cloudflare-challenge"

    for marker in _PX_BODY_MARKERS:
        if marker in low_html:
            return "perimeterx"
    for marker in _DD_BODY_MARKERS:
        if marker in low_html:
            return "datadome"

    if any(t in low_html for t in _WALL_TEXT) and any(
            w in low_html for w in _WALL_WIDGETS):
        return "captcha-wall"
    return ""


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
                         evidence: dict[str, str] | None = None) -> BrowserError:
    """Classify an HTTP error status into the taxonomy.

    403/429/503 (or any status whose body carries challenge markers)
    → :class:`BrowserBotDetectedError`; other 4xx and 5xx →
    :class:`BrowserSiteError`. A 429 without challenge markers is still a
    block (rate-limit), not a broken page. 2xx/3xx here is a caller bug
    and raises ValueError — never silently "classify" success.
    """
    status = int(status or 0)
    if status < 400:
        raise ValueError(
            f"classify_http_status got non-error status {status}")
    where = f" on {url}" if url else ""
    challenge = ""
    if status in _CHALLENGE_STATUSES or headers:
        challenge = detect_challenge(html, title=title, headers=headers)
    if challenge:
        label = challenge.replace("-", " ")
        return BrowserBotDetectedError(
            f"blocked by {label}{where} (HTTP {status}) — "
            f"the page is a {label} interstitial, not the site",
            detection=challenge, url=url, evidence=evidence)
    if status == 429:
        return BrowserBotDetectedError(
            f"rate-limited{where} (HTTP 429) — the site is throttling "
            f"this client",
            detection="rate-limit", url=url, evidence=evidence)
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
    return payload
