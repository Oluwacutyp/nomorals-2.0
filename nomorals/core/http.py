"""Minimal HTTP client on ``urllib``.

Lives in the kernel because providers (L3), tools (L4), and social (L4) all need
it, and a shared dependency must sit below all of them. Stdlib-only: no requests,
no httpx, so it works on Termux with nothing installed.

It adds exactly what the raw stdlib is missing and nothing else: timeouts that
actually apply, automatic redirect handling with a bound, JSON helpers, streaming
downloads with resume, and multipart uploads.
"""

from __future__ import annotations

import gzip
import ipaddress
import json
import mimetypes
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

from .errors import NoMoralsError, ProviderError, RateLimited, TimeoutError_, classify
from .logging_setup import get_logger

__all__ = ["HttpClient", "HttpResponse", "RequestError"]

_log = get_logger(__name__)

DEFAULT_UA = "NoMoralsCore/0.1 (+https://github.com/Oluwacutyp/No-morals-ai)"


# ── Retired endpoint registry ─────────────────────────────────────────
# APIs that training knowledge still suggests but no longer exist. Matched
# by (host, path-prefix); the request is rejected BEFORE any network I/O
# with redirect guidance, instead of dying on a cryptic DNS/HTTP error.
# Add entries here as dead endpoints are discovered — never silently drop.
DEAD_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    # CoinDesk v1 price API was retired; the host no longer serves it.
    ("api.coindesk.com", "/v1/bpi/",
     "CoinDesk v1 price API is retired. Use the finance_price tool "
     "(Binance → CoinGecko fallback chain) for crypto prices."),
)


def _check_dead_endpoint(url: str) -> None:
    """Reject known-retired API endpoints with redirect guidance."""
    try:
        parts = urllib.parse.urlparse(url)
        host = (parts.hostname or "").lower()
        path = parts.path or "/"
    except Exception:  # noqa: BLE001 - unparseable; SSRF guard handles it
        return
    for dead_host, dead_prefix, guidance in DEAD_ENDPOINTS:
        if host == dead_host and path.startswith(dead_prefix):
            raise RequestError(
                f"Retired endpoint {host}{dead_prefix}* — {guidance}")
    return None


class RequestError(ProviderError):
    code = "http.request"


@dataclass
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        # Respect the charset from Content-Type if present; fall back to
        # UTF-8. Only use errors="replace" as a last resort — silent
        # replacement corrupts non-UTF-8 responses (e.g. OpenRouter).
        charset = "utf-8"
        ctype = self.headers.get("content-type", "")
        if "charset=" in ctype:
            charset = ctype.split("charset=")[-1].split(";")[0].strip().strip('"')
        try:
            return self.body.decode(charset, errors="strict")
        except (UnicodeDecodeError, LookupError):
            _log.warning("http: response not decodable as %s, using replace", charset)
            return self.body.decode(charset, errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise RequestError(
                f"response was not valid JSON: {exc}",
                details={"body_head": self.text[:200]},
            ) from exc

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    def raise_for_status(self) -> "HttpResponse":
        if self.ok:
            return self
        raise http_error(self.status, self.text, self.url, self.headers)


def _error_snippet(body: str) -> str:
    """The server's own reason, if it gave one.

    OpenAI-shaped error JSON ({"error": {"message": ...}}) is unwrapped;
    otherwise the raw body trimmed. Empty/HTML bodies stay empty so the
    message never ends in a dangling dash."""
    body = (body or "").strip()
    if not body or body[0] == "<":
        return ""
    if body[0] == "{":
        try:
            import json

            data = json.loads(body)
            message = ((data.get("error") or {}).get("message")
                       if isinstance(data.get("error"), dict)
                       else data.get("error")) or ""
            if message:
                return str(message)[:200]
        except (ValueError, TypeError):  # noqa: E103 - falls through to raw body
            pass
    return body[:200]


def _safe_detail(body: str, limit: int = 500) -> str:
    """Make an error detail from a response body that may be binary.

    Response bodies are decoded with ``errors="replace"``, so a gzip or
    other binary body shows up as mojibake: U+FFFD replacement characters
    mixed with printable garbage. Quoting that corrupts logs and
    terminals, so detect it (a text body has essentially no U+FFFD) and
    say so instead of dumping it.
    """
    text = body or ""
    if text:
        fffd = text.count("�")
        if fffd / len(text) > 0.03:
            return "<non-text (likely compressed) response body>"
    text = "".join(
        ch if (ch.isprintable() or ch in " \t\n\r") else ""
        for ch in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text or "<no readable error detail>"


def _parse_retry_after(headers: Mapping[str, str] | None) -> float:
    """Seconds to wait per the server's rate-limit headers.

    Reads ``Retry-After`` (delay-seconds or HTTP-date) and the common
    ``X-RateLimit-Reset`` epoch variants. Falls back to 1.0s when nothing
    usable is present. Header names are matched case-insensitively so
    both plain dicts and ``http.client.HTTPMessage`` work.
    """
    if not headers:
        return 1.0
    lowered: dict[str, str] = {}
    try:
        items = headers.items()  # type: ignore[union-attr]
    except AttributeError:
        items = []
    for k, v in items:
        lowered[str(k).lower()] = str(v).strip()

    def _get(*names: str) -> str:
        for name in names:
            val = lowered.get(name, "")
            if val:
                return val
        return ""

    raw = _get("retry-after")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            _log.debug("retry-after %r is not seconds; trying HTTP date", raw)
        try:
            from datetime import datetime, timezone
            from email.utils import parsedate_to_datetime
            reset = parsedate_to_datetime(raw)
            if reset.tzinfo is None:
                reset = reset.replace(tzinfo=timezone.utc)
            delta = (reset - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, min(delta, 3600.0))
        except Exception:  # noqa: BLE001 - unparsable date, fall through
            pass
    for name in ("x-ratelimit-reset", "ratelimit-reset",
                 "x-rate-limit-reset"):
        raw = _get(name)
        if raw:
            try:
                delta = float(raw) - time.time()
            except ValueError:
                continue
            if delta > 0:
                return min(delta, 3600.0)
    return 1.0


def http_error(status: int, body: str, url: str = "",
               headers: Mapping[str, str] | None = None) -> NoMoralsError:
    """Map an HTTP status onto the framework error hierarchy.

    Every mapped error carries ``details["http_status"]`` so callers can
    branch on the status *machine-readably* (auth vs not-found vs
    rate-limit) instead of regexing the message.  :func:`classify` passes
    these errors through unchanged, so the status survives into providers.
    """
    detail = _safe_detail(body)
    if status == 429:
        retry_after = _parse_retry_after(headers)
        return RateLimited(f"429 from {url}: {detail}",
                           retry_after=retry_after,
                           details={"http_status": 429})
    if status in {401, 403}:
        # Auth is terminal for this key: never retried, never healed by
        # swapping models — a different model won't fix a bad key.
        return ProviderError(f"{status} unauthorized for {url}: {detail}",
                             retryable=False,
                             details={"http_status": status})
    if status == 404:
        # a 404 without the server's reason is unactionable — "model not
        # found: <retired-id>" is exactly what tells the owner to fix config
        why = _error_snippet(body)
        return ProviderError(
            f"404 not found: {url}" + (f" — {why}" if why else ""),
            retryable=False,
            details={"http_status": 404})
    if status in {408, 425} or status >= 500:
        # 503 included: the server is down or overloaded — back off and retry.
        return ProviderError(f"{status} from {url}: {detail}",
                             retryable=True,
                             details={"http_status": status})
    return ProviderError(f"HTTP {status} from {url}: {detail}",
                         retryable=False,
                         details={"http_status": status})


class HttpClient:
    """Small, predictable HTTP client."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_UA,
        headers: Mapping[str, str] | None = None,
        max_redirects: int = 5,
        verify_tls: bool = True,
        proxy_url: str = "",
        allow_private_ips: bool = False,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.headers = dict(headers or {})
        self.max_redirects = max_redirects
        self.verify_tls = verify_tls
        self.proxy_url = proxy_url
        #: When False (default), requests to private/loopback/link-local IPs
        #: are rejected — SSRF protection. Set True only for explicitly
        #: internal tooling that needs loopback access.
        self.allow_private_ips = allow_private_ips
        self.stats = {"requests": 0, "errors": 0, "bytes": 0}

    def _effective_proxy(self, target: str) -> str:
        """Which proxy (if any) this request goes through.

        Precedence: the client's explicit ``proxy_url`` → the
        registered rotation resolver (called WITH the target URL so
        per-domain affinity applies; zero-arg resolvers still work) →
        the process-wide default (``proxy_set``).
        """
        if self.proxy_url:
            return self.proxy_url
        resolver = get_proxy_resolver()
        if resolver is not None:
            try:
                try:
                    return resolver(target) or ""
                except TypeError:
                    return resolver() or ""
            except Exception:  # noqa: BLE001 - a broken resolver = direct
                _log.debug("proxy resolver raised; going direct",
                           exc_info=True)
        return get_default_proxy()

    def _check_ssrf(self, url: str) -> None:
        """Reject URLs resolving to private/loopback/link-local addresses."""
        if self.allow_private_ips:
            return
        try:
            host = urllib.parse.urlparse(url).hostname or ""
        except Exception:
            raise NoMoralsError(f"SSRF block: unparseable URL {url!r}")
        if not host:
            raise NoMoralsError(f"SSRF block: empty hostname in {url!r}")
        try:
            # Try as literal IP first
            addr = ipaddress.ip_address(host)
        except ValueError:
            # Resolve hostname → check all returned addresses
            try:
                infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC)
            except socket.gaierror:
                # DNS failure — let the actual request raise the real error
                return
            for fam, _, _, _, sockaddr in infos:
                try:
                    addr = ipaddress.ip_address(sockaddr[0])
                except ValueError:
                    continue
                if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved:
                    raise NoMoralsError(
                        f"SSRF block: {host} resolves to private address {addr}"
                    )
            return
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved:
            raise NoMoralsError(f"SSRF block: private address {addr} in {url!r}")

    # ── core ─────────────────────────────────────────────────────────────────
    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        stream_to: str | Path | None = None,
        progress: Callable[[int, int], None] | None = None,
        resume: bool = False,
    ) -> HttpResponse:
        target = url
        if params:
            separator = "&" if urllib.parse.urlparse(url).query else "?"
            target = f"{url}{separator}{urllib.parse.urlencode(params)}"

        # Retired-endpoint guard first: fail fast with redirect guidance
        # before SSRF/DNS (a dead endpoint stays dead whatever DNS says).
        _check_dead_endpoint(target)
        # SSRF guard: check the initial URL before connecting
        self._check_ssrf(target)

        merged = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            **self.headers,
            **(headers or {}),
        }

        opener_kwargs: dict[str, Any] = {}
        if not self.verify_tls:
            import ssl

            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            opener_kwargs["context"] = context

        # ── proxy routing: explicit client proxy → rotation resolver
        # (with the target URL, so domain affinity applies) → the
        # process-wide default.  This is what makes proxy_set and
        # proxy_rotate actually route traffic.
        effective_proxy = self._effective_proxy(target)

        destination = Path(stream_to) if stream_to else None
        start_byte = 0
        if destination is not None and resume and destination.exists():
            start_byte = destination.stat().st_size
            if start_byte:
                merged["Range"] = f"bytes={start_byte}-"

        request = urllib.request.Request(
            target, data=data, headers=merged, method=method.upper()
        )
        self.stats["requests"] += 1
        fetch_start = time.monotonic()
        try:
            if effective_proxy:
                raw_cm = _proxied_open(request, effective_proxy,
                                       timeout or self.timeout,
                                       opener_kwargs)
            else:
                raw_cm = urllib.request.urlopen(
                    request, timeout=timeout or self.timeout,
                    **opener_kwargs)
            with raw_cm as raw:
                final_url = raw.geturl()
                response_headers = {k.lower(): v for k, v in raw.headers.items()}
                if destination is not None:
                    total = start_byte + int(raw.headers.get("Content-Length") or 0)
                    mode = "ab" if start_byte and raw.status == 206 else "wb"
                    written = start_byte if mode == "ab" else 0
                    with destination.open(mode) as handle:
                        while chunk := raw.read(256 * 1024):
                            handle.write(chunk)
                            written += len(chunk)
                            self.stats["bytes"] += len(chunk)
                            if progress:
                                progress(written, total)
                    if effective_proxy:
                        _report_proxy_success(
                            effective_proxy,
                            (time.monotonic() - fetch_start) * 1000.0)
                    return HttpResponse(
                        status=raw.status, body=b"", headers=response_headers, url=final_url
                    )
                payload = self._decode_body(raw.read(), response_headers)
                self.stats["bytes"] += len(payload)
                if effective_proxy:
                    _report_proxy_success(
                        effective_proxy,
                        (time.monotonic() - fetch_start) * 1000.0)
                return HttpResponse(
                    status=raw.status, body=payload, headers=response_headers, url=final_url
                )
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # noqa: BLE001 - error path must not raise
                pass
            self.stats["errors"] += 1
            # an HTTP status came back THROUGH the proxy — it forwarded
            # fine, so this is a success signal for rotation, not a
            # failure (except 407: the proxy itself rejected our auth)
            if effective_proxy:
                if exc.code == 407:
                    _report_proxy_failure(
                        effective_proxy, "proxy authentication failed (407)")
                else:
                    _report_proxy_success(
                        effective_proxy,
                        (time.monotonic() - fetch_start) * 1000.0)
            try:
                err_headers = dict(exc.headers.items()) if exc.headers else None
            except Exception:  # noqa: BLE001 - error path must not raise
                err_headers = None
            raise http_error(exc.code, body.decode("utf-8", "replace"),
                             target, err_headers) from exc
        except urllib.error.URLError as exc:
            self.stats["errors"] += 1
            reason = str(exc.reason)
            _report_proxy_failure(effective_proxy, f"URLError: {reason}")
            if "timed out" in reason.lower():
                raise TimeoutError_(f"request to {target} timed out: {reason}", retryable=True) from exc
            raise RequestError(f"request to {target} failed: {reason}", retryable=True) from exc
        except TimeoutError as exc:
            self.stats["errors"] += 1
            _report_proxy_failure(effective_proxy, "request timed out")
            raise TimeoutError_(f"request to {target} timed out", retryable=True) from exc
        except OSError as exc:
            self.stats["errors"] += 1
            _report_proxy_failure(effective_proxy, f"OSError: {exc}")
            raise classify(exc) from exc

    def _decode_body(self, payload: bytes, headers: Mapping[str, str]) -> bytes:
        encoding = headers.get("content-encoding", "").lower()
        try:
            if "gzip" in encoding:
                return gzip.decompress(payload)
            if "deflate" in encoding:
                try:
                    return zlib.decompress(payload)
                except zlib.error:
                    return zlib.decompress(payload, -zlib.MAX_WBITS)
        except (OSError, zlib.error):
            _log.debug("could not decode body encoding %r; using raw bytes", encoding)
        return payload

    # ── conveniences ─────────────────────────────────────────────────────────
    def get(self, url: str, **kw: Any) -> HttpResponse:
        return self.request("GET", url, **kw)

    def head(self, url: str, **kw: Any) -> HttpResponse:
        return self.request("HEAD", url, **kw)

    def post_json(
        self, url: str, payload: Mapping[str, Any], **kw: Any
    ) -> HttpResponse:
        body = json.dumps(payload, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json", **(kw.pop("headers", None) or {})}
        return self.request("POST", url, data=body, headers=headers, **kw)

    def put_json(
        self, url: str, payload: Mapping[str, Any], **kw: Any
    ) -> HttpResponse:
        body = json.dumps(payload, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json", **(kw.pop("headers", None) or {})}
        return self.request("PUT", url, data=body, headers=headers, **kw)

    def post_form(self, url: str, form: Mapping[str, Any], **kw: Any) -> HttpResponse:
        body = urllib.parse.urlencode(form).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            **(kw.pop("headers", None) or {}),
        }
        return self.request("POST", url, data=body, headers=headers, **kw)

    def post_multipart(
        self,
        url: str,
        fields: Mapping[str, str] | None = None,
        files: Sequence[tuple[str, str | Path, str]] | None = None,
        **kw: Any,
    ) -> HttpResponse:
        """``files`` entries are ``(form_name, path, mime)``."""
        boundary = f"----nomorals{int(time.time() * 1000)}"
        chunks: list[bytes] = []
        for key, value in (fields or {}).items():
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
            )
        for name, path, mime in files or ():
            file_path = Path(path)
            guess = mime or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            chunks.append(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                    f'filename="{file_path.name}"\r\nContent-Type: {guess}\r\n\r\n'
                ).encode()
            )
            chunks.append(file_path.read_bytes())
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        body = b"".join(chunks)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **(kw.pop("headers", None) or {}),
        }
        return self.request("POST", url, data=body, headers=headers, **kw)

    def download(
        self,
        url: str,
        destination: str | Path,
        *,
        resume: bool = True,
        progress: Callable[[int, int], None] | None = None,
        chunk_bytes: int = 256 * 1024,
    ) -> Path:
        """Stream a URL to disk, resuming a partial download when possible."""
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.request(
            "GET",
            url,
            stream_to=target,
            progress=progress,
            resume=resume,
            timeout=max(self.timeout, 120.0),
        )
        return target


def url_filename(url: str, default: str = "download.bin") -> str:
    """Best-effort filename from a URL path."""
    path = urllib.parse.urlparse(url).path
    name = Path(path).name
    return name or default


def default_proxy_handler() -> dict[str, Any] | None:
    """Stub: default proxy handler (removed). Returns None (no proxy)."""
    return None


def apply_socks_proxy(proxy_url: str) -> bool:
    """Route raw-socket traffic through a SOCKS proxy via PySocks.

    Patches :mod:`socket` so urllib and raw-socket code honor the proxy.
    Returns True on success, False if PySocks is not installed.
    """
    global _ambient_socks_url
    try:
        import socks  # type: ignore[import-not-found]
        import socket as _socket
    except ImportError:
        return False
    parsed = urllib.parse.urlparse(proxy_url)
    if not parsed.hostname or not parsed.port:
        return False
    proxy_type = {
        "socks5": socks.SOCKS5,
        "socks5h": socks.SOCKS5,
        "socks4": socks.SOCKS4,
        "socks4a": socks.SOCKS4,
    }.get((parsed.scheme or "").lower(), socks.SOCKS5)
    socks.set_default_proxy(
        proxy_type,
        parsed.hostname,
        parsed.port,
        username=parsed.username,
        password=parsed.password,
    )
    _socket.socket = socks.socksocket  # type: ignore[assignment]
    _ambient_socks_url = proxy_url  # restored after per-request SOCKS windows
    return True


def reset_socks_proxy() -> None:
    """Restore direct socket routing (undo :func:`apply_socks_proxy`)."""
    global _ambient_socks_url
    try:
        import socks  # type: ignore[import-not-found]
        import socket as _socket
    except ImportError:
        _ambient_socks_url = ""
        return
    socks.set_default_proxy()
    _ambient_socks_url = ""
    # Restore the original socket class if it was patched.
    if getattr(_socket.socket, "__module__", "") == "socks":
        import _socket as _real_socket  # type: ignore[import-not-found]

        _socket.socket = _real_socket.socket


_default_proxy: str = ""


def set_default_proxy(proxy_url: str) -> None:
    """Set the process-wide default proxy URL."""
    global _default_proxy
    _default_proxy = proxy_url or ""


def get_default_proxy() -> str:
    """Get the process-wide default proxy URL."""
    return _default_proxy


_proxy_resolver = None


def set_proxy_resolver(resolver):
    """Register a callable that returns a proxy URL for a given target.

    Used by proxylab to route HTTP requests through rotating proxies.
    Pass None to clear.
    """
    global _proxy_resolver
    _proxy_resolver = resolver


def get_proxy_resolver():
    """Return the currently registered proxy resolver, or None."""
    return _proxy_resolver


_proxy_error_reporter = None


def set_proxy_error_reporter(reporter):
    """Register a callable that reports proxy errors for rotation/cooldown.

    Used by proxylab to track proxy failures and remove bad proxies from rotation.
    Pass None to clear.
    """
    global _proxy_error_reporter
    _proxy_error_reporter = reporter


def get_proxy_error_reporter():
    """Return the currently registered proxy error reporter, or None."""
    return _proxy_error_reporter


_proxy_success_reporter = None


def set_proxy_success_reporter(reporter):
    """Register a callable ``(proxy_url, latency_ms)`` for successful
    requests that went through a proxy — feeds the lab's decayed
    latency scoring with real traffic.  Pass None to clear.
    """
    global _proxy_success_reporter
    _proxy_success_reporter = reporter


def get_proxy_success_reporter():
    """Return the currently registered proxy success reporter, or None."""
    return _proxy_success_reporter


def _report_proxy_failure(proxy_url: str, reason: str) -> None:
    if not proxy_url:
        return
    reporter = get_proxy_error_reporter()
    if reporter is None:
        return
    try:
        reporter(proxy_url, reason)
    except Exception:  # noqa: BLE001 - reporting must never break requests
        _log.debug("proxy error reporter raised", exc_info=True)


def _report_proxy_success(proxy_url: str, latency_ms: float) -> None:
    if not proxy_url:
        return
    reporter = get_proxy_success_reporter()
    if reporter is None:
        return
    try:
        reporter(proxy_url, latency_ms)
    except Exception:  # noqa: BLE001 - reporting must never break requests
        _log.debug("proxy success reporter raised", exc_info=True)


_socks_lock = threading.RLock()
_socks_warned_no_pysocks = False


def _socks_available() -> bool:
    try:
        import socks  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


@contextmanager
def _temporary_socks_proxy(proxy_url: str):
    """Route one request through a SOCKS proxy, then restore the ambient
    SOCKS default.  Serialized: while one request holds the window, no
    other thread's proxy setting can interleave."""
    global _socks_warned_no_pysocks
    parsed = urllib.parse.urlparse(proxy_url)
    if not _socks_available():
        if not _socks_warned_no_pysocks:
            _socks_warned_no_pysocks = True
            _log.warning("SOCKS proxy %s requested but PySocks is not "
                         "installed — request goes direct (pip install "
                         "pysocks)", proxy_url)
        yield False  # direct fallback; the caller still performs the request
        return
    import socks  # type: ignore[import-not-found]

    proxy_type = {
        "socks5": socks.SOCKS5,
        "socks5h": socks.SOCKS5,
        "socks4": socks.SOCKS4,
        "socks4a": socks.SOCKS4,
    }.get((parsed.scheme or "").lower(), socks.SOCKS5)
    with _socks_lock:
        try:
            import socket as _socket

            if getattr(_socket.socket, "__module__", "") != "socks":
                _socket.socket = socks.socksocket  # type: ignore[assignment]
            socks.set_default_proxy(
                proxy_type, parsed.hostname, parsed.port,
                username=parsed.username, password=parsed.password)
            yield True
        finally:
            # restore the ambient SOCKS default (proxy_set's global patch
            # or nothing), never leave a per-request proxy behind
            if _ambient_socks_url:
                _restore_socks_url(_ambient_socks_url)
            else:
                socks.set_default_proxy()


_ambient_socks_url = ""


def _restore_socks_url(proxy_url: str) -> None:
    """Re-apply a SOCKS default without touching the ambient record."""
    try:
        import socks  # type: ignore[import-not-found]
    except ImportError:
        return
    parsed = urllib.parse.urlparse(proxy_url)
    proxy_type = {
        "socks5": socks.SOCKS5,
        "socks5h": socks.SOCKS5,
        "socks4": socks.SOCKS4,
        "socks4a": socks.SOCKS4,
    }.get((parsed.scheme or "").lower(), socks.SOCKS5)
    socks.set_default_proxy(
        proxy_type, parsed.hostname, parsed.port,
        username=parsed.username, password=parsed.password)


def _proxied_open(request, proxy_url: str, timeout: float,
                  opener_kwargs: dict):
    """Open ``request`` through ``proxy_url``.

    HTTP(S) proxies → urllib ProxyHandler on a dedicated opener (the
    TLS context still applies).  SOCKS proxies → a scoped PySocks
    default around a plain opener (urllib cannot speak SOCKS itself).
    Returns a context manager yielding the response.
    """
    scheme = (urllib.parse.urlparse(proxy_url).scheme or "").lower()
    if scheme in ("http", "https"):
        handlers: list = [urllib.request.ProxyHandler(
            {"http": proxy_url, "https": proxy_url})]
        ctx = opener_kwargs.get("context")
        if ctx is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        opener = urllib.request.build_opener(*handlers)
        return opener.open(request, timeout=timeout)
    if scheme not in ("socks4", "socks4a", "socks5", "socks5h", "socks"):
        _log.warning("unsupported proxy scheme %r — request goes direct",
                     scheme)
        return urllib.request.urlopen(request, timeout=timeout,
                                      **opener_kwargs)

    @contextmanager
    def _socks_open():
        with _temporary_socks_proxy(proxy_url):
            with urllib.request.urlopen(request, timeout=timeout,
                                        **opener_kwargs) as raw:
                yield raw

    return _socks_open()
