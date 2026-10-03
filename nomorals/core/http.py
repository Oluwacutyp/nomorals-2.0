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
import json
import mimetypes
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

from .errors import NoMoralsError, ProviderError, RateLimited, TimeoutError_, classify
from .logging_setup import get_logger

__all__ = ["HttpClient", "HttpResponse", "RequestError"]

_log = get_logger(__name__)

DEFAULT_UA = "NoMoralsCore/0.1 (+https://github.com/Oluwacutyp/No-morals-ai)"


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
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8", errors="replace"))
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
        raise http_error(self.status, self.text, self.url)


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


def http_error(status: int, body: str, url: str = "") -> NoMoralsError:
    """Map an HTTP status onto the framework error hierarchy."""
    detail = _safe_detail(body)
    if status == 429:
        retry_after = 1.0
        return RateLimited(f"429 from {url}: {detail}", retry_after=retry_after)
    if status in {401, 403}:
        return ProviderError(f"{status} unauthorized for {url}: {detail}", retryable=False)
    if status == 404:
        # a 404 without the server's reason is unactionable — "model not
        # found: <retired-id>" is exactly what tells the owner to fix config
        why = _error_snippet(body)
        return ProviderError(
            f"404 not found: {url}" + (f" — {why}" if why else ""),
            retryable=False)
    if status in {408, 425} or status >= 500:
        return ProviderError(f"{status} from {url}: {detail}", retryable=True)
    if status == 503:
        return ProviderError(f"503 unavailable: {url}", retryable=True)
    return ProviderError(f"HTTP {status} from {url}: {detail}", retryable=False)


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
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.headers = dict(headers or {})
        self.max_redirects = max_redirects
        self.verify_tls = verify_tls
        self.proxy_url = proxy_url
        self.stats = {"requests": 0, "errors": 0, "bytes": 0}

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
        try:
            with urllib.request.urlopen(
                request, timeout=timeout or self.timeout, **opener_kwargs
            ) as raw:
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
                    return HttpResponse(
                        status=raw.status, body=b"", headers=response_headers, url=final_url
                    )
                payload = self._decode_body(raw.read(), response_headers)
                self.stats["bytes"] += len(payload)
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
            raise http_error(exc.code, body.decode("utf-8", "replace"), target) from exc
        except urllib.error.URLError as exc:
            self.stats["errors"] += 1
            reason = str(exc.reason)
            if "timed out" in reason.lower():
                raise TimeoutError_(f"request to {target} timed out: {reason}", retryable=True) from exc
            raise RequestError(f"request to {target} failed: {reason}", retryable=True) from exc
        except TimeoutError as exc:
            self.stats["errors"] += 1
            raise TimeoutError_(f"request to {target} timed out", retryable=True) from exc
        except OSError as exc:
            self.stats["errors"] += 1
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
    return True


def reset_socks_proxy() -> None:
    """Restore direct socket routing (undo :func:`apply_socks_proxy`)."""
    try:
        import socks  # type: ignore[import-not-found]
        import socket as _socket
    except ImportError:
        return
    socks.set_default_proxy()
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
