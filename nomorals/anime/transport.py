"""One injectable HTTP seam for the anime organ.

Every source client in this package talks through a ``Transport``:
a callable taking ``(method, url, json_body=…, params=…, files=…,
timeout=…)`` and returning an :class:`HttpResult`.  The production
transport rides :class:`nomorals.core.http.HttpClient` (the house
client — proxy/TLS discipline included); tests substitute a fake
that answers from fixtures, so the whole suite runs network-free.

Network-level trouble (timeout, DNS, TLS, connection reset) raises
:class:`TransportError` — callers turn it into a named per-source
failure, mirroring the /play resolver's discipline: every source
tried is accounted for, none fails silently or fabricates a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "HttpResult",
    "Transport",
    "TransportError",
    "core_http_transport",
]

#: Per-request ceiling for anime sources: these are small JSON APIs;
#: a slow one is a failed source, not a hung chat.
DEFAULT_TIMEOUT = 12.0


class TransportError(Exception):
    """The request never produced an HTTP response (timeout, DNS…)."""


@dataclass
class HttpResult:
    """One completed HTTP exchange, JSON pre-parsed when possible."""

    status: int
    data: Any = None
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class Transport:
    """The transport protocol: one method, duck-typed on purpose."""

    def request(self, method: str, url: str, *,
                json_body: Any = None,
                params: dict[str, Any] | None = None,
                files: list[tuple[str, str, str]] | None = None,
                timeout: float = DEFAULT_TIMEOUT) -> HttpResult:
        raise NotImplementedError


class _CoreHttpTransport(Transport):
    """Production transport over the house HttpClient."""

    def __init__(self) -> None:
        self._client: Any = None

    def _client_or_build(self) -> Any:
        if self._client is None:
            from ..core.http import HttpClient

            self._client = HttpClient(
                timeout=DEFAULT_TIMEOUT,
                user_agent="Devon/anime (+https://graphql.anilist.co)",
            )
        return self._client

    def request(self, method: str, url: str, *,
                json_body: Any = None,
                params: dict[str, Any] | None = None,
                files: list[tuple[str, str, str]] | None = None,
                timeout: float = DEFAULT_TIMEOUT) -> HttpResult:
        from ..core.http import RequestError

        client = self._client_or_build()
        full_url = url
        if params:
            from urllib.parse import urlencode

            sep = "&" if "?" in url else "?"
            full_url = f"{url}{sep}{urlencode(params)}"
        try:
            if files:
                resp = client.post_multipart(full_url, files=files,
                                             timeout=timeout)
            elif method.upper() == "POST":
                resp = client.post_json(full_url, json_body or {},
                                        timeout=timeout)
            else:
                resp = client.get(full_url, timeout=timeout)
        except RequestError as exc:
            raise TransportError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - one seam, named failure
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        data = None
        if resp.body:
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001 - non-JSON body stays bytes
                data = None
        headers = {str(k).lower(): str(v)
                   for k, v in (resp.headers or {}).items()}
        return HttpResult(status=int(resp.status), data=data,
                          content=resp.body or b"", headers=headers,
                          url=full_url)


def core_http_transport() -> Transport:
    """The production transport (one shared instance)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = _CoreHttpTransport()
    return _SHARED


_SHARED: Transport | None = None
