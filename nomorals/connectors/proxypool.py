"""Proxy pool connector — Devon's own proxy rotation pool.

A LOCAL management interface (AuthMethod.NONE): Devon keeps its own pool
of proxies for rotation across research, fetching, and other outbound
work. Supported protocols: ``http``, ``https`` (forward proxies), and
``socks4`` / ``socks4a`` / ``socks5``.

SOCKS support is a real implementation, not a label: neither ``PySocks``
nor ``python-socks`` is installed in this runtime, so the SOCKS4/4a/5
handshakes are implemented directly on stdlib ``socket`` (SOCKS5
RFC 1928 greeting/auth/CONNECT, SOCKS4/4a CONNECT). SOCKS health probes
open a real tunnel through the proxy and fetch the check URL through it
(plain HTTP, or TLS-wrapped for https:// targets). SOCKS4/4a have no
password authentication by protocol design — ``add_proxy`` fails fast
when a password is given for them (use SOCKS5 for user/pass proxies).

Storage: one vault credential per proxy under ``"connector:proxypool"``.
The vault username is the proxy's endpoint id (``protocol://host:port``);
the encrypted vault secret is a small JSON blob holding the proxy's
auth username/password (so password-less proxies still store a
decryptable secret); the endpoint, tags, and health history ride in
metadata. Proxy passwords never appear in plaintext config, logs, or
list output.

Health is never guessed: :meth:`health_check` sends a real HTTP request
through each proxy (stdlib urllib + ProxyHandler for HTTP(S) proxies,
auth via the proxy URL's userinfo; manual SOCKS tunnel for SOCKS
proxies) and records latency, outcome, and timestamp. :meth:`rotate`
round-robins over proxies whose last recorded check was healthy.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import re
import socket
import ssl
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..accounts.vault import Credential
from ..core.errors import NotFound
from ..core.logging_setup import get_logger
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["ProxyPoolConnector", "ProxyPoolError"]

_log = get_logger(__name__)

#: Health states recorded per proxy.
HEALTHY = "healthy"
UNHEALTHY = "unhealthy"
UNKNOWN = "unknown"

#: Human-readable SOCKS5 CONNECT reply codes (RFC 1928 §6).
_SOCKS5_ERRORS = {
    0x01: "general SOCKS server failure",
    0x02: "connection not allowed by ruleset",
    0x03: "network unreachable",
    0x04: "host unreachable",
    0x05: "connection refused",
    0x06: "TTL expired",
    0x07: "command not supported",
    0x08: "address type not supported",
}

#: SOCKS4a reply codes.
_SOCKS4_ERRORS = {
    0x5B: "request rejected or failed",
    0x5C: "request rejected (no identd on client)",
    0x5D: "request rejected (identd identity mismatch)",
}


class ProxyPoolError(ConnectorError):
    """A proxy pool operation failed."""


@register_connector
class ProxyPoolConnector(Connector):
    """Devon's proxy pool: store, health-check, and rotate proxies.

    HTTP(S) forward proxies plus SOCKS4/SOCKS4a/SOCKS5 (handshakes
    implemented on stdlib socket — no PySocks dependency).
    """

    id = "proxypool"
    name = "Proxy Pool"
    description = (
        "Manage Devon's own pool of proxies: store endpoints with "
        "vault-held credentials, health-check them with real requests, and "
        "rotate across the healthy ones. HTTP(S) forward proxies and "
        "SOCKS4/SOCKS4a/SOCKS5. Local service — no external auth."
    )
    auth_methods = (AuthMethod.NONE,)
    CATEGORY = "network"
    PROVISIONABLE = ("proxy",)

    #: Proxy protocols this pool can store and actively health-check.
    #: ``"socks"`` is accepted as an alias for ``"socks5"`` on the way in.
    PROTOCOLS = ("http", "https", "socks4", "socks4a", "socks5")

    #: Default target for health checks: small, stable, plain HTTP.
    DEFAULT_CHECK_URL = "http://example.com/"

    #: Password placeholder used in every non-secret view.
    MASK = "***"

    #: Rotation strategies for select(): power-of-two-choices (proxyhive
    #: default — avoids hotspotting), round-robin, random, least-latency.
    STRATEGIES = ("p2c", "round_robin", "random", "least_latency")

    #: EWMA smoothing for latency (proxyhive default 0.3).
    LATENCY_EWMA_ALPHA = 0.3

    def __init__(self, vault: Any, http: Any = None) -> None:
        super().__init__(vault, http=http)
        self._rr = 0  # round-robin cursor (in-memory; resets on disconnect)
        #: sticky sessions: key -> (proxy_id, expires_at), in-memory.
        self._sticky: dict[str, tuple[str, float]] = {}

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self, *, check: bool = True) -> ConnectResult:
        """Validate the pool is usable.

        There is no external auth to perform — "connected" means the pool
        holds proxies. With ``check=True`` (default) every proxy is
        live-probed first, so a successful connect means at least one
        proxy actually works right now. An empty pool is an honest
        failure, never a fake success.
        """
        proxies = self._load_all()
        if not proxies:
            return ConnectResult(
                ok=False,
                account="local proxy pool",
                message=(
                    "proxy pool is empty — nothing to connect to. Add your "
                    "first proxy with "
                    "add_proxy(host, port, username=..., password=...) "
                    "or provision('proxy', host=..., port=...)."
                ),
            )
        if check:
            summary = self.health_check()
            healthy = summary["healthy"]
            if not healthy:
                return ConnectResult(
                    ok=False,
                    account="local proxy pool",
                    message=(
                        f"pool holds {len(proxies)} proxies but none passed "
                        "a live health check — check endpoints and "
                        "credentials, then run health_check() again."
                    ),
                )
            return ConnectResult(
                ok=True,
                account="local proxy pool",
                message=(
                    f"proxy pool ready: {healthy}/{len(proxies)} proxies "
                    "healthy (live-checked)."
                ),
            )
        return ConnectResult(
            ok=True,
            account="local proxy pool",
            message=(
                f"proxy pool holds {len(proxies)} proxies "
                "(not live-checked; pass check=True to probe them)."
            ),
        )

    def disconnect(self) -> None:
        """Reset the rotation cursor. Idempotent.

        Stored proxies are NOT removed — they live in the encrypted
        vault until explicitly deleted with :meth:`remove_proxy`.
        """
        self._rr = 0
        _log.info("proxypool disconnected (rotation cursor reset)")

    def status(self) -> ConnectorStatus:
        proxies = self._load_all()
        if not proxies:
            return ConnectorStatus(
                connected=False,
                account="local proxy pool",
                detail=(
                    "proxy pool is empty — add one with "
                    "add_proxy(host, port)"
                ),
            )
        healthy = unhealthy = unknown = 0
        last_checked = 0.0
        for cred in proxies:
            health = (cred.metadata or {}).get("health") or {}
            state = health.get("status", UNKNOWN)
            if state == HEALTHY:
                healthy += 1
            elif state == UNHEALTHY:
                unhealthy += 1
            else:
                unknown += 1
            last_checked = max(last_checked, float(health.get("last_checked", 0.0) or 0.0))
        parts = [f"{len(proxies)} proxies", f"{healthy} healthy"]
        if unhealthy:
            parts.append(f"{unhealthy} unhealthy")
        if unknown:
            parts.append(f"{unknown} unchecked")
        detail = ", ".join(parts)
        if last_checked:
            detail += f"; last check {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(last_checked))}"
        else:
            detail += "; never checked"
        return ConnectorStatus(
            connected=healthy > 0,
            account="local proxy pool",
            last_checked=last_checked,
            detail=detail,
        )

    def test_connection(self) -> bool:
        """True when at least one proxy passes a live health check now."""
        try:
            return self.health_check()["healthy"] > 0
        except ConnectorError:
            return False

    # ── provisioning ───────────────────────────────────────────

    def provision(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        """Provision a proxy into the pool (the pool's provisioning action)."""
        if kind == "proxy":
            return self.add_proxy(**kwargs)
        raise ConnectorError(
            f"proxypool cannot provision {kind!r} "
            f"(provisionable: {', '.join(self.PROVISIONABLE)})"
        )

    # ── pool management ────────────────────────────────────────

    def add_proxy(
        self,
        host: str,
        port: int | str,
        username: str | None = None,
        password: str | None = None,
        protocol: str = "http",
        tags: list[str] | tuple[str, ...] | None = None,
        country: str = "",
        city: str = "",
    ) -> dict[str, Any]:
        """Store a proxy endpoint. Re-adding an endpoint updates it in place.

        ``protocol`` is ``http``/``https`` (forward proxy), ``socks5``
        (``"socks"`` is accepted as an alias), ``socks4a`` (SOCKS4 with
        remote DNS), or ``socks4`` (SOCKS4, IPv4 literals only). SOCKS4
        and SOCKS4a have no password authentication by protocol design —
        passing a password with them raises immediately (use SOCKS5 for
        user/pass proxies).

        ``country``/``city`` (ISO-3166 / free text) power geo-filtered
        selection via :meth:`select` — proxyhive-style geo targeting.

        The proxy password is encrypted into the vault; list/status views
        never expose it. Returns the stored proxy with the password masked.
        """
        protocol = (protocol or "").strip().lower()
        if protocol == "socks":
            protocol = "socks5"  # alias: bare "socks" means SOCKS5
        if protocol not in self.PROTOCOLS:
            raise ProxyPoolError(
                f"unsupported proxy protocol {protocol!r}: "
                f"proxypool supports {', '.join(self.PROTOCOLS)}"
            )
        host = self._normalize_host(host)
        port_num = self._normalize_port(port)
        proxy_user = (username or "").strip() or ""
        if bool(proxy_user) != (password is not None and password != ""):
            raise ProxyPoolError(
                "proxy auth needs both username and password, or neither"
            )
        if protocol in ("socks4", "socks4a") and password:
            raise ProxyPoolError(
                f"{protocol} has no password authentication by protocol "
                "design — store this proxy as socks5 (user/pass) or drop "
                "the password"
            )
        clean_tags = self._normalize_tags(tags)
        proxy_id = f"{protocol}://{host}:{port_num}"

        # Preserve health history when an endpoint is re-added/updated.
        health = self._blank_health()
        with contextlib.suppress(NotFound):
            existing = self.vault.get(self._service, proxy_id, mark_used=False)
            health = dict((existing.metadata or {}).get("health") or {}) or self._blank_health()

        metadata = {
            "proxy_id": proxy_id,
            "host": host,
            "port": port_num,
            "protocol": protocol,
            "proxy_username": proxy_user,
            "tags": clean_tags,
            "country": (country or "").strip().upper(),
            "city": (city or "").strip(),
            "health": health,
        }
        cred = self.vault.store(
            service=self._service,
            username=proxy_id,
            # The vault cannot decrypt an empty secret, so the credential
            # is always a JSON blob — decryptable even with no auth.
            password=json.dumps(
                {"username": proxy_user, "password": password or ""}
            ),
            credential_type="proxy",
            tags=["connector", "proxypool", *clean_tags],
            metadata=metadata,
        )
        _log.info("proxypool: stored proxy %s", proxy_id)
        return self._public_view(cred, has_auth=bool(proxy_user))

    def list_proxies(self) -> list[dict[str, Any]]:
        """Every stored proxy with health status. Passwords are masked."""
        creds = sorted(self._load_all(), key=lambda c: c.username)
        return [self._public_view(c) for c in creds]

    def remove_proxy(self, proxy_id: str) -> None:
        """Delete a proxy from the pool. Idempotent — unknown ids are a no-op."""
        pid = self._normalize_id(proxy_id)
        with contextlib.suppress(NotFound):
            self.vault.delete(self._service, pid)
        _log.info("proxypool: removed proxy %s", pid)

    def get_proxy(self, proxy_id: str) -> dict[str, Any]:
        """Full connection details for one proxy, credential included.

        For use by other components (rotation, HttpClient wiring). The
        returned dict contains the real password and a ``url_with_auth``
        form — treat it as secret.
        """
        cred = self._require_proxy(proxy_id)
        return self._details(cred, include_secret=True)

    # ── health ─────────────────────────────────────────────────

    def health_check(
        self,
        proxy_id: str | None = None,
        *,
        url: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Probe proxies with a real HTTP request sent through each one.

        * ``proxy_id`` given: check one proxy. On failure the unhealthy
          state is recorded and a :class:`ProxyPoolError` is raised —
          failures are never silent marks.
        * ``proxy_id`` omitted: check every proxy concurrently and return
          a summary with per-proxy results. Individual failures are
          recorded on their proxies; the summary reports them.
        """
        url = url or self.DEFAULT_CHECK_URL
        if proxy_id is not None:
            return self._check_one(proxy_id, url=url, timeout=timeout)
        creds = self._load_all()
        if not creds:
            raise ProxyPoolError(
                "proxy pool is empty — add a proxy with "
                "add_proxy(host, port) before health-checking"
            )
        # Decrypt credentials on this thread first: vault reads happen
        # here, workers only run the (slow) network probes, and health
        # writes happen back here. Keeps every DB touch on one thread.
        targets: list[tuple[str, str]] = []
        for cred in creds:
            full = self.vault.get(self._service, cred.username, mark_used=False)
            targets.append((cred.username, self._proxy_url(full, with_auth=True)))
        results: dict[str, dict[str, Any]] = {}
        workers = max(1, min(8, len(targets)))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="proxypool-check"
        ) as pool:
            future_to_id = {
                pool.submit(self._probe_via_proxy, proxy_url, url, timeout): pid
                for pid, proxy_url in targets
            }
            for future in concurrent.futures.as_completed(future_to_id):
                pid = future_to_id[future]
                try:
                    status_code, latency_ms = future.result()
                    ok, error = True, ""
                except ProxyPoolError as exc:
                    status_code, latency_ms, ok, error = None, None, False, str(exc)
                updated = self._record_health(
                    pid,
                    ok=ok,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    error=error,
                )
                results[pid] = self._public_view(updated)
        healthy = sum(1 for r in results.values() if r["health"]["status"] == HEALTHY)
        summary = {
            "total": len(creds),
            "healthy": healthy,
            "unhealthy": sum(
                1 for r in results.values() if r["health"]["status"] == UNHEALTHY
            ),
            "results": results,
        }
        _log.info(
            "proxypool health check: %d/%d healthy", healthy, len(creds)
        )
        return summary

    def _check_one(
        self, proxy_id: str, *, url: str, timeout: float
    ) -> dict[str, Any]:
        result = self._probe_and_record(proxy_id, url, timeout)
        if result["health"]["status"] != HEALTHY:
            raise ProxyPoolError(
                f"proxy {result['id']} failed health check: "
                f"{result['health']['last_error']}"
            )
        return result

    def _probe_and_record(
        self, proxy_id: str, url: str, timeout: float
    ) -> dict[str, Any]:
        """Probe one proxy and persist its health. Returns the masked view."""
        cred = self._require_proxy(proxy_id)
        proxy_url = self._proxy_url(cred, with_auth=True)
        try:
            status_code, latency_ms = self._probe_via_proxy(
                proxy_url, url, timeout
            )
            error = ""
            ok = True
        except ProxyPoolError as exc:
            status_code, latency_ms, error, ok = None, None, str(exc), False
        updated = self._record_health(
            proxy_id,
            ok=ok,
            latency_ms=latency_ms,
            status_code=status_code,
            error=error,
        )
        return self._public_view(updated)

    def _probe_via_proxy(
        self, proxy_url: str, url: str, timeout: float
    ) -> tuple[int, float]:
        """Send one real HTTP request through the proxy.

        Returns (status code, latency_ms). Any completed HTTP response —
        whatever the status — proves the proxy forwarded the request, so
        it counts as healthy. Network/auth failures raise ProxyPoolError
        with a specific reason. SOCKS proxies go through the manual
        socket tunnel below; urllib's ProxyHandler cannot speak SOCKS.
        """
        parsed = urllib.parse.urlparse(proxy_url)
        if parsed.scheme in ("socks4", "socks4a", "socks5", "socks"):
            return self._probe_via_socks(parsed, url, timeout)
        handler = urllib.request.ProxyHandler(
            {"http": proxy_url, "https": proxy_url}
        )
        opener = urllib.request.build_opener(handler)
        start = time.monotonic()
        try:
            with opener.open(url, timeout=timeout) as resp:
                status = int(resp.status)
                resp.read(65536)  # drain enough to prove the body flows
        except urllib.error.HTTPError as exc:
            # We got an HTTP response through the proxy — it forwarded.
            # 407 means the proxy itself rejected our credentials.
            if exc.code == 407:
                raise ProxyPoolError(
                    "proxy authentication failed (407): bad username/password"
                ) from exc
            return int(exc.code), (time.monotonic() - start) * 1000.0
        except urllib.error.URLError as exc:
            raise ProxyPoolError(
                f"proxy request failed: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProxyPoolError(
                f"proxy request timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise ProxyPoolError(f"proxy connection failed: {exc}") from exc
        return status, (time.monotonic() - start) * 1000.0

    # ── SOCKS probing (stdlib socket, no PySocks) ──────────────────

    def _probe_via_socks(
        self, proxy: urllib.parse.ParseResult, url: str, timeout: float
    ) -> tuple[int, float]:
        """Probe the check URL through a SOCKS proxy's TCP tunnel.

        Handshakes SOCKS4/4a/5 manually, sends a real HTTP request
        through the tunnel (TLS-wrapped for https:// targets), and
        returns (status code, latency_ms) on any completed HTTP response.
        """
        target = urllib.parse.urlparse(url)
        if target.scheme not in ("http", "https"):
            raise ProxyPoolError(
                f"cannot probe {url!r} through SOCKS: only http/https "
                "targets are supported"
            )
        target_host = target.hostname or ""
        if not target_host:
            raise ProxyPoolError(f"cannot probe {url!r}: no target host")
        target_port = target.port or (443 if target.scheme == "https" else 80)
        path = target.path or "/"
        if target.query:
            path += "?" + target.query

        protocol = proxy.scheme
        proxy_host = proxy.hostname or ""
        proxy_port = proxy.port or 1080
        username = urllib.parse.unquote(proxy.username or "")
        password = urllib.parse.unquote(proxy.password or "")

        start = time.monotonic()
        sock: socket.socket | None = None
        try:
            try:
                sock = socket.create_connection(
                    (proxy_host, proxy_port), timeout=timeout
                )
            except OSError as exc:
                raise ProxyPoolError(
                    f"socks proxy {proxy_host}:{proxy_port} unreachable: "
                    f"{exc}"
                ) from exc
            sock.settimeout(timeout)
            if protocol in ("socks4", "socks4a"):
                self._socks4_handshake(
                    sock, protocol, target_host, target_port, username
                )
            else:
                self._socks5_handshake(
                    sock, target_host, target_port, username, password
                )
            stream: socket.socket = sock
            if target.scheme == "https":
                context = ssl.create_default_context()
                try:
                    stream = context.wrap_socket(
                        sock, server_hostname=target_host
                    )
                except (ssl.SSLError, OSError) as exc:
                    raise ProxyPoolError(
                        f"socks TLS to {target_host} failed: {exc}"
                    ) from exc
            request = (
                f"GET {path} HTTP/1.0\r\n"
                f"Host: {target_host}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
            try:
                stream.sendall(request)
                raw = self._read_http_head(stream)
            except (OSError, ssl.SSLError) as exc:
                raise ProxyPoolError(
                    f"socks tunnel request failed: {exc}"
                ) from exc
            finally:
                if stream is not sock:
                    with contextlib.suppress(OSError):
                        stream.close()
        finally:
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
        status = self._parse_http_status(raw, target_host)
        return status, (time.monotonic() - start) * 1000.0

    @staticmethod
    def _recv_exact(sock: socket.socket, count: int) -> bytes:
        """Read exactly ``count`` bytes or raise on EOF."""
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            try:
                chunk = sock.recv(remaining)
            except socket.timeout as exc:
                raise ProxyPoolError(
                    "socks proxy timed out mid-handshake"
                ) from exc
            if not chunk:
                raise ProxyPoolError(
                    "socks proxy closed the connection mid-handshake"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _socks5_handshake(
        self,
        sock: socket.socket,
        target_host: str,
        target_port: int,
        username: str,
        password: str,
    ) -> None:
        """RFC 1928: greeting (+ user/pass auth) then CONNECT."""
        # 1. greeting: version, method count, methods
        if username:
            sock.sendall(b"\x05\x02\x00\x02")  # no-auth or user/pass
        else:
            sock.sendall(b"\x05\x01\x00")  # no-auth only
        ver, method = self._recv_exact(sock, 2)
        if ver != 0x05:
            raise ProxyPoolError(
                f"socks5 proxy answered with bad version byte {ver:#x}"
            )
        if method == 0xFF:
            raise ProxyPoolError(
                "socks5 proxy accepts no offered auth method"
            )
        if method == 0x02:
            # RFC 1929 username/password subnegotiation
            user_b = username.encode("utf-8")
            pass_b = password.encode("utf-8")
            if len(user_b) > 255 or len(pass_b) > 255:
                raise ProxyPoolError(
                    "socks5 username/password must be <= 255 bytes"
                )
            sock.sendall(
                b"\x01"
                + bytes([len(user_b)]) + user_b
                + bytes([len(pass_b)]) + pass_b
            )
            auth_ver, auth_status = self._recv_exact(sock, 2)
            if auth_ver != 0x01 or auth_status != 0x00:
                raise ProxyPoolError(
                    "socks5 proxy authentication failed: bad "
                    "username/password"
                )
        elif method != 0x00:
            raise ProxyPoolError(
                f"socks5 proxy chose unsupported auth method {method:#x}"
            )
        # 2. CONNECT request: VER CMD RSV ATYP ADDR PORT
        try:
            addr = socket.inet_pton(socket.AF_INET, target_host)
            atyp, addr_field = b"\x01", addr
        except OSError:
            try:
                addr = socket.inet_pton(socket.AF_INET6, target_host)
                atyp, addr_field = b"\x04", addr
            except OSError:
                host_b = target_host.encode("idna")
                if len(host_b) > 255:
                    raise ProxyPoolError(
                        f"socks5 target hostname too long: {target_host!r}"
                    ) from None
                atyp = b"\x03"
                addr_field = bytes([len(host_b)]) + host_b
        sock.sendall(
            b"\x05\x01\x00" + atyp + addr_field
            + struct.pack(">H", target_port)
        )
        # 3. reply: VER REP RSV ATYP BND.ADDR BND.PORT
        ver, rep, _rsv, atyp_b = self._recv_exact(sock, 4)
        if ver != 0x05:
            raise ProxyPoolError(
                f"socks5 proxy answered CONNECT with bad version {ver:#x}"
            )
        if rep != 0x00:
            detail = _SOCKS5_ERRORS.get(rep, f"unknown reply {rep:#x}")
            raise ProxyPoolError(f"socks5 CONNECT failed: {detail}")
        atyp = atyp_b
        if atyp == 0x01:
            self._recv_exact(sock, 4)
        elif atyp == 0x04:
            self._recv_exact(sock, 16)
        elif atyp == 0x03:
            name_len = self._recv_exact(sock, 1)[0]
            self._recv_exact(sock, name_len)
        else:
            raise ProxyPoolError(
                f"socks5 proxy returned unknown address type {atyp:#x}"
            )
        self._recv_exact(sock, 2)  # bound port

    def _socks4_handshake(
        self,
        sock: socket.socket,
        protocol: str,
        target_host: str,
        target_port: int,
        username: str,
    ) -> None:
        """SOCKS4/4a CONNECT. SOCKS4 needs an IPv4 literal; 4a resolves
        the hostname at the proxy (0.0.0.1 marker)."""
        if protocol == "socks4":
            try:
                ip_bytes = socket.inet_pton(socket.AF_INET, target_host)
            except OSError as exc:
                raise ProxyPoolError(
                    f"socks4 cannot resolve hostnames — use socks4a or "
                    f"socks5 for {target_host!r}"
                ) from exc
            host_field = ip_bytes
            domain_field = b""
        else:  # socks4a: proxy-side DNS
            host_field = b"\x00\x00\x00\x01"
            domain_field = target_host.encode("idna") + b"\x00"
        user_b = username.encode("utf-8") + b"\x00"
        sock.sendall(
            b"\x04\x01"
            + struct.pack(">H", target_port)
            + host_field
            + user_b
            + domain_field
        )
        reply = self._recv_exact(sock, 8)
        if reply[0] != 0x00:
            raise ProxyPoolError(
                f"{protocol} proxy answered with bad version byte "
                f"{reply[0]:#x}"
            )
        if reply[1] != 0x5A:
            detail = _SOCKS4_ERRORS.get(
                reply[1], f"unknown reply {reply[1]:#x}"
            )
            raise ProxyPoolError(f"{protocol} CONNECT failed: {detail}")

    @staticmethod
    def _read_http_head(stream: socket.socket) -> bytes:
        """Read an HTTP response through the tunnel: headers + up to 64KB.

        Reading headers first keeps the probe honest about protocol-level
        failures; draining a body chunk proves bytes actually flow.
        """
        buf = b""
        while b"\r\n\r\n" not in buf:
            try:
                chunk = stream.recv(65536)
            except socket.timeout as exc:
                raise ProxyPoolError(
                    "socks tunnel timed out waiting for the response"
                ) from exc
            if not chunk:
                break
            buf += chunk
            if len(buf) > 1 << 20:
                break
        try:
            body = stream.recv(65536)
        except (OSError, socket.timeout):
            body = b""
        return buf + body

    @staticmethod
    def _parse_http_status(raw: bytes, target_host: str) -> int:
        """The status code off the response's status line."""
        head = raw.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        match = re.match(r"HTTP/\d(?:\.\d)?\s+(\d{3})", head)
        if not match:
            raise ProxyPoolError(
                f"socks tunnel to {target_host} returned a non-HTTP "
                f"response: {head[:80]!r}"
            )
        return int(match.group(1))

    def _record_health(
        self,
        proxy_id: str,
        *,
        ok: bool,
        latency_ms: float | None,
        status_code: int | None,
        error: str,
    ) -> Credential:
        cred = self.vault.get(self._service, proxy_id, mark_used=False)
        meta = dict(cred.metadata or {})
        health = dict(meta.get("health") or {}) or self._blank_health()
        now = time.time()
        prev_ewma = health.get("latency_ewma_ms")
        ewma = prev_ewma
        if ok and latency_ms is not None:
            alpha = self.LATENCY_EWMA_ALPHA
            ewma = (latency_ms if prev_ewma is None
                    else alpha * latency_ms + (1 - alpha) * prev_ewma)
        checks = int(health.get("checks", 0) or 0) + 1
        successes = int(health.get("successes", 0) or 0) + (1 if ok else 0)
        consecutive = (0 if ok
                       else int(health.get("consecutive_failures", 0) or 0) + 1)
        health.update(
            {
                "status": HEALTHY if ok else UNHEALTHY,
                "latency_ms": latency_ms,
                "latency_ewma_ms": ewma,
                "status_code": status_code,
                "last_checked": now,
                "last_error": error if not ok else "",
                "consecutive_failures": consecutive,
                "checks": checks,
                "successes": successes,
                "score": self._health_score(successes, checks, ewma),
            }
        )
        # Automatic exponential backoff: 3+ consecutive failures parks the
        # proxy in quarantine with a growing cooldown (production proxy
        # practice) instead of letting rotation keep tripping over it.
        if consecutive >= 3:
            cooldown = min(300.0 * (2 ** (consecutive - 3)), 3600.0)
            health["quarantined_until"] = now + cooldown
            health["quarantine_reason"] = (
                f"auto: {consecutive} consecutive failures"
                + (f" ({error[:80]})" if error else "")
            )
        elif ok:
            health["quarantined_until"] = 0.0
            health["quarantine_reason"] = ""
        meta["health"] = health
        return self.vault.store(
            service=self._service,
            username=proxy_id,
            password=cred.password,
            credential_type=cred.credential_type,
            tags=list(cred.tags or []),
            metadata=meta,
        )

    @staticmethod
    def _health_score(
        successes: int, checks: int, ewma_ms: float | None
    ) -> float:
        """0–100 proxy quality score (proxyhub formula).

        ``score = success_rate * 60 + latency_score * 40`` where the
        latency curve maps 100ms→~100 and 2000ms+→0. Unchecked proxies
        score a neutral 50 so new entries are selectable.
        """
        if checks <= 0:
            return 50.0
        success_rate = max(0.0, min(1.0, successes / checks))
        if ewma_ms is None:
            latency_score = 50.0
        else:
            latency_score = max(0.0, min(100.0, 100.0 * (2000.0 - ewma_ms) / 1900.0))
        return round(success_rate * 60.0 + latency_score * 0.4, 1)

    # ── rotation ───────────────────────────────────────────────

    def _quarantined(self, health: dict[str, Any],
                     now: float | None = None) -> bool:
        until = float(health.get("quarantined_until", 0.0) or 0.0)
        return until > (time.time() if now is None else now)

    def _eligible_proxies(
        self,
        *,
        country: str | None = None,
        protocol: str | None = None,
        max_latency_ms: float | None = None,
        min_score: float = 0.0,
        include_unchecked: bool = False,
    ) -> list[Credential]:
        """Healthy, un-quarantined proxies matching the filters.

        ``include_unchecked=True`` also admits never-checked proxies
        (neutral score 50) — useful for bootstrapping a fresh pool.
        """
        now = time.time()
        wanted_country = (country or "").strip().upper()
        wanted_proto = (protocol or "").strip().lower()
        out: list[Credential] = []
        for cred in sorted(self._load_all(), key=lambda c: c.id):
            meta = cred.metadata or {}
            health = dict(meta.get("health") or {})
            if self._quarantined(health, now):
                continue
            status = health.get("status", UNKNOWN)
            if status != HEALTHY and not (
                    include_unchecked and status == UNKNOWN):
                continue
            if wanted_country and str(meta.get("country", "")).upper() != wanted_country:
                continue
            if wanted_proto and str(meta.get("protocol", "")) != wanted_proto:
                continue
            if max_latency_ms is not None:
                ewma = health.get("latency_ewma_ms")
                if ewma is not None and float(ewma) > max_latency_ms:
                    continue
            if float(health.get("score", 50.0) or 50.0) < min_score:
                continue
            out.append(cred)
        return out

    @staticmethod
    def _proxy_score(cred: Credential) -> float:
        return float(((cred.metadata or {}).get("health") or {}).get(
            "score", 50.0) or 50.0)

    def select(
        self,
        strategy: str = "p2c",
        *,
        country: str | None = None,
        protocol: str | None = None,
        max_latency_ms: float | None = None,
        min_score: float = 0.0,
        include_unchecked: bool = False,
    ) -> dict[str, Any]:
        """Pick one proxy by strategy (proxyhive-style).

        Strategies: ``p2c`` (power of two choices — sample two at random,
        take the higher score; the default, avoids hotspotting),
        ``round_robin``, ``random``, ``least_latency`` (lowest EWMA).
        Filters: ``country`` (ISO-3166), ``protocol``, ``max_latency_ms``
        (EWMA ceiling), ``min_score``. Raises :class:`ProxyPoolError`
        when nothing is eligible.
        """
        import random as _random

        strategy = (strategy or "p2c").strip().lower()
        if strategy not in self.STRATEGIES:
            raise ProxyPoolError(
                f"unknown rotation strategy {strategy!r}: "
                f"use one of {', '.join(self.STRATEGIES)}"
            )
        eligible = self._eligible_proxies(
            country=country, protocol=protocol,
            max_latency_ms=max_latency_ms, min_score=min_score,
            include_unchecked=include_unchecked,
        )
        if not eligible:
            raise ProxyPoolError(
                "no eligible proxies for "
                f"strategy={strategy} country={country} protocol={protocol} "
                "— run health_check() or loosen the filters"
            )
        if strategy == "round_robin":
            pick = eligible[self._rr % len(eligible)]
            self._rr += 1
        elif strategy == "random":
            pick = _random.choice(eligible)
        elif strategy == "least_latency":
            def _lat(c: Credential) -> float:
                ewma = ((c.metadata or {}).get("health") or {}).get(
                    "latency_ewma_ms")
                return float(ewma) if ewma is not None else float("inf")
            pick = min(eligible, key=_lat)
        else:  # p2c: two random candidates, higher score wins
            a, b = (_random.choice(eligible), _random.choice(eligible))
            pick = a if self._proxy_score(a) >= self._proxy_score(b) else b
        cred = self.vault.get(self._service, pick.username, mark_used=False)
        _log.info("proxypool: selected %s via %s", pick.username, strategy)
        details = self._details(cred, include_secret=True)
        details["selection_strategy"] = strategy
        return details

    def sticky_acquire(
        self, key: str, *, ttl: float = 600.0, **filters: Any
    ) -> dict[str, Any]:
        """Pin ``key`` to one proxy for ``ttl`` seconds (sticky session).

        Login flows, carts, multi-step scrapes — anywhere rotating the IP
        mid-session would look incoherent (identity discipline: the proxy
        rotates as part of the session, not independently of it). The pin
        survives as long as the proxy stays eligible; a dead or
        quarantined proxy transparently re-pins to a fresh one.
        """
        key = (key or "").strip()
        if not key:
            raise ProxyPoolError("sticky_acquire needs a non-empty key")
        now = time.time()
        pinned = self._sticky.get(key)
        if pinned:
            proxy_id, expires_at = pinned
            if expires_at > now:
                try:
                    cred = self.vault.get(self._service, proxy_id,
                                          mark_used=False)
                    health = dict((cred.metadata or {}).get("health") or {})
                    if (health.get("status") == HEALTHY
                            and not self._quarantined(health, now)):
                        details = self._details(cred, include_secret=True)
                        details["sticky"] = True
                        details["sticky_key"] = key
                        return details
                except Exception:  # noqa: BLE001 - re-pin on any problem
                    pass
            self._sticky.pop(key, None)
        details = self.select(**filters)
        self._sticky[key] = (details["id"], now + ttl)
        details["sticky"] = True
        details["sticky_key"] = key
        details["sticky_ttl"] = ttl
        return details

    def sticky_release(self, key: str) -> bool:
        """Drop the sticky pin for ``key``. Returns True when one existed."""
        return self._sticky.pop((key or "").strip(), None) is not None

    def quarantine(
        self, proxy_id: str, seconds: float = 300.0, reason: str = ""
    ) -> dict[str, Any]:
        """Park a proxy for ``seconds`` (ban/block cooldown).

        Quarantined proxies are skipped by :meth:`select` and
        :meth:`rotate` until the cooldown expires; health status is
        untouched, so a later successful check re-admits them. Use for
        BLOCK/CHALLENGE classifications — the proxy isn't dead, the target
        just doesn't want to see it right now.
        """
        pid = self._normalize_id(proxy_id)
        cred = self.vault.get(self._service, pid, mark_used=False)
        meta = dict(cred.metadata or {})
        health = dict(meta.get("health") or {}) or self._blank_health()
        health["quarantined_until"] = time.time() + max(1.0, seconds)
        health["quarantine_reason"] = reason or "manual quarantine"
        meta["health"] = health
        self.vault.store(
            service=self._service, username=pid, password=cred.password,
            credential_type=cred.credential_type,
            tags=list(cred.tags or []), metadata=meta,
        )
        _log.info("proxypool: quarantined %s for %.0fs (%s)", pid, seconds,
                  health["quarantine_reason"])
        return self._public_view(
            self.vault.get(self._service, pid, mark_used=False))

    def stats(self) -> dict[str, Any]:
        """Pool rollup: totals, health mix, latency, geo/protocol spread."""
        creds = self._load_all()
        now = time.time()
        healthy = unhealthy = unknown = quarantined = 0
        latencies: list[float] = []
        countries: dict[str, int] = {}
        protocols: dict[str, int] = {}
        scores: list[float] = []
        for cred in creds:
            meta = cred.metadata or {}
            health = dict(meta.get("health") or {})
            status = health.get("status", UNKNOWN)
            if self._quarantined(health, now):
                quarantined += 1
            if status == HEALTHY:
                healthy += 1
            elif status == UNHEALTHY:
                unhealthy += 1
            else:
                unknown += 1
            ewma = health.get("latency_ewma_ms")
            if ewma is not None:
                latencies.append(float(ewma))
            scores.append(float(health.get("score", 50.0) or 50.0))
            c = str(meta.get("country", "")).upper() or "??"
            countries[c] = countries.get(c, 0) + 1
            p = str(meta.get("protocol", "http"))
            protocols[p] = protocols.get(p, 0) + 1
        return {
            "total": len(creds),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "unchecked": unknown,
            "quarantined": quarantined,
            "avg_latency_ms": (round(sum(latencies) / len(latencies), 1)
                               if latencies else None),
            "avg_score": (round(sum(scores) / len(scores), 1)
                          if scores else None),
            "countries": dict(sorted(countries.items())),
            "protocols": dict(sorted(protocols.items())),
            "sticky_pins": len(self._sticky),
        }

    def record_failure(self, proxy_id: str, error: str = "") -> dict[str, Any]:
        """Demote a proxy that just failed a real request.

        Marks it unhealthy (``rotate()`` skips it) and increments its
        consecutive-failure count.  Used by the download fallback chain
        and other consumers so one dead proxy never kills the work —
        the caller simply moves to the next proxy.
        """
        self._record_health(proxy_id, ok=False, latency_ms=None,
                            status_code=None, error=error or "")
        _log.info("proxypool: demoted %s (%s)", proxy_id,
                  (error or "")[:120])
        return self.get_proxy(proxy_id)

    def record_success(self, proxy_id: str,
                       latency_ms: float | None = None) -> dict[str, Any]:
        """Mark a proxy healthy again after a successful real request.

        Resets its consecutive-failure count so a proxy that recovered
        rejoins rotation.
        """
        self._record_health(proxy_id, ok=True, latency_ms=latency_ms,
                            status_code=None, error="")
        return self.get_proxy(proxy_id)

    def healthy_proxies(self) -> list[dict[str, Any]]:
        """Connection details for every currently-healthy proxy.

        Same shape as :meth:`rotate` (includes ``proxy_url`` with auth —
        treat the result as secret), in insertion order, for consumers
        that need protocol-aware selection instead of blind round-robin.
        """
        out: list[dict[str, Any]] = []
        for cred in sorted(self._load_all(), key=lambda c: c.id):
            if ((cred.metadata or {}).get("health") or {}).get("status") \
                    == HEALTHY:
                full = self.vault.get(self._service, cred.username,
                                      mark_used=False)
                out.append(self._details(full, include_secret=True))
        return out

    def rotate(self) -> dict[str, Any]:
        """Next healthy proxy, round-robin. Includes the credential.

        Skips proxies whose last recorded check was not healthy and any
        under quarantine. Raises :class:`ProxyPoolError` when no healthy
        proxy exists — run :meth:`health_check` to refresh the pool first.
        For strategy-based picking (p2c, least-latency, geo) use
        :meth:`select`.
        """
        healthy = self._eligible_proxies()
        if not healthy:
            total = len(self._load_all())
            raise ProxyPoolError(
                "no healthy proxies to rotate to"
                + (f" ({total} stored)" if total else " (pool is empty)")
                + " — run health_check() to refresh the pool"
            )
        pick = healthy[self._rr % len(healthy)]
        self._rr += 1
        cred = self.vault.get(self._service, pick.username, mark_used=False)
        _log.info("proxypool: rotated to %s", pick.username)
        return self._details(cred, include_secret=True)

    # ── internals ──────────────────────────────────────────────

    def _load_all(self) -> list[Credential]:
        return self.vault.list_all(service=self._service)

    def _require_proxy(self, proxy_id: str) -> Credential:
        pid = self._normalize_id(proxy_id)
        try:
            return self.vault.get(self._service, pid, mark_used=False)
        except NotFound as exc:
            raise ProxyPoolError(f"unknown proxy {proxy_id!r}") from exc

    @staticmethod
    def _blank_health() -> dict[str, Any]:
        return {
            "status": UNKNOWN,
            "latency_ms": None,
            "latency_ewma_ms": None,
            "status_code": None,
            "last_checked": 0.0,
            "last_error": "",
            "consecutive_failures": 0,
            "checks": 0,
            "successes": 0,
            "score": 50.0,
            "quarantined_until": 0.0,
            "quarantine_reason": "",
        }

    @staticmethod
    def _normalize_host(host: str) -> str:
        h = (host or "").strip().lower().rstrip(".")
        if not h or any(ch.isspace() or ord(ch) < 32 for ch in h):
            raise ProxyPoolError(
                f"invalid proxy host {host!r}: must be a hostname or IP"
            )
        if ":" in h and not (h.startswith("[") and h.endswith("]")):
            h = f"[{h}]"  # IPv6 literal
        return h

    @staticmethod
    def _normalize_port(port: int | str) -> int:
        try:
            num = int(str(port).strip())
        except (TypeError, ValueError) as exc:
            raise ProxyPoolError(
                f"invalid proxy port {port!r}: must be an integer 1-65535"
            ) from exc
        if not 1 <= num <= 65535:
            raise ProxyPoolError(
                f"invalid proxy port {port!r}: must be an integer 1-65535"
            )
        return num

    @staticmethod
    def _normalize_tags(
        tags: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if tags is None:
            return []
        if not isinstance(tags, (list, tuple)):
            raise ProxyPoolError("tags must be a list of strings")
        clean: list[str] = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ProxyPoolError("tags must be a list of strings")
            tag = tag.strip()
            if tag and tag not in clean:
                clean.append(tag)
        return clean

    @classmethod
    def _normalize_id(cls, proxy_id: str) -> str:
        """Normalize a free-form proxy id to the stored ``protocol://host:port`` form."""
        pid = (proxy_id or "").strip().lower()
        if not pid:
            raise ProxyPoolError("proxy id must not be empty")
        if "://" not in pid:
            raise ProxyPoolError(
                f"invalid proxy id {proxy_id!r}: expected "
                "'protocol://host:port'"
            )
        protocol, rest = pid.split("://", 1)
        if ":" not in rest:
            raise ProxyPoolError(
                f"invalid proxy id {proxy_id!r}: expected "
                "'protocol://host:port'"
            )
        host_part, _, port_part = rest.rpartition(":")
        host = cls._normalize_host(host_part)
        port = cls._normalize_port(port_part)
        if protocol == "socks":
            protocol = "socks5"  # alias, same as add_proxy()
        if protocol not in cls.PROTOCOLS:
            raise ProxyPoolError(
                f"invalid proxy id {proxy_id!r}: unsupported protocol "
                f"{protocol!r}"
            )
        return f"{protocol}://{host}:{port}"

    def _proxy_url(self, cred: Credential, *, with_auth: bool) -> str:
        meta = cred.metadata or {}
        host = meta.get("host", "")
        port = meta.get("port", 0)
        protocol = meta.get("protocol", "http")
        auth = ""
        if with_auth:
            username, password = self._proxy_auth(cred)
            if username:
                user = urllib.parse.quote(username, safe="")
                pw = urllib.parse.quote(password, safe="")
                auth = f"{user}:{pw}@"
        return f"{protocol}://{auth}{host}:{port}"

    @staticmethod
    def _proxy_auth(cred: Credential) -> tuple[str, str]:
        """The proxy's (username, password) from the decrypted vault secret."""
        try:
            blob = json.loads(cred.password or "")
        except (ValueError, TypeError):
            blob = {}
        if isinstance(blob, dict):
            return str(blob.get("username") or ""), str(blob.get("password") or "")
        return "", ""

    def _public_view(
        self, cred: Credential, *, has_auth: bool | None = None
    ) -> dict[str, Any]:
        """Owner-safe view: health + endpoint, password masked, never leaked."""
        meta = cred.metadata or {}
        authed = (
            has_auth if has_auth is not None else bool(meta.get("proxy_username"))
        )
        return {
            "id": cred.username,
            "host": meta.get("host", ""),
            "port": meta.get("port", 0),
            "protocol": meta.get("protocol", "http"),
            "country": str(meta.get("country", "") or ""),
            "city": str(meta.get("city", "") or ""),
            "username": str(meta.get("proxy_username") or ""),
            "password": self.MASK if authed else "",
            "has_auth": authed,
            "tags": list(meta.get("tags") or []),
            "health": dict(meta.get("health") or {}) or self._blank_health(),
            "url": self._proxy_url(cred, with_auth=False),
            "added_at": cred.created_at,
            "updated_at": cred.updated_at,
        }

    def _details(self, cred: Credential, *, include_secret: bool) -> dict[str, Any]:
        """Connection details for other components.

        With ``include_secret=True`` the real password and a
        ``url_with_auth`` form are included — treat the result as secret.
        """
        view = self._public_view(cred)
        if include_secret:
            username, password = self._proxy_auth(cred)
            view["username"] = username
            view["password"] = password
            view["url_with_auth"] = self._proxy_url(cred, with_auth=True)
            view["proxy_url"] = view["url_with_auth"]  # alias: hand to HttpClient etc.
        return view
