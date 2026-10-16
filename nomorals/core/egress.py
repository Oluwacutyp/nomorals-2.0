"""Gated egress: the ONLY sanctioned way for sandboxed code to reach the network.

L1 (``nomorals.tools.shell.run_sandboxed``) is offline. L2
(``nomorals.tools.sandbox_l2``) is L1 + this module: every outbound
request goes through an :class:`EgressToken` bound to one TASK, itself
backed by a :class:`~nomorals.core.policy.TimedGrant`, and is checked
**after DNS resolution, on every redirect hop**:

* scheme must be ``http`` or ``https``; no credentials in the URL;
* the hostname must match the token's domain allowlist (glob);
* the path must match the token's path prefixes;
* the hostname must NOT be an IP literal — including disguised forms
  like ``2130706433`` (decimal), ``0x7f000001`` (hex) or ``127.1``,
  which ``inet_aton`` accepts as an IP;
* every IP the hostname resolves to must be a public address — no
  loopback, no RFC1918, no link-local (this is what blocks
  ``169.254.169.254`` cloud-metadata theft), no multicast, no
  reserved/unspecified;
* redirects are followed manually and re-checked hop by hop — a
  redirect to an off-allowlist host, a private IP, or an IP literal
  is refused BEFORE connecting;
* budgets: max requests, max bytes, max redirects, TTL — all enforced
  on the token, counted per hop.

Kill switches: :meth:`EgressController.kill_all` (global) and
:meth:`EgressController.kill_token` (per-token) revoke the token(s),
snapshot what was in flight, and close the live sockets of in-flight
requests — a stalled download dies immediately, not at the next timeout.

Cassette replay: with ``cassette_mode="replay"`` an
:class:`EgressClient` serves recorded responses from the generic
cassette store (:mod:`nomorals.core.cassette`) and performs ZERO
network I/O — a replay miss is a denial, never a live fetch. With
``"record"`` it records the response (headers minus credentials, body
base64) so a later replay is byte-identical. This is the same
mechanism LLM cassettes use: one record/replay system for both planes.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import http.client
import ipaddress
import socket
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cassette import GenericCassetteStore, canonical_key
from .errors import NoMoralsError
from .logging_setup import get_logger
from .policy import Policy, TimedGrant

_log = get_logger(__name__)

__all__ = [
    "EgressClient",
    "EgressController",
    "EgressDenied",
    "EgressResponse",
    "EgressToken",
    "check_host_allowed",
    "check_ip_allowed",
    "default_controller",
    "is_ip_literal",
    "network_cassette_key",
    "resolve_host",
]

#: Headers never recorded to a cassette and never forwarded with secrets.
_SENSITIVE_HEADERS = frozenset({
    "authorization", "cookie", "proxy-authorization", "x-api-key",
    "x-auth-token", "set-cookie",
})
#: Hop-by-hop headers are connection-scoped, never forwarded end-to-end.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-connection", "te", "trailer",
    "transfer-encoding", "upgrade", "proxy-authenticate",
})


class EgressDenied(NoMoralsError):
    """Egress refused: no token, off-allowlist, private IP, kill switch, budget."""

    code = "egress.denied"


# ── address checks ─────────────────────────────────────────────────────────

def is_ip_literal(host: str) -> bool:
    """True when ``host`` is an IP address — including disguised forms.

    ``ipaddress`` catches dotted quads and IPv6. ``inet_aton`` catches
    the forms attackers use to dodge a naive literal check: ``127.1``,
    ``2130706433``, ``0x7f000001``, ``0177.0.0.1``. A real hostname
    (letters, or a TLD) fails ``inet_aton``.
    """
    host = (host or "").strip()
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False


def resolve_host(host: str) -> list[str]:
    """All IPs ``host`` resolves to (IPv4 first). Fail CLOSED on DNS failure."""
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC,
                                   type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise EgressDenied(f"egress denied: DNS failure for {host!r}: {exc}") from exc
    except OSError as exc:
        raise EgressDenied(f"egress denied: DNS failure for {host!r}: {exc}") from exc
    ips: list[str] = []
    for _fam, _type, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if ip not in ips:
            ips.append(ip)
    # IPv4 first: v4-mapped/dual-stack hosts connect more reliably, and
    # tests bind their servers to 127.0.0.1.
    ips.sort(key=lambda ip: (":" in ip, ip))
    if not ips:
        raise EgressDenied(f"egress denied: {host!r} resolves to no address")
    return ips


def check_ip_allowed(ip: str) -> None:
    """Raise :class:`EgressDenied` unless ``ip`` is a public address."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError as exc:
        raise EgressDenied(f"egress denied: unparseable IP {ip!r}") from exc
    if addr.is_unspecified:
        raise EgressDenied(f"egress denied: unspecified address {ip}")
    if addr.is_loopback:
        raise EgressDenied(f"egress denied: loopback address {ip}")
    if addr.is_link_local:
        raise EgressDenied(
            f"egress denied: link-local address {ip} "
            "(cloud-metadata / link-local exfiltration)")
    if addr.is_multicast:
        raise EgressDenied(f"egress denied: multicast address {ip}")
    if addr.is_private or addr.is_reserved:
        raise EgressDenied(f"egress denied: private/reserved address {ip}")


def check_host_allowed(host: str, *, allow_private_ips: bool = False) -> list[str]:
    """Validate ``host`` and return its resolved IPs.

    Refuses IP literals outright, resolves the name, and — unless
    ``allow_private_ips`` (explicit opt-in for local test servers /
    internal tooling) — refuses any resolution to a non-public address.
    Resolution happens here, BEFORE any connection: this is the
    post-resolution check, and callers must connect to one of the
    returned IPs, never re-resolve (that would re-open DNS rebinding).
    """
    if not host:
        raise EgressDenied("egress denied: empty hostname")
    if is_ip_literal(host):
        raise EgressDenied(
            f"egress denied: IP literals are not allowed ({host!r}) — "
            "use a hostname on the token's allowlist")
    ips = resolve_host(host)
    if not allow_private_ips:
        for ip in ips:
            check_ip_allowed(ip)
    return ips


# ── token ──────────────────────────────────────────────────────────────────

class EgressToken:
    """A network capability bound to one TASK.

    Wraps a :class:`~nomorals.core.policy.TimedGrant` (``net.out``) so
    expiry/revocation ride the policy machinery, plus the L2 bounds the
    grant cannot express: domain allowlist, path prefixes, request /
    byte / redirect budgets.
    """

    def __init__(
        self,
        task_id: str,
        grant: TimedGrant,
        *,
        domains: tuple[str, ...] | list[str],
        path_prefixes: tuple[str, ...] | list[str] = ("/",),
        max_requests: int = 100,
        max_bytes: int = 10_000_000,
        max_redirects: int = 5,
        timeout: float = 30.0,
        allow_private_ips: bool = False,
    ) -> None:
        if not domains:
            raise ValueError("EgressToken needs at least one allowed domain")
        self.task_id = task_id
        self.grant = grant
        self.token_id = grant.grant_id
        self.domains = tuple(d.lower() for d in domains)
        self.path_prefixes = tuple(path_prefixes)
        self.max_requests = max(0, int(max_requests))
        self.max_bytes = max(0, int(max_bytes))
        self.max_redirects = max(0, int(max_redirects))
        self.timeout = max(0.1, float(timeout))
        self.allow_private_ips = bool(allow_private_ips)
        # RLock: check_url / consume_bytes hold the lock while raising
        # via _deny, which re-enters to bump the denied counter — a plain
        # Lock would deadlock there.
        self._lock = threading.RLock()
        self._requests_used = 0
        self._bytes_used = 0
        self._denied = 0

    @classmethod
    def mint(
        cls,
        task_id: str,
        *,
        domains: tuple[str, ...] | list[str],
        ttl_s: float = 600.0,
        path_prefixes: tuple[str, ...] | list[str] = ("/",),
        max_requests: int = 100,
        max_bytes: int = 10_000_000,
        max_redirects: int = 5,
        timeout: float = 30.0,
        allow_private_ips: bool = False,
        policy: Policy | None = None,
    ) -> "EgressToken":
        """Mint a token for ``task_id`` with a fresh :class:`TimedGrant`."""
        grant = (policy or Policy()).issue_grant(
            "net.out", ttl_s=ttl_s, actor=task_id,
            note=f"egress for task {task_id}")
        return cls(task_id, grant, domains=tuple(domains),
                   path_prefixes=tuple(path_prefixes),
                   max_requests=max_requests, max_bytes=max_bytes,
                   max_redirects=max_redirects, timeout=timeout,
                   allow_private_ips=allow_private_ips)

    # ── state ──
    @property
    def expired(self) -> bool:
        return self.grant.expired

    @property
    def revoked(self) -> bool:
        return self.grant.revoked

    @property
    def remaining_s(self) -> float:
        return self.grant.remaining_s()

    def revoke(self) -> None:
        self.grant.revoke()

    @property
    def requests_used(self) -> int:
        with self._lock:
            return self._requests_used

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used

    @property
    def denied(self) -> int:
        with self._lock:
            return self._denied

    def _note_denied(self) -> None:
        with self._lock:
            self._denied += 1

    def usage(self) -> dict[str, Any]:
        with self._lock:
            return {
                "token_id": self.token_id,
                "task_id": self.task_id,
                "requests_used": self._requests_used,
                "requests_max": self.max_requests,
                "bytes_used": self._bytes_used,
                "bytes_max": self.max_bytes,
                "denied": self._denied,
                "expired": self.grant.expired,
                "revoked": self.grant.revoked,
                "remaining_s": round(self.grant.remaining_s(), 3),
            }

    # ── checks ──
    def _deny(self, reason: str) -> EgressDenied:
        self._note_denied()
        return EgressDenied(f"egress denied for task {self.task_id!r}: {reason}")

    def check_url(self, url: str) -> urllib.parse.ParseResult:
        """Validate one URL (one hop). Raises :class:`EgressDenied`.

        Order matters: grant state → scheme/credentials → domain →
        path → budget. The DNS/IP check is NOT here — it happens in
        :func:`check_host_allowed` at connect time, per hop, so a
        redirect's fresh resolution is checked against the CURRENT
        answer, not a cached one.
        """
        try:
            if not self.grant.grants("net.out"):
                raise self._deny("token expired or revoked")
            try:
                parsed = urllib.parse.urlparse(url)
            except ValueError as exc:
                raise self._deny(f"unparseable URL {url!r}: {exc}") from exc
            if parsed.scheme not in ("http", "https"):
                raise self._deny(
                    f"scheme {parsed.scheme!r} not allowed (http/https only)")
            if parsed.username or parsed.password:
                raise self._deny("credentials in URL are not allowed")
            host = (parsed.hostname or "").lower()
            if not host:
                raise self._deny(f"empty hostname in {url!r}")
            if is_ip_literal(host):
                raise self._deny(
                    f"IP literals are not allowed ({host!r}) — use a "
                    "hostname on the token's allowlist")
            if not any(fnmatch.fnmatchcase(host, pat) for pat in self.domains):
                raise self._deny(
                    f"host {host!r} not on allowlist {list(self.domains)}")
            path = parsed.path or "/"
            if not any(path.startswith(prefix) for prefix in self.path_prefixes):
                raise self._deny(
                    f"path {path!r} not under allowed prefixes "
                    f"{list(self.path_prefixes)}")
            with self._lock:
                if self._requests_used >= self.max_requests:
                    raise self._deny(
                        f"request budget exhausted "
                        f"({self._requests_used}/{self.max_requests})")
            return parsed
        except EgressDenied:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed on any surprise
            raise self._deny(f"validation failed: {exc}") from exc

    def consume_request(self) -> None:
        with self._lock:
            self._requests_used += 1

    def consume_bytes(self, n: int) -> None:
        with self._lock:
            self._bytes_used += max(0, int(n))
            if self._bytes_used > self.max_bytes:
                raise self._deny(
                    f"byte budget exceeded "
                    f"({self._bytes_used}/{self.max_bytes})")


# ── controller / kill switch ───────────────────────────────────────────────

class EgressController:
    """Global kill switch + per-token revocation + in-flight accounting.

    One controller per process (see :func:`default_controller`); tests
    build their own. ``kill_all`` / ``kill_token`` revoke tokens,
    snapshot what was in flight at kill time, and close the live
    sockets of in-flight requests so a stalled download dies now.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tokens: dict[str, EgressToken] = {}
        self._killed = False
        self._kill_reason = ""
        self._in_flight: dict[str, dict[str, Any]] = {}
        self._killed_in_flight: list[dict[str, Any]] = []
        self._sockets: set[socket.socket] = set()
        self._flight_seq = 0

    # ── kill switch ──
    @property
    def killed(self) -> bool:
        with self._lock:
            return self._killed

    @property
    def kill_reason(self) -> str:
        with self._lock:
            return self._kill_reason

    def register(self, token: EgressToken) -> None:
        with self._lock:
            self._tokens[token.token_id] = token

    def check(self, token: EgressToken) -> None:
        """Raise if the global kill switch fired or the token is dead."""
        with self._lock:
            killed, reason = self._killed, self._kill_reason
        if killed:
            raise EgressDenied(
                f"egress denied: global kill switch fired ({reason})")
        if not token.grant.grants("net.out"):
            raise EgressDenied(
                f"egress denied for task {token.task_id!r}: "
                "token expired or revoked")

    def kill_token(self, token: EgressToken, reason: str = "") -> dict[str, Any]:
        """Revoke one token; log what it had in flight. Returns the report."""
        token.revoke()
        with self._lock:
            flights = [f for f in self._in_flight.values()
                       if f["token_id"] == token.token_id]
            report = {
                "scope": "token",
                "token_id": token.token_id,
                "task_id": token.task_id,
                "reason": reason or "per-token kill switch",
                "in_flight": flights,
                "killed_at": time.time(),
            }
            self._killed_in_flight.append(report)
        _log.warning("egress kill_token %s: %d in flight",
                     token.token_id, len(flights))
        return report

    def kill_all(self, reason: str = "") -> dict[str, Any]:
        """Global kill switch: revoke everything, snapshot + close in-flight."""
        with self._lock:
            if self._killed:
                return {"scope": "global", "already_killed": True,
                        "reason": self._kill_reason,
                        "in_flight": list(self._killed_in_flight)}
            self._killed = True
            self._kill_reason = reason or "global kill switch"
            flights = list(self._in_flight.values())
            for token in self._tokens.values():
                token.revoke()
            sockets = list(self._sockets)
            report = {
                "scope": "global",
                "reason": self._kill_reason,
                "in_flight": flights,
                "killed_at": time.time(),
            }
            self._killed_in_flight.append(report)
        # Close live sockets OUTSIDE the lock: a blocked recv raises now.
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        _log.warning("egress kill_all: %d in flight, %d sockets closed",
                     len(flights), len(sockets))
        return report

    def reset(self) -> None:
        """Clear the global kill flag. Tests only — production never resets."""
        with self._lock:
            self._killed = False
            self._kill_reason = ""
            self._in_flight.clear()
            self._sockets.clear()

    def killed_in_flight(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._killed_in_flight)

    # ── in-flight accounting ──
    def begin(self, token: EgressToken, method: str, url: str) -> str:
        with self._lock:
            self._flight_seq += 1
            flight_id = f"flt_{self._flight_seq}"
            self._in_flight[flight_id] = {
                "flight_id": flight_id,
                "token_id": token.token_id,
                "task_id": token.task_id,
                "method": method,
                "url": url,
                "started_at": time.time(),
            }
            return flight_id

    def end(self, flight_id: str) -> None:
        with self._lock:
            self._in_flight.pop(flight_id, None)

    def in_flight_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._in_flight.values())

    # ── socket tracking (kill closes what is blocked in recv) ──
    def track_socket(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.add(sock)

    def untrack_socket(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.discard(sock)


_default_controller: EgressController | None = None
_default_lock = threading.Lock()


def default_controller() -> EgressController:
    """The process-wide controller."""
    global _default_controller
    with _default_lock:
        if _default_controller is None:
            _default_controller = EgressController()
        return _default_controller


# ── cassette keys ──────────────────────────────────────────────────────────

def network_cassette_key(method: str, url: str, body: bytes | None) -> str:
    """Key for one network exchange: method + URL + sha256(body).

    The body is hashed, never stored in the key — a POST body may carry
    secrets, and the key must not leak them.
    """
    digest = hashlib.sha256(body or b"").hexdigest()
    return canonical_key("net", {"method": method.upper(), "url": url,
                                 "body_sha256": digest})


def _sanitised_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if key.lower() in _SENSITIVE_HEADERS:
            continue
        out[key.lower()] = value
    return out


# ── response ───────────────────────────────────────────────────────────────

@dataclass
class EgressResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    url: str = ""
    hops: list[str] = field(default_factory=list)
    replayed: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        import json as _json

        return _json.loads(self.text)


# ── client ─────────────────────────────────────────────────────────────────

class _CheckedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to a CHECKED IP, with SNI + cert verification for the hostname.

    The caller resolved + validated the hostname already; this connection
    dials the checked IP (no second resolution — no DNS-rebinding window)
    while TLS still authenticates the original hostname.
    """

    def __init__(self, host: str, checked_ip: str, port: int,
                 timeout: float, context: ssl.SSLContext) -> None:
        super().__init__(host, port, timeout=timeout, context=context)
        self._checked_ip = checked_ip

    def connect(self) -> None:
        sock = socket.create_connection((self._checked_ip, self.port),
                                        timeout=self.timeout)
        try:
            self.sock = self.context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


class EgressClient:
    """Gated HTTP client: token + post-resolution checks + manual redirects + cassettes."""

    def __init__(
        self,
        *,
        controller: EgressController | None = None,
        cassette: GenericCassetteStore | None = None,
        cassette_mode: str = "record",
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if cassette_mode not in ("record", "replay", "off"):
            raise ValueError(f"cassette_mode must be record|replay|off, got {cassette_mode!r}")
        self.controller = controller or default_controller()
        self.cassette = cassette
        self.cassette_mode = cassette_mode
        self.ssl_context = ssl_context or ssl.create_default_context()

    # ── public ──
    def get(self, token: EgressToken, url: str, *,
            headers: dict[str, str] | None = None,
            timeout: float | None = None) -> EgressResponse:
        return self.fetch(token, "GET", url, headers=headers, timeout=timeout)

    def fetch(
        self,
        token: EgressToken,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> EgressResponse:
        """One gated exchange. Redirects are followed manually, re-checked per hop."""
        self.controller.register(token)
        self.controller.check(token)
        method = method.upper()
        key = network_cassette_key(method, url, data)

        # Replay: serve from cassette, ZERO network I/O. A miss is a
        # denial — replay mode never falls through to live traffic.
        if self.cassette_mode == "replay" and self.cassette is not None:
            record = self.cassette.lookup(key)
            if record is None:
                raise EgressDenied(
                    f"egress denied: cassette replay miss for {method} {url} "
                    "(replay mode never goes live)")
            return _response_from_record(record)

        flight_id = self.controller.begin(token, method, url)
        try:
            return self._fetch_live(token, method, url, headers, data,
                                    timeout or token.timeout, key)
        finally:
            self.controller.end(flight_id)

    # ── live path ──
    def _fetch_live(
        self,
        token: EgressToken,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        data: bytes | None,
        timeout: float,
        key: str,
    ) -> EgressResponse:
        current = url
        hops: list[str] = []
        sent_headers = _sanitised_headers(headers)
        for hop in range(token.max_redirects + 1):
            # Kill switch is consulted EVERY hop, not just at entry.
            self.controller.check(token)
            parsed = token.check_url(current)
            hops.append(current)
            ips = check_host_allowed(
                parsed.hostname or "", allow_private_ips=token.allow_private_ips)
            token.consume_request()
            status, resp_headers, body, location = self._one_hop(
                token, method, parsed, ips, sent_headers, data, timeout)
            if status in (301, 302, 303, 307, 308) and location:
                # Redirect bodies count against the byte budget too.
                token.consume_bytes(len(body))
                if hop >= token.max_redirects:
                    raise token._deny(
                        f"too many redirects (>{token.max_redirects})")
                # 303: the redirect is always a GET with no body.
                if status == 303:
                    method, data = "GET", None
                current = urllib.parse.urljoin(current, location)
                continue
            # Final response: account bytes, record cassette, return.
            token.consume_bytes(len(body))
            response = EgressResponse(
                status=status, headers=resp_headers, body=body,
                url=current, hops=hops, replayed=False)
            if self.cassette_mode == "record" and self.cassette is not None:
                self._record(key, method, url, sent_headers, response)
            return response
        raise token._deny(f"too many redirects (>{token.max_redirects})")

    def _one_hop(
        self,
        token: EgressToken,
        method: str,
        parsed: urllib.parse.ParseResult,
        ips: list[str],
        headers: dict[str, str],
        data: bytes | None,
        timeout: float,
    ) -> tuple[int, dict[str, str], bytes, str]:
        """One HTTP exchange to the first connectable CHECKED IP."""
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        last_error: Exception | None = None
        for ip in ips:
            conn: http.client.HTTPConnection
            try:
                if parsed.scheme == "https":
                    conn = _CheckedHTTPSConnection(host, ip, port, timeout,
                                                   self.ssl_context)
                else:
                    conn = http.client.HTTPConnection(ip, port, timeout=timeout)
                conn.connect()
            except OSError as exc:
                last_error = exc
                continue
            sock = getattr(conn, "sock", None)
            if sock is not None:
                self.controller.track_socket(sock)
            try:
                conn.putrequest(method, target, skip_host=True, skip_accept_encoding=True)
                conn.putheader("Host", host if port in (80, 443) else f"{host}:{port}")
                conn.putheader("User-Agent", "NoMoralsEgress/1.0")
                conn.putheader("Accept", "*/*")
                conn.putheader("Connection", "close")
                for key, value in headers.items():
                    if key.lower() in _HOP_BY_HOP or key.lower() == "host":
                        continue
                    conn.putheader(key, value)
                if data:
                    conn.putheader("Content-Length", str(len(data)))
                conn.endheaders(data)
                resp = conn.getresponse()
                status = resp.status
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                location = resp_headers.get("location", "")
                # Read the body in chunks: the kill switch closes the
                # socket mid-read, and the byte budget is enforced as
                # bytes arrive — a 10 GB response never lands in memory.
                chunks: list[bytes] = []
                total = 0
                while True:
                    try:
                        chunk = resp.read(65536)
                    except OSError as exc:
                        raise _killed_or_error(self.controller, token, exc) from exc
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > token.max_bytes:
                        raise token._deny(
                            f"byte budget exceeded during read "
                            f"({total}/{token.max_bytes})")
                    chunks.append(chunk)
                return status, resp_headers, b"".join(chunks), location
            except EgressDenied:
                raise
            except OSError as exc:
                raise _killed_or_error(self.controller, token, exc) from exc
            except (http.client.HTTPException, ssl.SSLError, ValueError) as exc:
                raise token._deny(f"request failed: {exc}") from exc
            finally:
                if sock is not None:
                    self.controller.untrack_socket(sock)
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - close must not mask the result
                    pass
        raise token._deny(
            f"could not connect to {host} "
            f"({last_error or 'no resolved address'})")

    def _record(self, key: str, method: str, url: str,
                headers: dict[str, str], response: EgressResponse) -> None:
        assert self.cassette is not None
        try:
            self.cassette.record(key, {
                "key": key,
                "kind": "net",
                "method": method,
                "url": url,
                "request_headers": dict(headers),
                "response": {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body_b64": base64.b64encode(response.body).decode("ascii"),
                    "url": response.url,
                    "hops": list(response.hops),
                },
            })
        except Exception:  # noqa: BLE001 - cassettes never break the live path
            _log.debug("egress cassette record failed", exc_info=True)


def _response_from_record(record: dict[str, Any]) -> EgressResponse:
    data = record["response"]
    return EgressResponse(
        status=int(data["status"]),
        headers=dict(data.get("headers") or {}),
        body=base64.b64decode(data.get("body_b64", "")),
        url=data.get("url", ""),
        hops=list(data.get("hops") or []),
        replayed=True,
    )


def _killed_or_error(controller: EgressController, token: EgressToken,
                     exc: OSError) -> EgressDenied:
    """Map a socket failure: kill-switch aborts say so explicitly."""
    if controller.killed or token.revoked or token.expired:
        return EgressDenied(
            f"egress aborted: kill switch fired during request ({exc})")
    return EgressDenied(f"egress denied: network error: {exc}")
