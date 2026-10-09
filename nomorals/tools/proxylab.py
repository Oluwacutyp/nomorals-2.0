"""God-tier proxy lab: scrape, test, rank, store, and serve working proxies.

This is the DISCOVERY half of the proxy stack (``tools/proxy.py`` is the
ROUTING half: it applies one proxy to all outbound traffic).  The lab:

1. **Scraps free public proxies** from several independent sources
   (plain ``host:port`` lists, ``scheme://host:port`` lists, and HTML
   proxy-list pages) — concurrent, source failures are isolated.
2. **Tests every candidate for real**, over raw sockets (no third-party
   dependencies):
     - liveness + latency, per protocol (HTTP / HTTPS / SOCKS4 / SOCKS5 —
       the SOCKS client handshakes are implemented here, stdlib only)
     - the **egress IP** — proof of where traffic actually goes
     - **anonymity class**: transparent / anonymous / elite (does the
       site see our real IP? does it see proxy headers?)
     - **country** of the egress (optional, capped per run)
3. **Ranks** (alive → anonymity → speed), **dedupes**, **drops dead**
   proxies, and **persists** the working pool (``working.txt`` +
   ``working.json``) plus the raw candidate set for cheap re-checks.
4. **Serves fresh proxies on demand** — the main AI / any agent calls
   ``proxy_pool`` and gets ranked, TTL-checked URLs it can hand to
   ``proxy_set`` (routing), the browser, web tools, or OSINT.
5. **Scheduled re-checks** through the normal scheduler: a
   ``proxy_refresh`` job re-tests the stored candidates every N minutes,
   so the pool never silently rots.

Everything network-touching takes an injectable transport, which is how
the test suite verifies the whole pipeline hermetically against local
fake proxies (a real HTTP forward proxy, a real SOCKS5 server, a real
SOCKS4 server — in-process).
"""
from __future__ import annotations

import base64
import concurrent.futures
import ipaddress
import json
import os
import re
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = [
    "Proxy",
    "ProxyScraper",
    "ProxyTester",
    "ProxyStore",
    "ProxyLab",
    "ProxyRotationManager",
    "DomainAffinityManager",
    "default_sources",
    "rank_proxies",
    "score_proxy",
    "empty_pool_guidance",
    "register",
    "proxy_discover",
    "proxy_sources",
]

SCHEMES = ("http", "https", "socks4", "socks5")

#: Built-in sources (wave 83): the health-tracked catalog from
#: :mod:`nomorals.tools.proxysources` — kept as DEFAULT_SOURCES for
#: compatibility; per-source failures retire entries for 24h and
#: proxy_discovered sources are added on top at runtime.
from .proxysources import BUILT_IN_SOURCES as _BUILTIN_CATALOG

DEFAULT_SOURCES: tuple[tuple[str, str, str], ...] = _BUILTIN_CATALOG

_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 "
       "Firefox/128.0")

#: per-profile concurrency (profile-gating, not designing down: the
#: lab does the same work everywhere, just with fewer parallel
#: connections on small devices)
_PROFILE_WORKERS = {
    "workstation": {"scrape": 8, "test": 12},
    "laptop": {"scrape": 6, "test": 8},
    "termux": {"scrape": 3, "test": 4},
}


def detect_profile(context: Any = None) -> str:
    """workstation | laptop | termux — settings pin first, then the
    Termux $PREFIX heuristic (mirrors agent_loop)."""
    try:
        settings = getattr(context, "settings", None) if context else None
        prof = (getattr(settings, "profile", "") or "").strip().lower()
        if prof in _PROFILE_WORKERS:
            return prof
        ctx_prof = (getattr(context, "resource_profile", "") or ""
                    ).strip().lower()
        if ctx_prof in _PROFILE_WORKERS:
            return ctx_prof
    except Exception:  # noqa: BLE001
        pass
    try:
        if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
            return "termux"
    except Exception:  # noqa: BLE001
        pass
    return "workstation"


def profile_workers(profile: str, kind: str) -> int:
    """Worker count for ``kind`` ("scrape" | "test") on ``profile``."""
    return _PROFILE_WORKERS.get((profile or "").lower(),
                                _PROFILE_WORKERS["workstation"])[kind]

_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
#: the ip:port token anywhere in a line — plain lists, decorated lists
#: (country flag, latency, ISP annotations), and html cells alike
_IPPORT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}):(\d{1,5})")
_TYPE_RE = re.compile(r"\b(HTTPS|HTTP|SOCKS5|SOCKS4)\b", re.IGNORECASE)


# ── model ────────────────────────────────────────────────────────────────────


@dataclass
class Proxy:
    """One proxy endpoint plus everything the tester has proven about it."""

    host: str
    port: int
    scheme: str = "http"          # http | https | socks4 | socks5
    country: str = ""
    egress_ip: str = ""
    latency_ms: int = 0
    anonymity: str = ""           # transparent | anonymous | elite | ""
    claimed_anonymity: str = ""   # the SOURCE's label (never overwrites
    # a lab test) — e.g. proxydb's "High Anonymous" → elite
    alive: bool = False
    tested_at: float = field(default_factory=time.time)
    error: str = ""
    # rolling latency samples (ts, ms) — recent ones weigh more in the
    # decayed average; capped so a long-lived record stays small
    latency_samples: list[tuple[float, float]] = field(default_factory=list)
    consecutive_failures: int = 0
    backoff_until: float = 0.0    # unix ts: excluded from routing until then
    last_success: float = 0.0

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.host, int(self.port), self.scheme)

    @property
    def age_hours(self) -> float:
        return max(0.0, (time.time() - self.tested_at) / 3600.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "host": self.host, "port": int(self.port),
            "scheme": self.scheme, "country": self.country,
            "egress_ip": self.egress_ip, "latency_ms": int(self.latency_ms),
            "anonymity": self.anonymity,
            "claimed_anonymity": self.claimed_anonymity,
            "alive": bool(self.alive),
            "tested_at": float(self.tested_at), "error": self.error,
            "latency_samples": [[float(t), float(ms)]
                                for t, ms in self.latency_samples[-12:]],
            "consecutive_failures": int(self.consecutive_failures),
            "backoff_until": float(self.backoff_until),
            "last_success": float(self.last_success),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Proxy":
        samples = data.get("latency_samples") or []
        clean: list[tuple[float, float]] = []
        for s in samples:
            try:
                clean.append((float(s[0]), float(s[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return cls(
            host=str(data.get("host") or ""),
            port=int(data.get("port") or 0),
            scheme=str(data.get("scheme") or "http"),
            country=str(data.get("country") or ""),
            egress_ip=str(data.get("egress_ip") or ""),
            latency_ms=int(data.get("latency_ms") or 0),
            anonymity=str(data.get("anonymity") or ""),
            claimed_anonymity=str(data.get("claimed_anonymity") or ""),
            alive=bool(data.get("alive")),
            tested_at=float(data.get("tested_at") or time.time()),
            error=str(data.get("error") or ""),
            latency_samples=clean[-12:],
            consecutive_failures=int(data.get("consecutive_failures") or 0),
            backoff_until=float(data.get("backoff_until") or 0.0),
            last_success=float(data.get("last_success") or 0.0),
        )

    # -- latency with time decay -------------------------------------------
    def record_latency(self, ms: float,
                       at: float | None = None) -> None:
        """One more measurement; old samples age out of influence."""
        self.latency_samples.append((at if at is not None else time.time(),
                                     max(0.0, float(ms))))
        del self.latency_samples[:-12]
        self.latency_ms = round(self.decayed_latency())

    def decayed_latency(self, at: float | None = None,
                        half_life_hours: float = 6.0) -> float:
        """Exponentially-decayed latency average: recent measurements
        weigh more.  Empty history falls back to the last point value."""
        now = at if at is not None else time.time()
        half = max(0.25, float(half_life_hours)) * 3600.0
        total = weight = 0.0
        for ts, ms in self.latency_samples:
            age = max(0.0, now - float(ts))
            w = 0.5 ** (age / half)
            total += float(ms) * w
            weight += w
        if weight > 0:
            return total / weight
        return float(self.latency_ms or 0)

    @property
    def in_backoff(self) -> bool:
        return float(self.backoff_until or 0.0) > time.time()

    def record_success(self, ms: float,
                       at: float | None = None) -> None:
        """A request through this proxy worked: sample the latency,
        clear the failure streak, drop any backoff."""
        now = at if at is not None else time.time()
        self.record_latency(ms, at=now)
        self.consecutive_failures = 0
        self.backoff_until = 0.0
        self.last_success = now
        self.alive = True
        self.error = ""

    def record_failure(self, reason: str = "",
                       backoff_seconds: float = 300.0,
                       at: float | None = None) -> float:
        """A request through this proxy failed at the connection level:
        the failure streak grows, the proxy is demoted with an
        exponential backoff (base * 2^streak, capped at 1h), and the
        reason is kept — nothing is dropped silently.  ``alive`` is
        left alone: a routing failure is transient (retry after the
        backoff), only the lab's own test marks a proxy dead.
        Returns the unix timestamp until which the proxy stays out of
        routing."""
        now = at if at is not None else time.time()
        self.consecutive_failures = int(self.consecutive_failures) + 1
        backoff = min(float(backoff_seconds)
                      * (2 ** (self.consecutive_failures - 1)), 3600.0)
        self.backoff_until = now + backoff
        if reason:
            self.error = reason[:200]
        return self.backoff_until


# ── transport (raw-socket HTTP + SOCKS4/5 — stdlib only) ────────────────────


def _http_exchange(sock: socket.socket, request_line: str,
                   headers: dict[str, str] | None = None,
                   timeout: float = 6.0) -> tuple[int, dict[str, str], str]:
    """Send one raw HTTP/1.1 request, read the full response (the request
    always carries Connection: close, so EOF-delimited is exact)."""
    sock.settimeout(timeout)
    head = request_line + "\r\n"
    for k, v in (headers or {}).items():
        head += f"{k}: {v}\r\n"
    head += "Connection: close\r\n\r\n"
    sock.sendall(head.encode("latin-1", "replace"))
    buf = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > 4 * 1024 * 1024:
            break
    raw = bytes(buf).decode("utf-8", "replace")
    split = raw.split("\r\n\r\n", 1)
    head_part = split[0]
    body = split[1] if len(split) > 1 else ""
    lines = head_part.split("\r\n")
    status = 0
    resp_headers: dict[str, str] = {}
    if lines:
        parts = lines[0].split(" ", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            status = int(parts[1])
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            resp_headers[k.strip().lower()] = v.strip()
    return status, resp_headers, body


def _connect(host: str, port: int, timeout: float) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:  # noqa: E103 - socket tuning is best-effort
        pass
    return sock


def socks5_connect(proxy_host: str, proxy_port: int, target_host: str,
                   target_port: int, timeout: float = 6.0) -> socket.socket:
    """SOCKS5 handshake (no-auth) → tunneled socket to the target.

    Implements the RFC 1928 minimum: greeting (methods: no-auth), then a
    domain-name connect request.  Raises on any non-success reply.
    """
    sock = _connect(proxy_host, proxy_port, timeout)
    try:
        sock.sendall(b"\x05\x01\x00")  # VER=5, 1 method, NO AUTH
        reply = _recv_exact(sock, 2, timeout)
        if reply[:1] != b"\x05" or reply[1] != 0x00:
            raise ConnectionError(f"socks5 auth rejected: {reply.hex()}")
        domain = target_host.encode("idna")
        if len(domain) > 255:
            raise ConnectionError("socks5 target name too long")
        request = (b"\x05\x01\x00\x03" + bytes([len(domain)]) + domain
                   + struct.pack("!H", target_port))
        sock.sendall(request)
        head = _recv_exact(sock, 4, timeout)
        if head[:1] != b"\x05" or head[1] != 0x00:
            raise ConnectionError(f"socks5 connect failed: {head.hex()}")
        atyp = head[3]
        if atyp == 0x01:
            _recv_exact(sock, 4 + 2, timeout)
        elif atyp == 0x03:
            ln = _recv_exact(sock, 1, timeout)[0]
            _recv_exact(sock, ln + 2, timeout)
        elif atyp == 0x04:
            _recv_exact(sock, 16 + 2, timeout)
        else:
            raise ConnectionError(f"socks5 unknown reply atyp {atyp}")
        return sock
    except Exception:
        sock.close()
        raise


def socks4_connect(proxy_host: str, proxy_port: int, target_ip: str,
                   target_port: int, userid: str = "",
                   timeout: float = 6.0) -> socket.socket:
    """SOCKS4 handshake (RFC 1928 appendix) → tunneled socket.

    SOCKS4 has no domain type, so the target must be an IPv4 address.
    """
    packed = socket.inet_aton(target_ip)
    sock = _connect(proxy_host, proxy_port, timeout)
    try:
        user = (userid or "").encode("latin-1", "replace")[:255]
        # SOCKS4: VN=0x04, CD=0x01 (CONNECT), DSTPORT, DSTIP, USERID, NULL
        request = (b"\x04\x01" + struct.pack("!H", target_port)
                   + packed + user + b"\x00")
        sock.sendall(request)
        reply = _recv_exact(sock, 8, timeout)
        # SOCKS4 reply: VN=0x00, CD=0x5A (granted)
        if reply[0] != 0x00 or reply[1] != 0x5A:
            raise ConnectionError(f"socks4 connect failed: {reply.hex()}")
        return sock
    except Exception:
        sock.close()
        raise


def _recv_exact(sock: socket.socket, n: int, timeout: float) -> bytes:
    buf = bytearray()
    deadline = time.monotonic() + timeout
    while len(buf) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("socks read timed out")
        sock.settimeout(remaining)
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socks connection closed mid-handshake")
        buf.extend(chunk)
    return bytes(buf)


def _valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    # the unspecified address is never a real proxy (some lists pad
    # with 0.0.0.0 rows) — drop it at parse time, not test time
    return value not in ("0.0.0.0", "::")


# ── scraper ──────────────────────────────────────────────────────────────────


def default_sources() -> list[tuple[str, str, str]]:
    return list(DEFAULT_SOURCES)


class ProxyScraper:
    """Concurrent multi-source scraper with per-source failure isolation."""

    def __init__(self, *, fetcher: Callable[[str], bytes] | None = None,
                 timeout: float = 10.0, user_agent: str = _UA,
                 sources: list[tuple[str, str, str]] | None = None,
                 max_workers: int = 0, profile: str = "") -> None:
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self.sources = list(sources or DEFAULT_SOURCES)
        # explicit max_workers wins; otherwise the runtime profile picks
        # (termux = fewer parallel connections, same work)
        self.max_workers = max(1, int(max_workers)) if max_workers else \
            profile_workers(profile or detect_profile(), "scrape")
        self.profile = profile or detect_profile()
        self._fetcher = fetcher or self._http_get

    # -- transport ----------------------------------------------------------
    def _http_get(self, url: str) -> bytes:
        import urllib.request

        request = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.8",
        })
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            return resp.read()

    # -- parsing ------------------------------------------------------------
    @staticmethod
    def parse_list(text: str, scheme: str) -> list[Proxy]:
        """Plain ``host:port`` lines — with decorations.

        Handles every flavor that appears in the wild: bare
        ``ip:port``, ``host port``, annotated lines (country flag,
        latency, ISP in brackets — roosterkid's format), commented
        lines, and header/banners that must be skipped.  The first
        valid ``ip:port`` token on each line wins; a bare hostname
        line (``host port``) still works for non-IP sources.
        """
        out: list[Proxy] = []
        for line in text.splitlines():
            line = line.split("#", 1)[0]
            stripped = line.strip()
            if not stripped or len(stripped) > 200:
                continue
            match = _IPPORT_RE.search(stripped)
            if match:
                host = match.group(1)
                port = int(match.group(2))
                if 0 < port < 65536 and _valid_ip(host):
                    out.append(Proxy(host=host, port=port, scheme=scheme))
                continue
            # not an ip:port line — try a bare "host port" pair
            low = stripped.lower()
            if low.startswith(("http", "scheme", "fromat", "format",
                               "website", "support", "btc", "eth ",
                               "ltc", "doge", "socks", "from", "note",
                               "updated", "list")):
                continue
            if ":" in stripped or stripped.count(" ") != 1:
                continue
            host, _, port_s = stripped.rpartition(" ")
            if not port_s.isdigit():
                continue
            port = int(port_s)
            if not (0 < port < 65536) or not _valid_hostname(host.strip()):
                continue
            out.append(Proxy(host=host.strip(), port=port, scheme=scheme))
        return out

    @staticmethod
    def parse_protocol(text: str) -> list[Proxy]:
        """One ``scheme://host:port`` per line."""
        out: list[Proxy] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.lower().startswith(("http", "socks")):
                continue
            parsed = urlparse(line)
            if parsed.scheme not in SCHEMES or not parsed.hostname \
                    or not parsed.port:
                continue
            if not _valid_ip(parsed.hostname) and not _valid_hostname(parsed.hostname):
                continue
            out.append(Proxy(host=parsed.hostname, port=parsed.port,
                             scheme=parsed.scheme))
        return out

    @staticmethod
    def parse_json(text: str) -> list[Proxy]:
        """A JSON array of proxy records (monosans proxies.json and
        friends): {protocol, host, port, exit_ip, geolocation.country
        .iso_code, ...}.  The per-record protocol + country beat any
        source-level defaults.

        Also handles the geonode shape ({ip, port, protocols: [...],
        country: "DE"}) and proxy-free's shape ({ip, port, protocol,
        country_code: "KR"} inside {"proxies": [...]}).
        """
        out: list[Proxy] = []
        try:
            data = json.loads(text)
        except ValueError:
            return []
        if isinstance(data, dict):
            data = data.get("proxies") or data.get("data") or []
        if not isinstance(data, list):
            return []
        for rec in data:
            if not isinstance(rec, dict):
                continue
            host = str(rec.get("host") or rec.get("ip") or "").strip()
            try:
                port = int(rec.get("port"))
            except (TypeError, ValueError):
                continue
            if not (0 < port < 65536):
                continue
            if not _valid_ip(host) and not _valid_hostname(host):
                continue
            scheme = str(rec.get("protocol") or rec.get("scheme")
                         or "").strip().lower()
            if not scheme:
                # geonode-style: "protocols": ["socks4", "http"]
                protos = rec.get("protocols")
                if isinstance(protos, list):
                    for p in protos:
                        if str(p).strip().lower() in SCHEMES:
                            scheme = str(p).strip().lower()
                            break
            if scheme not in SCHEMES:
                scheme = "http"
            country = ""
            geo = rec.get("geolocation") or rec.get("geo")
            if isinstance(geo, dict):
                c = geo.get("country")
                if isinstance(c, dict):
                    country = str(c.get("iso_code") or "").upper()
            if not country:
                # "country_code": "KR" (proxy-free) or "country": "DE"
                # (geonode) — only trust bare 2-letter codes
                for key in ("country_code", "country"):
                    cc = rec.get(key)
                    if isinstance(cc, str) and re.fullmatch(
                            r"[A-Za-z]{2}", cc.strip()):
                        country = cc.strip().upper()
                        break
            out.append(Proxy(host=host, port=port, scheme=scheme,
                             country=country))
        return out

    @staticmethod
    def parse_html(text: str, scheme: str = "http") -> list[Proxy]:
        """Generic proxy-list <table>.  Three cell layouts, all seen in
        the wild:

        1. one cell holding ``ip:port`` (spys.one)
        2. separate IP cell + numeric Port cell (free-proxy-list.net)
        3. a per-row type cell (HTTP / HTTPS / SOCKS4 / SOCKS5) that
           overrides the source-level scheme (spys mixed pages)
        4. base64 ``data-ip`` / ``data-port`` attributes on the cells
           (advanced.name)

        A 2-3 letter cell is treated as the country code.
        """
        out: list[Proxy] = []
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text,
                              re.IGNORECASE | re.DOTALL):
            host, port = "", 0
            # anti-scraper junk first: strip display:none elements (proxydb
            # hides port-poisoning digits in them; tag-stripping alone
            # would concatenate them onto the real port)
            row = _HIDDEN_ELT_RE.sub("", row)
            # layout 4: base64-encoded ip/port cell attributes — check
            # the raw row HTML before the tags are stripped
            m_ip = re.search(r'data-ip="([^"]+)"', row)
            m_port = re.search(r'data-port="([^"]+)"', row)
            if m_ip and m_port:
                try:
                    host = base64.b64decode(m_ip.group(1)).decode(
                        "utf-8", "replace").strip()
                    port = int(base64.b64decode(m_port.group(1)).decode(
                        "utf-8", "replace").strip())
                except (ValueError, base64.binascii.Error):
                    host, port = "", 0
                if not _valid_ip(host) or not 0 < port < 65536:
                    host, port = "", 0
            cells = [c.strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", row,
                                         re.IGNORECASE | re.DOTALL)]
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            cells = [c for c in cells if c]
            if not host:
                if len(cells) < 2:
                    continue
                # layout 1: an ip:port token inside one cell
                for c in cells:
                    m = _IPPORT_RE.search(c)
                    if m and 0 < int(m.group(2)) < 65536:
                        host, port = m.group(1), int(m.group(2))
                        break
            if not host:
                # layout 2: separate IP cell, then a numeric port cell
                ip_idx = next((i for i, c in enumerate(cells)
                               if _valid_ip(c)), -1)
                if ip_idx < 0:
                    continue
                host = cells[ip_idx]
                for c in cells[ip_idx + 1:]:
                    if c.isdigit() and 0 < int(c) < 65536:
                        port = int(c)
                        break
                if not port:
                    continue
            if not (0 < port < 65536):
                continue
            # layout 3: per-row protocol from a type cell
            row_scheme = scheme
            for c in cells:
                if ":" in c or _valid_ip(c):
                    continue  # never let the address cells count
                t = _TYPE_RE.search(c)
                if t:
                    row_scheme = t.group(1).lower()
                    break
            country = next((c for c in cells
                            if re.fullmatch(r"[A-Z]{2,3}", c)), "")
            # the site's claimed anonymity label (proxydb: "High
            # Anonymous") — the lab's own tested anonymity goes on
            # Proxy.anonymity later; this never overwrites a test
            claimed = ""
            for c in cells:
                if _valid_ip(c) or c.isdigit():
                    continue
                m = _CLAIMED_ANON_RE.search(c)
                if m:
                    claimed = _CLAIMED_ANON_MAP[m.group(1).lower()
                                               .replace("  ", " ")]
                    break
            out.append(Proxy(host=host, port=port, scheme=row_scheme,
                             country=country, claimed_anonymity=claimed))
        return out

    # -- driver -------------------------------------------------------------
    def scrape(self, schemes: str = "http,https,socks4,socks5",
               *, sources: list[tuple[str, str, str]] | None = None,
               limit: int = 2000) -> dict[str, Any]:
        wanted = [s.strip().lower() for s in schemes.split(",") if s.strip()]
        wanted = [s for s in wanted if s in SCHEMES] or list(SCHEMES)
        pool = sources or self.sources
        picked = [(n, u, k) for (n, u, k) in pool
                  if self._source_scheme(n, k) in wanted
                  or k in ("html", "protocol")]
        seen: dict[tuple[str, int, str], Proxy] = {}
        by_source: dict[str, int] = {}
        errors: dict[str, str] = {}
        started = time.time()

        def _one(source: tuple[str, str, str]) -> tuple[str, list[Proxy], str]:
            name, url, kind = source
            try:
                raw = self._fetcher(url)
                text = raw.decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001 - one dead source is fine
                return name, [], str(exc)[:200]
            try:
                if kind == "list":
                    proxies = self.parse_list(text, self._source_scheme(name, kind))
                elif kind == "protocol":
                    proxies = self.parse_protocol(text)
                elif kind == "json":
                    proxies = self.parse_json(text)
                else:
                    proxies = self.parse_html(text, self._source_scheme(name, kind))
            except Exception as exc:  # noqa: BLE001
                return name, [], f"parse: {exc}"[:200]
            return name, proxies, ""

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as pool_ex:
            for name, proxies, error in pool_ex.map(_one, picked):
                if error:
                    errors[name] = error
                    continue
                added = 0
                for p in proxies:
                    if p.scheme not in wanted:
                        continue
                    if p.key in seen:
                        continue
                    if len(seen) >= max(1, int(limit)):
                        break
                    seen[p.key] = p
                    added += 1
                by_source[name] = added
        return {
            "proxies": list(seen.values()),
            "total": len(seen),
            "by_source": by_source,
            "errors": errors,
            "seconds": round(time.time() - started, 1),
            "sources": len(picked),
        }

    @staticmethod
    def _source_scheme(name: str, kind: str) -> str:
        for token in ("socks4", "socks5", "https", "http"):
            if name.endswith(token):
                return token
        return "http"

    # -- wave 83: internet-wide source discovery -----------------------------
    def discover(self, seeds: list[str] | None = None, *,
                 max_repos: int = 12, max_urls: int = 30,
                 min_proxies: int = 5) -> dict[str, Any]:
        """Find NEW proxy-list endpoints from the internet at large.

        Seed pages (aggregator/topic pages that *list* proxy projects)
        are mined for candidate endpoints: GitHub repos linked from the
        page (each .txt file in the repo → a raw list endpoint) and
        direct .txt links.  Every candidate is fetched and auto-parsed;
        the ones that actually yield proxies become registered sources
        for every future scrape.
        """
        from .proxysources import (DISCOVERY_SEEDS, extract_list_candidates,
                                   github_search_repos)

        seed_list = list(seeds or DISCOVERY_SEEDS)
        repo_candidates: list[str] = []
        url_candidates: list[str] = []
        seed_errors: dict[str, str] = {}

        # primary engine: the GitHub search API — proxy-list repos pushed
        # in the last week, freshest first.  No hardcoded repo list: the
        # catalog refreshes itself from the internet on every run.
        search_repos = github_search_repos(days=7, limit=max_repos,
                                           fetch=self._fetcher)
        repo_candidates.extend(search_repos)
        seed_errors["github-search"] = (
            "ok" if search_repos else "no results (offline or rate-limited)")

        for seed in seed_list:
            try:
                raw = self._fetcher(seed)
                text = raw.decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001
                seed_errors[seed] = str(exc)[:160]
                continue
            candidates = extract_list_candidates(seed, text,
                                                 repo_limit=max_repos,
                                                 url_limit=max_urls)
            repo_candidates.extend(candidates["github"])
            url_candidates.extend(candidates["urls"])

        # GitHub repos → their raw .txt files (HEAD works for any
        # default branch)
        endpoints: list[tuple[str, str]] = []  # (name, url)
        seen_urls: set[str] = set()
        for slug in repo_candidates:
            api = f"https://api.github.com/repos/{slug}/contents/"
            try:
                listing = self._fetcher(api)
                files = json.loads(listing.decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(files, list):
                continue
            for item in files:
                fname = str(item.get("name", ""))
                if not fname.lower().endswith((".txt", ".list")):
                    continue
                raw_url = (f"https://raw.githubusercontent.com/{slug}"
                           f"/HEAD/{fname}")
                if raw_url in seen_urls:
                    continue
                seen_urls.add(raw_url)
                endpoints.append((self._name_for(slug, fname), raw_url))
        for url in url_candidates:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            endpoints.append((self._name_for(url, ""), url))

        # test every endpoint; keep the ones that yield proxies
        tested = 0
        registered: list[dict[str, Any]] = []
        for name, url in endpoints[:max_repos * 4 + max_urls]:
            tested += 1
            try:
                raw = self._fetcher(url)
                text = raw.decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                continue
            proxies, kind = self._auto_parse(text)
            if len(proxies) < max(1, int(min_proxies)):
                continue
            registered.append({"name": name, "url": url, "kind": kind,
                               "found": len(proxies)})
        return {
            "seeds": seed_list,
            "seed_errors": seed_errors,
            "endpoints_tested": tested,
            "registered": registered,
        }

    @staticmethod
    def _name_for(slug_or_url: str, filename: str) -> str:
        # repo-aware names: found-<owner>-<file> — two repos both
        # hosting http.txt must not collide into one source
        if filename:
            base = re.sub(r"\.[^.]+$", "", Path(filename).name).lower()
            if "/" in slug_or_url:
                owner = slug_or_url.split("/", 1)[0].lower()
                return f"found-{owner}-{base}"[:48]
            return f"found-{base}"[:48]
        host = urlparse(slug_or_url).hostname or slug_or_url
        base = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")
        return f"found-{base}"[:48]

    @classmethod
    def _auto_parse(cls, text: str) -> tuple[list["Proxy"], str]:
        """What kind of list is this?  JSON array > protocol lines >
        bare host:port lines > an html table."""
        head = text.lstrip()[:1]
        if head in "{[":
            try:
                json.loads(text)
            except ValueError:  # noqa: E103 - falls through to protocol/list parsers
                pass
            else:
                proxies = cls.parse_json(text)
                if proxies:
                    return proxies, "json"
        if re.search(r"^(?:http|socks)[a-z0-9]*://\S+", text,
                     re.IGNORECASE | re.MULTILINE):
            return cls.parse_protocol(text), "protocol"
        if _IPPORT_RE.search(text):
            return cls.parse_list(text, "http"), "list"
        if "<tr" in text.lower():
            return cls.parse_html(text), "html"
        return [], ""


def _valid_hostname(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                             value.lower()))


# ── tester ───────────────────────────────────────────────────────────────────

_ANON_SCORE = {"elite": 3, "anonymous": 2, "unverified": 1, "transparent": 0}

#: headers whose presence means "the site knows a proxy is involved"
_LEAK_HEADERS = ("x-forwarded-for", "x-real-ip", "proxy-connection",
                 "proxy-client_ip", "proxy-authorization", "forwarded",
                 "via", "x-forwarded-proto")

#: proxydb.net poisons its port cells with <div style="display:none">NN</div>
#: junk that concatenates onto the real port after tag-stripping
#: ("12"+"1090" = "121090" — an invalid port, so the whole row was
#: dropped; ("12"+"80" = "1280" — a WRONG port that passed).  Strip
#: hidden elements before any cell parsing.
_HIDDEN_ELT_RE = re.compile(
    r"<([a-z][a-z0-9]*)[^>]*style=[\"'][^\"']*display\s*:\s*none"
    r"[^\"']*[\"'][^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL)

#: proxydb.net's per-row anonymity label (the site's CLAIM, not the
#: lab's tested result — recorded separately on the Proxy).
_CLAIMED_ANON_RE = re.compile(
    r"\b(high\s*anonymous|elite|anonymous|transparent)\b", re.IGNORECASE)
_CLAIMED_ANON_MAP = {"high anonymous": "elite", "elite": "elite",
                     "anonymous": "anonymous", "transparent": "transparent"}


class ProxyTester:
    """Tests proxies for real: liveness, latency, egress IP, anonymity,
    country.  All probes go over raw sockets so HTTP and both SOCKS
    versions are handled with zero third-party dependencies."""

    def __init__(self, *,
                 ip_url: str = "http://api.ipify.org",
                 echo_url: str = "http://httpbin.org/headers",
                 country_url: str = "http://ip-api.com/json?fields=country",
                 local_ip_provider: Callable[[], str] | None = None,
                 timeout: float = 6.0,
                 max_workers: int = 0,
                 profile: str = "",
                 detect_country: bool = True,
                 country_budget: int = 60) -> None:
        self.ip_url = ip_url
        self.echo_url = echo_url
        self.country_url = country_url
        self.timeout = float(timeout)
        self.max_workers = max(1, int(max_workers)) if max_workers else \
            profile_workers(profile or detect_profile(), "test")
        self.profile = profile or detect_profile()
        self.detect_country = bool(detect_country)
        self.country_budget = max(0, int(country_budget))
        self._local_ip_provider = local_ip_provider
        self._local_ip_cache: str = ""
        self._country_spent = 0
        self._country_lock = threading.Lock()

    # -- helpers -------------------------------------------------------------
    def _local_ip(self) -> str:
        if not self._local_ip_cache:
            if self._local_ip_provider is not None:
                candidate = (self._local_ip_provider() or "").strip()
            else:
                try:
                    candidate = self._direct_get(self.ip_url).strip()
                except Exception:  # noqa: BLE001
                    candidate = ""
            # same guard as the probe: a rate-limit page is not an IP
            self._local_ip_cache = candidate if _valid_ip(candidate) else ""
        return self._local_ip_cache

    def _direct_get(self, url: str) -> str:
        parsed = urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        sock = _connect(parsed.hostname, port, self.timeout)
        try:
            if parsed.scheme == "https":
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                sock = ctx.wrap_socket(sock, server_hostname=parsed.hostname)
            _, _, body = _http_exchange(
                sock, f"GET {path} HTTP/1.1",
                {"Host": parsed.hostname,
                 "User-Agent": _UA})
            return body
        finally:
            sock.close()

    def _route(self, proxy: Proxy, target_url: str) -> tuple[socket.socket, str, dict[str, str]]:
        """Open a connection to ``target_url`` THROUGH the proxy.
        Returns (socket, request_line, extra-headers)."""
        parsed = urlparse(target_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        host_header = parsed.hostname
        if proxy.scheme in ("socks4", "socks5"):
            if proxy.scheme == "socks4":
                target_ip = socket.gethostbyname(parsed.hostname)
                sock = socks4_connect(proxy.host, proxy.port, target_ip,
                                      port, timeout=self.timeout)
            else:
                sock = socks5_connect(proxy.host, proxy.port,
                                      parsed.hostname, port,
                                      timeout=self.timeout)
            request_line = f"GET {path} HTTP/1.1"
            return sock, request_line, {"Host": host_header, "User-Agent": _UA}
        # HTTP(S) proxies use the absolute-form request line
        request_line = f"GET {target_url} HTTP/1.1"
        sock = _connect(proxy.host, proxy.port, self.timeout)
        return sock, request_line, {"Host": host_header, "User-Agent": _UA}

    def _probe(self, proxy: Proxy, target_url: str,
               marker: str) -> tuple[int, dict[str, str], str]:
        sock, line, headers = self._route(proxy, target_url)
        headers["X-ProxyLab-Marker"] = marker
        try:
            return _http_exchange(sock, line, headers, self.timeout)
        finally:
            sock.close()

    # -- the test ------------------------------------------------------------
    def test_one(self, proxy: Proxy) -> Proxy:
        marker = f"nl-{os.getpid()}-{threading.get_ident() & 0xffff:x}"
        started = time.monotonic()
        try:
            status, _, body = self._probe(proxy, self.ip_url, marker)
        except Exception as exc:  # noqa: BLE001 - a dead proxy is a result
            proxy.alive = False
            proxy.error = str(exc)[:200]
            proxy.latency_ms = round((time.monotonic() - started) * 1000)
            proxy.tested_at = time.time()
            return proxy
        latency = round((time.monotonic() - started) * 1000)
        egress = body.strip()[:64]
        # the probe must come back as an IP — a rate-limit JSON page or
        # block notice ({"error": "Too Many Requests"}) is NOT proof of
        # relay; without this a throttled probe URL marks dead proxies
        # alive with garbage as their "egress IP"
        if status != 200 or not _valid_ip(egress):
            proxy.alive = False
            proxy.error = (f"probe status {status}" if status != 200
                           else f"probe returned non-IP: {egress[:40]}")
            proxy.latency_ms = latency
            proxy.tested_at = time.time()
            return proxy
        proxy.alive = True
        proxy.record_success(latency)
        proxy.egress_ip = egress
        proxy.tested_at = time.time()

        # anonymity: does the site see our real IP / proxy headers?
        try:
            local_ip = self._local_ip()
        except Exception:  # noqa: BLE001
            local_ip = ""
        if local_ip and proxy.egress_ip == local_ip:
            proxy.anonymity = "transparent"
            self._maybe_country(proxy)
            return proxy
        try:
            status2, _, echo_body = self._probe(proxy, self.echo_url, marker)
            if status2 == 200:
                try:
                    headers = (json.loads(echo_body) or {}).get("headers") or {}
                    lowered = {k.lower() for k in headers}
                except (ValueError, TypeError):
                    lowered = set()
                proxy.anonymity = ("anonymous"
                                   if any(h in lowered for h in _LEAK_HEADERS)
                                   else "elite")
            else:
                proxy.anonymity = "unverified"
        except Exception:  # noqa: BLE001 - anonymity is best-effort
            proxy.anonymity = "unverified"
        self._maybe_country(proxy)
        return proxy

    def _maybe_country(self, proxy: Proxy) -> None:
        if not self.detect_country or proxy.country:
            return
        with self._country_lock:
            if self._country_spent >= self.country_budget:
                return
            self._country_spent += 1
        try:
            status, _, body = self._probe(proxy, self.country_url, "c")
            if status == 200:
                proxy.country = str(json.loads(body).get("country") or "")[:2]
        except Exception:  # noqa: BLE001 - country is cosmetic
            pass

    def test_many(self, proxies: list[Proxy], *,
                  limit: int = 150) -> list[Proxy]:
        """Concurrent testing.  Returns the SAME proxy objects, mutated."""
        targets = list(proxies)[:max(1, int(limit))]
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as pool:
            list(pool.map(self.test_one, targets))
        return targets


#: EU members + UK/CH/NO (low-latency routes from Nigeria on most
#: backbones; the NG bonus is scored first, EU second).
_EU_COUNTRIES = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
    "RO", "SK", "SI", "ES", "SE", "GB", "CH", "NO",
})


def score_proxy(proxy: Proxy, at: float | None = None, *,
                prefer_countries: frozenset[str] | set[str] | None = None,
                half_life_hours: float = 6.0) -> float:
    """One number, higher is better.  Alive first; anonymity outranks
    speed (an elite 800ms proxy beats a transparent 100ms one); speed
    uses the time-decayed latency so a proxy that GOT slow recently
    sinks; failures and backoff demote; a country preference (NG first,
    EU second for Nigeria-friendly routing) adds a bonus.

    Returns ``float("-inf")`` for proxies that must not route now
    (dead, in backoff) — callers filter on that, never on silence.
    """
    now = at if at is not None else time.time()
    if not proxy.alive:
        return float("-inf")
    if float(proxy.backoff_until or 0.0) > now:
        return float("-inf")
    anon = _ANON_SCORE.get(proxy.anonymity, 1)
    if proxy.anonymity == "transparent":
        anon = -2  # leaks the real IP — worse than no anonymity data
    # latency: decayed ms → a 0..1-ish term (5s is effectively "unusable")
    decayed = proxy.decayed_latency(at=now, half_life_hours=half_life_hours)
    latency_term = max(0.0, 1.0 - (decayed / 5000.0))
    # failure streaks drag the score even after the backoff expires
    failure_term = -0.5 * min(int(proxy.consecutive_failures or 0), 6)
    country_bonus = 0.0
    cc = (proxy.country or "").upper()
    if prefer_countries and cc in prefer_countries:
        country_bonus = 0.75
    elif cc == "NG":
        country_bonus = 1.0      # Nigeria-first, always
    elif cc in _EU_COUNTRIES:
        country_bonus = 0.5      # EU second — Nigeria-friendly routes
    return (anon * 2.0) + (latency_term * 3.0) + failure_term + country_bonus


def rank_proxies(proxies: list[Proxy], *,
                 prefer_countries: frozenset[str] | set[str] | None = None,
                 at: float | None = None) -> list[Proxy]:
    """Alive first, then score (anonymity → decayed speed → failures →
    country preference), dead last.  Backed-off proxies sort with the
    dead — they must not route until the backoff expires."""
    now = at if at is not None else time.time()
    scored: list[tuple[float, Proxy]] = []
    dead: list[Proxy] = []
    for p in proxies:
        s = score_proxy(p, at=now, prefer_countries=prefer_countries)
        if s == float("-inf"):
            dead.append(p)
        else:
            scored.append((s, p))
    scored.sort(key=lambda t: (-t[0], t[1].url))
    dead.sort(key=lambda p: (-int(p.alive), p.url))
    return [p for _, p in scored] + dead


# ── store ───────────────────────────────────────────────────────────────────


class ProxyStore:
    """Durable pool: working.txt / working.json / candidates.json under
    the NoMorals home dir (``$NM_HOME/proxies``)."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.working_txt = self.dir / "working.txt"
        self.working_json = self.dir / "working.json"
        self.candidates_json = self.dir / "candidates.json"
        self._lock = threading.Lock()

    def save(self, working: list[Proxy],
             candidates: list[Proxy] | None = None) -> dict[str, int]:
        with self._lock:
            alive = sorted((p for p in working if p.alive),
                           key=lambda p: p.url)
            # working.txt: the clean alive-only list for consumers;
            # working.json: the FULL record (dead + backing-off proxies
            # keep their failure history — nothing dropped silently)
            self.working_txt.write_text(
                "".join(p.url + "\n" for p in alive), encoding="utf-8")
            self.working_json.write_text(
                json.dumps([p.to_dict()
                            for p in sorted(working, key=lambda p: p.url)],
                           indent=1),
                encoding="utf-8")
            if candidates is not None:
                dedup: dict[tuple[str, int, str], Proxy] = {}
                for p in candidates:
                    dedup.setdefault(p.key, p)
                self.candidates_json.write_text(
                    json.dumps([p.to_dict() for p in dedup.values()],
                               indent=1), encoding="utf-8")
            return {"working": len(alive),
                    "candidates": len(self.load_candidates())}

    def load_working(self) -> list[Proxy]:
        try:
            data = json.loads(self.working_json.read_text(encoding="utf-8"))
            return [Proxy.from_dict(d) for d in data if isinstance(d, dict)]
        except (OSError, ValueError, TypeError):
            return []

    def load_candidates(self) -> list[Proxy]:
        try:
            data = json.loads(self.candidates_json.read_text(encoding="utf-8"))
            return [Proxy.from_dict(d) for d in data if isinstance(d, dict)]
        except (OSError, ValueError, TypeError):
            return []

    def clear(self) -> None:
        with self._lock:
            for path in (self.working_txt, self.working_json,
                         self.candidates_json):
                try:
                    path.unlink()
                except OSError:  # noqa: E103 - cache file may not exist
                    pass

    # -- pool queries ---------------------------------------------------------
    def pool(self, *, scheme: str = "", country: str = "",
             anonymity: str = "", max_age_hours: float = 24.0,
             limit: int = 10,
             prefer_countries: frozenset[str] | set[str] | None = None,
             ) -> list[Proxy]:
        """Ranked, fresh, working proxies (dead and stale are dropped).

        Ranking uses :func:`score_proxy` — anonymity outranks speed,
        speed is the time-decayed latency, failure streaks demote, and
        NG/EU countries get a Nigeria-friendly bonus.
        """
        now = time.time()
        proxies = self.load_working()
        out = [p for p in proxies
               if p.alive and p.age_hours <= max_age_hours
               and float(p.backoff_until or 0.0) <= now]
        if scheme:
            out = [p for p in out if p.scheme == scheme]
        if country:
            out = [p for p in out if p.country.lower() == country.lower()]
        if anonymity:
            out = [p for p in out if p.anonymity == anonymity]
        return rank_proxies(out, prefer_countries=prefer_countries,
                            at=now)[:max(1, int(limit))]

    def by_url(self, url: str) -> Proxy | None:
        """One stored proxy by its ``scheme://host:port`` URL."""
        for p in self.load_working():
            if p.url == url:
                return p
        return None

    def record_routing_result(self, url: str, *, ok: bool,
                              latency_ms: float = 0.0,
                              reason: str = "",
                              backoff_seconds: float = 300.0) -> Proxy | None:
        """Failure-triggered rotation feeds back here: a proxy that
        failed a real request gets demoted with backoff (recorded on
        the proxy itself — never dropped silently); a success clears
        the streak and samples the latency.  Persists the pool."""
        with self._lock:
            working = self.load_working()
            target = next((p for p in working if p.url == url), None)
            if target is None:
                return None
            if ok:
                target.record_success(latency_ms)
            else:
                target.record_failure(reason,
                                      backoff_seconds=backoff_seconds)
            # working.txt stays the clean alive-only list; working.json
            # keeps EVERYTHING (including backing-off proxies) so the
            # failure streak / backoff is never dropped silently
            alive = sorted((p for p in working if p.alive),
                           key=lambda p: p.url)
            all_sorted = sorted(working, key=lambda p: p.url)
            self.working_txt.write_text(
                "".join(p.url + "\n" for p in alive), encoding="utf-8")
            self.working_json.write_text(
                json.dumps([p.to_dict() for p in all_sorted], indent=1),
                encoding="utf-8")
            return target

    def urls(self, **filters: Any) -> list[str]:
        return [p.url for p in self.pool(**filters)]

    def active_file(self, *, limit: int = 1000, max_age_hours: float = 24.0,
                    filename: str = "proxies-active.txt") -> dict[str, Any]:
        """A CLEAN file of the currently working proxies (wave 68):
        one ``scheme://ip:port`` per line, dead and stale entries
        dropped, fastest-first.  Refreshes the file from the live pool
        (never a stale copy) and reports the path so any layer — chat
        (sent as a file), devon, the CLI — can hand it to the owner.

        Returns ``{"path": ..., "count": ..., "written": True}`` or
        ``{"path": ..., "count": 0, "written": False, "note": ...}``
        when the pool is empty (the file is left untouched).
        """
        proxies = self.pool(limit=limit, max_age_hours=max_age_hours)
        target = self.dir / filename
        if not proxies:
            return {"path": str(target), "count": 0, "written": False,
                    "note": "no working proxies right now — /proxy refresh "
                            "scrapes + tests first"}
        lines = "".join(p.url + "\n" for p in proxies)
        target.write_text(lines, encoding="utf-8")
        return {"path": str(target), "count": len(proxies), "written": True,
                "fastest": proxies[0].url,
                "latency_ms": proxies[0].latency_ms}


# ── per-domain proxy affinity ──────────────────────────────────────────────


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    except ValueError:
        host = ""
    return host.strip().lower().lstrip(".")


class DomainAffinityManager:
    """Sticky proxy per site: the same domain keeps using the same proxy
    (sessions, rate-limit buckets, and login flows all survive rotation).

    Bindings are persisted (``affinity.json`` next to the pool) with a
    TTL — a binding to a dead / backing-off / stale proxy is dropped on
    lookup and the caller falls back to a fresh pick, which re-binds.
    """

    def __init__(self, path: str | os.PathLike[str],
                 default_ttl_hours: float = 6.0) -> None:
        self.path = Path(path)
        self.default_ttl = max(0.5, float(default_ttl_hours or 6.0))
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {str(k): v for k, v in raw.items()
                                  if isinstance(v, dict)}
        except Exception:  # noqa: BLE001 - corrupt affinity = start clean
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=1),
                                 encoding="utf-8")
        except OSError:  # noqa: E103 - affinity is best-effort
            pass

    def bind(self, domain: str, proxy_url: str,
             ttl_hours: float | None = None) -> dict[str, Any]:
        domain = _domain_of(domain)
        if not domain or not proxy_url:
            raise ToolError("affinity bind needs a domain and a proxy URL")
        with self._lock:
            self._data[domain] = {
                "proxy_url": proxy_url,
                "bound_at": time.time(),
                "ttl_hours": max(0.5, float(ttl_hours or self.default_ttl)),
            }
            self._save()
            return {"domain": domain, **self._data[domain]}

    def lookup(self, domain: str,
               store: "ProxyStore | None" = None) -> str:
        """The bound proxy URL, or ``""``.  A binding is only honored
        when the proxy is still alive, fresh, and out of backoff —
        otherwise it is dropped (with the reason logged) and the
        caller re-picks."""
        domain = _domain_of(domain)
        if not domain:
            return ""
        with self._lock:
            entry = self._data.get(domain)
            if not entry:
                return ""
            age_h = (time.time() - float(entry.get("bound_at", 0))) / 3600.0
            if age_h > float(entry.get("ttl_hours", self.default_ttl)):
                _log.info("affinity: %s binding expired (%.1fh old)",
                          domain, age_h)
                del self._data[domain]
                self._save()
                return ""
            url = str(entry.get("proxy_url") or "")
        if store is not None:
            proxy = store.by_url(url)
            if proxy is None or not proxy.alive or proxy.in_backoff \
                    or proxy.age_hours > 24:
                with self._lock:
                    self._data.pop(domain, None)
                    self._save()
                _log.info("affinity: %s binding dropped (proxy %s)",
                          domain,
                          "gone" if proxy is None
                          else "in backoff" if proxy.in_backoff
                          else "dead/stale")
                return ""
        return url

    def release(self, domain: str) -> bool:
        with self._lock:
            removed = self._data.pop(_domain_of(domain), None) is not None
            if removed:
                self._save()
            return removed

    def bindings(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.time()
            return [
                {"domain": d,
                 "proxy_url": e.get("proxy_url", ""),
                 "age_hours": round(
                     (now - float(e.get("bound_at", now))) / 3600.0, 1),
                 "ttl_hours": float(e.get("ttl_hours", self.default_ttl))}
                for d, e in sorted(self._data.items())
            ]

    def prune(self, store: "ProxyStore | None" = None) -> int:
        """Drop expired / dead-proxy bindings; returns the drop count."""
        with self._lock:
            domains = list(self._data)
        dropped = 0
        for d in domains:
            before = self._data.get(d, {}).get("proxy_url", "")
            if not self.lookup(d, store=store):
                dropped += 1
            elif self._data.get(d, {}).get("proxy_url") != before:
                dropped += 1
        return dropped


# ── honest empty-pool guidance ─────────────────────────────────────────────


def empty_pool_guidance(store: "ProxyStore", *, want: str = "") -> str:
    """Exactly why the pool is empty and what to run — never a bare
    ``pool empty``.  ``want`` describes the filters that found nothing."""
    try:
        all_known = store.load_working()
        alive = [p for p in all_known if p.alive]
        backing_off = sum(1 for p in alive if p.in_backoff)
        candidates = len(store.load_candidates())
        last_test = max((p.tested_at for p in all_known), default=0.0)
    except Exception:  # noqa: BLE001
        alive, backing_off, candidates, last_test = [], 0, 0, 0.0
    want_s = f" for {want}" if want else ""
    if backing_off and not [p for p in alive if not p.in_backoff]:
        return (f"every working proxy{want_s} is in failure backoff right "
                f"now ({backing_off}) — they cool down automatically, or "
                f"tap `/proxy refresh` to re-test and rebuild the pool")
    if not alive:
        age = ""
        if last_test:
            hours = (time.time() - last_test) / 3600.0
            age = f" (last test {hours:.1f}h ago)"
        if candidates:
            return (f"no working proxies{want_s}{age} — {candidates} "
                    f"candidates are stored but none passed. Tap "
                    f"`/proxy refresh` to re-test them, or `/proxy scrape` "
                    f"to pull fresh lists first")
        return (f"no working proxies{want_s}{age} — nothing stored yet. "
                f"Tap `/proxy refresh` (scrapes + tests in one go)")
    return (f"no proxy matches{want_s} ({len(alive)} working, none fit) — "
            f"loosen the filters or tap `/proxy refresh` for fresh stock")


# ── facade used by the tools ─────────────────────────────────────────────────


class ProxyLab:
    """Scrape → test → rank → store → serve.  One object, clean API."""

    def __init__(self, context: Any) -> None:
        self.context = context
        settings = getattr(context, "settings", None)
        home = str(getattr(settings, "home", "~/.nomorals")) if settings \
            else "~/.nomorals"
        self.store = ProxyStore(Path(os.path.expanduser(home)) / "proxies")
        from .proxysources import SourceRegistry
        self.registry = SourceRegistry(self.store.dir / "sources.json")
        self._profile = detect_profile(context)
        self._scraper = ProxyScraper(sources=self.registry.sources(),
                                     profile=self._profile)
        self._tester: ProxyTester | None = None

    @property
    def profile(self) -> str:
        return self._profile

    def scraper(self) -> ProxyScraper:
        return self._scraper

    def tester(self) -> ProxyTester:
        if self._tester is None:
            self._tester = ProxyTester(profile=self._profile)
        return self._tester

    def scrape(self, schemes: str = "http,https,socks4,socks5",
               limit: int = 2000,
               sources: list[tuple[str, str, str]] | None = None) -> dict[str, Any]:
        pool = sources or self.registry.sources()
        self._scraper.sources = pool
        report = self._scraper.scrape(schemes, sources=pool, limit=limit)
        try:
            self.store.save([], candidates=report["proxies"])
        except OSError as exc:
            report["store_error"] = str(exc)
        # wave 83: every source's outcome feeds its health record —
        # three straight failures retire it for a day, a good harvest
        # reinstates it
        errors = report.get("errors", {})
        by_source = report.get("by_source", {})
        for name in set(list(errors) + list(by_source)):
            found = int(by_source.get(name, 0))
            self.registry.record(name, ok=found > 0, found=found,
                                 error=str(errors.get(name, ""))[:200])
        report["disabled_sources"] = [d["name"] for d in
                                      self.registry.disabled()]
        report["source_stats"] = self.registry.stats()
        return report

    def discover_sources(self, seeds: list[str] | None = None, *,
                         min_proxies: int = 5) -> dict[str, Any]:
        """Wave 83: learn NEW sources from the internet, register the
        working ones.  See ProxyScraper.discover for the mechanics."""
        report = self._scraper.discover(seeds, min_proxies=min_proxies)
        registered = []
        for cand in report.get("registered", []):
            self.registry.add_discovered(
                cand["name"], cand["url"], cand["kind"],
                seed="; ".join(report.get("seeds", [])[:3]),
                found=cand["found"])
            registered.append(cand)
        report["registered"] = registered
        report["source_stats"] = self.registry.stats()
        self._scraper.sources = self.registry.sources()
        return report

    def source_health(self) -> dict[str, Any]:
        return {"sources": self.registry.health(),
                "stats": self.registry.stats()}

    def refresh(self, schemes: str = "http,https,socks4,socks5",
                limit: int = 150, *, detect_country: bool = True,
                scrape_if_empty: bool = True) -> dict[str, Any]:
        """The full cycle: candidates (stored or freshly scraped) → test →
        store the working pool → summary."""
        candidates = self.store.load_candidates()
        scraped = 0
        errors: dict[str, str] = {}
        if not candidates and scrape_if_empty:
            report = self.scrape(schemes, limit=limit * 3)
            candidates = report["proxies"]
            scraped = report["total"]
            errors = report.get("errors", {})
        ranked_in = rank_proxies(candidates)  # fastest-known first
        tester = self.tester()
        tester.detect_country = bool(detect_country)
        tested = tester.test_many(ranked_in, limit=limit)
        working = self.store.load_working()
        by_key = {p.key: p for p in working}
        for p in tested:
            by_key[p.key] = p
        all_known = list(by_key.values())
        self.store.save(all_known, candidates=candidates)
        fresh = rank_proxies([p for p in all_known if p.alive])
        by_scheme: dict[str, int] = {}
        by_anon: dict[str, int] = {}
        for p in fresh:
            by_scheme[p.scheme] = by_scheme.get(p.scheme, 0) + 1
            by_anon[p.anonymity or "?"] = by_anon.get(p.anonymity or "?", 0) + 1
        return {
            "scraped": scraped,
            "candidates": len(candidates),
            "tested": len(tested),
            "working": len(fresh),
            "by_scheme": by_scheme,
            "by_anonymity": by_anon,
            "errors": errors,
            "best": [p.to_dict() for p in fresh[:10]],
            "store": str(self.store.dir),
        }

    def pool(self, **filters: Any) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.store.pool(**filters)]

    def active_file(self, *, limit: int = 1000, max_age_hours: float = 24.0,
                    filename: str = "proxies-active.txt") -> dict[str, Any]:
        """A clean file of the currently working proxies (wave 68) —
        delegates to the store, which owns the pool and the file."""
        return self.store.active_file(limit=limit,
                                      max_age_hours=max_age_hours,
                                      filename=filename)

    def status(self) -> dict[str, Any]:
        working = self.store.load_working()
        alive = [p for p in working if p.alive]
        fresh = [p for p in alive if p.age_hours <= 24]
        return {
            "store": str(self.store.dir),
            "working_total": len(alive),
            "fresh_24h": len(fresh),
            "candidates": len(self.store.load_candidates()),
            "by_scheme": {s: sum(1 for p in alive if p.scheme == s)
                          for s in SCHEMES
                          if any(p.scheme == s for p in alive)},
            "best": [p.url for p in rank_proxies(alive)[:5]],
        }

    def health(self) -> dict[str, Any]:
        """The dashboard: pool size by protocol / anonymity / country,
        decayed-latency average, last refresh, dead-proxy counts,
        rotation + affinity + schedule state.  Empty pool → honest
        guidance, never a bare zero."""
        now = time.time()
        working = self.store.load_working()
        alive = [p for p in working if p.alive]
        fresh = [p for p in alive if p.age_hours <= 24]
        routable = [p for p in fresh if not p.in_backoff]
        backing_off = [p for p in alive if p.in_backoff]
        dead = [p for p in working if not p.alive]
        tested = [p for p in alive if p.latency_samples or p.latency_ms]
        avg_latency = (round(sum(p.decayed_latency(at=now) for p in tested)
                             / len(tested)) if tested else 0)

        def _counts(key: Callable[[Proxy], str]) -> dict[str, int]:
            out: dict[str, int] = {}
            for p in routable:
                k = key(p) or "?"
                out[k] = out.get(k, 0) + 1
            return dict(sorted(out.items(), key=lambda kv: -kv[1]))

        by_scheme = _counts(lambda p: p.scheme)
        by_anonymity = _counts(lambda p: p.anonymity or "unverified")
        by_country = _counts(lambda p: (p.country or "?").upper())
        ng_eu = sum(1 for p in routable
                    if (p.country or "").upper() in ({"NG"} | _EU_COUNTRIES))

        last_refresh = 0.0
        try:
            last_refresh = self.store.working_json.stat().st_mtime
        except OSError:  # noqa: E103 - never refreshed
            pass

        # rotation + affinity + schedule state (best-effort; the lab
        # dashboard must not die because rotation state is missing)
        rotation: dict[str, Any] = {}
        affinity_n = 0
        try:
            rot = ProxyRotationManager(self.context)
            rotation = rot.status()
            aff = DomainAffinityManager(self.store.dir / "affinity.json")
            affinity_n = len(aff.bindings())
        except Exception:  # noqa: BLE001
            pass
        schedule: dict[str, Any] = {}
        try:
            from ..agents.scheduler import Scheduler

            jobs = Scheduler(self.context).list_jobs(include_disabled=True)
            job = next((j for j in jobs if j.get("name") == "proxy_refresh"),
                       None)
            if job:
                schedule = {"active": bool(job.get("enabled")),
                            "spec": job.get("spec", ""),
                            "last_run": job.get("last_run", "")}
        except Exception:  # noqa: BLE001
            pass

        report: dict[str, Any] = {
            "pool": {
                "routable": len(routable),
                "fresh_24h": len(fresh),
                "alive_total": len(alive),
                "in_backoff": len(backing_off),
                "dead": len(dead),
                "candidates": len(self.store.load_candidates()),
                "by_scheme": by_scheme,
                "by_anonymity": by_anonymity,
                "top_countries": dict(list(by_country.items())[:12]),
                "ng_or_eu": ng_eu,
                "avg_latency_ms": avg_latency,
                "last_refresh": (time.strftime("%Y-%m-%d %H:%M",
                                               time.localtime(last_refresh))
                                 if last_refresh else "never"),
                "last_refresh_hours_ago": (round((now - last_refresh) / 3600.0, 1)
                                           if last_refresh else None),
            },
            "rotation": {
                "enabled": bool(rotation.get("enabled")),
                "strategy": rotation.get("strategy", ""),
                "healthy": rotation.get("healthy", 0),
                "cooling": len(rotation.get("cooling", [])),
                "direct_fallbacks": (rotation.get("stats") or {})
                .get("direct_fallbacks", 0),
            },
            "affinity_bindings": affinity_n,
            "schedule": schedule,
            "store": str(self.store.dir),
        }
        if not routable:
            report["guidance"] = empty_pool_guidance(self.store)
            report["tap"] = "/proxy refresh"
        return report


# ── rotation ─────────────────────────────────────────────────────────────────


_ROTATE_KV = "proxy.rotate"
_STRATEGIES = ("round_robin", "random", "sticky", "least_used")


class ProxyRotationManager:
    """Cycles the working pool through outbound requests — with failover.

    While enabled, it registers itself as the core.http per-request
    proxy resolver: every ``HttpClient(proxy_url="")`` (i.e. every web
    tool, OSINT call, research fetch, social media request) gets the
    NEXT healthy proxy according to the strategy instead of one static
    proxy.  Connection-level failures are reported back by core.http,
    which puts the offender on cooldown and fails over to the next one —
    the system keeps routing instead of stalling on a dead proxy.

    Strategies:
      round_robin — strict cycle in ranked order (default)
      random      — uniform pick from the healthy set
      sticky      — keep one proxy until it fails, then fail over
      least_used  — always take the least-served healthy proxy

    The pool = the lab store's fresh working proxies plus every live
    SSH tunnel SOCKS5 URL (tunnels are always fresh).  When everything
    is in cooldown, traffic falls back to DIRECT (and the fallback is
    counted, so silence is never mistaken for health).  State persists
    in kv_store and re-registers itself on boot.
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        self.state = self._load_state()
        if self.state.get("enabled"):
            self._register_hooks()

    # -- state --------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        default = {
            "enabled": False,
            "strategy": "round_robin",
            "cooldown_seconds": 300,
            "failover_seconds": 60,
            "max_age_hours": 24.0,
            "index": 0,
            "current": "",
            "usage": {},
            "cooldown": {},
            "stats": {"requests": 0, "failovers": 0,
                      "direct_fallbacks": 0, "since": 0.0},
        }
        db = getattr(self.context, "db", None)
        if db is None:
            return default
        try:
            row = db.query_one("SELECT value FROM kv_store WHERE key = ?",
                               (_ROTATE_KV,))
            if row:
                merged = dict(default)
                merged.update(json.loads(row["value"]))
                merged.setdefault("usage", {})
                merged.setdefault("cooldown", {})
                merged.setdefault("stats", default["stats"])
                return merged
        except Exception:  # noqa: BLE001
            pass
        return default

    def _save_state(self) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            with db.transaction():
                db.execute(
                    "INSERT INTO kv_store (key, value, kind, updated_at) "
                    "VALUES (?, ?, 'json', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                    "updated_at = excluded.updated_at",
                    (_ROTATE_KV, json.dumps(self.state, default=str),
                     time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.warning("could not persist rotation state: %s", exc)

    # -- hooks ----------------------------------------------------------------
    def _register_hooks(self) -> None:
        from ..core import http as core_http

        core_http.set_proxy_resolver(self._resolver)
        core_http.set_proxy_error_reporter(self.report_error)
        core_http.set_proxy_success_reporter(self.report_success)

    def _unregister_hooks(self) -> None:
        from ..core import http as core_http

        core_http.set_proxy_resolver(None)
        core_http.set_proxy_error_reporter(None)
        core_http.set_proxy_success_reporter(None)

    def _resolver(self, target: str = "") -> str:
        """core.http calls this per request (with the target URL when it
        knows it).  Domain affinity wins: a site keeps its sticky proxy;
        otherwise the strategy picks."""
        if not self.state.get("enabled"):
            return ""
        try:
            lab = self._lab()
            if target:
                bound = self._affinity().lookup(target, store=lab.store)
                if bound:
                    self._count_served(bound)
                    return bound
            chosen = self.pick()
            if chosen and target:
                self._affinity().bind(target, chosen)
            return chosen
        except Exception as exc:  # noqa: BLE001 - routing must never raise
            _log.warning("rotation resolver failed: %s", exc)
            return ""

    # -- lab / affinity ---------------------------------------------------------
    def _lab(self) -> "ProxyLab":
        lab = getattr(self, "_lab_cache", None)
        if lab is None:
            lab = ProxyLab(self.context)
            self._lab_cache = lab
        return lab

    def _affinity(self) -> DomainAffinityManager:
        aff = getattr(self, "_affinity_cache", None)
        if aff is None:
            aff = DomainAffinityManager(self._lab().store.dir / "affinity.json")
            self._affinity_cache = aff
        return aff

    def _count_served(self, proxy_url: str) -> None:
        entry = self.state["usage"].setdefault(proxy_url,
                                               {"served": 0, "failures": 0})
        entry["served"] = int(entry.get("served", 0)) + 1
        self.state["current"] = proxy_url
        self.state["stats"]["requests"] += 1
        self._save_state()

    # -- pool ------------------------------------------------------------------
    def _pool_urls(self) -> list[str]:
        """Fresh working proxies (store) + live SSH tunnel SOCKS5 URLs."""
        urls: list[str] = []
        try:
            lab = ProxyLab(self.context)
            for p in lab.store.pool(max_age_hours=self.state["max_age_hours"],
                                    limit=20):
                urls.append(p.url)
        except Exception as exc:  # noqa: BLE001
            _log.warning("rotation pool load failed: %s", exc)
        try:
            from .ssh_socks import SshSocksManager

            urls.extend(SshSocksManager._live_urls())
        except Exception:  # noqa: BLE001 - tunnels are optional
            pass
        seen: dict[str, None] = {}
        for u in urls:
            seen.setdefault(u, None)
        return list(seen)

    def _healthy(self, urls: list[str]) -> list[str]:
        now = time.time()
        cooldown = self.state["cooldown"]
        for u in list(cooldown):
            if float(cooldown[u]) <= now:
                del cooldown[u]
        return [u for u in urls if float(cooldown.get(u, 0)) <= now]

    # -- core action ------------------------------------------------------------
    def _eligible(self, *, schemes: tuple[str, ...] = (),
                  min_anonymity: str = "",
                  countries: set[str] | frozenset[str] | None = None,
                  ) -> list[Proxy]:
        """Fresh, alive, out-of-backoff proxies, protocol/anonymity/
        country-filtered, ranked best-first (score: anonymity → decayed
        speed → failures → NG/EU preference)."""
        try:
            lab = self._lab()
        except Exception as exc:  # noqa: BLE001
            _log.warning("rotation pool load failed: %s", exc)
            return []
        proxies = lab.store.pool(max_age_hours=self.state["max_age_hours"],
                                 limit=200,
                                 prefer_countries=countries)
        out = []
        for p in proxies:
            if schemes and p.scheme not in schemes:
                continue
            if min_anonymity and _ANON_SCORE.get(p.anonymity, 1) < \
                    _ANON_SCORE.get(min_anonymity, 0):
                continue
            out.append(p)
        return out

    def pick(self, *, schemes: tuple[str, ...] = (),
             target: str = "") -> str:
        """The next proxy for this request ('' = direct fallback).

        ``schemes`` restricts the protocol (protocol-aware routing);
        ``target`` lets domain affinity pin one proxy per site.
        """
        st = self.state
        st["stats"]["requests"] += 1
        lab = self._lab()
        if target:
            bound = self._affinity().lookup(target, store=lab.store)
            if bound:
                self._count_served(bound)
                return bound
        healthy = self._healthy(self._pool_urls())
        # protocol-aware filter on the URL list (cheap) …
        if schemes:
            wanted = set(schemes)
            healthy = [u for u in healthy
                       if urlparse(u).scheme in wanted]
        # …but prefer the scored order from the store when we can
        ranked = [p.url for p in self._eligible(schemes=schemes)]
        ordered = [u for u in ranked if u in set(healthy)]
        ordered += [u for u in healthy if u not in set(ranked)]
        if not ordered:
            st["stats"]["direct_fallbacks"] += 1
            self._save_state()
            return ""
        strategy = st.get("strategy", "round_robin")
        usage = st["usage"]
        if strategy == "round_robin":
            st["index"] = (int(st.get("index", 0)) + 1) % len(ordered)
            chosen = ordered[st["index"]]
        elif strategy == "random":
            import random

            chosen = random.choice(ordered)
        elif strategy == "least_used":
            chosen = min(ordered,
                         key=lambda u: int((usage.get(u) or {}).get("served", 0)))
        else:  # sticky
            if st.get("current") in ordered:
                chosen = st["current"]
            else:
                chosen = ordered[0]
        chosen = str(chosen)
        entry = usage.setdefault(chosen, {"served": 0, "failures": 0})
        entry["served"] = int(entry.get("served", 0)) + 1
        st["current"] = chosen
        if target:
            self._affinity().bind(target, chosen)
        self._save_state()
        return chosen

    def pick_for_job(self, *, needs_udp: bool = False,
                     needs_auth: bool = False,
                     country: str = "", min_anonymity: str = "",
                     prefer_ng_eu: bool = True,
                     target: str = "") -> dict[str, Any]:
        """Protocol-aware routing: choose the right proxy TYPE for the
        job, not just the next one in line.

          needs_udp  → SOCKS5 only (HTTP proxies can't do UDP)
          needs_auth → honest answer: the free pool has no authenticated
                       proxies; returns guidance instead of a fake pick
          country / min_anonymity → hard filters
          prefer_ng_eu → NG first, EU second (Nigeria-friendly)
        """
        if needs_auth:
            return {"proxy": "",
                    "note": "the free-proxy pool holds no authenticated "
                            "proxies — add your own via the proxypool "
                            "connector (vault-held credentials) or "
                            "/proxy set socks5://user:pass@host:port"}
        schemes: tuple[str, ...] = ()
        if needs_udp:
            schemes = ("socks5",)
        prefer: set[str] | None = {"NG"} | set(_EU_COUNTRIES) \
            if prefer_ng_eu else None
        try:
            lab = self._lab()
        except Exception as exc:  # noqa: BLE001
            return {"proxy": "",
                    "note": f"pool unavailable: {exc}"}
        proxies = lab.store.pool(
            max_age_hours=self.state.get("max_age_hours", 24.0),
            limit=200, prefer_countries=prefer)
        if schemes:
            proxies = [p for p in proxies if p.scheme in schemes]
        if country:
            proxies = [p for p in proxies
                       if p.country.lower() == country.lower()]
        if min_anonymity:
            floor = _ANON_SCORE.get(min_anonymity, 0)
            proxies = [p for p in proxies
                       if _ANON_SCORE.get(p.anonymity, 1) >= floor]
        if not proxies:
            return {"proxy": "",
                    "note": empty_pool_guidance(
                        lab.store,
                        want=f"scheme={schemes or 'any'} "
                             f"country={country or 'any'} "
                             f"anonymity>={min_anonymity or 'any'}")}
        best = rank_proxies(proxies, prefer_countries=prefer)[0]
        why = (f"{best.scheme} · {best.anonymity or 'unverified'} · "
               f"{round(best.decayed_latency())}ms decayed · "
               f"{best.country or 'country unknown'}")
        if needs_udp:
            why += " · SOCKS5 (UDP-capable)"
        if target:
            self._affinity().bind(target, best.url)
            why += f" · pinned to { _domain_of(target)}"
        return {"proxy": best.url, "why": why,
                "details": best.to_dict()}

    def report_success(self, proxy_url: str,
                       latency_ms: float = 0.0) -> None:
        """A request through ``proxy_url`` worked: sample the latency,
        clear the failure streak.  Feeds the store so the decayed
        average keeps improving with real traffic."""
        proxy_url = (proxy_url or "").strip()
        if not proxy_url:
            return
        try:
            self._lab().store.record_routing_result(
                proxy_url, ok=True, latency_ms=latency_ms)
        except Exception as exc:  # noqa: BLE001
            _log.warning("rotation success record failed: %s", exc)

    def report_error(self, proxy_url: str, reason: str = "") -> None:
        """A request through ``proxy_url`` failed at the connection level.

        The proxy is demoted, not dropped: exponential backoff on the
        store record (300s * 2^streak, capped at 1h) plus the rotation
        cooldown — and the failure REASON is recorded, so the health
        dashboard shows exactly why.  Sticky fails over immediately.
        """
        st = self.state
        proxy_url = (proxy_url or "").strip()
        if not proxy_url:
            return
        entry = st["usage"].setdefault(proxy_url,
                                       {"served": 0, "failures": 0})
        entry["failures"] = int(entry.get("failures", 0)) + 1
        # exponential backoff on the rotation cooldown too: repeated
        # offenders stay out longer (capped at 1h)
        base = max(1.0, float(st.get("cooldown_seconds", 300)))
        cooldown = min(base * (2 ** (entry["failures"] - 1)), 3600.0)
        st["cooldown"][proxy_url] = time.time() + cooldown
        st["stats"]["failovers"] += 1
        if st.get("strategy") == "sticky" and st.get("current") == proxy_url:
            st["current"] = ""
        try:
            self._lab().store.record_routing_result(
                proxy_url, ok=False, reason=reason,
                backoff_seconds=base)
        except Exception as exc:  # noqa: BLE001
            _log.warning("rotation failure record failed: %s", exc)
        _log.info("rotation: %s demoted (backoff %.0fs, streak %d) — %s",
                  proxy_url, cooldown, entry["failures"],
                  (reason or "connection failure")[:120])
        self._save_state()

    def next(self) -> str:
        """Force rotation to the next proxy now (returns the new pick)."""
        return self.pick()

    # -- lifecycle ---------------------------------------------------------------
    def enable(self, *, strategy: str = "round_robin",
               cooldown_seconds: float = 300,
               failover_seconds: float = 60,
               max_age_hours: float = 24.0) -> dict[str, Any]:
        strategy = (strategy or "round_robin").strip().lower()
        if strategy not in _STRATEGIES:
            raise ToolError(
                f"unknown rotation strategy {strategy!r} — "
                f"one of: {', '.join(_STRATEGIES)}")
        st = self.state
        st.update({
            "enabled": True,
            "strategy": strategy,
            "cooldown_seconds": max(1.0, float(cooldown_seconds or 300)),
            "failover_seconds": max(5.0, float(failover_seconds or 60)),
            "max_age_hours": max(0.5, float(max_age_hours or 24.0)),
        })
        if not st["stats"].get("since"):
            st["stats"]["since"] = time.time()
        self._register_hooks()
        self._save_state()
        pool = self._pool_urls()
        note = f"pool has {len(pool)} working proxy(ies)" if pool else (
            "pool is EMPTY — /proxy refresh first; traffic goes direct "
            "until proxies are tested in")
        return {"enabled": True, "strategy": strategy, "pool": len(pool),
                "note": note}

    def disable(self) -> dict[str, Any]:
        self.state["enabled"] = False
        self._unregister_hooks()
        self._save_state()
        return {"enabled": False,
                "stats": self.state["stats"]}

    # -- status --------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        st = self.state
        now = time.time()
        urls = self._pool_urls()
        healthy = self._healthy(urls)
        cooling = [
            {"proxy": u, "seconds_left": max(0, int(float(cool) - now))}
            for u, cool in sorted(st["cooldown"].items(),
                                  key=lambda kv: -float(kv[1]))
            if float(cool) > now
        ]
        usage = [
            {"proxy": u, "served": int(v.get("served", 0)),
             "failures": int(v.get("failures", 0))}
            for u, v in sorted(st["usage"].items(),
                               key=lambda kv: -int(kv[1].get("served", 0)))
        ][:15]
        return {
            "enabled": bool(st.get("enabled")),
            "strategy": st.get("strategy", ""),
            "cooldown_seconds": st.get("cooldown_seconds", 300),
            "pool_size": len(urls),
            "healthy": len(healthy),
            "cooling": cooling,
            "current": st.get("current", ""),
            "usage": usage,
            "stats": st.get("stats", {}),
            "direct_routing_when": ("disabled" if st.get("enabled")
                                    else "rotation is off"),
        }


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "proxy_scrape",
        description=(
            "Scrape free public proxies from multiple live sources "
            "(HTTP/HTTPS/SOCKS4/SOCKS5 lists + proxy-list pages), concurrent, "
            "deduped. Saves the candidate set for proxy_test. Returns counts "
            "per source plus per-source errors."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "schemes": "str (optional) — comma list, default all four",
            "limit": "int (optional, 2000) — max unique proxies to keep",
        },
    )
    def proxy_scrape(schemes: str = "", *, limit: str = "") -> dict[str, Any]:
        lab = ProxyLab(context)
        try:
            cap = max(10, min(int(limit or 2000), 10000))
        except ValueError:
            cap = 2000
        report = lab.scrape(schemes or "http,https,socks4,socks5", limit=cap)
        return {
            "total": report["total"],
            "by_source": report["by_source"],
            "errors": report["errors"],
            "seconds": report["seconds"],
            "sample": [p.url for p in report["proxies"][:15]],
        }

    @registry.register(
        "proxy_refresh",
        description=(
            "Test the stored proxy candidates for real (liveness, latency, "
            "egress IP, anonymity class, country — SOCKS4/5 included) and "
            "store the working pool. Use proxy_pool to get them afterwards. "
            "Scrapes first when no candidates are stored yet."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "limit": "int (optional, 150) — how many candidates to test",
            "schemes": "str (optional) — for the scrape fallback",
            "detect_country": "bool (optional, true) — per-proxy country probe",
        },
    )
    def proxy_refresh(*, limit: str = "", schemes: str = "",
                      detect_country: str = "") -> dict[str, Any]:
        lab = ProxyLab(context)
        try:
            cap = max(1, min(int(limit or 150), 500))
        except ValueError:
            cap = 150
        do_country = str(detect_country or "true").lower() \
            not in {"0", "false", "no", "off"}
        return lab.refresh(schemes or "http,https,socks4,socks5",
                           limit=cap, detect_country=do_country)

    @registry.register(
        "proxy_pool",
        description=(
            "Working proxies on demand: ranked (alive → anonymity → speed), "
            "TTL-filtered, deduped. action=list returns records, action=urls "
            "returns scheme://host:port strings ready for proxy_set / the "
            "browser / OSINT. action=clear wipes the pool."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — list | urls | clear (default list)",
            "scheme": "str (optional) — http | https | socks4 | socks5",
            "country": "str (optional, 2-letter code)",
            "anonymity": "str (optional) — elite | anonymous | transparent",
            "max_age_hours": "float (optional, 24)",
            "limit": "int (optional, 10)",
        },
    )
    def proxy_pool(*, action: str = "list", scheme: str = "", country: str = "",
                   anonymity: str = "", max_age_hours: str = "",
                   limit: str = "") -> dict[str, Any]:
        lab = ProxyLab(context)
        action = (action or "list").strip().lower()
        if action == "clear":
            lab.store.clear()
            return {"cleared": True}
        try:
            age = float(max_age_hours or 24)
        except ValueError:
            age = 24.0
        try:
            cap = max(1, min(int(limit or 10), 100))
        except ValueError:
            cap = 10
        proxies = lab.store.pool(scheme=scheme, country=country,
                                 anonymity=anonymity, max_age_hours=age,
                                 limit=cap)
        if action == "urls":
            return {"urls": [p.url for p in proxies], "count": len(proxies)}
        return {"proxies": [p.to_dict() for p in proxies],
                "count": len(proxies)}

    @registry.register(
        "proxy_lab_status",
        description=(
            "Proxy lab state: working pool size, freshness, per-scheme "
            "counts, best-5 list, store location. (Routing/active-proxy "
            "state is the proxy_status in tools/proxy.py.)"
        ),
        capability=Capability.FS_READ,
    )
    def proxy_lab_status() -> dict[str, Any]:
        return ProxyLab(context).status()

    @registry.register(
        "proxy_health",
        description=(
            "Proxy pool health dashboard: routable pool size by protocol / "
            "anonymity / country, decayed-latency average, last refresh, "
            "dead + backing-off counts, rotation state, per-domain "
            "affinity bindings, and the auto-refresh schedule. When the "
            "pool is empty it says exactly why and what to run."
        ),
        capability=Capability.FS_READ,
        parameters={},
    )
    def proxy_health() -> dict[str, Any]:
        return ProxyLab(context).health()

    @registry.register(
        "proxy_pick",
        description=(
            "Protocol-aware proxy pick: choose the right proxy TYPE for "
            "the job. needs_udp=true → SOCKS5 only (HTTP proxies can't do "
            "UDP); country/min_anonymity filter hard; NG/EU preferred by "
            "default (Nigeria-friendly). Empty pool → honest guidance, "
            "never a fake pick."
        ),
        capability=Capability.FS_READ,
        parameters={
            "needs_udp": "bool (optional) — UDP-capable proxy required",
            "needs_auth": "bool (optional) — authenticated proxy required "
                          "(the free pool has none; says so honestly)",
            "country": "str (optional) — 2-letter code",
            "min_anonymity": "str (optional) — anonymous | elite",
            "prefer_ng_eu": "bool (optional, true) — Nigeria-friendly",
            "target": "str (optional) — target URL; pins the pick to the "
                      "domain (per-domain affinity)",
        },
    )
    def proxy_pick(*, needs_udp: str = "", needs_auth: str = "",
                   country: str = "", min_anonymity: str = "",
                   prefer_ng_eu: str = "", target: str = "") -> dict[str, Any]:
        manager = ProxyRotationManager(context)
        flag = lambda v, d=False: str(v or ("true" if d else "")).lower() \
            not in {"", "0", "false", "no", "off"}
        return manager.pick_for_job(
            needs_udp=flag(needs_udp),
            needs_auth=flag(needs_auth),
            country=(country or "").strip(),
            min_anonymity=(min_anonymity or "").strip().lower(),
            prefer_ng_eu=flag(prefer_ng_eu, True),
            target=(target or "").strip(),
        )

    @registry.register(
        "proxy_affinity",
        description=(
            "Per-domain proxy affinity: pin one proxy per site so "
            "sessions, logins, and rate-limit buckets survive rotation. "
            "bind <domain> <proxy_url> | lookup <domain> | release "
            "<domain> | list — bindings expire (TTL, default 6h) and drop "
            "automatically when the proxy dies or backs off."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — bind | lookup | release | list (default list)",
            "domain": "str (bind/lookup/release) — site domain or URL",
            "proxy_url": "str (bind) — scheme://host:port",
            "ttl_hours": "float (bind, optional, 6)",
        },
    )
    def proxy_affinity(*, action: str = "list", domain: str = "",
                       proxy_url: str = "", ttl_hours: str = "") -> dict[str, Any]:
        lab = ProxyLab(context)
        aff = DomainAffinityManager(lab.store.dir / "affinity.json")
        action = (action or "list").strip().lower()
        if action == "bind":
            try:
                ttl = float(ttl_hours or 6)
            except ValueError:
                ttl = 6.0
            bound = aff.bind(domain, proxy_url, ttl_hours=ttl)
            return {"bound": True, **bound}
        if action == "lookup":
            url = aff.lookup(domain, store=lab.store)
            return {"domain": _domain_of(domain), "proxy_url": url,
                    "note": "" if url else
                    "no live binding — use proxy_pick with target= to get "
                    "and pin one"}
        if action == "release":
            return {"released": aff.release(domain),
                    "domain": _domain_of(domain)}
        bindings = aff.bindings()
        return {"bindings": bindings, "count": len(bindings)}

    @registry.register(
        "proxy_schedule",
        description=(
            "Continuous/scheduled proxy re-checks through the normal "
            "scheduler. enabled=true registers a proxy_refresh job that "
            "re-tests the stored candidate pool every N minutes (so the "
            "working pool never silently rots); enabled=false removes it."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "enabled": "bool (default true) — register or remove the job",
            "every_minutes": "int (optional, 30) — re-check interval",
        },
    )
    def proxy_schedule(*, enabled: str = "true",
                       every_minutes: str = "") -> dict[str, Any]:
        from ..agents.scheduler import Scheduler

        want = str(enabled or "true").lower() not in {"0", "false", "no", "off"}
        try:
            every = max(5, min(int(every_minutes or 30), 1440))
        except ValueError:
            every = 30
        scheduler = Scheduler(context)
        job_name = "proxy_refresh"
        jobs = scheduler.list_jobs(include_disabled=True)
        existing = next((j for j in jobs if j.get("name") == job_name), None)
        if want:
            if existing:
                return {"scheduled": True, "id": existing["id"],
                        "note": "already scheduled"}
            created = scheduler.add(
                job_name, f"every {every}m", "tool",
                {"tool": "proxy_refresh", "limit": "150"})
            return {"scheduled": True, "id": created["id"],
                    "every_minutes": every}
        if existing:
            scheduler.remove(existing["id"])
            return {"scheduled": False, "removed": existing["id"]}
        return {"scheduled": False, "note": "no proxy_refresh job existed"}

    @registry.register(
        "proxy_file",
        description=(
            "Write a CLEAN file of the currently working proxies — one "
            "scheme://host:port per line, dead and stale entries dropped, "
            "fastest first (refreshes from the live pool, never a stale "
            "copy). Returns the file path + count. The chat layer can send "
            "this file to the owner directly (/proxy file)."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "limit": "int (optional, 1000) — max proxies in the file",
            "max_age_hours": "float (optional, 24) — drop proxies tested older than this",
        },
    )
    def proxy_file(*, limit: str = "", max_age_hours: str = "") -> dict[str, Any]:
        lab = ProxyLab(context)
        try:
            cap = max(1, min(int(limit or 1000), 10000))
        except ValueError:
            cap = 1000
        try:
            age = float(max_age_hours or 24)
        except ValueError:
            age = 24.0
        return lab.active_file(limit=cap, max_age_hours=age)

    @registry.register(
        "proxy_rotate",
        description=(
            "Proxy rotation: cycle the working pool through outbound "
            "requests (every HttpClient without an explicit proxy). "
            "start [strategy]: round_robin | random | sticky | least_used, "
            "with failover — a proxy that fails a connection is put on "
            "cooldown and the next request takes the next healthy proxy. "
            "stop: route via the static proxy_set default again. "
            "status: strategy, pool/healthy sizes, cooldowns, per-proxy "
            "served/failed counts. next: rotate immediately."
        ),
        capability=Capability.SYS_CONFIG,
        parameters={
            "action": "str — start | stop | status | next (default status)",
            "strategy": "str (start) — round_robin | random | sticky | least_used",
            "cooldown_seconds": "float (start, optional, 300) — failed-proxy cooldown",
            "failover_seconds": "float (start, optional, 60) — sticky failover window",
            "max_age_hours": "float (start, optional, 24) — max proxy age in the pool",
        },
    )
    def proxy_rotate(*, action: str = "status", strategy: str = "",
                     cooldown_seconds: str = "", failover_seconds: str = "",
                     max_age_hours: str = "") -> dict[str, Any]:
        manager = ProxyRotationManager(context)
        action = (action or "status").strip().lower()
        if action == "start":
            try:
                cooldown = float(cooldown_seconds or 300)
            except ValueError:
                cooldown = 300.0
            try:
                failover = float(failover_seconds or 60)
            except ValueError:
                failover = 60.0
            try:
                age = float(max_age_hours or 24)
            except ValueError:
                age = 24.0
            return manager.enable(strategy=strategy,
                                  cooldown_seconds=cooldown,
                                  failover_seconds=failover,
                                  max_age_hours=age)
        if action == "stop":
            return manager.disable()
        if action == "next":
            if not manager.state.get("enabled"):
                return {"rotated": False,
                        "note": "rotation is off — proxy_rotate start first"}
            return {"rotated": True, "proxy": manager.next()}
        return manager.status()

    @registry.register(
        "proxy_discover",
        description=(
            "Wave 83: learn NEW proxy-list sources from the internet at "
            "large — not just the built-in catalog. Mines aggregator pages "
            "(github topics, list directories) for proxy-list repos and .txt "
            "endpoints, fetches + parses every candidate, and registers the "
            "ones that actually yield proxies. They join every future "
            "scrape. Pass seeds to mine specific pages."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "seeds": "str (optional) — space/comma-separated seed page URLs",
            "min_proxies": "int (optional, 5) — a source must yield at "
                           "least this many proxies to be kept",
        },
    )
    def proxy_discover(*, seeds: str = "", min_proxies: str = "") -> dict[str, Any]:
        lab = ProxyLab(context)
        seed_list = [s.strip() for s in re.split(r"[,\s]+", seeds) if s.strip()]
        try:
            floor = max(1, min(int(min_proxies or 5), 100))
        except ValueError:
            floor = 5
        report = lab.discover_sources(seed_list or None, min_proxies=floor)
        return {
            "seeds": report.get("seeds", []),
            "seed_errors": report.get("seed_errors", {}),
            "endpoints_tested": report.get("endpoints_tested", 0),
            "registered": report.get("registered", []),
            "source_stats": report.get("source_stats", {}),
        }

    @registry.register(
        "proxy_sources",
        description=(
            "Health of every proxy source the scraper uses: built-in + "
            "discovered, with test counts, consecutive failures, last "
            "harvest size, and disabled sources (3 straight failures "
            "retire a source for 24h; a good harvest reinstates it)."
        ),
        capability=Capability.FS_WRITE,
        parameters={},
    )
    def proxy_sources() -> dict[str, Any]:
        lab = ProxyLab(context)
        return lab.source_health()

    # boot: if rotation was enabled before a restart, re-arm the hooks so
    # routing picks up where it left off (state lives in kv_store).
    try:
        ProxyRotationManager(context)
    except Exception as exc:  # noqa: BLE001 - never break tool registration
        _log.warning("rotation boot re-arm failed: %s", exc)
