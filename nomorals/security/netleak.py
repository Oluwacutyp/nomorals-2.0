"""Non-DNS leak surface — WebRTC-style and IPv6 checks.

The standard leak-test trio (ipleak.net, mullvad, browserleaks) is
**DNS, WebRTC, IPv6**; ``dnsleak.py`` covers DNS. This module covers the
other two from pure stdlib, no browser needed:

**WebRTC-style check.** A browser leak test gathers ICE candidates:
``host`` (local interface IPs), ``srflx`` (public IP discovered via STUN),
``relay`` (TURN). The ``srflx`` half is reproducible exactly: send an
RFC 5389 STUN Binding Request over UDP (STUN uses UDP — which is why it
bypasses TCP-only SOCKS proxies and exposes the real interface) to
``stun.l.google.com:19302`` and read the XOR-MAPPED-ADDRESS. If that IP
differs from the HTTP egress IP, traffic is leaving via two paths — the
same verdict a browser test gives.

**Host candidates.** Enumerate local interface IPs (what WebRTC would
offer as ``host`` candidates) and classify them (RFC1918/loopback/
link-local). Informational: modern browsers mask these as mDNS ``.local``,
older ones do not.

**IPv6.** If the VPN tunnels IPv4 only, a live IPv6 egress leaks everything.
Detected via ``api64.ipify.org`` plus a local interface scan.

``check_net_leak`` returns a report, never raises.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import struct
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "NetLeakReport",
    "check_net_leak",
    "stun_public_ip",
    "local_interface_ips",
    "ipv6_egress",
    "format_report",
]

_UA = {"User-Agent": "Mozilla/5.0"}

# RFC 5389 §6: Binding Request = 0x0001, Binding Success Response = 0x0101,
# magic cookie 0x2112A442, 96-bit transaction id.
_STUN_BINDING_REQUEST = 0x0001
_STUN_BINDING_RESPONSE = 0x0101
_STUN_MAGIC_COOKIE = 0x2112A442
_ATTR_XOR_MAPPED_ADDRESS = 0x0020
_ATTR_MAPPED_ADDRESS = 0x0001

_DEFAULT_STUN_SERVERS = [
    ("stun.l.google.com", 19302),   # what the mined tools use
    ("stun1.l.google.com", 19302),
]


def _http_text(url: str, timeout: float = 10) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace").strip()


# ── stdlib STUN client ───────────────────────────────────────────────────

def _build_binding_request() -> tuple[bytes, bytes]:
    txn_id = os.urandom(12)
    header = struct.pack("!HHI12s", _STUN_BINDING_REQUEST, 0,
                         _STUN_MAGIC_COOKIE, txn_id)
    return header, txn_id


def _parse_binding_response(data: bytes, txn_id: bytes) -> str:
    """Extract the reflexive address from a Binding Success Response."""
    if len(data) < 20:
        raise ValueError("STUN response too short")
    msg_type, msg_len, cookie, rx_txn = struct.unpack("!HHI12s", data[:20])
    if msg_type != _STUN_BINDING_RESPONSE:
        raise ValueError(f"not a Binding Success Response: {msg_type:#06x}")
    if cookie != _STUN_MAGIC_COOKIE or rx_txn != txn_id:
        raise ValueError("STUN cookie/transaction mismatch (spoofed?)")
    pos = 20
    end = 20 + msg_len
    while pos + 4 <= min(end, len(data)):
        attr_type, attr_len = struct.unpack("!HH", data[pos:pos + 4])
        val = data[pos + 4:pos + 4 + attr_len]
        if attr_type == _ATTR_XOR_MAPPED_ADDRESS and len(val) >= 8:
            family = val[1]
            xport = struct.unpack("!H", val[2:4])[0]
            port = xport ^ (_STUN_MAGIC_COOKIE >> 16)
            if family == 0x01 and len(val) >= 8:  # IPv4
                xaddr = struct.unpack("!I", val[4:8])[0]
                addr = xaddr ^ _STUN_MAGIC_COOKIE
                ip = socket.inet_ntoa(struct.pack("!I", addr))
                return ip
            # IPv6 XOR uses cookie+txn_id; report unsupported honestly
            raise ValueError("IPv6 XOR-MAPPED-ADDRESS not decoded")
        if attr_type == _ATTR_MAPPED_ADDRESS and len(val) >= 8:
            family = val[1]
            if family == 0x01:
                port = struct.unpack("!H", val[2:4])[0]
                return socket.inet_ntoa(val[4:8])
        pos += 4 + attr_len + (attr_len % 2)  # attributes are 32-bit padded
    raise ValueError("no MAPPED-ADDRESS in STUN response")


def stun_public_ip(server: str = "stun.l.google.com", port: int = 19302,
                   timeout: float = 3.0, attempts: int = 2) -> str:
    """RFC 5389 Binding → server-reflexive IP. Empty string on failure."""
    try:
        infos = socket.getaddrinfo(server, port, socket.AF_INET,
                                   socket.SOCK_DGRAM)
    except OSError as exc:
        _log.debug("STUN dns failed for %s: %s", server, exc)
        return ""
    if not infos:
        return ""
    req, txn_id = _build_binding_request()
    for attempt in range(attempts):
        for fam, _, _, _, sockaddr in infos:
            sock = None
            try:
                sock = socket.socket(fam, socket.SOCK_DGRAM)
                # RFC 5389 RTO: start 500ms, double each retransmit
                sock.settimeout(min(timeout, 0.5 * (2 ** attempt)))
                sock.sendto(req, sockaddr)
                data, _ = sock.recvfrom(2048)
                return _parse_binding_response(data, txn_id)
            except (OSError, ValueError) as exc:
                _log.debug("STUN attempt %d via %s failed: %s",
                           attempt, sockaddr, exc)
            finally:
                if sock is not None:
                    sock.close()
    return ""


# ── local interface enumeration (host-candidate equivalent) ──────────────

def _classify(ip: str) -> str:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return "invalid"
    if a.is_loopback:
        return "loopback"
    if a.is_link_local:
        return "link-local"
    if isinstance(a, ipaddress.IPv4Address) and a.is_private:
        return "rfc1918-private"
    if isinstance(a, ipaddress.IPv6Address) and a.is_private:
        return "unique-local"
    if a.is_multicast:
        return "multicast"
    return "global"


def local_interface_ips() -> list[dict[str, str]]:
    """Enumerate local interface IPs (what WebRTC offers as host candidates)."""
    found: dict[str, str] = {}
    # 1) hostname resolution
    try:
        for fam, _, _, _, sockaddr in socket.getaddrinfo(
                socket.gethostname(), None):
            ip = sockaddr[0]
            if "%" in ip:  # strip IPv6 zone id
                ip = ip.split("%")[0]
            found.setdefault(ip, _classify(ip))
    except OSError:
        pass
    # 2) default-route source address (UDP connect sends nothing)
    for remote in (("8.8.8.8", 53), ("2001:4860:4860::8888", 53)):
        sock = None
        try:
            fam = socket.AF_INET6 if ":" in remote[0] else socket.AF_INET
            sock = socket.socket(fam, socket.SOCK_DGRAM)
            sock.settimeout(2)
            sock.connect(remote)
            ip = sock.getsockname()[0].split("%")[0]
            found.setdefault(ip, _classify(ip))
        except OSError:
            pass
        finally:
            if sock is not None:
                sock.close()
    return [{"ip": ip, "class": cls} for ip, cls in sorted(found.items())]


def ipv6_egress() -> dict[str, Any]:
    """IPv6 leak surface: egress address (if any) + local IPv6 interfaces."""
    out: dict[str, Any] = {"egress": "", "local": [], "supported": False}
    try:
        if socket.has_ipv6:
            out["supported"] = True
    except AttributeError:
        pass
    try:
        ip = _http_text("https://api64.ipify.org", timeout=10)
        if ":" in ip:
            out["egress"] = ip
    except Exception as exc:  # noqa: BLE001
        _log.debug("ipv6 egress check failed: %s", exc)
    out["local"] = [e["ip"] for e in local_interface_ips()
                    if ":" in e["ip"] and e["class"] not in ("loopback",)]
    return out


# ── report ───────────────────────────────────────────────────────────────

@dataclass
class NetLeakReport:
    """Result of the WebRTC-style + IPv6 leak check."""
    ok: bool = False
    http_public_ip: str = ""
    stun_ip: str = ""
    stun_server: str = ""
    local_ips: list[dict[str, str]] = field(default_factory=list)
    ipv6_egress: str = ""
    ipv6_local: list[str] = field(default_factory=list)
    webrtc_leak: bool = False           # srflx IP != HTTP egress IP
    webrtc_detail: str = ""
    ipv6_leak: bool = False             # live IPv6 egress (advisory)
    local_exposure: bool = False        # private IPs enumerable
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "http_public_ip": self.http_public_ip,
            "stun_ip": self.stun_ip, "stun_server": self.stun_server,
            "local_ips": self.local_ips, "ipv6_egress": self.ipv6_egress,
            "ipv6_local": self.ipv6_local, "webrtc_leak": self.webrtc_leak,
            "webrtc_detail": self.webrtc_detail, "ipv6_leak": self.ipv6_leak,
            "local_exposure": self.local_exposure, "note": self.note,
        }


def check_net_leak(*, expected_ip: str = "",
                   stun_servers: list[tuple[str, int]] | None = None
                   ) -> NetLeakReport:
    """Run the WebRTC-style (STUN) + IPv6 leak check. Never raises."""
    rep = NetLeakReport()
    try:
        return _check(rep, expected_ip=expected_ip,
                      stun_servers=stun_servers or _DEFAULT_STUN_SERVERS)
    except Exception as exc:  # noqa: BLE001
        _log.warning("net leak check failed: %s", exc)
        rep.note = f"check failed: {exc}"
        return rep


def _check(rep: NetLeakReport, *, expected_ip: str,
           stun_servers: list[tuple[str, int]]) -> NetLeakReport:
    try:
        rep.http_public_ip = _http_text("https://api.ipify.org", timeout=10)
    except Exception as exc:  # noqa: BLE001
        _log.debug("public ip check failed: %s", exc)
    for server, port in stun_servers:
        ip = stun_public_ip(server, port)
        if ip:
            rep.stun_ip = ip
            rep.stun_server = f"{server}:{port}"
            break
    rep.local_ips = local_interface_ips()
    v6 = ipv6_egress()
    rep.ipv6_egress = v6["egress"]
    rep.ipv6_local = v6["local"]

    rep.local_exposure = any(e["class"] in ("rfc1918-private", "unique-local",
                                            "link-local")
                             for e in rep.local_ips)

    # ── verdicts ─────────────────────────────────────────────────────────
    if rep.stun_ip and rep.http_public_ip:
        if expected_ip:
            # strict: STUN must see the expected egress
            if rep.stun_ip != expected_ip:
                rep.webrtc_leak = True
                rep.webrtc_detail = (
                    f"STUN sees {rep.stun_ip} but expected egress is "
                    f"{expected_ip} — UDP bypasses the tunnel")
        elif rep.stun_ip != rep.http_public_ip:
            rep.webrtc_leak = True
            rep.webrtc_detail = (
                f"STUN-discovered IP {rep.stun_ip} differs from HTTP egress "
                f"{rep.http_public_ip} — UDP traffic (WebRTC/STUN path) "
                "leaves via a different route")
    elif not rep.stun_ip:
        rep.note = ((rep.note + " " if rep.note else "")
                    + "STUN unreachable — WebRTC-style verdict unavailable "
                      "(UDP may be blocked; treat as unknown, not clean).")

    # IPv6 egress is advisory: only a leak if the tunnel is IPv4-only.
    # We cannot know the tunnel's intent, so report, don't convict —
    # unless it differs from an expected (tunnel) egress.
    if rep.ipv6_egress:
        rep.ipv6_leak = True  # surfaced as advisory in format_report
        if expected_ip and ":" in expected_ip \
                and rep.ipv6_egress != expected_ip:
            rep.note = ((rep.note + " " if rep.note else "")
                        + f"IPv6 egress {rep.ipv6_egress} differs from "
                          f"expected {expected_ip}.")

    rep.ok = not rep.webrtc_leak and bool(rep.stun_ip or rep.http_public_ip)
    rep.note = (rep.note.strip() + " " if rep.note else "") + (
        "Python STUN mirrors the srflx half of a browser WebRTC test "
        "(RFC 5389); it cannot see browser-level mDNS masking of host "
        "candidates — run a browser test (ipleak.net) for the full picture.")
    rep.note = rep.note.strip()
    return rep


def format_report(rep: NetLeakReport) -> str:
    """Human-readable rendering (chat/terminal)."""
    L: list[str] = []
    verdict = ("✅ NO LEAK" if rep.ok and not rep.webrtc_leak
               else "🚨 LEAK DETECTED" if rep.webrtc_leak
               else "⚠️ INCONCLUSIVE")
    L.append(f"🌐 WebRTC/IPv6 leak check — {verdict}")
    if rep.http_public_ip:
        L.append(f"   HTTP egress: {rep.http_public_ip}")
    if rep.stun_ip:
        L.append(f"   STUN srflx:  {rep.stun_ip}  (via {rep.stun_server})")
        L.append("   " + ("✅ STUN path matches HTTP egress — no WebRTC-style leak"
                          if not rep.webrtc_leak else
                          f"   🚨 {rep.webrtc_detail}"))
    else:
        L.append("   STUN srflx:  unreachable (UDP blocked?)")
    if rep.local_ips:
        priv = [e["ip"] for e in rep.local_ips
                if e["class"] in ("rfc1918-private", "unique-local")]
        if priv:
            L.append("   host candidates (private, WebRTC-visible): "
                     + ", ".join(priv))
    if rep.ipv6_egress:
        L.append(f"   ⚠️ IPv6 egress LIVE: {rep.ipv6_egress} — confirm it is "
                 "inside the tunnel, else this leaks everything")
    elif rep.ipv6_local:
        L.append("   IPv6: no public egress, local addresses: "
                 + ", ".join(rep.ipv6_local))
    else:
        L.append("   IPv6: no egress detected")
    if rep.note:
        L.append(f"   note: {rep.note}")
    return "\n".join(L)
