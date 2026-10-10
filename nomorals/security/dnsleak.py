"""DNS leak detection — does the machine's DNS bypass the VPN/tunnel?

Technique (from dnsleaktest.com / macvk/dnsleaktest / dns-leak-detector-cli):
1. Generate a unique nonce subdomain.
2. Force the OS to resolve it via ``socket.getaddrinfo()`` — this goes
   through whatever DNS the system is actually using.
3. Ask the open bash.ws leak-detection endpoint which resolver IP(s)
   asked its authoritative server for that nonce.
4. Compare resolver IPs/ASNs against the public egress IP: if resolvers
   belong to the ISP while traffic should be tunneled → leak.

Pure stdlib (socket + urllib + json). No API key. If bash.ws is
unreachable the module falls back to a local heuristic (configured
resolvers from /etc/resolv.conf vs known-public resolvers) and says
so honestly.
"""

from __future__ import annotations

import json
import socket
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["DnsLeakReport", "check_dns_leak"]

_UA = {"User-Agent": "Mozilla/5.0"}


def _http_json(url: str, timeout: float = 15) -> Any:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_text(url: str, timeout: float = 15) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace").strip()


#: well-known public resolvers — seeing ONLY these is not a leak
_PUBLIC_RESOLVERS = {
    "1.1.1.1", "1.0.0.1",           # Cloudflare
    "8.8.8.8", "8.8.4.4",           # Google
    "9.9.9.9", "149.112.112.112",   # Quad9
    "208.67.222.222", "208.67.220.220",  # OpenDNS
}


def _public_ip() -> str:
    for url in ("https://api.ipify.org", "https://api64.ipify.org"):
        try:
            return _http_text(url, timeout=10)
        except Exception:  # noqa: BLE001
            continue
    return ""


def _ip_info(ip: str) -> dict[str, Any]:
    try:
        data = _http_json(f"https://ipwho.is/{ip}", timeout=10)
        if isinstance(data, dict) and data.get("success"):
            conn = data.get("connection", {})
            return {"asn": conn.get("asn"), "org": conn.get("org"),
                    "isp": conn.get("isp"), "country": data.get("country")}
    except Exception as exc:  # noqa: BLE001
        _log.debug("ipwho.is failed: %s", exc)
    return {}


def _configured_resolvers() -> list[str]:
    """Parse /etc/resolv.conf for configured nameservers."""
    out: list[str] = []
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    out.append(parts[1])
    except OSError:
        pass
    return out


@dataclass
class DnsLeakReport:
    """Result of a DNS leak check."""
    ok: bool = False                    # True = no leak detected
    method: str = ""                    # "bash.ws" or "local-heuristic"
    public_ip: str = ""
    public_asn: str = ""
    resolvers: list[str] = field(default_factory=list)
    resolver_info: list[dict[str, Any]] = field(default_factory=list)
    leak: bool = False
    leak_detail: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "method": self.method,
            "public_ip": self.public_ip, "public_asn": self.public_asn,
            "resolvers": self.resolvers, "resolver_info": self.resolver_info,
            "leak": self.leak, "leak_detail": self.leak_detail,
            "note": self.note,
        }


def check_dns_leak(*, expected_asn: str = "") -> DnsLeakReport:
    """Run a DNS leak check. Returns a report, never raises."""
    rep = DnsLeakReport()
    try:
        return _check(rep, expected_asn=expected_asn)
    except Exception as exc:  # noqa: BLE001
        _log.warning("dns leak check failed: %s", exc)
        rep.note = f"check failed: {exc}"
        return rep


def _check(rep: DnsLeakReport, *, expected_asn: str) -> DnsLeakReport:
    rep.public_ip = _public_ip()
    if rep.public_ip:
        info = _ip_info(rep.public_ip)
        rep.public_asn = str(info.get("asn") or "")

    # ── primary: bash.ws nonce technique ─────────────────────────────
    nonce = uuid.uuid4().hex[:12]
    probe_host = f"{nonce}.bash.ws"
    try:
        socket.getaddrinfo(probe_host, 80)
    except OSError:
        pass  # resolution "failure" is fine — the query still went out
    try:
        data = _http_json(f"https://bash.ws/dnsleak/test/{nonce}", timeout=15)
        resolvers = _extract_bashws_resolvers(data)
        if resolvers:
            rep.method = "bash.ws"
            rep.resolvers = resolvers
            return _assess(rep, expected_asn)
    except Exception as exc:  # noqa: BLE001
        _log.debug("bash.ws probe failed: %s", exc)

    # ── fallback: local heuristic ────────────────────────────────────
    rep.method = "local-heuristic"
    rep.resolvers = _configured_resolvers()
    rep.note = ("bash.ws unreachable — reporting configured resolvers only. "
                "This cannot prove which resolver answered; treat as advisory.")
    return _assess(rep, expected_asn)


def _extract_bashws_resolvers(data: Any) -> list[str]:
    """Pull resolver IPs out of a bash.ws response (shape-tolerant)."""
    out: list[str] = []
    if isinstance(data, dict):
        for key in ("resolvers", "dns", "servers", "ips"):
            val = data.get(key)
            if isinstance(val, list):
                out.extend(str(x) for x in val if x)
        # some shapes nest: {"test": {"resolvers": [...]}}
        for val in data.values():
            if isinstance(val, dict):
                out.extend(_extract_bashws_resolvers(val))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item:
                out.append(item)
            elif isinstance(item, dict):
                ip = item.get("ip") or item.get("address")
                if ip:
                    out.append(str(ip))
    # dedupe, keep order
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def _assess(rep: DnsLeakReport, expected_asn: str) -> DnsLeakReport:
    for r in rep.resolvers:
        info = _ip_info(r)
        rep.resolver_info.append({"ip": r, **info})
    if not rep.resolvers:
        rep.note = (rep.note + " " if rep.note else "") + "no resolvers detected."
        rep.ok = False
        return rep
    # leak logic
    if expected_asn:
        # strict mode: every resolver must belong to the expected ASN
        bad = [ri for ri in rep.resolver_info
               if str(ri.get("asn") or "") != expected_asn]
        if bad:
            rep.leak = True
            rep.leak_detail = ("resolvers outside expected ASN "
                               f"{expected_asn}: " + ", ".join(b["ip"] for b in bad))
    else:
        # heuristic mode: leak if a resolver is NOT a known public resolver
        # AND its ASN differs from the egress ASN (ISP resolver leaking
        # through the tunnel)
        bad = []
        for ri in rep.resolver_info:
            ip = ri["ip"]
            asn = str(ri.get("asn") or "")
            if ip in _PUBLIC_RESOLVERS:
                continue
            if rep.public_asn and asn and asn != rep.public_asn:
                bad.append(ri)
        if bad:
            rep.leak = True
            rep.leak_detail = ("possible ISP resolvers bypassing tunnel: "
                               + ", ".join(f"{b['ip']} ({b.get('org', '?')})" for b in bad))
    rep.ok = not rep.leak
    if not rep.leak and not rep.leak_detail:
        rep.leak_detail = "all detected resolvers look consistent — no leak."
    return rep
