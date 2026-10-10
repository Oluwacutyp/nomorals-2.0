"""DNS leak detection — does the machine's DNS bypass the VPN/tunnel?

Technique (macvk/dnsleaktest, the reference implementation):
1. ``GET https://bash.ws/id`` — the server issues a leak-test id.
2. Resolve N hosts ``1..<id>.bash.ws`` … ``N..<id>.bash.ws`` **in parallel**
   via ``socket.getaddrinfo()`` — these go through whatever DNS the system
   actually uses. Many probes catch resolvers that round-robin.
3. ``GET https://bash.ws/dnsleak/test/<id>?json`` — the server reports which
   resolver IP(s) asked its authoritative server for those names. Items are
   typed: ``"ip"`` (egress IP), ``"dns"`` (a resolver), ``"conclusion"``
   (server-side verdict); resolvers already carry ``country_name``/``asn``.
4. Compare resolver IPs/ASNs against the public egress IP, the *configured*
   resolvers (transparent-proxy detection), and the expected ASN.

Pure stdlib (socket + urllib + json + concurrent.futures). No API key.
If bash.ws is unreachable/rate-limited the module falls back to a local
heuristic (OS-configured resolvers vs known-public resolvers) and says
so honestly. ``check_dns_leak`` returns a report, never raises.
"""

from __future__ import annotations

import concurrent.futures
import json
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "DnsLeakReport",
    "check_dns_leak",
    "configured_resolvers",
    "detect_smhnr_risk",
    "format_report",
]

_UA = {"User-Agent": "Mozilla/5.0"}

_BASH_WS = "https://bash.ws"


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

#: systemd-resolved stub — not a real upstream resolver
_STUB_RESOLVERS = {"127.0.0.53", "127.0.0.54", "::1", "127.0.0.1"}


def _public_ip() -> str:
    for url in ("https://api.ipify.org", "https://api64.ipify.org"):
        try:
            return _http_text(url, timeout=10)
        except Exception:  # noqa: BLE001
            continue
    return ""


def _public_ipv6() -> str:
    """Best-effort IPv6 egress address (empty string if none/unreachable)."""
    try:
        ip = _http_text("https://api64.ipify.org", timeout=10)
        return ip if ":" in ip else ""
    except Exception:  # noqa: BLE001
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


# ── per-OS configured resolvers ──────────────────────────────────────────

def _resolvers_linux() -> list[str]:
    out: list[str] = []
    # systemd-resolved knows the real upstreams (not the 127.0.0.53 stub)
    if shutil.which("resolvectl"):
        try:
            txt = subprocess.run(
                ["resolvectl", "status"], capture_output=True, text=True,
                timeout=8).stdout
            for m in re.finditer(r"DNS Servers?:\s*([0-9a-fA-F.: ]+)", txt):
                out.extend(s for s in m.group(1).split() if s)
        except Exception:  # noqa: BLE001
            pass
    if not out and shutil.which("nmcli"):
        try:
            txt = subprocess.run(
                ["nmcli", "dev", "show"], capture_output=True, text=True,
                timeout=8).stdout
            for m in re.finditer(r"IP4\.DNS\[\d+\]:\s*(\S+)", txt):
                out.append(m.group(1))
            for m in re.finditer(r"IP6\.DNS\[\d+\]:\s*(\S+)", txt):
                out.append(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    if not out:
        out.extend(_resolvers_resolv_conf())
    return out


def _resolvers_resolv_conf() -> list[str]:
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


def _resolvers_macos() -> list[str]:
    out: list[str] = []
    try:
        txt = subprocess.run(
            ["scutil", "--dns"], capture_output=True, text=True,
            timeout=8).stdout
        for m in re.finditer(r"nameserver\[\d+\]\s*:\s*(\S+)", txt):
            out.append(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return out


def _resolvers_windows() -> list[str]:
    out: list[str] = []
    try:
        import winreg
        # per-interface NameServer values under Tcpip\Interfaces
        base = (r"SYSTEM\CurrentControlSet\Services\Tcpip\Interfaces")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    name = winreg.EnumKey(key, i)
                    with winreg.OpenKey(key, name) as sub:
                        for val in ("NameServer", "DhcpNameServer"):
                            try:
                                data, _ = winreg.QueryValueEx(sub, val)
                                out.extend(s for s in str(data).replace(
                                    ",", " ").split() if s)
                            except OSError:
                                pass
                except OSError:
                    continue
    except Exception:  # noqa: BLE001
        pass
    if not out:
        try:
            txt = subprocess.run(
                ["ipconfig", "/all"], capture_output=True, text=True,
                timeout=10).stdout
            for m in re.finditer(
                    r"DNS Servers[.\s]*:\s*([0-9a-fA-F.:]+)", txt):
                out.append(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    return out


def configured_resolvers() -> list[str]:
    """Resolvers the OS is configured to use (best effort, never raises)."""
    system = platform.system().lower()
    try:
        if system == "windows":
            raw = _resolvers_windows()
        elif system == "darwin":
            raw = _resolvers_macos()
        else:
            raw = _resolvers_linux()
    except Exception:  # noqa: BLE001
        raw = []
    seen: set[str] = set()
    return [x for x in raw if x and not (x in seen or seen.add(x))]


def detect_smhnr_risk() -> dict[str, Any]:
    """Windows Smart Multi-Homed Name Resolution risk (winreg, no admin).

    SMHNR sends DNS queries to ALL adapters in parallel — a classic,
    invisible leak source (ValdikSS; pavellizunov/vpnrouter). Returns
    ``{"applicable": bool, "risk": bool, "detail": str}``.
    """
    if platform.system().lower() != "windows":
        return {"applicable": False, "risk": False,
                "detail": "SMHNR is a Windows-only behavior."}
    disabled = False
    detail = "could not read SMHNR registry state"
    try:
        import winreg
        checks = [
            (r"SOFTWARE\Policies\Microsoft\Windows NT\DNSClient",
             "DisableSmartNameResolution"),
            (r"SYSTEM\CurrentControlSet\Services\Dnscache\Parameters",
             "DisableParallelAandAAAA"),
        ]
        values = []
        for path, name in checks:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
                    v, _ = winreg.QueryValueEx(k, name)
                    values.append(int(v))
            except OSError:
                values.append(0)
        disabled = all(v != 0 for v in values)
        detail = ("SMHNR explicitly disabled via registry"
                  if disabled else
                  "SMHNR not disabled — Windows may send DNS to all "
                  "adapters in parallel (set DisableSmartNameResolution=1 "
                  "under HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows NT\\"
                  "DNSClient and DisableParallelAandAAAA=1 under HKLM\\"
                  "SYSTEM\\CurrentControlSet\\Services\\Dnscache\\Parameters)")
    except Exception as exc:  # noqa: BLE001
        detail = f"registry read failed: {exc}"
    return {"applicable": True, "risk": not disabled, "detail": detail}


# ── report ───────────────────────────────────────────────────────────────

@dataclass
class DnsLeakReport:
    """Result of a DNS leak check."""
    ok: bool = False                    # True = no leak detected
    method: str = ""                    # "bash.ws" or "local-heuristic"
    public_ip: str = ""
    public_asn: str = ""
    egress_ips: list[str] = field(default_factory=list)   # bash.ws "ip" items
    resolvers: list[str] = field(default_factory=list)
    resolver_info: list[dict[str, Any]] = field(default_factory=list)
    configured_resolvers: list[str] = field(default_factory=list)
    leak: bool = False
    leak_detail: str = ""
    hijack: bool = False                # transparent DNS proxy detected
    hijack_detail: str = ""
    ipv6_egress: str = ""
    smhnr_risk: bool = False
    conclusion: str = ""                # bash.ws server-side verdict text
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "method": self.method,
            "public_ip": self.public_ip, "public_asn": self.public_asn,
            "egress_ips": self.egress_ips,
            "resolvers": self.resolvers, "resolver_info": self.resolver_info,
            "configured_resolvers": self.configured_resolvers,
            "leak": self.leak, "leak_detail": self.leak_detail,
            "hijack": self.hijack, "hijack_detail": self.hijack_detail,
            "ipv6_egress": self.ipv6_egress, "smhnr_risk": self.smhnr_risk,
            "conclusion": self.conclusion, "note": self.note,
        }


def check_dns_leak(*, expected_asn: str = "", probes: int = 30,
                   timeout: float = 90) -> DnsLeakReport:
    """Run a DNS leak check. Returns a report, never raises.

    :param expected_asn: strict mode — every detected resolver must belong
        to this ASN.
    :param probes: how many ``N.<id>.bash.ws`` names to resolve in parallel
        (macvk's reference uses 30; more probes catch round-robin resolvers).
    :param timeout: overall budget in seconds for the bash.ws leg.
    """
    rep = DnsLeakReport()
    try:
        return _check(rep, expected_asn=expected_asn, probes=probes,
                      timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        _log.warning("dns leak check failed: %s", exc)
        rep.note = f"check failed: {exc}"
        return rep


def _check(rep: DnsLeakReport, *, expected_asn: str, probes: int,
           timeout: float) -> DnsLeakReport:
    rep.public_ip = _public_ip()
    if rep.public_ip:
        info = _ip_info(rep.public_ip)
        rep.public_asn = str(info.get("asn") or "")
    rep.configured_resolvers = configured_resolvers()
    smhnr = detect_smhnr_risk()
    rep.smhnr_risk = bool(smhnr.get("risk"))
    if rep.smhnr_risk:
        rep.note = (rep.note + " " if rep.note else "") + smhnr["detail"]
    rep.ipv6_egress = _public_ipv6()

    # ── primary: bash.ws nonce technique (real protocol) ────────────────
    leak_id = _bash_ws_id()
    if leak_id:
        items = _bash_ws_probe(leak_id, probes=probes, timeout=timeout)
        if items is None:
            # rate-limited / empty — one retry after a pause (ricco020 notes
            # bash.ws throttles repeated runs)
            _log.debug("bash.ws empty, retrying once after backoff")
            time.sleep(5)
            items = _bash_ws_probe(leak_id, probes=max(6, probes // 2),
                                   timeout=timeout)
        if items:
            rep.method = "bash.ws"
            _apply_bash_ws_items(rep, items)
            return _assess(rep, expected_asn)

    # ── fallback: local heuristic ───────────────────────────────────────
    rep.method = "local-heuristic"
    rep.resolvers = [r for r in rep.configured_resolvers
                     if r not in _STUB_RESOLVERS]
    rep.note = ((rep.note + " " if rep.note else "")
                + "bash.ws unreachable — reporting configured resolvers only. "
                  "This cannot prove which resolver answered; treat as advisory.")
    return _assess(rep, expected_asn)


def _bash_ws_id() -> str:
    """Ask bash.ws for a leak-test id (empty string on failure)."""
    try:
        leak_id = _http_text(f"{_BASH_WS}/id", timeout=10)
        # sanity: ids are short alnum tokens
        if leak_id and re.fullmatch(r"[A-Za-z0-9_-]{4,64}", leak_id):
            return leak_id
        _log.debug("bash.ws /id returned unexpected payload: %r", leak_id[:40])
    except Exception as exc:  # noqa: BLE001
        _log.debug("bash.ws /id failed: %s", exc)
    return ""


def _resolve_one(host: str) -> None:
    try:
        socket.getaddrinfo(host, 80)
    except OSError:
        pass  # resolution "failure" is fine — the query still went out


def _bash_ws_probe(leak_id: str, *, probes: int,
                   timeout: float) -> list[dict[str, Any]] | None:
    """Resolve N nonce hosts in parallel, then fetch the typed result items."""
    hosts = [f"{i}.{leak_id}.bash.ws" for i in range(1, probes + 1)]
    deadline = time.monotonic() + timeout
    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(probes, 30)) as ex:
            list(ex.map(_resolve_one, hosts))
    except Exception as exc:  # noqa: BLE001
        _log.debug("parallel probe failed: %s", exc)
    remaining = max(5.0, deadline - time.monotonic())
    try:
        data = _http_json(f"{_BASH_WS}/dnsleak/test/{leak_id}?json",
                          timeout=remaining)
    except Exception as exc:  # noqa: BLE001
        _log.debug("bash.ws result fetch failed: %s", exc)
        return None
    items = _extract_bash_ws_items(data)
    return items or None


def _extract_bash_ws_items(data: Any) -> list[dict[str, Any]]:
    """Normalize the bash.ws ``?json`` response to typed item dicts."""
    out: list[dict[str, Any]] = []
    if isinstance(data, dict):
        # tolerate wrapped shapes: {"test": [...]} / {"result": [...]}
        for val in data.values():
            if isinstance(val, list):
                data = val
                break
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type") or "").lower()
            if typ not in ("ip", "dns", "conclusion"):
                # legacy/unknown shapes: an item with an ip but no type is
                # most likely a resolver
                if item.get("ip"):
                    typ = "dns"
                else:
                    continue
            out.append({
                "type": typ,
                "ip": str(item.get("ip") or ""),
                "country": str(item.get("country_name")
                               or item.get("country") or ""),
                "asn": str(item.get("asn") or ""),
                "text": str(item.get("text") or item.get("conclusion") or ""),
            })
    return out


def _apply_bash_ws_items(rep: DnsLeakReport,
                         items: list[dict[str, Any]]) -> None:
    for it in items:
        if it["type"] == "ip" and it["ip"]:
            if it["ip"] not in rep.egress_ips:
                rep.egress_ips.append(it["ip"])
        elif it["type"] == "dns" and it["ip"]:
            if it["ip"] not in rep.resolvers:
                rep.resolvers.append(it["ip"])
                rep.resolver_info.append({
                    "ip": it["ip"], "asn": it["asn"],
                    "country": it["country"], "org": "", "isp": "",
                })
        elif it["type"] == "conclusion" and (it["text"] or it["ip"]):
            rep.conclusion = it["text"] or it["ip"]
    # enrich resolvers that lack ASN (server usually provides it already)
    for ri in rep.resolver_info:
        if not ri.get("asn"):
            info = _ip_info(ri["ip"])
            ri.update({k: info.get(k, "") for k in ("asn", "org", "isp")})
            if info.get("country"):
                ri["country"] = info["country"]


def _assess(rep: DnsLeakReport, expected_asn: str) -> DnsLeakReport:
    # ── transparent DNS proxy: detected resolver not among configured ───
    if rep.resolvers and rep.configured_resolvers:
        real_cfg = {c for c in rep.configured_resolvers
                    if c not in _STUB_RESOLVERS}
        foreign = [r for r in rep.resolvers
                   if r not in real_cfg and r not in _PUBLIC_RESOLVERS]
        if foreign and real_cfg:
            rep.hijack = True
            rep.hijack_detail = (
                "resolvers answering differ from OS-configured resolvers "
                f"({', '.join(sorted(real_cfg))}) — something is intercepting "
                f"DNS: {', '.join(foreign)}")

    if not rep.resolvers:
        rep.note = (rep.note + " " if rep.note else "") + "no resolvers detected."
        rep.ok = False
        return rep

    # ── leak logic ──────────────────────────────────────────────────────
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
                               + ", ".join(f"{b['ip']} ({b.get('org') or b.get('country') or '?'})"
                                           for b in bad))
    rep.ok = not rep.leak and not rep.hijack
    if not rep.leak and not rep.leak_detail:
        rep.leak_detail = "all detected resolvers look consistent — no leak."
    return rep


def format_report(rep: DnsLeakReport) -> str:
    """Human-readable rendering of a leak report (chat/terminal)."""
    L: list[str] = []
    verdict = ("✅ NO LEAK" if rep.ok else
               "🚨 LEAK DETECTED" if rep.leak else
               "⚠️ INCONCLUSIVE")
    L.append(f"🔒 DNS leak check — {verdict}  (method: {rep.method or 'n/a'})")
    if rep.public_ip:
        L.append(f"   egress: {rep.public_ip}"
                 + (f"  [AS{rep.public_asn}]" if rep.public_asn else ""))
    if rep.egress_ips and rep.egress_ips != [rep.public_ip]:
        L.append(f"   bash.ws saw egress: {', '.join(rep.egress_ips)}")
    if rep.resolvers:
        L.append(f"   resolvers ({len(rep.resolvers)}):")
        for ri in rep.resolver_info or [{"ip": r} for r in rep.resolvers]:
            tag = []
            if ri.get("asn"):
                tag.append(f"AS{ri['asn']}")
            if ri.get("country"):
                tag.append(ri["country"])
            if ri.get("org"):
                tag.append(str(ri["org"]))
            mark = " ⚠️" if rep.leak and ri["ip"] in rep.leak_detail else ""
            L.append(f"     • {ri['ip']}"
                     + (f"  [{' · '.join(tag)}]" if tag else "") + mark)
    if rep.configured_resolvers:
        L.append("   OS-configured: " + ", ".join(rep.configured_resolvers))
    if rep.hijack:
        L.append(f"   🕵️ transparent proxy: {rep.hijack_detail}")
    if rep.leak:
        L.append(f"   detail: {rep.leak_detail}")
    elif rep.leak_detail:
        L.append(f"   {rep.leak_detail}")
    if rep.ipv6_egress:
        L.append(f"   IPv6 egress live: {rep.ipv6_egress} "
                 "(check it is inside the tunnel too)")
    if rep.smhnr_risk:
        L.append("   ⚠️ Windows SMHNR active — DNS may fan out to all adapters")
    if rep.conclusion:
        L.append(f"   bash.ws: {rep.conclusion}")
    if rep.note:
        L.append(f"   note: {rep.note.strip()}")
    return "\n".join(L)
