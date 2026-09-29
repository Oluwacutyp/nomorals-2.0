"""Network tools: DNS, port awareness, request crafting, whois, local services.

Two classes of capability, deliberately separated:

* **Passive / own-machine** — DNS resolution (pure-Python UDP wire protocol,
  no dependencies), RDAP whois (public registry data), local listener
  inventory (``/proc/net/tcp``), raw request crafting. Always available.

* **Active probing** — TCP port scans and service banners. Scoped to the
  operator's own infrastructure: loopback and private ranges pass by
  default; anything public must be explicitly allowlisted in
  ``NM_NET_ALLOWED_TARGETS``. This is the standing agreement — your
  endpoints, not strangers'.

Everything else a "recon" kit would normally do against third parties is
deliberately absent.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..core.errors import ToolError
from ..core.http import HttpClient, default_proxy_handler
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = [
    "dns_query", "parse_dns_response", "is_scannable", "tcp_probe",
    "local_listeners", "rdap_domain", "http_craft", "register",
]

_PUBLIC_RESOLVERS = ("1.1.1.1", "8.8.8.8", "9.9.9.9")
_TYPE_IDS = {
    "a": 1, "ns": 2, "cname": 5, "soa": 6, "ptr": 12, "mx": 15,
    "txt": 16, "aaaa": 28, "srv": 33, "caa": 257, "spf": 16,
}
class _IdCounter:
    """Tiny monotonically-increasing DNS transaction-id source."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> int:
        self._n = (self._n + 1) & 0xFFFF
        return self._n or 1


_ID_COUNTER = _IdCounter()


# ── DNS: minimal wire protocol (stdlib UDP, works on Termux with nothing) ───


def _encode_name(domain: str) -> bytes:
    out = b""
    for label in domain.rstrip(".").lower().split("."):
        label = label.encode("idna") if any(ord(c) > 127 for c in label) else label.encode("ascii")
        if not 0 < len(label) < 64:
            raise ToolError(f"bad DNS label {label!r}")
        out += bytes([len(label)]) + label
    return out + b"\x00"


def dns_query(domain: str, record: str = "A", *, resolvers: tuple[str, ...] = _PUBLIC_RESOLVERS,
              timeout: float = 3.0) -> list[str]:
    """Resolve a record type. Pure-Python: builds the UDP query, sends it to
    the first public resolver that answers, parses the answer section."""
    record = (record or "A").upper()
    if record == "SPF":
        record, domain = "TXT", f"spf.{domain}" if not domain.startswith("_spf.") else domain
    type_id = _TYPE_IDS.get(record.lower())
    if type_id is None:
        raise ToolError(f"unsupported record type {record!r}; use one of {sorted(set(_TYPE_IDS))}")

    header = struct.pack(">HHHHHH", 0x1000, 0, 1, 0, 0, 0)
    tid = _ID_COUNTER()
    header = struct.pack(">H", tid) + header[2:]
    query = header + _encode_name(domain) + struct.pack(">HH", type_id, 1)

    last_error = ""
    got_response = False
    for resolver in resolvers:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.sendto(query, (resolver, 53))
            data, _ = sock.recvfrom(4096)
        except (socket.timeout, OSError) as exc:
            last_error = f"{resolver}: {exc}"
            continue
        finally:
            sock.close()
        try:
            answers, rcode = _parse_response(data)
        except ToolError:
            continue  # unparseable (middlebox garbage) — next resolver
        got_response = True
        if rcode == 3:  # NXDOMAIN is definitive
            return []
        if answers:
            return answers
        # empty answer — try the next resolver before trusting it
    if got_response:
        return []
    raise ToolError(f"DNS query failed for {domain} {record}: {last_error}")


def _parse_name(data: bytes, offset: int) -> tuple[str, int]:
    """Parse a (possibly compressed) DNS name. Returns (name, next_offset)."""
    labels: list[str] = []
    jumped = False
    end = offset
    while True:
        if offset >= len(data):
            raise ToolError("truncated DNS name")
        length = data[offset]
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(data):
                raise ToolError("truncated DNS pointer")
            pointer = struct.unpack(">H", data[offset:offset + 2])[0] & 0x3FFF
            if not jumped:
                end = offset + 2
            offset = pointer
            jumped = True
            continue
        if length == 0:
            offset += 1
            break
        offset += 1
        if offset + length > len(data):
            raise ToolError("truncated DNS label")
        labels.append(data[offset:offset + length].decode("ascii", "replace"))
        offset += length
    if not jumped:
        end = offset  # plain name: we simply walked to the end
    return ".".join(labels) or ".", end


def _rdata(data: bytes, offset: int, rtype: int, end: int | None = None) -> list[str]:
    """Decode one RDATA blob. ``end`` bounds length-walking types (TXT)."""
    if end is None:
        end = len(data)
    if rtype in (1,):  # A
        return [".".join(str(b) for b in data[offset:offset + 4])]
    if rtype == 28:  # AAAA
        return [str(ipaddress.ip_address(data[offset:offset + 16]))]
    if rtype in (5, 12, 2):  # CNAME / PTR / NS
        name, _ = _parse_name(data, offset)
        return [name]
    if rtype == 15:  # MX
        preference = struct.unpack(">H", data[offset:offset + 2])[0]
        name, _ = _parse_name(data, offset + 2)
        return [f"{preference} {name}"]
    if rtype == 16:  # TXT (possibly multi-string)
        out, pos = [], offset
        while pos < end:
            length = data[pos]
            if pos + 1 + length > len(data):
                break
            out.append(data[pos + 1:pos + 1 + length].decode("utf-8", "replace"))
            pos += 1 + length
        # RFC 7208: TXT character-strings are concatenated without separators
        return ["".join(out)] if out else []
    if rtype == 6:  # SOA
        mname, pos = _parse_name(data, offset)
        rname, pos = _parse_name(data, pos)
        serial, refresh, retry, expire, minimum = struct.unpack(">IIIII", data[pos:pos + 20])
        return [f"{mname} {rname} {serial} {refresh} {retry} {expire} {minimum}"]
    if rtype == 33:  # SRV (full)
        pref, weight, port = struct.unpack(">HHH", data[offset:offset + 6])
        name, _ = _parse_name(data, offset + 6)
        return [f"{pref} {weight} {port} {name}"]
    if rtype == 257:  # CAA: flags, length, property, tag, value
        length = data[offset + 1]
        prop = data[offset + 2:offset + 2 + length].decode("ascii", "replace")
        tag = data[offset + 2 + length]
        value = data[offset + 3 + length:].decode("ascii", "replace").strip()
        return [f"{prop} {chr(tag)} {value}"]
    return [data[offset:offset + 16].hex()]


def _parse_response(data: bytes) -> tuple[list[str], int]:
    """Parse the answer section. Returns (answers, rcode)."""
    if len(data) < 12:
        raise ToolError("response too short")
    flags = struct.unpack(">H", data[2:4])[0]
    rcode = flags & 0x0F
    ancount = struct.unpack(">H", data[6:8])[0]
    # Skip the question section (name + 4 bytes of type/class).
    _, offset = _parse_name(data, 12)
    offset += 4
    answers: list[str] = []
    for _ in range(ancount):
        try:
            _, offset = _parse_name(data, offset)
            rtype, _rclass, _ttl, rdlength = struct.unpack(">HHIH", data[offset:offset + 10])
            offset += 10
            answers.extend(_rdata(data, offset, rtype, offset + rdlength))
            offset += rdlength
        except (struct.error, ToolError, IndexError):
            break
    return answers, rcode


parse_dns_response = _parse_response  # public alias for tests


# ── scope guard: who may be actively probed ─────────────────────────────────


def is_scannable(target: str, allowed_targets: str = "") -> bool:
    """True when the target is within the operator's own infrastructure.

    Loopback, private, link-local, and CGNAT ranges always pass; public
    targets only when explicitly allowlisted (comma-separated in settings).
    """
    allowed = {a.strip() for a in (allowed_targets or "").split(",") if a.strip()}
    target = (target or "").strip()
    if not target:
        return False
    try:
        ip = ipaddress.ip_address(target)
        if not ip.is_global:  # loopback, private, link-local, CGNAT, test, reserved
            return True
        return target in allowed
    except ValueError:
        pass  # hostname — resolved by the caller; literal match on the allowlist
    return target.lower() in {a.lower() for a in allowed}


def _assert_scannable(target: str, allowed_targets: str) -> list[str]:
    """Resolve hostnames, then enforce the scope. Returns the IP list to scan."""
    target = (target or "").strip()
    if not target:
        raise ToolError("port_scan needs a target")
    try:
        ipaddress.ip_address(target)
        ips = [target]
    except ValueError:
        try:
            ips = dns_query(target, "A", timeout=3.0) or dns_query(target, "AAAA", timeout=3.0)
        except ToolError as exc:
            raise ToolError(f"cannot resolve {target!r}: {exc}") from exc
        if not ips:
            raise ToolError(f"cannot resolve {target!r}")
    for ip in ips:
        if not is_scannable(ip, allowed_targets):
            raise ToolError(
                f"active probing of {ip} is outside the allowed scope (own "
                f"infrastructure only). It is your system, not a stranger's — "
                f"add it to NM_NET_ALLOWED_TARGETS to permit it explicitly."
            )
    return ips


# ── port probing ─────────────────────────────────────────────────────────────


def tcp_probe(host: str, port: int, timeout: float = 1.5, banner: bool = False) -> str:
    """'open' | 'closed' | 'filtered' (connect-scan semantics).

    With ``banner`` an open port returns 'open:<service banner>' — the first
    greeting bytes, printable-stripped, capped at 80 chars (own targets only).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        if result == 0:
            if banner:
                try:
                    sock.settimeout(min(2.0, timeout))
                    raw = sock.recv(64)
                    text = "".join(
                        ch for ch in raw.decode("utf-8", "replace")
                        if ch.isprintable() or ch in "\r\n"
                    ).strip()[:80]
                    if text:
                        return f"open:{text}"
                except (socket.timeout, OSError):
                    pass
            return "open"
        if result in (111, 61, 10061):  # ECONNREFUSED family
            return "closed"
        return "filtered"
    except socket.timeout:
        return "filtered"
    except OSError:
        return "closed"
    finally:
        sock.close()


def local_listeners() -> list[dict[str, Any]]:
    """What is LISTENing on this machine (/proc/net/tcp{,6}, ss fallback)."""
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for path, family in (("/proc/net/tcp", "ipv4"), ("/proc/net/tcp6", "ipv6")):
        try:
            with open(path, encoding="ascii") as fh:
                text = fh.read()
        except OSError:
            continue
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4 or parts[3] != "0A":  # 0A = LISTEN
                continue
            addr_hex, port_hex = parts[1].rsplit(":", 1)
            port = int(port_hex, 16)
            if family == "ipv4":
                ip = socket.inet_ntoa(struct.pack("<I", int(addr_hex, 16)))
            else:
                ip = f"[{addr_hex}]"
            found.setdefault((ip, port), {"address": ip, "port": port, "family": family})
    if not found:
        # non-Linux fallback: ss if present
        import shutil
        import subprocess

        ss = shutil.which("ss")
        if ss:
            try:
                out = subprocess.run([ss, "-tln"], capture_output=True, timeout=5, check=False)
                for line in out.stdout.decode("utf-8", "replace").splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        addr, port = parts[3].rsplit(":", 1)
                        if port.isdigit():
                            found.setdefault((addr, int(port)),
                                             {"address": addr, "port": int(port), "family": "?"})
            except (OSError, subprocess.SubprocessError):
                pass
    return sorted(found.values(), key=lambda d: d["port"])


# ── whois via RDAP (public HTTPS JSON, no binary needed) ─────────────────────


def rdap_domain(domain: str, *, client: HttpClient | None = None,
                timeout: float = 15.0) -> dict[str, Any]:
    client = client or HttpClient(timeout=timeout)
    url = f"https://rdap.org/domain/{domain.strip().lower()}"
    try:
        response = client.get(url)
    except Exception as exc:  # noqa: BLE001 - whois is best-effort
        raise ToolError(f"RDAP lookup failed for {domain!r}: {exc}") from exc
    if response.status == 404:
        raise ToolError(f"no RDAP registration found for {domain!r} (not a gTLD? try whois)")
    if not response.ok:
        raise ToolError(f"RDAP returned {response.status} for {domain!r}")
    data = response.json()
    out: dict[str, Any] = {"domain": domain.lower(), "handle": data.get("handle", "")}
    for event in data.get("events") or []:
        action = str(event.get("eventAction") or "")
        if action in {"registration", "expiration", "last changed", "last update of rdap database"}:
            out[action.replace(" ", "_")] = str(event.get("eventDate", ""))[:10]
    for entity in data.get("entities") or []:
        roles = [str(r) for r in entity.get("roles") or []]
        vcard = ((entity.get("vcardArray") or [None, []])[1] or [])
        for item in vcard:
            if isinstance(item, list) and item and item[0] == "fn":
                out["registrar" if "registrar" in roles else "entity"] = item[-1]
    nameservers = [str(n.get("ldhName")).lower()
                   for n in data.get("nameservers") or [] if n.get("ldhName")]
    out["nameservers"] = nameservers
    status = [s for s in data.get("status") or []]
    out["status"] = status
    return out


# ── raw request crafting ─────────────────────────────────────────────────────


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


def http_craft(
    method: str,
    url: str,
    *,
    headers: str = "",
    body: str = "",
    timeout: float = 20.0,
    follow_redirects: bool = False,
    max_body_chars: int = 8000,
) -> dict[str, Any]:
    """Send one hand-crafted request; report exactly what came back.

    headers: "Name: value" per line. body: raw request body. The response is
    never followed by default — 3xx stays visible, which is the point of
    crafting.
    """
    method = (method or "GET").upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        raise ToolError(f"unsupported method {method!r}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("url must be http(s)")
    header_map: dict[str, str] = {}
    for line in (headers or "").splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            header_map[key.strip()] = value.strip()
    data = body.encode("utf-8") if body else None
    request = urllib.request.Request(url, data=data, method=method, headers=header_map)
    handlers: list[urllib.request.BaseHandler] = []
    if not follow_redirects:
        handlers.append(_NoRedirect())
    proxy_handler = default_proxy_handler()
    if proxy_handler is not None:
        handlers.append(proxy_handler)
    opener = urllib.request.build_opener(*handlers) if handlers \
        else urllib.request.build_opener()
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            resp_headers = {k: v for k, v in response.headers.items()}
            body_bytes = response.read(max_body_chars + 1)
    except urllib.error.HTTPError as exc:
        # 4xx/5xx are ANSWERS, not failures for a crafting tool
        status = exc.code
        resp_headers = {k: v for k, v in (exc.headers.items() if exc.headers else [])}
        try:
            body_bytes = exc.read(max_body_chars + 1)
        except Exception:  # noqa: BLE001
            body_bytes = b""
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"request failed: {exc}") from exc
    truncated = len(body_bytes) > max_body_chars
    return {
        "method": method,
        "url": url,
        "status": status,
        "ok": 200 <= status < 300,
        "location": resp_headers.get("Location", ""),
        "server": resp_headers.get("Server", ""),
        "content_type": resp_headers.get("Content-Type", ""),
        "request_headers": header_map,
        "response_headers": resp_headers,
        "body": body_bytes[:max_body_chars].decode("utf-8", "replace"),
        "body_truncated": truncated,
        "seconds": round(time.monotonic() - started, 3),
    }


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "dns_lookup",
        description="DNS resolution (A/AAAA/MX/NS/TXT/SOA/PTR/CNAME/SPF/SRV/CAA) — pure-Python, no deps.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str", "record": "str (optional, default A)"},
    )
    def dns_lookup(domain: str, *, record: str = "A") -> dict[str, Any]:
        started = time.monotonic()
        answers = dns_query(domain, record)
        return {"domain": domain, "record": (record or "A").upper(),
                "answers": answers, "seconds": round(time.monotonic() - started, 3)}

    @registry.register(
        "reverse_dns",
        description="PTR lookup for an IP address.",
        capability=Capability.NET_OUT,
        parameters={"ip": "str"},
    )
    def reverse_dns(ip: str) -> dict[str, Any]:
        try:
            ipaddress.ip_address(ip.strip())
        except ValueError:
            raise ToolError(f"not an IP address: {ip!r}") from None
        answers = dns_query(ip, "PTR")
        return {"ip": ip.strip(), "ptr": answers or None}

    @registry.register(
        "port_scan",
        description=(
            "TCP connect port scan — own infrastructure only (loopback/private, or "
            "NM_NET_ALLOWED_TARGETS). Returns open/closed/filtered per port."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "target": "str — host or IP",
            "ports": "str (optional) — comma list or 'start-end' range; default from settings",
            "banner": "bool (optional, default off) — read a service banner on open ports",
        },
    )
    def port_scan(target: str, *, ports: str = "", banner: str = "") -> dict[str, Any]:
        settings = getattr(context, "settings", None)
        net = getattr(settings, "net", None) if settings is not None else None
        allowed = getattr(net, "allowed_targets", "") if net is not None else ""
        timeout = float(getattr(net, "probe_timeout", 1.5)) if net is not None else 1.5
        max_ports = int(getattr(net, "max_probe_ports", 1024)) if net is not None else 1024
        use_banner = (banner or "").lower() in {"1", "true", "yes", "on"}

        ips = _assert_scannable(target, allowed)
        port_list = _expand_ports(ports or (getattr(net, "default_ports", "") if net else ""))
        if not port_list:
            raise ToolError("no ports to scan")
        if len(port_list) > max_ports:
            raise ToolError(f"too many ports ({len(port_list)} > {max_ports})")

        started = time.monotonic()
        host = ips[0]
        with ThreadPoolExecutor(max_workers=48, thread_name_prefix="scan") as pool:
            futures = {port: pool.submit(tcp_probe, host, port, timeout, use_banner)
                       for port in port_list}
            outcomes = {port: fut.result() for port, fut in futures.items()}
        open_ports = sorted(p for p, s in outcomes.items() if s.startswith("open"))
        closed = sorted(p for p, s in outcomes.items() if s == "closed")
        filtered = sorted(p for p, s in outcomes.items() if s == "filtered")
        return {
            "target": target, "scanned": host,
            "open": open_ports, "closed": len(closed), "filtered": len(filtered),
            "seconds": round(time.monotonic() - started, 2),
            "scope": "own-infrastructure",
        }

    @registry.register(
        "local_services",
        description="List what is LISTENing on this machine (ports + addresses).",
        capability=Capability.FS_READ,
    )
    def local_services() -> dict[str, Any]:
        listeners = local_listeners()
        return {"count": len(listeners), "listeners": listeners[:64]}

    @registry.register(
        "whois_lookup",
        description="Registration data via public RDAP: registrar, dates, nameservers, status.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str"},
    )
    def whois_lookup(domain: str) -> dict[str, Any]:
        return rdap_domain(domain)

    @registry.register(
        "http_craft",
        description=(
            "Hand-craft one HTTP request: any method, headers, raw body; 3xx are not "
            "followed by default. For controlled interactions (webhooks, API tests)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "method": "str — GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS",
            "url": "str",
            "headers": "str (optional) — 'Name: value' per line",
            "body": "str (optional) — raw request body",
            "timeout": "float (optional, default 20)",
            "follow_redirects": "bool (optional, default off)",
        },
    )
    def http_craft_tool(method: str, url: str, *, headers: str = "", body: str = "",
                        timeout: float = 20.0, follow_redirects: str = "") -> dict[str, Any]:
        return http_craft(
            method, url, headers=headers, body=body, timeout=float(timeout),
            follow_redirects=(follow_redirects or "").lower() in {"1", "true", "yes", "on"},
        )


def _expand_ports(spec: str) -> list[int]:
    out: set[int] = set()
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            try:
                start_i, end_i = int(start), int(end)
            except ValueError:
                continue
            if 0 < start_i <= end_i < 65536:
                out.update(range(start_i, end_i + 1))
        else:
            try:
                port = int(part)
                if 0 < port < 65536:
                    out.add(port)
            except ValueError:
                continue
    return sorted(out)
