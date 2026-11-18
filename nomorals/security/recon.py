"""Passive surface discovery: domain -> asset inventory.

Order is deliberate — cheapest and least intrusive first, and every
finding is filtered through a wildcard control so a catch-all DNS record
cannot invent 500 fake subdomains.

* certificate transparency (crt.sh) — the whole historical name set
* DNS over HTTPS (Cloudflare JSON) — A/AAAA/CNAME/MX/TXT/NS
* AXFR — real zone transfer attempt against each authoritative NS
* takeover — CNAME against the provider table, then the provider's own
  unclaimed marker in the HTTP body (a CNAME alone proves nothing)

Nothing is brute-forced here; that is `subbrute.py`.
"""
from __future__ import annotations

import json
import random
import re
import socket
import string
import struct
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)
__all__ = ["recon", "crt_sh", "dns_lookup", "axfr", "takeover_check", "register"]

_DOH = "https://cloudflare-dns.com/dns-query"
_RRTYPE = {"A": 1, "NS": 2, "CNAME": 5, "SOA": 6, "MX": 15, "TXT": 16,
           "AAAA": 28, "AXFR": 252}

#: CNAME suffix -> (provider, regex marking an UNCLAIMED resource).
#: Source list: can-i-take-over-xyz.  Statuses drift; a hit is a lead,
#: not a verdict.
_TAKEOVER: tuple[tuple[str, str, str], ...] = (
    ("s3.amazonaws.com", "AWS S3", r"NoSuchBucket"),
    ("s3-website", "AWS S3 website", r"NoSuchBucket|NoSuchWebsiteConfiguration"),
    ("cloudfront.net", "CloudFront", r"Bad request|ERROR: The request could not be satisfied"),
    ("github.io", "GitHub Pages", r"There isn't a GitHub Pages site here"),
    ("herokuapp.com", "Heroku", r"No such app"),
    ("herokudns.com", "Heroku DNS", r"No such app"),
    ("netlify.app", "Netlify", r"Not Found - Request ID"),
    ("netlify.com", "Netlify", r"Not Found - Request ID"),
    ("surge.sh", "Surge", r"project not found"),
    ("bitbucket.io", "Bitbucket", r"Repository not found"),
    ("ghost.io", "Ghost", r"Domain is not configured"),
    ("myshopify.com", "Shopify", r"Sorry, this shop is currently unavailable"),
    ("fastly.net", "Fastly", r"Fastly error: unknown domain"),
    ("pantheonsite.io", "Pantheon", r"The gods are wise|404 error unknown site"),
    ("readme.io", "ReadMe", r"Project doesnt exist"),
    ("zendesk.com", "Zendesk", r"Help Center Closed"),
    ("unbouncepages.com", "Unbounce", r"The requested URL was not found"),
    ("tumblr.com", "Tumblr", r"Whatever you were looking for doesn't currently exist"),
    ("cargo.site", "Cargo", r"404 Not Found"),
    ("wpengine.com", "WP Engine", r"Couldn't find this site"),
    ("azurewebsites.net", "Azure", r"Error 404 - Web app not found"),
    ("cloudapp.azure.com", "Azure VM", r"404 Web Site not found"),
    ("trafficmanager.net", "Azure Traffic Manager", r"not found"),
    ("digitaloceanspaces.com", "DO Spaces", r"NoSuchBucket"),
    ("storage.googleapis.com", "GCS", r"NoSuchBucket"),
)

_SAFE = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)+$")


@dataclass
class Asset:
    host: str
    records: dict[str, list[str]] = field(default_factory=dict)
    cname_chain: list[str] = field(default_factory=list)
    takeover: Optional[dict[str, Any]] = None
    sources: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {"host": self.host, "records": self.records,
                "cname_chain": self.cname_chain, "takeover": self.takeover,
                "sources": sorted(self.sources)}


# ── certificate transparency ────────────────────────────────────────────────
def crt_sh(domain: str, *, timeout: float = 30.0) -> list[str]:
    """Every name ever issued for the domain, deduped, wildcards stripped."""
    client = HttpClient(timeout=timeout)
    url = "https://crt.sh/?q=%25." + urllib.parse.quote(domain) + "&output=json"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        _log.info("crt.sh failed for %s: %s", domain, exc)
        return []
    if not resp.ok:
        return []
    try:
        rows = json.loads(resp.text or "[]")
    except ValueError:
        return []
    out: set[str] = set()
    for row in rows:
        for name in str(row.get("name_value", "")).splitlines():
            name = name.strip().lower().lstrip("*.")
            if name.endswith(domain) and _SAFE.match(name):
                out.add(name)
    return sorted(out)


# ── DNS over HTTPS ──────────────────────────────────────────────────────────
def dns_lookup(name: str, rtype: str = "A", *, timeout: float = 10.0) -> list[str]:
    if rtype.upper() not in _RRTYPE:
        raise ToolError(f"unsupported record type {rtype!r}")
    client = HttpClient(timeout=timeout)
    url = f"{_DOH}?name={urllib.parse.quote(name)}&type={rtype.upper()}"
    try:
        resp = client.get(url, headers={"Accept": "application/dns-json"})
    except Exception as exc:  # noqa: BLE001
        _log.info("doh %s %s failed: %s", name, rtype, exc)
        return []
    if not resp.ok:
        return []
    try:
        data = json.loads(resp.text or "{}")
    except ValueError:
        return []
    out: list[str] = []
    for ans in data.get("Answer", []) or []:
        if int(ans.get("type", 0)) == _RRTYPE[rtype.upper()]:
            value = str(ans.get("data", "")).strip().rstrip(".")
            if value:
                out.append(value)
    return out


def wildcard_control(domain: str) -> Optional[str]:
    """Resolve a random label. Any IP here is a catch-all and proves
    nothing about a real subdomain."""
    label = "".join(random.choices(string.ascii_lowercase, k=20))
    hits = dns_lookup(f"{label}.{domain}", "A")
    return hits[0] if hits else None


# ── AXFR over raw TCP ───────────────────────────────────────────────────────
def _enc_name(name: str) -> bytes:
    out = b""
    for label in name.rstrip(".").split("."):
        out += bytes([len(label)]) + label.encode("ascii")
    return out + b"\x00"


def _read_name(buf: bytes, off: int) -> tuple[str, int]:
    labels: list[str] = []
    jumped = False
    end = off
    while True:
        if off >= len(buf):
            break
        length = buf[off]
        if length == 0:
            off += 1
            if not jumped:
                end = off
            break
        if length & 0xC0 == 0xC0:
            pointer = struct.unpack("!H", buf[off:off + 2])[0] & 0x3FFF
            if not jumped:
                end = off + 2
            jumped = True
            off = pointer
            continue
        labels.append(buf[off + 1:off + 1 + length].decode("ascii", "replace"))
        off += 1 + length
        if not jumped:
            end = off
    return ".".join(labels), end


def axfr(domain: str, ns: str, *, timeout: float = 20.0) -> list[dict[str, str]]:
    """Attempt a real zone transfer. Returns [] on REFUSED/empty."""
    query_id = random.randint(1, 65535)
    packet = (struct.pack("!HHHHHH", query_id, 0x0000, 1, 0, 0, 0)
              + _enc_name(domain) + struct.pack("!HH", _RRTYPE["AXFR"], 1))
    records: list[dict[str, str]] = []
    soa_seen = 0
    try:
        with socket.create_connection((ns, 53), timeout=timeout) as sock:
            sock.sendall(struct.pack("!H", len(packet)) + packet)
            while True:
                head = sock.recv(2)
                if len(head) < 2:
                    break
                size = struct.unpack("!H", head)[0]
                buf = b""
                while len(buf) < size:
                    chunk = sock.recv(size - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < 12:
                    break
                ancount = struct.unpack("!H", buf[6:8])[0]
                off = 12
                qd = struct.unpack("!H", buf[4:6])[0]
                for _ in range(qd):
                    _, off = _read_name(buf, off)
                    off += 4
                for _ in range(ancount):
                    name, off = _read_name(buf, off)
                    if off + 10 > len(buf):
                        break
                    rtype, _, _, rdlen = struct.unpack("!HHIH", buf[off:off + 10])
                    off += 10
                    rdata = buf[off:off + rdlen]
                    off += rdlen
                    label = next((k for k, v in _RRTYPE.items() if v == rtype), str(rtype))
                    if label == "SOA":
                        soa_seen += 1
                    value = ""
                    if label == "A" and rdlen == 4:
                        value = ".".join(str(b) for b in rdata)
                    elif label == "CNAME" or label == "NS":
                        value, _ = _read_name(buf, off - rdlen)
                    records.append({"name": name, "type": label, "value": value})
                if soa_seen >= 2:
                    break
    except (OSError, struct.error) as exc:
        _log.info("axfr %s @%s failed: %s", domain, ns, exc)
    return records


# ── takeover ────────────────────────────────────────────────────────────────
def takeover_check(host: str, *, timeout: float = 15.0) -> Optional[dict[str, Any]]:
    chain = dns_lookup(host, "CNAME")
    if not chain:
        return None
    target = chain[0].lower()
    for suffix, provider, marker in _TAKEOVER:
        if suffix in target:
            client = HttpClient(timeout=timeout)
            try:
                resp = client.get(f"http://{host}/")
                body = (resp.text or "")[:20_000]
            except Exception:  # noqa: BLE001
                body = ""
            unclaimed = bool(re.search(marker, body, re.I)) if marker else False
            return {"host": host, "cname": target, "provider": provider,
                    "unclaimed_marker": unclaimed,
                    "state": "likely" if unclaimed else "cname-only"}
    return None


# ── orchestrator ────────────────────────────────────────────────────────────
def recon(domain: str, *, include_ct: bool = True, do_axfr: bool = True,
          timeout: float = 20.0) -> dict[str, Any]:
    domain = (domain or "").strip().lower().strip(".")
    if not _SAFE.match(domain):
        raise ToolError(f"not a domain: {domain!r}")
    wildcard = wildcard_control(domain)
    names: set[str] = {domain}
    sources: dict[str, set[str]] = {}
    if include_ct:
        for name in crt_sh(domain, timeout=timeout):
            names.add(name)
            sources.setdefault(name, set()).add("crt.sh")
    for sub in ("www", "mail", "api", "dev", "staging", "admin", "vpn"):
        if dns_lookup(f"{sub}.{domain}", "A"):
            names.add(f"{sub}.{domain}")
            sources.setdefault(f"{sub}.{domain}", set()).add("common")
    assets: list[Asset] = []
    for name in sorted(names):
        asset = Asset(host=name, sources=sources.get(name, set()) or {"seed"})
        for rtype in ("A", "AAAA", "CNAME", "MX", "TXT"):
            values = dns_lookup(name, rtype, timeout=timeout)
            if rtype == "CNAME":
                asset.cname_chain = values
            if values:
                asset.records[rtype] = values
        if wildcard and asset.records.get("A") == [wildcard] and name != domain:
            asset.sources.add("wildcard-match")
        asset.takeover = takeover_check(name, timeout=timeout)
        assets.append(asset)
    zones: dict[str, int] = {}
    if do_axfr:
        for ns in dns_lookup(domain, "NS", timeout=timeout):
            rows = axfr(domain, ns, timeout=timeout)
            if rows:
                zones[ns] = len(rows)
                for row in rows:
                    host = row["name"].lower()
                    if host.endswith(domain) and host not in names:
                        extra = Asset(host=host, sources={"axfr"})
                        extra.records[row["type"]] = [row["value"]]
                        assets.append(extra)
                        names.add(host)
    return {
        "domain": domain,
        "wildcard": wildcard,
        "host_count": len(assets),
        "axfr": zones,
        "takeover_candidates": [a.as_dict() for a in assets if a.takeover],
        "assets": [a.as_dict() for a in assets],
    }


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "recon_domain",
        description="Passive surface discovery for a domain: CT logs, DNS over "
                    "HTTPS, AXFR attempt, wildcard suppression, takeover leads.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str — apex domain",
                    "no_ct": "bool (optional) — skip certificate transparency",
                    "no_axfr": "bool (optional) — skip zone transfer"},
    )
    def recon_domain(domain: str, *, no_ct: str = "",
                     no_axfr: str = "") -> dict[str, Any]:
        def flag(v: str) -> bool:
            return (v or "").lower() in {"1", "true", "yes", "on"}
        return recon(domain, include_ct=not flag(no_ct), do_axfr=not flag(no_axfr))
