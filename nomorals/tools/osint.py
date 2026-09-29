"""OSINT toolkit: read-only intelligence from public sources.

Every source here is public by definition: DNS, RDAP registries, certificate
transparency logs, IP allocation data, TLS handshakes. Optional depth comes
from the operator's own keys (AbuseIPDB, Shodan). No active probing, no
authenticated scraping, no breach data, no social-media harvesting behind
login walls — that line is the same standing one.

``osint_report`` auto-detects the target kind (domain / IP / URL / email)
and composes one consolidated report, which is what the main AI and sub-agents
mostly call.
"""

from __future__ import annotations

import ipaddress
import json
import re
import ssl
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..core.errors import ToolError
from ..core.http import HttpClient, default_proxy_handler
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .network import dns_query, rdap_domain

_log = get_logger(__name__)

__all__ = ["osint_domain", "osint_ip", "osint_url", "osint_email", "osint_report", "register"]


def _settings(context: Any) -> tuple[float, str, str, int]:
    s = getattr(context, "settings", None)
    osint = getattr(s, "osint", None) if s else None
    return (
        float(getattr(osint, "request_timeout", 15.0)) if osint else 15.0,
        (getattr(osint, "abuseipdb_key", "") if osint else "") or "",
        (getattr(osint, "shodan_key", "") if osint else "") or "",
        int(getattr(osint, "crtsh_days", 3650)) if osint else 3650,
    )


def _http(context: Any, url: str, timeout: float) -> Any:
    return HttpClient(timeout=timeout).get(url)


# ── domain ───────────────────────────────────────────────────────────────────


def osint_domain(context: Any, domain: str) -> dict[str, Any]:
    timeout, _abuse, _shodan, _days = _settings(context)
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or " " in domain:
        raise ToolError(f"bad domain {domain!r}")
    out: dict[str, Any] = {"target": domain, "kind": "domain"}
    records: dict[str, list[str]] = {}
    for record in ("A", "AAAA", "NS", "MX", "TXT", "SPF", "CAA"):
        try:
            records[record] = dns_query(domain, record, timeout=min(4.0, timeout))
        except ToolError:
            records[record] = []
    out["dns"] = records
    spf = records.get("SPF") or []
    out["spf"] = spf[0] if spf else None
    try:
        out["registration"] = rdap_domain(domain, timeout=timeout)
    except ToolError as exc:
        out["registration"] = f"unavailable: {exc}"
    # Certificate transparency (crt.sh public API)
    try:
        response = _http(context, f"https://crt.sh/?q={urllib.parse.quote(domain)}&output=json", timeout)
        certs = response.json() if response.ok else []
        if isinstance(certs, list) and certs:
            names: set[str] = set()
            for entry in certs:
                for name in str(entry.get("name_value") or "").split("\n"):
                    names.add(name.strip())
            out["cert_transparency"] = {
                "distinct_names": sorted(n for n in names if n)[:40],
                "sample_cert": certs[0].get("common_name", ""),
                "issuer": certs[0].get("issuer_name", ""),
                "not_before": certs[0].get("not_before", "")[:10],
                "count": len(certs),
            }
        else:
            out["cert_transparency"] = None
    except Exception as exc:  # noqa: BLE001 - CT is best-effort
        out["cert_transparency"] = f"unavailable: {exc}"
    return out


# ── IP ───────────────────────────────────────────────────────────────────────


def osint_ip(context: Any, ip: str) -> dict[str, Any]:
    timeout, abuse_key, shodan_key, _ = _settings(context)
    ip = (ip or "").strip()
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        raise ToolError(f"not an IP address: {ip!r}") from None
    out: dict[str, Any] = {"target": ip, "kind": "ip",
                           "version": parsed.version,
                           "private": parsed.is_private}
    try:
        ptr = dns_query(ip, "PTR", timeout=min(4.0, timeout))
        out["reverse_dns"] = ptr[0] if ptr else None
    except ToolError:
        out["reverse_dns"] = None
    try:
        response = _http(context, f"https://rdap.org/ip/{ip}", timeout)
        if response.ok:
            data = response.json()
            reg: dict[str, Any] = {"handle": data.get("handle", "")}
            for entity in data.get("entities") or []:
                roles = [str(r) for r in entity.get("roles") or []]
                vcard = ((entity.get("vcardArray") or [None, []])[1] or [])
                name = ""
                for item in vcard:
                    if isinstance(item, list) and item and item[0] == "fn":
                        name = str(item[-1])
                if name:
                    reg["entity" if "registrar" not in roles else "registrar"] = name
            out["rdap"] = reg
        else:
            out["rdap"] = f"status {response.status}"
    except Exception as exc:  # noqa: BLE001
        out["rdap"] = f"unavailable: {exc}"
    try:
        response = _http(context, f"https://ipwho.is/{ip}", timeout)
        data = response.json() if response.ok else {}
        out["geo"] = {
            "country": data.get("country"), "city": data.get("city"),
            "isp": data.get("connection", {}).get("isp") if isinstance(data.get("connection"), dict) else data.get("isp"),
            "org": data.get("connection", {}).get("org") if isinstance(data.get("connection"), dict) else None,
        }
    except Exception as exc:  # noqa: BLE001
        out["geo"] = f"unavailable: {exc}"
    if abuse_key:
        try:
            client = HttpClient(timeout=timeout, headers={"Key": abuse_key,
                                                          "Accept": "application/json"})
            response = client.get(f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}")
            data = response.json() if response.ok else {}
            out["abuseipdb"] = {
                "score": (data.get("data") or {}).get("abuseScore"),
                "total_reports": (data.get("data") or {}).get("totalReports"),
            }
        except Exception as exc:  # noqa: BLE001
            out["abuseipdb"] = f"unavailable: {exc}"
    return out


# ── URL ─────────────────────────────────────────────────────────────────────


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


def osint_url(context: Any, url: str) -> dict[str, Any]:
    timeout, _a, _s, _ = _settings(context)
    url = (url or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError(f"not an http(s) url: {url!r}")
    out: dict[str, Any] = {"target": url, "kind": "url", "chain": []}
    current = url
    seen: set[str] = set()
    handlers: list[urllib.request.BaseHandler] = [_NoRedirect()]
    proxy_handler = default_proxy_handler()
    if proxy_handler is not None:
        handlers.append(proxy_handler)
    opener = urllib.request.build_opener(*handlers)
    try:
        for _ in range(6):
            if current in seen or not current:
                break
            seen.add(current)
            request = urllib.request.Request(current, method="GET")
            try:
                with opener.open(request, timeout=timeout) as response:
                    entry = {"url": current, "status": response.status}
                    location = response.headers.get("Location")
                    if location:
                        entry["location"] = urllib.parse.urljoin(current, location)
                    out["chain"].append(entry)
                    if response.status in (301, 302, 303, 307, 308):
                        current = urllib.parse.urljoin(current, location or "")
                        continue
                    out["final"] = entry
                    out["final"]["content_type"] = response.headers.get("Content-Type", "")
                    out["final"]["server"] = response.headers.get("Server", "")
                    out["final"]["length"] = response.headers.get("Content-Length", "")
                    body = response.read(4096)
                    out["final"]["body_start"] = body[:500].decode("utf-8", "replace")
                    break
            except urllib.error.HTTPError as exc:
                out["chain"].append({"url": current, "status": exc.code,
                                     "error": str(exc.reason)[:120]})
                break
            except Exception as exc:  # noqa: BLE001
                out["chain"].append({"url": current, "error": f"{type(exc).__name__}: {exc}"})
                break
    finally:
        pass
    # TLS certificate (only for https endpoints)
    if (out.get("chain") or [{}])[0].get("url", url).startswith("https://"):
        host = parsed.hostname or ""
        port = parsed.port or 443
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, port), timeout=min(8.0, timeout)) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as tls:
                    cert = tls.getpeercert()
            out["tls"] = {
                "subject": dict(x[0] for x in cert.get("subject", ())),
                "issuer": dict(x[0] for x in cert.get("issuer", ())),
                "not_before": cert.get("notBefore", ""),
                "not_after": cert.get("notAfter", ""),
                "san": cert.get("subjectAltName", ()),
            }
        except Exception as exc:  # noqa: BLE001
            out["tls"] = f"unavailable: {exc}"
    return out


# ── email (passive only) ─────────────────────────────────────────────────────


def osint_email(context: Any, email: str) -> dict[str, Any]:
    timeout, _a, _s, _ = _settings(context)
    email = (email or "").strip().lower()
    match = re.fullmatch(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)", email)
    if not match:
        raise ToolError(f"not an email address: {email!r}")
    local, domain = match.groups()
    out: dict[str, Any] = {"target": email, "kind": "email",
                           "note": "passive DNS only — no SMTP probing, no account checks"}
    try:
        mx = dns_query(domain, "MX", timeout=min(4.0, timeout))
        out["mx"] = mx
    except ToolError:
        out["mx"] = []
    try:
        spf = dns_query(domain, "SPF", timeout=min(4.0, timeout))
        out["spf"] = spf[0] if spf else None
    except ToolError:
        out["spf"] = None
    try:
        dmarc = dns_query(f"_dmarc.{domain}", "TXT", timeout=min(4.0, timeout))
        out["dmarc"] = dmarc[0] if dmarc else None
    except ToolError:
        out["dmarc"] = None
    if not out["mx"]:
        out["deliverable"] = "no MX — mail for this domain is likely undeliverable"
    else:
        out["deliverable"] = "MX present; active deliverability was intentionally not probed"
    return out


# ── report: auto-detect + compose ────────────────────────────────────────────


def osint_report(context: Any, target: str) -> dict[str, Any]:
    target = (target or "").strip()
    if not target:
        raise ToolError("osint_report needs a target")
    if target.startswith(("http://", "https://")):
        return osint_url(context, target)
    if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", target):
        return osint_email(context, target)
    try:
        ipaddress.ip_address(target)
        return osint_ip(context, target)
    except ValueError:
        pass
    if re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", target):
        return osint_domain(context, target)
    raise ToolError(f"unrecognized target kind: {target!r} (domain | ip | url | email)")


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "osint_report",
        description=(
            "Consolidated OSINT report, auto-detects the target: domain (DNS+RDAP+"
            "cert-transparency), IP (PTR+RDAP+geo+optional abuse score), URL (redirect "
            "chain+headers+TLS cert), or email (MX/SPF/DMARC, passive only)."
        ),
        capability=Capability.NET_OUT,
        parameters={"target": "str — domain, IP, URL, or email"},
    )
    def osint_report_tool(target: str) -> dict[str, Any]:
        return osint_report(context, target)

    @registry.register(
        "osint_domain",
        description="Domain bundle: DNS records, RDAP registration, certificate transparency.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str"},
    )
    def osint_domain_tool(domain: str) -> dict[str, Any]:
        return osint_domain(context, domain)

    @registry.register(
        "osint_ip",
        description="IP bundle: reverse DNS, RDAP allocation, geo/ISP, optional AbuseIPDB score.",
        capability=Capability.NET_OUT,
        parameters={"ip": "str"},
    )
    def osint_ip_tool(ip: str) -> dict[str, Any]:
        return osint_ip(context, ip)

    @registry.register(
        "osint_url",
        description="URL bundle: redirect chain, response headers, TLS certificate details.",
        capability=Capability.NET_OUT,
        parameters={"url": "str"},
    )
    def osint_url_tool(url: str) -> dict[str, Any]:
        return osint_url(context, url)

    @registry.register(
        "osint_email",
        description="Email domain checks — passive DNS only: MX, SPF, DMARC. No SMTP probing.",
        capability=Capability.NET_OUT,
        parameters={"email": "str"},
    )
    def osint_email_tool(email: str) -> dict[str, Any]:
        return osint_email(context, email)
