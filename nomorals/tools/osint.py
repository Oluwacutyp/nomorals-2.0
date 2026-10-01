"""OSINT toolkit: read-only intelligence from public sources.

Every source here is public by definition: DNS (direct and DNS-over-HTTPS),
RDAP registries, certificate transparency logs, IP allocation data, TLS
handshakes, and the keyless threat-intel feeds urlscan.io, urlhaus, and
ThreatFox plus the hackertarget free utilities. Optional depth comes from
the operator's own keys (AbuseIPDB, Shodan).

``osint_report`` auto-detects the target kind (domain / IP / URL / email)
and composes one consolidated report; ``osint_sweep`` fans out to every
applicable source with per-source timing and error isolation, which is what
the main AI and sub-agents mostly call.
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

__all__ = [
    "osint_domain", "osint_ip", "osint_url", "osint_email",
    "osint_dns", "osint_threat", "osint_sweep", "osint_report",
    "register",
]


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


# ── target classification (shared by report + sweep) ──────────────────────────


def _classify(target: str) -> str:
    if target.startswith(("http://", "https://")):
        return "url"
    if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", target):
        return "email"
    try:
        ipaddress.ip_address(target)
        return "ip"
    except ValueError:
        pass
    if re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", target):
        return "domain"
    raise ToolError(f"unrecognized target kind: {target!r} (domain | ip | url | email)")


def _sweep_domain(kind: str, target: str) -> str:
    """The domain-shaped term to feed host-oriented feeds for a given kind."""
    if kind == "domain":
        return target.lower().rstrip(".")
    if kind == "email":
        return target.split("@", 1)[1].lower()
    if kind == "url":
        return (urllib.parse.urlparse(target).hostname or "").lower()
    return ""


# ── keyless enrichment sources ───────────────────────────────────────────────

_DOHTYPES = ("A", "AAAA", "MX", "TXT", "NS", "CAA", "SOA")
_HT_SLEEP = 0.6  # politeness pause between hackertarget calls


def _doh_lookup(name: str, rtype: str, timeout: float) -> tuple[list[str], str]:
    """DNS-over-HTTPS with provider fallback. Returns (records, provider host).

    Raises ToolError only when both providers fail for this record type.
    """
    last: Exception | None = None
    for base in ("https://cloudflare-dns.com/dns-query",
                 "https://dns.google/resolve"):
        url = f"{base}?name={urllib.parse.quote(name)}&type={rtype}"
        try:
            response = HttpClient(
                timeout=timeout,
                headers={"Accept": "application/dns-json"},
            ).get(url)
            if not response.ok:
                last = ToolError(f"{base} -> HTTP {response.status}")
                continue
            records: list[str] = []
            for answer in (response.json().get("Answer") or []):
                value = str(answer.get("data", "")).strip()
                if rtype == "TXT":
                    value = value.strip('"')
                if value:
                    records.append(value)
            return records, base.split("//", 1)[1].split("/", 1)[0]
        except Exception as exc:  # noqa: BLE001 - fall through to next provider
            last = exc
    raise ToolError(f"DoH {rtype} lookup for {name!r} failed: {last}")


def osint_dns(context: Any, domain: str) -> dict[str, Any]:
    """DNS record enumeration over DNS-over-HTTPS (keyless, fail-soft per type)."""
    timeout, *_ = _settings(context)
    src_timeout = min(timeout, 8.0)
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or " " in domain:
        raise ToolError(f"bad domain {domain!r}")
    records: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    providers: set[str] = set()
    for rtype in _DOHTYPES:
        try:
            recs, provider = _doh_lookup(domain, rtype, src_timeout)
            records[rtype] = recs
            providers.add(provider)
        except ToolError as exc:
            records[rtype] = []
            errors[rtype] = str(exc)
    out: dict[str, Any] = {
        "target": domain, "kind": "dns",
        "records": records, "providers": sorted(providers),
    }
    if errors:
        out["errors"] = errors
    return out


def _urlscan_search(target: str, kind: str, timeout: float) -> dict[str, Any]:
    """urlscan.io search API — no key needed for search."""
    if kind == "url":
        query = f'page.url:"{target}"'
    elif kind == "ip":
        query = f"ip:{target}"
    else:
        query = f"domain:{target}"
    url = f"https://urlscan.io/api/v1/search/?q={urllib.parse.quote(query)}"
    response = HttpClient(timeout=timeout).get(url)
    data = response.json() if response.ok else {}
    scans: list[dict[str, Any]] = []
    for item in (data.get("results") or [])[:10]:
        page = item.get("page") or {}
        overall = (item.get("verdicts") or {}).get("overall") or {}
        scans.append({
            "url": page.get("url"), "domain": page.get("domain"),
            "ip": page.get("ip"), "country": page.get("country"),
            "time": (item.get("task") or {}).get("time"),
            "malicious": bool(overall.get("malicious")),
            "score": overall.get("score"),
            "uuid": item.get("_id"),
        })
    return {"total": data.get("total", 0), "scans": scans}


def _urlhaus_host(host: str, timeout: float) -> dict[str, Any]:
    """urlhaus host feed — POST, no key."""
    response = HttpClient(timeout=timeout).post_form(
        "https://urlhaus-api.abuse.ch/v1/host/", {"host": host})
    data = response.json() if response.ok else {}
    status = str(data.get("query_status", ""))
    if status != "ok":
        return {"query_status": status}
    urls = data.get("urls") or []
    tags: set[str] = set()
    sample: list[dict[str, Any]] = []
    for entry in urls[:8]:
        sample.append({
            "url": entry.get("url"), "threat": entry.get("threat"),
            "url_status": entry.get("url_status"),
        })
        for tag in entry.get("tags") or []:
            tags.add(str(tag))
    return {
        "query_status": "ok", "host": data.get("host"),
        "firstseen": data.get("firstseen"), "lastseen": data.get("lastseen"),
        "url_count": data.get("url_count", len(urls)),
        "sample_urls": sample, "tags": sorted(tags),
    }


def _urlhaus_url(url: str, timeout: float) -> dict[str, Any]:
    """urlhaus url feed — POST, no key."""
    response = HttpClient(timeout=timeout).post_form(
        "https://urlhaus-api.abuse.ch/v1/url/", {"url": url})
    data = response.json() if response.ok else {}
    status = str(data.get("query_status", ""))
    if status != "ok":
        return {"query_status": status}
    return {
        "query_status": "ok",
        "urlhaus_reference": data.get("urlhaus_reference"),
        "threat": data.get("threat"),
        "tags": data.get("tags") or [],
        "url_status": data.get("url_status"),
        "firstseen": data.get("firstseen"),
    }


def _threatfox_search(term: str, timeout: float) -> dict[str, Any]:
    """ThreatFox IOC lookup — POST query get_iocs, no key."""
    response = HttpClient(timeout=timeout).post_json(
        "https://threatfox-api.abuse.ch/api/v1/",
        {"query": "get_iocs", "search_term": term})
    data = response.json() if response.ok else {}
    status = str(data.get("query_status", ""))
    if status != "ok":
        return {"query_status": status}
    iocs: list[dict[str, Any]] = []
    for item in (data.get("data") or [])[:15]:
        iocs.append({
            "ioc": item.get("ioc"),
            "ioc_type": item.get("ioc_type_desc"),
            "threat_type": item.get("threat_type_desc") or item.get("threat_type"),
            "malware": item.get("malware"),
            "confidence": item.get("confidence_level"),
            "first_seen": item.get("first_seen"),
            "last_seen": item.get("last_seen"),
            "tags": item.get("tags") or [],
        })
    return {"query_status": "ok", "count": len(data.get("data") or []), "iocs": iocs}


def _hackertarget(endpoint: str, query: str, timeout: float) -> str:
    """One hackertarget free-API call. Raises ToolError on failure/rate-limit."""
    url = f"https://api.hackertarget.com/{endpoint}/?q={urllib.parse.quote(query)}"
    response = HttpClient(timeout=timeout).get(url)
    if not response.ok:
        raise ToolError(f"hackertarget {endpoint} -> HTTP {response.status}")
    text = response.text.strip()
    if text.lower().startswith("error"):
        raise ToolError(f"hackertarget {endpoint}: {text[:120]}")
    return text[:4000]


def _hackertarget_bundle(kind: str, target: str, timeout: float) -> dict[str, Any]:
    """Minimal hackertarget call set per target kind, spaced by a politeness delay."""
    out: dict[str, Any] = {}
    if kind == "domain":
        calls = [("dnslookup", target), ("hostsearch", target)]
    elif kind == "ip":
        calls = [("reverseiplookup", target)]
    elif kind == "url":
        calls = [("httpheaders", target), ("pagelinks", target)]
    else:
        calls = [("dnslookup", target)]
    for index, (endpoint, query) in enumerate(calls):
        if index:
            time.sleep(_HT_SLEEP)
        try:
            raw = _hackertarget(endpoint, query, timeout)
        except ToolError as exc:
            out[endpoint] = f"unavailable: {exc}"
            continue
        if endpoint == "hostsearch":
            entries: list[dict[str, str]] = []
            for line in raw.splitlines():
                parts = line.split(",", 1)
                if parts and parts[0].strip():
                    entries.append({
                        "ip": parts[0].strip(),
                        "host": parts[1].strip() if len(parts) > 1 else "",
                    })
            out[endpoint] = entries[:25]
        else:
            out[endpoint] = raw
    return out


def _ipapi(ip: str, timeout: float) -> dict[str, Any]:
    """ip-api.com free geo/ASN/ISP — no key (free tier, short timeout)."""
    fields = "status,message,country,regionName,city,lat,lon,isp,org,as,asn,reverse,query"
    url = f"http://ip-api.com/json/{ip}?fields={fields}"
    response = HttpClient(timeout=timeout).get(url)
    data = dict(response.json()) if response.ok else {}
    if data.get("status") != "success":
        return {"status": "fail", "message": data.get("message", "lookup failed")}
    data.pop("status", None)
    return data


def osint_threat(context: Any, target: str) -> dict[str, Any]:
    """Threat-intel bundle across the keyless feeds: urlscan.io, urlhaus, ThreatFox.

    Fail-soft per feed — a dead feed shows as an "unavailable: ..." string.
    """
    timeout, *_ = _settings(context)
    target = (target or "").strip()
    if not target:
        raise ToolError("osint_threat needs a target")
    kind = _classify(target)
    domain = _sweep_domain(kind, target)
    src_timeout = min(timeout, 8.0)
    out: dict[str, Any] = {"target": target, "kind": kind}
    feeds: list[tuple[str, Any]] = [
        ("urlscan", lambda: _urlscan_search(target, kind, src_timeout)),
        ("urlhaus", lambda: (_urlhaus_url(target, src_timeout) if kind == "url"
                             else _urlhaus_host(domain or target, src_timeout))),
        ("threatfox", lambda: _threatfox_search(
            target if kind in {"ip", "url"} else domain or target, src_timeout)),
    ]
    for name, fn in feeds:
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 - fail-soft per feed
            out[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


# ── sweep: fan-out with per-source isolation ──────────────────────────────────


def _run_source(name: str, fn: Any) -> dict[str, Any]:
    """Run one source, isolating errors and capturing wall time. Never raises."""
    started = time.monotonic()
    try:
        data = fn()
    except Exception as exc:  # noqa: BLE001 - per-source isolation
        return {"ok": False, "seconds": round(time.monotonic() - started, 2),
                "error": f"{type(exc).__name__}: {exc}", "data": None}
    return {"ok": True, "seconds": round(time.monotonic() - started, 2),
            "error": None, "data": data}


def _sweep_sections(context: Any, kind: str, target: str,
                    domain: str, timeout: float) -> dict[str, dict[str, Any]]:
    """All applicable keyless sources for a target kind, each isolated."""
    src_timeout = min(timeout, 8.0)
    jobs: list[tuple[str, Any]] = []
    if kind in {"domain", "email"}:
        jobs = [
            ("dns_doh", lambda: osint_dns(context, domain)),
            ("urlscan", lambda: _urlscan_search(domain, "domain", src_timeout)),
            ("urlhaus", lambda: _urlhaus_host(domain, src_timeout)),
            ("threatfox", lambda: _threatfox_search(domain, src_timeout)),
            ("hackertarget", lambda: _hackertarget_bundle("domain", domain, src_timeout)),
        ]
    elif kind == "ip":
        jobs = [
            ("ipapi", lambda: _ipapi(target, src_timeout)),
            ("urlscan", lambda: _urlscan_search(target, "ip", src_timeout)),
            ("urlhaus", lambda: _urlhaus_host(target, src_timeout)),
            ("threatfox", lambda: _threatfox_search(target, src_timeout)),
            ("hackertarget", lambda: _hackertarget_bundle("ip", target, src_timeout)),
        ]
    elif kind == "url":
        jobs = [
            ("urlscan", lambda: _urlscan_search(target, "url", src_timeout)),
            ("urlhaus", lambda: _urlhaus_url(target, src_timeout)),
            ("threatfox", lambda: _threatfox_search(target, src_timeout)),
            ("hackertarget", lambda: _hackertarget_bundle("url", target, src_timeout)),
        ]
    return {name: _run_source(name, fn) for name, fn in jobs}


def osint_sweep(context: Any, target: str) -> dict[str, Any]:
    """Unified sweep: auto-classifies the target, runs the base bundle plus every
    applicable keyless source, merges into one report. One dead source never
    kills the sweep — failures are isolated per source with timings recorded.
    """
    timeout, *_ = _settings(context)
    target = (target or "").strip()
    if not target:
        raise ToolError("osint_sweep needs a target")
    kind = _classify(target)
    base = {"domain": osint_domain, "ip": osint_ip,
            "url": osint_url, "email": osint_email}[kind](context, target)
    sections = _sweep_sections(context, kind, target, _sweep_domain(kind, target), timeout)
    ok = sum(1 for section in sections.values() if section["ok"])
    return {
        "target": target, "kind": kind, "base": base, "sources": sections,
        "sources_ok": ok, "sources_failed": len(sections) - ok,
    }


# ── report: auto-detect + compose ────────────────────────────────────────────


def osint_report(context: Any, target: str) -> dict[str, Any]:
    target = (target or "").strip()
    if not target:
        raise ToolError("osint_report needs a target")
    kind = _classify(target)
    out = {"domain": osint_domain, "ip": osint_ip,
           "url": osint_url, "email": osint_email}[kind](context, target)
    # Sweep enrichment: keyless intel sections, each fail-soft and isolated.
    timeout, *_ = _settings(context)
    sections = _sweep_sections(context, kind, target, _sweep_domain(kind, target), timeout)
    out["intel"] = {name: section["data"] for name, section in sections.items()
                    if section["ok"]}
    failed = {name: section["error"] for name, section in sections.items()
              if not section["ok"]}
    if failed:
        out["intel_errors"] = failed
    out["intel_seconds"] = {name: section["seconds"] for name, section in sections.items()}
    return out


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "osint_report",
        description=(
            "Consolidated OSINT report, auto-detects the target: domain (DNS+RDAP+"
            "cert-transparency), IP (PTR+RDAP+geo+optional abuse score), URL (redirect "
            "chain+headers+TLS cert), or email (MX/SPF/DMARC, passive only). Enriched "
            "with keyless intel sections: DNS-over-HTTPS records, urlscan.io, "
            "urlhaus, ThreatFox, hackertarget, ip-api — each fail-soft."
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

    @registry.register(
        "osint_dns",
        description=(
            "DNS record enumeration over DNS-over-HTTPS (Cloudflare, fallback Google): "
            "A, AAAA, MX, TXT, NS, CAA, SOA. Keyless, fail-soft per record type."
        ),
        capability=Capability.NET_OUT,
        parameters={"domain": "str"},
    )
    def osint_dns_tool(domain: str) -> dict[str, Any]:
        return osint_dns(context, domain)

    @registry.register(
        "osint_threat",
        description=(
            "Threat-intel bundle across the keyless feeds: urlscan.io search, "
            "urlhaus host/URL, ThreatFox IOCs. Fail-soft per feed."
        ),
        capability=Capability.NET_OUT,
        parameters={"target": "str — domain, IP, URL, or email"},
    )
    def osint_threat_tool(target: str) -> dict[str, Any]:
        return osint_threat(context, target)

    @registry.register(
        "osint_sweep",
        description=(
            "Unified OSINT sweep: auto-classifies the target (domain/ip/url/email), "
            "runs the base bundle plus every applicable keyless source "
            "(DNS-over-HTTPS, urlscan, urlhaus, ThreatFox, hackertarget, ip-api) "
            "with per-source timings and isolated errors."
        ),
        capability=Capability.NET_OUT,
        parameters={"target": "str — domain, IP, URL, or email"},
    )
    def osint_sweep_tool(target: str) -> dict[str, Any]:
        return osint_sweep(context, target)
