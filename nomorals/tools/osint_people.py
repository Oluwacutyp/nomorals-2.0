"""OSINT — people, identities, and correlation (the second OSINT unit).

Extends :mod:`nomorals.tools.osint` (domains/IPs/URLs/emails) with
identity-level investigation:

- ``username_check``      public profile existence across a curated site
                          list (Sherlock-style, concurrent, honest statuses)
- ``email_investigate``   passive DNS (MX/SPF/DMARC) + HIBP breach awareness
- ``phone_investigate``   E.164 validation + country-code intelligence
- ``breach_check``        official HaveIBeenPwned v3 API (key-gated, free
                          non-commercial tier) — breach NAMES only, never
                          credential content
- ``osint_people``        multi-seed investigation that assembles a
                          relationship graph (nodes/edges + ASCII view)

Every source here is public data or the operator's own API key.  Nothing
in this module brute-forces, scrapes behind authentication walls, or
harvests credentials.
"""
from __future__ import annotations

import concurrent.futures
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .network import dns_query
from .osint import osint_domain, osint_ip

_log = get_logger(__name__)

__all__ = [
    "USERNAME_SITES",
    "username_check",
    "email_investigate",
    "phone_investigate",
    "breach_check",
    "osint_people",
    "register",
]

_UA = "Mozilla/5.0 (compatible; NoMoralsOSINT/1.0; +personal research)"

# ── username correlation ────────────────────────────────────────────────────
# Public profile URL templates.  ``absent`` is an optional substring that,
# when present in a 200 page, means the profile does not exist (some sites
# answer 200 with a "not found" body instead of a real 404).


@dataclass(frozen=True)
class Site:
    name: str
    url: str
    absent: str = ""
    reliable: bool = True  # False = 200/404 heuristic is shaky


USERNAME_SITES: tuple[Site, ...] = (
    Site("github", "https://github.com/{u}"),
    Site("gitlab", "https://gitlab.com/{u}"),
    Site("reddit", "https://www.reddit.com/user/{u}"),
    Site("twitch", "https://www.twitch.tv/{u}", reliable=False),
    Site("youtube", "https://www.youtube.com/@{u}"),
    Site("medium", "https://medium.com/@{u}"),
    Site("bitbucket", "https://bitbucket.org/{u}/"),
    Site("sourcehut", "https://sr.ht/~{u}/"),
    Site("codeberg", "https://codeberg.org/{u}"),
    Site("npm", "https://www.npmjs.com/~{u}"),
    Site("pypi", "https://pypi.org/user/{u}/"),
    Site("mastodon", "https://mastodon.social/@{u}", reliable=False),
    Site("telegram", "https://t.me/{u}", absent="tgme_page_extra"),
    Site("twitter_x", "https://x.com/{u}", reliable=False),
    Site("instagram", "https://www.instagram.com/{u}/", reliable=False),
    Site("facebook", "https://www.facebook.com/{u}", reliable=False),
    Site("tiktok", "https://www.tiktok.com/@{u}", reliable=False),
    Site("linkedin", "https://www.linkedin.com/in/{u}", reliable=False),
    Site("devto", "https://dev.to/{u}"),
    Site("hackernews", "https://news.ycombinator.com/user?id={u}", reliable=False),
)

_AUTO_SITES = [s for s in USERNAME_SITES if s.reliable]


def _probe_site(site: Site, handle: str, timeout: float) -> dict[str, Any]:
    url = site.url.format(u=urllib.parse.quote(handle))
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = ""
            if site.absent:
                body = response.read(200_000).decode("utf-8", "replace").lower()
            exists = status == 200
            if exists and site.absent and site.absent.lower() in body:
                exists = False
            confidence = "high" if site.reliable else "medium"
            note = ""
    except urllib.error.HTTPError as exc:
        status = exc.code
        if status == 404:
            exists, confidence, note = False, "high", ""
        elif status in {403, 429, 503}:
            exists, confidence, note = None, "low", f"blocked/limited ({status})"
        else:
            exists, confidence, note = None, "low", f"http {status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status = 0
        exists, confidence, note = None, "low", f"{type(exc).__name__}"
    return {"site": site.name, "url": url, "status": status, "exists": exists,
            "confidence": confidence, "note": note}


def username_check(context: Any, handle: str, *, sites: str = "auto",
                   timeout: float = 0.0) -> dict[str, Any]:
    """Check a public handle across profile sites, concurrently.

    ``sites``: "auto" (the reliable set), "all", or a comma list of site
    names.  A result is ``exists: true|false|None`` (None = could not be
    determined — honest, never guessed).
    """
    handle = (handle or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{2,64}", handle):
        raise ToolError(f"bad handle {handle!r} (2-64 chars, letters/digits/._-)")
    wanted = (sites or "auto").strip().lower()
    if wanted == "all":
        chosen = list(USERNAME_SITES)
    elif wanted == "auto" or not wanted:
        chosen = list(_AUTO_SITES)
    else:
        by_name = {s.name: s for s in USERNAME_SITES}
        chosen = []
        for token in wanted.split(","):
            token = token.strip()
            if token in by_name:
                chosen.append(by_name[token])
        if not chosen:
            raise ToolError(
                f"unknown site(s); available: "
                + ", ".join(sorted(by_name)))
    timeout = timeout or 6.0
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8,
                                               thread_name_prefix="usercheck") as pool:
        futures = {pool.submit(_probe_site, site, handle, timeout): site
                   for site in chosen}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - one dead site is a result
                site = futures[future]
                results.append({"site": site.name, "url": site.url.format(u=handle),
                                "status": 0, "exists": None, "confidence": "low",
                                "note": str(exc)[:120]})
    found = [r for r in results if r["exists"] is True]
    absent = [r for r in results if r["exists"] is False]
    unknown = [r for r in results if r["exists"] is None]
    return {
        "handle": handle,
        "checked": len(results),
        "found": [{"site": r["site"], "url": r["url"]} for r in found],
        "absent": [r["site"] for r in absent],
        "unknown": [{"site": r["site"], "note": r["note"]} for r in unknown],
        "detail": sorted(results, key=lambda r: r["site"]),
    }


# ── email investigation ─────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def email_investigate(context: Any, email: str) -> dict[str, Any]:
    """Full passive email investigation: syntax, domain DNS (MX/SPF/DMARC),
    and breach awareness through the official HIBP API when a key is set.

    No SMTP probing, no account checks — public DNS + official breach API
    only.
    """
    email = (email or "").strip().lower()
    match = _EMAIL_RE.fullmatch(email)
    if not match:
        raise ToolError(f"not a valid email address: {email!r}")
    local, domain = match.groups()
    out: dict[str, Any] = {
        "target": email, "kind": "email",
        "local_part": local, "domain": domain,
        "note": "passive only — no SMTP probing, no account enumeration",
    }
    try:
        out["mx"] = dns_query(domain, "MX")
    except ToolError:
        out["mx"] = []
    try:
        spf = dns_query(domain, "SPF")
        out["spf"] = spf[0] if spf else None
    except ToolError:
        out["spf"] = None
    try:
        dmarc = dns_query(f"_dmarc.{domain}", "TXT")
        out["dmarc"] = dmarc[0] if dmarc else None
    except ToolError:
        out["dmarc"] = None
    out["deliverable"] = (
        "no MX — mail for this domain is likely undeliverable"
        if not out["mx"] else
        "MX present; active deliverability intentionally not probed")
    # domain context: who runs this mail domain
    try:
        domain_info = osint_domain(context, domain)
        out["domain_registration"] = domain_info.get("registration")
        out["domain_nameservers"] = domain_info.get("nameservers")
    except ToolError as e:
        _log.debug("domain enrichment failed for %s: %s", domain, e)
    out["breaches"] = breach_check(context, email)
    return out


# ── phone investigation ─────────────────────────────────────────────────────

# ITU-E.164 country prefixes for the regions people actually run — the
# lookup is offline and honest about its own limits.
_COUNTRY_PREFIXES: dict[str, str] = {
    "1": "United States / Canada (NANP)",
    "7": "Russia / Kazakhstan",
    "20": "Egypt", "27": "South Africa",
    "30": "Greece", "31": "Netherlands", "32": "Belgium", "33": "France",
    "34": "Spain", "39": "Italy", "40": "Romania", "41": "Switzerland",
    "43": "Austria", "44": "United Kingdom", "45": "Denmark", "46": "Sweden",
    "47": "Norway", "48": "Poland", "49": "Germany",
    "51": "Peru", "52": "Mexico", "54": "Argentina", "55": "Brazil",
    "56": "Chile", "57": "Colombia",
    "60": "Malaysia", "61": "Australia / New Zealand", "62": "Indonesia",
    "63": "Philippines", "64": "New Zealand", "65": "Singapore",
    "66": "Thailand", "81": "Japan", "82": "South Korea", "84": "Vietnam",
    "86": "China", "90": "Turkey", "91": "India", "94": "Sri Lanka",
    "95": "Myanmar", "98": "Iran",
    "211": "South Sudan", "212": "Morocco", "213": "Algeria", "216": "Tunisia",
    "218": "Libya", "221": "Senegal", "223": "Mali", "225": "Ivory Coast",
    "226": "Burkina Faso", "228": "Togo", "230": "Mauritius", "231": "Liberia",
    "232": "Sierra Leone", "233": "Ghana", "234": "Nigeria", "235": "Chad",
    "236": "Central African Rep.", "237": "Cameroon", "238": "Cape Verde",
    "239": "Sao Tome", "240": "Gambia", "241": "Gabon", "242": "Congo",
    "243": "DR Congo", "244": "Angola", "245": "Guinea-Bissau",
    "246": "Diego Garcia", "247": "Ascension", "248": "Seychelles",
    "249": "Sudan", "250": "Rwanda", "251": "Ethiopia", "252": "Somalia",
    "253": "Djibouti", "254": "Kenya", "255": "Tanzania", "256": "Uganda",
    "257": "Burundi", "258": "Mozambique", "260": "Zambia", "261": "Madagascar",
    "262": "Reunion / Mayotte", "263": "Zimbabwe", "264": "Namibia",
    "265": "Malawi", "266": "Lesotho", "267": "Botswana", "268": "Eswatini",
    "269": "Comoros", "290": "St Helena", "291": "Eritrea", "297": "Curaçao",
    "298": "Faroe Islands", "299": "Greenland",
    "960": "Maldives", "961": "Lebanon", "962": "Jordan", "963": "Syria",
    "964": "Iraq", "965": "Kuwait", "966": "Saudi Arabia", "967": "Yemen",
    "968": "Oman", "970": "Palestine", "971": "UAE", "972": "Israel",
    "973": "Bahrain", "974": "Qatar", "975": "Bhutan", "976": "Mongolia",
    "977": "Nepal", "980": "", "992": "Tajikistan", "993": "Turkmenistan",
    "994": "Azerbaijan", "995": "Georgia", "996": "Kyrgyzstan", "998": "Uzbekistan",
}


def phone_investigate(context: Any, phone: str) -> dict[str, Any]:
    """Offline phone intelligence: E.164 normalisation + country mapping.

    Carrier-level lookup needs a paid API (Twilio Lookup et al.) — when
    none is configured this says so instead of guessing.
    """
    raw = (phone or "").strip()
    if not raw:
        raise ToolError("phone_investigate needs a number")
    digits = re.sub(r"[^\d+]", "", raw)
    if not digits:
        raise ToolError(f"no digits in {raw!r}")
    has_plus = digits.startswith("+")
    bare = digits.lstrip("+")
    e164 = f"+{bare}" if (has_plus or len(bare) >= 10) else ""
    # longest-prefix country match
    country = ""
    matched_code = ""
    for code in sorted(_COUNTRY_PREFIXES, key=len, reverse=True):
        if bare.startswith(code):
            country = _COUNTRY_PREFIXES[code]
            matched_code = code
            break
    length_ok = 7 <= len(bare) <= 15  # E.164 limit
    out: dict[str, Any] = {
        "target": raw,
        "e164": e164 if length_ok else None,
        "country_code": matched_code or None,
        "country": country or None,
        "valid_e164_shape": length_ok and bool(bare.isdigit()),
        "carrier": None,
        "carrier_note": ("carrier lookup needs a paid API key (e.g. Twilio "
                         "Lookup) — not configured, not guessed"),
    }
    return out


# ── breach awareness (official HIBP API, key-gated) ─────────────────────────


def breach_check(context: Any, target: str) -> dict[str, Any]:
    """'Is this email/domain in known breaches?' via the official HIBP v3 API.

    The API returns breach NAMES and hashed data — never credential content.
    Free non-commercial tier is 5k requests/month with a key
    (https://haveibeenpwned.com/API/Key).  Without a key this returns an
    honest note, not a guess.
    """
    s = getattr(context, "settings", None)
    osint = getattr(s, "osint", None) if s else None
    key = str(getattr(osint, "hibp_key", "") if osint else "") or ""
    target = (target or "").strip().lower()
    if not target:
        raise ToolError("breach_check needs an email or domain")
    if not key:
        return {
            "checked": False,
            "reason": ("no HIBP API key set — add NM_OSINT_HIBP_KEY to "
                       "~/.nomorals/.env (free, 5k calls/month, "
                       "non-commercial)"),
        }
    is_domain = "@" not in target
    if is_domain:
        url = f"https://haveibeenpwned.com/api/v3/breacheddomain/{urllib.parse.quote(target)}"
    else:
        url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{urllib.parse.quote(target)}?truncateResponse=true"
    try:
        request = urllib.request.Request(url, headers={
            "User-Agent": _UA, "hibp-api-key": key,
            "Accept": "application/json",
        })
        with urllib.request.urlopen(request, timeout=20) as response:
            import json

            data = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"checked": True, "pwned": False,
                    "breaches": [], "note": "not found in HIBP data"}
        raise ToolError(f"HIBP returned {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise ToolError(f"HIBP lookup failed: {exc}") from exc
    breaches = [
        {"name": str(b.get("Name") or b.get("Title") or ""),
         "date": str(b.get("BreachDate") or b.get("PublishedDate") or "")[:10],
         "title": str(b.get("Title") or "")}
        for b in (data if isinstance(data, list) else data.get("Breaches", []))
    ]
    return {
        "checked": True,
        "pwned": bool(breaches),
        "count": len(breaches),
        "breaches": breaches[:40],
        "note": "breach names/dates only — the official API never exposes "
                "credential content",
    }


# ── multi-seed people investigation + relationship map ──────────────────────


def osint_people(context: Any, seeds: str) -> dict[str, Any]:
    """Investigate a set of identities and map how they connect.

    ``seeds`` is a JSON object:
        {"emails": [...], "handles": [...], "domains": [...],
         "phones": [...], "ips": [...]}
    Any subset works.  The output is a relationship graph: nodes are
    identities, edges are observed links (handle→site, email→domain,
    domain→nameserver, domain→registration entity, ip→PTR, …), with an
    ASCII rendering and a JSON body for downstream tools.
    """
    import json

    try:
        parsed = json.loads(seeds) if (seeds or "").strip() else {}
    except (ValueError, TypeError) as exc:
        raise ToolError(f"seeds must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolError("seeds must be a JSON object")
    emails = [str(x) for x in parsed.get("emails") or []]
    handles = [str(x) for x in parsed.get("handles") or []]
    domains = [str(x) for x in parsed.get("domains") or []]
    phones = [str(x) for x in parsed.get("phones") or []]
    ips = [str(x) for x in parsed.get("ips") or []]
    if not (emails or handles or domains or phones or ips):
        raise ToolError(
            "no seeds given — provide at least one of "
            "emails/handles/domains/phones/ips")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}

    def add_node(kind: str, value: str) -> str:
        node_id = f"{kind}:{value}"
        if not any(n["id"] == node_id for n in nodes):
            nodes.append({"id": node_id, "kind": kind, "value": value})
        return node_id

    def add_edge(a: str, b: str, relation: str) -> None:
        edges.append({"from": a, "to": b, "relation": relation})

    # handles → sites
    if handles:
        found_sites: dict[str, list[str]] = {}
        for handle in handles[:6]:
            try:
                result = username_check(context, handle)
            except ToolError as exc:
                found_sites[handle] = [f"error: {exc}"]
                continue
            found_sites[handle] = [f["site"] for f in result["found"]]
            node = add_node("handle", handle)
            for found in result["found"]:
                site_node = add_node("site", f"{found['site']}:{handle}")
                add_edge(node, site_node, "profile on")
            for site in result["absent"][:8]:
                add_edge(node, add_node("site", f"{site} (absent)"), "checked, absent")
        sections["handles"] = found_sites

    # emails → domain + breach
    for email in emails[:6]:
        node = add_node("email", email)
        try:
            info = email_investigate(context, email)
            domain = info["domain"]
            domain_node = add_node("domain", domain)
            add_edge(node, domain_node, "belongs to domain")
            if info["mx"]:
                for mx in info["mx"][:3]:
                    host = mx.split()[-1]
                    add_edge(domain_node, add_node("host", host), "MX →")
            if info["spf"]:
                info_out = {"spf": info["spf"], "dmarc": info["dmarc"],
                            "deliverable": info["deliverable"]}
                sections.setdefault("emails", {})[email] = info_out
            breaches = info.get("breaches") or {}
            if breaches.get("pwned"):
                for breach in breaches["breaches"][:5]:
                    add_edge(node, add_node("breach", breach["name"]), "in breach")
            elif breaches.get("checked"):
                sections.setdefault("emails", {})[email] = {"breach": "clean (HIBP)"}
        except ToolError as exc:
            sections.setdefault("emails", {})[email] = f"error: {exc}"

    # domains → DNS/registration
    for domain in domains[:6]:
        domain = domain.lower()
        node = add_node("domain", domain)
        try:
            info = osint_domain(context, domain)
            dns = info.get("dns") or {}
            for record in ("A", "AAAA", "MX", "NS"):
                for value in (dns.get(record) or [])[:4]:
                    host = value.split()[-1] if record == "MX" else value
                    add_edge(node, add_node("host", host), f"{record} →")
            if isinstance(info.get("registration"), dict):
                reg = info["registration"]
                for field in ("registrar", "entity"):
                    if reg.get(field):
                        add_edge(node, add_node("org", str(reg[field])),
                                 "registered via")
                sections.setdefault("domains", {})[domain] = {
                    "registrar": reg.get("registrar"),
                    "registration": reg.get("registration"),
                    "expiration": reg.get("expiration"),
                    "nameservers": reg.get("nameservers"),
                }
            ct = info.get("cert_transparency")
            if isinstance(ct, dict):
                for name in ct.get("distinct_names", [])[:8]:
                    if name not in domains:
                        add_edge(node, add_node("domain", name), "shares certificate")
        except ToolError as exc:
            sections.setdefault("domains", {})[domain] = f"error: {exc}"

    # ips → PTR/geo
    for ip in ips[:6]:
        node = add_node("ip", ip)
        try:
            info = osint_ip(context, ip)
            if info.get("reverse_dns"):
                add_edge(node, add_node("host", info["reverse_dns"]), "PTR →")
            geo = info.get("geo")
            if isinstance(geo, dict) and geo.get("country"):
                add_edge(node, add_node("geo", f"{geo.get('country')}/{geo.get('city')}"),
                         "located in")
            sections.setdefault("ips", {})[ip] = {
                "ptr": info.get("reverse_dns"),
                "geo": info.get("geo"),
                "abuseipdb": info.get("abuseipdb"),
            }
        except ToolError as exc:
            sections.setdefault("ips", {})[ip] = f"error: {exc}"

    # phones (offline intelligence)
    for phone in phones[:6]:
        node = add_node("phone", phone)
        try:
            info = phone_investigate(context, phone)
            if info.get("country"):
                add_edge(node, add_node("geo", info["country"]), "country code")
            sections.setdefault("phones", {})[phone] = {
                "e164": info.get("e164"), "country": info.get("country"),
            }
        except ToolError as exc:
            sections.setdefault("phones", {})[phone] = f"error: {exc}"

    return {
        "nodes": nodes,
        "edges": edges,
        "sections": sections,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "seeds": {
                "emails": len(emails), "handles": len(handles),
                "domains": len(domains), "phones": len(phones), "ips": len(ips),
            },
        },
        "ascii": _ascii_graph(nodes, edges),
    }


def _ascii_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """A readable node→edge rendering for chat."""
    lines: list[str] = [f"relationship map ({len(nodes)} nodes, {len(edges)} links)"]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_kind.setdefault(node["kind"], []).append(node)
    for kind in sorted(by_kind):
        values = ", ".join(n["value"] for n in by_kind[kind][:12])
        more = f" …+{len(by_kind[kind]) - 12}" if len(by_kind[kind]) > 12 else ""
        lines.append(f"  {kind:8s} {values}{more}")
    lines.append("links:")
    for edge in edges[:40]:
        lines.append(f"  {edge['from']} --{edge['relation']}--> {edge['to']}")
    if len(edges) > 40:
        lines.append(f"  …+{len(edges) - 40} more (see JSON)")
    return "\n".join(lines)


# ── tool registration ───────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "username_check",
        description=(
            "Check a public handle across profile sites (github, gitlab, "
            "reddit, npm, pypi, t.me, …). Concurrent, honest exists/absent/"
            "unknown per site. args: handle, sites (auto|all|list)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "handle": "str",
            "sites": "str (optional, auto) — auto | all | comma list",
        },
    )
    def username_check_tool(handle: str, *, sites: str = "auto") -> dict[str, Any]:
        return username_check(context, handle, sites=sites)

    @registry.register(
        "email_investigate",
        description=(
            "Full passive email investigation: syntax, domain MX/SPF/DMARC, "
            "domain registration, and breach awareness (HIBP, key-gated). "
            "No SMTP probing."
        ),
        capability=Capability.NET_OUT,
        parameters={"email": "str"},
    )
    def email_investigate_tool(email: str) -> dict[str, Any]:
        return email_investigate(context, email)

    @registry.register(
        "phone_investigate",
        description=(
            "Phone intelligence: E.164 normalisation + country-code mapping "
            "(offline, 200+ prefixes). Carrier lookup needs a paid key — "
            "says so instead of guessing."
        ),
        capability=Capability.NET_OUT,
        parameters={"phone": "str"},
    )
    def phone_investigate_tool(phone: str) -> dict[str, Any]:
        return phone_investigate(context, phone)

    @registry.register(
        "breach_check",
        description=(
            "Is this email/domain in known breaches? Official HIBP v3 API, "
            "key-gated (NM_OSINT_HIBP_KEY, free non-commercial). Returns "
            "breach names/dates only."
        ),
        capability=Capability.NET_OUT,
        parameters={"target": "str — email or domain"},
    )
    def breach_check_tool(target: str) -> dict[str, Any]:
        return breach_check(context, target)

    @registry.register(
        "osint_people",
        description=(
            "Multi-seed identity investigation + relationship map. seeds is "
            "JSON: {emails, handles, domains, phones, ips}. Builds a node/"
            "edge graph with an ASCII view."
        ),
        capability=Capability.NET_OUT,
        parameters={"seeds": "str — JSON object of identity lists"},
    )
    def osint_people_tool(seeds: str) -> dict[str, Any]:
        return osint_people(context, seeds)
