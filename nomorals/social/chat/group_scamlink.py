"""Scam-link OSINT sensor (DESIGN §4 beyond-parity / §9.4).

Links posted in a claimed group get dns/whois/look checks against
the keyless threat feeds — urlhaus (known-bad URL/host lists),
ThreatFox (IOC search), urlscan.io (past scan verdicts), the
domain's own DNS (does it resolve) and RDAP registration age.

Verdicts, in evidence order:
- ``bad`` — a feed NAMES the link: urlhaus lists the URL or its
  host, a high-confidence ThreatFox IOC, or a urlscan scan marked
  malicious.  The surface auto-deletes + warns on the shared
  escalation spine, exactly like the other sensors.
- ``suspicious`` — no feed names it, but the domain does not
  resolve or was registered recently (under ``young_days``).  The
  surface stages a ``scam`` flag proposal — a human confirms it,
  the sensor brands nobody — and never deletes or strikes.
- ``clean`` — the feeds answered and nothing fired.
- ``unknown`` — every feed failed.  Fail-open: no action, never
  cached, re-attempted on the next message carrying the link.

Why this organ runs the feeds itself instead of calling
``osint_threat`` verbatim: that bundle's per-feed budget is up to
8s (three feeds ≈ 24s worst case) — built for DM investigation,
not the live message path.  The sensor runs the SAME feeds and
the SAME dns/rdap organs (``tools.network``) under a ~6s per-link
deadline with early exit (a urlhaus hit returns after one call),
and a per-group verdict cache makes repeat links free.

The transport is injectable (``fetch``/``dns``/``rdap``) so tests
never touch the network; the defaults are the real feeds.
"""
from __future__ import annotations

import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from ...security.payload_inspection import MessageEnvelope
from .group_defense import DefenseConfig, SensorHit

URLHAUS_URL = "https://urlhaus-api.abuse.ch/v1/url/"
URLHAUS_HOST = "https://urlhaus-api.abuse.ch/v1/host/"
THREATFOX_URL = "https://threatfox-api.abuse.ch/api/v1/"
URLSCAN_SEARCH = "https://urlscan.io/api/v1/search/?q="

#: Per-feed timeout (seconds).  A feed slower than this is
#: treated as failed — fail-open, never a stall.
FEED_TIMEOUT_S = 2.5
#: Whole-link deadline (seconds) across all feeds.
LINK_BUDGET_S = 6.0
#: Verdict cache TTLs: a clean/suspicious verdict is re-checked
#: daily; a bad one stands a week (a delisted host is re-proven
#: then, not trusted forever).
CACHE_TTL_CLEAN_S = 24 * 3600
CACHE_TTL_BAD_S = 7 * 24 * 3600
CACHE_MAX_ENTRIES = 200

SCAMLINK_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "action": "delete_warn",
    "suspicious": "flag",       # flag | off — what a suspicious
                                # link does (never a delete)
    "max_links": 3,             # distinct domains checked per
                                # message (the rest are skipped)
    "young_days": 30,           # registration age under this is
                                # suspicious, never bad
}

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_TRAILING_PUNCT = ".,;:!?)]}\u201d\u2019"


@dataclass
class LinkVerdict:
    verdict: str                # bad | suspicious | clean | unknown
    url: str
    domain: str
    sources: list[str] = field(default_factory=list)
    detail: str = ""


def _normalize_url(raw: str) -> str:
    url = (raw or "").strip().rstrip(_TRAILING_PUNCT)
    try:
        parts = urllib.parse.urlparse(url)
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https"):
        return ""
    host = (parts.hostname or "").lower().rstrip(".")
    if not host or " " in host:
        return ""
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    return urllib.parse.urlunparse((
        parts.scheme.lower(), netloc, parts.path or "",
        parts.params or "", parts.query or "", ""))


def extract_links(env: MessageEnvelope) -> list[tuple[str, str]]:
    """Every http(s) URL in the message, in post order, deduped:
    url/text_link entities first (the platform's own parse), then
    bare URLs in the text.  Returns (url, domain) pairs."""
    found: list[str] = []

    def _add(raw: str) -> None:
        url = _normalize_url(raw)
        if url and url not in found:
            found.append(url)

    for ent in env.entities or []:
        if ent.get("type") not in ("url", "text_link"):
            continue
        if ent.get("url"):
            _add(str(ent["url"]))
            continue
        start = int(ent.get("offset") or 0)
        length = int(ent.get("length") or 0)
        if length:
            _add(env.text[start:start + length])
    for match in _URL_RE.finditer(env.text or ""):
        _add(match.group(0))
    out: list[tuple[str, str]] = []
    for url in found:
        host = urllib.parse.urlparse(url).hostname or ""
        if host:
            out.append((url, host.lower().rstrip(".")))
    return out


# ── default transport (real feeds, lazy imports) ───────────────


def _default_fetch(method: str, url: str, payload: Any = None,
                   timeout: float = FEED_TIMEOUT_S) -> dict[str, Any]:
    """One feed call.  Returns the parsed JSON object ({} when the
    feed answers non-JSON or non-200 — the caller reads
    ``query_status``/shape, never the status).  Raises on transport
    failure so the ladder can mark the feed down."""
    from ...core.http import HttpClient

    client = HttpClient(timeout=timeout)
    if method == "GET":
        response = client.get(url)
    elif method == "POST_FORM":
        response = client.post_form(url, payload or {})
    elif method == "POST_JSON":
        response = client.post_json(url, payload or {})
    else:
        raise ValueError(f"unknown fetch method {method!r}")
    try:
        data = response.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _default_dns(domain: str, timeout: float = 2.0) -> list[str]:
    from ...tools.network import dns_query

    return dns_query(domain, "A", timeout=timeout)


def _default_rdaps(domain: str,
                   timeout: float = FEED_TIMEOUT_S) -> dict[str, Any]:
    from ...tools.network import rdap_domain

    return rdap_domain(domain, timeout=timeout)


def _high_confidence(value: Any) -> bool:
    """ThreatFox confidence: the classic API returns the words
    high/medium/low; newer shapes return 0–100.  Only the top
    rung names a link bad."""
    if isinstance(value, str):
        return value.strip().lower() == "high"
    try:
        return float(value) >= 75
    except (TypeError, ValueError):
        return False


def _domain_age_days(registration: str, *, now: float) -> int | None:
    """Days since an RDAP registration date (YYYY-MM-DD).  None
    when the date is absent or unparseable — age is evidence,
    never a guess."""
    if not registration:
        return None
    try:
        stamp = time.mktime(time.strptime(
            str(registration)[:10], "%Y-%m-%d"))
    except (ValueError, OverflowError):
        return None
    return max(0, int((now - stamp) // 86400))


def check_link(url: str, *, fetch: Any = None, dns: Any = None,
               rdap: Any = None, young_days: int = 30,
               now: float | None = None) -> LinkVerdict:
    """One link through the feed ladder, earliest exit first.
    Every feed is fail-soft: a dead feed is recorded as down and
    the ladder moves on.  ``fetch`` is ``(method, url, payload,
    timeout) -> dict``; ``dns`` is ``(domain, timeout) -> [ips]``;
    ``rdap`` is ``(domain, timeout) -> dict``."""
    now = now if now is not None else time.time()
    fetch = fetch or _default_fetch
    dns = dns or _default_dns
    rdap = rdap or _default_rdaps
    domain = urllib.parse.urlparse(url).hostname or ""
    domain = domain.lower().rstrip(".")
    deadline = time.monotonic() + LINK_BUDGET_S

    def _budget() -> float:
        return max(0.2, deadline - time.monotonic())

    answered: list[str] = []

    def _call(name: str, fn: Any) -> Any:
        try:
            out = fn()
        except Exception:
            return None
        answered.append(name)
        return out

    # 1. urlhaus exact URL — the authoritative known-bad list.
    data = _call("urlhaus", lambda: fetch(
        "POST_FORM", URLHAUS_URL, {"url": url},
        timeout=min(FEED_TIMEOUT_S, _budget())))
    if isinstance(data, dict) and data.get("query_status") == "ok":
        threat = str(data.get("threat") or "malware")
        return LinkVerdict("bad", url, domain, ["urlhaus"],
                           f"listed by urlhaus as {threat}")
    # 2. urlhaus host — any known-bad URL on the host condemns it.
    data = _call("urlhaus", lambda: fetch(
        "POST_FORM", URLHAUS_HOST, {"host": domain},
        timeout=min(FEED_TIMEOUT_S, _budget())))
    if isinstance(data, dict) and data.get("query_status") == "ok":
        count = data.get("url_count", 0)
        return LinkVerdict(
            "bad", url, domain, ["urlhaus"],
            f"host listed by urlhaus ({count} known-bad URLs)")
    # 3. ThreatFox IOCs — high confidence only.
    data = _call("threatfox", lambda: fetch(
        "POST_JSON", THREATFOX_URL,
        {"query": "get_iocs", "search_term": url},
        timeout=min(FEED_TIMEOUT_S, _budget())))
    if isinstance(data, dict) and data.get("query_status") == "ok":
        for item in data.get("data") or []:
            if _high_confidence((item or {}).get(
                    "confidence_level")):
                kind = str((item or {}).get("threat_type_desc")
                           or (item or {}).get("threat_type")
                           or "malicious")
                return LinkVerdict(
                    "bad", url, domain, ["threatfox"],
                    f"ThreatFox IOC ({kind}, high confidence)")
    # 4. urlscan.io past scans — a malicious verdict names it.
    data = _call("urlscan", lambda: fetch(
        "GET", URLSCAN_SEARCH + urllib.parse.quote(
            f'page.url:"{url}"'),
        timeout=min(FEED_TIMEOUT_S, _budget())))
    if isinstance(data, dict):
        for scan in (data.get("results") or [])[:10]:
            overall = ((scan or {}).get("verdicts") or {}).get(
                "overall") or {}
            if overall.get("malicious"):
                return LinkVerdict(
                    "bad", url, domain, ["urlscan"],
                    "urlscan.io scan marked malicious")
    # 5. DNS — a domain that does not resolve is suspicious,
    # never bad: dead links are common, scams are not proven
    # by a missing record.
    ips = _call("dns", lambda: dns(domain, timeout=2.0))
    if ips is not None and not ips:
        return LinkVerdict("suspicious", url, domain, ["dns"],
                           "domain does not resolve")
    # 6. RDAP registration age — a days-old domain is suspicious,
    # never bad: a new legit site can at most earn a proposal
    # that dies quietly unconfirmed.
    reg = _call("rdap", lambda: rdap(
        domain, timeout=min(FEED_TIMEOUT_S, _budget())))
    if isinstance(reg, dict):
        age = _domain_age_days(str(reg.get("registration") or ""),
                               now=now)
        if age is not None and age < max(1, young_days):
            return LinkVerdict(
                "suspicious", url, domain, ["rdap"],
                f"domain registered {age} days ago")
    if not answered:
        return LinkVerdict("unknown", url, domain, [],
                           "all feeds failed")
    return LinkVerdict("clean", url, domain,
                       list(dict.fromkeys(answered)),
                       "no feed fired")


# ── verdict cache ───────────────────────────────────────────────


def _cache_ttl(verdict: str) -> float:
    return (CACHE_TTL_BAD_S if verdict == "bad"
            else CACHE_TTL_CLEAN_S)


def _cache_get(cache: dict[str, Any] | None, url: str,
               *, now: float) -> LinkVerdict | None:
    if not isinstance(cache, dict):
        return None
    row = cache.get(url)
    if not isinstance(row, dict):
        return None
    if now - float(row.get("ts") or 0) > _cache_ttl(
            str(row.get("verdict") or "")):
        return None
    return LinkVerdict(str(row.get("verdict") or "unknown"), url,
                       str(row.get("domain") or ""),
                       list(row.get("sources") or []),
                       str(row.get("detail") or ""))


def _cache_put(cache: dict[str, Any] | None, verdict: LinkVerdict,
               *, now: float) -> None:
    if not isinstance(cache, dict) or verdict.verdict == "unknown":
        return
    if len(cache) >= CACHE_MAX_ENTRIES:
        stale = [k for k, row in cache.items()
                 if isinstance(row, dict) and now - float(
                     row.get("ts") or 0) > _cache_ttl(
                         str(row.get("verdict") or ""))]
        for key in stale:
            cache.pop(key, None)
    if len(cache) >= CACHE_MAX_ENTRIES:
        oldest = min(
            cache, key=lambda k: float(
                (cache[k] or {}).get("ts") or 0)
            if isinstance(cache[k], dict) else 0.0)
        cache.pop(oldest, None)
    cache[verdict.url] = {
        "verdict": verdict.verdict, "domain": verdict.domain,
        "sources": list(verdict.sources), "detail": verdict.detail,
        "ts": now}


# ── the sensor ──────────────────────────────────────────────────


def check_scamlink(env: MessageEnvelope, cfg: DefenseConfig, *,
                   checker: Any = None,
                   cache: dict[str, Any] | None = None,
                   now: float | None = None) -> SensorHit | None:
    """The stage-3 sensor.  Returns the first BAD hit (a later
    link in the same message can still outrank a suspicious one),
    else the first SUSPICIOUS hit, else None.  ``checker`` is
    ``(url) -> LinkVerdict``; the default is the real feed ladder
    with the group's young-domain threshold."""
    if not cfg.scamlink_enabled:
        return None
    links = extract_links(env)
    if not links:
        return None
    now = now if now is not None else time.time()
    check = checker or (lambda u: check_link(
        u, young_days=cfg.scamlink_young_days, now=now))
    suspect: SensorHit | None = None
    seen_domains: set[str] = set()
    for url, domain in links:
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        if len(seen_domains) > cfg.scamlink_max_links:
            break
        cached = _cache_get(cache, url, now=now)
        verdict = cached or check(url)
        if cached is None:
            _cache_put(cache, verdict, now=now)
        if verdict.verdict == "bad":
            reason = (f"known scam link: {url[:120]} "
                      f"({verdict.detail})")
            return SensorHit(
                "scamlink", reason, cfg.scamlink_action,
                {"verdict": "bad", "url": url,
                 "domain": domain, "sources": verdict.sources,
                 "detail": verdict.detail})
        if verdict.verdict == "suspicious" and suspect is None:
            suspect = SensorHit(
                "scamlink",
                f"suspicious link: {url[:120]} ({verdict.detail})",
                cfg.scamlink_suspicious,
                {"verdict": "suspicious", "url": url,
                 "domain": domain, "sources": verdict.sources,
                 "detail": verdict.detail})
    return suspect
