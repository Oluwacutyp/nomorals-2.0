"""Keyless seed sources.

Every fetcher here answers anonymously. Keyed providers (HIBP account,
DeHashed, Shodan full) are listed in KEYED with the header they need so
the gap is named rather than hidden.

Normalised item kinds: host, url, email, port, vuln. `harvest()` runs all
sources concurrently, dedupes, and splits by kind. `wordlist()` turns
harvested URL paths into brute words — the loop back into subbrute.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["SOURCES", "KEYED", "harvest", "wordlist", "bin_check", "register"]

_HOST = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)+$")
_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I)
_PATH = re.compile(r"^/([a-z0-9_\-]{2,40})(?:/|$)", re.I)


@dataclass
class Item:
    kind: str
    value: str
    source: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value,
                "source": self.source, "extra": self.extra}


#: Keyed providers, listed so the gap is explicit.
KEYED: dict[str, str] = {
    "hibp_account": "NM_OSINT_HIBP_KEY — breach membership by address",
    "dehashed": "NM_DEHASHED_KEY — credential dumps (email/password pairs)",
    "shodan_full": "NM_SHODAN_KEY — banners, orgs, host detail",
    "censys": "NM_CENSYS_ID/SECRET — certificates + services",
    "securitytrails": "NM_SECURITYTRAILS_KEY — historical DNS",
    "intelx": "NM_INTELX_KEY — dark-web index search",
    "leakcheck": "NM_LEAKCHECK_KEY — combo/leak search",
}


def _clean_host(value: str) -> str:
    return (value or "").strip().lower().lstrip("*.").rstrip(".")


# ── fetchers: (name, url builder, parser) ───────────────────────────────────
def _otx(domain: str, client: HttpClient) -> list[Item]:
    url = (f"https://otx.alienvault.com/api/v1/indicators/domain/"
           f"{urllib.parse.quote(domain)}/passive_dns")
    resp = client.get(url)
    if not resp.ok:
        return []
    data = json.loads(resp.text or "{}")
    out = []
    for row in data.get("passive_dns", []) or []:
        host = _clean_host(str(row.get("hostname", "")))
        if host.endswith(domain) and _HOST.match(host):
            out.append(Item("host", host, "otx",
                            {"first": row.get("first"), "last": row.get("last")}))
    return out


def _hackertarget(domain: str, client: HttpClient) -> list[Item]:
    resp = client.get("https://api.hackertarget.com/hostsearch/"
                      f"?q={urllib.parse.quote(domain)}")
    if not resp.ok or "error" in (resp.text or "").lower()[:80]:
        return []
    out = []
    for line in (resp.text or "").splitlines():
        host = _clean_host(line.split(",")[0])
        if host.endswith(domain) and _HOST.match(host):
            out.append(Item("host", host, "hackertarget"))
    return out


def _rapiddns(domain: str, client: HttpClient) -> list[Item]:
    resp = client.get(f"https://rapiddns.io/subdomain/{urllib.parse.quote(domain)}"
                      "?full=1")
    if not resp.ok:
        return []
    out = []
    for host in re.findall(r"<td>([a-z0-9.\-]+\." + re.escape(domain) + r")</td>",
                           resp.text or "", re.I):
        host = _clean_host(host)
        if _HOST.match(host):
            out.append(Item("host", host, "rapiddns"))
    return out


def _wayback(domain: str, client: HttpClient, limit: int = 400) -> list[Item]:
    url = ("http://web.archive.org/cdx/search/cdx?url=*." +
           urllib.parse.quote(domain) + "/*&output=json&collapse=urlkey"
           f"&fl=original&limit={limit}")
    resp = client.get(url)
    if not resp.ok:
        return []
    try:
        rows = json.loads(resp.text or "[]")
    except ValueError:
        return []
    out = []
    for row in rows[1:]:
        original = row[0] if row else ""
        if not original:
            continue
        host = _clean_host(urllib.parse.urlsplit(original).hostname or "")
        if host.endswith(domain) and _HOST.match(host):
            out.append(Item("url", original, "wayback"))
    return out


def _urlscan(domain: str, client: HttpClient) -> list[Item]:
    resp = client.get("https://urlscan.io/api/v1/search/"
                      f"?q=domain%3A{urllib.parse.quote(domain)}&size=100")
    if not resp.ok:
        return []
    data = json.loads(resp.text or "{}")
    out = []
    for row in data.get("results", []) or []:
        page = (row.get("page") or {})
        url = page.get("url") or ""
        host = _clean_host(page.get("domain") or "")
        if host.endswith(domain):
            out.append(Item("url", url, "urlscan",
                            {"ip": page.get("ip"), "asn": page.get("asnname")}))
    return out


def _commoncrawl(domain: str, client: HttpClient, limit: int = 200) -> list[Item]:
    idx = client.get("https://index.commoncrawl.org/collinfo.json")
    if not idx.ok:
        return []
    try:
        newest = json.loads(idx.text or "[]")[0]["id"]
    except (ValueError, IndexError, KeyError):
        return []
    resp = client.get(f"https://index.commoncrawl.org/{newest}-index"
                      f"?url=*.{urllib.parse.quote(domain)}/*&output=json"
                      f"&limit={limit}")
    if not resp.ok:
        return []
    out = []
    for line in (resp.text or "").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        url = row.get("url") or ""
        host = _clean_host(urllib.parse.urlsplit(url).hostname or "")
        if host.endswith(domain):
            out.append(Item("url", url, "commoncrawl"))
    return out


def _internetdb(ip: str, client: HttpClient) -> list[Item]:
    resp = client.get(f"https://internetdb.shodan.io/{ip}")
    if not resp.ok:
        return []
    data = json.loads(resp.text or "{}")
    out = [Item("port", str(p), "internetdb", {"ip": ip})
           for p in data.get("ports", []) or []]
    out += [Item("vuln", str(v), "internetdb", {"ip": ip})
            for v in data.get("vulns", []) or []]
    return out


def _emailrep(email: str, client: HttpClient) -> list[Item]:
    resp = client.get(f"https://emailrep.io/{urllib.parse.quote(email)}",
                      headers={"User-Agent": "security-test"})
    if not resp.ok:
        return []
    try:
        data = json.loads(resp.text or "{}")
    except ValueError:
        return []
    return [Item("email", email, "emailrep", {
        "reputation": data.get("reputation"),
        "suspicious": data.get("suspicious"),
        "breaches": (data.get("details") or {}).get("breaches"),
        "profiles": (data.get("details") or {}).get("profiles")})]


def _gravatar(email: str, client: HttpClient) -> list[Item]:
    digest = hashlib.md5(email.strip().lower().encode()).hexdigest()
    resp = client.get(f"https://www.gravatar.com/{digest}.json")
    if not resp.ok:
        return []
    try:
        entry = (json.loads(resp.text or "{}").get("entry") or [{}])[0]
    except (ValueError, IndexError):
        return []
    return [Item("profile", entry.get("profileUrl") or "gravatar", "gravatar",
                 {"name": (entry.get("name") or {}).get("formatted"),
                  "accounts": [a.get("shortname") for a in
                               entry.get("accounts") or []]})]


SOURCES: dict[str, Callable[..., list[Item]]] = {
    "otx": _otx, "hackertarget": _hackertarget, "rapiddns": _rapiddns,
    "wayback": _wayback, "urlscan": _urlscan, "commoncrawl": _commoncrawl,
    "internetdb": _internetdb, "emailrep": _emailrep, "gravatar": _gravatar,
}


def harvest(domain: str = "", *, email: str = "", ips: Iterable[str] = (),
            only: Optional[Iterable[str]] = None, workers: int = 6,
            timeout: float = 25.0,
            proxy_factory: Optional[Callable[[], Optional[str]]] = None
            ) -> dict[str, Any]:
    """Run every applicable source concurrently. Deduped, source-tagged."""
    wanted = [n for n in (only or SOURCES) if n in SOURCES]
    jobs: list[tuple[str, Callable[[], list[Item]]]] = []
    for name in wanted:
        fn = SOURCES[name]
        if name in ("otx", "hackertarget", "rapiddns", "wayback",
                    "urlscan", "commoncrawl"):
            if domain:
                jobs.append((name, lambda f=fn: f(
                    domain, HttpClient(timeout=timeout,
                                       proxy_url=(proxy_factory() if proxy_factory else "") or ""))))
        elif name in ("emailrep", "gravatar"):
            if email:
                jobs.append((name, lambda f=fn: f(
                    email, HttpClient(timeout=timeout,
                                      proxy_url=(proxy_factory() if proxy_factory else "") or ""))))
        elif name == "internetdb":
            for ip in ips:
                jobs.append((f"{name}:{ip}", lambda i=ip, f=fn: f(
                    i, HttpClient(timeout=timeout,
                                  proxy_url=(proxy_factory() if proxy_factory else "") or ""))))

    items: list[Item] = []
    failed: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                items.extend(fut.result() or [])
            except Exception as exc:  # noqa: BLE001 - a dead source is data
                failed.append(f"{name}: {type(exc).__name__}")
                _log.info("seed source %s failed: %s", name, exc)

    seen: dict[str, Item] = {}
    for item in items:
        key = f"{item.kind}:{item.value.lower()}"
        if key not in seen:
            seen[key] = item
    unique = list(seen.values())
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in unique:
        buckets.setdefault(item.kind, []).append(item.as_dict())
    return {"domain": domain, "email": email, "sources_run": len(jobs),
            "sources_failed": failed, "total": len(unique), **buckets,
            "keyed_not_used": sorted(KEYED)}


def wordlist(items: Iterable[dict[str, Any]], *, max_words: int = 500) -> list[str]:
    """Path segments from harvested URLs, deduped — brute words."""
    words: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.get("kind") != "url":
            continue
        path = urllib.parse.urlsplit(item.get("value", "")).path
        for segment in path.strip("/").split("/"):
            segment = segment.lower()
            m = _PATH.match("/" + segment)
            word = (m.group(1) if m else segment)[:40]
            if word and word not in seen and not word.isdigit():
                seen.add(word)
                words.append(word)
        if len(words) >= max_words:
            break
    return words


def bin_check(ip: str, *, timeout: float = 12.0) -> dict[str, Any]:
    return {"ip": ip, "items": [i.as_dict() for i in
                                _internetdb(ip, HttpClient(timeout=timeout))]}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "seed_harvest",
        description="Harvest keyless seed sources (passive DNS, URL history, "
                    "scan indexes, host intel, email reputation). Deduped.",
        capability=Capability.NET_OUT,
        parameters={"domain": "str (optional)", "email": "str (optional)",
                    "ips": "str (optional) — comma list",
                    "only": "str (optional) — comma list of source names"},
    )
    def seed_harvest(domain: str = "", *, email: str = "", ips: str = "",
                     only: str = "") -> dict[str, Any]:
        return harvest(domain, email=email,
                       ips=[i.strip() for i in ips.split(",") if i.strip()],
                       only=[o.strip() for o in only.split(",") if o.strip()])

    @registry.register(
        "seed_wordlist",
        description="Turn harvested URLs into brute words for subbrute.",
        capability=Capability.DB_READ,
        parameters={"urls": "list[str]"},
    )
    def seed_wordlist(urls: list[str]) -> dict[str, Any]:
        words = wordlist([{"kind": "url", "value": u} for u in urls])
        return {"count": len(words), "words": words}

    @registry.register(
        "ip_intel",
        description="Ports, CVEs and hostnames for an IP (Shodan InternetDB).",
        capability=Capability.NET_OUT,
        parameters={"ip": "str"},
    )
    def ip_intel(ip: str) -> dict[str, Any]:
        return bin_check(ip)
