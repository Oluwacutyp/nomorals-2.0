"""Proxy source catalog + health (wave 83).

The scraper's source list is no longer a frozen constant.  Every source —
built-in or learned from the internet by ``proxy_discover`` — keeps
health state: consecutive failures, last success, last proxy count.
Three consecutive failures disable a source for a cooldown (24h); a
success clears the counter.  Dead endpoints (a deleted repo file, a
retired API) stop poisoning every scrape with the same 404 instead of
showing up in the error list forever.

Discovered sources live in the same registry, tagged with the seed page
that found them, and they scrape alongside the built-ins from the next
run on.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "BUILT_IN_SOURCES",
    "DISCOVERY_SEEDS",
    "SourceRegistry",
    "extract_list_candidates",
    "github_repos",
    "github_search_repos",
    "list_url_candidates",
]

#: (name, url, kind).  kinds: "list" = bare host:port lines (decorations
#: tolerated), "protocol" = scheme://host:port lines, "html" = a table
#: page, "json" = a JSON array of {protocol,host,port} records.
#: The name suffix encodes the protocol so the scheme filter works.
#:
#: Every entry below was live-verified on 2026-09-20 or 2026-10-01
#: (fetched and parsed).  Dead ones were cut: proxy-list.de (domain
#: parked), openproxylist.xd4dd.com (dead), spys /en/proxy-list/5/
#: (site restructured); 2026-10-01 pass also cut proxy-list.download
#: (502), openproxy.space (521), proxyscan.io API (404), freeproxylists
#: .net + list.proxylistplus.com (403 bot-block), cool-proxy.net (dead),
#: and a dozen renamed/deleted GitHub repos (ALIILAPROXY, yemixzy,
#: ProxySurf, 0x192/some-proxies, r00tee, specterxyz, ArrayIterator,
#: proxylist-to [empty], mertguvencli [moved], sunny9577 [moved]).
#: The catalog is a floor, not a ceiling — proxy_discover learns new
#: sources from the internet (GitHub search API + aggregator pages)
#: and they join every future scrape.
BUILT_IN_SOURCES: tuple[tuple[str, str, str], ...] = (
    # TheSpeedX — the big community repo (per-protocol files, updated daily)
    ("thespeedx-http",
     "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
     "list"),
    ("thespeedx-https",
     "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/https.txt",
     "list"),
    ("thespeedx-socks4",
     "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks4.txt",
     "list"),
    ("thespeedx-socks5",
     "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
     "list"),
    # monosans — the highest-quality public list: every proxy must fetch a
    # real URL in full to make it, re-checked every hour, fastest first
    ("monosans-http",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
     "list"),
    ("monosans-socks4",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks4.txt",
     "list"),
    ("monosans-socks5",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
     "list"),
    # all.txt carries the protocol prefix — one source for every scheme
    ("monosans-all",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/all.txt",
     "protocol"),
    # proxies.json carries exit_ip, timeout, ASN, geolocation
    ("monosans-json",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies.json",
     "json"),
    # roosterkid — hourly updates (hundreds of thousands of commits),
    # flag/latency/ISP decorated lines (the parser handles them)
    ("roosterkid-https",
     "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS.txt",
     "list"),
    ("roosterkid-socks4",
     "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS4.txt",
     "list"),
    ("roosterkid-socks5",
     "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS5.txt",
     "list"),
    # proxy-scrape v4 API — one source per protocol
    ("proxyscrape-http",
     "https://api.proxyscrape.com/v4/free-proxy-list/get"
     "?request=displayproxies&proxy_format=ipport&format=text"
     "&country=all&proxy_type=http",
     "list"),
    ("proxyscrape-https",
     "https://api.proxyscrape.com/v4/free-proxy-list/get"
     "?request=displayproxies&proxy_format=ipport&format=text"
     "&country=all&proxy_type=https",
     "list"),
    ("proxyscrape-socks4",
     "https://api.proxyscrape.com/v4/free-proxy-list/get"
     "?request=displayproxies&proxy_format=ipport&format=text"
     "&country=all&proxy_type=socks4",
     "list"),
    ("proxyscrape-socks5",
     "https://api.proxyscrape.com/v4/free-proxy-list/get"
     "?request=displayproxies&proxy_format=ipport&format=text"
     "&country=all&proxy_type=socks5",
     "list"),
    # html list sites (ip:port cells, separate IP+Port columns, and a
    # per-row type column are all handled)
    ("spys-http", "https://spys.one/en/http-proxy-list/", "html"),
    ("spys-mixed", "https://spys.one/en/free-proxy-list/", "html"),
    ("spys-socks", "https://spys.one/en/socks-proxy-list/", "html"),
    ("fpl-http", "https://free-proxy-list.net/", "html"),
    ("fpl-ssl", "https://free-proxy-list.net/ssl-proxy.html", "html"),
    # ---- 2026-10-01 expansion: every entry live-verified (fetched +
    # parsed) the same day; counts are proxies parsed from the live
    # response -----------------------------------------------
    # clarketm — the classic mixed list
    ("clarketm-http",
     "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
     "list"),
    # mmpx12 — per-protocol files
    ("mmpx12-http",
     "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
     "list"),
    ("mmpx12-https",
     "https://raw.githubusercontent.com/mmpx12/proxy-list/master/https.txt",
     "list"),
    ("mmpx12-socks4",
     "https://raw.githubusercontent.com/mmpx12/proxy-list/master/socks4.txt",
     "list"),
    ("mmpx12-socks5",
     "https://raw.githubusercontent.com/mmpx12/proxy-list/master/socks5.txt",
     "list"),
    # ShiftyTR — hourly updates
    ("shiftytr-http",
     "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
     "list"),
    ("shiftytr-https",
     "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
     "list"),
    ("shiftytr-socks4",
     "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks4.txt",
     "list"),
    ("shiftytr-socks5",
     "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",
     "list"),
    # TheSpeedX/SOCKS-List — the sister repo
    ("speedx-httpb",
     "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
     "list"),
    ("speedx-socks5b",
     "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
     "list"),
    # jetkai — per-protocol files under online-proxies/txt
    ("jetkai-http",
     "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
     "list"),
    ("jetkai-https",
     "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-https.txt",
     "list"),
    ("jetkai-socks4",
     "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks4.txt",
     "list"),
    ("jetkai-socks5",
     "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
     "list"),
    # hookzof — the big SOCKS5 list
    ("hookzof-socks5",
     "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
     "list"),
    # ProxyScrape's own GitHub mirror (5-min refresh; scheme:// lines)
    ("psgh-http",
     "https://raw.githubusercontent.com/proxyscrape/free-proxy-list/main/proxies/protocols/http/data.txt",
     "protocol"),
    ("psgh-https",
     "https://raw.githubusercontent.com/proxyscrape/free-proxy-list/main/proxies/protocols/https/data.txt",
     "protocol"),
    ("psgh-socks4",
     "https://raw.githubusercontent.com/proxyscrape/free-proxy-list/main/proxies/protocols/socks4/data.txt",
     "protocol"),
    ("psgh-socks5",
     "https://raw.githubusercontent.com/proxyscrape/free-proxy-list/main/proxies/protocols/socks5/data.txt",
     "protocol"),
    # proxy-free — 30-min tested lists + a structured proxies.json
    ("proxyfree-http",
     "https://raw.githubusercontent.com/proxy-free/free-proxy-list/main/http.txt",
     "list"),
    ("proxyfree-https",
     "https://raw.githubusercontent.com/proxy-free/free-proxy-list/main/https.txt",
     "list"),
    ("proxyfree-socks4",
     "https://raw.githubusercontent.com/proxy-free/free-proxy-list/main/socks4.txt",
     "list"),
    ("proxyfree-socks5",
     "https://raw.githubusercontent.com/proxy-free/free-proxy-list/main/socks5.txt",
     "list"),
    ("proxyfree-json",
     "https://raw.githubusercontent.com/proxy-free/free-proxy-list/main/proxies.json",
     "json"),
    # maximilianfeix — verified hourly, GitHub Pages + jsDelivr mirrors
    ("maxfeix-http",
     "https://raw.githubusercontent.com/maximilianfeix/proxy-scraper/proxy-list/http.txt",
     "list"),
    ("maxfeix-socks5",
     "https://raw.githubusercontent.com/maximilianfeix/proxy-scraper/proxy-list/socks5.txt",
     "list"),
    # cyberh4ck3r — big unchecked per-protocol banks
    ("cyberh4ck3r-http",
     "https://raw.githubusercontent.com/cyberh4ck3r/free-proxy-list/main/proxies/unchecked/http-proxies.txt",
     "list"),
    ("cyberh4ck3r-https",
     "https://raw.githubusercontent.com/cyberh4ck3r/free-proxy-list/main/proxies/unchecked/https-proxies.txt",
     "list"),
    ("cyberh4ck3r-socks4",
     "https://raw.githubusercontent.com/cyberh4ck3r/free-proxy-list/main/proxies/unchecked/socks4-proxies.txt",
     "list"),
    ("cyberh4ck3r-socks5",
     "https://raw.githubusercontent.com/cyberh4ck3r/free-proxy-list/main/proxies/unchecked/socks5-proxies.txt",
     "list"),
    # roosterkid's raw (undecorated) HTTPS file
    ("roosterkid-raw-https",
     "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
     "list"),
    # MuRongPIG — per-protocol banks
    ("murongpig-http",
     "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/http.txt",
     "list"),
    ("murongpig-socks4",
     "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/socks4.txt",
     "list"),
    ("murongpig-socks5",
     "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/socks5.txt",
     "list"),
    # Anonym0usWork1221 — http file carries scheme:// lines
    ("anonym0us-http",
     "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/http_proxies.txt",
     "protocol"),
    ("anonym0us-socks4",
     "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/socks4_proxies.txt",
     "list"),
    ("anonym0us-socks5",
     "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/socks5_proxies.txt",
     "list"),
    # prxchk — small but live
    ("prxchk-http",
     "https://raw.githubusercontent.com/prxchk/proxy-list/main/http.txt",
     "list"),
    ("prxchk-socks4",
     "https://raw.githubusercontent.com/prxchk/proxy-list/main/socks4.txt",
     "list"),
    ("prxchk-socks5",
     "https://raw.githubusercontent.com/prxchk/proxy-list/main/socks5.txt",
     "list"),
    # zevtyardt, SevenworksDev, proxy4parsing, rdavydov, zloi-user
    ("zevtyardt-http",
     "https://raw.githubusercontent.com/zevtyardt/proxy-list/main/http.txt",
     "list"),
    ("sevenworks-http",
     "https://raw.githubusercontent.com/SevenworksDev/proxy-list/main/proxies/http.txt",
     "list"),
    ("proxy4parsing-http",
     "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt",
     "list"),
    ("rdavydov-http",
     "https://raw.githubusercontent.com/rdavydov/proxy-list/main/proxies/http.txt",
     "list"),
    ("zloi-http",
     "https://raw.githubusercontent.com/zloi-user/hideip.me/main/http.txt",
     "list"),
    # dpangestuw — scheme:// lines in every file
    ("dpangestuw-http",
     "https://raw.githubusercontent.com/dpangestuw/Free-Proxy/main/http_proxies.txt",
     "protocol"),
    ("dpangestuw-socks4",
     "https://raw.githubusercontent.com/dpangestuw/Free-Proxy/main/socks4_proxies.txt",
     "protocol"),
    ("dpangestuw-socks5",
     "https://raw.githubusercontent.com/dpangestuw/Free-Proxy/main/socks5_proxies.txt",
     "protocol"),
    # thordata, proxygenerator1, vakhov, ClearProxy — SOCKS5 banks
    ("thordata-socks5",
     "https://raw.githubusercontent.com/Thordata/awesome-free-proxy-list/main/proxies/socks5.txt",
     "list"),
    ("proxygen-socks5",
     "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/socks5.txt",
     "list"),
    ("vakhov-socks5",
     "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/socks5.txt",
     "list"),
    ("clearproxy-socks5",
     "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/all.txt",
     "list"),
    # vmheaven, SoliSpirit — per-protocol banks
    ("vmheaven-http",
     "https://raw.githubusercontent.com/vmheaven/VMHeaven.io-Free-Proxy-List/main/http.txt",
     "list"),
    ("vmheaven-socks5",
     "https://raw.githubusercontent.com/vmheaven/VMHeaven.io-Free-Proxy-List/main/socks5.txt",
     "list"),
    ("solispirit-http",
     "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/http.txt",
     "list"),
    ("solispirit-socks5",
     "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/socks5.txt",
     "list"),
    # iplocate — mixed scheme:// bank
    ("iplocate-all",
     "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/all-proxies.txt",
     "protocol"),
    # claude89757 — https bank
    ("claude89757-https",
     "https://raw.githubusercontent.com/claude89757/free_https_proxies/main/https_proxies.txt",
     "list"),
    # ErcinDedeoglu — per-protocol banks
    ("ercin-http",
     "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/http.txt",
     "list"),
    ("ercin-https",
     "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt",
     "list"),
    ("ercin-socks4",
     "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/socks4.txt",
     "list"),
    ("ercin-socks5",
     "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/socks5.txt",
     "list"),
    # KangProxy — per-protocol dirs + the big xResults bank
    ("kangproxy-http",
     "https://raw.githubusercontent.com/officialputuid/KangProxy/main/http/http.txt",
     "list"),
    ("kangproxy-xresults",
     "https://raw.githubusercontent.com/officialputuid/KangProxy/main/xResults/Proxies.txt",
     "list"),
    # ---- JSON / text APIs -----------------------------------------
    # geonode — free JSON API, {data: [{ip, port, protocols[], country}]}
    ("geonode-http",
     "https://proxylist.geonode.com/api/proxy-list"
     "?limit=500&page=1&sort_by=lastChecked&sort_type=desc",
     "json"),
    # proxyscrape v2 API (still serving)
    ("proxyscrape-v2-http",
     "https://api.proxyscrape.com/v2/?request=displayproxies"
     "&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
     "list"),
    ("proxyscrape-v2-socks5",
     "https://api.proxyscrape.com/v2/?request=displayproxies"
     "&protocol=socks5&timeout=10000&country=all&ssl=all&anonymity=all",
     "list"),
    # proxyscrape v3 API — scheme:// lines
    ("proxyscrape-v3-http",
     "https://api.proxyscrape.com/v3/free-proxy-list/get"
     "?request=displayproxies&proxy_format=protocolipport&format=text"
     "&timeout=10000",
     "protocol"),
    # openproxylist.xyz API
    ("openproxylist-http",
     "https://api.openproxylist.xyz/http.txt",
     "list"),
    # spys.me plain lists
    ("spysme-http", "https://spys.me/proxy.txt", "list"),
    ("spysme-socks5", "https://spys.me/socks.txt", "list"),
    # ---- HTML list sites ------------------------------------------
    # the fpl sister sites (same table family as free-proxy-list.net)
    ("sslproxies-http", "https://www.sslproxies.org/", "html"),
    ("usproxy-http", "https://www.us-proxy.org/", "html"),
    ("socksproxy-socks5", "https://www.socks-proxy.net/", "html"),
    ("freeproxyworld-http", "https://www.freeproxy.world/?type=http", "html"),
    # advanced.name — base64 data-ip/data-port cells (parser layout 4)
    ("advancedname-http", "https://advanced.name/freeproxy", "html"),
)

#: pages that LIST proxy-list projects / endpoints — the seeds for
#: internet-wide source discovery.  discovery mines them for new
#: working list endpoints, not for proxies directly.  The primary
#: discovery engine is the GitHub search API (github_search_repos):
#: thousands of proxy-list repos, filtered to ones pushed this week.
DISCOVERY_SEEDS: tuple[str, ...] = (
    "https://github.com/topics/proxy-list",
    "https://github.com/topics/free-proxy",
    "https://spys.one/en/",
    "https://proxy-list.org/",
)

#: GitHub search for proxy-list repos pushed within the last N days.
_GITHUB_SEARCH_URL = ("https://api.github.com/search/repositories"
                      "?q=proxy-list+in:name,description,readme"
                      "+pushed:{start}..{end}"
                      "&sort=pushed&order=desc&per_page={limit}")


def github_search_repos(days: int = 7, limit: int = 20,
                        fetch: Callable[[str], bytes] | None = None,
                        ) -> list[str]:
    """Repo slugs (owner/name) of proxy-list repos pushed in the last
    ``days`` days — the continuously-refreshing source catalog.

    Unauthenticated GitHub search is rate-limited but needs no key;
    a failed fetch returns [] (discovery degrades to seed mining).
    """
    import datetime

    end = datetime.date.today()
    start = end - datetime.timedelta(days=max(1, int(days)))
    url = _GITHUB_SEARCH_URL.format(
        start=start.isoformat(), end=end.isoformat(),
        limit=max(3, int(limit)))
    try:
        raw = (fetch or _default_fetch)(url)
        data = json.loads(raw.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - no network = no search = no repos
        return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        slug = str(item.get("full_name", ""))
        if not slug or item.get("archived"):
            continue
        if slug.lower() in seen:
            continue
        seen.add(slug.lower())
        out.append(slug)
    return out


def _default_fetch(url: str) -> bytes:
    import urllib.request

    request = urllib.request.Request(url, headers={
        "User-Agent": "nomorals-proxy-lab/1.0",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(request, timeout=15) as resp:
        return resp.read()

_GITHUB_REPO_RE = re.compile(
    r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_GITHUB_SKIP = {
    "topics", "features", "about", "settings", "search", "sponsors",
    "orgs", "login", "signup", "new", "notifications",
}
_LIST_URL_RE = re.compile(
    r"""["'](https?://[^"'\s]+?\.txt(?:\?[^"'\s]*)?)["']""",
    re.IGNORECASE)
_REPO_TOKENS = re.compile(
    r"proxy|proxies|socks|spoofed", re.IGNORECASE)


def github_repos(page_text: str, limit: int = 12) -> list[str]:
    """Repo slugs (owner/name) linked from a page, list-ish first."""
    repos: list[str] = []
    seen: set[str] = set()
    for owner, name in _GITHUB_REPO_RE.findall(page_text or ""):
        if owner.lower() in _GITHUB_SKIP:
            continue
        slug = f"{owner}/{name}"
        if slug.lower() in seen:
            continue
        seen.add(slug)
        repos.append(slug)
    # prefer repos whose name talks about proxies — they are the
    # likely list hosts; the rest still get tested, just later
    repos.sort(key=lambda r: 0 if _REPO_TOKENS.search(r) else 1)
    return repos[:limit]


def list_url_candidates(seed_url: str, page_text: str,
                        limit: int = 30) -> list[str]:
    """Direct list endpoints referenced from a page (.txt links)."""
    out: list[str] = []
    seen: set[str] = set()
    for url in _LIST_URL_RE.findall(page_text or ""):
        url = url.rstrip("\"'")
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out[:limit]


def extract_list_candidates(seed_url: str, page_text: str,
                            *, repo_limit: int = 12,
                            url_limit: int = 30) -> dict[str, list[str]]:
    """One seed page → the candidate ENDPOINTS to test.

    Two kinds of candidates come back:
      github: repo slugs whose file listings the caller fetches via the
              GitHub API (each .txt file becomes a raw list endpoint)
      urls:   direct .txt / list links found on the page
    """
    return {
        "github": github_repos(page_text, limit=repo_limit),
        "urls": list_url_candidates(seed_url, page_text, limit=url_limit),
    }


class SourceRegistry:
    """Persistent source catalog with health tracking.

    Layout: one JSON file (``sources.json``) next to the proxy pool:

        {name: {url, kind, origin: builtin|discovered, seed,
                fails, last_ok, last_found, disabled_until, tests,
                error}}
    """

    def __init__(self, path: str | Path, *, cooldown_hours: float = 24.0,
                 fail_threshold: int = 3) -> None:
        self.path = Path(path)
        self.cooldown_seconds = float(cooldown_hours) * 3600.0
        self.fail_threshold = int(fail_threshold)
        self._data: dict[str, dict[str, Any]] = {}

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        self._data = {}
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt registry means: rebuild
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, indent=1, sort_keys=True),
                encoding="utf-8")
        except OSError:  # noqa: E103 - health is best-effort; scraping must not die on it
            pass

    # -- catalog ---------------------------------------------------------------
    def _entry(self, name: str, url: str, kind: str,
               origin: str = "builtin") -> dict[str, Any]:
        entry = self._data.get(name)
        if entry is None:
            entry = {"url": url, "kind": kind, "origin": origin,
                     "seed": "", "fails": 0, "last_ok": 0.0, "last_found": 0,
                     "disabled_until": 0.0, "tests": 0}
            self._data[name] = entry
        return entry

    def sources(self, now: float | None = None) -> list[tuple[str, str, str]]:
        """Active sources (built-in + discovered, minus disabled)."""
        self._load()
        now = now or time.time()
        out: list[tuple[str, str, str]] = []
        for name, url, kind in BUILT_IN_SOURCES:
            self._entry(name, url, kind)
        for name in list(self._data):
            entry = self._data[name]
            if float(entry.get("disabled_until") or 0) > now:
                continue
            out.append((name, entry["url"], entry["kind"]))
        # stable order: built-ins first (catalog order), then learned
        order = {n: i for i, (n, _, _) in enumerate(BUILT_IN_SOURCES)}
        out.sort(key=lambda t: (0 if t[0] in order else 1,
                                order.get(t[0], 10_000)))
        # persist the catalog so health state survives across processes
        self._save()
        return out

    def record(self, name: str, *, ok: bool, found: int = 0,
               error: str = "") -> None:
        """One scrape attempt's outcome → health state."""
        self._load()
        entry = self._data.get(name) or self._entry(name, "", "", "builtin")
        now = time.time()
        entry["tests"] = int(entry.get("tests") or 0) + 1
        if ok and found > 0:
            entry["fails"] = 0
            entry["last_ok"] = now
            entry["last_found"] = int(found)
            entry["disabled_until"] = 0.0
        else:
            entry["fails"] = int(entry.get("fails") or 0) + 1
            entry["error"] = error[:200]
            if entry["fails"] >= self.fail_threshold:
                entry["disabled_until"] = now + self.cooldown_seconds
        self._save()

    def add_discovered(self, name: str, url: str, kind: str, *,
                       seed: str = "", found: int = 0) -> dict[str, Any]:
        """Register a source the internet taught us about."""
        self._load()
        entry = self._entry(name, url, kind, origin="discovered")
        entry["seed"] = seed
        entry["fails"] = 0
        entry["disabled_until"] = 0.0
        entry["last_ok"] = time.time()
        entry["last_found"] = int(found)
        self._save()
        return entry

    def forget(self, name: str) -> bool:
        self._load()
        removed = self._data.pop(name, None) is not None
        if removed:
            self._save()
        return removed

    # -- reporting ---------------------------------------------------------------
    def disabled(self, now: float | None = None) -> list[dict[str, Any]]:
        self._load()
        now = now or time.time()
        out = []
        for name, entry in self._data.items():
            until = float(entry.get("disabled_until") or 0)
            if until > now:
                out.append({
                    "name": name,
                    "origin": entry.get("origin", "builtin"),
                    "fails": entry.get("fails", 0),
                    "last_error": str(entry.get("error", ""))[:120],
                    "disabled_for_hours": round((until - now) / 3600.0, 1),
                })
        return out

    def health(self) -> list[dict[str, Any]]:
        self._load()
        now = time.time()
        out = []
        for name, entry in self._data.items():
            out.append({
                "name": name,
                "origin": entry.get("origin", "builtin"),
                "tests": int(entry.get("tests") or 0),
                "fails": int(entry.get("fails") or 0),
                "last_found": int(entry.get("last_found") or 0),
                "last_ok": round(now - float(entry.get("last_ok") or now), 1)
                if entry.get("last_ok") else None,
                "disabled": float(entry.get("disabled_until") or 0) > now,
                "seed": entry.get("seed", ""),
            })
        out.sort(key=lambda d: (d["disabled"], -d["last_found"] or 0, d["name"]))
        return out

    def stats(self) -> dict[str, Any]:
        self._load()
        active = [n for n, e in self._data.items()
                  if float(e.get("disabled_until") or 0) <= time.time()]
        return {
            "total": len(self._data),
            "builtin": sum(1 for e in self._data.values()
                           if e.get("origin") == "builtin"),
            "discovered": sum(1 for e in self._data.values()
                              if e.get("origin") == "discovered"),
            "active": len(active),
            "disabled": len(self._data) - len(active),
        }
