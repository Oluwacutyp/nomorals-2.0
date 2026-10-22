"""Film download sources — the movie equivalent of sources.py.

Same contract: search(query) -> candidates, download_url(candidate) -> file.
Same defensive posture: every hop has fallbacks, failures are honest,
the chain keeps moving.

Mined sources (Nigerian context, per user):
- NetNaija movies (thenetnaija.net) — WordPress ?s= search, post pages
- Nkiri.com — Nollywood/Hollywood/Korean, direct links
- FzMovies — mobile-friendly direct download links
- Internet Archive (archive.org) — public-domain / licensed films,
  JSON API, verified-live 2026-10-18 (see below) — FIRST in the chain
  because it is the only source verified-live from the verifier
  network.

Verification status, 2026-10-18 (SourceHealth probes from the
verifier network):
- archive.org — VERIFIED LIVE (HTTP 200, real movie docs).
- netnaija-movies — bot-blocked (HTTP 403) from the verifier network.
  NOT dead: the block is evidence about the verifier network (a
  datacenter IP), not about the site. Kept in the chain: it may work
  from the owner's (residential/Nigerian) network.
- nkiri / fzmovies — unverifiable (transport errors) from the verifier
  network. Same caveat: NOT dead, kept in the chain for the owner's
  network. Re-verify from a residential/Nigerian path before relying
  on them.

Pass #93 (open-sources law): this chain is a source class in the
ONE source registry (``media/source_registry.py``), like the music
chain (#88) and the book sites (#89) before it.  Until this pass
the chain was CLOSED: a fixed order, and every outcome — served,
catalogue miss, bot-wall, dead host — flattened to the same ``[]``,
so a walled or dead source was retried in the same order forever
and two of the four hosts (nkiri.com, fzmovies.net) were not
registered anywhere.  Now:

* every chain source's host is a registry descriptor
  (:data:`FILM_SOURCE_DOMAINS` is the class membership list;
  archive.org / thenetnaija.net are shared descriptors whose
  kinds the film class extends, nkiri / fzmovies are film seeds);
* each source keeps its last-fetch evidence (HTTP status, wall
  markers in the body) so the seam can classify the outcome
  honestly: served / no-result (weak) / blocked / rate-limited /
  operational failure — the one vocabulary;
* with a registry in reach, :func:`search_films` runs the chain
  in the registry's learned order (day one: no evidence → the
  chain order below, byte-for-byte), records every attempt, and
  appends health-promoted GROWN film sources, served through
  :class:`DiscoveredFilmSource` (the generic WordPress film
  shape); foreign hosts in search results intake as untrusted
  candidates (``kinds=("film",)``) that serve nothing until the
  health loop promotes them;
* with NO registry in reach (no injected registry and a context
  without a database), behaviour is exactly the pre-#93 chain —
  the #88/#89 discipline.
"""

from __future__ import annotations

import abc
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .sources import BROWSER_UA, _HTTP_TIMEOUT

_log = get_logger(__name__)

__all__ = [
    "FilmCandidate",
    "FilmSource",
    "BaseFilmScrape",
    "ArchiveOrgFilmSource",
    "NetNaijaMoviesSource",
    "NkiriSource",
    "FzMoviesSource",
    "DiscoveredFilmSource",
    "FILM_CHAIN",
    "FILM_SEARCH_SOURCES",
    "FILM_SOURCE_DOMAINS",
    "film_registry_for_context",
    "search_films",
    "resolve_film_url",
]


@dataclass
class FilmCandidate:
    """One film search hit."""

    source: str = ""          # "netnaija-movies" | "nkiri" | "fzmovies"
    title: str = ""
    year: str = ""
    url: str = ""             # post page
    direct_url: str = ""      # actual video file when resolved
    quality: str = ""         # "480p" | "720p" | "1080p"
    size_hint: str = ""
    downloadable: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class FilmSource(abc.ABC):
    """One film strategy. Never raises."""

    name: str = "film-base"
    #: Registry probe kind: "html-posts" for scrapers, "json" for APIs.
    search_kind: str = "html-posts"

    # ── last-fetch evidence (Pass #93) ──
    # ``search`` returns [] for a catalogue miss AND for a dead or
    # walled host; the registry seam needs to tell those apart, so
    # every fetch records what actually happened: the HTTP status
    # (0 = transport failure before any response, None = nothing
    # fetched yet), an error class, and the body (wall-marker
    # checks).  Class-level defaults; each fetch shadows them on
    # the instance.
    _last_status: int | None = None
    _last_error: str = ""
    _last_body: str = ""
    _fetch_count: int = 0

    @abc.abstractmethod
    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        """Find film candidates. [] on any failure."""

    def download_url(self, candidate: FilmCandidate) -> str:
        return candidate.direct_url or candidate.url

    def _get(self, url: str, timeout: float = _HTTP_TIMEOUT) -> str:
        import urllib.error
        import urllib.request
        self._fetch_count = getattr(self, "_fetch_count", 0) + 1
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": BROWSER_UA,
                              "Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            self._last_status = status
            self._last_error = ""
            self._last_body = text
            return text
        except urllib.error.HTTPError as exc:
            # An HTTP error still proves the host answered — keep
            # the status (and any wall page) as fetch evidence;
            # the contract is unchanged: "" to the caller.
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                body = ""
            self._last_status = int(exc.code or 0)
            self._last_error = f"http_{exc.code}"
            self._last_body = body
            _log.info("%s: GET %s -> HTTP %s", self.name,
                      url[:80], exc.code)
            return ""
        except Exception as exc:  # noqa: BLE001
            self._last_status = 0
            self._last_error = type(exc).__name__
            self._last_body = ""
            _log.info("%s: GET failed for %s: %s", self.name,
                      url[:80], exc)
            return ""


class BaseFilmScrape(FilmSource):
    """WordPress/blog-style film scrapers: search → post → download hops."""

    _HOP_HINTS = ("download now", "download", "download link",
                  "click here to download", "get movie")

    def _abs(self, base: str, href: str) -> str:
        return urllib.parse.urljoin(base, href)

    @staticmethod
    def _clean(s: str) -> str:
        import html as html_mod
        return html_mod.unescape(re.sub(r"\s+", " ", s or "")).strip()

    def _hop_links(self, html: str, base: str) -> list[str]:
        """Links whose text hints at a download."""
        out = []
        for href, text in re.findall(
                r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
            t = self._clean(re.sub(r"<[^>]+>", "", text)).lower()
            if any(h in t for h in self._HOP_HINTS):
                out.append(self._abs(base, href))
        return out

    def _direct_file(self, html: str) -> str:
        """A direct .mp4/.mkv link in the page."""
        m = re.search(
            r'href="([^"]+\.(?:mp4|mkv|avi)(?:\?[^"]*)?)"', html, re.I)
        return m.group(1) if m else ""


class NetNaijaMoviesSource(BaseFilmScrape):
    """NetNaija movies section. UNVERIFIED LIVE (geo-block caveat)."""

    name = "netnaija-movies"
    _BASE = "https://thenetnaija.net"

    def search_url(self, query: str) -> str:
        return f"{self._BASE}/?s={urllib.parse.quote_plus(query)}"

    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        html = self._get(self.search_url(query))
        if not html:
            return []
        out: list[FilmCandidate] = []
        seen: set[str] = set()
        for href, title in re.findall(
                r'<h[12][^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                html, re.S | re.I):
            url = self._abs(self._BASE, href)
            title = self._clean(re.sub(r"<[^>]+>", "", title))
            if url in seen or len(title) < 4:
                continue
            seen.add(url)
            # year hint in title like "(2024)"
            ym = re.search(r"\((19|20)\d{2}\)", title)
            out.append(FilmCandidate(
                source=self.name, title=title,
                year=ym.group(0).strip("()") if ym else "", url=url))
            if len(out) >= limit:
                break
        return out

    def download_url(self, candidate: FilmCandidate) -> str:
        if candidate.direct_url:
            return candidate.direct_url
        html = self._get(candidate.url)
        if not html:
            return candidate.url
        direct = self._direct_file(html)
        if direct:
            return direct
        # follow one hop
        for hop in self._hop_links(html, self._BASE)[:3]:
            h2 = self._get(hop)
            if h2:
                direct = self._direct_file(h2)
                if direct:
                    return direct
        return candidate.url


class NkiriSource(BaseFilmScrape):
    """Nkiri.com — Nollywood/Hollywood/Korean free downloads. UNVERIFIED."""

    name = "nkiri"
    _BASE = "https://nkiri.com"

    def search_url(self, query: str) -> str:
        """The search URL this source fetches (registry probe)."""
        return f"{self._BASE}/?s={urllib.parse.quote_plus(query)}"

    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        html = self._get(self.search_url(query))
        if not html:
            return []
        out: list[FilmCandidate] = []
        seen: set[str] = set()
        for href, title in re.findall(
                r'<a[^>]*href="([^"]+)"[^>]*>([^<]{6,140})</a>',
                html, re.S | re.I):
            url = self._abs(self._BASE, href)
            title = self._clean(title)
            if url in seen or self._BASE not in url:
                continue
            # skip nav junk
            if any(x in title.lower() for x in
                   ("home", "contact", "privacy", "disclaimer")):
                continue
            seen.add(url)
            ym = re.search(r"\((19|20)\d{2}\)", title)
            out.append(FilmCandidate(
                source=self.name, title=title,
                year=ym.group(0).strip("()") if ym else "", url=url))
            if len(out) >= limit:
                break
        return out

    def download_url(self, candidate: FilmCandidate) -> str:
        if candidate.direct_url:
            return candidate.direct_url
        html = self._get(candidate.url)
        if not html:
            return candidate.url
        direct = self._direct_file(html)
        return direct or candidate.url


class FzMoviesSource(BaseFilmScrape):
    """FzMovies — mobile-friendly direct links. UNVERIFIED."""

    name = "fzmovies"
    _BASE = "https://fzmovies.net"

    def search_url(self, query: str) -> str:
        """The primary search URL this source fetches (registry probe)."""
        return (f"{self._BASE}/search.php?searchname="
                f"{urllib.parse.quote_plus(query)}")

    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        html = self._get(self.search_url(query))
        if not html:
            # try alternate search path
            html = self._get(
                f"{self._BASE}/?s={urllib.parse.quote_plus(query)}")
        if not html:
            return []
        out: list[FilmCandidate] = []
        seen: set[str] = set()
        for href, title in re.findall(
                r'<a[^>]*href="([^"]*(?:movie|film)[^"]*)"[^>]*>'
                r"([^<]{6,140})</a>", html, re.S | re.I):
            url = self._abs(self._BASE, href)
            title = self._clean(title)
            if url in seen:
                continue
            seen.add(url)
            # quality hint like "720p" in title
            qm = re.search(r"(480p|720p|1080p|2160p)", title, re.I)
            out.append(FilmCandidate(
                source=self.name, title=title, url=url,
                quality=qm.group(1) if qm else ""))
            if len(out) >= limit:
                break
        return out

    def download_url(self, candidate: FilmCandidate) -> str:
        if candidate.direct_url:
            return candidate.direct_url
        html = self._get(candidate.url)
        if not html:
            return candidate.url
        direct = self._direct_file(html)
        if direct:
            return direct
        for hop in self._hop_links(html, self._BASE)[:3]:
            h2 = self._get(hop)
            if h2:
                direct = self._direct_file(h2)
                if direct:
                    return direct
        return candidate.url


class ArchiveOrgFilmSource(FilmSource):
    """Internet Archive films — JSON API, verified-live 2026-10-18.

    Searches ``mediatype:(movies)`` via ``advancedsearch.php`` and
    resolves a download URL from the item's metadata (first MPEG4 /
    .mp4 file). Public-domain / licensed films only — the honest,
    always-reachable film source in the chain.
    """

    name = "archiveorg"
    search_kind = "json"
    _BASE = "https://archive.org"

    def search_url(self, query: str) -> str:
        q = (f"title:({query}) AND mediatype:(movies)")
        return (f"{self._BASE}/advancedsearch.php?q="
                f"{urllib.parse.quote_plus(q)}"
                f"&fl[]=identifier&fl[]=title&fl[]=year"
                f"&fl[]=description&rows=50&page=1&output=json")

    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        raw = self._get(self.search_url(query))
        if not raw:
            return []
        try:
            docs = (json.loads(raw).get("response", {}) or {}).get(
                "docs", []) or []
        except Exception:  # noqa: BLE001 - malformed JSON = no results
            return []
        out: list[FilmCandidate] = []
        for doc in docs:
            ident = str(doc.get("identifier", "") or "")
            title = str(doc.get("title", "") or "")
            if not ident or not title:
                continue
            out.append(FilmCandidate(
                source=self.name, title=title,
                year=str(doc.get("year", "") or ""),
                url=f"{self._BASE}/details/{ident}",
                extra={"identifier": ident,
                       "description": str(doc.get("description", ""))[:300]}))
            if len(out) >= limit:
                break
        return out

    def download_url(self, candidate: FilmCandidate) -> str:
        if candidate.direct_url:
            return candidate.direct_url
        ident = (candidate.extra or {}).get("identifier") or ""
        if not ident and candidate.url:
            ident = candidate.url.rstrip("/").rsplit("/", 1)[-1]
        if not ident:
            return candidate.url
        raw = self._get(f"{self._BASE}/metadata/{ident}")
        if not raw:
            return candidate.url
        try:
            files = (json.loads(raw).get("files", []) or [])
        except Exception:  # noqa: BLE001 - malformed metadata
            return candidate.url
        for f in files:
            name = str(f.get("name", "") or "")
            fmt = str(f.get("format", "") or "")
            if name.lower().endswith(".mp4") or "MPEG4" in fmt:
                return (f"{self._BASE}/download/{ident}/"
                        f"{urllib.parse.quote(name)}")
        return candidate.url


class DiscoveredFilmSource(NetNaijaMoviesSource):
    """A film blog DISCOVERED on the film path, promoted by the
    source registry's health loop (Pass #93, open-sources law).

    The registry (``source_registry.py``) is the growth pipeline:
    a host encountered on the real path is noted as an UNTRUSTED
    candidate, serves nothing, and only the health loop can
    promote it (origin ``grown``).  A promoted host becomes
    searchable through this class — the generic WordPress shape
    the NetNaija movies source already implements (``?s=`` search,
    entry-title result parsing, download-page hops), pointed at
    the discovered domain.  Sites that are not WordPress-shaped
    simply return [] / the post URL and the chain moves on,
    honestly.
    """

    def __init__(self, domain: str) -> None:
        super().__init__()
        host = (domain or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        self.domain = host
        self._BASE = f"https://{host}"
        self.name = f"grown:{host}"


FILM_CHAIN: list[FilmSource] = [
    ArchiveOrgFilmSource(),
    NetNaijaMoviesSource(),
    NkiriSource(),
    FzMoviesSource(),
]

#: The film chain's sources, in day-one chain order.
FILM_SEARCH_SOURCES: tuple[str, ...] = (
    "archiveorg",
    "netnaija-movies",
    "nkiri",
    "fzmovies",
)

#: The host each film chain source actually fetches from — the
#: source's membership in the ONE source registry (Pass #93).
#: This map is the film class's membership list: the search seam
#: credits outcomes to the registry descriptor for this domain,
#: the registry's status surface renders the class from it, and
#: the health loop probes these hosts through their real search
#: endpoints.  Domains are evidence from this module (each
#: source's ``_BASE``), NOT marketing domains — and note the film
#: chain's NetNaija is ``thenetnaija.net``, the same host as the
#: download-blog seed (unlike the music chain's netnaija.com).
#: Grown sources are not listed here — they are not chain
#: members; the seam appends them from the registry itself.
FILM_SOURCE_DOMAINS: dict[str, str] = {
    "archiveorg": "archive.org",
    "netnaija-movies": "thenetnaija.net",
    "nkiri": "nkiri.com",
    "fzmovies": "fzmovies.net",
}


# ── source registry: outcomes, ordering, growth (Pass #93) ───────

#: Bot-wall markers in a 200 body — the registry's own vocabulary
#: (``source_registry._BOT_SIGNALS``): a challenge page is a wall,
#: not a catalogue miss.
_WALL_MARKERS = ("captcha", "cloudflare", "just a moment",
                 "verify you are human", "are you a robot",
                 "access denied", "attention required")


def film_registry_for_context(context: Any = None,
                              injected: Any = None) -> Any:
    """The source registry film outcomes land in, or None.

    Mirrors the #88 resolver's / #89 books' resolution exactly: an
    injected registry wins; otherwise the shared
    ``DEFAULT_REGISTRY`` bound to the context's KV store — and
    only when the context carries a database, so db-less callers
    keep the pre-#93 behaviour exactly.  Never raises."""
    if injected is not None:
        return injected
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return None
        from .source_registry import DEFAULT_REGISTRY
        reg = DEFAULT_REGISTRY
        try:
            from ..storage.kv import KVStore
            reg.bind_store(KVStore(db))
        except Exception:  # noqa: BLE001 - persistence is an
            # enhancement; its absence never sinks a search
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _desc_for_film_source(reg: Any, src: Any) -> Any:
    """The registry descriptor a film source serves from
    (:data:`FILM_SOURCE_DOMAINS`; ``grown:<domain>`` for promoted
    sources), or None.  Never raises."""
    try:
        name = str(getattr(src, "name", "") or "")
        if name.startswith("grown:"):
            return reg.for_domain(name[len("grown:"):])
        domain = FILM_SOURCE_DOMAINS.get(name)
        if not domain:
            return None
        return reg.for_domain(domain)
    except Exception:  # noqa: BLE001
        return None


def _film_fetch_evidence(src: Any) -> tuple[str, str]:
    """Classify one source's last-fetch evidence into the registry
    vocabulary: ``(kind, error_class)`` where kind is
    ``rate_limited`` | ``blocked`` | ``failure`` | ``no_result`` |
    ``none``.  ``served`` is the caller's call (results landed);
    this reads only what the fetch itself proved:

    * HTTP 429 → rate-limited (its own class, #90 — a throttled
      moment, never death);
    * HTTP 401/403, or a 200 page carrying wall markers →
      blocked (a wall is evidence about the wall, #89);
    * HTTP 200 with no wall → the source answered: a catalogue
      miss is weak evidence (``no_result``);
    * transport failure (status 0) → ``network_failure``; any
      other HTTP status → ``http_<status>`` (the search route
      itself failed — the query was never answered);
    * nothing fetched → ``none``: no evidence, nothing recorded.
    """
    try:
        status = getattr(src, "_last_status", None)
        if status is None:
            return "none", ""
        if status == 429:
            return "rate_limited", ""
        if status in (401, 403):
            return "blocked", ""
        if status == 200:
            body = (getattr(src, "_last_body", "") or "")[:4000].lower()
            if any(m in body for m in _WALL_MARKERS):
                return "blocked", ""
            return "no_result", ""
        if status == 0:
            return "failure", "network_failure"
        return "failure", f"http_{status}"
    except Exception:  # noqa: BLE001
        return "none", ""


def _apply_film_evidence(reg: Any, desc: Any, kind: str,
                         error_class: str, latency_s: float) -> None:
    """Land one classified outcome in the registry.  Best-effort;
    never raises — bookkeeping never breaks a search."""
    try:
        if desc is None or kind in ("", "none"):
            return
        if kind == "served":
            reg.record_outcome(desc, True, latency_s=latency_s)
        elif kind == "blocked":
            reg.record_blocked(desc, latency_s=latency_s)
        elif kind == "rate_limited":
            reg.record_rate_limited(desc, latency_s=latency_s)
        elif kind == "no_result":
            reg.record_no_result(desc)
        else:
            reg.record_outcome(desc, False, latency_s=latency_s,
                               error_class=error_class or "site_error")
    except Exception:  # noqa: BLE001
        pass


def _record_film_search(reg: Any, src: Any, desc: Any,
                        results: list[FilmCandidate],
                        latency_s: float, raised: bool) -> None:
    """Credit/blame one film search attempt.  On a serve, foreign
    hosts among the result URLs intake as UNTRUSTED film
    candidates (post pages genuinely live on hosts the registry
    has never seen — mirrors, CDNs, sister blogs); known sources
    refuse themselves inside the registry.  Best-effort; never
    raises."""
    try:
        if desc is None:
            return
        if results:
            _apply_film_evidence(reg, desc, "served", "", latency_s)
            for cand in results:
                for url in (cand.url, cand.direct_url):
                    if url:
                        reg.note_candidate(
                            url, via="film-search-result",
                            kinds=("film",))
            return
        if raised:
            _apply_film_evidence(reg, desc, "failure",
                                 "search_failed", latency_s)
            return
        kind, error_class = _film_fetch_evidence(src)
        _apply_film_evidence(reg, desc, kind, error_class, latency_s)
    except Exception:  # noqa: BLE001
        pass


def _order_film(reg: Any, items: list[tuple[Any, Any]]
                ) -> list[tuple[Any, Any]]:
    """Outcome-aware chain order (Pass #93), through the registry's
    existing score/health semantics — the #88 resolver's discipline:

    * NO learned state anywhere in the chain → the list is
      returned untouched: day-one behaviour is today's chain
      order, byte-for-byte;
    * otherwise active sources sort by registry score (stable —
      ties keep chain order) and stood-down / disabled / dead
      sources are demoted to the end, in chain order: they leave
      the active ordering (the #86 vocabulary) but the chain
      still tries them last rather than pretending they do not
      exist — the attempt log stays truthful.
    Never raises; a registry hiccup returns the chain as-is."""
    try:
        evidence = False
        for _src, desc in items:
            if desc is None:
                continue
            st = reg.stats_for(desc)
            if st["ok"] or st["fail"] or st["no_result"] \
                    or not reg.is_active(desc):
                evidence = True
                break
        if not evidence:
            return items
        active = [it for it in items
                  if it[1] is None or reg.is_active(it[1])]
        inactive = [it for it in items
                    if it[1] is not None and not reg.is_active(it[1])]
        active.sort(key=lambda it: -reg.score(it[1])
                    if it[1] is not None else 0.0)
        return active + inactive
    except Exception:  # noqa: BLE001
        return items


def _grown_film_sources(reg: Any) -> list[FilmSource]:
    """Serving shapes for GROWN film sources: candidates the
    health loop promoted (origin ``grown``, kind ``film``).
    Candidates that were never promoted are NOT here — they serve
    nothing.  Never raises."""
    try:
        return [DiscoveredFilmSource(d.domain) for d in reg.all()
                if reg.origin_for(d) == "grown"
                and "film" in d.kinds]
    except Exception:  # noqa: BLE001
        return []


def search_films(query: str, limit: int = 8, *,
                 context: Any = None,
                 registry: Any = None) -> list[FilmCandidate]:
    """Search all film sources in priority order. Never raises.

    Pass #93: with a registry in reach (injected, or the shared
    ``DEFAULT_REGISTRY`` when *context* carries a database) the
    chain runs in the registry's learned order, every attempt's
    outcome lands in the registry (served / no-result / blocked /
    rate-limited / failure, classified from fetch evidence), film
    search results intake foreign hosts as candidates, and
    health-promoted grown film sources serve after the seed chain.
    With no registry in reach the behaviour is the pre-#93 chain,
    exactly."""
    reg = film_registry_for_context(context, registry)
    if reg is None:
        out: list[FilmCandidate] = []
        for src in FILM_CHAIN:
            try:
                out.extend(src.search(query, limit))
            except Exception:  # noqa: BLE001 — chain keeps moving
                continue
            if len(out) >= limit:
                break
        return out[:limit]
    items: list[tuple[Any, Any]] = [
        (src, _desc_for_film_source(reg, src)) for src in FILM_CHAIN]
    for src in _grown_film_sources(reg):
        items.append((src, _desc_for_film_source(reg, src)))
    items = _order_film(reg, items)
    out = []
    for src, desc in items:
        t0 = time.perf_counter()
        raised = False
        try:
            results = src.search(query, limit)
        except Exception:  # noqa: BLE001 — chain keeps moving
            results, raised = [], True
        _record_film_search(reg, src, desc, results,
                            time.perf_counter() - t0, raised)
        out.extend(results)
        if len(out) >= limit:
            break
    return out[:limit]


def resolve_film_url(candidate: FilmCandidate, *,
                     context: Any = None,
                     registry: Any = None) -> str:
    """Best download URL for a candidate. Never raises.

    Pass #93: with a registry in reach, the serving source's
    resolve fetch is also evidence — a source that answers is
    credited served; a wall / throttle / failure during the
    download-page hops lands in the registry with its class.  A
    candidate that already carries a direct URL fetches nothing
    and earns nothing (nothing was observed)."""
    reg = film_registry_for_context(context, registry)
    sources: list[FilmSource] = list(FILM_CHAIN)
    if reg is not None:
        sources.extend(_grown_film_sources(reg))
    for src in sources:
        if src.name == candidate.source:
            if reg is None:
                try:
                    return src.download_url(candidate)
                except Exception:  # noqa: BLE001
                    return candidate.url
            desc = _desc_for_film_source(reg, src)
            before = getattr(src, "_fetch_count", 0)
            t0 = time.perf_counter()
            try:
                url = src.download_url(candidate)
            except Exception:  # noqa: BLE001
                _apply_film_evidence(
                    reg, desc, "failure", "resolve_failed",
                    time.perf_counter() - t0)
                return candidate.url
            latency = time.perf_counter() - t0
            if getattr(src, "_fetch_count", 0) > before:
                kind, error_class = _film_fetch_evidence(src)
                if kind in ("blocked", "rate_limited", "failure"):
                    _apply_film_evidence(reg, desc, kind,
                                         error_class, latency)
                elif kind == "no_result":
                    # the source fetched and answered during the
                    # hops — that is a serve, whatever the hops
                    # found (the download itself is judged on the
                    # download path, #86)
                    _apply_film_evidence(reg, desc, "served", "",
                                         latency)
            return url
    return candidate.direct_url or candidate.url
