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
import contextlib
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .sources import BROWSER_UA, _HTTP_TIMEOUT
from .failures import record_failure

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
    # ── availability layer (JustWatch native build, 52-websites #5) ──
    "DEFAULT_TERRITORY",
    "FRESH_WITHIN_S",
    "STALE_AFTER_S",
    "OFFER_KINDS",
    "OFFER_RECONCILE_SPEC",
    "OFFER_RECONCILE_JOB",
    "OFFER_RECONCILE_TOOL",
    "Offer",
    "OfferChange",
    "OfferStore",
    "OfferAdapter",
    "ArchiveOrgOfferAdapter",
    "TMDBOfferAdapter",
    "default_offer_adapters",
    "normalize_title",
    "parse_title_query",
    "reconcile_offers",
    "where_to_watch",
    "format_offers_reply",
    "run_offer_reconcile_tick_once",
    "ensure_offer_reconcile_job",
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
        except Exception as exc:  # noqa: BLE001 - persistence is an
            # enhancement; its absence never sinks a search
            record_failure("film_sources.registry_bind", exc)
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


# ── availability layer (JustWatch native build, 52-websites #5) ─────
#
# The missing relation: title -> (service, territory, stream|rent|buy,
# price, quality) kept fresh by reconciliation, never answered from
# memory. Every answer carries checked_at per offer plus a staleness
# label — an availability claim without a date is exactly the
# ungrounded shape Devon avoids elsewhere. Change tracking (new /
# leaving) runs on the watchlist; the daily tick pages the owner
# once per unseen change, never twice.
#
# Adapters are the honest boundary: an adapter either returns offers
# it actually observed (Archive.org's free streams; TMDB's
# watch-provider data, keyed) or says why it cannot. No adapter
# invents an offer, a price, or a verification date. A failed
# adapter fetch never deletes stored offers — they go stale instead.

#: The owner's territory — the default every answer assumes.
DEFAULT_TERRITORY = "NG"

#: An offer verified within the last day is fresh.
FRESH_WITHIN_S = 24 * 3600
#: An offer older than a week is stale — catalogues churn daily.
STALE_AFTER_S = 7 * 24 * 3600

OFFER_KINDS = ("stream", "rent", "buy")
OFFER_FILE = "film_offers.json"

#: Daily reconcile: catalogues churn daily, so the watchlist is
#: re-checked once a day, at 06:00 local.
OFFER_RECONCILE_SPEC = "0 6 * * *"
OFFER_RECONCILE_JOB = "offer reconcile tick"
OFFER_RECONCILE_TOOL = "__offer_reconcile_tick__"

_MAX_CHANGES = 100


def normalize_title(title: str) -> str:
    """The title key: lowercase, year stripped, punctuation folded.

    "Dune: Part Two (2024)" and "dune part two" are the same title;
    "Dune" and "Dune: Part Two" are not.
    """
    text = (title or "").strip().lower()
    text = re.sub(r"\(\s*(?:19|20)\d{2}\s*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_title_query(text: str) -> tuple[str, int | None]:
    """Split "Title (2024)" / "Title 2024" into (title, year|None)."""
    raw = (text or "").strip()
    match = re.search(r"\(\s*((?:19|20)\d{2})\s*\)\s*$", raw)
    if match:
        return raw[: match.start()].strip(), int(match.group(1))
    match = re.search(r"\s+((?:19|20)\d{2})\s*$", raw)
    if match:
        return raw[: match.start()].strip(), int(match.group(1))
    return raw, None


def _valid_territory(territory: str) -> str:
    """Upper-case ISO alpha-2 territory, or "" when it is not one."""
    code = (territory or "").strip().upper()
    return code if re.fullmatch(r"[A-Z]{2}", code) else ""


def _quality_hint(title: str) -> str:
    match = re.search(r"(480p|720p|1080p|2160p|4k)", title or "", re.I)
    if not match:
        return ""
    found = match.group(1).lower()
    return "4K" if found == "4k" else found


@dataclass
class Offer:
    """One observed streaming offer: a title on a service, somewhere.

    ``price`` is None when there is no per-title price — a
    subscription stream, a free copy, or a catalogue that does not
    list prices (TMDB); ``note`` says which, in words. ``quality``
    is "" when the catalogue did not state one — never guessed.
    ``checked_at`` is when the offer was last observed, epoch
    seconds; 0.0 means never verified.
    """

    title_ref: str = ""
    service: str = ""
    territory: str = DEFAULT_TERRITORY
    kind: str = "stream"
    price: float | None = None
    currency: str = ""
    quality: str = ""
    note: str = ""
    checked_at: float = 0.0
    source_adapter: str = ""

    def __post_init__(self) -> None:
        self.territory = (self.territory or "").strip().upper()
        self.kind = (self.kind or "").strip().lower()

    def key(self) -> tuple[str, str, str, str]:
        return (normalize_title(self.title_ref), self.service,
                self.territory, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {"title_ref": self.title_ref, "service": self.service,
                "territory": self.territory, "kind": self.kind,
                "price": self.price, "currency": self.currency,
                "quality": self.quality, "note": self.note,
                "checked_at": float(self.checked_at),
                "source_adapter": self.source_adapter}

    @classmethod
    def from_dict(cls, raw: Any) -> Offer:
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            title_ref=str(raw.get("title_ref", "") or ""),
            service=str(raw.get("service", "") or ""),
            territory=str(raw.get("territory", "") or DEFAULT_TERRITORY),
            kind=str(raw.get("kind", "") or "stream"),
            price=(None if raw.get("price") is None
                   else float(raw.get("price"))),
            currency=str(raw.get("currency", "") or ""),
            quality=str(raw.get("quality", "") or ""),
            note=str(raw.get("note", "") or ""),
            checked_at=float(raw.get("checked_at", 0.0) or 0.0),
            source_adapter=str(raw.get("source_adapter", "") or ""))


@dataclass
class OfferChange:
    """One availability change on a watchlisted title."""

    title: str = ""
    territory: str = DEFAULT_TERRITORY
    service: str = ""
    kind: str = "new"  # "new" | "leaving"
    at: float = 0.0
    seen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "territory": self.territory,
                "service": self.service, "kind": self.kind,
                "at": float(self.at), "seen": bool(self.seen)}

    @classmethod
    def from_dict(cls, raw: Any) -> OfferChange:
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            title=str(raw.get("title", "") or ""),
            territory=str(raw.get("territory", "") or DEFAULT_TERRITORY),
            service=str(raw.get("service", "") or ""),
            kind=str(raw.get("kind", "") or "new"),
            at=float(raw.get("at", 0.0) or 0.0),
            seen=bool(raw.get("seen", False)))


def _offer_key_str(key: tuple[str, str, str, str]) -> str:
    return "|".join(key)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no store is an empty store
        return {}
    return raw if isinstance(raw, dict) else {}


def _resolve_home(home: Any, context: Any) -> Path:
    if home is not None:
        return Path(home)
    settings = getattr(context, "settings", None) if context else None
    found = None
    if settings is not None:
        found = getattr(settings, "home_path", None) or getattr(
            settings, "home", None)
    if found is not None:
        return Path(found)
    return Path("~/.nomorals").expanduser()


class OfferStore:
    """Durable offer index + watchlist + change journal (atomic JSON).

    ``<home>/data/film_offers.json`` — offers keyed by
    (title, service, territory, kind), the watchlist, unseen
    availability changes, per-title reconcile outcomes, last tick.
    A corrupt store reads as empty, never a crash.
    """

    def __init__(self, home: Any) -> None:
        self._path = Path(home) / "data" / OFFER_FILE
        self._data = _read_json(self._path)

    # -- persistence ---------------------------------------------------
    def _save(self) -> None:
        with contextlib.suppress(Exception):
            _atomic_json_write(self._path, self._data)

    def _offers(self) -> dict[str, Any]:
        raw = self._data.get("offers")
        return raw if isinstance(raw, dict) else {}

    def _watchlist(self) -> list[dict[str, Any]]:
        raw = self._data.get("watchlist")
        return list(raw) if isinstance(raw, list) else []

    def _changes(self) -> list[dict[str, Any]]:
        raw = self._data.get("changes")
        return list(raw) if isinstance(raw, list) else []

    # -- staleness ------------------------------------------------------
    @staticmethod
    def staleness(offer: Offer, *, now: float | None = None) -> str:
        """fresh (<24h) / aging (1–7d) / stale (>7d) / never."""
        if not offer.checked_at:
            return "never"
        age = float(now if now is not None else time.time()) - offer.checked_at
        if age <= FRESH_WITHIN_S:
            return "fresh"
        if age <= STALE_AFTER_S:
            return "aging"
        return "stale"

    # -- offers ---------------------------------------------------------
    def upsert_offers(self, offers: list[Offer],
                      *, now: float | None = None) -> None:
        now_f = float(now if now is not None else time.time())
        rows = self._offers()
        for offer in offers:
            if not offer.checked_at:
                offer.checked_at = now_f
            rows[_offer_key_str(offer.key())] = offer.to_dict()
        self._data["offers"] = rows
        self._save()

    def remove_offers(self, keys: list[tuple[str, str, str, str]]) -> None:
        rows = self._offers()
        changed = False
        for key in keys:
            if _offer_key_str(key) in rows:
                del rows[_offer_key_str(key)]
                changed = True
        if changed:
            self._data["offers"] = rows
            self._save()

    def offers_for(self, title: str, territory: str) -> list[Offer]:
        """Stored offers for one title in one territory, by service."""
        want = normalize_title(title)
        code = _valid_territory(territory) or DEFAULT_TERRITORY
        out = []
        for raw in self._offers().values():
            offer = Offer.from_dict(raw)
            if (normalize_title(offer.title_ref) == want
                    and offer.territory == code):
                out.append(offer)
        out.sort(key=lambda o: (o.service.lower(), o.kind))
        return out

    # -- watchlist ------------------------------------------------------
    def watch(self, title: str, territory: str, *,
              now: float | None = None) -> bool:
        """Watch a title in a territory; False when already watched."""
        code = _valid_territory(territory) or DEFAULT_TERRITORY
        key = normalize_title(title)
        rows = self._watchlist()
        for row in rows:
            if (normalize_title(str(row.get("title", ""))) == key
                    and str(row.get("territory", "")).upper() == code):
                return False
        rows.append({"title": (title or "").strip(), "territory": code,
                     "added_at": float(now if now is not None
                                       else time.time())})
        self._data["watchlist"] = rows
        self._save()
        return True

    def unwatch(self, title: str, territory: str) -> bool:
        code = _valid_territory(territory) or DEFAULT_TERRITORY
        key = normalize_title(title)
        rows = self._watchlist()
        kept = [row for row in rows
                if not (normalize_title(str(row.get("title", ""))) == key
                        and str(row.get("territory", "")).upper() == code)]
        if len(kept) == len(rows):
            return False
        self._data["watchlist"] = kept
        self._save()
        return True

    def is_watched(self, title: str, territory: str) -> bool:
        code = _valid_territory(territory) or DEFAULT_TERRITORY
        key = normalize_title(title)
        return any(
            normalize_title(str(row.get("title", ""))) == key
            and str(row.get("territory", "")).upper() == code
            for row in self._watchlist())

    def list_watchlist(self) -> list[tuple[str, str]]:
        return [(str(row.get("title", "")),
                 str(row.get("territory", "")).upper())
                for row in self._watchlist()]

    # -- changes ---------------------------------------------------------
    def record_change(self, change: OfferChange) -> None:
        rows = self._changes()
        rows.append(change.to_dict())
        self._data["changes"] = rows[-_MAX_CHANGES:]
        self._save()

    def unseen_changes(self) -> list[OfferChange]:
        return [OfferChange.from_dict(raw) for raw in self._changes()
                if not raw.get("seen")]

    def mark_changes_seen(self, changes: list[OfferChange]) -> None:
        wanted = {(c.title, c.territory, c.service, c.kind, c.at)
                  for c in changes}
        rows = self._changes()
        changed = False
        for row in rows:
            ident = (str(row.get("title", "")), str(row.get("territory", "")),
                     str(row.get("service", "")), str(row.get("kind", "")),
                     float(row.get("at", 0.0) or 0.0))
            if ident in wanted and not row.get("seen"):
                row["seen"] = True
                changed = True
        if changed:
            self._data["changes"] = rows
            self._save()

    # -- reconcile bookkeeping ------------------------------------------
    def set_last_reconcile(self, title: str, territory: str,
                           outcomes: dict[str, Any], *,
                           now: float | None = None) -> None:
        rows = self._data.get("last_reconcile")
        rows = dict(rows) if isinstance(rows, dict) else {}
        key = f"{normalize_title(title)}|{_valid_territory(territory) or DEFAULT_TERRITORY}"
        rows[key] = {"at": float(now if now is not None else time.time()),
                     "adapters": outcomes}
        self._data["last_reconcile"] = rows
        self._save()

    def last_reconcile(self, title: str, territory: str) -> dict[str, Any]:
        rows = self._data.get("last_reconcile")
        if not isinstance(rows, dict):
            return {}
        key = f"{normalize_title(title)}|{_valid_territory(territory) or DEFAULT_TERRITORY}"
        raw = rows.get(key)
        return dict(raw) if isinstance(raw, dict) else {}

    def last_tick(self) -> float | None:
        value = self._data.get("last_tick")
        return None if value is None else float(value)

    def set_last_tick(self, now: float) -> None:
        self._data["last_tick"] = float(now)
        self._save()


# ── adapters ──────────────────────────────────────────────────────────


class OfferAdapter(abc.ABC):
    """One streaming catalogue that can answer for a title.

    ``available()`` is the honesty gate: an adapter that cannot
    run (no key, no network shape) says so and is skipped, never
    faked. ``fetch_offers`` returns only offers actually observed.
    ``last_note`` names the outcome of the last fetch in words, so
    a reconcile can say why an adapter came back empty. Never
    raises — the reconcile wraps every fetch.
    """

    name: str = "offer-base"
    last_note: str = ""

    @abc.abstractmethod
    def available(self) -> tuple[bool, str]:
        """(True, "") when the adapter can run, else (False, reason)."""

    @abc.abstractmethod
    def fetch_offers(self, title: str, territory: str, *,
                     now: float | None = None) -> list[Offer]:
        """Offers actually observed for this title, or []."""

    def _safe_fetch(self, title: str, territory: str,
                    *, now: float) -> tuple[list[Offer], bool, str]:
        try:
            offers = self.fetch_offers(title, territory, now=now)
        except Exception as exc:  # noqa: BLE001 - one dead adapter
            # never kills a reconcile
            return [], False, f"{self.name} fetch failed: {exc}"
        return list(offers or []), True, self.last_note


class ArchiveOrgOfferAdapter(OfferAdapter):
    """archive.org — the free, keyless, verified-live film source.

    The archive's films stream free worldwide; an offer here is
    stream / free, quality only when the title itself states it.
    """

    name = "archive.org"

    def __init__(self, source: Any = None) -> None:
        self._source = source or ArchiveOrgFilmSource()

    def available(self) -> tuple[bool, str]:
        return True, ""

    def fetch_offers(self, title: str, territory: str, *,
                     now: float | None = None) -> list[Offer]:
        now_f = float(now if now is not None else time.time())
        candidates = self._source.search(title, limit=8)
        offers = []
        for cand in candidates:
            if normalize_title(cand.title) != normalize_title(title):
                continue
            offers.append(Offer(
                title_ref=cand.title or title, service="archive.org",
                territory=_valid_territory(territory) or DEFAULT_TERRITORY,
                kind="stream", price=None, currency="",
                quality=_quality_hint(cand.title), note="free",
                checked_at=now_f, source_adapter=self.name))
        self.last_note = (f"{len(offers)} free stream(s) on archive.org"
                          if offers else "no archive.org copy")
        return offers


def _tmdb_key_from_context(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return os.environ.get("NM_TMDB_API_KEY", "")
    tmdb = getattr(settings, "tmdb", None)
    if tmdb is None:
        return os.environ.get("NM_TMDB_API_KEY", "")
    return str(getattr(tmdb, "api_key", "") or "") or os.environ.get(
        "NM_TMDB_API_KEY", "")


_TMDB_BASE = "https://api.themoviedb.org/3"


def _tmdb_get(url: str, *, timeout: float = 20.0) -> str:
    """One TMDB GET; "" on any failure (the adapter judges emptiness)."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Devon-offer-check/1.0",
                          "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - "" is the honest empty
        return ""


class TMDBOfferAdapter(OfferAdapter):
    """TMDB watch providers — the industry availability feed.

    TMDB's ``/watch/providers`` endpoint lists, per territory,
    which services stream / rent / sell a title — the same provider
    data JustWatch itself supplies. Keyed (free TMDB key):
    without ``NM_TMDB_API_KEY`` the adapter is unavailable and
    says so. TMDB lists no prices and no quality, so those stay
    empty with the note naming the gap. Title matching is strict:
    the normalized title must match exactly, and a year in the
    query must match the release year — a near miss returns [],
    never the wrong film's offers.
    """

    name = "tmdb"

    def __init__(self, api_key: str = "",
                 fetcher: Any = None) -> None:
        self.api_key = (api_key or "").strip()
        self._fetcher = fetcher or _tmdb_get

    def available(self) -> tuple[bool, str]:
        if self.api_key:
            return True, ""
        return False, ("no TMDB API key — set NM_TMDB_API_KEY "
                       "(tmdb.api_key in config)")

    def fetch_offers(self, title: str, territory: str, *,
                     now: float | None = None) -> list[Offer]:
        now_f = float(now if now is not None else time.time())
        code = _valid_territory(territory) or DEFAULT_TERRITORY
        query, year = parse_title_query(title)
        search_raw = self._fetcher(
            f"{_TMDB_BASE}/search/movie?api_key={self.api_key}"
            f"&query={urllib.parse.quote_plus(query)}&include_adult=false")
        try:
            results = (json.loads(search_raw).get("results", []) or []
                       if search_raw else [])
        except Exception:  # noqa: BLE001 - malformed JSON = no match
            results = []
        match = None
        for row in results:
            row_title = str(row.get("title", "") or "")
            if normalize_title(row_title) != normalize_title(query):
                continue
            if year is not None:
                release = str(row.get("release_date", "") or "")
                if not release.startswith(str(year)):
                    continue
            match = row
            break
        if match is None:
            self.last_note = "no exact title match on TMDB"
            return []
        movie_id = match.get("id")
        prov_raw = self._fetcher(
            f"{_TMDB_BASE}/movie/{movie_id}/watch/providers"
            f"?api_key={self.api_key}")
        try:
            per_territory = (json.loads(prov_raw).get("results", {}) or {}
                             if prov_raw else {})
        except Exception:  # noqa: BLE001 - malformed JSON = no offers
            per_territory = {}
        entry = per_territory.get(code) or {}
        offers: list[Offer] = []
        for provider_field, kind in (("flatrate", "stream"), ("rent", "rent"),
                            ("buy", "buy")):
            for provider in entry.get(provider_field) or []:
                service = str(provider.get("provider_name", "") or "")
                if not service:
                    continue
                offers.append(Offer(
                    title_ref=str(match.get("title", "") or query),
                    service=service, territory=code, kind=kind,
                    price=None, currency="", quality="",
                    note="price not listed by TMDB — check the service",
                    checked_at=now_f, source_adapter=self.name))
        self.last_note = (f"{len(offers)} TMDB provider offer(s) in {code}"
                          if offers
                          else f"TMDB lists no providers in {code}")
        return offers


def default_offer_adapters(context: Any = None) -> list[OfferAdapter]:
    """The production adapters: archive.org (keyless) + TMDB (keyed).

    The TMDB rung is unavailable without a key and says so; the
    archive rung always runs. Callers may inject their own list.
    """
    return [ArchiveOrgOfferAdapter(),
            TMDBOfferAdapter(api_key=_tmdb_key_from_context(context))]


# ── reconcile + answer ────────────────────────────────────────────────


def _offer_dict(offer: Offer, *, now: float) -> dict[str, Any]:
    row = offer.to_dict()
    row["staleness"] = OfferStore.staleness(offer, now=now)
    return row


def reconcile_offers(store: OfferStore, title: str, territory: str, *,
                     adapters: list[OfferAdapter] | None = None,
                     context: Any = None,
                     now: float | None = None) -> dict[str, Any]:
    """Re-check one title in one territory against the catalogues.

    Fresh offers are stored; offers whose adapter fetched cleanly
    but no longer lists the title leave (with a change event when
    the title is watchlisted); offers from a failed or unavailable
    adapter stay, going stale — a failed fetch never deletes.
    Never raises.
    """
    now_f = float(now if now is not None else time.time())
    code = _valid_territory(territory)
    if not (title or "").strip():
        return {"ok": False, "title": title, "territory": territory,
                "offers": [], "new": [], "leaving": [], "adapters": {},
                "stale": False, "error": "a title is needed"}
    if not code:
        return {"ok": False, "title": title, "territory": territory,
                "offers": [], "new": [], "leaving": [], "adapters": {},
                "stale": False,
                "error": ("territory must be a 2-letter code "
                          f"like NG, not {territory!r}")}
    try:
        runs = adapters if adapters is not None else default_offer_adapters(context)
        fresh: dict[tuple[str, str, str, str], Offer] = {}
        outcomes: dict[str, Any] = {}
        fetched_ok: set[str] = set()
        for adapter in runs:
            ok, reason = adapter.available()
            if not ok:
                outcomes[adapter.name] = {"ok": False, "reason": reason}
                continue
            offers, fetched, note = adapter._safe_fetch(
                title, code, now=now_f)
            outcomes[adapter.name] = {"ok": fetched, "reason": note}
            if not fetched:
                continue
            fetched_ok.add(adapter.name)
            for offer in offers:
                if not offer.source_adapter:
                    offer.source_adapter = adapter.name
                if not offer.checked_at:
                    offer.checked_at = now_f
                fresh.setdefault(offer.key(), offer)
        stored = store.offers_for(title, code)
        stored_by_key = {offer.key(): offer for offer in stored}
        new_keys = [key for key in fresh if key not in stored_by_key]
        leaving_keys = [key for key, offer in stored_by_key.items()
                        if key not in fresh
                        and offer.source_adapter in fetched_ok]
        store.upsert_offers(list(fresh.values()), now=now_f)
        store.remove_offers(leaving_keys)
        watched = store.is_watched(title, code)
        if watched:
            for key in new_keys:
                offer = fresh[key]
                store.record_change(OfferChange(
                    title=offer.title_ref, territory=code,
                    service=offer.service, kind="new", at=now_f))
            for key in leaving_keys:
                offer = stored_by_key[key]
                store.record_change(OfferChange(
                    title=offer.title_ref, territory=code,
                    service=offer.service, kind="leaving", at=now_f))
        store.set_last_reconcile(title, code, outcomes, now=now_f)
        offers = store.offers_for(title, code)
        return {"ok": True, "title": title, "territory": code,
                "offers": [_offer_dict(o, now=now_f) for o in offers],
                "new": sorted({fresh[key].service for key in new_keys}),
                "leaving": sorted({stored_by_key[key].service
                                   for key in leaving_keys}),
                "adapters": outcomes,
                "stale": any(OfferStore.staleness(o, now=now_f) == "stale"
                             for o in offers),
                "error": ""}
    except Exception as exc:  # noqa: BLE001 - reconcile never raises
        return {"ok": False, "title": title, "territory": code,
                "offers": [], "new": [], "leaving": [], "adapters": {},
                "stale": False, "error": f"reconcile failed: {exc}"}


def where_to_watch(store: OfferStore, title: str, territory: str, *,
                   adapters: list[OfferAdapter] | None = None,
                   context: Any = None, now: float | None = None,
                   refresh: bool = False) -> dict[str, Any]:
    """Where can this title be watched, in this territory?

    Stored offers answer without a fetch; an empty store (or
    ``refresh``) reconciles first. Every offer carries its
    ``checked_at`` and staleness — an offer with no fresh check
    says so and names its last check. Never raises.
    """
    now_f = float(now if now is not None else time.time())
    code = _valid_territory(territory)
    if not (title or "").strip():
        return {"ok": False, "title": title, "territory": territory,
                "offers": [], "stale": False,
                "error": "a title is needed"}
    if not code:
        return {"ok": False, "title": title, "territory": territory,
                "offers": [], "stale": False,
                "error": ("territory must be a 2-letter code "
                          f"like NG, not {territory!r}")}
    stored = store.offers_for(title, code)
    if stored and not refresh:
        return {"ok": True, "title": title, "territory": code,
                "offers": [_offer_dict(o, now=now_f) for o in stored],
                "stale": any(OfferStore.staleness(o, now=now_f) == "stale"
                             for o in stored),
                "error": ""}
    result = reconcile_offers(store, title, code, adapters=adapters,
                              context=context, now=now_f)
    if not result["ok"]:
        return result
    if not result["offers"]:
        notes = "; ".join(
            f"{name}: {info.get('reason', '')}"
            for name, info in (result.get("adapters") or {}).items()
            if info.get("reason"))
        return {"ok": False, "title": title, "territory": code,
                "offers": [], "stale": False,
                "error": (f"no streaming offer found for {title!r} in "
                          f"{code}"
                          + (f" ({notes})" if notes else ""))}
    return result


def format_offers_reply(result: dict[str, Any]) -> str:
    """Plain-words reply for a where/reconcile result."""
    title = str(result.get("title", "") or "that title")
    territory = str(result.get("territory", "") or "")
    if not result.get("ok", False):
        return (f"Could not find where to watch {title}"
                + (f" in {territory}" if territory else "")
                + f": {result.get('error') or 'unknown reason'}.")
    offers = result.get("offers") or []
    lines = [f"{title}" + (f" in {territory}" if territory else "") + ":"]
    for offer in offers:
        service = str(offer.get("service", ""))
        kind = str(offer.get("kind", ""))
        price = offer.get("price")
        currency = str(offer.get("currency", "") or "")
        if price is None:
            price_text = str(offer.get("note", "") or "no price listed")
        else:
            price_text = f"{float(price):.2f} {currency}".strip()
        quality = str(offer.get("quality", "") or "")
        checked = offer.get("checked_at") or 0
        when = (time.strftime("%Y-%m-%d", time.gmtime(float(checked)))
                if checked else "never")
        staleness = str(offer.get("staleness", "") or "")
        lines.append(f"- {service}: {kind}"
                     + (f", {quality}" if quality else "")
                     + f" — {price_text}"
                     + f" (checked {when}, {staleness})")
    if result.get("stale"):
        lines.append("Some listings are stale — last verified more than "
                     "7 days ago; ask me to refresh to re-check.")
    return "\n".join(lines)


# ── tick + scheduler ──────────────────────────────────────────────────


def run_offer_reconcile_tick_once(
        *, context: Any = None, notifier: Any = None,
        adapters: list[OfferAdapter] | None = None,
        home: Any = None, now: float | None = None) -> dict[str, Any]:
    """One offer-reconcile tick: re-check every watchlisted title.

    One Notifier note per unseen change (new / leaving), never
    twice; a quiet tick (empty watchlist) makes no network call
    at all — no adapter is even built. Never raises.
    """
    now_f = float(now if now is not None else time.time())
    try:
        home_path = _resolve_home(home, context)
        store = OfferStore(home_path)
        watchlist = store.list_watchlist()
        if not watchlist:
            return {"ok": True, "checked": 0, "alerts": 0,
                    "ran_at": now_f, "error": ""}
        runs = adapters if adapters is not None else default_offer_adapters(context)
        for title, territory in watchlist:
            reconcile_offers(store, title, territory, adapters=runs,
                             context=context, now=now_f)
        unseen = store.unseen_changes()
        alerts = 0
        published: list[OfferChange] = []
        for change in unseen:
            if notifier is None:
                continue
            with contextlib.suppress(Exception):
                when = time.strftime(
                    "%Y-%m-%d", time.gmtime(change.at))
                if change.kind == "leaving":
                    body = (f"{change.service} no longer lists "
                            f"{change.title} in {change.territory} "
                            f"(last seen {when}).")
                else:
                    body = (f"{change.service} now lists {change.title} "
                            f"in {change.territory} (seen {when}).")
                notifier.publish(
                    "offer_change",
                    title=(f"{change.title}: {change.kind} — "
                           f"{change.service}"),
                    body=body)
                published.append(change)
                alerts += 1
        with contextlib.suppress(Exception):
            store.mark_changes_seen(published)
            store.set_last_tick(now_f)
        return {"ok": True, "checked": len(watchlist), "alerts": alerts,
                "ran_at": now_f, "error": ""}
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {"ok": False, "checked": 0, "alerts": 0,
                "ran_at": now_f, "error": f"offer tick failed: {exc}"}


def ensure_offer_reconcile_job(scheduler: Any, *,
                               spec: str = OFFER_RECONCILE_SPEC) -> Any | None:
    """Register the daily offer-reconcile job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != OFFER_RECONCILE_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            OFFER_RECONCILE_JOB, spec, "tool",
            {"tool": OFFER_RECONCILE_TOOL, "args": {}}, routine=True)
    except Exception:
        return None
