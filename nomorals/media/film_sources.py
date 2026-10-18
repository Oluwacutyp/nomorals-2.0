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
"""

from __future__ import annotations

import abc
import json
import re
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
    "FILM_CHAIN",
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

    @abc.abstractmethod
    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        """Find film candidates. [] on any failure."""

    def download_url(self, candidate: FilmCandidate) -> str:
        return candidate.direct_url or candidate.url

    def _get(self, url: str, timeout: float = _HTTP_TIMEOUT) -> str:
        import urllib.request
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": BROWSER_UA,
                              "Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
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


FILM_CHAIN: list[FilmSource] = [
    ArchiveOrgFilmSource(),
    NetNaijaMoviesSource(),
    NkiriSource(),
    FzMoviesSource(),
]


def search_films(query: str, limit: int = 8) -> list[FilmCandidate]:
    """Search all film sources in priority order. Never raises."""
    out: list[FilmCandidate] = []
    for src in FILM_CHAIN:
        try:
            out.extend(src.search(query, limit))
        except Exception:  # noqa: BLE001 — chain keeps moving
            continue
        if len(out) >= limit:
            break
    return out[:limit]


def resolve_film_url(candidate: FilmCandidate) -> str:
    """Best download URL for a candidate. Never raises."""
    for src in FILM_CHAIN:
        if src.name == candidate.source:
            try:
                return src.download_url(candidate)
            except Exception:  # noqa: BLE001
                return candidate.url
    return candidate.direct_url or candidate.url
