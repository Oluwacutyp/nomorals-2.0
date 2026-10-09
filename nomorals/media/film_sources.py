"""Film download sources — the movie equivalent of sources.py.

Same contract: search(query) -> candidates, download_url(candidate) -> file.
Same defensive posture: every hop has fallbacks, failures are honest,
the chain keeps moving.

Mined sources (Nigerian context, per user):
- NetNaija movies (thenetnaija.net) — WordPress ?s= search, post pages
- Nkiri.com — Nollywood/Hollywood/Korean, direct links
- FzMovies — mobile-friendly direct download links

All marked UNVERIFIED until live-tested — these sites change markup and
geo-block non-NG networks (same caveat as the audio NetNaija source).
"""

from __future__ import annotations

import abc
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

    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        html = self._get(
            f"{self._BASE}/?s={urllib.parse.quote_plus(query)}")
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

    def search(self, query: str, limit: int = 8) -> list[FilmCandidate]:
        html = self._get(
            f"{self._BASE}/search.php?searchname="
            f"{urllib.parse.quote_plus(query)}")
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


FILM_CHAIN: list[FilmSource] = [
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
