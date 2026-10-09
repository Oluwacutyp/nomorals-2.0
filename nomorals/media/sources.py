"""Audio download sources — a strategy chain for /play.

Each source is an :class:`AudioSource` with the same two operations::

    search(query, limit) -> [candidate, ...]   # never raises
    download_url(candidate) -> str             # permalink / direct file URL

The resolver tries sources in priority order; the first one that yields
a downloadable candidate wins.  Every failure is honest and the chain
keeps moving — no source is ever tried twice for the same query.

Research grounding (2026-10-09, verified — not guessed):
- yt-dlp 2026.08.19 ``--list-extractors``: ``audiomack`` + ``audiomack:album``
  exist; there is NO ``boomplay`` extractor, NO naija/notjustok/tooxclusive
  extractors, and NO ``audiomack:search`` prefix.
- Audiomack ``api.audiomack.com/v1/search`` -> 401 without OAuth 1.0a;
  ``/v1/music/search`` -> 404 (dead); ``audiomack.com/search?q=`` is a
  client-side SPA with no server-rendered results.  The public stream
  endpoint ``audiomack.com/api/music/url/song/<artist>/<slug>`` needs no
  auth (confirmed in yt-dlp's extractor source).
- Boomplay: no official public API (confirmed); streams are protected;
  no yt-dlp extractor.  Search metadata only — downloads are honestly
  refused, the same way Spotify is handled.
- NetNaija: WordPress-based download blog (``?s=`` search, post page,
  multi-hop download pages, direct MP3).  Unreachable from non-NG
  networks (geo-block); the scraper is defensive and marked unverified.

Layering: media is L4.  No connector imports here.
"""

from __future__ import annotations

import abc
import html as html_mod
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AudioSource",
    "BaseScrapeSource",
    "AudiomackSource",
    "NetNaijaSource",
    "BoomplaySource",
    "SOURCE_CHAIN",
    "BROWSER_UA",
]

#: User-Agent that looks like a real phone browser — download blogs and
#: SPAs routinely 403 curl/python-urllib defaults.
BROWSER_UA = (
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
)

#: Hard ceiling for any single source HTTP round-trip.
_HTTP_TIMEOUT = 20.0


@dataclass
class SourceCandidate:
    """One search hit from an :class:`AudioSource`."""

    source: str = ""          # "audiomack" | "netnaija" | "boomplay"
    title: str = ""
    artist: str = ""
    url: str = ""             # permalink (page) or direct file URL
    direct_url: str = ""      # when known: the actual media file URL
    duration: float = 0.0
    downloadable: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class AudioSource(abc.ABC):
    """One strategy in the download chain.  Never raises."""

    name: str = "base"

    @abc.abstractmethod
    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        """Find candidates for *query*.  Returns [] on any failure."""

    def download_url(self, candidate: SourceCandidate) -> str:
        """Best URL to hand to the downloader for *candidate*."""
        return candidate.direct_url or candidate.url

    # ── shared HTTP helper ──────────────────────────────────────────
    def _get(self, url: str, timeout: float = _HTTP_TIMEOUT) -> str:
        """GET *url*, return text.  Empty string on any failure."""
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": BROWSER_UA,
                              "Accept": "text/html,application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            charset = "utf-8"
            try:
                ctype = resp.headers.get_content_charset()
                if ctype:
                    charset = ctype
            except Exception:  # noqa: BLE001
                pass
            return raw.decode(charset, errors="replace")
        except Exception as exc:  # noqa: BLE001 — network is hostile
            _log.info("%s: GET failed for %s: %s", self.name,
                      url[:80], exc)
            return ""


# ── scrape framework ────────────────────────────────────────────────────

class BaseScrapeSource(AudioSource):
    """Framework for WordPress-style download blogs.

    Subclasses declare:
    - ``search_url(query)`` — the site's search URL
    - ``parse_search(html)`` — extract [(title, post_url)] from results
    - ``extract_download(post_html, post_url)`` — the direct file URL
      (may follow further hops via ``self._get``)

    Everything is defensive: empty results, changed markup, and dead
    links all degrade to [] / "" instead of raising.
    """

    name = "scrape-base"

    @abc.abstractmethod
    def search_url(self, query: str) -> str:
        """The site's search URL for *query*."""

    @abc.abstractmethod
    def parse_search(self, html: str) -> list[tuple[str, str]]:
        """[(title, post_url)] from a search-results page."""

    @abc.abstractmethod
    def extract_download(self, post_html: str,
                         post_url: str) -> str:
        """Direct media file URL from a post page (may hop further)."""

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            html = self._get(self.search_url(q))
            if not html:
                return []
            out: list[SourceCandidate] = []
            for title, post_url in self.parse_search(html)[:limit]:
                title = (title or "").strip()
                post_url = (post_url or "").strip()
                if not title or not post_url:
                    continue
                out.append(SourceCandidate(
                    source=self.name, title=title, url=post_url))
            _log.info("%s: %d candidates for %r", self.name, len(out), q)
            return out
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("%s: search failed for %r: %s", self.name, q, exc)
            return []

    # shared helpers for subclasses ──────────────────────────────
    @staticmethod
    def _abs(base: str, href: str) -> str:
        return urllib.parse.urljoin(base, (href or "").strip())

    @staticmethod
    def _clean(text: str) -> str:
        return html_mod.unescape(re.sub(r"\s+", " ", text or "")).strip()

    @staticmethod
    def _direct_media_links(html: str, base_url: str) -> list[str]:
        """href/src URLs in *html* that look like audio files."""
        found: list[str] = []
        for m in re.finditer(
                r'''(?:href|src)\s*=\s*["']([^"']+)["']''', html, re.I):
            u = urllib.parse.urljoin(base_url, html_mod.unescape(m.group(1)))
            path = u.split("?", 1)[0].lower()
            if path.endswith((".mp3", ".m4a", ".ogg", ".opus", ".wav",
                              ".flac", ".aac")):
                if u not in found:
                    found.append(u)
        return found


# ── Audiomack ─────────────────────────────────────────────────────────────

_AM_TRACK_RE = re.compile(
    r"https?://(?:www\.)?audiomack\.com/([\w-]+)/song/([\w-]+)",
    re.IGNORECASE)


class AudiomackSource(AudioSource):
    """Audiomack — huge for Afrobeats/hip-hop.

    Download: yt-dlp's ``audiomack`` extractor handles permalink URLs
    (verified in yt-dlp 2026.08.19).  Search: the public search API is
    dead (404) and the OAuth API needs a key, so text search degrades
    gracefully to [] and the chain moves on — Audiomack stays a
    first-class *URL* source (paste a link, get the track).
    """

    name = "audiomack"

    # public stream endpoint (no auth) — confirmed in yt-dlp's extractor
    _URL_API = ("https://audiomack.com/api/music/url/song/{artist}/{slug}"
                "?extended=1")

    def handles_url(self, url: str) -> bool:
        return bool(_AM_TRACK_RE.search(url or ""))

    def track_api_url(self, page_url: str) -> str:
        """Public stream-API URL for an Audiomack track page URL."""
        m = _AM_TRACK_RE.search(page_url or "")
        if not m:
            return ""
        return self._URL_API.format(artist=m.group(1), slug=m.group(2))

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        # No working keyless search endpoint (verified 2026-10-09):
        #  /v1/music/search -> 404, /v1/search -> 401 (OAuth), SPA has no SSR.
        # Return [] so the chain falls through to sources that can search.
        _log.info("audiomack: no keyless search endpoint; skipping text "
                  "search for %r (paste an audiomack.com link instead)",
                  (query or "")[:60])
        return []

    def download_url(self, candidate: SourceCandidate) -> str:
        # yt-dlp routes the permalink itself via the audiomack extractor.
        return candidate.url


# ── NetNaija ──────────────────────────────────────────────────────────────

class NetNaijaSource(BaseScrapeSource):
    """NetNaija — Nigerian download blog (music, often otherwise hard to
    find).  WordPress ``?s=`` search, post pages, multi-hop download
    pages ending in a direct MP3 link.

    UNVERIFIED LIVE: the site geo-blocks non-Nigerian networks, so this
    could not be tested end-to-end from the build environment.  Every
    hop is defensive with multiple selector fallbacks; failures are
    honest and the chain continues.
    """

    name = "netnaija"
    _BASE = "https://netnaija.com"

    # link text that leads toward the file on download-hop pages
    _HOP_HINTS = ("download now", "download", "alternative link",
                  "click here to download")

    def search_url(self, query: str) -> str:
        return f"{self._BASE}/?s={urllib.parse.quote_plus(query)}"

    def parse_search(self, html: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        # WordPress search results: <h2 class="entry-title"><a href=...>Title</a>
        # plus generic fallbacks for theme variations.
        patterns = [
            r'<h[12][^>]*class="[^"]*entry-title[^"]*"[^>]*>\s*'
            r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<article[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*class="[^"]*'
            r'(?:entry|post)-title[^"]*"[^>]*>(.*?)</a>',
            r'<a[^>]*href="((?:https?://netnaija\.com)?/[^"]+)"[^>]*>'
            r'([^<]{8,120})</a>',
        ]
        seen: set[str] = set()
        for pat in patterns:
            for href, title in re.findall(pat, html, re.S | re.I):
                url = self._abs(self._BASE, href)
                title = self._clean(re.sub(r"<[^>]+>", "", title))
                if not url.startswith(self._BASE) or len(title) < 4:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                out.append((title, url))
            if out:
                break
        return out

    def extract_download(self, post_html: str, post_url: str) -> str:
        # Hop 1: direct media links already on the post page.
        direct = self._direct_media_links(post_html, post_url)
        if direct:
            return direct[0]
        # Hop 2+: follow "download" buttons up to 3 hops deep.
        url: str | None = post_url
        html: str | None = post_html
        for _ in range(3):
            assert html is not None and url is not None
            direct = self._direct_media_links(html, url)
            if direct:
                return direct[0]
            nxt = self._next_hop(html, url)
            if not nxt:
                break
            url = nxt
            html = self._get(url)
            if not html:
                break
        return ""

    def _next_hop(self, html: str, base_url: str) -> str:
        """The most likely 'continue to download' link on a hop page."""
        cands: list[tuple[int, str]] = []
        for m in re.finditer(
                r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
            href, text = m.group(1), self._clean(
                re.sub(r"<[^>]+>", "", m.group(2))).lower()
            if not href or href.startswith(("#", "javascript:")):
                continue
            score = 0
            for hint in self._HOP_HINTS:
                if hint in text:
                    score = max(score, len(hint))
            href_l = href.lower()
            if any(x in href_l for x in ("download", "dl.", ".mp3")):
                score += 2
            if score:
                cands.append((score, self._abs(base_url, href)))
        if not cands:
            return ""
        cands.sort(key=lambda t: -t[0])
        return cands[0][1]

    def download_url(self, candidate: SourceCandidate) -> str:
        if candidate.direct_url:
            return candidate.direct_url
        try:
            html = self._get(candidate.url)
            if not html:
                return ""
            return self.extract_download(html, candidate.url)
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("netnaija: download_url failed for %s: %s",
                      candidate.url[:80], exc)
            return ""


# ── Boomplay ──────────────────────────────────────────────────────────────

class BoomplaySource(AudioSource):
    """Boomplay — massive in Nigeria/Africa.

    Honest limitation (verified 2026-10-09): there is no official public
    API, no yt-dlp extractor, and streams are protected.  This source
    provides *search metadata only* — like Spotify, candidates are
    clearly marked non-downloadable so the chain (and the user) knows
    exactly what to expect.  Useful for discovery ("what's hot on
    Boomplay"), not for file delivery.
    """

    name = "boomplay"
    _SEARCH = "https://www.boomplay.com/search"

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            html = self._get(
                f"{self._SEARCH}?q={urllib.parse.quote_plus(q)}")
            if not html:
                return []
            out: list[SourceCandidate] = []
            # Boomplay song pages: /songs/<id>_<slug>
            seen: set[str] = set()
            for m in re.finditer(
                    r'href="(/songs/\d+[^"]*)"[^>]*>(.*?)</a',
                    html, re.S | re.I):
                path = m.group(1)
                title = BaseScrapeSource._clean(
                    re.sub(r"<[^>]+>", "", m.group(2)))
                if len(title) < 3 or path in seen:
                    continue
                seen.add(path)
                out.append(SourceCandidate(
                    source=self.name,
                    title=title,
                    url=urllib.parse.urljoin("https://www.boomplay.com",
                                             path),
                    downloadable=False,
                    extra={"note": "Boomplay streams are protected — "
                                   "open in the Boomplay app"}))
                if len(out) >= limit:
                    break
            _log.info("boomplay: %d metadata candidates for %r", len(out), q)
            return out
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("boomplay: search failed for %r: %s", q, exc)
            return []

    def download_url(self, candidate: SourceCandidate) -> str:
        return ""  # protected streams — honestly undownloadable


# ── the chain ─────────────────────────────────────────────────────────────

#: Text-search priority for /play.  Nigerian sources first (direct MP3s
#: beat metadata), proven globals next, metadata-only last.
#: Each entry: (strategy name, source instance or None for legacy paths).
SOURCE_CHAIN: list[tuple[str, AudioSource]] = [
    ("netnaija-search", NetNaijaSource()),
    ("boomplay-search", BoomplaySource()),
    # soundcloud-search / youtube-search / spotify-search stay on their
    # existing paths in resolver.py (API adapters + yt-dlp) — they are
    # proven and this module does not duplicate them.
    # audiomack is URL-first-class (no keyless text search exists).
]
