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
    "ITunesSource",
    "DeezerSource",
    "ArchiveOrgSource",
    "BandcampSource",
    "NotJustOkSource",
    "TooXclusiveSource",
    "MusicBrainzSource",
    "OpenverseSource",
    "WikimediaSource",
    "SOURCE_CHAIN",
    "BROWSER_UA",
    "MUSIC_SEARCH_SOURCES",
    "MUSIC_SOURCE_REGIONS",
    "GLOBAL_REGIONS",
    "music_covered_regions",
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


# ── keyless JSON music APIs (no key, no account) ───────────────────────

class _JsonApiSource(AudioSource):
    """Base for keyless JSON search APIs: GET → parse JSON → candidates.

    Subclasses declare ``search_url`` + ``parse``.  Never raises; any
    transport/parse failure degrades to [] so the chain keeps moving.
    """

    name = "json-api"

    def search_url(self, query: str) -> str:
        raise NotImplementedError

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        raise NotImplementedError

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            import json as _json
            raw = self._get(self.search_url(q))
            if not raw:
                return []
            payload = _json.loads(raw)
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("%s: search failed for %r: %s", self.name, q, exc)
            return []
        try:
            out = self.parse(payload, q)
        except Exception as exc:  # noqa: BLE001
            _log.info("%s: parse failed for %r: %s", self.name, q, exc)
            return []
        _log.info("%s: %d candidates for %r", self.name, len(out), q)
        return out[:limit]


class ITunesSource(_JsonApiSource):
    """iTunes Search API — keyless, official Apple endpoint.

    Returns 30-second preview MP3s (``previewUrl``) + the store page
    (``trackViewUrl``).  Previews are real downloadable files but only
    30s — candidates carry that in ``extra`` so the resolver treats
    them as last-resort (full tracks first, previews only when nothing
    else hit).
    """

    name = "itunes"

    def search_url(self, query: str) -> str:
        return ("https://itunes.apple.com/search?term="
                + urllib.parse.quote_plus(query)
                + "&media=music&limit=10")

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for row in (payload or {}).get("results") or []:
            if not isinstance(row, dict):
                continue
            preview = str(row.get("previewUrl") or "")
            page = str(row.get("trackViewUrl") or "")
            title = str(row.get("trackName") or "")
            artist = str(row.get("artistName") or "")
            if not title and not preview:
                continue
            out.append(SourceCandidate(
                source=self.name, title=title or query, artist=artist,
                url=page or preview, direct_url=preview,
                duration=float(row.get("trackTimeMillis") or 0) / 1000.0
                if row.get("trackTimeMillis") else 0.0,
                downloadable=bool(preview),
                extra={"preview_only": True,
                       "note": "iTunes 30-second preview"}))
        return out


class DeezerSource(_JsonApiSource):
    """Deezer public search API — keyless, no account.

    Same shape as iTunes: 30-second preview MP3s (``preview``) + the
    track page (``link``).  Previews are last-resort in the resolver.
    """

    name = "deezer"

    def search_url(self, query: str) -> str:
        return ("https://api.deezer.com/search?q="
                + urllib.parse.quote_plus(query))

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for row in (payload or {}).get("data") or []:
            if not isinstance(row, dict):
                continue
            preview = str(row.get("preview") or "")
            page = str(row.get("link") or "")
            title = str(row.get("title") or "")
            artist = str((row.get("artist") or {}).get("name") or "")
            if not title and not preview:
                continue
            out.append(SourceCandidate(
                source=self.name, title=title or query, artist=artist,
                url=page or preview, direct_url=preview,
                duration=float(row.get("duration") or 0),
                downloadable=bool(preview),
                extra={"preview_only": True,
                       "note": "Deezer 30-second preview"}))
        return out


class MusicBrainzSource(_JsonApiSource):
    """MusicBrainz — keyless, worldwide metadata catalog.

    The largest open music database (recordings from every country,
    every language).  Metadata only — no audio files — so candidates
    are marked non-downloadable, like Boomplay: useful for discovery
    ("what is this track / who made it"), and the chain then finds the
    audio through a full-track source.
    """

    name = "musicbrainz"

    def search_url(self, query: str) -> str:
        return ("https://musicbrainz.org/ws/2/recording/?query=recording:"
                + urllib.parse.quote_plus(query)
                + "&fmt=json&limit=8")

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for row in (payload or {}).get("recordings") or []:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "")
            artist = ""
            credits = row.get("artist-credit") or []
            if credits and isinstance(credits[0], dict):
                artist = str((credits[0].get("artist") or {}).get("name")
                             or credits[0].get("name") or "")
            if not title:
                continue
            mbid = str(row.get("id") or "")
            out.append(SourceCandidate(
                source=self.name, title=title, artist=artist,
                url=f"https://musicbrainz.org/recording/{mbid}" if mbid
                else "",
                downloadable=False,
                extra={"note": "MusicBrainz metadata only — no audio; "
                               "find the track via the full-track sources"}))
        return out


class OpenverseSource(_JsonApiSource):
    """Openverse — keyless CC-licensed audio from across the web.

    api.openverse.org aggregates openly licensed audio (ccMixter,
    Jamendo, Freesound, Wikimedia and more) from every region.  The
    ``url`` field is the direct audio file — full downloadable files,
    not previews.  Anonymous (no key) requests are rate-limited but
    work; a key only raises the limit.
    """

    name = "openverse"

    def search_url(self, query: str) -> str:
        return ("https://api.openverse.org/v1/audio/?q="
                + urllib.parse.quote_plus(query) + "&page_size=8")

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for row in (payload or {}).get("results") or []:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "")
            creator = str(row.get("creator") or "")
            direct = str(row.get("url") or "")
            landing = str(row.get("foreign_landing_url") or "")
            if not title and not direct:
                continue
            out.append(SourceCandidate(
                source=self.name, title=title or query, artist=creator,
                url=direct or landing, direct_url=direct,
                downloadable=bool(direct),
                extra={"note": "CC-licensed audio via Openverse"}))
        return out


class WikimediaSource(_JsonApiSource):
    """Wikimedia Commons — keyless, freely licensed audio worldwide.

    The Commons MediaWiki API searches audio files (music, spoken
    word, field recordings) uploaded from every country.  Candidates
    point at the direct file URL — full downloadable files.
    """

    name = "wikimedia"

    def search_url(self, query: str) -> str:
        return ("https://commons.wikimedia.org/w/api.php?"
                "action=query&generator=search&gsrsearch=filetype%3Aaudio+"
                + urllib.parse.quote_plus(query)
                + "&gsrnamespace=6&gsrlimit=8&prop=imageinfo"
                "&iiprop=url%7Csize%7Cmime&format=json")

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        pages = ((payload or {}).get("query") or {}).get("pages") or {}
        for page in pages.values():
            if not isinstance(page, dict):
                continue
            info = (page.get("imageinfo") or [{}])[0]
            mime = str(info.get("mime") or "")
            url = str(info.get("url") or "")
            if not mime.startswith("audio/") or not url:
                continue
            title = str(page.get("title") or "").split(":", 1)[-1]
            out.append(SourceCandidate(
                source=self.name, title=title or query,
                url=url, direct_url=url, downloadable=True,
                extra={"note": "Freely licensed audio via Wikimedia Commons"}))
        return out


class ArchiveOrgSource(AudioSource):
    """Internet Archive audio — keyless, FULL downloadable files.

    Search hits the ``advancedsearch`` JSON endpoint (title + mediatype
    audio).  Candidates point at the details page; ``download_url``
    resolves via the metadata API to the first real audio file
    (mp3/ogg/flac/m4a) — never guessed, the file list comes from the
    archive itself.
    """

    name = "archiveorg"
    _SEARCH = "https://archive.org/advancedsearch.php"
    _META = "https://archive.org/metadata/{ident}"

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            import json as _json
            url = (self._SEARCH + "?q="
                   + urllib.parse.quote_plus(
                       f"title:({q}) AND mediatype:audio")
                   + "&fl[]=identifier&fl[]=title&fl[]=creator"
                   + f"&rows={max(1, min(limit, 20))}&output=json")
            raw = self._get(url)
            if not raw:
                return []
            docs = (( _json.loads(raw) or {}).get("response") or {}
                    ).get("docs") or []
        except Exception as exc:  # noqa: BLE001
            _log.info("archiveorg: search failed for %r: %s", q, exc)
            return []
        out: list[SourceCandidate] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            ident = str(doc.get("identifier") or "")
            if not ident:
                continue
            title = str(doc.get("title") or ident)
            artist = str(doc.get("creator") or "")
            out.append(SourceCandidate(
                source=self.name, title=title, artist=artist,
                url=f"https://archive.org/details/{ident}",
                extra={"identifier": ident,
                       "note": "Internet Archive — full file, free"}))
        return out

    def download_url(self, candidate: SourceCandidate) -> str:
        ident = (candidate.extra or {}).get("identifier") or ""
        if not ident:
            return ""
        try:
            import json as _json
            raw = self._get(self._META.format(ident=ident))
            if not raw:
                return ""
            files = (_json.loads(raw) or {}).get("files") or []
        except Exception as exc:  # noqa: BLE001
            _log.info("archiveorg: metadata failed for %s: %s", ident, exc)
            return ""
        for f in files:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name") or "")
            if name.lower().endswith((".mp3", ".ogg", ".flac", ".m4a",
                                       ".opus", ".wav")):
                return (f"https://archive.org/download/{ident}/"
                        + urllib.parse.quote(name))
        return ""


class BandcampSource(AudioSource):
    """Bandcamp search — HTML, keyless.  Full-track streams via yt-dlp.

    Bandcamp pages are server-rendered search results; candidates point
    at the album/track page and yt-dlp's bandcamp extractor handles
    download.  Honest: if the markup changes, this returns [] and the
    chain moves on.
    """

    name = "bandcamp"
    _SEARCH = "https://bandcamp.com/search"

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        html = self._get(f"{self._SEARCH}?q={urllib.parse.quote_plus(q)}")
        if not html:
            return []
        out: list[SourceCandidate] = []
        seen: set[str] = set()
        for m in re.finditer(
                r'href="(https?://[a-z0-9.-]+\.bandcamp\.com/(?:album|track)/[^"]+)"[^>]*>(.*?)</a',
                html, re.S | re.I):
            url = m.group(1)
            title = BaseScrapeSource._clean(re.sub(r"<[^>]+>", "", m.group(2)))
            if len(title) < 3 or url in seen:
                continue
            seen.add(url)
            out.append(SourceCandidate(source=self.name, title=title, url=url))
            if len(out) >= limit:
                break
        _log.info("bandcamp: %d candidates for %r", len(out), q)
        return out


class NotJustOkSource(NetNaijaSource):
    """NotJustOk — Nigerian music/news blog (WordPress ?s= search)."""

    name = "notjustok"
    _BASE = "https://notjustok.com"


class TooXclusiveSource(NetNaijaSource):
    """TooXclusive — Nigerian music/charts blog (WordPress ?s= search)."""

    name = "tooxclusive"
    _BASE = "https://tooxclusive.com"


# ── the chain ─────────────────────────────────────────────────────────────

#: Text-search priority for /play.  Regional blogs first (direct MP3s
#: beat metadata), proven globals next, metadata-only last.
#: Each entry: (strategy name, source instance or None for legacy paths).
SOURCE_CHAIN: list[tuple[str, AudioSource]] = [
    ("netnaija-search", NetNaijaSource()),
    ("notjustok-search", NotJustOkSource()),
    ("tooxclusive-search", TooXclusiveSource()),
    ("boomplay-search", BoomplaySource()),
    ("bandcamp-search", BandcampSource()),
    ("archiveorg-search", ArchiveOrgSource()),
    ("musicbrainz-search", MusicBrainzSource()),
    ("openverse-search", OpenverseSource()),
    ("wikimedia-search", WikimediaSource()),
    ("itunes-search", ITunesSource()),
    ("deezer-search", DeezerSource()),
    # soundcloud-search / youtube-search / youtube-music-search /
    # spotify-search stay on their existing paths in resolver.py (API
    # adapters + yt-dlp) — they are proven and this module does not
    # duplicate them.
    # audiomack is URL-first-class (no keyless text search exists).
]

#: Every text-search source /play tries, in chain order.  The resolver
#: builds its chain from this list so the count is single-sourced and
#: testable — the owner rule is MANY sources, never 1–2.
MUSIC_SEARCH_SOURCES: tuple[str, ...] = (
    "workspace-scan",
    "netnaija-search",
    "notjustok-search",
    "tooxclusive-search",
    "soundcloud-search",
    "bandcamp-search",
    "youtube-search",
    "youtube-music-search",
    "archiveorg-search",
    "musicbrainz-search",
    "openverse-search",
    "wikimedia-search",
    "itunes-search",
    "deezer-search",
    "boomplay-search",
    "spotify-search",
)

#: The six world regions coverage is measured against.  Nigeria/Africa
#: is ONE region among many, not the focus — the owner rule is GLOBAL
#: coverage: every region gets sources.
GLOBAL_REGIONS: tuple[str, ...] = (
    "americas", "europe", "asia", "africa", "middle_east", "oceania",
)

#: Region each /play source covers.  "global" = a worldwide catalog
#: (covers every region in GLOBAL_REGIONS).  "local" = the owner's own
#: files, not a region at all.
MUSIC_SOURCE_REGIONS: dict[str, str] = {
    "workspace-scan": "local",
    "netnaija-search": "africa",
    "notjustok-search": "africa",
    "tooxclusive-search": "africa",
    "soundcloud-search": "global",
    "bandcamp-search": "global",
    "youtube-search": "global",
    "youtube-music-search": "global",
    "archiveorg-search": "global",
    "musicbrainz-search": "global",
    "openverse-search": "global",
    "wikimedia-search": "global",
    "itunes-search": "global",
    "deezer-search": "global",
    "boomplay-search": "africa",
    "spotify-search": "global",
}


def music_covered_regions() -> set[str]:
    """World regions /play covers: "global" expands to every region."""
    out: set[str] = set()
    for region in MUSIC_SOURCE_REGIONS.values():
        if region == "global":
            out.update(GLOBAL_REGIONS)
        elif region in GLOBAL_REGIONS:
            out.add(region)
    return out
