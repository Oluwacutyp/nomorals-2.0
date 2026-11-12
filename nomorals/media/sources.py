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
import os
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
    "JamendoSource",
    "FMASource",
    "MdundoSource",
    "BandcampSource",
    "NotJustOkSource",
    "TooXclusiveSource",
    "MusicBrainzSource",
    "OpenverseSource",
    "WikimediaSource",
    "SOURCE_CHAIN",
    "BROWSER_UA",
    "MUSIC_SEARCH_SOURCES",
    "MUSIC_SOURCE_DOMAINS",
    "DiscoveredBlogSource",
    "MUSIC_SOURCE_REGIONS",
    "GLOBAL_REGIONS",
    "music_covered_regions",
    "PREVIEW_ONLY_SOURCES",
    "candidate_asset_class",
]

#: Sources whose APIs serve ONLY a ~30-second preview as their audio
#: field — proven from their response shapes, not their names:
#: the iTunes Search API's only audio field is ``previewUrl`` (an
#: ``AudioPreview…`` asset, ~30s) and Deezer's only audio field is
#: ``preview`` (a ~30s MP3 clip).  Neither API has a full-track audio
#: field at any tier.  Candidates from these sources are PREVIEW-class:
#: identification/metadata (they prove the exact track exists, carry
#: its real full-track duration and artwork) — never download
#: candidates, never pick-list entries, never deliverables.
PREVIEW_ONLY_SOURCES: frozenset[str] = frozenset({"itunes", "deezer"})


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


def candidate_asset_class(candidate: SourceCandidate) -> str:
    """Classify what a candidate's downloadable asset actually IS.

    Returns ``"full"`` (the whole track), ``"preview"`` (a ~30s clip
    from a preview-only field) or ``"metadata"`` (no audio at all —
    protected streams / catalog records).  The classification reads
    the source's own markings: ``extra["preview_only"]`` is set at
    parse time from the API field the URL came from (``previewUrl`` /
    ``preview``), and ``downloadable=False`` is how metadata-only
    sources (Boomplay, MusicBrainz) declare themselves.
    """
    extra = candidate.extra or {}
    if extra.get("preview_only") or candidate.source in PREVIEW_ONLY_SOURCES:
        return "preview"
    if not candidate.downloadable:
        return "metadata"
    return "full"


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
    """Boomplay — massive in Nigeria/Africa. Browser-backed search.

    Honest limitation: there is no official public API, no yt-dlp
    extractor, and streams are protected (candidates are metadata
    only — open in the Boomplay app). The search page is a JS SPA
    shell, so plain-HTTP text search returns NO results (verified
    2026-10-09, re-verified 2026-10-18).

    The working path (verified live 2026-10-18) is the BROWSER path:
    the same requests a real browser makes to the SPA endpoints —
    a session-cookie visit to the homepage (sets ``JSESSIONID`` /
    ``imei`` / ``songLibraryNumStr``), then GET
    ``/search/music/<query>`` with a browser UA + ``Referer`` +
    ``X-Requested-With: XMLHttpRequest``. The returned HTML carries
    real ``songWrap`` buttons whose ``data-data`` payload splits on
    ``@+#`` into ``[0]=song id, [4]=title, [5]=artist``. Back in the
    active chain since the browser-backed search returns real
    candidates (verified live: "Burna Boy" → 10 keywords via
    ``/searchItem``; "Last Last" → 7 songs; "Asake" → 12; "Davido"
    → 12).
    """

    name = "boomplay"
    _BASE = "https://www.boomplay.com"
    _SEARCH = "https://www.boomplay.com/search"
    _DATA_SEP = "@+#"
    #: ``(url) -> html`` — injectable (tests / a real browser fetch).
    #: Class-level so tests can patch it on the class; None = the
    #: default browser-session fetch.
    _browser_fetch: Any = None

    def __init__(self, fetcher: Any | None = None) -> None:
        if fetcher is not None:
            self._browser_fetch = fetcher
        self._opener: Any = None

    # ── browser-session fetch (cookie jar + browser headers) ────
    def _session_fetch(self, url: str) -> str:
        """GET ``url`` the way the SPA does: session cookies + browser
        headers. First call visits the homepage to collect cookies
        (``JSESSIONID``, ``imei``, ``songLibraryNumStr``). Never raises."""
        import http.cookiejar
        import urllib.request

        try:
            if self._opener is None:
                jar = http.cookiejar.CookieJar()
                self._opener = urllib.request.build_opener(
                    urllib.request.HTTPCookieProcessor(jar))
                # Homepage visit: collect the session cookies the SPA
                # endpoints require.
                try:
                    req = urllib.request.Request(
                        self._BASE + "/", headers={
                            "User-Agent": BROWSER_UA,
                            "Accept": "text/html,application/xhtml+xml"})
                    with self._opener.open(req, timeout=_HTTP_TIMEOUT) as resp:
                        resp.read()
                except Exception:  # noqa: BLE001 - cookies best-effort
                    pass
            req = urllib.request.Request(url, headers={
                "User-Agent": BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Referer": self._BASE + "/",
                "X-Requested-With": "XMLHttpRequest"})
            with self._opener.open(req, timeout=_HTTP_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("boomplay: browser fetch failed for %s: %s",
                      url[:80], exc)
            return ""

    # ── parsing ───────────────────────────────────────────────────
    @classmethod
    def _parse_song_wrap(cls, html: str,
                         limit: int) -> list[SourceCandidate]:
        """Parse SPA ``songWrap`` buttons: ``data-data`` payloads split
        on ``@+#`` → ``[0]=song id, [4]=title, [5]=artist``. Falls
        back to ``songName``/``singerName`` + ``data-id`` markup."""
        out: list[SourceCandidate] = []
        seen: set[str] = set()
        for m in re.finditer(r'data-data="([^"]+)"', html, re.S | re.I):
            parts = m.group(1).split(cls._DATA_SEP)
            if len(parts) < 6:
                continue
            song_id = parts[0].strip()
            title = BaseScrapeSource._clean(parts[4])
            artist = BaseScrapeSource._clean(parts[5])
            if not song_id.isdigit() or len(title) < 2 or not artist:
                continue
            if song_id in seen:
                continue
            seen.add(song_id)
            out.append(SourceCandidate(
                source="boomplay", title=title, artist=artist,
                url=f"https://www.boomplay.com/songs/{song_id}",
                downloadable=False,
                extra={"note": "Boomplay streams are protected — "
                               "open in the Boomplay app",
                       "boomplay_id": song_id}))
            if len(out) >= limit:
                return out
        if out:
            return out
        # Fallback: songName/singerName text + a nearby data-id.
        for m in re.finditer(
                r'data-id="(\d+)"[^>]*>.*?'
                r'class="songName"[^>]*>([^<]+)<.*?'
                r'class="singerName"[^>]*>([^<]+)<',
                html, re.S | re.I):
            song_id, title, artist = (m.group(1),
                                      BaseScrapeSource._clean(m.group(2)),
                                      BaseScrapeSource._clean(m.group(3)))
            if song_id in seen or len(title) < 2:
                continue
            seen.add(song_id)
            out.append(SourceCandidate(
                source="boomplay", title=title, artist=artist,
                url=f"https://www.boomplay.com/songs/{song_id}",
                downloadable=False,
                extra={"note": "Boomplay streams are protected — "
                               "open in the Boomplay app",
                       "boomplay_id": song_id}))
            if len(out) >= limit:
                break
        return out

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            # Browser-backed SPA search (the working path).
            fetch = self._browser_fetch or self._session_fetch
            html = fetch(
                f"{self._BASE}/search/music/{urllib.parse.quote(q)}")
            out = self._parse_song_wrap(html or "", limit)
            if out:
                _log.info("boomplay: %d browser-backed candidates for %r",
                          len(out), q)
                return out
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("boomplay: browser search failed for %r: %s", q, exc)
        try:
            # Plain-HTTP fallback (legacy path; usually empty for SPA).
            html = self._get(
                f"{self._SEARCH}?q={urllib.parse.quote_plus(q)}")
            if not html:
                return []
            out = []
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

    Its ONLY audio field is ``previewUrl`` — a ~30-second preview
    asset (``AudioPreview…`` path).  There is no full-track field.
    Candidates are therefore PREVIEW-class: ``preview_only`` in
    ``extra`` and ``downloadable=False``.  iTunes identifies the
    exact track (title, artist, the track's real full length in
    ``trackTimeMillis``, artwork, store page) — it may never supply
    the audio.  The resolver excludes preview-class candidates from
    downloads and pick-lists entirely.
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
                # The preview asset is NOT a deliverable: identification
                # only.  downloadable stays False so no consumer can
                # mistake the 30s clip for the song.
                downloadable=False,
                extra={"preview_only": True,
                       "note": "iTunes 30-second preview — "
                               "identification only, not the track"}))
        return out


class DeezerSource(_JsonApiSource):
    """Deezer public search API — keyless, no account.

    Same shape as iTunes: its only audio field is ``preview``, a
    ~30-second MP3 clip (``duration`` is the FULL track's length —
    identification data, not the clip's).  Candidates are
    PREVIEW-class: ``preview_only`` in ``extra``,
    ``downloadable=False`` — the resolver never downloads them and
    never offers them as picks.
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
                downloadable=False,
                extra={"preview_only": True,
                       "note": "Deezer 30-second preview — "
                               "identification only, not the track"}))
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

    def search_url(self, query: str, limit: int = 8) -> str:
        """The advancedsearch URL this source fetches (registry probe)."""
        return (self._SEARCH + "?q="
                + urllib.parse.quote_plus(
                    f"title:({(query or '').strip()}) AND mediatype:audio")
                + "&fl[]=identifier&fl[]=title&fl[]=creator"
                + f"&rows={max(1, min(limit, 20))}&output=json")

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            import json as _json
            url = self.search_url(q, limit=limit)
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


class JamendoSource(_JsonApiSource):
    """Jamendo — Creative Commons catalogue, full downloadable tracks.

    The v3.0 tracks API is free but NOT keyless: every request needs
    a ``client_id`` (verified live 2026-11-12 — a no-key call fails
    with "Jamendo Api Invalid Client Id Error").  Register a free
    application at devportal.jamendo.com and set
    ``NM_JAMENDO_CLIENT_ID``; without it ``search()`` returns []
    immediately (no wasted round-trip, no invented key — the chain
    moves on and the registry records a no-result, never a failure).
    The ``audio`` / ``audiodownload`` fields are the FULL track as
    MP3 — Jamendo serves whole CC songs, never 30-second previews —
    so candidates are FULL-class (``candidate_asset_class`` == "full").
    """

    name = "jamendo"
    _API = "https://api.jamendo.com/v3.0/tracks/"

    @staticmethod
    def _client_id() -> str:
        return os.environ.get("NM_JAMENDO_CLIENT_ID", "").strip()

    def search_url(self, query: str) -> str:
        """The tracks URL this source fetches (registry probe)."""
        params = ("format=json&search="
                  + urllib.parse.quote_plus((query or "").strip())
                  + "&limit=8")
        cid = self._client_id()
        if cid:
            params = ("client_id=" + urllib.parse.quote_plus(cid)
                      + "&" + params)
        return self._API + "?" + params

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        if not self._client_id():
            _log.info("jamendo: no NM_JAMENDO_CLIENT_ID set — register a "
                      "free app at devportal.jamendo.com and set it; "
                      "skipping text search for %r", (query or "")[:60])
            return []
        return super().search(query, limit=limit)

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for row in (payload or {}).get("results") or []:
            if not isinstance(row, dict):
                continue
            direct = (str(row.get("audiodownload") or "")
                      or str(row.get("audio") or ""))
            if not direct:
                continue  # no audio field = not a candidate, never metadata
            title = str(row.get("name") or "")
            artist = str(row.get("artist_name") or "")
            page = str(row.get("shareurl") or "")
            out.append(SourceCandidate(
                source=self.name, title=title or query, artist=artist,
                url=direct, direct_url=direct,
                duration=float(row.get("duration") or 0),
                downloadable=True,
                extra={"page": page,
                       "album": str(row.get("album_name") or ""),
                       "license": str(row.get("license_ccurl") or ""),
                       "note": "Jamendo — full track, CC-licensed"}))
        return out


class FMASource(AudioSource):
    """Free Music Archive — CC catalogue, HTML search, metadata-class.

    The WFMU-era JSON API is DEAD — verified live 2026-11-12, NOT
    assumed: ``/api``, ``/api/trackSearch`` and ``/recent.json``
    all answer 404, and there is no documented public API left on
    the current (Tribe of Noise) site.  So this source does NOT
    speak a JSON API: it reads the live, server-rendered search
    page (``/search?quicksearch=<q>``, verified live 2026-11-12 —
    174,634 tracks in the unfiltered index) the way BandcampSource
    reads Bandcamp: track rows carry the title, artist, album and
    ``MM:SS`` duration, each title linking to its track page at
    ``/music/<Artist>/<album>/<track>/``.

    Honest classification: METADATA-class
    (``candidate_asset_class`` == "metadata",
    ``downloadable=False``).  The search page proves the track
    exists and where its page lives; it does NOT expose a direct
    audio URL, and downloads on the current site may need an
    account — so no audio URL is invented here and the chain must
    never WIN on an FMA hit for audio.  The resolver routes FMA
    through the metadata path (MusicBrainz shape): identification
    + track page, never a download.  Promotion to full-track needs
    a live-browser probe of a track page pinning the download href
    (or the account wall) — named here, not silently assumed.

    Parser shape, from the live page (2026-11-12): a track row is
    a 3-segment ``/music/`` anchor (artist/album/track), followed
    by a 1-segment artist anchor, a 2-segment album anchor and a
    ``M:SS`` duration (minutes may exceed 59 — "65:36" observed).
    Anchor inner markup is tag-stripped; a row missing artist,
    album or duration still yields its track (title + page URL),
    never dropped for missing decoration.  Changed markup degrades
    to [] and the chain moves on (Bandcamp contract).
    """

    name = "fma"
    _SEARCH = "https://freemusicarchive.org/search"
    _SITE = "https://freemusicarchive.org"

    #: A track link has exactly three /music/ segments; artist links
    #: have one, album links two.  Segment counts are the discriminator.
    _TRACK_RE = re.compile(
        r'<a\s+[^>]*href=["\']((?:https?://freemusicarchive\.org)?'
        r'/music/[^/"\'<>]+/[^/"\'<>]+/[^/"\'<>]+/?)["\'][^>]*>'
        r'(.*?)</a>', re.I | re.S)
    _ANCHOR_RE = re.compile(
        r'<a\s+[^>]*href=["\']((?:https?://freemusicarchive\.org)?'
        r'/music/[^/"\'<>]+(?:/[^/"\'<>]+){0,2}/?)["\'][^>]*>'
        r'(.*?)</a>', re.I | re.S)
    _DURATION_RE = re.compile(r"(\d+):(\d{2})")

    def search_url(self, query: str) -> str:
        """The search URL this source fetches (registry probe)."""
        return (f"{self._SEARCH}?quicksearch="
                + urllib.parse.quote_plus((query or "").strip())
                + "&pageSize=20&sort=track")

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        html = self._get(self.search_url(q))
        if not html:
            return []
        try:
            out = self._parse_rows(html, q)
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("fma: parse failed for %r: %s", q, exc)
            return []
        _log.info("fma: %d candidates for %r", len(out), q)
        return out[:limit]

    def _parse_rows(self, html: str,
                    query: str) -> list[SourceCandidate]:
        tracks = list(self._TRACK_RE.finditer(html))
        out: list[SourceCandidate] = []
        seen: set[str] = set()
        for i, m in enumerate(tracks):
            href = m.group(1)
            url = (href if href.startswith("http") else self._SITE + href)
            if url in seen:
                continue
            seen.add(url)
            title = BaseScrapeSource._clean(
                re.sub(r"<[^>]+>", "", m.group(2)))
            if len(title) < 2:
                continue
            # Row window: until the next track anchor (or 800 chars).
            end = (tracks[i + 1].start() if i + 1 < len(tracks)
                   else m.end() + 800)
            window = html[m.end():min(end, m.end() + 800)]
            artist = ""
            album = ""
            for am in self._ANCHOR_RE.finditer(window):
                ahref = am.group(1)
                segs = [s for s in ahref.split("/music/", 1)[-1]
                        .strip("/").split("/") if s]
                text = BaseScrapeSource._clean(
                    re.sub(r"<[^>]+>", "", am.group(2)))
                if len(segs) == 1 and not artist:
                    artist = text
                elif len(segs) == 2 and not album:
                    album = text
                if artist and album:
                    break
            duration = 0.0
            dm = self._DURATION_RE.search(window)
            if dm:
                duration = float(int(dm.group(1)) * 60
                                 + int(dm.group(2)))
            out.append(SourceCandidate(
                source=self.name, title=title or query, artist=artist,
                url=url, duration=duration, downloadable=False,
                extra={"album": album,
                       "note": "Free Music Archive — CC catalogue; "
                               "track identified, no audio: download "
                               "via the track page (audio URL "
                               "unverified, account may be required); "
                               "check the track page for the licence"}))
        return out


class JioSaavnSource(_JsonApiSource):
    """JioSaavn — Indian streaming catalogue, metadata-class.

    Verified live 2026-11-12, NOT assumed: the official search API
    answers keyless and anonymous — GET
    ``https://www.jiosaavn.com/api.php?__call=search.getResults&q=<q>&p=1&n=<n>&_format=json&_marker=0&ctx=web6dot0``
    returns HTTP 200 JSON ``{"total": N, "start": 1, "results":
    [...]}`` with title, artist, album, year, duration, track page
    and cover per row.  No account, no key, no cookies — a plain GET
    with a browser UA.  (The third-party wrapper saavn.dev is NOT
    used: it answers an empty reply from this box, curl 52, on both
    the root and the api subdomain, 2026-11-12.)

    Honest classification: METADATA-class
    (``candidate_asset_class`` == "metadata",
    ``downloadable=False``).  The search response carries NO direct
    audio URL: the audio lives behind ``encrypted_media_url`` (an
    encrypted token) plus a per-track ``song.generateAuthToken``
    POST exchange — the path yt-dlp's JioSaavn extractor uses to get
    an ``auth_url`` — and ``media_preview_url`` is a preview only,
    never a full track.  The token exchange was NOT verified live
    from this box (the POST timed out after 30s with 0 bytes,
    2026-11-12), so no audio is claimed here and the chain must
    never WIN on a JioSaavn hit for audio.  The resolver routes
    JioSaavn through the metadata path (MusicBrainz shape):
    identification + track page, never a download.  Promotion to
    full-track needs the token exchange proven live (an ``auth_url``
    answering audio) — named here, not silently assumed.  Every row
    observed so far carries ``is_drm: 1`` — recorded, never chased.

    Search is TOKEN-LENIENT (verified 2026-11-12: the nonsense query
    "xyqz totally unknown 999" returns total=329 with 5 rows, the
    same class as Mdundo's token-lenient search) — a JioSaavn hit
    never proves the query matched; the metadata path only
    identifies, so leniency is recorded, not hidden.

    Parser shape, from the live response (2026-11-12): each row has
    ``song`` (HTML-entity encoded — unescape it), ``primary_artists``
    (comma string), ``album``, ``year``, ``duration`` (seconds as a
    string), ``perma_url`` (track page), ``image`` (150x150 cover),
    ``language``.  A row missing title or track page is skipped; a
    changed shape degrades to [] and the chain moves on.
    """

    name = "jiosaavn"
    _API = "https://www.jiosaavn.com/api.php"

    def search_url(self, query: str) -> str:
        """The search URL this source fetches (registry probe)."""
        return (self._API + "?__call=search.getResults&q="
                + urllib.parse.quote_plus((query or "").strip())
                + "&p=1&n=10&_format=json&_marker=0&ctx=web6dot0")

    def parse(self, payload: Any, query: str) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        seen: set[str] = set()
        for row in (payload or {}).get("results") or []:
            if not isinstance(row, dict):
                continue
            title = html_mod.unescape(str(row.get("song") or "")).strip()
            page = str(row.get("perma_url") or "").strip()
            if len(title) < 2 or not page.startswith("http"):
                continue
            if page in seen:
                continue
            seen.add(page)
            try:
                duration = float(str(row.get("duration") or "0"))
            except (TypeError, ValueError):
                duration = 0.0
            out.append(SourceCandidate(
                source=self.name, title=title or query,
                artist=html_mod.unescape(
                    str(row.get("primary_artists") or "")).strip(),
                url=page, duration=duration, downloadable=False,
                extra={"album": html_mod.unescape(
                    str(row.get("album") or "")).strip(),
                       "year": str(row.get("year") or ""),
                       "language": str(row.get("language") or ""),
                       "image": str(row.get("image") or ""),
                       "note": "JioSaavn — Indian streaming catalogue; "
                               "track identified, no audio: the search "
                               "response carries no direct audio URL "
                               "(audio needs a per-track auth-token "
                               "exchange, unverified live); open the "
                               "track page or the JioSaavn app"}))
        return out


class MdundoSource(AudioSource):
    """Mdundo — East African music downloads, HTML search, full-track.

    The live site (verified 2026-11-12, NOT assumed) is
    ``play.mdundo.com`` — ``mdundo.com`` geo-redirects there — and
    there is NO public search API: ``/api/search`` answers 404 and
    the app JS exposes only ``/api/songlist/<type>/<id>`` (artist /
    playlist / genre, no search), so this source reads the live,
    server-rendered search page (``/search-results?term=<q>``,
    32 songs on a real results page for "diamond platnumz") the
    way FMASource reads FMA.  Each song item carries
    ``data-url-play="https://mdundo.com/stream/<id>"``,
    ``data-url-download="https://mdundo.com/dl/<id>?pwa=1"``,
    ``data-title`` / ``data-artist`` / ``data-duration-seconds`` /
    ``data-is-premium`` on one element.

    Honest classification: FULL-class (``candidate_asset_class``
    == "full", ``downloadable=True``).  The stream URL is a REAL
    full-track MP3, verified end to end 2026-11-12:
    ``/stream/1571564`` 302s to a signed
    ``mdundosound.com/song_get/?...&t=Diamond-Platnumz-Sikomi.mp3``
    answering HTTP 200, ``Content-Type: audio/mpeg``,
    ``Content-Length: 1872715``, ``Accept-Ranges: bytes`` — a whole
    file, not a preview.  The candidate URL is the STABLE
    ``/stream/<id>`` redirector, never the signed URL: the
    signature is fresh per request and minutes-lived (``e=`` is a
    Unix timestamp ~now), so a stored signed URL would rot.
    The ``/dl/<id>`` link is NOT a direct file — it 302s through
    ``/dl/<id>/high/go`` to the artist page with ``#autoplay`` —
    so it is recorded in ``extra`` as the download page, never
    claimed as the audio URL.  No login wall exists (every fetch
    above was anonymous, no cookies); auth is none, no key, no
    account.  Premium rows (``data-is-premium="1"``) are skipped:
    they cannot stream without a subscription.  Search is
    TOKEN-LENIENT — a nonsense query still returns token-matched
    songs (verified 2026-11-12: "xyqz totally unknown 999" returns
    43 rows, incl. songs titled "TOTALLY" and "unknown"), the same
    class as the WordPress blogs' ?s= search — so hermetic tests
    must neutralize this source like the other live ones, and the
    chain's first-hit-wins order is what bounds it.  Changed markup
    degrades to [] and the chain moves on (Bandcamp contract).
    """

    name = "mdundo"
    _SEARCH = "https://play.mdundo.com/search-results"
    _SITE = "https://play.mdundo.com"

    _PLAY_RE = re.compile(
        r'data-url-play="(https://mdundo\.com/stream/(\d+))"')
    _TITLE_RE = re.compile(r'data-title="([^"]*)"')
    _ARTIST_RE = re.compile(r'data-artist="([^"]*)"')
    _DURATION_RE = re.compile(r'data-duration-seconds="(\d+)"')
    _PREMIUM_RE = re.compile(r'data-is-premium="(\d+)"')
    _DOWNLOAD_RE = re.compile(r'data-url-download="([^"]*)"')

    def search_url(self, query: str) -> str:
        """The search URL this source fetches (registry probe)."""
        return (f"{self._SEARCH}?term="
                + urllib.parse.quote_plus((query or "").strip()))

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        html = self._get(self.search_url(q))
        if not html:
            return []
        try:
            out = self._parse_rows(html, q)
        except Exception as exc:  # noqa: BLE001 — never raises
            _log.info("mdundo: parse failed for %r: %s", q, exc)
            return []
        _log.info("mdundo: %d candidates for %r", len(out), q)
        return out[:limit]

    @staticmethod
    def _unescape(html: str) -> str:
        """Decode the song list's escaped markup to plain HTML.

        The search page embeds the song list as an escaped payload
        (``\\/``, ``\\&quot;``, ``\\u0022``, ``&gt;``/``&lt;``/
        ``&amp;`` — verified against the live page 2026-11-12).
        Plain HTML passes through unchanged, so this is safe on
        either rendering.
        """
        norm = re.sub(r"\\+/", "/", html)
        norm = re.sub(r"\\+&quot;", '"', norm)
        norm = re.sub(r"\\+u0022", '"', norm)
        return (norm.replace("&quot;", '"').replace("&gt;", ">")
                    .replace("&lt;", "<").replace("&amp;", "&"))

    def _parse_rows(self, html: str,
                    query: str) -> list[SourceCandidate]:
        norm = self._unescape(html)
        out: list[SourceCandidate] = []
        seen: set[str] = set()
        # One song item = one element tag; every attribute of the
        # song sits in that tag, so split on ">" and read each tag.
        for chunk in norm.split(">"):
            m = self._PLAY_RE.search(chunk)
            if not m:
                continue
            url = m.group(1)
            if url in seen:
                continue
            seen.add(url)
            premium = self._PREMIUM_RE.search(chunk)
            if premium and premium.group(1) == "1":
                continue  # subscription-only: cannot stream, never a hit
            title_m = self._TITLE_RE.search(chunk)
            title = BaseScrapeSource._clean(
                title_m.group(1)) if title_m else ""
            if len(title) < 2:
                continue
            artist_m = self._ARTIST_RE.search(chunk)
            artist = BaseScrapeSource._clean(
                artist_m.group(1)) if artist_m else ""
            duration = 0.0
            dur_m = self._DURATION_RE.search(chunk)
            if dur_m:
                duration = float(int(dur_m.group(1)))
            dl_m = self._DOWNLOAD_RE.search(chunk)
            download = dl_m.group(1) if dl_m else ""
            out.append(SourceCandidate(
                source=self.name, title=title or query, artist=artist,
                url=url, direct_url=url, duration=duration,
                downloadable=True,
                extra={"download_page": download,
                       "note": "Mdundo — full track MP3 via the "
                               "stable /stream/<id> redirector "
                               "(signed mdundosound.com URL, "
                               "minutes-lived — never stored); the "
                               "/dl/ link is a page flow, not a "
                               "direct file"}))
        return out


class BandcampSource(AudioSource):
    """Bandcamp search — HTML, keyless.  Full-track streams via yt-dlp.

    Bandcamp pages are server-rendered search results; candidates point
    at the album/track page and yt-dlp's bandcamp extractor handles
    download.  Honest: if the markup changes, this returns [] and the
    chain moves on.
    """

    name = "bandcamp"
    _SEARCH = "https://bandcamp.com/search"

    def search_url(self, query: str) -> str:
        """The search URL this source fetches (registry probe)."""
        return (f"{self._SEARCH}?q="
                f"{urllib.parse.quote_plus((query or '').strip())}")

    def search(self, query: str, limit: int = 8) -> list[SourceCandidate]:
        q = (query or "").strip()
        if not q:
            return []
        html = self._get(self.search_url(q))
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


class DiscoveredBlogSource(NetNaijaSource):
    """A download blog DISCOVERED on the music path, promoted by the
    source registry's health loop (Pass #88, open-sources law).

    The registry (``source_registry.py``) is the growth pipeline: a
    host encountered on the real path is noted as an UNTRUSTED
    candidate, serves nothing, and only the health loop can promote
    it (origin ``grown``).  A promoted host becomes searchable through
    this class — the generic WordPress shape the scrape framework
    already implements (``?s=`` search, entry-title result parsing,
    multi-hop download pages ending in a direct media link), pointed
    at the discovered domain.  Sites that are not WordPress-shaped
    simply return [] / "" and the chain moves on, honestly.
    """

    def __init__(self, domain: str) -> None:
        super().__init__()
        host = (domain or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        self.domain = host
        self._BASE = f"https://{host}"
        self.name = f"grown:{host}"


# ── the chain ─────────────────────────────────────────────────────────────

#: Text-search priority for /play.  Regional blogs first (direct MP3s
#: beat metadata), proven globals next, metadata-only last.
#: Each entry: (strategy name, source instance or None for legacy paths).
SOURCE_CHAIN: list[tuple[str, AudioSource]] = [
    ("netnaija-search", NetNaijaSource()),
    ("notjustok-search", NotJustOkSource()),
    ("tooxclusive-search", TooXclusiveSource()),
    ("mdundo-search", MdundoSource()),
    ("bandcamp-search", BandcampSource()),
    ("archiveorg-search", ArchiveOrgSource()),
    ("jamendo-search", JamendoSource()),
    ("fma-search", FMASource()),
    ("jiosaavn-search", JioSaavnSource()),
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
    "mdundo-search",
    "soundcloud-search",
    "bandcamp-search",
    "youtube-search",
    "youtube-music-search",
    "archiveorg-search",
    "jamendo-search",
    "fma-search",
    "jiosaavn-search",
    "boomplay-search",
    "musicbrainz-search",
    "openverse-search",
    "wikimedia-search",
    "itunes-search",
    "deezer-search",
    "spotify-search",
)

#: The host each music chain strategy actually fetches from — the
#: strategy's source in the ONE source registry (Pass #88).  This map
#: is the music class's membership list: the resolver credits search /
#: resolve outcomes to the registry descriptor for this domain, the
#: registry's status surface renders the class from it, and the health
#: loop verifies these hosts.  Domains are evidence from this module
#: (each source's ``_BASE`` / API host), NOT the site's marketing
#: domain — Deezer search fetches ``api.deezer.com``, Openverse
#: ``api.openverse.org``, and the chain's NetNaija is ``netnaija.com``
#: (the registry's older download-blog seed is ``thenetnaija.net``, a
#: different host that also exists).  ``None`` = no host (the owner's
#: own files).  Grown sources are not listed here — they are not
#: strategies; the resolver appends them from the registry itself.
MUSIC_SOURCE_DOMAINS: dict[str, str | None] = {
    "workspace-scan": None,
    "netnaija-search": "netnaija.com",
    "notjustok-search": "notjustok.com",
    "tooxclusive-search": "tooxclusive.com",
    "mdundo-search": "play.mdundo.com",
    "soundcloud-search": "soundcloud.com",
    "bandcamp-search": "bandcamp.com",
    "youtube-search": "youtube.com",
    "youtube-music-search": "youtube.com",
    "archiveorg-search": "archive.org",
    "jamendo-search": "api.jamendo.com",
    "fma-search": "freemusicarchive.org",
    "jiosaavn-search": "jiosaavn.com",
    "boomplay-search": "boomplay.com",
    "musicbrainz-search": "musicbrainz.org",
    "openverse-search": "api.openverse.org",
    "wikimedia-search": "commons.wikimedia.org",
    "itunes-search": "itunes.apple.com",
    "deezer-search": "api.deezer.com",
    "spotify-search": "open.spotify.com",
    "audiomack": "audiomack.com",
}

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
    "mdundo-search": "africa",
    "soundcloud-search": "global",
    "bandcamp-search": "global",
    "youtube-search": "global",
    "youtube-music-search": "global",
    "archiveorg-search": "global",
    "jamendo-search": "global",
    "fma-search": "global",
    "jiosaavn-search": "asia",
    "boomplay-search": "africa",
    "musicbrainz-search": "global",
    "openverse-search": "global",
    "wikimedia-search": "global",
    "itunes-search": "global",
    "deezer-search": "global",
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
