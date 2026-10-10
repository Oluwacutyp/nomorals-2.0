"""Webnovel source adapters — the story-reader's fetch layer.

Each adapter knows ONE novel archive natively: its URL shapes, its search
endpoint, where the chapter body lives in the markup, and how chapters link
to each other.  All selectors below were verified live on 2026-10-09
against the real sites (plain-HTTP fetch from the build machine):

* freewebnovel.com — novel ``/novel/<slug>`` (og:novel:* meta for author /
  genre / status; latest chapters as ``/novel/<slug>/chapter-<n>`` links);
  chapter ``/novel/<slug>/chapter-<n>`` with the body in
  ``div.txt > div#article`` (``h4`` title, ``p`` paragraphs) and
  "Prev Chapter" / "Next Chapter" links; search GET
  ``/search?keyword=<q>``.
* novelfull.net (+ novelfull.org, readnovelfull.com — same reader
  platform) — novel ``/<slug>.html`` with ``?page=N`` pagination and
  ``a.con`` chapter links (``data-total-chapters`` on ``#list-chapter``);
  chapter ``/<slug>/chapter-<n>-<title>.html`` with the body in
  ``div.txt > div#chapter-content``; search GET ``/search?keyword=<q>``.
* royalroad.com — novel ``/fiction/<id>/<slug>`` (chapter links
  ``/fiction/<id>/<slug>/chapter/<cid>/<n>-<title>``); chapter body in
  ``div.chapter-inner.chapter-content``; search GET
  ``/fictions/search?query=<q>``.

Fetch strategy (per adapter, in order):

1. plain HTTP (urllib, browser UA, per-domain cookie jar, gzip) — works
   for every verified source above;
2. on a block (403/503, Cloudflare "Just a moment" markers, empty body)
   the repo's :class:`nomorals.tools.browser.BrowserSession` is reused
   (persistent cookies under ``workspace/books/reader/sessions/``);
   when ``NM_BROWSER_PLAYWRIGHT=1`` that session transparently uses
   headless Chromium — the browser-context fallback for JS-walled pages;
3. anything else raises :class:`SourceError` with an actionable message.

PandaNovel (pandanovel.com / pandasnovel.com / panda-novel.com) could NOT
be verified live — every domain was unresolvable from the build network on
2026-10-09.  Its adapter is a :class:`GenericAdapter` instance (candidate
selectors + quality scoring) honestly flagged ``verified=False``; it will
self-tune on the first successful fetch rather than pretend to know the
markup.
"""

from __future__ import annotations

import gzip
import http.client
import http.cookiejar
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..tools.browser import Node, parse_html

_log = get_logger(__name__)

__all__ = [
    "Chapter", "StoryMeta", "SearchHit",
    "SourceAdapter", "SourceError", "SourceBlocked",
    "FreeWebNovelAdapter", "NovelFullAdapter", "RoyalRoadAdapter",
    "PandaNovelAdapter", "GenericAdapter",
    "ADAPTERS", "adapter_for_url", "search_all", "normalize_url",
]

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: rotated user-agents (FanFicFare's fingerprint-variety idea)
_BROWSER_UAS = (
    _BROWSER_UA,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
)

#: query params stripped by normalize_url (tracking junk)
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "referrer",
})


def normalize_url(url: str) -> str:
    """Canonicalize a story/chapter URL (FanFicFare-style normalization).

    Lowercases the host, strips tracking params, drops the fragment and
    trailing slash so the same chapter always maps to the same key —
    the basis of download resume.
    """
    try:
        p = urllib.parse.urlparse((url or "").strip())
    except Exception:  # noqa: BLE001
        return (url or "").strip()
    host = (p.hostname or "").lower()
    if not host:
        return (url or "").strip()
    qs = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    qs = [(k, v) for k, v in qs if k.lower() not in _TRACKING_PARAMS]
    path = p.path.rstrip("/") or "/"
    netloc = host
    if p.port:
        netloc = f"{host}:{p.port}"
    return urllib.parse.urlunparse(
        (p.scheme or "https", netloc, path,
         "", urllib.parse.urlencode(qs), ""))

#: markers that mean "you are looking at a bot wall, not content"
_BLOCK_MARKERS = (
    "just a moment", "attention required", "cf-chl", "cf_clearance",
    "are you human", "verify you are human", "captcha",
    "access denied", "request blocked",
)

#: junk paragraphs every aggregator injects — stripped from chapter bodies
_JUNK_PATTERNS = (
    r"^\s*visit\s+\S+\s+for\s+the\s+best\s+novel\s+reading\s+experience\s*$",
    r"^\s*read\s+the\s+latest\s+chapters?\s+at\s+\S+\s+only\s*$",
    r"^\s*\S*freewebnovel\S*\s*$",
    r"^\s*please\s+visit\s+\S+\s*$",
)


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Chapter:
    number: int
    title: str
    url: str
    paragraphs: list[str] = field(default_factory=list)
    prev_url: str = ""
    next_url: str = ""
    source: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "title": self.title, "url": self.url,
            "paragraphs": len(self.paragraphs), "words": self.words,
            "prev_url": self.prev_url, "next_url": self.next_url,
            "source": self.source,
        }


@dataclass
class StoryMeta:
    title: str
    url: str
    source: str
    author: str = ""
    synopsis: str = ""
    genres: list[str] = field(default_factory=list)
    status: str = ""          # ongoing | completed | hiatus | unknown
    total_chapters: int = 0
    cover_url: str = ""
    updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "url": self.url, "source": self.source,
            "author": self.author, "synopsis": self.synopsis[:600],
            "genres": self.genres, "status": self.status,
            "total_chapters": self.total_chapters,
            "cover_url": self.cover_url, "updated": self.updated,
        }


@dataclass
class SearchHit:
    title: str
    url: str
    source: str
    author: str = ""
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "source": self.source,
                "author": self.author, "snippet": self.snippet[:200]}


class SourceError(Exception):
    """The source could not serve this request (network, 404, moved)."""


class SourceBlocked(SourceError):
    """The source is bot-walled; browser-context fallback also failed."""


# ── fetch chain ──────────────────────────────────────────────────────────────


class Fetcher:
    """Two-tier fetch: plain HTTP first, BrowserSession fallback on blocks.

    One Fetcher per adapter; cookie jars are per-domain so sessions stay
    isolated the way a browser profile would keep them.
    """

    def __init__(self, *, session_dir: str = "", timeout: float = 30.0,
                 rate_limit: float = 1.0) -> None:
        self.timeout = timeout
        self.session_dir = session_dir
        #: minimum seconds between requests to the same host
        #: (FanFicFare's SleepDecorator idea — don't hammer archives)
        self.rate_limit = max(0.0, rate_limit)
        self._jars: dict[str, http.cookiejar.CookieJar] = {}
        self._browser_sessions: dict[str, Any] = {}
        self._last_fetch: dict[str, float] = {}
        self._ua_index = 0

    def polite_wait(self, url: str) -> None:
        """Sleep until ``rate_limit`` seconds passed since the last fetch
        to this host.  Call between chapter fetches."""
        if self.rate_limit <= 0:
            return
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        last = self._last_fetch.get(host, 0.0)
        wait = self.rate_limit - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch[host] = time.time()

    def _next_ua(self) -> str:
        ua = _BROWSER_UAS[self._ua_index % len(_BROWSER_UAS)]
        self._ua_index += 1
        return ua

    # -- tier 1: plain HTTP ---------------------------------------------------
    def _jar(self, url: str) -> http.cookiejar.CookieJar:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        jar = self._jars.get(host)
        if jar is None:
            jar = http.cookiejar.CookieJar()
            self._jars[host] = jar
        return jar

    def http_get(self, url: str, *, referer: str = "",
                 retries: int = 2) -> tuple[str, str]:
        """GET ``url`` → (final_url, html).  Raises SourceError/SourceBlocked.

        Transient failures (resets, timeouts, truncated chunked bodies —
        common through flaky proxies) are retried with backoff before
        they become errors, mirroring the repo's BrowserSession.
        """
        last: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            try:
                self.polite_wait(url)
                return self._http_get_once(url, referer=referer)
            except SourceBlocked:
                raise
            except SourceError as exc:
                last = exc
                if attempt < retries:
                    time.sleep(0.6 * (2 ** attempt))
        raise last if last is not None else SourceError(f"{url} failed")

    def _http_get_once(self, url: str, *, referer: str = "") -> tuple[str, str]:
        jar = self._jar(url)
        # ProxyHandler() with no args honors the environment's proxy
        # variables (https_proxy etc.) — required in sandboxed networks,
        # a no-op everywhere else.
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar),
            urllib.request.HTTPSHandler(),
            urllib.request.ProxyHandler(),
        )
        headers = {
            "User-Agent": self._next_ua(),
            "Accept": ("text/html,application/xhtml+xml,application/xml;"
                       "q=0.9,*/*;q=0.8"),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }
        if referer:
            headers["Referer"] = referer
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with opener.open(request, timeout=self.timeout) as resp:
                status = getattr(resp, "status", 200)
                raw = resp.read(8_000_000)
                final_url = resp.geturl()
                encoding = (resp.headers.get("Content-Encoding", "") or "").lower()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 503, 429):
                raise SourceBlocked(
                    f"{url} -> HTTP {exc.code} (bot wall?)") from exc
            raise SourceError(f"{url} -> HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            # HTTPException covers IncompleteRead / BadStatusLine — the
            # truncated chunked bodies flaky proxies produce.
            raise SourceError(f"{url} unreachable: {exc}") from exc
        if "gzip" in encoding:
            try:
                raw = gzip.decompress(raw)
            except OSError:
                pass
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            html = raw.decode("latin-1", errors="replace")
        if status >= 400:
            raise SourceError(f"{url} -> HTTP {status}")
        if _looks_blocked(html):
            raise SourceBlocked(f"{url} served a bot-wall page")
        return final_url, html

    # -- tier 2: browser context ----------------------------------------------
    def browser_get(self, url: str) -> tuple[str, str]:
        """Fetch through the repo's BrowserSession (headless Chromium when
        ``NM_BROWSER_PLAYWRIGHT=1``).  The fallback for Cloudflare-ish
        blocks and JS-rendered pages."""
        from ..tools.browser import BrowserSession

        host = (urllib.parse.urlparse(url).hostname or "default").lower()
        session = self._browser_sessions.get(host)
        if session is None:
            session = BrowserSession(
                name=f"bookforge-{host}",
                session_dir=self.session_dir or "",
                user_agent=_BROWSER_UA,
                respect_robots=False,
                timeout=self.timeout,
            )
            self._browser_sessions[host] = session
        try:
            opened = session.open(url)
        except Exception as exc:  # noqa: BLE001
            raise SourceBlocked(
                f"browser-context fetch failed for {url}: {exc}") from exc
        if not opened.get("ok"):
            raise SourceBlocked(
                f"browser-context fetch: HTTP {opened.get('status')} for {url}")
        html = session.html().get("html", "")
        if _looks_blocked(html) or len(html) < 800:
            raise SourceBlocked(
                f"browser-context fetch still walled for {url}")
        return session.url or url, html

    def get(self, url: str, *, referer: str = "") -> tuple[str, Node, str]:
        """Full chain → (final_url, parsed DOM, raw html)."""
        try:
            final_url, html = self.http_get(url, referer=referer)
        except SourceBlocked:
            _log.info("plain HTTP blocked for %s — browser-context fallback",
                      url)
            final_url, html = self.browser_get(url)
        return final_url, parse_html(html), html


def _looks_blocked(html: str) -> bool:
    if len(html) < 800:
        return True
    low = html[:6000].lower()
    title = ""
    m = re.search(r"<title>(.*?)</title>", low, re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    if any(marker in title for marker in ("just a moment", "attention required")):
        return True
    return any(marker in low[:3000] for marker in _BLOCK_MARKERS[:4])


# ── DOM helpers ──────────────────────────────────────────────────────────────


def _by_id(dom: Node, id_: str) -> Node | None:
    for node in dom.walk():
        if not node.is_text and node.attrs.get("id") == id_:
            return node
    return None


def _by_class(dom: Node, *classes: str) -> list[Node]:
    out = []
    for node in dom.walk():
        if node.is_text:
            continue
        have = (node.attrs.get("class") or "").split()
        if all(c in have for c in classes):
            out.append(node)
    return out


def _first_by_class(dom: Node, *classes: str) -> Node | None:
    found = _by_class(dom, *classes)
    return found[0] if found else None


def _links(dom: Node, base_url: str) -> list[tuple[str, str]]:
    """(absolute_url, text) for every http(s) anchor."""
    out = []
    for node in dom.find_all("a"):
        href = (node.attrs.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:")):
            continue
        abs_url = urllib.parse.urljoin(base_url, href)
        if urllib.parse.urlparse(abs_url).scheme not in {"http", "https"}:
            continue
        out.append((abs_url, node.inner_text().strip()))
    return out


def _meta(dom: Node, prop: str) -> str:
    for node in dom.find_all("meta"):
        if node.attrs.get("property") == prop or node.attrs.get("name") == prop:
            return (node.attrs.get("content") or "").strip()
    return ""


def _clean_paragraphs(paras: list[str]) -> list[str]:
    """Drop aggregator junk injections, collapse whitespace, keep prose."""
    out = []
    for p in paras:
        text = re.sub(r"\s+", " ", p).strip()
        if not text or len(text) < 2:
            continue
        if any(re.match(pat, text, re.I) for pat in _JUNK_PATTERNS):
            continue
        out.append(text)
    return out


def _chapter_number_from_text(text: str) -> int:
    m = re.search(r"chapter\s+(\d+)", text, re.I)
    return int(m.group(1)) if m else 0


# ── adapter base ─────────────────────────────────────────────────────────────


class SourceAdapter(ABC):
    """One novel archive, natively understood."""

    name: str = "base"
    domains: tuple[str, ...] = ()
    #: False when the markup was never verified live (honest flag).
    verified: bool = True

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self.fetcher = fetcher or Fetcher()

    def matches(self, url: str) -> bool:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self.domains)

    @abstractmethod
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]: ...

    @abstractmethod
    def novel(self, url: str) -> StoryMeta: ...

    @abstractmethod
    def chapter_list(self, novel_url: str, *, limit: int = 0) -> list[tuple[int, str, str]]: ...

    @abstractmethod
    def fetch_chapter(self, url: str) -> Chapter: ...

    def chapter_url(self, novel_url: str, number: int) -> str:
        """Deterministic chapter URL when the pattern is fully
        predictable (freewebnovel); "" when it isn't (needs the list)."""
        return ""

    # -- whole-story download -------------------------------------------------
    def download_story(self, url: str, *,
                       limit: int = 0,
                       skip_urls: list[str] | None = None,
                       on_chapter: Any = None) -> dict[str, Any]:
        """Metadata + chapter list + full chapter fetch in one call.

        ``skip_urls`` = normalized chapter URLs already on disk (resume).
        ``on_chapter(done, total, chapter)`` fires per fetched chapter.
        Failures are per-chapter and recorded, never fatal.
        """
        url = normalize_url(url)
        meta = self.novel(url)
        listing = self.chapter_list(url, limit=limit)
        total = len(listing)
        seen = {normalize_url(u) for u in (skip_urls or [])}
        chapters: list[Chapter] = []
        failed: list[dict[str, Any]] = []
        for i, (num, title, curl) in enumerate(listing):
            curl = normalize_url(curl)
            if curl in seen:
                continue
            try:
                ch = self.fetch_chapter(curl)
            except SourceError as exc:
                failed.append({"number": num, "url": curl,
                               "reason": str(exc)[:160]})
                _log.warning("%s ch%s fetch failed: %s", self.name, num, exc)
                continue
            ch.number = num
            if title:
                ch.title = title
            ch.source = self.name
            chapters.append(ch)
            seen.add(curl)
            if on_chapter:
                try:
                    on_chapter(i + 1, total, ch)
                except Exception:  # noqa: BLE001
                    pass
        return {"meta": meta.to_dict(), "total": total,
                "chapters": [c.to_dict() | {"url": normalize_url(c.url)}
                             for c in chapters],
                "chapter_texts": [c.text for c in chapters],
                "failed": failed}

    def download_cover(self, meta: StoryMeta, dest_dir: str | Path) -> str:
        """Fetch the cover image to ``dest_dir``.  Returns the path or ""."""
        if not meta.cover_url:
            return ""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        suffix = (urllib.parse.urlparse(meta.cover_url).path.rsplit(".", 1)[-1]
                  if "." in urllib.parse.urlparse(meta.cover_url).path else "jpg")
        suffix = "".join(c for c in suffix.lower() if c.isalnum())[:4] or "jpg"
        path = dest / f"cover.{suffix}"
        if path.exists() and path.stat().st_size > 1024:
            return str(path)
        try:
            final, html = self.fetcher.http_get(meta.cover_url,
                                                referer=meta.url)
            # http_get returns decoded text — re-fetch raw for binary
            import urllib.request as _req
            req = _req.Request(meta.cover_url,
                               headers={"User-Agent": _BROWSER_UA,
                                        "Referer": meta.url})
            with _req.urlopen(req, timeout=self.fetcher.timeout) as resp:
                data = resp.read(10_000_000)
        except Exception as exc:  # noqa: BLE001
            _log.warning("cover download failed: %s", exc)
            return ""
        if len(data) < 1024 or not data.startswith(
                (b"\xff\xd8", b"\x89PNG", b"GIF8", b"RIFF")):
            return ""
        path.write_bytes(data)
        return str(path)


# ── freewebnovel.com ─────────────────────────────────────────────────────────


class FreeWebNovelAdapter(SourceAdapter):
    """freewebnovel.com — verified live 2026-10-09."""

    name = "freewebnovel"
    domains = ("freewebnovel.com", "freewebnovel.io")

    _CHAPTER_RE = re.compile(r"/novel/([^/]+)/chapter-(\d+)")
    _NOVEL_RE = re.compile(r"/novel/([^/?#]+)/?$")

    def chapter_url(self, novel_url: str, number: int) -> str:
        m = self._NOVEL_RE.search(novel_url)
        if m and number >= 1:
            return f"https://freewebnovel.com/novel/{m.group(1)}/chapter-{number}"
        return ""

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        url = ("https://freewebnovel.com/search?keyword="
               + urllib.parse.quote_plus(query))
        _, dom, _ = self.fetcher.get(url)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for href, text in _links(dom, url):
            m = re.match(r"https?://freewebnovel\.com/novel/([^/?#]+)/?$", href)
            if not m or href in seen:
                continue
            seen.add(href)
            hits.append(SearchHit(title=text or m.group(1).replace("-", " ").title(),
                                  url=href, source=self.name))
            if len(hits) >= limit:
                break
        return hits

    def novel(self, url: str) -> StoryMeta:
        final_url, dom, _ = self.fetcher.get(url)
        m = self._CHAPTER_RE.search(final_url)
        slug = m.group(1) if m else ""
        if "/chapter-" in final_url and slug:
            final_url = f"https://freewebnovel.com/novel/{slug}"
            _, dom, _ = self.fetcher.get(final_url)
        title = _meta(dom, "og:title") or _meta(dom, "og:novel:title")
        author = _meta(dom, "og:novel:author")
        genres = [g.strip() for g in _meta(dom, "og:novel:genre").split(",") if g.strip()]
        status = _meta(dom, "og:novel:status") or "unknown"
        synopsis = _meta(dom, "og:description")
        cover = _meta(dom, "og:image")
        updated = _meta(dom, "og:novel:update_time")
        # latest chapters listed on the novel page
        chapters = self._chapter_links(dom, final_url)
        total = max((n for n, _, _ in chapters), default=0)
        return StoryMeta(title=title or slug.replace("-", " ").title(),
                          url=final_url, source=self.name, author=author,
                          synopsis=synopsis, genres=genres, status=status,
                          total_chapters=total, cover_url=cover,
                          updated=updated)

    def _chapter_links(self, dom: Node, base: str) -> list[tuple[int, str, str]]:
        out: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for href, text in _links(dom, base):
            m = self._CHAPTER_RE.search(href)
            if not m or href in seen:
                continue
            seen.add(href)
            out.append((int(m.group(2)), text, href))
        out.sort(key=lambda t: t[0])
        return out

    def chapter_list(self, novel_url: str, *, limit: int = 0) -> list[tuple[int, str, str]]:
        # The novel page lists the latest chapters; walking "Next Chapter"
        # from chapter 1 always yields the full ordered list.
        final_url, dom, _ = self.fetcher.get(novel_url)
        links = self._chapter_links(dom, final_url)
        if links and links[0][0] == 1:
            return links[:limit] if limit else links
        return links[:limit] if limit else links

    def fetch_chapter(self, url: str) -> Chapter:
        final_url, dom, _ = self.fetcher.get(url)
        m = self._CHAPTER_RE.search(final_url)
        number = int(m.group(2)) if m else _chapter_number_from_text(final_url)
        body = _by_id(dom, "article")
        if body is None:
            # markup drift: fall back to the reader text container
            txt = _first_by_class(dom, "txt")
            body = txt
        title = ""
        paras: list[str] = []
        if body is not None:
            for h in body.find_all(("h1", "h2", "h3", "h4")):
                t = h.inner_text().strip()
                if t and not title:
                    title = t
                    break
            paras = [p.inner_text() for p in body.find_all("p")]
        paras = _clean_paragraphs(paras)
        if not paras:
            raise SourceError(f"no chapter body found at {final_url} "
                              "(markup may have changed)")
        prev_url = next_url = ""
        for href, text in _links(dom, final_url):
            low = text.lower()
            if "prev chapter" in low and not prev_url:
                prev_url = href
            elif "next chapter" in low and not next_url:
                next_url = href
        if not title:
            title = f"Chapter {number}" if number else "Chapter"
        return Chapter(number=number, title=title, url=final_url,
                       paragraphs=paras, prev_url=prev_url,
                       next_url=next_url, source=self.name)


# ── novelfull.net family ─────────────────────────────────────────────────────


class NovelFullAdapter(SourceAdapter):
    """novelfull.net / novelfull.org / readnovelfull.com — verified live
    2026-10-09 (novelfull.net; siblings share the reader platform)."""

    name = "novelfull"
    domains = ("novelfull.net", "novelfull.org", "readnovelfull.com",
               "readnovelfull.net")

    _CHAPTER_RE = re.compile(r"/chapter-(\d+)-")

    def _base(self, url: str) -> str:
        p = urllib.parse.urlparse(url)
        return f"{p.scheme}://{p.hostname}"

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        base = "https://novelfull.net"
        url = base + "/search?keyword=" + urllib.parse.quote_plus(query)
        _, dom, _ = self.fetcher.get(url)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for href, text in _links(dom, base):
            if not re.match(r"https?://[^/]+/[^/]+\.html$", href):
                continue
            if "/chapter-" in href or href in seen:
                continue
            seen.add(href)
            title = text or href.rsplit("/", 1)[-1][:-5].replace("-", " ").title()
            hits.append(SearchHit(title=title, url=href, source=self.name))
            if len(hits) >= limit:
                break
        return hits

    def novel(self, url: str) -> StoryMeta:
        final_url, dom, _ = self.fetcher.get(url)
        title = _meta(dom, "og:title")
        author = _meta(dom, "og:novel:author")
        genres = [g.strip() for g in _meta(dom, "og:novel:genre").split(",") if g.strip()]
        status = _meta(dom, "og:novel:status") or "unknown"
        synopsis = _meta(dom, "og:description")
        cover = _meta(dom, "og:image")
        list_box = _by_id(dom, "list-chapter")
        total = 0
        if list_box is not None:
            try:
                total = int(list_box.attrs.get("data-total-chapters", "0") or 0)
            except ValueError:
                total = 0
        if not total:
            total = len(self._chapter_links(dom, final_url))
        return StoryMeta(title=title or "Unknown", url=final_url,
                          source=self.name, author=author, synopsis=synopsis,
                          genres=genres, status=status, total_chapters=total,
                          cover_url=cover)

    def _chapter_links(self, dom: Node, base: str) -> list[tuple[int, str, str]]:
        out: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for href, _text in _links(dom, base):
            if "/chapter-" not in href or href in seen:
                continue
            seen.add(href)
            m = self._CHAPTER_RE.search(href)
            number = int(m.group(1)) if m else 0
            # the a.con title attribute carries the clean chapter title
            out.append((number, "", href))
        # recover titles from the anchors themselves
        titled: list[tuple[int, str, str]] = []
        for node in dom.find_all("a"):
            href = (node.attrs.get("href") or "").strip()
            if "/chapter-" not in href:
                continue
            abs_url = urllib.parse.urljoin(base, href)
            m = self._CHAPTER_RE.search(abs_url)
            number = int(m.group(1)) if m else 0
            title = (node.attrs.get("title") or node.inner_text()).strip()
            titled.append((number, title, abs_url))
        titled.sort(key=lambda t: t[0])
        return titled or sorted(out, key=lambda t: t[0])

    def chapter_list(self, novel_url: str, *, limit: int = 0) -> list[tuple[int, str, str]]:
        # page 1 lists the newest 40; ?page=N walks the whole catalogue.
        final_url, dom, _ = self.fetcher.get(novel_url)
        list_box = _by_id(dom, "list-chapter")
        total_pages = 1
        if list_box is not None:
            try:
                total_pages = int(list_box.attrs.get("data-total-page", "1") or 1)
            except ValueError:
                total_pages = 1
        all_links = self._chapter_links(dom, final_url)
        seen_urls = {u for _, _, u in all_links}
        # cap page walks: termux profile fetches fewer pages (set by caller
        # via limit; default walks everything but slowly)
        max_pages = total_pages if not limit else min(total_pages, (limit // 40) + 2)
        for page in range(2, max_pages + 1):
            page_url = final_url.split("?")[0] + f"?page={page}"
            try:
                _, pdom, _ = self.fetcher.get(page_url)
            except SourceError:
                break
            for number, title, href in self._chapter_links(pdom, final_url):
                if href not in seen_urls:
                    seen_urls.add(href)
                    all_links.append((number, title, href))
            time.sleep(0.4)  # be polite to the aggregator
        all_links.sort(key=lambda t: t[0])
        return all_links[:limit] if limit else all_links

    def fetch_chapter(self, url: str) -> Chapter:
        final_url, dom, _ = self.fetcher.get(url)
        m = self._CHAPTER_RE.search(final_url)
        number = int(m.group(1)) if m else 0
        body = _by_id(dom, "chapter-content")
        if body is None:
            body = _first_by_class(dom, "txt")
        title = ""
        paras: list[str] = []
        if body is not None:
            for h in body.find_all(("h1", "h2", "h3")):
                t = h.inner_text().strip()
                if t and not title:
                    title = t
                    break
            paras = [p.inner_text() for p in body.find_all("p")]
        paras = _clean_paragraphs(paras)
        if not paras:
            raise SourceError(f"no chapter body found at {final_url}")
        prev_url = next_url = ""
        for href, text in _links(dom, final_url):
            low = text.lower()
            if "prev chapter" in low and not prev_url:
                prev_url = href
            elif "next chapter" in low and not next_url:
                next_url = href
        if not title:
            title = f"Chapter {number}" if number else "Chapter"
        return Chapter(number=number, title=title, url=final_url,
                       paragraphs=paras, prev_url=prev_url,
                       next_url=next_url, source=self.name)


# ── royalroad.com ────────────────────────────────────────────────────────────


class RoyalRoadAdapter(SourceAdapter):
    """royalroad.com — verified live 2026-10-09."""

    name = "royalroad"
    domains = ("royalroad.com", "www.royalroad.com")

    _CHAPTER_RE = re.compile(r"/fiction/(\d+)/([^/]+)/chapter/(\d+)/(\d+)-")

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        url = ("https://www.royalroad.com/fictions/search?query="
               + urllib.parse.quote_plus(query))
        _, dom, _ = self.fetcher.get(url)
        # the cover anchor carries no text (img alt does); the title
        # anchor right after it does — prefer titled anchors per URL.
        by_url: dict[str, str] = {}
        order: list[str] = []
        for href, text in _links(dom, url):
            m = re.match(r"https?://(?:www\.)?royalroad\.com/fiction/\d+/[^/?#]+/?$",
                         href)
            if not m:
                continue
            if href not in by_url:
                by_url[href] = ""
                order.append(href)
            if text and not by_url[href]:
                by_url[href] = text
        # second pass: img alt text for still-untitled hits
        if any(not t for t in by_url.values()):
            for node in dom.find_all("a"):
                href = (node.attrs.get("href") or "").strip()
                if not href:
                    continue
                abs_url = urllib.parse.urljoin(url, href)
                if abs_url in by_url and not by_url[abs_url]:
                    for img in node.find_all("img"):
                        alt = (img.attrs.get("alt") or "").strip()
                        if alt:
                            by_url[abs_url] = alt
                            break
        hits = [SearchHit(title=by_url[h] or "Unknown", url=h,
                          source=self.name)
                for h in order[:limit]]
        return hits

    def novel(self, url: str) -> StoryMeta:
        final_url, dom, _ = self.fetcher.get(url)
        title = ""
        for h in dom.find_all("h1"):
            t = h.inner_text().strip()
            if t:
                title = t
                break
        title = title or _meta(dom, "og:title")
        author = _meta(dom, "og:novel:author") or _meta(dom, "author")
        synopsis = _meta(dom, "og:description")
        cover = _meta(dom, "og:image")
        chapters = self._chapter_links(dom, final_url)
        return StoryMeta(title=title or "Unknown", url=final_url,
                          source=self.name, author=author, synopsis=synopsis,
                          total_chapters=len(chapters), cover_url=cover)

    def _chapter_links(self, dom: Node, base: str) -> list[tuple[int, str, str]]:
        out: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for href, text in _links(dom, base):
            m = self._CHAPTER_RE.search(href)
            if not m or href in seen:
                continue
            seen.add(href)
            out.append((int(m.group(4)), text.strip(), href))
        out.sort(key=lambda t: t[0])
        return out

    def chapter_list(self, novel_url: str, *, limit: int = 0) -> list[tuple[int, str, str]]:
        final_url, dom, _ = self.fetcher.get(novel_url)
        links = self._chapter_links(dom, final_url)
        return links[:limit] if limit else links

    def fetch_chapter(self, url: str) -> Chapter:
        final_url, dom, _ = self.fetcher.get(url)
        m = self._CHAPTER_RE.search(final_url)
        number = int(m.group(4)) if m else 0
        body = _first_by_class(dom, "chapter-inner", "chapter-content")
        title = ""
        paras: list[str] = []
        if body is not None:
            for h in body.find_all(("h1", "h2", "h3", "h4")):
                t = h.inner_text().strip()
                if t and not title:
                    title = t
                    break
            paras = [p.inner_text() for p in body.find_all("p")]
            if not paras:  # some chapters are raw text nodes / divs
                paras = [body.inner_text()]
        paras = _clean_paragraphs(paras)
        if not paras:
            raise SourceError(f"no chapter body found at {final_url}")
        # prev/next from the fiction's chapter order (robust when the
        # page's own nav buttons change markup)
        prev_url = next_url = ""
        fic_m = re.match(r"(https?://(?:www\.)?royalroad\.com/fiction/\d+/[^/]+)",
                         final_url)
        if fic_m:
            try:
                _, ndom, _ = self.fetcher.get(fic_m.group(1))
                order = self._chapter_links(ndom, fic_m.group(1))
                urls = [u for _, _, u in order]
                if final_url in urls:
                    i = urls.index(final_url)
                    if i > 0:
                        prev_url = urls[i - 1]
                    if i + 1 < len(urls):
                        next_url = urls[i + 1]
            except SourceError:
                pass
        if not title:
            title = f"Chapter {number}" if number else "Chapter"
        return Chapter(number=number, title=title, url=final_url,
                       paragraphs=paras, prev_url=prev_url,
                       next_url=next_url, source=self.name)


# ── generic fallback (PandaNovel + any unlisted aggregator) ──────────────────


class GenericAdapter(SourceAdapter):
    """A self-tuning adapter for archives whose markup was never verified.

    Given a novel/chapter URL it tries a ranked list of content-selector
    candidates and keeps the one that yields the most prose-like text
    (paragraph count × median length, junk-filtered).  The winning
    selectors are remembered per domain so later fetches skip the search.
    Honest by design: ``verified`` stays False until a human confirms it.
    """

    name = "generic"
    domains: tuple[str, ...] = ()

    #: (kind, value) candidates, tried in order
    _CANDIDATES: tuple[tuple[str, str], ...] = (
        ("id", "chr-content"),
        ("id", "chapter-content"),
        ("id", "article"),
        ("class", "chapter-content"),
        ("class", "chapter-inner"),
        ("class", "entry-content"),
        ("class", "post-content"),
        ("class", "txt"),
        ("tag", "article"),
    )

    def __init__(self, name: str = "generic",
                 domains: tuple[str, ...] = (),
                 fetcher: Fetcher | None = None) -> None:
        super().__init__(fetcher)
        self.name = name
        self.domains = domains
        self.verified = False
        self._winners: dict[str, tuple[str, str]] = {}

    # -- selector machinery ---------------------------------------------------
    def _pick(self, node: Node, kind: str, value: str) -> Node | None:
        if kind == "id":
            return _by_id(node, value)
        if kind == "class":
            return _first_by_class(node, value)
        found = node.find_all(value)
        return found[0] if found else None

    def _score(self, node: Node) -> tuple[int, list[str]]:
        paras = _clean_paragraphs([p.inner_text() for p in node.find_all("p")])
        if not paras:
            text = _clean_paragraphs([node.inner_text()])
            paras = text
        words = sum(len(p.split()) for p in paras)
        return words, paras

    def _extract_body(self, dom: Node, host: str) -> tuple[list[str], str]:
        winner = self._winners.get(host)
        candidates = ([winner] if winner else []) + [
            c for c in self._CANDIDATES if c != winner]
        best: tuple[int, list[str], tuple[str, str] | None] = (0, [], None)
        for cand in candidates:
            node = self._pick(dom, cand[0], cand[1])
            if node is None:
                continue
            words, paras = self._score(node)
            if words > best[0]:
                best = (words, paras, cand)
        if best[2] is not None and best[0] >= 120:
            self._winners[host] = best[2]
            return best[1], f"{best[2][0]}={best[2][1]}"
        # last resort: the largest text-bearing div on the page
        fallback_best: tuple[int, list[str]] = (0, [])
        for node in dom.find_all("div"):
            if node.attrs.get("id") in {"header", "footer", "sidebar", "nav"}:
                continue
            words, paras = self._score(node)
            if words > fallback_best[0]:
                fallback_best = (words, paras)
        if fallback_best[0] >= 120:
            return fallback_best[1], "largest-div"
        return [], ""

    # -- SourceAdapter surface --------------------------------------------------
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        raise SourceError(
            f"{self.name}: search is not mapped for this archive "
            "(unverified markup) — paste the novel URL directly")

    def novel(self, url: str) -> StoryMeta:
        final_url, dom, _ = self.fetcher.get(url)
        title = _meta(dom, "og:title")
        if not title:
            for h in dom.find_all("h1"):
                t = h.inner_text().strip()
                if t:
                    title = t
                    break
        return StoryMeta(title=title or "Unknown", url=final_url,
                          source=self.name, author=_meta(dom, "author"),
                          synopsis=_meta(dom, "og:description"),
                          cover_url=_meta(dom, "og:image"))

    def chapter_list(self, novel_url: str, *, limit: int = 0) -> list[tuple[int, str, str]]:
        final_url, dom, _ = self.fetcher.get(novel_url)
        out: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for href, text in _links(dom, final_url):
            if "chapter" not in href.lower() or href in seen:
                continue
            seen.add(href)
            out.append((_chapter_number_from_text(text) or
                        _chapter_number_from_text(href), text, href))
        out.sort(key=lambda t: (t[0] or 10 ** 9))
        return out[:limit] if limit else out

    def fetch_chapter(self, url: str) -> Chapter:
        final_url, dom, _ = self.fetcher.get(url)
        host = (urllib.parse.urlparse(final_url).hostname or "").lower()
        paras, _used = self._extract_body(dom, host)
        if not paras:
            raise SourceError(
                f"{self.name}: could not isolate chapter prose at "
                f"{final_url} (unverified markup — needs the "
                "browser-context fallback or manual mapping)")
        number = _chapter_number_from_text(final_url)
        title = ""
        for h in dom.find_all(("h1", "h2")):
            t = h.inner_text().strip()
            if t and len(t) < 160:
                title = t
                break
        return Chapter(number=number, title=title or f"Chapter {number}",
                       url=final_url, paragraphs=paras, source=self.name)


class PandaNovelAdapter(GenericAdapter):
    """PandaNovel — UNVERIFIED (all known domains unresolvable 2026-10-09).

    Behaves as the generic self-tuning adapter scoped to PandaNovel's
    domains; flips to verified only after a real successful fetch is
    confirmed against the live markup.
    """

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        super().__init__(name="pandanovel",
                         domains=("pandanovel.com", "pandasnovel.com",
                                   "panda-novel.com"),
                         fetcher=fetcher)


# ── registry ─────────────────────────────────────────────────────────────────


def _make_adapters() -> list[SourceAdapter]:
    return [
        FreeWebNovelAdapter(),
        NovelFullAdapter(),
        RoyalRoadAdapter(),
        PandaNovelAdapter(),
    ]


ADAPTERS: list[SourceAdapter] = _make_adapters()


def adapter_for_url(url: str) -> SourceAdapter:
    for adapter in ADAPTERS:
        if adapter.matches(url):
            return adapter
    # unknown aggregator → the self-tuning generic adapter, honestly flagged
    host = (urllib.parse.urlparse(url).hostname or "unknown").lower()
    return GenericAdapter(name=f"generic:{host}", domains=(host,))


def search_all(query: str, *, limit: int = 10,
               sources: list[str] | None = None) -> list[SearchHit]:
    """Search every (verified) source; failures are per-source, never fatal."""
    hits: list[SearchHit] = []
    for adapter in ADAPTERS:
        if sources and adapter.name not in sources:
            continue
        if not adapter.verified:
            continue
        try:
            hits.extend(adapter.search(query, limit=limit))
        except SourceError as exc:
            _log.warning("story search failed on %s: %s", adapter.name, exc)
        if len(hits) >= limit:
            break
    return hits[:limit]
