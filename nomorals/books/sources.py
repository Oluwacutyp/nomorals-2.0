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

Open sources (Pass #89, the owner's open-sources law): this layer used
to be a CLOSED-SOURCE class — no fetch outcome was recorded, no site
held health state, no new story site could arrive without a code
change, and a restart forgot everything.  Book/story sources now join
the ONE media source registry (``media/source_registry.py``) as their
own class — the registry's model fits without distortion (every book
source is one host serving novel/chapter fetches; the same outcome /
health / candidate vocabulary; the same KV persistence; the same
health loop and scheduler job, so no parallel machinery anywhere):

* every ``search`` / ``novel`` / ``chapter_list`` / ``fetch_chapter``
  call on a registry-bound adapter records its outcome against the
  serving site (see :meth:`SourceAdapter.bind_book_registry`):
  success credits it, a failure records its error class, a bot-wall
  (``SourceBlocked``) records as ``bot_block`` / health
  ``bot-blocked`` — never dead — and a catalogue miss (empty search
  or chapter list, a landed page with no chapter body) records as
  weak evidence via ``record_no_result``: it sinks the score, never
  trips the breaker;
* :func:`search_all` orders its pool outcome-aware when a registry
  is supplied (with zero learned state the order is exactly the
  pre-#89 order);
* discovery channels that genuinely exist on this path — redirect
  targets, story-shaped links embedded in fetched pages (the
  :class:`Fetcher` seam), and foreign hosts carried by search /
  fetch results — are noted UNTRUSTED as registry candidates
  (``kinds=("book",)``): they serve nothing and earn nothing until
  the health loop promotes them, after which
  :func:`grown_book_adapters` serves them through the universal
  :class:`GenericAdapter` shape.  Channels that do NOT exist here:
  no sitemap/feed/discovery listing is fetched by any book consumer,
  and cover-image hosts are CDNs, not story sites — never intaken.
"""

from __future__ import annotations

import functools
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
from ..media.source_registry import BOOK_SOURCE_DOMAINS
from ..tools.browser import Node, parse_html

_log = get_logger(__name__)

__all__ = [
    "Chapter", "StoryMeta", "SearchHit",
    "SourceAdapter", "SourceError", "SourceBlocked",
    "FreeWebNovelAdapter", "NovelFullAdapter", "RoyalRoadAdapter",
    "PandaNovelAdapter", "GenericAdapter",
    "ADAPTERS", "adapter_for_url", "search_all", "normalize_url",
    "BOOK_SOURCE_DOMAINS", "BOOK_SOURCE_BY_DOMAIN", "BOOK_SEARCH_SOURCES",
    "book_registry_for_context", "registry_desc_for_adapter",
    "grown_book_adapters", "ordered_book_adapters",
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


# ── book/story source class in the ONE registry (Pass #89) ─────────────
#
# Membership maps for the class, mirroring media/sources.py's
# MUSIC_SOURCE_DOMAINS (#88): the primary host each source serves
# from, and every known domain (mirrors included, from the adapters'
# own ``domains`` tuples + the descriptor table) mapped back to its
# source id.  The media registry seeds exactly these domains.
#
# Pass #95: :data:`BOOK_SOURCE_DOMAINS` itself is defined in
# ``media/source_registry.py`` — the table is shared with the
# registry's own status views, so it lives at the registry's layer
# and is imported back here (downward); the name, contents and order
# are unchanged for every consumer of this module.

#: every known book-source domain (mirrors included) → source id
BOOK_SOURCE_BY_DOMAIN: dict[str, str] = {
    "freewebnovel.com": "freewebnovel",
    "freewebnovel.io": "freewebnovel",
    "novelfull.net": "novelfull",
    "novelfull.org": "novelfull",
    "readnovelfull.com": "novelfull",
    "readnovelfull.net": "novelfull",
    "royalroad.com": "royalroad",
    "pandanovel.com": "pandanovel",
    "pandasnovel.com": "pandanovel",
    "panda-novel.com": "pandanovel",
    "gutenberg.org": "gutenberg",
    "scribblehub.com": "scribblehub",
    "archiveofourown.org": "ao3",
    "wattpad.com": "wattpad",
    "webnovel.com": "webnovel",
}

#: The dedicated-adapter search pool, in search_all's day-one order.
BOOK_SEARCH_SOURCES: tuple[str, ...] = (
    "freewebnovel", "novelfull", "royalroad", "pandanovel",
)


def _book_host(url_or_domain: str) -> str:
    """Bare host for a URL/domain (``www.``-insensitive), or ""."""
    try:
        text = str(url_or_domain or "").strip()
        if "://" in text:
            host = urllib.parse.urlparse(text).hostname or ""
        elif "/" in text:
            host = text.split("/", 1)[0]
        else:
            host = text
        host = host.lower().split(":")[0]
        return host[4:] if host.startswith("www.") else host
    except Exception:  # noqa: BLE001
        return ""


def book_registry_for_context(context: Any = None,
                              injected: Any = None) -> Any:
    """The source registry book outcomes land in, or None.

    Mirrors the #88 resolver's resolution exactly: an injected
    registry wins; otherwise the shared ``DEFAULT_REGISTRY`` bound
    to the context's KV store — and only when the context carries a
    database, so db-less callers (pure/offline contexts, the fake-
    adapter test harnesses) keep the pre-#89 behaviour exactly.
    Never raises."""
    if injected is not None:
        return injected
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return None
        from ..media.source_registry import DEFAULT_REGISTRY
        reg = DEFAULT_REGISTRY
        try:
            from ..storage.kv import KVStore
            reg.bind_store(KVStore(db))
        except Exception:  # noqa: BLE001 - persistence is an
            # enhancement; its absence never sinks a fetch
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _ensure_book_source(reg: Any, adapter: Any) -> Any:
    """Register a descriptor-backed adapter's primary domain as a
    book source (origin ``registered``) — the path by which an
    owner-supplied descriptor (``descriptors.register_descriptor``)
    becomes a first-class registry source.  Dedicated/generic
    adapters never reach here.  Never raises."""
    try:
        from ..media.source_registry import (
            Licence, SourceDescriptor, TrustTier)

        domains = getattr(adapter, "domains", ()) or ()
        if not domains:
            return None
        desc = SourceDescriptor(
            nation="GLOBAL", language="en", domain=domains[0],
            name=str(getattr(adapter, "name", "") or domains[0]),
            kinds=("book", "book-search"), licence=Licence.UNKNOWN,
            trust_tier=TrustTier.UNVERIFIED,
            notes="descriptor-backed story source (books/descriptors.py)")
        if reg.register(desc):
            reg._persist()  # a registered source survives restarts too
            return desc
        return reg.for_domain(domains[0])
    except Exception:  # noqa: BLE001
        return None


def registry_desc_for_adapter(reg: Any, adapter: Any,
                              *urls: str) -> Any:
    """The registry descriptor that should be credited for *adapter*
    (optionally resolving via the URLs it served): a book-class
    source, or None when the registry does not know the serving
    source (candidates and ghosts earn nothing — they must be
    promoted by the health loop first, the #87 precedent).  Never
    raises."""
    try:
        if reg is None or adapter is None:
            return None
        for url in urls:
            host = _book_host(url or "")
            if not host:
                continue
            desc = reg.for_domain(host)
            if desc is not None and (
                    "book" in desc.kinds
                    or host in BOOK_SOURCE_BY_DOMAIN):
                return desc
        name = str(getattr(adapter, "name", "") or "")
        if name.startswith("grown:"):
            desc = reg.for_domain(name[len("grown:"):])
            if desc is not None:
                return desc
        primary = BOOK_SOURCE_DOMAINS.get(name)
        if primary:
            desc = reg.for_domain(primary)
            if desc is not None:
                return desc
        for domain in getattr(adapter, "domains", ()) or ():
            desc = reg.for_domain(domain)
            if desc is not None and "book" in desc.kinds:
                return desc
        if getattr(adapter, "descriptor", None) is not None:
            return _ensure_book_source(reg, adapter)
        return None
    except Exception:  # noqa: BLE001
        return None


def _book_error_evidence(exc: Exception) -> tuple[str, str]:
    """Classify one adapter exception into registry evidence:
    ``(kind, error_class)`` where kind is ``blocked`` | ``failure`` |
    ``no_result`` | ``none``.

    * ``SourceBlocked`` → blocked (bot-wall/CF: distinct from dead);
    * configuration facts (an API key is not set, search is not
      mapped for an unverified archive, a malformed story URL) →
      ``none``: facts about this install/query, not the source;
    * a landed page with no chapter body, a 404/410, a "not found"
      → ``no_result``: the site answered, the story/chapter is what
      is missing — weak evidence (#88 precedent);
    * unreachable/timeout → failure, ``network_failure``; anything
      else → failure, ``site_error`` (the downloader's vocabulary).
    """
    if isinstance(exc, SourceBlocked):
        return "blocked", "bot_block"
    msg = str(exc or "").lower()
    if ("api needs" in msg or "not mapped" in msg
            or "cannot parse" in msg or "no plain-text format" in msg):
        return "none", ""
    if ("no chapter body found" in msg
            or "could not isolate chapter prose" in msg
            or "no chapters detected" in msg
            or "-> http 404" in msg or "-> http 410" in msg
            or "not found" in msg):
        return "no_result", ""
    if ("unreachable" in msg or "timed out" in msg or "timeout" in msg
            or "urlopen error" in msg):
        return "failure", "network_failure"
    return "failure", "site_error"


def _book_result_urls(op: str, result: Any) -> list[str]:
    """URLs a successful adapter call carries — the discovery
    surface of the result itself (search hits, a chapter's
    prev/next links, a chapter list)."""
    try:
        if op == "search":
            return [h.url for h in result or [] if getattr(h, "url", "")]
        if op == "novel":
            return [result.url] if result and result.url else []
        if op == "chapter_list":
            return [u for _n, _t, u in result or [] if u]
        if op == "fetch_chapter" and result is not None:
            return [u for u in (result.url, result.prev_url,
                                result.next_url) if u]
    except Exception:  # noqa: BLE001
        return []
    return []


def _note_book_result_urls(reg: Any, op: str, result: Any) -> None:
    """Note foreign hosts carried by a fetch/search result as
    UNTRUSTED book candidates.  Known sources refuse themselves in
    the registry; a candidate serves and earns nothing until the
    health loop promotes it.  Best-effort; never raises."""
    try:
        via = "book-search-result" if op == "search" \
            else "book-fetch-result"
        for url in _book_result_urls(op, result):
            reg.note_candidate(url, via=via, kinds=("book",))
    except Exception:  # noqa: BLE001
        pass


def _record_book_success(reg: Any, adapter: Any, op: str,
                         result: Any, latency_s: float,
                         arg_url: str) -> None:
    """Credit one successful adapter call (or record its catalogue
    miss as weak evidence).  Best-effort; never raises."""
    try:
        urls = _book_result_urls(op, result)
        desc = registry_desc_for_adapter(
            reg, adapter, *(urls or ([arg_url] if arg_url else [])))
        _note_book_result_urls(reg, op, result)
        if desc is None:
            return
        if op in ("search", "chapter_list") and not result:
            reg.record_no_result(desc)
        else:
            reg.record_outcome(desc, True, latency_s=latency_s)
    except Exception:  # noqa: BLE001 - bookkeeping never breaks a fetch
        pass


def _record_book_failure(reg: Any, adapter: Any, op: str,
                         exc: Exception, latency_s: float,
                         arg_url: str) -> None:
    """Blame one failed adapter call, classified by
    :func:`_book_error_evidence`.  Best-effort; never raises."""
    try:
        kind, error_class = _book_error_evidence(exc)
        if kind == "none":
            return
        desc = registry_desc_for_adapter(
            reg, adapter, arg_url if op != "search" else "")
        if desc is None:
            return
        if kind == "blocked":
            reg.record_blocked(desc, latency_s=latency_s)
        elif kind == "no_result":
            reg.record_no_result(desc)
        else:
            reg.record_outcome(desc, False, latency_s=latency_s,
                               error_class=error_class)
    except Exception:  # noqa: BLE001 - bookkeeping never breaks a fetch
        pass


def _wrap_book_op(op: str, fn: Any) -> Any:
    """Wrap one concrete adapter operation so its outcome lands in
    the registry the adapter is bound to (Pass #89).  Transparent:
    with no registry bound, or nested inside another recorded call
    on the same adapter (DescriptorAdapter delegating to
    GenericAdapter), the call is the original, byte-for-byte."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        reg = getattr(self, "_book_registry", None)
        if reg is None or getattr(self, "_book_rec_depth", 0):
            return fn(self, *args, **kwargs)
        self._book_rec_depth = getattr(self, "_book_rec_depth", 0) + 1
        arg_url = str(args[0]) if args and isinstance(args[0], str) else ""
        t0 = time.time()
        try:
            result = fn(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            _record_book_failure(reg, self, op, exc,
                                 time.time() - t0, arg_url)
            raise
        else:
            _record_book_success(reg, self, op, result,
                                 time.time() - t0, arg_url)
            return result
        finally:
            self._book_rec_depth -= 1

    wrapper._book_recorded = True  # type: ignore[attr-defined]
    return wrapper


def grown_book_adapters(reg: Any) -> list[Any]:
    """Adapters for GROWN book sources: candidates the health loop
    promoted (origin ``grown``, kind ``book``), served through the
    universal :class:`GenericAdapter` shape — the descriptor
    reader's generic shape, the #88 ``DiscoveredBlogSource``
    precedent.  Pending candidates are NOT here: they serve
    nothing.  Never raises."""
    try:
        out = []
        for desc in reg.all():
            if reg.origin_for(desc) != "grown" \
                    or "book" not in desc.kinds:
                continue
            adapter = GenericAdapter(name=f"grown:{desc.domain}",
                                     domains=(desc.domain,))
            adapter.bind_book_registry(reg)
            out.append(adapter)
        out.sort(key=lambda a: a.domains[0] if a.domains else a.name)
        return out
    except Exception:  # noqa: BLE001
        return []


def ordered_book_adapters(adapters: list[Any], reg: Any) -> list[Any]:
    """Outcome-aware adapter order (Pass #89), through the registry's
    existing score/health semantics, mirroring the #88 resolver:

    * NO learned state anywhere in the pool → the list is returned
      untouched: day-one behaviour is the pre-#89 order, byte-for-byte;
    * otherwise active sources sort by registry score (stable — ties
      keep pool order) and stood-down / disabled / dead sources are
      demoted to the end, in pool order: they leave the active
      ordering but are still tried last rather than pretending they
      do not exist.
    Never raises; a registry hiccup returns the pool as-is."""
    try:
        items: list[tuple[Any, Any]] = []
        evidence = False
        for adapter in adapters:
            desc = registry_desc_for_adapter(reg, adapter)
            items.append((adapter, desc))
            if desc is None:
                continue
            st = reg.stats_for(desc)
            if st["ok"] or st["fail"] or st["no_result"] \
                    or st.get("blocked") or not reg.is_active(desc):
                evidence = True
        if not evidence:
            return list(adapters)
        active = [it for it in items
                  if it[1] is None or reg.is_active(it[1])]
        inactive = [it for it in items
                    if it[1] is not None and not reg.is_active(it[1])]
        active.sort(key=lambda it: -reg.score(it[1])
                    if it[1] is not None else 0.0)
        return [a for a, _d in active] + [a for a, _d in inactive]
    except Exception:  # noqa: BLE001
        return list(adapters)


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
    licence: str = ""  # rights label, recorded on every fetch

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "url": self.url, "source": self.source,
            "author": self.author, "synopsis": self.synopsis[:600],
            "genres": self.genres, "status": self.status,
            "total_chapters": self.total_chapters,
            "cover_url": self.cover_url, "updated": self.updated,
            "licence": self.licence,
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
        #: the source registry this fetcher's adapter is bound to
        #: (Pass #89) — discovery intake only; None = pre-#89 behaviour.
        self._book_registry: Any = None

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
        dom = parse_html(html)
        self._note_book_discoveries(url, final_url, dom)
        return final_url, dom, html

    def _note_book_discoveries(self, requested_url: str,
                               final_url: str, dom: Node) -> None:
        """Discovery intake at the fetch seam (Pass #89): the raw
        page is only visible here, so this is where the book path's
        genuine discovery channels are noted as UNTRUSTED registry
        candidates (``kinds=("book",)``; they serve nothing until
        the health loop promotes them):

        * the redirect target — a fetch that lands on a foreign host
          (the site's own mirror answering);
        * story-shaped links embedded in the fetched page — links
          whose path carries a novel/chapter/fiction/series/read
          shape on a foreign host (a site linking its mirror, or the
          same novel elsewhere).  Share/social/plain links carry no
          story shape and are never intaken, and cover/CDN hosts are
          not links at all here — not story sites, never intaken.

        Best-effort; never raises, never changes the fetch result."""
        reg = getattr(self, "_book_registry", None)
        if reg is None:
            return
        try:
            req_host = _book_host(requested_url)
            fin_host = _book_host(final_url)
            if fin_host and req_host and fin_host != req_host:
                reg.note_candidate(final_url, via="book-redirect",
                                   kinds=("book",))
            seen: set[str] = set()
            for href, _text in _links(dom, final_url):
                host = _book_host(href)
                if not host or host == fin_host or host in seen:
                    continue
                seen.add(host)
                low = href.lower()
                if any(tok in low for tok in
                       ("/novel", "chapter", "/fiction", "/series",
                        "/read/")):
                    reg.note_candidate(href, via="book-page-link",
                                       kinds=("book",))
                if len(seen) >= 20:
                    break
        except Exception:  # noqa: BLE001 - intake never breaks a fetch
            pass


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
    #: The source registry this adapter's fetch outcomes land in
    #: (Pass #89); None = unbound, pre-#89 behaviour exactly.
    _book_registry: Any = None
    _book_rec_depth: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Pass #89: wrap each concrete fetch operation ONCE, at class
        # creation, so every real fetch path (reader, download_story,
        # direct adapter use) records its outcome through the one
        # central seam — no per-adapter, per-callsite duplication.
        super().__init_subclass__(**kwargs)
        for _op in ("search", "novel", "chapter_list", "fetch_chapter"):
            _fn = cls.__dict__.get(_op)
            if callable(_fn) and not getattr(_fn, "_book_recorded", False):
                setattr(cls, _op, _wrap_book_op(_op, _fn))

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self.fetcher = fetcher or Fetcher()

    def bind_book_registry(self, reg: Any) -> "SourceAdapter":
        """Bind the ONE media source registry: this adapter's fetch
        outcomes, blocked walls and catalogue misses land in it, and
        its fetcher notes book-path discoveries into it (Pass #89).
        Returns self for chaining.  Never raises."""
        try:
            self._book_registry = reg
            fetcher = getattr(self, "fetcher", None)
            if isinstance(fetcher, Fetcher):
                fetcher._book_registry = reg
        except Exception:  # noqa: BLE001 - binding never breaks a fetch
            pass
        return self

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


def adapter_for_url(url: str,
                    source_registry: Any = None) -> SourceAdapter:
    """The adapter serving *url*: a dedicated adapter, a descriptor
    adapter, a health-promoted GROWN book source (when a registry is
    supplied — served through the universal generic shape, named
    ``grown:<host>``), else the self-tuning generic adapter for an
    unknown host, honestly flagged.  When *source_registry* is given
    the returned dedicated/descriptor/grown adapter is bound to it
    so its fetch outcomes land in the one registry (Pass #89); a
    pending candidate is never returned here — candidates serve
    nothing, and the anonymous generic fallback for an explicitly
    pasted URL is the pre-#89 owner-directed path, unbound, earning
    no registry credit."""
    for adapter in ADAPTERS:
        if adapter.matches(url):
            if source_registry is not None:
                adapter.bind_book_registry(source_registry)
            return adapter
    # descriptor-driven legal sources (lazy import: descriptors imports
    # this module, so a top-level import would be circular)
    try:
        from .descriptors import descriptor_adapters
        for adapter in descriptor_adapters():
            if adapter.matches(url):
                if source_registry is not None:
                    adapter.bind_book_registry(source_registry)
                return adapter
    except Exception:  # noqa: BLE001 - descriptor failure never breaks lookup
        pass
    if source_registry is not None:
        host = _book_host(url)
        desc = source_registry.for_domain(host) if host else None
        if desc is not None \
                and source_registry.origin_for(desc) == "grown" \
                and "book" in desc.kinds:
            grown = GenericAdapter(name=f"grown:{host}", domains=(host,))
            grown.bind_book_registry(source_registry)
            return grown
    # unknown aggregator → the self-tuning generic adapter, honestly flagged
    host = (urllib.parse.urlparse(url).hostname or "unknown").lower()
    return GenericAdapter(name=f"generic:{host}", domains=(host,))


def search_all(query: str, *, limit: int = 10,
               sources: list[str] | None = None,
               source_registry: Any = None) -> list[SearchHit]:
    """Search every (verified) source; failures are per-source, never fatal.

    With *source_registry* (Pass #89) the pool is bound for the
    duration of the search (bindings restored afterwards — the
    global adapters keep no stale registry) and ordered
    outcome-aware via :func:`ordered_book_adapters`: with zero
    learned state the order, and therefore the result, is exactly
    the pre-#89 behaviour."""
    pool = list(ADAPTERS)
    previous: list[tuple[SourceAdapter, Any]] = []
    if source_registry is not None:
        pool = ordered_book_adapters(pool, source_registry)
        for adapter in pool:
            previous.append((adapter, adapter._book_registry))
            adapter.bind_book_registry(source_registry)
    hits: list[SearchHit] = []
    try:
        for adapter in pool:
            if sources and adapter.name not in sources:
                continue
            if not adapter.verified:
                continue
            try:
                hits.extend(adapter.search(query, limit=limit))
            except SourceError as exc:
                _log.warning("story search failed on %s: %s",
                             adapter.name, exc)
            if len(hits) >= limit:
                break
    finally:
        for adapter, old in previous:
            adapter._book_registry = old
            fetcher = getattr(adapter, "fetcher", None)
            if isinstance(fetcher, Fetcher):
                fetcher._book_registry = old
    return hits[:limit]
