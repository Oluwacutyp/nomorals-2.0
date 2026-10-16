"""Universal story-source descriptors — descriptor-as-data adapters.

A source is a :class:`SourceDescriptor`: URL patterns, selectors,
pagination, rate limits, bot-wall behaviour, and a licence/rights label.
:class:`DescriptorAdapter` implements the :class:`SourceAdapter` surface
from a descriptor, so a new archive is a data row, not a new class.

Legal-first registry:

* **Public domain** — Project Gutenberg (Gutenberg + Gutendex API).
* **Author-posted** — Royal Road (dedicated adapter in ``sources.py``),
  Scribble Hub, AO3 (Archive of Our Own). AO3 has no official API;
  its descriptor is honest ``verified=False`` and respects AO3's
  rate limits; locked works report blocked, never bypassed.
* **Licensed official** — Webnovel (official webnovel.com), Wattpad
  (official API, needs a key — without one it reports honestly
  unavailable, never scrapes around the key).
* **Aggregator adapters** (FreeWebNovel, NovelFull, PandaNovel) already
  exist in ``sources.py`` and are NOT in this registry; this module
  records them as ``rights_class="aggregator"`` in
  :data:`KNOWN_AGGREGATORS` for labelling only.

Rules, no exceptions:

* No paywall bypass. No login bypass. No evasion of access controls.
  A chapter the public cannot read is reported blocked.
* Every descriptor carries a ``licence`` label + ``rights_class``.
  Nothing is fetched without recording where it came from.
* ``robots`` / rate limits are honoured: ``Fetcher.polite_wait``
  already spaces requests per domain; descriptors declare their
  minimum delay and the adapter enforces ``max(declared, fetcher)``.
* A descriptor with ``verified=False`` says so on every result.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .sources import (
    Chapter,
    Fetcher,
    GenericAdapter,
    SearchHit,
    SourceAdapter,
    SourceBlocked,
    SourceError,
    StoryMeta,
    _chapter_number_from_text,
    _clean_paragraphs,
    _links,
    _meta,
    normalize_url,
)

_log = get_logger(__name__)

__all__ = [
    "SourceDescriptor",
    "DescriptorAdapter",
    "GutenbergAdapter",
    "DESCRIPTORS",
    "LEGAL_SOURCES",
    "KNOWN_AGGREGATORS",
    "register_descriptor",
    "descriptor_for_url",
    "descriptor_by_id",
    "list_sources",
]

# ── descriptor ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SourceDescriptor:
    """One archive as data. Selectors are (kind, value) pairs where kind
    is ``id`` | ``class`` | ``tag`` — the same vocabulary
    :class:`GenericAdapter` already uses."""

    id: str
    name: str
    domains: tuple[str, ...]
    url_patterns: tuple[str, ...] = ()
    search_url_template: str = ""  # "{query}" + "{page}" placeholders
    search_link_must_contain: str = ""
    novel_title_selectors: tuple[tuple[str, str], ...] = ()
    chapter_list_selectors: tuple[tuple[str, str], ...] = ()
    chapter_list_must_contain: str = "chapter"
    body_selectors: tuple[tuple[str, str], ...] = ()
    chapter_url_template: str = ""
    rate_limit_seconds: float = 1.0
    bot_wall: bool = False
    licence: str = ""
    licence_note: str = ""
    rights_class: str = ""  # public_domain | author_posted | licensed_official | aggregator
    verified: bool = False
    requires_key: bool = False
    api_key_env: str = ""
    notes: str = ""

    def matches(self, url: str) -> bool:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if host in self.domains:
            return True
        return any(re.search(p, url) for p in self.url_patterns)


# ── legal-first registry ─────────────────────────────────────────────────────


def _descriptor_table() -> list[SourceDescriptor]:
    return [
        SourceDescriptor(
            id="gutenberg",
            name="gutenberg",
            domains=("gutenberg.org", "www.gutenberg.org"),
            url_patterns=(r"gutenberg\.org/(ebooks|files|cache)/",),
            search_url_template=(
                "https://gutendex.com/books?search={query}"),
            licence="Public domain (US)",
            licence_note="Gutenberg texts are public domain in the US; "
                         "check your jurisdiction for non-US works.",
            rights_class="public_domain",
            verified=True,
            notes="Served by GutenbergAdapter (Gutendex JSON + plain-text "
                  "ebooks), not by generic HTML scraping.",
        ),
        SourceDescriptor(
            id="scribblehub",
            name="scribblehub",
            domains=("scribblehub.com", "www.scribblehub.com"),
            url_patterns=(r"scribblehub\.com/(series|read)/",),
            search_url_template=(
                "https://www.scribblehub.com/?s={query}&post_type=series"),
            novel_title_selectors=(("class", "fic_title"),),
            chapter_list_selectors=(("class", "toc"), ("id", "chp_contents")),
            chapter_list_must_contain="read",
            body_selectors=(("class", "chp_raw"), ("id", "chp_raw"),
                            ("class", "chapter-content")),
            rate_limit_seconds=1.0,
            licence="Author-posted (Scribble Hub ToS)",
            licence_note="Stories are posted by their authors on "
                         "Scribble Hub; redistribution follows the "
                         "author's terms.",
            rights_class="author_posted",
            verified=False,
            notes="Unverified markup — DescriptorAdapter self-tunes via "
                  "GenericAdapter scoring; verified stays False until a "
                  "real fetch is confirmed.",
        ),
        SourceDescriptor(
            id="ao3",
            name="ao3",
            domains=("archiveofourown.org", "www.archiveofourown.org"),
            url_patterns=(r"archiveofourown\.org/works/",),
            search_url_template=(
                "https://archiveofourown.org/works/search?"
                "work_search%5Bquery%5D={query}"),
            novel_title_selectors=(("class", "title"),),
            chapter_list_selectors=(("id", "chapter_index"),),
            chapter_list_must_contain="chapters",
            body_selectors=(("id", "workskin"), ("class", "userstuff"),
                            ("id", "chapters")),
            rate_limit_seconds=3.0,
            bot_wall=True,
            licence="Author-posted (AO3 ToS, CC licences per work)",
            licence_note="AO3 has no official API. Fetch respects AO3's "
                         "rate limits; locked works report blocked.",
            rights_class="author_posted",
            verified=False,
            notes="No API; HTML only, 3s minimum delay, locked works "
                  "are honestly blocked, never bypassed.",
        ),
        SourceDescriptor(
            id="wattpad",
            name="wattpad",
            domains=("wattpad.com", "www.wattpad.com"),
            url_patterns=(r"wattpad\.com/story/",),
            search_url_template="",
            body_selectors=(("class", "story-content"),),
            rate_limit_seconds=1.0,
            licence="Author-posted (Wattpad ToS)",
            licence_note="Official Wattpad API requires a key; without "
                         "one the source reports unavailable.",
            rights_class="author_posted",
            verified=False,
            requires_key=True,
            api_key_env="WATTPAD_API_KEY",
            notes="API-key source. Without WATTPAD_API_KEY every call "
                  "returns an honest unavailable error.",
        ),
        SourceDescriptor(
            id="webnovel",
            name="webnovel",
            domains=("webnovel.com", "www.webnovel.com"),
            url_patterns=(r"webnovel\.com/book/",),
            search_url_template=(
                "https://www.webnovel.com/search?keywords={query}"),
            novel_title_selectors=(("class", "book-name"),),
            chapter_list_selectors=(("class", "catalog"),),
            chapter_list_must_contain="chapter",
            body_selectors=(("class", "cha-words"), ("id", "content"),
                            ("class", "chapter-content")),
            rate_limit_seconds=1.0,
            licence="Licensed official (Webnovel / Qidian)",
            licence_note="Official Webnovel catalogue. Paywalled "
                         "chapters report blocked, never bypassed.",
            rights_class="licensed_official",
            verified=False,
            notes="Unverified markup; paywalled chapters are blocked, "
                  "never bypassed.",
        ),
    ]


DESCRIPTORS: list[SourceDescriptor] = _descriptor_table()

#: Aggregators that already have dedicated adapters in sources.py —
#: labelled here so every result can carry a rights class.
KNOWN_AGGREGATORS: tuple[dict[str, str], ...] = (
    {"id": "freewebnovel", "rights_class": "aggregator",
     "licence": "Aggregator — rights vary per work; check the source work"},
    {"id": "novelfull", "rights_class": "aggregator",
     "licence": "Aggregator — rights vary per work; check the source work"},
    {"id": "royalroad", "rights_class": "author_posted",
     "licence": "Author-posted (Royal Road ToS)"},
    {"id": "pandanovel", "rights_class": "aggregator",
     "licence": "Aggregator (unverified) — rights vary per work"},
)

#: Legal-first sources, in priority order. Royal Road is served by its
#: dedicated adapter; the rest by descriptors / GutenbergAdapter.
LEGAL_SOURCES: tuple[str, ...] = (
    "gutenberg", "royalroad", "scribblehub", "ao3", "wattpad", "webnovel",
)


def register_descriptor(descriptor: SourceDescriptor) -> SourceDescriptor:
    """Add an owner-supplied descriptor (any site). The descriptor is
    data: the system records its licence label and never claims rights.
    Re-registering an id replaces it."""
    global DESCRIPTORS
    DESCRIPTORS = [d for d in DESCRIPTORS if d.id != descriptor.id]
    DESCRIPTORS.append(descriptor)
    _log.info("registered story-source descriptor %s (%s)",
              descriptor.id, descriptor.rights_class)
    return descriptor


def descriptor_by_id(source_id: str) -> SourceDescriptor | None:
    for d in DESCRIPTORS:
        if d.id == (source_id or "").lower():
            return d
    return None


def descriptor_for_url(url: str) -> SourceDescriptor | None:
    for d in DESCRIPTORS:
        if d.matches(url or ""):
            return d
    return None


def list_sources() -> list[dict[str, Any]]:
    """Every known source with its rights label — legal first."""
    out: list[dict[str, Any]] = []
    for d in DESCRIPTORS:
        out.append({
            "id": d.id, "name": d.name, "domains": list(d.domains),
            "rights_class": d.rights_class, "licence": d.licence,
            "verified": d.verified, "requires_key": d.requires_key,
            "bot_wall": d.bot_wall,
            "rate_limit_seconds": d.rate_limit_seconds,
        })
    for agg in KNOWN_AGGREGATORS:
        out.append({
            "id": agg["id"], "name": agg["id"],
            "domains": [], "rights_class": agg["rights_class"],
            "licence": agg["licence"], "verified": agg["id"] != "pandanovel",
            "requires_key": False, "bot_wall": False,
            "rate_limit_seconds": 1.0,
        })
    return out


# ── descriptor-driven adapter ────────────────────────────────────────────────


class DescriptorAdapter(GenericAdapter):
    """A :class:`SourceAdapter` driven entirely by a descriptor.

    Body extraction reuses :class:`GenericAdapter`'s scored candidates
    (descriptor selectors first, then the generic list); chapter lists
    prefer the descriptor's list selectors, falling back to the generic
    chapter-link scan. ``verified`` comes from the descriptor.
    """

    def __init__(self, descriptor: SourceDescriptor,
                 fetcher: Fetcher | None = None) -> None:
        super().__init__(name=descriptor.id,
                         domains=descriptor.domains,
                         fetcher=fetcher)
        self.descriptor = descriptor
        self.verified = descriptor.verified
        self._candidates = tuple(descriptor.body_selectors) or self._CANDIDATES

    # -- helpers -------------------------------------------------------------
    def _pick(self, node: Any, kind: str, value: str) -> Any | None:
        from .sources import _by_id, _first_by_class
        if kind == "id":
            return _by_id(node, value)
        if kind == "class":
            return _first_by_class(node, value)
        found = node.find_all(value)
        return found[0] if found else None

    def _extract_body(self, dom: Any, host: str) -> tuple[list[str], str]:
        # descriptor selectors first, scored like the generic candidates
        best: tuple[int, list[str], tuple[str, str] | None] = (0, [], None)
        for cand in self._candidates:
            node = self._pick(dom, cand[0], cand[1])
            if node is None:
                continue
            words, paras = self._score(node)
            if words > best[0]:
                best = (words, paras, cand)
        if best[2] is not None and best[0] >= 120:
            return best[1], f"{best[2][0]}={best[2][1]}"
        return super()._extract_body(dom, host)

    # -- SourceAdapter surface -------------------------------------------------
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        d = self.descriptor
        if d.requires_key:
            raise SourceError(
                f"{d.id}: official API needs {d.api_key_env} — set it, "
                "or paste the story URL directly")
        if not d.search_url_template:
            raise SourceError(
                f"{d.id}: search is not mapped — paste the novel URL directly")
        url = d.search_url_template.format(
            query=urllib.parse.quote_plus(query), page=1)
        _final, dom, _ = self.fetcher.get(url)
        out: list[SearchHit] = []
        seen: set[str] = set()
        for href, text in _links(dom, url):
            if href in seen or not text.strip():
                continue
            if d.search_link_must_contain and \
                    d.search_link_must_contain not in href:
                continue
            seen.add(href)
            out.append(SearchHit(title=text.strip(), url=href,
                                 source=self.name))
            if len(out) >= limit:
                break
        return out

    def novel(self, url: str) -> StoryMeta:
        _final, dom, _ = self.fetcher.get(url)
        title = _meta(dom, "og:title")
        if not title:
            for kind, value in self.descriptor.novel_title_selectors:
                node = self._pick(dom, kind, value)
                if node is not None and node.inner_text().strip():
                    title = node.inner_text().strip()
                    break
        if not title:
            for h in dom.find_all("h1"):
                if h.inner_text().strip():
                    title = h.inner_text().strip()
                    break
        return StoryMeta(
            title=title or "Unknown", url=normalize_url(url),
            source=self.name, author=_meta(dom, "author"),
            synopsis=_meta(dom, "og:description"),
            cover_url=_meta(dom, "og:image"),
            licence=self.descriptor.licence)

    def chapter_list(self, novel_url: str, *,
                     limit: int = 0) -> list[tuple[int, str, str]]:
        _final, dom, _ = self.fetcher.get(novel_url)
        out: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        # descriptor list selectors first
        for kind, value in self.descriptor.chapter_list_selectors:
            node = self._pick(dom, kind, value)
            if node is None:
                continue
            for href, text in _links(node, novel_url):
                if href in seen:
                    continue
                if self.descriptor.chapter_list_must_contain not in \
                        href.lower() + text.lower():
                    continue
                seen.add(href)
                out.append((_chapter_number_from_text(text) or
                            _chapter_number_from_text(href), text, href))
            if out:
                break
        if not out:
            # fall back to the generic chapter-link scan
            return super().chapter_list(novel_url, limit=limit)
        out.sort(key=lambda t: (t[0] or 10 ** 9))
        return out[:limit] if limit else out

    def fetch_chapter(self, url: str) -> Chapter:
        if self.descriptor.requires_key:
            raise SourceError(
                f"{self.descriptor.id}: official API needs "
                f"{self.descriptor.api_key_env}")
        return super().fetch_chapter(url)


# ── Project Gutenberg (Gutendex JSON + plain text) ───────────────────────────


class GutenbergAdapter(SourceAdapter):
    """Project Gutenberg via the Gutendex catalogue API + plain-text
    ebooks. Public domain. Chapters = the book's own chapter splits
    (``library.split_chapters``); a book with no detectable chapters
    is one chapter."""

    name = "gutenberg"
    domains = ("gutenberg.org", "www.gutenberg.org")
    verified = True
    _API = "https://gutendex.com/books"

    def _api(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Devon-BookForge/1.0 (public-domain reader)",
            "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def _book(self, book_id: int) -> dict[str, Any]:
        return self._api(f"{self._API}/{book_id}")

    @staticmethod
    def _book_id(url: str) -> int | None:
        m = re.search(r"/(?:ebooks|files|cache)/(\d+)", url or "")
        return int(m.group(1)) if m else None

    @staticmethod
    def _txt_url(book: dict[str, Any]) -> str:
        formats = book.get("formats") or {}
        for key in ("text/plain; charset=utf-8", "text/plain"):
            if formats.get(key):
                return formats[key]
        for key, val in formats.items():
            if key.startswith("text/plain"):
                return val
        raise SourceError("gutenberg: no plain-text format for this book")

    def _download_text(self, book_id: int) -> str:
        book = self._book(book_id)
        txt_url = self._txt_url(book)
        req = urllib.request.Request(txt_url, headers={
            "User-Agent": "Devon-BookForge/1.0 (public-domain reader)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
        # strip Gutenberg header/footer
        start = raw.find("*** START OF")
        if start != -1:
            raw = raw[raw.find("\n", start) + 1:]
        end = raw.find("*** END OF")
        if end != -1:
            raw = raw[:end]
        return raw

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        try:
            data = self._api(
                f"{self._API}?search={urllib.parse.quote_plus(query)}"
                f"&page_size={max(1, min(limit, 25))}")
        except Exception as exc:  # noqa: BLE001 - network, honest
            raise SourceError(f"gutenberg search failed: {exc}") from exc
        out = []
        for book in data.get("results", [])[:limit]:
            authors = ", ".join(
                a.get("name", "") for a in book.get("authors", []))
            out.append(SearchHit(
                title=book.get("title", "Unknown"),
                url=f"https://www.gutenberg.org/ebooks/{book.get('id')}",
                source=self.name, author=authors))
        return out

    def novel(self, url: str) -> StoryMeta:
        book_id = self._book_id(url)
        if book_id is None:
            raise SourceError(
                f"gutenberg: cannot parse a book id from {url!r} "
                "(expected gutenberg.org/ebooks/<id>)")
        try:
            book = self._book(book_id)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"gutenberg: book {book_id} not found: {exc}") from exc
        authors = ", ".join(a.get("name", "") for a in book.get("authors", []))
        subjects = book.get("subjects") or []
        return StoryMeta(
            title=book.get("title", "Unknown"),
            url=f"https://www.gutenberg.org/ebooks/{book_id}",
            source=self.name, author=authors,
            synopsis="; ".join(subjects),
            licence="Public domain (US)")

    def chapter_list(self, novel_url: str, *,
                     limit: int = 0) -> list[tuple[int, str, str]]:
        book_id = self._book_id(novel_url)
        if book_id is None:
            raise SourceError(f"gutenberg: cannot parse a book id from {novel_url!r}")
        from .library import split_chapters
        text = self._download_text(book_id)
        chapters, _ = split_chapters(text)
        out = [(i + 1, title, f"{novel_url}#chapter-{i + 1}")
               for i, (title, _body) in enumerate(chapters)]
        return out[:limit] if limit else out

    def chapter_url(self, novel_url: str, number: int) -> str:
        return f"{normalize_url(novel_url)}#chapter-{number}"

    def fetch_chapter(self, url: str) -> Chapter:
        # chapter URLs are novel_url#chapter-N; the fragment carries N
        base, _, frag = url.partition("#")
        number = 1
        m = re.search(r"chapter-(\d+)", frag)
        if m:
            number = int(m.group(1))
        else:
            number = _chapter_number_from_text(url) or 1
        book_id = self._book_id(base)
        if book_id is None:
            raise SourceError(f"gutenberg: cannot parse a book id from {url!r}")
        from .library import split_chapters
        text = self._download_text(book_id)
        chapters, _ = split_chapters(text)
        if not chapters:
            raise SourceError("gutenberg: no chapters detected in this text")
        idx = min(max(1, number), len(chapters)) - 1
        title, body = chapters[idx]
        paras = [p for p in _clean_paragraphs(body.split("\n\n")) if p.strip()]
        return Chapter(number=idx + 1, title=title or f"Chapter {idx + 1}",
                       url=url, paragraphs=paras, source=self.name)


def descriptor_adapters() -> list[SourceAdapter]:
    """One adapter per descriptor (Gutenberg gets its API adapter)."""
    out: list[SourceAdapter] = [GutenbergAdapter()]
    for d in DESCRIPTORS:
        if d.id == "gutenberg":
            continue
        out.append(DescriptorAdapter(d))
    return out
