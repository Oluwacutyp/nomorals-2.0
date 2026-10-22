"""ArchiveIngestor — the fetch → parse → corpus ingest machine.

Phase 2 of the WisdomKeeper module (spec §2, §3D, §4).

Owns one :class:`CanonCorpus` and turns manifest entries into corpus
texts: fetch the bytes, verify the sha256, parse into text, hand the text
to the corpus. Fetch failures raise :class:`IngestError` — never silent,
never empty bytes.

Library search is deliberately *not* ingestion: :meth:`search` returns
candidates from public archives (archive.org first, a plain web search as
fallback); moving a candidate into the corpus is a separate explicit
:meth:`ingest_entry` step. No logins are ever scraped.

Pass #94 (open-sources law): the hunt's backends are a source class in
the ONE source registry (``media/source_registry.py``), like the music
chain (#88), the book sites (#89) and the film chain (#93) before them.
Until this pass the class was CLOSED: :meth:`search` tried
:data:`SEARCH_SOURCES` in a fixed order, a backend's failure lived only
in a local list that was thrown away unless *every* backend failed, no
outcome ever landed anywhere, and the ingest fetch path credited no
host for anything. Now, when a registry is in reach (injected, or the
shared ``DEFAULT_REGISTRY`` when the context carries a database):

* every backend's host is a registry descriptor
  (:data:`WISDOM_SOURCE_DOMAINS` is the class membership list;
  archive.org and lite.duckduckgo.com are shared descriptors whose
  kinds the wisdom class extends, gutendex.com and
  en.wikisource.org are wisdom seeds);
* :meth:`search` records every attempt's outcome in the one
  vocabulary — served / no-result (weak) / blocked / rate-limited /
  operational failure — classified from the error evidence the
  backends already raise (HTTP status via the cause chain, the
  robots-denial message), and the *default* hunt runs in the
  registry's learned order (day one: no evidence → today's order,
  byte-for-byte; a caller-passed ``sources`` order is the caller's
  explicit instruction and is never reordered);
* discovery is genuine on two real flows: foreign hosts among the
  search results intake as untrusted candidates
  (``kinds=("wisdom-archive",)``), and archive hosts the ingest
  fetch path successfully serves from — manifest entries and the
  research-fed queue bring new archives into the corpus — intake
  the same way.  Candidates serve nothing until the health loop
  promotes them; a promoted archive serves through the generic
  ingest fetch path itself (:meth:`fetch_bytes` is host-agnostic),
  its outcomes credited like any seed's.  There is no generic
  archive-*search* shape (nothing like the film chain's WordPress
  shape exists for archives), so grown sources do not join
  :meth:`search` — named absence, not invention;
* :meth:`fetch_bytes` credits the fetched host's descriptor
  (served, or the classified failure) when the registry knows the
  host — cache hits fetch nothing and earn nothing;
* with NO registry in reach, behaviour is exactly the pre-#94
  behaviour — the #88/#89/#93 discipline.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.errors import NoMoralsError, http_status_of
from ..core.http import HttpClient, HttpResponse
from ..core.logging_setup import get_logger
from ..documents.errors import DocumentError
from ..documents.model import full_text
from ..documents.parsers import parse_bytes
from ..tools.web import RobotsCache
from .corpus import CanonCorpus, ManifestEntry
from .errors import CorpusError, IngestError

__all__ = [
    "ArchiveIngestor",
    "WISDOM_KIND",
    "WISDOM_SOURCE_DOMAINS",
    "wisdom_registry_for_context",
]

_log = get_logger(__name__)

#: archive.org advanced search endpoint (no key needed).
_ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"

#: Item landing page for a search candidate.
_ARCHIVE_DETAILS = "https://archive.org/details/"

#: DuckDuckGo's lightweight HTML endpoint, used only as a fallback when
#: archive.org search is unreachable. No JS, no login, robots-respected.
_DDG_LITE = "https://lite.duckduckgo.com/lite/"

#: User-Agent identifying the WisdomKeeper crawler.
_USER_AGENT = "DevonWisdom/1.0 (WisdomKeeper archive ingestor)"

#: Gutendex - free, no-key JSON API over the Project Gutenberg catalog.
#: Structured search with authors/subjects/formats metadata (the clean
#: discovery path: archive.org search is token-soup by comparison).
_GUTENDEX = "https://gutendex.com/books/"

#: Wikisource MediaWiki API - structured full-text search over the
#: proofread public-domain library (curates individual works, links
#: translations to originals).
_WIKISOURCE_API = "https://en.wikisource.org/w/api.php"

#: Canonical plain-text URL for a Gutenberg ebook id.
_GUTENBERG_TEXT = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

#: Minimum seconds between fetches to the same host (politeness).
_POLITE_DELAY = 1.0

#: Fetch retry policy: attempts and the backoff slept before attempts 2..n.
_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF = (0.5, 1.5)

#: Blob cache directory and its URL→sha index, both under workspace/wisdom/.
_BLOBS_DIR = "blobs"
_BLOB_INDEX = "index.json"

#: Upper bound on a single fetched blob — protects against runaway downloads.
_MAX_BLOB_BYTES = 100 * 1024 * 1024

#: archive.org search fields we request.
_SEARCH_FIELDS = ("identifier", "title", "description")

_LINK_RE = re.compile(
    r'(?is)<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>')

#: The registry kind the wisdom archive class carries (Pass #94).
WISDOM_KIND = "wisdom-archive"

#: The host each wisdom search backend actually fetches from — the
#: backend's membership in the ONE source registry (Pass #94).
#: This map is the wisdom class's membership list: the search seam
#: credits outcomes to the registry descriptor for this domain,
#: the registry's status surface renders the class from it, and
#: the health loop probes the two template-carrying seeds through
#: their real search endpoints.  Domains are evidence from this
#: module (each backend's endpoint constant), NOT marketing
#: domains: the web backend's host is lite.duckduckgo.com (the
#: #90 web-search seed — the same host serving both classes), and
#: the archive backend's is archive.org (the #88 seed).  Keys are
#: in :attr:`ArchiveIngestor.SEARCH_SOURCES` order.
WISDOM_SOURCE_DOMAINS: dict[str, str] = {
    "gutendex": "gutendex.com",
    "wikisource": "en.wikisource.org",
    "archive": "archive.org",
    "web": "lite.duckduckgo.com",
}


class ArchiveIngestor:
    """Fetch, cache, parse, and ingest public-domain esoteric texts."""

    def __init__(self, context: Any, *, registry: Any = None) -> None:
        self.context = context
        # Pass #94: the source registry this ingestor's outcomes
        # land in (None = no registry in reach → pre-#94 behaviour,
        # exactly).  Resolved once: the context is fixed per
        # instance, and the resolver binds the shared registry to
        # the context's KV store only when it carries a database.
        self._registry = wisdom_registry_for_context(context, registry)
        self.corpus = CanonCorpus(context)
        self._http = HttpClient(timeout=30.0, user_agent=_USER_AGENT)
        self._robots = RobotsCache()
        self._blobs = self._wisdom_root() / _BLOBS_DIR
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._index_path = self._blobs / _BLOB_INDEX
        self._index: dict[str, dict[str, Any]] | None = None
        # host -> last fetch epoch (per-domain politeness delay)
        self._last_fetch: dict[str, float] = {}

    # ── paths ─────────────────────────────────────────────────────────
    def _wisdom_root(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = None
        if settings is not None:
            root = getattr(settings, "workspace_dir", None)
        if not root:
            root = Path.cwd() / "workspace"
        d = Path(root) / "wisdom"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── blob cache ────────────────────────────────────────────────────
    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            self._index = {}
            if self._index_path.is_file():
                try:
                    raw = json.loads(self._index_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raw = {}
                if isinstance(raw, dict):
                    self._index = {
                        str(k): v for k, v in raw.items()
                        if isinstance(v, dict) and "sha" in v
                    }
        return self._index

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._load_index(), indent=2, ensure_ascii=False),
            encoding="utf-8")

    def _blob_path(self, sha: str) -> Path:
        return self._blobs / sha

    def _cached_bytes(self, url: str) -> bytes | None:
        """Return cached bytes for a URL, or None on any miss/corruption."""
        record = self._load_index().get(url)
        if not record:
            return None
        path = self._blob_path(str(record["sha"]))
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def _store_blob(self, url: str, data: bytes) -> str:
        """Store fetched bytes keyed by sha256; returns the digest."""
        sha = hashlib.sha256(data).hexdigest()
        path = self._blob_path(sha)
        if not path.is_file():
            path.write_bytes(data)
        index = self._load_index()
        index[url] = {"sha": sha, "at": time.time(), "bytes": len(data)}
        self._save_index()
        return sha

    # ── fetch ─────────────────────────────────────────────────────────
    @staticmethod
    def _check_url(url: str) -> str:
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            raise IngestError(url, "refusing to fetch a non-http(s) URL")
        return url

    def _check_robots(self, url: str) -> None:
        try:
            allowed = self._robots.allowed(url, _USER_AGENT, self._http)
        except Exception:  # noqa: BLE001 - a robots check must never hard-fail
            allowed = True
        if not allowed:
            raise IngestError(url, "robots.txt disallows fetching this URL")

    def _attempt(self, url: str) -> bytes:
        response: HttpResponse = self._http.get(url)
        if response.body is None:
            raise IngestError(url, "fetch returned no body")
        if len(response.body) > _MAX_BLOB_BYTES:
            raise IngestError(
                url, f"blob too large ({len(response.body)} bytes)")
        return response.body

    def _polite_wait(self, url: str) -> None:
        """Sleep just enough to keep >= _POLITE_DELAY between fetches
        to the same host. Never sleeps on cache hits."""
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if not host:
            return
        wait = _POLITE_DELAY - (time.time() - self._last_fetch.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last_fetch[host] = time.time()

    def fetch_bytes(self, url: str) -> bytes:
        """GET a URL with retries and blob caching.

        Raises :class:`IngestError` on HTTP errors, robots denials,
        empty bodies, or exhausted retries — never returns empty bytes.

        Pass #94: with a registry in reach, a real fetch is evidence
        for the fetched host — a serve credits its descriptor (or
        intakes the host as an untrusted wisdom candidate when the
        registry has never seen it: the corpus's own manifest and
        research queue are a genuine discovery flow), a failure
        lands with its class.  A cache hit fetches nothing and
        earns nothing.  The recording is best-effort bookkeeping
        around the unchanged raise/return contract.
        """
        url = self._check_url(url)
        cached = self._cached_bytes(url)
        if cached is not None:
            return cached
        reg = self._registry
        t0 = time.perf_counter()
        try:
            self._check_robots(url)
        except IngestError as exc:
            _record_wisdom_fetch(reg, url, None,
                                 time.perf_counter() - t0, exc)
            raise
        self._polite_wait(url)
        try:
            data = self._get_with_retry(url)
        except IngestError as exc:
            _record_wisdom_fetch(reg, url, None,
                                 time.perf_counter() - t0, exc)
            raise
        if not data:
            # Defensive: _get_with_retry should already have raised, but a
            # silent empty result must never escape this boundary.
            raise IngestError(url, "fetch returned empty bytes")
        self._store_blob(url, data)
        _record_wisdom_fetch(reg, url, data,
                             time.perf_counter() - t0, None)
        return data

    def fetch(self, entry: ManifestEntry) -> bytes:
        """Fetch the bytes for a manifest entry."""
        entry.validate()
        return self.fetch_bytes(entry.source_url)

    def _get_with_retry(self, url: str) -> bytes:
        last: Exception | None = None
        for attempt in range(1, _FETCH_ATTEMPTS + 1):
            try:
                return self._attempt(url)
            except IngestError:
                raise  # fail fast: HTTP errors, robots, size — not retried
            except NoMoralsError as exc:
                last = exc
                if not exc.retryable:
                    raise IngestError(url, str(exc)) from exc
            except (OSError, TimeoutError) as exc:
                last = exc
            if attempt < _FETCH_ATTEMPTS:
                time.sleep(_FETCH_BACKOFF[attempt - 1])
        raise IngestError(url, f"fetch failed after {_FETCH_ATTEMPTS} "
                               f"attempts: {last}") from last

    # ── parse ─────────────────────────────────────────────────────────
    @staticmethod
    def _filename_for(url: str, fallback: str) -> str:
        name = urllib.parse.unquote(
            Path(urllib.parse.urlparse(url).path).name).strip()
        if not name or name in {".", ".."}:
            return fallback
        return name

    # Project Gutenberg boilerplate markers. Ingesting raw PG text
    # without stripping these pollutes the corpus with license headers
    # (the long-noted Gutenberg quality criticism).
    _PG_START = "*** START OF"
    _PG_END = "*** END OF"

    @classmethod
    def strip_gutenberg_boilerplate(cls, text: str) -> str:
        """Remove Project Gutenberg header/footer boilerplate.

        Cuts everything up to and including the "*** START OF ..."
        line, and everything from the "*** END OF ..." line on.
        Returns the text unchanged when the markers are absent.
        """
        if cls._PG_START not in text and cls._PG_END not in text:
            return text
        lines = text.splitlines(keepends=True)
        start = 0
        for i, line in enumerate(lines):
            if cls._PG_START in line and "PROJECT GUTENBERG" in line:
                start = i + 1
                break
        end = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if cls._PG_END in lines[i] and "PROJECT GUTENBERG" in lines[i]:
                end = i
                break
        return "".join(lines[start:end]).strip() + "\n"

    def parse(self, data: bytes, filename: str) -> str:
        """Parse raw bytes into full text via the document engine.

        Raises :class:`IngestError` on empty input, unparseable input,
        or empty extracted text.
        """
        if not data:
            raise IngestError(filename, "refusing to parse empty bytes")
        if not (filename or "").strip():
            raise IngestError("<unknown>", "parse needs a filename")
        try:
            document = parse_bytes(data, filename=filename)
        except DocumentError as exc:
            raise IngestError(filename, f"could not parse document: {exc}"
                             ) from exc
        text = full_text(document)
        if not text.strip():
            raise IngestError(filename, "document parsed to empty text")
        # Plain-text sources (Gutenberg et al.) carry license
        # boilerplate — strip it before the text reaches the corpus.
        text = self.strip_gutenberg_boilerplate(text)
        if not text.strip():
            raise IngestError(
                filename, "document parsed to empty text after "
                "boilerplate stripping")
        return text

    # ── ingest ────────────────────────────────────────────────────────
    def ingest_entry(self, entry: ManifestEntry, *,
                     edition: str = "") -> ManifestEntry:
        """Fetch, verify, parse, and ingest one manifest entry.

        Idempotent: if the manifest sha matches the fetched bytes and the
        entry is already ingested, the entry is returned unchanged.
        Registers the entry in the manifest first when it is new.

        ``edition`` records source-edition provenance (which edition /
        printing the text came from — the Gutenberg scholarly-rigor
        criticism) into the manifest notes when the entry has none.
        """
        entry.validate()
        data = self.fetch_bytes(entry.source_url)
        digest = hashlib.sha256(data).hexdigest()
        if entry.sha256 == digest and entry.ingested_at:
            return entry  # already ingested, unchanged bytes
        filename = self._filename_for(entry.source_url, f"{entry.slug}.bin")
        text = self.parse(data, filename)
        if edition and not entry.notes:
            entry.notes = f"source edition: {edition[:300]}"
        try:
            self.corpus.get(entry.slug)
        except CorpusError:
            self.corpus.register(entry)
        return self.corpus.ingest_text(entry.slug, text)

    def ingest_text(self, slug: str, text: str, *,
                    translator: str = "") -> ManifestEntry:
        """Direct text ingest passthrough to the corpus."""
        return self.corpus.ingest_text(slug, text, translator=translator)

    # ── search ────────────────────────────────────────────────────────
    #: Search backends tried in order (each can be disabled via the
    #: ``sources`` parameter of :meth:`search`).
    SEARCH_SOURCES = ("gutendex", "wikisource", "archive", "web")

    def search(self, query: str, tradition: str = "",
               max_results: int = 10,
               sources: "tuple[str, ...] | list[str] | None" = None
               ) -> list[dict[str, Any]]:
        """Find candidate texts in public archives. No auto-ingest.

        Tries each backend in order — Gutendex (structured Gutenberg
        catalog), Wikisource (proofread library), archive.org advanced
        search, then a plain web search — merging candidates and
        deduplicating by URL. Every candidate carries
        identifier/title/url/description (+ authors/subjects/edition
        when the backend provides them); ingestion is a separate step.
        Raises :class:`IngestError` only when every enabled backend fails.

        Pass #94: with a registry in reach, every attempt's outcome
        lands in the registry (served / no-result / blocked /
        rate-limited / failure, classified from the raised error's
        evidence) and foreign hosts among the results intake as
        untrusted wisdom candidates.  The DEFAULT hunt (``sources``
        not passed) runs in the registry's learned order once any
        evidence exists; an explicit ``sources`` order is the
        caller's instruction and is never reordered.
        """
        query = (query or "").strip()
        if not query:
            raise IngestError("<search>", "search needs a non-empty query")
        if max_results < 1:
            raise IngestError("<search>", "max_results must be >= 1")
        wanted = tuple(sources) if sources else self.SEARCH_SOURCES
        for s in wanted:
            if s not in self.SEARCH_SOURCES:
                raise IngestError(
                    "<search>", f"unknown search source {s!r} "
                    f"(known: {', '.join(self.SEARCH_SOURCES)})")
        _backends = {
            "gutendex": self._search_gutendex,
            "wikisource": self._search_wikisource,
            "archive": self._search_archive,
            "web": self._search_web,
        }
        reg = self._registry
        order: list[str] = list(wanted)
        if reg is not None and sources is None:
            # The default hunt learns; an explicit sources order is
            # the caller's instruction and stays exactly as passed.
            order = _order_wisdom(reg, order)
        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        failures: list[str] = []
        for name in order:
            t0 = time.perf_counter()
            try:
                found = _backends[name](query, tradition, max_results)
            except IngestError as exc:
                # One backend down must not kill the hunt; the failure
                # is recorded and the next backend is tried.
                failures.append(f"{name}: {exc}")
                _log.debug("wisdom search backend %s failed: %s", name, exc)
                _record_wisdom_search(reg, name, [],
                                      time.perf_counter() - t0, exc)
                continue
            _record_wisdom_search(reg, name, found,
                                  time.perf_counter() - t0, None)
            for cand in found:
                url = str(cand.get("url") or "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    cand.setdefault("source", name)
                    merged.append(cand)
                if len(merged) >= max_results:
                    break
            if len(merged) >= max_results:
                break
        if not merged:
            raise IngestError(
                query, "every search backend failed: "
                + "; ".join(failures))
        return merged[:max_results]

    def _search_gutendex(self, query: str, tradition: str,
                         max_results: int) -> list[dict[str, Any]]:
        """Project Gutenberg catalog via the Gutendex JSON API (no key).

        Returns candidates with authors, subjects, and a direct
        plain-text URL — the highest-signal source for public-domain
        books.
        """
        params = {"search": f"{query} {tradition}".strip()}
        url = f"{_GUTENDEX}?{urllib.parse.urlencode(params)}"
        try:
            response = self._http.get(url)
            payload = json.loads(response.text)
        except NoMoralsError as exc:
            raise IngestError(url, f"gutendex search failed: {exc}") from exc
        except (ValueError, OSError) as exc:
            raise IngestError(url, f"gutendex search failed: {exc}") from exc
        results: list[dict[str, Any]] = []
        for book in (payload.get("results") or [])[:max_results]:
            gid = book.get("id")
            title = str(book.get("title") or "").strip()
            if not gid or not title:
                continue
            authors = ", ".join(
                a.get("name", "") for a in (book.get("authors") or [])
                if a.get("name"))[:200]
            subjects = "; ".join(
                str(s) for s in (book.get("subjects") or [])[:6])
            results.append({
                "identifier": f"gutenberg-{gid}",
                "title": title,
                "url": _GUTENBERG_TEXT.format(id=gid),
                "description": subjects,
                "authors": authors,
                "subjects": subjects,
                "edition": f"Project Gutenberg ebook #{gid}",
            })
        return results

    def _search_wikisource(self, query: str, tradition: str,
                           max_results: int) -> list[dict[str, Any]]:
        """English Wikisource via the MediaWiki API.

        Proofread, curated-at-work-level texts — the best source for
        individual poems/treatises rather than whole volumes.
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{query} {tradition}".strip(),
            "srlimit": str(max_results),
            "srnamespace": "0",
            "format": "json",
        }
        url = f"{_WIKISOURCE_API}?{urllib.parse.urlencode(params)}"
        try:
            response = self._http.get(url)
            payload = json.loads(response.text)
        except NoMoralsError as exc:
            raise IngestError(
                url, f"wikisource search failed: {exc}") from exc
        except (ValueError, OSError) as exc:
            raise IngestError(
                url, f"wikisource search failed: {exc}") from exc
        results: list[dict[str, Any]] = []
        query_block = payload.get("query") or {}
        for hit in (query_block.get("search") or [])[:max_results]:
            title = str(hit.get("title") or "").strip()
            if not title:
                continue
            page = urllib.parse.quote(title.replace(" ", "_"))
            snippet = re.sub(r"(?s)<[^>]*>", " ",
                             str(hit.get("snippet") or "")).strip()
            results.append({
                "identifier": f"wikisource:{title}",
                "title": title,
                "url": f"https://en.wikisource.org/wiki/{page}",
                "description": snippet[:300],
                "edition": "English Wikisource (proofread)",
            })
        return results

    def _search_archive(self, query: str, tradition: str,
                        max_results: int) -> list[dict[str, Any]]:
        url = self._archive_query(query, tradition, max_results)
        self._check_url(url)
        if not self._robots.allowed(url, _USER_AGENT, self._http):
            raise IngestError(url, "robots.txt disallows this search")
        try:
            response = self._http.get(url)
            payload = json.loads(response.text)
        except NoMoralsError as exc:
            raise IngestError(url, f"archive.org search failed: {exc}"
                              ) from exc
        except (ValueError, OSError) as exc:
            raise IngestError(url, f"archive.org search failed: {exc}"
                              ) from exc
        docs = (payload.get("response") or {}).get("docs") or []
        results: list[dict[str, Any]] = []
        for doc in docs:
            identifier = str(doc.get("identifier") or "").strip()
            if not identifier:
                continue
            results.append({
                "identifier": identifier,
                "title": str(doc.get("title") or identifier),
                "url": f"{_ARCHIVE_DETAILS}{identifier}",
                "description": str(doc.get("description") or ""),
            })
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _archive_query(query: str, tradition: str,
                       max_results: int) -> str:
        # Quote multi-word queries as a phrase so archive.org's token OR
        # semantics don't drown exact-title matches in token soup.
        terms = query.strip()
        if " " in terms and not (
                terms.startswith('"') and terms.endswith('"')):
            terms = f'"{terms}"'
        if tradition:
            terms = f"{terms} AND {tradition}"
        params = {
            "q": f'({terms}) AND mediatype:texts',
            "fl[]": list(_SEARCH_FIELDS),
            "rows": str(max_results),
            "output": "json",
        }
        # doseq=True: each field in fl[] becomes its own query param.
        # Without it urlencode stringifies the list into one param
        # (fl[]=%5B%27identifier%27%2C+...%5D), which archive.org ignores —
        # docs come back with no fields and every result is skipped.
        return f"{_ARCHIVE_SEARCH}?{urllib.parse.urlencode(params, doseq=True)}"

    def _search_web(self, query: str, tradition: str,
                    max_results: int) -> list[dict[str, Any]]:
        """Plain web-search fallback via DuckDuckGo lite HTML."""
        terms = f"{query} {tradition} public domain text".strip()
        target = f"{_DDG_LITE}?{urllib.parse.urlencode({'q': terms})}"
        if not self._robots.allowed(target, _USER_AGENT, self._http):
            raise IngestError(target, "robots.txt disallows this search")
        try:
            response = self._http.get(target)
            markup = response.text
        except NoMoralsError as exc:
            raise IngestError(target, f"web search failed: {exc}") from exc
        results: list[dict[str, Any]] = []
        for match in _LINK_RE.finditer(markup):
            href = html.unescape(match.group(1))
            title = re.sub(r"(?s)<[^>]*>", " ", match.group(2)).strip()
            if not href.startswith(("http://", "https://")):
                continue
            identifier = urllib.parse.urlparse(href).netloc or ""
            results.append({
                "identifier": identifier,
                "title": title or href,
                "url": href,
                "description": "",
            })
            if len(results) >= max_results:
                break
        return results


# ── source registry: outcomes, ordering, growth (Pass #94) ───────


def wisdom_registry_for_context(context: Any = None,
                                injected: Any = None) -> Any:
    """The source registry wisdom outcomes land in, or None.

    Mirrors the #88 resolver's / #93 film resolution exactly: an
    injected registry wins; otherwise the shared
    ``DEFAULT_REGISTRY`` bound to the context's KV store — and
    only when the context carries a database, so db-less callers
    keep the pre-#94 behaviour exactly.  Never raises."""
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
            # enhancement; its absence never sinks a hunt
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _desc_for_wisdom_backend(reg: Any, name: str) -> Any:
    """The registry descriptor a hunt backend serves from
    (:data:`WISDOM_SOURCE_DOMAINS`), or None.  Never raises."""
    try:
        domain = WISDOM_SOURCE_DOMAINS.get(name)
        if not domain:
            return None
        return reg.for_domain(domain)
    except Exception:  # noqa: BLE001
        return None


def _wisdom_error_evidence(exc: BaseException) -> tuple[str, str]:
    """Classify one raised backend/fetch error into the registry
    vocabulary: ``(kind, error_class)`` where kind is
    ``rate_limited`` | ``blocked`` | ``failure``.  The backends
    raise :class:`IngestError` with the HTTP layer's error chained
    as the cause, so the status is read off the cause chain via
    :func:`http_status_of` (structured ``details`` first, the
    message's status code as fallback):

    * HTTP 429 → rate-limited (its own class, #90 — a throttled
      moment, never death);
    * HTTP 401/403 → blocked (a wall is evidence about the wall,
      #89); a robots.txt denial is the host's own published wall
      and classifies the same way;
    * any other HTTP status → ``http_<status>``; anything else →
      the root cause's exception type (``network_failure`` is not
      used here: the ingestor's transport errors surface as their
      real types — TimeoutError_, RequestError, OSError).
    """
    try:
        if "robots.txt disallows" in str(exc):
            return "blocked", ""
        status: int | None = None
        cur: BaseException | None = exc
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            status = http_status_of(cur)
            if status is not None:
                break
            cur = cur.__cause__ or cur.__context__
        if status == 429:
            return "rate_limited", ""
        if status in (401, 403):
            return "blocked", ""
        if status is not None:
            return "failure", f"http_{status}"
        root: BaseException = exc
        seen = set()
        while root.__cause__ is not None and id(root) not in seen:
            seen.add(id(root))
            root = root.__cause__
        return "failure", type(root).__name__ or "search_failed"
    except Exception:  # noqa: BLE001
        return "failure", "search_failed"


def _apply_wisdom_evidence(reg: Any, desc: Any, kind: str,
                           error_class: str, latency_s: float) -> None:
    """Land one classified outcome in the registry.  Best-effort;
    never raises — bookkeeping never breaks a hunt."""
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


def _record_wisdom_search(reg: Any, name: str,
                          results: list[dict[str, Any]],
                          latency_s: float,
                          exc: BaseException | None) -> None:
    """Credit/blame one hunt backend attempt.  On a serve, foreign
    hosts among the result URLs intake as UNTRUSTED wisdom
    candidates (the web backend's results genuinely live on
    archives the registry has never seen — sacred-texts and its
    kin; known hosts refuse themselves inside the registry).
    Best-effort; never raises."""
    try:
        if reg is None:
            return
        desc = _desc_for_wisdom_backend(reg, name)
        if desc is None:
            return
        if exc is not None:
            kind, error_class = _wisdom_error_evidence(exc)
            _apply_wisdom_evidence(reg, desc, kind, error_class,
                                   latency_s)
            return
        if results:
            _apply_wisdom_evidence(reg, desc, "served", "", latency_s)
            for cand in results:
                url = str(cand.get("url") or "")
                if url:
                    reg.note_candidate(
                        url, via="wisdom-search-result",
                        kinds=(WISDOM_KIND,))
            return
        _apply_wisdom_evidence(reg, desc, "no_result", "", latency_s)
    except Exception:  # noqa: BLE001
        pass


def _record_wisdom_fetch(reg: Any, url: str, data: bytes | None,
                         latency_s: float,
                         exc: BaseException | None) -> None:
    """Credit/blame one ingest fetch.  The fetched host's
    descriptor takes the outcome when the registry knows the host
    (seed or health-promoted grown archive alike — the fetch path
    is host-agnostic, so a grown archive serves here); a
    successfully served host the registry has never seen intakes
    as an UNTRUSTED wisdom candidate (the manifest and the
    research-fed queue are a genuine discovery flow — they bring
    new archives into the corpus).  A failed fetch of an unknown
    host records nothing: there is no descriptor to blame and a
    failure is not discovery evidence.  Best-effort; never
    raises."""
    try:
        if reg is None:
            return
        desc = reg.for_url(url)
        if exc is not None:
            if desc is not None:
                kind, error_class = _wisdom_error_evidence(exc)
                _apply_wisdom_evidence(reg, desc, kind, error_class,
                                       latency_s)
            return
        if desc is not None:
            _apply_wisdom_evidence(reg, desc, "served", "", latency_s)
        elif data:
            reg.note_candidate(url, via="wisdom-ingest-fetch",
                               kinds=(WISDOM_KIND,))
    except Exception:  # noqa: BLE001
        pass


def _order_wisdom(reg: Any, names: list[str]) -> list[str]:
    """Outcome-aware hunt order (Pass #94), through the registry's
    existing score/health semantics — the #88/#93 discipline:

    * NO learned state anywhere in the hunt → the list is
      returned untouched: day-one behaviour is today's
      :data:`SEARCH_SOURCES` order, byte-for-byte;
    * otherwise active backends sort by registry score (stable —
      ties keep hunt order) and stood-down / disabled / dead
      backends are demoted to the end, in hunt order: they leave
      the active ordering (the #86 vocabulary) but the hunt still
      tries them last rather than pretending they do not exist —
      the attempt log stays truthful.
    Never raises; a registry hiccup returns the hunt as-is."""
    try:
        items = [(n, _desc_for_wisdom_backend(reg, n)) for n in names]
        evidence = False
        for _name, desc in items:
            if desc is None:
                continue
            st = reg.stats_for(desc)
            if st["ok"] or st["fail"] or st["no_result"] \
                    or not reg.is_active(desc):
                evidence = True
                break
        if not evidence:
            return list(names)
        active = [it for it in items
                  if it[1] is None or reg.is_active(it[1])]
        inactive = [it for it in items
                    if it[1] is not None and not reg.is_active(it[1])]
        active.sort(key=lambda it: -reg.score(it[1])
                    if it[1] is not None else 0.0)
        return [n for n, _desc in active + inactive]
    except Exception:  # noqa: BLE001
        return list(names)
