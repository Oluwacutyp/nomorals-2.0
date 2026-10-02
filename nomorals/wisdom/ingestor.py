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

from ..core.errors import NoMoralsError
from ..core.http import HttpClient, HttpResponse
from ..core.logging_setup import get_logger
from ..documents.errors import DocumentError
from ..documents.model import full_text
from ..documents.parsers import parse_bytes
from ..tools.web import RobotsCache
from .corpus import CanonCorpus, ManifestEntry
from .errors import CorpusError, IngestError

__all__ = ["ArchiveIngestor"]

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


class ArchiveIngestor:
    """Fetch, cache, parse, and ingest public-domain esoteric texts."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.corpus = CanonCorpus(context)
        self._http = HttpClient(timeout=30.0, user_agent=_USER_AGENT)
        self._robots = RobotsCache()
        self._blobs = self._wisdom_root() / _BLOBS_DIR
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._index_path = self._blobs / _BLOB_INDEX
        self._index: dict[str, dict[str, Any]] | None = None

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

    def fetch_bytes(self, url: str) -> bytes:
        """GET a URL with retries and blob caching.

        Raises :class:`IngestError` on HTTP errors, robots denials,
        empty bodies, or exhausted retries — never returns empty bytes.
        """
        url = self._check_url(url)
        cached = self._cached_bytes(url)
        if cached is not None:
            return cached
        self._check_robots(url)
        data = self._get_with_retry(url)
        if not data:
            # Defensive: _get_with_retry should already have raised, but a
            # silent empty result must never escape this boundary.
            raise IngestError(url, "fetch returned empty bytes")
        self._store_blob(url, data)
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
        return text

    # ── ingest ────────────────────────────────────────────────────────
    def ingest_entry(self, entry: ManifestEntry) -> ManifestEntry:
        """Fetch, verify, parse, and ingest one manifest entry.

        Idempotent: if the manifest sha matches the fetched bytes and the
        entry is already ingested, the entry is returned unchanged.
        Registers the entry in the manifest first when it is new.
        """
        entry.validate()
        data = self.fetch_bytes(entry.source_url)
        digest = hashlib.sha256(data).hexdigest()
        if entry.sha256 == digest and entry.ingested_at:
            return entry  # already ingested, unchanged bytes
        filename = self._filename_for(entry.source_url, f"{entry.slug}.bin")
        text = self.parse(data, filename)
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
    def search(self, query: str, tradition: str = "",
               max_results: int = 10) -> list[dict[str, Any]]:
        """Find candidate texts in public archives. No auto-ingest.

        Queries archive.org's advanced search first; falls back to a plain
        web search if archive.org is unreachable. Every candidate carries
        identifier/title/url/description; ingestion is a separate step.
        Raises :class:`IngestError` only when both backends fail.
        """
        query = (query or "").strip()
        if not query:
            raise IngestError("<search>", "search needs a non-empty query")
        if max_results < 1:
            raise IngestError("<search>", "max_results must be >= 1")
        try:
            return self._search_archive(query, tradition, max_results)
        except IngestError as archive_exc:
            # archive.org unreachable — fall through to the plain web
            # fallback below; the failure is recorded on the exception chain.
            _log.debug("archive.org search failed, trying web fallback: %s",
                       archive_exc)
        try:
            return self._search_web(query, tradition, max_results)
        except IngestError as web_exc:
            raise IngestError(
                query, f"archive.org and web search both failed: {web_exc}"
            ) from web_exc

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
        terms = query
        if tradition:
            terms = f"{terms} AND {tradition}"
        params = {
            "q": f'({terms}) AND mediatype:texts',
            "fl[]": list(_SEARCH_FIELDS),
            "rows": str(max_results),
            "output": "json",
        }
        return f"{_ARCHIVE_SEARCH}?{urllib.parse.urlencode(params)}"

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
