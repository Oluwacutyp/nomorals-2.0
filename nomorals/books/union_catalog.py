"""Union catalogue over many libraries (52-websites #33, Anna's Archive).

The pattern, not the corpus: Anna's Archive hosts no book files itself —
it aggregates and normalises metadata from many source libraries,
deduplicates records into one catalogue, tracks which source holds which
file, and links out. Devon already runs the organs that pattern maps
onto (``books/library.py`` catalogue, ``documents/index.py`` FTS5 index,
``search/federated.py`` fan-out); what she did not have is a *union
catalogue with provenance*: one record per work, merged from many
sources, each holding attributed to its source with a health state.
This module is that layer.

Data shape (per the report): ``WorkRecord{canonical_id, title, authors,
identifiers (isbn/doi/oclc...), holdings[{source_id, source_ref, format,
health}]}``. The merge/reconciliation pass keys on identifiers first,
then fuzzy title+author; every merge is reversible and provenance is
never discarded — a merged-away record's full snapshot rides inside
its :class:`MergeRecord`, so ``unmerge`` restores it whole.

Honest limits, hard: this indexes METADATA and LAWFUL holdings only —
the owner's own library first, then open collections (Open Library,
Internet Archive, Project Gutenberg) via the licence-labelled
discipline: no source without a legal basis gets a row (the
``LAWFUL_SOURCES`` table carries each source's licence + basis, and
``register_source`` refuses a row without one). This is also the home
for the #3 Sci-Hub, #9 LibGen and #11 PDFDrive folds — those are folds
into this catalogue, not separate builds, and they fold in as lawful
holdings only: a shadow-library corpus is never indexed here. The
engineering worth porting is the union + reconciliation + bulk-export
shape, not the corpus.

Storage: the structured catalogue is durable JSON at
``<home>/data/union_catalog.json`` (atomic writes; a corrupt store is
quarantined aside and re-seeded, never silently emptied). The search
leg runs over :class:`~nomorals.documents.index.DocumentIndex` — the
FTS5 BM25 index from the documents organ — rebuilt in memory from the
store, so a catalogue query is a BM25 query over title + authors +
identifiers. Bulk export is the standardised container shape: one
``WorkRecord`` per line, line-delimited JSON (AAC's shape); zstd
compression rides along only when the ``zstandard`` module is actually
importable on the host, and the receipt says so honestly when it is not.

Health vocabulary follows :mod:`nomorals.agency.sourcehealth`:
``live`` / ``dead`` / ``bot-blocked`` / ``unverifiable`` / ``unknown``.
A dead source's holdings degrade in ranking — they sort last and are
labelled unavailable — but they never silently vanish.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..documents.index import DocumentIndex
from ..documents.model import Document, Section

__all__ = [
    "Candidate",
    "Holding",
    "LAWFUL_SOURCES",
    "MergeRecord",
    "UnionCatalog",
    "UnionCatalogStore",
    "WorkRecord",
    "default_store_path",
    "format_catalog_reply",
    "format_work",
    "parse_catalog_request",
    "register_source",
]

_log = get_logger(__name__)

#: A live search leg waits at most this long for one source.
_LIVE_TIMEOUT_S = 8.0
#: A probe query a source must answer to count as live.
_PROBE_QUERY = "pride and prejudice"
#: Fuzzy title+author merge needs at least this SequenceMatcher ratio.
_FUZZY_TITLE_RATIO = 0.88
#: Health states — the sourcehealth vocabulary, kept in sync by word.
HEALTH_LIVE = "live"
HEALTH_DEAD = "dead"
HEALTH_BOT_BLOCKED = "bot-blocked"
HEALTH_UNVERIFIABLE = "unverifiable"
HEALTH_UNKNOWN = "unknown"
_HEALTH_STATES = (
    HEALTH_LIVE, HEALTH_DEAD, HEALTH_BOT_BLOCKED,
    HEALTH_UNVERIFIABLE, HEALTH_UNKNOWN,
)

#: Holding score in ranking: live holdings lead, unknown next, dead and
#: walled holdings sink to the bottom but stay listed.
_HOLDING_SCORE = {
    HEALTH_LIVE: 2.0,
    HEALTH_UNKNOWN: 1.0,
    HEALTH_DEAD: 0.25,
    HEALTH_BOT_BLOCKED: 0.25,
    HEALTH_UNVERIFIABLE: 0.25,
}

#: The lawful-holdings discipline: every source row carries its licence
#: and its legal basis. No source without a legal basis gets a row —
#: ``register_source`` refuses one, and the live legs only ever query
#: rows in this table. The owner's own library is first, always.
LAWFUL_SOURCES: dict[str, dict[str, str]] = {
    "owner": {
        "label": "Your library",
        "licence": "owner's own books",
        "legal_basis": "owner",
        "health_url": "",
    },
    "openlibrary": {
        "label": "Open Library",
        "licence": "public-domain + open ebooks (openlibrary.org)",
        "legal_basis": "open collection",
        "health_url": "https://openlibrary.org/search.json?q=pride+and+prejudice&limit=1",
    },
    "archiveorg": {
        "label": "Internet Archive",
        "licence": "public-domain / lending (archive.org)",
        "legal_basis": "open collection",
        "health_url": ("https://archive.org/advancedsearch.php?q=title%3A"
                       "%28pride+and+prejudice%29+AND+mediatype%3A%28texts"
                       "%29&fl%5B%5D=identifier&rows=1&output=json"),
    },
    "gutenberg": {
        "label": "Project Gutenberg",
        "licence": "Public domain (US)",
        "legal_basis": "open collection",
        "health_url": "https://gutendex.com/books?search=pride+and+prejudice&page_size=1",
    },
}

#: Live-search endpoints, keyless JSON. Each leg returns candidates with
#: identifiers the merge pass can key on — a leg that cannot carry
#: identifiers cannot merge, so it does not exist here.
_OPENLIBRARY_SEARCH = "https://openlibrary.org/search.json?q={q}&limit={n}"
_ARCHIVE_SEARCH = (
    "https://archive.org/advancedsearch.php?q="
    "title%3A%28{q}%29+AND+mediatype%3A%28texts%29"
    "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=creator"
    "&fl%5B%5D=date&rows={n}&output=json"
)
_GUTENDEX_SEARCH = "https://gutendex.com/books?search={q}&page_size={n}"

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ISBN_RE = re.compile(r"[^0-9xX]")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def default_store_path(home: str | None = None) -> str:
    """Durable catalogue path for a home (``<home>/data/...``)."""
    base = Path(home) if home else Path.home() / ".nomorals"
    return str(base / "data" / "union_catalog.json")


def register_source(source_id: str, *, label: str, licence: str,
                    legal_basis: str, health_url: str = "") -> dict[str, str]:
    """Add a lawful source row. A row without a legal basis is refused.

    The registry grows by discovery; the discipline does not move:
    ``legal_basis`` must name why this source may be indexed (``owner``,
    ``open collection``, ``publisher OA`` …). Returns the row.
    """
    source_id = (source_id or "").strip().lower()
    if not source_id:
        raise ValueError("register_source needs a source id")
    if not (legal_basis or "").strip():
        raise ValueError(
            f"refusing source {source_id!r}: no legal basis — "
            "no source without a legal basis gets a row")
    row = {
        "label": (label or source_id).strip(),
        "licence": (licence or "").strip(),
        "legal_basis": legal_basis.strip(),
        "health_url": (health_url or "").strip(),
    }
    LAWFUL_SOURCES[source_id] = row
    return row


# ── normalisation ────────────────────────────────────────────────────────


def _norm_title(title: str) -> str:
    """Title merge key: lowercase, no leading article, punctuation out."""
    text = (title or "").strip().lower()
    text = _ARTICLE_RE.sub("", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def _norm_author(author: str) -> str:
    """Author surname, lowercase, punctuation out — the fuzzy-match key."""
    text = (author or "").strip().lower()
    if "," in text:  # "Austen, Jane" → surname first
        text = text.split(",", 1)[0]
    else:
        parts = text.split()
        text = parts[-1] if parts else ""
    return _NON_ALNUM_RE.sub("", text)


def _isbn13(isbn: str) -> str:
    """Normalise an ISBN to ISBN-13 digits, converting ISBN-10 properly.

    Returns "" when the value is not an ISBN at all.
    """
    digits = _ISBN_RE.sub("", (isbn or "").upper())
    if len(digits) == 13 and digits.isdigit():
        return digits
    if len(digits) == 10 and digits[:9].isdigit() and (
            digits[9].isdigit() or digits[9] == "X"):
        body = "978" + digits[:9]
        total = sum((1 if i % 2 == 0 else 3) * int(ch)
                    for i, ch in enumerate(body))
        check = (10 - total % 10) % 10
        return body + str(check)
    return ""


def _norm_doi(doi: str) -> str:
    text = (doi or "").strip().lower()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text if _DOI_RE.match(text) else ""


def _norm_oclc(oclc: str) -> str:
    digits = re.sub(r"\D", "", oclc or "")
    return digits


def _slug(text: str) -> str:
    return _NON_ALNUM_RE.sub("-", _norm_title(text)).strip("-") or "untitled"


# ── records ──────────────────────────────────────────────────────────────


@dataclass
class Holding:
    """One source's claim to hold a work — provenance, never discarded."""

    source_id: str
    source_ref: str  # URL, or the owner's library slug
    format: str = ""
    health: str = HEALTH_UNKNOWN
    fetched_at: float = 0.0
    title_as_listed: str = ""
    authors_as_listed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Holding:
        return cls(
            source_id=str(data.get("source_id", "")),
            source_ref=str(data.get("source_ref", "")),
            format=str(data.get("format", "") or ""),
            health=str(data.get("health", HEALTH_UNKNOWN) or HEALTH_UNKNOWN),
            fetched_at=float(data.get("fetched_at", 0.0) or 0.0),
            title_as_listed=str(data.get("title_as_listed", "") or ""),
            authors_as_listed=[str(a) for a in
                               (data.get("authors_as_listed") or [])],
        )


@dataclass
class WorkRecord:
    """One work in the union catalogue: merged from many sources."""

    canonical_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    identifiers: dict[str, str] = field(default_factory=dict)
    holdings: list[Holding] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["holdings"] = [h.to_dict() for h in self.holdings]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkRecord:
        return cls(
            canonical_id=str(data.get("canonical_id", "")),
            title=str(data.get("title", "") or ""),
            authors=[str(a) for a in (data.get("authors") or [])],
            identifiers={str(k): str(v) for k, v in
                         (data.get("identifiers") or {}).items()},
            holdings=[Holding.from_dict(h) for h in
                      (data.get("holdings") or [])],
            created_at=float(data.get("created_at", 0.0) or 0.0),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
        )

    def live_holdings(self) -> list[Holding]:
        """Holdings whose source is live right now."""
        return [h for h in self.holdings if h.health == HEALTH_LIVE]

    def rank_score(self) -> float:
        """Dead holdings degrade the score; they never vanish."""
        return sum(_HOLDING_SCORE.get(h.health, 1.0)
                   for h in self.holdings)


@dataclass
class MergeRecord:
    """One reversible merge: the merged-away record rides along whole."""

    canonical_id: str  # the surviving work
    merged_id: str  # the work folded into it
    reason: str  # "identifier:<kind>" or "fuzzy:title+author:<ratio>"
    at: float = 0.0
    snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeRecord:
        return cls(
            canonical_id=str(data.get("canonical_id", "")),
            merged_id=str(data.get("merged_id", "")),
            reason=str(data.get("reason", "")),
            at=float(data.get("at", 0.0) or 0.0),
            snapshot=dict(data.get("snapshot") or {}),
        )


@dataclass
class Candidate:
    """One raw record from one source, before the merge pass."""

    title: str
    authors: list[str] = field(default_factory=list)
    identifiers: dict[str, str] = field(default_factory=dict)
    source_id: str = ""
    source_ref: str = ""
    format: str = ""


# ── durable store ────────────────────────────────────────────────────────


class UnionCatalogStore:
    """Durable JSON catalogue: works + merges + per-source health.

    Seeds empty when missing; a corrupt store is quarantined aside
    (``<path>.corrupt-<epoch>``) and re-seeded, never silently emptied.
    Writes are atomic (tmp + replace).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path(default_store_path())
        self._works: dict[str, WorkRecord] = {}
        self._merges: list[MergeRecord] = []
        self._health: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("catalogue root is not an object")
            for wid, data in (raw.get("works") or {}).items():
                self._works[str(wid)] = WorkRecord.from_dict(data)
            self._merges = [MergeRecord.from_dict(m)
                            for m in (raw.get("merges") or [])]
            self._health = {str(k): dict(v) for k, v in
                            (raw.get("source_health") or {}).items()}
        except (OSError, ValueError, TypeError) as exc:
            _log.warning("union catalogue corrupt at %s: %s — "
                         "quarantining and re-seeding", self.path, exc)
            try:
                stamp = int(time.time())
                self.path.replace(
                    self.path.with_name(
                        f"{self.path.name}.corrupt-{stamp}"))
            except OSError:
                _log.debug("could not quarantine corrupt catalogue",
                           exc_info=True)
            self._works, self._merges, self._health = {}, [], {}

    def save(self) -> None:
        """Atomic save of works + merges + source health."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "works": {wid: w.to_dict() for wid, w in self._works.items()},
            "merges": [m.to_dict() for m in self._merges],
            "source_health": self._health,
        }
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # -- works --------------------------------------------------------------

    def get(self, canonical_id: str) -> WorkRecord | None:
        return self._works.get(canonical_id)

    def list(self) -> list[WorkRecord]:
        return list(self._works.values())

    def put(self, work: WorkRecord) -> None:
        self._works[work.canonical_id] = work

    def drop(self, canonical_id: str) -> None:
        self._works.pop(canonical_id, None)

    # -- merges -------------------------------------------------------------

    def record_merge(self, merge: MergeRecord) -> None:
        self._merges.append(merge)

    def merges(self) -> list[MergeRecord]:
        return list(self._merges)

    def merges_for(self, canonical_id: str) -> list[MergeRecord]:
        return [m for m in self._merges
                if m.canonical_id == canonical_id]

    def forget_merge(self, canonical_id: str, merged_id: str) -> None:
        self._merges = [m for m in self._merges
                        if not (m.canonical_id == canonical_id
                                and m.merged_id == merged_id)]

    # -- source health ------------------------------------------------------

    def health(self, source_id: str) -> dict[str, Any]:
        return dict(self._health.get(source_id) or {})

    def record_health(self, source_id: str, status: str, *,
                      error: str = "") -> None:
        row = self._health.get(source_id) or {
            "status": HEALTH_UNKNOWN, "last_checked": 0.0,
            "last_ok": 0.0, "fail_count": 0, "last_error": "",
        }
        now = time.time()
        row["status"] = status if status in _HEALTH_STATES else HEALTH_UNKNOWN
        row["last_checked"] = now
        if status == HEALTH_LIVE:
            row["last_ok"] = now
            row["fail_count"] = 0
            row["last_error"] = ""
        else:
            row["fail_count"] = int(row.get("fail_count", 0) or 0) + 1
            row["last_error"] = error
        self._health[source_id] = row


# ── live legs (keyless JSON, never raise) ──────────────────────────────────


def _fetch_json(url: str, fetcher: Callable[..., Any] | None,
                timeout: float = _LIVE_TIMEOUT_S) -> dict[str, Any]:
    """One bounded JSON GET. An injected fetcher wins (tests); else urllib.

    Returns {} on any failure — a dead leg answers empty, never raises.
    """
    try:
        if fetcher is not None:
            data = fetcher(url, timeout=timeout)
            if isinstance(data, dict):
                return data
            if isinstance(data, (bytes, str)):
                return json.loads(data.decode("utf-8", "replace")
                                  if isinstance(data, bytes) else data)
            return {}
        req = urllib.request.Request(url, headers={
            "User-Agent": "Devon-UnionCatalog/1.0 (open-collections reader)",
            "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — a dead leg answers empty
        _log.debug("union catalogue fetch failed: %s", url, exc_info=True)
        return {}


def _classify_fetch(data: dict[str, Any]) -> tuple[str, str]:
    """Health verdict for one fetch: live / bot-blocked / dead / unverifiable.

    A search that returns zero hits is still a live source — zero
    results for one query is not a dead source. Only the transport
    outcome classifies here; the probe path (refresh_health) is the
    one that requires a hit.
    """
    if data:
        return HEALTH_LIVE, ""
    # Empty payload: the evidence is about the network, not the
    # source — an empty dict here is unverifiable, not dead.
    return HEALTH_UNVERIFIABLE, "empty response"


def search_openlibrary(query: str, *, limit: int = 5,
                       fetcher: Callable[..., Any] | None = None
                       ) -> tuple[list[Candidate], str, str]:
    """Open Library search API: public-domain + open ebooks, keyless."""
    query = (query or "").strip()
    if not query:
        return [], HEALTH_UNKNOWN, ""
    url = _OPENLIBRARY_SEARCH.format(
        q=urllib.parse.quote_plus(query), n=max(limit, 0))
    data = _fetch_json(url, fetcher)
    status, error = _classify_fetch(data)
    out: list[Candidate] = []
    for doc in (data.get("docs") or []):
        title = str(doc.get("title", "") or "").strip()
        key = str(doc.get("key", "") or "").strip()
        if not title or not key.startswith("/works/"):
            continue
        identifiers: dict[str, str] = {"openlibrary": key}
        for raw_isbn in (doc.get("isbn") or [])[:4]:
            isbn13 = _isbn13(str(raw_isbn))
            if isbn13:
                identifiers["isbn"] = isbn13
                break
        out.append(Candidate(
            title=title,
            authors=[str(a) for a in (doc.get("author_name") or [])],
            identifiers=identifiers,
            source_id="openlibrary",
            source_ref=f"https://openlibrary.org{key}",
            format="ebook",
        ))
        if len(out) >= max(limit, 0):
            break
    return out, status, error


def search_archiveorg(query: str, *, limit: int = 5,
                      fetcher: Callable[..., Any] | None = None
                      ) -> tuple[list[Candidate], str, str]:
    """Archive.org advanced search: identifier + title, keyless JSON."""
    query = (query or "").strip()
    if not query:
        return [], HEALTH_UNKNOWN, ""
    url = _ARCHIVE_SEARCH.format(
        q=urllib.parse.quote_plus(query), n=max(limit, 0))
    data = _fetch_json(url, fetcher)
    status, error = _classify_fetch(data)
    out: list[Candidate] = []
    for doc in ((data.get("response") or {}).get("docs") or []):
        ident = str(doc.get("identifier", "") or "").strip()
        title = str(doc.get("title", "") or "").strip()
        if not ident or not title:
            continue
        creator = str(doc.get("creator", "") or "").strip()
        out.append(Candidate(
            title=title,
            authors=[creator] if creator else [],
            identifiers={"archive": ident},
            source_id="archiveorg",
            source_ref=f"https://archive.org/details/{ident}",
            format="ebook (Archive.org)",
        ))
        if len(out) >= max(limit, 0):
            break
    return out, status, error


def _gutenberg_formats(formats: dict[str, Any]) -> str:
    """Friendly format label for a Gutendex formats map."""
    labels = []
    keys = list((formats or {}).keys())
    for prefix, label in (
            ("application/epub+zip", "EPUB"),
            ("text/plain", "Plain text"),
            ("application/x-mobipocket-ebook", "Kindle"),
            ("application/pdf", "PDF"),
            ("text/html", "HTML")):
        if any(k.startswith(prefix) for k in keys):
            labels.append(label)
    return ", ".join(labels) or "ebook"


def search_gutenberg(query: str, *, limit: int = 5,
                     fetcher: Callable[..., Any] | None = None
                     ) -> tuple[list[Candidate], str, str]:
    """Project Gutenberg via Gutendex: public domain, keyless JSON."""
    query = (query or "").strip()
    if not query:
        return [], HEALTH_UNKNOWN, ""
    url = _GUTENDEX_SEARCH.format(
        q=urllib.parse.quote_plus(query), n=max(limit, 0))
    data = _fetch_json(url, fetcher)
    status, error = _classify_fetch(data)
    out: list[Candidate] = []
    for book in (data.get("results") or []):
        title = str(book.get("title", "") or "").strip()
        book_id = book.get("id")
        if not title or book_id is None:
            continue
        authors = [str(a.get("name", "") or "").strip()
                   for a in (book.get("authors") or [])
                   if str(a.get("name", "") or "").strip()]
        out.append(Candidate(
            title=title,
            authors=authors,
            identifiers={"gutenberg": str(book_id)},
            source_id="gutenberg",
            source_ref=f"https://www.gutenberg.org/ebooks/{book_id}",
            format=_gutenberg_formats(book.get("formats") or {}),
        ))
        if len(out) >= max(limit, 0):
            break
    return out, status, error


#: source_id → live leg, in fan-out order (owner's library is local,
#: never a leg here).
LIVE_LEGS: dict[str, Callable[..., tuple[list[Candidate], str, str]]] = {
    "openlibrary": search_openlibrary,
    "archiveorg": search_archiveorg,
    "gutenberg": search_gutenberg,
}


# ── the catalogue ────────────────────────────────────────────────────────


class UnionCatalog:
    """One record per work, merged from many sources, provenance kept.

    ``library`` is the owner's :class:`nomorals.books.library.Library`
    (injected; the tool/hook builds it from the brain context) — the
    owner's library is always searched first and its holdings always
    rank first. ``fetcher`` is the injected transport for the live
    legs (tests); production passes None and the legs use urllib.
    """

    def __init__(self, home: str | None = None, *,
                 store: UnionCatalogStore | None = None,
                 library: Any = None,
                 fetcher: Callable[..., Any] | None = None) -> None:
        self.home = home
        self.store = store or UnionCatalogStore(default_store_path(home))
        self._library = library
        self._fetcher = fetcher
        self._index = DocumentIndex()
        self._rebuild_index()

    # -- index over documents/index.py storage -----------------------------

    def _rebuild_index(self) -> None:
        self._index = DocumentIndex()
        for work in self.store.list():
            text = " ".join(
                [work.title, " ".join(work.authors)]
                + [f"{k}:{v}" for k, v in work.identifiers.items()])
            try:
                self._index.add(Document(
                    id=work.canonical_id, title=work.title,
                    author=", ".join(work.authors),
                    sections=[Section(text=text)]))
            except Exception:  # noqa: BLE001 — one bad doc never kills search
                _log.debug("union catalogue index skip %s",
                           work.canonical_id, exc_info=True)

    def _index_search(self, query: str, limit: int) -> list[WorkRecord]:
        try:
            hits = self._index.search(query, limit=limit)
        except Exception:  # noqa: BLE001 — index failure degrades to a scan
            _log.debug("union catalogue index search failed", exc_info=True)
            return self._scan_search(query, limit)
        out: list[WorkRecord] = []
        for hit in hits:
            work = self.store.get(str(hit.get("doc_id", "")))
            if work is not None:
                out.append(work)
        return out

    def _scan_search(self, query: str, limit: int) -> list[WorkRecord]:
        """Token scan fallback when the FTS index cannot answer."""
        tokens = [t for t in _NON_ALNUM_RE.split(query.lower()) if len(t) >= 2]
        if not tokens:
            return []
        scored: list[tuple[int, WorkRecord]] = []
        for work in self.store.list():
            hay = _norm_title(work.title) + " " + " ".join(
                _norm_author(a) for a in work.authors)
            hits = sum(1 for t in tokens if t in hay)
            if hits:
                scored.append((hits, work))
        scored.sort(key=lambda pair: (-pair[0], pair[1].title))
        return [w for _, w in scored[:limit]]

    # -- merge / reconciliation ---------------------------------------------

    @staticmethod
    def _canonical_id(candidate: Candidate) -> str:
        """Deterministic id: strongest identifier wins, else title slug."""
        identifiers = candidate.identifiers
        if identifiers.get("isbn"):
            return f"work:isbn:{identifiers['isbn']}"
        if identifiers.get("doi"):
            digest = hashlib.sha256(
                identifiers["doi"].encode("utf-8")).hexdigest()[:12]
            return f"work:doi:{digest}"
        if identifiers.get("openlibrary"):
            key = identifiers["openlibrary"].strip("/").replace("/", "-")
            return f"work:{key}"
        if identifiers.get("gutenberg"):
            return f"work:gutenberg:{identifiers['gutenberg']}"
        if identifiers.get("archive"):
            return f"work:archive:{identifiers['archive']}"
        if identifiers.get("oclc"):
            return f"work:oclc:{identifiers['oclc']}"
        author = _norm_author(candidate.authors[0]) if candidate.authors else ""
        return f"work:title:{_slug(candidate.title)}" + (
            f"-{author}" if author else "")

    def _find_by_identifier(self, candidate: Candidate,
                            ) -> tuple[WorkRecord | None, str]:
        """Identifier-keyed match. Returns (work, identifier kind)."""
        mine = candidate.identifiers
        if not mine:
            return None, ""
        for work in self.store.list():
            theirs = work.identifiers
            for kind in ("isbn", "doi", "oclc", "openlibrary",
                         "gutenberg", "archive"):
                if mine.get(kind) and theirs.get(kind) == mine[kind]:
                    return work, kind
        return None, ""

    def _find_fuzzy(self, candidate: Candidate,
                    ) -> tuple[WorkRecord | None, float]:
        """Fuzzy title+author match. Returns (work, ratio)."""
        title_key = _norm_title(candidate.title)
        if not title_key:
            return None, 0.0
        mine_authors = {_norm_author(a) for a in candidate.authors
                        if _norm_author(a)}
        best: WorkRecord | None = None
        best_ratio = 0.0
        for work in self.store.list():
            ratio = difflib.SequenceMatcher(
                None, title_key, _norm_title(work.title)).ratio()
            if ratio < _FUZZY_TITLE_RATIO:
                continue
            theirs = {_norm_author(a) for a in work.authors
                      if _norm_author(a)}
            if mine_authors and theirs and not (mine_authors & theirs):
                continue  # same-ish title, different author: not a merge
            if ratio > best_ratio:
                best, best_ratio = work, ratio
        return best, best_ratio

    def ingest(self, candidates: list[Candidate], source_id: str,
               *, health: str = HEALTH_UNKNOWN) -> dict[str, int]:
        """Fold raw records from one source into the catalogue.

        Identifier-keyed merge first, then fuzzy title+author; every
        merge records a reversible :class:`MergeRecord`. A holding that
        already exists (same source + ref) is not duplicated. Returns
        ``{created, merged, holdings_added}``.
        """
        created = merged = holdings_added = 0
        now = time.time()
        for cand in candidates:
            if not _norm_title(cand.title):
                continue
            cand.source_id = source_id
            work, kind = self._find_by_identifier(cand)
            reason = ""
            if work is not None:
                reason = f"identifier:{kind}"
            else:
                work, ratio = self._find_fuzzy(cand)
                if work is not None:
                    reason = f"fuzzy:title+author:{ratio:.2f}"
            if work is None:
                work = WorkRecord(
                    canonical_id=self._canonical_id(cand),
                    title=cand.title.strip(),
                    authors=list(dict.fromkeys(
                        a.strip() for a in cand.authors if a.strip())),
                    identifiers=dict(cand.identifiers),
                    created_at=now, updated_at=now)
                self.store.put(work)
                created += 1
            else:
                # A merge folds the candidate's identity into the record;
                # the folded-away identity is recorded, reversibly — the
                # snapshot carries the candidate's own holding, so an
                # unmerge restores a whole record, not a bare title.
                if self._canonical_id(cand) != work.canonical_id:
                    snapshot = WorkRecord(
                        canonical_id=self._canonical_id(cand),
                        title=cand.title.strip(),
                        authors=list(cand.authors),
                        identifiers=dict(cand.identifiers),
                        holdings=[Holding(
                            source_id=source_id,
                            source_ref=cand.source_ref,
                            format=cand.format, health=health,
                            fetched_at=now,
                            title_as_listed=cand.title.strip(),
                            authors_as_listed=list(cand.authors))],
                        created_at=now, updated_at=now).to_dict()
                    self.store.record_merge(MergeRecord(
                        canonical_id=work.canonical_id,
                        merged_id=self._canonical_id(cand),
                        reason=reason, at=now, snapshot=snapshot))
                    merged += 1
                for key, value in cand.identifiers.items():
                    work.identifiers.setdefault(key, value)
                for author in cand.authors:
                    if author.strip() and author.strip() not in work.authors:
                        work.authors.append(author.strip())
            if not any(h.source_id == source_id
                       and h.source_ref == cand.source_ref
                       for h in work.holdings):
                work.holdings.append(Holding(
                    source_id=source_id, source_ref=cand.source_ref,
                    format=cand.format, health=health, fetched_at=now,
                    title_as_listed=cand.title.strip(),
                    authors_as_listed=list(cand.authors)))
                holdings_added += 1
            work.updated_at = now
            self.store.put(work)
        if created or merged or holdings_added:
            # The holding health is an observation about the source:
            # record it, so status() tells the truth after any ingest.
            if health != HEALTH_UNKNOWN:
                self.store.record_health(source_id, health)
            self.store.save()
            self._rebuild_index()
        return {"created": created, "merged": merged,
                "holdings_added": holdings_added}

    def unmerge(self, canonical_id: str, merged_id: str) -> bool:
        """Reverse one merge: the folded-away record stands alone again.

        The snapshot inside the :class:`MergeRecord` is the full
        pre-merge record, so provenance survives the round trip: the
        reinstated work keeps its own holdings, and the surviving work
        keeps everything else.
        """
        merge = next(
            (m for m in self.store.merges()
             if m.canonical_id == canonical_id and m.merged_id == merged_id),
            None)
        if merge is None or not merge.snapshot:
            return False
        survivor = self.store.get(canonical_id)
        if survivor is None:
            return False
        restored = WorkRecord.from_dict(merge.snapshot)
        # The restored work's holdings are its own; drop them from the
        # survivor so the same holding is never listed twice.
        restored_refs = {(h.source_id, h.source_ref)
                         for h in restored.holdings}
        survivor.holdings = [h for h in survivor.holdings
                             if (h.source_id, h.source_ref)
                             not in restored_refs]
        survivor.updated_at = time.time()
        self.store.put(survivor)
        self.store.put(restored)
        self.store.forget_merge(canonical_id, merged_id)
        self.store.save()
        self._rebuild_index()
        return True

    # -- owner library leg ---------------------------------------------------

    def _owner_candidates(self, query: str) -> list[Candidate]:
        """The owner's own books, matched by title/author tokens."""
        if self._library is None:
            return []
        try:
            books = self._library.list_books()
        except Exception:  # noqa: BLE001 — a broken library leg degrades
            _log.debug("union catalogue owner leg failed", exc_info=True)
            return []
        tokens = [t for t in _NON_ALNUM_RE.split(query.lower()) if len(t) >= 3]
        if not tokens:
            return []
        out: list[Candidate] = []
        for book in books or []:
            title = str(book.get("title", "") or "").strip()
            author = str(book.get("author", "") or "").strip()
            hay = _norm_title(title) + " " + _norm_author(author)
            if not title or not all(t in hay for t in tokens):
                continue
            out.append(Candidate(
                title=title, authors=[author] if author else [],
                identifiers={},
                source_id="owner",
                source_ref=str(book.get("slug", "") or title),
                format="your library"))
        return out

    # -- live legs ------------------------------------------------------------

    def _live_candidates(self, query: str, *,
                         limit: int = 5) -> list[Candidate]:
        """Fan out to the lawful live legs; one dead leg never kills search."""
        out: list[Candidate] = []
        for source_id, leg in LIVE_LEGS.items():
            try:
                found, status, error = leg(query, limit=limit,
                                           fetcher=self._fetcher)
            except Exception:  # noqa: BLE001 — a dead leg answers empty
                _log.debug("union catalogue leg %s failed", source_id,
                           exc_info=True)
                found, status, error = [], HEALTH_UNVERIFIABLE, "raised"
            self.store.record_health(source_id, status, error=error)
            for cand in found:
                out.append(cand)
        if out:
            self.store.save()
        return out

    # -- search ----------------------------------------------------------------

    def find(self, query: str, *, live: bool = True,
             limit: int = 8) -> list[WorkRecord]:
        """Find a work, any format: owner's library first, then the catalogue,
        then the live open collections (ingested as they answer).

        Returns ranked works: live holdings first, dead holdings last —
        degraded, never silently dropped.
        """
        query = (query or "").strip()
        if not query:
            return []
        seen: dict[str, WorkRecord] = {}
        for work in self._index_search(query, limit):
            seen[work.canonical_id] = work
        owner = self._owner_candidates(query)
        if owner:
            self.ingest(owner, "owner", health=HEALTH_LIVE)
            for work in self._index_search(query, limit):
                seen[work.canonical_id] = work
        if live:
            found = self._live_candidates(query, limit=5)
            if found:
                by_source: dict[str, list[Candidate]] = {}
                for cand in found:
                    by_source.setdefault(cand.source_id, []).append(cand)
                for source_id, cands in by_source.items():
                    health = (self.store.health(source_id).get("status")
                              or HEALTH_UNKNOWN)
                    self.ingest(cands, source_id, health=health)
                for work in self._index_search(query, limit):
                    seen[work.canonical_id] = work
        ranked = sorted(seen.values(),
                        key=lambda w: (-w.rank_score(),
                                       -len(w.holdings), w.title.lower()))
        return ranked[:max(limit, 0)]

    def get(self, canonical_id: str) -> WorkRecord | None:
        return self.store.get(canonical_id)

    def list_works(self, *, limit: int = 20) -> list[WorkRecord]:
        works = sorted(self.store.list(),
                       key=lambda w: (-w.rank_score(), w.title.lower()))
        return works[:max(limit, 0)]

    # -- health ------------------------------------------------------------------

    def refresh_health(self, *, prober: Callable[[str], dict[str, Any]]
                       | None = None) -> dict[str, dict[str, Any]]:
        """Re-check every lawful live source with a probe it must answer.

        A probe is a real query for a work the source must hold
        (``_PROBE_QUERY``); a source that answers it with ≥1 hit is
        live, one that answers empty is dead, a wall is bot-blocked,
        a transport failure is unverifiable. Never raises.
        """
        results: dict[str, dict[str, Any]] = {}
        for source_id, row in LAWFUL_SOURCES.items():
            if source_id == "owner" or not row.get("health_url"):
                continue
            leg = LIVE_LEGS.get(source_id)
            try:
                if prober is not None:
                    probe = prober(source_id)
                    hits = int(probe.get("hits", 0) or 0)
                    status = (HEALTH_LIVE if hits > 0 else HEALTH_DEAD)
                    error = str(probe.get("error", "") or "")
                elif leg is not None:
                    found, status, error = leg(
                        _PROBE_QUERY, limit=1, fetcher=self._fetcher)
                    if status == HEALTH_LIVE and not found:
                        status, error = HEALTH_DEAD, "probe returned no hits"
                else:
                    status, error = HEALTH_UNKNOWN, "no live leg"
            except Exception as exc:  # noqa: BLE001 — health never raises
                status, error = HEALTH_UNVERIFIABLE, str(exc)
            self.store.record_health(source_id, status, error=error)
            results[source_id] = self.store.health(source_id)
        self.store.save()
        return results

    def status(self) -> dict[str, Any]:
        """Catalogue size + per-source health, for the tool/hook."""
        return {
            "works": len(self.store.list()),
            "holdings": sum(len(w.holdings) for w in self.store.list()),
            "merges": len(self.store.merges()),
            "sources": {
                sid: {
                    "label": row.get("label", sid),
                    "licence": row.get("licence", ""),
                    "health": (self.store.health(sid).get("status")
                               or HEALTH_UNKNOWN),
                }
                for sid, row in LAWFUL_SOURCES.items()
            },
        }

    # -- bulk export ---------------------------------------------------------------

    def export_aac(self, path: str | Path) -> dict[str, Any]:
        """Bulk export: one WorkRecord per line, line-delimited JSON.

        The AAC container shape (line-delimited JSON, zstd-compressed
        when the host can). ``zstandard`` is not installed on this box,
        so the export lands as plain ``.jsonl`` and the receipt says
        so — the container is still the standard shape, and a host
        with zstandard gets the ``.zst`` beside it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        works = sorted(self.store.list(), key=lambda w: w.canonical_id)
        lines = [json.dumps(w.to_dict(), sort_keys=True) for w in works]
        path.write_text("\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8")
        compressed = None
        try:
            import zstandard  # type: ignore[import-untyped]

            zpath = path.with_suffix(path.suffix + ".zst")
            cctx = zstandard.ZstdCompressor()
            zpath.write_bytes(cctx.compress(
                ("\n".join(lines) + "\n").encode("utf-8")))
            compressed = str(zpath)
        except ImportError:
            compressed = None
        return {
            "path": str(path),
            "compressed": compressed,
            "works": len(works),
            "zstd": compressed is not None,
            "note": ("zstd compression unavailable on this host "
                     "(no zstandard module); plain .jsonl written"
                     if compressed is None else ""),
        }


# ── request parsing + replies ─────────────────────────────────────────────


_CATALOG_PATTERNS = (
    re.compile(
        r"^\s*find\s+(?:me\s+)?(?:the\s+book\s+)?(.+?)\s*,?\s+any\s+format\s*$",
        re.IGNORECASE),
    re.compile(
        r"^\s*find\s+(?:me\s+)?(?:the\s+book\s+)?(.+?)\s+any\s+edition\s*$",
        re.IGNORECASE),
    re.compile(
        r"^\s*find\s+(?:me\s+)?the\s+book\s+(.+?)\s*$",
        re.IGNORECASE),
    re.compile(
        r"^\s*where\s+can\s+i\s+(?:read|find)\s+(.+?)\s*$",
        re.IGNORECASE),
)


def parse_catalog_request(text: str) -> dict[str, str] | None:
    """Parse a union-catalogue request. Returns {"query": title} or None.

    Fires only on a book-shaped ask — "find me <title>, any format",
    "find the book <title>", "where can I read <title>". A bare
    "find me ..." with no book cue is not a catalogue ask: "find me a
    restaurant" must never land here.
    """
    text = (text or "").strip()
    if not text or text.startswith("/"):
        return None
    for pattern in _CATALOG_PATTERNS:
        match = pattern.match(text)
        if match:
            query = match.group(1).strip(" \"'.,")
            if len(query) >= 2:
                return {"query": query}
    return None


def _holding_line(holding: Holding) -> str:
    source = LAWFUL_SOURCES.get(holding.source_id, {})
    label = source.get("label", holding.source_id)
    detail = f"{label} — {holding.format}" if holding.format else label
    if holding.health == HEALTH_LIVE:
        state = "live"
    elif holding.health == HEALTH_DEAD:
        state = "unavailable right now"
    elif holding.health == HEALTH_BOT_BLOCKED:
        state = "walled right now"
    elif holding.health == HEALTH_UNVERIFIABLE:
        state = "couldn't check right now"
    else:
        state = "health unknown"
    ref = f" ({holding.source_ref})" if holding.source_ref else ""
    return f"  • {detail} — {state}{ref}"


def format_work(work: WorkRecord) -> str:
    """One work with its holdings and their liveness, dead ones included."""
    authors = ", ".join(work.authors) if work.authors else "unknown author"
    lines = [f"{work.title} — {authors}"]
    identifiers = work.identifiers
    if identifiers.get("isbn"):
        lines[0] += f" (ISBN {identifiers['isbn']})"
    live = work.live_holdings()
    lines.append(
        f"  held by {len(work.holdings)} source(s), "
        f"{len(live)} live right now:")
    for holding in sorted(work.holdings,
                          key=lambda h: (-_HOLDING_SCORE.get(h.health, 1.0),
                                         h.source_id)):
        lines.append(_holding_line(holding))
    return "\n".join(lines)


def format_catalog_reply(query: str, works: list[WorkRecord]) -> str:
    """The NL answer: which sources hold it, and which are alive now."""
    if not works:
        return (
            f"I couldn't find {query!r} in your library or the open "
            "collections (Open Library, Internet Archive, Project "
            "Gutenberg). Nothing invented — try the full title, or an "
            "author with it."
        )
    lines = [f"Found {len(works)} work(s) for {query!r}:"]
    for work in works:
        lines.append("")
        lines.append(format_work(work))
    return "\n".join(lines)
