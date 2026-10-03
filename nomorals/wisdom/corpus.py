"""CanonCorpus — the curated esoteric collection.

Wraps ``books.Library`` (the ingest → FTS5 BM25 → ranked passage search
machine) with curation on top: a versioned manifest of texts, provenance
on every hit, and canon-status metadata. The Library owns mechanics;
the Corpus owns curation.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..books.library import Library
from .errors import CorpusError

# Canon statuses recognized by the manifest validator.
CANON_STATUSES = frozenset({
    "canon", "apocrypha", "pseudepigrapha", "gnostic", "dss",
    "eastern", "esoteric", "secondary",
})

MANIFEST_VERSION = 1


@dataclass
class ManifestEntry:
    """One text in the corpus manifest."""
    slug: str
    title: str
    tradition: str
    canon_status: str
    source_url: str
    license: str = "public-domain"
    translator: str = ""
    sha256: str = ""
    ingested_at: float = 0.0
    #: Operator triage notes (e.g. why a URL was changed, or why a text
    #: is currently unmirrorable). Informational only; never affects fetch.
    notes: str = ""

    def validate(self) -> None:
        if not self.slug or not self.slug.replace("-", "").replace("_", "").isalnum():
            raise CorpusError(f"manifest entry has bad slug: {self.slug!r}")
        if not self.title:
            raise CorpusError(f"manifest entry {self.slug!r} has no title")
        if self.canon_status not in CANON_STATUSES:
            raise CorpusError(
                f"manifest entry {self.slug!r} has unknown canon_status "
                f"{self.canon_status!r} (expected one of "
                f"{sorted(CANON_STATUSES)})")
        if not self.source_url.startswith(("http://", "https://")):
            raise CorpusError(
                f"manifest entry {self.slug!r} has bad source_url: "
                f"{self.source_url!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "tradition": self.tradition,
            "canon_status": self.canon_status,
            "translator": self.translator,
            "source_url": self.source_url,
            "license": self.license,
            "sha256": self.sha256,
            "ingested_at": self.ingested_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ManifestEntry":
        try:
            entry = cls(
                slug=str(d["slug"]),
                title=str(d["title"]),
                tradition=str(d.get("tradition", "")),
                canon_status=str(d["canon_status"]),
                source_url=str(d["source_url"]),
                license=str(d.get("license", "public-domain")),
                translator=str(d.get("translator", "")),
                sha256=str(d.get("sha256", "")),
                ingested_at=float(d.get("ingested_at", 0.0)),
                notes=str(d.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusError(f"malformed manifest entry: {exc}") from exc
        entry.validate()
        return entry


@dataclass
class ProvenanceHit:
    """A search hit enriched with corpus provenance. Every claim traces
    back to one of these — no passage, no claim."""
    work: str
    translator: str
    section: str
    url: str
    snippet: str
    score: float = 0.0
    canon_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "work": self.work,
            "translator": self.translator,
            "section": self.section,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "canon_status": self.canon_status,
        }


@dataclass
class Answer:
    """The result of ``keeper.ask``: passages plus a synthesis stub.

    The synthesis is intentionally thin in this phase — it states what
    the passages say, not what is true. Deeper synthesis is Phase 6.
    """
    query: str
    passages: list[ProvenanceHit] = field(default_factory=list)
    synthesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "synthesis": self.synthesis,
            "passages": [p.to_dict() for p in self.passages],
        }


class CanonCorpus:
    """The curated collection. Owns the manifest; delegates storage and
    search to ``books.Library``."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.library = Library(context)
        self._manifest: dict[str, ManifestEntry] = {}
        self._manifest_path = self._wisdom_root() / "manifest.json"
        self._load_manifest()

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

    # ── manifest ──────────────────────────────────────────────────────
    def _load_manifest(self) -> None:
        if not self._manifest_path.is_file():
            return
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CorpusError(
                f"corrupt manifest at {self._manifest_path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("version") != MANIFEST_VERSION:
            raise CorpusError(
                f"unsupported manifest version at {self._manifest_path}")
        entries = raw.get("entries", [])
        if not isinstance(entries, list):
            raise CorpusError("manifest 'entries' must be a list")
        for item in entries:
            entry = ManifestEntry.from_dict(item)
            if entry.slug in self._manifest:
                raise CorpusError(
                    f"duplicate manifest slug: {entry.slug!r}")
            self._manifest[entry.slug] = entry

    def _save_manifest(self) -> None:
        payload = {
            "version": MANIFEST_VERSION,
            "entries": [e.to_dict() for e in self._manifest.values()],
        }
        self._manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")

    def seed(self) -> int:
        """Register every entry in the packaged seed manifest.

        Idempotent: slugs already in the manifest are skipped.
        Returns the number of newly registered entries.
        """
        from pathlib import Path
        seed_path = Path(__file__).parent / "data" / "manifest.json"
        if not seed_path.is_file():
            raise CorpusError(f"seed manifest missing: {seed_path}")
        try:
            raw = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CorpusError(f"corrupt seed manifest: {exc}") from exc
        added = 0
        for item in raw.get("entries", []):
            entry = ManifestEntry.from_dict(item)
            if entry.slug not in self._manifest:
                self._manifest[entry.slug] = entry
                added += 1
        if added:
            self._save_manifest()
        return added

    def register(self, entry: ManifestEntry) -> None:
        """Add a text to the manifest (validates; fails fast)."""
        entry.validate()
        if entry.slug in self._manifest:
            raise CorpusError(
                f"manifest slug already registered: {entry.slug!r}")
        self._manifest[entry.slug] = entry
        self._save_manifest()

    def get(self, slug: str) -> ManifestEntry:
        try:
            return self._manifest[slug]
        except KeyError:
            raise CorpusError(f"unknown corpus slug: {slug!r}") from None

    def list(self) -> list[ManifestEntry]:
        return list(self._manifest.values())

    def status(self) -> dict[str, Any]:
        ingested = [e for e in self._manifest.values() if e.ingested_at]
        return {
            "texts": len(self._manifest),
            "ingested": len(ingested),
            "pending": len(self._manifest) - len(ingested),
            "by_tradition": self._count_by("tradition"),
            "by_canon_status": self._count_by("canon_status"),
        }

    def _count_by(self, attr: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._manifest.values():
            key = getattr(e, attr) or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ── ingest ────────────────────────────────────────────────────────
    def ingest_text(self, slug: str, text: str, *,
                    translator: str = "") -> ManifestEntry:
        """Ingest raw text for a registered manifest entry.

        Writes the text to a temp file, hands it to ``Library.ingest``,
        records the sha256 + timestamp in the manifest. Re-ingesting
        unchanged bytes is a no-op (sha256 match).
        """
        entry = self.get(slug)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if entry.sha256 == digest and entry.ingested_at:
            return entry  # already ingested, unchanged
        if len(text.strip()) < 200:
            raise CorpusError(
                f"text for {slug!r} is too small to ingest "
                f"({len(text.strip())} chars)")
        tmp = self._wisdom_root() / f".ingest-{slug}.md"
        try:
            tmp.write_text(text, encoding="utf-8")
            # The Library slugs by title; we pass the corpus slug via the
            # title prefix so search hits map back to manifest provenance.
            self.library.ingest(
                tmp, title=f"[{slug}] {entry.title}",
                author=translator or entry.translator)
        except Exception as exc:
            raise CorpusError(
                f"library ingest failed for {slug!r}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
        entry.sha256 = digest
        entry.ingested_at = time.time()
        if translator:
            entry.translator = translator
        self._save_manifest()
        return entry

    # ── ask ───────────────────────────────────────────────────────────
    def ask(self, query: str, *, top: int = 5,
            tradition: str = "", mode: str = "keyword") -> Answer:
        """Search the corpus; every hit carries provenance.

        ``mode`` selects the retrieval path:

        - ``"keyword"`` (default) — the existing FTS5/BM25 lexical search.
        - ``"semantic"`` — dense-vector cosine search over passage
          embeddings (requires :meth:`build_semantic_index` first).
        - ``"hybrid"`` — reciprocal-rank fusion of keyword + semantic.
        """
        query = (query or "").strip()
        if not query:
            raise CorpusError("ask() needs a non-empty query")
        if mode not in ("keyword", "semantic", "hybrid"):
            raise CorpusError(
                f"unknown ask() mode {mode!r} "
                "(expected 'keyword', 'semantic', or 'hybrid')")
        passages: list[ProvenanceHit] = []
        if mode == "keyword":
            hits = self.library.search(query, top=top)
            passages = self._to_provenance(hits, tradition)
        else:
            passages = self._ask_semantic(query, top=top,
                                          tradition=tradition,
                                          hybrid=mode == "hybrid")
        synthesis = self._synthesize(query, passages)
        return Answer(query=query, passages=passages, synthesis=synthesis)

    def _to_provenance(self, hits: Any,
                       tradition: str) -> list[ProvenanceHit]:
        """Map raw search hits to provenance-carrying passages."""
        passages: list[ProvenanceHit] = []
        for h in hits:
            # The corpus slug is embedded as "[slug]" at the title start.
            slug = ""
            title = h.title or h.book
            if title.startswith("["):
                end = title.find("]")
                if end > 1:
                    slug = title[1:end]
                    title = title[end + 1:].strip()
            entry = self._manifest.get(slug)
            if tradition and entry and entry.tradition != tradition:
                continue
            passages.append(ProvenanceHit(
                work=title,
                translator=entry.translator if entry else "",
                section=h.chapter,
                url=entry.source_url if entry else "",
                snippet=h.passage,
                score=h.score,
                canon_status=entry.canon_status if entry else "",
            ))
        return passages

    # ── semantic search ─────────────────────────────────────────────
    def _embed_backend(self, name: str = "") -> Any:
        """Best available embedding backend (lazy, cached per name)."""
        from .embeddings import auto_backend
        cache = self.__dict__.setdefault("_backend_cache", {})
        key = name.strip().lower() or "auto"
        if key not in cache:
            cache[key] = auto_backend("" if key == "auto" else key)
        return cache[key]

    def _vector_index(self, backend: Any) -> Any:
        from .vectorstore import open_index
        return open_index(str(self._wisdom_root() / "vectors.db"), backend)

    def _iter_passages(self) -> list[tuple[str, str]]:
        """All (passage_key, text) pairs in the library.

        Key is ``"<book_slug>#<chapter_number>"`` — stable across DB
        rebuilds, unlike rowids. Works for both the FTS5 and the plain
        passages table.
        """
        import sqlite3
        con = sqlite3.connect(self.library.db_path())
        try:
            rows = con.execute(
                "SELECT book_slug, number, text FROM passages").fetchall()
        finally:
            con.close()
        return [(f"{slug}#{int(num)}", text or "")
                for slug, num, text in rows]

    def _passage_rows(self, keys: list[str]) -> dict[str, Any]:
        """Fetch library rows for vector-hit keys → SearchHit-shaped rows."""
        import sqlite3
        from types import SimpleNamespace
        if not keys:
            return {}
        con = sqlite3.connect(self.library.db_path())
        try:
            out: dict[str, Any] = {}
            for key in keys:
                if "#" not in key:
                    continue
                slug, _, num = key.rpartition("#")
                try:
                    number = int(num)
                except ValueError:
                    continue
                row = con.execute(
                    "SELECT book_slug, book_title, chapter, number, text "
                    "FROM passages WHERE book_slug = ? AND number = ?",
                    (slug, number)).fetchone()
                if row:
                    out[key] = SimpleNamespace(
                        book=row[0], title=row[1], chapter=row[2],
                        chapter_number=row[3], passage=row[4], score=0.0)
            return out
        finally:
            con.close()

    def _semantic_search(self, query: str, *, top: int) -> list[Any]:
        """Vector KNN → SearchHit-shaped rows ordered by cosine."""
        backend = self._embed_backend()
        try:
            index = self._vector_index(backend)
        except Exception as exc:
            raise CorpusError(
                f"semantic search unavailable: {exc}; run "
                f"build_semantic_index() first") from exc
        try:
            if index.count() == 0:
                raise CorpusError(
                    "semantic index is empty; run build_semantic_index() "
                    "after ingesting texts")
            qvec = backend.embed_one(query)
            hits = index.search(qvec, top=top)
        finally:
            index.close()
        rows = self._passage_rows([k for k, _ in hits])
        ordered = []
        for key, cosine in hits:
            row = rows.get(key)
            if row is None:
                continue
            row.score = cosine
            ordered.append(row)
        return ordered

    def _ask_semantic(self, query: str, *, top: int,
                      tradition: str, hybrid: bool) -> list[ProvenanceHit]:
        sem_hits = self._semantic_search(query, top=top * 2 if hybrid else top)
        if not hybrid:
            return self._to_provenance(sem_hits, tradition)
        from .hybrid import fuse_hits
        kw_hits = self.library.search(query, top=top * 2)
        fused = fuse_hits(
            kw_hits, sem_hits,
            key_fn=lambda h: (
                h.title or h.book, h.chapter_number, h.passage[:64]),
            top=top * 2)
        return self._to_provenance(fused[:top], tradition)

    def build_semantic_index(self, backend_name: str = "", *,
                             rebuild: bool = False,
                             progress: Any = None) -> dict[str, Any]:
        """Embed every library passage and (re)build the vector index.

        ``backend_name`` forces one provider (``""`` = best available).
        Idempotent-ish: skipped only when an identical index already
        exists *and* ``rebuild`` is False. Returns a status dict.
        """
        backend = self._embed_backend(backend_name)
        from .vectorstore import VectorStoreError, open_index
        path = str(self._wisdom_root() / "vectors.db")
        pairs = self._iter_passages()
        if not pairs:
            raise CorpusError(
                "no passages to index; ingest texts first")
        # An existing index built by the same backend+dim is only
        # skipped when the passage set is unchanged.
        try:
            index = open_index(path, backend)
        except VectorStoreError:
            if not rebuild:
                raise
            Path(path).unlink(missing_ok=True)
            index = open_index(path, backend)
        try:
            if not rebuild and index.count() == len(pairs):
                return self.semantic_index_status()
            keys = [k for k, _ in pairs]
            texts = [t for _, t in pairs]
            n = index.build(keys, texts, backend, progress=progress)
        finally:
            index.close()
        status = self.semantic_index_status()
        status["vectors"] = n
        return status

    def semantic_index_status(self) -> dict[str, Any]:
        """Backend, engine, and vector counts for the semantic index."""
        import sqlite3
        from .embeddings import available_backends
        path = self._wisdom_root() / "vectors.db"
        status: dict[str, Any] = {
            "built": False,
            "backend": "",
            "engine": "",
            "vectors": 0,
            "available_backends": available_backends(),
        }
        if not path.is_file():
            return status
        con = sqlite3.connect(str(path))
        try:
            meta = dict(con.execute(
                "SELECT key, value FROM meta").fetchall())
            count = con.execute(
                "SELECT COUNT(*) FROM vectors").fetchone()[0]
        except Exception:
            return status
        finally:
            con.close()
        status.update({
            "built": True,
            "backend": meta.get("backend", ""),
            "engine": meta.get("engine", ""),
            "vectors": int(count),
        })
        return status

    @staticmethod
    def _synthesize(query: str, passages: list[ProvenanceHit]) -> str:
        if not passages:
            return (
                f"No passages in the corpus speak to {query!r} yet. "
                "Ingest more texts or try different terms.")
        works = sorted({p.work for p in passages})
        return (
            f"{len(passages)} passage(s) from {len(works)} work(s) "
            f"({', '.join(works[:3])}{'…' if len(works) > 3 else ''}) "
            f"speak to {query!r}. Each passage below carries its source; "
            f"WisdomKeeper reports what the texts say, not what is true.")
