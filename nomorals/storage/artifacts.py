"""First-class artifacts: agent outputs as addressable, provenanced objects.

Instead of agents passing huge strings around, every significant output
becomes an :class:`Artifact` — content-addressed through the blob store,
recorded in the ``artifacts`` table, and referenced by URI::

    store = ArtifactStore(db, blob_store)
    art = store.put_text(report, type="report", creator="analyst",
                         mission_id=mission.id, task_id=task.id)
    # downstream agents receive "artifact://01J..." instead of the blob

:func:`ArtifactStore.resolve` turns a reference back into the artifact;
:meth:`ArtifactStore.read` returns its bytes.  Provenance records where the
content came from and what it was derived from, so the agent can distinguish
"I know this" from "I inferred this".
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines
from .failures import record_failure
from .fts import FTSIndex

_log = get_logger(__name__)

ARTIFACT_URI_SCHEME = "artifact://"
_URI_RE = re.compile(r"artifact://([A-Za-z0-9_-]+)")

# ── content search (WS1 P4) ──────────────────────────────────────────────
#
# Artifact bodies are searchable through the repo's existing FTS organ
# (storage/fts.py — the same FTS5 index memory and documents ride).
# The index is DERIVED state over the artifacts table: rows are keyed
# by the artifacts table's own rowid, maintained eagerly on put /
# delete / supersede and reconciled lazily on every search (a store
# instance with a stronger text extractor backfills bodies an earlier
# instance could not read).  Text extraction is INJECTED: this layer
# (L2) cannot import the documents engine (L4), so the store ships a
# plain-text extractor and higher layers pass
# ``nomorals.documents.artifact_body_text`` for document formats.

CONTENT_FTS_TABLE = "artifacts_fts"
CONTENT_FTS_COLUMNS = ("title", "body")

#: Bodies larger than this are counted as skipped, never read for
#: indexing (a search index is not a streaming reader).
_MAX_INDEX_BODY_BYTES = 4_000_000
#: Indexed text is capped per artifact; the tail of a huge text body
#: is not searchable (the head, where titles/abstracts live, is).
_MAX_INDEX_TEXT_CHARS = 1_000_000

_PLAIN_TEXT_MIMES = {
    "text/plain", "text/markdown", "text/x-markdown", "application/json",
    "text/json", "text/csv", "text/tab-separated-values",
    "application/xml", "text/xml",
}
_PLAIN_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".xml",
    ".log", ".py", ".js", ".ts",
}

_FORMAT_BY_MIME = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/rtf": "rtf", "text/rtf": "rtf",
    "application/epub+zip": "epub",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "text/html": "html",
    "text/plain": "txt", "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "application/json": "json", "text/json": "json",
    "application/xml": "xml", "text/xml": "xml",
    "text/csv": "csv", "text/tab-separated-values": "tsv",
}
_FORMAT_BY_EXT = {
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xls": "xls",
    ".pptx": "pptx", ".html": "html", ".htm": "html", ".md": "markdown",
    ".markdown": "markdown", ".csv": "csv", ".tsv": "tsv",
    ".txt": "txt", ".rtf": "rtf", ".epub": "epub", ".odt": "odt",
    ".ods": "ods", ".json": "json", ".xml": "xml",
}
#: Formats whose bodies are plain text (decoded directly).
_PLAIN_FORMATS = {"txt", "markdown", "json", "csv", "tsv", "xml"}
#: Formats the documents engine extracts (never decodable raw).
_DOCUMENT_FORMATS = {"pdf", "docx", "xlsx", "xls", "pptx", "html",
                     "rtf", "epub", "odt", "ods"}
#: Metadata keys that name an artifact's file (extension source) and
#: its display title, in priority order.
_FILENAME_KEYS = ("filename", "name", "title")
_TITLE_KEYS = ("title", "name", "filename")


def _plain_text_extractor(data: bytes, *, mime: str = "",
                          filename: str = "") -> str | None:
    """The store's built-in extractor: plain-text bodies only.

    Document formats return None here — wire
    ``nomorals.documents.artifact_body_text`` as the store's
    ``text_extractor`` to index those too.  Unknown-typed bodies get
    one honest attempt: clean UTF-8 with no NUL bytes is text.
    """
    if not data:
        return None
    ext = Path(filename).suffix.lower() if filename else ""
    fmt = _FORMAT_BY_MIME.get((mime or "").strip().lower()) \
        or _FORMAT_BY_EXT.get(ext)
    if fmt in _DOCUMENT_FORMATS:
        return None  # document formats need the documents engine
    if (mime.startswith("text/") or mime in _PLAIN_TEXT_MIMES
            or ext in _PLAIN_TEXT_EXTS or (not mime and not ext)):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if text.strip() and "\x00" not in text:
            return text
    return None


def _declared_format(art: Artifact) -> str:
    """The body's format key from its mime, else its filename's
    extension, else "" (unknown)."""
    fmt = _FORMAT_BY_MIME.get((art.mime or "").strip().lower())
    if fmt:
        return fmt
    for key in _FILENAME_KEYS:
        value = (art.metadata or {}).get(key)
        if isinstance(value, str) and value.strip():
            fmt = _FORMAT_BY_EXT.get(Path(value.strip()).suffix.lower())
            if fmt:
                return fmt
    return ""


def _body_kind(art: Artifact) -> tuple[str, str]:
    """Classify a body for search accounting: (category, kind).

    Category is ``plain`` (directly decodable text), ``document``
    (needs the documents engine) or ``nontext``.  Kind is the name
    the accounting reports — a format (``pdf``), a mime
    (``image/png``), or the artifact type when nothing else is known.
    """
    fmt = _declared_format(art)
    if fmt in _PLAIN_FORMATS:
        return "plain", fmt
    if fmt in _DOCUMENT_FORMATS:
        return "document", fmt
    if (art.mime or "").startswith("text/"):
        return "plain", art.mime
    if fmt:
        return "nontext", fmt
    if art.mime:
        return "nontext", art.mime
    if art.type in ("text", "json", "code", "report", "markdown"):
        return "plain", art.type
    return "nontext", art.type or "blob"


def _emit_created(art: Artifact) -> None:
    """Publish ``artifact.created`` on the process bus. Best-effort: a
    broken subscriber must never break storage writes."""
    try:
        global_bus.publish(Event(
            topic="artifact.created",
            data={"artifact_id": art.id, "uri": art.uri, "type": art.type,
                  "creator": art.creator, "mission_id": art.mission_id,
                  "task_id": art.task_id},
            source="nomorals.storage.artifacts",
        ))
    except Exception:  # noqa: BLE001 - events never break storage
        _log.debug("artifact.created event failed", exc_info=True)


@dataclass
class Provenance:
    """Where an artifact's content came from."""

    source_type: str = ""        # tool | model | user | memory | inference | ...
    source_id: str = ""          # tool name, model id, chat id, ...
    source_timestamp: float = 0.0
    confidence: float = 1.0
    derived_from: list[str] = field(default_factory=list)  # artifact ids
    supersedes: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    verification_state: str = "unverified"  # unverified|verified|contradicted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Artifact:
    """One addressable output."""

    id: str
    type: str                    # text | json | code | image | file | report | ...
    content_hash: str            # sha256 in the blob store
    size: int = 0
    mime: str = ""
    creator: str = ""            # agent/role/tool that made it
    mission_id: str = ""
    task_id: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def uri(self) -> str:
        return f"{ARTIFACT_URI_SCHEME}{self.id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uri"] = self.uri
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Artifact:
        return cls(
            id=row["id"],
            type=row.get("type") or "blob",
            content_hash=row["content_hash"],
            size=int(row.get("size") or 0),
            mime=row.get("mime") or "",
            creator=row.get("creator") or "",
            mission_id=row.get("mission_id") or "",
            task_id=row.get("task_id") or "",
            created_at=float(row.get("created_at") or 0.0),
            metadata=json.loads(row.get("metadata") or "{}"),
            provenance=Provenance.from_dict(
                json.loads(row.get("provenance") or "{}")),
        )


@dataclass
class ContentHit:
    """One full-text search result over artifact bodies."""

    artifact: Artifact
    score: float                 # bm25, negated: higher is better
    snippet: str = ""            # body window around the match


class ArtifactStore:
    """Create, persist, and resolve artifacts on top of a BlobStore."""

    def __init__(self, db: Any, blob_store: Any, *,
                 text_extractor: Any = None) -> None:
        self.db = db
        self.blobs = blob_store
        # Body-text extraction for the content index: a callable
        # ``(data, *, mime, filename) -> str | None``.  Defaults to
        # plain-text bodies only; pass
        # ``nomorals.documents.artifact_body_text`` to index document
        # formats too (the documents engine is L4, this store is L2 —
        # extraction is injected, never imported).
        self.text_extractor = text_extractor or _plain_text_extractor
        self._content_fts_cache: FTSIndex | None = None
        self._content_fts_failed = False

    # ── writes ───────────────────────────────────────────────────────────────
    def put(self, data: bytes, *, type: str = "blob", creator: str = "",
            mission_id: str = "", task_id: str = "", mime: str = "",
            metadata: dict[str, Any] | None = None,
            provenance: Provenance | dict[str, Any] | None = None) -> Artifact:
        info = self.blobs.put_bytes(data, mime=mime)
        if isinstance(provenance, dict):
            provenance = Provenance.from_dict(provenance)
        art = Artifact(
            id=ulid_now(),
            type=type,
            content_hash=info.sha256,
            size=len(data),
            mime=mime,
            creator=creator,
            mission_id=mission_id,
            task_id=task_id,
            metadata=dict(metadata or {}),
            provenance=provenance or Provenance(),
        )
        self.db.execute(
            "INSERT INTO artifacts (id, type, content_hash, size, mime, creator,"
            " mission_id, task_id, created_at, metadata, provenance)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (art.id, art.type, art.content_hash, art.size, art.mime,
             art.creator, art.mission_id, art.task_id, art.created_at,
             json.dumps(art.metadata), json.dumps(art.provenance.to_dict())),
        )
        self._index_content(art, data)
        _emit_created(art)
        return art

    def put_text(self, text: str, **kwargs: Any) -> Artifact:
        kwargs.setdefault("mime", "text/plain")
        return self.put(text.encode("utf-8"), type=kwargs.pop("type", "text"), **kwargs)

    def put_json(self, obj: Any, **kwargs: Any) -> Artifact:
        kwargs.setdefault("mime", "application/json")
        return self.put(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
                        type=kwargs.pop("type", "json"), **kwargs)

    def derive(self, data: bytes, *, from_ids: list[str],
               source_type: str = "inference", **kwargs: Any) -> Artifact:
        """Create an artifact derived from other artifacts (provenance link)."""
        prov = kwargs.pop("provenance", None)
        if isinstance(prov, dict):
            prov = Provenance.from_dict(prov)
        prov = prov or Provenance(source_type=source_type)
        prov.derived_from = list(dict.fromkeys([*prov.derived_from, *from_ids]))
        return self.put(data, provenance=prov, **kwargs)

    def supersede(self, artifact_id: str, data: bytes, *,
                  type: str = "", creator: str = "", mime: str = "",
                  mission_id: str | None = None,
                  task_id: str | None = None,
                  metadata: dict[str, Any] | None = None,
                  provenance: Provenance | dict[str, Any] | None = None,
                  ) -> Artifact:
        """Atomically replace ``artifact_id`` with a new version.

        One operation: the new artifact is written carrying
        ``supersedes=[artifact_id]`` (and ``derived_from`` — so
        :meth:`lineage` and :meth:`superseded_by` both see the chain)
        inside a single transaction, and the old body leaves the
        content index in the same call, so full-text search surfaces
        the new version only.  The old artifact itself is untouched —
        its bytes and row stay readable; only its search presence and
        its "current version" status change.  Type, mime, creator,
        mission/task links and metadata default to the old artifact's
        (explicit arguments and metadata keys win).
        """
        old = self.get(artifact_id)
        if old is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        if isinstance(provenance, dict):
            provenance = Provenance.from_dict(provenance)
        prov = provenance or Provenance()
        prov.supersedes = list(dict.fromkeys([*prov.supersedes, artifact_id]))
        prov.derived_from = list(
            dict.fromkeys([*prov.derived_from, artifact_id]))
        merged_meta = dict(old.metadata)
        merged_meta.update(metadata or {})
        txn = (self.db.transaction() if hasattr(self.db, "transaction")
               else contextlib.nullcontext())
        with txn:
            new = self.put(
                data, type=type or old.type, creator=creator or old.creator,
                mission_id=old.mission_id if mission_id is None else mission_id,
                task_id=old.task_id if task_id is None else task_id,
                mime=mime or old.mime, metadata=merged_meta,
                provenance=prov)
        self._deindex_content(artifact_id)
        return new

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, artifact_id: str) -> Artifact | None:
        row = self.db.query_one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        return Artifact.from_row(row) if row else None

    def read(self, artifact_id: str) -> bytes:
        art = self.get(artifact_id)
        if art is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return self.blobs.get_bytes(art.content_hash)

    def read_text(self, artifact_id: str) -> str:
        return self.read(artifact_id).decode("utf-8", errors="replace")

    def verify(self, artifact_id: str) -> bool:
        """Verify an artifact's blob against its recorded content hash.

        Reads the blob back and compares its SHA-256 with the hash stored
        at write time.  Returns False when the blob is missing, unreadable,
        or corrupted after the write — i.e. a failed blob write is caught
        here instead of silently serving bad bytes later.
        """
        import hashlib

        art = self.get(artifact_id)
        if art is None:
            return False
        try:
            data = self.blobs.get_bytes(art.content_hash)
        except Exception as e:  # missing/unreadable blob is a failure
            record_failure('artifacts.ArtifactStore.verify', e)
            return False
        return hashlib.sha256(data).hexdigest() == art.content_hash

    def resolve(self, ref: str) -> Artifact | None:
        """Resolve ``artifact://<id>`` or a bare id to the artifact."""
        ref = (ref or "").strip()
        if ref.startswith(ARTIFACT_URI_SCHEME):
            ref = ref[len(ARTIFACT_URI_SCHEME):]
        if not ref:
            return None
        return self.get(ref)

    @staticmethod
    def find_references(text: str) -> list[str]:
        """All ``artifact://`` URIs mentioned in ``text`` (deduped, in order)."""
        return list(dict.fromkeys(_URI_RE.findall(text or "")))

    def resolve_all(self, text: str) -> dict[str, Artifact]:
        """Map every artifact URI in ``text`` to its artifact (skips unknown)."""
        out: dict[str, Artifact] = {}
        for uri in self.find_references(text):
            art = self.get(uri)
            if art is not None:
                out[f"{ARTIFACT_URI_SCHEME}{uri}"] = art
        return out

    def for_mission(self, mission_id: str) -> list[Artifact]:
        rows = self.db.query(
            "SELECT * FROM artifacts WHERE mission_id = ? ORDER BY created_at",
            (mission_id,))
        return [Artifact.from_row(r) for r in rows]

    def for_task(self, task_id: str) -> list[Artifact]:
        rows = self.db.query(
            "SELECT * FROM artifacts WHERE task_id = ? ORDER BY created_at",
            (task_id,))
        return [Artifact.from_row(r) for r in rows]

    # ── provenance graph ─────────────────────────────────────────────────
    #
    # The ``derived_from`` / ``supersedes`` lists in each artifact's
    # provenance form a directed graph over the artifacts table. These
    # traversals answer "where did this come from?" (lineage) and "what was
    # built on top of this?" (descendants) without any new schema.

    def lineage(self, artifact_id: str) -> list[Artifact]:
        """All ancestors of ``artifact_id`` via ``derived_from`` (BFS).

        Nearest parents come first. Cycle-safe: provenance written by hand
        or by a buggy tool may loop back, and a lineage query must still
        terminate. Missing ancestors are skipped. The artifact itself is
        not included — only its ancestors.
        """
        seen = {artifact_id}
        ancestors: list[Artifact] = []
        queue = [artifact_id]
        while queue:
            current = self.get(queue.pop(0))
            if current is None:
                continue
            for parent_id in current.provenance.derived_from or []:
                if not parent_id or parent_id in seen:
                    continue
                seen.add(parent_id)
                parent = self.get(parent_id)
                if parent is not None:
                    ancestors.append(parent)
                    queue.append(parent_id)
        return ancestors

    def descendants(self, artifact_id: str) -> list[Artifact]:
        """Every artifact whose provenance names ``artifact_id`` in
        ``derived_from`` or ``supersedes`` — the reverse of :meth:`lineage`.
        Ordered oldest-first."""
        out: list[Artifact] = []
        for row in self.db.query("SELECT * FROM artifacts ORDER BY created_at"):
            art = Artifact.from_row(row)
            prov = art.provenance
            if (artifact_id in (prov.derived_from or [])
                    or artifact_id in (prov.supersedes or [])):
                out.append(art)
        return out

    def superseded_by(self, artifact_id: str) -> list[Artifact]:
        """Artifacts that explicitly supersede ``artifact_id`` (a subset of
        :meth:`descendants`: only the ``supersedes`` link, not ``derived_from``)."""
        return [a for a in self.descendants(artifact_id)
                if artifact_id in (a.provenance.supersedes or [])]

    def rebuild(self, artifact_id: str, *, creator: str = "") -> Artifact:
        """Create a fresh artifact re-derived from the same bytes as
        ``artifact_id``, with ``derived_from=[artifact_id]``.

        The new artifact carries the same type, mime, mission/task links
        and metadata; ``creator`` overrides the original creator when given.
        Emits ``artifact.created`` like any other write.
        """
        art = self.get(artifact_id)
        if art is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        data = self.blobs.get_bytes(art.content_hash)
        return self.derive(
            data,
            from_ids=[artifact_id],
            type=art.type,
            mime=art.mime,
            creator=creator or art.creator,
            mission_id=art.mission_id,
            task_id=art.task_id,
            metadata=dict(art.metadata),
        )

    # ── search ─────────────────────────────────────────────────────────────
    def search(
        self,
        *,
        type: str = "",
        creator: str = "",
        mission_id: str = "",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Artifact]:
        """Find artifacts by facets (MLflow ``search_runs`` for artifacts).

        ``metadata`` matches key/value pairs inside the JSON metadata column.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if type:
            clauses.append("type = ?")
            params.append(type)
        if creator:
            clauses.append("creator = ?")
            params.append(creator)
        if mission_id:
            clauses.append("mission_id = ?")
            params.append(mission_id)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        sql = "SELECT * FROM artifacts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        rows = self.db.query(sql, tuple(params))
        out = [Artifact.from_row(r) for r in rows]
        if metadata:
            out = [
                a for a in out
                if all(a.metadata.get(k) == v for k, v in metadata.items())
            ]
        return out

    # ── content search (full-text over bodies, WS1 P4) ─────────────────
    def _content_fts(self) -> FTSIndex | None:
        """The shared FTS organ over artifact bodies, lazily built.

        None when the SQLite build lacks FTS5 or the db cannot carry
        the index — search then degrades to no hits with the stats
        saying so, never to an exception.
        """
        if self._content_fts_cache is None and not self._content_fts_failed:
            try:
                idx = FTSIndex(self.db, CONTENT_FTS_TABLE,
                               columns=list(CONTENT_FTS_COLUMNS),
                               auto_create=True)
                if idx.available:
                    self._content_fts_cache = idx
                else:
                    self._content_fts_failed = True
            except Exception as exc:  # noqa: BLE001 - index is derived state
                record_failure("artifacts.ArtifactStore._content_fts", exc)
                self._content_fts_failed = True
        return self._content_fts_cache

    def _rowid_of(self, artifact_id: str) -> int | None:
        """The artifacts table's rowid for ``artifact_id`` — the key
        the FTS index rows carry (the store's ids are ULIDs; FTS5
        rowids are integers)."""
        try:
            row = self.db.query_one(
                "SELECT rowid AS rid FROM artifacts WHERE id = ?",
                (artifact_id,))
        except Exception as exc:  # noqa: BLE001 - derived state only
            record_failure("artifacts.ArtifactStore._rowid_of", exc)
            return None
        if row is None or row.get("rid") is None:
            return None
        return int(row["rid"])

    @staticmethod
    def _content_title(art: Artifact) -> str:
        for key in _TITLE_KEYS:
            value = (art.metadata or {}).get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return art.type

    def _extract_content_text(self, art: Artifact,
                              data: bytes | None = None) -> str | None:
        """Searchable text for ``art``'s body via the injected
        extractor; None when the body is not text-bearing (binary,
        oversized, unparseable) — the accounting names those by kind.
        """
        if art.size > _MAX_INDEX_BODY_BYTES:
            return None
        if data is None:
            try:
                data = self.blobs.get_bytes(art.content_hash)
            except Exception as exc:  # noqa: BLE001 - missing blob: skip
                record_failure(
                    "artifacts.ArtifactStore._extract_content_text", exc)
                return None
        filename = ""
        for key in _FILENAME_KEYS:
            value = (art.metadata or {}).get(key)
            if isinstance(value, str) and value.strip():
                filename = value.strip()
                break
        try:
            text = self.text_extractor(data, mime=art.mime,
                                       filename=filename)
        except Exception as exc:  # noqa: BLE001 - extractor must not break writes
            record_failure(
                "artifacts.ArtifactStore._extract_content_text", exc)
            return None
        if not text:
            return None
        text = text.strip()
        return text[:_MAX_INDEX_TEXT_CHARS] if text else None

    def _index_content(self, art: Artifact,
                       data: bytes | None = None) -> bool:
        """Index ``art``'s body text (replacing any prior entry).
        Returns True when the body is now searchable.  Best-effort:
        the content index is derived state and never fails a write.
        """
        fts = self._content_fts()
        if fts is None:
            return False
        rid = self._rowid_of(art.id)
        if rid is None:
            return False
        try:
            text = self._extract_content_text(art, data)
            if text:
                fts.put(rid, [self._content_title(art), text])
                return True
            fts.delete(rid)
        except Exception as exc:  # noqa: BLE001 - derived state only
            record_failure("artifacts.ArtifactStore._index_content", exc)
        return False

    def _deindex_content(self, artifact_id: str) -> None:
        """Remove ``artifact_id``'s body from the content index."""
        fts = self._content_fts()
        if fts is None:
            return
        rid = self._rowid_of(artifact_id)
        if rid is None:
            return
        try:
            fts.delete(rid)
        except Exception as exc:  # noqa: BLE001 - derived state only
            record_failure("artifacts.ArtifactStore._deindex_content", exc)

    def _superseded_ids(self) -> set[str]:
        """Every artifact id named in some artifact's
        ``provenance.supersedes`` — the replaced versions."""
        ids: set[str] = set()
        for row in self.db.query("SELECT provenance FROM artifacts"):
            try:
                prov = json.loads(row.get("provenance") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(prov, dict):
                for old_id in prov.get("supersedes") or []:
                    if isinstance(old_id, str) and old_id:
                        ids.add(old_id)
        return ids

    def _sync_content_index(self) -> dict[str, Any]:
        """Reconcile the content index with the artifacts table and
        report the accounting.

        Prunes index rows whose artifact is gone or superseded,
        backfills live artifacts an earlier store instance could not
        index (e.g. written before a documents-backed extractor was
        wired).  The returned dict is the honest census:
        ``total == indexed + skipped + superseded`` — every body is
        accounted for, and skipped bodies are named by kind.
        """
        rows = self.db.query("SELECT rowid AS rid, id FROM artifacts")
        superseded = self._superseded_ids()
        live = {int(r["rid"]): r["id"] for r in rows
                if r["id"] not in superseded}
        stats: dict[str, Any] = {
            "available": False, "total": len(rows), "indexed": 0,
            "superseded": len([r for r in rows
                               if r["id"] in superseded]),
            "skipped": 0, "skipped_kinds": {},
        }
        fts = self._content_fts()
        if fts is None:
            for art_id in live.values():
                art = self.get(art_id)
                if art is not None:
                    _cat, kind = _body_kind(art)
                    kinds = stats["skipped_kinds"]
                    kinds[kind] = kinds.get(kind, 0) + 1
            stats["skipped"] = sum(stats["skipped_kinds"].values())
            return stats
        stats["available"] = True
        try:
            indexed_rids = {
                int(r["rowid"]) for r in self.db.query(
                    f'SELECT rowid FROM "{CONTENT_FTS_TABLE}"')}
        except Exception as exc:  # noqa: BLE001 - treat as empty index
            record_failure("artifacts.ArtifactStore._sync_content_index",
                           exc)
            indexed_rids = set()
        for rid in indexed_rids - set(live):
            try:
                fts.delete(rid)
            except Exception as exc:  # noqa: BLE001 - derived state only
                record_failure(
                    "artifacts.ArtifactStore._sync_content_index", exc)
        indexed_rids &= set(live)
        for rid, art_id in live.items():
            if rid in indexed_rids:
                continue
            art = self.get(art_id)
            if art is not None and self._index_content(art):
                indexed_rids.add(rid)
        stats["indexed"] = len(indexed_rids)
        for rid, art_id in live.items():
            if rid in indexed_rids:
                continue
            art = self.get(art_id)
            if art is None:
                continue
            _cat, kind = _body_kind(art)
            kinds = stats["skipped_kinds"]
            kinds[kind] = kinds.get(kind, 0) + 1
        stats["skipped"] = sum(stats["skipped_kinds"].values())
        return stats

    def search_content(self, query: str, *,
                       limit: int = 20) -> list[ContentHit]:
        """Full-text search over artifact BODIES (bm25-ranked).

        Facet :meth:`search` finds artifacts by what they are; this
        finds them by what they SAY.  The index is reconciled first,
        so results reflect the store as it stands now: superseded
        versions never match (their replacements do), deleted bodies
        never match, and bodies written by other store instances are
        backfilled on the way in.
        """
        if not (query or "").strip():
            return []
        self._sync_content_index()
        fts = self._content_fts()
        if fts is None:
            return []
        hits = fts.search(query, limit=max(1, limit),
                          snippet_column="body",
                          weights={"title": 2.0, "body": 1.0})
        if not hits:
            return []
        id_by_rowid = {
            int(r["rid"]): r["id"] for r in self.db.query(
                "SELECT rowid AS rid, id FROM artifacts")}
        out: list[ContentHit] = []
        for hit in hits:
            art_id = id_by_rowid.get(hit.rowid)
            art = self.get(art_id) if art_id else None
            if art is not None:
                out.append(ContentHit(artifact=art, score=hit.score,
                                      snippet=hit.snippet))
        return out

    def content_index_stats(self) -> dict[str, Any]:
        """The content index's census (see :meth:`_sync_content_index`):
        how many bodies are searchable, how many are superseded, and
        how many are skipped — named by kind, never silently absent.
        """
        return self._sync_content_index()

    # ── tags ───────────────────────────────────────────────────────────────
    def tag(self, artifact_id: str, key: str, value: Any) -> bool:
        """Set a metadata tag on an artifact (MLflow tags)."""
        art = self.get(artifact_id)
        if art is None:
            return False
        meta = dict(art.metadata)
        meta[key] = value
        self.db.execute(
            "UPDATE artifacts SET metadata = ? WHERE id = ?",
            (json.dumps(meta), artifact_id),
        )
        if key in _TITLE_KEYS and not self.superseded_by(artifact_id):
            art.metadata = meta
            self._index_content(art)  # the indexed title follows the tag
        return True

    def untag(self, artifact_id: str, key: str) -> bool:
        art = self.get(artifact_id)
        if art is None or key not in art.metadata:
            return False
        meta = dict(art.metadata)
        del meta[key]
        self.db.execute(
            "UPDATE artifacts SET metadata = ? WHERE id = ?",
            (json.dumps(meta), artifact_id),
        )
        if key in _TITLE_KEYS and not self.superseded_by(artifact_id):
            art.metadata = meta
            self._index_content(art)
        return True

    # ── delete ─────────────────────────────────────────────────────────────
    def delete(self, artifact_id: str, *, drop_blob: bool = True) -> bool:
        """Delete an artifact row; decrements the blob refcount (DVC ``gc``).

        With ``drop_blob=True`` the blob itself is purged when its refcount
        reaches zero; aliases pointing at the artifact are removed too.
        """
        art = self.get(artifact_id)
        if art is None:
            return False
        self._deindex_content(artifact_id)  # rowid dies with the row
        self.db.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        self._ensure_alias_table()
        self.db.execute(
            "DELETE FROM artifact_aliases WHERE artifact_id = ?", (artifact_id,)
        )
        if drop_blob and hasattr(self.blobs, "release"):
            try:
                self.blobs.release(art.content_hash, delete_at_zero=True)
            except Exception as exc:  # noqa: BLE001 - row is already gone
                _log.debug("blob release after artifact delete failed: %s", exc)
        return True

    # ── aliases ────────────────────────────────────────────────────────────
    #
    # MLflow model-registry-style named pointers: "latest", "champion",
    # "production". ``set_alias`` moves the pointer; ``resolve_alias``
    # follows it. The table is created lazily so older databases keep
    # working without a migration.

    ALIAS_TABLE = "artifact_aliases"

    def _ensure_alias_table(self) -> None:
        self.db.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.ALIAS_TABLE}" ('
            "name TEXT PRIMARY KEY, "
            "artifact_id TEXT NOT NULL, "
            "updated_at REAL NOT NULL)"
        )

    def set_alias(self, name: str, artifact_id: str) -> None:
        """Point ``name`` at ``artifact_id`` (creates or moves the alias)."""
        name = (name or "").strip()
        if not name:
            raise ValueError("alias name must not be empty")
        if self.get(artifact_id) is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        self._ensure_alias_table()
        self.db.execute(
            f'INSERT INTO "{self.ALIAS_TABLE}" (name, artifact_id, updated_at) '
            "VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "artifact_id = excluded.artifact_id, updated_at = excluded.updated_at",
            (name, artifact_id, time.time()),
        )

    def resolve_alias(self, name: str) -> Artifact | None:
        """Follow alias ``name`` to its artifact (None when unknown)."""
        self._ensure_alias_table()
        row = self.db.query_one(
            f'SELECT artifact_id FROM "{self.ALIAS_TABLE}" WHERE name = ?',
            (name,),
        )
        if row is None:
            return None
        return self.get(row["artifact_id"])

    def delete_alias(self, name: str) -> bool:
        self._ensure_alias_table()
        return (
            self.db.delete(self.ALIAS_TABLE, "name = ?", (name,)) > 0
        )

    def aliases(self) -> dict[str, str]:
        """All aliases: name → artifact id."""
        self._ensure_alias_table()
        rows = self.db.query(
            f'SELECT name, artifact_id FROM "{self.ALIAS_TABLE}" ORDER BY name'
        )
        return {r["name"]: r["artifact_id"] for r in rows}

    # ── bundle export ──────────────────────────────────────────────────────
    def export_bundle(
        self,
        artifact_ids: list[str],
        dest: str | os.PathLike[str],
    ) -> Path:
        """Export artifacts as a ``.tar.gz`` bundle (bytes + manifest.json).

        The DVC-``get``/fetch story: one file carries the artifacts, their
        metadata, and provenance anywhere.
        """
        import tarfile

        target = Path(dest).expanduser()
        if target.suffix != ".gz":
            target = target.with_suffix(target.suffix + ".tar.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []
        with tarfile.open(target, "w:gz") as tar:
            for artifact_id in artifact_ids:
                art = self.get(artifact_id)
                if art is None:
                    continue
                data = self.blobs.get_bytes(art.content_hash)
                entry = art.to_dict()
                manifest.append(entry)
                info = tarfile.TarInfo(name=f"{artifact_id}.bin")
                info.size = len(data)
                info.mtime = int(art.created_at)
                import io as _io

                tar.addfile(info, _io.BytesIO(data))
            manifest_bytes = json.dumps(
                manifest, indent=2, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_bytes)
            import io as _io2

            tar.addfile(info, _io2.BytesIO(manifest_bytes))
        return target

    def import_bundle(self, bundle: str | os.PathLike[str], *,
                      creator: str = "") -> list[Artifact]:
        """Import a bundle created by :meth:`export_bundle`."""
        import io as _io
        import tarfile

        source = Path(bundle).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"bundle not found: {source}")
        imported: list[Artifact] = []
        with tarfile.open(source, "r:gz") as tar:
            manifest_member = tar.getmember("manifest.json")
            manifest_file = tar.extractfile(manifest_member)
            if manifest_file is None:
                raise ValueError(f"bundle {source} has no manifest.json")
            manifest = json.loads(manifest_file.read().decode("utf-8"))
            by_id = {m["id"]: m for m in manifest if isinstance(m, dict)}
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".bin"):
                    continue
                artifact_id = member.name[: -len(".bin")]
                entry = by_id.get(artifact_id, {})
                data_file = tar.extractfile(member)
                if data_file is None:
                    continue
                art = self.put(
                    data_file.read(),
                    type=entry.get("type", "blob"),
                    creator=creator or entry.get("creator", ""),
                    mission_id=entry.get("mission_id", ""),
                    task_id=entry.get("task_id", ""),
                    mime=entry.get("mime", ""),
                    metadata=entry.get("metadata", {}),
                    provenance=entry.get("provenance", {}),
                )
                imported.append(art)
        return imported

    # ── presentation ───────────────────────────────────────────────────────
    def format_card(self, art: Artifact, theme: Any = None) -> str:
        """One-artifact pretty card through the shared style layer."""
        theme = theme or active_theme()
        prov = art.provenance
        return "\n".join([
            header(f"artifact {art.id}", theme=theme),
            *kv_lines(
                {
                    "uri": art.uri,
                    "type": art.type,
                    "mime": art.mime or "–",
                    "size": f"{art.size} bytes",
                    "sha256": art.content_hash[:16] + "…",
                    "creator": art.creator or "–",
                    "mission": art.mission_id or "–",
                    "task": art.task_id or "–",
                    "source": f"{prov.source_type}:{prov.source_id}"
                    if prov.source_type else "–",
                    "verification": prov.verification_state,
                    "derived from": ", ".join(prov.derived_from) or "–",
                },
                theme=theme,
            ),
        ])

    def stats(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT type, COUNT(*) AS n, COALESCE(SUM(size), 0) AS bytes "
            "FROM artifacts GROUP BY type ORDER BY n DESC"
        )
        return {
            "total": sum(r["n"] for r in rows),
            "bytes": sum(r["bytes"] for r in rows),
            "by_type": {r["type"]: {"count": r["n"], "bytes": r["bytes"]}
                        for r in rows},
        }
