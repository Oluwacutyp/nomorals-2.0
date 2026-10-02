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

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.ids import ulid_now
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

ARTIFACT_URI_SCHEME = "artifact://"
_URI_RE = re.compile(r"artifact://([A-Za-z0-9_-]+)")


def _emit_created(art: "Artifact") -> None:
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
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
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
    def from_row(cls, row: dict[str, Any]) -> "Artifact":
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


class ArtifactStore:
    """Create, persist, and resolve artifacts on top of a BlobStore."""

    def __init__(self, db: Any, blob_store: Any) -> None:
        self.db = db
        self.blobs = blob_store

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
        except Exception:  # noqa: BLE001 - missing/unreadable blob is a failure
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
