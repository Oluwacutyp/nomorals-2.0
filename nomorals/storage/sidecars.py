"""The sidecar federation registry (Mongo program P5).

Devon does not have one database. Beside the central store — whose
tables carry the dispositions in :mod:`nomorals.storage.carry`
(PORTED / PORT_RESIDUE / PINNED) — a federation of sidecar stores
keeps durable state of its own: per-organ SQLite files opened with raw
``sqlite3.connect``, ``Database`` instances on organ-owned files,
JSON/JSONL file stores, vector engines, blob files. The census counted
67 raw-connect files outside ``storage/`` alone. A Mongo program that
only classifies the central file leaves most of Devon's persistence
unaccounted; this module is the accounting.

Every sidecar store has exactly one verdict, recorded in
:data:`SIDECAR_STORES` and surfaced by
:func:`nomorals.storage.docstore.storage_backend_status` (the
``sidecars`` key), so "where does every class of durable state live on
this install" is one call, not an archaeology dig:

* ``ported`` — the live path routes through the DocumentStore seam
  whenever a store is resolved (the boot-published ambient store, or
  — for the L0 preferences module, which may not import storage at
  all — a port injected by the composition layer), behind the
  organ's existing public API; on the SQLite default the organ runs
  its native file path byte-for-byte. Collections are namespaced ``<store>__<table>``
  (the census §5.6 rule), documents are carry-v1 shaped via
  :class:`~nomorals.storage.rowstore.RowTable`, and the organ's
  existing file data is imported lazily, once, through the same
  mapping the first time the seam path constructs (see
  :class:`SidecarPort`) — flipping the live path never strands the
  file's history.
* ``federated`` — the store stays a file/blob store BY NATURE (large
  binary artifacts and append-only journals are not documents), but
  it is registered here and its metadata lives in a backend-aware
  place (the central store's carried tables, the P3-ported dataset
  registry), so a Mongo-primary install still has one coherent
  picture of it.
* ``pinned`` — SQLite (or its own engine) BY DESIGN, a terminal
  disposition in the spirit of :data:`nomorals.storage.carry.PINNED_DOMAINS`,
  with the reason named. Several pins *follow* a central pin (the
  recall core, the games settlement graph, FTS): their rows key off
  the pinned system's identity, so porting them alone would fork
  them from the system they describe.
* ``residue`` — classified, live path still on its own file, port not
  yet done. Residue is NOT a pin: each entry names what its port
  takes. The drift-guard test (tests/test_mongo_p5.py) scans the tree
  mechanically — every raw ``sqlite3.connect`` file and every
  ``VectorStore`` instantiation outside ``storage/`` must be covered
  by exactly one entry here — so a NEW sidecar can never appear
  unclassified again.
* ``ephemeral`` / ``reader`` — classified non-stores: in-memory
  scratchpads and read-only consumers of other stores' files. They
  own no durable state; they are recorded so the scan's coverage is
  total and honest.

The factory half of this module is the port mechanism the census
asked for ("a shared document-store factory keyed by store name"):
:func:`sidecar_port` hands a sidecar organ its namespaced,
mapping-derived :class:`RowTable` set over the resolved store, or
``None`` on the SQLite default — the KV/attached_rows pickup rule,
generalized from the central handle to organs that never see it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .docstore import ambient_document_store
from .failures import record_failure

__all__ = [
    "SIDECAR_STORES",
    "SidecarPort",
    "SidecarSpec",
    "VERDICTS",
    "scan_sidecar_modules",
    "sidecar_port",
    "sidecar_status",
    "uncovered_modules",
]

#: The verdict vocabulary. ``pinned`` is terminal-by-design;
#: ``residue`` is explicitly not.
VERDICTS = ("ported", "federated", "pinned", "residue", "ephemeral", "reader")


@dataclass(frozen=True)
class SidecarSpec:
    """One sidecar store's classification."""

    name: str
    verdict: str  # one of VERDICTS
    kind: str  # "sqlite-file" | "database-file" | "json-file" | "jsonl" |
    #            "blob-files" | "vector" | "mixed" | "none"
    modules: tuple[str, ...] = ()  # owner modules (package-relative)
    reason: str = ""
    tables: tuple[str, ...] = ()  # the store's own tables, where it has them
    follows: str = ""  # central pin/registry this verdict follows, if any
    #: (module, marker) anchors for stores the mechanical scan cannot
    #: see (no raw connect of their own): the drift test asserts the
    #: marker still exists, so a rewrite forces reclassification.
    anchors: tuple[tuple[str, str], ...] = ()
    #: Modules in this entry that instantiate VectorStore (scan anchor).
    vector_sites: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "kind": self.kind,
            "modules": list(self.modules),
            "tables": list(self.tables),
            "reason": self.reason,
            "follows": self.follows or None,
        }


# ── the mechanical scan (shared by the drift-guard test) ──────────────────

_CONNECT_MARK = "sqlite3.connect"
_VECTOR_RE = re.compile(r"VectorStore\(")


def scan_sidecar_modules(root: str | Path) -> dict[str, set[str]]:
    """Mechanically find sidecar-shaped modules under ``root``.

    ``root`` is the ``nomorals`` package directory. Returns the set of
    package-relative module paths that open raw sqlite connections
    (``connect``) and the set that instantiate ``VectorStore``
    (``vector_sites``), excluding the ``storage/`` package itself —
    the two classes a new durable store cannot hide from.
    """
    base = Path(root)
    connect: set[str] = set()
    vector_sites: set[str] = set()
    for path in sorted(base.rglob("*.py")):
        rel = path.relative_to(base).as_posix()
        if rel.startswith("storage/"):
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        if _CONNECT_MARK in src:
            connect.add(rel)
        if _VECTOR_RE.search(src):
            vector_sites.add(rel)
    return {"connect": connect, "vector_sites": vector_sites}


def uncovered_modules(root: str | Path) -> dict[str, list[str]]:
    """Scan results NOT covered by the registry (the drift report).

    Empty lists are the invariant the test pins: a new sidecar file
    appears here the day it is written, until it is classified.
    """
    scan = scan_sidecar_modules(root)
    covered_connect: set[str] = set()
    covered_vector: set[str] = set()
    for spec in SIDECAR_STORES.values():
        covered_connect.update(spec.modules)
        covered_vector.update(spec.vector_sites)
    return {
        "connect": sorted(scan["connect"] - covered_connect),
        "vector_sites": sorted(scan["vector_sites"] - covered_vector),
    }


# ── the factory: namespaced seam tables for one sidecar store ─────────────

def _collection(store_name: str, table: str) -> str:
    return f"{store_name}__{table}"


def mapping_from_ddl(ddl: str, table: str, collection: str):
    """The carry mapping for a sidecar table, derived from its own DDL.

    The sidecar's ``CREATE TABLE`` statements are executed against a
    scratch in-memory :class:`~nomorals.storage.db.Database` and the
    mapping is derived by the same PRAGMA route the central inventory
    uses — a sidecar port can never drift from the DDL its native path
    actually runs.
    """
    from .db import Database
    from .rowstore import mapping_for

    scratch = Database(":memory:")
    try:
        scratch.executescript(ddl)
        base = mapping_for(scratch, table)
    finally:
        scratch.close()
    from .carry import TableMapping

    return TableMapping(table=table, collection=collection,
                        columns=list(base.columns))


class SidecarPort:
    """One sidecar store's seam path: namespaced tables over a store.

    Constructed by :func:`sidecar_port` only when a document store is
    resolved; the organ keeps its native implementation for the
    ``None`` case. ``tables`` maps table name → :class:`RowTable` for
    primary-keyed tables; PK-less or AUTOINCREMENT-child tables are
    written through :meth:`put_raw` with caller-allocated ids (the
    carry's ``rowid:<n>`` ids have no live-write meaning — the
    rowstore rule). :meth:`ensure_imported` performs the one-time
    lazy import of the organ's existing SQLite file through the same
    mapping, so the history the file holds is the history the seam
    serves from the first construction on.
    """

    def __init__(self, store_name: str, store: Any, ddl: str,
                 tables: tuple[str, ...], sqlite_path: str | Path | None,
                 defaults: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        from .db import Database
        from .rowstore import RowTable, column_defaults

        self.store_name = store_name
        self.store = store
        self._sqlite_path = Path(sqlite_path) if sqlite_path else None
        self.tables: dict[str, Any] = {}
        scratch = Database(":memory:")
        try:
            scratch.executescript(ddl)
            derived_defaults = {t: column_defaults(scratch, t) for t in tables}
        finally:
            scratch.close()
        for table in tables:
            mapping = mapping_from_ddl(ddl, table, _collection(store_name, table))
            merged = dict(derived_defaults.get(table, {}))
            merged.update((defaults or {}).get(table, {}))
            self.tables[table] = RowTable(store, mapping, defaults=merged)
        self._meta_collection = _collection(store_name, "meta")

    @property
    def meta_collection(self) -> str:
        """The store's meta collection (import marker, id counters)."""
        return self._meta_collection

    # ── raw document access (PK-less tables, counters) ───────────────────
    def put_raw(self, table: str, doc_id: str, fields: Mapping[str, Any]) -> None:
        self.store.put(_collection(self.store_name, table), doc_id, dict(fields))

    def find_raw(self, table: str,
                 filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.store.find(_collection(self.store_name, table), filter)

    def next_id(self, counter: str) -> int:
        """Allocate the next integer id for an AUTOINCREMENT-shaped table.

        One atomic ``inc`` on the store's meta document — the same
        single-document atomicity the seam advertises on both rungs.
        """
        return int(self.store.inc(self._meta_collection, "counters", counter))

    # ── the one-time lazy file import ────────────────────────────────────
    def ensure_imported(
        self,
        writers: Mapping[str, Callable[[list[dict[str, Any]]], int]],
        *,
        after_import: Callable[[dict[str, int]], None] | None = None,
    ) -> dict[str, int]:
        """Import the organ's SQLite file into the seam, once.

        ``writers`` maps table name → a callable that takes the file's
        rows (plain dicts, in rowid order) and writes them to the seam,
        returning how many it wrote; ports supply them because
        PK-less tables need caller-shaped ids. Idempotent by
        construction: completion is recorded in the meta collection,
        a store that is already live (documents present, e.g. written
        before an import marker existed) is adopted, not overwritten,
        and every write is an upsert by document id. Returns the
        per-table imported counts ({} when there was nothing to do).
        """
        meta = self.store.get(self._meta_collection, "import")
        if meta and meta.get("done"):
            return {}
        live = any(
            self.store.count(_collection(self.store_name, t))
            for t in writers)
        counts: dict[str, int] = {}
        if not live and self._sqlite_path is not None \
                and self._sqlite_path.exists():
            from .db import Database

            source = Database(str(self._sqlite_path))
            try:
                for table, writer in writers.items():
                    if not source.table_exists(table):
                        counts[table] = 0
                        continue
                    rows = source.query(
                        f'SELECT * FROM "{table}" ORDER BY rowid')
                    counts[table] = writer([dict(r) for r in rows])
            finally:
                source.close()
            if after_import is not None and any(counts.values()):
                after_import(counts)
        self.store.put(self._meta_collection, "import", {
            "done": True, "tables": counts, "at": time.time()})
        return counts


def sidecar_port(store_name: str, ddl: str, tables: tuple[str, ...],
                 sqlite_path: str | Path | None, *,
                 store: Any = None,
                 defaults: Mapping[str, Mapping[str, Any]] | None = None,
                 ) -> SidecarPort | None:
    """The seam path for a registered sidecar store, or ``None``.

    ``None`` is the SQLite default: no store resolved (neither an
    explicit ``store`` nor the boot ambient), so the organ runs its
    native file path. A store that fails to construct the port is a
    loud error, never a silent fallback — the resolution already
    degraded at boot if the backend was unusable.
    """
    resolved = store if store is not None else ambient_document_store()
    if resolved is None:
        return None
    if store_name not in SIDECAR_STORES:
        raise KeyError(f"unregistered sidecar store: {store_name!r}")
    spec = SIDECAR_STORES[store_name]
    if spec.verdict != "ported":
        raise ValueError(
            f"sidecar store {store_name!r} is classified {spec.verdict!r}, "
            "not 'ported' — a port must be registered before it is wired")
    try:
        return SidecarPort(store_name, resolved, ddl, tables, sqlite_path,
                           defaults=defaults)
    except Exception as exc:  # noqa: BLE001 - journaled, then raised
        record_failure("storage.sidecars.port", f"{store_name}: {exc}")
        raise


# ── the registry ──────────────────────────────────────────────────────────
#
# Grouped by verdict. Every entry names its owner module(s) exactly as
# the mechanical scan finds them (package-relative, storage/ excluded).

_S = SidecarSpec

SIDECAR_STORES: dict[str, SidecarSpec] = {}

def _register(spec: SidecarSpec) -> None:
    if spec.name in SIDECAR_STORES:
        raise ValueError(f"duplicate sidecar store: {spec.name}")
    if spec.verdict not in VERDICTS:
        raise ValueError(f"sidecar {spec.name}: bad verdict {spec.verdict!r}")
    SIDECAR_STORES[spec.name] = spec


# ── ported (Mongo P5) ─────────────────────────────────────────────────────

_register(_S(
    name="challenge-memory", verdict="ported", kind="sqlite-file",
    modules=("browser/challenge_memory.py",),
    tables=("encounters", "attempts"),
    reason=(
        "ported in P5: ChallengeMemory routes through the seam "
        "(challenge-memory__encounters / __attempts) whenever a store "
        "is resolved, behind its unchanged API; encounter ids are "
        "allocated by one atomic inc on the meta document (the "
        "AUTOINCREMENT shape, preserved); the attempts child table is "
        "PK-less, so its documents carry caller-shaped ids "
        "(<encounter_id>:<ordinal>) — reads are always by encounter. "
        "The native sqlite3 path serves the SQLite default."),
))

_register(_S(
    name="preferences", verdict="ported", kind="sqlite-file",
    modules=("preferences/store.py",),
    tables=("preferences", "preference_signals"),
    reason=(
        "ported in P5 for the SQLite-file variant: weights and the "
        "signal log route through the seam (preferences__preferences "
        "by its composite key, __preference_signals by allocated id) "
        "behind the unchanged API. Pickup is by INJECTED port, not "
        "the ambient: preferences/store.py is layer L0 (stdlib only), "
        "so it cannot import storage — the composition layer builds "
        "the port with sidecar_port() and passes it as port=. The "
        "class's other two shapes stay what they are by caller "
        "choice: the .json path is a JSON file backend "
        "(file-by-design), and path=None is an ephemeral :memory: "
        "store — neither is silently made persistent by the port."),
))

_register(_S(
    name="social-drafts", verdict="ported", kind="sqlite-file",
    modules=("social/drafts.py",),
    tables=("social_drafts",),
    reason=(
        "ported in P5: DraftQueue's single table routes through the "
        "seam (social-drafts__social_drafts) whenever a store is "
        "resolved, behind its unchanged lifecycle API — transitions "
        "are single-row updates, exactly the seam's single-document "
        "guarantee; the transition graph itself is untouched."),
))

_register(_S(
    name="agent-identities", verdict="ported", kind="sqlite-file",
    modules=("agents/identities.py",),
    tables=("identities",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 1): "
        "IdentityStore's single table routes through the seam "
        "(agent-identities__identities) whenever a store is "
        "resolved, behind its unchanged never-raises API; task-"
        "scoped delegated identities are caller-keyed documents. "
        "The native sqlite3 path serves the SQLite default."),
))

_register(_S(
    name="audio-audiobooks", verdict="ported", kind="sqlite-file",
    modules=("audio/audiobook.py",),
    tables=("audiobooks",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 1): "
        "AudiobookStore's single table routes through the seam "
        "(audio-audiobooks__audiobooks) whenever a store is "
        "resolved, behind its unchanged API; production-state rows "
        "are documents, the disclosure payload rides as its JSON "
        "text exactly as the file stores it. The native sqlite3 "
        "path serves the SQLite default."),
))

_register(_S(
    name="audio-characters", verdict="ported", kind="sqlite-file",
    modules=("audio/characters.py",),
    tables=("characters",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 1): "
        "CharacterStore's single table routes through the seam "
        "(audio-characters__characters) whenever a store is "
        "resolved, behind its unchanged API; voice definitions "
        "are self-contained documents and the JSON payload "
        "columns ride as text exactly as the file stores them. "
        "The native path's in-file column migration stays a "
        "native-path concern — the seam mapping is derived from "
        "the module's final-shape DDL. The native sqlite3 path "
        "serves the SQLite default."),
))

_register(_S(
    name="audio-overviews", verdict="ported", kind="sqlite-file",
    modules=("audio/overview.py",),
    tables=("overviews",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 1): "
        "OverviewStore's single table routes through the seam "
        "(audio-overviews__overviews) whenever a store is "
        "resolved, behind its unchanged API; the script/sources "
        "payloads ride as their JSON text exactly as the file "
        "stores them. The native sqlite3 path serves the SQLite "
        "default."),
))

_register(_S(
    name="audio-keyterms", verdict="ported", kind="sqlite-file",
    modules=("audio/pipeline.py",),
    tables=("keyterms",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 1): "
        "KeytermStore's single table routes through the seam "
        "(audio-keyterms__keyterms) whenever a store is resolved, "
        "behind its unchanged API. The index is derived data "
        "(rebuildable from transcripts), but its history is the "
        "file's, so the lazy import still runs. The native "
        "sqlite3 path serves the SQLite default."),
))

_register(_S(
    name="audio-melodic", verdict="ported", kind="sqlite-file",
    modules=("audio/melodic.py",),
    tables=("works",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 3): "
        "MelodyCatalog's single table routes through the seam "
        "(audio-melodic__works) whenever a store is resolved, "
        "behind its unchanged never-raises API. The store name, "
        "file and collections stay deliberately distinct from "
        "audio-fingerprints (SongRecon class-1): class separation "
        "is structural — a class-3 work can never masquerade as a "
        "class-1 recording. The segmented melody/chroma payloads "
        "ride as document fields, their JSON text exactly as the "
        "file stores it. The native sqlite3 path (its lazy "
        "connection) serves the SQLite default."),
))

_register(_S(
    name="property-scam", verdict="ported", kind="sqlite-file",
    modules=("property/scam.py",),
    tables=("seen_listings",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 3): "
        "ScamStore's seen-listing registry routes through the "
        "seam (property-scam__seen_listings) whenever a store is "
        "resolved, behind its unchanged API; duplicate lookups "
        "are finds by photo hash / normalized address, and the "
        "INSERT OR IGNORE remember goes through the row "
        "adapter's insert_ignore. The photo/address indexes stay "
        "native-path DDL — the seam reads are finds. The native "
        "sqlite3 path serves the SQLite default."),
))

_register(_S(
    name="property-value", verdict="ported", kind="sqlite-file",
    modules=("property/value.py",),
    tables=("value_estimates",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 3): "
        "ValueStore's single table routes through the seam "
        "(property-value__value_estimates) whenever a store is "
        "resolved, behind its unchanged never-raises API; "
        "estimates are documents and their comps / kept-id "
        "payloads ride as their JSON text exactly as the file "
        "stores them. The native sqlite3 path serves the SQLite "
        "default."),
))

_register(_S(
    name="social-voice", verdict="ported", kind="sqlite-file",
    modules=("social/voice.py",),
    tables=("evergreen",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 3): "
        "EvergreenQueue's single table routes through the seam "
        "(social-voice__evergreen) whenever a store is resolved, "
        "behind its unchanged API; next_due's engagement-floor / "
        "cooldown filter and NULLS-FIRST ordering re-express "
        "over finds in Python. The category index stays "
        "native-path DDL. The native sqlite3 path serves the "
        "SQLite default."),
))

_register(_S(
    name="social-voice-notes", verdict="ported", kind="sqlite-file",
    modules=("social/voice_notes.py",),
    tables=("voice_notes",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 3): "
        "VoiceNoteStore's single table routes through the seam "
        "(social-voice-notes__voice_notes) whenever a store is "
        "resolved, behind its unchanged never-raises API; the "
        "stats language rollup (its one GROUP BY) and the "
        "transcript substring search re-express over finds in "
        "Python. The vault copy and STT stay outside the store "
        "seam, unchanged; the chat index stays native-path DDL. "
        "The native sqlite3 path serves the SQLite default."),
))

_register(_S(
    name="calendar-integration", verdict="ported", kind="sqlite-file",
    modules=("integrations/calendar_integration.py",),
    tables=("sync_cursors",),
    reason=(
        "ported in the Mongo residue Batch 1 pass (slice 3): "
        "the calendar sync cursors route through the seam "
        "(calendar-integration__sync_cursors) whenever a store "
        "is resolved, behind CalendarIntegration's unchanged "
        "API; the composite (account, calendar_id) key rides "
        "the carry's joined-id shape (the preferences "
        "precedent). A cursor is only valid against the "
        "external system's sync token stream, so the port "
        "carries the continuity pin in test_mongo_p5: a cursor "
        "advanced on the native path is the cursor the seam "
        "resumes from after the lazy import. The module's "
        "native _sync_db() path serves the SQLite default."),
))

# ── federated by reference (files/blobs by nature; metadata accounted) ────

_register(_S(
    name="personal-vault", verdict="federated", kind="jsonl",
    modules=(),
    anchors=(("social/personal.py", "class PersonalVault"),),
    reason=(
        "the owner's personal-surface vault is append-only JSONL "
        "(journal.jsonl + recoveries.jsonl + dm_cache.json under the "
        "workspace) — the journal /pair reads, file-by-design per "
        "census §2.3. It is not a database and does not become one; "
        "it is registered so the federation picture is complete."),
))

_register(_S(
    name="artifact-blobs", verdict="federated", kind="blob-files",
    modules=(),
    anchors=(("storage/artifacts.py", "class ArtifactStore"),
             ("storage/blob.py", "class BlobStore")),
    reason=(
        "artifact/binary content is content-addressed FILES (local "
        "BlobStore or its S3 twin behind open_blob_store) — large "
        "binaries are not documents. The metadata half already lives "
        "in the central store's artifacts/blobs tables, which the P2 "
        "carry moves and the parity gate proves; ArtifactStore's "
        "live path stays native SQL over that table this pass. This "
        "is the pattern the federation verdict names."),
))

_register(_S(
    name="training-data", verdict="federated", kind="jsonl",
    modules=(),
    anchors=(("training/dataset.py", "class"),
             ("training/collect.py", "jsonl")),
    reason=(
        "training datasets are JSONL files on disk — an interchange "
        "format, not a store (census §2.3). Their registry half "
        "(datasets / training_runs tables) ported to the seam in "
        "P3; the items themselves stay files (carry.PORT_RESIDUE "
        "records dataset_items the same way)."),
))

# ── pinned by design ──────────────────────────────────────────────────────

_register(_S(
    name="incidents", verdict="pinned", kind="sqlite-file",
    modules=("core/incidents.py",),
    tables=("incidents", "recoveries", "verified_fixes"),
    reason=(
        "the incident journal is failure-recording infrastructure: "
        "it must be able to record the document store's own failures, "
        "so it may not depend on the store it reports on. It stays "
        "on the most primitive rung — a direct sqlite file (or "
        ":memory:) — by the same never-raises logic as "
        "storage/failures.py."),
))

_register(_S(
    name="backup-machinery", verdict="pinned", kind="mixed",
    modules=("backup/snapshot.py", "backup/verify.py", "os/snapshots.py"),
    reason=(
        "snapshot/verify machinery is FILE machinery by design: it "
        "copies, integrity-checks and restores files, and backups are "
        "files on every install. It is backend-aware since P6, as "
        "shipped: BackupManager dispatches on the active backend — a "
        "SQLite install snapshots its file exactly as before, while a "
        "Mongo-primary install's payload is a document dump exported "
        "through the seam (storage/backup.py's write_document_dump) "
        "recorded in the same manifest format, verified by "
        "scratch-restore + spot parity and restored through the "
        "seam; os/snapshots.py carries the same document leg (a "
        "snapshot missing it on a Mongo-configured install fails "
        "verification rather than silently omitting the primary "
        "store). The pin is on the machinery's medium, not on "
        "database blindness."),
))

_register(_S(
    name="doctor-file-probe", verdict="pinned", kind="sqlite-file",
    modules=("core/verify.py",),
    reason=(
        "the doctor's raw connect is a FILE-integrity probe: it "
        "opens ~/.nomorals/nomorals.db to verify the SQLite rung "
        "itself. Its backend-aware counterpart already exists — "
        "core/verify reports the document-store state from "
        "storage_backend_status beside the file probe."),
))

_register(_S(
    name="documents-index", verdict="pinned", kind="sqlite-file",
    modules=("documents/index.py",),
    follows="fts",
    reason=(
        "the documents index is an FTS5 lane (its own Database over "
        "the index file, FTS5 virtuals inside): it follows the "
        "central fts pin — FTS5 stays on the SQLite rung by design, "
        "no self-hosted Mongo equivalent at BM25 parity."),
))

_register(_S(
    name="memory-recall-riders", verdict="pinned", kind="sqlite-file",
    modules=("memory/delivery.py", "memory/repetition.py"),
    tables=("known_memories", "surfacings", "review_cards"),
    follows="memory-recall",
    reason=(
        "delivery scoring and spaced repetition are riders on the "
        "recall core: their rows key off memory identity and are "
        "read/written in the recall core's flows. They follow the "
        "memory-recall pin (carry.PINNED_DOMAINS) — porting them "
        "alone would fork them from the memories they schedule."),
))

_register(_S(
    name="vector-engines", verdict="pinned", kind="vector",
    modules=("memory/vector_backends.py", "wisdom/vectorstore.py",
             "wisdom/corpus.py"),
    tables=("vectors", "embed_cache", "meta"),
    vector_sites=("memory/vector_backends.py", "memory/manager.py",
                  "memory/tiers.py", "tools/code_indexer.py"),
    follows="memory-recall",
    reason=(
        "embeddings stay in vector engines, whose backends are "
        "already plural by design (storage/vectors.py VectorStore "
        "over pure-Python/numpy, the usearch/sqlite-vec backends, "
        "the wisdom corpus's own store): an embedding index is "
        "engine state, not documents. The central instantiations "
        "ride the recall pin's embeddings note; VectorStore sites "
        "on passed handles (code_indexer) follow their caller's "
        "database disposition."),
))

_register(_S(
    name="games-film", verdict="pinned", kind="sqlite-file",
    modules=("games/film.py",),
    tables=("film_breakdowns", "fingerprints"),
    follows="games-arena",
    reason=(
        "game-film breakdowns belong to the games domain: the "
        "domain's state graph is SQLite-pinned by design (carry "
        "games-arena pin — settlement atomicity), and its sidecar "
        "follows the domain's disposition."),
))

_register(_S(
    name="maps-gazetteer", verdict="pinned", kind="sqlite-file",
    modules=("maps/gazetteer.py",),
    tables=("gazetteer_packs", "gazetteer_entries",
            "gazetteer_staging", "gazetteer_staging_corrupt",
            "gazetteer_build_state"),
    reason=(
        "the gazetteer index (gazetteer.sqlite3) is derived data "
        "by design: it lives IN the packs directory, is built "
        "from the pack tiles by `nm maps index`, is keyed per "
        "pack by the pack digest, and is disposable — a corrupt "
        "index file is deleted and rebuilt, because the packs "
        "are the source of truth (Atlas doctrine: packs are the "
        "basemap, local-first). No owner data lives here — "
        "marks live in maps/marks.json and the map receipt "
        "store. Porting a rebuildable index into the document "
        "store would fork it from the packs it indexes and "
        "strand it when the packs move."),
))

_register(_S(
    name="maps-routing-graphs", verdict="pinned", kind="sqlite-file",
    modules=("maps/routing.py",),
    tables=("routing_graphs", "routing_nodes", "routing_arcs"),
    reason=(
        "the routing graph store (routing.sqlite3) is the same "
        "Atlas shape as the gazetteer index: a disposable cache "
        "of road graphs extracted from pack tiles, living IN "
        "the packs directory, keyed by (pack digest, corridor) "
        "cache keys, and rebuilt from the packs on demand. It "
        "holds no owner data and no state the packs cannot "
        "reproduce."),
))

# ── classified non-stores ─────────────────────────────────────────────────

_register(_S(
    name="datasci-scratch", verdict="ephemeral", kind="none",
    modules=("datasci/workspace.py",),
    reason=(
        "not a store: the data-science workspace runs read-only SQL "
        "over caller dataframes through an in-memory SQLite "
        "scratchpad. No durable state exists to federate."),
))

_register(_S(
    name="sidecar-readers", verdict="reader", kind="none",
    modules=("search/sources.py", "tools/vision.py",
             "agents/healing_ticket.py", "agents/repair_pipeline.py",
             "goals/providers.py", "maps/mbtiles.py"),
    reason=(
        "read-only consumers of other stores' files, owning no "
        "state of their own: federated search reads the books "
        "library file, the vision tool resolves inbox:<id> "
        "references from the inbox file, the healing ticket and "
        "the repair pipeline read the pinned incident journal's "
        "file read-only (observation writes nothing — the ticket "
        "is an envelope over the journal, never a parallel "
        "store), the goals provider reads the health timeline's "
        "file read-only (health is L5, above goals' L4, so the "
        "store cannot be imported; its DDL mirror is pinned by "
        "the goals P6 suite), and the mbtiles reader streams "
        "builder-emitted pack artifacts — an interchange file, "
        "converted into packs, never a store. Their reads follow "
        "the target store's disposition (books-library / "
        "workspace-inbox / incidents)."),
))

# ── Database-class hybrids (own Database instance on an own file) ─────────

_register(_S(
    name="books-library", verdict="residue", kind="database-file",
    modules=("books/library.py",),
    tables=("books", "annotations", "bookmarks", "book_tags",
            "book_series", "collections", "collection_books",
            "digest_flags"),
    vector_sites=("books/library.py",),
    reason=(
        "the library organ owns library.db through the Database "
        "class (plus a raw connect for its FTS5 search lane and a "
        "VectorStore on the same file). The catalog is "
        "document-shaped, but its search lane is embedded FTS5: "
        "the port waits on the search-parity story that also owns "
        "the documents-index pin, so catalog and search do not "
        "fork. Until then the file is its rung, and federated "
        "search reads it there (sidecar-readers)."),
    follows="fts",
))

_register(_S(
    name="workspace-inbox", verdict="residue", kind="database-file",
    modules=("workspace/inbox.py",),
    anchors=(("workspace/inbox.py", "inbox.db"),),
    reason=(
        "the workspace inbox owns inbox.db through the Database "
        "class. A Database-class sidecar ports by attaching the "
        "resolved store to its own handle and routing its tables "
        "via attached_rows — the pattern is proven (P3/P4); this "
        "organ's port is a later domain pass."),
))

_register(_S(
    name="shopping-deals", verdict="residue", kind="database-file",
    modules=("tools/deals.py",),
    anchors=(("tools/deals.py", "shopping.db"),),
    reason=(
        "the deals tool opens shopping.db through the Database "
        "class per call. Same disposition as workspace-inbox: a "
        "Database-class sidecar awaiting its domain port pass."),
))

_register(_S(
    name="health-timeline", verdict="residue", kind="database-file",
    modules=("health/timeline.py",),
    anchors=(("health/timeline.py", "timeline.db"),),
    reason=(
        "the health timeline owns timeline.db through the Database "
        "class. Same disposition as workspace-inbox: a "
        "Database-class sidecar awaiting its domain port pass."),
))

# ── residue: contained domain file stores, classified, port queued ────────
#
# Each of these is a real durable store whose live path stays on its
# own SQLite file for this pass. The reason is stated per store; the
# shared shape — and what the port takes — is the sentence appended
# by _residue(): the P5 factory pattern applied per organ, the same
# surgery the three P5 ports demonstrate. Residue is a queue, not a
# verdict of unimportance, and never a pin.

_RESIDUE_TAIL = (
    " Classified in P5 with its live path still on the file: the "
    "port is the P5 factory pattern (namespaced collections over "
    "the resolved store, one-time lazy file import, native path on "
    "the SQLite default) applied in a later domain pass.")


def _residue(name: str, module: str, tables: tuple[str, ...],
             clause: str, *, kind: str = "sqlite-file",
             follows: str = "") -> None:
    _register(_S(
        name=name, verdict="residue", kind=kind, modules=(module,),
        tables=tables, follows=follows,
        reason=clause + _RESIDUE_TAIL))


_residue("audio-fingerprints", "audio/fingerprint.py", ("hashes", "tracks"),
         "audio fingerprint hashes per track in fingerprints.db; "
         "hash rows are lookup-keyed blobs, document-shaped but "
         "bulk-binary in character.")
_residue("commerce-cart-recovery", "commerce/cart_recovery.py",
         ("carts", "cart_steps", "recoveries"),
         "abandoned-cart recovery flows and their step state.")
_residue("commerce-medusa", "commerce/medusa.py",
         ("stores", "webhook_subscribers"),
         "Medusa commerce backend links and webhook subscribers — "
         "state that mirrors an external system's registry.")
_residue("health-challenges", "health/challenges.py",
         ("challenges", "challenge_members", "challenge_pbs", "squads",
          "streaks", "streak_freezes", "completions", "proofs"),
         "health challenge groups, streaks and proofs — the "
         "largest health sidecar, eight tables with cross-table "
         "streak flows; its port wants the domain pass to itself.")
_residue("health-coach", "health/coach.py", ("readiness_log",),
         "coaching readiness log.")
_residue("health-form", "health/form.py", ("form_analyses",),
         "exercise form analyses in form.db.")
_residue("health-nutrition", "health/nutrition.py", ("meals", "water"),
         "meal and water logs in nutrition.db.")
_residue("health-training", "health/training.py",
         ("plans", "workout_log", "set_feedback"),
         "training plans, workout log and per-set feedback.")
_residue("household", "household/household.py",
         ("shared_assets", "shared_calendar", "shared_commitments",
          "household_audit"),
         "household shared state in household_shared.db — "
         "multi-member shared records whose write flows are "
         "audited as a set; the port reviews its shared-write "
         "composition first.")
_residue("identity-documents", "identity/documents.py",
         ("identity_documents",),
         "identity document registry — sensitive records; its port "
         "pass carries the vault/encryption review with it.")
_residue("immune-corpus", "immune/corpus.py", (),
         "the self-corpus: Devon's harvested self-evidence "
         "(verified-fix and owner-directed-receipt exemplars, "
         "each with provenance and the corpus version that "
         "admitted it) as one versioned JSON file in the settings "
         "home's data/ dir, written atomically (tmp+replace) by "
         "CorpusStore. Its only sqlite touch is a read-only "
         "harvest of the pinned incident journal's verified "
         "fixes.", kind="json-file")
_residue("digital-twin", "integrations/digital_twin.py",
         ("states", "desired", "entity_meta"),
         "digital-twin entity states and desired-state records.")
_residue("email-integration", "integrations/email_integration.py",
         ("cursors", "watches"),
         "email sync cursors and watches — same cursor-continuity "
         "note as calendar-integration.")
_residue("legal-portfolio", "legal/portfolio.py",
         ("contracts", "obligations", "sla_terms"),
         "legal contract portfolio in portfolio.db.")
_residue("legal-regulatory", "legal/regulatory.py",
         ("reg_items", "reg_seen", "reg_watches"),
         "regulatory tracking items and seen-state in regulatory.db.")
_residue("life-archive", "life/archive.py", ("life_events",),
         "the life archive's event records.")
_residue("llm-cost-display", "llm/cost_display.py", ("cost_display",),
         "LLM cost display aggregates in cost_display.db.")
_residue("marketing-aeo", "marketing/aeo.py", ("aeo_reports",),
         "answer-engine-optimization reports in aeo.db.")
_residue("marketing-briefs", "marketing/briefs.py",
         ("briefs", "drafts", "score_feedback"),
         "marketing briefs, their drafts and score feedback.")
_residue("marketing-competitor", "marketing/competitor.py",
         ("tracked", "snapshots"),
         "competitor tracking and snapshots.")
_residue("marketing-guardrails", "marketing/guardrails.py",
         ("rules", "audit", "metric_history"),
         "marketing guardrail rules with their audit trail.")
_residue("marketing-send", "marketing/send.py",
         ("send_queue", "send_log", "send_providers", "send_templates",
          "send_variants", "send_suppressions", "send_blocklist",
          "send_costs"),
         "the send pipeline's store in send_queue.db — it "
         "contains a QUEUE (send_queue) whose claim semantics need "
         "the P4 fencing treatment, so this port is a coordination "
         "port, not a plain CRUD port.")
_residue("matching-batches", "matching/batches.py",
         ("batches", "candidates", "shown", "feedback"),
         "matching batches with shown/feedback state.")
_residue("matching-questionnaire", "matching/questionnaire.py",
         ("answers", "ranks", "dealbreakers", "capacities"),
         "matching questionnaire answers and rankings.")
_residue("os-timeline", "os/timeline.py", ("event_log",),
         "the persistent event timeline: an append-only event log "
         "written by an EventBus subscriber — append-mostly and "
         "document-shaped; it ports with the os domain.")
_residue("people", "people/store.py",
         ("persons", "notes", "links", "gifts", "obligations"),
         "the people store: five tables with JOIN-shaped reads "
         "(person + notes rollups); the port re-expresses those "
         "reads over _id-ordered finds, a domain pass of its own.")
_residue("planning-congestion", "planning/congestion.py",
         ("congestion_resources", "congestion_events",
          "congestion_holds", "congestion_breakers"),
         "congestion resources, holds and circuit breakers in "
         "congestion.db — holds are lease-shaped state; the port "
         "reviews hold expiry semantics against the seam's "
         "store-managed expiry.")
_residue("planning-estimates", "planning/estimates.py",
         ("estimate_segments", "estimate_stats"),
         "estimate segments and their rolled-up stats in "
         "estimates.db.")
_residue("planning-graph", "planning/graph.py",
         ("graph_nodes", "graph_edges"),
         "the world graph in world_graph.db — graph-shaped "
         "nodes/edges; its port follows the graph story (the kg "
         "tables' disposition in carry.PORT_RESIDUE), not the "
         "document-fit pattern.")
_residue("planning-route", "planning/route.py",
         ("route_stops", "route_actuals", "route_globals"),
         "route plans, actuals and global stats.")
_residue("property-passport", "property/passport.py",
         ("passports", "history", "refs", "doc_refs"),
         "property passports and their history in passport.db.")
_residue("social-leads", "social/leads.py",
         ("leads", "lead_comments", "comment_seen", "lead_magnets",
          "lead_triggers", "trigger_meta"),
         "lead capture state in leads.db.")
_residue("social-profiles", "social/profiles.py",
         ("profiles", "elements", "likes"),
         "prompt-scaffolded profiles and element likes.")
_residue("social-whatsapp-cost", "social/whatsapp_cost.py",
         ("wa_costs", "wa_budgets"),
         "WhatsApp cost tracking and budgets.")
_residue("travel-groups", "travel/groups.py",
         ("group_trips", "legs", "expenses", "payments", "budgets",
          "proposals", "votes", "polls"),
         "group trip state — eight tables with money-split flows "
         "(expenses/payments); the port reviews its split "
         "composition, like household.")
_residue("travel-itinerary", "travel/itinerary.py", ("trips",),
         "trip itineraries.")
_residue("travel-watchers", "travel/watchers.py",
         ("watches", "price_history", "alert_log"),
         "travel price watches and their history.")
_residue("travel-whitelabel", "travel/whitelabel.py",
         ("travel_clients", "client_events"),
         "white-label travel client records and events.")
_residue("triggers-saved-search", "triggers/saved_search.py",
         ("saved_searches", "seen_matches", "pending_digest"),
         "saved searches with seen-match dedup and pending digests.")


def sidecar_status() -> dict[str, Any]:
    """JSON-able federation summary for status surfaces.

    The one-call answer to "where does every class of durable state
    live on this install": per-verdict counts plus every store's
    verdict, kind, owner modules, tables and reason.
    """
    counts: dict[str, int] = {}
    for spec in SIDECAR_STORES.values():
        counts[spec.verdict] = counts.get(spec.verdict, 0) + 1
    return {
        "counts": counts,
        "stores": {
            name: SIDECAR_STORES[name].to_dict()
            for name in sorted(SIDECAR_STORES)
        },
    }
