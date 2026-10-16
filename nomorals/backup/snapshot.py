"""Snapshots: consistent local copies, before anything is encrypted/shipped.

Databases go through the SQLite online backup API — never
``shutil.copy`` on a live WAL database — exactly the way
:class:`nomorals.storage.replication.SqliteReplicator` and
:class:`nomorals.storage.backup.BackupManager` do it. This module is the
multi-database coordinator over those two organs: use it when the caller
has file paths; use :func:`snapshot_via_backup_manager` /
:func:`snapshot_via_replicator` when the caller already holds a live
:class:`~nomorals.storage.db.Database`.

Every snapshot is verified with ``PRAGMA integrity_check`` *before* it
is accepted. A corrupt snapshot aborts the run — it is never encrypted,
never shipped, never counted as a generation.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import CorruptBackupError, DiskFullError
from .plan import BackupPlan, build_plan

__all__ = [
    "SnapshotEntry",
    "SnapshotResult",
    "snapshot_sqlite_file",
    "snapshot_databases",
    "snapshot_files",
    "snapshot_via_backup_manager",
    "snapshot_via_replicator",
    "take_snapshot",
]

_log = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe_db(path: Path) -> dict[str, Any]:
    """integrity_check + table count + schema version for a DB file."""
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        integrity = row[0] if row else "no result"
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        try:
            ver_row = conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
            schema_version = int(ver_row[0] or 0)
        except sqlite3.Error:
            schema_version = int(
                conn.execute("PRAGMA user_version").fetchone()[0] or 0
            )
        return {"integrity": integrity, "tables": int(count),
                "schema_version": schema_version}
    finally:
        conn.close()


@dataclass
class SnapshotEntry:
    """One file inside a snapshot staging dir."""

    rel: str
    path: str
    kind: str  # "database" | "file" | "vault"
    size: int
    sha256: str
    schema_version: int = 0
    tables: int = 0
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel, "path": self.path, "kind": self.kind,
            "size": self.size, "sha256": self.sha256,
            "schema_version": self.schema_version, "tables": self.tables,
            "status": self.status,
        }


@dataclass
class SnapshotResult:
    """A verified local snapshot, staged and ready to protect/ship."""

    backup_id: str
    staging_dir: Path
    created_at: float
    entries: list[SnapshotEntry] = field(default_factory=list)
    plan: BackupPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "staging_dir": str(self.staging_dir),
            "created_at": self.created_at,
            "entries": [e.to_dict() for e in self.entries],
        }


def snapshot_sqlite_file(src: Path, dest: Path) -> SnapshotEntry:
    """Online-backup ``src`` -> ``dest``, verified, atomic publish.

    The destination is written to a ``.tmp`` sibling first and renamed
    into place, so a Termux kill / disk-full mid-copy never leaves a
    half-written snapshot masquerading as a generation.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        source = sqlite3.connect(str(src))
        try:
            destination = sqlite3.connect(str(tmp))
            try:
                source.backup(destination, pages=0)
            finally:
                destination.close()
        finally:
            source.close()
    except sqlite3.Error as exc:
        tmp.unlink(missing_ok=True)
        raise CorruptBackupError(
            f"snapshot of {src} failed (not a readable SQLite database): "
            f"{exc} — not shipped, not counted"
        ) from exc
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:  # ENOSPC
            raise DiskFullError(
                f"disk full while snapshotting {src}: previous generation "
                "intact, this run aborted"
            ) from exc
        raise
    info = _describe_db(tmp)
    if info["integrity"] != "ok":
        tmp.unlink(missing_ok=True)
        raise CorruptBackupError(
            f"snapshot of {src} failed integrity_check: {info['integrity']} "
            "— not shipped, not counted"
        )
    tmp.rename(dest)
    return SnapshotEntry(
        rel=dest.name, path=str(dest), kind="database",
        size=dest.stat().st_size, sha256=_sha256_file(dest),
        schema_version=info["schema_version"], tables=info["tables"],
    )


def snapshot_databases(
    root: Path, staging: Path, plan: BackupPlan
) -> list[SnapshotEntry]:
    """Snapshot every present database in the plan via the backup API."""
    entries: list[SnapshotEntry] = []
    db_dir = staging / "db"
    for item in plan.databases:
        if item.get("status") != "present":
            if item.get("required"):
                raise CorruptBackupError(
                    f"required database missing: {item['rel']}"
                )
            _log.warning("backup: optional database absent: %s", item["rel"])
            continue
        src = root / item["rel"]
        safe = item["rel"].replace("/", "__")
        entry = snapshot_sqlite_file(src, db_dir / safe)
        entry.rel = item["rel"]
        entries.append(entry)
    return entries


def snapshot_files(
    root: Path, staging: Path, plan: BackupPlan
) -> list[SnapshotEntry]:
    """Copy planned plain + vault files into staging (preserving rel path)."""
    entries: list[SnapshotEntry] = []
    for item in list(plan.files) + list(plan.vault_files):
        src = root / item["rel"]
        if not src.is_file():
            continue
        kind = "vault" if item in plan.vault_files else "file"
        dest = staging / "files" / item["rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest)
        except OSError as exc:
            if getattr(exc, "errno", None) == 28:
                raise DiskFullError(
                    f"disk full while staging {src}: previous generation "
                    "intact, this run aborted"
                ) from exc
            raise
        entries.append(SnapshotEntry(
            rel=item["rel"], path=str(dest), kind=kind,
            size=dest.stat().st_size, sha256=_sha256_file(dest),
        ))
    return entries


def snapshot_via_backup_manager(db: Any, directory: str | os.PathLike[str],
                                *, label: str = "coordinator") -> Any:
    """Delegate a single live-DB snapshot to storage/backup.py.

    Wires into :class:`~nomorals.storage.backup.BackupManager` (SQLite
    online backup API, gzip, SHA-256, manifest, rotation) instead of
    re-implementing it. Returns the organ's ``BackupInfo``.
    """
    from ..storage.backup import BackupManager

    return BackupManager(db, directory).create(label=label)


def snapshot_via_replicator(db: Any, replica_dir: str | os.PathLike[str],
                            *, name: str = "replica") -> Any:
    """Delegate continuous replication to storage/replication.py.

    Wires into :class:`~nomorals.storage.replication.SqliteReplicator`
    (copy 1 of 3-2-1: continuous local replication). Returns the organ's
    ``SnapshotInfo``.
    """
    from ..storage.replication import SqliteReplicator

    return SqliteReplicator(db, replica_dir, name=name).snapshot()


def take_snapshot(
    root: str | os.PathLike[str],
    work_dir: str | os.PathLike[str],
    *,
    label: str = "",
) -> SnapshotResult:
    """Plan + snapshot + verify one generation into a fresh staging dir."""
    root = Path(root).expanduser()
    work_dir = Path(work_dir).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    backup_id = f"backup-{stamp}" + (f"-{label}" if label else "")
    staging = work_dir / backup_id
    if staging.exists():
        staging = work_dir / f"{backup_id}-{os.getpid()}"
        backup_id = staging.name
    staging.mkdir(parents=True)
    plan = build_plan(root)
    entries = snapshot_databases(root, staging, plan)
    entries.extend(snapshot_files(root, staging, plan))
    result = SnapshotResult(
        backup_id=backup_id, staging_dir=staging,
        created_at=time.time(), entries=entries, plan=plan,
    )
    _log.info("backup snapshot %s: %d entries staged at %s",
              backup_id, len(entries), staging)
    return result
