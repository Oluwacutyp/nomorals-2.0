"""Continuous SQLite replication, Litestream-style, with no new dependencies.

Litestream's insight: a SQLite database is a single file, so continuous
replication can be dead simple — take consistent snapshots on a schedule, keep
a window of generations, ship them to cheap storage. This module implements
that pattern natively with the ``sqlite3`` backup API:

* :class:`SqliteReplicator` — snapshots a live :class:`~nomorals.storage.db.Database`
  via the online backup API (consistent while writers keep running), verifies
  each snapshot with ``PRAGMA integrity_check``, rotates old generations, and
  optionally ships every snapshot through a ``ship`` callable (e.g. upload to
  an :class:`~nomorals.storage.s3blob.S3BlobStore` — the S3 side of Litestream).
* Point-in-time recovery: ``keep`` generations of timestamped snapshots give
  "restore to last Tuesday" without keeping every snapshot forever.
* ``run_loop`` runs the whole thing on a daemon thread; RPO = the interval.

This is deliberately *not* a WAL-frame shipper: per-interval snapshots are
simpler, need no sidecar process, and at agent scale (megabytes, not
terabytes) the copy cost is negligible. The honesty trade is documented on
:meth:`SqliteReplicator.snapshot`: recovery granularity is one snapshot.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import StorageError
from ..core.logging_setup import get_logger
from .db import Database

__all__ = ["SnapshotInfo", "SqliteReplicator"]

_log = get_logger(__name__)

CHUNK = 1024 * 1024
SIDECAR_SUFFIX = ".replica.json"


@dataclass
class SnapshotInfo:
    name: str
    path: str
    size: int
    sha256: str
    pages: int
    created_at: float
    source_path: str = ""
    schema_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "pages": self.pages,
            "created_at": self.created_at,
            "source_path": self.source_path,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotInfo:
        return cls(
            name=data["name"],
            path=data["path"],
            size=int(data.get("size", 0)),
            sha256=data.get("sha256", ""),
            pages=int(data.get("pages", 0)),
            created_at=float(data.get("created_at", 0.0)),
            source_path=data.get("source_path", ""),
            schema_version=int(data.get("schema_version", 0)),
        )


class SqliteReplicator:
    """Snapshot-based continuous replication for a SQLite database.

        replicator = SqliteReplicator(db, "data/replicas", keep=7,
                                      ship=lambda p: s3.put_file(p, move=False))
        replicator.replicate_once()          # snapshot + rotate + ship
        # …or run forever on a daemon thread:
        replicator.run_loop(60, should_stop=stop_event.is_set)
    """

    def __init__(
        self,
        db: Database,
        replica_dir: str | os.PathLike[str],
        *,
        name: str = "replica",
        keep: int = 7,
        ship: Callable[[Path], None] | None = None,
    ) -> None:
        self.db = db
        self.replica_dir = Path(replica_dir).expanduser()
        self.replica_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.keep = max(1, int(keep))
        self.ship = ship
        self.stats = {"snapshots": 0, "rotated": 0, "shipped": 0, "errors": 0}

    # ── snapshots ──────────────────────────────────────────────────────────
    def snapshot(self) -> SnapshotInfo:
        """Take a consistent snapshot of the live database.

        Uses the SQLite online backup API (never ``shutil.copy`` on a live
        WAL database), checkpoints first so the snapshot is self-contained,
        then verifies the copy with ``PRAGMA integrity_check`` before it is
        accepted. Recovery granularity is one snapshot: RPO = snapshot
        interval.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        target = self.replica_dir / f"{self.name}-{stamp}.db"
        if target.exists():  # two snapshots inside the same second
            target = self.replica_dir / f"{self.name}-{stamp}-{os.getpid()}.db"

        with contextlib.suppress(StorageError):  # not in WAL mode; nothing to do
            self.db.checkpoint("TRUNCATE")
        source = self.db._connection()  # noqa: SLF001 - intentional: same-process snapshot
        destination = sqlite3.connect(str(target))
        try:
            source.backup(destination, pages=0)
        finally:
            destination.close()

        info = self._describe(target)
        sidecar = target.with_suffix(target.suffix + SIDECAR_SUFFIX)
        sidecar.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
        self.stats["snapshots"] += 1
        _log.info("replica snapshot: %s (%d bytes, %d pages)", info.name, info.size, info.pages)
        return info

    def _describe(self, target: Path) -> SnapshotInfo:
        pages = 0
        schema_version = 0
        conn = sqlite3.connect(str(target))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                target.unlink(missing_ok=True)
                raise StorageError(f"replica snapshot failed integrity_check: {result}")
            pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
            try:
                row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
                schema_version = int(row[0] or 0)
            except sqlite3.Error:
                schema_version = 0
        finally:
            conn.close()
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
        return SnapshotInfo(
            name=target.name,
            path=str(target),
            size=target.stat().st_size,
            sha256=digest.hexdigest(),
            pages=pages,
            created_at=time.time(),
            source_path=str(self.db.path) if self.db.path else ":memory:",
            schema_version=schema_version,
        )

    # ── listing / rotation ─────────────────────────────────────────────────
    def list_snapshots(self) -> list[SnapshotInfo]:
        """Snapshots on disk that still have a valid sidecar, oldest first."""
        out: list[SnapshotInfo] = []
        for sidecar in sorted(self.replica_dir.glob(f"{self.name}-*{SIDECAR_SUFFIX}")):
            target = Path(str(sidecar)[: -len(SIDECAR_SUFFIX)])
            if not target.is_file():
                continue
            try:
                info = SnapshotInfo.from_dict(
                    json.loads(sidecar.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                _log.warning("unreadable replica sidecar: %s", sidecar.name)
                continue
            info.path = str(target)
            out.append(info)
        return sorted(out, key=lambda s: s.created_at)

    def latest(self) -> SnapshotInfo | None:
        snapshots = self.list_snapshots()
        return snapshots[-1] if snapshots else None

    def rotate(self) -> list[str]:
        """Delete the oldest snapshots beyond ``keep``. Returns removed names."""
        snapshots = self.list_snapshots()
        if len(snapshots) <= self.keep:
            return []
        removed: list[str] = []
        for info in snapshots[: len(snapshots) - self.keep]:
            Path(info.path).unlink(missing_ok=True)
            Path(info.path + SIDECAR_SUFFIX).unlink(missing_ok=True)
            removed.append(info.name)
        if removed:
            self.stats["rotated"] += len(removed)
            _log.info("rotated %d old replica snapshots", len(removed))
        return removed

    # ── restore ────────────────────────────────────────────────────────────
    def restore(
        self,
        snapshot: SnapshotInfo | str | os.PathLike[str] | None = None,
        *,
        target: str | os.PathLike[str] | None = None,
    ) -> Path:
        """Copy a snapshot to ``target`` (default: ``<name>.restored.db``)."""
        if snapshot is None:
            snapshot = self.latest()
            if snapshot is None:
                raise StorageError("no replica snapshots available")
        if isinstance(snapshot, (str, os.PathLike)):
            wanted = Path(str(snapshot)).name
            matches = [s for s in self.list_snapshots() if s.name == wanted]
            if not matches:
                raise StorageError(f"replica snapshot not found: {snapshot}")
            snapshot = matches[-1]
        source = Path(snapshot.path)
        if not source.is_file():
            raise StorageError(f"replica snapshot missing on disk: {source}")
        destination = Path(target) if target else self.replica_dir / f"{self.name}.restored.db"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _log.info("restored replica %s -> %s", snapshot.name, destination)
        return destination

    # ── shipping ───────────────────────────────────────────────────────────
    def _ship(self, info: SnapshotInfo) -> None:
        if self.ship is None:
            return
        try:
            self.ship(Path(info.path))
        except Exception as exc:
            self.stats["errors"] += 1
            _log.error("replica ship failed for %s: %s", info.name, exc)
            raise StorageError(f"replica ship failed for {info.name}: {exc}") from exc
        self.stats["shipped"] += 1

    # ── one-shot & loop ────────────────────────────────────────────────────
    def replicate_once(self) -> SnapshotInfo:
        """Snapshot, rotate old generations, and ship the new snapshot."""
        info = self.snapshot()
        self.rotate()
        self._ship(info)
        return info

    def run_loop(
        self,
        interval_seconds: float,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Blocking loop; intended to run on a daemon thread."""
        interval = max(1.0, float(interval_seconds))
        while should_stop is None or not should_stop():
            try:
                self.replicate_once()
            except Exception as exc:  # noqa: BLE001 - a replicator must never die
                self.stats["errors"] += 1
                _log.error("replication cycle failed: %s", exc)
            waited = 0.0
            while waited < interval:
                if should_stop is not None and should_stop():
                    return
                time.sleep(min(5.0, interval - waited))
                waited += 5.0

    def stats_snapshot(self) -> dict[str, Any]:
        snapshots = self.list_snapshots()
        return {
            **self.stats,
            "generations": len(snapshots),
            "bytes": sum(s.size for s in snapshots),
            "latest": snapshots[-1].to_dict() if snapshots else None,
            "replica_dir": str(self.replica_dir),
            "keep": self.keep,
        }
