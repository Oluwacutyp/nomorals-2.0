"""Point-in-time snapshots of Devon's live state.

A snapshot is a directory under ``<state>/snapshots/<id>/`` containing:

* ``db.sqlite`` — a consistent copy of the main state database, taken with
  ``VACUUM INTO`` so it is valid even while the database is open.
* ``blobs/`` — a copy of the blob store directory (artifact payloads).
* ``config.json`` — the resolved settings plus the raw config file when known.
* ``manifest.json`` — id, timestamps, checksums, schema version, sizes.

Restore is transactional: the snapshot is staged into a temp directory,
verified (checksums + ``PRAGMA integrity_check``), and only then swapped
into place.  If anything fails mid-swap the previous state is moved back,
so the system is never left half-restored.

Restore refuses to run over a live or dirty system unless ``force=True``:

* *running* — another process holds the exclusive lock on
  ``<state>/nomorals.lock``.
* *dirty* — the database has un-checkpointed WAL/SHM sidecar files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "Snapshot",
    "SnapshotError",
    "SnapshotRefused",
    "SnapshotManager",
    "system_state",
]

_log = get_logger(__name__)

SNAPSHOT_DIRNAME = "snapshots"
LOCK_FILENAME = "nomorals.lock"
MANIFEST_FILENAME = "manifest.json"
DB_FILENAME = "db.sqlite"
BLOBS_DIRNAME = "blobs"
CONFIG_FILENAME = "config.json"


class SnapshotError(Exception):
    """Base error for snapshot operations."""


class SnapshotRefused(SnapshotError):
    """Restore refused: the live system is running or dirty."""


@dataclass
class Snapshot:
    """One recorded snapshot."""

    id: str
    label: str = ""
    created_at: float = 0.0
    path: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return int(self.manifest.get("total_bytes", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "created_at": self.created_at,
            "path": str(self.path) if self.path else "",
            "size_bytes": self.size_bytes,
            "schema_version": self.manifest.get("schema_version"),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:  # pragma: no cover - raced deletion
                continue
    return total


def system_state(home: Path, db_path: Path) -> dict[str, bool]:
    """Report whether the live system is running and/or dirty.

    *running* — another process currently holds the exclusive
    ``nomorals.lock`` in ``home``.  *dirty* — the database has
    un-checkpointed WAL/SHM sidecars.
    """
    running = False
    lock_path = home / LOCK_FILENAME
    try:
        import fcntl

        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            running = True
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - cannot probe; treat as unknown-safe
        _log.debug("could not probe %s", lock_path, exc_info=True)

    dirty = False
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(db_path) + suffix)
        try:
            if sidecar.exists() and sidecar.stat().st_size > 0:
                dirty = True
                break
        except OSError:  # pragma: no cover - raced deletion
            continue
    return {"running": running, "dirty": dirty}


def _new_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    rand = hashlib.sha256(os.urandom(8)).hexdigest()[:6]
    return f"{ts}-{rand}"


class SnapshotManager:
    """Create, verify, list, restore and delete state snapshots."""

    def __init__(
        self,
        home: str | os.PathLike[str],
        db_path: str | os.PathLike[str],
        *,
        blob_dir: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        settings_dict: dict[str, Any] | None = None,
    ) -> None:
        self.home = Path(home)
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir) if blob_dir else None
        self.config_path = Path(config_path) if config_path else None
        self.settings_dict = settings_dict or {}
        self.root = self.home / SNAPSHOT_DIRNAME

    # ── create ─────────────────────────────────────────────────────────────

    def create(self, label: str = "") -> Snapshot:
        """Record a new snapshot of the current live state."""
        snap_id = _new_id()
        dest = self.root / snap_id
        dest.mkdir(parents=True, exist_ok=False)

        files: dict[str, dict[str, Any]] = {}

        # 1. database — VACUUM INTO gives a consistent copy even while open.
        db_dest = dest / DB_FILENAME
        self._copy_database(db_dest)
        files[DB_FILENAME] = {
            "sha256": _sha256_file(db_dest),
            "bytes": db_dest.stat().st_size,
        }

        # 2. blob store.
        if self.blob_dir is not None and self.blob_dir.exists():
            blobs_dest = dest / BLOBS_DIRNAME
            shutil.copytree(self.blob_dir, blobs_dest, symlinks=False)
            files[BLOBS_DIRNAME] = {
                "sha256": "",  # directory: checksum lives per-blob
                "bytes": _dir_size(blobs_dest),
            }

        # 3. config.
        config_payload = {
            "settings": self.settings_dict,
            "db_path": str(self.db_path),
            "blob_dir": str(self.blob_dir) if self.blob_dir else "",
            "config_source": str(self.config_path) if self.config_path else "",
        }
        config_dest = dest / CONFIG_FILENAME
        config_dest.write_text(
            json.dumps(config_payload, indent=2, default=str), encoding="utf-8"
        )
        if self.config_path is not None and self.config_path.exists():
            raw_dest = dest / "config.raw"
            shutil.copy2(self.config_path, raw_dest)
            files["config.raw"] = {
                "sha256": _sha256_file(raw_dest),
                "bytes": raw_dest.stat().st_size,
            }
        files[CONFIG_FILENAME] = {
            "sha256": _sha256_file(config_dest),
            "bytes": config_dest.stat().st_size,
        }

        manifest = {
            "id": snap_id,
            "label": label,
            "created_at": time.time(),
            "nomorals_version": self._version(),
            "schema_version": self._schema_version(db_dest),
            "files": files,
            "total_bytes": sum(f["bytes"] for f in files.values()),
        }
        (dest / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        _log.info("snapshot %s created (%s bytes)", snap_id, manifest["total_bytes"])
        return Snapshot(
            id=snap_id,
            label=label,
            created_at=manifest["created_at"],
            path=dest,
            manifest=manifest,
        )

    def _copy_database(self, dest: Path) -> None:
        src = str(self.db_path)
        if not self.db_path.exists():
            # No live database yet — an empty snapshot DB is still a valid
            # restore source; migrations will build the schema on boot.
            sqlite3.connect(str(dest)).close()
            return
        conn = sqlite3.connect(src)
        try:
            conn.execute("VACUUM INTO ?", (str(dest),))
        finally:
            conn.close()

    @staticmethod
    def _schema_version(db_file: Path) -> int | None:
        try:
            conn = sqlite3.connect(str(db_file))
            try:
                row = conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            return int(row[0]) if row else None
        except Exception:  # noqa: BLE001 - best effort metadata
            return None

    @staticmethod
    def _version() -> str:
        try:
            from ..version import __version__

            return str(__version__)
        except Exception:  # noqa: BLE001 - never break snapshots on metadata
            return "unknown"

    # ── list / get / delete ────────────────────────────────────────────────

    def list(self) -> list[Snapshot]:
        snaps: list[Snapshot] = []
        if not self.root.exists():
            return snaps
        for child in sorted(self.root.iterdir()):
            manifest_path = child / MANIFEST_FILENAME
            if not child.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            snaps.append(
                Snapshot(
                    id=manifest.get("id", child.name),
                    label=manifest.get("label", ""),
                    created_at=float(manifest.get("created_at", 0.0)),
                    path=child,
                    manifest=manifest,
                )
            )
        return snaps

    def get(self, snapshot_id: str) -> Snapshot:
        for snap in self.list():
            if snap.id == snapshot_id or snap.id.startswith(snapshot_id):
                return snap
        raise SnapshotError(f"no snapshot {snapshot_id!r}")

    def latest(self) -> Snapshot | None:
        snaps = self.list()
        return snaps[-1] if snaps else None

    def delete(self, snapshot_id: str) -> None:
        snap = self.get(snapshot_id)
        assert snap.path is not None
        shutil.rmtree(snap.path, ignore_errors=False)
        _log.info("snapshot %s deleted", snap.id)

    # ── verify ─────────────────────────────────────────────────────────────

    def verify(self, snapshot_id: str) -> list[str]:
        """Check a snapshot's integrity. Returns a list of problems (empty = ok)."""
        snap = self.get(snapshot_id)
        assert snap.path is not None
        problems: list[str] = []
        manifest = snap.manifest
        files = manifest.get("files", {})
        for name, meta in files.items():
            target = snap.path / name
            if not target.exists():
                problems.append(f"missing file: {name}")
                continue
            if name == BLOBS_DIRNAME:
                continue  # directory; blobs are content-addressed already
            expected = meta.get("sha256", "")
            if expected and _sha256_file(target) != expected:
                problems.append(f"checksum mismatch: {name}")
        db_file = snap.path / DB_FILENAME
        if db_file.exists():
            try:
                conn = sqlite3.connect(str(db_file))
                try:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                finally:
                    conn.close()
                if not result or str(result[0]).lower() != "ok":
                    problems.append(f"db integrity_check failed: {result}")
            except Exception as exc:  # noqa: BLE001 - report, don't raise
                problems.append(f"db integrity_check error: {exc}")
        else:
            problems.append("missing db.sqlite")
        return problems

    # ── restore ────────────────────────────────────────────────────────────

    def restore(self, snapshot_id: str, *, force: bool = False) -> Snapshot:
        """Restore a snapshot transactionally.

        The snapshot is staged into a temp dir and verified first; the live
        files are only swapped after verification passes.  If the swap fails
        partway, the previous live files are moved back.
        """
        snap = self.get(snapshot_id)
        assert snap.path is not None
        problems = self.verify(snap.id)
        if problems:
            raise SnapshotError(
                f"snapshot {snap.id} failed verification: {'; '.join(problems)}"
            )

        state = system_state(self.home, self.db_path)
        if not force and (state["running"] or state["dirty"]):
            why = " and ".join(
                k for k, v in state.items() if v
            )
            raise SnapshotRefused(
                f"live system is {why}; stop it first or pass force=True"
            )

        # Hold the lock for the whole restore so a second restorer (or a
        # booting system) cannot interleave with us.
        lock_fd = self._acquire_lock()
        try:
            stage = self.home / f".restore-{snap.id}"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            try:
                self._stage_snapshot(snap, stage)
                self._swap_into_place(stage)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        finally:
            self._release_lock(lock_fd)

        _log.info("snapshot %s restored", snap.id)
        return snap

    def _acquire_lock(self, timeout: float = 30.0) -> int:
        """Take the restore lock, waiting up to ``timeout`` seconds.

        Serialises concurrent restores instead of interleaving them, but
        never hangs forever behind a live server: ``--force`` overrides a
        *dirty* database, never a *running* system writing to it.
        """
        import fcntl

        lock_path = self.home / LOCK_FILENAME
        self.home.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.time() + timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError as err:
                    if time.time() >= deadline:
                        raise SnapshotRefused(
                            "another process holds the state lock; "
                            "stop the running system first"
                        ) from err
                    time.sleep(0.1)
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _release_lock(fd: int) -> None:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _stage_snapshot(self, snap: Snapshot, stage: Path) -> None:
        """Copy the snapshot's payload into the staging dir, re-verified."""
        assert snap.path is not None
        for name in (DB_FILENAME, BLOBS_DIRNAME, CONFIG_FILENAME):
            src = snap.path / name
            if not src.exists():
                continue
            dest = stage / name
            if src.is_dir():
                shutil.copytree(src, dest, symlinks=False)
            else:
                shutil.copy2(src, dest)
        # Re-verify staged DB before it ever touches the live path.
        staged_db = stage / DB_FILENAME
        conn = sqlite3.connect(str(staged_db))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
        if not result or str(result[0]).lower() != "ok":
            raise SnapshotError("staged database failed integrity_check")

    @staticmethod
    def _clear_sidecars(db_path: Path) -> None:
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = Path(str(db_path) + suffix)
            try:
                if sidecar.exists():
                    sidecar.unlink()
            except OSError:
                _log.warning("could not remove sidecar %s", sidecar,
                             exc_info=True)

    def _swap_into_place(self, stage: Path) -> None:
        """Atomically swap staged files into the live paths.

        Each live target is first moved aside into a backup dir; on any
        failure the asides are moved back.  os.replace() is atomic per file.
        """
        backup_dir = self.home / f".pre-restore-{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        moved: list[tuple[Path, Path]] = []  # (live, backup)

        def swap(live: Path, staged: Path) -> None:
            backup = backup_dir / live.name
            if live.exists():
                os.replace(live, backup)
                moved.append((live, backup))
            if staged.is_dir():
                if live.exists():  # pragma: no cover - defensive
                    shutil.rmtree(live)
                shutil.copytree(staged, live, symlinks=False)
            else:
                os.replace(staged, live)

        targets = [
            (self.db_path, stage / DB_FILENAME),
        ]
        if self.blob_dir is not None:
            targets.append((self.blob_dir, stage / BLOBS_DIRNAME))
        try:
            for live, staged in targets:
                if staged.exists():
                    live.parent.mkdir(parents=True, exist_ok=True)
                    swap(live, staged)
            # A staged database is a clean VACUUM INTO copy with no WAL;
            # stale sidecars from the pre-restore database must not survive
            # or SQLite would replay them onto the restored file.
            self._clear_sidecars(self.db_path)
        except Exception:
            # Roll back: move every aside back into place, newest first.
            for live, backup in reversed(moved):
                try:
                    if live.exists():
                        if live.is_dir() and not backup.is_dir():
                            shutil.rmtree(live)
                        elif not live.is_dir():
                            live.unlink()
                    os.replace(backup, live)
                except OSError:
                    _log.error("rollback of %s failed", live, exc_info=True)
            raise
        shutil.rmtree(backup_dir, ignore_errors=True)
