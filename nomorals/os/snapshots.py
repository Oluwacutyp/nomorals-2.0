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

import errno
import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.filelock import lock_exclusive_nb, unlock as _unlock_file
from ..core.logging_setup import get_logger

__all__ = [
    "Snapshot",
    "SnapshotError",
    "SnapshotLockedError",
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


class SnapshotLockedError(SnapshotRefused):
    """The live database file is held open (the WinError 32 shape).

    A refusal, not a corruption: the restore did not run because a
    process holds the database file. The message names the holder the
    state lock can identify — or says plainly that the holder sits
    outside it — so the operator stops the right thing instead of
    retrying the restore blind."""


def _is_file_in_use(exc: BaseException) -> bool:
    """True when `exc` is the file-in-use shape rather than a real
    failure: Windows WinError 32/33 (sharing violation / lock), or a
    permission/busy errno from a holder. Mirrors the classification
    the updater's deferral path already uses."""
    if getattr(exc, "winerror", None) in (32, 33):
        return True
    if isinstance(exc, PermissionError):
        return True
    return isinstance(exc, OSError) and exc.errno in (
        errno.EACCES, errno.EBUSY, errno.EPERM)


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
            "tags": list(self.manifest.get("tags", []) or []),
        }


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, n))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TB"  # pragma: no cover - unreachable


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


def _locking_backend_available() -> bool:
    """True when ``fcntl`` (POSIX) or ``msvcrt`` (Windows) can be imported.

    Used by the restore path to fail closed: without a locking backend
    we cannot prove no other process is restoring concurrently.
    """
    try:
        import fcntl  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import msvcrt  # noqa: F401
        return True
    except ImportError:
        return False


def system_state(home: Path, db_path: Path) -> dict[str, bool]:
    """Report whether the live system is running and/or dirty.

    *running* — another process currently holds the exclusive
    ``nomorals.lock`` in ``home``.  *dirty* — the database has
    un-checkpointed WAL/SHM sidecars.
    """
    running = False
    lock_path = home / LOCK_FILENAME
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if lock_exclusive_nb(fd):
                _unlock_file(fd)
            else:
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
        #: Bounded wait for a transient file-in-use failure during the
        #: restore swap (see :meth:`_replace_file`). Attributes, not
        #: constants, so tests can shrink the window.
        self.swap_retry_s = 10.0
        self.swap_retry_interval_s = 0.25

    # ── create ─────────────────────────────────────────────────────────────

    def create(self, label: str = "") -> Snapshot:
        """Record a new snapshot of the current live state."""
        return self._create(label=label, tags=())

    def _create(self, *, label: str = "",
                tags: tuple[str, ...] | list[str] = ()) -> Snapshot:
        """Record a new snapshot of the current live state, with tags."""
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
            "tags": sorted(set(tags)),
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
        # Creation order, not filesystem order: snapshot ids created within
        # the same second sort by their random suffix otherwise.
        snaps.sort(key=lambda s: (s.created_at, s.id))
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

    # ── tags & retention ─────────────────────────────────────────────────
    def _write_manifest(self, snap: Snapshot) -> None:
        assert snap.path is not None
        (snap.path / MANIFEST_FILENAME).write_text(
            json.dumps(snap.manifest, indent=2), encoding="utf-8")

    def tag(self, snapshot_id: str, *tags: str) -> Snapshot:
        """Add Timeshift-style tags to a snapshot (idempotent)."""
        snap = self.get(snapshot_id)
        current = set(snap.manifest.get("tags", []) or [])
        current.update(t for t in tags if t)
        snap.manifest["tags"] = sorted(current)
        self._write_manifest(snap)
        return snap

    def untag(self, snapshot_id: str, *tags: str) -> Snapshot:
        """Remove tags from a snapshot."""
        snap = self.get(snapshot_id)
        current = set(snap.manifest.get("tags", []) or [])
        current.difference_update(tags)
        snap.manifest["tags"] = sorted(current)
        self._write_manifest(snap)
        return snap

    def with_tag(self, tag: str) -> list[Snapshot]:
        """All snapshots carrying ``tag``."""
        return [s for s in self.list()
                if tag in (s.manifest.get("tags", []) or [])]

    def create_tagged(self, label: str = "", *tags: str) -> Snapshot:
        """Create a snapshot already carrying tags (e.g. "pre-update")."""
        return self._create(label=label, tags=tags)

    def prune(self, *, keep_last: int = 5,
              keep_tags: tuple[str, ...] | list[str] = ("pre-update",),
              dry_run: bool = False) -> list[str]:
        """Retention policy: keep the newest ``keep_last`` snapshots plus
        every snapshot carrying one of ``keep_tags``; delete the rest.

        Returns the ids that were (or, with ``dry_run=True``, would be)
        deleted.
        """
        keep_tags = set(keep_tags)
        snaps = self.list()
        keep: set[str] = set()
        for snap in snaps[-max(0, int(keep_last)):]:
            keep.add(snap.id)
        for snap in snaps:
            if keep_tags & set(snap.manifest.get("tags", []) or []):
                keep.add(snap.id)
        victims = [s.id for s in snaps if s.id not in keep]
        if not dry_run:
            for snap_id in victims:
                try:
                    self.delete(snap_id)
                except Exception:  # noqa: BLE001 — prune keeps going
                    _log.warning("prune could not delete %s", snap_id,
                                 exc_info=True)
        return victims

    def prune_pre_update(self, *, keep: int = 3,
                         protect: tuple[str, ...] | list[str] = (),
                         ) -> list[str]:
        """Bound the pre-update snapshot pile (disk discipline,
        2026-11-08).

        A snapshot created by the updater carries the label
        "pre-update" and exists to roll back ONE update.  Left
        alone the pile grows by a full database + blob-store copy
        per update until the disk is full — and the update itself
        is what discovers it, failing at its own snapshot step.
        Keep the newest ``keep`` pre-update snapshots (the
        in-flight update's, a deferred restore's, one spare) plus
        any id in ``protect`` — the snapshot a pending
        deferred-restore marker names is a rollback anchor, not
        clutter, however old it is.  Snapshots with any other
        label (manual takes, pre/post pairs) are not this pile
        and are never touched here.  Deletion failures are
        logged, never raised: retention must not fail the update
        that triggered it.  Returns the deleted ids, oldest
        first."""
        snaps = [s for s in self.list() if s.label == "pre-update"]
        keep_ids = {s.id for s in snaps[-max(0, int(keep)):]}
        keep_ids.update(str(p) for p in protect if p)
        victims = [s.id for s in snaps if s.id not in keep_ids]
        deleted: list[str] = []
        for snap_id in victims:
            try:
                self.delete(snap_id)
                deleted.append(snap_id)
            except Exception:  # noqa: BLE001 — retention keeps going
                _log.warning("prune_pre_update could not delete %s",
                             snap_id, exc_info=True)
        return deleted

    # ── diff ─────────────────────────────────────────────────────────────
    def diff(self, old_id: str, new_id: str) -> dict[str, Any]:
        """What changed between two snapshots: files added / removed /
        changed (by checksum), blob count delta, size delta."""
        old = self.get(old_id)
        new = self.get(new_id)
        old_files = old.manifest.get("files", {})
        new_files = new.manifest.get("files", {})
        added = sorted(set(new_files) - set(old_files))
        removed = sorted(set(old_files) - set(new_files))
        changed = sorted(
            name for name in set(old_files) & set(new_files)
            if old_files[name].get("sha256") != new_files[name].get("sha256")
            and name != BLOBS_DIRNAME)
        return {
            "old": old.id,
            "new": new.id,
            "added": added,
            "removed": removed,
            "changed": changed,
            "size_delta_bytes": int(new.manifest.get("total_bytes", 0))
            - int(old.manifest.get("total_bytes", 0)),
            "old_tags": list(old.manifest.get("tags", []) or []),
            "new_tags": list(new.manifest.get("tags", []) or []),
        }

    # ── export / import ──────────────────────────────────────────────────
    def export_tar(self, snapshot_id: str, dest: str | os.PathLike[str],
                   *, compress: str = "gz") -> Path:
        """Pack a snapshot into a portable tarball for off-machine backup."""
        import tarfile

        snap = self.get(snapshot_id)
        assert snap.path is not None
        dest = Path(dest)
        mode = {"gz": "w:gz", "bz2": "w:bz2", "": "w"}.get(compress, "w:gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, mode) as tar:
            tar.add(snap.path, arcname=snap.id)
        _log.info("snapshot %s exported to %s", snap.id, dest)
        return dest

    def import_tar(self, tarball: str | os.PathLike[str]) -> Snapshot:
        """Import a snapshot from an :meth:`export_tar` tarball."""
        import tarfile

        self.root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:*") as tar:
            members = tar.getmembers()
            top = {m.name.split("/")[0] for m in members if m.name}
            if len(top) != 1:
                raise SnapshotError("tarball must contain exactly one snapshot")
            snap_id = next(iter(top))
            if (self.root / snap_id).exists():
                raise SnapshotError(f"snapshot {snap_id} already exists")
            tar.extractall(self.root, filter="data")
        snap = self.get(snap_id)
        problems = self.verify(snap.id)
        if problems:
            raise SnapshotError(
                f"imported snapshot failed verification: {'; '.join(problems)}")
        _log.info("snapshot %s imported", snap.id)
        return snap

    # ── pre/post pairs ───────────────────────────────────────────────────
    @contextmanager
    def pre_post(self, label: str = "", *, tag: str = "auto"):
        """Take a pre snapshot, run the block, take a post snapshot.

        Yields a dict ``{"pre": Snapshot, "post": Snapshot | None}`` — the
        post entry is filled when the block exits.  On exception the post
        snapshot is still taken (tagged ``"failed"``) so the damage is
        reviewable, then the exception propagates.  Timeshift-style
        pre/post pairs around risky operations.
        """
        holder: dict[str, Snapshot | None] = {
            "pre": self._create(label=f"{label} (pre)".strip(),
                                tags=(tag, "pre")),
            "post": None,
        }
        try:
            yield holder
        except Exception:
            holder["post"] = self._create(
                label=f"{label} (post)".strip(), tags=(tag, "post", "failed"))
            raise
        holder["post"] = self._create(label=f"{label} (post)".strip(),
                                      tags=(tag, "post"))

    # ── presentation ─────────────────────────────────────────────────────
    def render_list(self) -> str:
        """Plain-text snapshot table."""
        snaps = self.list()
        lines = [f"snapshots ({len(snaps)})"]
        for snap in snaps:
            created = time.strftime("%Y-%m-%d %H:%M",
                                    time.localtime(snap.created_at or 0))
            size = _human_bytes(snap.size_bytes)
            tags = snap.manifest.get("tags", []) or []
            tag_s = f" [{','.join(tags)}]" if tags else ""
            label = f" — {snap.label}" if snap.label else ""
            lines.append(f"  {snap.id}  {created}  {size}{tag_s}{label}")
        return "\n".join(lines)

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
            except OSError as exc:
                if _is_file_in_use(exc):
                    raise SnapshotLockedError(self._locked_message(
                        exc, running=state["running"])) from exc
                raise
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
        if not _locking_backend_available():
            # Fail closed: without a locking backend we cannot prove no
            # other process is restoring concurrently.
            raise SnapshotRefused(
                "no file-locking backend available on this platform; "
                "refusing to restore without the state lock"
            )
        lock_path = self.home / LOCK_FILENAME
        self.home.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.time() + timeout
        try:
            while True:
                if lock_exclusive_nb(fd):
                    return fd
                if time.time() >= deadline:
                    raise SnapshotRefused(
                        "another process holds the state lock; "
                        "stop the running system first"
                    )
                time.sleep(0.1)
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _release_lock(fd: int) -> None:
        try:
            _unlock_file(fd)
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

    def _replace_file(self, src: Path, dst: Path) -> None:
        """``os.replace`` with a bounded retry for file-in-use failures.

        On Windows, replacing a file a process still holds open raises
        WinError 32 where POSIX would simply succeed — the failure the
        deferred boot restore kept paging on the EC2 host. A holder
        that is on its way out (the previous Devon still finishing its
        stop, a scanner releasing the file) frees it within seconds,
        so the swap waits it out briefly instead of failing the whole
        restore on the race. A holder that stays surfaces as the
        error itself, and :meth:`restore` names it.
        """
        deadline = time.time() + max(0.0, self.swap_retry_s)
        while True:
            try:
                os.replace(src, dst)
                return
            except OSError as exc:
                if not _is_file_in_use(exc) or time.time() >= deadline:
                    raise
                time.sleep(self.swap_retry_interval_s)

    def _locked_message(self, exc: BaseException, *,
                        running: bool) -> str:
        """Name the holder of the locked live database, as far as the
        state lock can identify it — the sentence the boot page and
        the restore report carry instead of a bare WinError 32.

        ``running`` is the state probe taken BEFORE this restore took
        the state lock: afterwards the probe would see our own lock
        and blame a phantom second process."""
        if running:
            holder = ("another Devon process is running and holds it "
                      "(the state lock is held) — stop that process "
                      "fully, then restore")
        else:
            holder = ("no Devon state lock is held, so the holder "
                      "sits outside it — a leftover Devon process, an "
                      "antivirus or backup scan, or this process's "
                      "own open database connection — close it, then "
                      "restore")
        return (f"cannot restore: the live database {self.db_path} "
                f"is locked ({type(exc).__name__}: {exc}); {holder}")

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
                self._replace_file(live, backup)
                moved.append((live, backup))
            if staged.is_dir():
                if live.exists():  # pragma: no cover - defensive
                    shutil.rmtree(live)
                shutil.copytree(staged, live, symlinks=False)
            else:
                self._replace_file(staged, live)

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
