"""Versioned backups with optional push to a Git repository.

Correctness comes from one decision: backups use the SQLite **online backup API**
(``sqlite3.Connection.backup``), never ``shutil.copy``. Copying a live WAL database
can produce a torn file that looks fine until you need it. The backup API takes a
consistent snapshot while writers keep running.

Each snapshot is gzipped, SHA-256 checksummed, and recorded in a manifest so a
restore can verify what it is about to load. Rotation keeps the last N, plus a
daily sample so a month of history does not cost a month of disk.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, styled_table
from .db import Database
from .failures import record_failure

__all__ = ["BackupInfo", "BackupManager", "RetentionPolicy"]

_log = get_logger(__name__)

MANIFEST = "manifest.json"
CHUNK = 1024 * 1024


@dataclass
class BackupInfo:
    name: str
    path: str
    size: int
    sha256: str
    pages: int
    created_at: float
    schema_version: int = 0
    label: str = ""
    # Collision-proof snapshot identity (F07): the reserved base name the
    # snapshot was created under. The timestamp+label prefix is metadata
    # for humans; this identity is what makes the snapshot unique.
    snapshot_id: str = ""
    # Blob sidecar (artifact blobs tarred alongside the database snapshot).
    blob_name: str = ""
    blob_sha256: str = ""
    blob_count: int = 0
    blob_files: dict[str, str] = field(default_factory=dict)  # rel path -> sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "pages": self.pages,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "label": self.label,
            "snapshot_id": self.snapshot_id,
            "blob_name": self.blob_name,
            "blob_sha256": self.blob_sha256,
            "blob_count": self.blob_count,
            "blob_files": self.blob_files,
        }


@dataclass
class RetentionPolicy:
    """Borg-style retention: keep the last N plus periodic samples.

    ``keep_last`` covers the recent window; ``keep_daily``/``keep_weekly``/
    ``keep_monthly`` each keep one backup per period *outside* the recent
    window. ``keep_within`` (seconds) keeps everything newer than the
    cutoff regardless of the other rules (borg ``--keep-within``).
    """

    keep_last: int = 14
    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 6
    keep_within: float = 0.0

    def select_keepers(self, backups: list["BackupInfo"]) -> set[str]:
        """Paths to keep, given backups oldest-first."""
        if not backups:
            return set()
        now = time.time()
        keepers: set[str] = set()
        if self.keep_within > 0:
            for backup in backups:
                if now - backup.created_at <= self.keep_within:
                    keepers.add(backup.path)
        ordered = sorted(backups, key=lambda b: b.created_at, reverse=True)
        for backup in ordered[: max(0, self.keep_last)]:
            keepers.add(backup.path)
        rest = ordered[max(0, self.keep_last):]
        # One per day, then one per ISO week, then one per month.
        for attr, count, key in (
            ("keep_daily", self.keep_daily, lambda b: time.strftime(
                "%Y-%m-%d", time.gmtime(b.created_at))),
            ("keep_weekly", self.keep_weekly, lambda b: time.strftime(
                "%G-W%V", time.gmtime(b.created_at))),
            ("keep_monthly", self.keep_monthly, lambda b: time.strftime(
                "%Y-%m", time.gmtime(b.created_at))),
        ):
            _ = attr
            seen: set[str] = set()
            kept = 0
            for backup in rest:
                if backup.path in keepers:
                    continue
                bucket = key(backup)
                if bucket in seen:
                    continue
                if kept >= count:
                    break
                seen.add(bucket)
                keepers.add(backup.path)
                kept += 1
        return keepers


@dataclass
class BackupManager:
    """Creates, verifies, rotates, restores, and ships database backups."""

    db: Database
    directory: str | os.PathLike[str]
    keep: int = 14
    compress: bool = True
    git_repo: str = ""
    git_branch: str = "backups"
    git_author: str = "NoMorals Core <nm@localhost>"
    include_blobs: bool = False
    blob_dir: str | os.PathLike[str] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Optional blob store (S3-compatible or local) receiving every scheduled
    #: backup via :meth:`push_to_s3` — the offsite leg of the 3-2-1 story.
    offsite_store: Any = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    # ── create ───────────────────────────────────────────────────────────────
    def create(self, label: str = "") -> BackupInfo:
        """Take a consistent snapshot of the database.

        Snapshot identities are collision-proof (F07): the human-readable
        ``nomorals-<stamp>[-<label>]`` prefix is metadata, not identity —
        a random suffix is appended and the name is *reserved* with
        ``O_EXCL`` before any bytes are written. Two creates in the same
        second, or two labels that sanitize to the same string, therefore
        produce distinct snapshots instead of overwriting one path while
        the manifest records both. The reserved base is recorded in the
        manifest as ``snapshot_id``.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        base = f"nomorals-{stamp}"
        if label:
            safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
            base = f"{base}-{safe}"
        raw_path, base = self._reserve_snapshot(base)
        try:
            return self._create_at(raw_path, base, label)
        except BaseException:
            # A failed create must not leave a phantom snapshot behind:
            # the reservation and any partial payloads go with it.
            raw_path.unlink(missing_ok=True)
            raw_path.with_suffix(".db.gz").unlink(missing_ok=True)
            (self.directory / f"{base}-blobs.tar.gz").unlink(missing_ok=True)
            raise

    def _reserve_snapshot(self, base: str) -> tuple[Path, str]:
        """Atomically reserve a unique snapshot base under ``base``.

        Returns ``(raw_path, unique_base)``; the raw file exists (empty)
        on return, claimed with ``O_CREAT | O_EXCL`` so no other create —
        thread or process — can take the same identity.
        """
        for _ in range(100):
            candidate = f"{base}-{secrets.token_hex(3)}"
            raw_path = self.directory / f"{candidate}.db"
            try:
                fd = os.open(raw_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                continue
            os.close(fd)
            return raw_path, candidate
        raise StorageError(f"cannot reserve a unique snapshot name for {base!r}")

    def _create_at(self, raw_path: Path, base: str, label: str) -> BackupInfo:
        """Write the snapshot into an already-reserved identity."""
        # Checkpoint first so the snapshot is self-contained (no -wal sidecar
        # needed): everything committed is in the main database file.
        with contextlib.suppress(StorageError):  # not in WAL mode; nothing to checkpoint
            self.db.checkpoint("TRUNCATE")
        source = self.db._connection()  # noqa: SLF001 - intentional: same-process snapshot
        destination = sqlite3.connect(str(raw_path))
        try:
            source.backup(destination, pages=0)
        finally:
            destination.close()

        pages = int(
            sqlite3.connect(str(raw_path)).execute("PRAGMA page_count").fetchone()[0]
        )
        schema_version = self._schema_version(raw_path)

        if self.compress:
            final_path = raw_path.with_suffix(".db.gz")
            with raw_path.open("rb") as src, gzip.open(final_path, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            raw_path.unlink()
        else:
            final_path = raw_path

        digest = _sha256_file(final_path)
        blob_name, blob_sha256, blob_count, blob_files = self._archive_blobs(base)
        info = BackupInfo(
            name=final_path.name,
            path=str(final_path),
            size=final_path.stat().st_size,
            sha256=digest,
            pages=pages,
            created_at=time.time(),
            schema_version=schema_version,
            label=label,
            snapshot_id=base,
            blob_name=blob_name,
            blob_sha256=blob_sha256,
            blob_count=blob_count,
            blob_files=blob_files,
        )
        self._append_manifest(info)
        _log.info(
            "backup created: %s (%d bytes, %d pages, %d blobs)",
            info.name, info.size, info.pages, info.blob_count,
        )
        return info

    def _schema_version(self, path: Path) -> int:
        try:
            conn = sqlite3.connect(str(path))
            try:
                row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
                return int(row[0] or 0)
            finally:
                conn.close()
        except sqlite3.Error:
            return 0

    # ── blob sidecar ─────────────────────────────────────────────────────────
    def _archive_blobs(self, base: str) -> tuple[str, str, int, dict[str, str]]:
        """Tar+gzip the artifact blob dir next to the database snapshot.

        Returns ``(archive_name, sha256, file_count, {rel_path: sha256})``.
        The per-file map is recorded in the manifest so ``verify()`` can check
        the *presence and integrity of every blob*, not just the archive file.
        """
        if not self.include_blobs or not self.blob_dir:
            return "", "", 0, {}
        source = Path(self.blob_dir).expanduser()
        if not source.is_dir():
            _log.debug("blob dir %s missing; skipping blob backup", source)
            return "", "", 0, {}
        archive = self.directory / f"{base}-blobs.tar.gz"
        files: dict[str, str] = {}
        try:
            with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
                for path in sorted(source.rglob("*")):
                    if not path.is_file() or path.is_symlink():
                        continue
                    rel = path.relative_to(source).as_posix()
                    digest = _sha256_file(path)
                    files[rel] = digest
                    tarinfo = tar.gettarinfo(str(path), arcname=rel)
                    with path.open("rb") as handle:
                        tar.addfile(tarinfo, handle)
        except OSError as exc:
            archive.unlink(missing_ok=True)
            raise StorageError(f"cannot archive blobs from {source}: {exc}") from exc
        return archive.name, _sha256_file(archive), len(files), files

    def _extract_blob_archive(self, archive: Path, dest: Path) -> Path:
        """Extract a blob tarball, refusing path-traversal entries."""
        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    name = member.name
                    if not name or name.startswith("/") or ".." in Path(name).parts:
                        raise StorageError(f"unsafe blob archive entry: {name!r}")
                    if not member.isfile():
                        continue
                    target = dest / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tar.extractfile(member)
                    if src is None:
                        raise StorageError(f"unreadable blob archive entry: {name!r}")
                    with src, target.open("wb") as out:
                        shutil.copyfileobj(src, out, CHUNK)
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise StorageError(f"cannot extract blob archive {archive.name}: {exc}") from exc
        return dest

    def _verify_blobs(self, backup: BackupInfo) -> list[str]:
        """Verify the blob sidecar: archive checksum, then every blob file.

        Extraction failures, missing blobs, and per-file checksum mismatches
        are all reported as problems — verify() must *catch* corruption, not
        just confirm the archive file exists.
        """
        problems: list[str] = []
        if not backup.blob_name:
            return problems  # backup predates blob support, or blobs disabled
        archive = self.directory / backup.blob_name
        if not archive.is_file():
            return [f"blob archive missing: {backup.blob_name}"]
        if backup.blob_sha256:
            actual = _sha256_file(archive)
            if actual != backup.blob_sha256:
                return [f"blob archive checksum mismatch for {backup.blob_name}"]
        tmpdir = Path(tempfile.mkdtemp(prefix="nm-blob-verify-"))
        try:
            try:
                self._extract_blob_archive(archive, tmpdir)
            except StorageError as exc:
                return [f"cannot extract blob archive {backup.blob_name}: {exc}"]
            for rel, digest in sorted((backup.blob_files or {}).items()):
                candidate = (tmpdir / rel).resolve()
                if tmpdir not in candidate.parents:
                    problems.append(f"unsafe blob path in manifest: {rel}")
                    continue
                if not candidate.is_file():
                    problems.append(f"blob missing from archive: {rel}")
                elif _sha256_file(candidate) != digest:
                    problems.append(f"blob corrupted (checksum mismatch): {rel}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return problems

    # ── manifest ─────────────────────────────────────────────────────────────
    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            _log.warning("backup manifest unreadable; starting fresh")
            return []

    @contextlib.contextmanager
    def _manifest_lock(self) -> Any:
        """Serialize manifest read-modify-write across threads/processes.

        The manifest is a JSON file rewritten whole on every append and
        rotation; without a lock, two simultaneous creates can each read
        the old manifest and one entry is silently lost — the manifest
        then disagrees with the snapshots on disk (F07's consistency
        half). The lock is the ``O_EXCL`` lockfile beside the manifest.
        """
        lock_path = self.directory / ".manifest.lock"
        fd: int | None = None
        deadline = time.monotonic() + 30.0
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise StorageError(
                        f"backup manifest is locked by another writer: {lock_path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            lock_path.unlink(missing_ok=True)

    def _append_manifest(self, info: BackupInfo) -> None:
        with self._manifest_lock():
            entries = self._load_manifest()
            entries.append(info.to_dict())
            self.manifest_path.write_text(
                json.dumps(entries, indent=2, default=str), encoding="utf-8"
            )
        self.history.append(info.to_dict())

    def list(self) -> list[BackupInfo]:
        """Backups recorded in the manifest that still exist on disk."""
        out: list[BackupInfo] = []
        for entry in self._load_manifest():
            path = Path(entry["path"])
            if not path.is_file():
                continue
            out.append(
                BackupInfo(
                    name=entry["name"],
                    path=str(path),
                    size=entry.get("size", path.stat().st_size),
                    sha256=entry.get("sha256", ""),
                    pages=entry.get("pages", 0),
                    created_at=entry.get("created_at", 0.0),
                    schema_version=entry.get("schema_version", 0),
                    label=entry.get("label", ""),
                    snapshot_id=entry.get("snapshot_id")
                    or _snapshot_id_from_name(entry["name"]),
                    blob_name=entry.get("blob_name", ""),
                    blob_sha256=entry.get("blob_sha256", ""),
                    blob_count=entry.get("blob_count", 0),
                    blob_files=entry.get("blob_files", {}) or {},
                )
            )
        return sorted(out, key=lambda b: b.created_at)

    def latest(self) -> BackupInfo | None:
        backups = self.list()
        return backups[-1] if backups else None

    # ── rotation ─────────────────────────────────────────────────────────────
    def rotate(
        self,
        *,
        policy: RetentionPolicy | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete backups outside the retention policy.

        Without ``policy`` this keeps the last ``self.keep`` plus one per day
        (the legacy behavior). With a :class:`RetentionPolicy` the full
        borg-style daily/weekly/monthly ladder applies. ``dry_run=True``
        reports what *would* be deleted without touching disk.
        """
        backups = self.list()
        if policy is None:
            if len(backups) <= self.keep:
                return []
            keepers: set[str] = {b.path for b in backups[-self.keep:]}
            seen_days: set[str] = set()
            for backup in reversed(backups):
                day = time.strftime("%Y-%m-%d", time.gmtime(backup.created_at))
                if day not in seen_days:
                    seen_days.add(day)
                    keepers.add(backup.path)
        else:
            keepers = policy.select_keepers(backups)
        removed: list[str] = []
        for backup in backups:
            if backup.path in keepers:
                continue
            removed.append(backup.name)
            if dry_run:
                continue
            Path(backup.path).unlink(missing_ok=True)
            # A pruned snapshot's blob sidecar goes with it, or disk leaks.
            if backup.blob_name:
                (self.directory / backup.blob_name).unlink(missing_ok=True)
        if removed and not dry_run:
            removed_names = set(removed)
            with self._manifest_lock():
                entries = [
                    e for e in self._load_manifest()
                    if e["name"] not in removed_names and Path(e["path"]).is_file()
                ]
                self.manifest_path.write_text(
                    json.dumps(entries, indent=2), encoding="utf-8")
            _log.info("rotated %d old backups", len(removed))
        return removed

    # ── verification & restore ───────────────────────────────────────────────
    def verify(self, backup: BackupInfo | None = None) -> list[str]:
        """Verify a backup by restoring it into a temp dir and checking it.

        Checks, in order: archive checksum, successful extraction, SQLite
        openability + ``PRAGMA integrity_check`` + non-empty schema, then the
        blob sidecar (archive checksum, extraction, per-blob presence and
        checksums). Returns a list of human-readable problems; empty means
        the backup is restorable.
        """
        target = backup or self.latest()
        if target is None:
            return ["no backups present"]
        problems: list[str] = []
        path = Path(target.path)
        if not path.is_file():
            return [f"missing: {path}"]
        # Manifest consistency (F07): several entries claiming one path
        # with different checksums means identities aliased — at most one
        # of those entries can describe the file actually on disk.
        siblings = [
            e for e in self._load_manifest() if e.get("path") == target.path
        ]
        claimed = {e.get("sha256") for e in siblings if e.get("sha256")}
        if len(claimed) > 1:
            problems.append(
                f"manifest alias: {len(siblings)} entries claim {target.name} "
                "with different checksums — snapshot identities collided"
            )
        if target.sha256:
            actual = _sha256_file(path)
            if actual != target.sha256:
                problems.append(f"checksum mismatch for {target.name}")
                # A checksum failure means the bytes are wrong; every later check
                # would be noise, and decompression may itself explode.
                return problems

        tmpdir = Path(tempfile.mkdtemp(prefix="nm-backup-verify-"))
        try:
            probe = tmpdir / "restore.db"
            try:
                self._extract(target, probe)
            except Exception as exc:
                # Corrupt gzip, truncated file, unreadable archive: that IS the
                # finding. verify() must report it, never raise it at the caller.
                problems.append(f"cannot extract {target.name}: {type(exc).__name__}: {exc}")
                return problems
            try:
                conn = sqlite3.connect(str(probe))
                try:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                    if result is None or result[0] != "ok":
                        problems.append(f"integrity_check failed for {target.name}: {result}")
                    count = conn.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    ).fetchone()[0]
                    if count == 0:
                        problems.append(f"backup {target.name} contains no tables")
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                problems.append(f"cannot open backup {target.name} as sqlite: {exc}")
            problems.extend(self._verify_blobs(target))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return problems

    def verify_all(self) -> dict[str, list[str]]:
        """Deep-check every backup (restic ``check``): name → problems.

        Empty problem lists are omitted from the result, so ``{}`` means
        every backup is restorable.
        """
        report: dict[str, list[str]] = {}
        for backup in self.list():
            problems = self.verify(backup)
            if problems:
                report[backup.name] = problems
        return report

    def restore(self, backup: BackupInfo | str | os.PathLike[str], *, target: str | os.PathLike[str] | None = None, overwrite: bool = False, safety_copy: bool = True) -> Path:
        """Restore a backup to ``target`` (default: alongside the backup).

        Verify-before-publish (F06): nothing is written to the destination
        until the snapshot has passed :meth:`verify` (archive checksum,
        extraction, SQLite integrity, blob sidecar) *and* the payload
        extracted into a staging directory beside the destination has
        passed its own integrity check. Publication is an atomic
        ``os.replace``. Any failure before publication leaves the previous
        target untouched and raises — a damaged snapshot can never be
        reported as a successful recovery.

        WAL contract: snapshots are self-contained (checkpointed at
        creation), and any ``-wal``/``-shm`` sidecars beside the
        destination belong to the *previous* file generation, so
        publication removes them — a stale WAL must never replay into the
        restored database. The pre-restore safety copy is taken through
        the SQLite backup API, capturing committed WAL content a plain
        file copy would silently lose; a target that is not an openable
        database is preserved byte-for-byte, sidecars included.

        Refuses to overwrite an existing target unless ``overwrite=True``.
        An overwrite restore is an offline operation for the target:
        stop its writer first — the mechanics protect the file, they do
        not coordinate with a live connection.
        """
        if isinstance(backup, (str, os.PathLike)):
            candidates = [b for b in self.list() if b.name == Path(str(backup)).name or b.path == str(backup)]
            if not candidates:
                raise StorageError(f"backup not found: {backup}")
            backup = candidates[-1]
        destination = Path(target) if target else Path(backup.path).with_suffix("").with_suffix(".restored.db")
        if str(destination).endswith(".gz"):
            destination = destination.with_suffix("")
        if destination.exists() and not overwrite:
            raise StorageError(
                f"restore target exists: {destination} "
                "(pass overwrite=True to replace it)"
            )
        problems = self.verify(backup)
        if problems:
            raise StorageError(
                f"restore refused: backup {backup.name} failed verification: "
                + "; ".join(problems)
            )
        if backup.blob_name and not self.blob_dir:
            raise StorageError(
                f"restore refused: backup {backup.name} carries a blob "
                f"sidecar ({backup.blob_name}) but this manager has no "
                "blob_dir to restore it into"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".restore-{destination.name}-", dir=destination.parent))
        try:
            staged = staging / destination.name
            self._extract(backup, staged)
            staged_problems = _database_file_problems(
                staged, expected_pages=backup.pages)
            if staged_problems:
                raise StorageError(
                    f"restore refused: staged payload of {backup.name} is "
                    "not a sound database: " + "; ".join(staged_problems)
                )
            blob_stage: Path | None = None
            if backup.blob_name:
                blob_stage = staging / "blobs"
                self._extract_blob_archive(
                    self.directory / backup.blob_name, blob_stage)
                self._publish_blobs(blob_stage, backup)
            if destination.exists() and safety_copy:
                spare = self._safety_copy(destination)
                _log.info("pre-restore safety copy: %s", spare)
            os.replace(staged, destination)
            for suffix in ("-wal", "-shm"):
                Path(f"{destination}{suffix}").unlink(missing_ok=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        _log.info("restored %s -> %s", backup.name, destination)
        return destination

    def _safety_copy(self, destination: Path) -> Path:
        """Preserve the pre-restore target as ``<target>.pre-restore-*``.

        Through the SQLite backup API when the target opens as a database
        (a live WAL target's committed rows may exist only in its ``-wal``
        sidecar — a plain copy of the main file would preserve an older,
        torn state and call it a safety copy). A target that does not
        open as a database is copied byte-for-byte, sidecars included.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        spare = destination.with_name(
            f"{destination.name}.pre-restore-{stamp}-{secrets.token_hex(3)}")
        try:
            source = sqlite3.connect(
                f"{destination.absolute().as_uri()}?mode=ro", uri=True)
            try:
                out = sqlite3.connect(str(spare))
                try:
                    source.backup(out)
                finally:
                    out.close()
            finally:
                source.close()
        except sqlite3.Error:
            spare.unlink(missing_ok=True)
            shutil.copy2(destination, spare)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{destination}{suffix}")
                if sidecar.is_file():
                    shutil.copy2(sidecar, Path(f"{spare}{suffix}"))
        return spare

    def _publish_blobs(self, staged_dir: Path, backup: BackupInfo) -> None:
        """Copy staged blob files into ``blob_dir``, checksum-verified.

        Runs before the database is published, so a blob failure aborts
        the restore with the previous database target still untouched.
        Files the snapshot does not carry are left alone — restore is
        not a sync-delete.
        """
        blob_dir = Path(self.blob_dir).expanduser()  # type: ignore[arg-type]
        blob_dir.mkdir(parents=True, exist_ok=True)
        for rel, digest in sorted((backup.blob_files or {}).items()):
            target = blob_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_dir / rel, target)
            if _sha256_file(target) != digest:
                raise StorageError(f"restored blob failed checksum: {rel}")

    def _extract(self, backup: BackupInfo, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(backup.path)
        try:
            if source.suffix == ".gz":
                with gzip.open(source, "rb") as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
            else:
                shutil.copy2(source, destination)
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            destination.unlink(missing_ok=True)
            raise StorageError(f"cannot extract backup {backup.name}: {exc}") from exc
        return destination

    # ── git shipping ─────────────────────────────────────────────────────────
    def push_to_git(self, *, backup: BackupInfo | None = None, message: str = "") -> dict[str, Any]:
        """Commit and push a backup to a separate Git repository.

        Backups go to their own repo, never the code repo: a nightly database
        snapshot would otherwise bury the project history and blow past hosting
        limits. Returns the subprocess outcome so callers can react to a failed
        push rather than discovering it a week later.
        """
        if not self.git_repo:
            return {"pushed": False, "reason": "no git_repo configured"}
        target = backup or self.latest()
        if target is None:
            return {"pushed": False, "reason": "no backup to push"}

        workdir = self.directory / ".git-stage"
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)

        def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *args],
                cwd=str(cwd or workdir),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=300,
                check=False,
            )

        init = run("clone", "--depth", "1", "--branch", self.git_branch, self.git_repo, "repo")
        repo_dir = workdir / "repo"
        if init.returncode != 0:
            # Fresh remote (no branch yet): start a local repo and wire it up.
            repo_dir.mkdir(parents=True, exist_ok=True)
            run("init", cwd=repo_dir)
            run("remote", "add", "origin", self.git_repo, cwd=repo_dir)
            run("checkout", "-B", self.git_branch, cwd=repo_dir)
        shutil.copy2(target.path, repo_dir / target.name)
        (repo_dir / MANIFEST).write_text(
            json.dumps(self._load_manifest(), indent=2), encoding="utf-8"
        )

        run("config", "user.name", self.git_author.split(" <")[0], cwd=repo_dir)
        run("config", "user.email", self.git_author.split("<")[-1].rstrip(">"), cwd=repo_dir)
        run("add", "-A", cwd=repo_dir)
        commit_message = message or f"backup {target.name} ({target.size} bytes)"
        commit = run("commit", "-m", commit_message, cwd=repo_dir)
        if commit.returncode != 0 and "nothing to commit" in (commit.stdout + commit.stderr):
            return {"pushed": False, "reason": "nothing changed", "backup": target.name}
        push = run("push", "origin", self.git_branch, cwd=repo_dir)
        result = {
            "pushed": push.returncode == 0,
            "backup": target.name,
            "sha256": target.sha256,
            "stdout": push.stdout[-2000:],
            "stderr": push.stderr[-2000:],
        }
        if push.returncode != 0:
            _log.error("backup push failed: %s", push.stderr.strip()[:500])
        shutil.rmtree(workdir, ignore_errors=True)
        return result

    def push_to_s3(self, store: Any, *, backup: BackupInfo | None = None) -> dict[str, Any]:
        """Upload a backup (and its blob sidecar, if any) to a blob store.

        ``store`` is anything with the blob API — an
        :class:`~nomorals.storage.s3blob.S3BlobStore` for the offsite leg, or
        the local :class:`~nomorals.storage.blob.BlobStore`; build it with
        :func:`~nomorals.storage.s3blob.open_blob_store`. Uploads are
        content-addressed, so re-pushing the same backup is a dedup hit, not a
        second copy. The store's returned SHA-256 is checked against the
        manifest's — a mismatch fails fast instead of recording a phantom
        offsite copy. Returns a result dict mirroring :meth:`push_to_git`.
        """
        target = backup or self.latest()
        if target is None:
            return {"pushed": False, "reason": "no backup to push"}
        objects: list[dict[str, Any]] = []

        def _ship(path: str | os.PathLike[str], name: str, mime: str, expected_sha: str) -> None:
            info = store.put_file(path, mime=mime)
            if info.sha256 != expected_sha:
                raise StorageError(
                    f"offsite upload corrupted {name}: "
                    f"store sha256 {info.sha256} != manifest {expected_sha}"
                )
            objects.append({"file": name, "sha256": info.sha256, "size": info.size})

        _ship(
            target.path,
            target.name,
            "application/gzip" if str(target.path).endswith(".gz") else "application/octet-stream",
            target.sha256,
        )
        if target.blob_name:
            sidecar = self.directory / target.blob_name
            if sidecar.is_file():
                _ship(sidecar, target.blob_name, "application/gzip", target.blob_sha256)
            else:
                _log.warning("blob sidecar %s missing; shipping database only", target.blob_name)
        _log.info("backup pushed to blob store: %s (%d objects)", target.name, len(objects))
        return {
            "pushed": True,
            "backup": target.name,
            "sha256": target.sha256,
            "objects": objects,
        }

    # ── scheduled backups ────────────────────────────────────────────────────
    def backup_if_due(self, interval_seconds: float) -> BackupInfo | None:
        """Create a backup only if the newest one is older than ``interval_seconds``."""
        latest = self.latest()
        if latest is not None and (time.time() - latest.created_at) < interval_seconds:
            return None
        info = self.create(label="scheduled")
        self.rotate()
        return info

    def run_scheduler(self, interval_seconds: float, *, should_stop: Any = None) -> None:
        """Blocking loop; intended to run on a daemon thread."""
        while should_stop is None or not should_stop():
            try:
                self.backup_if_due(interval_seconds)
                if self.git_repo:
                    self.push_to_git()
                if self.offsite_store is not None:
                    self.push_to_s3(self.offsite_store)
            except Exception as exc:  # noqa: BLE001 - a scheduler must never die
                _log.error("scheduled backup failed: %s", exc)
            for _ in range(int(max(1, interval_seconds / 5))):
                if should_stop is not None and should_stop():
                    return
                time.sleep(5)

    def stats_snapshot(self) -> dict[str, Any]:
        backups = self.list()
        return {
            "count": len(backups),
            "bytes": sum(b.size for b in backups),
            "latest": backups[-1].to_dict() if backups else None,
            "git_repo": self.git_repo or None,
            "directory": str(self.directory),
        }

    def estimate(self) -> dict[str, Any]:
        """Project the next backup's size before taking it (borg ``info``)."""
        pages = int(self.db.scalar("PRAGMA page_count", default=0))
        page_size = int(self.db.scalar("PRAGMA page_size", default=0))
        raw_bytes = pages * page_size
        blob_bytes = 0
        blob_files = 0
        if self.include_blobs and self.blob_dir:
            source = Path(self.blob_dir).expanduser()
            if source.is_dir():
                for path in source.rglob("*"):
                    if path.is_file() and not path.is_symlink():
                        blob_files += 1
                        try:
                            blob_bytes += path.stat().st_size
                        except OSError as e:
                            record_failure('backup.BackupManager.estimate', e)
        return {
            "db_pages": pages,
            "db_raw_bytes": raw_bytes,
            "db_compressed_estimate_bytes": int(raw_bytes * 0.35) if self.compress else raw_bytes,
            "blob_files": blob_files,
            "blob_bytes": blob_bytes,
            "compressed": self.compress,
        }

    def format_table(self, theme: Any = None) -> str:
        """borg-``list``-style backup table through the shared style layer."""
        theme = theme or active_theme()
        backups = self.list()
        now = time.time()
        rows = []
        for backup in backups:
            age_s = now - backup.created_at
            if age_s < 3600:
                age = f"{int(age_s // 60)}m"
            elif age_s < 86400:
                age = f"{age_s / 3600:.1f}h"
            else:
                age = f"{age_s / 86400:.1f}d"
            rows.append([
                backup.name,
                age,
                _backup_bytes(backup.size),
                str(backup.blob_count) if backup.blob_count else "–",
                f"v{backup.schema_version}",
                backup.label or "–",
            ])
        table = styled_table(
            ["name", "age", "size", "blobs", "schema", "label"],
            rows, theme=theme,
        )
        total = sum(b.size for b in backups)
        return (
            header(f"backups ({len(backups)}, {_backup_bytes(total)})",
                   theme=theme)
            + "\n" + table
        )


def _backup_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024
    return f"{num:.1f}TB"  # pragma: no cover


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_id_from_name(name: str) -> str:
    """Snapshot identity for a manifest entry that predates ``snapshot_id``:
    the file name minus its ``.db``/``.db.gz`` payload suffix."""
    stem = name
    if stem.endswith(".gz"):
        stem = stem[: -len(".gz")]
    if stem.endswith(".db"):
        stem = stem[: -len(".db")]
    return stem


def _database_file_problems(path: Path, *, expected_pages: int = 0) -> list[str]:
    """Integrity problems of an extracted database file, or ``[]``.

    The restore path runs this on the *staged* bytes it is about to
    publish — the same checks :meth:`BackupManager.verify` performs on
    its own probe extraction (integrity, non-empty schema), plus the
    page count the manifest recorded at creation when it is known.
    """
    problems: list[str] = []
    try:
        conn = sqlite3.connect(str(path))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                problems.append(f"integrity_check failed: {result}")
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            if count == 0:
                problems.append("database contains no tables")
            if expected_pages:
                pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
                if pages != int(expected_pages):
                    problems.append(
                        f"page count {pages} != manifest pages {expected_pages}"
                    )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        problems.append(f"cannot open as sqlite: {exc}")
    return problems
