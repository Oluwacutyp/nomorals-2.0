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

import gzip
import hashlib
import json
import os
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
from .db import Database

__all__ = ["BackupInfo", "BackupManager"]

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
            "blob_name": self.blob_name,
            "blob_sha256": self.blob_sha256,
            "blob_count": self.blob_count,
            "blob_files": self.blob_files,
        }


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

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    # ── create ───────────────────────────────────────────────────────────────
    def create(self, label: str = "") -> BackupInfo:
        """Take a consistent snapshot of the database."""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        base = f"nomorals-{stamp}"
        if label:
            safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
            base = f"{base}-{safe}"

        raw_path = self.directory / f"{base}.db"
        source = self.db._connection()  # noqa: SLF001 - intentional: same-process snapshot
        destination = sqlite3.connect(str(raw_path))
        try:
            source.backup(destination, pages=0)
        finally:
            destination.close()

        # Checkpoint first so the snapshot is self-contained (no -wal sidecar needed).
        try:
            self.db.checkpoint("TRUNCATE")
        except StorageError:  # noqa: E103 - not in WAL mode; nothing to checkpoint (pragma: no cover)
            pass

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

    def _append_manifest(self, info: BackupInfo) -> None:
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
    def rotate(self) -> list[str]:
        """Delete the oldest backups beyond ``keep``, keeping one per day.

        The daily sample is what makes "restore to last Tuesday" possible without
        keeping every hourly snapshot forever.
        """
        backups = self.list()
        if len(backups) <= self.keep:
            return []
        keepers: set[str] = {b.path for b in backups[-self.keep :]}
        seen_days: set[str] = set()
        for backup in reversed(backups):
            day = time.strftime("%Y-%m-%d", time.gmtime(backup.created_at))
            if day not in seen_days:
                seen_days.add(day)
                keepers.add(backup.path)
        removed: list[str] = []
        for backup in backups:
            if backup.path in keepers:
                continue
            Path(backup.path).unlink(missing_ok=True)
            # A pruned snapshot's blob sidecar goes with it, or disk leaks.
            if backup.blob_name:
                (self.directory / backup.blob_name).unlink(missing_ok=True)
            removed.append(backup.name)
        if removed:
            removed_names = set(removed)
            entries = [
                e for e in self._load_manifest()
                if e["name"] not in removed_names and Path(e["path"]).is_file()
            ]
            self.manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
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

    def restore(self, backup: BackupInfo | str | os.PathLike[str], *, target: str | os.PathLike[str] | None = None) -> Path:
        """Extract a backup to ``target`` (default: alongside the backup)."""
        if isinstance(backup, (str, os.PathLike)):
            candidates = [b for b in self.list() if b.name == Path(str(backup)).name or b.path == str(backup)]
            if not candidates:
                raise StorageError(f"backup not found: {backup}")
            backup = candidates[-1]
        destination = Path(target) if target else Path(backup.path).with_suffix("").with_suffix(".restored.db")
        if str(destination).endswith(".gz"):
            destination = destination.with_suffix("")
        extracted = self._extract(backup, destination)
        _log.info("restored %s -> %s", backup.name, extracted)
        return extracted

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
                text=True,
                timeout=300,
                check=False,
            )

        init = run("clone", "--depth", "1", "--branch", self.git_branch, self.git_repo, "repo")
        if init.returncode != 0:
            workdir.mkdir(parents=True, exist_ok=True)
            run("init", cwd=workdir / "repo")
            (workdir / "repo").mkdir(parents=True, exist_ok=True)
            run("remote", "add", "origin", self.git_repo, cwd=workdir / "repo")
            run("checkout", "-B", self.git_branch, cwd=workdir / "repo")
        repo_dir = workdir / "repo"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
