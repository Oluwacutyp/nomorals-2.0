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
        except StorageError:  # pragma: no cover - not in WAL mode
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
        info = BackupInfo(
            name=final_path.name,
            path=str(final_path),
            size=final_path.stat().st_size,
            sha256=digest,
            pages=pages,
            created_at=time.time(),
            schema_version=schema_version,
            label=label,
        )
        self._append_manifest(info)
        _log.info("backup created: %s (%d bytes, %d pages)", info.name, info.size, info.pages)
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
        """Check a backup is a readable SQLite database with a matching checksum."""
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

        probe = self.directory / f".verify-{int(time.time())}.db"
        try:
            try:
                self._extract(target, probe)
            except (OSError, EOFError, Exception) as exc:
                # Corrupt gzip, truncated file, unreadable archive: that IS the
                # finding. verify() must report it, never raise it at the caller.
                problems.append(f"cannot extract {target.name}: {type(exc).__name__}: {exc}")
                return problems
            conn = sqlite3.connect(str(probe))
            try:
                result = conn.execute("PRAGMA integrity_check").fetchone()
                if result is None or result[0] != "ok":
                    problems.append(f"integrity_check returned {result}")
                count = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()[0]
                if count == 0:
                    problems.append("backup contains no tables")
            finally:
                conn.close()
        except sqlite3.Error as exc:
            problems.append(f"cannot open backup as sqlite: {exc}")
        finally:
            probe.unlink(missing_ok=True)
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
