"""Package / export scaffolded projects as verifiable ``.tar.gz`` archives.

:func:`export_project` tars a project directory (excluding ``.git``,
``__pycache__``, ``.venv``, ``node_modules``, and the archive itself)
and embeds a ``MANIFEST.json`` listing every included file with its
sha256.  :func:`verify_export` re-opens an archive, recomputes the
hashes, and reports mismatches -- a tamper-evident round trip.

``reproducible=True`` applies the reproducible-builds.org discipline:
fixed mtimes from ``SOURCE_DATE_EPOCH``, uid/gid 0, normalized modes,
``gzip -n`` semantics — so the same project exported twice is
**byte-identical**.  :func:`verify_reproducible` proves it.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["ExportResult", "VerifyResult", "export_project", "verify_export",
           "verify_reproducible", "source_date_epoch"]

#: Directory names never packaged.
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
                ".mypy_cache", ".pytest_cache", ".tox", "dist", "build"}

MANIFEST_NAME = "MANIFEST.json"


def source_date_epoch() -> int | None:
    """Honor ``SOURCE_DATE_EPOCH`` (reproducible-builds.org standard)."""
    raw = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        _log.warning("ignoring invalid SOURCE_DATE_EPOCH=%r", raw)
        return None


def _excluded(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(
        part in EXCLUDE_DIRS or part.endswith(".egg-info")
        for part in parts[:-1]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ExportResult:
    archive: Path
    files: list[str] = field(default_factory=list)
    bytes: int = 0
    #: sha256 of the archive itself (lets callers compare rebuilds)
    sha256: str = ""
    #: True when reproducible discipline was applied
    reproducible: bool = False

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        return {"archive": str(self.archive), "files": list(self.files),
                "bytes": self.bytes, "sha256": self.sha256,
                "file_count": self.file_count,
                "reproducible": self.reproducible}


@dataclass
class VerifyResult:
    ok: bool
    files_checked: int = 0
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "files_checked": self.files_checked,
                "problems": list(self.problems)}


def _add_deterministic(tar: tarfile.TarFile, path: Path,
                       arcname: str, epoch: int) -> None:
    """Add one file with normalized metadata (reproducible builds)."""
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    info.pax_headers = {}
    with path.open("rb") as fh:
        tar.addfile(info, fh)


def export_project(project_dir: str | Path,
                   dest: str | Path | None = None,
                   reproducible: bool = False,
                   epoch: int | None = None) -> ExportResult:
    """Create ``<name>.tar.gz`` from ``project_dir``; return an :class:`ExportResult`.

    ``dest`` may be a directory (archive lands inside it) or a full file
    path.  Defaults to ``<parent>/<name>.tar.gz``.

    ``reproducible=True`` applies the reproducible-builds.org
    discipline — sorted entries (already), mtimes clamped to
    ``epoch`` (or ``SOURCE_DATE_EPOCH``, or ``time.time()`` when neither
    is set), uid/gid 0, normalized modes, ``gzip -n`` — so exporting the
    same project twice yields byte-identical archives.  The manifest
    records ``reproducible`` and the toolchain (python version,
    platform) for provenance.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise ToolError(f"not a directory: {project_dir}")

    if dest is None:
        archive = project_dir.parent / f"{project_dir.name}.tar.gz"
    else:
        dest = Path(dest).expanduser()
        archive = dest / f"{project_dir.name}.tar.gz" if dest.suffix != ".gz" else dest
    if archive.exists():
        raise ToolError(f"archive already exists: {archive}")

    epoch_val = epoch if epoch is not None else source_date_epoch()
    if reproducible and epoch_val is None:
        epoch_val = int(time.time())
        _log.info("export_project: reproducible without SOURCE_DATE_EPOCH; "
                  "using current time as epoch (set SOURCE_DATE_EPOCH for "
                  "cross-machine reproducibility)")

    entries: list[tuple[str, Path]] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file() or _excluded(path, project_dir):
            continue
        if path.resolve() == archive.resolve():
            continue
        entries.append((path.relative_to(project_dir).as_posix(), path))
    if not entries:
        raise ToolError(f"nothing to export in {project_dir}")

    exported_at = (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_val))
                   if reproducible and epoch_val is not None
                   else time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()))
    manifest = {
        "project": project_dir.name,
        "exported_at": exported_at,
        "reproducible": reproducible,
        "toolchain": {"python": platform.python_version(),
                      "implementation": platform.python_implementation(),
                      "platform": sys.platform},
        "files": [
            {"path": rel, "sha256": _sha256(path), "size": path.stat().st_size}
            for rel, path in entries
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                      ).encode("utf-8")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, path in entries:
            arcname = f"{project_dir.name}/{rel}"
            if reproducible:
                assert epoch_val is not None
                _add_deterministic(tar, path, arcname, epoch_val)
            else:
                tar.add(path, arcname=arcname)
        info = tarfile.TarInfo(f"{project_dir.name}/{MANIFEST_NAME}")
        info.size = len(manifest_bytes)
        if reproducible:
            assert epoch_val is not None
            info.mtime = epoch_val
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            info.pax_headers = {}
        else:
            info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest_bytes))

    # gzip -n semantics: fixed mtime in the gzip header when reproducible
    data = gzip.compress(buf.getvalue(), compresslevel=9,
                         mtime=(epoch_val if reproducible
                                else int(time.time())))
    archive.write_bytes(data)

    size = archive.stat().st_size
    digest = hashlib.sha256(data).hexdigest()
    files = [rel for rel, _ in entries]
    _log.info("exported %s -> %s (%d files, %d bytes%s)",
              project_dir, archive, len(files), size,
              ", reproducible" if reproducible else "")
    return ExportResult(archive=archive, files=files, bytes=size,
                        sha256=digest, reproducible=reproducible)


def verify_export(archive: str | Path) -> VerifyResult:
    """Re-open ``archive`` and check every file against the embedded manifest."""
    archive = Path(archive).expanduser().resolve()
    problems: list[str] = []
    if not archive.is_file():
        return VerifyResult(ok=False, problems=[f"archive not found: {archive}"])
    try:
        tar = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        return VerifyResult(ok=False, problems=[f"cannot open archive: {exc}"])

    with tar:
        names = tar.getnames()
        manifest_name = next((n for n in names if n.endswith(MANIFEST_NAME)), None)
        if manifest_name is None:
            return VerifyResult(ok=False, problems=["MANIFEST.json missing from archive"])
        try:
            manifest = json.loads(tar.extractfile(manifest_name).read().decode("utf-8"))  # type: ignore[union-attr]
            expected_files = manifest["files"]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as exc:
            return VerifyResult(ok=False, problems=[f"manifest unreadable: {exc}"])

        prefix = manifest_name[: -len(MANIFEST_NAME)]
        seen: set[str] = set()
        for entry in expected_files:
            rel = entry["path"]
            arcname = f"{prefix}{rel}"
            seen.add(arcname)
            member = tar.getmember(arcname) if arcname in names else None
            if member is None:
                problems.append(f"missing from archive: {rel}")
                continue
            data = tar.extractfile(member).read()  # type: ignore[union-attr]
            actual = hashlib.sha256(data).hexdigest()
            if actual != entry["sha256"]:
                problems.append(f"sha256 mismatch: {rel}")
            if len(data) != entry["size"]:
                problems.append(f"size mismatch: {rel} "
                                f"(manifest {entry['size']}, actual {len(data)})")
        for name in names:
            if name not in seen and name != manifest_name and not name.endswith("/"):
                problems.append(f"unexpected extra file in archive: {name}")

    ok = not problems
    _log.info("verify_export %s -> %s (%d files)",
              archive, "OK" if ok else "FAIL", len(expected_files))
    return VerifyResult(ok=ok, files_checked=len(expected_files), problems=problems)


def verify_reproducible(project_dir: str | Path,
                        epoch: int | None = None) -> dict[str, Any]:
    """Export twice with ``reproducible=True`` and compare archive hashes.

    The reproducible-builds.org acceptance gate: byte-identical rebuilds.
    Returns ``{"ok", "sha256", "bytes", "detail"}``.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    epoch_val = epoch if epoch is not None else source_date_epoch() \
        or 1700000000
    digests: list[str] = []
    sizes: list[int] = []
    with tempfile.TemporaryDirectory(prefix="nm-repro-a-") as da, \
            tempfile.TemporaryDirectory(prefix="nm-repro-b-") as db:
        for d in (da, db):
            result = export_project(project_dir, dest=d,
                                    reproducible=True, epoch=epoch_val)
            digests.append(result.sha256)
            sizes.append(result.bytes)
    ok = digests[0] == digests[1]
    return {
        "ok": ok,
        "sha256": digests[0],
        "bytes": sizes[0],
        "detail": ("byte-identical across two exports"
                   if ok else
                   f"NON-DETERMINISTIC: {digests[0][:16]} != {digests[1][:16]}"),
    }
