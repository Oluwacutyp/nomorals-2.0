"""Package / export scaffolded projects as verifiable ``.tar.gz`` archives.

:func:`export_project` tars a project directory (excluding ``.git``,
``__pycache__``, ``.venv``, ``node_modules``, and the archive itself)
and embeds a ``MANIFEST.json`` listing every included file with its
sha256.  :func:`verify_export` re-opens an archive, recomputes the
hashes, and reports mismatches -- a tamper-evident round trip.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["ExportResult", "VerifyResult", "export_project", "verify_export"]

#: Directory names never packaged.
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
                ".mypy_cache", ".pytest_cache", ".tox", "dist", "build"}

MANIFEST_NAME = "MANIFEST.json"


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

    def to_dict(self) -> dict[str, Any]:
        return {"archive": str(self.archive), "files": list(self.files),
                "bytes": self.bytes}


@dataclass
class VerifyResult:
    ok: bool
    files_checked: int = 0
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "files_checked": self.files_checked,
                "problems": list(self.problems)}


def export_project(project_dir: str | Path,
                   dest: str | Path | None = None) -> ExportResult:
    """Create ``<name>.tar.gz`` from ``project_dir``; return an :class:`ExportResult`.

    ``dest`` may be a directory (archive lands inside it) or a full file
    path.  Defaults to ``<parent>/<name>.tar.gz``.
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

    entries: list[tuple[str, Path]] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file() or _excluded(path, project_dir):
            continue
        if path.resolve() == archive.resolve():
            continue
        entries.append((path.relative_to(project_dir).as_posix(), path))
    if not entries:
        raise ToolError(f"nothing to export in {project_dir}")

    manifest = {
        "project": project_dir.name,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "files": [
            {"path": rel, "sha256": _sha256(path), "size": path.stat().st_size}
            for rel, path in entries
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    with tarfile.open(archive, "w:gz") as tar:
        for rel, path in entries:
            tar.add(path, arcname=f"{project_dir.name}/{rel}")
        info = tarfile.TarInfo(f"{project_dir.name}/{MANIFEST_NAME}")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest_bytes))

    size = archive.stat().st_size
    files = [rel for rel, _ in entries]
    _log.info("exported %s -> %s (%d files, %d bytes)",
              project_dir, archive, len(files), size)
    return ExportResult(archive=archive, files=files, bytes=size)


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
