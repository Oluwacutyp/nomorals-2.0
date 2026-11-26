"""The FileStore — where files live, by shelf.

Shelves are real directories under ``<home>/data/files/<shelf>/``.
A file is shelved by moving it there (atomic rename on one
filesystem, copy + unlink across them); nothing is ever "shelved"
in a database while the bytes sit somewhere else. The downloads
directory (``<home>/data/downloads`` — where Devon already fetches
files) is the intake: files are seen there, then shelved into the
store. Packs live in ``<root>/packs``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .file import File, FileError, resolve
from .kinds import kind_for

__all__ = ["FileStore"]


class FileStore:
    """Shelves over one root. ``root`` is explicit in tests; in
    production it resolves from ``context.settings``."""

    def __init__(self, context: Any, root: str | Path | None = None
                 ) -> None:
        self.context = context
        if root is not None:
            self.root = Path(root)
        else:
            settings = getattr(context, "settings", None)
            if settings is None or not hasattr(settings, "resolve"):
                raise FileError(
                    "FileStore needs a context with settings.resolve "
                    "or an explicit root")
            self.root = Path(settings.resolve("data/files"))
        self.root.mkdir(parents=True, exist_ok=True)

    # ── roots ────────────────────────────────────────────────────────
    @property
    def packs_dir(self) -> Path:
        d = self.root / "packs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def downloads_dir(self) -> Path | None:
        settings = getattr(self.context, "settings", None)
        if settings is None or not hasattr(settings, "resolve"):
            return None
        return Path(settings.resolve("data/downloads"))

    # ── shelves ──────────────────────────────────────────────────────
    def shelf_dir(self, shelf: str) -> Path:
        name = (shelf or "").strip().strip("/")
        if not name or ".." in Path(name).parts or "/" in name:
            raise FileError(
                f"shelf {shelf!r} is not a plain shelf name")
        return self.root / name

    def shelves(self) -> list[dict[str, Any]]:
        """Every shelf that holds files, with counts and bytes."""
        out: list[dict[str, Any]] = []
        try:
            dirs = sorted(d for d in self.root.iterdir()
                          if d.is_dir())
        except OSError:
            return out
        for d in dirs:
            files = [f for f in d.iterdir() if f.is_file()]
            total = 0
            for f in files:
                try:
                    total += f.stat().st_size
                except OSError:
                    continue
            out.append({"shelf": d.name, "files": len(files),
                        "bytes": total, "path": str(d)})
        return out

    def shelf(self, name: str) -> list[File]:
        """Every file on one shelf, as Files."""
        d = self.shelf_dir(name)
        if not d.is_dir():
            raise FileError(f"no shelf {name!r} in the file store")
        files = []
        for p in sorted(d.iterdir()):
            if p.is_file():
                files.append(File(p, shelf=d.name))
        return files

    def downloads(self) -> list[File]:
        """The intake: every file Devon fetched, as Files (read-only
        view — shelving moves them into the store)."""
        d = self.downloads_dir
        if d is None or not d.is_dir():
            return []
        files = []
        try:
            found = sorted(f for f in d.rglob("*") if f.is_file())
        except OSError:
            return []
        for p in found:
            kind, shelf = kind_for(p)
            files.append(File(p, kind=kind, shelf=shelf,
                              provenance={"intake": "downloads"}))
        return files

    # ── moves ────────────────────────────────────────────────────────
    def shelve(self, ref: str, shelf: str = "") -> File:
        """Move a resolved file onto a shelf (its kind's shelf when
        none is named). The file's new home is the returned File;
        the old path stops existing."""
        f = resolve(self.context, ref)
        if f.is_archive_member:
            raise FileError(
                "an archive member cannot be shelved — unpack the "
                "archive first, then shelve the file")
        target_shelf = (shelf or "").strip() or f.shelf
        dest_dir = self.shelf_dir(target_shelf)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name
        if dest.resolve() == f.path.resolve():
            return File(dest, kind=f.kind, shelf=target_shelf,
                        provenance={"ref": ref, "tool": "files",
                                    "shelved": True})
        if dest.exists():
            raise FileError(
                f"shelf {target_shelf!r} already holds {f.name!r}")
        try:
            os.replace(f.path, dest)
        except OSError:
            shutil.copy2(f.path, dest)
            os.unlink(f.path)
        return File(dest, kind=f.kind, shelf=target_shelf,
                    provenance={"ref": ref, "tool": "files",
                                "shelved": True})

    def unshelve(self, ref: str, dest: str = "") -> File:
        """Move a shelved file back out — to ``dest`` (a directory),
        or to the downloads intake when none is given."""
        f = resolve(self.context, ref)
        if f.is_archive_member:
            raise FileError(
                "an archive member cannot be unshelved")
        if dest:
            target_dir = Path(os.path.expanduser(dest))
            if not target_dir.is_absolute():
                from ..tools.filesystem import safe_path

                target_dir = safe_path(self.context, dest)
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.downloads_dir
            if target_dir is None:
                raise FileError("no downloads intake to unshelve to")
            target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f.name
        if target.resolve() == f.path.resolve():
            return f
        if target.exists():
            raise FileError(f"{target_dir} already holds {f.name!r}")
        try:
            os.replace(f.path, target)
        except OSError:
            shutil.copy2(f.path, target)
            os.unlink(f.path)
        return File(target, kind=f.kind, shelf=f.shelf,
                    provenance={"ref": ref, "tool": "files",
                                "unshelved": True})

    def all_files(self, *, cap: int = 500) -> list[File]:
        """Every shelved file, for the exhibitor. Capped, with the
        cap named by the caller — a census never silently drops the
        tail."""
        files: list[File] = []
        try:
            found = sorted(f for f in self.root.rglob("*")
                           if f.is_file())
        except OSError:
            return []
        for p in found[:cap]:
            kind, shelf = kind_for(p)
            shelf_name = p.parent.name
            files.append(File(p, kind=kind, shelf=shelf_name,
                              provenance={"store": str(self.root)}))
        return files
